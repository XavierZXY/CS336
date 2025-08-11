import logging
import os
from collections import defaultdict
from multiprocessing import Process, Queue
from typing import BinaryIO

import regex as re
from rich.logging import RichHandler

# Configure rich logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)

log = logging.getLogger("rich")


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def split_by_special_tokens(text: str, special_tokens: list[str]) -> list[str]:
    """Split text by special tokens.
    example:
        text = "Hello <|endoftext|> world"
        special_tokens = ["<|endoftext|>"]
        split_by_special_tokens(text, special_tokens)
        # returns ['Hello ', '<|endoftext|>', ' world']

    Args:
        text (str): _description_
        special_tokens (list[str]): _description_

    Returns:
        list[str]: _description_
    """
    special_tokens_sorted = sorted(special_tokens, key=lambda x: -len(x))
    if not special_tokens_sorted:
        parts = [text]
    else:
        pattern = "|".join(re.escape(tok) for tok in special_tokens_sorted)
        parts = re.split("(" + pattern + ")", text)

    return parts


def pretokenize(
    text: str, special_tokens: list[str], drop_special_token: bool = True
) -> list[bytes]:
    """Seperating text into pretokens, Special tokens are independent pretokens.

    Args:
        text (str): input text
        special_token (list[str]): _description_
        drop_special_token (bool, optional): _description_. Defaults to True.

    Returns:
        list[bytes]: _description_
    """
    parts = split_by_special_tokens(text, special_tokens)
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    tokens_list = []
    for part in parts:
        if part in special_tokens:
            if not drop_special_token:
                spec_tok_bytes = part.encode("utf-8")
                tokens_list.append([spec_tok_bytes])
        else:
            str_tokens = re.findall(PAT, part)
            part_tokens = [s.encode("utf-8") for s in str_tokens]
            tokens_list.append(part_tokens)
    tokens = [token for sublist in tokens_list for token in sublist]
    return tokens


def merge(
    counts: dict[tuple[int, int], int],
    index_dict: dict[tuple[int, int], set[int]],
    pretokens: list[list[int]],
    max_pair: tuple[int, int],
    new_index: int,
):
    """Merge the pairs with highest frequency and update counts, index_dict

    Args:
        counts (dict[tuple[int, int], int]): A dictionary mapping token pairs to their merge counts.
        index_dict (dict[tuple[int, int], set[int]]): A dictionary mapping token pairs to their indices in the pretokens list.
        pretokens (list[list[int]]): The list of pretokenized input.
        max_pair (tuple[int, int]): The token pair with the highest merge count.
        new_index (int): The index of the newly created token.
    """
    index_set = index_dict[max_pair]
    for i in index_set:
        pretoken = pretokens[i]
        new_pretoken = []

        pos_list = []  # store positions of max_pair for each new pretoken after merge
        pos = 0  # record the new position of new_index in new_pretoken
        j = 0
        # replace max_pair with new_index in each pretoken
        while j < len(pretoken):
            # use the new_index to replace
            if (j < len(pretoken) - 1) and ((pretoken[j], pretoken[j + 1]) == max_pair):
                new_pretoken.append(new_index)
                pos_list.append(pos)
                j += 2  # Skip the next token as it is part of the pair
            else:
                new_pretoken.append(pretoken[j])
                j += 1
            pos += 1

        # update counts and index_dict
        for pos in pos_list:
            counts[max_pair] -= 1

            if pos > 0:
                # deal with the left position
                if new_pretoken[pos - 1] == new_index:
                    counts[(max_pair[1], max_pair[0])] -= 1
                else:
                    counts[(new_pretoken[pos - 1], max_pair[0])] -= 1

                counts[(new_pretoken[pos - 1], new_pretoken[pos])] += 1
                index_dict[(new_pretoken[pos - 1], new_pretoken[pos])].add(i)

            if pos < len(new_pretoken) - 1:
                # deal with the right position
                if new_pretoken[pos + 1] == new_index:
                    counts[(max_pair[1], max_pair[0])] -= 1
                else:
                    counts[(max_pair[1], new_pretoken[pos + 1])] -= 1

                counts[(new_pretoken[pos], new_pretoken[pos + 1])] += 1
                index_dict[(new_pretoken[pos], new_pretoken[pos + 1])].add(i)

        pretokens[i] = new_pretoken


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, train a BPE tokenizer.

    Args:
        input_path (str | os.PathLike): The path to the input corpus.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_token (list[str]): A list of special tokens to include in the vocabulary.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]: _description
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    special_tokens = special_tokens or []
    num_merges = max(0, vocab_size - len(special_tokens) - 256)

    # Initialize the vocabulary with special tokens
    vocab = {x: bytes([x]) for x in range(256)}
    for i, token in enumerate(special_tokens):
        vocab[256 + i] = token.encode("utf-8")
    merges = []

    # chunk the file
    ## the number of processes is set to 4 by default
    num_processes = kwargs.get("num_processes", 4)
    chunk_list = []
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            chunk_list.append(chunk)

    # Parallelize pretokenization
    def _worker(text: str, special_token: list[str], q: Queue):
        """Worker function to pretokenize text."""
        pretokens = pretokenize(text, special_token)
        q.put(pretokens)

    pretokens_list = []
    processes = []
    q = Queue()
    for chunk in chunk_list:
        p = Process(target=_worker, args=(chunk, special_tokens, q))
        p.start()
        processes.append(p)

    pretokens_list = [q.get() for _ in processes]

    for p in processes:
        p.join()

    pretokens = [token for tokens in pretokens_list for token in tokens]
    log.info(f"Total pretokens: {len(pretokens)}")
    # Merge pretokens into a single string
    counts = defaultdict(int)
    ## store pretoken location for each pair
    index_dict = defaultdict(set)
    for j, pretoken in enumerate(pretokens):
        for index1, index2 in zip(pretoken, pretoken[1:]):
            counts[index1, index2] += 1
            index_dict[index1, index2].add(j)
    for i in range(num_merges):
        # for i in range(5):
        # Prefer lexicographically greater tokens
        max_pair = max(
            counts.items(),
            key=lambda x: (
                x[1],
                vocab[x[0][0]].decode("utf-8", errors="ignore"),
                vocab[x[0][1]].decode("utf-8", errors="ignore"),
            ),
        )[0]

        index1, index2 = max_pair
        new_index = 256 + len(special_tokens) + i
        vocab[new_index] = vocab[index1] + vocab[index2]
        merges.append((vocab[index1], vocab[index2]))
        log.info(
            f"Counts: {counts}, \n"
            f"index_dict: {index_dict}, \n"
            f"pretokens: {pretokens}, \n"
            f"max_pair: {max_pair}, \n"
            f"new_index: {new_index}"
        )
        merge(counts, index_dict, pretokens, max_pair, new_index)

    return (vocab, merges)


class BPETokenizer:
    def __init__(
        self,
        vocab_size: int,
    ):
        raise NotImplementedError


def main():
    test_file = "test.txt"
    if not os.path.exists(test_file):
        with open(test_file, "w") as f:
            f.write("Hello <|endoftext|> world. This is a test file for BPE tokenizer.")

    special_tokens = ["<|endoftext|>"]
    vocab_size = 1000
    vocab, merges = train_bpe(test_file, vocab_size, special_tokens)


if __name__ == "__main__":
    main()
