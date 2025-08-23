import logging
import os
from collections import defaultdict
from collections.abc import Iterable, Iterator
from multiprocessing import Process, Queue
from typing import BinaryIO

import regex as re
from rich.logging import RichHandler
from tqdm import tqdm  # type: ignore

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
    special_tokens: list[str],
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    Boundaries are set at the beginning of any special token.
    May return fewer chunks if the boundaries end up overlapping.
    """
    if not special_tokens:
        split_pattern_bytes = b""
    else:
        encoded_tokens = [re.escape(tok.encode("utf-8")) for tok in special_tokens]
        split_pattern_bytes = b"|".join(encoded_tokens)

    split_pattern = re.compile(split_pattern_bytes) if split_pattern_bytes else None

    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    if file_size == 0:
        return [0, 0]

    chunk_size = file_size // desired_num_chunks
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size
    mini_chunk_size = 8192

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)

        if not split_pattern:
            continue

        while True:
            mini_chunk = file.read(mini_chunk_size)
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            match = split_pattern.search(mini_chunk)
            if match:
                chunk_boundaries[bi] = initial_position + match.start()
                break

            initial_position += len(mini_chunk)
            if len(mini_chunk) < mini_chunk_size:
                chunk_boundaries[bi] = file_size
                break

    return sorted(set(chunk_boundaries))


def split_by_special_tokens(text: str, special_tokens: list[str]) -> list[str]:
    """Split text by special tokens."""
    if not special_tokens:
        return [text]

    special_tokens_sorted = sorted(special_tokens, key=lambda x: -len(x))
    pattern = "|".join(re.escape(tok) for tok in special_tokens_sorted)
    parts = re.split(f"({pattern})", text)
    return [p for p in parts if p]


def pretokenize(text: str, special_tokens: list[str]) -> list[bytes]:
    """
    Separates text into pretokens. Special tokens become single, atomic pretokens.
    """
    parts = split_by_special_tokens(text, special_tokens)
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    tokens = []
    for part in parts:
        if part in special_tokens:
            tokens.append(part.encode("utf--8"))
        else:
            str_tokens = re.findall(PAT, part)
            tokens.extend(s.encode("utf-8") for s in str_tokens)
    return tokens


def merge(
    counts: dict[tuple[int, int], int],
    index_dict: dict[tuple[int, int], set[int]],
    pretokens: list[list[int]],
    max_pair: tuple[int, int],
    new_index: int,
):
    """Merge the pairs with highest frequency and update counts, index_dict."""
    index_set = index_dict[max_pair]
    for i in index_set:
        pretoken = pretokens[i]
        new_pretoken = []

        pos_list = []
        pos = 0
        j = 0
        while j < len(pretoken):
            if (j < len(pretoken) - 1) and ((pretoken[j], pretoken[j + 1]) == max_pair):
                new_pretoken.append(new_index)
                pos_list.append(pos)
                j += 2
            else:
                new_pretoken.append(pretoken[j])
                j += 1
            pos += 1

        for pos in pos_list:
            counts[max_pair] -= 1

            if pos > 0:
                left_neighbor = new_pretoken[pos - 1]
                old_left_pair = (left_neighbor, max_pair[0])
                if counts.get(old_left_pair, 0) > 0:
                    counts[old_left_pair] -= 1

                new_left_pair = (left_neighbor, new_index)
                counts[new_left_pair] = counts.get(new_left_pair, 0) + 1
                index_dict[new_left_pair].add(i)

            if pos < len(new_pretoken) - 1:
                right_neighbor = new_pretoken[pos + 1]
                old_right_pair = (max_pair[1], right_neighbor)
                if counts.get(old_right_pair, 0) > 0:
                    counts[old_right_pair] -= 1

                new_right_pair = (new_index, right_neighbor)
                counts[new_right_pair] = counts.get(new_right_pair, 0) + 1
                index_dict[new_right_pair].add(i)

        pretokens[i] = new_pretoken


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    *,
    show_progress: bool = True,
    progress_lib: str = "tqdm",
    log_interval: int = 200,
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, train a BPE tokenizer."""
    special_tokens = special_tokens or []
    num_merges = max(0, vocab_size - len(special_tokens) - 256)

    vocab = {x: bytes([x]) for x in range(256)}
    for i, token in enumerate(special_tokens):
        vocab[256 + i] = token.encode("utf-8")

    if num_merges <= 0:
        log.info(f"No merges to perform; returning base vocabulary.")
        return (vocab, [])

    merges = []
    num_processes = kwargs.get("num_processes", os.cpu_count() or 4)
    log.info(
        f"Starting BPE training: vocab_size={vocab_size}, merges={num_merges}, special_tokens={special_tokens}"
    )

    chunk_list = []
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, special_tokens)
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            if start >= end:
                continue
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            chunk_list.append(chunk)

    def _worker(text: str, special_tokens_list: list[str], q: Queue):
        tokens = pretokenize(text, special_tokens_list)
        q.put(tokens)

    q = Queue()
    processes = [
        Process(target=_worker, args=(chunk, special_tokens, q)) for chunk in chunk_list
    ]
    for p in processes:
        p.start()

    pretokens_list = []
    pbar_desc = "Pretokenizing"
    if show_progress and progress_lib == "tqdm":
        for _ in tqdm(range(len(processes)), desc=pbar_desc, leave=False):
            pretokens_list.extend(q.get())
    else:
        for _ in range(len(processes)):
            pretokens_list.extend(q.get())

    for p in processes:
        p.join()

    byte_to_id = {v: k for k, v in vocab.items()}
    pretokens = []
    for byte_pretoken in pretokens_list:
        if byte_pretoken in byte_to_id:
            pretokens.append([byte_to_id[byte_pretoken]])
        else:
            pretokens.append([b for b in byte_pretoken])

    log.info(f"Pretokenization complete: {len(pretokens)} pretokens.")

    counts = defaultdict(int)
    index_dict = defaultdict(set)
    for j, pretoken in enumerate(pretokens):
        for index1, index2 in zip(pretoken, pretoken[1:]):
            counts[index1, index2] += 1
            index_dict[index1, index2].add(j)

    # --- MODIFIED MERGE LOOP TO PROTECT SPECIAL TOKENS ---
    special_token_ids = {byte_to_id[tok.encode("utf-8")] for tok in special_tokens}
    log.info("Beginning merge loop ...")

    merge_iterable = range(num_merges)
    if show_progress and progress_lib == "tqdm":
        merge_iterable = tqdm(merge_iterable, desc="Merging BPE", leave=True)

    for i in merge_iterable:
        if not counts:
            log.warning("No more pairs to merge. Stopping early.")
            break

        # Find the best pair, EXCLUDING any pairs that involve a special token.
        max_pair = max(
            counts.keys(),
            key=lambda p: (
                counts[p]
                if p[0] not in special_token_ids and p[1] not in special_token_ids
                else -1,
                vocab.get(p[0], b"").decode("utf-8", errors="ignore"),
                vocab.get(p[1], b"").decode("utf-8", errors="ignore"),
            ),
        )

        # If the best pair has a count of -1, it means all remaining pairs involve special tokens.
        if counts.get(max_pair, 0) <= 0 and (
            max_pair[0] in special_token_ids or max_pair[1] in special_token_ids
        ):
            log.warning("Only pairs with special tokens remain. Stopping early.")
            break

        index1, index2 = max_pair
        new_index = 256 + len(special_tokens) + i
        vocab[new_index] = vocab[index1] + vocab[index2]
        merges.append((vocab[index1], vocab[index2]))

        merge(counts, index_dict, pretokens, max_pair, new_index)

        if counts.get(max_pair) == 0:
            del counts[max_pair]

    log.info(
        f"Merge loop finished: total_merges={len(merges)}, final_vocab_size={len(vocab)}"
    )
    return (vocab, merges)


class BPETokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        """Byte Pair Encoding Tokenizer."""
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        self.vocab_reversed = {v: k for k, v in self.vocab.items()}
        # Create a lookup for merges based on integer IDs for faster encoding
        self.merges_map = {
            (self.vocab_reversed[p1], self.vocab_reversed[p2]): self.vocab_reversed[
                p1 + p2
            ]
            for p1, p2 in self.merges
            if p1 in self.vocab_reversed
            and p2 in self.vocab_reversed
            and p1 + p2 in self.vocab_reversed
        }

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | os.PathLike,
        merges_filepath: str | os.PathLike,
        special_tokens: list[str] | None = None,
    ):
        """Create a BPE tokenizer from vocabulary and merges files."""
        raise NotImplementedError

    def encode(self, text: str) -> list[int]:
        """Encode an input text into a sequence of token IDs."""
        byte_pretokens = pretokenize(text, self.special_tokens)

        all_tokens = []
        for byte_word in byte_pretokens:
            # If the pretoken is a special token, just add its ID
            if byte_word in self.vocab_reversed:
                all_tokens.append(self.vocab_reversed[byte_word])
                continue

            # Otherwise, convert to byte IDs and apply merges
            tokens = [b for b in byte_word]
            while len(tokens) >= 2:
                # Find the first possible merge
                pairs = list(zip(tokens, tokens[1:]))
                try:
                    first_merge_idx = min(
                        idx for idx, p in enumerate(pairs) if p in self.merges_map
                    )
                except ValueError:
                    break  # No more merges possible in this word

                # Perform the merge
                p1, p2 = pairs[first_merge_idx]
                new_id = self.merges_map[(p1, p2)]
                tokens = (
                    tokens[:first_merge_idx] + [new_id] + tokens[first_merge_idx + 2 :]
                )
            all_tokens.extend(tokens)

        return all_tokens

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Lazily tokenize an iterable of strings."""
        for line in iterable:
            yield from self.encode(line)

    def decode(self, ids: list[int]) -> str:
        """Decode a sequence of token IDs into text."""
        tokens = b"".join(self.vocab.get(token_id, b"") for token_id in ids)
        return tokens.decode("utf-8", errors="replace")
