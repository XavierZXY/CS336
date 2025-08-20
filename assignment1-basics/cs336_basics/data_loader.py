from typing import Generator

import numpy as np
import numpy.typing as npt
import torch


def load_dataset(
    file_path: str,
    dtype: np.dtype = np.int64,
    batch_size: int = 4,
    context_length: int = 1024,
    device: str = "cuda",
) -> Generator[tuple[torch.Tensor, torch.Tensor], None, None]:
    """Load a dataset from a file.

    Args:
        file_path (str): The path to the dataset file.
        dtype (np.dtype, optional): The data type of the dataset. Defaults to np.int64.
        batch_size (int, optional): The number of samples per batch. Defaults to 4.
        context_length (int, optional): The length of the context window. Defaults to 1024.
        device (str, optional): The device to load the dataset onto. Defaults to "cuda".
    """
    dataset = np.memmap(file_path, dtype=dtype, mode="r")
    yield from get_batch(
        dataset,
        batch_size=batch_size,
        context_length=context_length,
        device=device,
    )


def get_batch(
    dataset: npt.NDArray,
    batch_size: int = 1,
    context_length: int = 1024,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load data. Sample batch sequences of token data with a length of context_length from a token
    sequence dataset using the sliding window method.

    Args:
        dataset (npt.NDArray): The input dataset.
        batch_size (int, optional): The number of samples per batch. Defaults to 1.
        context_length (int, optional): The length of the context window. Defaults to 1024.
        device (str, optional): Device. Defaults to "cuda".

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing the input and target tensors.
    """
    dataset_len = dataset.shape[0]
    assert dataset_len > context_length, (
        "Dataset length must be greater than context_length."
    )
    # Make sure we don't go out of bounds by checking if start+context_length exceeds dataset_len
    starts = np.random.randint(0, dataset_len - context_length, size=batch_size)
    inputs = np.stack(
        [dataset[start : start + context_length] for start in starts], dtype=np.int64
    )
    targets = np.stack(
        [dataset[start + 1 : start + context_length + 1] for start in starts],
        dtype=np.int64,
    )
    return (
        torch.tensor(inputs, dtype=torch.long, device=device),
        torch.tensor(targets, dtype=torch.long, device=device),
    )
