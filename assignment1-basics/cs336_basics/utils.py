import logging
import os
from collections import OrderedDict
from typing import IO, Any, BinaryIO, Dict

import safetensors.torch as safetensors
import torch
from rich.logging import RichHandler

__all__ = ["setup_logging", "save_checkpoint", "load_checkpoint"]


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )


log = logging.getLogger("rich")


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Given a tensor of inputs, return the output of softmaxing the given `dim` of the input.

    Args:
        x (torch.Tensor): Input features to softmax.
        dim (int): Dimension to softmax over.

    Returns:
        torch.Tensor: Softmax output.
    """
    x_max = torch.max(x, dim, keepdim=True).values
    x_stable = x - x_max
    x_exp = torch.exp(x_stable)
    output = x_exp / torch.sum(x_exp, dim=dim, keepdim=True)
    return output


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """Save model checkpoint.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer to save.
        iteration (int): The current iteration number.
        out (str | os.PathLike | BinaryIO | IO[bytes]): The output path or file-like object.
    """
    checkpoint = {
        "iteration": iteration,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    try:
        torch.save(checkpoint, out)
        log.info(f"Save checkpoint to  {out}, The iteraion is: {iteration}")
    except Exception as e:
        log.error(f"Failed to Save: {str(e)}")
        raise


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """Load model checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): The source path or file-like object.
        model (torch.nn.Module): The model to load.
        optimizer (torch.optim.Optimizer): The optimizer to load.

    Returns:
        int: The previously serialized number of iterations.
    """
    try:
        checkpoint = torch.load(src)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        iteration = checkpoint["iteration"]

        log.info(f"Successfully loaded checkpoint from {src}, iteration: {iteration}")
        return iteration

    except Exception as e:
        log.error(f"Failed to load checkpoint: {str(e)}")
        raise
