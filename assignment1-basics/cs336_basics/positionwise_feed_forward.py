import torch
import torch.nn as nn

from .linear import Linear


def silu(x: torch.Tensor) -> torch.Tensor:
    """Sigmoid Linear Unit (SiLU) activation function.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Output tensor after applying SiLU activation.
    """
    return x * torch.sigmoid(x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        """Position-wise Feed-Forward Network.

        Args:
            d_model (int): Dimensionality of the input tensor.
            d_ff (int): Dimensionality of the feed-forward layer.
        """
        super().__init__()

        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Implement the SwiGLU feedward network

        Args:
            x (torch.Tensor): Input embeddings to the feed-forward layer

        Returns:
            torch.Tensor: Output embeddings after the feed-forward layer
        """
        output = self.w2(silu(self.w1(x)) * self.w3(x))
        return output
