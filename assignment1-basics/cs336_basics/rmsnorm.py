import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """RMSNorm layer.
        RMSNorm(X) = (X / sqrt(mean(X^2) + eps)) @ W
        Args:
            d_model (int): Dimensionality of the input tensor.
            eps (float, optional): Small value to avoid division by zero. Defaults to 1e-5.
            device (torch.device | None, optional): Device to create the layer on. Defaults to None.
            dtype (torch.dtype | None, optional): Data type of the layer's parameters. Defaults to None.
        """
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        self.d_model = d_model
        self.eps = eps

        w_init = self.initialize_Weights(d_model, factory_kwargs)
        self.weight = nn.Parameter(w_init)

    def initialize_Weights(self, d_model: int, factory_kwargs: dict):
        """Initialize the weights w using truncated normal method"""
        w = torch.ones(d_model, **factory_kwargs)
        return w

    def RMS(
        self,
        x: torch.Tensor,
        d_model: int,
        eps: float,
    ) -> torch.Tensor:
        """Applies RMS normalization to the input tensor.
        rms(x) = x / sqrt(mean(x^2) + eps)

        Args:
            x (torch.Tensor): Input tensor.
            d_model (int): Dimensionality of the input tensor.
            eps (float): Small value to avoid division by zero.

        Returns:
            torch.Tensor: RMS normalized tensor.
        """
        x = x.to(torch.float32)
        rms_x = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + eps)
        return x / rms_x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms_x = self.RMS(x, self.d_model, self.eps)
        output = rms_x * self.weight
        return output.to(dtype=x.dtype, device=x.device)
