import numpy as np
import torch
import torch.nn as nn


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """Initializes a linear layer.

        Args:
            in_features (int): final dimension of the input tensor
            out_features (int): final dimension of the output tensor
            device (torch.device | None, optional): device to store the parameters on. Defaults to None.
            dtype (torch.dtype | None, optional): Data type of parameters. Defaults to None.
        """
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype

        w_init = self.initialize_Weights(out_features, in_features, factory_kwargs)
        self.weight = nn.Parameter(w_init)

    def initialize_Weights(
        self, out_dim: int, in_dim: int, factory_kwargs: dict
    ) -> torch.Tensor:
        """Initializes the weights of the linear layer.

        Args:
            out_dim (int): output dimension of the linear layer
            in_dim (int): input dimension of the linear layer
            factory_kwargs (dict): factory arguments for tensor creation

        Returns:
            torch.Tensor: _description_
        """
        W = torch.empty(out_dim, in_dim, **factory_kwargs)
        W_mean = 0.0
        W_std: float = np.sqrt(2.0 / (in_dim + out_dim))
        nn.init.trunc_normal_(W, W_mean, W_std, -3.0 * W_std, 3.0 * W_std)

        return W

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        output = x @ self.weight.T
        return output
