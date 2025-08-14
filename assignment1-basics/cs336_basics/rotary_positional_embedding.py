import numpy as np
import torch
import torch.nn as nn


class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ):
        """Rotary Positional Embedding

        Args:
            theta (float): Rotation angle
            d_k (int): Dimension of the key
            max_seq_len (int): Maximum sequence length
            device (torch.device | None, optional): Device. Defaults to None.
        """
        super().__init__()

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device
        self.rotation_matrix_table = self.generate_rotation_martix(
            theta, d_k, max_seq_len
        )
        self.register_buffer(
            "rotation_matrix", self.rotation_matrix_table, persistent=False
        )

    def generate_rotation_block(
        self, theta: float, block_index: int, seq_pos: int, d_k: int
    ) -> torch.Tensor:
        """Generate a rotation matrix for a specific block and sequence position.
        angle = \frac{seq\_pos}{\theta^{2 \cdot block\_index / d\_k}}
        Args:
            theta (float): Base rotation angle default to 10000.0
            block_index (int): Block index
            seq_pos (int): Sequence position
            d_k (int): Dimension of the key

        Returns:
            torch.Tensor: Rotation matrix
        """
        angle = torch.tensor(
            seq_pos / (theta ** (2 * block_index / d_k)), device=self.device
        )
        cos = torch.cos(angle)
        sin = torch.sin(angle)
        r_matrix = torch.Tensor([[cos, -sin], [sin, cos]]).to(self.device)
        return r_matrix

    def generate_rotation_martix(
        self, theta: float, d_k: int, max_seq_len: int
    ) -> torch.Tensor:
        """Generate a rotation matrix table.

        Args:
            theta (float): Rotation angle
            d_k (int): Dimension of the key
            max_seq_len (int): Maximum sequence length

        Returns:
            torch.Tensor: Rotation matrix table
        """
        rotation_matrix_table = torch.zeros(max_seq_len, d_k, d_k)
        for i in range(max_seq_len):
            blocks = [
                self.generate_rotation_block(theta, j, i, d_k) for j in range(d_k // 2)
            ]
            rotation_matrix_table[i, :, :] = torch.block_diag(*blocks)

        return rotation_matrix_table

    def forward(
        self, x: torch.Tensor, token_positions: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Apply rotary positional embedding.

        Args:
            x (torch.Tensor):[..., seq_len, d_k]. Input tensor of shape (batch_size, seq_len, d_k).
            token_positions (torch.Tensor):[..., seq_len] Token positions of shape (batch_size, seq_len).

        Returns:
            torch.Tensor: Output tensor with rotary positional embedding applied.
        """
        *prefix_dims, seq_len, d_k = x.shape
        if token_positions is None:
            token_positions = torch.arange(seq_len, device=self.device)

        rotation_matrix = self.rotation_matrix_table[
            token_positions
        ]  # (batch_size, seq_len, d_k, d_k)
        x_rotated = rotation_matrix @ x.unsqueeze(-1)
        x_rotated = x_rotated.squeeze(-1)
        return x_rotated
