import math

import torch
import torch.nn as nn
from einops import einsum, rearrange

from .linear import Linear
from .rotary_positional_embedding import RotaryPositionalEmbedding
from .utils import softmax


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Given key (K), query (Q), and value (V) tensors, return
    the output of scaled dot product attention.

    Args:
        Q (torch.Tensor): [..., queries, d_k].Query tensor.
        K (torch.Tensor): [..., keys, d_k].Key tensor.
        V (torch.Tensor): [..., values, d_v].Values tensor.
        mask (torch.Tensor | None, optional):[..., queries, d_k]. Attention mask.

    Returns:
        torch.Tensor: Output tensor.
    """
    d_k = K.shape[-1]
    attention_score = einsum(
        Q, K, "... queries d_k, ... keys d_k -> ... queries keys"
    ) / math.sqrt(d_k)

    if mask is not None:
        attention_score = attention_score.masked_fill(mask == 0, float("-inf"))
    attention_weight = softmax(attention_score, dim=-1)
    output = einsum(
        attention_weight, V, "... queries keys, ... keys d_v -> ... queries d_v"
    )

    return output


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        use_rope: bool = False,
        max_seq_len: int | None = None,
        theta: float | None = None,
        token_positions: torch.Tensor | None = None,
    ):
        """Multi-head self-attention module.

        Args:
            d_model (int): Dimensionality of the feedforward input and output.
            num_heads (int): Number of heads to use in multi-headed attention.
            use_rope (bool, optional): Whether to use rotary positional encoding. Defaults to False.
            max_seq_len (int | None, optional): Maximum sequence length for attention. Defaults to None.
            theta (float | None, optional): Rotary positional encoding parameter. Defaults to None.
            token_positions (torch.Tensor | None, optional): Token positions for rotary positional encoding. Defaults to None.
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.use_rope = use_rope
        self.rope = (
            RotaryPositionalEmbedding(theta, d_model // num_heads, max_seq_len)
            if use_rope
            else None
        )
        self.token_positions = token_positions
        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.o_proj = Linear(d_model, d_model)

    def forward(
        self,
        in_features: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for multi-head self-attention.

        Args:
            in_features (torch.Tensor): Input features of shape (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: Output features of shape (batch_size, seq_len, d_model).
        """

        seq_len = in_features.shape[-2]
        qkv_proj = torch.cat(
            [self.q_proj.weight, self.k_proj.weight, self.v_proj.weight]
        )
        qkv = in_features @ qkv_proj.T
        q, k, v = qkv.chunk(3, dim=-1)
        # Rearrange for multi-head attention
        q = rearrange(
            q, "... seq_len (h d_head) -> ... h seq_len d_head", h=self.num_heads
        )
        k = rearrange(
            k, "... seq_len (h d_head) -> ... h seq_len d_head", h=self.num_heads
        )
        v = rearrange(
            v, "... seq_len (h d_head) -> ... h seq_len d_head", h=self.num_heads
        )

        if self.use_rope:
            q = self.rope(q, self.token_positions)
            k = self.rope(k, self.token_positions)

        casual_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        casual_mask = casual_mask[None, None, :, :]
        output = scaled_dot_product_attention(q, k, v, ~casual_mask)
        output = rearrange(output, "... h seq_len d_head ->  ... seq_len (h d_head)")
        return self.o_proj(output)
