import torch
import torch.nn as nn

from .attention import MultiHeadSelfAttention, softmax
from .embedding import Embedding
from .linear import Linear
from .positionwise_feed_forward import PositionwiseFeedForward
from .rmsnorm import RMSNorm


class Transformer(nn.Module):
    def __init__(
        self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float
    ):
        """Transformer model.

        Args:
            d_model (int): Dimensionality of the transformer block inputs.
            num_heads (int): Number of heads to use in multi-head self-attention.
            d_ff (int): Dimensionality of the position-wise feed-forward inner layer.
            max_seq_len (int): Maximum sequence length to pre-cache.
            theta (float): RoPe parameter.
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rms_norm1 = RMSNorm(d_model=d_model)
        self.rms_norm2 = RMSNorm(d_model=d_model)
        self.attn = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            use_rope=True,
            max_seq_len=max_seq_len,
            theta=theta,
        )
        self.ff = PositionwiseFeedForward(d_model, d_ff)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Forward methods

        Args:
            x (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """
        y = x + self.attn(self.rms_norm1(x))

        output = y + self.ff(self.rms_norm2(y))
        return output


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
    ):
        """The LM model config.

        Args:
            vocab_size (int): Vocabulary size.
            context_length (int): Context length.
            d_model (int): Dimensionality of the model.
            num_layers (int): Number of layers.
            num_heads (int): Number of attention heads.
            d_ff (int): Dimensionality of the feed-forward layer.
            rope_theta (float): RoPE theta parameter.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers

        self.token_embeddings = Embedding(
            num_embeddings=vocab_size, embedding_dim=d_model
        )
        self.layers = nn.ModuleList(
            Transformer(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                max_seq_len=context_length,
                theta=rope_theta,
            )
            for _ in range(num_layers)
        )
        self.rms_norm = RMSNorm(d_model=d_model)
        self.output_embeddings = Linear(in_features=d_model, out_features=vocab_size)

    def forward(
        self,
        in_indices: torch.Tensor,
    ) -> torch.Tensor:
        """

        Args:
            in_indices (torch.Tensor): (Int[Tensor, "batch_size sequence_length"]) Tensor with input indices to run the language model on.
            Shape is (batch_size, sequence_length), where `sequence_length` is at most `context_length`.

        Returns:
            torch.Tensor: (Int[Tensor, "batch_size sequence_length"]) Tensor with input indices to run the language model on.
            Shape is (batch_size, sequence_length), where `sequence_length` is at most `context_length`.
        """
        x = self.token_embeddings(in_indices)
        for layer in self.layers:
            x = layer(x)

        x_norm = self.rms_norm(x)
        output_embed = self.output_embeddings(x_norm)

        return output_embed
