import numpy as np
import torch
import torch.nn as nn


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """_summary_

        Args:
            num_embeddings (int): Number of unique tokens in the vocabulary.
            embedding_dim (int): Dimension of the embedding vectors.
            device (torch.device | None, optional): Device. Defaults to None.
            dtype (torch.dtype | None, optional): Data type. Defaults to None.
        """
        super().__init__()

        factory_kwargs = {"device": device, "dtype": dtype}
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        w_init = self.initialize_Weights(num_embeddings, embedding_dim, factory_kwargs)
        self.weight = nn.Parameter(w_init)

    def initialize_Weights(
        self,
        vocab_size: int,
        d_model: int,
        factory_kwargs: dict,
    ) -> torch.Tensor:
        """Initializes the weights of the embedding layer.

        Args:
            vocab_size (int): number of unique tokens in the vocabulary
            d_model (int): dimension of the embedding vectors
            factory_kwargs (dict): factory arguments for tensor creation

        Returns:
            torch.Tensor: _description_
        """
        W = torch.empty(vocab_size, d_model, **factory_kwargs)
        W_mean = 0.0
        W_std: float = np.sqrt(1.0 / d_model)
        nn.init.trunc_normal_(W, W_mean, W_std, -3.0 * W_std, 3.0 * W_std)

        return W

    def forward(
        self,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of the embedding layer.

        Args:
            token_ids (torch.Tensor): (batch, seq_len).Indices of tokens to be embedded.

        Returns:
            torch.Tensor: (batch, seq_len, embed_dim).Embedded representations of the input tokens.
        """
        return self.weight[token_ids]
