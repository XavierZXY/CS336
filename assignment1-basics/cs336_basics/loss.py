import numpy as np
import torch


def cross_entropy_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Given a tensor of inputs and target, compute the average cross-entropy loss.

    Args:
        inputs (torch.Tensor): [batch_size, vocab_size].inputs[i][j] is the unnormalized logit of jth class for the ith example.
        targets (torch.Tensor): [batch_size].

    Returns:
        torch.Tensor:  The average cross-entropy loss across examples.
    """

    target_logits = inputs.gather(dim=-1, index=targets.unsqueeze(-1))

    # log-sum-exp trick for numercial stability by subtracting the larget element
    log_sum_exp = torch.logsumexp(inputs, dim=-1, keepdim=True)

    # cancel out log and exp after softmax when calculating loss
    loss_matrix = -target_logits + log_sum_exp

    # average loss
    loss = loss_matrix.mean()
    return loss
