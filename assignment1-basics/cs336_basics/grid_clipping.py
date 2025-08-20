from typing import Iterable

import torch


def clip_grad_norm(
    parameters: Iterable[torch.nn.Parameter],
    max_l2_norm: float,
    eps: float = 1e-6,
):
    """Clip gradient norm of parameters.

    Args:
        parameters (Iterable[torch.nn.Parameter]): The model parameters to clip.
        max_l2_norm (float): The maximum L2 norm value.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-6.
    """
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return
    total_norm = torch.norm(torch.stack([g.detach().flatten() for g in grads]), p=2)
    if total_norm < max_l2_norm:
        return
    scale = max_l2_norm / (total_norm + eps)
    for p in parameters:
        if p.grad is not None:
            p.grad.data.mul_(scale)
