from collections.abc import Callable
from typing import Iterable, Optional

import torch


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 0.01,
        eps: float = 1e-8,
    ):
        """The AdamW optimizer.

        Args:
            params (Iterable): The parameters to optimize.
            lr (float, optional): The learning rate. Defaults to 1e-3.
            betas (tuple[float, float], optional): The coefficients used for computing running averages of gradient and its square. Defaults to (0.9, 0.999).
            weight_decay (float, optional): The weight decay (L2 penalty). Defaults to 0.01.
            eps (float, optional): A small constant for numerical stability. Defaults to 1e-8.
        """
        assert lr > 0, "Learning rate must be positive."
        defaults = {"lr": lr}
        super().__init__(params, defaults)
        self.betas = betas
        self.weight_decay = weight_decay
        self.eps = eps

    def step(
        self,
        closure: Optional[Callable] = None,
    ):
        loss = closure() if closure else None
        beta1, beta2 = self.betas

        for group in self.param_groups:
            lr = group["lr"]
            params = group["params"]
            if not params:
                continue
            for p in params:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                # Initialize state
                if len(state) == 0:
                    state["step"] = 1
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)
                m, v = state["m"], state["v"]

                # update first and second moment
                m_new = beta1 * m + (1 - beta1) * grad
                v_new = beta2 * v + (1 - beta2) * grad**2

                # comput adjusted learning rate
                lr_t = (
                    lr
                    * ((1 - beta2 ** state["step"]) ** 0.5)
                    / (1 - beta1 ** state["step"])
                )

                # update the parameters
                p.data -= lr_t * (m_new / (v_new.sqrt() + self.eps))
                # apply weight decay
                p.data -= lr * self.weight_decay * p.data

                # update the state
                state["step"] += 1
                state["m"] = m_new
                state["v"] = v_new
        return loss
