import math
from typing import Optional


class LR_Scheduler:
    def __init__(
        self,
        step: int = 0,
        max_learning_rate: float = 1e-3,
        min_learning_rate: float = 1e-5,
        warmup_steps: int = 0,
        cosine_steps: int = 1000,
    ):
        """Learning Rate Scheduler

        Args:
            step (int): Current training step (default 0)
            max_learning_rate (float): Maximum learning rate (default 1e-3)
            min_learning_rate (float): Minimum learning rate (default 1e-5)
            warmup_steps (int): Number of warmup steps (default 0)
            cosine_steps (int): Number of cosine annealing steps (default 1000)

        Notes:
            - Does not change the original LR computation logic.
            - Raises ValueError on invalid parameter combinations (e.g. cosine_steps < warmup_steps
              or cosine_steps == warmup_steps which would cause division by zero).
        """
        # basic validation
        if step < 0:
            raise ValueError("step must be non-negative")
        if warmup_steps < 0 or cosine_steps < 0:
            raise ValueError("warmup_steps and cosine_steps must be non-negative")
        if cosine_steps < warmup_steps:
            raise ValueError("cosine_steps must be >= warmup_steps")
        if cosine_steps == warmup_steps and cosine_steps != 0:
            # if both zero it's fine (no cosine region), otherwise would cause division by zero
            raise ValueError(
                "cosine_steps must be different from warmup_steps to avoid division by zero"
            )

        self._step = int(step)
        self.max_learning_rate = float(max_learning_rate)
        self.min_learning_rate = float(min_learning_rate)
        self.cosine_steps = int(cosine_steps)
        self.warmup_steps = int(warmup_steps)

    @property
    def step(self) -> int:
        """Current step (read/write)."""
        return self._step

    @step.setter
    def step(self, value: int) -> None:
        if value < 0:
            raise ValueError("step must be non-negative")
        self._step = int(value)

    def advance(self, n: int = 1) -> None:
        """Advance current step by n (default 1)."""
        if n < 0:
            raise ValueError("n must be non-negative")
        self._step += int(n)

    def reset(self, step: int = 0) -> None:
        """Reset step to given value (default 0)."""
        if step < 0:
            raise ValueError("step must be non-negative")
        self._step = int(step)

    def get_lr(self, step: Optional[int] = None) -> float:
        """Get learning rate for `step`. If step is None, use the scheduler's current step.

        The computation logic is unchanged; this method merely allows querying arbitrary steps.
        """
        s = self._step if step is None else int(step)
        # same computation logic as original
        if s < self.warmup_steps:
            lr = (
                self.max_learning_rate * (s / self.warmup_steps)
                if self.warmup_steps > 0
                else self.max_learning_rate
            )
        elif self.warmup_steps <= s <= self.cosine_steps:
            lr = self.min_learning_rate + 0.5 * (
                self.max_learning_rate - self.min_learning_rate
            ) * (
                1
                + math.cos(
                    (s - self.warmup_steps)
                    / (self.cosine_steps - self.warmup_steps)
                    * math.pi
                )
            )
        else:
            lr = self.min_learning_rate

        return lr

    def __call__(self, step: Optional[int] = None) -> float:
        """Alias to get_lr so scheduler(step) returns LR."""
        return self.get_lr(step)

    def __repr__(self) -> str:
        return (
            f"LR_Scheduler(step={self._step}, max_lr={self.max_learning_rate}, "
            f"min_lr={self.min_learning_rate}, warmup={self.warmup_steps}, cosine={self.cosine_steps})"
        )
