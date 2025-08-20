import math


class LR_Scheduler:
    def __init__(
        self,
        step,
        max_learning_rate,
        min_learning_rate,
        warmup_steps,
        cosine_steps,
    ):
        """Learning Rate Scheduler

        Args:
            step (int): Current training step
            max_learning_rate (float): Maximum learning rate
            min_learning_rate (float): Minimum learning rate
            warmup_steps (int): Number of warmup steps
            cosine_steps (int): Number of cosine annealing steps
        """
        self.step = step
        self.max_learning_rate = max_learning_rate
        self.min_learning_rate = min_learning_rate
        self.cosine_steps = cosine_steps
        self.warmup_steps = warmup_steps

    def get_lr(self):
        if self.step < self.warmup_steps:
            lr = self.max_learning_rate * (self.step / self.warmup_steps)
        elif self.warmup_steps <= self.step <= self.cosine_steps:
            lr = self.min_learning_rate + 0.5 * (
                self.max_learning_rate - self.min_learning_rate
            ) * (
                1
                + math.cos(
                    (self.step - self.warmup_steps)
                    / (self.cosine_steps - self.warmup_steps)
                    * math.pi
                )
            )
        else:
            lr = self.min_learning_rate

        return lr
