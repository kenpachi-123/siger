import math
from typing import Optional

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class InverseSquareRootSchedule(_LRScheduler):
    """PyTorch-native variant of EAGER's inverse square-root scheduler."""

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_updates: int,
        warmup_init_lr: float = -1.0,
        last_epoch: int = -1,
    ):
        self.warmup_updates = max(int(warmup_updates), 1)
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.warmup_init_lr = (
            float(warmup_init_lr)
            if warmup_init_lr >= 0
            else min(self.base_lrs)
        )
        self.lr_steps = [
            (base_lr - self.warmup_init_lr) / self.warmup_updates
            for base_lr in self.base_lrs
        ]
        self.decay_factors = [
            base_lr * math.sqrt(self.warmup_updates)
            for base_lr in self.base_lrs
        ]
        for param_group in optimizer.param_groups:
            param_group["lr"] = self.warmup_init_lr
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        step_num = max(self.last_epoch, 0)
        lrs = []
        for base_lr, lr_step, decay_factor in zip(
            self.base_lrs,
            self.lr_steps,
            self.decay_factors,
        ):
            if step_num < self.warmup_updates:
                lr = self.warmup_init_lr + step_num * lr_step
            else:
                lr = decay_factor * (step_num ** -0.5)
            lrs.append(lr if math.isfinite(lr) else base_lr)
        return lrs
