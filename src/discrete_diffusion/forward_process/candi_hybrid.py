"""Hybrid Noising process for CANDI."""

from __future__ import annotations

import torch
from .utils import _mask_token_id
from .base import ForwardProcess
from ..noise_schedules.base import NoiseSchedule
import torch.nn.functional as F


class HybridForwardCANDI(ForwardProcess):
    """Hybrid noising kernel for CANDI.
    https://arxiv.org/pdf/2510.22510 

    Selects positions with probability `(1 - alpha_t)` to add Gaussian noise, with variance sigma_t.
    Leaves others the same.

    """

    def __init__(
        self, tokenizer, schedule: NoiseSchedule, name: str | None = None
    ) -> None:
        print(schedule)
        
        assert hasattr(schedule, "r_t") and hasattr(schedule, "sigma_t"), (
            "CANDI schedule must implement r_t and sigma_t methods."
        )

        super().__init__(tokenizer=tokenizer, schedule=schedule, name=name)
        self.mask_id = _mask_token_id(tokenizer)

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor, t: torch.Tensor):
        """Applies hybrid noising process to input_ids at time t. Implements Equation 11 of CANDI paper."""
        alpha_t = self.schedule.alpha_t(t).view(-1, 1)
        dalpha_t = self.schedule.alpha_prime_t(t).view(-1, 1)
        sigma_t = self.schedule.sigma_t(t).view(-1, 1, 1)
        p_mask = (1.0 - alpha_t).to(dtype=torch.float32)

        # get noisy positiosn
        move_mask = (torch.rand_like(input_ids, dtype=torch.float32) < p_mask).float()

        # get one-hot representations 
        X_0 = F.one_hot(input_ids, num_classes=len(self.tokenizer)).to(input_ids.device)

        X_t_prime = X_0 + torch.randn_like(X_0, dtype=torch.float32) * sigma_t

        X_t = X_0 * (1 - move_mask.unsqueeze(-1)) + X_t_prime * move_mask.unsqueeze(-1)

        return {"xt": X_t, "reveal_mask": 1-move_mask, "continuous_noise": sigma_t, "discrete_noise": t, "alpha_t": alpha_t, "dalpha_t": dalpha_t}  
