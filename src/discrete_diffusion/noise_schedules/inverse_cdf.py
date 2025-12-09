"""Continuous Noise schedule following inverse cdf from https://patrickpynadath1.github.io/candi-lander/candi.pdf."""

import torch

from .base import NoiseSchedule


class InverseCDF(NoiseSchedule):
    """Uses inverse Gaussian cdf to calibrate Gaussian noise schedule. sigma_t = - 1 / (phi_inverse(r*(t)) sqrt(2))"""

    def __init__(
        self, eps: float = 1e-3, r_max: float = 0.45, r_min: float = 0.05, **kwargs
    ):
        super().__init__()
        self.eps = float(eps)
        self.r_max = float(r_max)
        self.r_min = float(r_min)

    def r_t(self, t: torch.Tensor) -> torch.Tensor:
        r_star = t * (self.r_max - self.r_min) + self.r_min
        return r_star
    
    def sigma_t(self, t: torch.Tensor) -> torch.Tensor:
        r_star = self.r_t(t)
        # Compute Φ^(-1)(r*(t)) - inverse CDF of standard normal
        # Using the normal distribution's inverse CDF (quantile function)
        # equation 10 of CANDI
        phi_inv = torch.distributions.Normal(0, 1).icdf(r_star)
        
        # Compute σ(t) = -1 / (Φ^(-1)(r*(t)) * √2)
        sigma_t = -1.0 / (phi_inv * torch.sqrt(torch.tensor(2.0)))
        return sigma_t

    def alpha_t(self, t: torch.Tensor) -> torch.Tensor:
        scaled_t = (1 - self.eps) * t
        return 1 - scaled_t

    def alpha_prime_t(self, t: torch.Tensor) -> torch.Tensor:
        return -(1 - self.eps) * torch.ones_like(t)

__all__ = ["InverseCDF"]
