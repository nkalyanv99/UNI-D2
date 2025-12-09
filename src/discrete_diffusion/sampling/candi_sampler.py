"""Hybrid sampler for CANDI."""

from __future__ import annotations

import torch

from ..forward_process.utils import sample_categorical
from .base import Sampler


class CANDI_Sampler(Sampler):
    """Base inference method that implements helper functions needed for CANDI"""

    def __init__(self, config, forward_process=None, **kwargs):
        self.config = config
        self.forward_process = forward_process
        self.num_steps = config.sampling.steps


    def _continuous_step(
        self,
        model,
        x: torch.Tensor,
        time_t: torch.Tensor,
        time_s: torch.Tensor,
        sigma_s: torch.Tensor,
        sigma_t: torch.Tensor,
        embedding_cache: torch.Tensor,
        reveal_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        dt = sigma_s - sigma_t
        time_t_vec = torch.ones(x.shape[0], device=x.device) * time_t.item()
        sigma_t_vec = torch.ones(x.shape[0], device=x.device) * sigma_t.item()
        if reveal_mask is None:
            reveal_mask = torch.zeros(x.shape[:-1], device=x.device)
        cond_denoised = model.forward(
            xt=x,
            discrete_noise=time_t_vec,
            reveal_mask=reveal_mask,
            continuous_noise=sigma_t_vec[:, None, None],
            embedding=embedding_cache,
        ).double()

        denoised = cond_denoised.exp()
        x0_hat = sample_categorical(denoised)
        embedding_hat = model.backbone.get_embedding(x0_hat)
        d = (embedding_cache - embedding_hat) / (sigma_t**2)
        new_embedding_cache = embedding_cache - dt * d
        return new_embedding_cache, x0_hat

    def _discrete_step(
        self, x0_hat, xt, t, dt, prev_clean_mask, noise_removal_step=False
    ):

        if noise_removal_step:
            s = 0
        else:
            s = t - dt

        # unmasking correspends to 1-alpha(s) / 1-alpha(t)
        # this is just s / t under a log linear schedule
        unmask = (
            torch.rand(prev_clean_mask.shape, device=prev_clean_mask.device)
            < (t - s) / t
        )
        xt[~prev_clean_mask] = x0_hat[~prev_clean_mask]
        new_clean_mask = prev_clean_mask | unmask
        return xt, new_clean_mask

    @torch.no_grad()
    def generate(self, model, *, num_samples, num_steps, eps, inject_bos):
        if num_steps is None:
            num_steps = self.config.sampling.steps

        x = model.prior_sample(num_samples, model.num_tokens)
        embedding_cache = model.backbone.get_embedding(x)
        timesteps = torch.linspace(0.999, eps, num_steps + 1, device=model.device)
        continuous_noise = model.noise.sigma_t(timesteps)
        clean_mask = torch.zeros(
            (num_samples, model.num_tokens), device=x.device, dtype=torch.bool
        )

        timesteps = torch.linspace(0.999, eps, num_steps + 1, device=model.device)
        dt = (1 - eps) / (num_steps)

        self.max_sigma = continuous_noise.max().item()
        x = x.argmax(dim=-1)
        for i in range(num_steps):
            t = timesteps[i]
            s = timesteps[i + 1]

            sigma_s = continuous_noise[i]
            sigma_t = continuous_noise[i + 1]
            embedding_cache, x0_hat = self._continuous_step(
                model=model, 
                x=x,
                time_t=t,
                time_s=s,
                sigma_s=sigma_s,
                sigma_t=sigma_t,
                reveal_mask=clean_mask.float(),
                embedding_cache=embedding_cache,
            )
            x, clean_mask = self._discrete_step(
                x0_hat, x, t, dt, prev_clean_mask=clean_mask, noise_removal_step=False
            )
        return x
    
    
__all__ = ["CANDI_Sampler"]