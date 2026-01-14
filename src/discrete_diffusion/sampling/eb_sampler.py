"""Entropy-Bounded (EB) Sampler.

This module implements the *Entropy Bounded Sampler* proposed in:

  Heli Ben-Hamu, Itai Gat, Daniel Severo, Niklas Nolte, Brian Karrer
  "Accelerated Sampling from Masked Diffusion Models via Entropy Bounded Unmasking".

EB-Sampler is designed for **masked diffusion models** (MDMs) where generation
proceeds by iteratively replacing *mask* tokens. Unlike fixed Top-k samplers
that unmask a constant number of tokens per step, EB-Sampler chooses how many
tokens to unmask *adaptively* based on an entropy budget.

Integration notes for UNI-D²:
  * This sampler expects a time-*independent* masked model (i.e.
    ``algo.time_conditioning=False``) that exposes a ``mask_id``.
  * The model is assumed to return per-position logits (unnormalized) over the
    full vocabulary. EB-Sampler internally normalizes with ``log_softmax``.
  * No new dependencies are introduced (only PyTorch).
"""

from __future__ import annotations

from typing import Literal, Optional

import torch
import torch.nn.functional as F

from .. import utils
from ..forward_process.utils import sample_categorical
from .base import Sampler


ErrorProxy = Literal["entropy", "confidence", "margin"]


def _entropy_from_log_probs(log_probs: torch.Tensor) -> torch.Tensor:
  """Compute categorical entropy from log-probabilities.

  Args:
    log_probs: [..., vocab] log-probabilities.

  Returns:
    [...]-shaped entropy in nats.
  """
  probs = log_probs.exp()
  return -(probs * log_probs).sum(dim=-1)


class EBSampler(Sampler):
  """Entropy-Bounded unmasking sampler.

  Parameters (Hydra):
    gamma: Non-negative entropy budget controlling how many tokens can be
      unmasked per forward pass. ``gamma=0`` reduces to 1-token-per-step.
    error_proxy: Which proxy to use for ordering masked tokens.
      - ``entropy``: unmask lowest-entropy tokens first.
      - ``confidence``: unmask highest max-prob tokens first.
      - ``margin``: unmask largest (top1 - top2) margin first.
    temperature: Optional temperature applied to logits before sampling and
      proxy computations.

  Notes:
    EB-Sampler is a *planning* sampler: it performs one model forward pass per
    step and can unmask multiple tokens in parallel. It does not use the
    diffusion-time DDPM predictor used by :class:`~discrete_diffusion.sampling.absorbing.AbsorbingSampler`.
  """

  def __init__(
    self,
    config,
    forward_process=None,
    *,
    gamma: float = 0.0,
    error_proxy: ErrorProxy = "entropy",
    temperature: float = 1.0,
    **kwargs,
  ):
    # `forward_process` is injected by UNI-D²'s sampler factory for all samplers.
    # EB-Sampler does not require it (it samples directly from the model's
    # factorized clean-data conditionals), but we accept and store it to keep the
    # constructor signature compatible with the rest of the codebase.
    self.forward_process = forward_process

    # Accept unknown kwargs for forward compatibility with Hydra configs.
    # (e.g. if someone adds sampler.* knobs in YAML later)
    _ = kwargs
    self.config = config
    self.gamma = float(gamma)
    if self.gamma < 0:
      raise ValueError(f"EB-Sampler requires gamma >= 0, got {self.gamma}")
    self.error_proxy: ErrorProxy = error_proxy
    if self.error_proxy not in {"entropy", "confidence", "margin"}:
      raise ValueError(
        f"Unknown error_proxy={error_proxy!r}. Expected one of: entropy|confidence|margin"
      )
    self.temperature = float(temperature)
    if self.temperature <= 0:
      raise ValueError(
        f"EB-Sampler requires temperature > 0, got {self.temperature}"
      )

  @torch.no_grad()
  def generate(
    self,
    model,
    *,
    num_samples: int,
    num_steps: Optional[int],
    eps: float,
    inject_bos: bool,
  ) -> torch.Tensor:
    """Generate samples.

    Args:
      model: A UNI-D² algorithm module (e.g. MDLM) exposing ``forward``,
        ``prior_sample``, and ``mask_id``.
      num_samples: Number of samples to generate.
      num_steps: Maximum number of unmasking iterations. If ``None``, defaults
        to ``model.num_tokens``.
      eps: Unused (kept for interface compatibility with other samplers).
      inject_bos: Whether to inject the tokenizer BOS token at position 0.

    Returns:
      [num_samples, seq_len] generated token sequences.
    """
    del eps  # not used by this sampler

    # EB-Sampler relies on clean-data conditionals p(x^l | x^{\bar M}).
    # In UNI-D² this is best matched by time-independent masked models.
    if getattr(model, "time_conditioning", False):
      raise ValueError(
        "EBSampler expects algo.time_conditioning=False (time-independent masked model)."
      )

    device = model.device
    seq_len = int(model.num_tokens)
    mask_id = getattr(model, "mask_id", None)
    if mask_id is None:
      raise ValueError("Model must expose `mask_id` for EB sampling")
    mask_id = int(mask_id)

    # Initialise from the prior (typically all masks).
    x = model.prior_sample(num_samples, seq_len).to(device)
    if inject_bos and getattr(model.tokenizer, "bos_token_id", None) is not None:
      x[:, 0] = int(model.tokenizer.bos_token_id)

    # Maximum number of unmasking iterations.
    max_steps = int(num_steps) if num_steps is not None else seq_len
    max_steps = max(max_steps, 1)

    # Sigma is ignored when `time_conditioning=False` but we keep the call
    # signature compatible with UNI-D² backbones.
    sigma = torch.zeros((num_samples, 1), device=device)

    for _ in range(max_steps):
      masked = x.eq(mask_id)
      if not bool(masked.any()):
        break

      logits = model.forward(x, sigma)
      if getattr(self.config.sampling, "use_float64", False):
        logits = logits.to(torch.float64)

      # Optional nucleus filtering for consistency with other samplers.
      top_p = float(getattr(self.config.sampling, "p_nucleus", 1.0))
      if top_p < 1.0:
        logits = utils.top_k_top_p_filtering(logits, top_p=top_p)

      # Prevent sampling the mask token as an output.
      logits = logits.clone()
      neg_inf = getattr(model, "neg_infinity", -float("inf"))
      logits[:, :, mask_id] = neg_inf

      # Temperature + normalized log-probs.
      log_probs = F.log_softmax(logits / self.temperature, dim=-1)
      entropy = _entropy_from_log_probs(log_probs)  # [B, L]

      # Error proxy for ordering masked tokens.
      if self.error_proxy == "entropy":
        err = entropy
      else:
        probs = log_probs.exp()
        if self.error_proxy == "confidence":
          # Higher max-prob means lower error.
          err = -probs.max(dim=-1).values
        elif self.error_proxy == "margin":
          top2 = probs.topk(k=2, dim=-1).values
          margin = top2[..., 0] - top2[..., 1]
          err = -margin
        else:  # pragma: no cover
          raise RuntimeError("Unhandled error_proxy")

      # Only plan over currently masked positions.
      err = err.masked_fill(~masked, float("inf"))

      # Sort tokens by increasing error (best first).
      ids_sorted = err.argsort(dim=-1)  # [B, L]
      entropy_sorted = entropy.gather(1, ids_sorted)

      # Entropy budget rule: find the largest prefix U s.t.
      #   sum(H_U) - max(H_U) <= gamma
      acc_entropy = entropy_sorted.cumsum(dim=-1)
      cummax_entropy = entropy_sorted.cummax(dim=-1).values
      cond = (acc_entropy - cummax_entropy) <= self.gamma

      k = cond.sum(dim=-1)  # [B]
      num_masked = masked.sum(dim=-1)
      k = torch.minimum(k, num_masked)
      # Ensure progress whenever there is at least one mask.
      k = torch.where(num_masked > 0, torch.clamp(k, min=1), k)

      # Turn per-sample k into a boolean mask of positions to unmask.
      # rank[pos] = rank of pos in sorted order.
      L = x.shape[1]
      rank_vals = torch.arange(L, device=device).unsqueeze(0).expand(num_samples, L)
      ranks = torch.empty_like(ids_sorted)
      ranks.scatter_(1, ids_sorted, rank_vals)
      to_unmask = (ranks < k.unsqueeze(1)) & masked

      # Sample token values (independently) for selected positions.
      probs = log_probs.exp()
      sampled = sample_categorical(probs)
      x = torch.where(to_unmask, sampled, x)

    return x


__all__ = ["EBSampler"]
