"""StarShape sampler - extends MDLM with two-phase masking strategy.

StarShape uses a hyperparameter t_on to control a transition between two phases:
  - Phase 1 (t > t_on): Standard MDLM denoising (irreversible, masks only masked tokens)
  - Phase 2 (t <= t_on): Random masking from sampled x0 (can re-mask any token)

Two remasking modes are supported:
  - "default": Mask ratio decreases following MDLM schedule
  - "plato": Mask ratio stays fixed at the level at t_on (plateau)
"""

from __future__ import annotations

from typing import Literal

import torch

from ..forward_process.utils import sample_categorical
from .absorbing import AbsorbingSampler


class StarShapeSampler(AbsorbingSampler):
  """StarShape sampler with two-phase masking controlled by t_on.
  
  Extends AbsorbingSampler (MDLM) with a phase transition:
    - Phase 1 (t > t_on): Use MDLM masking strategy
    - Phase 2 (t <= t_on): Random masking from sampled x0
  
  Args:
    config: Hydra config object containing sampling parameters.
    forward_process: Optional forward diffusion process (unused in sampling).
    t_on: Transition point for phase change. Default 0.1 means transition 
          happens at 10% through the diffusion process.
    remasker_schedule: Controls mask ratio behavior after t_on.
          "default": Mask ratio continues decreasing following MDLM schedule.
          "plato": Mask ratio stays fixed at alpha(t_on) level.
  """

  def __init__(self, config, forward_process=None, t_on=0.1, 
               remasker_schedule: Literal["default", "plato"] = "default"):
    super().__init__(config, forward_process)
    self.t_on = t_on
    self.remasker_schedule = remasker_schedule

  def _get_mistake_confidences(self, model, sampled_x0, t):
    """Return confidence scores indicating likelihood each token is a mistake.
    
    Higher values indicate higher confidence that the token should be remasked.
    This base implementation returns random values for random remasking.
    
    Args:
      model: The diffusion model.
      sampled_x0: Sampled x0 from model [batch, seq_len].
      t: Current timestep [batch, 1].
      
    Returns:
      Tensor: Confidence scores [batch, seq_len]. Higher = more likely mistake.
    """
    return torch.randn(sampled_x0.shape, device=sampled_x0.device)

  def _mask_tokens_starshape(self, model, x, sampled_x0, alpha_s, t, noise_removal_step):
    """Apply StarShape masking: guided masking from sampled x0.
    
    This implements the StarShape Phase 2 strategy:
    - Start with fully denoised sampled_x0
    - Select floor(num_tokens * (1 - alpha)) positions with highest mistake confidence
    - Keep remaining positions as sampled_x0 values
    - Special case: if noise_removal_step=True, mask 0 tokens (return full x0)
    
    Args:
      model: The diffusion model.
      x: Current sequence [batch, length] (not used in StarShape masking).
      sampled_x0: Sampled x0 from model [batch, length].
      alpha_s: Noise level at time s = t - dt [batch, 1].
      t: Current timestep [batch, 1].
      noise_removal_step: If True, return sampled_x0 with 0 masks.
      
    Returns:
      out: Partially masked sequence [batch, length].
    """
    if noise_removal_step:
      # Final step: no masking
      return sampled_x0
    
    batch_size, seq_length = sampled_x0.shape
    
    # Calculate number of tokens to mask based on remasking mode
    if self.remasker_schedule == "plato":
      # Use alpha(t_on) for fixed mask ratio
      alpha_for_mask = model.noise.alpha_t(
        torch.tensor([[self.t_on]], device=t.device, dtype=t.dtype)
      )
      num_tokens_to_mask = int(torch.floor(seq_length * (1 - alpha_for_mask[0, 0])).item())
    else:
      # Default: use alpha_s (continuing MDLM schedule)
      num_tokens_to_mask = int(torch.floor(seq_length * (1 - alpha_s[0, 0])).item())
    
    if num_tokens_to_mask == 0:
      return sampled_x0
    
    # Get mistake confidences for each position
    confidences = self._get_mistake_confidences(model, sampled_x0, t)
    
    # Get indices of top-k positions with highest confidence (these will be masked)
    _, mask_positions = torch.topk(confidences, k=num_tokens_to_mask, dim=1)
    
    # Create batch indices for advanced indexing
    batch_indices = torch.arange(batch_size, device=sampled_x0.device).unsqueeze(1).expand_as(mask_positions)
    
    # Apply masking using advanced indexing
    sampled_x0[batch_indices, mask_positions] = model.mask_id
    return sampled_x0

  def compute_posterior(self, model, x, t, dt, p_x0=None,
                        noise_removal_step=False):
    """Compute posterior for StarShape sampling.
    
    Implements two-phase masking:
      - If t > t_on: Use MDLM masking (Phase 1)
      - If t <= t_on: Use StarShape guided masking (Phase 2)
    
    Args:
      model: The diffusion model.
      x: Current sequence [batch, length].
      t: Current timestep [batch, 1].
      dt: Timestep increment.
      p_x0: Optional cached probability distribution over x0.
      noise_removal_step: If True, this is the final denoising step.
      
    Returns:
      p_x0: Probability distribution over x0.
      out: Next sequence after applying masking strategy.
    """
    alpha_t = model.noise.alpha_t(t)
    if noise_removal_step:
      alpha_s = torch.ones_like(alpha_t)
    else:
      alpha_s = model.noise.alpha_t(t - dt)
    assert alpha_t.ndim == 2
    
    # Sample x0 (shared between both phases)
    p_x0, sampled_x0 = self._sample_x0(model, x, t, p_x0)
    
    # Determine phase based on t
    # t is a tensor [batch, 1], compare with scalar t_on
    # All batch elements share same t, so use torch.all to avoid GPU->CPU transfer
    if torch.all(t > self.t_on):
      # Phase 1: Use MDLM masking (irreversible denoising)
      out = self._mask_tokens_mdlm(model, x, sampled_x0, alpha_t, alpha_s)
    else:
      # Phase 2: Use StarShape guided masking
      out = self._mask_tokens_starshape(model, x, sampled_x0, alpha_s, t, noise_removal_step)
    
    return p_x0, out
