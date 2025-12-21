"""GStarSampler - extends StarShape with remasker-guided token selection.

GStarSampler uses a trained remasker model to identify which tokens are likely
mistakes, then preferentially remasks those tokens instead of random selection.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .starshape import StarShapeSampler


class GStarSampler(StarShapeSampler):
  """GStar-guided remasking using trained remasker predictions.
  
  Extends StarShapeSampler by overriding _get_mistake_confidences() to use
  the remasker model's predictions instead of random values.
  
  Args:
    config: Hydra config object containing sampling parameters.
    forward_process: Optional forward diffusion process (unused in sampling).
    t_on: Transition point for phase change. Default 0.1 means transition 
          happens at 10% through the diffusion process.
    remasker_schedule: Controls mask ratio behavior after t_on.
          "default": Mask ratio continues decreasing following MDLM schedule.
          "plato": Mask ratio stays fixed at alpha(t_on) level.
  """

  def _get_mistake_confidences(self, model, sampled_x0, t):
    """Use remasker logits to identify likely mistakes.
    
    Calls the model's _remasker_forward() to get binary classification logits,
    then returns the softmax probability of class 1 (mistake) as confidence.
    
    Args:
      model: The GStar diffusion model with _remasker_forward().
      sampled_x0: Sampled x0 from model [batch, seq_len].
      t: Current timestep [batch, 1].
      
    Returns:
      Tensor: Mistake confidence scores [batch, seq_len]. Higher = more likely mistake.
      
    Raises:
      ValueError: If model does not have _remasker_forward method.
    """
    if not hasattr(model, '_remasker_forward'):
      raise ValueError(
        "GStarSampler requires a GStar model with _remasker_forward() method. "
        "Make sure you're using a GStar checkpoint, not an MDLM checkpoint."
      )
    
    # Compute sigma from t
    alpha_t = model.noise.alpha_t(t)
    sigma = model._sigma_from_alphat(alpha_t)
    
    # Get remasker predictions [batch, seq_len, 2]
    with torch.no_grad():
      remasker_logits = model._remasker_forward(sampled_x0, sigma)
    
    # Softmax and take class 1 (mistake) confidence
    confidences = F.softmax(remasker_logits, dim=-1)[:, :, 1]
    
    return confidences  # [batch, seq_len]

