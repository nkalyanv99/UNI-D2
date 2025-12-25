"""MDLM algorithm implementation extracted from :mod:`algorithms.algo`."""

import torch
import torch.nn.functional as F

from . import base as trainer_base
from ..utils.utils import liger_cross_entropy


class MDLM(trainer_base.AbsorbingState):
  """Masked Diffusion Language Model (MDLM) algorithm.

  Implements the MDLM objective where tokens are masked and reconstructed.
  """
  def __init__(self, config, tokenizer):
    super().__init__(config, tokenizer)
    self.shift_loss_targets = getattr(config.algo, 'shift_loss_targets', False)
    super()._validate_configuration()

  def _process_model_output(self, model_output, xt, sigma):
    index = torch.full((xt.shape[0], xt.shape[1], 1), self.mask_id, device=xt.device)
    model_output = torch.scatter(model_output, -1, index, self.neg_infinity)
    model_output = torch.where((xt != self.mask_id)[..., None], self.neg_infinity, model_output)
    model_output = torch.scatter(model_output, -1, xt[..., None], 0.0)
    return model_output

  def nll_per_token(self, log_x_theta, xt, x0, alpha_t, dalpha_t, low_var=False):
    ce_loss = liger_cross_entropy(
        log_x_theta.flatten(0, 1),
        x0.flatten(0, 1),
        reduction='none'
    ).view_as(x0)
    loss_coefficient = -1 if low_var else dalpha_t / (1 - alpha_t)
    return -loss_coefficient * ce_loss

  def _get_score(self, x, sigma, group_idxs=None):
    model_output = self.forward(x, sigma, group_idxs)
    # score(x, t) = p_t(y) / p_t(x) => log score(x, t) = log p_t(y) - log p_t(x)

    log_k = -torch.log(torch.expm1(sigma)).squeeze(-1)
    assert log_k.ndim == 1

    # Compute scores for masked positions
    masked_score = model_output + log_k[:, None, None]
    masked_score[:, :, self.mask_id] = 0

    unmasked_score = self.neg_infinity * torch.ones_like(model_output)
    unmasked_score = torch.scatter(
      unmasked_score,
      -1,
      x[..., None],
      torch.zeros_like(unmasked_score[..., :1]))
    
    unmasked_score[:, :, self.mask_id] = -(log_k[:, None] * torch.ones_like(x))
    
    masked_indices = (x == self.mask_id).to(model_output.dtype)[:, :, None]
    model_output = (masked_score * masked_indices + unmasked_score * (1 - masked_indices))
    return model_output.exp()
