"""Block Diffusion algorithm adapted from the BD3-LMs reference implementation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..evaluations import BD3Metrics
from .base import AbsorbingState


# ---------------------------------------------------------------------------
# Loss container
# ---------------------------------------------------------------------------
@dataclass
class Loss:
  loss: torch.FloatTensor
  nlls: torch.FloatTensor
  token_mask: torch.FloatTensor


class BD3LM(AbsorbingState):
  """Block-diffusion trainer mirroring BD3-LMs behaviour."""

  def __init__(self, config, tokenizer):
    # BD3LM always uses 'subs' parameterization
    from omegaconf import OmegaConf
    OmegaConf.set_struct(config.algo, False)
    config.algo.parameterization = 'subs'
    OmegaConf.set_struct(config.algo, True)
    super().__init__(config, tokenizer)
    self.cross_attn = getattr(self.config.algo, 'cross_attn', False)
    self.mdlm_loss_scale = getattr(self.config.algo, 'mdlm_loss_scale', False)
    self.block_size = getattr(config, 'block_size', self.config.model.length)
    self.var_min = getattr(self.config.algo, 'var_min', False)
    
    # Validate noise schedule type (self.noise is set by TrainerBase)
    from ..noise_schedules import LogLinear
    if not isinstance(self.noise, LogLinear):
      raise ValueError(
        f'BD3LM requires LogLinear noise schedule, got {type(self.noise).__name__}'
      )
    
    # Compute sigma bounds once (for _sigma_from_p method)
    self.sigma_max = -torch.log(self.noise.alpha_t(torch.tensor(1.0)))
    self.sigma_min = torch.tensor(self.noise.eps, dtype=torch.float32)

    if self.var_min:
      self.register_buffer('sampling_eps_min', torch.tensor(
        self.config.training.sampling_eps_min, dtype=torch.float32))
      self.register_buffer('sampling_eps_max', torch.tensor(
        self.config.training.sampling_eps_max, dtype=torch.float32))

    self.time_conditioning = getattr(self.config.algo, 'time_conditioning', False)
    self.fast_forward_epochs = None
    self.fast_forward_batches = None

  # -------------------------------------------------------------------------
  # Noise schedule helper methods
  # -------------------------------------------------------------------------
  def _total_noise(self, t):
    """Compute sigma(t) = -log(alpha(t))"""
    return -torch.log(self.noise.alpha_t(t))
  
  def _rate_noise(self, t):
    """Compute dsigma/dt = -alpha'(t) / alpha(t)"""
    alpha = self.noise.alpha_t(t)
    alpha_prime = self.noise.alpha_prime_t(t)
    return -alpha_prime / alpha
  
  def _compute_loss_scaling_and_move_chance(self, t):
    """BD3LM loss scaling: -1/t, and move probability: t"""
    return -1 / t, t

  # -------------------------------------------------------------------------
  # Lightning hooks
  # -------------------------------------------------------------------------
  def _initialize_metrics(self):
    """Override to use BD3LM-specific metrics."""
    self.metrics = BD3Metrics(self.config)

  def to(self, *args, **kwargs):
    self = super().to(*args, **kwargs)
    if hasattr(self.backbone, 'block_diff_mask'):
      self.backbone.block_diff_mask = self.backbone.block_diff_mask.to(self.device)
    return self

  def on_train_epoch_start(self):
    super().on_train_epoch_start()
    self._train_mode()

  def training_step(self, batch, batch_idx):
    del batch_idx
    losses = self._loss(batch['input_ids'], batch['attention_mask'])
    self.metrics.train_nlls.update(losses.nlls, losses.token_mask)
    self.log(name='trainer/loss',
             value=losses.loss.item(),
             on_step=True,
             on_epoch=False,
             sync_dist=True,
             prog_bar=True)
    return losses.loss

  def on_validation_epoch_start(self):
    super().on_validation_epoch_start()
    if self.var_min:
      self.sampling_eps = self.config.training.sampling_eps

  def validation_step(self, batch, batch_idx):
    del batch_idx
      
    if self.var_min:
      valid_loss = None
      for noise_clip_start in self.metrics.valid_vars.keys():
        sampling_eps_min, sampling_eps_max = noise_clip_start
        losses_clip = self._loss(
          batch['input_ids'],
          batch['attention_mask'],
          sampling_eps_min=sampling_eps_min,
          sampling_eps_max=sampling_eps_max)
        if self._check_val_sampling_intvl(sampling_eps_min, sampling_eps_max):
          valid_loss = losses_clip
        if len(self.metrics.valid_vars[noise_clip_start]) < 100:
          nlls = losses_clip.nlls
          per_block = nlls.reshape(nlls.shape[0], -1, self.block_size).mean(-1)
          self.metrics.valid_vars[noise_clip_start].append(per_block)
      if valid_loss is not None:
        self.metrics.valid_nlls.update(valid_loss.nlls, valid_loss.token_mask)
      return valid_loss.loss if valid_loss is not None else losses_clip.loss
    else:
      losses = self._loss(
        batch['input_ids'],
        batch['attention_mask'],
        sampling_eps_min=1e-3 if self.block_size > 1 else 1,
        sampling_eps_max=1 if self.block_size > 1 else 1)
      self.metrics.valid_nlls.update(losses.nlls, losses.token_mask)
      return losses.loss

  def on_validation_epoch_end(self):
    if self.var_min and not self.trainer.sanity_checking:
      self._clipped_schedule_search()
    for k, v in self.metrics.valid_nlls.items():
      self.log(name=k, value=v.compute(), on_step=False,
               on_epoch=True, sync_dist=True)
    self._train_mode()

  def configure_optimizers(self):
    return super().configure_optimizers()

  # -------------------------------------------------------------------------
  # Forward helpers
  # -------------------------------------------------------------------------
  def _subs_parameterization(self, logits, xt):
    logits[:, :, self.mask_id] += self.neg_infinity
    logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    unmasked_indices = (xt != self.mask_id)
    logits[unmasked_indices] = self.neg_infinity
    logits[unmasked_indices, xt[unmasked_indices]] = 0
    return logits

  def forward(self, x, sigma, sample_mode=False, store_kv=False):
    sigma = self._process_sigma(sigma)
    with torch.amp.autocast('cuda', dtype=torch.float32):
      logits = self.backbone(x, sigma,
                             sample_mode=sample_mode,
                             store_kv=store_kv)
    if self.cross_attn:
      x = x[:, :self.config.model.length]
    return self._subs_parameterization(logits, xt=x)

  # -------------------------------------------------------------------------
  # Noise helpers
  # -------------------------------------------------------------------------
  def _sigma_from_p(self, p):
    return torch.min(- torch.log(1 - p), self.sigma_max)

  def _sample_t(self, batch_dims, device, sampling_eps_min, sampling_eps_max, block_size=None):
    if block_size is None:
      block_size = self.block_size
    n = batch_dims[-1]
    num_blocks = n // block_size
    _eps_b = torch.rand((batch_dims[0], num_blocks), device=device)
    if self.antithetic_sampling:
      offset_b = torch.arange(batch_dims[0] * num_blocks, device=device) / (batch_dims[0] * num_blocks)
      offset_b = offset_b.view(batch_dims[0], num_blocks)
      _eps_b = (_eps_b / (batch_dims[0] * num_blocks) + offset_b) % 1
    t = _eps_b
    if block_size != self.config.model.length:
      t = t.repeat_interleave(block_size, dim=-1)
    if sampling_eps_max >= 1 and sampling_eps_min >= 1:
      return torch.ones_like(t)
    t = t * (sampling_eps_max - sampling_eps_min) + sampling_eps_min
    return t

  # -------------------------------------------------------------------------
  # Forward diffusion helpers
  # -------------------------------------------------------------------------
  def _resample_q_xt(self, x, xt, move_indices, p, block_size, sampling_eps_min, sampling_eps_max):
    perc_masked = (xt == self.mask_id).float().sum(-1) / block_size
    while (perc_masked < sampling_eps_min).any() or (perc_masked > sampling_eps_max).any():
      if sampling_eps_min == 1e-3 and sampling_eps_max != 1:
        regen_idx = (perc_masked > sampling_eps_max)
        if regen_idx.max() == 0:
          break
      elif sampling_eps_min != 1e-3 and sampling_eps_max == 1:
        regen_idx = (perc_masked < sampling_eps_min)
        if regen_idx.max() == 0:
          break
      else:
        regen_idx = (perc_masked < sampling_eps_min) | (perc_masked > sampling_eps_max)
      regen_idx = regen_idx.repeat_interleave(block_size, dim=-1)
      move_indices[regen_idx] = (torch.rand(*x.shape, device=x.device) < p)[regen_idx]
      xt = torch.where(move_indices, self.mask_id, x)
      xt = xt.reshape(xt.shape[0], -1, block_size)
      perc_masked = (xt == self.mask_id).float().sum(-1) / block_size
    return xt

  def q_xt(self, x, p, block_size=None, sampling_eps_min=None, sampling_eps_max=None):
    if block_size is None:
      block_size = self.block_size
    move_indices = torch.rand(*x.shape, device=x.device) <= p
    xt = torch.where(move_indices, self.mask_id, x)
    if block_size == 1 and sampling_eps_min == 1.0:
      return torch.full_like(x, self.mask_id)
    if self.config.training.resample and not (sampling_eps_min == 1e-3 and sampling_eps_max == 1.0):
      xt = xt.reshape(xt.shape[0], -1, block_size)
      xt = self._resample_q_xt(x, xt, move_indices, p, block_size, sampling_eps_min, sampling_eps_max)
      xt = xt.reshape(xt.shape[0], -1)
    return xt

  def _maybe_sub_sample(self, x0, attention_mask):
    seqlen = x0.shape[1]
    if seqlen > self.num_tokens:
      start = np.random.choice(self.num_tokens)
      end = start + self.num_tokens
      input_tokens = x0[:, start: end]
      output_tokens = x0[:, start + 1: end + 1]
      new_attention_mask = attention_mask[:, start: end]
      insert_special = getattr(self.config.data, 'insert_train_special', False)
      insert_eos = getattr(self.config.data, 'insert_train_eos', False)
      if insert_special or insert_eos:
        input_tokens[:, 0] = self.tokenizer.bos_token_id
        output_tokens[:, -1] = self.tokenizer.eos_token_id
    else:
      input_tokens = x0
      output_tokens = None
      new_attention_mask = attention_mask
    return input_tokens, output_tokens, new_attention_mask

  def _forward_pass_diffusion(self, x0, t=None, sampling_eps_min=None, sampling_eps_max=None):
    if sampling_eps_min is None:
      sampling_eps_min = 1e-3
      sampling_eps_max = 1.0
    if t is None:
      t = self._sample_t(x0.shape, x0.device, sampling_eps_min, sampling_eps_max)
    loss_scale, p = self._compute_loss_scaling_and_move_chance(t)
    sigma = self._sigma_from_p(p[:, 0].unsqueeze(-1))
    dsigma = - loss_scale * torch.expm1(sigma)
    if self.mdlm_loss_scale:
      sigma, dsigma = self._total_noise(t), self._rate_noise(t)
      p = 1 - torch.exp(-sigma)
      loss_scale = - (dsigma / torch.expm1(sigma))
    xt = self.q_xt(x0, p, sampling_eps_min=sampling_eps_min, sampling_eps_max=sampling_eps_max)
    if sampling_eps_min is not None and sampling_eps_min > 0.5:
      loss_scale = - torch.ones_like(loss_scale)
    if self.config.algo.ignore_bos:
      xt[:, 0] = x0[:, 0]
    x_input = xt
    if self.cross_attn:
      x_input = torch.cat((xt, x0), dim=-1)
    model_output = self.forward(x_input, sigma=sigma)
    log_p_theta = torch.gather(model_output, -1, x0[:, :, None]).squeeze(-1)
    loss = loss_scale * log_p_theta
    return loss

  # -------------------------------------------------------------------------
  # Loss computation
  # -------------------------------------------------------------------------
  def _loss(self, x0, attention_mask, t=None, sampling_eps_min=None, sampling_eps_max=None):
    if sampling_eps_min is None and self.var_min:
      sampling_eps_min = self.sampling_eps_min
      sampling_eps_max = self.sampling_eps_max
    elif sampling_eps_min is None:
      sampling_eps_min = 1e-3
      sampling_eps_max = 1.0

    (input_tokens, output_tokens, attention_mask) = self._maybe_sub_sample(x0, attention_mask)
    loss = self._forward_pass_diffusion(
      input_tokens,
      t=t,
      sampling_eps_min=sampling_eps_min,
      sampling_eps_max=sampling_eps_max)

    if self.ignore_bos and not self.training:
      attention_mask[:, 0] = 0
      
    nlls = loss * attention_mask
    token_nll = nlls.sum() / attention_mask.sum()
    return Loss(loss=token_nll,
                nlls=nlls,
                token_mask=attention_mask)

  # -------------------------------------------------------------------------
  # Validation helpers
  # -------------------------------------------------------------------------
  def _clipped_schedule_search(self):
    best_var = float('inf')
    for (eps_min, eps_max), var in self.metrics.valid_vars.items():
      all_vars = torch.tensor(0., device=self.device)
      for value in var:
        agg_var = value.to(self.device)
        agg_var = self.all_gather(agg_var)
        all_vars += agg_var.var()
      if all_vars < best_var:
        best_var = all_vars
        sampling_eps_min_best = eps_min
        sampling_eps_max_best = eps_max
      self.log(f'valid_var_{round(eps_min, 2)} - {round(eps_max, 2)}',
               all_vars / max(len(var), 1),
               on_epoch=True,
               on_step=False,
               sync_dist=True)
    if getattr(self.config.algo, 'fix_clipping', False) is False:
      self.sampling_eps_min.fill_(sampling_eps_min_best)
      self.sampling_eps_max.fill_(sampling_eps_max_best)

  def _check_val_sampling_intvl(self, sampling_eps_min, sampling_eps_max):
    if (sampling_eps_min == 1e-3 and sampling_eps_max == 1
        and not (self.block_size == 1 and self.config.training.eval_nll)):
      return True
    elif (self.block_size == 1 and sampling_eps_min >= 1):
      return True
    return False

__all__ = ['BD3LM']
