import itertools
import os
from dataclasses import dataclass
from pathlib import Path

import hydra.utils
import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
import transformers

from ..evaluations import Metrics
from ..models import create_ema
from .. import utils
import omegaconf
from ..forward_process.utils import _effective_vocab_size, _unsqueeze


def ensure_mask_token(tokenizer):
  """Return mask token id and vocab size, ensuring the tokenizer exposes the mask."""
  vocab_size = _effective_vocab_size(tokenizer)
  if getattr(tokenizer, 'mask_token', None) is None:
    mask_id = vocab_size
    vocab_size += 1
  else:
    mask_id = tokenizer.mask_token_id
  if getattr(tokenizer, 'mask_token_id', None) is None:
    setattr(tokenizer, 'mask_token_id', int(mask_id))
  return int(mask_id), vocab_size

@dataclass
class Loss:
  loss: torch.FloatTensor
  nlls: torch.FloatTensor
  num_tokens: torch.FloatTensor


class TrainerBase(L.LightningModule):
  """Base Trainer class for discrete diffusion models.
  
  Handles initialization of backbone, noise schedule, and sampler, as well as
  Lightning hooks for training and validation loops.
  """
  def __init__(self, config, tokenizer: transformers.PreTrainedTokenizer, vocab_size=None):
    super().__init__()
    self.save_hyperparameters()
    self.config = config
    self.ignore_bos = getattr(self.config.algo, 'ignore_bos', False)
    self.loss_type = getattr(self.config.algo, 'loss_type', None)
    self.tokenizer = tokenizer
    if vocab_size is None:
      self.vocab_size = len(self.tokenizer)
    else:
      self.vocab_size = vocab_size
    self.sampler = self.config.sampling.predictor
    self.antithetic_sampling = self.config.training.antithetic_sampling
    self.parameterization = self.config.algo.parameterization
    self._sampler_cfg = self._resolve_sampler_config()
    self._sampler = None
    
    target = self.config.model._target_
    instantiate_config = omegaconf.OmegaConf.create({'_target_': target})
    self.backbone = hydra.utils.instantiate(
      instantiate_config,
      self.config,
      self.vocab_size,
      _recursive_=False
    )
    self.model = self.backbone

    self.T = self.config.algo.T
    self.num_tokens = self.config.model.length
    self.softplus = torch.nn.Softplus()
    self.p_nucleus = self.config.sampling.p_nucleus
    # Noise schedule - use Hydra instantiation
    # HybridDiffusion needs tokenizer passed at runtime
    if hasattr(self.config.noise, '_target_') and 'HybridDiffusion' in self.config.noise._target_:
      self.noise = hydra.utils.instantiate(self.config.noise, tokenizer=self.tokenizer)
    else:
      self.noise = hydra.utils.instantiate(self.config.noise)

    # Metrics will be initialized via _initialize_metrics() to allow override
    self.metrics = None
    self._initialize_metrics()

    # EMA will be initialized in setup() to avoid initialization order issues
    self.ema = None
    self.lr = self.config.optim.lr
    self.sampling_eps = self.config.training.sampling_eps
    self.time_conditioning = self.config.algo.time_conditioning
    if config.neg_infinity_mode == 'large-finite':
      self.neg_infinity = -1000000.0
    elif config.neg_infinity_mode == 'true-inf':
      self.neg_infinity = -float('inf')
    else:
      raise ValueError(f"neg_infinity_mode must be 'large-finite' or 'true-inf', got '{config.neg_infinity_mode}'")
    self.fast_forward_epochs = None
    self.fast_forward_batches = None

  def _initialize_metrics(self):
    """Initialize metrics. Override in subclasses for custom metrics."""
    self.metrics = Metrics()

  def _prepare_ema(self):
    if self.config.training.ema > 0:
      self.ema = create_ema(self._get_parameters(), decay=self.config.training.ema)
    else:
      self.ema = None
  
  def setup(self, stage: str):
    """Called by Lightning before training/validation/test.
    
    Initializes EMA here to ensure all model components (including subclass
    additions like remasker_head in GStar) are ready before EMA wraps parameters.
    """
    if self.ema is None:  # Only prepare once
      self._prepare_ema()

  def _validate_configuration(self):
    if self.config.algo.parameterization == 'ar':
      assert not self.config.algo.time_conditioning
      assert self.config.prior.type == 'none'

    if self.parameterization in {'score', 'mean'}:
      assert self.time_conditioning
    if self.T > 0:
      assert self.parameterization != 'score'

  def to(self, *args, **kwargs):
    self = super().to(*args, **kwargs) 
    self.metrics.to(*args, **kwargs)
    return self

  def q_xt(self, x, t):
    raise NotImplementedError
  
  def _get_parameters(self):
    return itertools.chain(self.backbone.parameters(), self.noise.parameters())

  def _eval_mode(self):
    if self.ema:
      self.ema.store(self._get_parameters())
      self.ema.copy_to(self._get_parameters())
    self.backbone.eval()
    self.noise.eval()

  def _train_mode(self):
    if self.ema:
      self.ema.restore(self._get_parameters())
    self.backbone.train()
    self.noise.train()

  def on_load_checkpoint(self, checkpoint):
    if self.ema:
      self.ema.load_state_dict(checkpoint['ema'])
    # Copied from:
    # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py#L41
    self.fast_forward_epochs = checkpoint['loops'][
      'fit_loop']['epoch_progress']['current']['completed']
    self.fast_forward_batches = checkpoint['loops'][
      'fit_loop']['epoch_loop.batch_progress'][
        'current']['completed']

  def on_save_checkpoint(self, checkpoint):
    if self.ema:
      checkpoint['ema'] = self.ema.state_dict()
    # Copied from:
    # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/tasks/seq.py
    # ['epoch_loop.batch_progress']['total']['completed']
    # is 1 iteration behind, so we're using the optimizer's progress.
    checkpoint['loops']['fit_loop'][
      'epoch_loop.batch_progress']['total'][
        'completed'] = checkpoint['loops']['fit_loop'][
          'epoch_loop.automatic_optimization.optim_progress'][
            'optimizer']['step']['total'][
              'completed'] * self.trainer.accumulate_grad_batches
    checkpoint['loops']['fit_loop'][
      'epoch_loop.batch_progress']['current'][
        'completed'] = checkpoint['loops']['fit_loop'][
          'epoch_loop.automatic_optimization.optim_progress'][
            'optimizer']['step']['current'][
              'completed'] * self.trainer.accumulate_grad_batches
    # _batches_that_stepped tracks the number of global steps,
    # not the number of local steps, so we don't multiply with
    # self.trainer.accumulate_grad_batches here.
    checkpoint['loops']['fit_loop'][
      'epoch_loop.state_dict'][
        '_batches_that_stepped'] = checkpoint['loops']['fit_loop'][
          'epoch_loop.automatic_optimization.optim_progress'][
            'optimizer']['step']['total']['completed']

  def on_train_start(self):
    if self.ema:
      self.ema.move_shadow_params_to_device(self.device)

  def optimizer_step(self, *args, **kwargs):
    super().optimizer_step(*args, **kwargs)
    if self.ema: self.ema.update(self._get_parameters())

  def _process_sigma(self, sigma):
    raise NotImplementedError

  def _process_model_output(self, model_output, xt, sigma):
    """Process raw model output into log-probabilities or scores.
    
    Args:
        model_output: Raw output from the backbone model.
        xt: Noisy input tokens.
        sigma: Noise level.
        
    Returns:
        Tensor: Processed output (e.g. log-probs).
    """
    raise NotImplementedError

  def forward(self, xt, sigma, group_idxs=None):
    sigma = self._process_sigma(sigma)
    with torch.amp.autocast('cuda', dtype=torch.float32):
      if group_idxs is None:
        model_output = self.backbone(xt, sigma)
      else:
        model_output = self.backbone(xt, group_idxs, sigma)
    return self._process_model_output(model_output=model_output, xt=xt, sigma=sigma)

  def _loss(self, x0, valid_tokens,
            current_accumulation_step=None,
            train_mode=False):
    """Generic loss aggregation for all trainer modules."""
    (input_tokens, output_tokens,
     valid_tokens) = self._process_model_input(x0, valid_tokens)
    loss = self.nll(input_tokens, output_tokens,
                    current_accumulation_step, train_mode)
    assert loss.ndim == 2
    if self.ignore_bos:
      loss[:, 0] = 0
      valid_tokens[:, 0] = 0
    if (getattr(self, 'shift_loss_targets', False)
        and valid_tokens.size(-1) == loss.size(-1) + 1):
      valid_tokens = valid_tokens[:, 1:]

    nlls = (loss * valid_tokens).sum()
    num_tokens = valid_tokens.sum()
    token_nll = nlls / num_tokens

    return Loss(loss=token_nll,
                nlls=nlls,
                num_tokens=num_tokens)

  def _reset_train_metrics(self):
    """Reset train metrics at epoch start. Override for custom behavior."""
    self.metrics.reset()

  def on_train_epoch_start(self):
    self._reset_train_metrics()

  def _update_train_metrics(self, losses):
    """Update train metrics with batch losses. Override for custom metrics."""
    self.metrics.update_train(losses.nlls, losses.num_tokens)

  def training_step(self, batch, batch_idx):
    current_accumulation_step = (
      batch_idx % self.trainer.accumulate_grad_batches)
    losses = self._loss(batch['input_ids'], batch['attention_mask'], current_accumulation_step, train_mode=True)
    self._update_train_metrics(losses)
    self.log(name='trainer/loss', value=losses.loss, on_step=True, on_epoch=False, sync_dist=True, prog_bar=True)
    return losses.loss

  def _log_train_epoch_metrics(self):
    """Log train metrics at epoch end. Override for custom logging."""
    train_metrics = {}
    for k, v in self.metrics.train_nlls.items():
      if getattr(v, 'weight', 0) > 0:
        train_metrics[k] = v.compute()
    if train_metrics:
      self.log_dict(train_metrics, on_step=False, on_epoch=True, sync_dist=True)
    if hasattr(self.metrics, 'train_aux') and self.metrics.train_aux.weight > 0:
      self.log(name='train/aux', value=self.metrics.train_aux.compute(), on_step=False, on_epoch=True, sync_dist=True)

  def on_train_epoch_end(self):
    self._log_train_epoch_metrics()

  def _reset_valid_metrics(self):
    """Reset valid metrics at validation start. Override for custom behavior."""
    self.metrics.reset()

  def on_validation_epoch_start(self):
    self._reset_valid_metrics()
    self._eval_mode()

  def _update_valid_metrics(self, losses):
    """Update valid metrics with batch losses. Override for custom metrics."""
    self.metrics.update_valid(losses.nlls, losses.num_tokens)

  def validation_step(self, batch, batch_idx):
    losses = self._loss(batch['input_ids'], batch['attention_mask'])
    self._update_valid_metrics(losses)
    return losses.loss

  def _log_valid_epoch_metrics(self):
    """Log valid metrics at epoch end. Override for custom logging."""
    valid_metrics = {}
    for k, v in self.metrics.valid_nlls.items():
      if getattr(v, 'weight', 0) > 0:
        valid_metrics[k] = v.compute()
    if valid_metrics:
      self.log_dict(valid_metrics, on_step=False, on_epoch=True, sync_dist=True)
    if hasattr(self.metrics, 'valid_aux') and self.metrics.valid_aux.weight > 0:
      self.log(name='val/aux', value=self.metrics.valid_aux.compute(), on_step=False, on_epoch=True, sync_dist=True)

  def on_validation_epoch_end(self):
    self._log_valid_epoch_metrics()
    if ((self.config.eval.compute_perplexity_on_sanity
         or not self.trainer.sanity_checking)
         and self.config.eval.generate_samples):
      try:
        samples, text_samples = None, None
        for _ in range(
          self.config.sampling.num_sample_batches):
          samples = self.generate_samples(num_samples=self.config.loader.eval_batch_size)
          
          # Record entropy if metrics class supports it
          if hasattr(self.metrics, 'record_entropy'):
            self.metrics.record_entropy(samples)
          # For logging and optional saving only
          text_samples = self.tokenizer.batch_decode(samples)
        if text_samples is not None:
          if self.trainer.global_rank == 0 and hasattr(
            self.trainer.logger, 'log_table'):
            # Log the last generated samples
            text_samples = text_samples[
              : self.config.sampling.num_sample_log]
            self.trainer.logger.log_table(
              key=f'samples@global_step{self.global_step}',
              columns=['Generated Samples'],
              data=[[s] for s in text_samples])
          if hasattr(self.metrics, 'sample_entropy'):
            # Always log sample entropy (cheap and useful)
            self.log('val/sample_entropy', self.metrics.sample_entropy.compute(), on_epoch=True, on_step=False, sync_dist=True)

          # Optionally save validation samples for later gen-PPL evaluation
          if getattr(self.config.eval, 'save_validation_samples', False):
            save_dir = Path(os.getcwd()) / 'validation_samples'
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / f'step_{self.global_step}.pt'
            torch.save(samples.detach().cpu(), save_path.as_posix())
      except Exception as e:
        print(f"Sampling failed at step {self.global_step}: {e}")
    self._train_mode()

  def configure_optimizers(self):
    optimizer = torch.optim.AdamW(
      self._get_parameters(),
      lr=self.config.optim.lr,
      betas=(self.config.optim.beta1,
             self.config.optim.beta2),
      eps=self.config.optim.eps,
      weight_decay=self.config.optim.weight_decay)

    scheduler = hydra.utils.instantiate(self.config.lr_scheduler, optimizer=optimizer)
    scheduler_dict = {'scheduler': scheduler,
                      'interval': 'step',
                      'monitor': 'val/loss',
                      'name': 'trainer/lr'}
    return [optimizer], [scheduler_dict]

  def _create_sampler(self):
    """Instantiate (and cache) the configured sampler."""
    if self._sampler is not None:
      return self._sampler
    sampler_cfg = self._sampler_cfg
    if sampler_cfg is None:
      return None
    self._sampler = hydra.utils.instantiate(
      sampler_cfg,
      self.config,
      forward_process=getattr(self, '_forward_process', None),
      _recursive_=False,
    )
    return self._sampler

  def _resolve_sampler_config(self):
    """Return the first sampler config specifying a Hydra target."""
    algo_sampler = getattr(self.config.algo, 'sampler', None)
    if getattr(algo_sampler, '_target_', None):
      return algo_sampler
    sampling_sampler = getattr(self.config.sampling, 'sampler', None)
    if getattr(sampling_sampler, '_target_', None):
      return sampling_sampler
    return None

  @torch.no_grad()
  def generate_samples(self, num_samples, num_steps=None, eps=None):
    """Generate samples from the model using the new sampler system.
    
    Subclasses should not need to override this method if they have a 
    corresponding Sampler implementation registered in the sampling registry.
    """
    if num_steps is None:
      num_steps = self.config.sampling.steps
    if eps is None:
      eps = 1e-5
    inject_bos = getattr(self.config.sampling, 'inject_bos', True)
    
    sampler = self._create_sampler()
    if sampler is None:
      raise NotImplementedError(
        f"Algorithm {self.config.algo.name} does not have a configured sampler. "
        "Set 'sampling.sampler._target_' or 'algo.sampler._target_' in the config "
        "to select a Sampler, or override generate_samples().")
    
    return sampler.generate(model=self, num_samples=num_samples, num_steps=num_steps, eps=eps, inject_bos=inject_bos)

  def _process_model_input(self, x0, valid_tokens):
    raise NotImplementedError

  def nll(self, input_tokens, output_tokens,
          current_accumulation_step=None, train_mode=False):
    """Compute negative log likelihood for the given input and output tokens.

    Args:
        input_tokens: Input token indices.
        output_tokens: Target token indices.
        current_accumulation_step: Current gradient accumulation step index.
        train_mode: Whether the model is in training mode.

    Returns:
        Tensor: NLL loss.
    """
    raise NotImplementedError

class Diffusion(TrainerBase):
  """Base class for diffusion-based algorithms.
  
  Implements continuous-time diffusion logic including time sampling,
  sigma processing, and generic NLL computation.
  """
  def _validate_configuration(self):
    super()._validate_configuration()
    assert self.loss_type in {'elbo', 'low_var'}

  def _process_model_input(self, x0, valid_tokens):
    return x0, None, valid_tokens

  def nll(self, x0, output_tokens,
          current_accumulation_step=None, train_mode=False):
    """Implements diffusion-style NLL evaluation."""
    del output_tokens
    t = self._sample_t(x0.shape[0], current_accumulation_step)
    assert t.shape[0] == x0.shape[0]
    if self.T > 0:
      t = (t * self.T).to(torch.int)
      t = t / self.T
      t += (1 / self.T)

    alpha_t = self.noise.alpha_t(t)
    dalpha_t = self.noise.alpha_prime_t(t)
    alpha_t = alpha_t.unsqueeze(-1)
    dalpha_t = dalpha_t.unsqueeze(-1)
    assert alpha_t.ndim == 2
    sigma = self._sigma_from_alphat(alpha_t)

    xt = self.q_xt(x0, t)
    # Optional next-token shift: align logits[..., :-1, :] with targets[..., 1:]
    if getattr(self, 'shift_loss_targets', False):
      # MD4-style: compute CE on raw logits, mask to xt==mask, weight by dalpha/(1-alpha).
      # 1) Get raw logits from backbone (bypass post-processing)
      raw_logits = self.backbone(xt, sigma)
      # 2) Apply next-token shift to align logits and targets, also shift xt
      raw_logits, x0, xt = utils.shift_for_next_token(raw_logits, x0, xt)
      # 3) Per-token CE (use log_softmax to avoid adding new imports)
      ce = - raw_logits.log_softmax(-1).gather(-1, x0.unsqueeze(-1)).squeeze(-1)
      # 4) Mask to only count positions where xt was masked
      mask_positions = (xt == self.mask_id).to(ce.dtype)
      masked_neg_ce = mask_positions * (-ce)
      # 5) Weight by alpha_prime / (1 - alpha)
      weighting = dalpha_t / (1 - alpha_t)
      while weighting.dim() < masked_neg_ce.dim():
        weighting = weighting.unsqueeze(-1)
      return weighting * masked_neg_ce
    else:
      log_x_theta = self.forward(xt, sigma=sigma)
      return self.nll_per_token(
        log_x_theta=log_x_theta,
        xt=xt,
        x0=x0,
        alpha_t=alpha_t,
        dalpha_t=dalpha_t,
        low_var=train_mode and self.loss_type == 'low_var')

  def _process_sigma(self, sigma):
    assert sigma.ndim == 2
    sigma = sigma.mean(-1).squeeze()
    if sigma.ndim == 0:
      sigma = sigma.unsqueeze(0)
    if not self.time_conditioning:
      sigma = torch.zeros_like(sigma)
    assert sigma.ndim == 1, sigma.shape
    return sigma

  def _sample_t(self, n, accum_step):
    if accum_step is not None:
      # During training
      batch_dim = n
      n = self.config.loader.global_batch_size
    _eps_t = torch.rand(n, device=self.device)
    if self.antithetic_sampling:
      offset = torch.arange(n, device=self.device) / n
      _eps_t = (_eps_t / n + offset) % 1
    t = (1 - self.sampling_eps) * _eps_t + self.sampling_eps
    if accum_step is not None:
      t = t.chunk(self.trainer.num_nodes)[self.trainer.node_rank]
      t = t.chunk(self.trainer.num_devices)[self.trainer.local_rank]
      t = t.chunk(self.trainer.accumulate_grad_batches)[
        accum_step]
      # corner case for the last datapoint
      t = t[:batch_dim]
    return t

  def _sigma_from_alphat(self, alpha_t):
    return -torch.log(alpha_t)

  def _reconstruction_loss(self, x0):
    t0 = torch.zeros(1, x0.shape[0], dtype=self.dtype, device=self.device)
    sigma_t0 = self._sigma_from_alphat(self.noise.alpha_t(t0))
    model_output_t0 = self.forward(x0, sigma_t0)
    return -torch.gather(input=model_output_t0, dim=-1, index=x0[:, :, None]).squeeze(-1)

  def nll_per_token(self, model_output, xt, x0, alpha_t, dalpha_t, low_var):
    """Compute per-token negative log likelihood.

    Args:
        model_output: Model predictions (logits or scores).
        xt: Noisy input tokens.
        x0: Target clean tokens.
        alpha_t: Signal schedule value at time t.
        dalpha_t: Derivative of alpha_t at time t.
        low_var: Whether to use low-variance loss formulation.

    Returns:
        Tensor: Per-token NLL.
    """
    raise NotImplementedError

  def _get_score(self, x, sigma, group_idxs=None):
    raise NotImplementedError


class AbsorbingState(Diffusion):
  """Base class for absorbing state diffusion models (e.g. MDLM).
  
  Handles mask token management and forward process instantiation for
  masking-based methods.
  """
  def __init__(self, config, tokenizer):
    # NOTE: Ideally, we should do 
    # vocab_size = len(tokenizer), so that we account
    # for the special tokens added in data/loaders.py.
    # But we use tokenizer.vocab_size so as to to be
    # consistent with the prior checkpoints.
    self.mask_id, vocab_size = ensure_mask_token(tokenizer)
    super().__init__(config, tokenizer, vocab_size=vocab_size)
    self.save_hyperparameters()

    # Instantiate forward process using Hydra
    fp_cfg = getattr(self.config.algo, 'forward_process', None)
    if fp_cfg is None or not hasattr(fp_cfg, '_target_'):
      raise ValueError(
        "Forward process must be configured with '_target_' field. "
        "Example: forward_process._target_=discrete_diffusion.forward_process.AbsorbingForwardProcess"
      )
    fp_config = omegaconf.OmegaConf.create(fp_cfg)
    self._forward_process = hydra.utils.instantiate(
      fp_config,
      tokenizer=self.tokenizer,
      schedule=self.noise,
      _recursive_=False
    )

  def _validate_configuration(self):
    super()._validate_configuration()
    if self.parameterization in {'score', 'mean'}:
      assert self.time_conditioning
    assert not (self.parameterization == 'mean' and self.T == 0)
    if self.T > 0:
      assert self.parameterization in {'mean', 'subs'}

  def q_xt(self, x, t):
    """Computes the noisy sample xt by delegating to the configured forward process.
    
    Args:
        x: Clean input tokens [batch, length].
        t: Time values [batch] or float.
        
    Returns:
        Tensor: Noisy tokens xt.
    """
    if not isinstance(t, torch.Tensor):
      t = torch.as_tensor(t, device=x.device, dtype=torch.float32)
    elif t.device != x.device:
      t = t.to(device=x.device)
    out = self._forward_process(x, t)
    xt = out[0] if isinstance(out, (tuple, list)) else out
    if self.ignore_bos:
      xt[:, 0] = x[:, 0]
    return xt

  def prior_sample(self, *batch_dims):
    size = batch_dims[0] if len(batch_dims) == 1 and isinstance(batch_dims[0], (tuple, list)) else batch_dims
    return torch.full(tuple(size), self.mask_id, dtype=torch.int64, device=self.device)
