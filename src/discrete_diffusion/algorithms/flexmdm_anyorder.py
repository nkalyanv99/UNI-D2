"""FlexMDM Any-Order Algorithm Implementation.

This module implements the FlexMDM any-order mask insertion flow algorithm,
which jointly models insertion (length) and masking (content) processes.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import transformers

from .base import TrainerBase, ensure_mask_token
from ..utils.utils import liger_cross_entropy
from ..forward_process.flexmdm import (
  FlexMDMForwardProcess,
  ModelPrediction,
)
from ..noise_schedules.flex import build_flex_schedule


class FlexMDMAnyOrder(TrainerBase):
  """FlexMDM Any-Order Mask Insertion Flow algorithm.
  
  This algorithm uses a joint interpolant for variable-length discrete
  diffusion with both insertion (length) and masking (content) processes.
  """

  def __init__(self, config, tokenizer: transformers.PreTrainedTokenizer):
    """Initialize FlexMDM any-order algorithm.
    
    Args:
      config: Hydra config object
      tokenizer: Tokenizer for the dataset
    """
    # Ensure tokenizer has mask token
    self.mask_id, vocab_size = ensure_mask_token(tokenizer)

    # Set special tokens in config BEFORE calling super().__init__()
    # so the model gets instantiated with correct token IDs
    config.model.mask_token = self.mask_id
    config.model.pad_token = tokenizer.pad_token_id

    super().__init__(config, tokenizer, vocab_size=vocab_size)
    self.save_hyperparameters()
    
    # Get algorithm-specific config
    algo_cfg = self.config.algo
    
    # Loss function types
    self.unmask_loss_fn = getattr(algo_cfg, 'unmask_loss_fn', 'elbo')
    self.insert_loss_fn = getattr(algo_cfg, 'insert_loss_fn', 'expectation')
    
    # Create insertion and unmasking schedules (default to linear)
    insert_cfg = getattr(algo_cfg, 'insert_schedule', None)
    unmask_cfg = getattr(algo_cfg, 'unmask_schedule', None)
    self.insertion_schedule = build_flex_schedule(insert_cfg)
    self.unmask_schedule = build_flex_schedule(unmask_cfg)
    
    # Create interpolant
    self.interpolant = FlexMDMForwardProcess(
      insertion_schedule=self.insertion_schedule,
      unmask_schedule=self.unmask_schedule,
      tokenizer=self.tokenizer,
      max_length=self.num_tokens,
      pad_token=self.tokenizer.pad_token_id,
    )
    
    # Only embed insert time (not both insert and unmask)
    self.only_embed_insert = getattr(algo_cfg, 'only_embed_insert', True)
    
    self._validate_configuration()

  def _validate_configuration(self):
    """Validate algorithm configuration."""
    assert self.unmask_loss_fn == 'elbo', (
      f"Only 'elbo' unmask loss supported, got {self.unmask_loss_fn}"
    )
    assert self.insert_loss_fn in {'expectation', 'distribution'}, (
      f"Insert loss must be 'expectation' or 'distribution', "
      f"got {self.insert_loss_fn}"
    )

  @staticmethod
  def _jump_kernel_elbo(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-6):
    """Compute KL divergence for Poisson process (jump kernel ELBO).
    
    This is the insertion loss for the length prediction.
    
    Args:
      x: True gap lengths [...]
      y: Predicted gap lengths [...]
      eps: Small constant for numerical stability
      
    Returns:
      KL divergence per position [...]
    """
    x = x.float()
    y = y.float()
    
    x_safe = torch.clamp(x, min=eps)
    y_safe = torch.clamp(y, min=eps)
    return y_safe - x_safe + x_safe * (torch.log(x_safe) - torch.log(y_safe))

  def sample_time(self, batch_size: int, device: torch.device) -> torch.Tensor:
    """Sample time values with low-discrepancy sampling.
    
    Args:
      batch_size: Number of samples
      device: Device for tensors
      
    Returns:
      Time values in [0, 1] [B]
    """
    eps = 1e-6
    interval = 1.0 - eps
    interval_size = interval / batch_size
    u = torch.rand(batch_size, device=device)
    return (torch.arange(batch_size, device=device, dtype=u.dtype) + u
            ) * interval_size

  def forward(self, x: torch.Tensor, t: torch.Tensor) -> ModelPrediction:
    """Forward pass through the model.
    
    Args:
      x: Token indices [B, L]
      t: Time values [B]
      
    Returns:
      ModelPrediction with token_logits and expected_gaps/length_posterior
    """
    if self.only_embed_insert:
      t_embed = self.insertion_schedule.at(t)
      return self.backbone(x, t_embed)
    else:
      return self.backbone(x, t)

  def training_loss(
    self, x1: torch.Tensor, t: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute training loss.
    
    Args:
      x1: Clean token sequences [B, L]
      t: Time values [B]
      
    Returns:
      Tuple of (unmask_loss, insertion_loss, total_loss)
    """
    # Sample interpolant
    interpolant_sample = self.interpolant.sample_interpolant(t, x1)
    unmask_weight, insert_weight = self.interpolant.elbo_weight(t, x1)

    # Forward pass
    prediction: ModelPrediction = self.forward(interpolant_sample.xt, t)

    scale_factor = x1.shape[0] * self.num_tokens

    # Unmask loss (token prediction)
    if self.unmask_loss_fn == "elbo":
      mask_indices = interpolant_sample.mask_indices
      unmask_loss = unmask_weight[mask_indices] * liger_cross_entropy(
        prediction.token_logits[mask_indices],
        interpolant_sample.unmasked[mask_indices],
        reduction="none",
      )
      unmask_loss = unmask_loss.sum() / scale_factor
    else:
      raise ValueError(f"Invalid unmask loss type: {self.unmask_loss_fn}")

    # Insertion loss (length prediction)
    gaps, gaps_mask = interpolant_sample.gaps_and_mask
    if self.insert_loss_fn == "expectation":
      insertion_loss = insert_weight[gaps_mask] * self._jump_kernel_elbo(
        gaps[gaps_mask], prediction.expected_gaps[gaps_mask]
      )
      insertion_loss = insertion_loss.sum() / scale_factor
    elif self.insert_loss_fn == "distribution":
      insertion_loss = insert_weight[gaps_mask] * liger_cross_entropy(
        prediction.length_posterior[gaps_mask],
        gaps[gaps_mask],
        reduction="none",
      )
      insertion_loss = insertion_loss.sum() / scale_factor
    else:
      raise ValueError(f"Invalid insert loss type: {self.insert_loss_fn}")

    total_loss = unmask_loss + insertion_loss
    return unmask_loss, insertion_loss, total_loss

  def training_step(self, batch, batch_idx):
    """Training step.
    
    Args:
      batch: Batch dictionary with 'input_ids' and 'attention_mask'
      batch_idx: Batch index
      
    Returns:
      Total loss
    """
    del batch_idx  # Unused
    
    # Extract input data
    if isinstance(batch, dict):
      x1 = batch["input_ids"]
    else:
      x1 = batch

    # Sample time
    t = self.sample_time(x1.shape[0], x1.device)

    # Calculate losses
    unmask_loss, len_loss, loss = self.training_loss(x1, t)

    # Log component losses
    self.log("train/unmask_loss", unmask_loss, prog_bar=True, sync_dist=True)
    self.log("train/len_loss", len_loss, prog_bar=True, sync_dist=True)
    self.log("trainer/loss", loss, prog_bar=True, sync_dist=True)

    return loss

  def validation_step(self, batch, batch_idx):
    """Validation step.
    
    Args:
      batch: Batch dictionary with 'input_ids' and 'attention_mask'
      batch_idx: Batch index
      
    Returns:
      Total loss
    """
    del batch_idx  # Unused
    
    if isinstance(batch, dict):
      x1 = batch["input_ids"]
    else:
      x1 = batch

    # Sample time
    t = self.sample_time(x1.shape[0], x1.device)
    unmask_loss, len_loss, loss = self.training_loss(x1, t)

    self.log("val/unmask_loss", unmask_loss, prog_bar=True, sync_dist=True)
    self.log("val/len_loss", len_loss, prog_bar=True, sync_dist=True)
    self.log("val_loss", loss, prog_bar=True, sync_dist=True)

    return loss

  def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure=None):
    """Optimizer step with logging.
    
    Args:
      epoch: Current epoch
      batch_idx: Current batch index
      optimizer: Optimizer
      optimizer_closure: Closure for optimizer step
    """
    super().optimizer_step(
      epoch, batch_idx, optimizer, optimizer_closure=optimizer_closure
    )
    
    # Log learning rate and gradient norm
    lr = optimizer.param_groups[0]["lr"]
    self.log("train/lr", lr, on_step=True, prog_bar=True)
    grad_norm = torch.sqrt(
      sum(p.grad.norm(2) ** 2 for p in self.parameters() if p.grad is not None)
    )
    self.log("train/grad_norm", grad_norm, on_step=True, prog_bar=False)




__all__ = ['FlexMDMAnyOrder']
