from __future__ import annotations

import math
from typing import Dict, List, Tuple, Union

import torch
import torchmetrics

LOG2 = math.log(2)

Value = Union[float, torch.Tensor]


class NLL(torchmetrics.aggregation.MeanMetric):
  def update(self,
             value: Value,
             weight: Value = 1.0) -> None:
    """Update state with data.

    Args:
      value: Either a float or tensor containing data.
        Additional tensor dimensions will be flattened
      weight: Either a float or tensor containing weights
        for calculating the average. Shape of weight should
        be able to broadcast with the shape of `value`.
        Default to `1.0` corresponding to simple harmonic
        average.
    """
    # broadcast weight to value shape
    if not isinstance(value, torch.Tensor):
      value = torch.as_tensor(value,
                              dtype=self.dtype,
                              device=self.device)
    else:
      value = value.to(dtype=self.dtype, device=self.device)

    if (weight is not None and
        not isinstance(weight, torch.Tensor)):
      weight = torch.as_tensor(weight,
                               dtype=self.dtype,
                               device=self.device)
    else:
      weight = weight.to(dtype=self.dtype, device=self.device)

    # Handle edge case where torch.compile infers scalar value but sees tensor inputs
    if value.ndim == 0 and weight.ndim > 0:
      weight = weight.squeeze()
    if value.ndim > 0:
      weight = torch.broadcast_to(weight, value.shape)

    if value.numel() == 0:
      return
    self.mean_value += value.sum()
    self.weight += weight.sum()


class BPD(NLL):
  def compute(self) -> torch.Tensor:
    """Computes the bits per dimension.

    Returns:
      bpd
    """
    return self.mean_value / self.weight / LOG2


class Perplexity(NLL):
  def compute(self) -> torch.Tensor:
    """Computes the Perplexity.

    Returns:
     Perplexity
    """
    return torch.exp(self.mean_value / self.weight)


class Metrics:
  def __init__(self, *_args, **_kwargs) -> None:
    metrics = torchmetrics.MetricCollection({
        'nll': NLL(), 'bpd': BPD(), 'ppl': Perplexity()})
    metrics.set_dtype(torch.float64)
    self.train_nlls = metrics.clone(prefix='train/')
    self.train_aux = BPD()
    self.valid_nlls = metrics.clone(prefix='val/')
    self.valid_aux = BPD()
    # Keep sample entropy as a lightweight generative signal during training
    self.sample_entropy = torchmetrics.aggregation.MeanMetric()
    self.sample_entropy.set_dtype(torch.float64)

  def to(self, *args, **kwargs):
    self.sample_entropy = self.sample_entropy.to(*args, **kwargs)
    self.train_nlls = self.train_nlls.to(*args, **kwargs)
    self.train_aux = self.train_aux.to(*args, **kwargs)
    self.valid_nlls = self.valid_nlls.to(*args, **kwargs)
    self.valid_aux = self.valid_aux.to(*args, **kwargs)

  def reset(self):
    self.sample_entropy.reset()
    self.train_nlls.reset()
    self.train_aux.reset()
    self.valid_nlls.reset()
    self.valid_aux.reset()

  def update_train(self, nll, num_tokens):
    self.train_nlls.update(nll, num_tokens)

  def update_valid(self, nll, num_tokens):
    self.valid_nlls.update(nll, num_tokens)


  # Generative ppl logic removed; use standalone evaluation script instead.

  @torch.no_grad()
  def record_entropy(self, tokens):
    for sample in tokens:
      entropy = _token_entropy(sample)
      self._record_entropy_value(entropy)

  def _record_entropy_value(self, entropy: float) -> None:
    self.sample_entropy.update(entropy)


class NFEs(torchmetrics.aggregation.MeanMetric):
  """Average number of function evaluations per sample."""


class BD3Metrics(Metrics):
  """Extension of Metrics with BD3-specific variance tracking."""

  def __init__(self, config) -> None:
    super().__init__()
    self.config = config
    self.block_size = getattr(config, 'block_size', config.model.length)
    self.nfes = NFEs()
    self.gen_entropy = NLL()
    self.gen_nfes: List[float] = []
    self.gen_entropies: List[float] = []
    self.gen_lengths: List[int] = []

    self.sampling_eps = config.training.sampling_eps
    self.clip_search_delta = getattr(config.algo, 'clip_search_delta', None)
    self.valid_vars: Dict[Tuple[float, float], List[torch.Tensor]] = {
      (self.sampling_eps, 1.0): []
    }
    if getattr(config.algo, 'var_min', None):
      self.init_valid_vars()

  def init_valid_vars(self):
    eps = self.sampling_eps
    if self.block_size > 1:
      self.valid_vars = {(eps, 1): []}
      for width in self.config.algo.clip_search_widths:
        for i in torch.arange(0, 1 - width + self.clip_search_delta,
                              self.clip_search_delta):
          eps_min = torch.clamp(i, min=self.sampling_eps).item()
          eps_max = torch.clamp(i + width, min=self.sampling_eps).item()
          self.valid_vars[(eps_min, eps_max)] = []
    else:
      self.valid_vars = {
        (eps, 1): [],
        (1, 1): []
      }

  def update_train(self,
                   nll_sum: torch.Tensor,
                   num_tokens: torch.Tensor):
    self.train_nlls.update(nll_sum, num_tokens)

  def update_valid(self,
                   nll_sum: torch.Tensor,
                   num_tokens: torch.Tensor):
    self.valid_nlls.update(nll_sum, num_tokens)

  def to(self, *args, **kwargs):
    super().to(*args, **kwargs)
    self.nfes = self.nfes.to(*args, **kwargs)
    self.gen_entropy = self.gen_entropy.to(*args, **kwargs)

  def reset(self):
    super().reset()
    self.gen_nfes, self.gen_entropies, self.gen_lengths = [], [], []
    self.nfes.reset()
    self.gen_entropy.reset()
    if getattr(self.config.algo, 'var_min', None):
      self.init_valid_vars()

  @torch.no_grad()
  def record_entropy(self, tokens):
    for sample in tokens:
      entropy = _token_entropy(sample)
      self._record_entropy_value(entropy)

  def _record_entropy_value(self, entropy: float) -> None:
    self.sample_entropy.update(entropy)
    self.gen_entropies.append(entropy)
    self.gen_entropy.update(entropy, 1)


def _token_entropy(sample: torch.Tensor) -> float:
  _, counts = torch.unique(sample, return_counts=True, sorted=False)
  probs = counts.float() / counts.sum()
  return torch.special.entr(probs).sum().item()


class GStarMetrics:
  """Metrics for GStar binary classification (remasker error detection)."""
  
  def __init__(self):
    # Train metrics
    self.train_accuracy = torchmetrics.classification.BinaryAccuracy()
    self.train_precision = torchmetrics.classification.BinaryPrecision()
    self.train_recall = torchmetrics.classification.BinaryRecall()
    self.train_f1 = torchmetrics.classification.BinaryF1Score()
    
    # Validation metrics
    self.val_accuracy = torchmetrics.classification.BinaryAccuracy()
    self.val_precision = torchmetrics.classification.BinaryPrecision()
    self.val_recall = torchmetrics.classification.BinaryRecall()
    self.val_f1 = torchmetrics.classification.BinaryF1Score()

  def update_train(self, preds: torch.Tensor, targets: torch.Tensor):
    """Update train metrics with batch predictions and targets.
    
    Args:
        preds: Binary predictions [batch, seq_len] (0 or 1)
        targets: Binary targets [batch, seq_len] (0 or 1)
    """
    # Flatten to [batch * seq_len] for binary classification metrics
    preds_flat = preds.flatten()
    targets_flat = targets.flatten()
    
    self.train_accuracy.update(preds_flat, targets_flat)
    self.train_precision.update(preds_flat, targets_flat)
    self.train_recall.update(preds_flat, targets_flat)
    self.train_f1.update(preds_flat, targets_flat)

  def update_valid(self, preds: torch.Tensor, targets: torch.Tensor):
    """Update validation metrics with batch predictions and targets.
    
    Args:
        preds: Binary predictions [batch, seq_len] (0 or 1)
        targets: Binary targets [batch, seq_len] (0 or 1)
    """
    # Flatten to [batch * seq_len] for binary classification metrics
    preds_flat = preds.flatten()
    targets_flat = targets.flatten()
    
    self.val_accuracy.update(preds_flat, targets_flat)
    self.val_precision.update(preds_flat, targets_flat)
    self.val_recall.update(preds_flat, targets_flat)
    self.val_f1.update(preds_flat, targets_flat)

  def compute_train(self) -> Dict[str, torch.Tensor]:
    """Compute train metrics.
    
    Returns:
        Dictionary with train/accuracy, train/precision, train/recall, train/f1
    """
    return {
      'train/accuracy': self.train_accuracy.compute(),
      'train/precision': self.train_precision.compute(),
      'train/recall': self.train_recall.compute(),
      'train/f1': self.train_f1.compute(),
    }

  def compute_valid(self) -> Dict[str, torch.Tensor]:
    """Compute validation metrics.
    
    Returns:
        Dictionary with val/accuracy, val/precision, val/recall, val/f1
    """
    return {
      'val/accuracy': self.val_accuracy.compute(),
      'val/precision': self.val_precision.compute(),
      'val/recall': self.val_recall.compute(),
      'val/f1': self.val_f1.compute(),
    }

  def reset(self):
    """Reset all metrics."""
    self.train_accuracy.reset()
    self.train_precision.reset()
    self.train_recall.reset()
    self.train_f1.reset()
    self.val_accuracy.reset()
    self.val_precision.reset()
    self.val_recall.reset()
    self.val_f1.reset()

  def to(self, *args, **kwargs):
    """Move all metrics to specified device/dtype."""
    self.train_accuracy = self.train_accuracy.to(*args, **kwargs)
    self.train_precision = self.train_precision.to(*args, **kwargs)
    self.train_recall = self.train_recall.to(*args, **kwargs)
    self.train_f1 = self.train_f1.to(*args, **kwargs)
    self.val_accuracy = self.val_accuracy.to(*args, **kwargs)
    self.val_precision = self.val_precision.to(*args, **kwargs)
    self.val_recall = self.val_recall.to(*args, **kwargs)
    self.val_f1 = self.val_f1.to(*args, **kwargs)
    return self


__all__ = ['Metrics', 'BD3Metrics', 'GStarMetrics']
