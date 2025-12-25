"""Console logger utilities.

Copied from https://github.com/HazyResearch/transformers/blob/master/src/utils/utils.py
Copied from https://docs.python.org/3/howto/logging-cookbook.html#using-a-context-manager-for-selective-logging
"""
import logging
import pickle
import hashlib
import base64

import fsspec
import lightning
import numpy as np
import torch
from timm.scheduler import CosineLRScheduler


def fsspec_exists(filename):
  """Check if a file exists using fsspec."""
  fs, _ = fsspec.core.url_to_fs(filename)
  return fs.exists(filename)


def fsspec_listdir(dirname):
  """Listdir in manner compatible with fsspec."""
  fs, _ = fsspec.core.url_to_fs(dirname)
  return fs.ls(dirname)


def fsspec_mkdirs(dirname, exist_ok=True):
  """Mkdirs in manner compatible with fsspec."""
  fs, _ = fsspec.core.url_to_fs(dirname)
  fs.makedirs(dirname, exist_ok=exist_ok)


class LoggingContext:
  """Context manager for selective logging."""
  def __init__(self, logger, level=None, handler=None, close=True):
    self.logger = logger
    self.level = level
    self.handler = handler
    self.close = close

  def __enter__(self):
    if self.level is not None:
      self.old_level = self.logger.level
      self.logger.setLevel(self.level)
    if self.handler:
      self.logger.addHandler(self.handler)

  def __exit__(self, et, ev, tb):
    if self.level is not None:
      self.logger.setLevel(self.old_level)
    if self.handler:
      self.logger.removeHandler(self.handler)
    if self.handler and self.close:
      self.handler.close()


def get_logger(name=__name__, level=logging.INFO) -> logging.Logger:
  """Initializes multi-GPU-friendly python logger."""

  logger = logging.getLogger(name)
  logger.setLevel(level)

  # this ensures all logging levels get marked with the rank zero decorator
  # otherwise logs would get multiplied for each GPU process in multi-GPU setup
  for level in ('debug', 'info', 'warning', 'error',
                'exception', 'fatal', 'critical'):
    setattr(logger,
            level,
            lightning.pytorch.utilities.rank_zero_only(
              getattr(logger, level)))

  return logger


class CosineDecayWarmupLRScheduler(
  CosineLRScheduler,
  torch.optim.lr_scheduler._LRScheduler):
  """Wrap timm.scheduler.CosineLRScheduler to match GIDD's cosine schedule.
  Enables calling scheduler.step() without passing in epoch.
  Supports resuming as well.
  Adapted from:
    https://github.com/HazyResearch/hyena-dna/blob/main/src/utils/optim/schedulers.py
  """

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._last_epoch = -1
    self.step(epoch=0)

  def step(self, epoch=None):
    if epoch is None:
      self._last_epoch += 1
    else:
      self._last_epoch = epoch
    # We call either step or step_update, depending on
    # whether we're using the scheduler every epoch or every
    # step.
    # Otherwise, lightning will always call step (i.e.,
    # meant for each epoch), and if we set scheduler
    # interval to "step", then the learning rate update will
    # be wrong.
    if self.t_in_epochs:
      super().step(epoch=self._last_epoch)
    else:
      super().step_update(num_updates=self._last_epoch)


# Copied from https://github.com/jdeschena/sdtt/blob/bbc54d5b3c5fcffd79602cff17ed34dde1f3eff6/src/sdtt/core/sampling/utils.py#L10
def top_k_top_p_filtering(
    logits,
    top_k=0,
    top_p=0.0,
    filter_value=-float("Inf"),
    dim=-1):
    """Apply top-k/top-p filtering to logits."""
    if dim != -1:
      logits = torch.transpose(logits, dim, -1)

    assert top_k < logits.size(dim)
    if top_k > 0:
      values, _ = torch.topk(logits, k=top_k, dim=-1)
      to_remove_mask = (
          logits < torch.min(values, dim=-1, keepdim=True)[0]
      )  # min returns a tuple (values, indices)
      logits[to_remove_mask] = filter_value

    if top_p > 0.0:
      sorted_logits, sorted_indices = torch.sort(
        logits, descending=True, dim=-1)
      cum_probs = torch.cumsum(
        torch.softmax(sorted_logits, dim=-1), dim=-1)

      sorted_indices_to_remove = cum_probs > top_p
      # Ensures at least one token is kept
      sorted_indices_to_remove[..., 1:] = \
        sorted_indices_to_remove[..., :-1].clone()
      sorted_indices_to_remove[..., 0] = 0

      mask_to_remove = torch.empty_like(sorted_indices_to_remove)
      mask_to_remove.scatter_(dim=-1,
                              index=sorted_indices,
                              src=sorted_indices_to_remove)
      logits[mask_to_remove] = filter_value

    if dim != -1:
      logits = torch.transpose(logits, dim, -1)

    return logits


def vars_to_fname(sep='#', **kwargs):
  bool_entries = []
  non_bool_entries = []

  for k, v in kwargs.items():
    if type(v) is bool:
      bool_entries.append((k, v))
    else:
      non_bool_entries.append((k, v))

  bool_entries.sort(key=lambda x: x[0])
  non_bool_entries.sort(key=lambda x: x[0])

  name = ''
  for k, v in non_bool_entries:
    name += str(k) + '=' + str(v)
    name += sep

  for i, (k, v) in enumerate(bool_entries):
    if v:
      name += k
    else:
      name += 'no_' + k

    if i != len(bool_entries) - 1:
      name += sep

  return name


def short_hash(value: str):
  digest = hashlib.md5(value.encode()).digest()
  h = base64.b64encode(digest).decode()
  h = h.replace('/', 'AA')
  assert h[-2:] == '=='
  return h[:-2]


def np_to_base64(arr: np.ndarray) -> str:
  arr_bytes = pickle.dumps(arr)
  base64_bytes = base64.b64encode(arr_bytes)
  return base64_bytes.decode('ascii')


def base64_to_np(b64_str: str) -> np.ndarray:
  base64_bytes = b64_str.encode('ascii')
  arr_bytes = base64.b64decode(base64_bytes)
  return pickle.loads(arr_bytes)


def shift_for_next_token(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *extra_targets: torch.Tensor,
):
  """Align logits/targets so position i predicts token at i+1.

  Args:
    logits: Tensor shaped (..., seq_len, vocab) where the penultimate dim is the
      time/sequence dimension to shift.
    targets: Tensor shaped (..., seq_len) that provides the next-token targets.
    *extra_targets: Optional tensors that share the same sequence shape as
      `targets` (e.g., attention masks) and should be shifted identically.

  Returns:
    Tuple with shifted logits, shifted targets, followed by each shifted tensor
    from `extra_targets` (preserving order). All outputs are contiguous.
  """
  if logits.size(-2) != targets.size(-1):
    raise ValueError(
      'Logits sequence length does not match targets: '
      f'{logits.size(-2)} != {targets.size(-1)}')
  if logits.size(-2) < 2:
    raise ValueError('Need at least two tokens to apply next-token shift.')

  shifted_logits = logits[..., :-1, :].contiguous()
  shifted_targets = targets[..., 1:].contiguous()

  shifted_extras = []
  for tensor in extra_targets:
    if tensor is None:
      shifted_extras.append(None)
      continue
    if tensor.size(-1) != targets.size(-1):
      raise ValueError(
        'Extra tensor sequence length does not match targets: '
        f'{tensor.size(-1)} != {targets.size(-1)}')
    shifted_extras.append(tensor[..., 1:].contiguous())

  return (shifted_logits, shifted_targets, *shifted_extras)


# Try to import Liger Kernel for optimized cross-entropy
try:
  from liger_kernel.transformers.cross_entropy import LigerCrossEntropyLoss
  _LIGER_AVAILABLE = True
except ImportError:
  _LIGER_AVAILABLE = False


def liger_cross_entropy(
    input: torch.Tensor,
    target: torch.Tensor,
    weight=None,
    ignore_index: int = -100,
    reduction: str = "mean",
    label_smoothing: float = 0.0,
) -> torch.Tensor:
  """Cross-entropy loss with optional Liger Kernel optimization.

  Uses Liger Kernel's optimized implementation when available and on CUDA,
  otherwise falls back to PyTorch's F.cross_entropy.

  Args:
    input: Logits tensor of shape (N, C) or (N, C, d1, d2, ..., dk).
    target: Target tensor of shape (N,) or (N, d1, d2, ..., dk).
    weight: Optional manual rescaling weight for each class.
    ignore_index: Target value that is ignored.
    reduction: 'none', 'mean', or 'sum'.
    label_smoothing: Amount of label smoothing in [0.0, 1.0].

  Returns:
    The computed cross-entropy loss.
  """
  # Use Liger if available and input is on CUDA
  if _LIGER_AVAILABLE and input.is_cuda:
    # Flatten to (N, C) format for Liger
    if input.dim() > 2:
      input_flat = input.transpose(1, -1).flatten(0, -2)
      target_flat = target.flatten()
    else:
      input_flat = input
      target_flat = target

    liger_ce = LigerCrossEntropyLoss(
      weight=weight,
      ignore_index=ignore_index,
      label_smoothing=label_smoothing,
      reduction=reduction,
    )
    loss = liger_ce(input_flat, target_flat)

    # Reshape if reduction='none'
    if reduction == "none":
      loss = loss.view_as(target)
    return loss
  else:
    # Fall back to PyTorch
    return torch.nn.functional.cross_entropy(
      input=input,
      target=target,
      weight=weight,
      ignore_index=ignore_index,
      reduction=reduction,
      label_smoothing=label_smoothing,
    )
