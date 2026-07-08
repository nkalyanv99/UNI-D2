"""Shared model primitives for UNI-D² backbones.

This module centralizes lightweight, backend-agnostic helpers used across
multiple backbones (DiT, BlockDiT, Encoder-Decoder). Attention backend
selection (flash-attn vs SDPA vs flex) remains in each backbone.
"""

from __future__ import annotations

import math
import typing

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

try:  # flash-attn is optional but recommended
  import flash_attn
  import flash_attn.layers.rotary
  FLASH_ATTN_AVAILABLE = True
except (ImportError, RuntimeError):  # pragma: no cover - flash_attn required in production
  flash_attn = None  # type: ignore
  FLASH_ATTN_AVAILABLE = False


def supports_flash_attention() -> bool:
  """Check if flash-attn is available and functional."""
  return FLASH_ATTN_AVAILABLE


def supports_flex_attention() -> bool:
  """Check if torch flex attention is available (PyTorch 2.4+)."""
  return hasattr(torch.nn.functional, 'flex_attention')


# -----------------------------------------------------------------------------
# Fused bias + dropout + residual + scale utilities
# -----------------------------------------------------------------------------
def bias_dropout_add_scale(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float,
    training: bool) -> torch.Tensor:
  if bias is not None:
    out = scale * F.dropout(x + bias, p=prob, training=training)
  else:
    out = scale * F.dropout(x, p=prob, training=training)

  if residual is not None:
    out = residual + out
  return out


def get_bias_dropout_add_scale(training: bool):
  def _bias_dropout_add(x, bias, scale, residual, prob):
    return bias_dropout_add_scale(
      x, bias, scale, residual, prob, training)

  return _bias_dropout_add


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
  return x * (1 + scale) + shift


@torch.jit.script
def bias_dropout_add_scale_fused_train(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float) -> torch.Tensor:
  return bias_dropout_add_scale(x, bias, scale, residual, prob, True)


@torch.jit.script
def bias_dropout_add_scale_fused_inference(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float) -> torch.Tensor:
  return bias_dropout_add_scale(x, bias, scale, residual, prob, False)


@torch.jit.script
def modulate_fused(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
  return modulate(x, shift, scale)


# -----------------------------------------------------------------------------
# Normalization and small utils
# -----------------------------------------------------------------------------
class LayerNorm(nn.Module):
  def __init__(self, dim: int):
    super().__init__()
    self.weight = nn.Parameter(torch.ones([dim]))
    self.dim = dim

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    with torch.amp.autocast('cuda', enabled=False):
      x = F.layer_norm(x.float(), [self.dim])
    return x * self.weight[None, None, :]


def residual_linear(x: torch.Tensor, W: torch.Tensor, x_skip: torch.Tensor, residual_scale: float) -> torch.Tensor:
  """Compute x_skip + residual_scale * (W @ x) efficiently via addmm.

  Shapes:
    - x: (..., dim_in)
    - W: (dim_out, dim_in)
    - returns: (..., dim_out)
  """
  dim_out, dim_in = W.shape[0], W.shape[1]
  return torch.addmm(
    x_skip.view(-1, dim_out),
    x.view(-1, dim_in),
    W.T,
    alpha=residual_scale).view(*x.shape[:-1], dim_out)


# -----------------------------------------------------------------------------
# Rotary embeddings and helpers
# -----------------------------------------------------------------------------
class Rotary(torch.nn.Module):
  def __init__(self, dim: int, base: int = 10_000):
    super().__init__()
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    self.register_buffer('inv_freq', inv_freq)
    self.seq_len_cached = None
    self.cos_cached: typing.Optional[torch.Tensor] = None
    self.sin_cached: typing.Optional[torch.Tensor] = None

  def forward(
      self,
      x: typing.Optional[torch.Tensor] = None,
      seq_len: typing.Optional[int] = None,
      device: typing.Optional[torch.device] = None,
      seq_dim: int = 1):
    # Support both x-provided and direct seq_len usage
    if seq_len is not None and seq_len == self.seq_len_cached:
      # Return cached tensors
      assert self.cos_cached is not None and self.sin_cached is not None
      return self.cos_cached, self.sin_cached

    if x is not None:
      seq_len = x.shape[seq_dim]
      device = x.device

    assert seq_len is not None, "seq_len must be provided if x is None"
    assert device is not None, "device must be provided if x is None"

    if seq_len != self.seq_len_cached:
      self.seq_len_cached = seq_len
      inv_freq = self.inv_freq.to(device=device, dtype=torch.float32)
      t = torch.arange(seq_len, device=device, dtype=torch.float32)
      freqs = torch.einsum("i,j->ij", t, inv_freq)
      emb = torch.cat((freqs, freqs), dim=-1).to(device)
      # dims are: batch, seq_len, qkv, head, dim
      self.cos_cached = emb.cos()[None, :, None, None, :].repeat(1, 1, 3, 1, 1)
      self.sin_cached = emb.sin()[None, :, None, None, :].repeat(1, 1, 3, 1, 1)
      # This makes the transformation on v an identity.
      self.cos_cached[:, :, 2, :, :].fill_(1.)
      self.sin_cached[:, :, 2, :, :].fill_(0.)

    assert self.cos_cached is not None and self.sin_cached is not None
    return self.cos_cached, self.sin_cached


def rotate_half(x: torch.Tensor) -> torch.Tensor:
  x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
  return torch.cat((-x2, x1), dim=-1)


def split_and_apply_rotary_pos_emb(qkv: torch.Tensor, rotary_cos_sin: typing.Tuple[torch.Tensor, torch.Tensor]):
  """Apply rotary to q,k slices of packed qkv.

  Expects qkv shaped (B, S, 3, H, D). Returns (q, k, v) with shapes
  (B, S, H, D), (B, S, H, D), (B, S, H, D).
  """
  if flash_attn is None:
    raise RuntimeError("flash_attn is required for rotary split helpers")
  with torch.amp.autocast('cuda', enabled=False):
    cos, sin = rotary_cos_sin
    cos = cos.to(qkv.dtype)
    sin = sin.to(qkv.dtype)
    # Align cached length/batch with qkv if needed
    if qkv.shape[1] < cos.shape[1]:
      cos = cos[:, :qkv.shape[1]]
      sin = sin[:, :qkv.shape[1]]
    if cos.shape[0] == 1:
      cos_in = cos[0, :, 0, 0, :cos.shape[-1] // 2]
      sin_in = sin[0, :, 0, 0, :sin.shape[-1] // 2]
    else:
      cos_in = cos[:, :, 0, 0, :cos.shape[-1] // 2]
      sin_in = sin[:, :, 0, 0, :sin.shape[-1] // 2]
    q, k, v = qkv.chunk(3, dim=2)
    q = flash_attn.layers.rotary.apply_rotary_emb_torch(q.squeeze(dim=2), cos_in, sin_in)
    k = flash_attn.layers.rotary.apply_rotary_emb_torch(k.squeeze(dim=2), cos_in, sin_in)
    v = v.squeeze(dim=2)
  return q, k, v


def apply_rotary_pos_emb_torchscript(qkv: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
  """TorchScript-friendly rotary application for SDPA/backends without flash-attn.

  Expects qkv shaped (B, S, 3, H, D). Returns transformed qkv.
  """
  return (qkv * cos) + (rotate_half(qkv) * sin)


def apply_rotary_pos_emb(qkv: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
  """In-place rotary application for qkv-packed tensors using flash-attn helper."""
  if flash_attn is None:
    raise RuntimeError("flash_attn is required for rotary qkv application")
  cos = cos[0, :, 0, 0, :cos.shape[-1] // 2]
  sin = sin[0, :, 0, 0, :sin.shape[-1] // 2]
  return flash_attn.layers.rotary.apply_rotary_emb_qkv_(qkv, cos, sin)


def apply_rotary_pos_emb_single(vec: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
  """Apply rotary to a single tensor (q or k) with shape (B, S, H, D)."""
  if flash_attn is None:
    raise RuntimeError("flash_attn is required for rotary single-vector application")
  with torch.amp.autocast('cuda', enabled=False):
    cos = cos.to(vec.dtype)
    sin = sin.to(vec.dtype)
    if vec.shape[1] < cos.shape[1]:
      cos = cos[:, :vec.shape[1]]
      sin = sin[:, :vec.shape[1]]
    if cos.shape[0] == 1:
      cos_in = cos[0, :, 0, 0, :cos.shape[-1] // 2]
      sin_in = sin[0, :, 0, 0, :sin.shape[-1] // 2]
    else:
      cos_in = cos[:, :, 0, 0, :cos.shape[-1] // 2]
      sin_in = sin[:, :, 0, 0, :sin.shape[-1] // 2]
    vec = flash_attn.layers.rotary.apply_rotary_emb_torch(vec, cos_in, sin_in)
  return vec


# -----------------------------------------------------------------------------
# Embeddings and final layer
# -----------------------------------------------------------------------------
class TimestepEmbedder(nn.Module):
  """Embeds scalar timesteps into vector representations."""

  def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
    super().__init__()
    self.mlp = nn.Sequential(
      nn.Linear(frequency_embedding_size, hidden_size, bias=True),
      nn.SiLU(),
      nn.Linear(hidden_size, hidden_size, bias=True))
    self.frequency_embedding_size = frequency_embedding_size

  @staticmethod
  def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000):
    # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
    half = dim // 2
    freqs = torch.exp(
      - math.log(max_period)
      * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
      / half)
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
      embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

  def forward(self, t: torch.Tensor) -> torch.Tensor:
    t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
    t_emb = self.mlp(t_freq)
    return t_emb


class EmbeddingLayer(nn.Module):
  def __init__(self, dim: int, vocab_dim: int):
    super().__init__()
    self.embedding = nn.Parameter(torch.empty((vocab_dim, dim)))
    torch.nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 2:
      return self.embedding[x]
    assert x.ndim == 3
    return torch.einsum(
      "blv,ve->ble",
      torch.nn.functional.softmax(x, dim=-1).float(),
      self.embedding.float()).to(x.dtype)


class LabelEmbedder(nn.Module):
  """Embeds class labels into vector representations."""

  def __init__(self, num_classes, cond_size):
    super().__init__()
    self.embedding_table = nn.Embedding(num_classes + 1, cond_size)
    self.num_classes = num_classes

  def forward(self, labels):
    return self.embedding_table(labels)


class DDiTFinalLayer(nn.Module):
  def __init__(self, hidden_size: int, out_channels: int, cond_dim: int,
               adaLN: bool, tie_word_embeddings: bool = False):
    super().__init__()
    self.norm_final = LayerNorm(hidden_size)
    self.linear = nn.Linear(hidden_size, out_channels)
    self.linear.weight.data.zero_()
    self.linear.bias.data.zero_()
    self.adaLN = adaLN
    if self.adaLN:
      self.adaLN_modulation = nn.Linear(cond_dim, 2 * hidden_size, bias=True)
      self.adaLN_modulation.weight.data.zero_()
      self.adaLN_modulation.bias.data.zero_()
    # tie_word_embeddings is handled by backbones that choose to tie weights
    self.tie_word_embeddings = tie_word_embeddings

  def forward(self, x: torch.Tensor, c: typing.Optional[torch.Tensor]):
    x = self.norm_final(x)
    if c is not None:
      if c.shape[0] == x.shape[0]:
        shift, scale = self.adaLN_modulation(c)[:, None].chunk(2, dim=2)
      else:
        shift, scale = rearrange(
          self.adaLN_modulation(c), '(b h) d -> b h d', b=x.shape[0]).chunk(2, dim=-1)
      x = modulate_fused(x, shift, scale)
    x = self.linear(x)
    return x


def _prepare_ada_modulation(modulation: torch.Tensor, batch_size: int):
  if modulation.shape[0] == batch_size:
    modulated = modulation[:, None]
  else:
    modulated = rearrange(modulation, '(b h) d -> b h d', b=batch_size)
    modulated = modulated[:, None]
  return modulated.chunk(6, dim=2)


class DDiTBlock(nn.Module):
  def __init__(self, dim, n_heads, adaLN,
               cond_dim=None, mlp_ratio=4,
               dropout=0.1, attn_backend='auto'):
    super().__init__()
    self.n_heads = n_heads
    self.adaLN = adaLN
    self.attn_backend = attn_backend

    self.norm1 = LayerNorm(dim)
    self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
    self.attn_out = nn.Linear(dim, dim, bias=False)
    self.dropout1 = nn.Dropout(dropout)

    self.norm2 = LayerNorm(dim)
    self.mlp = nn.Sequential(
      nn.Linear(dim, mlp_ratio * dim, bias=True),
      nn.GELU(approximate='tanh'),
      nn.Linear(mlp_ratio * dim, dim, bias=True))
    self.dropout2 = nn.Dropout(dropout)
    self.dropout = dropout

    if self.adaLN:
      self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim)
      self.adaLN_modulation.weight.data.zero_()
      self.adaLN_modulation.bias.data.zero_()

  def _get_bias_dropout_scale(self):
    if self.training:
      return bias_dropout_add_scale_fused_train
    else:
      return bias_dropout_add_scale_fused_inference

  def _extract_ada_params(self, c, batch_size):
    if not self.adaLN or c is None:
      return None
    modulation = self.adaLN_modulation(c)
    return _prepare_ada_modulation(modulation, batch_size)

  def _apply_attention(self, qkv, rotary_cos_sin, attn_mask):
    cos, sin = rotary_cos_sin
    cos = cos.to(qkv.dtype)
    sin = sin.to(qkv.dtype)
    if self.attn_backend == 'flash_attn' or (self.attn_backend == 'auto' and supports_flash_attention()):
      qkv = apply_rotary_pos_emb(qkv, cos, sin)
      return flash_varlen_attention_qkvpacked(qkv, causal=False)
    # Fallback to SDPA
    qkv = apply_rotary_pos_emb_torchscript(qkv, cos, sin)
    q, k, v = [x.squeeze(2) for x in qkv.chunk(3, dim=2)]
    if attn_mask is not None:
      return sdpa_attention_masked(q, k, v, attn_mask, causal=False)
    return sdpa_attention_unmasked(q, k, v)

  def forward(self, x, rotary_cos_sin, c=None, attn_mask=None):
    bias_dropout_scale_fn = self._get_bias_dropout_scale()
    shift_msa = scale_msa = gate_msa = None
    shift_mlp = scale_mlp = gate_mlp = None
    ada_params = self._extract_ada_params(c, x.shape[0])
    if ada_params is not None:
      (shift_msa, scale_msa, gate_msa,
       shift_mlp, scale_mlp, gate_mlp) = ada_params

    x_skip = x
    x = self.norm1(x)
    if shift_msa is not None:
      x = modulate_fused(x, shift_msa, scale_msa)

    qkv = rearrange(
      self.attn_qkv(x),
      'b s (three h d) -> b s three h d',
      three=3,
      h=self.n_heads)
    x = self._apply_attention(qkv, rotary_cos_sin, attn_mask)

    if gate_msa is not None:
      x = bias_dropout_scale_fn(
        self.attn_out(x), None, gate_msa, x_skip, self.dropout)
      x = bias_dropout_scale_fn(
        self.mlp(modulate_fused(
          self.norm2(x), shift_mlp, scale_mlp)),
        None, gate_mlp, x, self.dropout)
    else:
      scale = torch.ones(1, device=x.device, dtype=x.dtype)
      x = bias_dropout_scale_fn(
        self.attn_out(x), None, scale, x_skip, self.dropout)
      x = bias_dropout_scale_fn(
        self.mlp(self.norm2(x)), None, scale, x, self.dropout)
    return x


class DDiTBlockCausal(nn.Module):
  def __init__(self, dim, n_heads, mlp_ratio=4, dropout=0.1, attn_backend='auto'):
    super().__init__()
    self.n_heads = n_heads
    self.attn_backend = attn_backend

    self.norm1 = LayerNorm(dim)
    self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
    self.attn_out = nn.Linear(dim, dim, bias=False)
    self.dropout1 = nn.Dropout(dropout)

    self.norm2 = LayerNorm(dim)
    self.mlp = nn.Sequential(
      nn.Linear(dim, mlp_ratio * dim, bias=True),
      nn.GELU(approximate='tanh'),
      nn.Linear(mlp_ratio * dim, dim, bias=True))
    self.dropout2 = nn.Dropout(dropout)
    self.dropout = dropout

  def _get_bias_dropout_scale(self):
    if self.training:
      return bias_dropout_add_scale_fused_train
    else:
      return bias_dropout_add_scale_fused_inference

  def _apply_causal_attention(self, qkv, rotary_cos_sin):
    """Apply causal attention with fallback logic."""
    cos, sin = rotary_cos_sin
    cos = cos.to(qkv.dtype)
    sin = sin.to(qkv.dtype)

    # Try flash-attn first
    if self.attn_backend == 'flash_attn' or (self.attn_backend == 'auto' and supports_flash_attention()):
      with torch.amp.autocast('cuda', enabled=False):
        qkv_rotary = apply_rotary_pos_emb(qkv, cos, sin)
      return flash_varlen_attention_qkvpacked(qkv_rotary, causal=True)
    else:
      # Fallback to SDPA
      qkv_rotary = apply_rotary_pos_emb_torchscript(qkv, cos, sin)
      q, k, v = qkv_rotary.chunk(3, dim=2)
      return sdpa_attention(q, k, v, causal=True, dropout_p=0.0)

  def forward(self, x, rotary_cos_sin, **kwargs):
    del kwargs
    bias_dropout_scale_fn = self._get_bias_dropout_scale()
    x_skip = x
    x = self.norm1(x)

    qkv = self.attn_qkv(x)
    qkv = rearrange(
      qkv,
      'b s (three h d) -> b s three h d',
      three=3,
      h=self.n_heads)
    x = self._apply_causal_attention(qkv, rotary_cos_sin)

    scale = torch.ones(1, device=x.device, dtype=x.dtype)
    x = bias_dropout_scale_fn(
      self.attn_out(x), None, scale, x_skip, self.dropout)

    x = bias_dropout_scale_fn(
      self.mlp(self.norm2(x)), None, scale, x, self.dropout)
    return x


__all__ = [
  'supports_flash_attention', 'supports_flex_attention',
  'bias_dropout_add_scale', 'get_bias_dropout_add_scale',
  'bias_dropout_add_scale_fused_train', 'bias_dropout_add_scale_fused_inference',
  'modulate', 'modulate_fused',
  'LayerNorm', 'residual_linear',
  'Rotary', 'rotate_half', 'split_and_apply_rotary_pos_emb',
  'apply_rotary_pos_emb', 'apply_rotary_pos_emb_torchscript', 'apply_rotary_pos_emb_single',
  'TimestepEmbedder', 'EmbeddingLayer', 'LabelEmbedder', 'DDiTBlock', 'DDiTBlockCausal', 'DDiTFinalLayer',
  'sdpa_attention', 'sdpa_attention_unmasked', 'sdpa_attention_masked',
  'flash_varlen_attention_qkvpacked',
]


# -----------------------------------------------------------------------------
# Multi-head attention helpers (centralized)
# -----------------------------------------------------------------------------
def sdpa_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: typing.Optional[torch.Tensor] = None,
    causal: bool = False,
    dropout_p: float = 0.0,
    scale: typing.Optional[float] = None,
) -> torch.Tensor:
  """Scaled dot-product attention over packed heads using torch SDPA.

  Args:
    q, k, v: Tensors shaped (B, S, H, D)
    attn_mask: Optional mask shaped (B, S, S) or broadcastable.
    causal: Whether to use causal masking.
    dropout_p: Dropout probability (training only; kept for API parity).
    scale: Optional scale override (1/sqrt(D)).

  Returns:
    Tensor shaped (B, S, H*D) (flattened heads for downstream linear).
  """
  # torch SDPA expects (B, H, S, D)
  q = q.transpose(1, 2)
  k = k.transpose(1, 2)
  v = v.transpose(1, 2)
  x = F.scaled_dot_product_attention(
    q, k, v,
    attn_mask=attn_mask[:, None] if attn_mask is not None else None,
    dropout_p=dropout_p,
    is_causal=causal,
    scale=scale)
  x = x.transpose(1, 2)  # (B, S, H, D)
  return rearrange(x, 'b s h d -> b s (h d)')


def sdpa_attention_unmasked(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
  """Convenience wrapper for unmasked SDPA returning (B,S,H*D)."""
  return sdpa_attention(q, k, v, attn_mask=None, causal=False, dropout_p=0.0)


def sdpa_attention_masked(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: torch.Tensor,
    causal: bool = False) -> torch.Tensor:
  """Convenience wrapper for masked SDPA returning (B,S,H*D)."""
  return sdpa_attention(q, k, v, attn_mask=attn_mask, causal=causal, dropout_p=0.0)


def flash_varlen_attention_qkvpacked(
    qkv: torch.Tensor,
    causal: bool = False,
    dropout_p: float = 0.0,
) -> torch.Tensor:
  """FlashAttention varlen over packed qkv.

  Args:
    qkv: Tensor shaped (B, S, 3, H, D)
    causal: Whether to apply causal masking.
    dropout_p: Dropout probability (kept for parity; 0.0 for inference/eval).

  Returns:
    Tensor shaped (B, S, H*D)
  """
  if flash_attn is None:
    raise RuntimeError("flash_attn is not available for flash_varlen_attention_qkvpacked")
  bsz, seqlen = qkv.shape[0], qkv.shape[1]
  qkv_flat = rearrange(qkv, 'b s ... -> (b s) ...')
  cu_seqlens = torch.arange(
    0, (bsz + 1) * seqlen, step=seqlen, dtype=torch.int32, device=qkv.device)
  x = flash_attn.flash_attn_interface.flash_attn_varlen_qkvpacked_func(
    qkv_flat, cu_seqlens, seqlen, dropout_p, causal=causal)
  x = rearrange(x, '(b s) h d -> b s (h d)', b=bsz)
  return x
