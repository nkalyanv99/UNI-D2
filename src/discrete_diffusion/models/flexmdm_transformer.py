"""FlexMDM Transformer Model for Any-Order Mask Insertion Flow.

This module implements the transformer architecture for FlexMDM, including
adaptive layer normalization, rotary embeddings, and dual prediction heads
for both token logits and expected gap lengths.
"""

from __future__ import annotations

import math
from typing import Optional

import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

from ..forward_process.flexmdm import ModelPrediction


# Configure JIT for fusion
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)

# Compile flex_attention for performance (dynamic=True for variable batch sizes)
flex_attention = torch.compile(flex_attention, dynamic=True)


#################################################################################
#                          Helper Functions                                      #
#################################################################################


@torch.jit.script
def modulate_fused(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
  """Fused modulation: x * (1 + scale) + shift."""
  return x * (1 + scale) + shift


@torch.jit.script
def bias_dropout_add_scale_fused_train(
  x: Tensor,
  bias: Optional[Tensor],
  scale: Tensor,
  residual: Optional[Tensor],
  prob: float,
) -> Tensor:
  """Fused bias-dropout-add-scale for training."""
  if bias is not None:
    out = scale * F.dropout(x + bias, p=prob, training=True)
  else:
    out = scale * F.dropout(x, p=prob, training=True)
  if residual is not None:
    out = residual + out
  return out


@torch.jit.script
def bias_dropout_add_scale_fused_inference(
  x: Tensor,
  bias: Optional[Tensor],
  scale: Tensor,
  residual: Optional[Tensor],
  prob: float,
) -> Tensor:
  """Fused bias-dropout-add-scale for inference."""
  if bias is not None:
    out = scale * (x + bias)
  else:
    out = scale * x
  if residual is not None:
    out = residual + out
  return out


def rotate_half(x):
  """Rotate half the hidden dims of the input."""
  x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
  return torch.cat((-x2, x1), dim=-1)


@torch.jit.script
def _apply_rotary_pos_emb_torchscript(qkv, cos, sin):
  """Apply rotary positional embeddings (TorchScript fallback)."""
  return (qkv * cos) + (rotate_half(qkv) * sin)


def apply_rotary_pos_emb(qkv, cos, sin):
  """Apply rotary positional embeddings (uses flash_attn if available)."""
  try:
    import flash_attn.layers.rotary

    cos = cos[0, :, 0, 0, : cos.shape[-1] // 2]
    sin = sin[0, :, 0, 0, : sin.shape[-1] // 2]
    return flash_attn.layers.rotary.apply_rotary_emb_qkv_(qkv, cos, sin)
  except ImportError:
    return _apply_rotary_pos_emb_torchscript(qkv, cos, sin)


def get_mask_mod(seq_len: torch.Tensor):
  """Create mask function for variable-length sequences."""
  def mask_mod(b, h, q_idx, kv_idx):
    return (q_idx <= seq_len[b]) & (kv_idx <= seq_len[b])
  return mask_mod


#################################################################################
#                                  Layers                                       #
#################################################################################


class LayerNorm(nn.Module):
  """Layer normalization with learnable scale."""

  def __init__(self, dim):
    super().__init__()
    self.weight = nn.Parameter(torch.ones([dim]))
    self.dim = dim

  def forward(self, x):
    with torch.amp.autocast("cuda", enabled=False):
      x = F.layer_norm(x.float(), [self.dim])
    return x * self.weight[None, None, :]


class Rotary(torch.nn.Module):
  """Rotary positional embeddings."""

  def __init__(self, dim, base=10_000):
    super().__init__()
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    self.register_buffer("inv_freq", inv_freq)
    self.seq_len_cached = None
    self.cos_cached = None
    self.sin_cached = None

  def forward(self, x, seq_dim=1):
    seq_len = x.shape[seq_dim]
    if seq_len != self.seq_len_cached:
      self.seq_len_cached = seq_len
      t = torch.arange(x.shape[seq_dim], device=x.device).type_as(
        self.inv_freq
      )
      freqs = torch.einsum("i,j->ij", t, self.inv_freq.clone())
      emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
      # Dims: batch, seq_len, qkv, head, dim
      self.cos_cached = emb.cos()[None, :, None, None, :].repeat(1, 1, 3, 1, 1)
      self.sin_cached = emb.sin()[None, :, None, None, :].repeat(1, 1, 3, 1, 1)
      # Make transformation on v an identity
      self.cos_cached[:, :, 2, :, :].fill_(1.0)
      self.sin_cached[:, :, 2, :, :].fill_(0.0)

    return self.cos_cached, self.sin_cached


class TimestepEmbedder(nn.Module):
  """Embeds scalar timesteps into vector representations."""

  def __init__(self, hidden_size, frequency_embedding_size=256):
    super().__init__()
    self.mlp = nn.Sequential(
      nn.Linear(frequency_embedding_size, hidden_size, bias=True),
      nn.SiLU(),
      nn.Linear(hidden_size, hidden_size, bias=True),
    )
    self.frequency_embedding_size = frequency_embedding_size

  @staticmethod
  def timestep_embedding(t, dim, max_period=10000):
    """Create sinusoidal timestep embeddings."""
    half = dim // 2
    freqs = torch.exp(
      -math.log(max_period)
      * torch.arange(start=0, end=half, dtype=torch.float32)
      / half
    ).to(device=t.device)
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
      embedding = torch.cat(
        [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
      )
    return embedding

  def forward(self, t):
    t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
    t_emb = self.mlp(t_freq)
    return t_emb


class ScalarLengthHead(nn.Module):
  """Predicts expected gap lengths as scalars."""

  def __init__(
    self, d_model: int, normalized_len: int, cond_dim: Optional[int] = None
  ):
    super().__init__()
    self.has_cond = cond_dim is not None
    if self.has_cond:
      self.adaLN = nn.Linear(cond_dim, 2 * d_model, bias=True)
      self.adaLN.weight.data.zero_()
      self.adaLN.bias.data.zero_()

    self.norm = LayerNorm(d_model)
    self.proj1 = nn.Linear(d_model, d_model)
    self.act = nn.GELU()
    self.proj2 = nn.Linear(d_model, 1)
    self.softplus = nn.Softplus()
    self.normalized_len = normalized_len

  def forward(
    self, x: torch.Tensor, c: Optional[torch.Tensor] = None
  ) -> torch.Tensor:
    x_fp32 = x.float()
    c_fp32 = c.float() if (self.has_cond and c is not None) else None
    if self.has_cond and c_fp32 is not None:
      shift, scale = self.adaLN(c_fp32)[:, None].chunk(2, dim=2)
      x_fp32 = modulate_fused(self.norm(x_fp32), shift, scale)
    else:
      x_fp32 = self.norm(x_fp32)
    s = self.proj2(self.act(self.proj1(x_fp32)))
    out = self.softplus(s).squeeze(-1) * self.normalized_len
    return out.to(x.dtype)


class EmbeddingLayer(nn.Module):
  """Token embedding layer."""

  def __init__(self, dim, vocab_dim):
    super().__init__()
    self.embedding = nn.Parameter(torch.empty((vocab_dim, dim)))
    torch.nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))

  def forward(self, x):
    return self.embedding[x]


#################################################################################
#                          Transformer Blocks                                   #
#################################################################################


class DDiTBlock(nn.Module):
  """Diffusion Transformer block with adaptive layer norm."""

  def __init__(self, dim, n_heads, cond_dim, mlp_ratio=4, dropout=0.1):
    super().__init__()
    self.n_heads = n_heads

    self.norm1 = LayerNorm(dim)
    self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
    self.attn_out = nn.Linear(dim, dim, bias=False)
    self.dropout1 = nn.Dropout(dropout)

    self.norm2 = LayerNorm(dim)
    self.mlp = nn.Sequential(
      nn.Linear(dim, mlp_ratio * dim, bias=True),
      nn.GELU(approximate="tanh"),
      nn.Linear(mlp_ratio * dim, dim, bias=True),
    )
    self.dropout2 = nn.Dropout(dropout)

    self.dropout = dropout

    self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim, bias=True)
    self.adaLN_modulation.weight.data.zero_()
    self.adaLN_modulation.bias.data.zero_()

  def _get_bias_dropout_scale(self):
    return (
      bias_dropout_add_scale_fused_train
      if self.training
      else bias_dropout_add_scale_fused_inference
    )

  def forward(self, x, rotary_cos_sin, c, block_mask):
    batch_size = x.shape[0]

    bias_dropout_scale_fn = self._get_bias_dropout_scale()

    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
      self.adaLN_modulation(c)[:, None].chunk(6, dim=2)
    )

    # Attention operation
    x_skip = x
    x = modulate_fused(self.norm1(x), shift_msa, scale_msa)

    qkv = self.attn_qkv(x)
    qkv = rearrange(
      qkv, "b s (three h d) -> b s three h d", three=3, h=self.n_heads
    )
    with torch.amp.autocast("cuda", enabled=False):
      cos, sin = rotary_cos_sin
      qkv = apply_rotary_pos_emb(qkv, cos.to(qkv.dtype), sin.to(qkv.dtype))

    q, k, v = rearrange(qkv, "b s three h d -> three b h s d", three=3)

    x = flex_attention(q, k, v, block_mask=block_mask)

    x = rearrange(x, "b h s d -> b s (h d)", b=batch_size)

    x = bias_dropout_scale_fn(
      self.attn_out(x), None, gate_msa, x_skip, self.dropout
    )

    # MLP operation
    x = bias_dropout_scale_fn(
      self.mlp(modulate_fused(self.norm2(x), shift_mlp, scale_mlp)),
      None,
      gate_mlp,
      x,
      self.dropout,
    )

    return x


class DDitFinalLayer(nn.Module):
  """Final output layer with adaptive layer norm."""

  def __init__(self, hidden_size, out_channels, cond_dim):
    super().__init__()
    self.norm_final = LayerNorm(hidden_size)
    self.linear = nn.Linear(hidden_size, out_channels)
    self.linear.weight.data.zero_()
    self.linear.bias.data.zero_()

    self.adaLN_modulation = nn.Linear(cond_dim, 2 * hidden_size, bias=True)
    self.adaLN_modulation.weight.data.zero_()
    self.adaLN_modulation.bias.data.zero_()

  def forward(self, x, c):
    shift, scale = self.adaLN_modulation(c)[:, None].chunk(2, dim=2)
    x = modulate_fused(self.norm_final(x), shift, scale)
    x = self.linear(x)
    return x


#################################################################################
#                              Main Model                                       #
#################################################################################


class AnyOrderMaskInsertionFlow(nn.Module):
  """FlexMDM Any-Order Mask Insertion Flow model.
  
  This model predicts both token logits and expected gap lengths for
  the joint insertion-masking process.
  """

  def __init__(self, config, vocab_size: int):
    super().__init__()
    if isinstance(config, dict):
      config = omegaconf.OmegaConf.create(config)
    
    self.config = config
    self.vocab_size = vocab_size
    self.hidden_size = config.model.hidden_size
    self.n_heads = config.model.n_heads
    self.cond_dim = config.model.cond_dim
    self.n_blocks = config.model.n_blocks
    self.dropout = config.model.dropout

    max_length = getattr(config.model, 'max_length', None)
    if max_length is None:
      max_length = getattr(config.model, 'length', None)
    if max_length is None:
      raise ValueError(
        "AnyOrderMaskInsertionFlow requires 'max_length' or 'length' in the model config."
      )
    self.max_length = max_length
    
    # Get special tokens from config
    self.pad_token = getattr(config.model, 'pad_token', 0)
    self.mask_token = getattr(config.model, 'mask_token', None)
    
    # Get loss function type
    self.len_predict_type = getattr(config.model, 'len_predict_type', 'expectation')

    self.vocab_embed = EmbeddingLayer(self.hidden_size, self.vocab_size)
    self.sigma_map = TimestepEmbedder(self.cond_dim)
    self.rotary_emb = Rotary(self.hidden_size // self.n_heads)

    self.blocks = nn.ModuleList(
      [
        DDiTBlock(
          self.hidden_size,
          self.n_heads,
          self.cond_dim,
          dropout=self.dropout,
        )
        for _ in range(self.n_blocks)
      ]
    )

    self.output_layer = DDitFinalLayer(
      self.hidden_size, self.vocab_size, self.cond_dim
    )

    if self.len_predict_type == "distribution":
      self.len_pred = DDitFinalLayer(
        self.hidden_size,
        self.max_length + 1,
        self.cond_dim,
      )
    elif self.len_predict_type == "expectation":
      self.len_pred = ScalarLengthHead(
        self.hidden_size, self.max_length, self.cond_dim
      )
    else:
      raise ValueError(
        f"Invalid length prediction type: {self.len_predict_type}"
      )

  def forward(
    self, indices: torch.Tensor, t: torch.Tensor
  ) -> ModelPrediction:
    """Forward pass.
    
    Args:
      indices: Token indices [B, L]
      t: Timestep [B]
      
    Returns:
      ModelPrediction with token_logits and expected_gaps or length_posterior
    """
    B, L = indices.shape
    
    # Append padding token for length prediction
    indices = torch.cat(
      [
        indices,
        self.pad_token
        * torch.ones((B, 1), device=indices.device, dtype=torch.int64),
      ],
      dim=-1,
    )
    
    seq_lens = (indices != self.pad_token).sum(dim=-1)
    block_mask = create_block_mask(
      get_mask_mod(seq_lens),
      B=B,
      H=None,
      Q_LEN=indices.shape[1],
      KV_LEN=indices.shape[1],
    )

    x = self.vocab_embed(indices)
    c = F.silu(self.sigma_map(t))

    rotary_cos_sin = self.rotary_emb(x)

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
      for i in range(len(self.blocks)):
        x = self.blocks[i](x, rotary_cos_sin, c, block_mask)

      # Token logits (excluding the appended padding position)
      token_logits = self.output_layer(x[:, :-1], c)

      # Length prediction
      if self.len_predict_type == "distribution":
        length_posterior = self.len_pred(x, c)
        return ModelPrediction(
          token_logits=token_logits,
          length_posterior=length_posterior,
        )
      else:  # expectation
        return ModelPrediction(
          token_logits=token_logits,
          expected_gaps=self.len_pred(x, c),
        )


__all__ = ['AnyOrderMaskInsertionFlow']

