import math
import typing

import einops
import huggingface_hub
import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # flash-attn is optional but recommended
  import flash_attn
  import flash_attn.layers.rotary
except (ImportError, RuntimeError):
  flash_attn = None  # type: ignore
from .common import (
  bias_dropout_add_scale,
  get_bias_dropout_add_scale,
  bias_dropout_add_scale_fused_train,
  bias_dropout_add_scale_fused_inference,
  modulate,
  modulate_fused,
  Rotary,
  rotate_half,
  split_and_apply_rotary_pos_emb,
  apply_rotary_pos_emb,
  LayerNorm,
  residual_linear,
  TimestepEmbedder,
  EmbeddingLayer,
  LabelEmbedder,
  DDiTBlock,
  DDiTBlockCausal,
  DDiTFinalLayer,
  sdpa_attention_unmasked,
  flash_varlen_attention_qkvpacked,
)

# Flags required to enable jit fusion kernels
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)


## Moved to common: sdpa_attention_unmasked


#################################################################################
#                                  Layers                                       #
#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################


#################################################################################
#                                 Core Model                                    #
#################################################################################

class DIT(nn.Module, huggingface_hub.PyTorchModelHubMixin):
  """Diffusion Transformer (DiT) backbone model.
  
  A Transformer architecture optimized for diffusion, supporting both
  causal (GPT-style) and bidirectional (BERT-style) attention, with
  adaptive layer normalization for time conditioning.
  """
  def __init__(self, config, vocab_size: int):
    """Initialize the DiT model.
    
    Args:
        config: Hydra configuration object containing model hyperparameters.
        vocab_size: Size of the vocabulary.
    """
    super().__init__()
    if type(config) == dict:
      config = omegaconf.OmegaConf.create(config)
    self.causal = config.algo.causal_attention
    self.adaLN = not self.causal
    self.config = config
    self.vocab_size = vocab_size
    dim = config.model.hidden_size
    cond_dim = config.model.cond_dim
    self.vocab_embed = EmbeddingLayer(dim, vocab_size)
    if not self.causal:
      self.sigma_map = TimestepEmbedder(cond_dim)
    self.rotary_emb = Rotary(dim // config.model.n_heads)

    blocks = []
    for _ in range(config.model.n_blocks):
      if self.causal:
        block = DDiTBlockCausal(
          dim=dim,
          n_heads=config.model.n_heads,
          dropout=config.model.dropout)
      else:
        block = DDiTBlock(
          dim=dim,
          n_heads=config.model.n_heads,
          cond_dim=cond_dim,
          adaLN=self.adaLN,
          dropout=config.model.dropout)
      blocks.append(block)
    self.blocks = nn.ModuleList(blocks)

    self.output_layer = DDiTFinalLayer(
      hidden_size=dim,
      out_channels=vocab_size,
      cond_dim=cond_dim,
      adaLN=self.adaLN)
    self.scale_by_sigma = config.model.scale_by_sigma
    # Tie output projection to input embeddings if requested
    if getattr(config.model, 'tie_word_embeddings', False):
      self.output_layer.linear.weight = self.vocab_embed.embedding

  def _get_bias_dropout_scale(self):
    if self.training:
      return bias_dropout_add_scale_fused_train
    else:
      return  bias_dropout_add_scale_fused_inference

  def forward(self, x, sigma, return_hidden_states: bool = False):
    """Forward pass of the DiT.
    
    Args:
        x: Input token indices [batch, seq_len].
        sigma: Noise level/time embedding [batch] or [batch, seq_len].
        return_hidden_states: If True, return hidden states after blocks instead of logits.
        
    Returns:
        Tensor: Logits [batch, seq_len, vocab_size] if return_hidden_states=False,
                otherwise hidden states [batch, seq_len, hidden_size].
    """
    x = self.vocab_embed(x)
    if self.causal:
      t_cond = None
    else:
      t_cond = F.silu(self.sigma_map(sigma))

    rotary_cos_sin = self.rotary_emb(x)

    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
      for i in range(len(self.blocks)):
        x = self.blocks[i](x, rotary_cos_sin, c=t_cond)
      
      if return_hidden_states:
        return x  # Return hidden states after blocks, before output_layer
      
      x = self.output_layer(x, c=t_cond)

    return x
