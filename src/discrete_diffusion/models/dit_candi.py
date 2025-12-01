"""DiT model architecture adapted for continuous-discrete hybrid diffusion."""
import math
import pdb
import typing

import huggingface_hub
import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F

# Flags required to enable jit fusion kernels
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)

from .common import (
    bias_dropout_add_scale_fused_train,
    bias_dropout_add_scale_fused_inference,
    Rotary,
    TimestepEmbedder,
    DDiTBlock,
    DDiTBlockCausal,
    DDiTFinalLayer,
)


class EmbeddingLayer(nn.Module):
    def __init__(self, dim, vocab_dim):
        super().__init__()
        self.dim = dim

        self.embedding = nn.Parameter(torch.empty((vocab_dim, dim)))
        torch.nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))

    def forward(self, x):

        # Just for now 
        normalized_embeddings = F.normalize(self.embedding[:-1], p=2, dim=-1)
        # normalized_embeddings = normalized_embeddings * math.sqrt(self.dim)
        
        # normalized_embeddings = self.embedding[:-1]

        if x.ndim == 2:
            return normalized_embeddings[x]
        assert x.ndim == 3

        return torch.einsum(
            "blv,ve->ble",
            x.float(),
            normalized_embeddings.float(),
        ).to(x.dtype)


# assumes that the vocab size is counting the mask token
class DIT_CANDI(nn.Module, huggingface_hub.PyTorchModelHubMixin):
    def __init__(self, 
                 config, 
                 vocab_size: int, 
                 mixed_coeff: float=.5):
        super().__init__()
        if type(config) == dict:
            config = omegaconf.OmegaConf.create(config)
        self.causal = False
        self.adaLN = True
        self.config = config.model
        self.vocab_size = vocab_size
        dim = config.model.hidden_size
        self.dim=dim
        cond_dim = config.model.cond_dim
        self.vocab_embed = EmbeddingLayer(dim,vocab_size)
        self.mask_index = vocab_size


        if not self.causal:
            self.sigma_map = TimestepEmbedder(cond_dim)
        self.rotary_emb = Rotary(dim // config.model.n_heads)
        self.mixed_coeff = mixed_coeff


        blocks = []
        for _ in range(config.model.n_blocks):
            if self.causal:
                block = DDiTBlockCausal(
                    dim=dim, n_heads=config.model.n_heads, dropout=config.model.dropout
                )
            else:
                block = DDiTBlock(
                    dim=dim,
                    n_heads=config.model.n_heads,
                    cond_dim=cond_dim,
                    adaLN=self.adaLN,
                    dropout=config.model.dropout,
                )
            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)

        self.output_layer = DDiTFinalLayer(
            hidden_size=dim,
            out_channels=vocab_size,
            cond_dim=cond_dim,
            adaLN=self.adaLN,
        )

    def _get_bias_dropout_scale(self):
        if self.training:
            return bias_dropout_add_scale_fused_train
        else:
            return bias_dropout_add_scale_fused_inference
        
    def forward(self, xt, discrete_noise, reveal_mask, continuous_noise, **kwargs):
        x = xt 
        sigma = -1 * torch.log(1 - discrete_noise)
        c_in = 1 / (1 + continuous_noise ** 2) ** .5

        x = self.vocab_embed(x)

        special = self.vocab_embed.embedding[-1].view(1, 1, -1).expand_as(x)
        mask = reveal_mask.unsqueeze(-1).float()
        coeffs = torch.ones(mask.size(0), device=mask.device)[:, None, None] * self.mixed_coeff 

        # input preconditioning 
        # for noisy positions, add corruption bias and rescale by c_in
        x = x * mask + (1 - mask) * (coeffs * special / special.norm(dim=-1, keepdim=True)+ (1 - coeffs) * x * c_in)

        t_cond = F.silu(self.sigma_map(sigma))

        rotary_cos_sin = self.rotary_emb(x)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            for i in range(len(self.blocks)):
                x = self.blocks[i](x, rotary_cos_sin, c=t_cond)
            logits = self.output_layer(x, c=t_cond)

        # do not return the mask token logits
        return logits[:, :, :-1]
    
    def get_embedding(self, x): 
        return self.vocab_embed(x)
    
    
__all__ = ["DIT_CANDI"]
