"""GStar: Training a remasker to detect MDLM prediction errors."""

import itertools
from copy import deepcopy

import torch
import torch.nn.functional as F

from .mdlm import MDLM
from ..evaluations import GStarMetrics
from ..forward_process.utils import sample_categorical
from ..models.common import DDiTFinalLayer


class GStar(MDLM):
    """GStar meta-learning algorithm.
    
    Freezes a pretrained MDLM backbone and trains a binary classifier
    (remasker_head) to predict whether MDLM's sampled prediction differs
    from the ground truth.
    
    Training flow:
    1. Standard MDLM forward: xt -> log_x_theta
    2. Sample: sampled_x0 = sample_categorical(exp(log_x_theta))
    3. Remasker forward: sampled_x0 -> hidden_states -> remasker_logits
    4. Loss: CE(remasker_logits, sampled_x0 != x0)
    """
    
    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)
        
        # Store freeze_backbone config
        self.freeze_backbone = config.algo.freeze_backbone
        
        # Always freeze MDLM backbone (used for forward pass to get log_x_theta)
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.noise.parameters():
            param.requires_grad = False
        
        # Create remasker backbone:
        # - If freeze_backbone=True: share with MDLM backbone (no extra params)
        # - If freeze_backbone=False: clone backbone for remasker (separate trainable copy)
        if self.freeze_backbone:
            self.remasker_backbone = self.backbone  # Same object, frozen
        else:
            self.remasker_backbone = deepcopy(self.backbone)  # Separate trainable copy
            # Re-enable gradients (backbone was frozen before deepcopy)
            for param in self.remasker_backbone.parameters():
                param.requires_grad = True
            # Freeze only output_layer (we use remasker_head instead)
            for param in self.remasker_backbone.output_layer.parameters():
                param.requires_grad = False
        
        # Initialize remasker head (binary classification)
        hidden_size = config.model.hidden_size
        cond_dim = config.model.cond_dim
        adaLN = not config.algo.causal_attention
        
        self.remasker_head = DDiTFinalLayer(
            hidden_size=hidden_size,
            out_channels=2,  # Binary: correct (0) vs error (1)
            cond_dim=cond_dim,
            adaLN=adaLN
        )
        
        # Storage for predictions and targets (set in nll_per_token)
        self._last_preds = None
        self._last_targets = None
        
    def _initialize_metrics(self):
        """Override to use GStar-specific binary classification metrics."""
        self.metrics = GStarMetrics()
    
    def _update_train_metrics(self, losses):
        """Update train metrics with batch predictions and targets."""
        if self._last_preds is not None and self._last_targets is not None:
            self.metrics.update_train(self._last_preds, self._last_targets)
    
    def _update_valid_metrics(self, losses):
        """Update valid metrics with batch predictions and targets."""
        if self._last_preds is not None and self._last_targets is not None:
            self.metrics.update_valid(self._last_preds, self._last_targets)
    
    def _log_train_epoch_metrics(self):
        """Log train metrics at epoch end."""
        metrics = self.metrics.compute_train()
        self.log_dict(metrics, on_step=False, on_epoch=True, sync_dist=True)
    
    def _log_valid_epoch_metrics(self):
        """Log validation metrics at epoch end."""
        metrics = self.metrics.compute_valid()
        self.log_dict(metrics, on_step=False, on_epoch=True, sync_dist=True)
        
    def _get_parameters(self):
        """Return parameters for optimization based on freeze_backbone config."""
        if self.freeze_backbone:
            return self.remasker_head.parameters()
        else:
            # Train remasker_backbone (excluding output_layer) + remasker_head
            remasker_backbone_params = [
                p for name, p in self.remasker_backbone.named_parameters()
                if not name.startswith('output_layer.') and p.requires_grad
            ]
            return itertools.chain(
                remasker_backbone_params,
                self.remasker_head.parameters()
            )
    
    def forward(self, xt, sigma, group_idxs=None):
        """Override parent forward for MDLM predictions.
        
        When freeze_backbone=True, use no_grad since we don't need backbone gradients.
        When freeze_backbone=False, allow gradients so DDP sees the parameters as used.
        (The actual gradient flow to loss happens via _remasker_forward, but DDP
        tracks usage during forward().)
        """
        with torch.no_grad():
            return super().forward(xt, sigma, group_idxs)
    
    def _remasker_forward(self, sampled_x0, sigma):
        """Forward pass that returns remasker logits.
        
        Takes MDLM's sampled predictions, passes them through the remasker_backbone
        to get hidden states, then through the remasker head.
        
        Args:
            sampled_x0: Sampled tokens from MDLM [batch, seq_len]
            sigma: Noise level [batch, seq_len] or [batch]
        
        Returns:
            Tensor: Remasker logits [batch, seq_len, 2]
        """
        # Process sigma same way as parent class
        sigma_processed = self._process_sigma(sigma)
        
        # Get hidden states from remasker_backbone
        # When freeze_backbone=True, remasker_backbone is same as backbone (frozen)
        # When freeze_backbone=False, remasker_backbone is a separate trainable copy
        context = torch.no_grad() if self.freeze_backbone else torch.enable_grad()
        with context:
            with torch.amp.autocast('cuda', dtype=torch.float32):
                hidden_states = self.remasker_backbone(sampled_x0, sigma_processed, return_hidden_states=True)
            if self.remasker_backbone.causal:
                t_cond = None
            else:
                t_cond = F.silu(self.remasker_backbone.sigma_map(sigma_processed))
        
        # Apply remasker head (trainable)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            remasker_logits = self.remasker_head(hidden_states, c=t_cond)
        
        return remasker_logits
    
    def nll_per_token(self, log_x_theta, xt, x0, alpha_t, dalpha_t, low_var=False):
        """Compute remasker CE loss instead of MDLM NLL.
        
        Overrides parent nll_per_token to:
        1. Sample x0 from MDLM log probabilities
        2. Get remasker predictions on sampled_x0
        3. Compute binary CE loss (no rescheduling)
        
        Args:
            log_x_theta: Log-probabilities from MDLM [batch, seq_len, vocab]
            xt: Noisy input (not used in GStar)
            x0: Ground truth tokens [batch, seq_len]
            alpha_t: Schedule value (not used - no rescheduling)
            dalpha_t: Schedule derivative (not used - no rescheduling)
            low_var: Low variance flag (not used)
        
        Returns:
            Tensor: Raw per-token CE loss [batch, seq_len]
        """
        # Ignore xt, alpha_t, dalpha_t, low_var - we don't need them
        
        # Sample x0 from MDLM predictions
        with torch.no_grad():
            sampled_x0 = sample_categorical(log_x_theta.exp())
        
        # Compute sigma for remasker (reconstruct from alpha_t)
        sigma = self._sigma_from_alphat(alpha_t)
        
        # Get remasker predictions
        remasker_logits = self._remasker_forward(sampled_x0, sigma)
        
        # Compute binary targets: 1 if error, 0 if correct
        targets = (sampled_x0 != x0).long()  # [batch, seq_len]
        
        # Get predictions (argmax of logits)
        preds = remasker_logits.argmax(dim=-1)  # [batch, seq_len]
        
        # Store predictions and targets for metric computation
        self._last_preds = preds.detach()
        self._last_targets = targets.detach()
        
        # Cross-entropy loss per token (raw, no rescheduling)
        # remasker_logits: [batch, seq_len, 2]
        # targets: [batch, seq_len]
        ce_loss = F.cross_entropy(
            remasker_logits.transpose(1, 2),  # [batch, 2, seq_len]
            targets,
            reduction='none'
        )  # [batch, seq_len]
        
        # Return raw CE loss (no weighting by dalpha_t/(1-alpha_t))
        return ce_loss

