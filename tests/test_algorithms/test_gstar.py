"""Unit tests for GStar algorithm."""

import pytest
import torch
from hydra import initialize, compose

from discrete_diffusion.evaluations import GStarMetrics


@pytest.fixture
def gstar_model():
    """Create GStar model using Hydra configs."""
    with initialize(config_path="../../configs", version_base="1.3"):
        cfg = compose(config_name="config", overrides=[
            "model=small",
            "model.length=32",
            "model.n_blocks=2",  # Smaller for faster tests
            "algo=gstar",
            "data=openwebtext-split",
            "training.ema=0.0"  # Disable EMA for tests
        ])
        
        from discrete_diffusion.data import get_tokenizer
        from discrete_diffusion.algorithms.gstar import GStar
        
        tokenizer = get_tokenizer(cfg)
        model = GStar(cfg, tokenizer)
        return model, cfg


def test_gstar_initialization(gstar_model):
    """Test that GStar initializes correctly with frozen backbone."""
    model, cfg = gstar_model
    
    # Check that backbone parameters are frozen
    for param in model.backbone.parameters():
        assert not param.requires_grad, "Backbone should be frozen"
    
    # Check that noise parameters are frozen
    for param in model.noise.parameters():
        assert not param.requires_grad, "Noise schedule should be frozen"
    
    # Check that remasker_head exists and is trainable
    assert hasattr(model, 'remasker_head')
    for param in model.remasker_head.parameters():
        assert param.requires_grad, "Remasker head should be trainable"


def test_gstar_get_parameters(gstar_model):
    """Test that _get_parameters only returns remasker_head params."""
    model, cfg = gstar_model
    
    params = list(model._get_parameters())
    remasker_params = list(model.remasker_head.parameters())
    
    assert len(params) == len(remasker_params)
    assert len(params) > 0, "Should have trainable parameters"


def test_gstar_remasker_forward(gstar_model):
    """Test _remasker_forward returns correct shape."""
    model, cfg = gstar_model
    model.eval()
    
    batch_size = 2
    seq_len = 32
    sampled_x0 = torch.randint(0, model.vocab_size, (batch_size, seq_len))
    sigma = torch.randn(batch_size, seq_len)
    
    with torch.no_grad():
        remasker_logits = model._remasker_forward(sampled_x0, sigma)
    
    # Should return [batch, seq_len, 2] for binary classification
    assert remasker_logits.shape == (batch_size, seq_len, 2)


def test_gstar_nll_per_token_shape(gstar_model):
    """Test that nll_per_token returns correct shape."""
    model, cfg = gstar_model
    model.eval()
    
    batch_size = 2
    seq_len = 32
    vocab_size = model.vocab_size
    
    # Create mock inputs
    log_x_theta = torch.randn(batch_size, seq_len, vocab_size).log_softmax(dim=-1)
    xt = torch.randint(0, vocab_size, (batch_size, seq_len))
    x0 = torch.randint(0, vocab_size, (batch_size, seq_len))
    alpha_t = torch.rand(batch_size, 1)
    dalpha_t = torch.rand(batch_size, 1)
    
    with torch.no_grad():
        loss = model.nll_per_token(log_x_theta, xt, x0, alpha_t, dalpha_t, low_var=False)
    
    assert loss.shape == (batch_size, seq_len)
    assert not torch.isnan(loss).any()
    assert not torch.isinf(loss).any()
    assert (loss >= 0).all(), "CE loss should be non-negative"


def test_gstar_metrics_type(gstar_model):
    """Test that GStar uses GStarMetrics."""
    model, cfg = gstar_model
    
    assert isinstance(model.metrics, GStarMetrics), "GStar should use GStarMetrics"


def test_gstar_stores_predictions(gstar_model):
    """Test that nll_per_token stores predictions and targets."""
    model, cfg = gstar_model
    model.eval()
    
    batch_size = 2
    seq_len = 32
    vocab_size = model.vocab_size
    
    # Create mock inputs
    log_x_theta = torch.randn(batch_size, seq_len, vocab_size).log_softmax(dim=-1)
    xt = torch.randint(0, vocab_size, (batch_size, seq_len))
    x0 = torch.randint(0, vocab_size, (batch_size, seq_len))
    alpha_t = torch.rand(batch_size, 1)
    dalpha_t = torch.rand(batch_size, 1)
    
    with torch.no_grad():
        loss = model.nll_per_token(log_x_theta, xt, x0, alpha_t, dalpha_t, low_var=False)
    
    # Check that predictions and targets are stored
    assert model._last_preds is not None, "Predictions should be stored"
    assert model._last_targets is not None, "Targets should be stored"
    assert model._last_preds.shape == (batch_size, seq_len)
    assert model._last_targets.shape == (batch_size, seq_len)
    assert model._last_preds.dtype == torch.long
    assert model._last_targets.dtype == torch.long


def test_gstar_classification_metrics(gstar_model):
    """Test that classification metrics are computed correctly."""
    model, cfg = gstar_model
    model.eval()
    
    batch_size = 2
    seq_len = 8
    vocab_size = model.vocab_size
    
    # Create mock inputs
    log_x_theta = torch.randn(batch_size, seq_len, vocab_size).log_softmax(dim=-1)
    xt = torch.randint(0, vocab_size, (batch_size, seq_len))
    x0 = torch.randint(0, vocab_size, (batch_size, seq_len))
    alpha_t = torch.rand(batch_size, 1)
    dalpha_t = torch.rand(batch_size, 1)
    
    # Reset metrics first
    model.metrics.reset()
    
    with torch.no_grad():
        loss = model.nll_per_token(log_x_theta, xt, x0, alpha_t, dalpha_t, low_var=False)
    
    # Update metrics manually (simulating training_step)
    from discrete_diffusion.algorithms.base import Loss
    losses = Loss(loss=loss.mean(), nlls=loss.sum(), num_tokens=torch.tensor(batch_size * seq_len))
    model._update_train_metrics(losses)
    
    # Compute metrics
    train_metrics = model.metrics.compute_train()
    
    # Check that metrics are computed
    assert 'train/accuracy' in train_metrics
    assert 'train/precision' in train_metrics
    assert 'train/recall' in train_metrics
    assert 'train/f1' in train_metrics
    
    # All metrics should be tensors
    for v in train_metrics.values():
        assert isinstance(v, torch.Tensor)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

