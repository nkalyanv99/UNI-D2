"""Unit tests for DIT model modifications."""

import pytest
import torch
from hydra import initialize, compose


def test_dit_return_hidden_states():
    """Test that return_hidden_states=True returns correct shape."""
    with initialize(config_path="../../configs", version_base="1.3"):
        cfg = compose(config_name="config", overrides=[
            "model=small",
            "algo=mdlm",
            "data=openwebtext-split"
        ])
        
        from discrete_diffusion.models.dit import DIT
        
        vocab_size = 100
        model = DIT(cfg, vocab_size)
        
        batch_size = 2
        seq_len = 32
        x = torch.randint(0, vocab_size, (batch_size, seq_len))
        sigma = torch.randn(batch_size)
        
        # Normal forward
        logits = model(x, sigma, return_hidden_states=False)
        assert logits.shape == (batch_size, seq_len, vocab_size)
        
        # Hidden states forward
        hidden = model(x, sigma, return_hidden_states=True)
        assert hidden.shape == (batch_size, seq_len, cfg.model.hidden_size)


def test_dit_default_behavior_unchanged():
    """Test that default behavior (return_hidden_states=False) is unchanged."""
    with initialize(config_path="../../configs", version_base="1.3"):
        cfg = compose(config_name="config", overrides=[
            "model=small",
            "algo=mdlm"
        ])
        
        from discrete_diffusion.models.dit import DIT
        
        vocab_size = 100
        model = DIT(cfg, vocab_size)
        
        batch_size = 2
        seq_len = 32
        x = torch.randint(0, vocab_size, (batch_size, seq_len))
        sigma = torch.randn(batch_size)
        
        # Default should return logits
        output = model(x, sigma)
        assert output.shape == (batch_size, seq_len, vocab_size)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


