"""Unit tests for GStarSampler."""

import pytest
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from discrete_diffusion.sampling.gstar import GStarSampler
from discrete_diffusion.sampling.starshape import StarShapeSampler


class MockNoise:
    """Mock noise schedule."""
    def alpha_t(self, t):
        """Linear noise schedule for testing."""
        return t


class MockModel:
    """Mock model without remasker for testing."""
    
    def __init__(self, vocab_size=100, seq_length=32):
        self.vocab_size = vocab_size
        self.num_tokens = seq_length
        self.mask_id = 0
        self.device = torch.device("cpu")
        self.time_conditioning = False
        self.noise = MockNoise()
        
    def forward(self, x, sigma):
        """Mock forward pass - returns distribution that never predicts mask_id."""
        batch_size, seq_len = x.shape
        log_probs = torch.zeros(batch_size, seq_len, self.vocab_size)
        log_probs[:, :, self.mask_id] = -float('inf')
        return log_probs
    
    def _sigma_from_alphat(self, alpha_t):
        """Mock sigma computation."""
        return 1 - alpha_t


class MockGStarModel(MockModel):
    """Mock GStar model with remasker for testing."""
    
    def __init__(self, vocab_size=100, seq_length=32):
        super().__init__(vocab_size, seq_length)
        # Store the last sigma received for verification
        self._last_sigma = None
        self._last_sampled_x0 = None
    
    def _remasker_forward(self, sampled_x0, sigma):
        """Mock remasker forward - returns logits based on token values.
        
        For testing: tokens with higher values get higher mistake confidence.
        This allows deterministic testing of topk selection.
        """
        self._last_sampled_x0 = sampled_x0
        self._last_sigma = sigma
        
        batch_size, seq_len = sampled_x0.shape
        # Create logits where class 1 (mistake) probability is proportional to token value
        logits = torch.zeros(batch_size, seq_len, 2)
        # Higher token values -> higher mistake logit
        logits[:, :, 1] = sampled_x0.float() / self.vocab_size
        logits[:, :, 0] = 1 - logits[:, :, 1]
        return logits


@pytest.fixture
def mock_config():
    """Create mock config for sampler."""
    config = OmegaConf.create({
        "sampling": {
            "use_float64": False,
            "steps": 10,
            "predictor": "ddpm_cache",
            "inject_bos": False,
        }
    })
    return config


@pytest.fixture
def mock_model():
    """Create mock model without remasker."""
    return MockModel()


@pytest.fixture
def mock_gstar_model():
    """Create mock GStar model with remasker."""
    return MockGStarModel()


def test_gstar_sampler_inherits_from_starshape(mock_config):
    """GStarSampler should inherit from StarShapeSampler."""
    sampler = GStarSampler(mock_config, t_on=0.1)
    assert isinstance(sampler, StarShapeSampler)


def test_gstar_sampler_initialization(mock_config):
    """Test GStarSampler initializes with correct parameters."""
    sampler = GStarSampler(mock_config, t_on=0.2, remasker_schedule="plato")
    
    assert sampler.t_on == 0.2
    assert sampler.remasker_schedule == "plato"


def test_gstar_sampler_raises_without_remasker(mock_config, mock_model):
    """GStarSampler should raise error when model lacks _remasker_forward."""
    sampler = GStarSampler(mock_config, t_on=0.1)
    
    sampled_x0 = torch.randint(1, mock_model.vocab_size, (2, 32))
    t = torch.tensor([[0.05], [0.05]], dtype=torch.float32)
    
    with pytest.raises(ValueError, match="GStarSampler requires a GStar model"):
        sampler._get_mistake_confidences(mock_model, sampled_x0, t)


def test_gstar_sampler_get_mistake_confidences_shape(mock_config, mock_gstar_model):
    """_get_mistake_confidences should return correct shape."""
    sampler = GStarSampler(mock_config, t_on=0.1)
    
    batch_size = 4
    seq_length = 32
    sampled_x0 = torch.randint(1, mock_gstar_model.vocab_size, (batch_size, seq_length))
    t = torch.tensor([[0.05]] * batch_size, dtype=torch.float32)
    
    confidences = sampler._get_mistake_confidences(mock_gstar_model, sampled_x0, t)
    
    assert confidences.shape == (batch_size, seq_length), \
        f"Expected shape {(batch_size, seq_length)}, got {confidences.shape}"


def test_gstar_sampler_get_mistake_confidences_values(mock_config, mock_gstar_model):
    """_get_mistake_confidences should return softmax class 1 probabilities."""
    sampler = GStarSampler(mock_config, t_on=0.1)
    
    batch_size = 2
    seq_length = 8
    # Create tokens with known values
    sampled_x0 = torch.arange(seq_length).unsqueeze(0).expand(batch_size, -1)
    t = torch.tensor([[0.05]] * batch_size, dtype=torch.float32)
    
    confidences = sampler._get_mistake_confidences(mock_gstar_model, sampled_x0, t)
    
    # Confidences should be in [0, 1] (softmax outputs)
    assert (confidences >= 0).all() and (confidences <= 1).all(), \
        "Confidences should be probabilities in [0, 1]"
    
    # Higher token values should have higher confidence (per mock implementation)
    for i in range(batch_size):
        for j in range(seq_length - 1):
            # Token j+1 has higher value than token j, so should have higher confidence
            assert confidences[i, j] <= confidences[i, j + 1], \
                f"Higher token values should have higher confidence"


def test_gstar_sampler_calls_remasker_with_correct_args(mock_config, mock_gstar_model):
    """_get_mistake_confidences should call remasker with correct sigma."""
    sampler = GStarSampler(mock_config, t_on=0.1)
    
    batch_size = 2
    seq_length = 32
    sampled_x0 = torch.randint(1, mock_gstar_model.vocab_size, (batch_size, seq_length))
    t = torch.tensor([[0.2], [0.2]], dtype=torch.float32)
    
    sampler._get_mistake_confidences(mock_gstar_model, sampled_x0, t)
    
    # Verify remasker was called with sampled_x0
    assert mock_gstar_model._last_sampled_x0 is not None
    assert torch.equal(mock_gstar_model._last_sampled_x0, sampled_x0)
    
    # Verify sigma computation: sigma = 1 - alpha_t = 1 - t (linear schedule)
    expected_sigma = 1 - t
    assert torch.allclose(mock_gstar_model._last_sigma, expected_sigma)


def test_gstar_sampler_masks_highest_confidence_tokens(mock_config, mock_gstar_model):
    """GStarSampler should mask tokens with highest remasker confidence."""
    t_on = 0.3
    sampler = GStarSampler(mock_config, t_on=t_on)
    
    batch_size = 2
    seq_length = 32
    x = torch.randint(1, mock_gstar_model.vocab_size, (batch_size, seq_length), dtype=torch.long)
    
    t = torch.tensor([[0.2], [0.2]], dtype=torch.float32)  # t < t_on
    dt = 0.1
    
    p_x0, out = sampler.compute_posterior(
        mock_gstar_model, x, t, dt, p_x0=None, noise_removal_step=False
    )
    
    # Check the number of masked positions
    alpha_s = t - dt  # Linear schedule: alpha_s = t - dt
    expected_num_masks = int(torch.floor(seq_length * (1 - alpha_s[0, 0])).item())
    
    for i in range(batch_size):
        actual_num_masked = (out[i] == mock_gstar_model.mask_id).sum().item()
        assert actual_num_masked == expected_num_masks, \
            f"Batch {i}: expected {expected_num_masks} masked positions, got {actual_num_masked}"
    
    # Verify that the sampled_x0 passed to remasker was recorded
    assert mock_gstar_model._last_sampled_x0 is not None, \
        "Remasker should have been called with sampled_x0"


def test_gstar_sampler_phase1_matches_mdlm(mock_config, mock_gstar_model):
    """When t > t_on, GStarSampler should use MDLM masking (Phase 1)."""
    from discrete_diffusion.sampling.absorbing import AbsorbingSampler
    
    t_on = 0.1
    sampler_mdlm = AbsorbingSampler(mock_config)
    sampler_gstar = GStarSampler(mock_config, t_on=t_on)
    
    batch_size = 2
    seq_length = 32
    x = torch.full((batch_size, seq_length), mock_gstar_model.mask_id, dtype=torch.long)
    x[:, :5] = torch.randint(1, mock_gstar_model.vocab_size, (batch_size, 5))
    
    t = torch.tensor([[0.5], [0.5]], dtype=torch.float32)  # t > t_on
    dt = 0.1
    
    torch.manual_seed(42)
    p_x0_mdlm, out_mdlm = sampler_mdlm.compute_posterior(
        mock_gstar_model, x.clone(), t, dt, p_x0=None, noise_removal_step=False
    )
    
    torch.manual_seed(42)
    p_x0_gstar, out_gstar = sampler_gstar.compute_posterior(
        mock_gstar_model, x.clone(), t, dt, p_x0=None, noise_removal_step=False
    )
    
    # Results should be identical in Phase 1
    assert torch.allclose(p_x0_mdlm, p_x0_gstar)
    assert torch.equal(out_mdlm, out_gstar)


def test_gstar_sampler_plato_mode(mock_config, mock_gstar_model):
    """GStarSampler plato mode should maintain fixed mask ratio."""
    t_on = 0.3
    sampler = GStarSampler(mock_config, t_on=t_on, remasker_schedule="plato")
    
    batch_size = 2
    seq_length = 32
    
    # Expected masks based on alpha(t_on) = 0.3
    alpha_t_on = mock_gstar_model.noise.alpha_t(torch.tensor([[t_on]]))
    expected_num_masked = int(torch.floor(seq_length * (1 - alpha_t_on[0, 0])).item())
    
    # Test at multiple timesteps < t_on
    for t_val in [0.2, 0.1, 0.05]:
        x = torch.randint(1, mock_gstar_model.vocab_size, (batch_size, seq_length))
        t = torch.tensor([[t_val], [t_val]], dtype=torch.float32)
        dt = 0.05
        
        p_x0, out = sampler.compute_posterior(
            mock_gstar_model, x, t, dt, p_x0=None, noise_removal_step=False
        )
        
        for i in range(batch_size):
            actual_num_masked = (out[i] == mock_gstar_model.mask_id).sum().item()
            assert actual_num_masked == expected_num_masked, \
                f"At t={t_val}: expected {expected_num_masked} masked, got {actual_num_masked}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

