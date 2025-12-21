"""Unit tests for StarShape sampler."""

import pytest
import torch
from omegaconf import OmegaConf

from discrete_diffusion.sampling.starshape import StarShapeSampler
from discrete_diffusion.sampling.absorbing import AbsorbingSampler


class MockModel:
    """Mock model for testing samplers."""
    
    class MockNoise:
        """Mock noise schedule."""
        def alpha_t(self, t):
            """Linear noise schedule for testing."""
            return t
    
    def __init__(self, vocab_size=100, seq_length=32):
        self.vocab_size = vocab_size
        self.num_tokens = seq_length
        self.mask_id = 0
        self.device = torch.device("cpu")
        self.time_conditioning = False
        self.noise = self.MockNoise()
        
    def forward(self, x, sigma):
        """Mock forward pass - returns distribution that never predicts mask_id."""
        batch_size, seq_len = x.shape
        # Return log probabilities with 0 probability for mask_id
        log_probs = torch.zeros(batch_size, seq_len, self.vocab_size)
        log_probs[:, :, self.mask_id] = -float('inf')  # Never predict mask_id
        return log_probs
    
    def _sigma_from_alphat(self, alpha_t):
        """Mock sigma computation."""
        return 1 - alpha_t


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
    """Create mock model."""
    return MockModel()


def test_starshape_phase1_matches_mdlm(mock_config, mock_model):
    """When t > t_on, behavior should match MDLM exactly."""
    t_on = 0.1
    sampler_mdlm = AbsorbingSampler(mock_config)
    sampler_starshape = StarShapeSampler(mock_config, t_on=t_on)
    
    # Create test data
    batch_size = 2
    seq_length = 32
    x = torch.full((batch_size, seq_length), mock_model.mask_id, dtype=torch.long)
    # Unmask a few positions
    x[:, :5] = torch.randint(1, mock_model.vocab_size, (batch_size, 5))
    
    # Test with t > t_on (should use MDLM behavior)
    t = torch.tensor([[0.5], [0.5]], dtype=torch.float32)  # t=0.5 > t_on=0.1
    dt = 0.1
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    p_x0_mdlm, out_mdlm = sampler_mdlm.compute_posterior(
        mock_model, x.clone(), t, dt, p_x0=None, noise_removal_step=False
    )
    
    torch.manual_seed(42)
    p_x0_star, out_star = sampler_starshape.compute_posterior(
        mock_model, x.clone(), t, dt, p_x0=None, noise_removal_step=False
    )
    
    # Results should be identical in Phase 1
    assert torch.allclose(p_x0_mdlm, p_x0_star)
    assert torch.equal(out_mdlm, out_star)


def test_starshape_phase2_masks_correctly(mock_config, mock_model):
    """When t <= t_on, should mask exact number of tokens."""
    t_on = 0.3
    sampler = StarShapeSampler(mock_config, t_on=t_on)
    
    batch_size = 2
    seq_length = 32
    x = torch.randint(1, mock_model.vocab_size, (batch_size, seq_length), dtype=torch.long)
    
    # Test with t <= t_on (should use StarShape behavior)
    t = torch.tensor([[0.2], [0.2]], dtype=torch.float32)  # t=0.2 < t_on=0.3
    dt = 0.1
    
    torch.manual_seed(42)
    p_x0, out = sampler.compute_posterior(
        mock_model, x, t, dt, p_x0=None, noise_removal_step=False
    )
    
    # Check that the correct number of tokens are masked
    alpha_s = mock_model.noise.alpha_t(t - dt)
    expected_num_masked = torch.floor(seq_length * (1 - alpha_s)).long()
    
    for i in range(batch_size):
        actual_num_masked = (out[i] == mock_model.mask_id).sum().item()
        assert actual_num_masked == expected_num_masked[i, 0].item()


def test_starshape_final_step(mock_config, mock_model):
    """Final step should mask 0 tokens."""
    sampler = StarShapeSampler(mock_config, t_on=0.1)
    
    batch_size = 2
    seq_length = 32
    x = torch.full((batch_size, seq_length), mock_model.mask_id, dtype=torch.long)
    
    # Final step with noise_removal_step=True
    t = torch.tensor([[0.05], [0.05]], dtype=torch.float32)  # t < t_on (Phase 2)
    dt = 0.05
    
    torch.manual_seed(42)
    p_x0, out = sampler.compute_posterior(
        mock_model, x, t, dt, p_x0=None, noise_removal_step=True
    )
    
    # No tokens should be masked in final step
    for i in range(batch_size):
        assert (out[i] == mock_model.mask_id).sum().item() == 0


def test_starshape_t_on_boundary(mock_config, mock_model):
    """Test behavior at the t_on boundary."""
    t_on = 0.3
    sampler = StarShapeSampler(mock_config, t_on=t_on)
    
    batch_size = 2
    seq_length = 32
    x = torch.full((batch_size, seq_length), mock_model.mask_id, dtype=torch.long)
    
    # Test just below t_on (should definitely use StarShape)
    t = torch.tensor([[t_on - 0.01], [t_on - 0.01]], dtype=torch.float32)
    dt = 0.1
    
    torch.manual_seed(42)
    p_x0, out = sampler.compute_posterior(
        mock_model, x, t, dt, p_x0=None, noise_removal_step=False
    )
    
    # Should use StarShape masking (Phase 2) since t < t_on
    alpha_s = mock_model.noise.alpha_t(t - dt)
    expected_num_masked = torch.floor(seq_length * (1 - alpha_s)).long()
    
    for i in range(batch_size):
        actual_num_masked = (out[i] == mock_model.mask_id).sum().item()
        assert actual_num_masked == expected_num_masked[i, 0].item(), \
            f"Expected {expected_num_masked[i, 0].item()} masked tokens, got {actual_num_masked}"


def test_starshape_plato_mode_fixed_mask_ratio(mock_config, mock_model):
    """Plato mode should maintain fixed mask ratio at alpha(t_on)."""
    t_on = 0.3
    sampler = StarShapeSampler(mock_config, t_on=t_on, remasker_schedule="plato")
    
    batch_size = 2
    seq_length = 32
    x = torch.randint(1, mock_model.vocab_size, (batch_size, seq_length), dtype=torch.long)
    
    # Expected masks based on alpha(t_on) = t_on = 0.3 (linear schedule)
    # num_tokens_to_mask = floor(32 * (1 - 0.3)) = floor(32 * 0.7) = 22
    alpha_t_on = mock_model.noise.alpha_t(torch.tensor([[t_on]]))
    expected_num_masked = int(torch.floor(seq_length * (1 - alpha_t_on[0, 0])).item())
    
    # Test at multiple timesteps < t_on - all should have same mask count
    for t_val in [0.2, 0.1, 0.05]:
        t = torch.tensor([[t_val], [t_val]], dtype=torch.float32)
        dt = 0.05
        
        torch.manual_seed(42)
        p_x0, out = sampler.compute_posterior(
            mock_model, x.clone(), t, dt, p_x0=None, noise_removal_step=False
        )
        
        for i in range(batch_size):
            actual_num_masked = (out[i] == mock_model.mask_id).sum().item()
            assert actual_num_masked == expected_num_masked, \
                f"At t={t_val}: expected {expected_num_masked} masked, got {actual_num_masked}"


def test_starshape_default_mode_decreasing_mask_ratio(mock_config, mock_model):
    """Default mode should have decreasing mask count as t increases toward 1."""
    t_on = 0.3
    sampler = StarShapeSampler(mock_config, t_on=t_on, remasker_schedule="default")
    
    batch_size = 2
    seq_length = 32
    x = torch.randint(1, mock_model.vocab_size, (batch_size, seq_length), dtype=torch.long)
    
    mask_counts = []
    # Test with increasing t values (still < t_on for Phase 2)
    for t_val in [0.05, 0.15, 0.25]:
        t = torch.tensor([[t_val], [t_val]], dtype=torch.float32)
        dt = 0.05
        
        torch.manual_seed(42)
        p_x0, out = sampler.compute_posterior(
            mock_model, x.clone(), t, dt, p_x0=None, noise_removal_step=False
        )
        
        mask_counts.append((out[0] == mock_model.mask_id).sum().item())
    
    # As t increases, alpha_s = t - dt increases, so (1 - alpha_s) decreases
    # Therefore mask count should decrease as t increases
    assert mask_counts[0] > mask_counts[1] > mask_counts[2], \
        f"Expected decreasing mask counts as t increases, got {mask_counts}"


def test_starshape_get_mistake_confidences_shape(mock_config, mock_model):
    """_get_mistake_confidences should return correct shape."""
    sampler = StarShapeSampler(mock_config, t_on=0.1)
    
    batch_size = 4
    seq_length = 32
    sampled_x0 = torch.randint(1, mock_model.vocab_size, (batch_size, seq_length))
    t = torch.tensor([[0.05]] * batch_size, dtype=torch.float32)
    
    confidences = sampler._get_mistake_confidences(mock_model, sampled_x0, t)
    
    assert confidences.shape == (batch_size, seq_length), \
        f"Expected shape {(batch_size, seq_length)}, got {confidences.shape}"


def test_starshape_remasker_schedule_initialization(mock_config):
    """Test that remasker_schedule is properly initialized."""
    sampler_default = StarShapeSampler(mock_config, t_on=0.1)
    assert sampler_default.remasker_schedule == "default"
    
    sampler_plato = StarShapeSampler(mock_config, t_on=0.1, remasker_schedule="plato")
    assert sampler_plato.remasker_schedule == "plato"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])

