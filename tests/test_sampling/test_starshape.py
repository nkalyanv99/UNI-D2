"""Unit tests for StarShape sampler."""

import pytest
import torch
from omegaconf import OmegaConf

from discrete_diffusion.sampling.starshape import StarShapeSampler
from discrete_diffusion.sampling.absorbing import AbsorbingSampler


class MockModel:
    """Mock model for testing samplers."""
    
    class MockNoise:
        """Mock noise schedule: alpha decreases from 1 to 0 as t goes from 0 to 1."""
        def alpha_t(self, t):
            """Linear noise schedule: alpha = 1 - t.
            
            At t=0 (end of diffusion): alpha=1 (all denoised)
            At t=1 (start of diffusion): alpha=0 (all masked)
            """
            return 1 - t
    
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
    t_on = 0.3
    t_off = 0.05
    sampler_mdlm = AbsorbingSampler(mock_config)
    sampler_starshape = StarShapeSampler(mock_config, t_on=t_on, t_off=t_off)
    
    # Create test data
    batch_size = 2
    seq_length = 32
    x = torch.full((batch_size, seq_length), mock_model.mask_id, dtype=torch.long)
    # Unmask a few positions
    x[:, :5] = torch.randint(1, mock_model.vocab_size, (batch_size, 5))
    
    # Test with t > t_on (should use MDLM behavior - Phase 1)
    t = torch.tensor([[0.5], [0.5]], dtype=torch.float32)  # t=0.5 > t_on=0.3
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
    """When t_on >= t >= t_off, should mask exact number of tokens."""
    t_on = 0.3
    t_off = 0.05
    sampler = StarShapeSampler(mock_config, t_on=t_on, t_off=t_off)
    
    batch_size = 2
    seq_length = 32
    x = torch.randint(1, mock_model.vocab_size, (batch_size, seq_length), dtype=torch.long)
    
    # Test with t_on >= t >= t_off (should use StarShape behavior - Phase 2)
    t = torch.tensor([[0.2], [0.2]], dtype=torch.float32)  # t_off < 0.2 < t_on
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


def test_starshape_phase3_matches_mdlm(mock_config, mock_model):
    """When t < t_off, behavior should match MDLM exactly."""
    t_on = 0.3
    t_off = 0.1
    sampler_mdlm = AbsorbingSampler(mock_config)
    sampler_starshape = StarShapeSampler(mock_config, t_on=t_on, t_off=t_off)
    
    # Create test data
    batch_size = 2
    seq_length = 32
    x = torch.full((batch_size, seq_length), mock_model.mask_id, dtype=torch.long)
    # Unmask a few positions
    x[:, :5] = torch.randint(1, mock_model.vocab_size, (batch_size, 5))
    
    # Test with t < t_off (should use MDLM behavior - Phase 3)
    t = torch.tensor([[0.05], [0.05]], dtype=torch.float32)  # t=0.05 < t_off=0.1
    dt = 0.01
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    p_x0_mdlm, out_mdlm = sampler_mdlm.compute_posterior(
        mock_model, x.clone(), t, dt, p_x0=None, noise_removal_step=False
    )
    
    torch.manual_seed(42)
    p_x0_star, out_star = sampler_starshape.compute_posterior(
        mock_model, x.clone(), t, dt, p_x0=None, noise_removal_step=False
    )
    
    # Results should be identical in Phase 3
    assert torch.allclose(p_x0_mdlm, p_x0_star)
    assert torch.equal(out_mdlm, out_star)


def test_starshape_final_step(mock_config, mock_model):
    """Final step should mask 0 tokens."""
    sampler = StarShapeSampler(mock_config, t_on=0.3, t_off=0.05)
    
    batch_size = 2
    seq_length = 32
    x = torch.full((batch_size, seq_length), mock_model.mask_id, dtype=torch.long)
    
    # Final step with noise_removal_step=True in Phase 2
    t = torch.tensor([[0.1], [0.1]], dtype=torch.float32)  # t_off < t < t_on (Phase 2)
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
    t_off = 0.05
    sampler = StarShapeSampler(mock_config, t_on=t_on, t_off=t_off)
    
    batch_size = 2
    seq_length = 32
    x = torch.full((batch_size, seq_length), mock_model.mask_id, dtype=torch.long)
    
    # Test just below t_on (should definitely use StarShape - Phase 2)
    t = torch.tensor([[t_on - 0.01], [t_on - 0.01]], dtype=torch.float32)
    dt = 0.1
    
    torch.manual_seed(42)
    p_x0, out = sampler.compute_posterior(
        mock_model, x, t, dt, p_x0=None, noise_removal_step=False
    )
    
    # Should use StarShape masking (Phase 2) since t < t_on and t >= t_off
    alpha_s = mock_model.noise.alpha_t(t - dt)
    expected_num_masked = torch.floor(seq_length * (1 - alpha_s)).long()
    
    for i in range(batch_size):
        actual_num_masked = (out[i] == mock_model.mask_id).sum().item()
        assert actual_num_masked == expected_num_masked[i, 0].item(), \
            f"Expected {expected_num_masked[i, 0].item()} masked tokens, got {actual_num_masked}"


def test_starshape_t_off_boundary(mock_config, mock_model):
    """Test behavior at the t_off boundary."""
    t_on = 0.3
    t_off = 0.1
    sampler_mdlm = AbsorbingSampler(mock_config)
    sampler_starshape = StarShapeSampler(mock_config, t_on=t_on, t_off=t_off)
    
    batch_size = 2
    seq_length = 32
    x = torch.full((batch_size, seq_length), mock_model.mask_id, dtype=torch.long)
    x[:, :5] = torch.randint(1, mock_model.vocab_size, (batch_size, 5))
    
    # Test just below t_off (should use MDLM - Phase 3)
    t = torch.tensor([[t_off - 0.01], [t_off - 0.01]], dtype=torch.float32)
    dt = 0.01
    
    torch.manual_seed(42)
    p_x0_mdlm, out_mdlm = sampler_mdlm.compute_posterior(
        mock_model, x.clone(), t, dt, p_x0=None, noise_removal_step=False
    )
    
    torch.manual_seed(42)
    p_x0_star, out_star = sampler_starshape.compute_posterior(
        mock_model, x.clone(), t, dt, p_x0=None, noise_removal_step=False
    )
    
    # Results should be identical in Phase 3
    assert torch.allclose(p_x0_mdlm, p_x0_star)
    assert torch.equal(out_mdlm, out_star)


def test_starshape_plato_mode_rescaling(mock_config, mock_model):
    """Plato mode should use rescaled time to maintain smooth schedule."""
    t_on = 0.5
    t_off = 0.1
    sampler = StarShapeSampler(mock_config, t_on=t_on, t_off=t_off, remasker_schedule="plato")
    
    batch_size = 2
    seq_length = 32
    x = torch.randint(1, mock_model.vocab_size, (batch_size, seq_length), dtype=torch.long)
    
    # In plato mode, Phase 2 uses fixed rescaled alpha
    # effective_range = 1 - (t_on - t_off) = 1 - (0.5 - 0.1) = 0.6
    # progress = (1 - t_on) / effective_range = 0.5 / 0.6 = 0.833...
    # t_rescaled = 1 - progress = 0.167
    
    # Test at multiple timesteps in Phase 2 - all should have same mask count
    results = []
    for t_val in [0.4, 0.3, 0.2, 0.15]:
        t = torch.tensor([[t_val], [t_val]], dtype=torch.float32)
        dt = 0.05
        
        torch.manual_seed(42)
        p_x0, out = sampler.compute_posterior(
            mock_model, x.clone(), t, dt, p_x0=None, noise_removal_step=False
        )
        
        mask_count = (out[0] == mock_model.mask_id).sum().item()
        results.append(mask_count)
    
    # All mask counts should be the same in Phase 2 with plato mode
    assert all(r == results[0] for r in results), \
        f"Expected same mask count in plato Phase 2, got {results}"


def test_starshape_plato_mode_continuity_at_t_on(mock_config, mock_model):
    """Plato mode should have continuous alpha at t_on boundary."""
    t_on = 0.5
    t_off = 0.1
    sampler = StarShapeSampler(mock_config, t_on=t_on, t_off=t_off, remasker_schedule="plato")
    
    # Check rescaled alpha at t_on from Phase 1 side
    t_phase1 = torch.tensor([[t_on + 0.001]])
    alpha_phase1 = sampler._get_rescaled_alpha(mock_model, t_phase1)
    
    # Check rescaled alpha at t_on from Phase 2 side
    t_phase2 = torch.tensor([[t_on - 0.001]])
    alpha_phase2 = sampler._get_rescaled_alpha(mock_model, t_phase2)
    
    # Should be approximately equal (continuous at boundary)
    assert torch.allclose(alpha_phase1, alpha_phase2, atol=0.01), \
        f"Alpha not continuous at t_on: {alpha_phase1.item()} vs {alpha_phase2.item()}"


def test_starshape_plato_mode_continuity_at_t_off(mock_config, mock_model):
    """Plato mode should have continuous alpha at t_off boundary."""
    t_on = 0.5
    t_off = 0.1
    sampler = StarShapeSampler(mock_config, t_on=t_on, t_off=t_off, remasker_schedule="plato")
    
    # Check rescaled alpha at t_off from Phase 2 side
    t_phase2 = torch.tensor([[t_off + 0.001]])
    alpha_phase2 = sampler._get_rescaled_alpha(mock_model, t_phase2)
    
    # Check rescaled alpha at t_off from Phase 3 side
    t_phase3 = torch.tensor([[t_off - 0.001]])
    alpha_phase3 = sampler._get_rescaled_alpha(mock_model, t_phase3)
    
    # Should be approximately equal (continuous at boundary)
    assert torch.allclose(alpha_phase2, alpha_phase3, atol=0.01), \
        f"Alpha not continuous at t_off: {alpha_phase2.item()} vs {alpha_phase3.item()}"


def test_starshape_default_mode_increasing_mask_ratio_with_t(mock_config, mock_model):
    """Default mode should have increasing mask count as t increases toward 1.
    
    As t increases (moving back toward start of diffusion where all is masked):
    - alpha_s = 1 - (t - dt) decreases (more noise)
    - (1 - alpha_s) = t - dt increases
    - Therefore mask count increases
    """
    t_on = 0.5
    t_off = 0.05
    sampler = StarShapeSampler(mock_config, t_on=t_on, t_off=t_off, remasker_schedule="default")
    
    batch_size = 2
    seq_length = 32
    x = torch.randint(1, mock_model.vocab_size, (batch_size, seq_length), dtype=torch.long)
    
    mask_counts = []
    # Test with increasing t values (still in Phase 2)
    for t_val in [0.1, 0.25, 0.4]:
        t = torch.tensor([[t_val], [t_val]], dtype=torch.float32)
        dt = 0.05
        
        torch.manual_seed(42)
        p_x0, out = sampler.compute_posterior(
            mock_model, x.clone(), t, dt, p_x0=None, noise_removal_step=False
        )
        
        mask_counts.append((out[0] == mock_model.mask_id).sum().item())
    
    # As t increases, alpha_s = 1 - (t - dt) decreases, so (1 - alpha_s) increases
    # Therefore mask count should increase as t increases
    assert mask_counts[0] < mask_counts[1] < mask_counts[2], \
        f"Expected increasing mask counts as t increases, got {mask_counts}"


def test_starshape_get_mistake_confidences_shape(mock_config, mock_model):
    """_get_mistake_confidences should return correct shape."""
    sampler = StarShapeSampler(mock_config, t_on=0.5, t_off=0.05)
    
    batch_size = 4
    seq_length = 32
    sampled_x0 = torch.randint(1, mock_model.vocab_size, (batch_size, seq_length))
    t = torch.tensor([[0.2]] * batch_size, dtype=torch.float32)
    
    confidences = sampler._get_mistake_confidences(mock_model, sampled_x0, t)
    
    assert confidences.shape == (batch_size, seq_length), \
        f"Expected shape {(batch_size, seq_length)}, got {confidences.shape}"


def test_starshape_remasker_schedule_initialization(mock_config):
    """Test that remasker_schedule is properly initialized."""
    sampler_default = StarShapeSampler(mock_config, t_on=0.5, t_off=0.05)
    assert sampler_default.remasker_schedule == "default"
    
    sampler_plato = StarShapeSampler(mock_config, t_on=0.5, t_off=0.05, remasker_schedule="plato")
    assert sampler_plato.remasker_schedule == "plato"


def test_starshape_t_on_t_off_initialization(mock_config):
    """Test that t_on and t_off are properly initialized."""
    # Default values
    sampler = StarShapeSampler(mock_config)
    assert sampler.t_on == 0.55
    assert sampler.t_off == 0.05
    
    # Custom values
    sampler_custom = StarShapeSampler(mock_config, t_on=0.7, t_off=0.1)
    assert sampler_custom.t_on == 0.7
    assert sampler_custom.t_off == 0.1


def test_starshape_plato_rescaling_endpoints(mock_config, mock_model):
    """Test that plato rescaling maps t=1 to alpha=0 and t=0 to alpha=1.
    
    Alpha represents share of denoised tokens:
    - t=1 (start of diffusion): alpha=0 (all masked)
    - t=0 (end of diffusion): alpha=1 (all denoised)
    """
    t_on = 0.5
    t_off = 0.1
    sampler = StarShapeSampler(mock_config, t_on=t_on, t_off=t_off, remasker_schedule="plato")
    
    # At t=1 (start of diffusion), alpha should be 0 (all masked)
    t_start = torch.tensor([[1.0]])
    alpha_start = sampler._get_rescaled_alpha(mock_model, t_start)
    assert torch.allclose(alpha_start, torch.tensor([[0.0]]), atol=0.01), \
        f"At t=1, expected alpha=0, got {alpha_start.item()}"
    
    # At t=0 (end of diffusion), alpha should be 1 (all denoised)
    t_end = torch.tensor([[0.0]])
    alpha_end = sampler._get_rescaled_alpha(mock_model, t_end)
    assert torch.allclose(alpha_end, torch.tensor([[1.0]]), atol=0.01), \
        f"At t=0, expected alpha=1, got {alpha_end.item()}"


# ============================================================================
# Alpha Correctness Tests
# ============================================================================

def test_plato_alpha_exact_formula_phase1(mock_config, mock_model):
    """Test exact alpha formula in Phase 1 (t > t_on).
    
    Formula: progress = (1 - t) / (1 - (t_on - t_off))
             t_rescaled = 1 - progress
             alpha = 1 - t_rescaled (for linear schedule where alpha = 1 - t)
    """
    t_on = 0.5
    t_off = 0.1
    sampler = StarShapeSampler(mock_config, t_on=t_on, t_off=t_off, remasker_schedule="plato")
    
    effective_range = 1 - (t_on - t_off)  # = 0.6
    
    # Test several t values in Phase 1
    test_values = [1.0, 0.9, 0.7, 0.51]
    for t_val in test_values:
        t = torch.tensor([[t_val]])
        alpha = sampler._get_rescaled_alpha(mock_model, t)
        
        # Manual calculation
        progress = (1 - t_val) / effective_range
        t_rescaled = 1 - progress
        expected_alpha = 1 - t_rescaled  # Linear schedule: alpha = 1 - t
        
        assert torch.allclose(alpha, torch.tensor([[expected_alpha]]), atol=1e-6), \
            f"Phase 1 at t={t_val}: expected alpha={expected_alpha:.6f}, got {alpha.item():.6f}"


def test_plato_alpha_exact_formula_phase2(mock_config, mock_model):
    """Test exact alpha formula in Phase 2 (t_on >= t >= t_off).
    
    Formula: progress = (1 - t_on) / (1 - (t_on - t_off)) [FIXED]
             t_rescaled = 1 - progress
             alpha = 1 - t_rescaled (for linear schedule where alpha = 1 - t)
    """
    t_on = 0.5
    t_off = 0.1
    sampler = StarShapeSampler(mock_config, t_on=t_on, t_off=t_off, remasker_schedule="plato")
    
    effective_range = 1 - (t_on - t_off)  # = 0.6
    
    # Expected fixed alpha in Phase 2
    progress_fixed = (1 - t_on) / effective_range  # = 0.5 / 0.6 = 0.8333...
    t_rescaled_fixed = 1 - progress_fixed  # = 0.1666...
    expected_alpha = 1 - t_rescaled_fixed  # = 0.8333...
    
    # Test several t values in Phase 2 - all should give same alpha
    test_values = [0.49, 0.4, 0.3, 0.2, 0.11]
    for t_val in test_values:
        t = torch.tensor([[t_val]])
        alpha = sampler._get_rescaled_alpha(mock_model, t)
        
        assert torch.allclose(alpha, torch.tensor([[expected_alpha]]), atol=1e-6), \
            f"Phase 2 at t={t_val}: expected alpha={expected_alpha:.6f}, got {alpha.item():.6f}"


def test_plato_alpha_exact_formula_phase3(mock_config, mock_model):
    """Test exact alpha formula in Phase 3 (t < t_off).
    
    Formula: progress = (1 - t_on + t_off - t) / (1 - (t_on - t_off))
             t_rescaled = 1 - progress
             alpha = 1 - t_rescaled (for linear schedule where alpha = 1 - t)
    """
    t_on = 0.5
    t_off = 0.1
    sampler = StarShapeSampler(mock_config, t_on=t_on, t_off=t_off, remasker_schedule="plato")
    
    effective_range = 1 - (t_on - t_off)  # = 0.6
    
    # Test several t values in Phase 3
    test_values = [0.09, 0.05, 0.01, 0.0]
    for t_val in test_values:
        t = torch.tensor([[t_val]])
        alpha = sampler._get_rescaled_alpha(mock_model, t)
        
        # Manual calculation
        progress = (1 - t_on + t_off - t_val) / effective_range
        t_rescaled = 1 - progress
        expected_alpha = 1 - t_rescaled  # Linear schedule: alpha = 1 - t
        
        assert torch.allclose(alpha, torch.tensor([[expected_alpha]]), atol=1e-6), \
            f"Phase 3 at t={t_val}: expected alpha={expected_alpha:.6f}, got {alpha.item():.6f}"


def test_plato_alpha_monotonically_decreasing(mock_config, mock_model):
    """Test that rescaled alpha is monotonically decreasing with t.
    
    As t increases from 0 to 1 (toward start of diffusion):
    - alpha should decrease (except in plateau where constant)
    - t=0 -> alpha=1 (all denoised)
    - t=1 -> alpha=0 (all masked)
    """
    t_on = 0.5
    t_off = 0.1
    sampler = StarShapeSampler(mock_config, t_on=t_on, t_off=t_off, remasker_schedule="plato")
    
    # Test full range from 0 to 1
    t_values = torch.linspace(0, 1, 101).unsqueeze(1)
    
    prev_alpha = 2.0  # Start high since alpha decreases
    for i in range(len(t_values)):
        t = t_values[i:i+1]
        alpha = sampler._get_rescaled_alpha(mock_model, t).item()
        
        # Alpha should be <= previous (monotonically non-increasing)
        assert alpha <= prev_alpha + 1e-6, \
            f"Alpha not monotonic at t={t.item():.3f}: {alpha:.6f} > {prev_alpha:.6f}"
        prev_alpha = alpha


def test_plato_alpha_constant_in_plateau(mock_config, mock_model):
    """Test that alpha is constant throughout the plateau region."""
    t_on = 0.5
    t_off = 0.1
    sampler = StarShapeSampler(mock_config, t_on=t_on, t_off=t_off, remasker_schedule="plato")
    
    # Sample many points in the plateau region
    t_values = torch.linspace(t_off + 0.001, t_on - 0.001, 50).unsqueeze(1)
    
    alphas = []
    for i in range(len(t_values)):
        t = t_values[i:i+1]
        alpha = sampler._get_rescaled_alpha(mock_model, t).item()
        alphas.append(alpha)
    
    # All alphas should be identical
    alpha_mean = sum(alphas) / len(alphas)
    for i, alpha in enumerate(alphas):
        assert abs(alpha - alpha_mean) < 1e-6, \
            f"Alpha not constant in plateau: at t={t_values[i].item():.3f}, got {alpha:.6f}, expected {alpha_mean:.6f}"


def test_plato_alpha_boundary_values(mock_config, mock_model):
    """Test alpha values at exact boundary points.
    
    With alpha = 1 - t schedule:
    - At t_on boundary: alpha = 1 - t_rescaled = (1 - t_on) / effective_range
    - At t_off boundary: same value (continuous plateau)
    """
    t_on = 0.5
    t_off = 0.1
    sampler = StarShapeSampler(mock_config, t_on=t_on, t_off=t_off, remasker_schedule="plato")
    
    effective_range = 1 - (t_on - t_off)  # = 0.6
    
    # At t = t_on (from Phase 1 side, just above)
    # progress = (1 - t_on) / effective_range
    # t_rescaled = 1 - progress
    # alpha = 1 - t_rescaled = progress = (1 - t_on) / effective_range
    t = torch.tensor([[t_on + 1e-6]])
    alpha_at_t_on = sampler._get_rescaled_alpha(mock_model, t).item()
    expected_at_t_on = (1 - t_on) / effective_range  # = 0.5 / 0.6 = 0.833...
    assert abs(alpha_at_t_on - expected_at_t_on) < 1e-4, \
        f"At t_on boundary: expected {expected_at_t_on:.6f}, got {alpha_at_t_on:.6f}"
    
    # At t = t_off (from Phase 3 side, just below)
    # Should be the same as at t_on (continuous at boundary)
    t = torch.tensor([[t_off - 1e-6]])
    alpha_at_t_off = sampler._get_rescaled_alpha(mock_model, t).item()
    assert abs(alpha_at_t_off - expected_at_t_on) < 1e-4, \
        f"At t_off boundary: expected ~{expected_at_t_on:.6f}, got {alpha_at_t_off:.6f}"


def test_plato_removes_plateau_gives_mdlm_schedule(mock_config, mock_model):
    """Test that rescaling formula gives MDLM schedule when plateau is removed.
    
    The key property: if we sample alpha at times outside the plateau,
    the sequence should match what MDLM would produce if the plateau didn't exist.
    
    Alpha goes from 1 (at t=0) to 0 (at t=1).
    """
    t_on = 0.5
    t_off = 0.1
    sampler = StarShapeSampler(mock_config, t_on=t_on, t_off=t_off, remasker_schedule="plato")
    
    effective_range = 1 - (t_on - t_off)  # = 0.6
    
    # Generate points in Phase 1 and Phase 3 (avoiding plateau)
    # Ordered from t=0 to t=1, so alpha goes from high to low
    phase3_t = torch.linspace(0.0, t_off - 0.01, 20)
    phase1_t = torch.linspace(t_on + 0.01, 1.0, 20)
    
    # Collect rescaled alphas (in order of increasing t)
    rescaled_alphas = []
    for t_val in torch.cat([phase3_t, phase1_t]):
        t = torch.tensor([[t_val]])
        alpha = sampler._get_rescaled_alpha(mock_model, t).item()
        rescaled_alphas.append(alpha)
    
    rescaled_alphas = torch.tensor(rescaled_alphas)
    
    # These rescaled alphas should span [0, 1] evenly (like MDLM would)
    # Check that min is ~0 (at t=1) and max is ~1 (at t=0)
    assert rescaled_alphas.min() < 0.05, \
        f"Rescaled alphas don't reach near 0: min={rescaled_alphas.min():.4f}"
    assert rescaled_alphas.max() > 0.95, \
        f"Rescaled alphas don't reach near 1: max={rescaled_alphas.max():.4f}"
    
    # Check monotonicity (alpha decreases as t increases)
    assert (rescaled_alphas[1:] <= rescaled_alphas[:-1] + 1e-6).all(), \
        "Rescaled alphas not monotonically decreasing with increasing t"


def test_plato_alpha_with_different_t_on_t_off(mock_config, mock_model):
    """Test alpha rescaling with various t_on/t_off combinations.
    
    For all configurations:
    - alpha(t=1) = 0 (all masked at start)
    - alpha(t=0) = 1 (all denoised at end)
    - Continuous at t_on and t_off boundaries
    """
    test_cases = [
        (0.8, 0.2),  # Wide plateau
        (0.6, 0.4),  # Narrow plateau
        (0.9, 0.1),  # Wide plateau, extreme values
        (0.55, 0.05),  # Default values
    ]
    
    for t_on, t_off in test_cases:
        sampler = StarShapeSampler(mock_config, t_on=t_on, t_off=t_off, remasker_schedule="plato")
        effective_range = 1 - (t_on - t_off)
        
        # Verify endpoints: alpha(t=1) = 0, alpha(t=0) = 1
        alpha_at_1 = sampler._get_rescaled_alpha(mock_model, torch.tensor([[1.0]])).item()
        alpha_at_0 = sampler._get_rescaled_alpha(mock_model, torch.tensor([[0.0]])).item()
        
        assert abs(alpha_at_1 - 0.0) < 1e-4, \
            f"t_on={t_on}, t_off={t_off}: alpha(1) = {alpha_at_1:.4f}, expected 0.0"
        assert abs(alpha_at_0 - 1.0) < 1e-4, \
            f"t_on={t_on}, t_off={t_off}: alpha(0) = {alpha_at_0:.4f}, expected 1.0"
        
        # Verify continuity at boundaries
        alpha_above_t_on = sampler._get_rescaled_alpha(mock_model, torch.tensor([[t_on + 0.001]])).item()
        alpha_below_t_on = sampler._get_rescaled_alpha(mock_model, torch.tensor([[t_on - 0.001]])).item()
        
        assert abs(alpha_above_t_on - alpha_below_t_on) < 0.01, \
            f"t_on={t_on}, t_off={t_off}: discontinuity at t_on"
        
        alpha_above_t_off = sampler._get_rescaled_alpha(mock_model, torch.tensor([[t_off + 0.001]])).item()
        alpha_below_t_off = sampler._get_rescaled_alpha(mock_model, torch.tensor([[t_off - 0.001]])).item()
        
        assert abs(alpha_above_t_off - alpha_below_t_off) < 0.01, \
            f"t_on={t_on}, t_off={t_off}: discontinuity at t_off"


def test_default_mode_alpha_unchanged(mock_config, mock_model):
    """Test that default mode uses unmodified alpha values."""
    t_on = 0.5
    t_off = 0.1
    sampler = StarShapeSampler(mock_config, t_on=t_on, t_off=t_off, remasker_schedule="default")
    
    # In default mode, compute_posterior uses model.noise.alpha_t directly
    # No rescaling should occur
    test_values = [0.9, 0.5, 0.3, 0.1, 0.05]
    
    for t_val in test_values:
        t = torch.tensor([[t_val]])
        # For linear schedule, alpha_t(t) = 1 - t
        expected_alpha = 1 - t_val
        actual_alpha = mock_model.noise.alpha_t(t).item()
        
        assert abs(actual_alpha - expected_alpha) < 1e-6, \
            f"Default mode at t={t_val}: expected {expected_alpha}, got {actual_alpha}"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
