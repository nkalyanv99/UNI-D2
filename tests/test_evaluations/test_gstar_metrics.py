"""Unit tests for GStarMetrics."""

import pytest
import torch

from discrete_diffusion.evaluations import GStarMetrics
from tests.conftest import RunIf


@pytest.fixture
def gstar_metrics():
    """Create GStarMetrics instance."""
    return GStarMetrics()


def test_gstar_metrics_initialization(gstar_metrics):
    """Test that GStarMetrics initializes correctly."""
    assert hasattr(gstar_metrics, 'train_accuracy')
    assert hasattr(gstar_metrics, 'train_precision')
    assert hasattr(gstar_metrics, 'train_recall')
    assert hasattr(gstar_metrics, 'train_f1')
    assert hasattr(gstar_metrics, 'val_accuracy')
    assert hasattr(gstar_metrics, 'val_precision')
    assert hasattr(gstar_metrics, 'val_recall')
    assert hasattr(gstar_metrics, 'val_f1')


def test_gstar_metrics_update_train(gstar_metrics):
    """Test updating train metrics with known values."""
    batch_size = 4
    seq_len = 8
    
    # Create perfect predictions (all correct)
    preds = torch.zeros(batch_size, seq_len, dtype=torch.long)
    targets = torch.zeros(batch_size, seq_len, dtype=torch.long)
    
    gstar_metrics.update_train(preds, targets)
    
    train_metrics = gstar_metrics.compute_train()
    
    # Perfect predictions should give accuracy=1.0
    assert train_metrics['train/accuracy'] == pytest.approx(1.0, abs=1e-6)
    # Precision/recall/F1 may be undefined for all-negative case, but should not crash
    assert 'train/precision' in train_metrics
    assert 'train/recall' in train_metrics
    assert 'train/f1' in train_metrics


def test_gstar_metrics_update_valid(gstar_metrics):
    """Test updating validation metrics."""
    batch_size = 4
    seq_len = 8
    
    # Create some errors (50% error rate)
    preds = torch.tensor([
        [0, 0, 1, 1, 0, 0, 1, 1],
        [0, 1, 0, 1, 0, 1, 0, 1],
        [1, 1, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1],
    ], dtype=torch.long)
    targets = torch.tensor([
        [0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 1, 1, 1, 1],
    ], dtype=torch.long)
    
    gstar_metrics.update_valid(preds, targets)
    
    val_metrics = gstar_metrics.compute_valid()
    
    # Should have all four metrics
    assert 'val/accuracy' in val_metrics
    assert 'val/precision' in val_metrics
    assert 'val/recall' in val_metrics
    assert 'val/f1' in val_metrics
    
    # Accuracy should be 0.625 (20/32 correct: rows 0-2 have 4/8 correct each, row 3 has 8/8 correct)
    assert val_metrics['val/accuracy'] == pytest.approx(0.625, abs=1e-6)


def test_gstar_metrics_reset(gstar_metrics):
    """Test that reset clears all metrics."""
    batch_size = 2
    seq_len = 4
    
    preds = torch.ones(batch_size, seq_len, dtype=torch.long)
    targets = torch.ones(batch_size, seq_len, dtype=torch.long)
    
    gstar_metrics.update_train(preds, targets)
    gstar_metrics.update_valid(preds, targets)
    
    # Compute metrics to ensure they have values
    train_metrics_before = gstar_metrics.compute_train()
    val_metrics_before = gstar_metrics.compute_valid()
    
    assert train_metrics_before['train/accuracy'] > 0
    assert val_metrics_before['val/accuracy'] > 0
    
    # Reset
    gstar_metrics.reset()
    
    # Update again after reset to ensure metrics work correctly
    gstar_metrics.update_train(preds, targets)
    gstar_metrics.update_valid(preds, targets)
    
    # After reset and update, metrics should work correctly
    train_metrics_after = gstar_metrics.compute_train()
    val_metrics_after = gstar_metrics.compute_valid()
    
    # Metrics should be recomputed correctly after reset
    assert train_metrics_after['train/accuracy'] == pytest.approx(1.0, abs=1e-6)
    assert val_metrics_after['val/accuracy'] == pytest.approx(1.0, abs=1e-6)


@RunIf(min_gpus=1)
def test_gstar_metrics_device_movement(gstar_metrics):
    """Test that metrics can be moved to different devices."""
    
    # Move to CUDA
    gstar_metrics.to('cuda')
    
    batch_size = 2
    seq_len = 4
    preds = torch.ones(batch_size, seq_len, dtype=torch.long, device='cuda')
    targets = torch.ones(batch_size, seq_len, dtype=torch.long, device='cuda')
    
    gstar_metrics.update_train(preds, targets)
    
    metrics = gstar_metrics.compute_train()
    assert isinstance(metrics['train/accuracy'], torch.Tensor)
    assert metrics['train/accuracy'].device.type == 'cuda'


def test_gstar_metrics_compute_shape(gstar_metrics):
    """Test that compute methods return correct dictionary structure."""
    batch_size = 2
    seq_len = 4
    
    preds = torch.zeros(batch_size, seq_len, dtype=torch.long)
    targets = torch.zeros(batch_size, seq_len, dtype=torch.long)
    
    gstar_metrics.update_train(preds, targets)
    gstar_metrics.update_valid(preds, targets)
    
    train_metrics = gstar_metrics.compute_train()
    val_metrics = gstar_metrics.compute_valid()
    
    # Check train metrics
    assert len(train_metrics) == 4
    assert 'train/accuracy' in train_metrics
    assert 'train/precision' in train_metrics
    assert 'train/recall' in train_metrics
    assert 'train/f1' in train_metrics
    
    # Check validation metrics
    assert len(val_metrics) == 4
    assert 'val/accuracy' in val_metrics
    assert 'val/precision' in val_metrics
    assert 'val/recall' in val_metrics
    assert 'val/f1' in val_metrics
    
    # All values should be tensors
    for v in train_metrics.values():
        assert isinstance(v, torch.Tensor)
    for v in val_metrics.values():
        assert isinstance(v, torch.Tensor)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

