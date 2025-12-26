"""Pytest configuration and fixtures."""

import pytest
import torch


def RunIf(**kwargs):
    """Decorator to conditionally skip tests based on requirements.
    
    Usage:
        @RunIf(min_gpus=1)
        def test_requires_gpu():
            ...
    """
    def decorator(func):
        if 'min_gpus' in kwargs:
            min_gpus = kwargs['min_gpus']
            if not torch.cuda.is_available() or torch.cuda.device_count() < min_gpus:
                return pytest.mark.skip(reason=f"Requires at least {min_gpus} GPU(s)")(func)
        return func
    return decorator

