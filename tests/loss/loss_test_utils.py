"""
Shared test utilities for `scores.loss` tests.
"""

import pytest
import torch

TEST_DEVICE_PARAMS = [
    pytest.param(
        device,
        marks=pytest.mark.skipif(
            not getattr(torch, device).is_available(), reason=f"torch device {device} not available."
        ),
    )
    for device in ("cpu", "cuda", "mps")
]
