# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test basic image testing functionality."""

import numpy as np
import pytest
from falcor2.testing.image_test_plugin import ImageTest


def generate_checkerboard(width: int = 64, height: int = 64):
    """Generate a simple checkerboard pattern for testing."""
    # Create checkerboard pattern
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    pattern = ((x // 8) + (y // 8)) % 2

    # Convert to RGBA image
    image = np.zeros((height, width, 4), dtype=np.float32)
    image[:, :, 0] = pattern  # Red channel
    image[:, :, 1] = pattern  # Green channel
    image[:, :, 2] = pattern  # Blue channel
    image[:, :, 3] = 1.0  # Alpha channel

    return image


@pytest.mark.parametrize("width,height", [(32, 32), (64, 64), (128, 64)])
def test_checkerboard(image_test: ImageTest, width: int, height: int):
    """Test parameterized checkerboard generation."""
    image = generate_checkerboard(width, height)
    image_test(image)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
