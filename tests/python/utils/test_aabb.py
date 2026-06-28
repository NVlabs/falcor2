# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import numpy as np
import slangpy as spy
import falcor2 as f2


def test_aabb_properties():
    """Test AABB struct properties and methods."""
    # Test invalid AABB
    aabb = f2.AABB()
    assert not aabb.is_valid()

    # Test valid AABB
    min_pt = spy.float3(-1, -2, -3)
    max_pt = spy.float3(1, 2, 3)
    aabb = f2.AABB(min_pt, max_pt)
    assert aabb.is_valid()

    # Test center calculation
    center = aabb.center
    assert center.x == 0.0
    assert center.y == 0.0
    assert center.z == 0.0

    # Test size calculation
    size = aabb.size
    assert size.x == 2.0
    assert size.y == 4.0
    assert size.z == 6.0


def test_aabb_expand_point():
    """Test AABB expansion with points."""
    aabb = f2.AABB()
    assert not aabb.is_valid()

    # Expand with first point
    aabb.expand(spy.float3(1, 2, 3))
    assert aabb.is_valid()
    assert aabb.min.x == 1.0 and aabb.min.y == 2.0 and aabb.min.z == 3.0
    assert aabb.max.x == 1.0 and aabb.max.y == 2.0 and aabb.max.z == 3.0

    # Expand with second point
    aabb.expand(spy.float3(-1, 4, 1))
    assert aabb.min.x == -1.0 and aabb.min.y == 2.0 and aabb.min.z == 1.0
    assert aabb.max.x == 1.0 and aabb.max.y == 4.0 and aabb.max.z == 3.0


def test_aabb_expand_aabb():
    """Test AABB expansion with other AABBs."""
    aabb1 = f2.AABB(spy.float3(-1, -1, -1), spy.float3(1, 1, 1))
    aabb2 = f2.AABB(spy.float3(0, 0, 0), spy.float3(2, 3, 4))

    aabb1.expand(aabb2)
    assert aabb1.min.x == -1.0 and aabb1.min.y == -1.0 and aabb1.min.z == -1.0
    assert aabb1.max.x == 2.0 and aabb1.max.y == 3.0 and aabb1.max.z == 4.0


def test_aabb_transform():
    """Test AABB transformation."""
    aabb = f2.AABB(spy.float3(-1, -1, -1), spy.float3(1, 1, 1))

    # Create a simple translation matrix
    translation = spy.float4x4.identity()
    translation[0, 3] = 5.0  # Translate by 5 in X
    translation[1, 3] = 3.0  # Translate by 3 in Y
    translation[2, 3] = -2.0  # Translate by -2 in Z

    transformed = aabb.transform(translation)

    # Expected result: all corners translated
    assert transformed.is_valid()
    np.testing.assert_allclose(
        [transformed.min.x, transformed.min.y, transformed.min.z], [4.0, 2.0, -3.0], rtol=1e-6
    )
    np.testing.assert_allclose(
        [transformed.max.x, transformed.max.y, transformed.max.z], [6.0, 4.0, -1.0], rtol=1e-6
    )


def test_aabb_invalid_operations():
    """Test operations on invalid AABBs."""
    invalid_aabb = f2.AABB()

    # Invalid AABB should report zero center and size
    with pytest.raises(Exception):
        center = invalid_aabb.center

    with pytest.raises(Exception):
        size = invalid_aabb.size

    # Transform of invalid AABB should return invalid AABB
    with pytest.raises(Exception):
        matrix = spy.float4x4.identity()
        transformed = invalid_aabb.transform(matrix)


def test_aabb_edge_cases():
    """Test AABB edge cases."""
    # Test AABB with zero size
    point = spy.float3(1, 2, 3)
    zero_size_aabb = f2.AABB(point, point)
    assert zero_size_aabb.is_valid()
    assert zero_size_aabb.center.x == 1.0
    assert zero_size_aabb.center.y == 2.0
    assert zero_size_aabb.center.z == 3.0
    assert zero_size_aabb.size.x == 0.0
    assert zero_size_aabb.size.y == 0.0
    assert zero_size_aabb.size.z == 0.0

    # Test expanding invalid AABB with valid AABB should work
    invalid_aabb = f2.AABB()
    valid_aabb = f2.AABB(spy.float3(-1, -1, -1), spy.float3(1, 1, 1))

    invalid_aabb.expand(valid_aabb)
    assert invalid_aabb.is_valid()
    assert invalid_aabb.min.x == -1.0 and invalid_aabb.min.y == -1.0 and invalid_aabb.min.z == -1.0
    assert invalid_aabb.max.x == 1.0 and invalid_aabb.max.y == 1.0 and invalid_aabb.max.z == 1.0


if __name__ == "__main__":
    pytest.main([__file__])
