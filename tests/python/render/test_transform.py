# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import numpy as np
import slangpy as spy
import falcor2 as f2


def test_transform_default_construction():
    """Test default Transform construction."""
    transform = f2.Transform()

    # Default should be identity
    assert transform.is_identity()

    # Check default values
    assert transform.translation == spy.float3(0, 0, 0)
    assert transform.rotation == spy.quatf(0, 0, 0, 1)
    assert transform.scale == spy.float3(1, 1, 1)

    # Matrix should be identity
    assert transform.matrix == spy.float4x4.identity()


def test_transform_construction_with_parameters():
    """Test Transform construction with translation, rotation, and scale."""
    translation = spy.float3(1, 2, 3)
    rotation = spy.quatf(0, 0, 0, 1)  # Identity rotation
    scale = spy.float3(2, 3, 4)

    transform = f2.Transform(translation, rotation, scale)

    assert not transform.is_identity()

    assert transform.translation == translation
    assert transform.rotation == rotation
    assert transform.scale == scale


def test_transform_construction_from_matrix():
    """Test Transform construction from a matrix."""
    # Create a simple translation matrix
    matrix = spy.float4x4.identity()
    matrix[0, 3] = 5.0  # Translate by 5 in X
    matrix[1, 3] = 3.0  # Translate by 3 in Y
    matrix[2, 3] = -2.0  # Translate by -2 in Z

    transform = f2.Transform(matrix)

    # Check decomposed translation
    t = transform.translation
    np.testing.assert_allclose([t.x, t.y, t.z], [5.0, 3.0, -2.0], rtol=1e-6)

    # Rotation should be identity
    r = transform.rotation
    np.testing.assert_allclose([r.x, r.y, r.z, r.w], [0.0, 0.0, 0.0, 1.0], rtol=1e-6)

    # Scale should be identity
    s = transform.scale
    np.testing.assert_allclose([s.x, s.y, s.z], [1.0, 1.0, 1.0], rtol=1e-6)


def test_transform_copy_construction():
    """Test Transform copy construction."""
    original = f2.Transform(spy.float3(1, 2, 3), spy.quatf(0, 0, 0, 1), spy.float3(2, 2, 2))
    copy = f2.Transform(original)

    assert copy.translation == original.translation
    assert copy.rotation == original.rotation
    assert copy.scale == original.scale


def test_transform_reset():
    """Test Transform reset method."""
    transform = f2.Transform(spy.float3(1, 2, 3), spy.quatf(0, 0, 0, 1), spy.float3(2, 3, 4))
    assert not transform.is_identity()

    transform.reset()
    assert transform.is_identity()

    # Check reset values
    assert transform.translation == spy.float3(0, 0, 0)
    assert transform.rotation == spy.quatf(0, 0, 0, 1)
    assert transform.scale == spy.float3(1, 1, 1)


def test_transform_translation_property():
    """Test Transform translation property."""
    transform = f2.Transform()

    # Set translation
    translation = spy.float3(10, 20, 30)
    transform.translation = translation
    assert not transform.is_identity()

    # Get translation
    assert transform.translation == translation


def test_transform_rotation_property():
    """Test Transform rotation property."""
    transform = f2.Transform()

    # Set rotation (90 degrees around Z-axis)
    rotation = spy.quatf(0, 0, 0.7071068, 0.7071068)
    transform.rotation = rotation
    assert not transform.is_identity()

    # Get rotation
    assert transform.rotation == rotation


def test_transform_rotation_xyz_property():
    """Test Transform rotation_xyz property."""
    transform = f2.Transform()

    # Set rotation in euler angles (radians)
    euler = spy.float3(0.0, 0.0, np.pi / 2)  # 90 degrees around Z
    transform.rotation_xyz = euler
    assert not transform.is_identity()

    # Get rotation back as euler angles
    r = transform.rotation_xyz
    np.testing.assert_allclose([r.x, r.y, r.z], [0.0, 0.0, np.pi / 2], rtol=1e-6)


def test_transform_scale_property():
    """Test Transform scale property."""
    transform = f2.Transform()

    # Set scale
    scale = spy.float3(2, 3, 4)
    transform.scale = scale
    assert not transform.is_identity()

    # Get scale
    assert transform.scale == scale


def test_transform_composition_order():
    """Test Transform composition order."""
    translation = spy.float3(1, 0, 0)
    rotation = spy.quatf(0, 0, 0, 1)  # Identity
    scale = spy.float3(2, 2, 2)

    # Test different composition orders
    transform_srt = f2.Transform(translation, rotation, scale)
    transform_srt.composition_order = f2.Transform.CompositionOrder.srt

    transform_trs = f2.Transform(translation, rotation, scale)
    transform_trs.composition_order = f2.Transform.CompositionOrder.trs

    # For identity rotation, SRT and TRS give different results
    # SRT: first scale, then rotate, then translate
    # TRS: first translate, then rotate, then scale

    # Get composition order
    assert transform_srt.composition_order == f2.Transform.CompositionOrder.srt
    assert transform_trs.composition_order == f2.Transform.CompositionOrder.trs


def test_transform_translate_method():
    """Test Transform translate method."""
    transform = f2.Transform()

    # Translate by delta
    transform.translate(spy.float3(1, 2, 3))
    assert transform.translation == spy.float3(1, 2, 3)

    # Translate again (should accumulate)
    transform.translate(spy.float3(4, 5, 6))
    assert transform.translation == spy.float3(5, 7, 9)


def test_transform_rotate_method():
    """Test Transform rotate method."""
    transform = f2.Transform()

    # Rotate by delta (90 degrees around Z)
    delta = spy.quatf(0, 0, 0.7071068, 0.7071068)
    transform.rotate(delta)

    r = transform.rotation
    np.testing.assert_allclose(
        [r.x, r.y, r.z, r.w],
        [0.0, 0.0, 0.7071068, 0.7071068],
        rtol=1e-6,
    )

    # Rotate again (should accumulate)
    transform.rotate(delta)
    r = transform.rotation
    # After two 90-degree rotations around Z, we should have 180 degrees
    np.testing.assert_allclose([r.x, r.y, r.z, r.w], [0.0, 0.0, 1.0, 0.0], rtol=1e-5, atol=1e-6)


def test_transform_scale_by_method():
    """Test Transform scale_by method."""
    transform = f2.Transform()

    # Scale by factor
    transform.scale_by(spy.float3(2, 3, 4))
    assert transform.scale == spy.float3(2, 3, 4)

    # Scale again (should multiply)
    transform.scale_by(spy.float3(2, 2, 2))
    assert transform.scale == spy.float3(4, 6, 8)


def test_transform_matrix_computation():
    """Test Transform matrix computation."""
    # Pure translation
    transform = f2.Transform()
    transform.translation = spy.float3(5, 10, 15)

    matrix = transform.matrix
    # Check translation components
    assert matrix[0, 3] == 5.0
    assert matrix[1, 3] == 10.0
    assert matrix[2, 3] == 15.0

    # Pure scale
    transform = f2.Transform()
    transform.scale = spy.float3(2, 3, 4)

    matrix = transform.matrix
    # Check diagonal (scale components)
    assert matrix[0, 0] == 2.0
    assert matrix[1, 1] == 3.0
    assert matrix[2, 2] == 4.0


def test_transform_is_identity():
    """Test Transform is_identity method."""
    transform = f2.Transform()
    assert transform.is_identity()

    # Set non-identity translation
    transform.translation = spy.float3(1, 0, 0)
    assert not transform.is_identity()

    # Reset and set non-identity rotation
    transform.reset()
    transform.rotation = spy.quatf(0, 0, 0.7071068, 0.7071068)
    assert not transform.is_identity()

    # Reset and set non-identity scale
    transform.reset()
    transform.scale = spy.float3(2, 1, 1)
    assert not transform.is_identity()

    # Reset back to identity
    transform.reset()
    assert transform.is_identity()


def test_transform_chaining():
    """Test Transform method chaining."""
    transform = f2.Transform()

    # Chain multiple operations
    result = (
        transform.translate(spy.float3(1, 2, 3))
        .scale_by(spy.float3(2, 2, 2))
        .translate(spy.float3(1, 0, 0))
    )

    # The return value should be the transform itself
    assert result is transform

    # Check final state
    assert transform.translation == spy.float3(2, 2, 3)
    assert transform.rotation == spy.quatf(0, 0, 0, 1)
    assert transform.scale == spy.float3(2, 2, 2)


def test_transform_matrix_caching():
    """Test that Transform caches the matrix correctly."""
    transform = f2.Transform()
    transform.translation = spy.float3(1, 2, 3)

    # Get matrix twice - should be the same object (cached)
    matrix1 = transform.matrix
    matrix2 = transform.matrix

    # Values should be the same
    assert matrix1 == matrix2

    # Modify transform - matrix should be recomputed
    transform.translation = spy.float3(5, 6, 7)
    matrix3 = transform.matrix
    assert matrix3 != matrix2

    # New matrix should be different
    assert matrix3[0, 3] == 5.0
    assert matrix3[1, 3] == 6.0
    assert matrix3[2, 3] == 7.0


def test_transform_repr():
    """Test Transform string representation."""
    transform = f2.Transform(spy.float3(1, 2, 3), spy.quatf(0, 0, 0, 1), spy.float3(2, 3, 4))

    repr_str = repr(transform)
    assert "Transform" in repr_str
    assert "translation" in repr_str
    assert "rotation" in repr_str
    assert "scale" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-vs"])
