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


def test_matrix_rotation_preserved_with_large_child_translation() -> None:
    matrix = spy.float4x4(
        np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.float32)
    )
    original = f2.Transform(matrix)
    child = spy.float4(811685.375, 816883.6875, 6.166900157928467, 1)
    expected = spy.float4(child.x, -child.z, child.y, 1)
    for transform in (original, f2.Transform(original)):
        assert transform.matrix == matrix
        assert not transform.is_identity()
        assert spy.math.mul(transform.matrix, child) == expected


def test_matrix_matches_components_after_orthogonality_correction() -> None:
    matrix = np.eye(4, dtype=np.float32)
    matrix[0, 1] = 4 * np.finfo(np.float32).eps
    transform = f2.Transform(spy.float4x4(matrix))
    reconstructed = f2.Transform(transform.translation, transform.rotation, transform.scale)
    assert transform.matrix == reconstructed.matrix
    assert transform.matrix != spy.float4x4(matrix)
    assert transform.is_identity() == reconstructed.is_identity()


@pytest.mark.parametrize("edit", ["translation", "rotation", "scale", "composition_order", "reset"])
def test_matrix_cache_invalidated_by_edit(edit):
    matrix = spy.float4x4(
        np.array([[0, -1, 0, 2], [1, 0, 0, 3], [0, 0, 1, 4], [0, 0, 0, 1]], dtype=np.float32)
    )
    transform = f2.Transform(matrix)
    expected = f2.Transform(transform.translation, transform.rotation, transform.scale)
    for value in (transform, expected):
        if edit == "translation":
            value.translation = spy.float3(5, 6, 7)
        elif edit == "rotation":
            value.rotation = spy.quatf(0, 0, 0, 1)
        elif edit == "scale":
            value.scale = spy.float3(2, 3, 4)
        elif edit == "composition_order":
            value.composition_order = f2.Transform.CompositionOrder.str
        else:
            value.reset()
    assert transform.matrix == expected.matrix
    assert transform.is_identity() == expected.is_identity()


@pytest.mark.parametrize("scales", [[1, 1, 1], [-2, 3, 4], [1e-30, 1, 1e30]])
def test_transform_matrix_with_subnormal_decomposition_residual(scales: list[float]) -> None:
    # A glTF root rotation whose negligible skew residual underflows float.
    matrix = np.array(
        [
            [2.220446049250313e-16, 1, 2.465190328815662e-32, 2],
            [-5.053215392444312e-16, 7.395570986446986e-32, 1, -3],
            [1, -2.220446049250313e-16, 5.053215392444312e-16, 4],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )
    matrix[:3, :3] *= np.asarray(scales, dtype=np.float32)
    result = f2.Transform(spy.float4x4(matrix)).matrix
    actual = np.array([[result[i, j] for j in range(4)] for i in range(4)])
    # Measure each column in its own units, including widely different scales.
    np.testing.assert_allclose(
        actual[:3, :3] / np.abs(scales), matrix[:3, :3] / np.abs(scales), atol=1e-6, rtol=1e-6
    )
    np.testing.assert_allclose(actual[:, 3], matrix[:, 3], atol=1e-6)


@pytest.mark.parametrize("value", [0, np.inf, np.nan])
def test_transform_failed_decomposition_retains_identity_fallback(value: float) -> None:
    matrix = np.eye(4, dtype=np.float32)
    matrix[0, 0] = value
    assert f2.Transform(spy.float4x4(matrix)).is_identity()


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


def test_transform_rejects_perspective() -> None:
    matrix = np.eye(4, dtype=np.float32)
    matrix[3, 0] = 0.1
    transform = f2.Transform(spy.float4x4(matrix))
    assert transform.is_identity()
    assert transform.matrix == spy.float4x4.identity()


def test_transform_orthogonalizes_sheared_matrix() -> None:
    matrix = np.array([[2, 1.5, -2, 5], [0, 3, 1, 6], [0, 0, 4, 7], [0, 0, 0, 1]], dtype=np.float32)
    transform = f2.Transform(spy.float4x4(matrix))
    expected = np.array([[2, 0, 0, 5], [0, 3, 0, 6], [0, 0, 4, 7], [0, 0, 0, 1]], dtype=np.float32)
    np.testing.assert_allclose(transform.matrix.to_numpy(), expected, atol=1e-6)
    assert not transform.is_identity()


if __name__ == "__main__":
    pytest.main([__file__, "-vs"])
