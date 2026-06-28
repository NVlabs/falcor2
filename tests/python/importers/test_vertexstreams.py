# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for the new ImporterMesh bindings: get_stream, find_stream, and vertex_count
"""

from pathlib import Path
import pytest
import numpy as np
import falcor2 as f2

DATA = Path(__file__).parent.parent.parent.parent / "data"


def test_vertex_count_property():
    """Test that vertex_count property works correctly."""
    usd_importer = f2.UsdImporter()
    scene = usd_importer.load_scene(DATA / "assets/cornell-box/usdpreviewsurface/cornell-box.usda")
    assert scene is not None

    for mesh in scene.meshes:
        # vertex_count should be a non-negative integer
        assert isinstance(mesh.vertex_count, int)
        assert mesh.vertex_count >= 0

        # All existing arrays should have the same length as vertex_count
        if mesh.vertex_count > 0:
            assert mesh.positions is not None
            assert mesh.normals is not None
            assert mesh.tangents is not None
            assert mesh.handedness is not None
            tc0 = mesh.texcoords(0)
            assert tc0 is not None
            assert mesh.positions.shape[0] == mesh.vertex_count
            assert mesh.normals.shape[0] == mesh.vertex_count
            assert mesh.tangents.shape[0] == mesh.vertex_count
            assert mesh.handedness.shape[0] == mesh.vertex_count
            assert tc0.shape[0] == mesh.vertex_count


def test_find_stream():
    """Test find_stream method with various semantics."""
    usd_importer = f2.UsdImporter()
    scene = usd_importer.load_scene(DATA / "assets/cornell-box/usdpreviewsurface/cornell-box.usda")
    assert scene is not None

    mesh = scene.meshes[0]  # Get first mesh
    assert mesh.vertex_count > 0

    # Test position attribute
    positions = mesh.find_stream(f2.ImporterSemantic.position, 0)
    assert positions is not None
    assert positions.shape == (mesh.vertex_count, 3)
    assert positions.dtype == np.float32
    # Should match existing positions property
    assert np.array_equal(positions, mesh.positions)

    # Test normal attribute
    normals = mesh.find_stream(f2.ImporterSemantic.normal, 0)
    assert normals is not None
    assert normals.shape == (mesh.vertex_count, 3)
    assert normals.dtype == np.float32
    # Should match existing normals property
    assert np.array_equal(normals, mesh.normals)

    # Test tangent attribute
    tangents = mesh.find_stream(f2.ImporterSemantic.tangent, 0)
    assert tangents is not None
    assert tangents.shape == (mesh.vertex_count, 3)
    assert tangents.dtype == np.float32
    # Should match existing tangents property
    assert np.array_equal(tangents, mesh.tangents)

    # Test handedness attribute
    handedness = mesh.find_stream(f2.ImporterSemantic.handedness, 0)
    assert handedness is not None
    assert handedness.shape == (mesh.vertex_count,)  # Handedness is scalar per vertex
    assert handedness.dtype == np.float32
    # Should match existing handedness property
    assert np.array_equal(handedness, mesh.handedness)

    # Test texture coordinate attribute
    uvs = mesh.find_stream(f2.ImporterSemantic.tex_coord, 0)
    assert uvs is not None
    assert uvs.shape == (mesh.vertex_count, 2)
    assert uvs.dtype == np.float32
    # Should match existing uvs property
    assert np.array_equal(uvs, mesh.texcoords(0))

    # Test non-existent attribute (should return None)
    non_existent = mesh.find_stream(f2.ImporterSemantic.color, 0)
    # Color might or might not exist, but it should either return None or a valid array
    if non_existent is not None:
        assert len(non_existent.shape) == 2
        assert non_existent.shape[0] == mesh.vertex_count


def test_get_stream():
    """Test get_stream method with explicit attributes."""
    usd_importer = f2.UsdImporter()
    scene = usd_importer.load_scene(DATA / "assets/cornell-box/usdpreviewsurface/cornell-box.usda")
    assert scene is not None

    mesh = scene.meshes[0]
    assert mesh.vertex_count > 0

    # Get position attribute and test get_stream
    pos_attrib = mesh.find_attribute(f2.ImporterSemantic.position, 0)
    assert pos_attrib is not None
    assert pos_attrib.valid

    positions = mesh.get_stream(pos_attrib)
    assert positions is not None
    assert positions.shape == (mesh.vertex_count, 3)
    assert positions.dtype == np.float32
    # Should match existing positions property and find_stream result
    assert np.array_equal(positions, mesh.positions)

    positions_via_find = mesh.find_stream(f2.ImporterSemantic.position, 0)
    assert np.array_equal(positions, positions_via_find)

    # Test with normal attribute
    normal_attrib = mesh.find_attribute(f2.ImporterSemantic.normal, 0)
    assert normal_attrib is not None

    normals = mesh.get_stream(normal_attrib)
    assert normals is not None
    assert normals.shape == (mesh.vertex_count, 3)
    assert np.array_equal(normals, mesh.normals)


def test_attribute_consistency():
    """Test that all attribute access methods give consistent results."""
    usd_importer = f2.UsdImporter()
    scene = usd_importer.load_scene(DATA / "assets/cornell-box/usdpreviewsurface/cornell-box.usda")
    assert scene is not None

    for mesh in scene.meshes:
        if mesh.vertex_count == 0:
            continue

        # Test position consistency
        pos_attrib = mesh.find_attribute(f2.ImporterSemantic.position, 0)
        if pos_attrib:
            pos1 = mesh.positions  # existing property
            pos2 = mesh.find_stream(f2.ImporterSemantic.position, 0)  # new method 1
            pos3 = mesh.get_stream(pos_attrib)  # new method 2

            assert np.array_equal(pos1, pos2)
            assert np.array_equal(pos1, pos3)

        # Test normal consistency
        normal_attrib = mesh.find_attribute(f2.ImporterSemantic.normal, 0)
        if normal_attrib:
            norm1 = mesh.normals
            norm2 = mesh.find_stream(f2.ImporterSemantic.normal, 0)
            norm3 = mesh.get_stream(normal_attrib)

            assert np.array_equal(norm1, norm2)
            assert np.array_equal(norm1, norm3)

        # Test UV consistency
        uv_attrib = mesh.find_attribute(f2.ImporterSemantic.tex_coord, 0)
        if uv_attrib:
            uv1 = mesh.texcoords(0)  # existing property
            uv2 = mesh.find_stream(f2.ImporterSemantic.tex_coord, 0)  # new method 1
            uv3 = mesh.get_stream(uv_attrib)  # new method 2

            assert np.array_equal(uv1, uv2)
            assert np.array_equal(uv1, uv3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
