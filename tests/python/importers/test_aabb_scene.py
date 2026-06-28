# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import pytest
import numpy as np
import falcor2 as f2
from typing import Optional, Tuple, Any

DATA = Path(__file__).parent.parent.parent.parent / "data"

# Test files for AABB validation
TEST_FILES = [
    "assets/kronos/Box/glTF-Binary/Box.glb",
    "assets/kronos/Avocado/glTF-Binary/Avocado.glb",
]


def calculate_mesh_aabb_numpy(
    mesh: Any,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Calculate AABB using numpy for validation."""
    if mesh.vertex_count == 0:
        return None, None

    positions = mesh.positions
    min_pos = np.min(positions, axis=0)
    max_pos = np.max(positions, axis=0)
    return min_pos, max_pos


def transform_aabb_numpy(
    min_pos: Optional[np.ndarray], max_pos: Optional[np.ndarray], transform_matrix: np.ndarray
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Transform an AABB using numpy for validation."""
    if min_pos is None or max_pos is None:
        return None, None

    # Generate all 8 corners of the AABB
    corners = []
    for i in range(8):
        corner = np.array(
            [
                max_pos[0] if (i & 1) else min_pos[0],
                max_pos[1] if (i & 2) else min_pos[1],
                max_pos[2] if (i & 4) else min_pos[2],
                1.0,
            ]
        )
        corners.append(corner)

    # Transform all corners
    transformed_corners = []
    for corner in corners:
        transformed = transform_matrix @ corner
        transformed_corners.append(transformed[:3])  # Drop homogeneous coordinate

    # Find min/max of transformed corners
    transformed_corners = np.array(transformed_corners)
    min_transformed = np.min(transformed_corners, axis=0)
    max_transformed = np.max(transformed_corners, axis=0)

    return min_transformed, max_transformed


def calculate_node_aabb_recursive_numpy(
    scene: Any, node_idx: int, parent_transform: Optional[np.ndarray] = None
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Recursively calculate node AABBs using numpy for validation."""
    if node_idx < 0 or node_idx >= len(scene.nodes):
        return None, None

    node = scene.nodes[node_idx]

    # Calculate world transform
    if parent_transform is None:
        world_transform = node.transform.to_numpy()
    else:
        node_transform = node.transform.to_numpy()
        world_transform = parent_transform @ node_transform

    # Initialize AABB as invalid
    combined_min = None
    combined_max = None

    # If this node has a mesh, include its transformed AABB
    if node.mesh_index >= 0 and node.mesh_index < len(scene.meshes):
        mesh = scene.meshes[node.mesh_index]
        mesh_min, mesh_max = calculate_mesh_aabb_numpy(mesh)

        if mesh_min is not None and mesh_max is not None:
            transformed_min, transformed_max = transform_aabb_numpy(
                mesh_min, mesh_max, world_transform
            )
            combined_min = transformed_min
            combined_max = transformed_max

    # Include AABBs from children
    for child_idx in node.children:
        child_min, child_max = calculate_node_aabb_recursive_numpy(
            scene, child_idx, world_transform
        )

        if child_min is not None and child_max is not None:
            if combined_min is None or combined_max is None:
                combined_min = child_min
                combined_max = child_max
            else:
                combined_min = np.minimum(combined_min, child_min)
                combined_max = np.maximum(combined_max, child_max)

    return combined_min, combined_max


@pytest.mark.parametrize("test_file", TEST_FILES)
def test_mesh_aabb_calculation(test_file: str):
    """Test that mesh AABB calculation matches numpy implementation."""
    gltf_importer = f2.GltfImporter()
    scene = gltf_importer.load_scene(DATA / test_file)
    assert scene is not None

    # Calculate AABBs
    scene.calculate_aabbs()

    # Validate each mesh AABB
    for i, mesh in enumerate(scene.meshes):
        # Calculate expected AABB using numpy
        expected_min, expected_max = calculate_mesh_aabb_numpy(mesh)

        if expected_min is None or expected_max is None:
            # Empty mesh should have invalid AABB
            assert not mesh.local_aabb.is_valid()
            continue

        # Check that the AABB is valid
        assert mesh.local_aabb.is_valid()

        # Check that the calculated AABB matches numpy calculation
        actual_min = np.array([mesh.local_aabb.min.x, mesh.local_aabb.min.y, mesh.local_aabb.min.z])
        actual_max = np.array([mesh.local_aabb.max.x, mesh.local_aabb.max.y, mesh.local_aabb.max.z])

        np.testing.assert_allclose(
            actual_min, expected_min, rtol=1e-6, atol=1e-8, err_msg=f"Mesh {i} AABB min mismatch"
        )
        np.testing.assert_allclose(
            actual_max, expected_max, rtol=1e-6, atol=1e-8, err_msg=f"Mesh {i} AABB max mismatch"
        )


@pytest.mark.parametrize("test_file", TEST_FILES)
def test_node_aabb_calculation(test_file: str):
    """Test that node AABB calculation matches numpy implementation."""
    gltf_importer = f2.GltfImporter()
    scene = gltf_importer.load_scene(DATA / test_file)
    assert scene is not None

    # Calculate AABBs
    scene.calculate_aabbs()

    # Validate each node AABB by comparing with numpy implementation
    for root_idx in scene.root_nodes:
        _validate_node_aabb_recursive(scene, root_idx)


def _validate_node_aabb_recursive(
    scene: Any, node_idx: int, parent_transform: Optional[np.ndarray] = None
) -> None:
    """Helper function to recursively validate node AABBs."""
    if node_idx < 0 or node_idx >= len(scene.nodes):
        return

    node = scene.nodes[node_idx]

    # Calculate expected AABB using numpy
    expected_min, expected_max = calculate_node_aabb_recursive_numpy(
        scene, node_idx, parent_transform
    )

    if expected_min is None or expected_max is None:
        # Node should have invalid AABB
        assert not node.world_aabb.is_valid()
    else:
        # Check that the AABB is valid
        assert node.world_aabb.is_valid()

        # Check that the calculated AABB matches numpy calculation
        actual_min = np.array([node.world_aabb.min.x, node.world_aabb.min.y, node.world_aabb.min.z])
        actual_max = np.array([node.world_aabb.max.x, node.world_aabb.max.y, node.world_aabb.max.z])

        np.testing.assert_allclose(
            actual_min,
            expected_min,
            rtol=1e-5,
            atol=1e-7,
            err_msg=f"Node '{node.name}' AABB min mismatch",
        )
        np.testing.assert_allclose(
            actual_max,
            expected_max,
            rtol=1e-5,
            atol=1e-7,
            err_msg=f"Node '{node.name}' AABB max mismatch",
        )

    # Calculate world transform for children
    if parent_transform is None:
        world_transform = node.transform.to_numpy()
    else:
        node_transform = node.transform.to_numpy()
        world_transform = parent_transform @ node_transform

    # Recursively validate children
    for child_idx in node.children:
        _validate_node_aabb_recursive(scene, child_idx, world_transform)


@pytest.mark.parametrize("test_file", TEST_FILES)
def test_scene_aabb_calculation(test_file: str) -> None:
    """Test scene-wide AABB calculation."""
    gltf_importer = f2.GltfImporter()
    scene = gltf_importer.load_scene(DATA / test_file)
    assert scene is not None

    # Calculate AABBs
    scene.calculate_aabbs()

    # Get scene AABB
    scene_aabb = scene.get_scene_aabb()
    assert scene_aabb.is_valid()

    # Verify that scene AABB encompasses all root node AABBs
    for root_idx in scene.root_nodes:
        node = scene.nodes[root_idx]
        if node.world_aabb.is_valid():
            # Check that scene AABB min is <= node AABB min
            assert scene_aabb.min.x <= node.world_aabb.min.x
            assert scene_aabb.min.y <= node.world_aabb.min.y
            assert scene_aabb.min.z <= node.world_aabb.min.z

            # Check that scene AABB max is >= node AABB max
            assert scene_aabb.max.x >= node.world_aabb.max.x
            assert scene_aabb.max.y >= node.world_aabb.max.y
            assert scene_aabb.max.z >= node.world_aabb.max.z


@pytest.mark.parametrize("test_file", TEST_FILES)
def test_rescale_to_fit(test_file: str) -> None:
    """Test scene rescaling functionality."""
    gltf_importer = f2.GltfImporter()
    scene = gltf_importer.load_scene(DATA / test_file)
    assert scene is not None

    # Get original scene AABB
    scene.calculate_aabbs()
    original_aabb = scene.get_scene_aabb()
    assert original_aabb.is_valid()

    original_size = original_aabb.size
    original_max_dim = max(original_size.x, original_size.y, original_size.z)

    # Test rescaling to different sizes
    for target_size in [0.5, 1.0, 2.0]:
        # Make a copy to test (reload the scene)
        test_scene = gltf_importer.load_scene(DATA / test_file)

        # Rescale
        test_scene.rescale_to_fit(target_size)

        # Get new AABB
        new_aabb = test_scene.get_scene_aabb()
        assert new_aabb.is_valid()

        # Check that the scene is centered near the origin
        center = new_aabb.center
        np.testing.assert_allclose(
            [center.x, center.y, center.z],
            [0.0, 0.0, 0.0],
            atol=1e-6,
            err_msg=f"Scene not centered after rescaling to {target_size}",
        )

        # Check that the max dimension is approximately the target size
        new_size = new_aabb.size
        new_max_dim = max(new_size.x, new_size.y, new_size.z)
        np.testing.assert_allclose(
            new_max_dim,
            target_size,
            rtol=1e-6,
            err_msg=f"Max dimension {new_max_dim} != target {target_size}",
        )

        # Check that aspect ratio is preserved
        if original_max_dim > 0:
            expected_scale = target_size / original_max_dim
            expected_size_x = original_size.x * expected_scale
            expected_size_y = original_size.y * expected_scale
            expected_size_z = original_size.z * expected_scale

            np.testing.assert_allclose(
                new_size.x,
                expected_size_x,
                rtol=1e-5,
                err_msg=f"X dimension not scaled correctly for target {target_size}",
            )
            np.testing.assert_allclose(
                new_size.y,
                expected_size_y,
                rtol=1e-5,
                err_msg=f"Y dimension not scaled correctly for target {target_size}",
            )
            np.testing.assert_allclose(
                new_size.z,
                expected_size_z,
                rtol=1e-5,
                err_msg=f"Z dimension not scaled correctly for target {target_size}",
            )


def test_rescale_default_size() -> None:
    """Test rescaling with default parameter (size 1.0)."""
    gltf_importer = f2.GltfImporter()
    scene = gltf_importer.load_scene(DATA / TEST_FILES[0])
    assert scene is not None

    # Rescale with default parameter
    scene.rescale_to_fit()  # Should default to 1.0

    # Check result
    new_aabb = scene.get_scene_aabb()
    assert new_aabb.is_valid()

    new_size = new_aabb.size
    new_max_dim = max(new_size.x, new_size.y, new_size.z)
    np.testing.assert_allclose(new_max_dim, 1.0, rtol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__])
