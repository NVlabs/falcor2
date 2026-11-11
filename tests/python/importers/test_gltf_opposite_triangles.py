# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import slangpy as spy

import falcor2 as f2

POSITIONS = np.array([[-1, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
FRONT = (0, 1, 2)
BACK = (0, 2, 1)


def _make_mesh(
    subgeometries: list[list[tuple[int, int, int]]],
    *,
    positions: np.ndarray | None = None,
    materials: tuple[int, ...] | None = None,
) -> f2.ImporterMesh:
    mesh = f2.ImporterMesh.create(
        [np.array(indices, dtype=np.uint32) for indices in subgeometries],
        {"position": POSITIONS if positions is None else positions},
    )
    for index, subgeometry in enumerate(mesh.subgeometries):
        subgeometry.material_index = 0 if materials is None else materials[index]
    return mesh


def _node(mesh_index: int, *, mirrored: bool = False) -> f2.ImporterNode:
    node = f2.ImporterNode()
    node.mesh_index = mesh_index
    if mirrored:
        node.transform = spy.float4x4(np.diag([-1, 1, 1, 1]).astype(np.float32))
    return node


def _triangle_count(mesh: f2.ImporterMesh) -> int:
    return sum(len(subgeometry.indices) for subgeometry in mesh.subgeometries)


def test_mesh_collapse_retains_first_winding_and_same_winding_duplicates() -> None:
    positions = np.concatenate([POSITIONS, POSITIONS])
    mesh = _make_mesh([[FRONT, BACK, FRONT], [(3, 5, 4)]], positions=positions)
    assert mesh.collapse_exact_opposite_triangles() == 2
    assert len(mesh.subgeometries) == 1
    assert [tuple(index) for index in mesh.subgeometries[0].indices] == [FRONT, FRONT]
    np.testing.assert_array_equal(mesh.positions, positions)
    assert mesh.collapse_exact_opposite_triangles() == 0


def test_mesh_collapse_preserves_material_boundaries() -> None:
    mesh = _make_mesh([[FRONT], [BACK]], materials=(0, 1))
    assert mesh.collapse_exact_opposite_triangles() == 0
    assert [subgeometry.material_index for subgeometry in mesh.subgeometries] == [0, 1]
    assert _triangle_count(mesh) == 2


def test_scene_collapse_preserves_separate_meshes_and_shared_instances() -> None:
    scene = f2.ImporterScene()
    scene.meshes = [_make_mesh([[FRONT, BACK]]), _make_mesh([[BACK]])]
    scene.nodes = [_node(0), _node(0, mirrored=True), _node(1)]
    scene.root_nodes = [0, 1, 2]
    assert scene.collapse_exact_opposite_triangles() == 1
    assert [_triangle_count(mesh) for mesh in scene.meshes] == [1, 1]
    assert [node.mesh_index for node in scene.nodes] == [0, 0, 1]
    assert scene.collapse_exact_opposite_triangles() == 0


def test_scene_collapse_preserves_meshes_shared_with_skinning() -> None:
    scene = f2.ImporterScene()
    scene.meshes = [_make_mesh([[FRONT, BACK]]), _make_mesh([[FRONT, BACK]])]
    scene.nodes = [_node(0), _node(0), _node(1)]
    scene.nodes[1].skin_index = 0
    scene.skins = [f2.ImporterSkin()]
    scene.root_nodes = [0, 1, 2]
    assert scene.collapse_exact_opposite_triangles() == 1
    assert [_triangle_count(mesh) for mesh in scene.meshes] == [2, 1]
    assert [node.mesh_index for node in scene.nodes] == [0, 0, 1]
    assert len(scene.skins) == 1 and scene.nodes[1].skin_index == 0


def test_gltf_import_preserves_opposite_triangles_until_explicit_cleanup(tmp_path: Path) -> None:
    positions = np.concatenate([POSITIONS, POSITIONS[[0, 2, 1]]])
    (tmp_path / "positions.bin").write_bytes(positions.tobytes())
    path = tmp_path / "opposites.gltf"
    path.write_text(
        json.dumps(
            {
                "asset": {"version": "2.0"},
                "buffers": [{"uri": "positions.bin", "byteLength": positions.nbytes}],
                "bufferViews": [{"buffer": 0, "byteLength": positions.nbytes}],
                "accessors": [
                    {
                        "bufferView": 0,
                        "componentType": 5126,
                        "count": 6,
                        "type": "VEC3",
                        "min": positions.min(axis=0).tolist(),
                        "max": positions.max(axis=0).tolist(),
                    }
                ],
                "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
                "nodes": [{"mesh": 0}],
                "scenes": [{"nodes": [0]}],
                "scene": 0,
            }
        ),
        encoding="utf-8",
    )
    scene = f2.import_scene(path)
    assert len(scene.meshes) == 1
    assert _triangle_count(scene.meshes[0]) == 2
    assert scene.collapse_exact_opposite_triangles() == 1
    assert _triangle_count(scene.meshes[0]) == 1
