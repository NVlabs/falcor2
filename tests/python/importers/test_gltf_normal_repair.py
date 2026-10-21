# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Repair bad shading frames without replacing valid authored attributes."""

import json
from pathlib import Path

import numpy as np
import pytest

import falcor2 as f2


def _load(tmp_path: Path, primitives: list[dict], *, repair: bool = True) -> f2.ImporterMesh:
    blob = bytearray()
    views, accessors, entries = [], [], []
    for primitive in primitives:
        attributes = {}
        entry = {"attributes": attributes, "mode": 4}
        for name, values in primitive.items():
            indices = name == "indices"
            array = np.asarray(values, dtype=np.uint32 if indices else np.float32)
            views.append({"buffer": 0, "byteOffset": len(blob), "byteLength": array.nbytes})
            accessor = {
                "bufferView": len(views) - 1,
                "componentType": 5125 if indices else 5126,
                "count": array.size if indices else len(array),
                "type": "SCALAR" if indices else f"VEC{array.shape[1]}",
            }
            if name == "POSITION":
                accessor.update(min=array.min(axis=0).tolist(), max=array.max(axis=0).tolist())
            accessors.append(accessor)
            if indices:
                entry["indices"] = len(accessors) - 1
            else:
                attributes[name] = len(accessors) - 1
            blob.extend(array.tobytes())
        entries.append(entry)
    (tmp_path / "mesh.bin").write_bytes(blob)
    path = tmp_path / "mesh.gltf"
    path.write_text(
        json.dumps(
            {
                "asset": {"version": "2.0"},
                "buffers": [{"uri": "mesh.bin", "byteLength": len(blob)}],
                "bufferViews": views,
                "accessors": accessors,
                "meshes": [{"primitives": entries}],
                "nodes": [{"mesh": 0}],
                "scenes": [{"nodes": [0]}],
                "scene": 0,
            }
        )
    )
    options = f2.ImportOptions()
    options.repair_broken_geometry = repair
    return f2.GltfImporter().load_scene(path, options).meshes[0]


def _square() -> dict:
    return {
        "POSITION": [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]],
        "NORMAL": [[0.6, 0, 0.8]] * 4,
        "TANGENT": [[0, 1, 0, -1]] * 4,
        "TEXCOORD_0": [[0, 0], [1, 0], [0, 1], [1, 1]],
        "COLOR_0": [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1], [1, 1, 1, 1]],
        "indices": [[0, 1, 2], [1, 3, 2]],
    }


def _valid_frames(mesh: f2.ImporterMesh) -> None:
    for subgeometry in mesh.subgeometries:
        indices = np.asarray(subgeometry.indices)
        p = mesh.positions[indices].astype(np.float64)
        face = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
        n = mesh.normals[indices]
        assert np.isfinite(n).all()
        np.testing.assert_allclose(np.linalg.norm(n, axis=-1), 1, atol=1e-6)
        scale = np.linalg.norm(face, axis=-1)[:, None]
        assert (np.einsum("fci,fi->fc", n, face) >= -1e-4 * scale).all()
    assert np.isfinite(mesh.tangents).all()
    assert (np.linalg.norm(mesh.tangents, axis=-1) > 0.9).all()
    np.testing.assert_allclose(np.sum(mesh.normals * mesh.tangents, axis=-1), 0, atol=1e-5)


@pytest.mark.parametrize("bad", [[0, 0, -1], [0, 0, 0], [float("nan"), 0, 1], [float("inf"), 0, 1]])
@pytest.mark.parametrize("uvs", [True, False])
def test_only_bad_normal_and_its_tangent_change(tmp_path: Path, bad: list, uvs: bool) -> None:
    primitive = _square()
    primitive["NORMAL"][0] = bad
    if not uvs:
        primitive.pop("TEXCOORD_0")
    mesh = _load(tmp_path, [primitive])
    assert mesh.vertex_count == 4
    np.testing.assert_array_equal(mesh.subgeometries[0].indices, primitive["indices"])
    np.testing.assert_array_equal(mesh.positions, primitive["POSITION"])
    np.testing.assert_array_equal(mesh.normals[1:], np.asarray(primitive["NORMAL"], np.float32)[1:])
    np.testing.assert_array_equal(
        mesh.tangents[1:], np.asarray(primitive["TANGENT"], np.float32)[1:, :3]
    )
    np.testing.assert_array_equal(mesh.handedness[1:], -1)
    np.testing.assert_allclose(mesh.normals[0], [0, 0, 1])
    _valid_frames(mesh)


def test_bad_frame_across_mesh_does_not_require_splitting(tmp_path: Path) -> None:
    primitive = _square()
    primitive["NORMAL"] = [[0, 0, -1]] * 4
    mesh = _load(tmp_path, [primitive])
    assert mesh.vertex_count == 4
    np.testing.assert_allclose(mesh.normals, [[0, 0, 1]] * 4)
    _valid_frames(mesh)


def test_disabled_repair_preserves_authored_frame(tmp_path: Path) -> None:
    primitive = _square()
    primitive["NORMAL"][0] = [0, 0, -1]
    mesh = _load(tmp_path, [primitive], repair=False)
    np.testing.assert_array_equal(mesh.normals, np.asarray(primitive["NORMAL"], np.float32))
    np.testing.assert_array_equal(
        mesh.tangents, np.asarray(primitive["TANGENT"], np.float32)[:, :3]
    )


def test_valid_authored_mesh_is_unchanged_by_enabling_repair(tmp_path: Path) -> None:
    primitive = _square()
    original = _load(tmp_path, [primitive], repair=False)
    repaired = _load(tmp_path, [primitive], repair=True)
    for name in ("positions", "normals", "tangents", "handedness"):
        np.testing.assert_array_equal(getattr(original, name), getattr(repaired, name))
    np.testing.assert_array_equal(
        original.subgeometries[0].indices, repaired.subgeometries[0].indices
    )


@pytest.mark.parametrize("direction", [[1.0, 0.0], [0.6, 0.8]])
def test_hemisphere_repair_preserves_authored_tangential_direction(
    tmp_path: Path, direction: list[float]
) -> None:
    primitive = _square()
    authored = np.array([*direction, -0.1], dtype=np.float32)
    authored /= np.linalg.norm(authored)
    primitive["NORMAL"][0] = authored.tolist()
    mesh = _load(tmp_path, [primitive])

    # A slight hemisphere violation must not replace a near-tangent authored
    # normal with the face average, creating an abrupt change beside its neighbors.
    assert np.dot(mesh.normals[0], authored) > 0.99
    np.testing.assert_allclose(mesh.normals[0, :2], direction, atol=1e-5)
    assert 0.0 <= mesh.normals[0, 2] < 1e-5
    np.testing.assert_array_equal(mesh.normals[1:], np.asarray(primitive["NORMAL"], np.float32)[1:])
    np.testing.assert_array_equal(mesh.positions, primitive["POSITION"])
    np.testing.assert_array_equal(mesh.subgeometries[0].indices, primitive["indices"])
    _valid_frames(mesh)


def test_average_is_adjusted_before_splitting(tmp_path: Path) -> None:
    primitive = {
        "POSITION": [[0, 0, 0], [2, 0, 0], [0, 2, 0], [0, 0.1, 0], [0.1, 0, 0.1]],
        "NORMAL": [
            [0, 0, -1],
            [0, 0, 1],
            [0, 0, 1],
            [0.70710677, 0, -0.70710677],
            [0.70710677, 0, -0.70710677],
        ],
        "indices": [[0, 1, 2], [0, 3, 4]],
    }
    mesh = _load(tmp_path, [primitive])
    assert mesh.vertex_count == 5
    np.testing.assert_array_equal(mesh.subgeometries[0].indices, primitive["indices"])
    _valid_frames(mesh)


def test_incompatible_incident_faces_split_only_bad_vertex(tmp_path: Path) -> None:
    positions = [[0, 0, 0]]
    normals = [[0, 0, 0]]
    indices = []
    for n in np.concatenate([np.eye(3), -np.eye(3)]):
        u = np.roll(n, 1)
        v = np.cross(n, u)
        indices.append([0, len(positions), len(positions) + 1])
        positions.extend([u.tolist(), v.tolist()])
        normals.extend([n.tolist(), n.tolist()])
    mesh = _load(tmp_path, [{"POSITION": positions, "NORMAL": normals, "indices": indices}])
    _valid_frames(mesh)
    after = np.asarray(mesh.subgeometries[0].indices)
    assert len(np.unique(after[:, 0])) == 6
    np.testing.assert_array_equal(mesh.positions[after], np.asarray(positions)[indices])
    np.testing.assert_array_equal(
        mesh.normals[after[:, 1:]], np.asarray(normals)[np.asarray(indices)[:, 1:]]
    )


def test_degenerate_faces_and_uvs_produce_finite_repaired_frames(tmp_path: Path) -> None:
    primitive = _square()
    primitive["NORMAL"][0] = [0, 0, 0]
    primitive["TEXCOORD_0"] = [[0, 0]] * 4
    primitive["indices"].append([0, 0, 1])
    mesh = _load(tmp_path, [primitive])
    _valid_frames(mesh)


def test_mirrored_uv_repair_preserves_unaffected_corner_frames(tmp_path: Path) -> None:
    primitive = _square()
    primitive["NORMAL"] = [[0, 0, 1], [0, 0, -1], [0, 0, 1], [0, 0, 1]]
    primitive["TEXCOORD_0"][3] = [0, 0]
    mesh = _load(tmp_path, [primitive])
    source = np.asarray(primitive["indices"])
    output = np.asarray(mesh.subgeometries[0].indices)
    unaffected = source != 1
    np.testing.assert_array_equal(mesh.positions[output], np.asarray(primitive["POSITION"])[source])
    np.testing.assert_array_equal(
        mesh.tangents[output][unaffected],
        np.asarray(primitive["TANGENT"], np.float32)[source, :3][unaffected],
    )
    np.testing.assert_array_equal(mesh.handedness[output][unaffected], -1)
    assert set(mesh.handedness[output][source == 1].tolist()) == {-1, 1}
    assert mesh.vertex_count == 5
    _valid_frames(mesh)


@pytest.mark.parametrize("scale", [1e-12, 1e12])
def test_normal_repair_is_independent_of_source_scale(tmp_path: Path, scale: float) -> None:
    primitive = _square()
    primitive["POSITION"] = np.asarray(primitive["POSITION"]) * scale
    primitive["NORMAL"][0] = [0, 0, -1]
    mesh = _load(tmp_path, [primitive])
    np.testing.assert_allclose(mesh.normals[0], [0, 0, 1])
    np.testing.assert_array_equal(mesh.normals[1:], np.asarray(primitive["NORMAL"], np.float32)[1:])
    _valid_frames(mesh)
