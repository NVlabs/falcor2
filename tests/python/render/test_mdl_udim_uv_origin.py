# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import pytest
import slangpy as spy

import falcor2 as f2
import falcor2.testing.helpers as helpers
from falcor2.importers import import_scene
from falcor2.rendergraph import ContainerSpec
from falcor2.rendernodes import ReferencePathTracerNode


DATA = Path(__file__).parents[3] / "data"
USD_SCENE_PATH = DATA / "assets/test_uv_origin/udim.usdc"
GLB_SCENE_PATH = DATA / "assets/test_uv_origin/udim.glb"
DEBUG_OUTPUT_DIR = Path(".tmp") / "mdl_udim_uv_origin"
WIDTH = 512
HEIGHT = 512
PATCH_RADIUS = 24
TILE_ORIENTATION_MIN_SCORE = 0.025

FloatArray = NDArray[np.float32]
UIntArray = NDArray[np.uint32]

SCREEN_NORMALS = np.array([[0.0, 0.0, 1.0]] * 4, dtype=np.float32)
CAMERA_POSITION = spy.float3(0.0, 0.0, 4.0)
CAMERA_ROTATION = spy.math.quat_from_look_at(
    spy.float3(0.0, 0.0, -1.0),
    spy.float3(0.0, 1.0, 0.0),
)


@dataclass(frozen=True)
class ManualQuadCase:
    id: str
    uv_origin: f2.UVOrigin
    positions: FloatArray
    indices: UIntArray
    texcoords: FloatArray
    expected_tiles: dict[str, str]


MANUAL_QUAD_CASES = [
    ManualQuadCase(
        id="usd_lower_left",
        uv_origin=f2.UVOrigin.lower_left,
        positions=np.array(
            [
                [-1.0, -1.0, 0.0],
                [1.0, -1.0, 0.0],
                [1.0, 1.0, 0.0],
                [-1.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        ),
        indices=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32),
        texcoords=np.array(
            [[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]],
            dtype=np.float32,
        ),
        expected_tiles={
            "upper_left": "1011",
            "upper_right": "missing",
            "lower_left": "1001",
            "lower_right": "1002",
        },
    ),
    # Preserve the imported GLB plane's vertex order, triangle indices, and
    # UV-to-vertex association, but rotate the XZ plane into XY so the debug
    # PNG is not confounded by the original front-side camera's U mirror.
    ManualQuadCase(
        id="glb_upper_left",
        uv_origin=f2.UVOrigin.upper_left,
        positions=np.array(
            [
                [-1.0, -1.0, 0.0],
                [1.0, -1.0, 0.0],
                [-1.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        ),
        indices=np.array([[0, 1, 3], [0, 3, 2]], dtype=np.uint32),
        texcoords=np.array(
            [[0.0, 1.0], [2.0, 1.0], [0.0, -1.0], [2.0, -1.0]],
            dtype=np.float32,
        ),
        expected_tiles={
            "upper_left": "1011",
            "upper_right": "missing",
            "lower_left": "1001",
            "lower_right": "1002",
        },
    ),
]


def test_udim_test_assets_have_expected_imported_uvs() -> None:
    usd_scene = import_scene(USD_SCENE_PATH)
    glb_scene = import_scene(GLB_SCENE_PATH)
    assert usd_scene is not None
    assert glb_scene is not None

    assert usd_scene.uv_origin == f2.UVOrigin.lower_left
    assert glb_scene.uv_origin == f2.UVOrigin.upper_left

    usd_mesh = usd_scene.meshes[0]
    glb_mesh = glb_scene.meshes[0]
    assert usd_mesh.uv_origin == f2.UVOrigin.lower_left
    assert glb_mesh.uv_origin == f2.UVOrigin.upper_left

    np.testing.assert_allclose(
        usd_mesh.texcoords(0),
        np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        usd_mesh.subgeometries[0].indices_numpy,
        np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32),
    )

    np.testing.assert_allclose(
        glb_mesh.texcoords(0),
        np.array([[0.0, 1.0], [2.0, 1.0], [0.0, -1.0], [2.0, -1.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        glb_mesh.subgeometries[0].indices_numpy,
        np.array([[0, 1, 3], [0, 3, 2]], dtype=np.uint32),
    )


def _patch_bounds(name: str) -> tuple[slice, slice]:
    centers = {
        "upper_left": (HEIGHT // 4, WIDTH // 4),
        "upper_right": (HEIGHT // 4, WIDTH * 3 // 4),
        "lower_left": (HEIGHT * 3 // 4, WIDTH // 4),
        "lower_right": (HEIGHT * 3 // 4, WIDTH * 3 // 4),
    }
    center_y, center_x = centers[name]
    return (
        slice(center_y - PATCH_RADIUS, center_y + PATCH_RADIUS),
        slice(center_x - PATCH_RADIUS, center_x + PATCH_RADIUS),
    )


def _quadrant_bounds(name: str) -> tuple[slice, slice]:
    return {
        "upper_left": (slice(0, HEIGHT // 2), slice(0, WIDTH // 2)),
        "upper_right": (slice(0, HEIGHT // 2), slice(WIDTH // 2, WIDTH)),
        "lower_left": (slice(HEIGHT // 2, HEIGHT), slice(0, WIDTH // 2)),
        "lower_right": (slice(HEIGHT // 2, HEIGHT), slice(WIDTH // 2, WIDTH)),
    }[name]


def _geometry_mask(
    geometry_id: UIntArray,
    y_slice: slice,
    x_slice: slice,
    min_coverage: float,
) -> NDArray[np.bool_]:
    valid_geometry = geometry_id[y_slice, x_slice, 1] != np.uint32(
        int(f2.GeometryInstanceID.invalid)
    )
    assert valid_geometry.mean() > min_coverage
    return valid_geometry


def _classify_tile(color: FloatArray) -> str:
    red, green, blue = color
    if max(red, green, blue) < 0.01:
        return "missing"
    if max(abs(red - green), abs(green - blue), abs(red - blue)) < 0.04:
        return "1001"
    if red > green + 0.12 and green > blue + 0.05:
        return "1002"
    if blue > green + 0.10 and green > red + 0.05:
        return "1011"
    raise AssertionError(f"Could not classify MDL UDIM tile color {color}.")


def _tile_bounds(
    geometry_id: UIntArray,
    quadrant_name: str,
) -> tuple[slice, slice]:
    y_slice, x_slice = _quadrant_bounds(quadrant_name)
    valid_geometry = _geometry_mask(geometry_id, y_slice, x_slice, 0.10)
    y_indices, x_indices = np.nonzero(valid_geometry)

    y_min = int(y_indices.min()) + y_slice.start
    y_max = int(y_indices.max()) + y_slice.start + 1
    x_min = int(x_indices.min()) + x_slice.start
    x_max = int(x_indices.max()) + x_slice.start + 1

    y_margin = max(2, (y_max - y_min) // 12)
    x_margin = max(2, (x_max - x_min) // 12)
    return (
        slice(y_min + y_margin, y_max - y_margin),
        slice(x_min + x_margin, x_max - x_margin),
    )


def _tile_orientation_score(tile: FloatArray) -> float:
    luma = tile[..., :3].mean(axis=2)
    darkness = np.clip(np.percentile(luma, 95) - luma, 0.0, 1.0)
    assert float(darkness.sum()) > 1e-3

    y_weights = np.linspace(-1.0, 1.0, tile.shape[0], dtype=np.float32)[:, np.newaxis]
    return float((darkness * y_weights).sum() / darkness.sum())


def _assert_tile_is_upright(
    material_color: FloatArray,
    geometry_id: UIntArray,
    quadrant_name: str,
) -> None:
    y_slice, x_slice = _tile_bounds(geometry_id, quadrant_name)
    tile = material_color[y_slice, x_slice, :3]
    score = _tile_orientation_score(tile)
    flipped_score = _tile_orientation_score(tile[::-1, :, :])
    assert score > TILE_ORIENTATION_MIN_SCORE, (
        f"Rendered tile in {quadrant_name} appears vertically flipped "
        f"or too vertically symmetric; score={score:.4f}"
    )
    assert flipped_score < -TILE_ORIENTATION_MIN_SCORE, (
        f"Rendered tile in {quadrant_name} does not have enough vertical "
        f"asymmetry to catch a flipped image; flipped_score={flipped_score:.4f}"
    )


def _write_png(data: FloatArray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    png = spy.Bitmap(
        data=np.clip(data, 0.0, 1.0),
        pixel_format=spy.Bitmap.PixelFormat.rgba,
        srgb_gamma=False,
    ).convert(
        component_type=spy.Bitmap.ComponentType.uint8,
        srgb_gamma=True,
    )
    png.write(str(path), spy.Bitmap.FileFormat.png)
    print(f"Wrote debug PNG: {path}")


@pytest.mark.slow
@pytest.mark.parametrize(
    "case",
    MANUAL_QUAD_CASES,
    ids=[case.id for case in MANUAL_QUAD_CASES],
)
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_mdl_udim_manual_quad_uses_recorded_origin_conventions(
    device_type: spy.DeviceType,
    case: ManualQuadCase,
) -> None:
    device = helpers.get_device(device_type)

    mesh = f2.ImporterMesh.create(
        [np.array(case.indices, dtype=np.uint32, copy=True)],
        {
            "position": np.array(case.positions, dtype=np.float32, copy=True),
            "normal": np.array(SCREEN_NORMALS, dtype=np.float32, copy=True),
            "tex_coord": np.array(case.texcoords, dtype=np.float32, copy=True),
        },
    )
    mesh.name = f"{case.id}_manual_udim"
    mesh.uv_origin = case.uv_origin
    mesh.add_tangents_from_uvs()
    subgeometries = list(mesh.subgeometries)
    subgeometries[0].name = "manual_udim_quad"
    subgeometries[0].material_name = "mdl_sdk_example_tiled"
    mesh.subgeometries = subgeometries

    material = f2.ImporterMaterial()
    material.name = "mdl_sdk_example_tiled"
    material.params = f2.Properties(
        {
            "_scene_material_type": "MDLMaterial",
            "mdl_library_path": str(DATA / "assets/mdl_sdk_examples"),
            "mdl_material_name": "tutorials::example_tiled",
            "mdl_class_compilation": False,
        }
    )

    node = f2.ImporterNode()
    node.name = "Manual UDIM Quad"
    node.mesh_index = 0

    importer_scene = f2.ImporterScene()
    importer_scene.uv_origin = case.uv_origin
    importer_scene.materials = [material]
    importer_scene.meshes = [mesh]
    importer_scene.nodes = [node]
    importer_scene.root_nodes = [0]
    importer_scene.calculate_aabbs()

    scene = f2.Scene.create(device, importer_scene)
    scene.update()
    camera = helpers.create_test_camera(
        scene,
        width=WIDTH,
        height=HEIGHT,
        fov_y=45.0,
        position=CAMERA_POSITION,
        rotation=CAMERA_ROTATION,
    )
    scene.active_camera = camera
    scene.update()

    path_tracer = ReferencePathTracerNode.create(device)
    path_tracer.output_spec = ContainerSpec.texture2d(
        spy.Format.rgba32_float,
        (HEIGHT, WIDTH),
    )
    path_tracer.guide_output_specs = {
        "material_color": ContainerSpec.texture2d(spy.Format.rgba32_float),
        "geometry_id": ContainerSpec.texture2d(spy.Format.rgba32_uint),
    }
    _, guides = path_tracer(
        scene,
        camera,
        iteration=0,
        subpixel_offset=spy.float2(0.0, 0.0),
        subpixel_random_jitter=0.0,
    )
    material_color = guides["material_color"]
    geometry_id = guides["geometry_id"]
    assert isinstance(material_color, spy.Texture)
    assert isinstance(geometry_id, spy.Texture)
    material_color = material_color.to_numpy()
    geometry_id = geometry_id.to_numpy()

    assert material_color.shape == (HEIGHT, WIDTH, 4)
    assert geometry_id.shape == (HEIGHT, WIDTH, 4)
    assert np.isfinite(material_color).all()

    _write_png(
        material_color,
        DEBUG_OUTPUT_DIR / f"{case.id}_{device_type.name}_material_color.png",
    )
    hit_mask = geometry_id[..., 1] != np.uint32(int(f2.GeometryInstanceID.invalid))
    hit_mask_rgba = np.zeros((HEIGHT, WIDTH, 4), dtype=np.float32)
    hit_mask_rgba[..., :3] = hit_mask[..., np.newaxis].astype(np.float32)
    hit_mask_rgba[..., 3] = 1.0
    _write_png(
        hit_mask_rgba,
        DEBUG_OUTPUT_DIR / f"{case.id}_{device_type.name}_hit_mask.png",
    )

    actual_tiles = {}
    for quadrant_name, expected_tile in case.expected_tiles.items():
        y_slice, x_slice = _patch_bounds(quadrant_name)
        valid_geometry = _geometry_mask(geometry_id, y_slice, x_slice, 0.50)
        patch_mean = material_color[y_slice, x_slice, :3][valid_geometry].mean(axis=0)
        actual_tiles[quadrant_name] = _classify_tile(patch_mean)
        if expected_tile != "missing":
            _assert_tile_is_upright(material_color, geometry_id, quadrant_name)

    assert actual_tiles == case.expected_tiles
