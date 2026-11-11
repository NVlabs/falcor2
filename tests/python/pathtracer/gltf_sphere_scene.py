# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Self-contained imported sphere atlases, with unsanitized linear HDR readback."""

import base64
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
import slangpy as spy
import falcor2 as f2
import falcor2.testing.helpers as helpers
from falcor2.rendernodes import AccumulatorNode, ReferencePathTracerNode

from .gltf_sphere_cases import SphereCase

TILE_SIZE = 96
SPP = 512
SPACING = 3.2
CAMERA_DISTANCE = 100.0


def _write_scene(directory: Path, cases: list[SphereCase], columns: int) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    rows = math.ceil(len(cases) / columns)
    theta, phi = np.meshgrid(np.linspace(0, 2 * np.pi, 65), np.linspace(0, np.pi, 33))
    positions = (
        np.stack([np.sin(phi) * np.cos(theta), np.cos(phi), np.sin(phi) * np.sin(theta)], -1)
        .astype(np.float32)
        .reshape(-1, 3)
    )
    tangents = (
        np.stack([-np.sin(theta), np.zeros_like(theta), np.cos(theta), np.ones_like(theta)], -1)
        .astype(np.float32)
        .reshape(-1, 4)
    )
    uv = np.stack([theta / (2 * np.pi), phi / np.pi], -1).astype(np.float32).reshape(-1, 2)
    indices = []
    for y in range(32):
        for x in range(64):
            a = y * 65 + x
            if y > 0:
                indices.append([a, a + 1, a + 65])
            if y < 31:
                indices.append([a + 1, a + 66, a + 65])
    arrays = [
        positions,
        positions,
        tangents,
        uv,
        np.column_stack((1 - uv[:, 0], uv[:, 1] * 0.75 + 0.125)),
        np.tile([0.3, 0.7, 0.9, 1], (len(positions), 1)),
        tangents * np.array([1, 1, 1, -1]),
        np.asarray(indices, dtype=np.uint32).reshape(-1),
    ]
    binary = bytearray()
    accessors, views = [], []
    for i, array in enumerate(arrays):
        array = array.astype(np.uint32 if i == 7 else np.float32)
        payload = array.tobytes()
        views.append({"buffer": 0, "byteOffset": len(binary), "byteLength": len(payload)})
        accessor = {
            "bufferView": i,
            "componentType": 5125 if i == 7 else 5126,
            "count": len(array),
            "type": "SCALAR" if i == 7 else f"VEC{array.shape[1]}",
        }
        if i == 0:
            accessor.update(min=array.min(0).tolist(), max=array.max(0).tolist())
        accessors.append(accessor)
        binary.extend(payload)

    # Deliberately unequal channels and asymmetric patterns expose channel/UV mistakes.
    yy, xx = np.mgrid[:16, :16]
    checker = ((xx // 4 + yy // 4) % 2).astype(bool)
    color = np.where(checker[..., None], [220, 70, 30, 255], [35, 150, 215, 70]).astype(np.uint8)
    normal = np.where((xx < 8)[..., None], [170, 128, 248, 255], [128, 175, 246, 255]).astype(
        np.uint8
    )
    blue_zero = normal.copy()
    blue_zero[xx < 4, 2] = 0
    textures = [
        color,
        np.tile([40, 77, 204, 255], (16, 16, 1)).astype(np.uint8),
        normal,
        normal[..., [1, 0, 2, 3]],
        np.where(checker[..., None], [255, 128, 102, 255], [128, 255, 230, 255]).astype(np.uint8),
        np.tile([179, 77, 26, 166], (16, 16, 1)).astype(np.uint8),
        blue_zero,
        np.tile([26, 255, 255, 255], (16, 16, 1)).astype(np.uint8),
    ]
    images = []
    for i, data in enumerate(textures):
        path = directory / f"texture_{i}.png"
        Image.fromarray(data).save(path)
        images.append(
            {"uri": "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()}
        )
        path.unlink()

    meshes, nodes = [], []
    for i, case in enumerate(cases):
        attributes = {
            "POSITION": 0,
            "NORMAL": 1,
            "TANGENT": 6 if case.mirrored else 2,
            "TEXCOORD_0": 3,
            "TEXCOORD_1": 4,
        }
        if case.vertex_color:
            attributes["COLOR_0"] = 5
        meshes.append(
            {
                "name": case.name,
                "primitives": [{"attributes": attributes, "indices": 7, "material": i}],
            }
        )
        x, y = (i % columns - (columns - 1) / 2) * SPACING, (
            (rows - 1) / 2 - i // columns
        ) * SPACING
        # Changing radius preserves angular size by moving the physical sphere farther away.
        nodes.append(
            {
                "name": case.name,
                "mesh": i,
                "translation": [
                    x * case.radius,
                    y * case.radius,
                    CAMERA_DISTANCE * (1 - case.radius),
                ],
                "scale": [case.radius] * 3,
            }
        )

    def extensions(value: Any) -> set[str]:
        if isinstance(value, dict):
            found = set(value.get("extensions", {}))
            for child in value.values():
                found.update(extensions(child))
            return found
        if isinstance(value, list):
            return set().union(*(extensions(child) for child in value))
        return set()

    materials = [case.material for case in cases]
    document = {
        "asset": {"version": "2.0"},
        "buffers": [
            {
                "byteLength": len(binary),
                "uri": "data:application/octet-stream;base64," + base64.b64encode(binary).decode(),
            }
        ],
        "bufferViews": views,
        "accessors": accessors,
        "images": images,
        "samplers": [{"wrapS": 10497, "wrapT": 33648, "minFilter": 9729, "magFilter": 9729}],
        "textures": [{"source": i, "sampler": 0} for i in range(len(images))],
        "materials": materials,
        "meshes": meshes,
        "nodes": nodes,
        "scenes": [{"nodes": list(range(len(nodes)))}],
        "scene": 0,
        "extensionsUsed": sorted(extensions(materials)),
    }
    path = directory / "spheres.gltf"
    path.write_text(json.dumps(document))
    return path


def render_spheres(
    device: spy.Device,
    directory: Path,
    cases: list[SphereCase],
    *,
    columns: int = 4,
    spp: int = SPP,
    seed: int = 0,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, float]]:
    start = perf_counter()
    columns = min(columns, len(cases))
    rows = math.ceil(len(cases) / columns)
    assert len({case.mode for case in cases}) == 1
    options = f2.ImportOptions()
    options.gltf_material_mode = getattr(f2.GLTFMaterialMode, cases[0].mode)
    scene = f2.Scene.load(device, _write_scene(directory, cases, columns), options)
    expected_types = {
        "standard_material": "StandardMaterial",
        "openpbr": "OpenPBRMaterial",
        "materialx": "MaterialXMaterial",
        "mdl": "MDLMaterial",
    }
    for case in cases:
        material = scene.materials.find(case.name)
        expected = expected_types.get(
            case.mode,
            (
                "GLTFSpecGlossMaterial"
                if "KHR_materials_pbrSpecularGlossiness" in case.material.get("extensions", {})
                else "GLTFMaterial"
            ),
        )
        # Generated MDL/MaterialX shader type names are specialized; OpenPBR has no Python subclass.
        if case.mode in ("mdl", "materialx"):
            assert isinstance(material, getattr(f2, expected)), case.name
        else:
            assert material is not None, case.name
            actual = material.slang_type_name
            if expected in ("GLTFMaterial", "GLTFSpecGlossMaterial"):
                assert actual.startswith(f"{expected}<"), f"{case.name}: {actual}"
            else:
                assert actual == expected, f"{case.name}: {actual}"
        if case.bc5:
            # glTF PNG cannot store BC5. Replace only this imported texture with a real GPU BC5 block.
            block = np.array([160, 0, 0, 0, 0, 0, 0, 0, 144, 0, 0, 0, 0, 0, 0, 0], dtype=np.uint8)
            texture = device.create_texture(
                type=spy.TextureType.texture_2d,
                format=spy.Format.bc5_unorm,
                width=4,
                height=4,
                usage=spy.TextureUsage.shader_resource,
            )
            texture.copy_from_numpy(block)
            material.normal_texture = texture

    yy, xx = np.mgrid[:128, :256]
    v = yy / 128
    env = np.zeros((128, 256, 4), dtype=np.float32)
    env[..., :3] = 0.12 + 0.3 * (1 - v[..., None])
    env[..., :3] += (((xx // 32 + yy // 32) % 2) * 0.12)[..., None]
    env[20:55, 40:100, :3] += [3, 2.7, 2.4]
    env[35:80, 170:190, :3] += [0.8, 1.0, 1.5]
    env[..., 3] = 1
    env_path = directory / "studio.exr"
    spy.Bitmap(env).write(str(env_path))
    scene.create_entity().create_component(f2.EnvMapLight).env_map_path = str(env_path)
    camera = helpers.create_test_camera(
        scene,
        columns * TILE_SIZE,
        rows * TILE_SIZE,
        fov_y=math.degrees(2 * math.atan(rows * SPACING / (2 * CAMERA_DISTANCE))),
        position=spy.float3(0, 0, CAMERA_DISTANCE),
    )
    scene.update()
    node = ReferencePathTracerNode.create(device)
    node.output_spec = f2.ContainerSpec.texture2d(spy.Format.rgba32_float)
    # Show the same environment that the spheres reflect and transmit.
    node.use_background_color = False
    node.enable_depth_of_field = False
    node.enable_russian_roulette = False
    node.enable_precise_ray_offsets = True
    node.max_depth = 12
    node.guide_output_specs = {
        "nan_count": f2.ContainerSpec.auto(),
        "geometry_id": f2.ContainerSpec.auto(),
    }
    accumulator = AccumulatorNode.create(
        device, sanitize_non_finite=False, clamp_sample_luminance=False
    )
    prepared = perf_counter()
    # First dispatch includes specialization/compilation. Later samples reuse it.
    color, guides = node(scene, camera, iteration=seed)
    result = accumulator(color)
    device.wait()
    first = perf_counter()
    for iteration in range(1, spp):
        color, guides = node(scene, camera, iteration=seed + iteration)
        result = accumulator(color)
    image = result.to_numpy()
    guide_arrays = {name: value.to_numpy() for name, value in guides.items() if value is not None}
    assert np.isfinite(image).all(), f"NaN/Inf radiance in {[case.name for case in cases]}"
    assert not guide_arrays["nan_count"].any(), f"NaN samples in {[case.name for case in cases]}"
    for i, case in enumerate(cases):
        ids = guide_arrays["geometry_id"][
            i // columns * TILE_SIZE : (i // columns + 1) * TILE_SIZE,
            i % columns * TILE_SIZE : (i % columns + 1) * TILE_SIZE,
        ]
        material_id = int(scene.materials.find(case.name).material_id)
        visible = (ids[..., 1] != 0xFFFFFFFF) & (ids[..., 3] == material_id)
        assert (
            np.count_nonzero(visible) > TILE_SIZE * TILE_SIZE * 0.05
        ), f"Missing sphere {case.name}"
    return (
        image,
        guide_arrays,
        {
            "setup": prepared - start,
            "first_sample": first - prepared,
            "remaining_samples_and_readback": perf_counter() - first,
        },
    )


def save_contact_sheet(
    image: np.ndarray, cases: list[SphereCase], path: Path, columns: int = 4
) -> None:
    """Diagnostic display only; comparisons always use the untouched HDR grid."""
    columns = min(columns, len(cases))
    rgb = np.maximum(image[..., :3], 0)
    display = Image.fromarray(np.uint8(np.clip((rgb / (1 + rgb)) ** (1 / 2.2) * 255, 0, 255)))
    draw = ImageDraw.Draw(display)
    for i, case in enumerate(cases):
        draw.text(
            (i % columns * TILE_SIZE + 3, i // columns * TILE_SIZE + 3),
            case.name,
            fill="white",
            stroke_width=1,
            stroke_fill="black",
        )
    display.save(path)
