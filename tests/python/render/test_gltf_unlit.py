# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
from io import BytesIO
import json
from pathlib import Path

import numpy as np
import pytest
import slangpy as spy
from PIL import Image

import falcor2 as f2
import falcor2.testing.helpers as helpers
from falcor2.editor import SceneShaderHelper
from falcor2.rendernodes import ReferencePathTracerNode


def _write_unlit_geometry_fixture(tmp_path: Path) -> Path:
    image = BytesIO()
    Image.new("RGBA", (1, 1), (255, 255, 255, 255)).save(image, format="PNG")
    white_png = base64.b64encode(image.getvalue()).decode("ascii")
    front_positions = np.array(
        [
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [-1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        dtype="<f4",
    )
    front_normals = np.tile(np.array([0.0, 0.0, -1.0], dtype="<f4"), (4, 1))
    front_tangents = np.tile(np.array([1.0, 0.0, 0.0, 1.0], dtype="<f4"), (4, 1))
    uv0 = np.tile(np.array([0.25, 0.5], dtype="<f4"), (4, 1))
    uv1 = np.tile(np.array([0.75, 0.5], dtype="<f4"), (4, 1))
    colors = np.array(
        [
            [64, 128, 255, 255],
            [64, 128, 255, 0],
            [64, 128, 255, 255],
            [64, 128, 255, 0],
        ],
        dtype=np.uint8,
    )
    indices = np.array([0, 2, 1, 2, 3, 1], dtype="<u2")
    rear_positions = front_positions.copy()
    rear_positions[:, 2] = 0.2
    rear_normals = front_normals.copy()

    buffer_data = b"".join(
        value.tobytes()
        for value in (
            front_positions,
            front_normals,
            front_tangents,
            uv0,
            uv1,
            colors,
            indices,
            rear_positions,
            rear_normals,
            indices,
        )
    )
    assert len(buffer_data) == 360

    buffer_path = tmp_path / "unlit_geometry.bin"
    buffer_path.write_bytes(buffer_data)
    scene_path = tmp_path / "unlit_geometry.gltf"
    scene_path.write_text(
        json.dumps(
            {
                "asset": {"version": "2.0"},
                "extensionsUsed": ["KHR_materials_unlit"],
                "buffers": [{"uri": buffer_path.name, "byteLength": 360}],
                "bufferViews": [
                    {"buffer": 0, "byteOffset": 0, "byteLength": 48, "target": 34962},
                    {"buffer": 0, "byteOffset": 48, "byteLength": 48, "target": 34962},
                    {"buffer": 0, "byteOffset": 96, "byteLength": 64, "target": 34962},
                    {"buffer": 0, "byteOffset": 160, "byteLength": 32, "target": 34962},
                    {"buffer": 0, "byteOffset": 192, "byteLength": 32, "target": 34962},
                    {"buffer": 0, "byteOffset": 224, "byteLength": 16, "target": 34962},
                    {"buffer": 0, "byteOffset": 240, "byteLength": 12, "target": 34963},
                    {"buffer": 0, "byteOffset": 252, "byteLength": 48, "target": 34962},
                    {"buffer": 0, "byteOffset": 300, "byteLength": 48, "target": 34962},
                    {"buffer": 0, "byteOffset": 348, "byteLength": 12, "target": 34963},
                ],
                "accessors": [
                    {
                        "bufferView": 0,
                        "componentType": 5126,
                        "count": 4,
                        "type": "VEC3",
                        "min": [-1.0, -1.0, 0.0],
                        "max": [1.0, 1.0, 0.0],
                    },
                    {"bufferView": 1, "componentType": 5126, "count": 4, "type": "VEC3"},
                    {"bufferView": 2, "componentType": 5126, "count": 4, "type": "VEC4"},
                    {"bufferView": 3, "componentType": 5126, "count": 4, "type": "VEC2"},
                    {"bufferView": 4, "componentType": 5126, "count": 4, "type": "VEC2"},
                    {
                        "bufferView": 5,
                        "componentType": 5121,
                        "normalized": True,
                        "count": 4,
                        "type": "VEC4",
                    },
                    {"bufferView": 6, "componentType": 5123, "count": 6, "type": "SCALAR"},
                    {
                        "bufferView": 7,
                        "componentType": 5126,
                        "count": 4,
                        "type": "VEC3",
                        "min": [-1.0, -1.0, 0.2],
                        "max": [1.0, 1.0, 0.2],
                    },
                    {"bufferView": 8, "componentType": 5126, "count": 4, "type": "VEC3"},
                    {"bufferView": 9, "componentType": 5123, "count": 6, "type": "SCALAR"},
                ],
                "images": [{"uri": f"data:image/png;base64,{white_png}"}],
                "textures": [{"source": 0}],
                "materials": [
                    {
                        "name": "Front",
                        "pbrMetallicRoughness": {
                            "baseColorFactor": [0.8, 0.5, 0.25, 0.8],
                            "baseColorTexture": {"index": 0, "texCoord": 1},
                        },
                        "extensions": {"KHR_materials_unlit": {}},
                        "alphaMode": "MASK",
                        "alphaCutoff": 0.5,
                        "doubleSided": True,
                    },
                    {
                        "name": "Rear",
                        "pbrMetallicRoughness": {"baseColorFactor": [0.0, 1.0, 0.0, 1.0]},
                        "extensions": {"KHR_materials_unlit": {}},
                        "doubleSided": True,
                    },
                ],
                "meshes": [
                    {
                        "primitives": [
                            {
                                "attributes": {
                                    "POSITION": 0,
                                    "NORMAL": 1,
                                    "TANGENT": 2,
                                    "TEXCOORD_0": 3,
                                    "TEXCOORD_1": 4,
                                    "COLOR_0": 5,
                                },
                                "indices": 6,
                                "material": 0,
                            },
                            {
                                "attributes": {"POSITION": 7, "NORMAL": 8},
                                "indices": 9,
                                "material": 1,
                            },
                        ]
                    }
                ],
                "nodes": [{"mesh": 0}],
                "scenes": [{"nodes": [0]}],
                "scene": 0,
            }
        ),
        encoding="utf-8",
    )
    return scene_path


def _write_attribute_free_unlit_fixture(tmp_path: Path) -> Path:
    positions = np.array(
        [
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [-1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        dtype="<f4",
    )
    normals = np.tile(np.array([0.0, 0.0, -1.0], dtype="<f4"), (4, 1))
    indices = np.array([0, 2, 1, 2, 3, 1], dtype="<u2")
    buffer_data = positions.tobytes() + normals.tobytes() + indices.tobytes()
    assert len(buffer_data) == 108

    buffer_path = tmp_path / "attribute_free_unlit.bin"
    buffer_path.write_bytes(buffer_data)
    scene_path = tmp_path / "attribute_free_unlit.gltf"
    scene_path.write_text(
        json.dumps(
            {
                "asset": {"version": "2.0"},
                "extensionsUsed": ["KHR_materials_unlit"],
                "buffers": [{"uri": buffer_path.name, "byteLength": 108}],
                "bufferViews": [
                    {"buffer": 0, "byteOffset": 0, "byteLength": 48, "target": 34962},
                    {"buffer": 0, "byteOffset": 48, "byteLength": 48, "target": 34962},
                    {"buffer": 0, "byteOffset": 96, "byteLength": 12, "target": 34963},
                ],
                "accessors": [
                    {
                        "bufferView": 0,
                        "componentType": 5126,
                        "count": 4,
                        "type": "VEC3",
                        "min": [-1.0, -1.0, 0.0],
                        "max": [1.0, 1.0, 0.0],
                    },
                    {"bufferView": 1, "componentType": 5126, "count": 4, "type": "VEC3"},
                    {"bufferView": 2, "componentType": 5123, "count": 6, "type": "SCALAR"},
                ],
                "materials": [
                    {
                        "name": "AttributeFree",
                        "pbrMetallicRoughness": {
                            "baseColorFactor": [0.25, 0.5, 0.75, 0.8],
                        },
                        "extensions": {"KHR_materials_unlit": {}},
                        "alphaMode": "MASK",
                        "alphaCutoff": 0.5,
                        "doubleSided": True,
                    }
                ],
                "meshes": [
                    {
                        "primitives": [
                            {
                                "attributes": {"POSITION": 0, "NORMAL": 1},
                                "indices": 2,
                                "material": 0,
                            }
                        ]
                    }
                ],
                "nodes": [{"mesh": 0}],
                "scenes": [{"nodes": [0]}],
                "scene": 0,
            }
        ),
        encoding="utf-8",
    )
    return scene_path


def _create_primary_emission_renderer(device: spy.Device) -> ReferencePathTracerNode:
    renderer = ReferencePathTracerNode.create(device)
    renderer.output_spec = f2.ContainerSpec.texture2d(spy.Format.rgba32_float)
    renderer.enable_nee = False
    renderer.enable_mis = False
    renderer.enable_russian_roulette = False
    renderer.max_depth = 1
    renderer.use_background_color = True
    renderer.background_color = spy.float3(0.0)
    return renderer


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_unlit_opacity_properties(device_type: spy.DeviceType, device: spy.Device) -> None:
    scene = f2.Scene.create(device)
    material = scene.create_material(f2.UnlitMaterial)
    assert material.base_color_texture_texcoord == 0
    assert material.use_vertex_color is False
    material.alpha_mode = f2.AlphaMode.mask
    scene.update()
    assert scene.required_capabilities.has(f2.SceneCapabilityFlags.opacity_evaluation)
    assert not scene.required_capabilities.has(f2.SceneCapabilityFlags.geometry_properties)

    material.base_color_texture_texcoord = 1
    scene.update()
    assert scene.required_capabilities.has(f2.SceneCapabilityFlags.opacity_evaluation)
    assert scene.required_capabilities.has(f2.SceneCapabilityFlags.geometry_properties)

    material.base_color_texture_texcoord = 0
    scene.update()
    assert scene.required_capabilities.has(f2.SceneCapabilityFlags.opacity_evaluation)
    assert not scene.required_capabilities.has(f2.SceneCapabilityFlags.geometry_properties)

    material.use_vertex_color = True
    scene.update()
    assert scene.required_capabilities.has(f2.SceneCapabilityFlags.opacity_evaluation)
    assert scene.required_capabilities.has(f2.SceneCapabilityFlags.geometry_properties)

    material.alpha_mode = f2.AlphaMode.opaque
    scene.update()
    assert not scene.required_capabilities.has(f2.SceneCapabilityFlags.opacity_evaluation)
    assert scene.required_capabilities.has(f2.SceneCapabilityFlags.geometry_properties)

    material.alpha_mode = f2.AlphaMode.blend
    scene.update()
    assert scene.required_capabilities.has(f2.SceneCapabilityFlags.opacity_evaluation)
    assert scene.required_capabilities.has(f2.SceneCapabilityFlags.geometry_properties)

    material.use_vertex_color = False
    scene.update()
    assert scene.required_capabilities.has(f2.SceneCapabilityFlags.opacity_evaluation)
    assert not scene.required_capabilities.has(f2.SceneCapabilityFlags.geometry_properties)
    material.base_color_texture_texcoord = 7
    with pytest.raises(RuntimeError, match=r"range \[0, 7\]"):
        material.base_color_texture_texcoord = 8
    assert material.base_color_texture_texcoord == 7


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("image_source", ["embedded", "external"])
def test_gltf_scene_unlit_material_mapping(
    tmp_path: Path, device_type: spy.DeviceType, image_source: str
) -> None:
    image = BytesIO()
    Image.new("RGBA", (1, 1), (255, 255, 255, 255)).save(image, format="PNG")
    white_png = base64.b64encode(image.getvalue()).decode("ascii")
    image_uri = f"data:image/png;base64,{white_png}"
    if image_source == "external":
        image_path = tmp_path / "white.png"
        image_path.write_bytes(image.getvalue())
        image_uri = image_path.name
    scene_path = tmp_path / "unlit.gltf"
    scene_path.write_text(
        json.dumps(
            {
                "asset": {"version": "2.0"},
                "extensionsUsed": ["KHR_materials_unlit"],
                "images": [{"uri": image_uri}],
                "textures": [{"source": 0}],
                "materials": [
                    {
                        "name": "Unlit",
                        "pbrMetallicRoughness": {
                            "baseColorFactor": [0.25, 0.5, 0.75, 0.625],
                            "baseColorTexture": {"index": 0, "texCoord": 1},
                            "metallicFactor": 0.2,
                            "roughnessFactor": 0.3,
                        },
                        "extensions": {"KHR_materials_unlit": {}},
                        "alphaMode": "MASK",
                        "alphaCutoff": 0.375,
                        "doubleSided": False,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    device = helpers.get_device(device_type)
    scene = f2.Scene.load(device, scene_path)
    scene.update()
    material = scene.materials.find("Unlit")

    assert isinstance(material, f2.UnlitMaterial)
    assert material.slang_type_name == "UnlitMaterial"
    assert material.base_color_factor == spy.float3(0.25, 0.5, 0.75)
    assert material.alpha_factor == pytest.approx(0.625)
    assert material.alpha_mode == f2.AlphaMode.mask
    assert material.alpha_cutoff == pytest.approx(0.375)
    assert material.double_sided is True
    assert material.enable_backface_culling is True
    if image_source == "embedded":
        assert material.base_color_texture is not None
        assert material.base_color_texture.format == spy.Format.rgba8_unorm_srgb
        assert material.base_color_texture_path == Path()
    else:
        assert material.base_color_texture_path == tmp_path / "white.png"
    assert material.base_color_texture_texcoord == 1
    assert material.use_vertex_color is False
    assert material.flags == f2.MaterialFlags.unlit | f2.MaterialFlags.two_sided
    assert scene.required_capabilities.has(f2.SceneCapabilityFlags.opacity_evaluation)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_gltf_unlit_uv1_vertex_color_alpha_mask(
    tmp_path: Path,
    device_type: spy.DeviceType,
    device: spy.Device,
) -> None:
    scene = f2.Scene.load(device, _write_unlit_geometry_fixture(tmp_path))
    front = scene.materials.find("Front")
    rear = scene.materials.find("Rear")
    geometry_instance = scene.components.find(type=f2.GeometryInstance)

    assert isinstance(front, f2.UnlitMaterial)
    assert isinstance(rear, f2.UnlitMaterial)
    assert front.enable_backface_culling is False
    assert rear.enable_backface_culling is False
    assert geometry_instance is not None
    assert front.base_color_texture_texcoord == 1
    assert front.use_vertex_color is True

    front.base_color_texture = device.create_texture(
        type=spy.TextureType.texture_2d,
        format=spy.Format.rgba32_float,
        width=2,
        height=1,
        usage=spy.TextureUsage.shader_resource,
        data=np.array(
            [[[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]]],
            dtype=np.float32,
        ),
    )
    scene.update()
    assert scene.required_capabilities.has(f2.SceneCapabilityFlags.opacity_evaluation)

    helper = SceneShaderHelper(device)
    source_module = spy.Module.load_from_file(device, "render/test_geometry_properties.slang")
    module, requirements = helper.prepare(scene, source_module)
    opacity_probe = (
        module.multi_uv_opacity_probe.as_func()
        .type_conformances(requirements.type_conformances)
        .write(helper.bind_scene)
        .return_type(spy.Tensor)
    )
    geometry_instance_ids = np.full(
        2,
        int(geometry_instance.geometry_instance_id),
        dtype=np.uint32,
    )
    opacities = opacity_probe(
        spy.Tensor.from_numpy(device, geometry_instance_ids),
        spy.Tensor.from_numpy(device, np.array([0, 1], dtype=np.uint32)),
        spy.Tensor.from_numpy(
            device,
            np.array([[0.45, 0.1], [0.45, 0.45]], dtype=np.float32),
        ),
    ).to_numpy()
    np.testing.assert_allclose(opacities, [0.72, 0.08], rtol=1e-5, atol=1e-5)

    camera = helpers.create_test_camera(
        scene,
        width=8,
        height=8,
        fov_y=35.0,
        position=spy.float3(0.0, 0.0, -2.0),
        rotation=spy.math.quat_from_look_at(
            spy.float3(0.0, 0.0, 1.0),
            spy.float3(0.0, 1.0, 0.0),
        ),
    )
    renderer = _create_primary_emission_renderer(device)
    renderer.pixel_filter = f2.BoxReconstructionFilter(width=0)
    image, _ = renderer(
        scene,
        camera,
        iteration=0,
        pixel_offset=spy.float2(0.0),
    )
    image_data = image.to_numpy()[..., :3]
    expected_front = np.array([0.8 * 64.0 / 255.0, 0.5 * 128.0 / 255.0, 0.25])

    np.testing.assert_allclose(image_data[4, 6], expected_front, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(image_data[4, 1], [0.0, 1.0, 0.0], rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_gltf_unlit_without_optional_vertex_attributes(
    tmp_path: Path,
    device_type: spy.DeviceType,
    device: spy.Device,
) -> None:
    scene = f2.Scene.load(device, _write_attribute_free_unlit_fixture(tmp_path))
    material = scene.materials.find("AttributeFree")
    geometry_instance = scene.components.find(type=f2.GeometryInstance)

    assert isinstance(material, f2.UnlitMaterial)
    assert geometry_instance is not None
    assert material.use_vertex_color is False
    scene.update()

    helper = SceneShaderHelper(device)
    source_module = spy.Module.load_from_file(device, "render/test_geometry_properties.slang")
    module, requirements = helper.prepare(scene, source_module)
    opacity_probe = (
        module.multi_uv_opacity_probe.as_func()
        .type_conformances(requirements.type_conformances)
        .write(helper.bind_scene)
        .return_type(spy.Tensor)
    )
    opacity = opacity_probe(
        spy.Tensor.from_numpy(
            device,
            np.array([int(geometry_instance.geometry_instance_id)], dtype=np.uint32),
        ),
        spy.Tensor.from_numpy(device, np.array([0], dtype=np.uint32)),
        spy.Tensor.from_numpy(device, np.array([[0.2, 0.3]], dtype=np.float32)),
    ).to_numpy()
    np.testing.assert_allclose(opacity, [0.8], rtol=1e-5, atol=1e-5)

    camera = helpers.create_test_camera(
        scene,
        width=8,
        height=8,
        fov_y=35.0,
        position=spy.float3(0.0, 0.0, -2.0),
        rotation=spy.math.quat_from_look_at(
            spy.float3(0.0, 0.0, 1.0),
            spy.float3(0.0, 1.0, 0.0),
        ),
    )
    renderer = _create_primary_emission_renderer(device)
    renderer.pixel_filter = f2.BoxReconstructionFilter(width=0)
    image, _ = renderer(
        scene,
        camera,
        iteration=0,
        pixel_offset=spy.float2(0.0),
    )
    np.testing.assert_allclose(
        image.to_numpy()[4, 4, :3],
        [0.25, 0.5, 0.75],
        rtol=1e-4,
        atol=1e-4,
    )
