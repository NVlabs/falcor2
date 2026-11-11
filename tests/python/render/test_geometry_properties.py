# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import slangpy as spy

import falcor2 as f2
import falcor2.testing.helpers as helpers
from falcor2.editor import SceneShaderHelper
from falcor2.rendernodes import ReferencePathTracerNode
from falcor2.testing.image_test_plugin import ImageTest


@dataclass(frozen=True)
class _TriangleScene:
    scene: f2.Scene
    geometry_instance_id: int


@dataclass(frozen=True)
class _AdditionalAttributeScene:
    scene: f2.Scene
    geometries: list[f2.StaticMeshGeometry]
    geometry_instance_ids: list[int]
    attributes: list[dict[int, np.ndarray]]


from .geometry_property_helpers import (
    TRIANGLE_POSITIONS,
    TRIANGLE_INDICES,
    read_material_probe as _read_material_probe,
)

MULTI_UV_SCENE_PATH = Path("data/assets/kronos/MultiUVTest/glTF/MultiUVTest.gltf")
VERTEX_COLOR_SCENE_PATH = Path("data/assets/kronos/VertexColorTest/glTF/VertexColorTest.gltf")
USD_ADDITIONAL_GEOMETRY_PROPERTIES_PATH = Path(
    "data/assets/test_usd_importer/test_additional_geometry_properties.usda"
)
MATERIALX_GEOMETRY_PROPERTY_DOCUMENT = """<materialx version="1.39" colorspace="lin_rec709">
  <geompropdef name="Pref" type="vector3" />
  <geompropvalue name="pref" type="vector3">
    <input name="geomprop" type="string" value="Pref" />
    <input name="default" type="vector3" value="0.11, 0.12, 0.13" />
  </geompropvalue>
  <texcoord name="uv1" type="vector2"><input name="index" type="integer" value="1" /></texcoord>
  <convert name="uv3" type="vector3"><input name="in" type="vector2" nodename="uv1" /></convert>
  <geomcolor name="vertex_color" type="color3"><input name="index" type="integer" value="0" /></geomcolor>
  <convert name="color_vector" type="vector3"><input name="in" type="color3" nodename="vertex_color" /></convert>
  <add name="pref_uv" type="vector3">
    <input name="in1" type="vector3" nodename="pref" />
    <input name="in2" type="vector3" nodename="uv3" />
  </add>
  <add name="sum" type="vector3">
    <input name="in1" type="vector3" nodename="pref_uv" />
    <input name="in2" type="vector3" nodename="color_vector" />
  </add>
  <position name="position_object" type="vector3">
    <input name="space" type="string" value="object" />
  </position>
  <add name="with_position" type="vector3">
    <input name="in1" type="vector3" nodename="sum" />
    <input name="in2" type="vector3" nodename="position_object" />
  </add>
  <convert name="tint" type="color3"><input name="in" type="vector3" nodename="with_position" /></convert>
  <standard_surface name="surface" type="surfaceshader">
    <input name="base_color" type="color3" nodename="tint" />
    <input name="specular" type="float" value="0" />
  </standard_surface>
  <surfacematerial name="M" type="material">
    <input name="surfaceshader" type="surfaceshader" nodename="surface" />
  </surfacematerial>
</materialx>"""
SHARED_CUSTOM_ATTRIBUTE_NAME = "shared_custom"
OTHER_CUSTOM_ATTRIBUTE_NAME = "other_custom"


def _set_triangle_mesh(
    geometry: f2.StaticMeshGeometry,
    attributes: list[f2.MeshVertexAttribute],
) -> None:
    geometry.set_mesh_data(
        positions=TRIANGLE_POSITIONS,
        sub_mesh_indices=[TRIANGLE_INDICES],
        vertex_attributes=attributes,
    )


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


def _add_uv_opacity_quad(
    scene: f2.Scene,
    material: f2.Material,
    z: float,
    uv0: tuple[float, float],
    uv1: tuple[float, float] | None = None,
    color: tuple[float, float, float, float] | None = None,
) -> f2.GeometryInstance:
    geometry = scene.create_geometry(f2.StaticMeshGeometry)
    positions = np.array(
        [[-0.7, -0.7, z], [0.7, -0.7, z], [-0.7, 0.7, z], [0.7, 0.7, z]],
        dtype=np.float32,
    )
    vertex_attributes = []
    if uv1 is not None:
        vertex_attributes.append(
            f2.MeshVertexAttribute(
                f2.MeshAttributeSemantic.tex_coord,
                1,
                np.tile(np.asarray(uv1, dtype=np.float32), (4, 1)),
            )
        )
    if color is not None:
        vertex_attributes.append(
            f2.MeshVertexAttribute(
                f2.MeshAttributeSemantic.color,
                0,
                np.tile(np.asarray(color, dtype=np.float32), (4, 1)),
            )
        )
    geometry.set_mesh_data(
        positions=positions,
        sub_mesh_indices=[np.array([[0, 2, 1], [2, 3, 1]], dtype=np.uint32)],
        normals=np.tile(np.array([0.0, 0.0, -1.0], dtype=np.float32), (4, 1)),
        tangents=np.tile(np.array([1.0, 0.0, 0.0], dtype=np.float32), (4, 1)),
        handedness=np.ones(4, dtype=np.float32),
        texcoords=np.tile(np.asarray(uv0, dtype=np.float32), (4, 1)),
        vertex_attributes=vertex_attributes or None,
    )
    entity = scene.create_entity()
    geometry_instance = entity.create_component(f2.GeometryInstance)
    geometry_instance.geometry = geometry
    geometry_instance.materials = [material]
    return geometry_instance


@pytest.fixture
def additional_attribute_scene(device: spy.Device) -> _AdditionalAttributeScene:
    scene = f2.Scene.create(device)
    material = scene.create_material(f2.StandardMaterial)
    attributes = [
        {
            1: np.array([[2.0, 3.0], [4.0, 5.0], [6.0, 7.0]], dtype=np.float32),
            16: np.array([[71.0, 72.0], [73.0, 74.0], [75.0, 76.0]], dtype=np.float32),
        },
        {
            3: np.array(
                [[11.0, 12.0, 13.0], [14.0, 15.0, 16.0], [17.0, 18.0, 19.0]], dtype=np.float32
            ),
            8: np.array(
                [[21.0, 22.0, 23.0, 24.0], [25.0, 26.0, 27.0, 28.0], [29.0, 30.0, 31.0, 32.0]],
                dtype=np.float32,
            ),
            16: np.array([[81.0, 82.0], [83.0, 84.0], [85.0, 86.0]], dtype=np.float32),
        },
        {
            0: np.array(
                [[41.0, 42.0, 43.0], [44.0, 45.0, 46.0], [47.0, 48.0, 49.0]], dtype=np.float32
            ),
            17: np.array([[91.0], [92.0], [93.0]], dtype=np.float32),
        },
    ]
    attribute_definitions = [
        [
            f2.MeshVertexAttribute(f2.MeshAttributeSemantic.tex_coord, 1, attributes[0][1]),
            f2.MeshVertexAttribute(
                f2.MeshAttributeSemantic.unknown,
                0,
                attributes[0][16],
                name=SHARED_CUSTOM_ATTRIBUTE_NAME,
            ),
        ],
        [
            f2.MeshVertexAttribute(f2.MeshAttributeSemantic.tex_coord, 3, attributes[1][3]),
            f2.MeshVertexAttribute(f2.MeshAttributeSemantic.color, 0, attributes[1][8]),
            f2.MeshVertexAttribute(
                f2.MeshAttributeSemantic.unknown,
                0,
                attributes[1][16],
                name=SHARED_CUSTOM_ATTRIBUTE_NAME,
            ),
        ],
        [
            f2.MeshVertexAttribute(f2.MeshAttributeSemantic.tex_coord, 0, attributes[2][0]),
            f2.MeshVertexAttribute(
                f2.MeshAttributeSemantic.weights,
                0,
                attributes[2][17],
                name=OTHER_CUSTOM_ATTRIBUTE_NAME,
            ),
        ],
    ]
    geometries: list[f2.StaticMeshGeometry] = []
    geometry_instances: list[f2.GeometryInstance] = []
    for definitions in attribute_definitions:
        geometry = scene.create_geometry(f2.StaticMeshGeometry)
        _set_triangle_mesh(geometry, definitions)
        entity = scene.create_entity()
        geometry_instance = entity.create_component(f2.GeometryInstance)
        geometry_instance.geometry = geometry
        geometry_instance.materials = [material]
        geometries.append(geometry)
        geometry_instances.append(geometry_instance)
    scene.update()
    geometry_instance_ids = [int(instance.geometry_instance_id) for instance in geometry_instances]
    return _AdditionalAttributeScene(scene, geometries, geometry_instance_ids, attributes)


def _create_additional_attribute_probe(
    device: spy.Device, scene: f2.Scene
) -> tuple[SceneShaderHelper, Any]:
    helper = SceneShaderHelper(device)
    source_module = spy.Module.load_from_file(device, "render/test_geometry_properties.slang")
    module, requirements = helper.prepare(scene, source_module)
    probe = (
        module.additional_attribute_probe.as_func()
        .type_conformances(requirements.type_conformances)
        .write(helper.bind_scene)
        .return_type(spy.Tensor)
    )
    return helper, probe


def _read_additional_attribute(
    device: spy.Device,
    probe: Any,
    geometry_instance_id: int,
    mask_bit: int,
    word_count: int,
) -> Any:
    result = probe(
        spy.Tensor.from_numpy(device, np.array([geometry_instance_id], dtype=np.uint32)),
        spy.Tensor.from_numpy(device, np.array([mask_bit], dtype=np.uint32)),
        spy.Tensor.from_numpy(device, np.array([word_count], dtype=np.uint32)),
    )
    cursor = result.cursor()
    cursor.load()
    return cursor[0].read()


def _create_attribute_accessor_probe(device: spy.Device, scene: f2.Scene) -> Any:
    helper = SceneShaderHelper(device)
    source_module = spy.Module.load_from_file(device, "render/test_geometry_properties.slang")
    module, requirements = helper.prepare(scene, source_module)
    return (
        module.attribute_accessor_probe.as_func()
        .type_conformances(requirements.type_conformances)
        .write(helper.bind_scene)
        .return_type(spy.Tensor)
    )


def _create_lss_attribute_accessor_probe(device: spy.Device, scene: f2.Scene) -> Any:
    helper = SceneShaderHelper(device)
    source_module = spy.Module.load_from_file(device, "render/test_geometry_properties.slang")
    module, requirements = helper.prepare(scene, source_module)
    return (
        module.lss_attribute_accessor_probe.as_func()
        .type_conformances(requirements.type_conformances)
        .write(helper.bind_scene)
        .return_type(spy.Tensor)
    )


def _read_attribute_accessor(
    device: spy.Device,
    probe: Any,
    geometry_instance_id: int,
    attribute_id: int,
    barycentrics: np.ndarray,
) -> Any:
    result = probe(
        spy.Tensor.from_numpy(device, np.array([geometry_instance_id], dtype=np.uint32)),
        spy.Tensor.from_numpy(device, np.array([attribute_id], dtype=np.uint32)),
        spy.Tensor.from_numpy(device, barycentrics),
    )
    cursor = result.cursor()
    cursor.load()
    return cursor[0].read()


def _read_lss_attribute_accessors(
    device: spy.Device,
    probe: Any,
    geometry_instance_id: int,
    primitive_indices: np.ndarray,
    attribute_ids: np.ndarray,
    uvs: np.ndarray,
) -> list[Any]:
    result = probe(
        spy.Tensor.from_numpy(
            device,
            np.full(primitive_indices.shape, geometry_instance_id, dtype=np.uint32),
        ),
        spy.Tensor.from_numpy(device, primitive_indices),
        spy.Tensor.from_numpy(device, attribute_ids),
        spy.Tensor.from_numpy(device, uvs),
    )
    cursor = result.cursor()
    cursor.load()
    return [cursor[index].read() for index in range(primitive_indices.size)]


def _assert_zero_float_outputs(result: Any) -> None:
    assert result["value_float"] == 0.0
    for name, component_count in (
        ("value2", 2),
        ("value3", 3),
        ("value4", 4),
        ("additional_texcoord2", 2),
        ("additional_texcoord3", 3),
        ("additional_texcoord4", 4),
    ):
        np.testing.assert_array_equal(result[name], np.zeros(component_count, dtype=np.float32))


def _assert_missing_color(result: Any) -> None:
    assert result["color_success"] is False
    np.testing.assert_array_equal(result["color"], np.zeros(4, dtype=np.float32))


def _assert_zero_discrete_outputs(result: Any) -> None:
    assert result["value_int"] == 0
    assert result["value_bool"] is False
    assert result["int_vector_success_mask"] == 0
    assert result["bool_vector_success_mask"] == 0
    for name, component_count, dtype in (
        ("value_int2", 2, np.int32),
        ("value_int3", 3, np.int32),
        ("value_int4", 4, np.int32),
        ("value_bool2", 2, np.bool_),
        ("value_bool3", 3, np.bool_),
        ("value_bool4", 4, np.bool_),
    ):
        np.testing.assert_array_equal(result[name], np.zeros(component_count, dtype=dtype))


def _create_triangle_scene(device: spy.Device) -> _TriangleScene:
    scene = f2.Scene.create(device)
    material = scene.create_material(f2.StandardMaterial)

    geometry = scene.create_geometry(f2.StaticMeshGeometry)
    geometry.set_mesh_data(
        positions=TRIANGLE_POSITIONS,
        sub_mesh_indices=[TRIANGLE_INDICES],
    )

    entity = scene.create_entity()
    geometry_instance = entity.create_component(f2.GeometryInstance)
    geometry_instance.geometry = geometry
    geometry_instance.materials = [material]
    scene.update()

    return _TriangleScene(
        scene=scene,
        geometry_instance_id=int(geometry_instance.geometry_instance_id),
    )


@pytest.fixture
def triangle_scene(device: spy.Device) -> _TriangleScene:
    return _create_triangle_scene(device)


def _read_packed_hit_info(
    device: spy.Device,
    triangle_scene: _TriangleScene,
    barycentrics: np.ndarray,
    specified_capabilities: f2.SceneCapabilities = f2.SceneCapabilities.all(),
) -> Any:
    helper = SceneShaderHelper(device)
    source_module = spy.Module.load_from_file(device, "render/test_geometry_properties.slang")
    module, requirements = helper.prepare(
        triangle_scene.scene,
        source_module,
        policy=f2.SceneCapabilityPolicy.specified,
        specified_capabilities=specified_capabilities,
    )
    probe = (
        module.packed_hit_info_probe.as_func()
        .type_conformances(requirements.type_conformances)
        .write(helper.bind_scene)
        .return_type(spy.Tensor)
    )
    result = probe(
        spy.Tensor.from_numpy(
            device,
            np.array([triangle_scene.geometry_instance_id], dtype=np.uint32),
        ),
        spy.Tensor.from_numpy(device, barycentrics),
    )
    cursor = result.cursor()
    cursor.load()
    return cursor[0].read()


@pytest.mark.nightly_if_not_vk
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_triangle_surface_interaction_packs_hit_info(
    device_type: spy.DeviceType,
    device: spy.Device,
    triangle_scene: _TriangleScene,
) -> None:
    barycentrics = np.array([[0.2, 0.3]], dtype=np.float32)
    packed_hit_info = _read_packed_hit_info(device, triangle_scene, barycentrics)

    expected = np.array(
        [
            0,
            *barycentrics.view(np.uint32)[0],
        ],
        dtype=np.uint32,
    )
    assert int(packed_hit_info["geometry_type"]) == int(f2.GeometryType.triangle)
    assert int(packed_hit_info["geometry_instance_id"]) == triangle_scene.geometry_instance_id
    np.testing.assert_array_equal(packed_hit_info["payload"], expected)


@pytest.mark.nightly_if_not_vk
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_unused_geometry_properties_can_be_omitted(
    device_type: spy.DeviceType,
    device: spy.Device,
) -> None:
    triangle_scene = _create_triangle_scene(device)
    packed_hit_info = _read_packed_hit_info(
        device, triangle_scene, np.array([[0.2, 0.3]], dtype=np.float32), f2.SceneCapabilities()
    )
    assert int(packed_hit_info["geometry_type"]) == np.iinfo(np.uint32).max
    assert int(packed_hit_info["geometry_instance_id"]) == 0
    np.testing.assert_array_equal(packed_hit_info["payload"], np.zeros(3, dtype=np.uint32))


@pytest.mark.nightly_if_not_vk
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_geometry_accessor_convenience_defaults(
    device_type: spy.DeviceType,
    device: spy.Device,
) -> None:
    scene = f2.Scene.create(device)
    scene.update()

    helper = SceneShaderHelper(device)
    source_module = spy.Module.load_from_file(device, "render/test_geometry_properties.slang")
    module, requirements = helper.prepare(scene, source_module)
    probe = (
        module.attribute_fallback_probe.as_func()
        .type_conformances(requirements.type_conformances)
        .write(helper.bind_scene)
        .return_type(spy.Tensor)
    )
    result = probe(spy.Tensor.from_numpy(device, np.array([0], dtype=np.uint32)))
    cursor = result.cursor()
    cursor.load()
    value = cursor[0].read()

    np.testing.assert_array_equal(value["generic_zero"], [0.0, 0.0, 0.0])
    np.testing.assert_array_equal(value["generic_default"], [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(value["texcoord2_default"], [4.0, 5.0])
    np.testing.assert_array_equal(value["texcoord3_default"], [6.0, 7.0, 8.0])
    np.testing.assert_array_equal(value["texcoord4_default"], [9.0, 10.0, 11.0, 12.0])
    np.testing.assert_array_equal(value["color_default"], [13.0, 14.0, 15.0, 16.0])
    np.testing.assert_array_equal(value["identity_point"], [17.0, 18.0, 19.0])


@pytest.mark.nightly
@pytest.mark.parametrize(
    "language,device_type,layering_mode",
    [
        (language, device_type, mode)
        for language in ("materialx", "mdl")
        for device_type in helpers.DEFAULT_DEVICE_TYPES
        for mode in [
            None,
            # Alternate MaterialX modes are exercised only on this representative BSDF-backed probe.
            f2.MaterialXLayeringMode.closure_tree,
        ]
        if (language == "mdl" and mode is None)
        or (language == "materialx" and device_type != spy.DeviceType.cuda)
    ],
)
def test_materialx_reads_production_geometry_accessor(
    language: str,
    device_type: spy.DeviceType,
    device: spy.Device,
    layering_mode: f2.MaterialXLayeringMode | None,
) -> None:
    scene = f2.Scene.create(
        device,
        f2.SceneConfig(uv_origin=f2.UVOrigin.lower_left),
    )
    if language == "materialx":
        material = scene.create_material(
            f2.MaterialXMaterial,
            f2.Properties(
                {
                    "mtlx_buffer": MATERIALX_GEOMETRY_PROPERTY_DOCUMENT,
                    "mtlx_node_name": "M",
                    "mtlx_editable_params": "pref_default",
                    "inputs:pref_default": spy.float3(0.11, 0.12, 0.13),
                }
            ),
        )
    else:
        material = scene.create_material(
            f2.MDLMaterial,
            f2.Properties(
                {
                    "mdl_library_path": str(
                        Path(__file__).parent / "data/mdl_generated_material_pipeline"
                    ),
                    "mdl_material_name": "geometry_properties::Default",
                }
            ),
        )
    if layering_mode is not None:
        material.mtlx_layering_mode = layering_mode
    default_property = "inputs:pref_default" if language == "materialx" else "fallback"
    np.testing.assert_allclose(material[default_property], [0.11, 0.12, 0.13])

    uv1 = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=np.float32)
    colors = np.array(
        [[0.1, 0.1, 0.1, 0.8], [0.2, 0.3, 0.4, 0.9], [0.4, 0.5, 0.6, 1.0]],
        dtype=np.float32,
    )
    pref = np.array(
        [[0.01, 0.02, 0.03], [0.04, 0.05, 0.06], [0.07, 0.08, 0.09]],
        dtype=np.float32,
    )
    shared_attributes = [
        f2.MeshVertexAttribute(f2.MeshAttributeSemantic.tex_coord, 1, uv1),
        f2.MeshVertexAttribute(f2.MeshAttributeSemantic.color, 0, colors),
    ]

    instances = []
    for include_pref in (True, False):
        geometry = scene.create_geometry(f2.StaticMeshGeometry)
        attributes = list(shared_attributes)
        if include_pref:
            attributes.append(
                f2.MeshVertexAttribute(f2.MeshAttributeSemantic.unknown, 0, pref, name="Pref")
            )
        _set_triangle_mesh(geometry, attributes)
        entity = scene.create_entity()
        if not include_pref:
            transform = f2.Transform()
            transform.translation = spy.float3(3.0, -2.0, 5.0)
            entity.transform = transform
        instance = entity.create_component(f2.GeometryInstance)
        instance.geometry = geometry
        instance.materials = [material]
        instances.append(instance)
    scene.update()
    assert scene.required_capabilities.has(f2.SceneCapabilityFlags.geometry_properties)

    helper = SceneShaderHelper(device)
    source_module = spy.Module.load_from_file(device, "render/test_geometry_properties.slang")
    module, requirements = helper.prepare(scene, source_module)
    probe = (
        module.multi_uv_material_probe.as_func()
        .type_conformances(requirements.type_conformances)
        .write(helper.bind_scene)
        .return_type(spy.Tensor)
    )
    result = probe(
        spy.Tensor.from_numpy(
            device,
            np.array(
                [int(instance.geometry_instance_id) for instance in instances], dtype=np.uint32
            ),
        ),
        spy.Tensor.from_numpy(device, np.array([0, 0], dtype=np.uint32)),
        spy.Tensor.from_numpy(device, np.array([[0.2, 0.3], [0.2, 0.3]], dtype=np.float32)),
    )
    cursor = result.cursor()
    cursor.load()
    actual = np.asarray([cursor[index].read()["material_color"] for index in range(2)])

    # Common contract: exact Pref + indexed UV1 + object position. Semantic color belongs only to MaterialX.
    weights = np.array([0.5, 0.2, 0.3])
    expected = np.array([weights @ pref, [0.11, 0.12, 0.13]])
    expected += [*(weights @ uv1), 0]
    expected += [0.2, 0.3, 0]
    if language == "materialx":
        expected += (weights @ colors)[:3]
    expected = np.clip(expected, 0, 1)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.nightly_if_not_vk
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_triangle_geometry_attribute_accessor_handles_empty_descriptors(
    device_type: spy.DeviceType,
    device: spy.Device,
    triangle_scene: _TriangleScene,
) -> None:
    probe = _create_attribute_accessor_probe(device, triangle_scene.scene)
    result = _read_attribute_accessor(
        device,
        probe,
        triangle_scene.geometry_instance_id,
        1,
        np.array([[0.2, 0.3]], dtype=np.float32),
    )
    assert result["success_mask"] == 0
    assert result["additional_texcoord_success_mask"] == 0
    _assert_missing_color(result)
    _assert_zero_discrete_outputs(result)
    _assert_zero_float_outputs(result)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_triangle_geometry_typed_additional_attributes(
    device_type: spy.DeviceType,
    device: spy.Device,
) -> None:
    scene = f2.Scene.create(device)
    material = scene.create_material(f2.StandardMaterial)
    geometry = scene.create_geometry(f2.StaticMeshGeometry)
    half_values = np.array(
        [[1.0, 2.0, 3.0], [5.0, 6.0, 7.0], [9.0, 10.0, 11.0]],
        dtype=np.float16,
    )
    float_values = np.array([[2.0, 3.0], [4.0, 5.0], [6.0, 7.0]], dtype=np.float32)
    int_values = np.array(
        [[-7, -8, -9], [-11, -12, -13], [-15, -16, -17]],
        dtype=np.int32,
    )
    bool_values = np.array(
        [[False, True, False], [True, False, True], [False, False, True]],
        dtype=np.bool_,
    )
    attributes = [
        f2.MeshVertexAttribute(f2.MeshAttributeSemantic.tex_coord, 1, half_values),
        f2.MeshVertexAttribute(f2.MeshAttributeSemantic.tex_coord, 2, float_values),
        f2.MeshVertexAttribute(
            f2.MeshAttributeSemantic.unknown,
            0,
            int_values,
            name="typed_int",
        ),
        f2.MeshVertexAttribute(
            f2.MeshAttributeSemantic.unknown,
            0,
            bool_values,
            name="typed_bool",
        ),
    ]
    _set_triangle_mesh(geometry, attributes)
    entity = scene.create_entity()
    geometry_instance = entity.create_component(f2.GeometryInstance)
    geometry_instance.geometry = geometry
    geometry_instance.materials = [material]
    scene.update()

    _, probe = _create_additional_attribute_probe(device, scene)
    for attribute_id, expected_type, component_count in (
        (1, 3, 3),
        (2, 4, 2),
        (16, 2, 3),
        (17, 1, 3),
    ):
        stored = _read_additional_attribute(
            device,
            probe,
            int(geometry_instance.geometry_instance_id),
            attribute_id,
            1,
        )
        assert stored["format"] & 0xFF == component_count
        assert stored["format"] >> 8 == expected_type

    stored_bool = _read_additional_attribute(
        device,
        probe,
        int(geometry_instance.geometry_instance_id),
        17,
        3,
    )
    assert stored_bool["data"][0] == 0x01000100
    assert stored_bool["data"][1] == 0x00000100

    accessor_probe = _create_attribute_accessor_probe(device, scene)
    barycentrics = np.array([[0.2, 0.3]], dtype=np.float32)
    half_result = _read_attribute_accessor(
        device,
        accessor_probe,
        int(geometry_instance.geometry_instance_id),
        1,
        barycentrics,
    )
    half_expected = 0.5 * half_values[0] + 0.2 * half_values[1] + 0.3 * half_values[2]
    assert half_result["success_mask"] == 0b111100
    assert half_result["int_vector_success_mask"] == 0
    assert half_result["bool_vector_success_mask"] == 0
    assert half_result["additional_texcoord_success_mask"] == 0b111
    _assert_zero_discrete_outputs(half_result)
    np.testing.assert_allclose(half_result["value_float"], half_expected[0], rtol=1e-3, atol=1e-3)
    for value_name, component_count in (("value2", 2), ("value3", 3), ("value4", 4)):
        converted = np.zeros(component_count, dtype=np.float32)
        converted[: min(component_count, 3)] = half_expected[: min(component_count, 3)]
        np.testing.assert_allclose(half_result[value_name], converted, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(
        half_result["additional_texcoord2"], half_expected[:2], rtol=1e-3, atol=1e-3
    )
    np.testing.assert_allclose(
        half_result["additional_texcoord3"], half_expected, rtol=1e-3, atol=1e-3
    )
    np.testing.assert_allclose(
        half_result["additional_texcoord4"],
        np.append(half_expected, 0.0),
        rtol=1e-3,
        atol=1e-3,
    )

    int_result = _read_attribute_accessor(
        device,
        accessor_probe,
        int(geometry_instance.geometry_instance_id),
        16,
        barycentrics,
    )
    assert int_result["success_mask"] == 0b000001
    assert int_result["int_vector_success_mask"] == 0b111
    assert int_result["bool_vector_success_mask"] == 0
    assert int_result["additional_texcoord_success_mask"] == 0
    assert int_result["value_int"] == int_values[0, 0]
    np.testing.assert_array_equal(int_result["value_int2"], int_values[0, :2])
    np.testing.assert_array_equal(int_result["value_int3"], int_values[0])
    np.testing.assert_array_equal(int_result["value_int4"], [*int_values[0], 0])
    assert int_result["value_bool"] is False
    _assert_zero_float_outputs(int_result)

    for selected_barycentrics, expected_vertex in (
        (np.array([[0.5, 0.0]], dtype=np.float32), 0),
        (np.array([[0.8, 0.1]], dtype=np.float32), 1),
    ):
        bool_result = _read_attribute_accessor(
            device,
            accessor_probe,
            int(geometry_instance.geometry_instance_id),
            17,
            selected_barycentrics,
        )
        assert bool_result["success_mask"] == 0b000010
        assert bool_result["int_vector_success_mask"] == 0
        assert bool_result["bool_vector_success_mask"] == 0b111
        assert bool_result["additional_texcoord_success_mask"] == 0
        assert bool_result["value_bool"] is bool(bool_values[expected_vertex, 0])
        np.testing.assert_array_equal(bool_result["value_bool2"], bool_values[expected_vertex, :2])
        np.testing.assert_array_equal(bool_result["value_bool3"], bool_values[expected_vertex])
        np.testing.assert_array_equal(
            bool_result["value_bool4"],
            [*bool_values[expected_vertex], False],
        )
        assert bool_result["value_int"] == 0
        _assert_zero_float_outputs(bool_result)

    updated_values = half_values.astype(np.float32) + 10.0
    _set_triangle_mesh(
        geometry,
        [
            f2.MeshVertexAttribute(f2.MeshAttributeSemantic.tex_coord, 1, updated_values),
            f2.MeshVertexAttribute(f2.MeshAttributeSemantic.tex_coord, 2, float_values),
            f2.MeshVertexAttribute(
                f2.MeshAttributeSemantic.unknown,
                0,
                int_values,
                name="typed_int",
            ),
            f2.MeshVertexAttribute(
                f2.MeshAttributeSemantic.unknown,
                0,
                bool_values,
                name="typed_bool",
            ),
        ],
    )
    scene.update()
    stored_updated = _read_additional_attribute(
        device,
        probe,
        int(geometry_instance.geometry_instance_id),
        1,
        1,
    )
    assert stored_updated["format"] >> 8 == 4
    updated_result = _read_attribute_accessor(
        device,
        accessor_probe,
        int(geometry_instance.geometry_instance_id),
        1,
        barycentrics,
    )
    updated_expected = 0.5 * updated_values[0] + 0.2 * updated_values[1] + 0.3 * updated_values[2]
    np.testing.assert_allclose(updated_result["value3"], updated_expected, rtol=1e-6, atol=1e-6)

    _set_triangle_mesh(geometry, [])
    scene.update()
    removed_result = _read_attribute_accessor(
        device,
        accessor_probe,
        int(geometry_instance.geometry_instance_id),
        1,
        barycentrics,
    )
    assert removed_result["success_mask"] == 0
    assert removed_result["additional_texcoord_success_mask"] == 0
    _assert_zero_discrete_outputs(removed_result)
    _assert_zero_float_outputs(removed_result)


@pytest.mark.nightly_if_not_vk
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize(
    "indexing_mode",
    [f2.CurveIndexingMode.successive, f2.CurveIndexingMode.list],
    ids=["successive", "list"],
)
def test_lss_geometry_typed_additional_attributes(
    device_type: spy.DeviceType,
    device: spy.Device,
    indexing_mode: f2.CurveIndexingMode,
) -> None:
    scene = f2.Scene.create(device)
    material = scene.create_material(f2.StandardMaterial)
    geometry = scene.create_geometry(f2.StaticCurveGeometry)

    positions = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    radii = np.full(4, 0.1, dtype=np.float32)
    float_values = np.array([[0.0, 2.0], [4.0, 6.0], [8.0, 10.0], [12.0, 14.0]], dtype=np.float16)
    color_values = np.array(
        [[0.0, 0.0, 0.0], [0.2, 0.4, 0.6], [0.4, 0.6, 0.8], [0.6, 0.8, 1.0]],
        dtype=np.float32,
    )
    int_values = np.array([10, 20, 30, 40], dtype=np.int32)
    bool_values = np.array([False, True, False, True], dtype=np.bool_)

    # List mode deliberately reorders both segments. Successive mode authors the equivalent remapped data directly.
    source_order = np.array([2, 0, 1, 3], dtype=np.uint32)
    expected_color_values = color_values[source_order]
    if indexing_mode == f2.CurveIndexingMode.list:
        indices = source_order
    else:
        positions = positions[source_order]
        radii = radii[source_order]
        float_values = float_values[source_order]
        color_values = color_values[source_order]
        int_values = int_values[source_order]
        bool_values = bool_values[source_order]
        indices = np.array([0, 2], dtype=np.uint32)

    geometry.set_curve_data(
        positions,
        radii,
        indices,
        indexing_mode,
        vertex_attributes=[
            f2.MeshVertexAttribute(f2.MeshAttributeSemantic.tex_coord, 1, float_values),
            f2.MeshVertexAttribute(f2.MeshAttributeSemantic.color, 0, color_values),
            f2.MeshVertexAttribute(
                f2.MeshAttributeSemantic.unknown, 0, int_values, name="curve_int"
            ),
            f2.MeshVertexAttribute(
                f2.MeshAttributeSemantic.unknown, 0, bool_values, name="curve_bool"
            ),
        ],
    )

    entity = scene.create_entity()
    transform = f2.Transform()
    transform.translation = spy.float3(3.0, -2.0, 5.0)
    entity.transform = transform
    geometry_instance = entity.create_component(f2.GeometryInstance)
    geometry_instance.geometry = geometry
    geometry_instance.materials = [material]
    scene.update()

    probe = _create_lss_attribute_accessor_probe(device, scene)
    results = _read_lss_attribute_accessors(
        device,
        probe,
        int(geometry_instance.geometry_instance_id),
        primitive_indices=np.array([0, 1, 0, 1, 0, 1, 0], dtype=np.uint32),
        attribute_ids=np.array([1, 1, 16, 16, 17, 17, 18], dtype=np.uint32),
        uvs=np.array(
            [
                [0.25, 0.5],
                [0.75, 0.5],
                [0.25, 0.5],
                [0.75, 0.5],
                [0.25, 0.5],
                [0.75, 0.5],
                [0.25, 0.5],
            ],
            dtype=np.float32,
        ),
    )

    for result, expected in zip(results[:2], ([6.0, 8.0], [10.0, 12.0]), strict=True):
        assert result["success_mask"] == 0b111100
        assert result["additional_texcoord_success_mask"] == 0b111
        np.testing.assert_allclose(result["value2"], expected, rtol=1e-3, atol=1e-3)

    expected_colors = (
        0.75 * expected_color_values[0] + 0.25 * expected_color_values[1],
        0.25 * expected_color_values[2] + 0.75 * expected_color_values[3],
    )
    for result, expected in zip(results[:2], expected_colors, strict=True):
        assert result["color_success"] is True
        np.testing.assert_allclose(result["color"], np.append(expected, 1.0))

    for result, expected in zip(results[2:4], (30, 40), strict=True):
        assert result["success_mask"] == 0b1
        assert result["int_vector_success_mask"] == 0b111
        assert result["value_int"] == expected

    for result, expected in zip(results[4:6], (False, True), strict=True):
        assert result["success_mask"] == 0b10
        assert result["bool_vector_success_mask"] == 0b111
        assert result["value_bool"] is expected

    missing = results[6]
    assert missing["success_mask"] == 0
    assert missing["int_vector_success_mask"] == 0
    assert missing["bool_vector_success_mask"] == 0
    assert missing["additional_texcoord_success_mask"] == 0
    _assert_zero_discrete_outputs(missing)
    _assert_zero_float_outputs(missing)

    for result in results:
        np.testing.assert_allclose(result["object_origin_ws"], [3.0, -2.0, 5.0])
        np.testing.assert_allclose(result["world_origin_os"], [-3.0, 2.0, -5.0])


@pytest.mark.nightly_if_not_vk
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_triangle_geometry_additional_attribute_storage_multiple_meshes(
    device_type: spy.DeviceType,
    device: spy.Device,
    additional_attribute_scene: _AdditionalAttributeScene,
) -> None:
    _, probe = _create_additional_attribute_probe(device, additional_attribute_scene.scene)
    cases = [
        (0, 1, (1 << 1) | (1 << 16)),
        (0, 16, (1 << 1) | (1 << 16)),
        (1, 3, (1 << 3) | (1 << 8) | (1 << 16)),
        (1, 8, (1 << 3) | (1 << 8) | (1 << 16)),
        (1, 16, (1 << 3) | (1 << 8) | (1 << 16)),
        (2, 0, (1 << 0) | (1 << 17)),
        (2, 17, (1 << 0) | (1 << 17)),
    ]
    stored_attributes = []
    for geometry_index, mask_bit, expected_mask in cases:
        values = additional_attribute_scene.attributes[geometry_index][mask_bit]
        stored = _read_additional_attribute(
            device,
            probe,
            additional_attribute_scene.geometry_instance_ids[geometry_index],
            mask_bit,
            values.size,
        )
        assert stored["mask"] == expected_mask
        assert stored["byte_offset"] % 16 == 0
        assert stored["format"] & 0xFF == values.shape[1]
        np.testing.assert_array_equal(
            stored["data"][: values.size], values.view(np.uint32).reshape(-1)
        )
        stored_attributes.append((int(stored["byte_offset"]), values.size))

    for offset, word_count in stored_attributes:
        for other_offset, other_word_count in stored_attributes:
            if offset != other_offset:
                assert (
                    offset + ((word_count * 4 + 15) // 16) * 16 <= other_offset
                    or other_offset + ((other_word_count * 4 + 15) // 16) * 16 <= offset
                )


@pytest.mark.nightly_if_not_vk
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_triangle_geometry_additional_attribute_storage_updates(
    device_type: spy.DeviceType,
    device: spy.Device,
    additional_attribute_scene: _AdditionalAttributeScene,
) -> None:
    updated_color = np.array(
        [[61.0, 62.0, 63.0], [65.0, 66.0, 67.0], [69.0, 70.0, 71.0]],
        dtype=np.float32,
    )
    added_uv6 = np.array([[51.0, 52.0], [53.0, 54.0], [55.0, 56.0]], dtype=np.float32)
    updated_shared = np.array([[101.0, 102.0], [103.0, 104.0], [105.0, 106.0]], dtype=np.float32)
    _set_triangle_mesh(
        additional_attribute_scene.geometries[1],
        [
            f2.MeshVertexAttribute(f2.MeshAttributeSemantic.color, 0, updated_color),
            f2.MeshVertexAttribute(f2.MeshAttributeSemantic.tex_coord, 6, added_uv6),
            f2.MeshVertexAttribute(
                f2.MeshAttributeSemantic.unknown,
                0,
                updated_shared,
                name=SHARED_CUSTOM_ATTRIBUTE_NAME,
            ),
        ],
    )
    additional_attribute_scene.scene.update()

    _, probe = _create_additional_attribute_probe(device, additional_attribute_scene.scene)
    for mask_bit, values in ((6, added_uv6), (8, updated_color), (16, updated_shared)):
        stored = _read_additional_attribute(
            device,
            probe,
            additional_attribute_scene.geometry_instance_ids[1],
            mask_bit,
            values.size,
        )
        assert stored["mask"] == (1 << 6) | (1 << 8) | (1 << 16)
        assert stored["format"] & 0xFF == values.shape[1]
        np.testing.assert_array_equal(
            stored["data"][: values.size], values.view(np.uint32).reshape(-1)
        )

    untouched = _read_additional_attribute(
        device,
        probe,
        additional_attribute_scene.geometry_instance_ids[0],
        1,
        additional_attribute_scene.attributes[0][1].size,
    )
    np.testing.assert_array_equal(
        untouched["data"][: additional_attribute_scene.attributes[0][1].size],
        additional_attribute_scene.attributes[0][1].view(np.uint32).reshape(-1),
    )


@pytest.mark.nightly_if_not_vk
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_triangle_geometry_attribute_accessor_interpolates_and_converts(
    device_type: spy.DeviceType,
    device: spy.Device,
    additional_attribute_scene: _AdditionalAttributeScene,
) -> None:
    probe = _create_attribute_accessor_probe(device, additional_attribute_scene.scene)

    barycentrics = np.array([[0.2, 0.3]], dtype=np.float32)
    results = {}
    cases = ((0, 1), (0, 16), (1, 3), (1, 8), (1, 16), (2, 0), (2, 17))
    for geometry_index, attribute_id in cases:
        geometry_instance_id = additional_attribute_scene.geometry_instance_ids[geometry_index]
        stored = _read_attribute_accessor(
            device,
            probe,
            geometry_instance_id,
            attribute_id,
            barycentrics,
        )
        results[(geometry_index, attribute_id)] = stored

        source = additional_attribute_scene.attributes[geometry_index][attribute_id]
        expected = 0.5 * source[0] + 0.2 * source[1] + 0.3 * source[2]
        assert stored["success_mask"] == 0b111100
        assert stored["int_vector_success_mask"] == 0
        assert stored["bool_vector_success_mask"] == 0
        np.testing.assert_allclose(stored["value_float"], expected[0], rtol=1e-6, atol=1e-6)
        for value_name, component_count in (("value2", 2), ("value3", 3), ("value4", 4)):
            converted = np.zeros(component_count, dtype=np.float32)
            copied_component_count = min(component_count, expected.size)
            converted[:copied_component_count] = expected[:copied_component_count]
            np.testing.assert_allclose(stored[value_name], converted, rtol=1e-6, atol=1e-6)

        expected_texcoord_success = 0b111 if 0 < attribute_id < 8 else 0
        if attribute_id == 0:
            expected_texcoord_success = 0b110
        assert stored["additional_texcoord_success_mask"] == expected_texcoord_success
        for bit, value_name, component_count in (
            (0, "additional_texcoord2", 2),
            (1, "additional_texcoord3", 3),
            (2, "additional_texcoord4", 4),
        ):
            converted = np.zeros(component_count, dtype=np.float32)
            if expected_texcoord_success & (1 << bit):
                copied_component_count = min(component_count, expected.size)
                converted[:copied_component_count] = expected[:copied_component_count]
            np.testing.assert_allclose(stored[value_name], converted, rtol=1e-6, atol=1e-6)

    first = results[(0, 1)]
    assert first["value_int"] == 0
    assert first["value_bool"] is False
    color_expected = 0.5 * additional_attribute_scene.attributes[1][8][0]
    color_expected += 0.2 * additional_attribute_scene.attributes[1][8][1]
    color_expected += 0.3 * additional_attribute_scene.attributes[1][8][2]
    assert results[(1, 8)]["color_success"] is True
    np.testing.assert_allclose(results[(1, 8)]["color"], color_expected, rtol=1e-6, atol=1e-6)
    _assert_missing_color(results[(0, 1)])

    geometry_instance_id = additional_attribute_scene.geometry_instance_ids[2]
    missing = _read_attribute_accessor(
        device,
        probe,
        geometry_instance_id,
        16,
        barycentrics,
    )
    assert missing["success_mask"] == 0
    assert missing["additional_texcoord_success_mask"] == 0
    _assert_missing_color(missing)
    _assert_zero_discrete_outputs(missing)
    _assert_zero_float_outputs(missing)


@pytest.mark.nightly_if_not_vk
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_imported_usd_additional_geometry_properties(
    device_type: spy.DeviceType,
    device: spy.Device,
) -> None:
    config = f2.SceneConfig(uv_origin=f2.UVOrigin.lower_left)
    scene = f2.Scene.load(device, USD_ADDITIONAL_GEOMETRY_PROPERTIES_PATH, config=config)
    scene.update()

    geometry_instance = scene.components.find(type=f2.GeometryInstance)
    assert geometry_instance is not None
    geometry_instance_id = int(geometry_instance.geometry_instance_id)
    primitive_indices = np.array([0, 1], dtype=np.uint32)
    barycentrics = np.array([[0.2, 0.3], [0.25, 0.5]], dtype=np.float32)

    helper = SceneShaderHelper(device)
    source_module = spy.Module.load_from_file(device, "render/test_geometry_properties.slang")
    module, requirements = helper.prepare(scene, source_module)
    texcoord_probe = (
        module.imported_texcoord_probe.as_func()
        .type_conformances(requirements.type_conformances)
        .write(helper.bind_scene)
        .return_type(spy.Tensor)
    )
    geometry_instance_ids = spy.Tensor.from_numpy(
        device,
        np.full(primitive_indices.shape, geometry_instance_id, dtype=np.uint32),
    )
    primitive_index_tensor = spy.Tensor.from_numpy(device, primitive_indices)
    barycentric_tensor = spy.Tensor.from_numpy(device, barycentrics)
    for texture_coord_index in range(8):
        result = texcoord_probe(
            geometry_instance_ids,
            primitive_index_tensor,
            barycentric_tensor,
            spy.Tensor.from_numpy(
                device,
                np.full(primitive_indices.shape, texture_coord_index, dtype=np.uint32),
            ),
        )
        cursor = result.cursor()
        cursor.load()
        actual = np.asarray([cursor[index].read() for index in range(2)])
        expected = np.array(
            [
                [10.0 * texture_coord_index + 0.2, 0.3],
                [
                    10.5 if texture_coord_index == 1 else 10.0 * texture_coord_index + 2.25,
                    0.75 if texture_coord_index == 1 else 0.5,
                ],
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)

    accessor_probe = (
        module.imported_attribute_accessor_probe.as_func()
        .type_conformances(requirements.type_conformances)
        .write(helper.bind_scene)
        .return_type(spy.Tensor)
    )
    expected_pref = (
        np.array([0.2, 0.3, 10.0], dtype=np.float32),
        np.array([2.25, 0.5, 10.0], dtype=np.float32),
    )
    for primitive_index, expected in enumerate(expected_pref):
        result = accessor_probe(
            spy.Tensor.from_numpy(device, np.array([geometry_instance_id], dtype=np.uint32)),
            spy.Tensor.from_numpy(device, np.array([primitive_index], dtype=np.uint32)),
            spy.Tensor.from_numpy(device, np.array([16], dtype=np.uint32)),
            spy.Tensor.from_numpy(device, barycentrics[primitive_index : primitive_index + 1]),
        )
        cursor = result.cursor()
        cursor.load()
        value = cursor[0].read()
        assert value["success_mask"] & (1 << 4)
        np.testing.assert_allclose(value["value3"], expected, rtol=1e-6, atol=1e-6)


@pytest.mark.nightly_if_not_vk
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_triangle_geometry_color_accessor_widens_rgb(
    device_type: spy.DeviceType,
    device: spy.Device,
) -> None:
    scene = f2.Scene.create(device)
    material = scene.create_material(f2.StandardMaterial)
    geometry = scene.create_geometry(f2.StaticMeshGeometry)
    colors = np.array(
        [[0.2, 0.4, 0.6], [0.4, 0.6, 0.8], [0.6, 0.8, 1.0]],
        dtype=np.float32,
    )
    _set_triangle_mesh(
        geometry,
        [f2.MeshVertexAttribute(f2.MeshAttributeSemantic.color, 0, colors)],
    )
    entity = scene.create_entity()
    geometry_instance = entity.create_component(f2.GeometryInstance)
    geometry_instance.geometry = geometry
    geometry_instance.materials = [material]
    scene.update()

    result = _read_attribute_accessor(
        device,
        _create_attribute_accessor_probe(device, scene),
        int(geometry_instance.geometry_instance_id),
        8,
        np.array([[0.2, 0.3]], dtype=np.float32),
    )
    expected_rgb = 0.5 * colors[0] + 0.2 * colors[1] + 0.3 * colors[2]
    assert result["color_success"] is True
    np.testing.assert_allclose(
        result["color"],
        np.append(expected_rgb, 1.0),
        rtol=1e-6,
        atol=1e-6,
    )


@pytest.mark.nightly_if_not_vk
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_gltf_standard_material_samples_selected_uv_channel(
    device_type: spy.DeviceType,
    device: spy.Device,
) -> None:
    scene = f2.Scene.load(device, MULTI_UV_SCENE_PATH)
    scene.update()
    material = scene.materials[0]
    geometry_instance = scene.components.find(type=f2.GeometryInstance)
    assert isinstance(material, f2.GLTFMaterial)
    assert geometry_instance is not None
    assert material.base_color_texture_texcoord == 0
    assert material.emissive_texture_texcoord == 1

    helper = SceneShaderHelper(device)
    source_module = spy.Module.load_from_file(device, "render/test_geometry_properties.slang")
    module, requirements = helper.prepare(scene, source_module)
    probe = (
        module.multi_uv_material_probe.as_func()
        .type_conformances(requirements.type_conformances)
        .write(helper.bind_scene)
        .return_type(spy.Tensor)
    )

    primitive_indices = np.arange(12, dtype=np.uint32)
    geometry_instance_ids = np.full(
        primitive_indices.shape,
        int(geometry_instance.geometry_instance_id),
        dtype=np.uint32,
    )
    barycentrics = np.tile(np.array([[0.2, 0.3]], dtype=np.float32), (primitive_indices.size, 1))

    def sample_emission() -> np.ndarray:
        result = probe(
            spy.Tensor.from_numpy(device, geometry_instance_ids),
            spy.Tensor.from_numpy(device, primitive_indices),
            spy.Tensor.from_numpy(device, barycentrics),
        )
        cursor = result.cursor()
        cursor.load()
        values = [cursor[index].read() for index in range(primitive_indices.size)]
        return np.stack([value["emissive_radiance"] for value in values])

    selected_emission = sample_emission()

    material.emissive_texture_texcoord = 0
    scene.update()
    uv0_emission = sample_emission()

    assert np.max(np.abs(selected_emission - uv0_emission)) > 0.25


@pytest.mark.nightly_if_not_vk
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_gltf_standard_material_enables_available_vertex_color(
    device_type: spy.DeviceType,
    device: spy.Device,
) -> None:
    scene = f2.Scene.load(device, VERTEX_COLOR_SCENE_PATH)
    scene.update()
    standard_materials = [
        material for material in scene.materials if isinstance(material, f2.GLTFMaterial)
    ]
    assert standard_materials
    assert any(material.use_vertex_color for material in standard_materials)


@pytest.mark.nightly
@pytest.mark.parametrize("device_type", [spy.DeviceType.d3d12])
def test_gltf_standard_material_multi_uv_image(
    device_type: spy.DeviceType,
    image_test: ImageTest,
) -> None:
    if device_type not in helpers.DEFAULT_DEVICE_TYPES:
        pytest.skip(f"{device_type} is not available in this test environment")

    device = helpers.get_device(device_type)
    # This reference covers the StandardMaterial conversion of the glTF UV sets.
    import_options = f2.ImportOptions()
    import_options.gltf_material_mode = f2.GLTFMaterialMode.standard_material
    scene = f2.Scene.load(device, MULTI_UV_SCENE_PATH, import_options=import_options)
    cameras = scene.components.find_all(type=f2.Camera)
    assert len(cameras) == 1
    camera = cameras[0]
    camera.width = 512
    camera.height = 288

    light_entity = scene.create_entity()
    environment_light = light_entity.create_component(f2.ConstantLight)
    environment_light.color = spy.float3(0.5)
    scene.update()

    renderer = _create_primary_emission_renderer(device)
    renderer.max_depth = 2
    image_data = np.zeros((camera.height, camera.width, 4), dtype=np.float32)
    sample_count = 32
    renderer.pixel_filter = f2.BoxReconstructionFilter(width=0)
    for iteration in range(sample_count):
        image, _ = renderer(
            scene,
            camera,
            iteration=iteration,
            pixel_offset=spy.float2(0.0),
        )
        image_data += image.to_numpy()
    image_data /= sample_count
    assert np.isfinite(image_data).all()
    assert image_data[..., :3].max() > 0.1
    image_test(image_data, tolerance=1e-5)


@pytest.mark.nightly_if_not_vk
@pytest.mark.nightly_if(material_type=(f2.StandardSpecGlossMaterial,))
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize(
    "material_type", [f2.StandardMaterial, f2.StandardSpecGlossMaterial, f2.GLTFMaterial]
)
def test_standard_family_uv1_alpha_mask_opacity(
    device_type: spy.DeviceType,
    device: spy.Device,
    material_type: type[f2.Material],
) -> None:
    alpha_texture = device.create_texture(
        type=spy.TextureType.texture_2d,
        format=spy.Format.rgba32_float,
        width=2,
        height=1,
        usage=spy.TextureUsage.shader_resource,
        data=np.array(
            [[[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 1.0]]],
            dtype=np.float32,
        ),
    )

    scene = f2.Scene.create(device)
    front = scene.create_material(material_type)
    if isinstance(front, (f2.StandardMaterial, f2.GLTFMaterial)):
        front.base_color_factor = spy.float3(0.0)
        front.base_color_texture = alpha_texture
        front.base_color_texture_texcoord = 1
    else:
        assert isinstance(front, f2.StandardSpecGlossMaterial)
        front.diffuse_factor = spy.float3(0.0)
        front.diffuse_texture = alpha_texture
        front.diffuse_texture_texcoord = 1
    front.emissive_factor = spy.float3(1.0, 0.0, 0.0)
    front.alpha_mode = f2.AlphaMode.mask
    front.alpha_cutoff = 0.5
    front.double_sided = True

    rear = scene.create_material(material_type)
    if isinstance(rear, (f2.StandardMaterial, f2.GLTFMaterial)):
        rear.base_color_factor = spy.float3(0.0)
    else:
        assert isinstance(rear, f2.StandardSpecGlossMaterial)
        rear.diffuse_factor = spy.float3(0.0)
    rear.emissive_factor = spy.float3(0.0, 1.0, 0.0)
    rear.double_sided = True

    front_instance = _add_uv_opacity_quad(
        scene,
        front,
        0.0,
        (0.25, 0.5),
        (0.75, 0.5),
        (1.0, 1.0, 1.0, 0.25),
    )
    _add_uv_opacity_quad(scene, rear, 0.2, (0.0, 0.0))
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
    geometry_instance_ids = np.array(
        [int(front_instance.geometry_instance_id)],
        dtype=np.uint32,
    )
    primitive_indices = np.array([0], dtype=np.uint32)
    barycentrics = np.array([[0.2, 0.3]], dtype=np.float32)

    def evaluate_opacity() -> float:
        result = opacity_probe(
            spy.Tensor.from_numpy(device, geometry_instance_ids),
            spy.Tensor.from_numpy(device, primitive_indices),
            spy.Tensor.from_numpy(device, barycentrics),
        )
        return float(result.to_numpy()[0])

    renderer = _create_primary_emission_renderer(device)

    def render_center(iteration: int) -> np.ndarray:
        scene.update()
        renderer.pixel_filter = f2.BoxReconstructionFilter(width=0)
        image, _ = renderer(
            scene,
            camera,
            iteration=iteration,
            pixel_offset=spy.float2(0.0),
        )
        return image.to_numpy()[4, 4, :3]

    selected_opacity = evaluate_opacity()
    selected_uv_image = render_center(0)

    if isinstance(front, (f2.StandardMaterial, f2.GLTFMaterial)):
        front.base_color_texture_texcoord = 0
    else:
        front.diffuse_texture_texcoord = 0
    scene.update()
    uv0_opacity = evaluate_opacity()
    uv0_image = render_center(1)

    assert selected_opacity > 0.99
    assert uv0_opacity < 0.01
    assert selected_uv_image[0] > 0.1
    assert selected_uv_image[0] > selected_uv_image[1] * 10.0
    assert uv0_image[1] > 0.1
    assert uv0_image[1] > uv0_image[0] * 10.0

    if isinstance(front, (f2.StandardMaterial, f2.GLTFMaterial)):
        front.base_color_texture_texcoord = 1
    else:
        front.diffuse_texture_texcoord = 1
    front.use_vertex_color = True
    scene.update()
    vertex_color_opacity = evaluate_opacity()
    vertex_color_image = render_center(2)
    assert vertex_color_opacity == pytest.approx(0.25)
    assert vertex_color_image[1] > 0.1
    assert vertex_color_image[1] > vertex_color_image[0] * 10.0


@pytest.mark.nightly_if_not_vk
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_standard_specgloss_uses_selected_uv_and_vertex_color(
    device_type: spy.DeviceType,
    device: spy.Device,
) -> None:
    texture = device.create_texture(
        type=spy.TextureType.texture_2d,
        format=spy.Format.rgba32_float,
        width=2,
        height=1,
        usage=spy.TextureUsage.shader_resource,
        data=np.array(
            [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 1.0]]],
            dtype=np.float32,
        ),
    )
    scene = f2.Scene.create(device)
    material = scene.create_material(f2.StandardSpecGlossMaterial)
    material.diffuse_factor = spy.float3(1.0)
    material.specular_factor = spy.float3(0.0)
    material.diffuse_texture = texture
    material.diffuse_texture_texcoord = 1
    material.use_vertex_color = True
    material.alpha_mode = f2.AlphaMode.mask
    geometry_instance = _add_uv_opacity_quad(
        scene,
        material,
        0.0,
        (0.25, 0.5),
        (0.75, 0.5),
        (0.25, 0.5, 0.75, 0.5),
    )
    scene.update()

    selected = _read_material_probe(device, scene, geometry_instance)
    np.testing.assert_allclose(selected["diffuse_reflection_albedo"], [0.0, 0.5, 0.0], atol=1e-6)
    assert selected["opacity"] == pytest.approx(0.5)

    material.diffuse_texture_texcoord = 0
    scene.update()
    uv0 = _read_material_probe(device, scene, geometry_instance)
    np.testing.assert_allclose(uv0["diffuse_reflection_albedo"], [0.25, 0.0, 0.0], atol=1e-6)
    assert uv0["opacity"] == pytest.approx(0.0)


@pytest.mark.nightly_if_not_vk
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_openpbr_uses_independent_uv_selectors_and_vertex_color(
    device_type: spy.DeviceType,
    device: spy.Device,
) -> None:
    texture = device.create_texture(
        type=spy.TextureType.texture_2d,
        format=spy.Format.rgba32_float,
        width=2,
        height=1,
        usage=spy.TextureUsage.shader_resource,
        data=np.array(
            [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 1.0]]],
            dtype=np.float32,
        ),
    )
    scene = f2.Scene.create(device)
    material = scene.create_material(
        "OpenPBRMaterial",
        f2.Properties(
            {
                "base_color": spy.float3(1.0),
                "base_color_texture": texture,
                "base_color_texture_texcoord": 1,
                "emission_luminance": 1.0,
                "emission_color": spy.float3(1.0),
                "emission_color_texture": texture,
                "emission_color_texture_texcoord": 1,
                "normal_texture": texture,
                "normal_texture_texcoord": 0,
                "opacity_texture": texture,
                "opacity_texture_texcoord": 0,
                "opacity_texture_channel": 3,
                "use_vertex_color": True,
            }
        ),
    )
    color_only_material = scene.create_material(
        "OpenPBRMaterial",
        f2.Properties({"use_vertex_color": True}),
    )
    geometry_instance = _add_uv_opacity_quad(
        scene,
        material,
        0.0,
        (0.25, 0.5),
        (0.75, 0.5),
        (0.25, 0.5, 0.75, 0.5),
    )
    uncolored_instance = _add_uv_opacity_quad(
        scene,
        material,
        1.0,
        (0.25, 0.5),
        (0.75, 0.5),
    )
    color_only_instance = _add_uv_opacity_quad(
        scene,
        color_only_material,
        2.0,
        (0.25, 0.5),
        color=(1.0, 1.0, 1.0, 0.5),
    )
    scene.update()

    selected = _read_material_probe(device, scene, geometry_instance)
    uncolored = _read_material_probe(device, scene, uncolored_instance)
    color_only = _read_material_probe(device, scene, color_only_instance)
    np.testing.assert_allclose(selected["emissive_radiance"], [0.0, 1.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(selected["material_color"], [0.0, 0.5, 0.0], atol=1e-6)
    np.testing.assert_allclose(uncolored["material_color"], [0.0, 1.0, 0.0], atol=1e-6)
    assert selected["opacity"] == pytest.approx(0.0)
    assert color_only["opacity"] == pytest.approx(0.5)

    material["normal_texture_texcoord"] = 1
    scene.update()
    normal_uv1 = _read_material_probe(device, scene, geometry_instance)
    assert np.max(np.abs(normal_uv1["guide_normal"] - selected["guide_normal"])) > 0.5

    material["opacity_texture_texcoord"] = 1
    scene.update()
    opacity_uv1 = _read_material_probe(device, scene, geometry_instance)
    assert opacity_uv1["opacity"] == pytest.approx(0.5)
