# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import re
from typing import Any, Optional, cast

import numpy as np
import pytest
import slangpy as spy

import falcor2 as f2
import falcor2.testing.helpers as helpers
from falcor2.editor import SceneShaderHelper

from .geometry_property_helpers import TRIANGLE_POSITIONS, TRIANGLE_INDICES, read_material_probe


MDL_LIBRARY = Path(__file__).parent / "data/mdl_generated_material_pipeline"


def _named_attribute(name: str, values: np.ndarray) -> f2.MeshVertexAttribute:
    return f2.MeshVertexAttribute(f2.MeshAttributeSemantic.unknown, 0, values, name=name)


def _mdl_properties(name: str, shader_path: Path) -> f2.Properties:
    return f2.Properties(
        {
            "mdl_library_path": str(MDL_LIBRARY),
            "mdl_material_name": name,
            "debug_write_shader_path": str(shader_path),
        }
    )


@pytest.mark.nightly_if_not_vk
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_mdl_scene_data_callbacks(
    device_type: spy.DeviceType,
    tmp_path: Path,
) -> None:
    device = helpers.get_device(device_type)
    helper = SceneShaderHelper(device)
    scene = f2.Scene.create(device)
    shader_path = tmp_path / "scene_data.slang"
    material = scene.create_material(
        f2.MDLMaterial, _mdl_properties("scene_data::Default", shader_path)
    )
    attributes = [
        _named_attribute(
            "Pref", np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]], dtype=np.float32)
        ),
        _named_attribute("other", np.full((3, 3), 0.6, dtype=np.float32)),
        _named_attribute(
            "float_value", np.tile(np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32), (3, 1))
        ),
        _named_attribute(
            "int_value",
            np.array([[2, 3, 5, 7], [11, 13, 17, 19], [23, 29, 31, 37]], dtype=np.int32),
        ),
        _named_attribute("flag", np.array([True, False, False], dtype=np.bool_)),
        _named_attribute("pref", np.full((3, 3), 0.7, dtype=np.float32)),
        _named_attribute("texcoord_1", np.full((3, 3), 0.8, dtype=np.float32)),
        f2.MeshVertexAttribute(
            f2.MeshAttributeSemantic.tex_coord, 1, np.full((3, 2), 0.1, dtype=np.float32)
        ),
    ]
    instances = []
    for present in (True, False):
        geometry = scene.create_geometry(f2.StaticMeshGeometry)
        geometry.set_mesh_data(
            positions=TRIANGLE_POSITIONS,
            sub_mesh_indices=[TRIANGLE_INDICES],
            vertex_attributes=attributes if present else [],
        )
        instance = scene.create_entity().create_component(f2.GeometryInstance)
        instance.geometry = geometry
        instance.materials = [material]
        instances.append(instance)
    scene.update()
    assert scene.required_capabilities.has(f2.SceneCapabilityFlags.geometry_properties)
    source = shader_path.read_text(encoding="utf-8")
    assert "#define MDL_HAS_SCENE_DATA 1" in source
    assert "GeomPropProvider" not in source
    synthetic = read_material_probe(
        device, scene, instances[0], "synthetic_material_probe", helper=helper
    )
    assert synthetic["interior_ior"] == pytest.approx(1.12)
    assert synthetic["opacity"] == pytest.approx(0.25)

    # Interleave present/missing/present hits: callback context must belong to this material instance's hit.
    results = read_material_probe(
        device, scene, [instances[index] for index in (0, 1, 0)], helper=helper
    )
    for index, result in zip((0, 1, 0), results):
        available = index == 0
        fv = [0.1, 0.2, 0.3, 0.4] if available else [0.9, 0.8, 0.7, 0.6]
        iv = [2, 3, 5, 7] if available else [11, 13, 17, 19]
        floats = sum((1 << n) * sum(fv[: n + 1]) for n in range(4)) + 16 * sum(fv[:3])
        ints = sum((1 << n) * sum(iv[: n + 1]) for n in range(4))
        checks = int(available) + 0.17 + 0.23 + 3 + 0.31 + 0.37 + 5 + 7 + 0.41 + 0.43
        np.testing.assert_allclose(
            result["diffuse_reflection_albedo"], [floats / 128, ints / 1024, checks / 64], atol=1e-6
        )
        assert result["interior_ior"] == pytest.approx(1.44 if available else 1.12)
        assert result["opacity"] == pytest.approx(0.75 if available else 0.25)

    # One real curve hit covers the other production accessor and presence dispatch, without a new renderer.
    curve = scene.create_geometry(f2.StaticCurveGeometry)
    curve.set_curve_data(
        np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32),
        np.full(2, 0.1, dtype=np.float32),
        np.array([0], dtype=np.uint32),
        f2.CurveIndexingMode.successive,
        vertex_attributes=[
            _named_attribute("Pref", np.array([[0.1, 0.2, 0.3], [0.6, 0.7, 0.8]], dtype=np.float32))
        ],
    )
    curve_instance = scene.create_entity().create_component(f2.GeometryInstance)
    curve_instance.geometry = curve
    curve_instance.materials = [material]
    scene.update()
    curve_result = read_material_probe(
        device, scene, curve_instance, "lss_material_probe", helper=helper
    )
    assert curve_result["interior_ior"] == pytest.approx(1.3)
    assert curve_result["opacity"] == pytest.approx(0.75)

    # Keep these scenarios in regular Vulkan runs and on the original backend at night.
    if device_type not in (helpers.DEFAULT_DEVICE_TYPES[0], spy.DeviceType.vulkan):
        return

    unchanged_source = shader_path.read_text(encoding="utf-8")
    material["choose"] = False
    scene.update()
    assert shader_path.read_text(encoding="utf-8") == unchanged_source
    assert read_material_probe(device, scene, instances[0], helper=helper)[
        "interior_ior"
    ] == pytest.approx(1.6)
    assert material["choose"] is False
    # Missing defaults are editable numeric inputs; no string-editing interface is introduced.
    material["fallback"] = spy.float3(0.41, 0.12, 0.13)
    scene.update()
    assert read_material_probe(device, scene, instances[1], helper=helper)[
        "interior_ior"
    ] == pytest.approx(1.22)
    missing_geometry = instances[1].geometry
    missing_geometry.set_mesh_data(
        positions=TRIANGLE_POSITIONS,
        sub_mesh_indices=[TRIANGLE_INDICES],
        vertex_attributes=attributes,
    )
    scene.update()
    assert read_material_probe(device, scene, instances[1], helper=helper)[
        "interior_ior"
    ] == pytest.approx(1.6)

    # Captured default strings are absent from the target constant table but must have SDK runtime IDs.
    material.set_properties(
        f2.Properties({"mdl_material_name": "scene_data_metadata::DynamicName"})
    )
    scene.update()
    first_binding = re.findall(r"case \d+u: return \d+u;", shader_path.read_text(encoding="utf-8"))
    assert read_material_probe(device, scene, instances[0], helper=helper)[
        "interior_ior"
    ] == pytest.approx(0.44)
    # Changing source/name rebuilds this material's mapping; a differently cased name is not an alias.
    material.set_properties(f2.Properties({"mdl_material_name": "scene_data_metadata::ExactNames"}))
    scene.update()
    assert read_material_probe(device, scene, instances[0], helper=helper)[
        "interior_ior"
    ] == pytest.approx(1.5)

    # A second Scene offers geometry first and gives Pref a different ID. The same SDK source must rebind locally.
    second = f2.Scene.create(device)
    geometry = second.create_geometry(f2.StaticMeshGeometry)
    geometry.set_mesh_data(
        positions=TRIANGLE_POSITIONS,
        sub_mesh_indices=[TRIANGLE_INDICES],
        vertex_attributes=[
            _named_attribute("prior_reservation", np.ones(3, dtype=np.float32)),
            *attributes,
        ],
    )
    second.update()
    second_path = tmp_path / "second.slang"
    second_material = second.create_material(
        f2.MDLMaterial, _mdl_properties("scene_data_metadata::DynamicName", second_path)
    )
    instance = second.create_entity().create_component(f2.GeometryInstance)
    instance.geometry = geometry
    instance.materials = [second_material]
    second.update()
    assert read_material_probe(device, second, instance)["interior_ior"] == pytest.approx(0.44)
    assert first_binding != re.findall(
        r"case \d+u: return \d+u;", second_path.read_text(encoding="utf-8")
    )


def _compile_material(
    device_type: spy.DeviceType,
    module_name: str,
    *,
    class_compilation: bool,
    learnable: bool = False,
    debug_write_shader_path: Optional[Path] = None,
) -> tuple[f2.Scene, f2.Material]:
    scene = f2.Scene.create(helpers.get_device(device_type))
    properties = f2.Properties(
        {
            "mdl_library_path": str(MDL_LIBRARY),
            "mdl_material_name": f"{module_name}::Default",
            "mdl_class_compilation": class_compilation,
            "learnable": learnable,
        }
    )
    if debug_write_shader_path is not None:
        properties["debug_write_shader_path"] = str(debug_write_shader_path)

    material = scene.create_material(
        "MDLMaterial",
        properties,
    )
    scene.update()

    assert material.required_module() is not None
    return scene, material


def _read_extra_bsdf_properties(
    device_type: spy.DeviceType,
    scene: f2.Scene,
    material: f2.Material,
) -> dict[str, Any]:
    device = helpers.get_device(device_type)
    helper = SceneShaderHelper(device)
    module, _ = helper.prepare(scene, "falcor2.tools.materials.scene_sampling_tools")
    extra_func = (
        module["material_collect_extra_bsdf_properties_simple"].as_func().write(helper.bind_scene)
    )
    uvs = np.array([(0.5, 0.5)], dtype=np.float32)
    wi = np.array([(0.0, 1.0, 0.0)], dtype=np.float32)
    wo = np.array([(0.0, 1.0, 0.0)], dtype=np.float32)
    extra = cast(spy.Tensor, extra_func(material, uvs, wi, wo))
    extra_cursor = extra.cursor()
    extra_cursor.load()
    return cast(dict[str, Any], extra_cursor[0].read())


@pytest.mark.nightly
@pytest.mark.slow
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES[:1])
def test_modified_material_does_not_reanalyze_source_module(
    device_type: spy.DeviceType,
) -> None:
    _compile_material(
        device_type,
        "wildcard_import_leaf_diagnostics",
        class_compilation=True,
        learnable=True,
    )


@pytest.mark.slow
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES[:1])
def test_cached_modified_material_preserves_tagged_bsdf_metadata(
    device_type: spy.DeviceType,
) -> None:
    _compile_material(
        device_type,
        "wildcard_import_leaf_diagnostics",
        class_compilation=True,
        learnable=True,
    )
    cached_scene, cached_material = _compile_material(
        device_type,
        "wildcard_import_leaf_diagnostics",
        class_compilation=True,
        learnable=True,
    )

    result = _read_extra_bsdf_properties(device_type, cached_scene, cached_material)
    assert result["bsdf_count"] == 1
    assert result["bsdf_weight"][0] == pytest.approx(spy.float3(0.8, 0.7, 0.6))


@pytest.mark.slow
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES[:1])
def test_modified_material_parameter_names_are_collision_safe(
    device_type: spy.DeviceType,
) -> None:
    # The compiled tint components require generated aliases, while the explicit compiled_parameter_0 and mollify
    # parameters collide with the old alias scheme and Falcor's injected mollification parameter, respectively.
    _, material = _compile_material(
        device_type,
        "compiled_parameter_name_collisions",
        class_compilation=True,
        learnable=True,
    )

    assert material["tint.r"] == pytest.approx(0.2)
    assert material["tint.g"] == pytest.approx(0.4)
    assert material["tint.b"] == pytest.approx(0.8)
    assert material["compiled_parameter_0"] == pytest.approx(0.6)
    assert material["mollify"] == pytest.approx(0.37)
    assert material["mollify_1"] == pytest.approx(0.01)


@pytest.mark.nightly_if(uv_origin=(f2.UVOrigin.lower_left,))
@pytest.mark.slow
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES[:1])
@pytest.mark.parametrize("uv_origin", [f2.UVOrigin.lower_left, f2.UVOrigin.upper_left])
def test_state_dependent_ior_and_secondary_texture_spaces(
    device_type: spy.DeviceType,
    tmp_path: Path,
    uv_origin: f2.UVOrigin,
) -> None:
    device = helpers.get_device(device_type)
    scene = f2.Scene.create(
        device,
        f2.SceneConfig(uv_origin=uv_origin),
    )
    shader_path = tmp_path / "uv.slang"
    material = scene.create_material(
        f2.MDLMaterial,
        f2.Properties(
            {
                "mdl_library_path": str(MDL_LIBRARY),
                "mdl_material_name": "state_dependent_ior::Default",
                "mdl_class_compilation": False,
                "debug_write_shader_path": str(shader_path),
            }
        ),
    )
    instances = []
    for present in (True, False):
        geometry = scene.create_geometry(f2.StaticMeshGeometry)
        attributes = [
            f2.MeshVertexAttribute(
                f2.MeshAttributeSemantic.tex_coord,
                index,
                np.tile(np.array(uv, dtype=np.float32), (3, 1)),
            )
            for index, uv in [(0, (0.2, 0.3))]
            + ([(1, (0.6, 0.7)), (3, (0.8, 0.9))] if present else [])
        ]
        geometry.set_mesh_data(
            positions=TRIANGLE_POSITIONS,
            sub_mesh_indices=[TRIANGLE_INDICES],
            vertex_attributes=attributes,
        )
        instance = scene.create_entity().create_component(f2.GeometryInstance)
        instance.geometry = geometry
        instance.materials = [material]
        instances.append(instance)
    scene.update()
    assert scene.required_capabilities.has(f2.SceneCapabilityFlags.geometry_properties)
    source = shader_path.read_text(encoding="utf-8")
    assert "GeomPropProvider" not in source
    assert "#define MDL_HAS_SCENE_DATA 0" in source
    for present, instance in zip((True, False), instances):
        result = read_material_probe(device, scene, instance)
        available = present
        assert result["interior_ior"] == pytest.approx(1.2 + (0.15 if available else 0))
        v1, u3, v3 = (0.7, 0.8, 0.9) if available else (0, 0, 0)
        if uv_origin == f2.UVOrigin.upper_left:
            v1, v3 = 1 - v1, 1 - v3
        np.testing.assert_allclose(result["diffuse_reflection_albedo"], (v1, u3, v3), atol=1e-6)
