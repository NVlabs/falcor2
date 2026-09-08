# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Any, Optional, cast

import numpy as np
import pytest
import slangpy as spy

import falcor2 as f2
import falcor2.testing.helpers as helpers
from falcor2.editor import SceneShaderHelper


MDL_LIBRARY = Path(__file__).parent / "data/mdl_generated_material_pipeline"

GEOMPROP_SAMPLING_MODULE = """
import slangpy;
module mdl_geomprop_sampling_test;

__exported import falcor2.render;
__exported import falcor2.utils;
import falcor2.render.materials.mtlx.test_geomprop_provider;

public float material_interior_ior_with_test_geomprops(IMaterial material, float2 uv)
{
    SurfaceInteraction si;
    si.position_ws = float3(uv, 0);
    si.wi_ws = float3(0, 1, 0);
    si.uv = uv;
    si.shading_frame_ws = Frame(float3(0, 1, 0));
    si.normal_ws = si.shading_frame_ws.normal;
    si.front_facing = dot(si.wi_ws, si.normal_ws) >= 0;
    si.exterior_ior = 1.0;

    let material_instance = material.setup_material_instance(
        si, ExplicitLodSampler(0), MaterialInstanceHints::none
    );
    return material_instance.get_interior_ior();
}
"""


def _compile_material(
    device_type: spy.DeviceType,
    module_name: str,
    *,
    class_compilation: bool,
    learnable: bool = False,
    debug_write_shader_path: Optional[Path] = None,
    geomprop_streams: Optional[dict[str, int]] = None,
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
    if geomprop_streams is not None:
        properties["mdl_geomprop_names"] = list(geomprop_streams)
        properties["mdl_geomprop_ids"] = list(geomprop_streams.values())

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
    module = helper.get_module(scene, "falcor2.tools.materials.scene_sampling_tools")
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


@pytest.mark.slow
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES[:1])
def test_state_dependent_ior_and_secondary_texture_space_compile(
    device_type: spy.DeviceType,
    tmp_path: Path,
) -> None:
    scene, material = _compile_material(
        device_type,
        "state_dependent_ior",
        class_compilation=False,
        geomprop_streams={"texcoord_1": 11},
    )
    fallback_shader_path = tmp_path / "state_dependent_ior_fallback.slang"
    fallback_scene, fallback_material = _compile_material(
        device_type,
        "state_dependent_ior",
        class_compilation=False,
        debug_write_shader_path=fallback_shader_path,
    )

    assert "GeomPropProvider::get_float2" not in fallback_shader_path.read_text(encoding="utf-8")

    device = helpers.get_device(device_type)
    source_module = spy.Module(
        device.load_module_from_source("mdl_geomprop_sampling_test", GEOMPROP_SAMPLING_MODULE)
    )
    uvs = np.array([(0.0, 0.25), (0.5, 0.75), (1.0, 0.5)], dtype=np.float32)

    def evaluate_iors(evaluation_scene: f2.Scene, evaluation_material: f2.Material) -> list[float]:
        helper = SceneShaderHelper(device)
        module = helper.get_module(evaluation_scene, source_module)
        ior_func = (
            module["material_interior_ior_with_test_geomprops"].as_func().write(helper.bind_scene)
        )
        iors = cast(spy.Tensor, ior_func(evaluation_material, uvs))
        ior_cursor = iors.cursor()
        ior_cursor.load()
        return [cast(float, ior_cursor[index].read()) for index in range(len(uvs))]

    mapped_iors = evaluate_iors(scene, material)
    fallback_iors = evaluate_iors(fallback_scene, fallback_material)
    for index, uv in enumerate(uvs):
        assert mapped_iors[index] == pytest.approx(1.0 + uv[0] + 0.25 * uv[1])
        assert fallback_iors[index] == pytest.approx(1.0 + 1.25 * uv[0])
