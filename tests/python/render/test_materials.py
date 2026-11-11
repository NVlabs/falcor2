# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import slangpy as spy
import falcor2 as f2
import falcor2.testing.helpers as helpers
import numpy as np
import pytest

# Allow this test to run standalone.
if __name__ == "__main__":
    from python_material_sample import PythonMaterialSample
    from scattering_tests import BSDFFlags
else:
    from .python_material_sample import PythonMaterialSample
    from .scattering_tests import BSDFFlags


DATA_SCENES_DIR = Path(__file__).resolve().parents[3] / "data" / "scenes"
if str(DATA_SCENES_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_SCENES_DIR))

from checker_material_support import CheckerMaterial

DATA_DIR = Path(__file__).parent.parent.parent.parent / "data"


def test_materialx_discovery_uses_highest_priority_resolver_root(tmp_path: Path) -> None:
    high_root = tmp_path / "high"
    low_root = tmp_path / "low"
    high_root.mkdir()
    low_root.mkdir()
    shutil.copyfile(
        DATA_DIR / "assets/cornell-box/mtlx-ref/cornell-box.mtlx",
        high_root / "material.mtlx",
    )
    shutil.copyfile(
        DATA_DIR / "assets/test_mtlx_opacity/opacity_test.mtlx",
        low_root / "material.mtlx",
    )

    resolver = f2.AssetResolver([high_root, low_root])
    records = f2.discover_materialx_renderable_elements("material.mtlx", asset_resolver=resolver)
    names = {str(record["name"]) for record in records}

    assert "light" in names
    assert "white" in names
    assert "green" in names
    assert "red" in names


def test_materialx_canonical_names_are_independent_of_editable_inputs() -> None:
    device_type = next(
        (
            device_type
            for device_type in helpers.DEFAULT_DEVICE_TYPES
            if device_type != spy.DeviceType.cuda
        ),
        None,
    )
    if device_type is None:
        pytest.skip("MaterialXMaterial requires a non-CUDA device")

    device = helpers.get_device(device_type)
    document = """<materialx version="1.39" colorspace="lin_rec709">
  <standard_surface name="surface" type="surfaceshader" />
  <surfacematerial name="M" type="material">
    <input name="surfaceshader" type="surfaceshader" nodename="surface" />
  </surfacematerial>
</materialx>"""

    def make_props(
        *,
        canonicalize_names: bool,
        editable_params: str | None = None,
        surface_base: float = 0.25,
    ) -> f2.Properties:
        props = f2.Properties()
        props["mtlx_buffer"] = document
        props["mtlx_node_name"] = "M"
        props["mtlx_canonicalize_names"] = canonicalize_names
        props["inputs:surface_base"] = surface_base
        if editable_params is not None:
            props["mtlx_editable_params"] = editable_params
        return props

    # Keep reflection-only coverage out of the scene that is compiled for the
    # GPU witness below. Native generation tests cover the larger source-
    # identity matrix without paying for every module in cold Python CI.
    property_scene = f2.Scene.create(device)
    canonical_baked = property_scene.create_material(
        f2.MaterialXMaterial, make_props(canonicalize_names=True)
    )
    assert canonical_baked.mtlx_canonicalize_names
    assert "inputs:surface_base" not in canonical_baked

    authored_live = property_scene.create_material(
        f2.MaterialXMaterial,
        make_props(canonicalize_names=False, editable_params="*", surface_base=0.75),
    )
    assert not authored_live.mtlx_canonicalize_names
    assert authored_live["inputs:surface_base"] == pytest.approx(0.75)

    subset_material = property_scene.create_material(
        f2.MaterialXMaterial,
        make_props(
            canonicalize_names=True,
            editable_params="surface_base",
            surface_base=0.625,
        ),
    )
    assert subset_material["inputs:surface_base"] == pytest.approx(0.625)
    assert "inputs:surface_base_color" not in subset_material

    def make_textured_props(texture: Path, color_scale: float) -> f2.Properties:
        textured_document = f"""<materialx version="1.39" colorspace="lin_rec709">
  <image name="base_image" type="color3">
    <input name="file" type="filename" value="{texture.as_posix()}" />
  </image>
  <multiply name="scaled_color" type="color3">
    <input name="in1" type="color3" nodename="base_image" />
    <input name="in2" type="color3" value="1, 1, 1" />
  </multiply>
  <standard_surface name="surface" type="surfaceshader">
    <input name="base_color" type="color3" nodename="scaled_color" />
    <input name="specular" type="float" value="0" />
  </standard_surface>
  <surfacematerial name="M" type="material">
    <input name="surfaceshader" type="surfaceshader" nodename="surface" />
  </surfacematerial>
</materialx>"""
        props = f2.Properties()
        props["mtlx_buffer"] = textured_document
        props["mtlx_node_name"] = "M"
        props["mtlx_canonicalize_names"] = True
        props["mtlx_editable_params"] = "scaled_color_in2"
        props["inputs:scaled_color_in2"] = spy.float3(color_scale, color_scale, color_scale)
        return props

    scene = f2.Scene.create(device)
    texture_root = DATA_DIR / "assets/textures/test_texture_manager"
    blue_high = scene.create_material(
        f2.MaterialXMaterial,
        make_textured_props(texture_root / "regular_blue.png", 0.5),
    )
    blue_low = scene.create_material(
        f2.MaterialXMaterial,
        make_textured_props(texture_root / "regular_blue.png", 0.125),
    )
    white_high = scene.create_material(
        f2.MaterialXMaterial,
        make_textured_props(texture_root / "white_1k.png", 0.5),
    )

    scene.update()
    assert blue_high.required_module().name == blue_low.required_module().name
    assert blue_high.required_module().name == white_high.required_module().name

    # All three instances share generated code. Evaluate two pairs to verify
    # that both scalar and texture values still come from per-instance data.
    module, helper = load_scene_sampling_module(device, scene)
    material_eval = module["material_eval_simple"].as_func().write(helper.bind_scene)
    uvs = np.array([(0.5, 0.5)], dtype=np.float32)
    wi = np.array([(0.0, 1.0, 0.0)], dtype=np.float32)
    wo = np.array([(0.0, 1.0, 0.0)], dtype=np.float32)

    def evaluate(material: f2.Material) -> np.ndarray:
        result = cast(spy.Tensor, material_eval(material, uvs, wi, wo))
        value = _float3_array(result.cursor()[0].read())
        assert np.isfinite(value).all()
        return value

    blue_high_value = evaluate(blue_high)
    blue_low_value = evaluate(blue_low)
    white_high_value = evaluate(white_high)
    assert np.all(blue_high_value >= blue_low_value)
    assert np.any(blue_high_value > blue_low_value)
    assert not np.allclose(blue_high_value, blue_low_value)
    assert blue_high_value[2] > blue_high_value[0]
    assert np.allclose(white_high_value, white_high_value[0])
    assert white_high_value[0] > blue_high_value[0]
    assert white_high_value[1] > blue_high_value[1]
    assert not np.allclose(blue_high_value, white_high_value)


@dataclass
class SceneMaterialTestDesc:
    scene_material_type: type
    properties: dict[str, Any]
    expected_results: dict[str, Any]


MATERIALX_TILED_BRASS_PROPERTIES = {
    "mtlx_path": str(
        DATA_DIR.parent
        / "external/MaterialX/resources/Materials/Examples/StandardSurface/standard_surface_brass_tiled.mtlx"
    ),
    "mtlx_node_name": "Tiled_Brass",
    "mtlx_basepath": str(
        DATA_DIR.parent / "external/MaterialX/resources/Materials/Examples/StandardSurface"
    ),
}


MATERIALX_TILED_BRASS_BSDF_MIX_EXPECTED_RESULTS = {
    "opacity": 1.0,
    "sample": {
        "wo_ws": spy.float3(-0.022421893, 0.9994462, 0.024587013),
        "pdf": 52.19148254394531,
        "weight": spy.float3(0.5868579, 0.3289865, 0.10662885),
        "flags": 2,
    },
    "eval": spy.float3(52.730022, 29.559902, 9.580754),
    "eval_pdf": 89.8514023,
    "collect_properties": {
        "material_color": spy.float3(0.03386597, 0.01901918, 0.00621708),
        "emission": spy.float3(0, 0, 0),
        "metallic": 0.0,
        "roughness": 0.02977316826581955,
        "guide_normal": spy.float3(0, 1, 0),
        "diffuse_reflection_albedo": spy.float3(0.03386597, 0.01901918, 0.00621708),
        "diffuse_transmission_albedo": spy.float3(0, 0, 0),
        "specular_reflection_albedo": spy.float3(0.53486705, 0.3003822, 0.09819043),
        "specular_transmission_albedo": spy.float3(0, 0, 0),
        "specular_reflectance": spy.float3(0.53486705, 0.3003822, 0.09819043),
    },
    "get_lobe_types": BSDFFlags.glossy_reflection,
    "collect_extra_bsdf_properties": {
        "bsdf_count": 2,
        "bsdf_N": [spy.float3(0, 0, 1)] * 2,
        "bsdf_T": [spy.float3(1, 0, 0)] * 2,
        "bsdf_B": [spy.float3(0, 1, 0)] * 2,
        "bsdf_albedo": [
            spy.float3(0.040000003, 0.040000003, 0.040000003),
            spy.float3(0.98642945, 0.98642945, 0.98642945),
        ],
        "bsdf_weight": [
            spy.float3(1, 1, 1),
            spy.float3(0.5360069, 0.28324518, 0.06529358),
        ],
        "bsdf_roughness": [spy.float2(0.029773166, 0.029773166)] * 2,
    },
}


def _merge_expected_results(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in overrides.items():
        base_value = result.get(key)
        if isinstance(base_value, dict) and isinstance(value, dict):
            result[key] = _merge_expected_results(base_value, value)
        else:
            result[key] = value
    return result


def _materialx_tiled_brass_test_desc(
    layering_mode: f2.MaterialXLayeringMode | None,
    expected_overrides: dict[str, Any] | None = None,
) -> SceneMaterialTestDesc:
    properties = dict(MATERIALX_TILED_BRASS_PROPERTIES)
    if layering_mode is not None:
        properties["mtlx_layering_mode"] = layering_mode
    return SceneMaterialTestDesc(
        f2.MaterialXMaterial,
        properties,
        _merge_expected_results(
            MATERIALX_TILED_BRASS_BSDF_MIX_EXPECTED_RESULTS,
            expected_overrides or {},
        ),
    )


_MATERIALX_DEFAULT_EXPECTED_OVERRIDES: dict[str, Any] = {}

# Scene-based materials (Material) - excludes Python material which is not supported
SCENE_MATERIALS = [
    SceneMaterialTestDesc(
        CheckerMaterial,
        {
            "color_a": spy.float3(0.92, 0.62, 0.24),
            "color_b": spy.float3(0.10, 0.18, 0.35),
            "scale": 8.0,
        },
        {
            "opacity": 1.0,
            "sample": {
                "wo_ws": spy.float3(0.20243618, 0.85205954, 0.4827154),
                "pdf": 0.27121896,
                "weight": spy.float3(0.92, 0.62, 0.24),
                "flags": 1,
            },
            "eval": spy.float3(0.2928450, 0.19735223, 0.07639437),
            "eval_pdf": 0.318309873,
            "collect_properties": {
                "roughness": 0.0,
                "guide_normal": spy.float3(0, 1, 0),
                "diffuse_reflection_albedo": spy.float3(0.92, 0.62, 0.24),
                "specular_reflection_albedo": spy.float3(0, 0, 0),
            },
            "get_lobe_types": BSDFFlags.diffuse_reflection,
            "collect_extra_bsdf_properties": {
                "bsdf_count": 0,
            },
        },
    ),
    SceneMaterialTestDesc(
        PythonMaterialSample,
        {},
        {
            "opacity": 0.25,
            "sample": {
                "wo_ws": spy.float3(0.20243618, 0.85205954, 0.4827154),
                "pdf": 0.27121896,
                "weight": spy.float3(1, 0, 1),
                "flags": 1,
            },
            "eval": spy.float3(0.31830987, 0, 0.31830987),
            "eval_pdf": 0.318309873,
            "collect_properties": {
                "material_color": spy.float3(1, 0, 1),
                "metallic": 0.0,
                "roughness": 0.0,
                "guide_normal": spy.float3(0, 1, 0),
                "diffuse_reflection_albedo": spy.float3(1, 0, 1),
                "specular_reflection_albedo": spy.float3(0, 0, 0),
            },
            "get_lobe_types": BSDFFlags.diffuse_reflection,
            "collect_extra_bsdf_properties": {
                "bsdf_count": 1,
                "bsdf_N": [spy.float3(0, 0, 1)],
                "bsdf_T": [spy.float3(1, 0, 0)],
                "bsdf_B": [spy.float3(0, 1, 0)],
                "bsdf_albedo": [spy.float3(1, 0, 1)],
                "bsdf_weight": [spy.float3(1, 1, 1)],
                "bsdf_roughness": [spy.float2(0.5, 0.5)],
            },
        },
    ),
    SceneMaterialTestDesc(
        f2.StandardMaterial,
        {
            "base_color_factor": spy.float3(0.8, 0.4, 0.2),
            "metallic_factor": 0.0,
            "roughness_factor": 0.6,
            "ior": 1.5,
        },
        {
            "opacity": 1.0,
            "eval": spy.float3(0.27920887, 0.15188491, 0.088222936),
            "eval_pdf": 0.341449767,
            "collect_properties": {
                "material_color": spy.float3(0.8, 0.4, 0.2),
                "metallic": 0.0,
                "roughness": 0.6,
                "guide_normal": spy.float3(0, 1, 0),
                "diffuse_reflection_albedo": spy.float3(0.8, 0.4, 0.2),
                "diffuse_transmission_albedo": spy.float3(0, 0, 0),
                "specular_reflection_albedo": spy.float3(0.04, 0.04, 0.04),
                "specular_transmission_albedo": spy.float3(0, 0, 0),
                "specular_reflectance": spy.float3(0.04, 0.04, 0.04),
            },
            "get_lobe_types": BSDFFlags.diffuse_reflection | BSDFFlags.glossy_reflection,
            "collect_extra_bsdf_properties": {
                "bsdf_count": 4,
                "bsdf_albedo": [
                    spy.float3(0.8, 0.4, 0.2),
                    spy.float3(1, 1, 1),
                    spy.float3(0.04, 0.04, 0.04),
                    spy.float3(0.04, 0.04, 0.04),
                ],
                "bsdf_weight": [
                    spy.float3(1, 1, 1),
                    spy.float3(0, 0, 0),
                    spy.float3(1, 1, 1),
                    spy.float3(0, 0, 0),
                ],
                "bsdf_roughness": [
                    spy.float2(1, 1),
                    spy.float2(1, 1),
                    spy.float2(0.6, 0.6),
                    spy.float2(0.6, 0.6),
                ],
            },
        },
    ),
    SceneMaterialTestDesc(
        f2.StandardMaterial,
        {
            "base_color_factor": spy.float3(0.8, 0.8, 0.8),
            "metallic_factor": 0.0,
            "roughness_factor": 0.5,
            "ior": 1.5,
            "transmission_factor": spy.float3(0.8, 0.9, 1.0),
            "specular_transmission_factor": 0.75,
            "thin_walled": True,
        },
        {
            "opacity": 1.0,
            "eval": spy.float3(0.11459155, 0.11459155, 0.11459155),
            "eval_pdf": 0.119356081,
            "collect_properties": {
                "material_color": spy.float3(0.8, 0.8, 0.8),
                "metallic": 0.0,
                "roughness": 0.5,
                "guide_normal": spy.float3(0, 1, 0),
                "diffuse_reflection_albedo": spy.float3(0.2, 0.2, 0.2),
                "diffuse_transmission_albedo": spy.float3(0, 0, 0),
                "specular_reflection_albedo": spy.float3(0.01, 0.01, 0.01),
                "specular_transmission_albedo": spy.float3(0.03, 0.03, 0.03),
                "specular_reflectance": spy.float3(0.04, 0.04, 0.04),
            },
            "get_lobe_types": (
                BSDFFlags.diffuse_reflection
                | BSDFFlags.glossy_reflection
                | BSDFFlags.glossy_transmission
            ),
            "collect_extra_bsdf_properties": {
                "bsdf_count": 4,
                "bsdf_albedo": [
                    spy.float3(0.8, 0.8, 0.8),
                    spy.float3(0.8, 0.9, 1.0),
                    spy.float3(0.04, 0.04, 0.04),
                    spy.float3(0.04, 0.04, 0.04),
                ],
                "bsdf_weight": [
                    spy.float3(0.25, 0.25, 0.25),
                    spy.float3(0, 0, 0),
                    spy.float3(0.25, 0.25, 0.25),
                    spy.float3(0.75, 0.75, 0.75),
                ],
                "bsdf_roughness": [
                    spy.float2(1, 1),
                    spy.float2(1, 1),
                    spy.float2(0.5, 0.5),
                    spy.float2(0.5, 0.5),
                ],
            },
        },
    ),
    SceneMaterialTestDesc(
        f2.StandardSpecGlossMaterial,
        {
            "diffuse_factor": spy.float3(0.7, 0.3, 0.1),
            "specular_factor": spy.float3(0.2, 0.1, 0.05),
            "glossiness_factor": 0.65,
            "ior": 1.5,
        },
        {
            "opacity": 1.0,
            # At normal incidence: 0.8 * diffuse / pi + F0 / (4 * pi * roughness^4).
            "eval": spy.float3(1.23884458, 0.60668989, 0.29061255),
            # Mixed PDF with attenuated diffuse and the unchanged raw-diffuse metallic estimate.
            "eval_pdf": 1.95298496,
            "collect_properties": {
                "roughness": 0.35,
                "guide_normal": spy.float3(0, 1, 0),
                "diffuse_reflection_albedo": spy.float3(0.56, 0.24, 0.08),
                "diffuse_transmission_albedo": spy.float3(0, 0, 0),
                "specular_reflection_albedo": spy.float3(0.2, 0.1, 0.05),
                "specular_transmission_albedo": spy.float3(0, 0, 0),
                "specular_reflectance": spy.float3(0.2, 0.1, 0.05),
            },
            "get_lobe_types": BSDFFlags.diffuse_reflection | BSDFFlags.glossy_reflection,
            "collect_extra_bsdf_properties": {
                "bsdf_count": 4,
                "bsdf_albedo": [
                    spy.float3(0.56, 0.24, 0.08),
                    spy.float3(1, 1, 1),
                    spy.float3(0.2, 0.1, 0.05),
                    spy.float3(0.2, 0.1, 0.05),
                ],
                "bsdf_weight": [
                    spy.float3(1, 1, 1),
                    spy.float3(0, 0, 0),
                    spy.float3(1, 1, 1),
                    spy.float3(0, 0, 0),
                ],
                "bsdf_roughness": [
                    spy.float2(1, 1),
                    spy.float2(1, 1),
                    spy.float2(0.35, 0.35),
                    spy.float2(0.35, 0.35),
                ],
            },
        },
    ),
    _materialx_tiled_brass_test_desc(None, _MATERIALX_DEFAULT_EXPECTED_OVERRIDES),
    _materialx_tiled_brass_test_desc(f2.MaterialXLayeringMode.closure_tree),
    SceneMaterialTestDesc(
        f2.MDLMaterial,
        {
            "mdl_library_path": str(DATA_DIR / "assets/mdl_sdk_examples"),
            "mdl_material_name": "carbon_composite::carbon_composite",
            "mdl_class_compilation": False,
        },
        {
            "opacity": 1.0,
            "sample": {
                # MDL xi receives (u_dir.x, u_dir.y, u_lobe, raw seed coordinate).
                "wo_ws": spy.float3(-0.14830333, 0.57023865, 0.8079813),
                "pdf": 1.0471245050430298,
                "weight": spy.float3(0.0005298115, 0.0005298115, 0.0005298115),
                "flags": 2,
            },
            "eval": spy.float3(0.0028125402, 0.0028125402, 0.0028125402),
            "eval_pdf": 4.77464819,
            "collect_properties": {
                "material_color": spy.float3(0, 0, 0),
                "metallic": 0.0,
                "roughness": 0.21119996905326843,
                "guide_normal": spy.float3(-7.437747e-05, 1, 0.000107270644),
                "diffuse_reflection_albedo": spy.float3(0, 0, 0),
                "specular_reflection_albedo": spy.float3(0.040565446, 0.040565446, 0.040565446),
            },
            "get_lobe_types": BSDFFlags.all,
            "collect_extra_bsdf_properties": {
                "bsdf_count": 1,
                "bsdf_N": [spy.float3(-7.437747e-05, 1, 0.000107270644)],
                "bsdf_albedo": [spy.float3(0.040565446, 0.040565446, 0.040565446)],
                "bsdf_weight": [spy.float3(1, 1, 1)],
                "bsdf_roughness": [spy.float2(0.0384, 0.38399994)],
            },
        },
    ),
    SceneMaterialTestDesc(
        f2.MDLMaterial,
        {
            "mdl_library_path": str(DATA_DIR / "assets/mdl_sdk_examples"),
            "mdl_material_name": "carbon_composite::carbon_composite",
            "mdl_class_compilation": False,
            "learnable": True,
        },
        {
            "opacity": 1.0,
            "eval": spy.float3(26.002197, 26.002197, 26.002197),
            "eval_pdf": 30.7740364,
            "collect_properties": {
                "material_color": spy.float3(0, 0, 0),
                "metallic": 0.0,
                "roughness": 0.21159996092319489,
                "guide_normal": spy.float3(-0.00092971936, 0.9999986, 0.0013408844),
                "diffuse_reflection_albedo": spy.float3(0, 0, 0),
                "specular_reflection_albedo": spy.float3(0.040565446, 0.040565446, 0.040565446),
            },
            "get_lobe_types": BSDFFlags.all,
            "collect_extra_bsdf_properties": {
                "bsdf_count": 2,
            },
        },
    ),
]


MATERIALX_CUDA_SKIP_REASON = (
    "MaterialXMaterial on CUDA is reserved for nightly runs because of pathological "
    "NVRTC 12.6 compile times (use --nightly to run)."
)


_MATERIALX_LARGE_EDITABLE_DATA_DOCUMENT = """\
<materialx version="1.39">
  <gltf_pbr name="shader" type="surfaceshader" nodedef="ND_gltf_pbr_surfaceshader">
    <input name="base_color" type="color3" nodename="base_color_image" output="outcolor" />
    <input name="metallic" type="float" nodename="metallic_extract" />
    <input name="roughness" type="float" nodename="roughness_extract" />
    <input name="occlusion" type="float" nodename="occlusion_extract" />
    <input name="transmission" type="float" value="0" />
    <input name="specular" type="float" value="1" />
    <input name="specular_color" type="color3" value="1, 1, 1" />
    <input name="ior" type="float" value="1.5" />
    <input name="alpha" type="float" value="1" />
    <input name="alpha_mode" type="integer" value="0" />
    <input name="alpha_cutoff" type="float" value="0.5" />
    <input name="iridescence" type="float" value="0" />
    <input name="iridescence_ior" type="float" value="1.3" />
    <input name="iridescence_thickness" type="float" value="100" />
    <input name="sheen_color" type="color3" value="0, 0, 0" />
    <input name="sheen_roughness" type="float" value="0" />
    <input name="clearcoat" type="float" value="0" />
    <input name="clearcoat_roughness" type="float" value="0" />
    <input name="emissive" type="color3" nodename="emissive_image" output="outcolor" />
    <input name="emissive_strength" type="float" value="1" />
    <input name="thickness" type="float" value="0" />
    <input name="attenuation_color" type="color3" value="1, 1, 1" />
    <input name="anisotropy_strength" type="float" value="0" />
    <input name="anisotropy_rotation" type="float" value="0" />
    <input name="dispersion" type="float" value="0" />
    <input name="normal" type="vector3" nodename="normal_image" />
  </gltf_pbr>
  <gltf_colorimage name="base_color_image" type="multioutput">
    <input name="file" type="filename" value="" colorspace="srgb_texture" />
    <output name="outcolor" type="color3" />
    <output name="outa" type="float" />
  </gltf_colorimage>
  <gltf_colorimage name="emissive_image" type="multioutput">
    <input name="file" type="filename" value="" colorspace="srgb_texture" />
    <output name="outcolor" type="color3" />
    <output name="outa" type="float" />
  </gltf_colorimage>
  <gltf_image name="orm_image" type="vector3">
    <input name="file" type="filename" value="" />
  </gltf_image>
  <extract name="metallic_extract" type="float">
    <input name="in" type="vector3" nodename="orm_image" />
    <input name="index" type="integer" value="2" />
  </extract>
  <extract name="roughness_extract" type="float">
    <input name="in" type="vector3" nodename="orm_image" />
    <input name="index" type="integer" value="1" />
  </extract>
  <extract name="occlusion_extract" type="float">
    <input name="in" type="vector3" nodename="orm_image" />
    <input name="index" type="integer" value="0" />
  </extract>
  <gltf_normalmap name="normal_image" type="vector3">
    <input name="file" type="filename" value="" />
  </gltf_normalmap>
  <surfacematerial name="material" type="material">
    <input name="surfaceshader" type="surfaceshader" nodename="shader" />
  </surfacematerial>
</materialx>
"""


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_materialx_editable_data_buffer_uses_packed_reflected_stride(
    device_type: spy.DeviceType, tmp_path: Path
) -> None:
    shader_path = tmp_path / f"large_editable_data_{device_type.name}.slang"
    device = helpers.get_device(device_type, use_cache=False)
    try:
        scene = f2.Scene.create(device)
        material = scene.create_material(
            f2.MaterialXMaterial,
            f2.Properties(
                {
                    "mtlx_buffer": _MATERIALX_LARGE_EDITABLE_DATA_DOCUMENT,
                    "mtlx_node_name": "material",
                    "mtlx_editable_params": "*",
                    "debug_write_shader_path": str(shader_path),
                }
            ),
        )

        scene.update()

        assert material.slang_type_name
        shader = shader_path.read_text(encoding="utf-8")
        assert "BufferPtr data_buffer" in shader
        assert "data_buffer.load<" in shader

        module_name = next(
            line.removeprefix("module ").removesuffix(";")
            for line in shader.splitlines()
            if line.startswith("module ")
        )
        data_name = next(
            line.split()[2]
            for line in shader.splitlines()
            if line.startswith("    public struct ") and "_Data" in line
        )
        module = device.load_module_from_source(module_name, shader)
        buffer_type = module.layout.find_type_by_name(f"StructuredBuffer<mtlx::{data_name}>")
        assert buffer_type is not None
        data_layout = module.layout.get_type_layout(buffer_type).element_type_layout

        padding = 0
        previous_end = 0
        for field in data_layout.fields:
            assert field.offset >= previous_end
            padding += field.offset - previous_end
            previous_end = field.offset + field.type_layout.size
        assert data_layout.stride >= previous_end
        padding += data_layout.stride - previous_end

        # Baked geometry selectors are excluded from the editable data layout.
        expected_stride = 432 if device_type == spy.DeviceType.cuda else 428
        expected_padding = 4 if device_type == spy.DeviceType.cuda else 0
        assert data_layout.stride == expected_stride
        assert padding == expected_padding
    finally:
        device.close()


def _is_materialx_on_cuda(device_type: spy.DeviceType, material_type: type) -> bool:
    return device_type == spy.DeviceType.cuda and material_type is f2.MaterialXMaterial


def _materialx_skip_reason(
    device_type: spy.DeviceType, material_type: type, nightly: bool
) -> str | None:
    if not _is_materialx_on_cuda(device_type, material_type):
        return None
    if not nightly:
        return MATERIALX_CUDA_SKIP_REASON
    return None


def _skip_materialx_on_cuda(
    device_type: spy.DeviceType, material_type: type, *, nightly: bool
) -> None:
    if reason := _materialx_skip_reason(device_type, material_type, nightly):
        pytest.skip(reason)


def _scene_materials_for_device(
    device_type: spy.DeviceType, *, nightly: bool
) -> list[SceneMaterialTestDesc]:
    return [
        test_material
        for test_material in SCENE_MATERIALS
        if not _materialx_skip_reason(device_type, test_material.scene_material_type, nightly)
        and (
            nightly
            or test_material.scene_material_type is not f2.MaterialXMaterial
            or "mtlx_layering_mode" not in test_material.properties
        )
    ]


def create_scene_material(
    scene: f2.Scene, test_material: SceneMaterialTestDesc
) -> f2.Material | PythonMaterialSample:
    props = f2.Properties()
    for key, value in test_material.properties.items():
        props[key] = value
    return cast(
        f2.Material | PythonMaterialSample,
        scene.create_material(test_material.scene_material_type, props),
    )


def test_discover_materialx_renderable_elements_defaults_to_mtlx() -> None:
    materialx_root = DATA_DIR.parent / "external/MaterialX"
    material_path = (
        materialx_root
        / "resources/Materials/Examples/StandardSurface/standard_surface_brass_tiled.mtlx"
    )
    asset_resolver = f2.AssetResolver().push(materialx_root)

    elements = f2.discover_materialx_renderable_elements(
        material_path,
        asset_resolver=asset_resolver,
    )

    assert ("Tiled_Brass", "material") in {(item["name"], item["type"]) for item in elements}


def test_discover_materialx_renderable_elements_includes_output_types() -> None:
    materialx_root = DATA_DIR.parent / "external/MaterialX"
    material_path = materialx_root / "resources/Materials/TestSuite/pbrlib/bsdf/dielectric.mtlx"
    asset_resolver = f2.AssetResolver().push(materialx_root)

    elements = f2.discover_materialx_renderable_elements(
        material_path,
        asset_resolver=asset_resolver,
    )
    element_types = {item["name"]: item["type"] for item in elements}

    assert element_types["dielectric_bsdf/RT_out"] == "surfaceshader"
    assert element_types["dielectric_bsdf/layer_RT_out"] == "surfaceshader"


def _float3_array(value: Any) -> np.ndarray:
    return np.array([float(value.x), float(value.y), float(value.z)], dtype=np.float32)


def _assert_finite_float3(value: Any) -> None:
    assert np.all(np.isfinite(_float3_array(value)))


def assert_valid_sample_result(sample: dict[str, Any]) -> None:
    assert np.isfinite(sample["pdf"])
    assert sample["pdf"] >= 0.0
    _assert_finite_float3(sample["weight"])
    _assert_finite_float3(sample["wo_ws"])
    if sample["pdf"] > 0.0:
        assert np.linalg.norm(_float3_array(sample["wo_ws"])) == pytest.approx(
            1.0, rel=1e-4, abs=1e-4
        )


@pytest.mark.nightly
@pytest.mark.slow
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_materialx_mtlx_tiled_brass_scene_material_successor(
    device_type: spy.DeviceType,
    request: pytest.FixtureRequest,
) -> None:
    _skip_materialx_on_cuda(
        device_type,
        f2.MaterialXMaterial,
        nightly=bool(request.config.getoption("--nightly")),
    )

    materialx_root = DATA_DIR.parent / "external/MaterialX"
    material_path = (
        materialx_root
        / "resources/Materials/Examples/StandardSurface/standard_surface_brass_tiled.mtlx"
    )

    device = helpers.get_device(device_type)
    scene = f2.Scene.create(device)

    props = f2.Properties()
    props["mtlx_path"] = str(material_path)
    props["mtlx_node_name"] = "Tiled_Brass"
    props["mtlx_basepath"] = str(material_path.parent)
    props["mtlx_search_paths"] = str(materialx_root)
    props["mtlx_layering_mode"] = f2.MaterialXLayeringMode.bsdf_mix
    scene.create_material("MaterialXMaterial", props)
    scene.update()

    sample_count = 1
    seed = np.array([(7, 0, 7)], dtype=np.uint32)
    uvs = np.array([(0.5, 0.5)], dtype=np.float32)
    wi = np.array([(0, 1, 0)], dtype=np.float32)
    wo = np.array([(0, 1, 0)], dtype=np.float32)

    module, helper = load_scene_sampling_module(device, scene)
    material = scene.materials[0]

    sample_func = (
        module["material_sample_simple<TinyUniformSampleGenerator>"]
        .as_func()
        .write(helper.bind_scene)
    )
    sample_results = cast(spy.Tensor, sample_func(seed, material, uvs, wi))
    sample_cursor = sample_results.cursor()
    sample_cursor.load()
    for sample_index in range(sample_count):
        sample = sample_cursor[sample_index].read()
        assert_valid_sample_result(sample)

    eval_func = module["material_eval_simple"].as_func().write(helper.bind_scene)
    eval_results = cast(spy.Tensor, eval_func(material, uvs, wi, wo))
    eval_cursor = eval_results.cursor()
    eval_cursor.load()

    for sample_index in range(sample_count):
        _assert_finite_float3(eval_cursor[sample_index].read())

    lobe_types_func = (
        module["material_get_lobe_types_simple<TinyUniformSampleGenerator>"]
        .as_func()
        .write(helper.bind_scene)
    )
    lobe_types_results = cast(spy.Tensor, lobe_types_func(material, uvs, wi))
    lobe_types_cursor = lobe_types_results.cursor()
    lobe_types_cursor.load()
    has_lobes = False
    for sample_index in range(sample_count):
        has_lobes = has_lobes or lobe_types_cursor[sample_index].read() != BSDFFlags.none
    assert has_lobes

    extra_func = (
        module["material_collect_extra_bsdf_properties_simple"].as_func().write(helper.bind_scene)
    )
    extra_results = cast(spy.Tensor, extra_func(material, uvs, wi, wo))
    extra_cursor = extra_results.cursor()
    extra_cursor.load()
    extra = extra_cursor[0].read()
    # Current MTLX Tiled Brass reduces to the coat and metal BSDFs.
    assert extra["bsdf_count"] == 2
    for bsdf_index in range(extra["bsdf_count"]):
        _assert_finite_float3(extra["bsdf_N"][bsdf_index])
        _assert_finite_float3(extra["bsdf_albedo"][bsdf_index])
        _assert_finite_float3(extra["bsdf_weight"][bsdf_index])


def assert_expected_subset(
    test_type: str, result: Any, test_material: SceneMaterialTestDesc
) -> None:
    def assert_close(result_value: Any, expected_value: Any, field_path: str) -> None:
        if isinstance(expected_value, list):
            assert len(result_value) >= len(
                expected_value
            ), f"Expected at least {len(expected_value)} values for {field_path}, got {len(result_value)}"
            for index, (rv, ev) in enumerate(zip(result_value, expected_value)):
                assert_close(rv, ev, f"{field_path}[{index}]")
            return

        if isinstance(expected_value, tuple):
            assert (
                result_value in expected_value
            ), f"Expected one of {expected_value} for {field_path}, got {result_value}"
            return

        if isinstance(expected_value, (float, spy.float2, spy.float3)):
            assert result_value == pytest.approx(
                expected_value, rel=0.1, abs=1e-4
            ), f"Expected {expected_value} for {field_path}, got {result_value}"
            return

        assert (
            result_value == expected_value
        ), f"Expected {expected_value} for {field_path}, got {result_value}"

    expected = test_material.expected_results.get(test_type, {})
    if isinstance(result, dict):
        for k, v in expected.items():
            assert k in result, f"Missing key {k} in {test_type} result"
            assert_close(result[k], v, k)
    else:
        assert_close(result, expected, test_type)


def load_scene_sampling_module(device: spy.Device, scene: f2.Scene) -> tuple[spy.Module, Any]:
    from falcor2.editor import SceneShaderHelper

    helper = SceneShaderHelper(device)
    module, _ = helper.prepare(scene, "falcor2.tools.materials.scene_sampling_tools")
    return module, helper


def _single_bsdf_materialx(
    bsdf_name: str, weight: float, *, weight_through_nodegraph: bool = False
) -> str:
    bsdf_color = "1.0, 1.0, 1.0"
    if weight_through_nodegraph:
        return f"""<?xml version="1.0"?>
<materialx version="1.39" colorspace="lin_rec709">
  <nodedef name="ND_weighted_root" node="weighted_root" nodegroup="pbr">
    <input name="in" type="BSDF" />
    <input name="weight" type="color3" value="1.0, 1.0, 1.0" />
    <output name="out" type="BSDF" />
  </nodedef>
  <nodegraph name="NG_weighted_root" nodedef="ND_weighted_root">
    <multiply name="multiply" type="BSDF" nodedef="ND_multiply_bsdfC">
      <input name="in1" type="BSDF" interfacename="in" />
      <input name="in2" type="color3" interfacename="weight" />
    </multiply>
    <output name="out" type="BSDF" nodename="multiply" />
  </nodegraph>
  <{bsdf_name} name="root_bsdf" type="BSDF">
    <input name="weight" type="float" value="1.0" />
    <input name="color" type="color3" value="{bsdf_color}" />
  </{bsdf_name}>
  <weighted_root name="weighted" type="BSDF">
    <input name="in" type="BSDF" nodename="root_bsdf" />
    <input name="weight" type="color3" value="{weight}, {weight}, {weight}" />
  </weighted_root>
  <surface name="surface1" type="surfaceshader">
    <input name="bsdf" type="BSDF" nodename="weighted" />
  </surface>
  <surfacematerial name="M" type="material">
    <input name="surfaceshader" type="surfaceshader" nodename="surface1" />
  </surfacematerial>
</materialx>"""
    return f"""<?xml version="1.0"?>
<materialx version="1.39" colorspace="lin_rec709">
  <{bsdf_name} name="root_bsdf" type="BSDF">
    <input name="weight" type="float" value="{weight}" />
    <input name="color" type="color3" value="{bsdf_color}" />
  </{bsdf_name}>
  <surface name="surface1" type="surfaceshader">
    <input name="bsdf" type="BSDF" nodename="root_bsdf" />
  </surface>
  <surfacematerial name="M" type="material">
    <input name="surfaceshader" type="surfaceshader" nodename="surface1" />
  </surfacematerial>
</materialx>"""


def _assert_float3_scaled(value: Any, reference: Any, scale: float) -> None:
    assert _float3_array(value) == pytest.approx(
        _float3_array(reference) * scale, rel=0.08, abs=1e-4
    )


def _read_single_material_outputs(
    module: spy.Module,
    helper: Any,
    material: f2.Material,
    wi_tuple: tuple[float, float, float],
    wo_tuple: tuple[float, float, float],
) -> dict[str, Any]:
    seed = np.array([(7, 0, 7)], dtype=np.uint32)
    uvs = np.array([(0.5, 0.5)], dtype=np.float32)
    wi = np.array([wi_tuple], dtype=np.float32)
    wo = np.array([wo_tuple], dtype=np.float32)

    sample_func = (
        module["material_sample_simple<TinyUniformSampleGenerator>"]
        .as_func()
        .write(helper.bind_scene)
    )
    sample_results = cast(spy.Tensor, sample_func(seed, material, uvs, wi))
    sample_cursor = sample_results.cursor()
    sample_cursor.load()

    eval_func = module["material_eval_simple"].as_func().write(helper.bind_scene)
    eval_results = cast(spy.Tensor, eval_func(material, uvs, wi, wo))
    eval_cursor = eval_results.cursor()
    eval_cursor.load()

    eval_with_pdf_func = module["material_eval_with_pdf_simple"].as_func().write(helper.bind_scene)
    eval_with_pdf_results = cast(spy.Tensor, eval_with_pdf_func(material, uvs, wi, wo))
    eval_with_pdf_cursor = eval_with_pdf_results.cursor()
    eval_with_pdf_cursor.load()

    properties_func = (
        module["material_collect_properties_simple"].as_func().write(helper.bind_scene)
    )
    properties_results = cast(spy.Tensor, properties_func(material, uvs, wi))
    properties_cursor = properties_results.cursor()
    properties_cursor.load()

    extra_func = (
        module["material_collect_extra_bsdf_properties_simple"].as_func().write(helper.bind_scene)
    )
    extra_results = cast(spy.Tensor, extra_func(material, uvs, wi, wo))
    extra_cursor = extra_results.cursor()
    extra_cursor.load()

    return {
        "sample": sample_cursor[0].read(),
        "eval": eval_cursor[0].read(),
        "eval_with_pdf": eval_with_pdf_cursor[0].read(),
        "properties": properties_cursor[0].read(),
        "extra": extra_cursor[0].read(),
    }


@pytest.mark.nightly
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("test_material", SCENE_MATERIALS)
def test_scene_material_codegen(
    device_type: spy.DeviceType,
    test_material: SceneMaterialTestDesc,
    request: pytest.FixtureRequest,
) -> None:
    _skip_materialx_on_cuda(
        device_type,
        test_material.scene_material_type,
        nightly=bool(request.config.getoption("--nightly")),
    )

    device = helpers.get_device(device_type)
    scene = f2.Scene.create(device)

    material = create_scene_material(scene, test_material)
    scene.update()

    required_module = material.required_module()
    requirements = scene.requirements
    modules = cast(list[spy.SlangModule], requirements.modules)
    type_conformances = cast(list[spy.TypeConformance], requirements.type_conformances)

    assert len(type_conformances) > 0
    for module in modules:
        assert module is not None

    if required_module is not None:
        assert len(modules) > 0
        assert any(module.name == required_module.name for module in modules)


@pytest.mark.nightly
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_scene_material_opacity_dispatch(device_type: spy.DeviceType) -> None:
    device = helpers.get_device(device_type)
    scene = f2.Scene.create(device)
    material = scene.create_material(PythonMaterialSample)
    scene.update()

    module, helper = load_scene_sampling_module(device, scene)
    opacity_func = module["material_eval_opacity_simple"].as_func().write(helper.bind_scene)
    uvs = np.array([(0.5, 0.5)], dtype=np.float32)
    wi = np.array([(0.0, 1.0, 0.0)], dtype=np.float32)
    opacity = cast(spy.Tensor, opacity_func(material, uvs, wi))
    opacity_cursor = opacity.cursor()
    opacity_cursor.load()

    assert opacity_cursor[0].read() == pytest.approx(0.25)


@pytest.mark.nightly
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_mdl_material_opacity_evaluation(device_type: spy.DeviceType) -> None:
    # TODO: Test fails on CUDA due to Slang codegen issue.
    # https://github.com/shader-slang/slang/issues/11658
    if device_type == spy.DeviceType.cuda:
        pytest.skip("Stochastic opacity evaluation fails on CUDA due to Slang codegen issue")
    device = helpers.get_device(device_type)
    scene = f2.Scene.create(device)
    mdl_path = DATA_DIR / "assets/test_mdl_opacity"

    def create_material(name: str, opacity: float | None = None) -> f2.Material:
        props = f2.Properties()
        props["mdl_library_path"] = str(mdl_path)
        props["mdl_material_name"] = f"opacity_test::{name}"
        props["mdl_class_compilation"] = True
        if opacity is not None:
            props["opacity"] = opacity
        return scene.create_material("MDLMaterial", props)

    constant_materials = [
        create_material("opaque"),
        create_material("transparent"),
        create_material("constant_half"),
        create_material("parameterized", 0.25),
    ]
    textured_material = create_material("textured")
    scene.update()

    assert scene.required_capabilities.has(f2.SceneCapabilityFlags.opacity_evaluation)
    module, helper = load_scene_sampling_module(device, scene)
    opacity_func = module["material_eval_opacity_simple"].as_func().write(helper.bind_scene)
    uvs = np.array([(0.5, 0.5)], dtype=np.float32)
    wi = np.array([(0.0, 1.0, 0.0)], dtype=np.float32)

    for material, expected in zip(constant_materials, (1.0, 0.0, 0.5, 0.25)):
        opacity = cast(spy.Tensor, opacity_func(material, uvs, wi))
        opacity_cursor = opacity.cursor()
        opacity_cursor.load()
        assert opacity_cursor[0].read() == pytest.approx(expected)

    texture_uvs = np.array(
        [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)],
        dtype=np.float32,
    )
    texture_wi = np.repeat(wi, len(texture_uvs), axis=0)
    texture_opacity = cast(spy.Tensor, opacity_func(textured_material, texture_uvs, texture_wi))
    texture_opacity_cursor = texture_opacity.cursor()
    texture_opacity_cursor.load()

    expected_texture_opacity = (0.0, 64.0 / 255.0, 128.0 / 255.0, 1.0)
    for index, expected in enumerate(expected_texture_opacity):
        assert texture_opacity_cursor[index].read() == pytest.approx(expected)


@pytest.mark.nightly_if(device_type=(spy.DeviceType.cuda,))
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_materialx_material_opacity_evaluation(
    device_type: spy.DeviceType, request: pytest.FixtureRequest
) -> None:
    # TODO: Test fails on CUDA due to Slang codegen issue.
    # https://github.com/shader-slang/slang/issues/11658
    if device_type == spy.DeviceType.cuda:
        pytest.skip("Stochastic opacity evaluation fails on CUDA due to Slang codegen issue")
    device = helpers.get_device(device_type)
    scene = f2.Scene.create(device)
    materialx_root = DATA_DIR / "assets/test_mtlx_opacity"
    props = f2.Properties()
    props["mtlx_basepath"] = str(materialx_root)
    props["mtlx_path"] = "opacity_test.mtlx"
    props["mtlx_node_name"] = "M"
    props["mtlx_editable_params"] = " surface_opacity, diffuse_color, surface_opacity "
    props["inputs:surface_opacity"] = 0.25
    material = scene.create_material(f2.MaterialXMaterial, props)

    complete_props = f2.Properties()
    complete_props["mtlx_basepath"] = str(materialx_root)
    complete_props["mtlx_path"] = "opacity_test.mtlx"
    complete_props["mtlx_node_name"] = "M"
    complete_props["mtlx_editable_params"] = "*"
    complete_material = scene.create_material(f2.MaterialXMaterial, complete_props)

    invalid_props = f2.Properties()
    invalid_props["mtlx_basepath"] = str(materialx_root)
    invalid_props["mtlx_path"] = "opacity_test.mtlx"
    invalid_props["mtlx_node_name"] = "M"
    invalid_props["mtlx_editable_params"] = "surface_opacity,*"
    with pytest.raises(Exception) as exc_info:
        scene.create_material(f2.MaterialXMaterial, invalid_props)
    assert "'*' must be used by itself" in str(exc_info.value)

    textured_props = f2.Properties()
    textured_props["mtlx_basepath"] = str(materialx_root)
    textured_props["mtlx_path"] = "opacity_test.mtlx"
    textured_props["mtlx_node_name"] = "Textured"
    textured_material = scene.create_material(f2.MaterialXMaterial, textured_props)

    luminance_props = f2.Properties()
    luminance_props["mtlx_basepath"] = str(materialx_root)
    luminance_props["mtlx_path"] = "opacity_luminance.mtlx"
    luminance_props["mtlx_node_name"] = "M"
    luminance_material = scene.create_material(f2.MaterialXMaterial, luminance_props)
    scene.update()

    assert "inputs:surface_opacity" in material
    assert "inputs:diffuse_color" in material
    assert "inputs:emission_color" not in material
    assert "inputs:surface_opacity" in complete_material
    assert "inputs:diffuse_color" in complete_material
    assert "inputs:emission_color" in complete_material
    assert scene.required_capabilities.has(f2.SceneCapabilityFlags.opacity_evaluation)
    texture_handles = textured_material.build_texture_list()
    assert len(texture_handles) == 1
    assert texture_handles[0].is_valid()
    assert texture_handles[0].is_finalized()
    module, helper = load_scene_sampling_module(device, scene)
    opacity_func = module["material_eval_opacity_simple"].as_func().write(helper.bind_scene)
    uvs = np.array([(0.5, 0.5)], dtype=np.float32)
    wi = np.array([(0.0, 1.0, 0.0)], dtype=np.float32)

    for expected in (0.25, 0.75):
        material["inputs:surface_opacity"] = expected
        scene.update()
        opacity = cast(spy.Tensor, opacity_func(material, uvs, wi))
        opacity_cursor = opacity.cursor()
        opacity_cursor.load()
        assert opacity_cursor[0].read() == pytest.approx(expected)

    if request.config.getoption("--nightly") or request.config.getoption("--all"):
        material.mtlx_layering_mode = f2.MaterialXLayeringMode.closure_tree
        scene.update()
        module, helper = load_scene_sampling_module(device, scene)
        opacity_func = module["material_eval_opacity_simple"].as_func().write(helper.bind_scene)
        opacity = cast(spy.Tensor, opacity_func(material, uvs, wi))
        opacity_cursor = opacity.cursor()
        opacity_cursor.load()
        assert opacity_cursor[0].read() == pytest.approx(0.75)

    luminance_opacity = cast(spy.Tensor, opacity_func(luminance_material, uvs, wi))
    luminance_opacity_cursor = luminance_opacity.cursor()
    luminance_opacity_cursor.load()
    assert luminance_opacity_cursor[0].read() == pytest.approx(0.0)

    texture_uvs = np.array(
        [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)],
        dtype=np.float32,
    )
    texture_wi = np.repeat(wi, len(texture_uvs), axis=0)
    texture_opacity = cast(spy.Tensor, opacity_func(textured_material, texture_uvs, texture_wi))
    texture_opacity_cursor = texture_opacity.cursor()
    texture_opacity_cursor.load()

    expected_texture_opacity = (0.0, 64.0 / 255.0, 128.0 / 255.0, 1.0)
    for index, expected in enumerate(expected_texture_opacity):
        assert texture_opacity_cursor[index].read() == pytest.approx(expected)


@pytest.mark.nightly
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_scene_material_texture_lists(
    device_type: spy.DeviceType, request: pytest.FixtureRequest
) -> None:
    device = helpers.get_device(device_type)
    textured_material_types = (f2.MaterialXMaterial, f2.MDLMaterial)
    keep_alive: list[tuple[f2.Scene, f2.Material, list[Any]]] = []

    for test_material in SCENE_MATERIALS:
        if test_material.scene_material_type not in textured_material_types:
            continue
        if _materialx_skip_reason(
            device_type, test_material.scene_material_type, request.config.getoption("--nightly")
        ):
            continue

        scene = f2.Scene.create(device)
        material = create_scene_material(scene, test_material)
        scene.update()

        texture_handles = material.build_texture_list()
        assert len(texture_handles) > 0
        for handle in texture_handles:
            assert handle.is_valid()
            assert handle.is_finalized()
            assert handle.texture is not None
            assert handle.sampler is not None
        keep_alive.append((scene, material, texture_handles))


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize(
    "material_type, texture_property",
    [
        ("StandardMaterial", "base_color_texture"),
        ("StandardSpecGlossMaterial", "diffuse_texture"),
        ("OpenPBRMaterial", "base_color_texture"),
        ("OpenPBRMaterial", "opacity_texture"),
        ("OpenPBRMaterial", "normal_texture"),
    ],
)
def test_material_encoded_texture_property(
    device_type: spy.DeviceType, material_type: str, texture_property: str
) -> None:
    device = helpers.get_device(device_type)
    scene = f2.Scene.create(device)
    image = DATA_DIR / "assets/textures/test_texture_manager/regular_blue.png"
    source = f2.Blob(image.read_bytes(), image.name)
    material = scene.create_material(
        material_type, f2.Properties({f"{texture_property}_bytes": source})
    )
    before = scene.texture_manager.stats.total_texture_count

    assert material[texture_property] is None
    scene.update()

    assert material[f"{texture_property}_bytes"].name == image.name
    assert material[texture_property] is None
    assert scene.texture_manager.stats.total_texture_count == before + 1
    if material_type == "StandardMaterial":
        module, helper = load_scene_sampling_module(device, scene)
        collect = module["material_collect_properties_simple"].as_func().write(helper.bind_scene)
        result = cast(
            spy.Tensor,
            collect(
                material,
                np.array([[0.5, 0.5]], dtype=np.float32),
                np.array([[0.0, 1.0, 0.0]], dtype=np.float32),
            ),
        )
        cursor = result.cursor()
        cursor.load()
        np.testing.assert_allclose(
            _float3_array(cursor[0].read()["material_color"]), [0.0, 0.0, 1.0], atol=2e-3
        )


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_scene_material_cursor_writers(device_type: spy.DeviceType) -> None:
    device = helpers.get_device(device_type)
    scene = f2.Scene.create(device)

    standard = scene.create_material(
        f2.StandardMaterial,
        f2.Properties(
            {
                "base_color_factor": spy.float3(0.25, 0.5, 0.75),
                "metallic_factor": 0.125,
                "roughness_factor": 0.625,
                "volume_sigma_a": spy.float3(0.1, 0.2, 0.3),
                "volume_sigma_s": spy.float3(0.4, 0.5, 0.6),
                "volume_anisotropy": 0.25,
                "base_color_texture_texcoord": 1,
                "metallic_roughness_texture_texcoord": 2,
                "normal_texture_texcoord": 3,
                "emissive_texture_texcoord": 4,
                "transmission_texture_texcoord": 5,
                "use_vertex_color": True,
            }
        ),
    )
    spec_gloss = scene.create_material(
        f2.StandardSpecGlossMaterial,
        f2.Properties(
            {
                "diffuse_factor": spy.float3(0.25, 0.5, 0.75),
                "specular_factor": spy.float3(0.125, 0.25, 0.5),
                "glossiness_factor": 0.625,
                "volume_sigma_a": spy.float3(0.2, 0.3, 0.4),
                "volume_sigma_s": spy.float3(0.5, 0.6, 0.7),
                "volume_anisotropy": -0.25,
                "diffuse_texture_texcoord": 1,
                "specular_glossiness_texture_texcoord": 2,
                "normal_texture_texcoord": 3,
                "emissive_texture_texcoord": 4,
                "transmission_texture_texcoord": 5,
                "use_vertex_color": True,
            }
        ),
    )
    openpbr = scene.create_material(
        "OpenPBRMaterial",
        f2.Properties(
            {
                "normal_texture_texcoord": 2,
                "opacity_texture_texcoord": 3,
                "opacity_texture_channel": 1,
                "use_vertex_color": True,
            }
        ),
    )
    python_material = scene.create_material(PythonMaterialSample)
    scene.update()

    def material_cursor(layout: Any, slang_type_name: str) -> spy.BufferCursor:
        buffer_type = layout.find_type_by_name(f"StructuredBuffer<{slang_type_name}>")
        assert buffer_type is not None
        buffer_type_layout = layout.get_type_layout(buffer_type)
        return spy.BufferCursor(device_type, buffer_type_layout.element_type_layout, 1)

    standard_cursor = material_cursor(scene.render_module.layout, "StandardMaterial")
    standard_cursor[0] = standard
    standard_result = standard_cursor[0].read()
    assert _float3_array(standard_result["base_color_factor"]) == pytest.approx(
        np.array([0.25, 0.5, 0.75], dtype=np.float32)
    )
    assert float(standard_result["metallic_factor"]) == pytest.approx(0.125)
    assert float(standard_result["roughness_factor"]) == pytest.approx(0.625)
    assert _float3_array(standard_result["volume_sigma_a"]) == pytest.approx([0.1, 0.2, 0.3])
    assert _float3_array(standard_result["volume_sigma_s"]) == pytest.approx([0.4, 0.5, 0.6])
    assert float(standard_result["volume_anisotropy"]) == pytest.approx(0.25)
    assert int(standard_result["texture_coordinate_selectors"]) == 55505

    spec_gloss_cursor = material_cursor(scene.render_module.layout, "StandardSpecGlossMaterial")
    spec_gloss_cursor[0] = spec_gloss
    spec_gloss_result = spec_gloss_cursor[0].read()
    assert _float3_array(spec_gloss_result["diffuse_factor"]) == pytest.approx(
        np.array([0.25, 0.5, 0.75], dtype=np.float32)
    )
    assert _float3_array(spec_gloss_result["specular_factor"]) == pytest.approx(
        np.array([0.125, 0.25, 0.5], dtype=np.float32)
    )
    assert float(spec_gloss_result["glossiness_factor"]) == pytest.approx(0.625)
    assert _float3_array(spec_gloss_result["volume_sigma_a"]) == pytest.approx([0.2, 0.3, 0.4])
    assert _float3_array(spec_gloss_result["volume_sigma_s"]) == pytest.approx([0.5, 0.6, 0.7])
    assert float(spec_gloss_result["volume_anisotropy"]) == pytest.approx(-0.25)
    assert int(spec_gloss_result["texture_coordinate_selectors"]) == 55505

    openpbr_buffer_type = scene.render_module.layout.find_type_by_name(
        "StructuredBuffer<OpenPBRMaterial>"
    )
    assert openpbr_buffer_type is not None
    openpbr_layout = scene.render_module.layout.get_type_layout(openpbr_buffer_type)
    assert openpbr_layout.element_type_layout.stride <= 128
    attributes_buffer_type = scene.render_module.layout.find_type_by_name(
        "StructuredBuffer<OpenPBRMaterial::Attributes>"
    )
    assert attributes_buffer_type is not None
    attributes_layout = scene.render_module.layout.get_type_layout(
        attributes_buffer_type
    ).element_type_layout
    attribute_fields = {field.name: field for field in attributes_layout.fields}
    assert attribute_fields["base_weight"].type_layout.stride == 8
    expected_float3_attribute_stride = 16 if device_type == spy.DeviceType.cuda else 12
    assert attribute_fields["base_color"].type_layout.stride == expected_float3_attribute_stride
    openpbr_cursor = material_cursor(scene.render_module.layout, "OpenPBRMaterial")
    openpbr_cursor[0] = openpbr
    openpbr_result = openpbr_cursor[0].read()
    assert int(openpbr_result["geometry_texture_metadata"]) == 346
    if device_type == spy.DeviceType.d3d12:
        # OpenPBR's optimized D3D attribute loads require a dedicated buffer at byte offset zero.
        attributes_bits = int(openpbr_result["attributes_buffer"]["data"])
        assert attributes_bits != f2.BufferPtr().data
        assert attributes_bits >> 32 == 0

    required_module = python_material.required_module()
    assert required_module is not None
    python_cursor = material_cursor(required_module.layout, "PythonMaterialSample")
    python_cursor[0] = python_material
    python_result = python_cursor[0].read()
    assert python_result["texture_index"] == 3


def _read_interior_medium(module: spy.Module, helper: Any, material: f2.Material) -> dict[str, Any]:
    func = module["material_interior_medium_simple"].as_func().write(helper.bind_scene)
    uvs = np.array([(0.5, 0.5)], dtype=np.float32)
    wi = np.array([(0.0, 1.0, 0.0)], dtype=np.float32)
    result = cast(spy.Tensor, func(material, uvs, wi))
    cursor = result.cursor()
    cursor.load()
    return cast(dict[str, Any], cursor[0].read())


def _assert_interior_medium(
    result: dict[str, Any], sigma_a: Any, sigma_s: Any, anisotropy: float
) -> None:
    assert _float3_array(result["sigma_a"]) == pytest.approx(sigma_a, rel=2e-3, abs=1e-5)
    assert _float3_array(result["sigma_s"]) == pytest.approx(sigma_s, rel=2e-3, abs=1e-5)
    assert float(result["anisotropy"]) == pytest.approx(anisotropy, rel=2e-3, abs=1e-5)


@pytest.mark.nightly_if_not_vk
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_material_interior_medium(device_type: spy.DeviceType) -> None:
    device = helpers.get_device(device_type)
    scene = f2.Scene.create(device)

    standard = scene.create_material(
        f2.StandardMaterial,
        f2.Properties(
            {
                "volume_sigma_a": spy.float3(0.1, 0.2, 0.3),
                "volume_sigma_s": spy.float3(0.4, 0.5, 0.6),
                "volume_anisotropy": 0.25,
            }
        ),
    )
    spec_gloss = scene.create_material(
        f2.StandardSpecGlossMaterial,
        f2.Properties(
            {
                "volume_sigma_a": spy.float3(0.2, 0.3, 0.4),
                "volume_sigma_s": spy.float3(0.5, 0.6, 0.7),
                "volume_anisotropy": -0.25,
            }
        ),
    )
    thin_walled = scene.create_material(
        f2.StandardMaterial,
        f2.Properties(
            {
                "thin_walled": True,
                "volume_sigma_a": spy.float3(1.0),
                "volume_sigma_s": spy.float3(1.0),
                "volume_anisotropy": 0.5,
            }
        ),
    )

    sigma_a = np.array([0.15, 0.3, 0.6], dtype=np.float32)
    sigma_s = np.array([0.05, 0.1, 0.2], dtype=np.float32)
    depth = 2.0
    openpbr = scene.create_material(
        "OpenPBRMaterial",
        f2.Properties(
            {
                "transmission_weight": 1.0,
                "transmission_color": spy.float3(*np.exp(-(sigma_a + sigma_s) * depth)),
                "transmission_depth": depth,
                "transmission_scatter": spy.float3(*(sigma_s * depth)),
                "transmission_scatter_anisotropy": 0.4,
            }
        ),
    )
    openpbr_zero_depth = scene.create_material(
        "OpenPBRMaterial",
        f2.Properties(
            {
                "transmission_weight": 1.0,
                "transmission_color": spy.float3(0.5),
                "transmission_depth": 0.0,
            }
        ),
    )

    scene.update()
    module, helper = load_scene_sampling_module(device, scene)

    _assert_interior_medium(
        _read_interior_medium(module, helper, standard), [0.1, 0.2, 0.3], [0.4, 0.5, 0.6], 0.25
    )
    _assert_interior_medium(
        _read_interior_medium(module, helper, spec_gloss), [0.2, 0.3, 0.4], [0.5, 0.6, 0.7], -0.25
    )
    _assert_interior_medium(
        _read_interior_medium(module, helper, thin_walled), [1, 1, 1], [1, 1, 1], 0.5
    )
    _assert_interior_medium(_read_interior_medium(module, helper, openpbr), sigma_a, sigma_s, 0.4)
    _assert_interior_medium(
        _read_interior_medium(module, helper, openpbr_zero_depth), [0, 0, 0], [0, 0, 0], 0.0
    )


@pytest.mark.nightly_if_not_vk
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_mdl_material_interior_medium(device_type: spy.DeviceType) -> None:
    device = helpers.get_device(device_type)
    scene = f2.Scene.create(device)
    material = scene.create_material(
        "MDLMaterial",
        f2.Properties(
            {
                "mdl_library_path": str(DATA_DIR / "assets/mdl_sdk_examples"),
                "mdl_material_name": "tutorials::example_volume",
                "mdl_class_compilation": False,
            }
        ),
    )
    scene.update()

    module, helper = load_scene_sampling_module(device, scene)
    extinction = -np.log(0.234) / 0.36
    scatter_color = np.array([0.368, 0.691, 0.603], dtype=np.float32)
    _assert_interior_medium(
        _read_interior_medium(module, helper, material),
        extinction * (1.0 - scatter_color),
        extinction * scatter_color,
        0.0,
    )


@pytest.mark.nightly
@pytest.mark.slow
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("test_material", SCENE_MATERIALS)
def test_scene_material_sampler(
    device_type: spy.DeviceType,
    test_material: SceneMaterialTestDesc,
    request: pytest.FixtureRequest,
) -> None:
    """
    Test individual scene materials using cursor-writer binding.
    """

    _skip_materialx_on_cuda(
        device_type,
        test_material.scene_material_type,
        nightly=bool(request.config.getoption("--nightly")),
    )

    device = helpers.get_device(device_type)
    scene = f2.Scene.create(device)

    create_scene_material(scene, test_material)

    scene.update()

    # Golden API/binding witnesses; randomized sampling contracts live in test_material_sampling.py.
    seed = np.array([(7, 0, 7)], dtype=np.uint32)
    uvs = np.array([(0.5, 0.5)], dtype=np.float32)
    wi = np.array([(0, 1, 0)], dtype=np.float32)
    wo = np.array([(0, 1, 0)], dtype=np.float32)
    module, helper = load_scene_sampling_module(device, scene)

    material = scene.materials[0]

    # Test sample
    signature = "material_sample_simple<TinyUniformSampleGenerator>"
    material_sample = module[signature].as_func().write(helper.bind_scene)
    sample_results = cast(spy.Tensor, material_sample(seed, material, uvs, wi))
    sample_cursor = sample_results.cursor()
    sample_cursor.load()
    result = sample_cursor[0].read()
    assert_valid_sample_result(result)
    assert_expected_subset("sample", result, test_material)

    # Test eval
    signature = "material_eval_simple"
    material_eval = module[signature].as_func().write(helper.bind_scene)
    eval_results = cast(spy.Tensor, material_eval(material, uvs, wi, wo))
    eval_cursor = eval_results.cursor()
    eval_cursor.load()
    result = eval_cursor[0].read()
    assert_expected_subset("eval", result, test_material)

    # Test opacity
    signature = "material_eval_opacity_simple"
    material_eval_opacity = module[signature].as_func().write(helper.bind_scene)
    opacity_results = cast(spy.Tensor, material_eval_opacity(material, uvs, wi))
    opacity_cursor = opacity_results.cursor()
    opacity_cursor.load()
    result = opacity_cursor[0].read()
    assert np.isfinite(result)
    assert_expected_subset("opacity", result, test_material)

    # Test collect_properties
    signature = "material_collect_properties_simple"
    material_collect_properties = module[signature].as_func().write(helper.bind_scene)
    properties_results = cast(spy.Tensor, material_collect_properties(material, uvs, wi))
    properties_cursor = properties_results.cursor()
    properties_cursor.load()
    result = properties_cursor[0].read()
    assert_expected_subset("collect_properties", result, test_material)

    # Test get_lobe_types
    signature = "material_get_lobe_types_simple<TinyUniformSampleGenerator>"
    material_get_lobe_types = module[signature].as_func().write(helper.bind_scene)
    lobe_types_results = cast(spy.Tensor, material_get_lobe_types(material, uvs, wi))
    lobe_types_cursor = lobe_types_results.cursor()
    lobe_types_cursor.load()
    result = lobe_types_cursor[0].read()
    assert_expected_subset("get_lobe_types", result, test_material)

    # Test collect_extra_bsdf_properties
    signature = "material_collect_extra_bsdf_properties_simple"
    material_collect_extra_bsdf_properties = module[signature].as_func().write(helper.bind_scene)
    result_data = cast(spy.Tensor, material_collect_extra_bsdf_properties(material, uvs, wi, wo))
    result_cursor = result_data.cursor()
    result_cursor.load()
    result = result_cursor[0].read()
    assert_expected_subset("collect_extra_bsdf_properties", result, test_material)


@pytest.mark.nightly_if(device_type=(spy.DeviceType.d3d12,))
@pytest.mark.slow
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_scene_material_sampler_batch(
    device_type: spy.DeviceType, request: pytest.FixtureRequest
) -> None:
    """
    Test batch scene materials (sample, eval, collect_extra_bsdf_properties).
    """

    device = helpers.get_device(device_type)
    scene = f2.Scene.create(device)

    active_scene_materials = _scene_materials_for_device(
        device_type,
        nightly=bool(request.config.getoption("--nightly") or request.config.getoption("--all")),
    )

    for test_material in active_scene_materials:
        create_scene_material(scene, test_material)

    scene.update()

    material_ids = [int(material.material_id) for material in scene.materials]
    assert len(material_ids) == len(active_scene_materials)

    material_ids_np = np.asarray(material_ids, dtype=np.int32)
    # Golden API/binding witnesses; randomized sampling contracts live in test_material_sampling.py.
    seed = np.array([(7, 0, 7)], dtype=np.uint32)
    uvs = np.array([(0.5, 0.5)], dtype=np.float32)
    wi = np.array([(0, 1, 0)], dtype=np.float32)
    wo = np.array([(0, 1, 0)], dtype=np.float32)
    seed = np.concatenate([seed] * len(material_ids), axis=0)
    uvs = np.concatenate([uvs] * len(material_ids), axis=0)
    wi = np.concatenate([wi] * len(material_ids), axis=0)
    wo = np.concatenate([wo] * len(material_ids), axis=0)

    module, helper = load_scene_sampling_module(device, scene)
    type_conformances = cast(list[spy.TypeConformance], scene.requirements.type_conformances)

    # Test sample
    signature = "material_sample_system<TinyUniformSampleGenerator>"
    material_sample = (
        module[signature]
        .as_func()
        .type_conformances(type_conformances)
        .write(helper.bind_scene)
        .map(spy.uint3, int, spy.float2, spy.float3)
    )
    sample_results = cast(spy.Tensor, material_sample(seed, material_ids_np, uvs, wi))
    sample_cursor = sample_results.cursor()
    sample_cursor.load()
    for i, test_material in enumerate(active_scene_materials):
        result: dict[str, Any] = sample_cursor[i].read()  # type: ignore
        assert_valid_sample_result(result)
        assert_expected_subset("sample", result, test_material)

    # Test eval and PDF
    signature = "material_eval_with_pdf_system"
    material_eval_with_pdf = (
        module[signature]
        .as_func()
        .type_conformances(type_conformances)
        .write(helper.bind_scene)
        .map(int, spy.float2, spy.float3, spy.float3)
    )
    eval_results = cast(spy.Tensor, material_eval_with_pdf(material_ids_np, uvs, wi, wo))
    eval_cursor = eval_results.cursor()
    eval_cursor.load()
    for i, test_material in enumerate(active_scene_materials):
        result: dict[str, Any] = eval_cursor[i].read()  # type: ignore
        assert_expected_subset("eval", result["value"], test_material)
        assert_expected_subset("eval_pdf", result["pdf"], test_material)

    # Test opacity
    signature = "material_eval_opacity_system"
    material_eval_opacity = (
        module[signature]
        .as_func()
        .type_conformances(type_conformances)
        .write(helper.bind_scene)
        .map(int, spy.float2, spy.float3)
    )
    opacity_results = cast(spy.Tensor, material_eval_opacity(material_ids_np, uvs, wi))
    opacity_cursor = opacity_results.cursor()
    opacity_cursor.load()
    for i, test_material in enumerate(active_scene_materials):
        result: float = opacity_cursor[i].read()
        assert np.isfinite(result)
        assert_expected_subset("opacity", result, test_material)

    # Test collect_properties
    signature = "material_collect_properties_system"
    material_collect_properties = (
        module[signature]
        .as_func()
        .type_conformances(type_conformances)
        .write(helper.bind_scene)
        .map(int, spy.float2, spy.float3)
    )
    properties_results = cast(spy.Tensor, material_collect_properties(material_ids_np, uvs, wi))
    properties_cursor = properties_results.cursor()
    properties_cursor.load()
    for i, test_material in enumerate(active_scene_materials):
        result: dict[str, Any] = properties_cursor[i].read()  # type: ignore
        assert_expected_subset("collect_properties", result, test_material)

    # Test get_lobe_types
    signature = "material_get_lobe_types_system<TinyUniformSampleGenerator>"
    material_get_lobe_types = (
        module[signature]
        .as_func()
        .type_conformances(type_conformances)
        .write(helper.bind_scene)
        .map(int, spy.float2, spy.float3)
    )
    lobe_types_results = cast(spy.Tensor, material_get_lobe_types(material_ids_np, uvs, wi))
    lobe_types_cursor = lobe_types_results.cursor()
    lobe_types_cursor.load()
    for i, test_material in enumerate(active_scene_materials):
        result: int = lobe_types_cursor[i].read()  # type: ignore
        assert_expected_subset("get_lobe_types", result, test_material)

    # Test collect_extra_bsdf_properties
    signature = "material_collect_extra_bsdf_properties_system"
    material_collect_extra_bsdf_properties = (
        module[signature]
        .as_func()
        .type_conformances(type_conformances)
        .write(helper.bind_scene)
        .map(int, spy.float2, spy.float3, spy.float3)
    )
    result_data = cast(
        spy.Tensor, material_collect_extra_bsdf_properties(material_ids_np, uvs, wi, wo)
    )
    result_cursor = result_data.cursor()
    result_cursor.load()
    for i, test_material in enumerate(active_scene_materials):
        result: dict[str, Any] = result_cursor[i].read()  # type: ignore
        assert_expected_subset("collect_extra_bsdf_properties", result, test_material)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_example_python_material(device_type: spy.DeviceType) -> None:
    device = helpers.get_device(device_type)
    scene = f2.Scene.create(device)

    props = f2.Properties()

    uvs = spy.float2(0.5, 0.5)
    wi = spy.float3(0, 1, 0)
    wo = spy.float3(0, 1, 0)

    material = scene.create_material(PythonMaterialSample, props)
    scene.update()

    module, helper = load_scene_sampling_module(device, scene)
    type_conformances = cast(list[spy.TypeConformance], scene.requirements.type_conformances)

    def compare_evals(mat: f2.Material, expected_albedo: spy.float3):
        signature_simple = "material_eval_simple"
        material_eval_simple = module[signature_simple].as_func().write(helper.bind_scene)

        result_simple = cast(spy.float3, material_eval_simple(mat, uvs, wi, wo))
        device.flush_print()

        signature_system = "material_eval_system"
        material_eval_system = (
            module[signature_system]
            .as_func()
            .write(helper.bind_scene)
            .type_conformances(type_conformances)
            .map(int, spy.float2, spy.float3, spy.float3)
        )
        result_system = material_eval_system(int(mat.material_id), uvs, wi, wo)
        device.flush_print()

        assert result_simple == result_system
        albedo = result_simple * np.pi
        assert expected_albedo == albedo, f"Expected albedo {expected_albedo}, got {albedo}"

    compare_evals(material, expected_albedo=spy.float3(1, 0, 1))
    material.texture_index = 0
    scene.update()
    compare_evals(material, expected_albedo=spy.float3(1, 0, 0))
    material.texture_index = 1
    scene.update()
    compare_evals(material, expected_albedo=spy.float3(0, 0, 1))
    material.texture_index = 2
    scene.update()
    compare_evals(material, expected_albedo=spy.float3(1, 0, 1))


if __name__ == "__main__":
    # pytest.main([__file__, "-v", "--slow"])
    test_example_python_material(spy.DeviceType.d3d12)
