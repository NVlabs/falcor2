# SPDX-License-Identifier: Apache-2.0

import sys
from dataclasses import dataclass
from enum import IntFlag
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
else:
    from .python_material_sample import PythonMaterialSample


EXAMPLE_DIR = Path(__file__).resolve().parents[3] / "examples" / "pathtracer"
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

from checker_material import CheckerMaterial


DATA_DIR = Path(__file__).parent.parent.parent.parent / "data"


class LobeTypes(IntFlag):
    """Python enum matching Slang's LobeTypes definition."""

    none = 0x00
    diffuse_reflection = 0x01
    glossy_reflection = 0x02
    delta_reflection = 0x04
    diffuse_transmission = 0x10
    glossy_transmission = 0x20
    delta_transmission = 0x40
    glossy_curve = 0x80
    diffuse = 0x11
    glossy = 0x22
    delta = 0x44
    curve = 0x80
    reflection = 0x07
    transmission = 0x70
    non_delta = 0xB3
    non_delta_reflection = 0x03
    non_delta_transmission = 0x30
    all = 0xF7


@dataclass
class SceneMaterialTestDesc:
    scene_material_type: type
    properties: dict[str, Any]
    expected_results: dict[str, Any]


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
            "sample": {
                "wo_ws": spy.float3(-0.19386107, 0.9628616, -0.1879241),
                "pdf": 0.30648836493492126,
                "weight": spy.float3(0.92, 0.62, 0.24),
                "lobe_types": 1,
            },
            "eval": spy.float3(0.2928450, 0.19735223, 0.07639437),
            "collect_properties": {
                "roughness": 0.0,
                "guide_normal": spy.float3(0, 1, 0),
                "diffuse_reflection_albedo": spy.float3(0.92, 0.62, 0.24),
                "specular_reflection_albedo": spy.float3(0, 0, 0),
            },
            "get_lobe_types": LobeTypes.diffuse_reflection,
            "collect_extra_bsdf_properties": {
                "bsdf_count": 0,
            },
        },
    ),
    SceneMaterialTestDesc(
        PythonMaterialSample,
        {},
        {
            "sample": {
                "wo_ws": spy.float3(-0.19386107, 0.9628616, -0.1879241),
                "pdf": 0.30648836493492126,
                "weight": spy.float3(1, 0, 1),
                "lobe_types": 1,
            },
            "eval": spy.float3(0.31830987, 0, 0.31830987),
            "collect_properties": {
                "roughness": 0.0,
                "guide_normal": spy.float3(0, 1, 0),
                "diffuse_reflection_albedo": spy.float3(1, 0, 1),
                "specular_reflection_albedo": spy.float3(0, 0, 0),
            },
            "get_lobe_types": LobeTypes.diffuse_reflection,
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
            "eval": spy.float3(0.27920887, 0.15188491, 0.088222936),
            "collect_properties": {
                "roughness": 0.6,
                "guide_normal": spy.float3(0, 1, 0),
                "diffuse_reflection_albedo": spy.float3(0.8, 0.4, 0.2),
                "diffuse_transmission_albedo": spy.float3(0, 0, 0),
                "specular_reflection_albedo": spy.float3(0.04, 0.04, 0.04),
                "specular_transmission_albedo": spy.float3(0, 0, 0),
                "specular_reflectance": spy.float3(0.04, 0.04, 0.04),
            },
            "get_lobe_types": LobeTypes.diffuse_reflection | LobeTypes.glossy_reflection,
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
            "thin_surface": True,
        },
        {
            "eval": spy.float3(0.11459155, 0.11459155, 0.11459155),
            "collect_properties": {
                "roughness": 0.5,
                "guide_normal": spy.float3(0, 1, 0),
                "diffuse_reflection_albedo": spy.float3(0.2, 0.2, 0.2),
                "diffuse_transmission_albedo": spy.float3(0, 0, 0),
                "specular_reflection_albedo": spy.float3(0.01, 0.01, 0.01),
                "specular_transmission_albedo": spy.float3(0.03, 0.03, 0.03),
                "specular_reflectance": spy.float3(0.04, 0.04, 0.04),
            },
            "get_lobe_types": (
                LobeTypes.diffuse_reflection
                | LobeTypes.glossy_reflection
                | LobeTypes.glossy_transmission
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
            "eval": spy.float3(1.2834091, 0.62578905, 0.29697904),
            "collect_properties": {
                "roughness": 0.35,
                "guide_normal": spy.float3(0, 1, 0),
                "diffuse_reflection_albedo": spy.float3(0.7, 0.3, 0.1),
                "diffuse_transmission_albedo": spy.float3(0, 0, 0),
                "specular_reflection_albedo": spy.float3(0.2, 0.1, 0.05),
                "specular_transmission_albedo": spy.float3(0, 0, 0),
                "specular_reflectance": spy.float3(0.2, 0.1, 0.05),
            },
            "get_lobe_types": LobeTypes.diffuse_reflection | LobeTypes.glossy_reflection,
            "collect_extra_bsdf_properties": {
                "bsdf_count": 4,
                "bsdf_albedo": [
                    spy.float3(0.7, 0.3, 0.1),
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
    SceneMaterialTestDesc(
        f2.MaterialXMaterial,
        {
            "mtlx_path": str(
                DATA_DIR.parent
                / "external/MaterialX/resources/Materials/Examples/StandardSurface/standard_surface_brass_tiled.mtlx"
            ),
            "mtlx_node_name": "Tiled_Brass",
            "mtlx_basepath": str(
                DATA_DIR.parent / "external/MaterialX/resources/Materials/Examples/StandardSurface"
            ),
        },
        {
            "sample": {
                "wo_ws": spy.float3(-0.022421893, 0.9994462, 0.024587013),
                "pdf": 52.19148254394531,
                "weight": spy.float3(0.5868579, 0.3289865, 0.10662885),
                "lobe_types": 2,
            },
            "eval": spy.float3(51.228004, 28.764034, 9.393779),
            "collect_properties": {
                "roughness": 0.02977316826581955,
                "guide_normal": spy.float3(0, 0.99999994, 0),
                "diffuse_reflection_albedo": spy.float3(0, 0, 0),
                "specular_reflection_albedo": spy.float3(0.568733, 0.3194014, 0.104407504),
            },
            "get_lobe_types": LobeTypes.glossy_reflection,
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
        },
    ),
    SceneMaterialTestDesc(
        f2.MDLMaterial,
        {
            "mdl_library_path": str(DATA_DIR / "assets/mdl_sdk_examples"),
            "mdl_material_name": "carbon_composite::carbon_composite",
            "mdl_class_compilation": False,
        },
        {
            "sample": {
                "wo_ws": spy.float3(0.04189388, 0.90362144, -0.42627868),
                "pdf": 2.1251282691955566,
                "weight": spy.float3(0.0005833827, 0.0005833827, 0.0005833827),
                "lobe_types": 2,
            },
            "eval": spy.float3(0.0028125402, 0.0028125402, 0.0028125402),
            "collect_properties": {
                "roughness": 0.0,
                "diffuse_reflection_albedo": spy.float3(0, 0, 0),
                "specular_reflection_albedo": spy.float3(1, 1, 1),
            },
            "get_lobe_types": LobeTypes.all,
            "collect_extra_bsdf_properties": {
                "bsdf_count": 1,
                "bsdf_N": [spy.float3(-7.437747e-05, 1, 0.000107270644)],
                "bsdf_albedo": [spy.float3(0.040565446, 0.040565446, 0.040565446)],
                "bsdf_weight": [spy.float3(1, 1, 1)],
                "bsdf_roughness": [spy.float2(0.0384, 0.38399994)],
            },
        },
    ),
]


MATERIALX_CUDA_SKIP_REASON = (
    "MaterialXMaterial is disabled on CUDA while pathological NVRTC 12.6 compile times "
    "are investigated."
)


def _is_materialx_on_cuda(device_type: spy.DeviceType, material_type: type) -> bool:
    return device_type == spy.DeviceType.cuda and material_type is f2.MaterialXMaterial


def _skip_materialx_on_cuda(device_type: spy.DeviceType, material_type: type) -> None:
    if _is_materialx_on_cuda(device_type, material_type):
        pytest.skip(MATERIALX_CUDA_SKIP_REASON)


def _scene_materials_for_device(device_type: spy.DeviceType) -> list[SceneMaterialTestDesc]:
    return [
        test_material
        for test_material in SCENE_MATERIALS
        if not _is_materialx_on_cuda(device_type, test_material.scene_material_type)
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


def test_discover_materialx_material_node_names_defaults_to_mx139() -> None:
    materialx_root = DATA_DIR.parent / "external/MaterialX"
    material_path = (
        materialx_root
        / "resources/Materials/Examples/StandardSurface/standard_surface_brass_tiled.mtlx"
    )
    asset_resolver = f2.AssetResolver()
    asset_resolver.add_search_path(materialx_root)

    node_names = f2.discover_materialx_material_node_names(
        material_path,
        asset_resolver=asset_resolver,
    )

    assert "Tiled_Brass" in node_names


def test_discover_materialx_material_node_names_includes_renderable_outputs() -> None:
    materialx_root = DATA_DIR.parent / "external/MaterialX"
    material_path = materialx_root / "resources/Materials/TestSuite/pbrlib/bsdf/dielectric.mtlx"
    asset_resolver = f2.AssetResolver()
    asset_resolver.add_search_path(materialx_root)

    node_names = f2.discover_materialx_material_node_names(
        material_path,
        asset_resolver=asset_resolver,
    )

    assert "dielectric_bsdf/RT_out" in node_names
    assert "dielectric_bsdf/layer_RT_out" in node_names


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_materialx_rejects_cuda_by_default(device_type: spy.DeviceType) -> None:
    if device_type != spy.DeviceType.cuda:
        pytest.skip("CUDA-only MaterialX guard test.")

    device = helpers.get_device(device_type)
    scene = f2.Scene(device)

    with pytest.raises(Exception) as exc_info:
        scene.create_material("MaterialXMaterial", f2.Properties())

    message = str(exc_info.value)
    assert "MaterialXMaterial is disabled on CUDA" in message
    assert "NVRTC 12.6" in message


def _float3_array(value: Any) -> np.ndarray:
    return np.array([value.x, value.y, value.z], dtype=np.float32)


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


@pytest.mark.slow
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES[:1])
def test_materialx_mx139_tiled_brass_scene_material_successor(
    device_type: spy.DeviceType,
) -> None:
    _skip_materialx_on_cuda(device_type, f2.MaterialXMaterial)

    materialx_root = DATA_DIR.parent / "external/MaterialX"
    material_path = (
        materialx_root
        / "resources/Materials/Examples/StandardSurface/standard_surface_brass_tiled.mtlx"
    )

    device = helpers.get_device(device_type)
    scene = f2.Scene(device)

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

    eval_func = (
        module["material_eval_simple<TinyUniformSampleGenerator>"]
        .as_func()
        .write(helper.bind_scene)
    )
    eval_results = cast(spy.Tensor, eval_func(seed, material, uvs, wi, wo))
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
        has_lobes = has_lobes or lobe_types_cursor[sample_index].read() != LobeTypes.none
    assert has_lobes

    extra_func = (
        module["material_collect_extra_bsdf_properties_simple"].as_func().write(helper.bind_scene)
    )
    extra_results = cast(spy.Tensor, extra_func(material, uvs, wi, wo))
    extra_cursor = extra_results.cursor()
    extra_cursor.load()
    extra = extra_cursor[0].read()
    # Current MX139 Tiled Brass reduces to the coat and metal BSDFs.
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
    module = helper.get_module(scene, "falcor2.tools.materials.scene_sampling_tools")
    return module, helper


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("test_material", SCENE_MATERIALS)
def test_scene_material_codegen(
    device_type: spy.DeviceType, test_material: SceneMaterialTestDesc
) -> None:
    _skip_materialx_on_cuda(device_type, test_material.scene_material_type)

    device = helpers.get_device(device_type)
    scene = f2.Scene(device)

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


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_scene_material_texture_lists(device_type: spy.DeviceType) -> None:
    device = helpers.get_device(device_type)
    textured_material_types = (f2.MaterialXMaterial, f2.MDLMaterial)
    keep_alive: list[tuple[f2.Scene, f2.Material, list[Any]]] = []

    for test_material in SCENE_MATERIALS:
        if test_material.scene_material_type not in textured_material_types:
            continue
        if _is_materialx_on_cuda(device_type, test_material.scene_material_type):
            continue

        scene = f2.Scene(device)
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
def test_scene_material_cursor_writers(device_type: spy.DeviceType) -> None:
    device = helpers.get_device(device_type)
    scene = f2.Scene(device)

    standard = scene.create_material(
        f2.StandardMaterial,
        f2.Properties(
            {
                "base_color_factor": spy.float3(0.25, 0.5, 0.75),
                "metallic_factor": 0.125,
                "roughness_factor": 0.625,
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
    assert standard_result["base_color_factor"] == pytest.approx(spy.float3(0.25, 0.5, 0.75))
    assert standard_result["metallic_factor"] == pytest.approx(0.125)
    assert standard_result["roughness_factor"] == pytest.approx(0.625)

    required_module = python_material.required_module()
    assert required_module is not None
    python_cursor = material_cursor(required_module.layout, "PythonMaterialSample")
    python_cursor[0] = python_material
    python_result = python_cursor[0].read()
    assert python_result["texture_index"] == 3


@pytest.mark.slow
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("test_material", SCENE_MATERIALS)
def test_scene_material_sampler(
    device_type: spy.DeviceType, test_material: SceneMaterialTestDesc
) -> None:
    """
    Test individual scene materials using get_this().
    """

    _skip_materialx_on_cuda(device_type, test_material.scene_material_type)

    device = helpers.get_device(device_type)
    scene = f2.Scene(device)

    create_scene_material(scene, test_material)

    scene.update()

    samples_per_material = 100

    seed = np.random.randint(0, 2**32, size=(samples_per_material, 3), dtype=np.uint32)
    uvs = np.random.rand(samples_per_material, 2).astype(np.float32)
    wi = np.random.randn(samples_per_material, 3).astype(np.float32)
    wi /= np.linalg.norm(wi, axis=1, keepdims=True)
    wo = np.random.randn(samples_per_material, 3).astype(np.float32)
    wo /= np.linalg.norm(wo, axis=1, keepdims=True)

    # Values for which we know sample/eval outputs
    seed[0] = (7, 0, 7)
    uvs[0] = (0.5, 0.5)
    wi[0] = (0, 1, 0)
    wo[0] = (0, 1, 0)

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
    signature = "material_eval_simple<TinyUniformSampleGenerator>"
    material_eval = module[signature].as_func().write(helper.bind_scene)
    eval_results = cast(spy.Tensor, material_eval(seed, material, uvs, wi, wo))
    eval_cursor = eval_results.cursor()
    eval_cursor.load()
    result = eval_cursor[0].read()
    assert_expected_subset("eval", result, test_material)

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


@pytest.mark.slow
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_scene_material_sampler_batch(device_type: spy.DeviceType) -> None:
    """
    Test batch scene materials (sample, eval, collect_extra_bsdf_properties).
    """

    device = helpers.get_device(device_type)
    scene = f2.Scene(device)

    samples_per_material = 100
    active_scene_materials = _scene_materials_for_device(device_type)

    for test_material in active_scene_materials:
        create_scene_material(scene, test_material)

    scene.update()

    material_ids = [int(material.material_id) for material in scene.materials]
    assert len(material_ids) == len(active_scene_materials)

    material_ids_np = np.array([material_ids[0]] * samples_per_material, dtype=np.int32)
    seed = np.random.randint(0, 2**32, size=(samples_per_material, 3), dtype=np.uint32)
    uvs = np.random.rand(samples_per_material, 2).astype(np.float32)
    wi = np.random.randn(samples_per_material, 3).astype(np.float32)
    wi /= np.linalg.norm(wi, axis=1, keepdims=True)
    wo = np.random.randn(samples_per_material, 3).astype(np.float32)
    wo /= np.linalg.norm(wo, axis=1, keepdims=True)

    # Values for which we know sample/eval outputs for each material
    seed[0] = (7, 0, 7)
    uvs[0] = (0.5, 0.5)
    wi[0] = (0, 1, 0)
    wo[0] = (0, 1, 0)

    seed = np.concatenate([seed] * len(material_ids), axis=0)
    uvs = np.concatenate([uvs] * len(material_ids), axis=0)
    wi = np.concatenate([wi] * len(material_ids), axis=0)
    wo = np.concatenate([wo] * len(material_ids), axis=0)

    for material_id in material_ids[1:]:
        material_ids_np = np.concatenate(
            [material_ids_np, np.array([material_id] * samples_per_material, dtype=np.int32)],
            axis=0,
        )

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
        result: dict[str, Any] = sample_cursor[i * samples_per_material].read()  # type: ignore
        assert_valid_sample_result(result)
        assert_expected_subset("sample", result, test_material)

    # Test eval
    signature = "material_eval_system<TinyUniformSampleGenerator>"
    material_eval = (
        module[signature]
        .as_func()
        .type_conformances(type_conformances)
        .write(helper.bind_scene)
        .map(spy.uint3, int, spy.float2, spy.float3, spy.float3)
    )
    eval_results = cast(spy.Tensor, material_eval(seed, material_ids_np, uvs, wi, wo))
    eval_cursor = eval_results.cursor()
    eval_cursor.load()
    for i, test_material in enumerate(active_scene_materials):
        result: dict[str, Any] = eval_cursor[i * samples_per_material].read()  # type: ignore
        assert_expected_subset("eval", result, test_material)

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
        result: dict[str, Any] = properties_cursor[i * samples_per_material].read()  # type: ignore
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
        result: int = lobe_types_cursor[i * samples_per_material].read()  # type: ignore
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
        result: dict[str, Any] = result_cursor[i * samples_per_material].read()  # type: ignore
        assert_expected_subset("collect_extra_bsdf_properties", result, test_material)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_example_python_material(device_type: spy.DeviceType) -> None:
    device = helpers.get_device(device_type)
    scene = f2.Scene(device)

    props = f2.Properties()

    seed = spy.uint3(7, 0, 7)
    uvs = spy.float2(0.5, 0.5)
    wi = spy.float3(0, 1, 0)
    wo = spy.float3(0, 1, 0)

    material = scene.create_material(PythonMaterialSample, props)
    scene.update()

    module, helper = load_scene_sampling_module(device, scene)
    type_conformances = cast(list[spy.TypeConformance], scene.requirements.type_conformances)

    def compare_evals(mat: f2.Material, expected_albedo: spy.float3):
        signature_simple = f"material_eval_simple<TinyUniformSampleGenerator>"
        material_eval_simple = module[signature_simple].as_func().write(helper.bind_scene)

        result_simple = cast(spy.float3, material_eval_simple(seed, mat, uvs, wi, wo))
        device.flush_print()

        signature_system = f"material_eval_system<TinyUniformSampleGenerator>"
        material_eval_system = (
            module[signature_system]
            .as_func()
            .write(helper.bind_scene)
            .type_conformances(type_conformances)
            .map(spy.uint3, int, spy.float2, spy.float3, spy.float3)
        )
        result_system = material_eval_system(seed, int(mat.material_id), uvs, wi, wo)
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
