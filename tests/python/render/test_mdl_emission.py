# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MDL surface emission, light discovery and sampling on every backend."""

from pathlib import Path

import numpy as np
import pytest
import slangpy as spy

import falcor2 as f2
import falcor2.testing.helpers as helpers
from falcor2.editor import SceneShaderHelper
from falcor2.rendernodes import ReferencePathTracerNode
from pathtracer.test_pathtracer import (
    _add_quad,
    _half_image_means,
    _make_quad_camera,
    _render_mean,
)

pytestmark = pytest.mark.nightly


def _scene(device: spy.Device) -> tuple[f2.Scene, f2.StandardMaterial, f2.GeometryInstance]:
    scene = f2.Scene.create(device)
    material = scene.create_material(f2.StandardMaterial)
    geometry = scene.create_geometry(f2.StaticMeshGeometry)
    geometry.set_mesh_data(
        positions=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
        sub_mesh_indices=[np.array([[0, 1, 2]], dtype=np.uint32)],
        normals=np.tile(np.array([0, 0, 1], dtype=np.float32), (3, 1)),
        texcoords=np.tile(np.array([0.25, 0.5], dtype=np.float32), (3, 1)),
        vertex_attributes=[
            f2.MeshVertexAttribute(
                f2.MeshAttributeSemantic.unknown,
                0,
                np.tile(np.array([0.25, 0.5, 0.75], dtype=np.float32), (3, 1)),
                name="COLOR_0",
            ),
            f2.MeshVertexAttribute(
                f2.MeshAttributeSemantic.tex_coord,
                1,
                np.tile(np.array([0.75, 0.5], dtype=np.float32), (3, 1)),
            ),
        ],
    )
    instance = scene.create_entity().create_component(f2.GeometryInstance)
    instance.geometry = geometry
    instance.materials = [material]
    return scene, material, instance


def _material(scene: f2.Scene, definition: str = "Diffuse", **values: object) -> f2.MDLMaterial:
    return scene.create_material(
        f2.MDLMaterial,
        f2.Properties(
            {
                "mdl_library_path": str(
                    Path(__file__).parent / "data/mdl_generated_material_pipeline"
                ),
                "mdl_material_name": f"::emission::{definition}",
                "mdl_class_compilation": True,
                **values,
            }
        ),
    )


def _probe(
    scene: f2.Scene, instance: f2.GeometryInstance, outgoing: tuple[float, float, float] = (0, 0, 1)
) -> np.ndarray:
    scene.update()
    helper = SceneShaderHelper(scene.device)
    source = spy.Module.load_from_file(scene.device, "render/test_mdl_emission.slang")
    module, requirements = helper.prepare(scene, source)
    function = (
        module.probe.as_func()
        .type_conformances(requirements.type_conformances)
        .write(helper.bind_scene)
        .return_type(spy.Tensor)
    )
    return function(
        spy.Tensor.from_numpy(
            scene.device, np.array([int(instance.geometry_instance_id)], dtype=np.uint32)
        ),
        spy.float3(*outgoing),
    ).to_numpy()[0]


def _assert_emission(result: np.ndarray, expected: list[float] | np.ndarray) -> None:
    for row in (0, 1, 3):
        # MDL texture resources use half precision; allow their quantization.
        np.testing.assert_allclose(result[row, :3], expected, atol=3e-5, rtol=5e-4)
    np.testing.assert_allclose(result[2, :3], np.array(expected) * np.pi, atol=1e-4, rtol=5e-4)


def _candidate_triangle_count(scene: f2.Scene) -> int:
    helper = SceneShaderHelper(scene.device)
    source = spy.Module.load_from_file(scene.device, "render/test_mdl_emission.slang")
    module, _ = helper.prepare(scene, source)
    return int(module.candidate_triangle_count.as_func().write(helper.bind_scene)())


@pytest.mark.nightly_if_not_vk
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("definition", ["ConditionalSpatial", "ConditionalSpatialReverse"])
def test_mdl_conditional_spatial_emission(device: spy.Device, definition: str) -> None:
    scene, _, instance = _scene(device)
    reverse = definition == "ConditionalSpatialReverse"
    material = _material(scene, definition, nighttime=False, mdl_class_compilation=True)
    other_material = _material(scene, definition, nighttime=True, mdl_class_compilation=True)
    instance.materials = [material]
    other = scene.create_entity().create_component(f2.GeometryInstance)
    other.geometry = instance.geometry
    other.materials = [other_material]
    scene.update()
    module = material.required_module()
    assert other_material.required_module().name == module.name
    for nighttime, level in ((False, 1), (True, 1), (False, 1), (True, 0), (True, 2)):
        material["nighttime"] = nighttime
        material["level"] = level
        enabled = (nighttime and level > 0) != reverse
        result = _probe(scene, instance)
        assert result[1, 3] == enabled
        assert _candidate_triangle_count(scene) == int(enabled) + int(not reverse)
        _assert_emission(result, [0.25 if enabled else 0.0] * 3)
        assert material.required_module() is module
    _assert_emission(_probe(scene, other), [0.0 if reverse else 0.25] * 3)


@pytest.mark.nightly_if_not_vk
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("definition", ["SceneConditional", "StateConditional"])
def test_mdl_spatial_conditions_remain_conservative(device: spy.Device, definition: str) -> None:
    scene, _, instance = _scene(device)
    instance.materials = [_material(scene, definition, mdl_class_compilation=True)]
    result = _probe(scene, instance)
    _assert_emission(result, [0.0] * 3)
    assert result[1, 3] == 1
    assert _candidate_triangle_count(scene) == 1


@pytest.mark.nightly_if_not_vk
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("class_compilation", [True, False])
@pytest.mark.parametrize("definition", ["EmptyEdf", "ConstantBlack", "ConstantDim"])
def test_mdl_structural_emission_rejects_dark_triangle_candidates(
    device: spy.Device, class_compilation: bool, definition: str
) -> None:
    scene, _, instance = _scene(device)
    material = _material(scene, definition, mdl_class_compilation=class_compilation)
    instance.materials = [material]
    expected_candidate = definition == "ConstantDim"
    result = _probe(scene, instance)
    assert result[1, 3] == expected_candidate
    # Check candidates, not just active triangles: rejection must precede allocation/integration.
    assert _candidate_triangle_count(scene) == int(expected_candidate)
    if expected_candidate:
        assert result[1, 1] > 0
    else:
        np.testing.assert_array_equal(result[1, :3], 0)


@pytest.mark.nightly_if(definition=("Diffuse",), learnable=(True,))
@pytest.mark.nightly_if(definition=("Custom",), learnable=(False,))
@pytest.mark.nightly_if_not_vk(definition=("Diffuse",))
@pytest.mark.nightly_if(
    definition=("Custom",), device_type=(spy.DeviceType.d3d12, spy.DeviceType.cuda)
)
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("learnable", [False, True])
@pytest.mark.parametrize("definition", ["Diffuse", "Custom"])
def test_mdl_emission_sharing_and_live_light_updates(
    device: spy.Device, learnable: bool, definition: str
) -> None:
    scene, _, instance = _scene(device)
    materials = [
        _material(scene, definition, radiance=spy.float3(*value), learnable=learnable)
        for value in ((0, 0, 0), (2, 1, 0.5), (4, 2, 1))
    ]
    scene.update()
    assert len({m.slang_type_name for m in materials}) == 1
    module = materials[0].required_module()
    assert all(m.required_module().name == module.name for m in materials)
    instance.materials = [materials[0]]
    for value in ((0, 0, 0), (2, 1, 0.5), (0, 0, 0), (4, 2, 1)):
        materials[0]["radiance"] = spy.float3(*value)
        result = _probe(scene, instance)
        _assert_emission(result, list(value))
        assert result[0, 3] == result[3, 3] == bool(value[0])
        assert result[1, 3] == bool(value[0])
        assert _candidate_triangle_count(scene) == int(bool(value[0]))
        assert result[2, 3] == 100
        # Diffuse radiance is independent of angle: do not multiply by cosine twice.
        _assert_emission(_probe(scene, instance, (0.8, 0, 0.6)), list(value))
    assert materials[0].required_module() is module


@pytest.mark.nightly_if_not_vk
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_mdl_direct_emission_parameters_are_instance_local(device: spy.Device) -> None:
    scene, _, instance = _scene(device)
    dark = _material(scene, "Direct", radiance=spy.float3(0))
    bright = _material(scene, "Direct", radiance=spy.float3(0, 1, 0))
    other = scene.create_entity().create_component(f2.GeometryInstance)
    other.geometry = instance.geometry
    other.materials = [bright]
    instance.materials = [dark]
    scene.update()
    module = dark.required_module()
    assert bright.required_module().name == module.name
    for value in (0.0, 1e-8, 2.0, 0.0):
        dark["radiance"] = spy.float3(0, value, 0)
        result = _probe(scene, instance)
        assert result[1, 3] == (value != 0)
        assert _candidate_triangle_count(scene) == 1 + int(value != 0)
        np.testing.assert_allclose(result[1, :3], [0, value / np.pi, 0], atol=1e-10, rtol=1e-5)
        assert dark.required_module() is module
    np.testing.assert_allclose(_probe(scene, other)[1, :3], [0, 1 / np.pi, 0], atol=1e-6)


@pytest.mark.nightly_if_not_vk
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_mdl_emission_parameter_factors_and_sums(device: spy.Device) -> None:
    scene, _, instance = _scene(device)
    material = _material(scene, "Scaled")
    instance.materials = [material]
    scene.update()
    module = material.required_module()
    for radiance, strength, tint in (
        (1.0, 1.0, 1.0),
        (1.0, 0.0, 1.0),
        (1.0, 2.0, 1.0),
        (1.0, 2.0, 0.0),
        (1.0, 2.0, 0.5),
        (0.0, 2.0, 0.5),
    ):
        material["radiance"] = spy.float3(radiance)
        material["strength"] = strength
        material["tint"] = spy.float3(tint)
        expected = radiance * strength * tint
        result = _probe(scene, instance)
        assert result[1, 3] == (expected != 0)
        assert _candidate_triangle_count(scene) == int(expected != 0)
        _assert_emission(result, [expected] * 3)
        assert material.required_module() is module

    # Evaluate the entire sum: zero parameters can emit, and nonzero parameters can cancel.
    additive = _material(scene, "Additive")
    instance.materials = [additive]
    for value in (0.0, 1.0, -1.0):
        additive["radiance"] = spy.float3(value)
        result = _probe(scene, instance)
        assert result[1, 3] == (value != -1)
        assert _candidate_triangle_count(scene) == int(value != -1)
        _assert_emission(result, [value + 1] * 3)


@pytest.mark.nightly_if_not_vk
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("definition", ["Missing", "CustomMissing", "VertexColour"])
def test_mdl_absent_and_spatial_emission(device: spy.Device, definition: str) -> None:
    scene, _, instance = _scene(device)
    material = _material(scene, definition)
    instance.materials = [material]
    expected = [0.25, 0.5, 0.75] if definition == "VertexColour" else [0, 0, 0]
    result = _probe(scene, instance)
    _assert_emission(result, expected)
    assert result[0, 3] == result[3, 3] == (definition == "VertexColour")
    assert result[1, 3] == (definition == "VertexColour")
    assert _candidate_triangle_count(scene) == int(definition == "VertexColour")
    assert result[2, 3] == 100


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("definition", ["Power", "CustomPower"])
def test_mdl_power_emission_is_explicitly_rejected(device: spy.Device, definition: str) -> None:
    scene, _, _ = _scene(device)
    with pytest.raises(RuntimeError, match="intensity_power"):
        _material(scene, definition)


@pytest.mark.nightly_if_not_vk(definition=("Custom",))
@pytest.mark.nightly_if(
    definition=("Diffuse",), device_type=(spy.DeviceType.d3d12, spy.DeviceType.vulkan)
)
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("definition", ["Diffuse", "Custom"])
def test_mdl_emission_visible_beauty_and_guide(device: spy.Device, definition: str) -> None:
    scene = f2.Scene.create(device)
    emitter = _material(scene, definition, radiance=spy.float3(2, 1, 0.5))
    _add_quad(scene, z=0, normal_z=-1, material=emitter)
    scene.update()
    node = ReferencePathTracerNode.create(device)
    # Exact single-sample radiance checks require a unit filter weight.
    node.pixel_filter = f2.BoxReconstructionFilter()
    node.output_spec = f2.ContainerSpec.texture2d(spy.Format.rgba32_float)
    node.guide_output_specs = {"emission": f2.ContainerSpec.auto()}
    beauty, guides = node(scene, _make_quad_camera(scene), iteration=0)
    np.testing.assert_allclose(beauty.to_numpy()[4, 4, :3], [2, 1, 0.5], atol=1e-5)
    np.testing.assert_allclose(guides["emission"].to_numpy()[4, 4, :3], [2, 1, 0.5], atol=1e-5)


@pytest.mark.nightly_if_not_vk
@pytest.mark.nightly_if(definition=("Custom",))
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("definition", ["Diffuse", "Custom"])
def test_mdl_area_light_matches_standard_receiver(device: spy.Device, definition: str) -> None:
    images = []
    for mdl in (False, True):
        scene = f2.Scene.create(device)
        receiver = scene.create_material(
            f2.StandardMaterial,
            f2.Properties(
                {
                    "base_color_factor": spy.float3(0.6),
                    "metallic_factor": 0.0,
                    "roughness_factor": 1.0,
                }
            ),
        )
        if mdl:
            emitter = _material(scene, definition, radiance=spy.float3(20, 10, 5))
        else:
            emitter = scene.create_material(
                f2.StandardMaterial,
                f2.Properties(
                    {
                        "base_color_factor": spy.float3(0),
                        "emissive_factor": spy.float3(20, 10, 5),
                    }
                ),
            )
        _add_quad(scene, z=0, normal_z=-1, material=receiver)
        # The emitter is outside the camera frustum, facing the diffuse receiver.
        _add_quad(scene, z=-1, normal_z=1, material=emitter, size=0.4, center_x=1.0)
        camera = _make_quad_camera(scene, width=16, height=16)
        images.append(
            _render_mean(
                device,
                scene,
                camera,
                iterations=64,
                enable_nee=True,
                enable_mis=False,
                max_depth=2,
                enable_homogeneous_media=False,
            )
        )
        # Primary rays do not see the emitter: a one-interaction render is black.
        dark = _render_mean(device, scene, camera, max_depth=1, enable_nee=False)
        assert np.max(dark) == 0
    assert np.isfinite(images).all()
    assert np.mean(images[0]) > 0.01
    # The emitters can choose different light samples; compare regional energy.
    # Allow a few percent of sampling variance at 64 iterations.
    np.testing.assert_allclose(
        _half_image_means(images[1]),
        _half_image_means(images[0]),
        rtol=0.05,
    )


@pytest.mark.nightly_if_not_vk
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_mdl_textured_emission_and_live_resource_updates(
    device: spy.Device, tmp_path: Path
) -> None:
    pixels = np.zeros((4, 8, 3), dtype=np.uint8)
    pixels[:, :4, 0] = 128
    pixels[:, 4:, 1] = 128
    path = tmp_path / "emission.png"
    spy.Bitmap(pixels).write(path)
    scene, _, instance = _scene(device)
    material = _material(scene, "Textured", map=str(path), **{"map:srgb": True})
    instance.materials = [material]
    linear = ((128 / 255 + 0.055) / 1.055) ** 2.4
    first = _probe(scene, instance)
    _assert_emission(first, [linear, 0, 0])
    assert first[1, 3] == 1
    assert first[2, 3] == 100
    module = material.required_module()
    material["uv"] = 1
    _assert_emission(_probe(scene, instance), [0, linear, 0])
    material["uv"] = 0
    material["offset"] = spy.float2(0.5, 0)
    _assert_emission(_probe(scene, instance), [0, linear, 0])
    material["radiance"] = spy.float3(0)
    dark = _probe(scene, instance)
    _assert_emission(dark, [0, 0, 0])
    assert dark[1, 3] == 0
    assert _candidate_triangle_count(scene) == 0
    assert dark[3, 3] == 0
    material["radiance"] = spy.float3(1)
    material["strength"] = 0.0
    assert _probe(scene, instance)[1, 3] == 0
    assert _candidate_triangle_count(scene) == 0
    material["strength"] = 1.0
    material["map"] = ""
    absent = _probe(scene, instance)
    _assert_emission(absent, [0, 0, 0])
    # An invalid resource selects the live fallback before spatial lookup.
    assert absent[1, 3] == 0
    assert _candidate_triangle_count(scene) == 0
    assert absent[3, 3] == 0
    for value in (0.5, 0.0):
        material["fallback"] = spy.float3(value)
        fallback = _probe(scene, instance)
        _assert_emission(fallback, [value] * 3)
        assert fallback[1, 3] == bool(value)
        assert _candidate_triangle_count(scene) == int(bool(value))
    material["map"] = str(path)
    _assert_emission(_probe(scene, instance), [0, linear, 0])
    assert material.required_module() is module


@pytest.mark.nightly_if_not_vk
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_mdl_emission_metadata_survives_regeneration(device: spy.Device) -> None:
    scene, _, instance = _scene(device)
    material = _material(scene, "Scaled", radiance=spy.float3(0), strength=3.0)
    instance.materials = [material]
    assert _probe(scene, instance)[1, 3] == 0
    # AOV consumer changes no longer regenerate materials; changing learnability does.
    material.learnable = True
    assert _probe(scene, instance)[1, 3] == 0
    assert _candidate_triangle_count(scene) == 0
    assert material["strength"] == 3.0
    material["radiance"] = spy.float3(1)
    assert _probe(scene, instance)[1, 3] == 1
    material["strength"] = 0.0
    assert _probe(scene, instance)[1, 3] == 0

    # Definition changes must replace the old predicate.
    material.mdl_material_name = "::emission::VertexColour"
    assert _probe(scene, instance)[1, 3] == 1
    material.mdl_material_name = "::emission::Missing"
    assert _probe(scene, instance)[1, 3] == 0


@pytest.mark.nightly_if_not_vk
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_mdl_emission_typed_controls(device: spy.Device) -> None:
    scene, _, instance = _scene(device)
    material = _material(scene, "TypedControls")
    instance.materials = [material]
    scene.update()
    module = material.required_module()
    for enabled, level, strength in (
        (True, 2, 1.0),
        (True, 3, 2.0),
        (False, 3, 2.0),
        (True, 4, 0.0),
        (True, 4, 0.5),
        (True, 2, 0.5),
    ):
        material["enabled"] = enabled
        material["level"] = level
        material["strength"] = strength
        expected = (level - 2) * strength * strength if enabled else 0.0
        result = _probe(scene, instance)
        assert result[1, 3] == (expected != 0)
        assert _candidate_triangle_count(scene) == int(expected != 0)
        _assert_emission(result, [expected] * 3)
        assert material.required_module() is module


@pytest.mark.nightly_if_not_vk
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_mdl_emission_ignores_unrelated_shading_dependencies(device: spy.Device) -> None:
    scene, _, instance = _scene(device)
    material = _material(scene, "UnrelatedShading")
    instance.materials = [material]
    for value in (-1.0, 0.0, -1.0):
        material["radiance"] = spy.float3(value)
        result = _probe(scene, instance)
        assert result[1, 3] == (value != -1)
        assert _candidate_triangle_count(scene) == int(value != -1)


@pytest.mark.nightly_if_not_vk
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize(
    "definition", ["UvEmission", "PositionEmission", "ObjectEmission", "UniformSceneEmission"]
)
def test_mdl_emission_state_dependencies_remain_candidates(
    device: spy.Device, definition: str
) -> None:
    scene, _, instance = _scene(device)
    instance.materials = [_material(scene, definition)]
    # These can be black at a synthetic zero state. That is not a material-wide proof.
    assert _probe(scene, instance)[1, 3] == 1
    assert _candidate_triangle_count(scene) == 1


@pytest.mark.nightly_if_not_vk
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_mdl_emission_fixed_texture_and_spatial_dark_texels(
    device: spy.Device, tmp_path: Path
) -> None:
    pixels = np.zeros((4, 8, 3), dtype=np.uint8)
    pixels[:, 4:, 1] = 255
    path = tmp_path / "half_emitting.png"
    spy.Bitmap(pixels).write(path)
    scene, _, instance = _scene(device)
    fixed = _material(scene, "FixedTexture")
    instance.materials = [fixed]
    scene.update()
    module = fixed.required_module()
    bright = tmp_path / "all_emitting.png"
    spy.Bitmap(np.full((4, 8, 3), 255, dtype=np.uint8)).write(bright)
    for texture, candidate in (("", False), (str(bright), True), (str(path), False), ("", False)):
        fixed["map"] = texture
        result = _probe(scene, instance)
        assert result[1, 3] == candidate
        assert _candidate_triangle_count(scene) == int(candidate)
        assert fixed.required_module() is module
    spatial = _material(scene, "Textured", map=str(path))
    instance.materials = [spatial]
    result = _probe(scene, instance)
    np.testing.assert_array_equal(result[1, :3], 0)
    assert result[1, 3] == 1
    assert _candidate_triangle_count(scene) == 1
