# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import slangpy as spy

import falcor2 as f2
import falcor2.testing.helpers as helpers
from falcor2.editor import SceneShaderHelper


FLEXIBLE_MR = "GLTFMaterial<8191,131071,131071,131071>"
FLEXIBLE_SPEC_GLOSS = "GLTFSpecGlossMaterial<7,15,15,15>"


def _texture(device: spy.Device) -> spy.Texture:
    return device.create_texture(
        type=spy.TextureType.texture_2d,
        format=spy.Format.rgba32_float,
        width=1,
        height=1,
        usage=spy.TextureUsage.shader_resource,
        data=np.ones((1, 1, 4), dtype=np.float32),
    )


def _compile_masks(type_name: str) -> tuple[int, int, int, int]:
    return tuple(int(value) for value in type_name.partition("<")[2].removesuffix(">").split(","))


def test_gltf_specialization_scene_config_constructor() -> None:
    default = f2.SceneConfig()
    assert default.gltf_specialization_mode == f2.GLTFSpecializationMode.balanced
    assert default.gltf_allowed_features == f2.GLTFFeatureFlags.all
    assert default.gltf_core_features == f2.GLTFFeatureFlags.none
    assert not default.gltf_core_texture_transforms
    assert not default.gltf_core_alternate_uvs
    allowed = f2.GLTFFeatureFlags.all & ~f2.GLTFFeatureFlags.anisotropy
    core = (
        f2.GLTFFeatureFlags.normal | f2.GLTFFeatureFlags.emission | f2.GLTFFeatureFlags.multiscatter
    )
    config = f2.SceneConfig(
        f2.UVOrigin.lower_left, 5, f2.GLTFSpecializationMode.single, allowed, core, True, True
    )
    assert config.uv_origin == f2.UVOrigin.lower_left
    assert config.max_mip_count == 5
    assert config.gltf_specialization_mode == f2.GLTFSpecializationMode.single
    assert config.gltf_allowed_features == allowed
    assert config.gltf_core_features == core
    assert config.gltf_core_texture_transforms
    assert config.gltf_core_alternate_uvs
    with pytest.raises(AttributeError):
        config.gltf_specialization_mode = f2.GLTFSpecializationMode.all
    with pytest.raises(TypeError):
        f2.SceneConfig(specialize_gltf_materials=False)
    with pytest.raises(RuntimeError, match="feature"):
        f2.SceneConfig(gltf_allowed_features=f2.GLTFFeatureFlags(1 << 30))
    with pytest.raises(RuntimeError, match="gltf_core_features"):
        f2.SceneConfig(gltf_core_features=f2.GLTFFeatureFlags(1 << 30))
    with pytest.raises(RuntimeError, match="gltf_core_features"):
        f2.SceneConfig(
            gltf_allowed_features=f2.GLTFFeatureFlags.normal,
            gltf_core_features=f2.GLTFFeatureFlags.emission,
        )


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES[:1])
@pytest.mark.parametrize("material_type", [f2.GLTFMaterial, f2.GLTFSpecGlossMaterial])
def test_gltf_zero_normal_scale_keeps_bound_normal_variant(
    device: spy.Device, material_type: type[f2.Material]
) -> None:
    scene = f2.Scene.create(device)
    material = scene.create_material(material_type)
    material.normal_texture_scale = 0.0
    material.normal_texture = device.create_texture(
        type=spy.TextureType.texture_2d,
        format=spy.Format.rgba32_float,
        width=1,
        height=1,
        usage=spy.TextureUsage.shader_resource,
        data=np.array([[[0.5, 0.5, 0.0, 1.0]]], dtype=np.float32),
    )
    scene.update()

    features, textures, _, _ = _compile_masks(str(material.slang_type_name))
    assert features & (1 << 0)  # NORMAL
    assert textures & (1 << 2)  # Normal texture slot


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES[:1])
def test_gltf_required_variants_are_per_material_and_workflow_specific(
    device: spy.Device,
) -> None:
    scene = f2.Scene.create(
        device, f2.SceneConfig(gltf_specialization_mode=f2.GLTFSpecializationMode.required)
    )
    mr_base = scene.create_material(f2.GLTFMaterial)
    mr_copy = scene.create_material(f2.GLTFMaterial)
    mr_textured = scene.create_material(f2.GLTFMaterial)
    mr_textured.base_color_texture = _texture(device)

    spec_gloss_base = scene.create_material(f2.GLTFSpecGlossMaterial)
    spec_gloss_emissive = scene.create_material(f2.GLTFSpecGlossMaterial)
    spec_gloss_emissive.emissive_factor = spy.float3(0.25)
    scene.update()

    names = [
        str(material.slang_type_name)
        for material in (
            mr_base,
            mr_copy,
            mr_textured,
            spec_gloss_base,
            spec_gloss_emissive,
        )
    ]
    assert names[0] == names[1]
    assert len(set(names)) == 4
    assert names[0] != FLEXIBLE_MR
    assert names[2] != FLEXIBLE_MR
    assert names[3] != FLEXIBLE_SPEC_GLOSS
    assert names[4] != FLEXIBLE_SPEC_GLOSS
    assert names[0].startswith("GLTFMaterial<")
    assert names[3].startswith("GLTFSpecGlossMaterial<")

    material_types = [
        item for item in scene.requirements.type_conformances if item.interface_name == "IMaterial"
    ]
    assert len(material_types) == 5  # InvalidMaterial and four exact keys.


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES[:1])
def test_gltf_all_feature_variants_are_shared_per_workflow(device: spy.Device) -> None:
    scene = f2.Scene.create(
        device, f2.SceneConfig(gltf_specialization_mode=f2.GLTFSpecializationMode.all)
    )
    mr_base = scene.create_material(f2.GLTFMaterial)
    mr_textured = scene.create_material(f2.GLTFMaterial)
    mr_textured.base_color_texture = _texture(device)
    spec_gloss_base = scene.create_material(f2.GLTFSpecGlossMaterial)
    spec_gloss_emissive = scene.create_material(f2.GLTFSpecGlossMaterial)
    spec_gloss_emissive.emissive_factor = spy.float3(0.25)
    scene.update()

    assert mr_base.slang_type_name == mr_textured.slang_type_name == FLEXIBLE_MR
    assert (
        spec_gloss_base.slang_type_name
        == spec_gloss_emissive.slang_type_name
        == FLEXIBLE_SPEC_GLOSS
    )
    material_types = [
        item for item in scene.requirements.type_conformances if item.interface_name == "IMaterial"
    ]
    assert len(material_types) == 3  # InvalidMaterial and one type per glTF workflow.

    before = scene.requirements
    mr_base.metallic_factor = 0.0
    mr_base.feature_mask_filter = int(mr_base.feature_mask_filter) & ~(1 << 3)
    scene.update()
    assert mr_base.slang_type_name == FLEXIBLE_MR
    assert scene.requirements is before

    mr_normal = scene.create_material(f2.GLTFMaterial)
    mr_normal.normal_texture = _texture(device)
    scene.update()
    assert mr_normal.slang_type_name == FLEXIBLE_MR
    assert scene.requirements is before


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES[:1])
@pytest.mark.parametrize(
    "mode",
    [
        f2.GLTFSpecializationMode.required,
        f2.GLTFSpecializationMode.single,
        f2.GLTFSpecializationMode.balanced,
        f2.GLTFSpecializationMode.all,
    ],
)
def test_gltf_core_features_preserve_modes_and_required_uv(
    device: spy.Device, mode: f2.GLTFSpecializationMode
) -> None:
    core = (
        f2.GLTFFeatureFlags.normal | f2.GLTFFeatureFlags.emission | f2.GLTFFeatureFlags.multiscatter
    )
    scene = f2.Scene.create(
        device, f2.SceneConfig(gltf_specialization_mode=mode, gltf_core_features=core)
    )
    mr_base = scene.create_material(f2.GLTFMaterial)
    spec_gloss_base = scene.create_material(f2.GLTFSpecGlossMaterial)
    scene.update()

    assert not scene.required_capabilities.has(f2.SceneCapabilityFlags.geometry_properties)
    if mode == f2.GLTFSpecializationMode.all:
        expected_mr = (8191, 131071, 131071, 131071)
        expected_spec_gloss = (7, 15, 15, 15)
    else:
        expected_mr = (11, 15, 0, 0)
        expected_spec_gloss = (3, 15, 0, 0)
    assert _compile_masks(mr_base.slang_type_name) == expected_mr
    assert _compile_masks(spec_gloss_base.slang_type_name) == expected_spec_gloss

    mr_uv = scene.create_material(f2.GLTFMaterial)
    mr_uv.base_color_texture = _texture(device)
    mr_uv.base_color_texture_offset = spy.float2(0.25)
    mr_uv.base_color_texture_texcoord = 1
    spec_gloss_uv = scene.create_material(f2.GLTFSpecGlossMaterial)
    spec_gloss_uv.diffuse_texture = _texture(device)
    spec_gloss_uv.diffuse_texture_offset = spy.float2(0.25)
    spec_gloss_uv.diffuse_texture_texcoord = 1
    scene.update()

    assert scene.required_capabilities.has(f2.SceneCapabilityFlags.geometry_properties)
    if mode == f2.GLTFSpecializationMode.required:
        assert _compile_masks(mr_base.slang_type_name) == (11, 15, 0, 0)
        assert _compile_masks(spec_gloss_base.slang_type_name) == (3, 15, 0, 0)
        assert _compile_masks(mr_uv.slang_type_name) == (11, 15, 1, 1)
        assert _compile_masks(spec_gloss_uv.slang_type_name) == (3, 15, 1, 1)
    elif mode == f2.GLTFSpecializationMode.all:
        assert (
            _compile_masks(mr_base.slang_type_name)
            == _compile_masks(mr_uv.slang_type_name)
            == expected_mr
        )
        assert (
            _compile_masks(spec_gloss_base.slang_type_name)
            == _compile_masks(spec_gloss_uv.slang_type_name)
            == expected_spec_gloss
        )
    else:
        assert (
            _compile_masks(mr_base.slang_type_name)
            == _compile_masks(mr_uv.slang_type_name)
            == (11, 15, 1, 1)
        )
        assert (
            _compile_masks(spec_gloss_base.slang_type_name)
            == _compile_masks(spec_gloss_uv.slang_type_name)
            == (3, 15, 1, 1)
        )

    helper = SceneShaderHelper(device)
    _, prepared = helper.prepare(scene, "render/test_gltf_material.slang")
    assert prepared.matches_shader_inputs(scene.requirements)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES[:1])
@pytest.mark.parametrize(
    "mode",
    [
        f2.GLTFSpecializationMode.required,
        f2.GLTFSpecializationMode.single,
        f2.GLTFSpecializationMode.balanced,
        f2.GLTFSpecializationMode.all,
    ],
)
def test_gltf_core_can_compile_specular_transforms_and_alternate_uvs(
    device: spy.Device, mode: f2.GLTFSpecializationMode
) -> None:
    core = (
        f2.GLTFFeatureFlags.normal
        | f2.GLTFFeatureFlags.emission
        | f2.GLTFFeatureFlags.multiscatter
        | f2.GLTFFeatureFlags.specular
    )
    scene = f2.Scene.create(
        device,
        f2.SceneConfig(
            gltf_specialization_mode=mode,
            gltf_core_features=core,
            gltf_core_texture_transforms=True,
            gltf_core_alternate_uvs=True,
        ),
    )
    mr = scene.create_material(f2.GLTFMaterial)
    spec_gloss = scene.create_material(f2.GLTFSpecGlossMaterial)
    scene.update()

    if mode == f2.GLTFSpecializationMode.all:
        expected_mr = (8191, 131071, 131071, 131071)
        expected_spec_gloss = (7, 15, 15, 15)
    else:
        expected_mr = (27, 63, 63, 63)
        expected_spec_gloss = (3, 15, 15, 15)
    assert _compile_masks(mr.slang_type_name) == expected_mr
    assert _compile_masks(spec_gloss.slang_type_name) == expected_spec_gloss
    assert not scene.required_capabilities.has(f2.SceneCapabilityFlags.geometry_properties)

    mr.base_color_texture = _texture(device)
    mr.base_color_texture_texcoord = 1
    scene.update()
    assert _compile_masks(mr.slang_type_name) == expected_mr
    assert scene.required_capabilities.has(f2.SceneCapabilityFlags.geometry_properties)

    helper = SceneShaderHelper(device)
    _, prepared = helper.prepare(scene, "render/test_gltf_material.slang")
    assert prepared.matches_shader_inputs(scene.requirements)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES[:1])
def test_gltf_single_unions_workflows_separately(device: spy.Device) -> None:
    scene = f2.Scene.create(
        device, f2.SceneConfig(gltf_specialization_mode=f2.GLTFSpecializationMode.single)
    )
    mr_base = scene.create_material(f2.GLTFMaterial)
    mr_textured = scene.create_material(f2.GLTFMaterial)
    mr_textured.base_color_texture = _texture(device)
    mr_textured.base_color_texture_offset = spy.float2(0.25)
    mr_textured.base_color_texture_texcoord = 1
    spec_gloss_base = scene.create_material(f2.GLTFSpecGlossMaterial)
    spec_gloss_emissive = scene.create_material(f2.GLTFSpecGlossMaterial)
    spec_gloss_emissive.emissive_factor = spy.float3(0.25)
    scene.update()

    assert mr_base.slang_type_name == mr_textured.slang_type_name
    assert _compile_masks(mr_base.slang_type_name)[1:] == (1, 1, 1)
    assert spec_gloss_base.slang_type_name == spec_gloss_emissive.slang_type_name
    assert not _compile_masks(mr_base.slang_type_name)[0] & int(f2.GLTFFeatureFlags.emission)
    assert _compile_masks(spec_gloss_base.slang_type_name)[0] & int(f2.GLTFFeatureFlags.emission)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES[:1])
def test_gltf_single_union_grows_and_shrinks(device: spy.Device) -> None:
    scene = f2.Scene.create(
        device, f2.SceneConfig(gltf_specialization_mode=f2.GLTFSpecializationMode.single)
    )
    base = scene.create_material(f2.GLTFMaterial)
    emissive = scene.create_material(f2.GLTFMaterial)
    scene.update()
    original_type = base.slang_type_name
    original_requirements = scene.requirements

    emissive.emissive_factor = spy.float3(0.25)
    scene.update()
    assert base.slang_type_name == emissive.slang_type_name != original_type
    grown_requirements = scene.requirements
    base.emissive_factor = spy.float3(0.5)
    scene.update()
    assert scene.requirements is grown_requirements

    base.emissive_factor = spy.float3(0)
    emissive.remove()
    scene.update()
    assert base.slang_type_name == original_type
    assert scene.requirements.matches_shader_inputs(original_requirements)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES[:1])
def test_gltf_balanced_groups_materials_by_transmission(device: spy.Device) -> None:
    scene = f2.Scene.create(device)
    assert scene.config.gltf_specialization_mode == f2.GLTFSpecializationMode.balanced
    opaque = scene.create_material(f2.GLTFMaterial)
    emissive = scene.create_material(f2.GLTFMaterial)
    emissive.emissive_factor = spy.float3(0.25)
    transmission = scene.create_material(f2.GLTFMaterial)
    transmission.transmission_factor = 0.5
    diffuse = scene.create_material(f2.GLTFMaterial)
    diffuse.diffuse_transmission_factor = 0.5
    spec_gloss = scene.create_material(f2.GLTFSpecGlossMaterial)
    spec_gloss_emissive = scene.create_material(f2.GLTFSpecGlossMaterial)
    spec_gloss_emissive.emissive_factor = spy.float3(0.25)
    scene.update()

    assert opaque.slang_type_name == emissive.slang_type_name
    assert transmission.slang_type_name == diffuse.slang_type_name != opaque.slang_type_name
    assert spec_gloss.slang_type_name == spec_gloss_emissive.slang_type_name
    opaque_features = _compile_masks(opaque.slang_type_name)[0]
    transmissive_features = _compile_masks(transmission.slang_type_name)[0]
    transmission_bits = int(
        f2.GLTFFeatureFlags.transmission | f2.GLTFFeatureFlags.diffuse_transmission
    )
    assert not opaque_features & transmission_bits
    assert transmissive_features & transmission_bits == transmission_bits

    transmission.feature_mask_filter = int(transmission.feature_mask_filter) & ~int(
        f2.GLTFFeatureFlags.transmission
    )
    scene.update()
    assert transmission.slang_type_name == opaque.slang_type_name
    assert diffuse.slang_type_name != opaque.slang_type_name
    diffuse_type = diffuse.slang_type_name
    diffuse.remove()
    scene.update()
    assert transmission.slang_type_name == opaque.slang_type_name
    assert all(
        item.type_name != diffuse_type
        for item in scene.requirements.type_conformances
        if item.interface_name == "IMaterial"
    )


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES[:1])
def test_gltf_all_with_no_optional_features_keeps_core_textures(device: spy.Device) -> None:
    scene = f2.Scene.create(
        device,
        f2.SceneConfig(
            gltf_specialization_mode=f2.GLTFSpecializationMode.all,
            gltf_allowed_features=f2.GLTFFeatureFlags(0),
        ),
    )
    mr = scene.create_material(f2.GLTFMaterial)
    spec_gloss = scene.create_material(f2.GLTFSpecGlossMaterial)
    scene.update()
    assert _compile_masks(mr.slang_type_name) == (0, 3, 3, 3)
    assert _compile_masks(spec_gloss.slang_type_name) == (0, 3, 3, 3)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES[:1])
def test_gltf_scene_and_material_feature_caps_intersect(device: spy.Device) -> None:
    allowed = (
        f2.GLTFFeatureFlags.all
        & ~f2.GLTFFeatureFlags.normal
        & ~f2.GLTFFeatureFlags.anisotropy
        & ~f2.GLTFFeatureFlags.multiscatter
    )
    scene = f2.Scene.create(device, f2.SceneConfig(gltf_allowed_features=allowed))
    mr = scene.create_material(f2.GLTFMaterial)
    mr.normal_texture = _texture(device)
    mr.anisotropy_strength = 0.5
    mr.emissive_factor = spy.float3(0.25)
    mr.feature_mask_filter = int(mr.feature_mask_filter) & ~int(f2.GLTFFeatureFlags.emission)
    spec_gloss = scene.create_material(f2.GLTFSpecGlossMaterial)
    spec_gloss.normal_texture = _texture(device)
    spec_gloss.emissive_factor = spy.float3(0.25)
    scene.update()

    assert _compile_masks(mr.slang_type_name)[0] == 0
    assert _compile_masks(spec_gloss.slang_type_name)[0] == int(f2.GLTFFeatureFlags.emission)
    assert mr.anisotropy_strength == pytest.approx(0.5)
    np.testing.assert_allclose(mr.emissive_factor, [0.25] * 3)


def _imported_gltf_material(name: str) -> f2.ImporterScene:
    imported = f2.ImporterScene()
    material = f2.ImporterMaterial()
    material.name = name
    material.params = f2.Properties({"_scene_material_type": "GLTFMaterial"})
    imported.materials = [material]
    return imported


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES[:1])
def test_gltf_specialization_config_survives_import_and_append(device: spy.Device) -> None:
    config = f2.SceneConfig(
        gltf_specialization_mode=f2.GLTFSpecializationMode.all,
        gltf_allowed_features=f2.GLTFFeatureFlags(0),
    )
    scene = f2.Scene.from_importer_scene(device, _imported_gltf_material("First"), config=config)
    scene.append(_imported_gltf_material("Second"))
    scene.update()
    assert scene.config.gltf_specialization_mode == config.gltf_specialization_mode
    assert scene.config.gltf_allowed_features == config.gltf_allowed_features
    for name in ("First", "Second"):
        material = scene.materials.find(name)
        assert isinstance(material, f2.GLTFMaterial)
        assert _compile_masks(material.slang_type_name) == (0, 3, 3, 3)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES[:1])
def test_gltf_required_variant_key_tracks_feature_texture_transform_and_uv(
    device: spy.Device,
) -> None:
    scene = f2.Scene.create(
        device, f2.SceneConfig(gltf_specialization_mode=f2.GLTFSpecializationMode.required)
    )
    material = scene.create_material(f2.GLTFMaterial)
    scene.update()
    previous_name = material.slang_type_name
    previous_requirements = scene.requirements

    def check_changed() -> None:
        nonlocal previous_name, previous_requirements
        scene.update()
        assert material.slang_type_name != previous_name
        assert scene.requirements is not previous_requirements
        assert not scene.requirements.matches_shader_inputs(previous_requirements)
        previous_name = material.slang_type_name
        previous_requirements = scene.requirements

    material.metallic_factor = 0.0
    check_changed()
    material.base_color_texture = _texture(device)
    check_changed()
    material.base_color_texture_offset = spy.float2(0.125, 0.25)
    check_changed()
    material.base_color_texture_texcoord = 1
    check_changed()

    helper = SceneShaderHelper(device)
    _, prepared = helper.prepare(scene, "render/test_gltf_material.slang")
    assert prepared.matches_shader_inputs(scene.requirements)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES[:1])
def test_gltf_exact_feature_edits_update_requirements(device: spy.Device) -> None:
    scene = f2.Scene.create(
        device, f2.SceneConfig(gltf_specialization_mode=f2.GLTFSpecializationMode.required)
    )
    material = scene.create_material(f2.GLTFSpecGlossMaterial)
    scene.update()
    initial_name = material.slang_type_name
    initial_requirements = scene.requirements

    material.emissive_factor = spy.float3(0.25)
    scene.update()
    assert material.slang_type_name != initial_name
    assert not scene.requirements.matches_shader_inputs(initial_requirements)
    emissive_requirements = scene.requirements

    material.emissive_factor = spy.float3(0.0)
    scene.update()
    assert material.slang_type_name == initial_name
    assert not scene.requirements.matches_shader_inputs(emissive_requirements)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES[:1])
def test_gltf_feature_filter_changes_exact_key(device: spy.Device) -> None:
    scene = f2.Scene.create(
        device, f2.SceneConfig(gltf_specialization_mode=f2.GLTFSpecializationMode.required)
    )
    material = scene.create_material(f2.GLTFMaterial)
    scene.update()
    initial = material.slang_type_name

    material.feature_mask_filter = int(material.feature_mask_filter) & ~(1 << 3)
    scene.update()
    assert material.slang_type_name != initial
    assert material.slang_type_name.startswith("GLTFMaterial<0,")

    spec_gloss = scene.create_material(f2.GLTFSpecGlossMaterial)
    spec_gloss.emissive_factor = spy.float3(0.2)
    scene.update()
    initial = spec_gloss.slang_type_name

    spec_gloss.feature_mask_filter = int(spec_gloss.feature_mask_filter) & ~(1 << 1)
    scene.update()
    assert spec_gloss.slang_type_name != initial
    assert spec_gloss.slang_type_name.startswith("GLTFSpecGlossMaterial<0,")
