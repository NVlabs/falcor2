# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import gc
import sys

import numpy as np
import pytest
import slangpy as spy
import falcor2 as f2
from falcor2.editor import SceneShaderHelper
import falcor2.testing.helpers as helpers

DATA = Path(__file__).parent.parent.parent.parent / "data"


@pytest.fixture
def device(device_type: spy.DeviceType) -> spy.Device:
    return helpers.get_device(device_type)


@pytest.fixture
def scene(device: spy.Device) -> f2.Scene:
    return f2.Scene.create(device)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_scene_object_predicates(scene: f2.Scene) -> None:
    for obj in (scene.free_camera, scene.free_camera.entity):
        assert obj.is_valid() is True
        assert obj.is_builtin() is True

    entity = scene.create_entity()
    camera = entity.create_component(f2.Camera)
    for obj in (entity, camera):
        assert obj.is_valid() is True
        assert obj.is_builtin() is False

    entity.remove()
    for obj in (entity, camera):
        assert obj.is_valid() is False
        assert obj.is_builtin() is False


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_scene_default_capabilities(device: spy.Device) -> None:
    scene = f2.Scene.create(device)
    assert scene.requirements.capabilities == f2.SceneCapabilities()
    scene.update()
    assert scene.requirements == scene.resolve_requirements()
    assert scene.required_capabilities == f2.SceneCapabilities()
    assert (
        scene.resolve_requirements(policy=f2.SceneCapabilityPolicy.all).capabilities
        == f2.SceneCapabilities.all()
    )
    with pytest.raises(AttributeError):
        scene.requirements.capabilities = f2.SceneCapabilities()


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_capability_resolution(device: spy.Device) -> None:
    scene = f2.Scene.create(device)
    material = scene.create_material(f2.StandardMaterial)
    scene.update()
    empty = f2.SceneCapabilities()
    geometry = f2.SceneCapabilities(f2.SceneCapabilityFlags.geometry_properties)
    opacity = f2.SceneCapabilities(f2.SceneCapabilityFlags.opacity_evaluation)
    general = scene.resolve_requirements(policy=f2.SceneCapabilityPolicy.all)
    before = scene.resolve_requirements(policy=f2.SceneCapabilityPolicy.required)
    assert before.capabilities == empty
    assert scene.requirements is before
    generation = scene.requirements_generation
    with pytest.raises(RuntimeError, match="requires specified_capabilities"):
        scene.resolve_requirements(policy=f2.SceneCapabilityPolicy.specified)

    material.base_color_texture_texcoord = 1
    assert scene.resolve_requirements(policy=f2.SceneCapabilityPolicy.required) == before
    assert scene.update() & f2.SceneUpdateFlags.requirements
    assert scene.requirements_generation > generation
    assert scene.resolve_requirements(policy=f2.SceneCapabilityPolicy.all) is general
    required = scene.resolve_requirements(policy=f2.SceneCapabilityPolicy.required)
    assert required.capabilities == geometry
    assert scene.requirements is required
    assert before.capabilities == empty
    with pytest.raises(RuntimeError, match="omits required capabilities"):
        scene.resolve_requirements(
            policy=f2.SceneCapabilityPolicy.specified, specified_capabilities=empty
        )
    assert (
        scene.resolve_requirements(
            policy=f2.SceneCapabilityPolicy.specified, specified_capabilities=geometry
        )
        == required
    )
    assert scene.resolve_requirements(
        f2.SceneRenderRequest(required_capabilities=opacity),
        policy=f2.SceneCapabilityPolicy.required,
    ).capabilities == geometry.merged(opacity)
    with pytest.raises(RuntimeError, match="omits required capabilities"):
        scene.resolve_requirements(
            f2.SceneRenderRequest(required_capabilities=opacity),
            f2.SceneCapabilityPolicy.specified,
            geometry,
        )
    unknown = f2.SceneCapabilities(f2.SceneCapabilityFlags(1 << 16))
    with pytest.raises(RuntimeError, match="unsupported capabilities"):
        scene.resolve_requirements(
            policy=f2.SceneCapabilityPolicy.specified, specified_capabilities=unknown
        )
    for policy in (
        f2.SceneCapabilityPolicy.all,
        f2.SceneCapabilityPolicy.required,
        f2.SceneCapabilityPolicy.specified,
    ):
        with pytest.raises(RuntimeError, match="unsupported capabilities"):
            scene.resolve_requirements(
                f2.SceneRenderRequest(required_capabilities=unknown), policy=policy
            )

    material.remove()
    scene.update()
    assert (
        scene.resolve_requirements(policy=f2.SceneCapabilityPolicy.required).capabilities == empty
    )
    assert scene.requirements.capabilities == empty
    assert required.capabilities == geometry


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_solid_transmission_scene_setting(device: spy.Device) -> None:
    scene = f2.Scene.create(device)
    material = scene.create_material(f2.StandardMaterial)
    scene.update()

    def enabled(requirements: f2.SceneRequirements) -> bool:
        module = spy.Module.load_from_source(
            device,
            "solid_transmission_scene_setting_probe",
            "import falcor2.render; uint enabled() { return ENABLE_SOLID_TRANSMISSION ? 1u : 0u; }",
            link=requirements.modules,
        )
        return bool(module.enabled())

    assert not enabled(scene.requirements)
    assert enabled(scene.resolve_requirements(policy=f2.SceneCapabilityPolicy.all))

    material.specular_transmission_factor = 1.0
    scene.update()
    assert enabled(scene.requirements)


def _set_quad(geometry: f2.StaticMeshGeometry) -> None:
    geometry.set_mesh_data(
        positions=np.array([[-2, -2, 0], [2, -2, 0], [-2, 2, 0], [2, 2, 0]], dtype=np.float32),
        sub_mesh_indices=[np.array([[0, 2, 1], [2, 3, 1]], dtype=np.uint32)],
        normals=np.tile(np.array([0, 0, -1], dtype=np.float32), (4, 1)),
    )


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_consumer_geometry_queries(device: spy.Device) -> None:
    scene = f2.Scene.create(device)
    geometry = scene.create_geometry(f2.StaticMeshGeometry)
    _set_quad(geometry)
    entity = scene.create_entity()
    transform = f2.Transform()
    transform.translation = spy.float3(3, -2, 5)
    entity.transform = transform
    instance = entity.create_component(f2.GeometryInstance)
    instance.geometry = geometry
    instance.materials = [scene.create_material(f2.StandardMaterial)]
    scene.update()
    empty = f2.SceneCapabilities()
    attributes = f2.SceneCapabilities(f2.SceneCapabilityFlags.geometry_properties)
    assert scene.required_capabilities == empty
    generation = scene.requirements_generation
    helper = SceneShaderHelper(device)
    source = spy.Module.load_from_file(device, "render/test_scene.slang")
    geometry_ids = spy.Tensor.from_numpy(
        device, np.array([int(instance.geometry_instance_id)], dtype=np.uint32)
    )
    modules = []
    for consumer, expected in ((empty, [0, 0, 0]), (attributes, [3, -2, 5])):
        module, requirements = helper.prepare(
            scene,
            source,
            policy=f2.SceneCapabilityPolicy.required,
            required_capabilities=consumer,
        )
        repeated, repeated_requirements = helper.prepare(
            scene,
            source,
            policy=f2.SceneCapabilityPolicy.required,
            required_capabilities=consumer,
        )
        assert repeated is module
        assert repeated_requirements is requirements
        modules.append(module)
        probe = (
            module.object_origin.as_func()
            .type_conformances(requirements.type_conformances)
            .write(helper.bind_scene)
            .return_type(spy.Tensor)
        )
        np.testing.assert_allclose(probe(geometry_ids).to_numpy()[0], expected)
    assert modules[0] is not modules[1]
    assert scene.required_capabilities == empty
    assert scene.requirements_generation == generation
    # The default selection follows scene and consumer requirements.
    general, _ = helper.prepare(scene, source)
    assert general is not modules[1]
    assert scene.requirements.capabilities == empty
    with pytest.raises(RuntimeError, match="requires specified_capabilities"):
        helper.prepare(scene, source, policy=f2.SceneCapabilityPolicy.specified)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_acceleration_structure_opacity_follows_materials(device: spy.Device) -> None:
    if not device.has_feature(spy.Feature.ray_query):
        pytest.skip("Ray queries are not supported by this device")
    scene = f2.Scene.create(device)
    material = scene.create_material(f2.StandardMaterial)
    geometry = scene.create_geometry(f2.StaticMeshGeometry)
    _set_quad(geometry)
    instance = scene.create_entity().create_component(f2.GeometryInstance)
    instance.geometry = geometry
    instance.materials = [material]
    helper = SceneShaderHelper(device)
    source = spy.Module.load_from_file(device, "render/test_scene.slang")
    for mode, expected in (
        (f2.AlphaMode.opaque, 0),
        (f2.AlphaMode.mask, 1),
        (f2.AlphaMode.opaque, 0),
    ):
        material.alpha_mode = mode
        scene.update()
        assert scene.requirements.capabilities == scene.required_capabilities
        module, _ = helper.prepare(scene, source)
        count = module.nonopaque_triangle_count.as_func().write(helper.bind_scene)()
        assert count == expected


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("geometry_type", [f2.StaticMeshGeometry, f2.StaticCurveGeometry])
def test_unused_streams_do_not_change_capability_selection(
    device: spy.Device, geometry_type: type
) -> None:
    scene = f2.Scene.create(device)

    def set_geometry(geometry: f2.Geometry, with_attributes: bool) -> None:
        positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        # Ordinary UV0 is packed into the built-in vertex data and must not enable extra attributes.
        attributes = [
            f2.MeshVertexAttribute(
                f2.MeshAttributeSemantic.tex_coord, 0, np.zeros((3, 2), dtype=np.float32)
            )
        ]
        if with_attributes:
            attributes.append(
                f2.MeshVertexAttribute(
                    f2.MeshAttributeSemantic.tex_coord, 1, np.ones((3, 2), dtype=np.float32)
                )
            )
        if geometry_type == f2.StaticMeshGeometry:
            geometry.set_mesh_data(
                positions=positions,
                sub_mesh_indices=[np.array([[0, 1, 2]], dtype=np.uint32)],
                vertex_attributes=attributes,
            )
        else:
            geometry.set_curve_data(
                positions,
                np.full(3, 0.1, dtype=np.float32),
                np.array([0, 1], dtype=np.uint32),
                f2.CurveIndexingMode.successive,
                vertex_attributes=attributes,
            )

    # Keep this geometry alive so removing the others does not change the geometry-type requirements.
    basic = scene.create_geometry(geometry_type)
    first = scene.create_geometry(geometry_type)
    for geometry in (basic, first):
        set_geometry(geometry, False)
    scene.update()
    generation = scene.requirements_generation

    def check() -> None:
        scene.update()
        assert scene.required_capabilities == f2.SceneCapabilities()
        assert scene.requirements.capabilities == scene.required_capabilities
        required = scene.resolve_requirements(policy=f2.SceneCapabilityPolicy.required)
        assert required.capabilities == f2.SceneCapabilities()
        assert scene.requirements_generation == generation
        module = spy.Module.load_from_source(
            device,
            "unused_stream_capability_probe",
            "import falcor2.render; uint enabled() { return ENABLE_GEOMETRY_PROPERTIES ? 1u : 0u; }",
            link=required.modules,
        )
        assert not module.enabled()

    check()
    set_geometry(first, True)
    check()
    second = scene.create_geometry(geometry_type)
    set_geometry(second, True)
    check()
    set_geometry(first, False)
    check()
    second.remove()
    check()
    set_geometry(first, True)
    check()
    first.remove()
    check()


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_imported_scene_capabilities(device: spy.Device, tmp_path: Path) -> None:
    path = tmp_path / "empty.gltf"
    path.write_text('{"asset": {"version": "2.0"}}', encoding="utf-8")
    scene = f2.Scene.load(device, path)
    scene.update()
    assert scene.requirements.capabilities == scene.required_capabilities
    assert not scene.resolve_requirements(
        policy=f2.SceneCapabilityPolicy.required
    ).capabilities.has(f2.SceneCapabilityFlags.geometry_properties)

    scene = f2.Scene.load(device, DATA / "assets/kronos/MultiUVTest/glTF/MultiUVTest.gltf")
    scene.update()
    assert scene.required_capabilities.has(f2.SceneCapabilityFlags.geometry_properties)
    assert scene.resolve_requirements(policy=f2.SceneCapabilityPolicy.required).capabilities.has(
        f2.SceneCapabilityFlags.geometry_properties
    )


@pytest.mark.nightly_if_not_vk
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_scene_uv_origin_api(device_type: spy.DeviceType):
    device = helpers.get_device(device_type)
    scene = f2.Scene.create(device)
    assert scene.config.uv_origin == f2.UVOrigin.upper_left
    assert f2.SceneConfig().uv_origin == f2.UVOrigin.upper_left

    lower_left_config = f2.SceneConfig(f2.UVOrigin.lower_left)
    lower_left_scene = f2.Scene.create(device, lower_left_config)
    assert lower_left_scene.config.uv_origin == f2.UVOrigin.lower_left
    with pytest.raises(AttributeError):
        lower_left_scene.config.uv_origin = f2.UVOrigin.upper_left
    texture_manager = f2.TextureManager(device)
    point_sampler = device.create_sampler(
        min_filter=spy.TextureFilteringMode.point,
        mag_filter=spy.TextureFilteringMode.point,
    )
    red_top_green_bottom_texture = device.create_texture(
        type=spy.TextureType.texture_2d,
        format=spy.Format.rgba32_float,
        width=1,
        height=2,
        usage=spy.TextureUsage.shader_resource,
        data=np.array([[1, 0, 0, 1], [0, 1, 0, 1]], dtype=np.float32),
    )
    red_top_green_bottom_handle = texture_manager.register_texture(
        red_top_green_bottom_texture, point_sampler
    )
    texture_manager.update()

    def sample_texture(scene: f2.Scene) -> spy.float4:
        scene.update()
        module = spy.Module.load_from_file(
            device, "render/test_texture_manager.slang", link=scene.requirements.modules
        )
        return module.test_2d(
            handle={"data": red_top_green_bottom_handle.data},
            uv=spy.float2(0.5, 0.25),
            default_value=spy.float4(1, 0, 1, 1),
        )

    assert sample_texture(scene) == spy.float4(1, 0, 0, 1)
    assert sample_texture(lower_left_scene) == spy.float4(0, 1, 0, 1)

    path = DATA / "assets/cornell-box/usdpreviewsurface/cornell-box.usda"
    importer_scene = f2.UsdImporter().load_scene(path)
    assert importer_scene.uv_origin == f2.UVOrigin.lower_left

    loaded_scenes = [
        f2.Scene.from_importer_scene(device, importer_scene),
        f2.Scene.load(device, path),
    ]
    for loaded_scene in loaded_scenes:
        assert loaded_scene.config.uv_origin == f2.UVOrigin.lower_left
        assert sample_texture(loaded_scene) == spy.float4(0, 1, 0, 1)

    upper_left_config = f2.SceneConfig(f2.UVOrigin.upper_left)
    override_scenes = [
        f2.Scene.from_importer_scene(device, importer_scene, config=upper_left_config),
        f2.Scene.load(device, path, config=upper_left_config),
    ]
    for override_scene in override_scenes:
        assert override_scene.config.uv_origin == f2.UVOrigin.upper_left
        assert sample_texture(override_scene) == spy.float4(1, 0, 0, 1)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES[:1])
def test_scene_material_aov_registry_api(device_type: spy.DeviceType) -> None:
    device = helpers.get_device(device_type)
    scene = f2.Scene.create(device)
    scene_refcount = sys.getrefcount(scene)
    registry = scene.material_aov_registry
    assert registry.outputs == []
    assert registry.find_output("dust_mask_output") is None
    assert registry.output_generation == 0
    output_id = registry.register_output("dust_mask_output", spy.DataType.float32, 1)
    assert output_id == 0
    assert registry.register_output("dust_mask_output", spy.DataType.float32, 1) == output_id
    with pytest.raises(RuntimeError):
        registry.register_output("dust_mask_output", spy.DataType.float32, 2)
    with pytest.raises(RuntimeError):
        registry.register_output("", spy.DataType.float32, 1)
    assert registry.output_generation == 1
    snapshot = registry.find_output("dust_mask_output")
    assert isinstance(snapshot, f2.AOVDesc)
    assert snapshot.id == output_id
    assert snapshot.type == spy.DataType.float32
    assert isinstance(registry.outputs[0], f2.AOVDesc)
    registry.register_output("vector", spy.DataType.float32, 3)
    # Neither the subsystem binding nor copied metadata pin the Scene Python owner.
    assert sys.getrefcount(scene) == scene_refcount
    del scene
    assert registry.has_output("dust_mask_output")
    del registry
    assert snapshot.name == "dust_mask_output"
    assert snapshot.component_count == 1


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_scene_opacity_requirement_transitions(device_type: spy.DeviceType) -> None:
    device = helpers.get_device(device_type)
    scene = f2.Scene.create(device)
    first = scene.create_material(f2.StandardMaterial)
    second = scene.create_material(f2.StandardMaterial)

    scene.update()
    assert not scene.required_capabilities.has(f2.SceneCapabilityFlags.opacity_evaluation)
    initial_generation = scene.requirements_generation

    first.alpha_mode = f2.AlphaMode.mask
    update_flags = scene.update()
    assert scene.required_capabilities.has(f2.SceneCapabilityFlags.opacity_evaluation)
    assert update_flags & f2.SceneUpdateFlags.requirements
    assert scene.requirements_generation > initial_generation
    masked_generation = scene.requirements_generation

    second.alpha_mode = f2.AlphaMode.mask
    update_flags = scene.update()
    assert scene.required_capabilities.has(f2.SceneCapabilityFlags.opacity_evaluation)
    assert not (update_flags & f2.SceneUpdateFlags.requirements)
    assert scene.requirements_generation == masked_generation

    first.alpha_mode = f2.AlphaMode.opaque
    update_flags = scene.update()
    assert scene.required_capabilities.has(f2.SceneCapabilityFlags.opacity_evaluation)
    assert not (update_flags & f2.SceneUpdateFlags.requirements)
    assert scene.requirements_generation == masked_generation

    second.remove()
    update_flags = scene.update()
    assert not scene.required_capabilities.has(f2.SceneCapabilityFlags.opacity_evaluation)
    assert update_flags & f2.SceneUpdateFlags.requirements
    assert scene.requirements_generation > masked_generation
    opaque_generation = scene.requirements_generation

    first.alpha_mode = f2.AlphaMode.blend
    update_flags = scene.update()
    assert scene.required_capabilities.has(f2.SceneCapabilityFlags.opacity_evaluation)
    assert update_flags & f2.SceneUpdateFlags.requirements
    assert scene.requirements_generation > opaque_generation
    blend_generation = scene.requirements_generation

    first.alpha_mode = f2.AlphaMode.opaque
    update_flags = scene.update()
    assert not scene.required_capabilities.has(f2.SceneCapabilityFlags.opacity_evaluation)
    assert update_flags & f2.SceneUpdateFlags.requirements
    assert scene.requirements_generation > blend_generation


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
class TestSceneCollections:
    """Tests for scene collection views and iteration."""

    @pytest.mark.nightly_if_not_vk
    def test_empty_collections(self, scene: f2.Scene):
        assert len(scene.materials) == 0
        assert len(scene.geometries) == 0
        assert len(scene.entities) == 1
        assert len(scene.components) == 1

    @pytest.mark.nightly_if_not_vk
    def test_empty_iteration(self, scene: f2.Scene):
        assert list(scene.materials) == []
        assert list(scene.geometries) == []
        assert list(scene.entities) == [scene.free_camera.entity]
        assert list(scene.components) == [scene.free_camera]

    @pytest.mark.nightly_if_not_vk
    def test_materials_collection(self, scene: f2.Scene):
        m0 = scene.create_material("StandardMaterial")
        m1 = scene.create_material("StandardMaterial")

        assert len(scene.materials) == 2
        assert scene.materials[0] is m0
        assert scene.materials[1] is m1

    @pytest.mark.nightly_if_not_vk
    def test_materials_iteration(self, scene: f2.Scene):
        m0 = scene.create_material("StandardMaterial")
        m1 = scene.create_material("StandardMaterial")

        iterated = list(scene.materials)
        assert len(iterated) == 2
        assert iterated[0] is m0
        assert iterated[1] is m1

    @pytest.mark.nightly_if_not_vk
    def test_material_id(self, scene: f2.Scene):
        material = scene.create_material("StandardMaterial")

        assert isinstance(material.material_id, int)
        assert material.material_id == 0

        scene.update()

        assert isinstance(material.material_id, int)
        assert material.material_id > 0

    @pytest.mark.nightly_if_not_vk
    def test_entities_collection(self, scene: f2.Scene):
        e0 = scene.create_entity()
        e1 = scene.create_entity()
        e2 = scene.create_entity()

        assert len(scene.entities) == 4
        assert scene.entities[1] is e0
        assert scene.entities[2] is e1
        assert scene.entities[3] is e2

    @pytest.mark.nightly_if_not_vk
    def test_entities_iteration(self, scene: f2.Scene):
        e0 = scene.create_entity()
        e1 = scene.create_entity()

        iterated = list(scene.entities)
        assert iterated == [scene.free_camera.entity, e0, e1]

    @pytest.mark.nightly_if_not_vk
    def test_components_via_entity(self, scene: f2.Scene):
        entity = scene.create_entity()
        camera = entity.create_component(f2.Camera)

        assert len(scene.components) == 2
        assert scene.components[1] is camera

        iterated = list(scene.components)
        assert iterated == [scene.free_camera, camera]

    @pytest.mark.nightly_if_not_vk
    def test_negative_indexing(self, scene: f2.Scene):
        m0 = scene.create_material("StandardMaterial")
        m1 = scene.create_material("StandardMaterial")
        m2 = scene.create_material("StandardMaterial")

        assert scene.materials[-1] is m2
        assert scene.materials[-2] is m1
        assert scene.materials[-3] is m0

    @pytest.mark.nightly_if_not_vk
    def test_index_out_of_range(self, scene: f2.Scene):
        scene.create_material("StandardMaterial")

        with pytest.raises(IndexError):
            scene.materials[1]
        with pytest.raises(IndexError):
            scene.materials[-2]

    @pytest.mark.nightly_if_not_vk
    def test_for_loop(self, scene: f2.Scene):
        """Test that standard for-loop patterns work."""
        materials = [scene.create_material("StandardMaterial") for _ in range(3)]

        count = 0
        for mat in scene.materials:
            assert mat is materials[count]
            count += 1
        assert count == 3


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
class TestCollectionFind:
    """Tests for find/find_all on collection views."""

    @pytest.mark.nightly_if_not_vk
    def test_find_by_name(self, scene: f2.Scene):
        m0 = scene.create_material("StandardMaterial")
        m0.name = "wood"
        m1 = scene.create_material("StandardMaterial")
        m1.name = "metal"
        m2 = scene.create_material("StandardMaterial")
        m2.name = "wood"

        assert scene.materials.find("wood") is m0
        assert scene.materials.find("metal") is m1
        assert scene.materials.find("missing") is None

    @pytest.mark.nightly_if_not_vk
    def test_find_by_type(self, scene: f2.Scene):
        entity = scene.create_entity()
        camera = entity.create_component(f2.Camera)
        light = entity.create_component(f2.ConstantLight)

        assert scene.components.find(type=f2.Camera) is camera
        assert scene.components.find(type=f2.ConstantLight) is light

    @pytest.mark.nightly_if_not_vk
    def test_find_by_name_and_type(self, scene: f2.Scene):
        e0 = scene.create_entity()
        camera = e0.create_component(f2.Camera)
        camera.name = "main"
        light = e0.create_component(f2.ConstantLight)
        light.name = "main"

        assert scene.components.find("main", type=f2.Camera) is camera
        assert scene.components.find("main", type=f2.ConstantLight) is light
        assert scene.components.find("main", type=f2.PointLight) is None

    @pytest.mark.nightly_if_not_vk
    def test_find_no_args_returns_first(self, scene: f2.Scene):
        m0 = scene.create_material("StandardMaterial")
        m1 = scene.create_material("StandardMaterial")

        assert scene.materials.find() is m0

    @pytest.mark.nightly_if_not_vk
    def test_find_on_empty(self, scene: f2.Scene):
        assert scene.materials.find("anything") is None
        assert scene.materials.find() is None

    @pytest.mark.nightly_if_not_vk
    def test_find_all_by_name(self, scene: f2.Scene):
        m0 = scene.create_material("StandardMaterial")
        m0.name = "wood"
        m1 = scene.create_material("StandardMaterial")
        m1.name = "metal"
        m2 = scene.create_material("StandardMaterial")
        m2.name = "wood"

        result = scene.materials.find_all("wood")
        assert len(result) == 2
        assert result[0] is m0
        assert result[1] is m2

        result = scene.materials.find_all("metal")
        assert len(result) == 1
        assert result[0] is m1

        assert scene.materials.find_all("missing") == []

    @pytest.mark.nightly_if_not_vk
    def test_find_all_by_type(self, scene: f2.Scene):
        entity = scene.create_entity()
        camera = entity.create_component(f2.Camera)
        light = entity.create_component(f2.ConstantLight)
        light2 = entity.create_component(f2.ConstantLight)

        cameras = scene.components.find_all(type=f2.Camera)
        assert cameras == [camera]

        lights = scene.components.find_all(type=f2.ConstantLight)
        assert len(lights) == 2
        assert lights[0] is light
        assert lights[1] is light2

    @pytest.mark.nightly_if_not_vk
    def test_find_all_by_name_and_type(self, scene: f2.Scene):
        entity = scene.create_entity()
        cam = entity.create_component(f2.Camera)
        cam.name = "main"
        light = entity.create_component(f2.ConstantLight)
        light.name = "main"
        light2 = entity.create_component(f2.ConstantLight)
        light2.name = "fill"

        result = scene.components.find_all("main", type=f2.ConstantLight)
        assert len(result) == 1
        assert result[0] is light

    @pytest.mark.nightly_if_not_vk
    def test_find_all_no_args(self, scene: f2.Scene):
        m0 = scene.create_material("StandardMaterial")
        m1 = scene.create_material("StandardMaterial")

        result = scene.materials.find_all()
        assert len(result) == 2
        assert result[0] is m0
        assert result[1] is m1

    @pytest.mark.nightly_if_not_vk
    def test_find_all_on_empty(self, scene: f2.Scene):
        assert scene.materials.find_all("anything") == []
        assert scene.materials.find_all() == []

    def test_builtin_filtering(self, scene: f2.Scene) -> None:
        for collection, builtin in (
            (scene.entities, scene.free_camera.entity),
            (scene.components, scene.free_camera),
        ):
            assert collection.find() is None
            assert collection.find_all() == []
            assert collection.find(builtin.name) is None
            assert collection.find(include_builtins=True) is builtin
            assert collection.find_all(include_builtins=True) == [builtin]
            assert collection.find(builtin.name, include_builtins=True) is builtin
            assert list(collection.iter()) == []
            assert list(collection.iter(include_builtins=True)) == [builtin]

        first = scene.create_entity().create_component(f2.Camera)
        named = scene.create_entity().create_component(f2.Camera)
        named.name = scene.free_camera.name
        for kwargs in ({}, {"type": f2.Camera}):
            assert scene.components.find(**kwargs) is first
            assert scene.components.find_all(**kwargs) == [first, named]
            assert scene.components.find_all(include_builtins=True, **kwargs) == [
                scene.free_camera,
                first,
                named,
            ]
            assert scene.components.find("free_camera", **kwargs) is named
            assert scene.components.find_all("free_camera", **kwargs) == [named]
            assert (
                scene.components.find("free_camera", include_builtins=True, **kwargs)
                is scene.free_camera
            )
            assert scene.components.find_all("free_camera", include_builtins=True, **kwargs) == [
                scene.free_camera,
                named,
            ]

    def test_filtered_iteration_preserves_raw_access(self, scene: f2.Scene) -> None:
        first = scene.create_entity()
        second = scene.create_entity()
        assert list(scene.materials.iter()) == []
        assert list(scene.entities.iter()) == [first, second]
        raw = [scene.free_camera.entity, first, second]
        assert list(scene.entities.iter(include_builtins=True)) == raw
        assert list(scene.entities) == raw
        assert len(scene.entities) == len(raw)
        assert scene.entities[-1] is second
        for entity in raw:
            assert scene.entities[entity.collection_index] is entity

    def test_filtered_iterator_retains_owner(self, device: spy.Device) -> None:
        def make_iterator(include_builtins: bool):
            local_scene = f2.Scene.create(device)
            local_scene.create_entity().name = "authored"
            return local_scene.entities.iter(include_builtins=include_builtins)

        filtered = make_iterator(False)
        inclusive = make_iterator(True)
        gc.collect()
        assert [entity.name for entity in filtered] == ["authored"]
        assert [entity.name for entity in inclusive] == ["free_camera", "authored"]

    def test_filtered_queries_match_python_subclasses(self, scene: f2.Scene) -> None:
        class QueryMaterial(f2.Material):
            pass

        regular = scene.create_material("StandardMaterial")
        custom = scene.create_material(QueryMaterial)
        regular.name = custom.name = "shared"
        for include_builtins in (False, True):
            assert (
                scene.materials.find(
                    "shared", type=QueryMaterial, include_builtins=include_builtins
                )
                is custom
            )
            assert scene.materials.find_all(
                type=QueryMaterial, include_builtins=include_builtins
            ) == [custom]
            assert list(scene.materials.iter(include_builtins=include_builtins)) == [
                regular,
                custom,
            ]


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
class TestSceneObjectFactory:
    """Tests for creating scene objects by type name and by class."""

    @pytest.mark.nightly_if_not_vk
    def test_create_material_by_type_name(self, scene: f2.Scene):
        mat = scene.create_material("StandardMaterial")
        assert mat is not None
        assert len(scene.materials) == 1
        assert scene.materials[0] is mat

    @pytest.mark.nightly_if_not_vk
    def test_create_geometry_by_type_name(self, scene: f2.Scene):
        geo = scene.create_geometry("StaticMeshGeometry")
        assert geo is not None
        assert isinstance(geo, f2.StaticMeshGeometry)
        assert len(scene.geometries) == 1

    @pytest.mark.nightly_if_not_vk
    def test_create_geometry_by_class(self, scene: f2.Scene):
        geo = scene.create_geometry(f2.StaticMeshGeometry)
        assert geo is not None
        assert isinstance(geo, f2.StaticMeshGeometry)

    @pytest.mark.nightly_if_not_vk
    def test_create_entity(self, scene: f2.Scene):
        entity = scene.create_entity()
        assert entity is not None
        assert len(scene.entities) == 2

    @pytest.mark.nightly_if_not_vk
    def test_create_component_by_class(self, scene: f2.Scene):
        entity = scene.create_entity()
        camera = entity.create_component(f2.Camera)
        assert camera is not None
        assert isinstance(camera, f2.Camera)
        assert len(scene.components) == 2

    @pytest.mark.nightly_if_not_vk
    def test_create_component_by_type_name(self, scene: f2.Scene):
        entity = scene.create_entity()
        camera = entity.create_component("Camera")
        assert camera is not None
        assert isinstance(camera, f2.Camera)

    def test_builtin_entity_rejects_component_creation(self, scene: f2.Scene) -> None:
        constructed = []

        class CustomComponent(f2.Component):
            def __init__(self) -> None:
                super().__init__()
                constructed.append(self)

        entity = scene.free_camera.entity
        for component_type in ("Camera", f2.Camera, CustomComponent):
            with pytest.raises(RuntimeError, match="Cannot add components to a built-in object"):
                entity.create_component(component_type)
            assert list(scene.components) == [scene.free_camera]
            assert list(entity.components) == [scene.free_camera]
            assert list(scene.entities) == [entity]
        assert constructed == []

    @pytest.mark.nightly_if_not_vk
    def test_create_multiple_types(self, scene: f2.Scene):
        """Create a variety of object types and verify collections."""
        scene.create_material("StandardMaterial")
        scene.create_material("StandardMaterial")
        scene.create_geometry("StaticMeshGeometry")
        e = scene.create_entity()
        e.create_component(f2.Camera)
        e.create_component(f2.ConstantLight)

        assert len(scene.materials) == 2
        assert len(scene.geometries) == 1
        assert len(scene.entities) == 2
        assert len(scene.components) == 3


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
class TestEntityTransformPropagation:
    """Tests for entity transform and parent-child world matrix propagation."""

    @pytest.mark.nightly_if_not_vk
    def test_identity_transform(self, scene: f2.Scene):
        entity = scene.create_entity()
        m = entity.world_from_object_matrix
        # Identity matrix: diagonal is 1, translation is 0.
        assert m[0, 0] == pytest.approx(1.0)
        assert m[1, 1] == pytest.approx(1.0)
        assert m[2, 2] == pytest.approx(1.0)
        assert m[3, 3] == pytest.approx(1.0)
        assert m[0, 3] == pytest.approx(0.0)
        assert m[1, 3] == pytest.approx(0.0)
        assert m[2, 3] == pytest.approx(0.0)

    @pytest.mark.nightly_if_not_vk
    def test_set_translation(self, scene: f2.Scene):
        entity = scene.create_entity()
        t = f2.Transform()
        t.translation = spy.float3(10, 20, 30)
        entity.transform = t

        m = entity.world_from_object_matrix
        assert m[0, 3] == pytest.approx(10.0)
        assert m[1, 3] == pytest.approx(20.0)
        assert m[2, 3] == pytest.approx(30.0)

    @pytest.mark.nightly_if_not_vk
    def test_parent_child_translation(self, scene: f2.Scene):
        parent = scene.create_entity()
        child = scene.create_entity()

        pt = f2.Transform()
        pt.translation = spy.float3(10, 0, 0)
        parent.transform = pt

        ct = f2.Transform()
        ct.translation = spy.float3(0, 5, 0)
        child.transform = ct

        child.parent = parent

        m = child.world_from_object_matrix
        assert m[0, 3] == pytest.approx(10.0)
        assert m[1, 3] == pytest.approx(5.0)
        assert m[2, 3] == pytest.approx(0.0)

    @pytest.mark.nightly_if_not_vk
    def test_reparent_updates_world_matrix(self, scene: f2.Scene):
        parent_a = scene.create_entity()
        parent_b = scene.create_entity()
        child = scene.create_entity()

        ta = f2.Transform()
        ta.translation = spy.float3(10, 0, 0)
        parent_a.transform = ta

        tb = f2.Transform()
        tb.translation = spy.float3(0, 20, 0)
        parent_b.transform = tb

        tc = f2.Transform()
        tc.translation = spy.float3(1, 1, 1)
        child.transform = tc

        child.parent = parent_a
        m = child.world_from_object_matrix
        assert m[0, 3] == pytest.approx(11.0)
        assert m[1, 3] == pytest.approx(1.0)

        child.parent = parent_b
        m = child.world_from_object_matrix
        assert m[0, 3] == pytest.approx(1.0)
        assert m[1, 3] == pytest.approx(21.0)

    @pytest.mark.nightly_if_not_vk
    def test_deep_hierarchy(self, scene: f2.Scene):
        root = scene.create_entity()
        mid = scene.create_entity()
        leaf = scene.create_entity()

        mid.parent = root
        leaf.parent = mid

        t = f2.Transform()
        t.translation = spy.float3(1, 0, 0)
        root.transform = t
        mid.transform = t
        leaf.transform = t

        m = leaf.world_from_object_matrix
        assert m[0, 3] == pytest.approx(3.0)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
class TestObjectRemoval:
    """Tests for object removal and scene update compaction."""

    def test_remove_material(self, scene: f2.Scene):
        m0 = scene.create_material("StandardMaterial")
        m0.name = "m0"
        m1 = scene.create_material("StandardMaterial")
        m1.name = "m1"
        m2 = scene.create_material("StandardMaterial")
        m2.name = "m2"

        assert len(scene.materials) == 3

        m1.remove()
        scene.update()

        assert len(scene.materials) == 2
        assert scene.materials[0].name == "m0"
        assert scene.materials[1].name == "m2"

    def test_remove_entity(self, scene: f2.Scene):
        e0 = scene.create_entity()
        e0.name = "e0"
        e1 = scene.create_entity()
        e1.name = "e1"

        e0.remove()
        scene.update()

        assert len(scene.entities) == 2
        assert scene.entities[1].name == "e1"

    def test_remove_entity_removes_components(self, scene: f2.Scene):
        entity = scene.create_entity()
        entity.create_component(f2.Camera)
        entity.create_component(f2.ConstantLight)

        assert len(scene.components) == 3

        entity.remove()
        scene.update()

        assert len(scene.entities) == 1
        assert len(scene.components) == 1

    def test_remove_all_objects(self, scene: f2.Scene):
        scene.create_material("StandardMaterial")
        scene.create_material("StandardMaterial")
        e = scene.create_entity()
        e.create_component(f2.Camera)

        scene.materials[0].remove()
        scene.materials[1].remove()
        e.remove()
        scene.update()

        assert len(scene.materials) == 0
        assert len(scene.entities) == 1
        assert len(scene.components) == 1

    def test_collection_index_updated_after_compaction(self, scene: f2.Scene):
        m0 = scene.create_material("StandardMaterial")
        m1 = scene.create_material("StandardMaterial")
        m2 = scene.create_material("StandardMaterial")

        assert m0.collection_index == 0
        assert m1.collection_index == 1
        assert m2.collection_index == 2

        m0.remove()
        scene.update()

        # After compaction, remaining objects should have updated indices.
        assert len(scene.materials) == 2
        assert scene.materials[0].collection_index == 0
        assert scene.materials[1].collection_index == 1


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
class TestActiveCamera:
    """Tests for active camera management on the scene."""

    def test_active_camera_auto_selected(self, scene: f2.Scene):
        assert scene.active_camera is scene.free_camera

        entity = scene.create_entity()
        camera = entity.create_component(f2.Camera)
        scene.update()

        assert scene.active_camera is camera

    def test_active_camera_set_get(self, scene: f2.Scene):
        e1 = scene.create_entity()
        cam1 = e1.create_component(f2.Camera)
        e2 = scene.create_entity()
        cam2 = e2.create_component(f2.Camera)
        scene.update()

        assert scene.active_camera is cam1

        scene.active_camera = cam2
        assert scene.active_camera is cam2

        scene.active_camera = None
        assert scene.active_camera is scene.free_camera

    def test_active_camera_falls_back_on_removal(self, scene: f2.Scene):
        entity = scene.create_entity()
        camera = entity.create_component(f2.Camera)
        scene.update()

        assert scene.active_camera is camera

        entity.remove()
        scene.update()

        assert scene.active_camera is scene.free_camera

    def test_active_camera_auto_replacement(self, scene: f2.Scene):
        e1 = scene.create_entity()
        cam1 = e1.create_component(f2.Camera)
        e2 = scene.create_entity()
        cam2 = e2.create_component(f2.Camera)
        scene.update()

        scene.active_camera = cam1
        assert scene.active_camera is cam1

        e1.remove()
        scene.update()

        assert scene.active_camera is scene.free_camera


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
class TestSceneGlobals:
    """Tests for Python-implemented scene globals."""

    def test_python_scene_globals_bind_and_remove(self, scene: f2.Scene, device: spy.Device):
        class TestGlobals(f2.SceneGlobals):
            def __init__(self, value: int):
                super().__init__()
                self.value = value
                self.bind_calls = 0

            def bind(self, globals: spy.ShaderCursor) -> None:
                self.bind_calls += 1
                globals["scene_globals_value"] = self.value

        module = device.load_module_from_source(
            "test_scene_globals_bind",
            """
import falcor2.render;

    ParameterBlock<Scene> g_scene;
uint scene_globals_value;
""",
        )
        shader_object = device.create_shader_object(module.layout.globals_type_layout)
        cursor = spy.ShaderCursor(shader_object)

        scene_globals = TestGlobals(17)
        scene.add_scene_globals(scene_globals)
        scene.bind(cursor, f2.SceneBindFlags.all)

        assert scene_globals.bind_calls == 1

        scene.remove_scene_globals(scene_globals)
        scene.bind(cursor, f2.SceneBindFlags.all)

        assert scene_globals.bind_calls == 1


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES[:1])
@pytest.mark.parametrize("per_geometry_type", [False, True])
def test_pipeline_snapshot_and_shader_reuse(
    device_type: spy.DeviceType, per_geometry_type: bool
) -> None:
    device = helpers.get_device(device_type)
    scene = f2.Scene.create(device)
    scene.create_material(f2.StandardMaterial)
    scene.update()
    helper = SceneShaderHelper(device)
    source = "falcor2.rendernodes.reference_pathtracer"
    module, before = helper.prepare(scene, source)
    if per_geometry_type:
        ray = f2.SceneRayTracingSetup.PerGeometryTypeRayDesc()
        ray.hit_groups = {
            f2.GeometryType.triangle: spy.HitGroupDesc(
                hit_group_name="probe_triangle", closest_hit_entry_point="probe_closest_hit"
            )
        }
    else:
        ray = f2.SceneRayTracingSetup.RayDesc()
        ray.name = "probe"
        ray.has_closest_hit = True
    original_setup = f2.SceneRayTracingSetup.create(scene, [ray])

    geometry = scene.create_geometry(f2.StaticMeshGeometry)
    _set_quad(geometry)
    scene.update()
    reused, after = helper.prepare(scene, source)
    assert reused is module
    assert after.matches_shader_inputs(before)
    assert after != before
    assert not before.has_geometry_type(f2.GeometryType.triangle)
    assert after.has_geometry_type(f2.GeometryType.triangle)
    # Setup uses the saved snapshot even after the scene changes.
    repeated_setup = f2.SceneRayTracingSetup.create(scene, [ray], before)
    current_setup = f2.SceneRayTracingSetup.create(scene, [ray], requirements=None, options=None)
    assert repeated_setup.sbt_hit_group_names == original_setup.sbt_hit_group_names
    assert current_setup.sbt_hit_group_names != original_setup.sbt_hit_group_names
    options = f2.SceneRayTracingSetup.Options()
    options.skip_unused_geometry_types = False
    with_options = f2.SceneRayTracingSetup.create(scene, [ray], before, options)
    assert with_options.sbt_hit_group_names != original_setup.sbt_hit_group_names
    assert (
        with_options.sbt_hit_group_names
        == f2.SceneRayTracingSetup.create(
            scene, [ray], requirements=before, options=options
        ).sbt_hit_group_names
    )


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES[:1])
def test_shared_requirement_snapshots(device: spy.Device) -> None:
    scene = f2.Scene.create(device)
    material = scene.create_material(f2.StandardMaterial)
    scene.update()
    general = scene.resolve_requirements(policy=f2.SceneCapabilityPolicy.all)
    required = scene.requirements
    assert scene.resolve_requirements() is required
    assert scene.resolve_requirements(policy=f2.SceneCapabilityPolicy.required) is required
    assert (
        scene.resolve_requirements(
            policy=f2.SceneCapabilityPolicy.specified,
            specified_capabilities=scene.required_capabilities,
        )
        is required
    )

    entity = scene.create_entity()
    for x in (0.0, 1.0):
        transform = f2.Transform()
        transform.translation = spy.float3(x, 0, 0)
        entity.transform = transform
        scene.update()
        assert scene.resolve_requirements(policy=f2.SceneCapabilityPolicy.all) is general
        assert scene.requirements is required
        assert scene.resolve_requirements(policy=f2.SceneCapabilityPolicy.required) is required

    geometry = scene.create_geometry(f2.StaticMeshGeometry)
    _set_quad(geometry)
    scene.update()
    with_geometry = scene.resolve_requirements(policy=f2.SceneCapabilityPolicy.all)
    assert with_geometry is not general
    assert with_geometry.matches_shader_inputs(general)
    assert with_geometry != general
    assert not general.has_geometry_type(f2.GeometryType.triangle)
    assert with_geometry.has_geometry_type(f2.GeometryType.triangle)
    optimized = scene.requirements
    assert optimized.matches_shader_inputs(required)
    assert optimized != required

    # Material demand changes revalidate selections, even if the all-capabilities shader is unchanged.
    material.alpha_mode = f2.AlphaMode.mask
    scene.update()
    assert scene.resolve_requirements(policy=f2.SceneCapabilityPolicy.all) is with_geometry
    assert scene.requirements.capabilities == scene.required_capabilities
    for _ in range(2):
        with pytest.raises(RuntimeError, match="omits required capabilities"):
            scene.resolve_requirements(
                policy=f2.SceneCapabilityPolicy.specified,
                specified_capabilities=f2.SceneCapabilities(),
            )
    assert not scene.resolve_requirements(
        policy=f2.SceneCapabilityPolicy.required
    ).matches_shader_inputs(optimized)
    material.alpha_mode = f2.AlphaMode.opaque
    scene.update()
    assert scene.resolve_requirements(policy=f2.SceneCapabilityPolicy.required) is optimized

    geometry.remove()
    scene.update()
    assert scene.resolve_requirements(policy=f2.SceneCapabilityPolicy.all) == general
    assert scene.requirements == required
    other = f2.Scene.create(device)
    other.create_material(f2.StandardMaterial)
    other.update()
    assert other.requirements is not scene.requirements
    assert other.requirements == scene.requirements
    assert other.requirements.matches(scene.requirements)
    assert general.matches(general)
    assert general.matches_shader_inputs(general)
    assert not general.matches(None)
    assert not general.matches_shader_inputs(None)
    assert not general.matches(with_geometry)

    # Python views cannot mutate the shared shader storage.
    conformances = general.type_conformances
    original_id = conformances[0].id
    conformances[0].id = original_id + 100
    general.modules.clear()
    assert general.type_conformances[0].id == original_id
    assert general.modules
    with pytest.raises(AttributeError):
        general.hit_group_policy.geometry_type_count = 123
    with pytest.raises(AttributeError):
        general.geometry_type_mask = 123
    del scene, other
    assert general.matches_shader_inputs(with_geometry)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES[:1])
def test_render_request_resolution(device: spy.Device) -> None:
    scene = f2.Scene.create(device)
    scene.update()
    general = scene.requirements
    request = f2.SceneRenderRequest(
        material_aovs=["roughness_output", "albedo_output", "albedo_output"]
    )
    assert request.material_aovs == ["roughness_output", "albedo_output", "albedo_output"]
    for _ in range(2):
        with pytest.raises(RuntimeError, match="not provided"):
            scene.resolve_requirements(request)

    registry = scene.material_aov_registry
    registry.register_output("albedo_output", spy.DataType.float32, 3)
    registry.register_output("roughness_output", spy.DataType.float32, 1)
    with pytest.raises(RuntimeError, match="not provided"):
        scene.resolve_requirements(request)
    scene.update()
    required = scene.resolve_requirements(request)
    assert scene.resolve_requirements() is general
    assert not general.capabilities.has(f2.SceneCapabilityFlags.material_aovs)
    assert (
        scene.resolve_requirements(request, f2.SceneCapabilityPolicy.all).capabilities
        == f2.SceneCapabilities.all()
    )
    attributes = f2.SceneCapabilities(f2.SceneCapabilityFlags.geometry_properties)
    aov_capabilities = attributes.merged(
        f2.SceneCapabilities(f2.SceneCapabilityFlags.material_aovs)
    )
    assert required.capabilities == aov_capabilities
    assert scene.resolve_requirements(request, f2.SceneCapabilityPolicy.required) is required
    assert (
        scene.resolve_requirements(
            f2.SceneRenderRequest(material_aovs=["albedo_output"]),
            f2.SceneCapabilityPolicy.required,
        )
        is required
    )
    assert (
        scene.resolve_requirements(
            f2.SceneRenderRequest(required_capabilities=aov_capabilities),
            f2.SceneCapabilityPolicy.required,
        )
        is required
    )
    assert (
        scene.resolve_requirements(policy=f2.SceneCapabilityPolicy.required).capabilities
        == f2.SceneCapabilities()
    )

    for insufficient in (f2.SceneCapabilities(), attributes):
        with pytest.raises(RuntimeError, match="omits required capabilities"):
            scene.resolve_requirements(request, f2.SceneCapabilityPolicy.specified, insufficient)
    assert (
        scene.resolve_requirements(request, f2.SceneCapabilityPolicy.specified, aov_capabilities)
        is required
    )
    # Mutable value requests are revalidated on each explicit resolution.
    request.material_aovs = ["missing"]
    with pytest.raises(RuntimeError, match="not provided"):
        scene.resolve_requirements(request)
    request.material_aovs = []
    request.required_capabilities = aov_capabilities
    assert scene.resolve_requirements(request, f2.SceneCapabilityPolicy.required) is required
    request.required_capabilities = f2.SceneCapabilities()
    assert (
        scene.resolve_requirements(request, f2.SceneCapabilityPolicy.required).capabilities
        == f2.SceneCapabilities()
    )
    request.material_aovs = ["roughness_output", "albedo_output"]
    registry.register_output("later", spy.DataType.float32, 1)
    scene.update()
    assert scene.resolve_requirements(request, f2.SceneCapabilityPolicy.required) is required
    other = f2.Scene.create(device)
    other.update()
    with pytest.raises(RuntimeError, match="not provided"):
        other.resolve_requirements(request)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES[:1])
def test_scene_shader_helper_resolves_consumer_demand(
    device: spy.Device, monkeypatch: pytest.MonkeyPatch
) -> None:
    scene = f2.Scene.create(device)
    scene.material_aov_registry.register_output("first", spy.DataType.float32, 1)
    scene.material_aov_registry.register_output("second", spy.DataType.float32, 1)
    scene.update()
    helper = SceneShaderHelper(device)
    source = "falcor2.rendernodes.reference_pathtracer"
    policy = f2.SceneCapabilityPolicy.required
    requests: list[tuple[str, ...]] = []
    resolve = f2.Scene.resolve_requirements

    def counted_resolve(
        self: f2.Scene,
        request: f2.SceneRenderRequest,
        policy: f2.SceneCapabilityPolicy,
        specified_capabilities: f2.SceneCapabilities,
    ) -> f2.SceneRequirements:
        requests.append(tuple(request.material_aovs))
        return resolve(self, request, policy, specified_capabilities)

    monkeypatch.setattr(f2.Scene, "resolve_requirements", counted_resolve)
    module, requirements = helper.prepare(scene, source, policy=policy)
    repeated, snapshot = helper.prepare(scene, source, policy=policy)
    assert repeated is module and snapshot is requirements
    assert len(requests) == 1

    geometry = f2.SceneCapabilities(f2.SceneCapabilityFlags.geometry_properties)
    _, snapshot = helper.prepare(scene, source, policy=policy, required_capabilities=geometry)
    assert snapshot.capabilities == geometry
    assert len(requests) == 2

    # Changing the contents of the same list must revalidate demand, even when shaders stay the same.
    names = ["first"]
    aov_module, aov_requirements = helper.prepare(scene, source, policy=policy, material_aovs=names)
    names[:] = ["second"]
    repeated, snapshot = helper.prepare(scene, source, policy=policy, material_aovs=names)
    assert repeated is aov_module and snapshot is aov_requirements
    assert requests[-2:] == [("first",), ("second",)]
    assert len(requests) == 4
    helper.prepare(scene, source, policy=policy, material_aovs=names)
    assert len(requests) == 4

    names[:] = ["missing"]
    for _ in range(2):
        with pytest.raises(RuntimeError, match="not provided"):
            helper.prepare(scene, source, policy=policy, material_aovs=names)
    assert len(requests) == 6
    names[:] = ["second"]
    repeated, snapshot = helper.prepare(scene, source, policy=policy, material_aovs=names)
    assert repeated is aov_module and snapshot is aov_requirements
    assert len(requests) == 6

    # Switching source modules only needs composition, not a fresh requirement resolution.
    alternate = spy.Module.load_from_file(device, "render/test_scene.slang")
    changed, snapshot = helper.prepare(scene, alternate, policy=policy, material_aovs=names)
    assert changed is not aov_module and snapshot is aov_requirements
    assert len(requests) == 6

    # A failed composition must not publish a different scene's requirements or binding.
    other = f2.Scene.create(device)
    other.material_aov_registry.register_output("second", spy.DataType.float32, 1)
    other.update()
    assert scene.requirements_generation != other.requirements_generation
    for _ in range(2):
        with pytest.raises(spy.SlangCompileError, match="missing_scene_shader_module"):
            helper.prepare(other, "missing_scene_shader_module", policy=policy, material_aovs=names)
    assert len(requests) == 8
    repeated, snapshot = helper.prepare(scene, alternate, policy=policy, material_aovs=names)
    assert repeated is changed and snapshot is aov_requirements
    assert len(requests) == 8
    reused, other_requirements = helper.prepare(
        other, alternate, policy=policy, material_aovs=names
    )
    assert reused is changed
    assert other_requirements is not snapshot
    assert other_requirements == snapshot
    assert len(requests) == 9


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
