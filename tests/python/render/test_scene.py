# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import numpy as np
import pytest
import slangpy as spy
import falcor2 as f2
import falcor2.testing.helpers as helpers

DATA = Path(__file__).parent.parent.parent.parent / "data"


@pytest.fixture
def device(device_type: spy.DeviceType) -> spy.Device:
    return helpers.get_device(device_type)


@pytest.fixture
def scene(device: spy.Device) -> f2.Scene:
    return f2.Scene.create(device)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES[:1])
def test_scene_uv_origin_api(device_type: spy.DeviceType):
    device = helpers.get_device(device_type)
    scene = f2.Scene.create(device)
    assert scene.options.uv_origin == f2.UVOrigin.upper_left
    assert scene.requirements_generation != 0
    assert f2.SceneOptions().uv_origin == f2.UVOrigin.upper_left
    assert not hasattr(f2, "create_scene")
    with pytest.raises(TypeError):
        f2.Scene(device)
    with pytest.raises(AttributeError):
        scene.uv_origin = f2.UVOrigin.lower_left
    with pytest.raises(AttributeError):
        scene.options.uv_origin = f2.UVOrigin.lower_left
    scene.update()
    assert "falcor2_scene_uv_origin_upper_left" in [
        module.name for module in scene.requirements.modules
    ]

    lower_left_scene = f2.Scene(device, f2.SceneOptions(f2.UVOrigin.lower_left))
    assert lower_left_scene.options.uv_origin == f2.UVOrigin.lower_left
    lower_left_scene.update()
    assert "falcor2_scene_uv_origin_lower_left" in [
        module.name for module in lower_left_scene.requirements.modules
    ]

    module = spy.Module.load_from_file(
        device, "render/test_texture_manager.slang", link=lower_left_scene.requirements.modules
    )
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
    assert module.test_2d(
        handle={"data": red_top_green_bottom_handle.data},
        uv=spy.float2(0.5, 0.25),
        default_value=spy.float4(1, 0, 1, 1),
    ) == spy.float4(0, 1, 0, 1)

    path = DATA / "assets/cornell-box/usdpreviewsurface/cornell-box.usda"
    importer_scene = f2.UsdImporter().load_scene(path)
    assert importer_scene.uv_origin == f2.UVOrigin.lower_left

    loaded_scenes = [
        f2.Scene.create(device, importer_scene),
        f2.Scene.create(device, path),
    ]
    for loaded_scene in loaded_scenes:
        assert loaded_scene.options.uv_origin == f2.UVOrigin.lower_left
        loaded_scene.update()
        assert "falcor2_scene_uv_origin_lower_left" in [
            module.name for module in loaded_scene.requirements.modules
        ]

    override_scenes = [
        f2.Scene.create(device, importer_scene, f2.UVOrigin.upper_left),
        f2.Scene.create(device, path, uv_origin=f2.UVOrigin.upper_left),
    ]
    for override_scene in override_scenes:
        assert override_scene.options.uv_origin == f2.UVOrigin.upper_left
        override_scene.update()
        assert "falcor2_scene_uv_origin_upper_left" in [
            module.name for module in override_scene.requirements.modules
        ]


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
class TestSceneCollections:
    """Tests for scene collection views and iteration."""

    def test_empty_collections(self, scene: f2.Scene):
        assert len(scene.materials) == 0
        assert len(scene.geometries) == 0
        assert len(scene.entities) == 0
        assert len(scene.components) == 0

    def test_empty_iteration(self, scene: f2.Scene):
        assert list(scene.materials) == []
        assert list(scene.geometries) == []
        assert list(scene.entities) == []
        assert list(scene.components) == []

    def test_materials_collection(self, scene: f2.Scene):
        m0 = scene.create_material("StandardMaterial")
        m1 = scene.create_material("StandardMaterial")

        assert len(scene.materials) == 2
        assert scene.materials[0] is m0
        assert scene.materials[1] is m1

    def test_materials_iteration(self, scene: f2.Scene):
        m0 = scene.create_material("StandardMaterial")
        m1 = scene.create_material("StandardMaterial")

        iterated = list(scene.materials)
        assert len(iterated) == 2
        assert iterated[0] is m0
        assert iterated[1] is m1

    def test_material_id(self, scene: f2.Scene):
        material = scene.create_material("StandardMaterial")

        assert isinstance(material.material_id, int)
        assert material.material_id == 0

        scene.update()

        assert isinstance(material.material_id, int)
        assert material.material_id > 0

    def test_entities_collection(self, scene: f2.Scene):
        e0 = scene.create_entity()
        e1 = scene.create_entity()
        e2 = scene.create_entity()

        assert len(scene.entities) == 3
        assert scene.entities[0] is e0
        assert scene.entities[1] is e1
        assert scene.entities[2] is e2

    def test_entities_iteration(self, scene: f2.Scene):
        e0 = scene.create_entity()
        e1 = scene.create_entity()

        iterated = list(scene.entities)
        assert len(iterated) == 2
        assert iterated[0] is e0
        assert iterated[1] is e1

    def test_components_via_entity(self, scene: f2.Scene):
        entity = scene.create_entity()
        camera = entity.create_component(f2.Camera)

        assert len(scene.components) == 1
        assert scene.components[0] is camera

        iterated = list(scene.components)
        assert len(iterated) == 1
        assert iterated[0] is camera

    def test_negative_indexing(self, scene: f2.Scene):
        m0 = scene.create_material("StandardMaterial")
        m1 = scene.create_material("StandardMaterial")
        m2 = scene.create_material("StandardMaterial")

        assert scene.materials[-1] is m2
        assert scene.materials[-2] is m1
        assert scene.materials[-3] is m0

    def test_index_out_of_range(self, scene: f2.Scene):
        scene.create_material("StandardMaterial")

        with pytest.raises(IndexError):
            scene.materials[1]
        with pytest.raises(IndexError):
            scene.materials[-2]

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

    def test_find_by_type(self, scene: f2.Scene):
        entity = scene.create_entity()
        camera = entity.create_component(f2.Camera)
        light = entity.create_component(f2.ConstantLight)

        assert scene.components.find(type=f2.Camera) is camera
        assert scene.components.find(type=f2.ConstantLight) is light

    def test_find_by_name_and_type(self, scene: f2.Scene):
        e0 = scene.create_entity()
        camera = e0.create_component(f2.Camera)
        camera.name = "main"
        light = e0.create_component(f2.ConstantLight)
        light.name = "main"

        assert scene.components.find("main", type=f2.Camera) is camera
        assert scene.components.find("main", type=f2.ConstantLight) is light
        assert scene.components.find("main", type=f2.PointLight) is None

    def test_find_no_args_returns_first(self, scene: f2.Scene):
        m0 = scene.create_material("StandardMaterial")
        m1 = scene.create_material("StandardMaterial")

        assert scene.materials.find() is m0

    def test_find_on_empty(self, scene: f2.Scene):
        assert scene.materials.find("anything") is None
        assert scene.materials.find() is None

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

    def test_find_all_by_type(self, scene: f2.Scene):
        entity = scene.create_entity()
        camera = entity.create_component(f2.Camera)
        light = entity.create_component(f2.ConstantLight)
        light2 = entity.create_component(f2.ConstantLight)

        cameras = scene.components.find_all(type=f2.Camera)
        assert len(cameras) == 1
        assert cameras[0] is camera

        lights = scene.components.find_all(type=f2.ConstantLight)
        assert len(lights) == 2
        assert lights[0] is light
        assert lights[1] is light2

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

    def test_find_all_no_args(self, scene: f2.Scene):
        m0 = scene.create_material("StandardMaterial")
        m1 = scene.create_material("StandardMaterial")

        result = scene.materials.find_all()
        assert len(result) == 2
        assert result[0] is m0
        assert result[1] is m1

    def test_find_all_on_empty(self, scene: f2.Scene):
        assert scene.materials.find_all("anything") == []
        assert scene.materials.find_all() == []


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
class TestSceneObjectFactory:
    """Tests for creating scene objects by type name and by class."""

    def test_create_material_by_type_name(self, scene: f2.Scene):
        mat = scene.create_material("StandardMaterial")
        assert mat is not None
        assert len(scene.materials) == 1
        assert scene.materials[0] is mat

    def test_create_geometry_by_type_name(self, scene: f2.Scene):
        geo = scene.create_geometry("StaticMeshGeometry")
        assert geo is not None
        assert isinstance(geo, f2.StaticMeshGeometry)
        assert len(scene.geometries) == 1

    def test_create_geometry_by_class(self, scene: f2.Scene):
        geo = scene.create_geometry(f2.StaticMeshGeometry)
        assert geo is not None
        assert isinstance(geo, f2.StaticMeshGeometry)

    def test_create_entity(self, scene: f2.Scene):
        entity = scene.create_entity()
        assert entity is not None
        assert len(scene.entities) == 1

    def test_create_component_by_class(self, scene: f2.Scene):
        entity = scene.create_entity()
        camera = entity.create_component(f2.Camera)
        assert camera is not None
        assert isinstance(camera, f2.Camera)
        assert len(scene.components) == 1

    def test_create_component_by_type_name(self, scene: f2.Scene):
        entity = scene.create_entity()
        camera = entity.create_component("Camera")
        assert camera is not None
        assert isinstance(camera, f2.Camera)

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
        assert len(scene.entities) == 1
        assert len(scene.components) == 2


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
class TestEntityTransformPropagation:
    """Tests for entity transform and parent-child world matrix propagation."""

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

    def test_set_translation(self, scene: f2.Scene):
        entity = scene.create_entity()
        t = f2.Transform()
        t.translation = spy.float3(10, 20, 30)
        entity.transform = t

        m = entity.world_from_object_matrix
        assert m[0, 3] == pytest.approx(10.0)
        assert m[1, 3] == pytest.approx(20.0)
        assert m[2, 3] == pytest.approx(30.0)

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

        assert len(scene.entities) == 1
        assert scene.entities[0].name == "e1"

    def test_remove_entity_removes_components(self, scene: f2.Scene):
        entity = scene.create_entity()
        entity.create_component(f2.Camera)
        entity.create_component(f2.ConstantLight)

        assert len(scene.components) == 2

        entity.remove()
        scene.update()

        assert len(scene.entities) == 0
        assert len(scene.components) == 0

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
        assert len(scene.entities) == 0
        assert len(scene.components) == 0

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
        assert scene.active_camera is None

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
        assert scene.active_camera is None

    def test_active_camera_cleared_on_removal(self, scene: f2.Scene):
        entity = scene.create_entity()
        camera = entity.create_component(f2.Camera)
        scene.update()

        assert scene.active_camera is camera

        entity.remove()
        scene.update()

        assert scene.active_camera is None

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

        assert scene.active_camera is cam2


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


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
