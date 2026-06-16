# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
import slangpy as spy

import falcor2 as f2
import falcor2.testing.helpers as helpers
import falcor2.ui as ui


DATA = Path(__file__).resolve().parents[3] / "data"
SCENE_PATH = DATA / "assets/kronos/DamagedHelmet/glTF/DamagedHelmet.gltf"


@pytest.fixture
def device(device_type: spy.DeviceType) -> spy.Device:
    return helpers.get_device(device_type)


def _load_scene(device: spy.Device) -> f2.Scene:
    scene = f2.Scene(device, str(SCENE_PATH))
    scene.update()
    return scene


def _create_camera(scene: f2.Scene, width: int = 64, height: int = 64) -> f2.Camera:
    entity = scene.create_entity()
    camera = entity.create_component(f2.Camera)
    camera.width = width
    camera.height = height
    camera.fov_y = 70.0
    scene.update()
    return camera


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_find_entity_by_geometry_instance_id(device: spy.Device) -> None:
    scene = _load_scene(device)

    component = next(
        component for component in scene.components if isinstance(component, f2.GeometryInstance)
    )
    expected_entity = component.entity

    assert expected_entity is not None

    result = ui.ScenePicker.find_entity_by_geometry_instance_id(
        scene, component.geometry_instance_id
    )
    assert result is expected_entity

    invalid_result = ui.ScenePicker.find_entity_by_geometry_instance_id(
        scene, f2.GeometryInstanceID.invalid
    )
    assert invalid_result is None

    out_of_range_result = ui.ScenePicker.find_entity_by_geometry_instance_id(
        scene, f2.GeometryInstanceID(999999)
    )
    assert out_of_range_result is None


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_render(device: spy.Device) -> None:
    scene = _load_scene(device)
    camera = _create_camera(scene, width=32, height=24)
    picker = ui.ScenePicker(device)

    assert picker.geometry_instance_id_texture is None
    assert picker.pick(spy.uint2(0, 0)) == f2.GeometryInstanceID.invalid
    assert picker.pick_entity(scene, spy.uint2(0, 0)) is None

    command_encoder = device.create_command_encoder()
    picker.render(command_encoder, scene, camera)
    device.submit_command_buffer(command_encoder.finish())

    assert picker.geometry_instance_id_texture is not None
    assert picker.geometry_instance_id_texture.width == camera.width
    assert picker.geometry_instance_id_texture.height == camera.height

    # TODO: Add picking tests once we can setup cameras more easily.


if __name__ == "__main__":
    pytest.main([__file__, "-vs"])
