# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import slangpy as spy

import falcor2 as f2
from falcor2.editor import Editor, EditorConfig
from falcor2.testing.helpers import DEFAULT_DEVICE_TYPES, get_device


@pytest.fixture(params=DEFAULT_DEVICE_TYPES)
def camera_device(request: pytest.FixtureRequest) -> spy.Device:
    return get_device(request.param)


def _make_editor(scene: f2.Scene, device: spy.Device) -> Editor:
    return Editor(device, EditorConfig(width=320, height=240, window=False, mcp=False), scene)


def _frame(editor: Editor) -> None:
    assert editor.update()
    editor.present(None)


def test_scene_editor_update_camera_nullable_return() -> None:
    editor = f2.ui.SceneEditor()
    assert editor.update_camera(dt=0.0, width=320, height=240) is None
    signature = f2.ui.SceneEditor.update_camera.__doc__.splitlines()[0]
    assert signature.endswith("-> falcor2.Camera | None")


def test_editor_camera_preparation_on_attach_resize_and_unload(camera_device: spy.Device) -> None:
    scene = f2.load_scene(camera_device, "data/scenes/cornell-box.py")
    for camera in scene.components.find_all(type=f2.Camera):
        camera.entity.remove()
    scene.update()
    scene.frame_free_camera()
    expected = np.array(scene.free_camera.entity.transform.translation)
    transform = f2.Transform()
    transform.translation = spy.float3(20, 30, 40)
    scene.free_camera.entity.transform = transform

    editor = _make_editor(scene, camera_device)
    editor._scene_editor.visible = False
    try:
        _frame(editor)
        np.testing.assert_allclose(scene.free_camera.entity.transform.translation, expected)
        assert (scene.free_camera.width, scene.free_camera.height) == (320, 240)
        scene.free_camera.entity.transform = transform
        editor.config.width, editor.config.height = 240, 480
        editor._composite = None
        _frame(editor)
        np.testing.assert_allclose(scene.free_camera.entity.transform.translation, [20, 30, 40])
        assert (scene.free_camera.width, scene.free_camera.height) == (240, 480)

        next_scene = f2.Scene.create(camera_device)
        next_scene.free_camera.entity.transform = transform
        editor.scene = next_scene
        _frame(editor)
        np.testing.assert_allclose(next_scene.free_camera.entity.transform.translation, [0, 0, 5])
        assert editor._camera is next_scene.free_camera

        event = spy.MouseEvent()
        event.type, event.button = spy.MouseEventType.button_down, spy.MouseButton.right
        editor._interaction_controller.handle_mouse_event(event)
        assert editor._interaction_controller.has_pointer_capture()
        editor.scene = None
        _frame(editor)
        assert editor._camera is None
        assert not editor._interaction_controller.has_pointer_capture()
        assert not editor._camera_controller.is_interacting()
    finally:
        editor.close()


def test_camera_preparation_updates_gizmo_projection(camera_device: spy.Device) -> None:
    scene = f2.Scene.create(camera_device)
    light = scene.create_entity()
    light.create_component(f2.PointLight)
    transform = f2.Transform()
    transform.translation = spy.float3(0.3, 0.2, 0)
    light.transform = transform
    editor = Editor(
        camera_device, EditorConfig(width=640, height=480, window=False, mcp=False), scene
    )
    try:
        for _ in range(3):
            _frame(editor)
        viewport = editor._scene_editor.viewport_state
        assert viewport.valid

        # Update both pose and aspect without a separate gizmo-camera setter or a controller.
        editor._scene_editor.camera_controller = None
        transform.translation = spy.float3(-0.2, 0, 5)
        scene.free_camera.entity.transform = transform
        scene.update()
        camera = editor._scene_editor.update_camera(0.0, 300, 600)
        assert camera is scene.free_camera
        view = camera.calc_view_from_world()
        projection = camera.calc_clip_from_view()
        view = np.array([[view[row, col] for col in range(4)] for row in range(4)])
        projection = np.array([[projection[row, col] for col in range(4)] for row in range(4)])
        clip = projection @ view @ np.array([0.3, 0.2, 0, 1])
        ndc = clip[:2] / clip[3]
        assert np.max(np.abs(ndc)) < 1

        # Hidden UI routes input without requiring OS hover; the last viewport still supplies picking bounds.
        editor._scene_editor.visible = False
        editor._scene_editor.camera_controller = editor._camera_controller
        event = spy.MouseEvent()
        event.type, event.button = spy.MouseEventType.button_down, spy.MouseButton.left
        event.pos = (
            viewport.pos + spy.float2((ndc[0] + 1) * 0.5, (1 - ndc[1]) * 0.5) * viewport.size
        )
        editor._interaction_controller.handle_mouse_event(event)
        assert editor._scene_editor.selected_object is light
    finally:
        editor.close()


def _animate_translation(scene: f2.Scene, entity: f2.Entity) -> f2.TransformAnimation:
    animation = scene.create_animation()
    channel = f2.AnimationChannel()
    channel.times = [0.0, 1.0]
    channel.values = [0.0, 0.0, 0.0, 10.0, 0.0, 0.0]
    animation.channels = [channel]
    component = entity.create_component(f2.TransformAnimation)
    component.animation = animation
    component.translation_channel = 0
    return component


def test_editor_scroll_detaches_from_animated_camera(camera_device: spy.Device) -> None:
    scene = f2.Scene.create(camera_device)
    ancestor = scene.create_entity()
    parent = scene.create_entity()
    parent.parent = ancestor
    entity = scene.create_entity()
    entity.parent = parent
    transform = f2.Transform()
    transform.translation = spy.float3(1, 2, 5)
    entity.transform = transform
    camera = entity.create_component(f2.Camera)
    _animate_translation(scene, ancestor)
    scene.time = 0.25
    scene.update()
    editor = _make_editor(scene, camera_device)
    editor._camera_controller.smoothing = False
    editor._scene_editor.visible = False
    try:
        _frame(editor)
        scene.time = 0.5
        assert scene.active_camera is camera
        before = np.array(entity.transform.translation)
        event = spy.MouseEvent()
        event.type = spy.MouseEventType.scroll
        event.scroll = spy.float2(0, 1)
        editor._interaction_controller.handle_mouse_event(event)
        _frame(editor)
        assert scene.active_camera is scene.free_camera
        assert editor._camera is scene.free_camera
        np.testing.assert_allclose(entity.transform.translation, before)
        # Editor.update() must evaluate the animation before applying the queued scroll.
        assert scene.free_camera.calc_uniforms().position.x == pytest.approx(6.0)
        assert scene.free_camera.calc_uniforms().position.z < 5.0
    finally:
        editor.close()
