# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import json
import time
import pytest
import slangpy as spy
import falcor2 as f2
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

from falcor2.editor import Editor, EditorConfig, StreamingConfig, RenderMode
from falcor2.rendergraph import ContainerSpec
from falcor2.rendernodes import PathTracerPipeline
from falcor2.testing.helpers import DEFAULT_DEVICE_TYPES, get_device
from falcor2.streaming.input import RemoteInput


@pytest.fixture(params=DEFAULT_DEVICE_TYPES)
def editor_device(request: pytest.FixtureRequest) -> spy.Device:
    if not spy.Device.enumerate_adapters(request.param):
        pytest.skip(f"No GPU available for {request.param}")
    return get_device(request.param)


@pytest.mark.nightly
def test_headless_editor_composes_srgb_and_ui(editor_device: spy.Device) -> None:
    editor = Editor(editor_device, EditorConfig(width=640, height=360, window=False, mcp=False))
    try:
        assert editor.window is None and editor.surface is None
        editor._scene_editor.visible = False
        assert editor.update() and editor.needs_render
        assert editor.cmd is not None and editor.output is not None
        editor.cmd.clear_texture_float(
            editor.output.color_target, clear_value=spy.float4(0.18, 0.18, 0.18, 1)
        )
        editor._present_frame()
        assert editor.frame_texture is not None
        first = editor.frame_texture.to_numpy().copy()
        # 18% linear gray is ~118 in 8-bit sRGB, not 46 or double-encoded 181.
        np.testing.assert_allclose(first[200, 320, :3], 118, atol=2)
        event = spy.KeyboardEvent()
        event.type, event.key = spy.KeyboardEventType.key_press, spy.KeyCode.f5
        editor._on_keyboard_event(event)
        assert editor._scene_editor.visible
        for _ in range(3):
            assert editor.update()
            editor.present(None)
        second = editor.frame_texture.to_numpy()
        assert np.mean(np.abs(second.astype(float) - first)) > 5
        editor._cancel_interaction()
        if editor_device.info.type == spy.DeviceType.cuda:
            with pytest.raises(ValueError, match="D3D12 or Vulkan"):
                editor.render_mode = RenderMode.shaded
            editor._cycle_render_mode()
            assert editor.render_mode == RenderMode.path_traced
    finally:
        editor.close()
    editor.close()
    assert not editor.update()
    # The editor borrows the caller's device.
    assert editor_device.create_command_encoder() is not None


@pytest.mark.nightly
def test_deleted_selection_clears_before_scene_update(editor_device: spy.Device) -> None:
    scene = f2.load_scene(editor_device, "data/scenes/cornell-box.py")
    editor = Editor(
        editor_device,
        EditorConfig(width=128, height=72, window=False, mcp=False),
        scene=scene,
    )
    try:
        for entity in scene.entities:
            if any(isinstance(component, f2.GeometryInstance) for component in entity.components):
                editor._scene_editor.selected_object = entity
                break
        else:
            pytest.fail("Scene has no selectable geometry entity")

        assert editor.update()
        editor.present(None)
        editor._scene_editor.remove_selected_object()

        controller = editor._interaction_controller
        calls = []

        def update_selection_overlay() -> None:
            calls.append(entity.scene is not None)
            controller.update_selection_overlay()

        editor._interaction_controller = Mock(wraps=controller)
        editor._interaction_controller.update_selection_overlay.side_effect = (
            update_selection_overlay
        )
        try:
            assert editor.update()
            assert calls == [True]
        finally:
            editor._interaction_controller = controller
        assert entity.scene is None
        editor.present(None)
    finally:
        editor.close()


@pytest.mark.nightly
def test_editor_stream_starts_and_closes_without_a_viewer(editor_device: spy.Device) -> None:
    pytest.importorskip("aiohttp")

    from urllib.request import ProxyHandler, build_opener
    from falcor2 import streaming

    if not streaming.WebRTCPeer.available() or not streaming.VideoEncoder.check_available():
        pytest.skip("Requires WebRTC and OpenH264")
    editor = Editor(
        editor_device,
        EditorConfig(
            width=640, height=360, window=False, mcp=False, streaming=StreamingConfig(port=0)
        ),
    )
    try:
        assert editor.streaming_url is not None
        # Loopback requests must not use the CI machine's HTTP proxy.
        opener = build_opener(ProxyHandler({}))
        with opener.open(editor.streaming_url + "health", timeout=5) as response:
            assert response.status == 200
        assert editor.update()
        editor.present(None)
        assert editor._capture_image is not None and editor.frame_texture is not None
        np.testing.assert_array_equal(
            editor._capture_image.to_numpy(), editor.frame_texture.to_numpy()
        )
    finally:
        editor.close()
    assert editor.streaming_url is None


def test_headless_pathtracer_accumulates_and_remote_camera_resets(
    editor_device: spy.Device,
) -> None:
    scene = f2.load_scene(editor_device, "data/scenes/emissive-strength.py")
    pipeline = PathTracerPipeline.create(editor_device)
    pipeline.output_spec = ContainerSpec.texture2d(format=spy.Format.rgba32_float)
    pipeline.tonemapper.auto_exposure = False
    pipeline.path_tracer.max_depth = 3
    editor = Editor(editor_device, EditorConfig(width=128, height=72, window=False, mcp=False))
    editor.scene = scene
    editor._scene_editor.visible = False
    editor._camera_controller.smoothing = False
    # Fixed frame time makes remote keyboard movement independent of shader compilation.
    editor._timer = Mock(elapsed_s=Mock(return_value=0.1))
    remote = RemoteInput("test", 128, 72, editor._on_keyboard_event, editor._on_mouse_event)

    def frame() -> spy.Texture:
        assert editor.update() and editor.needs_render
        image = pipeline(scene, delta_time=editor.dt)
        editor.present(image)
        return image

    def send(kind: str, **fields: object) -> None:
        remote.receive(
            json.dumps(dict(v=1, generation="test", seq=remote.sequence + 1, type=kind, **fields)),
            time.monotonic(),
        )

    try:
        for _ in range(3):
            image = frame()
        assert (image.width, image.height) == (128, 72)
        assert pipeline._iteration == 3
        before = image.to_numpy().copy()
        position = editor._streaming_stats()["camera_position"]
        send("button", relative=False, button=2, down=True, x=64, y=36, mods=0)
        send("key", code="KeyW", down=True, repeat=False, mods=0)
        image = frame()
        assert editor._streaming_stats()["camera_position"] != position
        assert pipeline._iteration == 1
        assert np.isfinite(image.to_numpy()).all()
        assert np.any(before != image.to_numpy())
        remote.release()
        stopped = editor._streaming_stats()["camera_position"]
        frame()
        assert editor._streaming_stats()["camera_position"] == stopped
        assert pipeline._iteration == 2
        rotation = editor._camera_controller.transform.rotation
        send("button", relative=False, button=2, down=True, x=64, y=36, mods=0)
        send("motion", relative=False, x=74, y=36, mods=0)
        send("motion", relative=False, x=84, y=36, mods=0)
        frame()
        assert editor._camera_controller.transform.rotation != rotation
        assert pipeline._iteration == 1
    finally:
        remote.release()
        editor.close()


@pytest.mark.nightly
def test_remote_text_edits_native_ui(editor_device: spy.Device) -> None:
    editor = Editor(editor_device, EditorConfig(width=640, height=360, window=False, mcp=False))
    editor._scene_editor.visible = False
    window = spy.ui.Window(
        editor._ui_context.screen,
        "Remote text",
        position=spy.float2(20, 40),
        size=spy.float2(300, 150),
    )
    field = spy.ui.InputText(window, "Value", value="")
    remote = RemoteInput("test", 640, 360, editor._on_keyboard_event, editor._on_mouse_event)

    def frame() -> None:
        assert editor.update()
        editor.present(None)

    def send(kind: str, **fields: object) -> None:
        remote.receive(
            json.dumps(dict(v=1, generation="test", seq=remote.sequence + 1, type=kind, **fields)),
            time.monotonic(),
        )
        frame()

    try:
        frame()
        frame()
        send("motion", relative=False, x=60, y=80, mods=0)
        send("button", relative=False, button=0, down=True, x=60, y=80, mods=0)
        send("button", relative=False, button=0, down=False, x=60, y=80, mods=0)
        send("text", text="Remote UI")
        frame()
        assert field.value == "Remote UI"
        remote.release()
    finally:
        editor.close()


@pytest.mark.nightly
@pytest.mark.parametrize("key", [spy.KeyCode.f5, spy.KeyCode.f6, spy.KeyCode.f12])
def test_application_function_keys_work_with_ui_focus(key: spy.KeyCode) -> None:
    editor = Editor.__new__(Editor)
    editor._ui_context = Mock()
    editor._ui_context.handle_keyboard_event.return_value = True
    editor._interaction_controller = Mock()
    editor._interaction_controller.handle_keyboard_event.return_value = False
    editor._scene_editor = Mock(visible=True)
    editor._toggle_profiler = Mock()
    editor._cycle_render_mode = Mock()
    editor.on_keyboard_event = Mock()
    event = spy.KeyboardEvent()
    event.type, event.key = spy.KeyboardEventType.key_press, key
    editor._on_keyboard_event(event)
    assert editor._scene_editor.visible == (key != spy.KeyCode.f5)
    assert editor._toggle_profiler.call_count == int(key == spy.KeyCode.f6)
    assert editor._cycle_render_mode.call_count == int(key == spy.KeyCode.f12)
    editor._ui_context.handle_keyboard_event.assert_called_once_with(event)
    editor.on_keyboard_event.assert_not_called()


@pytest.mark.nightly
@pytest.mark.parametrize("failure", ["update", "render", "present"])
def test_run_closes_editor_after_failure(failure: str) -> None:
    editor = SimpleNamespace(
        _scene=None,
        needs_render=True,
        update=Mock(return_value=True),
        present=Mock(),
        close=Mock(),
        device=Mock(),
    )
    render = Mock()
    (render if failure == "render" else getattr(editor, failure)).side_effect = RuntimeError(
        "failed"
    )
    with pytest.raises(RuntimeError, match="failed"):
        Editor.run(cast(Editor, editor), render)
    editor.close.assert_called_once()
    editor.device.close.assert_not_called()


@pytest.mark.nightly
def test_run_passes_current_scene_and_closes_editor() -> None:
    scenes = [object(), object()]
    editor = SimpleNamespace(
        _scene=None, needs_render=False, present=Mock(), close=Mock(), device=Mock()
    )
    frames = iter([(False, None), (True, scenes[0]), (True, scenes[1])])

    def update() -> bool:
        try:
            editor.needs_render, editor._scene = next(frames)
        except StopIteration:
            return False
        return True

    editor.update = update
    render = Mock(side_effect=["first", "second"])
    Editor.run(cast(Editor, editor), render)
    assert [call.args for call in render.call_args_list] == [(scene,) for scene in scenes]
    assert [call.args for call in editor.present.call_args_list] == [("first",), ("second",)]
    editor.close.assert_called_once()
    editor.device.close.assert_not_called()
