# SPDX-License-Identifier: Apache-2.0

import pytest
import slangpy as spy
from typing import Optional

import falcor2 as f2
import falcor2.ui as ui
import falcor2.testing.helpers as helpers


def _mouse_event(
    event_type: spy.MouseEventType,
    *,
    button: spy.MouseButton = spy.MouseButton.unknown,
    pos: Optional[spy.float2] = None,
    scroll: Optional[spy.float2] = None,
    modifiers: spy.KeyModifierFlags = spy.KeyModifierFlags.none,
):
    event = spy.MouseEvent()
    event.type = event_type
    event.button = button
    event.pos = pos if pos is not None else spy.float2(0.0, 0.0)
    event.scroll = scroll if scroll is not None else spy.float2(0.0, 0.0)
    event.mods = modifiers
    return event


def _key_event(
    event_type: spy.KeyboardEventType,
    key: spy.KeyCode,
):
    event = spy.KeyboardEvent()
    event.type = event_type
    event.key = key
    return event


# ---------------------------------------------------------------------------
# SceneInteractionController basics
# ---------------------------------------------------------------------------


def test_default_state():
    ic = ui.SceneInteractionController(None, None, None, None)
    assert ic.scene_editor is None
    assert ic.scene_picker is None
    assert ic.selection_overlay is None
    assert ic.camera_controller is None
    assert ic.scene is None


def test_keyboard_not_consumed_without_camera_controller():
    ic = ui.SceneInteractionController(None, None, None, None)

    event = _key_event(spy.KeyboardEventType.key_press, spy.KeyCode.w)
    assert ic.handle_keyboard_event(event) is False


def test_mouse_not_consumed_without_camera_controller():
    ic = ui.SceneInteractionController(None, None, None, None)

    event = _mouse_event(
        spy.MouseEventType.move,
        pos=spy.float2(100.0, 100.0),
    )
    assert ic.handle_mouse_event(event) is False


def test_captures_mouse_during_camera_interaction():
    cc = ui.CameraController()
    editor = ui.SceneEditor()
    editor.visible = False  # No editor gating.
    ic = ui.SceneInteractionController(editor, None, None, cc)

    # Start right-click to enter first-person mode.
    down = _mouse_event(
        spy.MouseEventType.button_down,
        button=spy.MouseButton.right,
        pos=spy.float2(100.0, 100.0),
    )
    ic.handle_mouse_event(down)

    assert cc.is_captured() is True

    # Mouse move while captured should be consumed.
    move = _mouse_event(
        spy.MouseEventType.move,
        pos=spy.float2(110.0, 110.0),
    )
    assert ic.handle_mouse_event(move) is True

    # Release should uncapture.
    up = _mouse_event(
        spy.MouseEventType.button_up,
        button=spy.MouseButton.right,
        pos=spy.float2(110.0, 110.0),
    )
    ic.handle_mouse_event(up)

    assert cc.is_captured() is False


def test_focus_on_selection_returns_false_without_editor():
    cc = ui.CameraController()
    ic = ui.SceneInteractionController(None, None, None, cc)

    assert ic.focus_on_selection(camera=None) is False


def test_focus_on_selection_returns_false_with_no_selection():
    cc = ui.CameraController()
    editor = ui.SceneEditor()
    ic = ui.SceneInteractionController(editor, None, None, cc)

    assert ic.focus_on_selection(camera=None) is False


def test_reset_callback_not_called_without_selection():
    cc = ui.CameraController()
    editor = ui.SceneEditor()
    ic = ui.SceneInteractionController(editor, None, None, cc)

    reset_called = False

    def on_reset():
        nonlocal reset_called
        reset_called = True

    ic.reset_callback = on_reset

    # Without a selected object, focus won't trigger the callback.
    ic.focus_on_selection(camera=None)
    assert reset_called is False


# ---------------------------------------------------------------------------
# SceneEditor selection version
# ---------------------------------------------------------------------------


def test_selection_version_starts_at_zero():
    editor = ui.SceneEditor()
    assert editor.selection_version == 0


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_selection_version_increments_on_set_selected_object(
    device_type: spy.DeviceType,
):
    device = helpers.get_device(device_type)
    scene = f2.Scene(device)
    entity = scene.create_entity()

    editor = ui.SceneEditor()
    editor.scene = scene
    v0 = editor.selection_version

    # Setting to None when already None should not increment.
    editor.selected_object = None
    assert editor.selection_version == v0

    # Setting to a real object should increment.
    editor.selected_object = entity
    assert editor.selection_version == v0 + 1

    # Setting to the same object again should not increment.
    editor.selected_object = entity
    assert editor.selection_version == v0 + 1

    # Setting back to None should increment.
    editor.selected_object = None
    assert editor.selection_version == v0 + 2


# ---------------------------------------------------------------------------
# SceneEditor viewport helpers (without ImGui context, defaults apply)
# ---------------------------------------------------------------------------


def test_is_viewport_interactive_defaults_to_false():
    editor = ui.SceneEditor()
    assert editor.is_viewport_interactive() is False


if __name__ == "__main__":
    pytest.main([__file__, "-vs"])
