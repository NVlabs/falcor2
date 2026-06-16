# SPDX-License-Identifier: Apache-2.0

import pytest
import slangpy as spy
from typing import Optional

import falcor2 as f2
import falcor2.ui as ui


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


DT = 1.0 / 60.0


# --- Initial state ---


def test_initial_state():
    controller = ui.CameraController()

    assert controller.transform.translation == spy.float3(0.0, 0.0, 0.0)
    assert controller.transform.rotation == spy.quatf(0.0, 0.0, 0.0, 1.0)
    assert controller.update(DT) is False


# --- Scroll ---


def test_scroll_moves_forward():
    controller = ui.CameraController()
    event = _mouse_event(
        spy.MouseEventType.scroll,
        scroll=spy.float2(0.0, 1.0),
        pos=spy.float2(0.0, 0.0),
    )
    controller.handle_mouse_event(event)

    assert controller.update(DT) is True

    position = controller.transform.translation
    assert position.x == pytest.approx(0.0)
    assert position.y == pytest.approx(0.0)
    assert position.z < 0.0


def test_scroll_backward():
    controller = ui.CameraController()
    controller.smoothing = False
    event = _mouse_event(
        spy.MouseEventType.scroll,
        scroll=spy.float2(0.0, -1.0),
    )
    controller.handle_mouse_event(event)
    controller.update(DT)

    # Scrolling backward should move camera away from pivot (positive Z from origin).
    position = controller.transform.translation
    assert position.z > 0.0


# --- Transform ---


def test_set_and_get_transform():
    controller = ui.CameraController()

    transform = f2.Transform()
    transform.translation = spy.float3(1.0, 2.0, 3.0)
    controller.transform = transform

    assert controller.transform.translation == spy.float3(1.0, 2.0, 3.0)


# --- First-person mouse look ---


def test_right_mouse_drag_rotates_after_first_move():
    controller = ui.CameraController()

    controller.handle_mouse_event(
        _mouse_event(
            spy.MouseEventType.button_down,
            button=spy.MouseButton.right,
            pos=spy.float2(10.0, 20.0),
        )
    )
    assert controller.state == ui.CameraController.State.first_person
    controller.handle_mouse_event(_mouse_event(spy.MouseEventType.move, pos=spy.float2(10.0, 20.0)))

    assert controller.update(DT) is False
    assert controller.transform.rotation == spy.quatf(0.0, 0.0, 0.0, 1.0)

    controller.handle_mouse_event(_mouse_event(spy.MouseEventType.move, pos=spy.float2(26.0, 12.0)))

    assert controller.update(DT) is True
    assert controller.transform.rotation != spy.quatf(0.0, 0.0, 0.0, 1.0)

    # Release RMB -> idle.
    controller.handle_mouse_event(
        _mouse_event(
            spy.MouseEventType.button_up,
            button=spy.MouseButton.right,
            pos=spy.float2(26.0, 12.0),
        )
    )
    assert controller.state == ui.CameraController.State.idle


# --- Drag (MMB) ---


def test_middle_mouse_drag_moves_in_view_plane():
    controller = ui.CameraController()

    controller.handle_mouse_event(
        _mouse_event(
            spy.MouseEventType.button_down,
            button=spy.MouseButton.middle,
            pos=spy.float2(4.0, 4.0),
        )
    )
    assert controller.state == ui.CameraController.State.pan

    controller.handle_mouse_event(_mouse_event(spy.MouseEventType.move, pos=spy.float2(4.0, 4.0)))
    assert controller.update(DT) is False

    controller.handle_mouse_event(_mouse_event(spy.MouseEventType.move, pos=spy.float2(14.0, 16.0)))
    assert controller.update(DT) is True

    position = controller.transform.translation
    assert position.x == pytest.approx(-0.05)
    assert position.y == pytest.approx(0.06)
    assert position.z == pytest.approx(0.0)

    # Release MMB -> idle.
    controller.handle_mouse_event(
        _mouse_event(
            spy.MouseEventType.button_up,
            button=spy.MouseButton.middle,
            pos=spy.float2(14.0, 16.0),
        )
    )
    assert controller.state == ui.CameraController.State.idle


# --- Button release returns to idle ---


def test_mouse_button_release_returns_to_idle():
    controller = ui.CameraController()

    controller.handle_mouse_event(
        _mouse_event(
            spy.MouseEventType.button_down,
            button=spy.MouseButton.right,
            pos=spy.float2(0.0, 0.0),
        )
    )
    assert controller.state == ui.CameraController.State.first_person

    controller.handle_mouse_event(_mouse_event(spy.MouseEventType.move, pos=spy.float2(0.0, 0.0)))
    controller.update(DT)

    controller.handle_mouse_event(
        _mouse_event(
            spy.MouseEventType.button_up,
            button=spy.MouseButton.right,
            pos=spy.float2(0.0, 0.0),
        )
    )
    assert controller.state == ui.CameraController.State.idle

    controller.handle_mouse_event(_mouse_event(spy.MouseEventType.move, pos=spy.float2(20.0, 15.0)))

    assert controller.update(DT) is False
    assert controller.transform.rotation == spy.quatf(0.0, 0.0, 0.0, 1.0)


# --- dt <= 0 handling ---


def test_zero_dt_disables_smoothing():
    """update() with dt=0 should still process non-time-dependent input (e.g. scroll dolly)."""
    controller = ui.CameraController()
    controller.smoothing = True  # Smoothing would normally defer the scroll.
    controller.handle_mouse_event(
        _mouse_event(spy.MouseEventType.scroll, scroll=spy.float2(0.0, 5.0))
    )
    # With dt=0, smoothing is forced off and scroll applies directly.
    assert controller.update(0.0) is True
    assert controller.transform.translation.z < 0.0


def test_negative_dt_disables_smoothing():
    """update() with negative dt should still process non-time-dependent input."""
    controller = ui.CameraController()
    controller.smoothing = True
    controller.handle_mouse_event(
        _mouse_event(spy.MouseEventType.scroll, scroll=spy.float2(0.0, 5.0))
    )
    assert controller.update(-1.0) is True
    assert controller.transform.translation.z < 0.0


def test_zero_dt_first_person_rotation_works():
    """First-person mouse rotation should work at dt=0 (non-smooth path has no dt dependency)."""
    controller = ui.CameraController()
    controller.smoothing = True
    _setup_first_person(controller)

    controller.handle_mouse_event(_mouse_event(spy.MouseEventType.move, pos=spy.float2(20.0, 10.0)))
    assert controller.update(0.0) is True
    assert controller.transform.rotation != spy.quatf(0.0, 0.0, 0.0, 1.0)


def test_zero_dt_first_person_movement_is_zero():
    """WASD movement should produce no displacement at dt=0 since it's velocity * dt."""
    controller = ui.CameraController()
    controller.smoothing = False
    _setup_first_person(controller)

    controller.handle_keyboard_event(_key_event(spy.KeyboardEventType.key_press, spy.KeyCode.w))
    controller.update(0.0)

    pos = controller.transform.translation
    assert pos.x == pytest.approx(0.0)
    assert pos.y == pytest.approx(0.0)
    assert pos.z == pytest.approx(0.0)


# --- WASD keyboard movement ---


def _setup_first_person(controller: ui.CameraController):
    """Enter first-person mode by pressing RMB and anchoring the mouse."""
    controller.handle_mouse_event(
        _mouse_event(
            spy.MouseEventType.button_down,
            button=spy.MouseButton.right,
            pos=spy.float2(0.0, 0.0),
        )
    )
    controller.handle_mouse_event(_mouse_event(spy.MouseEventType.move, pos=spy.float2(0.0, 0.0)))
    controller.update(DT)


def test_wasd_forward_movement():
    controller = ui.CameraController()
    controller.smoothing = False
    _setup_first_person(controller)

    controller.handle_keyboard_event(_key_event(spy.KeyboardEventType.key_press, spy.KeyCode.w))
    controller.update(DT)

    pos = controller.transform.translation
    assert pos.z < 0.0  # Forward is -Z in default orientation.
    assert pos.x == pytest.approx(0.0)
    assert pos.y == pytest.approx(0.0)


def test_wasd_backward_movement():
    controller = ui.CameraController()
    controller.smoothing = False
    _setup_first_person(controller)

    controller.handle_keyboard_event(_key_event(spy.KeyboardEventType.key_press, spy.KeyCode.s))
    controller.update(DT)

    pos = controller.transform.translation
    assert pos.z > 0.0  # Backward is +Z.


def test_wasd_strafe_left():
    controller = ui.CameraController()
    controller.smoothing = False
    _setup_first_person(controller)

    controller.handle_keyboard_event(_key_event(spy.KeyboardEventType.key_press, spy.KeyCode.a))
    controller.update(DT)

    pos = controller.transform.translation
    assert pos.x < 0.0
    assert pos.z == pytest.approx(0.0)


def test_wasd_strafe_right():
    controller = ui.CameraController()
    controller.smoothing = False
    _setup_first_person(controller)

    controller.handle_keyboard_event(_key_event(spy.KeyboardEventType.key_press, spy.KeyCode.d))
    controller.update(DT)

    pos = controller.transform.translation
    assert pos.x > 0.0


def test_wasd_up_down():
    controller = ui.CameraController()
    controller.smoothing = False
    _setup_first_person(controller)

    controller.handle_keyboard_event(_key_event(spy.KeyboardEventType.key_press, spy.KeyCode.e))
    controller.update(DT)

    pos = controller.transform.translation
    assert pos.y > 0.0

    controller.handle_keyboard_event(_key_event(spy.KeyboardEventType.key_release, spy.KeyCode.e))
    controller.handle_keyboard_event(_key_event(spy.KeyboardEventType.key_press, spy.KeyCode.q))

    old_y = pos.y
    controller.update(DT)
    pos = controller.transform.translation
    assert pos.y < old_y  # Moved down.


def test_key_release_stops_movement():
    controller = ui.CameraController()
    controller.smoothing = False
    _setup_first_person(controller)

    controller.handle_keyboard_event(_key_event(spy.KeyboardEventType.key_press, spy.KeyCode.w))
    controller.update(DT)
    pos_after_press = controller.transform.translation

    controller.handle_keyboard_event(_key_event(spy.KeyboardEventType.key_release, spy.KeyCode.w))
    controller.update(DT)
    pos_after_release = controller.transform.translation

    # Position should not change after key release.
    assert pos_after_release.z == pytest.approx(pos_after_press.z)


# --- Orbit (Alt+LMB) ---


def test_orbit_rotates_around_pivot():
    controller = ui.CameraController()
    controller.smoothing = False

    # Enter orbit mode: Alt + LMB.
    controller.handle_mouse_event(
        _mouse_event(
            spy.MouseEventType.button_down,
            button=spy.MouseButton.left,
            pos=spy.float2(100.0, 100.0),
            modifiers=spy.KeyModifierFlags.alt,
        )
    )
    assert controller.state == ui.CameraController.State.orbit
    controller.handle_mouse_event(
        _mouse_event(
            spy.MouseEventType.move,
            pos=spy.float2(100.0, 100.0),
            modifiers=spy.KeyModifierFlags.alt,
        )
    )
    controller.update(DT)

    pivot_before = controller.pivot

    # Drag horizontally.
    controller.handle_mouse_event(
        _mouse_event(
            spy.MouseEventType.move,
            pos=spy.float2(150.0, 100.0),
            modifiers=spy.KeyModifierFlags.alt,
        )
    )
    assert controller.update(DT) is True

    # Camera should have moved but pivot stays the same.
    assert controller.pivot.x == pytest.approx(pivot_before.x)
    assert controller.pivot.y == pytest.approx(pivot_before.y)
    assert controller.pivot.z == pytest.approx(pivot_before.z)
    # Camera orientation should have changed.
    assert controller.transform.rotation != spy.quatf(0.0, 0.0, 0.0, 1.0)

    # Release LMB -> idle.
    controller.handle_mouse_event(
        _mouse_event(
            spy.MouseEventType.button_up,
            button=spy.MouseButton.left,
            pos=spy.float2(150.0, 100.0),
        )
    )
    assert controller.state == ui.CameraController.State.idle


# --- Dolly (Alt+RMB) ---


def test_dolly_changes_distance():
    controller = ui.CameraController()
    controller.smoothing = False

    initial_distance = controller.orbit_distance

    # Enter dolly mode: Alt + RMB.
    controller.handle_mouse_event(
        _mouse_event(
            spy.MouseEventType.button_down,
            button=spy.MouseButton.right,
            pos=spy.float2(100.0, 100.0),
            modifiers=spy.KeyModifierFlags.alt,
        )
    )
    assert controller.state == ui.CameraController.State.dolly
    controller.handle_mouse_event(
        _mouse_event(
            spy.MouseEventType.move,
            pos=spy.float2(100.0, 100.0),
            modifiers=spy.KeyModifierFlags.alt,
        )
    )
    controller.update(DT)

    # Drag to dolly.
    controller.handle_mouse_event(
        _mouse_event(
            spy.MouseEventType.move,
            pos=spy.float2(130.0, 130.0),
            modifiers=spy.KeyModifierFlags.alt,
        )
    )
    assert controller.update(DT) is True

    # Distance should have changed.
    assert controller.orbit_distance != pytest.approx(initial_distance)

    # Release RMB -> idle.
    controller.handle_mouse_event(
        _mouse_event(
            spy.MouseEventType.button_up,
            button=spy.MouseButton.right,
            pos=spy.float2(130.0, 130.0),
        )
    )
    assert controller.state == ui.CameraController.State.idle


# --- Track (Alt+MMB) ---


def test_track_pans_camera_and_pivot():
    controller = ui.CameraController()
    controller.smoothing = False

    initial_pos = controller.transform.translation
    initial_pivot = controller.pivot

    # Enter track mode: Alt + MMB.
    controller.handle_mouse_event(
        _mouse_event(
            spy.MouseEventType.button_down,
            button=spy.MouseButton.middle,
            pos=spy.float2(100.0, 100.0),
            modifiers=spy.KeyModifierFlags.alt,
        )
    )
    assert controller.state == ui.CameraController.State.track
    controller.handle_mouse_event(
        _mouse_event(
            spy.MouseEventType.move,
            pos=spy.float2(100.0, 100.0),
            modifiers=spy.KeyModifierFlags.alt,
        )
    )
    controller.update(DT)

    # Drag to track.
    controller.handle_mouse_event(
        _mouse_event(
            spy.MouseEventType.move,
            pos=spy.float2(120.0, 110.0),
            modifiers=spy.KeyModifierFlags.alt,
        )
    )
    assert controller.update(DT) is True

    # Both camera and pivot should have moved.
    pos = controller.transform.translation
    assert pos.x != pytest.approx(initial_pos.x) or pos.y != pytest.approx(initial_pos.y)
    pivot = controller.pivot
    assert pivot.x != pytest.approx(initial_pivot.x) or pivot.y != pytest.approx(initial_pivot.y)

    # Release MMB -> idle.
    controller.handle_mouse_event(
        _mouse_event(
            spy.MouseEventType.button_up,
            button=spy.MouseButton.middle,
            pos=spy.float2(120.0, 110.0),
        )
    )
    assert controller.state == ui.CameraController.State.idle


# --- Focus ---


def test_focus_sets_pivot_and_repositions():
    controller = ui.CameraController()

    target = spy.float3(5.0, 3.0, -10.0)
    distance = 8.0
    controller.focus(target, distance)

    assert controller.pivot.x == pytest.approx(target.x)
    assert controller.pivot.y == pytest.approx(target.y)
    assert controller.pivot.z == pytest.approx(target.z)
    assert controller.orbit_distance == pytest.approx(distance)

    # Camera should be `distance` away from pivot.
    pos = controller.transform.translation
    dx = pos.x - target.x
    dy = pos.y - target.y
    dz = pos.z - target.z
    actual_dist = (dx * dx + dy * dy + dz * dz) ** 0.5
    assert actual_dist == pytest.approx(distance)


def test_focus_clamps_minimum_distance():
    controller = ui.CameraController()
    controller.focus(spy.float3(0.0, 0.0, 0.0), 0.001)
    # Should be clamped to MIN_ORBIT_DISTANCE (0.01f).
    assert controller.orbit_distance == pytest.approx(0.01, abs=1e-6)


# --- Properties ---


def test_move_speed_property():
    controller = ui.CameraController()
    controller.move_speed = 20.0
    assert controller.move_speed == pytest.approx(20.0)


def test_move_speed_clamped():
    controller = ui.CameraController()
    controller.move_speed = 0.0001
    assert controller.move_speed == pytest.approx(0.01, abs=1e-6)

    controller.move_speed = 99999.0
    assert controller.move_speed == pytest.approx(1000.0)


def test_orbit_distance_property():
    controller = ui.CameraController()
    controller.orbit_distance = 10.0
    assert controller.orbit_distance == pytest.approx(10.0)


def test_orbit_distance_clamped():
    controller = ui.CameraController()
    controller.orbit_distance = 0.001
    assert controller.orbit_distance == pytest.approx(0.01, abs=1e-6)


def test_pivot_property():
    controller = ui.CameraController()
    controller.pivot = spy.float3(1.0, 2.0, 3.0)
    assert controller.pivot.x == pytest.approx(1.0)
    assert controller.pivot.y == pytest.approx(2.0)
    assert controller.pivot.z == pytest.approx(3.0)


def test_smoothing_property():
    controller = ui.CameraController()
    assert controller.smoothing is True  # Default.
    controller.smoothing = False
    assert controller.smoothing is False


# --- Smoothing / Inertia ---


def test_orbit_inertia_continues_after_release():
    controller = ui.CameraController()
    controller.smoothing = True

    # Enter orbit and drag quickly.
    controller.handle_mouse_event(
        _mouse_event(
            spy.MouseEventType.button_down,
            button=spy.MouseButton.left,
            pos=spy.float2(100.0, 100.0),
            modifiers=spy.KeyModifierFlags.alt,
        )
    )
    controller.handle_mouse_event(
        _mouse_event(
            spy.MouseEventType.move,
            pos=spy.float2(100.0, 100.0),
            modifiers=spy.KeyModifierFlags.alt,
        )
    )
    controller.update(DT)

    # Rapid drag to build velocity.
    controller.handle_mouse_event(
        _mouse_event(
            spy.MouseEventType.move,
            pos=spy.float2(200.0, 100.0),
            modifiers=spy.KeyModifierFlags.alt,
        )
    )
    controller.update(DT)

    # Release button.
    controller.handle_mouse_event(
        _mouse_event(
            spy.MouseEventType.button_up,
            button=spy.MouseButton.left,
            pos=spy.float2(200.0, 100.0),
        )
    )

    pos_before = controller.transform.translation
    # Should still be moving due to inertia.
    assert controller.update(DT) is True
    pos_after = controller.transform.translation
    assert (
        pos_after.x != pytest.approx(pos_before.x)
        or pos_after.y != pytest.approx(pos_before.y)
        or pos_after.z != pytest.approx(pos_before.z)
    )


def test_disabling_smoothing_clears_velocities():
    controller = ui.CameraController()
    controller.smoothing = True

    # Build some orbit velocity.
    controller.handle_mouse_event(
        _mouse_event(
            spy.MouseEventType.button_down,
            button=spy.MouseButton.left,
            pos=spy.float2(100.0, 100.0),
            modifiers=spy.KeyModifierFlags.alt,
        )
    )
    controller.handle_mouse_event(
        _mouse_event(
            spy.MouseEventType.move,
            pos=spy.float2(100.0, 100.0),
            modifiers=spy.KeyModifierFlags.alt,
        )
    )
    controller.update(DT)
    controller.handle_mouse_event(
        _mouse_event(
            spy.MouseEventType.move,
            pos=spy.float2(200.0, 100.0),
            modifiers=spy.KeyModifierFlags.alt,
        )
    )
    controller.update(DT)
    controller.handle_mouse_event(
        _mouse_event(
            spy.MouseEventType.button_up,
            button=spy.MouseButton.left,
            pos=spy.float2(200.0, 100.0),
        )
    )

    # Disable smoothing - should kill inertia.
    controller.smoothing = False
    assert controller.update(DT) is False


# --- First-person speed modifiers ---


def test_scroll_adjusts_move_speed_in_first_person():
    controller = ui.CameraController()
    controller.smoothing = False
    _setup_first_person(controller)

    initial_speed = controller.move_speed

    # Scroll up to increase speed.
    controller.handle_mouse_event(
        _mouse_event(spy.MouseEventType.scroll, scroll=spy.float2(0.0, 3.0))
    )
    controller.update(DT)

    assert controller.move_speed > initial_speed


# --- Capturing ---


def test_camera_controller_is_captured_starts_false():
    cc = ui.CameraController()
    assert cc.is_captured() is False


def test_camera_controller_is_captured_tracks_capture_state():
    cc = ui.CameraController()

    callback_values: list[bool] = []
    cc.capture_callback = lambda c: callback_values.append(c)

    # Simulate right mouse button down to enter first-person mode.
    down = _mouse_event(
        spy.MouseEventType.button_down,
        button=spy.MouseButton.right,
        pos=spy.float2(100.0, 100.0),
    )
    cc.handle_mouse_event(down)

    assert cc.state == ui.CameraController.State.first_person
    assert cc.is_captured() is True
    assert callback_values[-1] is True

    # Simulate right mouse button up to return to idle.
    up = _mouse_event(
        spy.MouseEventType.button_up,
        button=spy.MouseButton.right,
        pos=spy.float2(110.0, 110.0),
    )
    cc.handle_mouse_event(up)

    assert cc.state == ui.CameraController.State.idle
    assert cc.is_captured() is False
    assert callback_values[-1] is False


def test_capture_callback_called_on_mode_enter_and_exit():
    controller = ui.CameraController()
    captures = []
    controller.capture_callback = lambda capture: captures.append(capture)

    # Enter first-person mode (RMB down) should invoke callback with True.
    controller.handle_mouse_event(
        _mouse_event(
            spy.MouseEventType.button_down,
            button=spy.MouseButton.right,
            pos=spy.float2(0.0, 0.0),
        )
    )
    assert captures == [True]

    # Release RMB should invoke callback with False.
    controller.handle_mouse_event(
        _mouse_event(
            spy.MouseEventType.button_up,
            button=spy.MouseButton.right,
            pos=spy.float2(0.0, 0.0),
        )
    )
    assert captures == [True, False]


def test_capture_callback_default_is_none():
    controller = ui.CameraController()
    assert controller.capture_callback is None


def test_capture_callback_can_be_cleared():
    controller = ui.CameraController()
    captures = []
    controller.capture_callback = lambda capture: captures.append(capture)

    # Clear the callback.
    controller.capture_callback = None

    # Enter first-person mode should not invoke callback.
    controller.handle_mouse_event(
        _mouse_event(
            spy.MouseEventType.button_down,
            button=spy.MouseButton.right,
            pos=spy.float2(0.0, 0.0),
        )
    )
    assert captures == []


if __name__ == "__main__":
    pytest.main([__file__, "-vs"])
