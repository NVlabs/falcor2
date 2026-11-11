# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
from typing import Optional

import pytest
import slangpy as spy

import falcor2 as f2
import falcor2.ui as ui

DT = 1.0 / 60.0


def _mouse_event(
    event_type: spy.MouseEventType,
    *,
    button: spy.MouseButton = spy.MouseButton.unknown,
    pos: Optional[spy.float2] = None,
    scroll: Optional[spy.float2] = None,
    modifiers: spy.KeyModifierFlags = spy.KeyModifierFlags.none,
) -> spy.MouseEvent:
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
) -> spy.KeyboardEvent:
    event = spy.KeyboardEvent()
    event.type = event_type
    event.key = key
    return event


def _assert_float3_close(
    actual: spy.float3, expected: spy.float3, *, abs_tolerance: float = 1e-6
) -> None:
    assert actual.x == pytest.approx(expected.x, abs=abs_tolerance)
    assert actual.y == pytest.approx(expected.y, abs=abs_tolerance)
    assert actual.z == pytest.approx(expected.z, abs=abs_tolerance)


def _setup_first_person(controller: ui.CameraController) -> None:
    """Enter first-person mode and anchor the mouse."""
    assert controller.handle_mouse_event(
        _mouse_event(
            spy.MouseEventType.button_down,
            button=spy.MouseButton.right,
        )
    )
    assert controller.handle_mouse_event(_mouse_event(spy.MouseEventType.move))
    assert controller.update(DT) is False


def _setup_drag(
    controller: ui.CameraController,
    button: spy.MouseButton,
    *,
    modifiers: spy.KeyModifierFlags = spy.KeyModifierFlags.none,
) -> None:
    """Enter a mouse-drag interaction and anchor the mouse."""
    assert controller.handle_mouse_event(
        _mouse_event(
            spy.MouseEventType.button_down,
            button=button,
            modifiers=modifiers,
        )
    )
    assert controller.handle_mouse_event(_mouse_event(spy.MouseEventType.move, modifiers=modifiers))
    assert controller.update(DT) is False


def _build_orbit_velocity(controller: ui.CameraController) -> None:
    """Build orbit velocity and release the mouse to enable inertia."""
    controller.smoothing = True
    _setup_drag(
        controller,
        spy.MouseButton.left,
        modifiers=spy.KeyModifierFlags.alt,
    )
    assert controller.handle_mouse_event(
        _mouse_event(
            spy.MouseEventType.move,
            pos=spy.float2(100.0, 0.0),
            modifiers=spy.KeyModifierFlags.alt,
        )
    )
    assert controller.update(DT) is True
    assert controller.handle_mouse_event(
        _mouse_event(
            spy.MouseEventType.button_up,
            button=spy.MouseButton.left,
        )
    )


# --- Initial state ---


def test_initial_state():
    controller = ui.CameraController()

    assert controller.transform.translation == spy.float3(0.0, 0.0, 0.0)
    assert controller.transform.rotation == spy.quatf(0.0, 0.0, 0.0, 1.0)
    assert controller.world_up == spy.float3(0.0, 1.0, 0.0)
    assert controller.update(DT) is False


# --- World up ---


def test_world_up_normalizes_arbitrary_direction():
    controller = ui.CameraController()
    controller.world_up = spy.float3(0.0, 2.0, 2.0)

    expected = 1.0 / (2.0**0.5)
    assert controller.world_up.x == pytest.approx(0.0)
    assert controller.world_up.y == pytest.approx(expected)
    assert controller.world_up.z == pytest.approx(expected)


def test_z_up_first_person_yaw_uses_world_z_axis():
    controller = ui.CameraController()
    world_up = spy.float3(0.0, 0.0, 1.0)
    controller.world_up = world_up
    controller.smoothing = False

    transform = f2.Transform()
    transform.rotation = spy.math.quat_from_look_at(spy.float3(0.0, 1.0, 0.0), world_up)
    controller.transform = transform
    _setup_first_person(controller)

    initial_rotation = spy.math.matrix_from_quat(controller.transform.rotation)
    initial_forward = -initial_rotation.get_col(2)
    controller.handle_mouse_event(_mouse_event(spy.MouseEventType.move, pos=spy.float2(50.0, 0.0)))
    assert controller.update(DT) is True

    rotation = spy.math.matrix_from_quat(controller.transform.rotation)
    forward = -rotation.get_col(2)
    _assert_float3_close(rotation.get_col(1), world_up)
    assert spy.math.dot(forward, world_up) == pytest.approx(0.0, abs=1e-6)
    assert spy.math.dot(forward, initial_forward) < 1.0


def test_first_person_vertical_movement_uses_camera_local_up():
    controller = ui.CameraController()
    world_up = spy.float3(0.0, 0.0, 1.0)
    controller.world_up = world_up
    controller.smoothing = False

    transform = f2.Transform()
    transform.rotation = spy.math.quat_from_look_at(
        spy.math.normalize(spy.float3(0.0, 1.0, 1.0)), world_up
    )
    controller.transform = transform
    local_up = spy.math.matrix_from_quat(transform.rotation).get_col(1)
    assert abs(spy.math.dot(local_up, world_up)) < 1.0

    _setup_first_person(controller)
    controller.handle_keyboard_event(_key_event(spy.KeyboardEventType.key_press, spy.KeyCode.e))
    assert controller.update(DT) is True

    _assert_float3_close(controller.transform.translation, local_up * controller.move_speed * DT)


@pytest.mark.parametrize(
    "world_up,position",
    [
        (spy.float3(0.0, 1.0, 0.0), spy.float3(0.0, 0.0, 5.0)),
        (spy.float3(0.0, -1.0, 0.0), spy.float3(0.0, 0.0, 5.0)),
        (spy.float3(0.0, 0.0, 1.0), spy.float3(0.0, -5.0, 0.0)),
        (spy.float3(1.0, 0.0, 0.0), spy.float3(0.0, 0.0, 5.0)),
        (
            spy.math.normalize(spy.float3(1.0, 1.0, 1.0)),
            spy.math.normalize(spy.float3(1.0, -1.0, 0.0)) * 5.0,
        ),
    ],
    ids=["y-up", "negative-y-up", "z-up", "x-up", "arbitrary-up"],
)
def test_orbit_yaw_uses_world_up(world_up: spy.float3, position: spy.float3):
    controller = ui.CameraController()
    controller.smoothing = False
    controller.world_up = world_up

    transform = f2.Transform()
    transform.translation = position
    transform.rotation = spy.math.quat_from_look_at(-spy.math.normalize(position), world_up)
    controller.transform = transform
    controller.pivot = spy.float3(0.0)

    _setup_drag(
        controller,
        spy.MouseButton.left,
        modifiers=spy.KeyModifierFlags.alt,
    )
    initial_position = controller.transform.translation
    controller.handle_mouse_event(
        _mouse_event(
            spy.MouseEventType.move,
            pos=spy.float2(50.0, 0.0),
            modifiers=spy.KeyModifierFlags.alt,
        )
    )
    assert controller.update(DT) is True

    position = controller.transform.translation
    assert spy.math.length(position - initial_position) > 0.0
    assert spy.math.dot(position, world_up) == pytest.approx(0.0, abs=1e-6)
    assert spy.math.length(position) == pytest.approx(5.0)

    rotation = spy.math.matrix_from_quat(controller.transform.rotation)
    forward = -rotation.get_col(2)
    _assert_float3_close(forward, -spy.math.normalize(position))


def test_orbit_pitch_changes_height_and_clamps_away_from_pole():
    controller = ui.CameraController()
    controller.smoothing = False

    transform = f2.Transform()
    transform.translation = spy.float3(0.0, 0.0, 5.0)
    transform.rotation = spy.math.quat_from_look_at(spy.float3(0.0, 0.0, -1.0), controller.world_up)
    controller.transform = transform
    controller.pivot = spy.float3(0.0)
    _setup_drag(
        controller,
        spy.MouseButton.left,
        modifiers=spy.KeyModifierFlags.alt,
    )

    controller.handle_mouse_event(
        _mouse_event(
            spy.MouseEventType.move,
            pos=spy.float2(0.0, 100000.0),
            modifiers=spy.KeyModifierFlags.alt,
        )
    )
    assert controller.update(DT) is True

    position = controller.transform.translation
    direction = spy.math.normalize(position)
    assert position.x == pytest.approx(0.0, abs=1e-6)
    assert spy.math.length(position) == pytest.approx(5.0)
    assert abs(spy.math.dot(direction, controller.world_up)) == pytest.approx(
        math.sin(math.radians(89.0)), abs=1e-5
    )


# --- Scroll ---


def test_scroll_moves_forward():
    controller = ui.CameraController()
    controller.smoothing = False
    event = _mouse_event(
        spy.MouseEventType.scroll,
        scroll=spy.float2(0.0, 1.0),
        pos=spy.float2(0.0, 0.0),
    )
    assert controller.handle_mouse_event(event) is True

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
    assert controller.handle_mouse_event(event) is True
    assert controller.update(DT) is True

    # Scrolling backward should move camera away from pivot (positive Z from origin).
    position = controller.transform.translation
    assert position.z > 0.0


def test_scroll_smoothing_continues_after_input():
    controller = ui.CameraController()
    controller.smoothing = True
    controller.handle_mouse_event(
        _mouse_event(spy.MouseEventType.scroll, scroll=spy.float2(0.0, 1.0))
    )

    assert controller.update(DT) is True
    first_position = controller.transform.translation
    assert controller.update(DT) is True
    assert controller.transform.translation.z < first_position.z


# --- Transform ---


def test_set_and_get_transform():
    controller = ui.CameraController()

    transform = f2.Transform()
    transform.translation = spy.float3(1.0, 2.0, 3.0)
    controller.transform = transform

    assert controller.transform.translation == spy.float3(1.0, 2.0, 3.0)
    _assert_float3_close(controller.pivot, spy.float3(1.0, 2.0, -2.0))


# --- First-person mouse look ---


def test_right_mouse_drag_rotates_after_first_move():
    controller = ui.CameraController()
    _setup_first_person(controller)
    assert controller.state == ui.CameraController.State.first_person
    assert controller.is_interacting() is True

    controller.handle_mouse_event(_mouse_event(spy.MouseEventType.move, pos=spy.float2(16.0, -8.0)))

    assert controller.update(DT) is True
    assert controller.transform.rotation != spy.quatf(0.0, 0.0, 0.0, 1.0)

    assert (
        controller.handle_mouse_event(
            _mouse_event(
                spy.MouseEventType.button_up,
                button=spy.MouseButton.right,
                pos=spy.float2(16.0, -8.0),
            )
        )
        is True
    )
    assert controller.state == ui.CameraController.State.idle
    assert controller.is_interacting() is False


# --- Drag (MMB) ---


def test_middle_mouse_drag_moves_in_view_plane():
    controller = ui.CameraController()
    _setup_drag(controller, spy.MouseButton.middle)
    assert controller.state == ui.CameraController.State.pan

    controller.handle_mouse_event(_mouse_event(spy.MouseEventType.move, pos=spy.float2(10.0, 12.0)))
    assert controller.update(DT) is True

    _assert_float3_close(controller.transform.translation, spy.float3(-0.05, 0.06, 0.0))

    controller.handle_mouse_event(
        _mouse_event(
            spy.MouseEventType.button_up,
            button=spy.MouseButton.middle,
            pos=spy.float2(10.0, 12.0),
        )
    )
    assert controller.state == ui.CameraController.State.idle


# --- Button release returns to idle ---


def test_mouse_button_release_does_not_carry_motion_into_next_drag():
    controller = ui.CameraController()
    _setup_first_person(controller)
    controller.handle_mouse_event(_mouse_event(spy.MouseEventType.move, pos=spy.float2(10.0, 5.0)))

    controller.handle_mouse_event(
        _mouse_event(
            spy.MouseEventType.button_up,
            button=spy.MouseButton.right,
            pos=spy.float2(0.0, 0.0),
        )
    )
    assert controller.state == ui.CameraController.State.idle

    controller.handle_mouse_event(
        _mouse_event(spy.MouseEventType.button_down, button=spy.MouseButton.middle)
    )
    controller.handle_mouse_event(_mouse_event(spy.MouseEventType.move, pos=spy.float2(20.0, 15.0)))

    assert controller.state == ui.CameraController.State.pan
    assert controller.update(DT) is False
    assert controller.transform.rotation == spy.quatf(0.0, 0.0, 0.0, 1.0)
    _assert_float3_close(controller.transform.translation, spy.float3(0.0))


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


# --- WASDQE keyboard movement ---


@pytest.mark.parametrize(
    "key,direction",
    [
        (spy.KeyCode.w, spy.float3(0.0, 0.0, -1.0)),
        (spy.KeyCode.s, spy.float3(0.0, 0.0, 1.0)),
        (spy.KeyCode.a, spy.float3(-1.0, 0.0, 0.0)),
        (spy.KeyCode.d, spy.float3(1.0, 0.0, 0.0)),
        (spy.KeyCode.e, spy.float3(0.0, 1.0, 0.0)),
        (spy.KeyCode.q, spy.float3(0.0, -1.0, 0.0)),
    ],
    ids=["forward", "backward", "left", "right", "up", "down"],
)
def test_first_person_key_movement(key: spy.KeyCode, direction: spy.float3):
    controller = ui.CameraController()
    controller.smoothing = False
    _setup_first_person(controller)

    assert controller.handle_keyboard_event(_key_event(spy.KeyboardEventType.key_press, key))
    assert controller.update(DT) is True

    _assert_float3_close(
        controller.transform.translation,
        direction * controller.move_speed * DT,
    )


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

    initial_position = controller.transform.translation
    initial_pivot = controller.pivot
    initial_distance = controller.orbit_distance
    _setup_drag(
        controller,
        spy.MouseButton.left,
        modifiers=spy.KeyModifierFlags.alt,
    )
    assert controller.state == ui.CameraController.State.orbit

    controller.handle_mouse_event(
        _mouse_event(
            spy.MouseEventType.move,
            pos=spy.float2(50.0, 0.0),
            modifiers=spy.KeyModifierFlags.alt,
        )
    )
    assert controller.update(DT) is True

    position = controller.transform.translation
    assert spy.math.length(position - initial_position) > 0.0
    _assert_float3_close(controller.pivot, initial_pivot)
    assert spy.math.length(position - controller.pivot) == pytest.approx(initial_distance)

    rotation = spy.math.matrix_from_quat(controller.transform.rotation)
    forward = -rotation.get_col(2)
    _assert_float3_close(forward, spy.math.normalize(controller.pivot - position))

    controller.handle_mouse_event(
        _mouse_event(
            spy.MouseEventType.button_up,
            button=spy.MouseButton.left,
            pos=spy.float2(50.0, 0.0),
        )
    )
    assert controller.state == ui.CameraController.State.idle


# --- Dolly (Alt+RMB) ---


def test_dolly_changes_distance():
    controller = ui.CameraController()
    controller.smoothing = False

    initial_pivot = controller.pivot
    _setup_drag(
        controller,
        spy.MouseButton.right,
        modifiers=spy.KeyModifierFlags.alt,
    )
    assert controller.state == ui.CameraController.State.dolly

    controller.handle_mouse_event(
        _mouse_event(
            spy.MouseEventType.move,
            pos=spy.float2(30.0, 30.0),
            modifiers=spy.KeyModifierFlags.alt,
        )
    )
    assert controller.update(DT) is True

    _assert_float3_close(controller.pivot, initial_pivot)
    _assert_float3_close(controller.transform.translation, spy.float3(0.0, 0.0, -1.5))
    assert controller.orbit_distance == pytest.approx(3.5)

    # Release RMB -> idle.
    controller.handle_mouse_event(
        _mouse_event(
            spy.MouseEventType.button_up,
            button=spy.MouseButton.right,
            pos=spy.float2(30.0, 30.0),
        )
    )
    assert controller.state == ui.CameraController.State.idle


# --- Track (Alt+MMB) ---


def test_track_pans_camera_and_pivot():
    controller = ui.CameraController()
    controller.smoothing = False

    initial_pos = controller.transform.translation
    initial_pivot = controller.pivot

    _setup_drag(
        controller,
        spy.MouseButton.middle,
        modifiers=spy.KeyModifierFlags.alt,
    )
    assert controller.state == ui.CameraController.State.track

    controller.handle_mouse_event(
        _mouse_event(
            spy.MouseEventType.move,
            pos=spy.float2(20.0, 10.0),
            modifiers=spy.KeyModifierFlags.alt,
        )
    )
    assert controller.update(DT) is True

    expected_delta = spy.float3(-0.1, 0.05, 0.0)
    _assert_float3_close(controller.transform.translation, initial_pos + expected_delta)
    _assert_float3_close(controller.pivot, initial_pivot + expected_delta)

    # Release MMB -> idle.
    controller.handle_mouse_event(
        _mouse_event(
            spy.MouseEventType.button_up,
            button=spy.MouseButton.middle,
            pos=spy.float2(20.0, 10.0),
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
    _assert_float3_close(controller.pivot, spy.float3(0.0, 0.0, -10.0))


def test_orbit_distance_clamped():
    controller = ui.CameraController()
    controller.orbit_distance = 0.001
    assert controller.orbit_distance == pytest.approx(0.01, abs=1e-6)


def test_pivot_property():
    controller = ui.CameraController()
    controller.pivot = spy.float3(1.0, 2.0, 3.0)
    _assert_float3_close(controller.pivot, spy.float3(1.0, 2.0, 3.0))
    assert controller.orbit_distance == pytest.approx(math.sqrt(14.0))


def test_smoothing_property():
    controller = ui.CameraController()
    assert controller.smoothing is True  # Default.
    controller.smoothing = False
    assert controller.smoothing is False


# --- Smoothing / Inertia ---


def test_orbit_inertia_continues_after_release():
    controller = ui.CameraController()
    _build_orbit_velocity(controller)

    pos_before = controller.transform.translation
    assert controller.update(DT) is True
    pos_after = controller.transform.translation
    assert spy.math.length(pos_after - pos_before) > 0.0


def test_disabling_smoothing_clears_orbit_velocity():
    controller = ui.CameraController()
    _build_orbit_velocity(controller)

    controller.smoothing = False
    assert controller.update(DT) is False


def test_setting_world_up_clears_orbit_velocity():
    controller = ui.CameraController()
    _build_orbit_velocity(controller)

    controller.world_up = spy.float3(0.0, 0.0, 1.0)
    assert controller.update(DT) is False


def test_setting_world_up_clears_look_velocity():
    controller = ui.CameraController()
    controller.smoothing = True
    _setup_first_person(controller)
    controller.handle_mouse_event(_mouse_event(spy.MouseEventType.move, pos=spy.float2(100.0, 0.0)))
    assert controller.update(DT) is True

    controller.world_up = spy.float3(0.0, 0.0, 1.0)
    assert controller.update(DT) is False


# --- First-person speed modifiers ---


@pytest.mark.parametrize(
    "modifier,factor",
    [
        (spy.KeyCode.left_shift, 10.0),
        (spy.KeyCode.left_control, 0.1),
    ],
    ids=["fast", "slow"],
)
def test_first_person_speed_modifier(modifier: spy.KeyCode, factor: float):
    controller = ui.CameraController()
    controller.smoothing = False
    _setup_first_person(controller)

    assert controller.handle_keyboard_event(_key_event(spy.KeyboardEventType.key_press, modifier))
    assert controller.handle_keyboard_event(
        _key_event(spy.KeyboardEventType.key_press, spy.KeyCode.w)
    )
    assert controller.update(DT) is True

    expected_distance = controller.move_speed * factor * DT
    _assert_float3_close(
        controller.transform.translation,
        spy.float3(0.0, 0.0, -expected_distance),
    )


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


# --- Interaction ownership ---


def test_camera_controller_is_interacting_starts_false():
    cc = ui.CameraController()
    assert cc.is_interacting() is False


def test_camera_keyboard_consumption_is_selective():
    controller = ui.CameraController()

    assert (
        controller.handle_keyboard_event(_key_event(spy.KeyboardEventType.key_press, spy.KeyCode.w))
        is False
    )

    controller.handle_mouse_event(
        _mouse_event(
            spy.MouseEventType.button_down,
            button=spy.MouseButton.right,
            pos=spy.float2(0.0, 0.0),
        )
    )

    assert (
        controller.handle_keyboard_event(_key_event(spy.KeyboardEventType.key_press, spy.KeyCode.w))
        is True
    )
    assert (
        controller.handle_keyboard_event(
            _key_event(spy.KeyboardEventType.key_release, spy.KeyCode.w)
        )
        is True
    )
    assert (
        controller.handle_keyboard_event(
            _key_event(spy.KeyboardEventType.key_press, spy.KeyCode.space)
        )
        is False
    )


def test_idle_move_key_press_does_not_prime_first_person():
    controller = ui.CameraController()
    controller.smoothing = False
    initial_position = controller.transform.translation

    assert (
        controller.handle_keyboard_event(_key_event(spy.KeyboardEventType.key_press, spy.KeyCode.w))
        is False
    )
    assert (
        controller.handle_mouse_event(
            _mouse_event(
                spy.MouseEventType.button_down,
                button=spy.MouseButton.right,
                pos=spy.float2(0.0, 0.0),
            )
        )
        is True
    )

    assert controller.update(DT) is False
    position = controller.transform.translation
    assert position.x == pytest.approx(initial_position.x)
    assert position.y == pytest.approx(initial_position.y)
    assert position.z == pytest.approx(initial_position.z)

    assert (
        controller.handle_keyboard_event(_key_event(spy.KeyboardEventType.key_press, spy.KeyCode.w))
        is True
    )
    assert controller.update(DT) is True


def test_cancel_interaction_returns_to_idle_and_clears_keys():
    controller = ui.CameraController()
    controller.smoothing = False

    controller.handle_mouse_event(
        _mouse_event(
            spy.MouseEventType.button_down,
            button=spy.MouseButton.right,
            pos=spy.float2(0.0, 0.0),
        )
    )
    controller.handle_keyboard_event(_key_event(spy.KeyboardEventType.key_press, spy.KeyCode.w))
    controller.cancel_interaction()

    assert controller.state == ui.CameraController.State.idle
    assert controller.is_interacting() is False
    assert controller.update(DT) is False


if __name__ == "__main__":
    pytest.main([__file__, "-vs"])
