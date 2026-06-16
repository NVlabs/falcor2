# SPDX-License-Identifier: Apache-2.0

import slangpy as spy
from typing import Callable, Union
from enum import Enum

from falcor2.minitracer.camera import Camera


class State(Enum):
    IDLE = 0
    FIRST_PERSON = 1
    DRAG = 2


class CameraController:
    MOVE_KEYS = {
        spy.KeyCode.a: spy.float3(-1, 0, 0),
        spy.KeyCode.d: spy.float3(1, 0, 0),
        spy.KeyCode.e: spy.float3(0, 1, 0),
        spy.KeyCode.q: spy.float3(0, -1, 0),
        spy.KeyCode.w: spy.float3(0, 0, 1),
        spy.KeyCode.s: spy.float3(0, 0, -1),
    }
    MOVE_FAST_FACTOR = 10.0
    MOVE_SLOW_FACTOR = 0.1

    def __init__(self, camera: Camera):
        super().__init__()
        self.camera = camera
        self.on_disable_cursor: Union[Callable[[], None], None] = None
        self.on_enable_cursor: Union[Callable[[], None], None] = None

        self.move_speed = 5.0
        self.rotate_speed = 0.002

        self._state = State.IDLE
        self._mouse_pos = spy.float2()
        self._mouse_delta = spy.float2()
        self._first_mouse_delta = True
        self._mouse_scroll_delta = spy.float2()

        self._key_state = {k: False for k in CameraController.MOVE_KEYS.keys()}
        self._shift_down = False
        self._ctrl_down = False

    def handle_keyboard_event(self, event: spy.KeyboardEvent):
        if event.is_key_press() or event.is_key_release():
            down = event.is_key_press()
            if event.key in CameraController.MOVE_KEYS:
                self._key_state[event.key] = down
            elif event.key == spy.KeyCode.left_shift:
                self._shift_down = down
            elif event.key == spy.KeyCode.left_control:
                self._ctrl_down = down

    def handle_mouse_event(self, event: spy.MouseEvent):
        self._shift_down = event.has_modifier(spy.KeyModifier.shift)
        self._ctrl_down = event.has_modifier(spy.KeyModifier.ctrl)

        if self._state == State.IDLE:
            self._first_mouse_delta = True
            if event.is_button_down():
                if event.button == spy.MouseButton.right:
                    self._state = State.FIRST_PERSON
                    self._disable_cursor()
                elif event.button == spy.MouseButton.middle:
                    self._state = State.DRAG
                    self._disable_cursor()
        elif self._state == State.FIRST_PERSON:
            if event.is_button_up() and event.button == spy.MouseButton.right:
                self._state = State.IDLE
                self._enable_cursor()
        elif self._state == State.DRAG:
            if event.is_button_up() and event.button == spy.MouseButton.middle:
                self._state = State.IDLE
                self._enable_cursor()

        if event.is_move():
            self._mouse_delta = event.pos - self._mouse_pos
            if self._first_mouse_delta:
                self._mouse_delta = spy.float2()
                self._first_mouse_delta = False
            self._mouse_pos = event.pos

        if event.is_scroll():
            self._mouse_scroll_delta = event.scroll

    def update(self, dt: float) -> bool:
        changed = False
        if self._state == State.IDLE:
            changed = self._update_scroll(dt)
        elif self._state == State.FIRST_PERSON:
            changed = self._update_first_person(dt)
        elif self._state == State.DRAG:
            changed = self._update_drag(dt)

        self._mouse_delta = spy.float2()
        self._mouse_scroll_delta = spy.float2()
        return changed

    def _update_scroll(self, dt: float) -> bool:
        changed = False

        pos = self.camera.transform.pos
        rot = spy.math.matrix_from_quat(self.camera.transform.rot)
        fwd = rot.get_row(2)

        if self._mouse_scroll_delta.y != 0:
            pos += fwd * self._mouse_scroll_delta.y
            self.camera.transform.pos = pos
            changed = True

        return changed

    def _update_first_person(self, dt: float) -> bool:
        changed = False

        pos = self.camera.transform.pos
        rot = spy.math.matrix_from_quat(self.camera.transform.rot)
        fwd = -rot.get_col(2)
        right = rot.get_col(0)
        up = rot.get_col(1)

        # Move
        move_delta = spy.float3()
        for key, state in self._key_state.items():
            if state:
                move_delta += CameraController.MOVE_KEYS[key]
        speed = self.move_speed
        if self._shift_down:
            speed = speed * CameraController.MOVE_FAST_FACTOR
        if self._ctrl_down:
            speed = speed * CameraController.MOVE_SLOW_FACTOR
        move_delta = move_delta * speed
        if spy.math.length(move_delta) > 0:
            offset = right * move_delta.x
            offset += up * move_delta.y
            offset += fwd * move_delta.z
            pos += offset * dt
            self.camera.transform.pos = pos
            changed = True

        # Rotate
        if spy.math.length(self._mouse_delta) > 0:
            horiz = spy.math.quat_from_angle_axis(
                -self._mouse_delta.x * self.rotate_speed, spy.float3(0, 1, 0)
            )
            vert = spy.math.quat_from_angle_axis(
                -self._mouse_delta.y * self.rotate_speed, spy.float3(1, 0, 0)
            )
            self.camera.transform.rot = spy.math.mul(
                spy.math.mul(horiz, self.camera.transform.rot), vert
            )
            changed = True

        return changed

    def _update_drag(self, dt: float) -> bool:
        changed = False

        pos = self.camera.transform.pos
        rot = spy.math.matrix_from_quat(self.camera.transform.rot)
        right = rot.get_row(0)
        up = rot.get_row(1)

        # Drag
        if spy.math.length(self._mouse_delta) > 0:
            offset = -right * self._mouse_delta.x
            offset += up * self._mouse_delta.y
            pos += offset * 0.001
            self.camera.transform.pos = pos
            changed = True

        return changed

    def _disable_cursor(self):
        if self.on_disable_cursor:
            self.on_disable_cursor()

    def _enable_cursor(self):
        if self.on_enable_cursor:
            self.on_enable_cursor()
