# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validated browser input, independent of rendering and camera controls."""

import json
import math
from collections.abc import Callable
from typing import Any

import slangpy as spy


_FIELDS = {
    "key": {"code", "down", "repeat", "mods"},
    "text": {"text"},
    "button": {"button", "down", "relative", "x", "y", "mods"},
    "motion": {"relative", "x", "y", "mods"},
    "wheel": {"relative", "x", "y", "mode", "px", "py", "mods"},
    "capture": {"enabled"},
    "release": set(),
    "heartbeat": set(),
}

_KEY_NAMES = {
    "Space": "space",
    "Quote": "apostrophe",
    "Comma": "comma",
    "Minus": "minus",
    "Period": "period",
    "Slash": "slash",
    "Semicolon": "semicolon",
    "Equal": "equal",
    "BracketLeft": "left_bracket",
    "Backslash": "backslash",
    "BracketRight": "right_bracket",
    "Backquote": "grave_accent",
    "Escape": "escape",
    "Tab": "tab",
    "Enter": "enter",
    "Backspace": "backspace",
    "Insert": "insert",
    "Delete": "delete",
    "ArrowRight": "right",
    "ArrowLeft": "left",
    "ArrowDown": "down",
    "ArrowUp": "up",
    "PageUp": "page_up",
    "PageDown": "page_down",
    "Home": "home",
    "End": "end",
    "CapsLock": "caps_lock",
    "ScrollLock": "scroll_lock",
    "NumLock": "num_lock",
    "PrintScreen": "print_screen",
    "Pause": "pause",
    "ContextMenu": "menu",
    "NumpadDecimal": "keypad_del",
    "NumpadDivide": "keypad_divide",
    "NumpadMultiply": "keypad_multiply",
    "NumpadSubtract": "keypad_subtract",
    "NumpadAdd": "keypad_add",
    "NumpadEnter": "keypad_enter",
    "NumpadEqual": "keypad_equal",
    **{f"Key{chr(i)}": chr(i).lower() for i in range(ord("A"), ord("Z") + 1)},
    **{f"Digit{i}": f"key{i}" for i in range(10)},
    **{f"Numpad{i}": f"keypad{i}" for i in range(10)},
    **{f"F{i}": f"f{i}" for i in range(1, 13)},
    **{
        f"{key}{side}": f"{side.lower()}_{name}"
        for key, name in [
            ("Shift", "shift"),
            ("Control", "control"),
            ("Alt", "alt"),
            ("Meta", "super"),
        ]
        for side in ["Left", "Right"]
    },
}
_KEY_CODES = {code: getattr(spy.KeyCode, name) for code, name in _KEY_NAMES.items()}
_MOUSE_BUTTONS = (spy.MouseButton.left, spy.MouseButton.middle, spy.MouseButton.right)


def _number(value: Any, bound: float) -> bool:
    return type(value) in (int, float) and abs(value) <= bound and math.isfinite(value)


def _integer(value: Any, low: int, high: int) -> bool:
    return type(value) is int and low <= value <= high


def _text(value: Any, maximum: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) <= maximum
        and not any(0xD800 <= ord(char) <= 0xDFFF for char in value)
    )


class RemoteInput:
    """Validate browser messages and dispatch SGL events on the calling thread.

    Positions use stream pixels. Relative motion advances a virtual cursor.
    Wheel offsets use 100 pixels, three lines, or 100/height pages per notch;
    vertical DOM deltas are negated to match SGL. Unsupported physical keys are
    ignored. Call update with monotonic seconds, and release on transport reset.
    Callbacks must return promptly; failures propagate to the session owner.
    """

    def __init__(
        self,
        generation: str,
        width: int,
        height: int,
        on_keyboard_event: Callable[[spy.KeyboardEvent], Any],
        on_mouse_event: Callable[[spy.MouseEvent], Any],
    ) -> None:
        super().__init__()
        self.generation = generation
        self.width, self.height = width, height
        self.keys: set[str] = set()
        self.buttons: set[int] = set()
        self.sequence = -1
        self.rejected = 0
        self.last_received = 0.0
        self._on_keyboard_event = on_keyboard_event
        self._on_mouse_event = on_mouse_event
        self._pos = spy.float2(0)
        self._relative = False

    def _keyboard(
        self, kind: spy.KeyboardEventType, key: spy.KeyCode, mods: int = 0, codepoint: int = 0
    ) -> None:
        event = spy.KeyboardEvent()
        event.type, event.key = kind, key
        event.mods = spy.KeyModifierFlags(mods)
        event.codepoint = codepoint
        self._on_keyboard_event(event)

    def _mouse(
        self,
        kind: spy.MouseEventType,
        mods: int = 0,
        button: spy.MouseButton = spy.MouseButton.unknown,
        scroll: spy.float2 | None = None,
    ) -> None:
        event = spy.MouseEvent()
        event.type, event.pos, event.button = kind, self._pos, button
        event.mods = spy.KeyModifierFlags(mods)
        event.scroll = scroll if scroll is not None else spy.float2(0)
        self._on_mouse_event(event)

    def release(self) -> None:
        """Synthesize releases, including modifiers, so existing SGL consumers reset."""
        keys, buttons = sorted(self.keys), sorted(self.buttons)
        self.keys.clear()
        self.buttons.clear()
        self._relative = False
        # Attempt every release even if a consumer raises during cleanup.
        error = None
        for code in keys:
            try:
                self._keyboard(spy.KeyboardEventType.key_release, _KEY_CODES[code])
            except Exception as caught:
                error = caught
        for button in buttons:
            try:
                self._mouse(spy.MouseEventType.button_up, button=_MOUSE_BUTTONS[button])
            except Exception as caught:
                error = caught
        if error is not None:
            raise error

    def _validate(self, raw: str) -> dict[str, Any]:
        if len(raw.encode("utf-8")) > 4096:
            raise ValueError("Input message exceeds 4096 bytes")
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError("Input must be an object")
        kind = value.get("type")
        if not isinstance(kind, str) or kind not in _FIELDS:
            raise ValueError("Unknown input type")
        if set(value) != {"v", "generation", "seq", "type"} | _FIELDS[kind]:
            raise ValueError("Unexpected input fields")
        if type(value["v"]) is not int or value["v"] != 1 or value["generation"] != self.generation:
            raise ValueError("Wrong input version or generation")
        if not _integer(value["seq"], self.sequence + 1, 2**53 - 1):
            raise ValueError("Stale input sequence")
        for name in ("down", "relative", "enabled", "repeat"):
            if name in value and type(value[name]) is not bool:
                raise ValueError("Expected boolean")
        if "mods" in value and not _integer(value["mods"], 0, 7):
            raise ValueError("Invalid modifiers")
        if "button" in value and not _integer(value["button"], 0, 2):
            raise ValueError("Invalid mouse button")
        if kind == "key":
            if not _text(value["code"], 128):
                raise ValueError("Invalid key identity")
            if value["repeat"] and not value["down"]:
                raise ValueError("Invalid key state")
        if kind == "text" and not _text(value["text"], 512):
            raise ValueError("Invalid Unicode text")
        if "x" in value:
            bound = 1_000_000 if kind == "wheel" else 8192
            if not _number(value["x"], bound) or not _number(value["y"], bound):
                raise ValueError("Invalid pointer/scroll value")
            if kind != "wheel" and not value.get("relative", False):
                if not (0 <= value["x"] <= self.width and 0 <= value["y"] <= self.height):
                    raise ValueError("Pointer outside video")
        if kind == "wheel" and (
            not _integer(value["mode"], 0, 2)
            or not _number(value["px"], self.width)
            or not _number(value["py"], self.height)
            or value["px"] < 0
            or value["py"] < 0
        ):
            raise ValueError("Invalid wheel units or position")
        return value

    def receive(self, raw: str, now: float) -> None:
        """Reject malformed/stale input and release any held state."""
        try:
            value = self._validate(raw)
        except (ValueError, TypeError, RecursionError):
            self.rejected += 1
            self.release()
            return
        self.sequence = value["seq"]
        self.last_received = now
        kind = value["type"]
        mods = value.get("mods", 0)
        if kind in {"release", "capture"}:
            self.release()
        elif kind == "key":
            code = value["code"]
            key = _KEY_CODES.get(code)
            if key is None:
                return
            if value["down"]:
                # After reset, a repeat cannot recreate a lost key-down.
                if value["repeat"] and code not in self.keys:
                    return
                event_type = (
                    spy.KeyboardEventType.key_repeat
                    if code in self.keys
                    else spy.KeyboardEventType.key_press
                )
                self.keys.add(code)
            else:
                if code not in self.keys:
                    return
                self.keys.remove(code)
                event_type = spy.KeyboardEventType.key_release
            self._keyboard(event_type, key, mods)
        elif kind == "text":
            for char in value["text"]:
                self._keyboard(
                    spy.KeyboardEventType.input, spy.KeyCode.unknown, codepoint=ord(char)
                )
        elif kind == "motion":
            if value["relative"]:
                if not self._relative:
                    self._mouse(spy.MouseEventType.move, mods)
                self._pos = self._pos + spy.float2(value["x"], value["y"])
            else:
                if self._relative:
                    self.release()
                self._pos = spy.float2(value["x"], value["y"])
            self._relative = value["relative"]
            self._mouse(spy.MouseEventType.move, mods)
        elif kind == "button":
            if not value["relative"]:
                self._pos = spy.float2(value["x"], value["y"])
            button = value["button"]
            if value["down"]:
                if button in self.buttons:
                    return
                self.buttons.add(button)
                self._mouse(spy.MouseEventType.button_down, mods, _MOUSE_BUTTONS[button])
                # Prime position-difference controllers before the first drag.
                self._mouse(spy.MouseEventType.move, mods)
            elif button in self.buttons:
                self.buttons.remove(button)
                self._mouse(spy.MouseEventType.button_up, mods, _MOUSE_BUTTONS[button])
        elif kind == "wheel":
            if not value["relative"]:
                self._pos = spy.float2(value["px"], value["py"])
            scale = (0.01, 1 / 3, self.height / 100)[value["mode"]]
            self._mouse(
                spy.MouseEventType.scroll, mods, scroll=spy.float2(value["x"], -value["y"]) * scale
            )

    def update(self, now: float) -> None:
        """Release held input if events/heartbeats stop for more than 1.5 seconds."""
        if (self.keys or self.buttons) and now - self.last_received > 1.5:
            self.release()
