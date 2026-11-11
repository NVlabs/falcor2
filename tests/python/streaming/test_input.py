# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Any

import pytest
import slangpy as spy
from falcor2.streaming.input import RemoteInput


def _send(remote: RemoteInput, event_type: str, **fields: Any) -> None:
    remote.receive(
        json.dumps(
            {"v": 1, "generation": "test", "seq": remote.sequence + 1, "type": event_type, **fields}
        ),
        10.0,
    )


def _key(**fields: Any) -> dict[str, Any]:
    return dict(code="F12", down=True, repeat=False, mods=0) | fields


def test_key_mapping_repeat_text_and_unknown_keys() -> None:
    events = []
    remote = RemoteInput("test", 1280, 720, events.append, lambda event: None)
    for code, expected in [
        ("KeyW", spy.KeyCode.w),
        ("NumpadEnter", spy.KeyCode.keypad_enter),
        ("ShiftRight", spy.KeyCode.right_shift),
        ("Digit1", spy.KeyCode.key1),
        ("Semicolon", spy.KeyCode.semicolon),
        ("F12", spy.KeyCode.f12),
    ]:
        _send(remote, "key", **_key(code=code, mods=5))
        _send(remote, "key", **_key(code=code, repeat=True))
        _send(remote, "key", **_key(code=code, down=False))
        assert [event.key for event in events[-3:]] == [expected] * 3
        assert [event.type for event in events[-3:]] == [
            spy.KeyboardEventType.key_press,
            spy.KeyboardEventType.key_repeat,
            spy.KeyboardEventType.key_release,
        ]
        assert events[-3].has_modifier(spy.KeyModifier.shift)
        assert events[-3].has_modifier(spy.KeyModifier.alt)
    count = len(events)
    _send(remote, "key", **_key(code="Unknown"))
    _send(remote, "key", **_key(repeat=True))
    assert len(events) == count and not remote.keys
    _send(remote, "text", text="A\u65e5\U0001f642")
    assert [e.codepoint for e in events[-3:]] == [65, 0x65E5, 0x1F642]
    assert all(e.type == spy.KeyboardEventType.input for e in events[-3:])


@pytest.mark.parametrize("reset", ["release", "timeout", "capture", "invalid", "transport"])
def test_reset_synthesizes_releases(reset: str) -> None:
    keys, mouse = [], []
    remote = RemoteInput("test", 1280, 720, keys.append, mouse.append)
    _send(remote, "key", **_key(code="ShiftLeft", mods=1))
    _send(remote, "key", **_key(code="KeyW", mods=1))
    _send(remote, "button", relative=False, button=2, down=True, x=100, y=100, mods=1)
    if reset == "timeout":
        remote.update(12)
    elif reset == "capture":
        _send(remote, "capture", enabled=False)
    elif reset == "invalid":
        remote.receive("{}", 10)
    elif reset == "transport":
        remote.release()
    else:
        _send(remote, "release")
    assert not remote.keys and not remote.buttons
    assert all(e.type == spy.KeyboardEventType.key_release for e in keys[-2:])
    assert mouse[-1].type == spy.MouseEventType.button_up
    assert mouse[-1].button == spy.MouseButton.right
    assert mouse[-1].mods == spy.KeyModifierFlags.none
    count = len(keys), len(mouse)
    remote.update(20)
    assert (len(keys), len(mouse)) == count


def test_relative_motion_anchors_and_buttons_keep_virtual_position() -> None:
    events = []
    remote = RemoteInput("test", 1280, 720, lambda event: None, events.append)
    _send(remote, "motion", relative=False, x=100, y=200, mods=0)
    _send(remote, "capture", enabled=True)
    _send(remote, "button", relative=True, button=2, down=True, x=0, y=0, mods=0)
    assert events[-1].pos == spy.float2(100, 200)
    _send(remote, "motion", relative=True, x=-10, y=20, mods=0)
    assert events[-1].pos == spy.float2(90, 220)
    _send(remote, "motion", relative=True, x=5, y=-30, mods=0)
    assert events[-1].pos == spy.float2(95, 190)
    _send(remote, "button", relative=True, button=2, down=False, x=0, y=0, mods=0)
    assert events[-1].pos == spy.float2(95, 190)
    assert events[-1].button == spy.MouseButton.right


@pytest.mark.parametrize("mode, y, expected", [(0, 100, -1), (1, 3, -1), (2, 1, -7.2)])
def test_wheel_units(mode: int, y: float, expected: float) -> None:
    events = []
    remote = RemoteInput("test", 1280, 720, lambda event: None, events.append)
    _send(remote, "wheel", relative=False, x=0, y=y, mode=mode, px=100, py=200, mods=0)
    assert events[-1].type == spy.MouseEventType.scroll
    assert events[-1].scroll.y == pytest.approx(expected)
    assert events[-1].pos == spy.float2(100, 200)


@pytest.mark.parametrize(
    "mutation",
    [
        {"v": True},
        {"generation": "old"},
        {"seq": -1},
        {"seq": True},
        {"code": 1},
        {"mods": 8},
        {"down": 1},
        {"extra": 1},
        {"repeat": True, "down": False},
        {"code": "\ud800"},
    ],
)
def test_invalid_protocol_resets_without_dispatch(mutation: dict[str, Any]) -> None:
    events = []
    remote = RemoteInput("test", 1280, 720, events.append, events.append)
    value = {"v": 1, "generation": "test", "seq": 0, "type": "key", **_key()}
    remote.receive(json.dumps(value | mutation), 10)
    assert remote.rejected == 1 and remote.sequence == -1 and not events


@pytest.mark.parametrize("raw", ["[]", "null", "{", "x" * 4097, "\ud800"])
def test_malformed_input_does_not_escape(raw: str) -> None:
    remote = RemoteInput("test", 1280, 720, lambda event: None, lambda event: None)
    remote.receive(raw, 10)
    assert remote.rejected == 1


@pytest.mark.parametrize(
    "payload",
    [
        dict(type="motion", relative=True, x=10**1000, y=0, mods=0),
        dict(type="motion", relative=True, x=float("nan"), y=0, mods=0),
        dict(type="motion", relative=False, x=1281, y=0, mods=0),
        dict(type="text", text="\udfff"),
        dict(type="text", text="x" * 513),
        dict(type="composition", phase="start", text=""),
        dict(type="button", relative=False, button=3, down=True, x=0, y=0, mods=0),
        dict(type="wheel", relative=False, x=0, y=0, mode=3, px=0, py=0, mods=0),
        dict(type="button", relative="true", button=0, down=True, x=0, y=0, mods=0),
        dict(type="wheel", relative=1, x=0, y=0, mode=0, px=0, py=0, mods=0),
    ],
)
def test_payload_bounds(payload: dict[str, Any]) -> None:
    remote = RemoteInput("test", 1280, 720, lambda event: None, lambda event: None)
    _send(remote, payload["type"], **{k: v for k, v in payload.items() if k != "type"})
    assert remote.rejected == 1


def test_heartbeat_keeps_state_alive_but_stale_sequence_releases() -> None:
    events = []
    remote = RemoteInput("test", 1280, 720, events.append, lambda event: None)
    _send(remote, "key", **_key())
    heartbeat = json.dumps({"v": 1, "generation": "test", "seq": 1, "type": "heartbeat"})
    remote.receive(heartbeat, 11)
    remote.update(12)
    assert remote.keys == {"F12"}
    remote.receive(heartbeat, 12)
    assert remote.rejected == 1 and not remote.keys
    assert events[-1].is_key_release()


def test_release_attempts_all_handlers_when_one_fails() -> None:
    events = []

    def callback(event: spy.KeyboardEvent) -> None:
        events.append(event)
        if event.is_key_release():
            raise RuntimeError("consumer failed")

    remote = RemoteInput("test", 1280, 720, callback, events.append)
    _send(remote, "key", **_key())
    _send(remote, "button", relative=False, button=0, down=True, x=0, y=0, mods=0)
    with pytest.raises(RuntimeError, match="consumer failed"):
        remote.release()
    assert events[-1].is_button_up()
    assert not remote.keys and not remote.buttons


@pytest.mark.parametrize("reset", ["release", "timeout", "invalid", "transport"])
@pytest.mark.parametrize("relative", [False, True])
def test_pointer_mode_after_reset(reset: str, relative: bool) -> None:
    events = []
    remote = RemoteInput("test", 1280, 720, lambda event: None, events.append)
    _send(remote, "motion", relative=False, x=100, y=200, mods=0)
    _send(remote, "capture", enabled=True)
    _send(remote, "button", relative=True, button=0, down=True, x=0, y=0, mods=0)
    if reset == "timeout":
        remote.update(12)
    elif reset == "invalid":
        remote.receive("{}", 10)
    elif reset == "transport":
        remote.release()
    else:
        _send(remote, "release")
    assert not remote.buttons

    # Neither a fresh capture notification nor a motion event precedes recovery.
    _send(remote, "button", relative=relative, button=0, down=True, x=500, y=400, mods=0)
    expected = spy.float2(100, 200) if relative else spy.float2(500, 400)
    assert events[-1].pos == expected
    _send(remote, "wheel", relative=relative, x=0, y=100, mode=0, px=600, py=450, mods=0)
    expected = spy.float2(100, 200) if relative else spy.float2(600, 450)
    assert events[-1].pos == expected
    assert remote.rejected == (1 if reset == "invalid" else 0)
