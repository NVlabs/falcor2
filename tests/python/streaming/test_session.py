# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import threading
from pathlib import Path
import time
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import pytest
import slangpy as spy
from falcor2 import streaming
from falcor2.streaming.demo import PatternRenderer
from falcor2.streaming.config import ServerConfig
from falcor2.streaming.session import _STAT_FIELDS
from falcor2.testing.helpers import DEFAULT_DEVICE_TYPES, get_device


def test_slow_render_loop_captures_when_transport_has_drained(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from falcor2.streaming import session as module

    now = [0.0]
    native = SimpleNamespace(**{**dict.fromkeys(_STAT_FIELDS, 0), "connected": True})
    peer = Mock()
    peer.stats.return_value = native
    peer.poll_keyframe_request.return_value = False
    peer.poll_input_reset.return_value = False
    peer.poll_input.return_value = None
    encoder = Mock()
    encoder.try_submit.return_value = True
    packet = SimpleNamespace(
        byte_size=1000, gpu_prepare_ms=0.0, encode_wall_ms=0.0, submit_to_output_ms=0.0
    )

    def send(packet: object) -> None:
        native.queue_frames = 1

    peer.send.side_effect = send
    monkeypatch.setattr(module.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(module.streaming, "VideoEncoder", Mock(return_value=encoder))
    monkeypatch.setattr(module.streaming, "WebRTCPeer", Mock(return_value=peer))
    device = SimpleNamespace(info=SimpleNamespace(type=spy.DeviceType.d3d12))
    connection = module.StreamingConnection(cast(spy.Device, device), ServerConfig(fps=30), "test")
    connection.start(lambda event: None, lambda event: None)
    try:
        for frame in range(10):
            encoder.poll_output.side_effect = [packet, None] if frame else [None]
            connection.update()
            now[0] += 0.1  # A slow renderer lets the previous packet finish sending.
            native.queue_frames = 0
            assert connection.capture(cast(spy.Texture, object()), streaming.InputTransfer.srgb)
        assert connection.stats()["captured"] == 10
        assert connection.stats()["transport_skips"] == 0
        # Real queued work or a long paced tail must still block new captures.
        for queue_frames, delay in ((1, 0), (0, 100)):
            now[0] += 0.1
            native.queue_frames, native.pacing_delay_ms = queue_frames, delay
            assert not connection.has_capacity
            assert not connection.capture(cast(spy.Texture, object()), streaming.InputTransfer.srgb)
        assert encoder.try_submit.call_count == 10
        assert connection.stats()["transport_skips"] == 2
    finally:
        connection.close()


@pytest.fixture(params=DEFAULT_DEVICE_TYPES)
def streaming_device(request: pytest.FixtureRequest) -> spy.Device:
    if not spy.Device.enumerate_adapters(request.param):
        pytest.skip(f"No GPU available for {request.param}")
    return get_device(request.param)


def test_pattern_renderer_animates(streaming_device: spy.Device) -> None:
    import numpy as np

    renderer = PatternRenderer(streaming_device, 256, 144)
    try:
        first = renderer.render(0, 0.0).to_numpy().copy()
        second = renderer.render(1, 1 / 30).to_numpy()
        np.testing.assert_array_equal(first[:72], second[:72])
        assert np.any(first[72:] != second[72:])
    finally:
        renderer.close()
    with pytest.raises(RuntimeError, match="Renderer is closed"):
        renderer.render(2, 0.0)


@pytest.mark.parametrize("transport_stalled", [False, True])
@pytest.mark.parametrize("fps", [30, 60])
def test_stalled_encoder_bounds_capture_and_delivers_input(
    monkeypatch: pytest.MonkeyPatch, transport_stalled: bool, fps: int
) -> None:
    from falcor2.streaming import session as module

    drain = threading.Event()
    render_calls: list[int] = []
    captured_iterations: list[int] = []
    input_events: list[spy.KeyboardEvent] = []
    input_threads: list[int] = []
    updates: list[float] = []
    statistics_calls: list[float] = []
    connected_after = time.monotonic() + 0.15
    input_sent = False
    input_ready = threading.Event()
    active_session: module.StreamingSession | None = None

    class Resource:
        info = SimpleNamespace(type=spy.DeviceType.cuda)

        def close(self) -> None:
            pass

    class Encoder(Resource):
        def __init__(self, device: object, config: streaming.VideoEncoderConfig) -> None:
            super().__init__()
            assert config.fps == fps
            assert config.target_bitrate == config.max_bitrate == 50000000
            assert (config.width, config.height) == (1920, 1080)
            self.pending = 0

        def poll_output(self) -> object | None:
            if drain.is_set() and self.pending:
                self.pending -= 1
                return SimpleNamespace(
                    byte_size=1000, gpu_prepare_ms=0.2, encode_wall_ms=1.0, submit_to_output_ms=2.0
                )
            return None

        def try_submit(self, image: int, *args: object) -> bool:
            assert self.pending < 3
            # Every capture must use the newest completed render, including
            # after encoder backpressure. Iterations stand in for image data.
            assert image == len(render_calls)
            assert not captured_iterations or image > captured_iterations[-1]
            captured_iterations.append(image)
            self.pending += 1
            return True

    class Peer(Resource):
        def __init__(self, config: streaming.WebRTCConfig) -> None:
            super().__init__()
            assert config.enable_input
            assert config.max_bitrate == 50000000
            assert config.fps == fps
            assert (config.width, config.height) == (1920, 1080)

        def set_answer(self, sdp: str) -> None:
            pass

        def set_target_bitrate(self, bits_per_second: int) -> None:
            assert bits_per_second > 0

        def set_pacing_bitrate(self, bits_per_second: int) -> None:
            assert bits_per_second == 50000000

        def poll_offer(self) -> None:
            return None

        def stats(self) -> SimpleNamespace:
            stats: dict[str, Any] = dict.fromkeys(_STAT_FIELDS, 0)
            stats.update(connected=time.monotonic() >= connected_after, failed=False, diagnostic="")
            stats["in_flight_frames"] = int(transport_stalled and drain.is_set())
            stats["pacing_delay_ms"] = 100 if stats["in_flight_frames"] else 0
            return SimpleNamespace(**stats)

        def poll_keyframe_request(self) -> bool:
            return False

        def poll_input_reset(self) -> bool:
            return False

        def poll_input(self) -> str | None:
            nonlocal input_sent
            if input_ready.is_set() and not input_sent:
                import json

                assert active_session is not None
                input_sent = True
                return json.dumps(
                    {
                        "v": 1,
                        "generation": active_session.generation,
                        "seq": 0,
                        "type": "key",
                        "code": "F12",
                        "down": True,
                        "repeat": False,
                        "mods": 0,
                    }
                )
            return None

        def send(self, packet: object) -> bool:
            return True

    class Renderer:
        def __init__(self, *args: object) -> None:
            super().__init__()

        def render(self, frame_id: int, dt: float) -> int:
            time.sleep(0.002)  # Model completion of one bounded GPU iteration.
            render_calls.append(frame_id)
            return len(render_calls)

        def stats(self) -> dict[str, Any]:
            statistics_calls.append(time.monotonic())
            return {"mode": "test", "rendered": -1, "render_wall_ms": -1}

        def handle_keyboard_event(self, event: spy.KeyboardEvent) -> None:
            input_events.append(event)
            input_threads.append(threading.get_ident())

        def handle_mouse_event(self, event: spy.MouseEvent) -> None:
            pass

        def close(self) -> None:
            pass

        def update(self, dt: float) -> None:
            updates.append(dt)

    monkeypatch.setattr(module.f2, "create_device", lambda **kwargs: Resource())
    monkeypatch.setattr(module.streaming, "VideoEncoder", Encoder)
    monkeypatch.setattr(module.streaming, "WebRTCPeer", Peer)
    config = ServerConfig(
        fps=fps,
        width=1920,
        height=1080,
        bitrate=50000000,
        pacing_bitrate=50000000,
    )

    # This double passes render counters to the fake encoder instead of textures.
    session = active_session = module.StreamingSession(
        config, cast(module.RendererFactory, Renderer)
    )
    input_ready.set()
    try:
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            _, stats = session.snapshot()
            assert not stats.get("failed"), stats
            if stats.get("busy_drops", 0) >= 3:
                break
            time.sleep(0.01)
        assert stats["captured"] == 3
        assert stats["busy_drops"] >= 3
        assert stats["rendered"] == 3 and stats["render_wall_ms"] >= 0
        # Negotiation time must not advance the renderer's simulation.
        assert updates[0] == 0.0
        assert any(dt > 0 for dt in updates[1:])
        assert len(statistics_calls) >= 2
        assert min(b - a for a, b in zip(statistics_calls, statistics_calls[1:])) >= 0.2
        assert input_events[0].key == spy.KeyCode.f12
        assert input_threads[0] != threading.get_ident()
        assert len(render_calls) == 4  # Pattern warm-up plus the three bounded slots.
        drain.set()
        if transport_stalled:
            update_count = len(updates)
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                _, stats = session.snapshot()
                if stats.get("transport_skips", 0) > 0:
                    break
                time.sleep(0.01)
            assert not stats.get("failed"), stats
            assert stats["captured"] == 3
            assert stats["transport_skips"] > 0
            assert len(render_calls) == 4
            assert len(updates) > update_count
            assert stats["capture_skips"] == stats["busy_drops"] + stats["transport_skips"]
            return
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            _, stats = session.snapshot()
            assert not stats.get("failed"), stats
            if stats.get("captured", 0) > 3:
                break
            time.sleep(0.01)
        assert stats["captured"] > 3
    finally:
        session.close()
        assert input_events[-1].is_key_release()


@pytest.mark.parametrize(
    "settings, diagnostic",
    [
        ({"bitrate": 500000}, "minimum"),
        ({"min_bitrate": 99999}, "minimum"),
        ({"bitrate": 99999}, "minimum"),
        ({"bitrate": 50000001}, "bitrate"),
        ({"min_bitrate": 9000000, "bitrate": 8000000}, "minimum"),
        ({"pacing_bitrate": -1}, "Pacing bitrate"),
        ({"pacing_bitrate": 99999}, "Pacing bitrate"),
        ({"pacing_bitrate": 100000001}, "Pacing bitrate"),
        ({"fps": 0}, "FPS"),
        ({"fps": -1}, "FPS"),
        ({"fps": 61}, "FPS"),
        ({"width": 15}, "Dimensions"),
        ({"width": 1281}, "Dimensions"),
        ({"height": 721}, "Dimensions"),
        ({"width": 3842}, "Dimensions"),
        ({"height": 2162}, "Dimensions"),
        ({"port_begin": 50101}, "ICE port range"),
        ({"token": "x" * 257}, "256 characters"),
        ({"force_relay": True}, "TURN"),
        ({"port": -1}, "HTTP port"),
        ({"port": 65536}, "HTTP port"),
        ({"behind_proxy": True, "origins": set()}, "HTTPS origins"),
        ({"cert": Path("certificate.pem")}, "both cert and key"),
    ],
)
def test_programmatic_config_rejects_invalid_settings(
    settings: dict[str, Any],
    diagnostic: str,
) -> None:
    with pytest.raises(ValueError, match=diagnostic):
        ServerConfig(**settings)


@pytest.mark.parametrize("kind", ["motion", "wheel"])
def test_camera_demo_preserves_batched_mouse_input(
    monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    import json
    import numpy as np
    from falcor2.streaming.demo import CameraRenderer
    from falcor2.streaming.input import RemoteInput

    # Input behavior needs the real controller, but no GPU resources.
    monkeypatch.setattr(PatternRenderer, "__init__", lambda *args: None)

    def run(count: int) -> tuple[list[float], list[float]]:
        renderer = CameraRenderer(Mock(spec=spy.Device), 1280, 720)
        remote = RemoteInput(
            "test", 1280, 720, renderer.handle_keyboard_event, renderer.handle_mouse_event
        )

        def send(event_type: str, **fields: object) -> None:
            remote.receive(
                json.dumps(
                    dict(v=1, generation="test", seq=remote.sequence + 1, type=event_type, **fields)
                ),
                10.0,
            )

        if kind == "motion":
            send("button", relative=False, button=2, down=True, x=0, y=0, mods=0)
            send("key", code="KeyW", down=True, repeat=False, mods=0)
        for index in range(count):
            if kind == "motion":
                send("motion", relative=False, x=20 * (index + 1) / count, y=0, mods=0)
            else:
                send("wheel", relative=False, x=0, y=300 / count, mode=0, px=0, py=0, mods=0)
        if kind == "motion":
            assert renderer.stats()["camera_position"] == [0, 1, 5]
        renderer.update(1 / 30)
        rotation = renderer.controller.transform.rotation
        return renderer.stats()["camera_position"], [rotation.x, rotation.y, rotation.z, rotation.w]

    single, batched = run(1), run(2)
    np.testing.assert_allclose(batched[0], single[0], atol=1e-6)
    np.testing.assert_allclose(batched[1], single[1], atol=1e-6)
    if kind == "motion":
        assert abs(single[1][1]) > 0.01
    else:
        assert single[0] != [0, 1, 5]


def test_camera_demo_uses_sgl_input_and_release_stops_motion(streaming_device: spy.Device) -> None:
    import json
    import numpy as np
    from falcor2.streaming.demo import CameraRenderer
    from falcor2.streaming.input import RemoteInput

    renderer = CameraRenderer(streaming_device, 256, 144)
    remote = RemoteInput(
        "camera", 256, 144, renderer.handle_keyboard_event, renderer.handle_mouse_event
    )

    def send(kind: str, **fields: object) -> None:
        remote.receive(
            json.dumps(
                dict(v=1, generation="camera", seq=remote.sequence + 1, type=kind, **fields)
            ),
            time.monotonic(),
        )

    try:
        first = renderer.render(0, 0).to_numpy().copy()
        send("capture", enabled=True)
        send("button", relative=True, button=2, down=True, x=100, y=70, mods=0)
        send("key", code="KeyW", down=True, repeat=False, mods=0)
        renderer.update(0.2)
        assert renderer.stats()["camera_position"][2] < 5
        send("motion", relative=True, x=30, y=10, mods=0)
        renderer.update(1 / 30)
        second = renderer.render(1, 1 / 30).to_numpy().copy()
        assert np.mean(np.abs(first - second)) > 0.01
        remote.release()
        stopped = renderer.stats()["camera_position"]
        renderer.update(0.2)
        assert renderer.stats()["camera_position"] == stopped
        third = renderer.render(2, 0.2).to_numpy()
        np.testing.assert_array_equal(second, third)
    finally:
        renderer.close()

    with pytest.raises(RuntimeError, match="Renderer is closed"):
        renderer.render(3, 0.0)


@pytest.mark.parametrize("failure", ["factory", "render", "close"])
def test_renderer_failure_closes_resources_on_owner_thread(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    from falcor2.streaming import session as module

    calls = []

    def record(name: str) -> None:
        calls.append((name, threading.get_ident()))

    class Device:
        info = SimpleNamespace(type=spy.DeviceType.cuda)

        def close(self) -> None:
            record("device.close")

    class Encoder:
        def __init__(self, device: Device, config: object) -> None:
            super().__init__()
            record("encoder.create")

        def close(self) -> None:
            record("encoder.close")

    class Renderer:
        def __init__(self, device: Device, width: int, height: int) -> None:
            super().__init__()
            assert isinstance(device, Device) and (width, height) == (1280, 720)
            record("renderer.create")
            if failure == "factory":
                raise RuntimeError("factory failed")

        def render(self, frame_id: int, dt: float) -> None:
            raise RuntimeError("render failed")

        def close(self) -> None:
            record("renderer.close")
            if failure == "close":
                raise RuntimeError("close failed")

    monkeypatch.setattr(module.f2, "create_device", lambda **kwargs: Device())
    monkeypatch.setattr(module.streaming, "VideoEncoder", Encoder)
    # This partial double only implements initialization and failing teardown.
    session = module.StreamingSession(ServerConfig(), cast(module.RendererFactory, Renderer))
    session.close()
    _, stats = session.snapshot()
    assert stats["failed"]
    expected = ["encoder.create", "renderer.create", "encoder.close"]
    if failure != "factory":
        expected.append("renderer.close")
    expected.append("device.close")
    assert [name for name, thread in calls] == expected
    assert len({thread for name, thread in calls}) == 1
    assert calls[0][1] != threading.get_ident()
