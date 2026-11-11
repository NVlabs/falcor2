# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Owner-thread rendering, input, and streaming session lifecycle."""

from __future__ import annotations

import queue
import secrets
import threading
import time
from typing import Any, Protocol
from urllib.parse import urlsplit
from collections import deque

import slangpy as spy
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Callable

import falcor2 as f2
from falcor2 import streaming
from falcor2.streaming.config import ServerConfig
from falcor2.streaming.input import RemoteInput


class _BitratePolicy:
    """Conservative LAN policy, not a production congestion controller.

    Sample once per second. Sparse encoded output does not measure link capacity.
    A fresh falling REMB under load or confirmed interval loss cuts promptly;
    queue pressure needs two samples. Three healthy reports allow recovery.
    RTCP loss fractions are in units of 1/256, not percent.
    """

    def __init__(self, minimum: int, maximum: int) -> None:
        super().__init__()
        self.minimum, self.maximum, self.target = minimum, maximum, maximum
        self._bad = self._good = 0
        self._reports = self._recoveries = 0
        self._remb_reports = self._remb = 0
        self.reason = "startup"
        self.loss_fraction = 0
        self._receiver_sequence: int | None = None
        self._receiver_lost = 0

    def update(self, stats: dict[str, Any]) -> int:
        fresh = stats["receiver_reports"] > self._reports
        fresh_remb = stats["receiver_bitrate_reports"] > self._remb_reports
        remb = stats["receiver_bitrate"]
        actual = stats["encoded_bitrate"]
        loss = stats["loss_fraction"]
        receiving = True
        interval_congestion = False
        sequence = stats.get("receiver_sequence")
        if fresh and sequence is not None:
            lost = stats["receiver_lost"]
            if self._receiver_sequence is not None:
                expected = (sequence - self._receiver_sequence) & 0xFFFFFFFF
                # RR fractions describe only the last report's short interval.
                # Cumulative deltas cover the entire policy tick and credit late
                # repairs instead of aliasing alternating lossy/clean bursts.
                loss = (
                    min(255, max(0, lost - self._receiver_lost) * 256 // expected)
                    if expected
                    else 0
                )
                receiving = expected > 0
                interval_congestion = expected > 0 and loss >= 5
            self._receiver_sequence, self._receiver_lost = sequence, lost
        if fresh:
            self.loss_fraction = loss
        overloaded = stats["recoveries"] > self._recoveries or stats["queue_age_ms"] > 100
        bad = overloaded or (fresh and loss >= 5)
        # REMB can track the application's sparse output, rather than available
        # bandwidth. Act on a significant downward change under load, not its
        # absolute value at startup or a cached value on every policy tick.
        falling = (
            fresh_remb
            and 0 < remb < self._remb * 0.8
            and actual >= self.target * 0.5
            and actual > remb * 1.1
        )
        self._bad = self._bad + 1 if bad else 0
        healthy = fresh and receiving and not bad and not falling and loss <= 1 and actual > 0
        self._good = self._good + 1 if healthy else 0
        self.reason = "holding"
        if falling or interval_congestion or self._bad >= 2:
            target = self.target * 3 // 4
            if fresh_remb and remb > 0:
                target = min(target, remb)
            self.target = max(self.minimum, target)
            self._bad = self._good = 0
            self.reason = "receiver_decrease" if falling else "congestion"
        elif self._good >= 3:
            self.target = min(self.maximum, self.target + max(250000, self.target // 20))
            self._good = 0
            self.reason = "healthy_probe"
        if fresh_remb:
            self._remb = remb
        self._remb_reports = stats["receiver_bitrate_reports"]
        self._reports, self._recoveries = stats["receiver_reports"], stats["recoveries"]
        return self.target


class _EncoderSample(Protocol):
    @property
    def byte_size(self) -> int: ...

    @property
    def gpu_prepare_ms(self) -> float: ...

    @property
    def encode_wall_ms(self) -> float: ...

    @property
    def submit_to_output_ms(self) -> float: ...


class _EncoderDiagnostics:
    def __init__(self, start: float) -> None:
        super().__init__()
        self._start = start
        self._samples: deque[tuple[float, int, float, float, float]] = deque(maxlen=512)
        self._bytes = 0

    def add(self, packet: _EncoderSample, now: float) -> None:
        self._bytes += packet.byte_size
        self._samples.append(
            (
                now,
                packet.byte_size,
                packet.gpu_prepare_ms,
                packet.encode_wall_ms,
                packet.submit_to_output_ms,
            )
        )

    def snapshot(self, now: float) -> dict[str, float | int]:
        while self._samples and self._samples[0][0] <= now - 1.0:
            self._samples.popleft()
        elapsed = min(1.0, max(0.0, now - self._start))
        count = len(self._samples)
        result: dict[str, float | int] = {
            "encoded_bytes": self._bytes,
            "encoded_bitrate": (
                sum(sample[1] for sample in self._samples) * 8 / elapsed if elapsed else 0.0
            ),
            "encoded_fps": count / elapsed if elapsed else 0.0,
            "encoder_window_frames": count,
        }
        for index, name in enumerate(
            ("gpu_prepare_ms", "encode_wall_ms", "submit_to_output_ms"), 2
        ):
            values = sorted(sample[index] for sample in self._samples)
            result[name] = sum(values) / count if count else 0.0
            result[name + "_p95"] = values[(95 * count + 99) // 100 - 1] if count else 0.0
        return result


_STAT_FIELDS = (
    "receiver_sequence",
    "receiver_lost",
    "in_flight_frames",
    "pacing_bitrate",
    "pacing_delay_ms",
    "rtp_packets",
    "rtp_bytes",
    "repair_packets",
    "repair_bytes",
    "repair_dropped",
    "repair_queue_packets",
    "connected",
    "failed",
    "diagnostic",
    "sent_frames",
    "dropped_frames",
    "recoveries",
    "queue_bytes",
    "queue_frames",
    "queue_age_ms",
    "pli_count",
    "nack_count",
    "receiver_reports",
    "loss_fraction",
    "receiver_bitrate",
    "receiver_bitrate_reports",
    "local_candidate_type",
    "remote_candidate_type",
)


class StreamingConnection:
    """Borrow a device and serialize all calls on its rendering thread.

    Call start() after application warm-up, update() even when capture is blocked,
    and capture() after submitting the completed image on the default queue.
    close() releases input and drains the encoder, but never closes the device.
    """

    def __init__(self, device: spy.Device, config: ServerConfig, generation: str) -> None:
        super().__init__()
        self._config = config
        self._generation = generation
        self._backend = device.info.type.name
        self._encoder_config = config._encoder_config()
        self._encoder = streaming.VideoEncoder(device, self._encoder_config)
        self._peer = None
        self._input = None
        self._begin = self._next_frame = time.monotonic()
        self._next_policy = self._begin + 1
        self._policy = _BitratePolicy(config.min_bitrate, config.bitrate)
        self._bitrate = config.bitrate
        self._diagnostics = _EncoderDiagnostics(self._begin)
        self._captured = self._encoded = self._busy = self._transport_skips = 0
        self.frame_id = 0
        self._transport_busy = False
        self._stats: dict[str, Any] = {"connected": False, "failed": False}

    def start(
        self,
        on_keyboard: Callable[[spy.KeyboardEvent], Any],
        on_mouse: Callable[[spy.MouseEvent], Any],
    ) -> None:
        if self._peer is not None or self._encoder is None:
            raise RuntimeError("Streaming connection has already started or closed")
        config = self._config
        self._input = RemoteInput(
            self._generation, config.width, config.height, on_keyboard, on_mouse
        )
        transport = streaming.WebRTCConfig()
        transport.fps = config.fps
        transport.max_bitrate = config.bitrate
        transport.width, transport.height = config.width, config.height
        transport.enable_input = True
        transport.force_relay = config.force_relay
        transport.bind_address = config.ice_bind
        transport.port_begin, transport.port_end = config.port_begin, config.port_end
        servers = []
        for value in config.ice_servers:
            entry = streaming.ICEServerConfig()
            entry.url = urlsplit(value["url"]).geturl()
            entry.username = value.get("username", "")
            entry.credential = value.get("credential", "")
            servers.append(entry)
        transport.ice_servers = servers
        self._peer = streaming.WebRTCPeer(transport)
        self._peer.set_target_bitrate(config.bitrate)
        self._peer.set_pacing_bitrate(config.pacing_bitrate)
        self._begin = self._next_frame = time.monotonic()
        self._next_policy = self._begin + 1
        self._diagnostics = _EncoderDiagnostics(self._begin)

    @property
    def connected(self) -> bool:
        return bool(self._stats["connected"])

    @property
    def capture_due(self) -> bool:
        return self.connected and time.monotonic() >= self._next_frame

    @property
    def has_capacity(self) -> bool:
        if self._peer is None:
            return False
        # Rendering can outlast a packet's paced send. Check current pressure
        # rather than blocking a whole application frame after draining output.
        stats = self._peer.stats()
        self._transport_busy = stats.queue_frames > 0 or stats.pacing_delay_ms > 50
        return (
            not self._transport_busy
            and self._captured - self._encoded < self._encoder_config.max_frame_slots
        )

    def update(self, answer: str | None = None) -> str | None:
        """Pump signaling, output and input; return a newly available SDP offer."""
        peer, encoder, remote = self._peer, self._encoder, self._input
        if peer is None or encoder is None or remote is None:
            raise RuntimeError("Streaming connection is not running")
        if answer is not None:
            peer.set_answer(answer)
        offer = peer.poll_offer()
        native = peer.stats()
        was_connected = self.connected
        self._stats = {name: getattr(native, name) for name in _STAT_FIELDS}
        if self._stats["failed"]:
            raise RuntimeError(self._stats["diagnostic"] or "WebRTC connection failed")
        now = time.monotonic()
        if not self.connected:
            if now - self._begin > 30:
                raise TimeoutError("ICE/video connection timed out")
            return offer
        if not was_connected:
            self._next_frame = now
        if peer.poll_keyframe_request():
            encoder.request_keyframe()
        if now >= self._next_policy:
            stats = {**self._stats, **self._diagnostics.snapshot(now)}
            target = self._policy.update(stats)
            if target != self._bitrate:
                encoder.set_target_bitrate(target)
                peer.set_target_bitrate(target)
                if target < self._bitrate and self._policy.loss_fraction >= 5:
                    encoder.request_keyframe()
                self._bitrate = target
            self._next_policy = now + 1
        while (packet := encoder.poll_output()) is not None:
            self._encoded += 1
            self._diagnostics.add(packet, time.monotonic())
            peer.send(packet)
        if peer.poll_input_reset():
            remote.release()
        for _ in range(128):
            message = peer.poll_input()
            if message is None:
                break
            remote.receive(message, now)
        remote.update(now)
        return offer

    def capture(self, image: spy.Texture | None, transfer: streaming.InputTransfer) -> bool:
        """Capture at most one due frame, skipping missed deadlines and pressure."""
        if not self.capture_due:
            return False
        now = time.monotonic()
        accepted = False
        has_capacity = self.has_capacity
        if self._transport_busy:
            self._transport_skips += 1
        elif not has_capacity or image is None:
            self._busy += 1
        else:
            assert self._encoder is not None
            accepted = self._encoder.try_submit(
                image, self.frame_id, int((now - self._begin) * 1000000), transfer
            )
            self._captured += int(accepted)
            self._busy += int(not accepted)
        self.frame_id += 1
        self._next_frame += (
            int((now - self._next_frame) * self._config.fps) + 1
        ) / self._config.fps
        return accepted

    def stats(self) -> dict[str, Any]:
        now = time.monotonic()
        remote = self._input
        return {
            **self._stats,
            **self._diagnostics.snapshot(now),
            "elapsed_seconds": now - self._begin,
            "captured": self._captured,
            "encoded": self._encoded,
            "busy_drops": self._busy,
            "transport_skips": self._transport_skips,
            "capture_skips": self._busy + self._transport_skips,
            "target_bitrate": self._bitrate,
            "target_fps": self._config.fps,
            "bitrate_reason": self._policy.reason,
            "interval_loss_fraction": self._policy.loss_fraction,
            "encoder": "openh264",
            "backend": self._backend,
            "input_sequence": remote.sequence if remote else -1,
            "input_rejected": remote.rejected if remote else 0,
            "pressed_keys": sorted(remote.keys) if remote else [],
            "pressed_buttons": sorted(remote.buttons) if remote else [],
        }

    def close(self) -> None:
        remote, self._input = self._input, None
        peer, self._peer = self._peer, None
        encoder, self._encoder = self._encoder, None
        error = None
        actions = ([remote.release] if remote else []) + [
            resource.close for resource in (peer, encoder) if resource is not None
        ]
        for action in actions:
            try:
                action()
            except Exception as caught:
                error = caught
        self._stats["connected"] = False
        if error is not None:
            raise error


class Renderer(Protocol):
    """Session-owned renderer; every method runs on its owning thread.

    Return a tone-mapped, linear SDR texture on the supplied device, at the
    configured dimensions. The encoder retains in-flight textures. close() runs
    after the encoder drains, before the device closes. Factory failures must
    clean up resources that were not returned to the session.
    """

    def update(self, dt: float) -> None: ...
    def render(self, frame_id: int, dt: float) -> spy.Texture: ...
    def handle_keyboard_event(self, event: spy.KeyboardEvent) -> None: ...
    def handle_mouse_event(self, event: spy.MouseEvent) -> None: ...
    def stats(self) -> dict[str, Any]: ...
    def close(self) -> None: ...


RendererFactory = Callable[[spy.Device, int, int], Renderer]


class StreamingSession:
    """All renderer/encoder/peer API calls belong to one Python owner thread.

    The factory constructs the renderer on that thread. Input handlers receive
    SGL events and must return promptly. Held inputs are released on teardown.
    """

    def __init__(
        self,
        config: ServerConfig,
        renderer_factory: RendererFactory,
        *,
        executor: ThreadPoolExecutor | None = None,
    ) -> None:
        super().__init__()
        self._config = config
        self._renderer_factory = renderer_factory
        self.generation = secrets.token_hex(16)
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._answer: queue.Queue[str] = queue.Queue(maxsize=1)
        self._offer: str | None = None
        self._status: dict[str, Any] = {"connected": False, "failed": False}
        self._owns_executor = executor is None
        self._executor = executor or ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="falcor-stream-session"
        )
        self._future = self._executor.submit(self._run)

    def set_answer(self, sdp: str) -> None:
        self._answer.put_nowait(sdp)

    def snapshot(self) -> tuple[str | None, dict[str, Any]]:
        with self._lock:
            offer, self._offer = self._offer, None
            return offer, dict(self._status)

    def close(self) -> None:
        self._stop.set()
        try:
            self._future.result()
        finally:
            if self._owns_executor:
                self._executor.shutdown(wait=True)

    def _run(self) -> None:
        device = connection = renderer = None
        try:
            device = f2.create_device(device_type=self._config.backend)
            connection = StreamingConnection(device, self._config, self.generation)
            renderer = self._renderer_factory(device, self._config.width, self._config.height)
            renderer.render(0, 0.0)  # Warm shaders before connection timing.
            connection.start(renderer.handle_keyboard_event, renderer.handle_mouse_event)
            last_render = None
            next_status = 0.0
            rendered = 0
            render_ms = 0.0
            while not self._stop.is_set():
                try:
                    answer = self._answer.get_nowait()
                except queue.Empty:
                    answer = None
                offer = connection.update(answer)
                now = time.monotonic()
                if connection.connected:
                    if last_render is None:
                        last_render = now
                    if connection.capture_due:
                        dt = now - last_render
                        renderer.update(dt)
                        image = None
                        if connection.has_capacity:
                            render_start = time.monotonic()
                            image = renderer.render(connection.frame_id, dt)
                            render_ms = (time.monotonic() - render_start) * 1000
                            rendered += 1
                        last_render = now
                        connection.capture(image, streaming.InputTransfer.linear)
                if offer is not None:
                    with self._lock:
                        self._offer = offer
                if now >= next_status:
                    stats = connection.stats()
                    stats.update(renderer.stats())
                    stats.update(rendered=rendered, render_wall_ms=render_ms)
                    with self._lock:
                        self._status = stats
                    next_status = now + 0.25
                time.sleep(0.001)
        except Exception as error:
            with self._lock:
                self._status.update(failed=True, diagnostic=str(error))
        finally:
            for resource in (connection, renderer, device):
                if resource is not None:
                    try:
                        resource.close()
                    except Exception as error:
                        with self._lock:
                            self._status.update(failed=True, diagnostic=str(error))
            with self._lock:
                self._status.update(connected=False, pressed_keys=[], pressed_buttons=[])
