# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import threading
from types import SimpleNamespace
from typing import Any, cast

import pytest
import slangpy as spy

pytest.importorskip("aiohttp")

from falcor2.streaming import editor as module


@pytest.fixture
def fake_runtime(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, int]]:
    calls = []

    def record(name: str) -> None:
        calls.append((name, threading.get_ident()))

    class Connection:
        def __init__(self, *args: Any) -> None:
            super().__init__()
            record("create")

        def start(self, *args: Any) -> None:
            record("start")

        def update(self, answer: str | None) -> str:
            record("update")
            return "offer"

        def capture(self, *args: Any) -> None:
            record("capture")

        def stats(self) -> dict[str, Any]:
            return {"connected": True, "failed": False}

        def close(self) -> None:
            record("close")

    monkeypatch.setattr(module, "StreamingConnection", Connection)
    monkeypatch.setattr(module.streaming, "WebRTCPeer", SimpleNamespace(available=lambda: True))
    monkeypatch.setattr(
        module.streaming, "VideoEncoder", SimpleNamespace(check_available=lambda: True)
    )
    return calls


def _stream(config: module.StreamingConfig | None = None) -> module.EditorStream:
    # Any attempt to own/close this borrowed device fails the test.
    return module.EditorStream(
        cast(spy.Device, object()),
        config or module.StreamingConfig(port=0),
        640,
        360,
        lambda event: None,
        lambda event: None,
        lambda: None,
    )


def test_reconnect_preserves_application_and_owner_thread(
    fake_runtime: list[tuple[str, int]]
) -> None:
    stream = _stream()
    try:
        generations = set()
        for iteration in range(2):
            session = stream._new_session(stream._server_config)
            generations.add(session.generation)
            stream.update({"rendered": 100 + iteration})
            assert stream.controls_input
            offer, stats = session.snapshot()
            assert offer == "offer" and stats["rendered"] == 100 + iteration
            stream.capture(cast(spy.Texture, object()), module.streaming.InputTransfer.srgb)
            worker = threading.Thread(target=session.close)
            worker.start()
            assert session.stop.wait(2)
            stream.update({})
            worker.join(timeout=2)
            assert not worker.is_alive()
            assert not stream.controls_input
            _, stats = session.snapshot()
            assert not stats["connected"]
            assert stats["pressed_keys"] == stats["pressed_buttons"] == []
        assert len(generations) == 2
    finally:
        stream.close()
    stream.close()
    assert not stream._thread.is_alive()
    assert {owner for _, owner in fake_runtime} == {threading.get_ident()}
    assert [name for name, _ in fake_runtime].count("close") == 2


@pytest.mark.parametrize("started", [False, True])
def test_shutdown_acknowledges_pending_and_active_sessions(
    fake_runtime: list[tuple[str, int]], started: bool
) -> None:
    stream = _stream()
    session = stream._new_session(stream._server_config)
    if started:
        stream.update({})
    worker = threading.Thread(target=session.close)
    worker.start()
    assert session.stop.wait(2)
    stream.close()
    worker.join(timeout=2)
    assert session.disposed.is_set() and not worker.is_alive()
    assert not stream._thread.is_alive()
    with pytest.raises(RuntimeError, match="closing"):
        stream._new_session(stream._server_config)


def test_session_failure_releases_control_and_allows_reconnect(
    fake_runtime: list[tuple[str, int]], monkeypatch: pytest.MonkeyPatch
) -> None:
    stream = _stream()
    try:
        session = stream._new_session(stream._server_config)
        stream.update({})

        def fail(*args: Any) -> None:
            raise RuntimeError("encoder failed")

        monkeypatch.setattr(stream._connections[session], "capture", fail)
        stream.capture(cast(spy.Texture, object()), module.streaming.InputTransfer.srgb)
        _, stats = session.snapshot()
        assert stats["failed"] and stats["diagnostic"] == "encoder failed"
        assert session.disposed.is_set() and not stream.controls_input
        second = stream._new_session(stream._server_config)
        stream.update({})
        assert second.snapshot()[1]["connected"]
    finally:
        stream.close()
