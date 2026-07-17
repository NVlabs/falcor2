# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import socket
import threading
import time
from collections.abc import Callable
from pathlib import Path

from falcor2.editor.editor import Editor, EditorConfig, FrameOutput, RenderMode
from falcor2.mcp.bridge import Bridge
from falcor2.mcp.editor_bridge import create_editor_bridge
from falcor2.mcp.screenshot import capture_editor_screenshot


def _bridge_call(host: str, port: int, method: str, params: dict[str, object]) -> dict[str, object]:
    with socket.create_connection((host, port), timeout=5.0) as client:
        request = {
            "id": "editor-test",
            "method": method,
            "params": params,
        }
        client.sendall((json.dumps(request) + "\n").encode("utf-8"))
        data = b""
        while not data.endswith(b"\n"):
            chunk = client.recv(65536)
            assert chunk
            data += chunk
    response = json.loads(data.decode("utf-8"))
    assert response["id"] == "editor-test"
    return response


def _start_bridge_call(
    host: str,
    port: int,
    method: str,
    params: dict[str, object],
) -> tuple[threading.Thread, list[dict[str, object]], list[BaseException]]:
    responses: list[dict[str, object]] = []
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            responses.append(_bridge_call(host, port, method, params))
        except BaseException as exc:  # noqa: BLE001 - fail the test after the worker rejoins.
            errors.append(exc)

    thread = threading.Thread(target=worker)
    thread.start()
    return thread, responses, errors


def _wait_for_pending_request(bridge: Bridge) -> None:
    for _ in range(100):
        if bridge.pending_count > 0:
            return
        time.sleep(0.01)
    raise AssertionError("Timed out waiting for bridge request to be queued.")


def _bridge_call_after_update(
    bridge: Bridge,
    host: str,
    port: int,
    method: str,
    params: dict[str, object],
    update: Callable[[], None] | None = None,
) -> dict[str, object]:
    thread, responses, errors = _start_bridge_call(host, port, method, params)
    _wait_for_pending_request(bridge)

    assert thread.is_alive()
    assert responses == []

    if update is None:
        bridge.update()
    else:
        update()
    thread.join(timeout=1.0)

    assert not thread.is_alive()
    assert not errors
    assert len(responses) == 1
    return responses[0]


def _make_editor_shell(config: EditorConfig) -> Editor:
    editor = object.__new__(Editor)
    editor.config = config
    editor.device = None
    editor._output = None
    editor._mcp_bridge = None
    editor._scene = None
    editor._render_mode = RenderMode.path_traced
    return editor


class _FakeDevice:
    def __init__(self) -> None:
        self.wait_count = 0

    def wait(self) -> None:
        self.wait_count += 1


def test_editor_mcp_can_be_disabled():
    editor = _make_editor_shell(EditorConfig(mcp=False))
    editor._start_mcp_bridge()
    assert editor._mcp_bridge is None


def test_editor_starts_updates_and_stops_bridge(tmp_path, monkeypatch):
    monkeypatch.setenv("FALCOR2_MCP_REGISTRY_DIR", str(tmp_path / "registry"))
    editor = _make_editor_shell(EditorConfig(title="MCP Test Editor"))

    editor._start_mcp_bridge()
    try:
        bridge = editor._mcp_bridge
        assert bridge is not None
        endpoint = bridge.endpoint
        assert endpoint.host is not None
        assert endpoint.port is not None

        editor._update_mcp_bridge()
        status = _bridge_call_after_update(
            bridge,
            endpoint.host,
            endpoint.port,
            "falcor2.app.status",
            {},
            update=editor._update_mcp_bridge,
        )
        assert status["result"]["title"] == "MCP Test Editor"
        assert status["result"]["scene_loaded"] is False
        assert "falcor2.app.screenshot" in bridge.endpoint_names()

        heartbeat_path = tmp_path / "registry" / f"{bridge.instance_id}.json"
        assert heartbeat_path.exists()
    finally:
        editor._stop_mcp_bridge()

    assert editor._mcp_bridge is None


def test_editor_mcp_screenshot_writes_current_output(tmp_path, monkeypatch):
    editor = _make_editor_shell(EditorConfig())
    texture = object()
    device = _FakeDevice()
    editor.device = device
    editor._output = FrameOutput(width=3, height=2, color_target=texture)

    calls: list[tuple[object, Path]] = []

    def save_image(image: object, path: Path) -> None:
        calls.append((image, Path(path)))
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(b"fake png")

    monkeypatch.setattr("falcor2.editor.utils.save_image", save_image)

    result = capture_editor_screenshot(
        editor,
        {"path": "captures/current"},
        workspace_root=tmp_path,
    )

    expected_path = (tmp_path / "captures" / "current.png").resolve()
    assert calls == [(texture, expected_path)]
    assert device.wait_count == 1
    assert result == {
        "path": str(expected_path),
        "markdown_path": expected_path.as_posix(),
        "width": 3,
        "height": 2,
        "mime_type": "image/png",
    }


def test_editor_mcp_screenshot_request_is_processed_by_update_thread(tmp_path, monkeypatch):
    editor = _make_editor_shell(EditorConfig())
    editor.device = _FakeDevice()
    editor._output = FrameOutput(width=1, height=1, color_target=object())
    mcp_bridge = create_editor_bridge(
        editor,
        workspace_root=tmp_path,
        falcor2_root=tmp_path,
        config_path=Path(__file__),
    )

    monkeypatch.setattr(
        "falcor2.editor.utils.save_image",
        lambda _image, path: Path(path).write_bytes(b"fake png"),
    )

    with mcp_bridge:
        endpoint = mcp_bridge.endpoint
        assert endpoint.host is not None
        assert endpoint.port is not None

        thread, responses, errors = _start_bridge_call(
            endpoint.host,
            endpoint.port,
            "falcor2.app.screenshot",
            {"path": "queued"},
        )
        _wait_for_pending_request(mcp_bridge)

        assert thread.is_alive()
        assert responses == []

        mcp_bridge.update()
        thread.join(timeout=1.0)

        assert not thread.is_alive()
        assert not errors
        assert len(responses) == 1
        assert responses[0]["result"]["path"] == str((tmp_path / "queued.png").resolve())
