# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import socket
import threading
import time
from pathlib import Path

from falcor2.mcp.bridge import Bridge


def _bridge_call(host: str, port: int, method: str, params: dict[str, object]) -> dict[str, object]:
    with socket.create_connection((host, port), timeout=5.0) as client:
        request = {
            "id": "test",
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
    assert response["id"] == "test"
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
) -> dict[str, object]:
    thread, responses, errors = _start_bridge_call(host, port, method, params)
    _wait_for_pending_request(bridge)

    assert thread.is_alive()
    assert responses == []

    bridge.update()
    thread.join(timeout=1.0)

    assert not thread.is_alive()
    assert not errors
    assert len(responses) == 1
    return responses[0]


def test_bridge_ping_echo_and_heartbeat(tmp_path, monkeypatch):
    registry_dir = tmp_path / "registry"
    monkeypatch.setenv("FALCOR2_MCP_REGISTRY_DIR", str(registry_dir))
    config_path = Path(__file__).resolve()

    bridge = Bridge(
        workspace_root=tmp_path,
        falcor2_root=tmp_path,
        config_paths=[config_path],
    )
    bridge.register_handler("example.echo", lambda params: dict(params))

    with bridge:
        endpoint = bridge.endpoint
        assert endpoint.host is not None
        assert endpoint.port is not None

        ping = _bridge_call_after_update(
            bridge,
            endpoint.host,
            endpoint.port,
            "falcor2.bridge.ping",
            {},
        )
        assert ping["result"] == {"ok": True}

        echo = _bridge_call_after_update(
            bridge,
            endpoint.host,
            endpoint.port,
            "example.echo",
            {"message": "hello"},
        )
        assert echo["result"] == {"message": "hello"}

        heartbeat_path = registry_dir / f"{bridge.instance_id}.json"
        assert heartbeat_path.exists()
        heartbeat = json.loads(heartbeat_path.read_text(encoding="utf-8"))
        assert heartbeat["workspace_root"] == str(tmp_path.resolve())
        assert "falcor2.bridge.ping" in heartbeat["endpoints"]
        assert "example.echo" in heartbeat["endpoints"]


def test_bridge_handlers_run_on_update_thread(tmp_path):
    bridge = Bridge(workspace_root=tmp_path)
    update_thread_id = threading.get_ident()

    def thread_info(_params: dict[str, object]) -> dict[str, object]:
        return {"thread_id": threading.get_ident()}

    bridge.register_handler("example.thread", thread_info)

    with bridge:
        endpoint = bridge.endpoint
        assert endpoint.host is not None
        assert endpoint.port is not None

        response = _bridge_call_after_update(
            bridge,
            endpoint.host,
            endpoint.port,
            "example.thread",
            {},
        )
        assert response["result"] == {"thread_id": update_thread_id}


def test_bridge_reports_unknown_method(tmp_path):
    bridge = Bridge(workspace_root=tmp_path)

    with bridge:
        endpoint = bridge.endpoint
        assert endpoint.host is not None
        assert endpoint.port is not None

        response = _bridge_call_after_update(
            bridge,
            endpoint.host,
            endpoint.port,
            "missing.method",
            {},
        )
        assert response["error"]["code"] == "unknown_method"
