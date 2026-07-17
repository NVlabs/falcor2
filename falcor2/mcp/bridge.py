# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import socket
import threading
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from falcor2.mcp.registry import (
    BridgeEndpoint,
    HeartbeatRecord,
    default_registry_dir,
    utc_now_text,
    write_heartbeat,
)

JsonObject = dict[str, object]


@dataclass
class _QueuedRequest:
    client: socket.socket
    text: str = ""
    error: Optional[BaseException] = None


class Bridge:
    def __init__(
        self,
        workspace_root: Path,
        falcor2_root: Optional[Path] = None,
        config_paths: Optional[Sequence[Path]] = None,
        registry_dir: Optional[Path] = None,
        status_provider: Optional[Callable[[], JsonObject]] = None,
    ) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self.falcor2_root = Path(falcor2_root).resolve() if falcor2_root is not None else None
        self.config_paths = [str(Path(path).resolve()) for path in (config_paths or [])]
        self.registry_dir = registry_dir or default_registry_dir()
        self.instance_id = str(uuid.uuid4())
        self.created_at = utc_now_text()

        self._handlers: dict[str, Callable[[JsonObject], JsonObject]] = {}
        self._endpoint: Optional[BridgeEndpoint] = None
        self._server_socket: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._request_lock = threading.Lock()
        self._requests: list[_QueuedRequest] = []
        self._stop_event = threading.Event()
        self._last_heartbeat = 0.0
        self._heartbeat_interval_s = 3.0
        self._register_default_handlers(status_provider)

    @property
    def endpoint(self) -> BridgeEndpoint:
        if self._endpoint is None:
            raise RuntimeError("Bridge has not been started.")
        return self._endpoint

    def register_handler(
        self,
        method: str,
        handler: Callable[[JsonObject], JsonObject],
    ) -> None:
        self._handlers[method] = handler

    def endpoint_names(self) -> list[str]:
        return sorted(self._handlers.keys())

    @property
    def pending_count(self) -> int:
        with self._request_lock:
            return len(self._requests)

    def update(self) -> None:
        if self._endpoint is None:
            return

        self._process_queued_requests()

        now = time.monotonic()
        if now - self._last_heartbeat >= self._heartbeat_interval_s:
            self._write_heartbeat()
            self._last_heartbeat = now

    def start(self) -> BridgeEndpoint:
        if self._endpoint is not None:
            return self._endpoint

        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind(("127.0.0.1", 0))
        server_socket.listen()
        server_socket.settimeout(0.2)

        host, port = server_socket.getsockname()
        self._server_socket = server_socket
        self._endpoint = BridgeEndpoint(
            transport="tcp",
            host=str(host),
            port=int(port),
            path=None,
        )
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._serve_loop,
            name=f"falcor2-mcp-{self.instance_id}",
            daemon=True,
        )
        self._thread.start()
        self._write_heartbeat()
        self._last_heartbeat = time.monotonic()
        return self._endpoint

    def serve_forever(self) -> None:
        self.start()
        try:
            while not self._stop_event.is_set():
                self.update()
                time.sleep(0.1)
        finally:
            self.stop()

    def stop(self) -> None:
        self._stop_event.set()
        server_socket = self._server_socket
        self._server_socket = None
        if server_socket is not None:
            try:
                server_socket.close()
            except OSError:
                pass
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None
        self._close_queued_requests()
        self._endpoint = None

    def __enter__(self) -> "Bridge":
        self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.stop()

    def _serve_loop(self) -> None:
        while not self._stop_event.is_set():
            server_socket = self._server_socket
            if server_socket is None:
                return
            try:
                client, _addr = server_socket.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            self._queue_request(self._read_client_request(client))

    def _read_client_request(self, client: socket.socket) -> _QueuedRequest:
        try:
            client.settimeout(5.0)
            data = b""
            while not data.endswith(b"\n"):
                chunk = client.recv(65536)
                if not chunk:
                    break
                data += chunk
                if len(data) > 1024 * 1024:
                    raise ValueError("Request exceeds 1 MiB.")
            return _QueuedRequest(client=client, text=data.decode("utf-8"))
        except Exception as exc:  # noqa: BLE001 - convert bridge failures to JSON errors.
            return _QueuedRequest(client=client, error=exc)

    def _queue_request(self, request: _QueuedRequest) -> None:
        close_client = False
        with self._request_lock:
            if self._stop_event.is_set() or self._endpoint is None:
                close_client = True
            else:
                self._requests.append(request)

        if close_client:
            self._close_client(request.client)

    def _process_queued_requests(self) -> None:
        with self._request_lock:
            requests = self._requests
            self._requests = []

        for request in requests:
            if request.error is not None:
                response = self._error_response("bridge_error", str(request.error))
            else:
                try:
                    response = self._handle_request(request.text)
                except Exception as exc:  # noqa: BLE001 - convert handler failures to JSON errors.
                    response = self._error_response("bridge_error", str(exc))

            self._send_response(request.client, response)

    def _close_queued_requests(self) -> None:
        with self._request_lock:
            requests = self._requests
            self._requests = []

        for request in requests:
            self._close_client(request.client)

    def _send_response(self, client: socket.socket, response: JsonObject) -> None:
        try:
            client.sendall((json.dumps(response) + "\n").encode("utf-8"))
        except OSError:
            pass
        finally:
            self._close_client(client)

    def _close_client(self, client: socket.socket) -> None:
        try:
            client.close()
        except OSError:
            pass

    def _error_response(self, code: str, message: str) -> JsonObject:
        return {
            "id": None,
            "error": {
                "code": code,
                "message": message,
            },
        }

    def _handle_request(self, text: str) -> JsonObject:
        request = json.loads(text)
        if not isinstance(request, dict):
            raise ValueError("Bridge request must be a JSON object.")
        request_id = request.get("id")
        method = request.get("method")
        params = request.get("params", {})
        if not isinstance(method, str):
            raise ValueError("Bridge request method must be a string.")
        if not isinstance(params, dict):
            raise ValueError("Bridge request params must be a JSON object.")
        handler = self._handlers.get(method)
        if handler is None:
            return {
                "id": request_id,
                "error": {
                    "code": "unknown_method",
                    "message": f"No bridge handler is registered for method {method}.",
                },
            }
        result = handler(params)
        if not isinstance(result, dict):
            raise ValueError(f"Bridge handler {method} must return a JSON object.")
        return {
            "id": request_id,
            "result": result,
        }

    def _write_heartbeat(self) -> None:
        if self._endpoint is None:
            return
        package_path = Path(__file__).resolve().parents[1]
        record = HeartbeatRecord(
            instance_id=self.instance_id,
            pid=os.getpid(),
            workspace_root=str(self.workspace_root),
            falcor2_root=str(self.falcor2_root) if self.falcor2_root is not None else None,
            falcor2_package_path=str(package_path),
            falcor2_version=None,
            config_paths=self.config_paths,
            endpoint=self._endpoint,
            endpoints=self.endpoint_names(),
            created_at=self.created_at,
            last_heartbeat=utc_now_text(),
        )
        write_heartbeat(record, self.registry_dir)

    def _register_default_handlers(
        self,
        status_provider: Optional[Callable[[], JsonObject]] = None,
    ) -> None:
        def ping(_params: JsonObject) -> JsonObject:
            return {"ok": True}

        def echo(params: JsonObject) -> JsonObject:
            return dict(params)

        def status(_params: JsonObject) -> JsonObject:
            if status_provider is not None:
                return status_provider()
            return {
                "instance_id": self.instance_id,
                "workspace_root": str(self.workspace_root),
                "endpoints": self.endpoint_names(),
            }

        self.register_handler("falcor2.bridge.ping", ping)
        self.register_handler("falcor2.app.status", status)
        self.register_handler("falcor2.echo", echo)
