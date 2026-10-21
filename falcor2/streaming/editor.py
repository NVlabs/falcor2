# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Embedded browser server for applications that own their rendering loop."""

import asyncio
import ipaddress
import queue
import secrets
import socket
import ssl
import threading
from concurrent.futures import Future
from typing import Any
from collections.abc import Callable
from urllib.parse import urlencode

import slangpy as spy
from falcor2 import streaming
from falcor2.streaming.session import StreamingConnection
from falcor2.streaming.config import ServerConfig, StreamingConfig


class _EditorSession:
    """Signaling-thread handle; disposal is acknowledged by the rendering owner."""

    def __init__(self) -> None:
        super().__init__()
        self.generation = secrets.token_hex(16)
        self.stop = threading.Event()
        self.disposed = threading.Event()
        self.answers: queue.Queue[str] = queue.Queue(maxsize=1)
        self._lock = threading.Lock()
        self._offer: str | None = None
        self._stats: dict[str, Any] = {"connected": False, "failed": False}

    def set_answer(self, sdp: str) -> None:
        self.answers.put_nowait(sdp)

    def publish(self, offer: str | None, stats: dict[str, Any]) -> None:
        with self._lock:
            if offer is not None:
                self._offer = offer
            self._stats = stats

    def snapshot(self) -> tuple[str | None, dict[str, Any]]:
        with self._lock:
            offer, self._offer = self._offer, None
            return offer, dict(self._stats)

    def close(self) -> None:
        self.stop.set()
        self.disposed.wait()


class EditorStream:
    """Borrow the editor's device; update, capture and close on its owner thread."""

    def __init__(
        self,
        device: spy.Device,
        config: StreamingConfig,
        width: int,
        height: int,
        on_keyboard: Callable[[spy.KeyboardEvent], Any],
        on_mouse: Callable[[spy.MouseEvent], Any],
        on_control_change: Callable[[], Any],
    ) -> None:
        super().__init__()
        self._config = config
        self._server_config = config._server_config(width, height)
        if not streaming.WebRTCPeer.available():
            raise RuntimeError("Editor streaming requires FALCOR_ENABLE_STREAMING=ON")
        if not streaming.VideoEncoder.check_available():
            raise RuntimeError("Install OpenH264 with python -m falcor2.streaming.openh264 install")
        self._device = device
        self._on_keyboard, self._on_mouse = on_keyboard, on_mouse
        self._on_control_change = on_control_change
        self._pending: queue.Queue[_EditorSession] = queue.Queue(maxsize=config.max_viewers or 0)
        self._connections: dict[_EditorSession, StreamingConnection | None] = {}
        self._lock = threading.Lock()
        self._closed = False
        self._server_failure: BaseException | None = None
        self._ready: Future[str] = Future()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._shutdown: asyncio.Event | None = None
        self._thread = threading.Thread(target=self._run_server, name="falcor-editor-http")
        self._thread.start()
        try:
            self.url = self._ready.result(timeout=15)
        except BaseException:
            self.close()
            raise

    @property
    def controls_input(self) -> bool:
        # Keep local input suppressed while any browser owns remote input state.
        return bool(self._connections)

    def _new_session(self, config: ServerConfig) -> _EditorSession:
        with self._lock:
            if self._closed:
                raise RuntimeError("Editor is closing")
            session = _EditorSession()
            self._pending.put_nowait(session)
            return session

    def _run_server(self) -> None:
        try:
            asyncio.run(self._serve())
        except BaseException as error:
            self._server_failure = error
            if not self._ready.done():
                self._ready.set_exception(error)

    async def _serve(self) -> None:
        from aiohttp import web
        from falcor2.streaming.server import create_app

        self._loop = asyncio.get_running_loop()
        self._shutdown = asyncio.Event()
        config = self._config
        tls = None
        if config.cert:
            tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            tls.load_cert_chain(config.cert, config.key)
        family = (
            socket.AF_INET6 if ipaddress.ip_address(config.bind).version == 6 else socket.AF_INET
        )
        sock = socket.create_server((config.bind, config.port), family=family)
        runner = None
        try:
            port = sock.getsockname()[1]
            host = f"[{config.bind}]" if family == socket.AF_INET6 else config.bind
            if ipaddress.ip_address(config.bind).is_unspecified:
                host = "[::1]" if family == socket.AF_INET6 else "127.0.0.1"
            origin = f"{'https' if tls else 'http'}://{host}:{port}"
            app = create_app(self._server_config, session_factory=self._new_session)
            runner = web.AppRunner(app, access_log=None)
            await runner.setup()
            await web.SockSite(runner, sock, ssl_context=tls).start()
            url = f"{sorted(config.origins)[0] if config.origins else origin}/"
            if config.token:
                url += "#" + urlencode({"token": config.token})
            self._ready.set_result(url)
            with self._lock:
                closed = self._closed
            if not closed:
                await self._shutdown.wait()
        finally:
            if runner is not None:
                await runner.cleanup()
            sock.close()

    def _dispose(self, session: _EditorSession, error: Exception | None = None) -> None:
        connection = self._connections.pop(session, None)
        try:
            if connection is not None:
                connection.close()
        except Exception as caught:
            error = caught
        finally:
            try:
                if not self._connections:
                    self._on_control_change()
            except Exception as caught:
                error = caught
            finally:
                _, stats = session.snapshot()
                stats.update(connected=False, pressed_keys=[], pressed_buttons=[])
                if error is not None:
                    stats.update(failed=True, diagnostic=str(error))
                session.publish(None, stats)
                session.stop.set()
                session.disposed.set()

    def update(self, stats: dict[str, Any]) -> None:
        if self._closed:
            return
        if self._server_failure is not None:
            raise RuntimeError("Editor streaming server failed") from self._server_failure
        while True:
            try:
                session = self._pending.get_nowait()
            except queue.Empty:
                break
            self._connections[session] = None

        for session, connection in list(self._connections.items()):
            if session.stop.is_set():
                self._dispose(session)
                continue
            try:
                if connection is None:
                    if not any(self._connections.values()):
                        self._on_control_change()
                    connection = StreamingConnection(
                        self._device, self._server_config, session.generation
                    )
                    self._connections[session] = connection
                    connection.start(self._on_keyboard, self._on_mouse)
                try:
                    answer = session.answers.get_nowait()
                except queue.Empty:
                    answer = None
                offer = connection.update(answer)
                session.publish(offer, {**connection.stats(), **stats})
            except Exception as error:
                self._dispose(session, error)

    def capture(self, image: spy.Texture, transfer: streaming.InputTransfer) -> None:
        for session, connection in list(self._connections.items()):
            if connection is None:
                continue
            try:
                connection.capture(image, transfer)
            except Exception as error:
                self._dispose(session, error)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        for session in list(self._connections):
            self._dispose(session)
        while True:
            try:
                session = self._pending.get_nowait()
            except queue.Empty:
                break
            session.stop.set()
            session.disposed.set()
        if self._loop is not None and self._shutdown is not None and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._shutdown.set)
        self._thread.join(timeout=15)
        if self._thread.is_alive():
            raise RuntimeError("Editor streaming server did not stop")
