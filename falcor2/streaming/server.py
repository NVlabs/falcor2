# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Browser streaming with optional token authentication."""

from __future__ import annotations

import argparse
import asyncio
import ctypes
import errno
import ipaddress
import json
import logging
import os
from pathlib import Path
import queue
import secrets
import ssl
import sys
import time
from typing import Any, Callable, Protocol
from urllib.parse import urlencode
from concurrent.futures import ThreadPoolExecutor

from aiohttp import web, WSMsgType
import falcor2 as f2
import slangpy as spy
from falcor2 import streaming
from falcor2.streaming.config import ServerConfig, load_ice_config
from falcor2.streaming.session import StreamingSession, RendererFactory
from falcor2.streaming.demo import PatternRenderer, CameraRenderer


_LOGGER = logging.getLogger(__name__)


class _JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return json.dumps({"level": record.levelname.lower(), "message": record.getMessage()})


class _Session(Protocol):
    generation: str

    def set_answer(self, sdp: str) -> None: ...
    def snapshot(self) -> tuple[str | None, dict[str, Any]]: ...
    def close(self) -> None: ...


def create_app(
    config: ServerConfig,
    session_factory: Callable[[ServerConfig], _Session] | None = None,
    *,
    renderer_factory: RendererFactory = PatternRenderer,
) -> web.Application:
    app = web.Application(client_max_size=65536)
    sessions: dict[web.WebSocketResponse, _Session] = {}
    # Standalone sessions each own a renderer on one shared executor. Only the
    # embedded editor factory supports simultaneous viewers of a shared render.
    viewer_limit = config.max_viewers if session_factory is not None else 1
    last_status: dict[str, Any] = {}
    assets = Path(__file__).with_name("web")

    async def health(request: web.Request) -> web.Response:
        return web.json_response(
            {
                "ready": True,
                "busy": viewer_limit is not None and len(sessions) >= viewer_limit,
                "viewers": len(sessions),
                "max_viewers": viewer_limit,
                "last_session": last_status,
            }
        )

    async def asset(request: web.Request) -> web.StreamResponse:
        name = request.match_info.get("name", "index.html")
        if name not in {"index.html", "client.js", "input.js", "style.css"}:
            raise web.HTTPNotFound()
        response = web.FileResponse(assets / name)
        response.headers.update(
            {
                "Cache-Control": "no-store",
                "Referrer-Policy": "no-referrer",
                "X-Content-Type-Options": "nosniff",
                "Content-Security-Policy": "default-src 'self'; connect-src 'self'; media-src 'self' blob:; object-src 'none'; frame-ancestors 'none'",
            }
        )
        return response

    async def signaling(request: web.Request) -> web.WebSocketResponse:
        nonlocal last_status
        allowed_origins = config.origins or {f"{request.scheme}://{request.host}"}
        if request.headers.get("Origin") not in allowed_origins:
            raise web.HTTPForbidden(text="Signaling Origin is not allowed")
        ws = web.WebSocketResponse(max_msg_size=65536, heartbeat=10, receive_timeout=30)
        await ws.prepare(request)
        session = None
        try:
            message = await asyncio.wait_for(ws.receive(), timeout=5)
            if message.type in {
                WSMsgType.CLOSE,
                WSMsgType.CLOSING,
                WSMsgType.CLOSED,
                WSMsgType.ERROR,
            }:
                return ws
            auth = json.loads(message.data) if message.type == WSMsgType.TEXT else {}
            token = auth.get("token", "") if isinstance(auth, dict) else None
            if (
                not isinstance(token, str)
                or len(token) > 256
                or auth.get("type") != "auth"
                or (
                    config.token
                    and not secrets.compare_digest(
                        token.encode("utf-8"), config.token.encode("utf-8")
                    )
                )
            ):
                await ws.send_json(
                    {"type": "error", "code": "unauthorized", "message": "Invalid session token"}
                )
                return ws
            if viewer_limit is not None and len(sessions) >= viewer_limit:
                await ws.send_json(
                    {
                        "type": "error",
                        "code": "busy",
                        "message": "The viewer limit has been reached",
                    }
                )
                return ws
            session = (
                StreamingSession(config, renderer_factory, executor=session_executor)
                if session_factory is None
                else session_factory(config)
            )
            sessions[ws] = session
            _LOGGER.info("Viewer connected.")
            await ws.send_json(
                {
                    "type": "config",
                    "generation": session.generation,
                    "width": config.width,
                    "height": config.height,
                    "fps": config.fps,
                    "bitrate": config.bitrate,
                    "iceServers": [
                        {
                            "urls": entry["url"],
                            **{
                                key: entry[key]
                                for key in ("username", "credential")
                                if key in entry
                            },
                        }
                        for entry in config.ice_servers
                    ],
                    "iceTransportPolicy": "relay" if config.force_relay else "all",
                }
            )
            answered = False
            offer_sent = False
            next_status = 0.0
            receive = asyncio.create_task(ws.receive())
            try:
                while not ws.closed:
                    done, _ = await asyncio.wait({receive}, timeout=0.05)
                    if done:
                        message = receive.result()
                        if message.type in {
                            WSMsgType.CLOSE,
                            WSMsgType.CLOSING,
                            WSMsgType.CLOSED,
                            WSMsgType.ERROR,
                        }:
                            break
                        if message.type != WSMsgType.TEXT:
                            raise ValueError("Expected a JSON signaling message")
                        value = json.loads(message.data)
                        if not isinstance(value, dict):
                            raise ValueError("Signaling message must be an object")
                        if value.get("type") == "disconnect":
                            break
                        if (
                            value.get("type") != "answer"
                            or answered
                            or not offer_sent
                            or not isinstance(value.get("sdp"), str)
                        ):
                            raise ValueError("Expected one SDP answer after the offer")
                        if not 1 <= len(value["sdp"]) <= 60000:
                            raise ValueError("Invalid SDP length")
                        session.set_answer(value["sdp"])
                        answered = True
                        receive = asyncio.create_task(ws.receive())
                    offer, status = session.snapshot()
                    if status.get("failed"):
                        raise RuntimeError(status.get("diagnostic", "Streaming session failed"))
                    if offer is not None:
                        await ws.send_json({"type": "offer", "sdp": offer})
                        offer_sent = True
                    if time.monotonic() >= next_status:
                        await ws.send_json({"type": "stats", "stats": status})
                        next_status = time.monotonic() + 0.5
            finally:
                receive.cancel()
                await asyncio.gather(receive, return_exceptions=True)
        except ConnectionError:
            pass  # Transport disappeared; the finally block still releases the lease.
        except (ValueError, RuntimeError, asyncio.TimeoutError, queue.Full) as error:
            _LOGGER.error("Session error: %s", error)
            if not ws.closed:
                await ws.send_json(
                    {"type": "error", "code": "session_error", "message": str(error)}
                )
        finally:
            try:
                if session is not None:
                    await asyncio.to_thread(session.close)
                    _, last_status = session.snapshot()
            finally:
                if sessions.pop(ws, None) is not None:
                    _LOGGER.info("Viewer disconnected.")
                await ws.close()
        return ws

    async def shutdown(application: web.Application) -> None:
        await asyncio.gather(
            *(ws.close(code=1001, message=b"Server shutting down") for ws in list(sessions))
        )

    app.router.add_get("/health", health)
    app.router.add_get("/ws", signaling)
    app.router.add_get("/", asset)
    app.router.add_get("/{name}", asset)
    app.on_shutdown.append(shutdown)
    # CUDA/driver thread-local state can outlive short-lived owner threads.
    # Reuse one owner for sequential viewers, while still closing every
    # session's device, renderer, encoder and peer before releasing its lease.
    session_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="falcor-stream-owner")

    async def cleanup_executor(application: web.Application) -> None:
        for session in list(sessions.values()):
            await asyncio.to_thread(session.close)
        await asyncio.to_thread(session_executor.shutdown, wait=True)

    app.on_cleanup.append(cleanup_executor)
    return app


def check_ready(config: ServerConfig) -> str:
    if not streaming.WebRTCPeer.available():
        raise RuntimeError(
            "Configure FALCOR_ENABLE_STREAMING=ON to build WebRTC support through vcpkg"
        )
    device = f2.create_device(device_type=config.backend)
    try:
        encoder = streaming.VideoEncoder(device, config._encoder_config())
        encoder.close()
        return device.info.adapter_name
    finally:
        device.close()


def main(argv: list[str] | None = None) -> int:
    if os.name == "nt":
        ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002 | 0x8000)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend", choices=["automatic", "cuda", "d3d12", "vulkan"], default="automatic"
    )
    parser.add_argument("--demo", choices=["pattern", "camera"], default="pattern")
    parser.add_argument("--bind", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--log-format", choices=["text", "json"], default="text")
    parser.add_argument(
        "--token",
        nargs="?",
        const="",
        default=None,
        metavar="VALUE",
        help="Require a token (generate one if VALUE is omitted); default: no token",
    )
    parser.add_argument("--origin", action="append", default=[])
    parser.add_argument("--cert", type=Path)
    parser.add_argument("--key", type=Path)
    parser.add_argument("--behind-proxy", action="store_true")
    parser.add_argument("--ice-config", type=Path)
    parser.add_argument("--force-relay", action="store_true")
    parser.add_argument("--ice-bind", default="")
    parser.add_argument("--ice-port-begin", type=int, default=50000)
    parser.add_argument("--ice-port-end", type=int, default=50100)
    parser.add_argument(
        "--bitrate",
        type=int,
        default=8000000,
        help="Encoder target and session ceiling bits/s (100000..50000000; default: 8000000)",
    )
    parser.add_argument(
        "--pacing-bitrate",
        type=int,
        default=100000000,
        help="Transport bits/s (100000..100000000; default: 100000000); 0 uses 125%% of encoder bitrate",
    )
    parser.add_argument(
        "--width", type=int, default=1280, help="Even frame width (16..3840; default: 1280)"
    )
    parser.add_argument(
        "--height", type=int, default=720, help="Even frame height (16..2160; default: 720)"
    )
    parser.add_argument(
        "--fps", type=int, default=30, help="Frames per second (1..60; default: 30)"
    )
    parser.add_argument("--min-bitrate", type=int, default=1000000)
    parser.add_argument(
        "--run-seconds",
        type=float,
        default=0,
        help="Gracefully stop after this duration (0 runs until interrupted)",
    )
    args = parser.parse_args(argv)
    handler = logging.StreamHandler()
    handler.setFormatter(
        _JSONFormatter() if args.log_format == "json" else logging.Formatter("%(message)s")
    )
    previous_level = _LOGGER.level
    _LOGGER.setLevel(logging.INFO)
    _LOGGER.addHandler(handler)
    try:
        token = secrets.token_urlsafe(32) if args.token == "" else args.token or ""
        address = ipaddress.ip_address(args.bind)
        if not 1 <= args.port <= 65535:
            raise ValueError("Invalid HTTP port")
        if args.run_seconds < 0:
            raise ValueError("--run-seconds must be nonnegative")
        ice = load_ice_config(args.ice_config)
        config = ServerConfig(
            bind=args.bind,
            port=args.port,
            cert=args.cert,
            key=args.key,
            behind_proxy=args.behind_proxy,
            token=token,
            origins=set(args.origin),
            ice_servers=ice,
            force_relay=args.force_relay,
            ice_bind=args.ice_bind,
            port_begin=args.ice_port_begin,
            port_end=args.ice_port_end,
            bitrate=args.bitrate,
            pacing_bitrate=args.pacing_bitrate,
            fps=args.fps,
            width=args.width,
            height=args.height,
            min_bitrate=args.min_bitrate,
            backend=getattr(spy.DeviceType, args.backend),
        )
        tls = None
        if args.cert:
            tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            tls.load_cert_chain(args.cert, args.key)
        host = f"[{args.bind}]" if address.version == 6 else args.bind
        if address.is_unspecified:
            host = "[::1]" if address.version == 6 else "127.0.0.1"
        local_origin = f"{'https' if tls else 'http'}://{host}:{args.port}"
        gpu = check_ready(config)
        app = create_app(
            config, renderer_factory=CameraRenderer if args.demo == "camera" else PatternRenderer
        )
        if args.run_seconds:

            async def schedule_stop(application: web.Application) -> None:
                def stop() -> None:
                    raise web.GracefulExit()

                asyncio.get_running_loop().call_later(args.run_seconds, stop)

            app.on_startup.append(schedule_stop)

        # aiohttp calls this only after all listening sockets have started.
        def started(message: str) -> None:
            url = f"{sorted(config.origins)[0] if config.origins else local_origin}/" + (
                "#" + urlencode({"token": token}) if token else ""
            )
            if args.log_format == "json":
                print(
                    json.dumps(
                        {
                            "ready": True,
                            "url": url,
                            "gpu": gpu,
                            "encoder": "openh264",
                            "backend": args.backend,
                            "fps": config.fps,
                            "width": config.width,
                            "height": config.height,
                        }
                    ),
                    flush=True,
                )
            else:
                authentication = (
                    "Authentication is enabled. Keep this URL private."
                    if token
                    else "Authentication is disabled. Use --token to enable it."
                )
                print(
                    f"Falcor streaming - {config.width}x{config.height} at {config.fps} FPS\n"
                    f"GPU: {gpu} ({args.backend})\n\n"
                    f"Open {url} in your browser to start viewing.\n"
                    f"{authentication}\nPress Ctrl+C to stop.",
                    flush=True,
                )

        web.run_app(
            app, host=args.bind, port=args.port, ssl_context=tls, print=started, access_log=None
        )
        return 0
    except Exception as error:
        if args.log_format == "json":
            code = (
                "address_in_use"
                if isinstance(error, OSError) and error.errno == errno.EADDRINUSE
                else "startup_error"
            )
            print(json.dumps({"ready": False, "code": code, "diagnostic": str(error)}), flush=True)
        else:
            print(f"Error: {error}", file=sys.stderr, flush=True)
        return 1
    finally:
        _LOGGER.removeHandler(handler)
        _LOGGER.setLevel(previous_level)
        handler.close()


if __name__ == "__main__":
    raise SystemExit(main())
