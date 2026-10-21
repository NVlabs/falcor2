# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import threading
from urllib.parse import parse_qs, urlsplit
from types import SimpleNamespace
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("aiohttp")
from aiohttp import ClientSession, WSServerHandshakeError, WSMsgType
from aiohttp.test_utils import TestServer
from falcor2 import streaming
from falcor2.streaming.server import create_app, load_ice_config
from falcor2.streaming.config import ServerConfig
from falcor2.streaming.server import main
from falcor2.streaming.session import StreamingSession


@pytest.fixture
def cli_configs(monkeypatch: pytest.MonkeyPatch) -> list[ServerConfig]:
    from falcor2.streaming import server

    configs: list[ServerConfig] = []

    def check_ready(config: ServerConfig) -> str:
        configs.append(config)
        return "test GPU"

    def create_app(config: ServerConfig, **kwargs: Any) -> object:
        assert config is configs[-1]
        return object()

    monkeypatch.setattr(server, "check_ready", check_ready)
    monkeypatch.setattr(server, "create_app", create_app)
    monkeypatch.setattr(server.web, "run_app", lambda *args, **kwargs: kwargs["print"](""))
    return configs


@pytest.mark.parametrize("encoder_fails", [False, True])
@pytest.mark.parametrize("fps", [30, 60])
def test_readiness_validates_encoder_and_closes_device(
    monkeypatch: pytest.MonkeyPatch, encoder_fails: bool, fps: int
) -> None:
    from falcor2.streaming import server

    closed = []
    device = SimpleNamespace(
        info=SimpleNamespace(adapter_name="test GPU"), close=lambda: closed.append("device")
    )

    def create_encoder(value: object, config: streaming.VideoEncoderConfig) -> SimpleNamespace:
        assert value is device
        assert config.fps == fps
        assert config.target_bitrate == config.max_bitrate == 50000000
        assert (config.width, config.height) == (3840, 2160)
        if encoder_fails:
            raise RuntimeError("OpenH264 unavailable")
        return SimpleNamespace(close=lambda: closed.append("encoder"))

    monkeypatch.setattr(server.streaming.WebRTCPeer, "available", lambda: True)
    monkeypatch.setattr(server.f2, "create_device", lambda **kwargs: device)
    monkeypatch.setattr(server.streaming, "VideoEncoder", create_encoder)
    config = ServerConfig(fps=fps, width=3840, height=2160, bitrate=50000000)
    if encoder_fails:
        with pytest.raises(RuntimeError, match="OpenH264 unavailable"):
            server.check_ready(config)
        assert closed == ["device"]
    else:
        assert server.check_ready(config) == "test GPU"
        assert closed == ["encoder", "device"]


def test_cli_reports_configuration_error(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--fps", "0"]) == 1
    assert "FPS must be between 1 and 60" in capsys.readouterr().err


class _FakeSession:
    generation = "test-generation"
    closed = 0

    def __init__(self, config: ServerConfig) -> None:
        super().__init__()
        self._offered = False
        self._answered = False

    def set_answer(self, sdp: str) -> None:
        self._answered = True

    def snapshot(self) -> tuple[str | None, dict[str, Any]]:
        offer = None if self._offered else "unit-test-offer"
        self._offered = True
        return offer, {"connected": self._answered, "failed": False}

    def close(self) -> None:
        type(self).closed += 1


@pytest.mark.parametrize(
    "required, supplied, accepted",
    [
        ("", None, True),
        ("", "", True),
        ("", "unused", True),
        ("secret", None, False),
        ("secret", "", False),
        ("secret", "wrong", False),
        ("secret", "secret", True),
        ("secret", ["secret"], False),
    ],
)
@pytest.mark.parametrize("bind", ["127.0.0.1", "192.0.2.10"])
def test_optional_token(required: str, supplied: object, accepted: bool, bind: str) -> None:
    async def run() -> None:
        assert ServerConfig().token == ""
        config = ServerConfig(bind=bind, token=required, origins={f"http://{bind}:8080"})
        async with (
            TestServer(create_app(config, _FakeSession)) as server,
            ClientSession() as client,
        ):
            async with client.ws_connect(
                server.make_url("/ws"), headers={"Origin": next(iter(config.origins))}
            ) as ws:
                message: dict[str, object] = {"type": "auth"}
                if supplied is not None:
                    message["token"] = supplied
                await ws.send_json(message)
                response = await ws.receive_json()
                if accepted:
                    assert response["type"] == "config"
                    await ws.send_json({"type": "disconnect"})
                else:
                    assert response["code"] == "unauthorized"

    asyncio.run(run())


@pytest.mark.parametrize(
    "bind, origin",
    [
        ("192.0.2.10", "http://192.0.2.10:8080"),
        ("2001:db8::10", "http://[2001:db8::10]:8080"),
        ("0.0.0.0", "http://render.test:8080"),
        ("::", "http://render.test:8080"),
    ],
)
def test_cli_direct_http(
    cli_configs: list[ServerConfig],
    capsys: pytest.CaptureFixture[str],
    bind: str,
    origin: str,
) -> None:
    arguments = ["--bind", bind, "--log-format", "json"]
    if bind in {"0.0.0.0", "::"}:
        arguments += ["--origin", origin]
    assert main(arguments) == 0
    assert json.loads(capsys.readouterr().out)["url"] == origin + "/"
    assert cli_configs[0].origins == ({origin} if bind in {"0.0.0.0", "::"} else set())
    assert cli_configs[0].cert is None and not cli_configs[0].behind_proxy


@pytest.mark.parametrize("host", ["render.test:8080", "192.0.2.10:8080", "[2001:db8::10]:8080"])
@pytest.mark.parametrize("origin_kind", ["same", "other", "missing"])
def test_default_origin_follows_requested_host(host: str, origin_kind: str) -> None:
    async def run() -> None:
        config = ServerConfig()
        assert config.bind == "0.0.0.0" and not config.origins
        headers = {"Host": host}
        if origin_kind != "missing":
            headers["Origin"] = f"http://{host}" if origin_kind == "same" else "http://other.test"
        async with (
            TestServer(create_app(config, _FakeSession)) as server,
            ClientSession() as client,
        ):
            if origin_kind != "same":
                with pytest.raises(WSServerHandshakeError) as error:
                    await client.ws_connect(server.make_url("/ws"), headers=headers)
                assert error.value.status == 403
            else:
                async with client.ws_connect(server.make_url("/ws"), headers=headers) as ws:
                    await ws.send_json({"type": "auth"})
                    assert (await ws.receive_json())["type"] == "config"
                    await ws.send_json({"type": "disconnect"})

    asyncio.run(run())


@pytest.mark.parametrize("arguments", [[], ["--token"], ["--token", "a&b +/#"]])
@pytest.mark.parametrize("remote", [False, True])
def test_cli_optional_token(
    cli_configs: list[ServerConfig],
    capsys: pytest.CaptureFixture[str],
    arguments: list[str],
    remote: bool,
) -> None:
    options = (
        ["--bind", "0.0.0.0", "--behind-proxy", "--origin", "https://render.test"] if remote else []
    )
    assert main(options + arguments + ["--log-format", "json"]) == 0
    url = json.loads(capsys.readouterr().out)["url"]
    token = cli_configs[0].token
    if not arguments:
        assert token == "" and not urlsplit(url).fragment
    else:
        assert token and (len(token) >= 32 if len(arguments) == 1 else token == arguments[1])
        assert parse_qs(urlsplit(url).fragment)["token"] == [token]


@pytest.mark.parametrize("bind", ["127.0.0.1", "0.0.0.0"])
@pytest.mark.parametrize(
    "origins", [[], ["http://render.test"], ["https://render.test", "http://other.test"]]
)
def test_proxy_requires_explicit_https_origins(
    cli_configs: list[ServerConfig],
    capsys: pytest.CaptureFixture[str],
    bind: str,
    origins: list[str],
) -> None:
    arguments = ["--behind-proxy", "--bind", bind]
    for origin in origins:
        arguments.extend(["--origin", origin])
    assert main(arguments) == 1
    assert "HTTPS" in capsys.readouterr().err
    assert not cli_configs


def test_loopback_proxy_advertises_public_origin(
    cli_configs: list[ServerConfig], capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["--behind-proxy", "--origin", "https://render.test", "--log-format", "json"]) == 0
    assert cli_configs[0].origins == {"https://render.test"}
    assert json.loads(capsys.readouterr().out)["url"] == "https://render.test/"


@pytest.mark.parametrize("token", ["", "test-token"])
def test_cli_human_readable_startup(
    cli_configs: list[ServerConfig],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    token: str,
) -> None:
    from falcor2.streaming import server

    def run_app(*args: object, **kwargs: Any) -> None:
        assert capsys.readouterr().out == ""
        kwargs["print"]("aiohttp listener started")

    monkeypatch.setattr(server.web, "run_app", run_app)
    assert main(["--token", token] if token else []) == 0
    output = capsys.readouterr().out
    assert "Open http://127.0.0.1:8080/" in output
    if token:
        assert f"#token={token}" in output


def test_server_reuses_owner_and_joins_it_on_shutdown(monkeypatch: pytest.MonkeyPatch) -> None:
    owners: list[threading.Thread] = []
    ended: list[bool] = []

    def run_session(self: StreamingSession) -> None:
        owners.append(threading.current_thread())
        self._stop.wait()
        ended.append(True)

    monkeypatch.setattr(StreamingSession, "_run", run_session)

    async def run() -> None:
        config = ServerConfig(token="test-token", origins={"http://allowed.test"})
        async with TestServer(create_app(config)) as server, ClientSession() as client:
            for index in range(3):
                async with client.ws_connect(
                    server.make_url("/ws"), headers={"Origin": "http://allowed.test"}
                ) as ws:
                    await ws.send_json({"type": "auth", "token": config.token})
                    assert (await ws.receive_json())["type"] == "config"
                    await ws.send_json({"type": "disconnect"})
                    while (await ws.receive()).type not in {WSMsgType.CLOSE, WSMsgType.CLOSED}:
                        pass
                assert len(ended) == index + 1
            assert len(owners) == 3 and all(owner is owners[0] for owner in owners)
            assert owners[0].is_alive()
            # Shutdown must also drain an active session before joining its owner.
            async with client.ws_connect(
                server.make_url("/ws"), headers={"Origin": "http://allowed.test"}
            ) as ws:
                await ws.send_json({"type": "auth", "token": config.token})
                assert (await ws.receive_json())["type"] == "config"
                await server.close()
            assert len(ended) == 4 and owners[-1] is owners[0]
        assert not owners[0].is_alive()

    asyncio.run(run())


@pytest.mark.parametrize("fps", [30, 60])
def test_signaling_lease_auth_origin_reconnect_and_cleanup(fps: int) -> None:
    async def run() -> None:
        config = ServerConfig(
            token="test-token",
            origins={"http://allowed.test"},
            fps=fps,
            width=3840,
            height=2160,
            max_viewers=1,
        )
        _FakeSession.closed = 0
        async with (
            TestServer(create_app(config, _FakeSession)) as server,
            ClientSession() as client,
        ):
            url = server.make_url("/ws")
            with pytest.raises(WSServerHandshakeError) as error:
                await client.ws_connect(url, headers={"Origin": "http://wrong.test"})
            assert error.value.status == 403
            for token in ["wrong", chr(233)]:
                async with client.ws_connect(
                    url, headers={"Origin": "http://allowed.test"}
                ) as wrong:
                    await wrong.send_json({"type": "auth", "token": token})
                    assert (await wrong.receive_json())["code"] == "unauthorized"
            async with client.ws_connect(url, headers={"Origin": "http://allowed.test"}) as first:
                await first.send_json({"type": "auth", "token": config.token})
                configuration = await first.receive_json()
                assert configuration["type"] == "config"
                assert configuration["generation"] == "test-generation"
                assert configuration["fps"] == fps
                assert (configuration["width"], configuration["height"]) == (3840, 2160)
                assert (await first.receive_json())["type"] == "offer"
                health = await (await client.get(server.make_url("/health"))).json()
                assert health["ready"] and health["busy"]
                async with client.ws_connect(
                    url, headers={"Origin": "http://allowed.test"}
                ) as second:
                    await second.send_json({"type": "auth", "token": config.token})
                    assert (await second.receive_json())["code"] == "busy"
                # Invalid signaling ends the lease and closes native resources.
                await first.send_json({"type": "candidate", "candidate": "unsupported"})
                for _ in range(5):
                    message = await first.receive_json()
                    if message["type"] == "error":
                        assert message["code"] == "session_error"
                        break
                else:
                    pytest.fail("Malformed signaling was not rejected")
                assert (await first.receive()).type in {WSMsgType.CLOSE, WSMsgType.CLOSED}
            assert _FakeSession.closed == 1
            # New connection after teardown gets a new offer and owns a fresh lease.
            async with client.ws_connect(url, headers={"Origin": "http://allowed.test"}) as third:
                await third.send_json({"type": "auth", "token": config.token})
                assert (await third.receive_json())["type"] == "config"
                assert (await third.receive_json())["type"] == "offer"
                await third.send_json({"type": "answer", "sdp": "unit-test-answer"})
                await third.send_json({"type": "disconnect"})
                while (await third.receive()).type not in {WSMsgType.CLOSE, WSMsgType.CLOSED}:
                    pass
            assert _FakeSession.closed == 2
            assert not (await (await client.get(server.make_url("/health"))).json())["busy"]

    asyncio.run(run())


def test_large_signaling_and_asset_allowlist() -> None:
    async def run() -> None:
        config = ServerConfig(token="test-token", origins={"http://allowed.test"})
        async with (
            TestServer(create_app(config, _FakeSession)) as server,
            ClientSession() as client,
        ):
            assert (await client.get(server.make_url("/server.py"))).status == 404
            response = await client.get(server.make_url("/"))
            assert response.status == 200
            assert "frame-ancestors 'none'" in response.headers["Content-Security-Policy"]
            async with client.ws_connect(
                server.make_url("/ws"), headers={"Origin": "http://allowed.test"}
            ) as ws:
                await ws.send_str("x" * 70000)
                assert (await ws.receive()).type in {WSMsgType.CLOSE, WSMsgType.CLOSED}
            assert not (await (await client.get(server.make_url("/health"))).json())["busy"]

    asyncio.run(run())


@pytest.mark.parametrize("scheme", ["turn", "TURN", "TuRn"])
def test_ice_config_is_explicit_and_validated(tmp_path: Path, scheme: str) -> None:
    assert load_ice_config(None) == []
    path = tmp_path / "ice.json"
    path.write_text(
        json.dumps(
            [{"url": f"{scheme}:relay.test:3478", "username": "user", "credential": "secret"}]
        )
    )
    servers = load_ice_config(path)
    assert servers[0]["username"] == "user"
    ServerConfig(force_relay=True, ice_servers=servers)
    path.write_text(json.dumps([{"url": f"{scheme}:relay.test:3478"}]))
    with pytest.raises(ValueError, match="credential"):
        load_ice_config(path)
    path.write_text(json.dumps([{"url": "https://example.test"}]))
    with pytest.raises(ValueError, match="ICE URL"):
        load_ice_config(path)
