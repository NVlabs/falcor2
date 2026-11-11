# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from falcor2.editor import EditorConfig, EditorArgumentParser, StreamingConfig


@pytest.fixture
def parser() -> EditorArgumentParser:
    return EditorArgumentParser()


@pytest.mark.parametrize("bind", ["192.0.2.10", "2001:db8::10", "0.0.0.0", "::"])
def test_direct_http_streaming(parser: EditorArgumentParser, bind: str) -> None:
    options = ["--headless", "--stream", "--stream-bind", bind]
    config = parser.parse_args(options).editor_config
    assert config.streaming.bind == bind
    assert config.streaming.cert is None and config.streaming.key is None
    assert not config.streaming.behind_proxy
    assert config.streaming.token == ""
    assert not config.streaming.origins


def test_streaming_defaults_to_all_ipv4_interfaces(parser: EditorArgumentParser) -> None:
    config = parser.parse_args(["--stream"]).editor_config.streaming
    assert config.bind == "0.0.0.0"
    assert not config.origins


def test_supplied_options_preserve_and_copy_application_defaults() -> None:
    defaults = EditorConfig(
        width=1920,
        height=1080,
        title="Application",
        vsync=False,
        mcp=False,
        window=False,
        streaming=StreamingConfig(
            port=8088,
            fps=45,
            origins={"https://example.org"},
            ice_servers=[{"url": "stun:example.org"}],
        ),
    )
    parser = EditorArgumentParser(editor_defaults=defaults)
    assert defaults.streaming is not None
    untouched = parser.parse_args([]).editor_config
    assert untouched == defaults and untouched is not defaults
    config = parser.parse_args(
        ["--width", "960", "--vsync", "--mcp", "--window", "--stream-fps", "60"]
    ).editor_config
    assert (config.width, config.height, config.title) == (960, 1080, "Application")
    assert config.vsync and config.mcp and config.window
    assert config.streaming.port == 8088 and config.streaming.fps == 60
    config.streaming.origins.clear()
    config.streaming.ice_servers[0]["url"] = "stun:changed.org"
    assert defaults.streaming.origins == {"https://example.org"}
    assert defaults.streaming.ice_servers == [{"url": "stun:example.org"}]
    assert defaults.width == 1920 and defaults.streaming.fps == 45
    disabled = parser.parse_args(["--no-stream"]).editor_config
    assert disabled.streaming is None and defaults.streaming is not None
    assert parser.parse_args([]).editor_config == defaults
    defaults.width = 640
    assert parser.parse_args([]).editor_config.width == 1920


@pytest.mark.parametrize("token", [[], ["secret"], [""]])
def test_stream_tokens(parser: EditorArgumentParser, token: list[str]) -> None:
    config = parser.parse_args(["--stream", "--stream-token", *token]).editor_config
    if token:
        assert config.streaming.token == token[0]
    else:
        assert len(config.streaming.token) >= 32
        other = parser.parse_args(["--stream", "--stream-token"]).editor_config
        assert config.streaming.token != other.streaming.token
