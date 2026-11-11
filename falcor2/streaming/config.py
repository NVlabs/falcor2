# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared browser server and video configuration."""

import ipaddress
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

import slangpy as spy
from falcor2 import streaming


@dataclass
class StreamingConfig:
    """Shared browser viewers, with dimensions taken from EditorConfig.

    Port zero selects a free port. Local and remote binds use HTTP unless TLS is
    configured. Empty origins allow same-origin browser connections using the
    requested hostname or IP. Disconnect preserves the editor and its device.
    """

    bind: str = "0.0.0.0"
    port: int = 8080
    max_viewers: int | None = None
    fps: int = 30
    bitrate: int = 8000000
    min_bitrate: int = 1000000
    pacing_bitrate: int = 100000000
    token: str = ""
    origins: set[str] = field(default_factory=set)
    cert: Path | None = None
    key: Path | None = None
    behind_proxy: bool = False
    ice_servers: list[dict[str, str]] = field(default_factory=list)
    force_relay: bool = False
    ice_bind: str = ""
    port_begin: int = 50000
    port_end: int = 50100

    def _server_config(self, width: int, height: int) -> "ServerConfig":
        settings = asdict(self)
        settings.update(width=width, height=height)
        return ServerConfig(**settings)

    def _validate_server(self) -> None:
        ipaddress.ip_address(self.bind)
        if not 0 <= self.port <= 65535:
            raise ValueError("Invalid HTTP port")
        if self.max_viewers is not None and self.max_viewers < 1:
            raise ValueError("Maximum viewers must be positive")
        if bool(self.cert) != bool(self.key):
            raise ValueError("TLS requires both cert and key")
        if self.behind_proxy and not self.origins:
            raise ValueError("A reverse proxy requires explicit HTTPS origins")
        if self.behind_proxy and any(urlsplit(origin).scheme != "https" for origin in self.origins):
            raise ValueError("A reverse proxy requires explicit HTTPS origins")


@dataclass
class ServerConfig(StreamingConfig):
    """Validated streaming settings with frame dimensions and a standalone backend."""

    width: int = 1280
    height: int = 720
    backend: spy.DeviceType = spy.DeviceType.automatic

    def __post_init__(self) -> None:
        self._validate_server()
        if len(self.token) > 256:
            raise ValueError("Token must be at most 256 characters")
        if not 1 <= self.fps <= 60:
            raise ValueError("FPS must be between 1 and 60")
        if not (
            16 <= self.width <= 3840
            and 16 <= self.height <= 2160
            and self.width % 2 == 0
            and self.height % 2 == 0
        ):
            raise ValueError("Dimensions must be even, at least 16, and at most 3840x2160")
        if not 1 <= self.port_begin <= self.port_end <= 65535:
            raise ValueError("Invalid ICE port range")
        if not 100000 <= self.min_bitrate <= self.bitrate <= 50000000:
            raise ValueError("Require 100 kbit/s <= minimum <= bitrate <= 50 Mbit/s")
        if self.pacing_bitrate != 0 and not 100000 <= self.pacing_bitrate <= 100000000:
            raise ValueError("Pacing bitrate must be 0 or 100000..100000000 bits/s")
        if self.force_relay and not any(
            urlsplit(entry["url"]).scheme == "turn" for entry in self.ice_servers
        ):
            raise ValueError("Forced relay requires a TURN server")

    def _encoder_config(self) -> streaming.VideoEncoderConfig:
        config = streaming.VideoEncoderConfig()
        config.fps = self.fps
        config.width, config.height = self.width, self.height
        config.target_bitrate = config.max_bitrate = self.bitrate
        return config


def load_ice_config(path: Path | None) -> list[dict[str, str]]:
    """Read and validate STUN/TURN entries for the supported UDP transport."""
    if path is None:
        return []
    if path.stat().st_size > 16384:
        raise ValueError("ICE configuration exceeds 16 KiB")
    entries = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(entries, list) or len(entries) > 8:
        raise ValueError("ICE configuration must be a list of at most eight servers")
    for item in entries:
        if not isinstance(item, dict) or set(item) - {"url", "username", "credential"}:
            raise ValueError("ICE entries accept url, username and credential only")
        if not all(isinstance(value, str) and len(value) <= 2048 for value in item.values()):
            raise ValueError("Invalid ICE configuration field")
        if urlsplit(item.get("url", "")).scheme not in {"stun", "turn"}:
            raise ValueError(
                "ICE URL must use stun: or turn: (the pinned libjuice backend supports TURN/UDP only)"
            )
        if "transport=tcp" in item["url"].lower():
            raise ValueError("The pinned libjuice backend supports TURN/UDP only")
        if urlsplit(item["url"]).scheme == "turn" and not (
            item.get("username") and item.get("credential")
        ):
            raise ValueError("TURN requires username and credential")
    return entries
