# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit command-line configuration for applications using the editor."""

import argparse
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import fields
from pathlib import Path
import secrets
from typing import Any

from falcor2.editor.editor import EditorConfig
from falcor2.streaming.config import StreamingConfig, load_ice_config


class EditorArgumentParser(argparse.ArgumentParser):
    """Argument parser with shared editor, streaming, and device options.

    Standard ArgumentParser constructor arguments are forwarded unchanged.
    editor_defaults is copied at construction. Parsed editor options override it,
    including values supplied through set_defaults() or an existing namespace.
    parse_args() attaches a fresh, validated EditorConfig as editor_config on the
    returned namespace. Device creation stays under application control.

    Partial parsing with parse_known_args() retains argparse behavior and does
    not build editor_config.
    """

    def __init__(
        self,
        *args: Any,
        editor_defaults: EditorConfig | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._editor_defaults = deepcopy(editor_defaults)
        self._add_editor_arguments()
        self._add_device_arguments()

    def parse_args(
        self, args: Sequence[str] | None = None, namespace: argparse.Namespace | None = None
    ) -> argparse.Namespace:
        """Parse arguments and validate editor configuration before application startup.

        Configuration and ICE-file errors use normal argparse error handling:
        usage and exit status 2, or ArgumentError when exit_on_error is false.
        """
        if namespace is None:
            namespace = argparse.Namespace()
        result = super().parse_args(args, namespace)
        try:
            result.editor_config = self._editor_config_from_args(result)
        except (ValueError, OSError) as error:
            if not self.exit_on_error:
                raise argparse.ArgumentError(None, str(error)) from error
            self.error(str(error))
        return result

    def _add_editor_arguments(self) -> None:
        """Register editor options without parsing arguments or replacing app defaults.

        Convert the resulting namespace with _editor_config_from_args(). Unspecified
        settings are absent from the namespace.
        """
        editor = self.add_argument_group("Editor")
        editor.add_argument("--width", type=int, default=argparse.SUPPRESS, help="Frame width")
        editor.add_argument("--height", type=int, default=argparse.SUPPRESS, help="Frame height")
        window = editor.add_mutually_exclusive_group()
        window.add_argument(
            "--headless",
            dest="window",
            action="store_false",
            default=argparse.SUPPRESS,
            help="Render without a native window; add --stream for browser access",
        )
        window.add_argument(
            "--window",
            action="store_true",
            default=argparse.SUPPRESS,
            help="Show a native editor window",
        )
        editor.add_argument(
            "--vsync",
            action=argparse.BooleanOptionalAction,
            default=argparse.SUPPRESS,
            help="Enable native window vertical synchronization",
        )
        editor.add_argument(
            "--mcp",
            action=argparse.BooleanOptionalAction,
            default=argparse.SUPPRESS,
            help="Enable the editor MCP bridge",
        )
        stream = self.add_argument_group("Streaming")
        stream.add_argument(
            "--stream",
            action=argparse.BooleanOptionalAction,
            default=argparse.SUPPRESS,
            help="Enable browser streaming independently of the native window",
        )
        stream.add_argument(
            "--stream-port",
            type=int,
            default=argparse.SUPPRESS,
            help="HTTP port; 0 selects a free port (default: 8080)",
        )
        stream.add_argument(
            "--stream-max-viewers",
            type=int,
            default=argparse.SUPPRESS,
            help="Maximum simultaneous viewers, all with input enabled (default: unlimited)",
        )
        stream.add_argument(
            "--stream-fps",
            type=int,
            default=argparse.SUPPRESS,
            help="Video capture rate, 1..60 (default: 30); does not limit rendering",
        )
        stream.add_argument(
            "--stream-bitrate",
            type=int,
            default=argparse.SUPPRESS,
            help="Encoder target bits/s (default: 8000000)",
        )
        stream.add_argument(
            "--stream-min-bitrate",
            type=int,
            default=argparse.SUPPRESS,
            help="Minimum encoder bits/s (default: 1000000)",
        )
        stream.add_argument(
            "--stream-pacing-bitrate",
            type=int,
            default=argparse.SUPPRESS,
            help="Transport bits/s; 0 derives it from the encoder target",
        )
        stream.add_argument(
            "--stream-token",
            nargs="?",
            const=None,
            default=argparse.SUPPRESS,
            metavar="VALUE",
            help="Require a token; generate one when VALUE is omitted (default: no token)",
        )
        remote = self.add_argument_group("Streaming remote access")
        remote.add_argument(
            "--stream-bind",
            default=argparse.SUPPRESS,
            help="Signaling IP address (default: 0.0.0.0, all IPv4 interfaces)",
        )
        remote.add_argument(
            "--stream-origin",
            dest="stream_origins",
            action="append",
            default=argparse.SUPPRESS,
            help="Restrict browser origins; repeat for multiple origins (default: same-origin)",
        )
        remote.add_argument(
            "--stream-cert",
            type=Path,
            default=argparse.SUPPRESS,
            help="TLS certificate file",
        )
        remote.add_argument(
            "--stream-key",
            type=Path,
            default=argparse.SUPPRESS,
            help="TLS private key file",
        )
        remote.add_argument(
            "--stream-ice-config",
            type=Path,
            default=argparse.SUPPRESS,
            help="JSON file containing STUN/TURN server settings",
        )
        remote.add_argument(
            "--stream-behind-proxy",
            action=argparse.BooleanOptionalAction,
            default=argparse.SUPPRESS,
            help="Use an HTTPS reverse proxy; requires --stream-origin",
        )
        remote.add_argument(
            "--stream-force-relay",
            action=argparse.BooleanOptionalAction,
            default=argparse.SUPPRESS,
            help="Require a TURN relay from --stream-ice-config",
        )
        remote.add_argument(
            "--stream-ice-bind",
            default=argparse.SUPPRESS,
            help="Media bind address",
        )
        remote.add_argument(
            "--stream-ice-port-begin",
            dest="stream_port_begin",
            type=int,
            default=argparse.SUPPRESS,
            help="First media UDP port (default: 50000)",
        )
        remote.add_argument(
            "--stream-ice-port-end",
            dest="stream_port_end",
            type=int,
            default=argparse.SUPPRESS,
            help="Last media UDP port (default: 50100)",
        )

    def _editor_config_from_args(self, args: argparse.Namespace) -> EditorConfig:
        """Copy app defaults, apply supplied flags, and validate before GPU creation.

        Raises ValueError for invalid configuration, including streaming options
        supplied without streaming enabled. ICE file errors propagate. This function
        never parses sys.argv, creates devices, or starts services. Nested defaults
        are copied so the returned configuration can be edited independently.
        """
        config = (
            deepcopy(self._editor_defaults) if self._editor_defaults is not None else EditorConfig()
        )
        for name in ("width", "height", "window", "vsync", "mcp"):
            if hasattr(args, name):
                setattr(config, name, getattr(args, name))
        if config.width < 1 or config.height < 1:
            raise ValueError("Editor dimensions must be positive")
        overrides = {
            field.name: getattr(args, f"stream_{field.name}")
            for field in fields(StreamingConfig)
            if hasattr(args, f"stream_{field.name}")
        }
        enabled = getattr(args, "stream", config.streaming is not None)
        if not enabled:
            if overrides or hasattr(args, "stream_ice_config"):
                raise ValueError("Streaming options require --stream")
            config.streaming = None
            return config
        if "token" in overrides and overrides["token"] is None:
            overrides["token"] = secrets.token_urlsafe(32)
        if "origins" in overrides:
            overrides["origins"] = set(overrides["origins"])
        if hasattr(args, "stream_ice_config"):
            overrides["ice_servers"] = load_ice_config(args.stream_ice_config)
        config.streaming = config.streaming or StreamingConfig()
        for name, value in overrides.items():
            setattr(config.streaming, name, value)
        config.streaming._server_config(config.width, config.height)
        return config

    def _add_device_arguments(self) -> None:
        """Register device selection; the application still creates its own device."""
        group = self.add_argument_group("Device")
        group.add_argument(
            "--device-type",
            choices=("automatic", "d3d12", "vulkan", "cuda"),
            default="automatic",
            help="Device type used for rendering",
        )
