# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


@dataclass
class BridgeEndpoint:
    transport: str
    host: Optional[str]
    port: Optional[int]
    path: Optional[str]


@dataclass
class HeartbeatRecord:
    instance_id: str
    pid: int
    workspace_root: str
    falcor2_root: Optional[str]
    falcor2_package_path: Optional[str]
    falcor2_version: Optional[str]
    config_paths: list[str]
    endpoint: BridgeEndpoint
    endpoints: list[str]
    created_at: str
    last_heartbeat: str


def utc_now_text() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def default_registry_dir() -> Path:
    override = os.environ.get("FALCOR2_MCP_REGISTRY_DIR")
    if override:
        return Path(override)
    return Path(tempfile.gettempdir()) / "falcor2-mcp" / "instances"


def write_heartbeat(record: HeartbeatRecord, registry_dir: Optional[Path] = None) -> Path:
    directory = registry_dir or default_registry_dir()
    directory.mkdir(parents=True, exist_ok=True)

    path = directory / f"{record.instance_id}.json"
    temp_path = directory / f"{record.instance_id}.tmp"
    data = asdict(record)
    temp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temp_path.replace(path)
    return path
