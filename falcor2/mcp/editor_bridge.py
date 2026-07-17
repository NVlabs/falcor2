# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from falcor2.mcp.bridge import Bridge, JsonObject
from falcor2.mcp.screenshot import capture_editor_screenshot


def create_editor_bridge(
    editor: Any,
    *,
    workspace_root: Optional[Path] = None,
    falcor2_root: Optional[Path] = None,
    config_path: Optional[Path] = None,
) -> Bridge:
    """Create a Bridge with Falcor2 Editor-specific endpoint handlers."""

    config_path = config_path or Path(__file__).with_name("config.json")
    falcor2_root = falcor2_root or Path(__file__).resolve().parents[2]

    def status() -> JsonObject:
        return _editor_status(editor, bridge)

    bridge = Bridge(
        workspace_root=workspace_root or Path.cwd(),
        falcor2_root=falcor2_root,
        config_paths=[config_path],
        status_provider=status,
    )
    bridge.register_handler(
        "falcor2.app.screenshot",
        lambda params: capture_editor_screenshot(
            editor,
            params,
            workspace_root=bridge.workspace_root,
        ),
    )
    return bridge


def _editor_status(editor: Any, bridge: Bridge) -> JsonObject:
    scene = editor.scene
    scene_path = getattr(scene, "path", None) if scene is not None else None
    render_mode = editor.render_mode
    result: JsonObject = {
        "title": editor.config.title,
        "scene_loaded": scene is not None,
        "render_mode": getattr(render_mode, "name", str(render_mode)),
        "width": editor.config.width,
        "height": editor.config.height,
        "endpoints": bridge.endpoint_names(),
    }
    if scene_path is not None:
        result["scene_path"] = str(scene_path)
    return result
