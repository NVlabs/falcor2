# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

JsonObject = dict[str, object]


def capture_editor_screenshot(
    editor: Any,
    params: JsonObject,
    *,
    workspace_root: Optional[Path] = None,
) -> JsonObject:
    output = editor.output
    if output is None:
        raise RuntimeError("No editor output is available yet. Render and present a frame first.")

    path = resolve_screenshot_path(params, workspace_root=workspace_root)
    editor.device.wait()

    from falcor2.editor.utils import save_image

    save_image(output.color_target, path)
    return {
        "path": str(path),
        "markdown_path": path.as_posix(),
        "width": output.width,
        "height": output.height,
        "mime_type": mime_type_for_screenshot(path),
    }


def resolve_screenshot_path(
    params: JsonObject,
    *,
    workspace_root: Optional[Path] = None,
) -> Path:
    path_value = params.get("path")
    if path_value is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
        path = (
            Path(tempfile.gettempdir()) / "falcor2-mcp" / "screenshots" / f"falcor2-{timestamp}.png"
        )
    else:
        if not isinstance(path_value, str) or not path_value.strip():
            raise ValueError("Screenshot path must be a non-empty string.")
        path = Path(path_value)
        if not path.is_absolute():
            path = (workspace_root or Path.cwd()) / path

    if path.suffix == "":
        path = path.with_suffix(".png")
    return path.resolve()


def mime_type_for_screenshot(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".jpg" or suffix == ".jpeg":
        return "image/jpeg"
    if suffix == ".exr":
        return "image/x-exr"
    if suffix == ".hdr":
        return "image/vnd.radiance"
    return "image/png"
