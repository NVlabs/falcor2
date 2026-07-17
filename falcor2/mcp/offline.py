# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import falcor2 as f2
import slangpy as spy
from falcor2.pyscene.preview import (
    DEFAULT_DEVICE_TYPE,
    DEFAULT_HEADLESS_SPP,
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
    CameraRenderResult,
    render_scene_cameras,
)
from falcor2.editor.utils import create_device

JsonObject = dict[str, object]


def python_info(params: JsonObject, context: JsonObject) -> JsonObject:
    return {
        "python": sys.executable,
        "version": sys.version,
        "cwd": os.getcwd(),
        "params": dict(params),
        "context": dict(context),
    }


def _required_path(params: JsonObject, name: str) -> Path:
    value = params.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return Path(value)


def _positive_int(params: JsonObject, name: str, default: int) -> int:
    value = params.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _workspace_path(path: Path, workspace_root: Path) -> Path:
    if not path.is_absolute():
        path = workspace_root / path
    return path.resolve()


def _default_output_dir() -> Path:
    root = Path(tempfile.gettempdir()) / "falcor2-mcp" / "renders"
    root.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="render-", dir=root)).resolve()


def _image_result(result: CameraRenderResult) -> JsonObject:
    path = result.path.resolve()
    return {
        "camera_name": result.camera_name,
        "path": str(path),
        "markdown_path": path.as_posix(),
        "width": result.width,
        "height": result.height,
        "mime_type": "image/png",
    }


def render_scene(params: JsonObject, context: JsonObject) -> JsonObject:
    workspace_value = context.get("workspace_root")
    if not isinstance(workspace_value, str) or not workspace_value:
        raise ValueError("context.workspace_root must be a non-empty string")
    workspace_root = Path(workspace_value).resolve()

    scene_path = _workspace_path(_required_path(params, "scene_path"), workspace_root)
    if not scene_path.is_file():
        raise ValueError(f"scene_path does not exist or is not a file: {scene_path}")

    out_value = params.get("out")
    if out_value is None:
        output_dir = _default_output_dir()
    elif isinstance(out_value, str) and out_value:
        output_dir = _workspace_path(Path(out_value), workspace_root)
    else:
        raise ValueError("out must be a non-empty string when provided")

    width = _positive_int(params, "width", DEFAULT_WIDTH)
    height = _positive_int(params, "height", DEFAULT_HEIGHT)
    spp = _positive_int(params, "spp", DEFAULT_HEADLESS_SPP)
    device_type_value = params.get("device_type", DEFAULT_DEVICE_TYPE)
    if not isinstance(device_type_value, str) or device_type_value not in {
        "automatic",
        "d3d12",
        "vulkan",
        "cuda",
    }:
        raise ValueError("device_type must be automatic, d3d12, vulkan, or cuda")

    device = create_device(device_type=getattr(spy.DeviceType, device_type_value))
    scene = f2.Scene.create(device, scene_path)
    results = render_scene_cameras(
        device,
        scene,
        output_dir,
        width=width,
        height=height,
        spp=spp,
    )
    return {
        "scene_path": str(scene_path),
        "output_dir": str(output_dir),
        "images": [_image_result(result) for result in results],
    }
