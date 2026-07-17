# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from falcor2.mcp import offline
from falcor2.pyscene.preview import CameraRenderResult


def test_render_scene_resolves_paths_defaults_and_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    scene_path = workspace / "scene.py"
    scene_path.write_text("", encoding="utf-8")
    temp_root = tmp_path / "system-temp"
    temp_root.mkdir()
    device = object()
    scene = object()
    captured: dict[str, Any] = {}

    monkeypatch.setattr(offline.tempfile, "gettempdir", lambda: str(temp_root))
    monkeypatch.setattr(offline, "create_device", lambda device_type: device)
    monkeypatch.setattr(
        offline,
        "f2",
        SimpleNamespace(Scene=SimpleNamespace(create=lambda actual_device, path: scene)),
    )

    def fake_render(
        actual_device: Any,
        actual_scene: Any,
        output_dir: Path,
        **kwargs: Any,
    ) -> list[CameraRenderResult]:
        captured.update(
            device=actual_device,
            scene=actual_scene,
            output_dir=output_dir,
            kwargs=kwargs,
        )
        image_path = output_dir / "000-camera.png"
        return [CameraRenderResult("Camera", image_path, kwargs["width"], kwargs["height"])]

    monkeypatch.setattr(offline, "render_scene_cameras", fake_render)

    result = offline.render_scene(
        {"scene_path": "scene.py"},
        {"workspace_root": str(workspace)},
    )

    assert result["scene_path"] == str(scene_path.resolve())
    output_dir = Path(result["output_dir"])
    assert output_dir.parent == (temp_root / "falcor2-mcp" / "renders").resolve()
    assert captured["device"] is device
    assert captured["scene"] is scene
    assert captured["output_dir"] == output_dir
    assert captured["kwargs"] == {
        "width": offline.DEFAULT_WIDTH,
        "height": offline.DEFAULT_HEIGHT,
        "spp": offline.DEFAULT_HEADLESS_SPP,
    }
    assert result["images"] == [
        {
            "camera_name": "Camera",
            "path": str((output_dir / "000-camera.png").resolve()),
            "markdown_path": (output_dir / "000-camera.png").resolve().as_posix(),
            "width": offline.DEFAULT_WIDTH,
            "height": offline.DEFAULT_HEIGHT,
            "mime_type": "image/png",
        }
    ]


def test_render_scene_resolves_explicit_output_and_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path.resolve()
    scene_path = workspace / "scene.py"
    scene_path.write_text("", encoding="utf-8")
    captured: dict[str, Any] = {}

    monkeypatch.setattr(
        offline,
        "create_device",
        lambda device_type: captured.setdefault("device_type", device_type) or object(),
    )
    monkeypatch.setattr(
        offline,
        "f2",
        SimpleNamespace(Scene=SimpleNamespace(create=lambda device, path: object())),
    )
    monkeypatch.setattr(
        offline,
        "render_scene_cameras",
        lambda device, scene, output_dir, **kwargs: captured.update(
            output_dir=output_dir, kwargs=kwargs
        )
        or [],
    )

    result = offline.render_scene(
        {
            "scene_path": "scene.py",
            "out": "renders",
            "width": 64,
            "height": 32,
            "spp": 2,
            "device_type": "vulkan",
        },
        {"workspace_root": str(workspace)},
    )

    assert result["output_dir"] == str((workspace / "renders").resolve())
    assert captured["output_dir"] == (workspace / "renders").resolve()
    assert captured["kwargs"] == {"width": 64, "height": 32, "spp": 2}


def test_render_scene_validates_scene_before_device_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        offline,
        "create_device",
        lambda device_type: pytest.fail("device must not be created"),
    )

    with pytest.raises(ValueError, match="scene_path does not exist"):
        offline.render_scene(
            {"scene_path": "missing.py"},
            {"workspace_root": str(tmp_path)},
        )


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({}, "scene_path"),
        ({"scene_path": "scene.py", "width": 0}, "width"),
        ({"scene_path": "scene.py", "height": -1}, "height"),
        ({"scene_path": "scene.py", "spp": True}, "spp"),
        ({"scene_path": "scene.py", "device_type": "metal"}, "device_type"),
        ({"scene_path": "scene.py", "out": ""}, "out"),
    ],
)
def test_render_scene_validates_parameters(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    params: dict[str, object],
    message: str,
) -> None:
    scene_path = tmp_path / "scene.py"
    scene_path.write_text("", encoding="utf-8")
    monkeypatch.setattr(offline, "create_device", lambda device_type: object())

    with pytest.raises(ValueError, match=message):
        offline.render_scene(params, {"workspace_root": str(tmp_path)})
