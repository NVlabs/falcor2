# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import falcor2 as f2

from falcor2 import pyscene
from falcor2.editor import Editor

PROJECT_ROOT = Path(__file__).resolve().parents[3]
preview_module = importlib.import_module("falcor2.pyscene.preview")


def test_preview_parser_defaults() -> None:
    args = preview_module.parse_preview_args([])

    assert args.device_type == preview_module.DEFAULT_DEVICE_TYPE
    assert args.width == preview_module.DEFAULT_WIDTH
    assert args.height == preview_module.DEFAULT_HEIGHT
    assert args.spp == preview_module.DEFAULT_INTERACTIVE_SPP

    headless_args = preview_module.parse_preview_args(["--out", "renders"])
    assert headless_args.spp == preview_module.DEFAULT_HEADLESS_SPP


@pytest.mark.parametrize("option", ["--width", "--height", "--spp"])
def test_preview_parser_rejects_non_positive_values(option: str) -> None:
    with pytest.raises(SystemExit):
        preview_module.parse_preview_args([option, "0"])


def test_create_preview_pipeline_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline = SimpleNamespace(path_tracer=SimpleNamespace(), tone_map=False)
    monkeypatch.setattr(
        preview_module.PathTracerPipeline,
        "create",
        lambda device: pipeline,
    )

    result = preview_module.create_preview_pipeline(object())  # type: ignore[arg-type]

    assert result is pipeline
    assert pipeline.path_tracer.max_depth == 3
    assert pipeline.path_tracer.enable_nee is True
    assert pipeline.path_tracer.enable_mis is True
    assert pipeline.path_tracer.enable_analytic_lights is True
    assert pipeline.path_tracer.enable_environment_light is True
    assert pipeline.path_tracer.enable_emissive_triangles is True
    assert pipeline.path_tracer.env_map_as_background is True
    assert pipeline.tone_map is True


def test_preview_resolves_source_path_without_overwriting_explicit_value(tmp_path: Path) -> None:
    assert Path(preview_module._find_external_caller_source()).resolve() == Path(__file__).resolve()

    importer = f2.Importer.create()
    caller_path = tmp_path / "scene.py"
    preview_module._set_importer_source_path(importer, caller_path)
    assert importer.source_path == caller_path.resolve()

    explicit_path = tmp_path / "explicit.py"
    importer.source_path = explicit_path
    preview_module._set_importer_source_path(importer, tmp_path / "other.py")
    assert importer.source_path == explicit_path.resolve()


def test_preview_builds_scene_and_runs_editor(monkeypatch: pytest.MonkeyPatch) -> None:
    importer = SimpleNamespace(source_path=Path())
    scene = object()
    device = object()
    pipeline = SimpleNamespace(spp=0)
    calls: dict[str, Any] = {}

    class FakeScene:
        @staticmethod
        def create(*args: Any, **kwargs: Any) -> object:
            calls["scene_args"] = args
            calls["scene_kwargs"] = kwargs
            return scene

    class FakeImporter:
        @staticmethod
        def get() -> Any:
            return importer

    class FakeEditor:
        def run(self, renderer: Any) -> None:
            calls["renderer"] = renderer

    monkeypatch.setattr(
        preview_module,
        "f2",
        SimpleNamespace(Importer=FakeImporter, Scene=FakeScene),
    )
    monkeypatch.setattr(preview_module, "create_device", lambda device_type: device)
    monkeypatch.setattr(preview_module, "create_preview_pipeline", lambda actual_device: pipeline)
    monkeypatch.setattr(Editor, "create", lambda *args, **kwargs: FakeEditor())

    pyscene.preview(
        ["--device-type", "automatic", "--width", "640", "--height", "480", "--spp", "3"]
    )

    assert calls["scene_args"] == (device, importer)
    assert calls["scene_kwargs"] == {
        "add_default_camera_best_view": True,
        "camera_aspect": pytest.approx(4.0 / 3.0),
    }
    assert pipeline.spp == 3
    assert calls["renderer"] is pipeline
    assert importer.source_path == Path(__file__).resolve()


@pytest.mark.parametrize(
    ("width", "height", "spp", "message"),
    [(0, 1, 1, "width"), (1, 0, 1, "height"), (1, 1, 0, "spp")],
)
def test_render_scene_cameras_validates_positive_values(
    tmp_path: Path,
    width: int,
    height: int,
    spp: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        preview_module.render_scene_cameras(
            object(),  # type: ignore[arg-type]
            object(),  # type: ignore[arg-type]
            tmp_path,
            width=width,
            height=height,
            spp=spp,
        )


def test_render_scene_cameras_reuses_pipeline_and_writes_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cameras = [
        SimpleNamespace(name="Hero Camera", width=100, height=50),
        SimpleNamespace(name="Hero Camera", width=200, height=75),
        SimpleNamespace(name="", width=300, height=100),
        SimpleNamespace(name="Caf\u00e9 / Side", width=400, height=125),
    ]
    scene = SimpleNamespace(
        components=SimpleNamespace(find_all=lambda **kwargs: cameras),
        active_camera=cameras[0],
    )
    device = SimpleNamespace(wait=lambda: calls.append("wait"))
    calls: list[Any] = []

    class FakePipeline:
        def __init__(self) -> None:
            self.output_spec = None
            self.spp = 0
            self.reset_count = 0
            self.rendered: list[Any] = []

        def reset(self) -> None:
            self.reset_count += 1

        def __call__(self, actual_scene: Any, *, camera: Any) -> object:
            assert actual_scene is scene
            self.rendered.append(camera)
            return object()

    pipeline = FakePipeline()

    def fake_save_image(image: Any, path: Path) -> None:
        path.write_bytes(b"png")

    monkeypatch.setattr(preview_module, "create_preview_pipeline", lambda actual_device: pipeline)
    monkeypatch.setattr(preview_module, "save_image", fake_save_image)

    results = preview_module.render_scene_cameras(
        device,  # type: ignore[arg-type]
        scene,  # type: ignore[arg-type]
        tmp_path,
        width=64,
        height=32,
        spp=7,
    )

    assert [result.path.name for result in results] == [
        "000-hero-camera.png",
        "001-hero-camera.png",
        "002-camera.png",
        "003-caf-side.png",
    ]
    assert all(result.path.read_bytes() == b"png" for result in results)
    assert [result.camera_name for result in results] == [camera.name for camera in cameras]
    assert all((result.width, result.height) == (64, 32) for result in results)
    assert pipeline.output_spec.dims == (32, 64)
    assert pipeline.spp == 7
    assert pipeline.reset_count == len(cameras)
    assert pipeline.rendered == cameras
    assert calls == ["wait"] * len(cameras)
    assert scene.active_camera is cameras[0]
    assert [(camera.width, camera.height) for camera in cameras] == [
        (100, 50),
        (200, 75),
        (300, 100),
        (400, 125),
    ]


def test_guarded_scene_renders_headlessly_from_unrelated_cwd(tmp_path: Path) -> None:
    output_dir = tmp_path / "renders"
    child_cwd = tmp_path / "cwd"
    child_cwd.mkdir()
    env = os.environ.copy()
    existing_python_path = env.get("PYTHONPATH")
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(PROJECT_ROOT), existing_python_path) if part
    )

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "data" / "scenes" / "cornell-box-env.py"),
            "--out",
            str(output_dir),
            "--width",
            "64",
            "--height",
            "64",
            "--spp",
            "1",
        ],
        cwd=child_cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert "Rendering camera 1/2: Camera" in result.stdout
    assert "Rendering camera 2/2: camera" in result.stdout
    image_paths = [output_dir / "000-camera.png", output_dir / "001-camera.png"]
    assert all(path.exists() for path in image_paths)
    assert all(path.stat().st_size > 0 for path in image_paths)
