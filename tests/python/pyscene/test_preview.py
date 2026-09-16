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
import falcor2.testing.helpers as helpers

from falcor2 import pyscene
from falcor2.editor import Editor

PROJECT_ROOT = Path(__file__).resolve().parents[3]
preview_module = importlib.import_module("falcor2.pyscene.preview")


def test_preview_parser_defaults() -> None:
    assert preview_module.DEFAULT_HEADLESS_SPP == 128

    args = preview_module.parse_preview_args([])

    assert args.device_type == preview_module.DEFAULT_DEVICE_TYPE
    assert args.width == preview_module.DEFAULT_WIDTH
    assert args.height == preview_module.DEFAULT_HEIGHT
    assert args.spp == preview_module.DEFAULT_INTERACTIVE_SPP
    assert args.tone_map is True

    headless_args = preview_module.parse_preview_args(["--out", "renders"])
    assert headless_args.spp == 128

    help_text = " ".join(preview_module.create_preview_parser().format_help().split())
    assert "Defaults to 1 interactively and 128 when --out is set." in help_text
    assert "--max-sample-luminance" not in help_text

    linear_args = preview_module.parse_preview_args(["--no-tone-map"])
    assert linear_args.tone_map is False


@pytest.mark.parametrize(
    "option",
    ["--width", "--height", "--spp"],
)
def test_preview_parser_rejects_non_positive_values(option: str) -> None:
    with pytest.raises(SystemExit):
        preview_module.parse_preview_args([option, "0"])


def test_create_preview_pipeline_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline = SimpleNamespace(
        path_tracer=SimpleNamespace(),
        tone_map=False,
    )
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
    assert pipeline.path_tracer.use_background_color is False
    assert pipeline.tone_map is True

    preview_module.create_preview_pipeline(  # type: ignore[arg-type]
        object(),
        tone_map=False,
    )
    assert pipeline.tone_map is False


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
    class FakeImporterScene:
        def __init__(self) -> None:
            self.cameras: list[object] = []

        def add_default_camera_best_view(self, focal_length: float, camera_aspect: float) -> None:
            calls["best_view_args"] = (focal_length, camera_aspect)
            self.cameras.append(object())

    importer_scene = FakeImporterScene()
    importer = SimpleNamespace(source_path=Path())
    scene = object()
    device = object()
    calls: dict[str, Any] = {}
    rendered_image = object()

    class FakePipeline:
        def __init__(self) -> None:
            self.spp = 0

        def __call__(self, actual_scene: Any, *, delta_time: float) -> object:
            calls["render_scene"] = actual_scene
            calls["delta_time"] = delta_time
            return rendered_image

    pipeline = FakePipeline()

    class FakeScene:
        @staticmethod
        def from_importer_scene(*args: Any, **kwargs: Any) -> object:
            calls["scene_args"] = args
            calls["scene_kwargs"] = kwargs
            return scene

    class FakeImporter:
        @staticmethod
        def get() -> Any:
            return importer

    importer.build_importer_scene = lambda: importer_scene
    importer.run_scene_loaded_callbacks = lambda actual_scene: calls.update(
        callback_scene=actual_scene
    )

    class FakeEditor:
        needs_render = True
        dt = 0.125

        def __init__(self) -> None:
            self._update_count = 0

        def update(self) -> bool:
            self._update_count += 1
            return self._update_count == 1

        def present(self, image: Any) -> None:
            calls["presented_image"] = image

    monkeypatch.setattr(
        preview_module,
        "f2",
        SimpleNamespace(Importer=FakeImporter, Scene=FakeScene),
    )

    def fake_create_device(
        device_type: Any,
        *,
        enable_cuda_interop: bool = False,
    ) -> object:
        calls["device_type"] = device_type
        calls["enable_cuda_interop"] = enable_cuda_interop
        return device

    monkeypatch.setattr(preview_module, "create_device", fake_create_device)

    def fake_create_preview_pipeline(
        actual_device: Any,
        *,
        tone_map: bool = True,
    ) -> FakePipeline:
        return pipeline

    monkeypatch.setattr(
        preview_module,
        "create_preview_pipeline",
        fake_create_preview_pipeline,
    )
    monkeypatch.setattr(Editor, "create", lambda *args, **kwargs: FakeEditor())

    pyscene.preview(
        [
            "--device-type",
            "automatic",
            "--width",
            "640",
            "--height",
            "480",
            "--spp",
            "3",
        ]
    )

    assert calls["best_view_args"] == (50.0, pytest.approx(4.0 / 3.0))
    assert calls["scene_args"] == (device, importer_scene)
    assert calls["scene_kwargs"] == {}
    assert calls["callback_scene"] is scene
    assert calls["enable_cuda_interop"] is True
    assert pipeline.spp == 3
    assert calls["render_scene"] is scene
    assert calls["delta_time"] == pytest.approx(0.125)
    assert calls["presented_image"] is rendered_image
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
        SimpleNamespace(
            name="Hero Camera",
            entity=SimpleNamespace(name="/World/Cameras/CameraB"),
            width=100,
            height=50,
        ),
        SimpleNamespace(
            name="Hero Camera",
            entity=SimpleNamespace(name="/World/Cameras/CameraA"),
            width=200,
            height=75,
        ),
        SimpleNamespace(
            name="",
            entity=SimpleNamespace(name="/World/Cameras/CameraC"),
            width=300,
            height=100,
        ),
        SimpleNamespace(
            name="Caf\u00e9 / Side",
            entity=SimpleNamespace(name="/World/Cameras/CameraA"),
            width=400,
            height=125,
        ),
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

    captured_pipeline_settings: list[bool] = []

    def fake_create_pipeline(
        actual_device: Any,
        *,
        tone_map: bool = True,
    ) -> FakePipeline:
        captured_pipeline_settings.append(tone_map)
        return pipeline

    monkeypatch.setattr(preview_module, "create_preview_pipeline", fake_create_pipeline)
    monkeypatch.setattr(preview_module, "save_image", fake_save_image)

    results = preview_module.render_scene_cameras(
        device,  # type: ignore[arg-type]
        scene,  # type: ignore[arg-type]
        tmp_path,
        width=64,
        height=32,
        spp=7,
        tone_map=False,
    )

    assert [result.path.name for result in results] == [
        "000-caf-side.png",
        "001-hero-camera.png",
        "002-hero-camera.png",
        "003-camera.png",
    ]
    assert all(result.path.read_bytes() == b"png" for result in results)
    expected_cameras = [cameras[3], cameras[1], cameras[0], cameras[2]]
    assert [result.camera_name for result in results] == [
        camera.name for camera in expected_cameras
    ]
    assert [result.camera_path for result in results] == [
        camera.entity.name for camera in expected_cameras
    ]
    assert all((result.width, result.height) == (64, 32) for result in results)
    assert pipeline.output_spec.dims == (32, 64)
    assert pipeline.spp == 7
    assert pipeline.reset_count == len(cameras)
    assert pipeline.rendered == expected_cameras
    assert calls == ["wait"] * len(cameras)
    assert captured_pipeline_settings == [False]
    assert scene.active_camera is cameras[0]
    assert [(camera.width, camera.height) for camera in cameras] == [
        (100, 50),
        (200, 75),
        (300, 100),
        (400, 125),
    ]


def test_render_scene_cameras_batches_large_sample_counts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    camera = SimpleNamespace(
        name="Camera",
        entity=SimpleNamespace(name="/World/Camera"),
    )
    scene = SimpleNamespace(
        components=SimpleNamespace(find_all=lambda **kwargs: [camera]),
    )
    waits: list[None] = []
    device = SimpleNamespace(wait=lambda: waits.append(None))

    class FakePipeline:
        def __init__(self) -> None:
            self.output_spec = None
            self.spp = 0
            self.batch_sizes: list[int] = []

        def reset(self) -> None:
            pass

        def __call__(self, actual_scene: Any, *, camera: Any) -> object:
            self.batch_sizes.append(self.spp)
            return object()

    pipeline = FakePipeline()
    monkeypatch.setattr(
        preview_module,
        "create_preview_pipeline",
        lambda actual_device, *, tone_map=True: pipeline,
    )
    monkeypatch.setattr(preview_module, "save_image", lambda image, path: None)

    preview_module.render_scene_cameras(
        device,  # type: ignore[arg-type]
        scene,  # type: ignore[arg-type]
        tmp_path,
        width=64,
        height=32,
        spp=65,
    )

    assert pipeline.batch_sizes == [32, 32, 1]
    assert len(waits) == 3


@pytest.mark.skipif(
    sys.platform.startswith("linux"),
    reason="Headless render currently crashes during Falcor2 shutdown on Linux",
)
def test_guarded_scene_renders_headlessly_from_unrelated_cwd(tmp_path: Path) -> None:
    # The child owns its graphics device. Release any session-cached parent device
    # so the smoke test does not depend on which GPU-backed tests ran before it.
    helpers.close_all_devices()
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

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Rendering camera 1/2: camera" in result.stdout
    assert "Rendering camera 2/2: Camera" in result.stdout
    image_paths = [output_dir / "000-camera.png", output_dir / "001-camera.png"]
    assert all(path.exists() for path in image_paths)
    assert all(path.stat().st_size > 0 for path in image_paths)
