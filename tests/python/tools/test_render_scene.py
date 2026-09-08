# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import slangpy as spy

from tools import render_scene


def test_parser_defaults_to_high_quality_denoised_render() -> None:
    args = render_scene.create_parser().parse_args(["scene.usda"])

    assert args.output is None
    assert args.spp == 1024
    assert args.bounces == 8
    assert args.width == 1920
    assert args.height == 1080
    assert args.denoise is True

    args = render_scene.create_parser().parse_args(
        [
            "scene.usda",
            "output",
            "--spp",
            "16",
            "--bounces",
            "4",
            "--width",
            "640",
            "--height",
            "360",
            "--no-denoise",
        ]
    )
    assert args.output == Path("output")
    assert args.spp == 16
    assert args.bounces == 4
    assert args.width == 640
    assert args.height == 360
    assert args.denoise is False


def test_prepare_output_directory_replaces_existing_contents(tmp_path: Path) -> None:
    scene_path = tmp_path / "example.usda"
    scene_path.write_text("#usda 1.0\n", encoding="utf-8")
    output_dir = render_scene._default_output_directory(scene_path)
    (output_dir / "old").mkdir(parents=True)
    (output_dir / "old" / "stale.exr").write_bytes(b"stale")

    render_scene._prepare_output_directory(output_dir, scene_path)

    assert output_dir.is_dir()
    assert list(output_dir.iterdir()) == []


def test_render_scene_renders_each_camera_and_all_guides(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    cameras = [
        SimpleNamespace(
            name="Camera B",
            entity=SimpleNamespace(name="/World/CameraB"),
            width=20,
            height=10,
        ),
        SimpleNamespace(
            name="Camera A",
            entity=SimpleNamespace(name="/World/CameraA"),
            width=40,
            height=30,
        ),
    ]
    scene = SimpleNamespace(
        components=SimpleNamespace(find_all=lambda **_kwargs: cameras),
    )
    waits: list[None] = []
    device = SimpleNamespace(wait=lambda: waits.append(None))
    saved: list[tuple[Any, Path]] = []
    denoise_calls: list[tuple[Any, ...]] = []
    denoiser = object()

    class FakePathTracer:
        def __init__(self) -> None:
            self.max_depth = 0
            self.enable_nee = False

    class FakePipeline:
        def __init__(self) -> None:
            self.tone_map = True
            self.spp = 0
            self.path_tracer = FakePathTracer()
            self.guide_output_specs = {
                "diffuse_albedo": None,
                "normals": None,
                "depth": None,
            }
            self.output_spec = None
            self.calls: list[tuple[Any, tuple[int, ...]]] = []
            self.reset_count = 0

        def reset(self) -> None:
            self.reset_count += 1

        def __call__(
            self,
            actual_scene: Any,
            *,
            camera: Any,
            output_guides: bool,
        ) -> tuple[str, dict[str, str]]:
            assert actual_scene is scene
            assert output_guides is True
            self.calls.append((camera, self.output_spec.dims))
            return (
                f"linear:{camera.name}",
                {
                    "diffuse_albedo": f"albedo:{camera.name}",
                    "normals": f"normal:{camera.name}",
                    "depth": f"depth:{camera.name}",
                },
            )

        def tonemapper(self, image: Any) -> str:
            return f"tonemapped:{image}"

    pipeline = FakePipeline()
    monkeypatch.setattr(
        render_scene.PathTracerPipeline,
        "create",
        lambda _device: pipeline,
    )
    monkeypatch.setattr(
        render_scene,
        "save_image",
        lambda image, path: saved.append((image, path)),
    )
    monkeypatch.setattr(
        render_scene,
        "_create_denoiser",
        lambda actual_device, width, height: denoise_calls.append(
            ("create", actual_device, width, height)
        )
        or denoiser,
    )

    def fake_denoise(
        actual_denoiser: Any,
        image: Any,
        albedo: Any,
        normal: Any,
        width: int,
        height: int,
    ) -> str:
        denoise_calls.append(("denoise", actual_denoiser, image, albedo, normal, width, height))
        return f"denoised:{image}"

    monkeypatch.setattr(render_scene, "_denoise", fake_denoise)

    scene_path = tmp_path / "scene.usda"
    scene_path.write_text("#usda 1.0\n", encoding="utf-8")
    output_dir = tmp_path / "render"
    render_scene.render_scene(
        device,  # type: ignore[arg-type]
        scene,  # type: ignore[arg-type]
        scene_path,
        output_dir,
        spp=64,
        bounces=6,
        width=80,
        height=45,
        denoise=True,
    )

    assert pipeline.tone_map is False
    assert pipeline.spp == 64
    assert pipeline.path_tracer.max_depth == 6
    assert pipeline.path_tracer.enable_nee is True
    assert pipeline.reset_count == 2
    assert pipeline.calls == [
        (cameras[1], (45, 80)),
        (cameras[0], (45, 80)),
    ]
    assert len(waits) == 8
    assert denoise_calls == [
        ("create", device, 80, 45),
        (
            "denoise",
            denoiser,
            "linear:Camera A",
            "albedo:Camera A",
            "normal:Camera A",
            80,
            45,
        ),
        (
            "denoise",
            denoiser,
            "linear:Camera B",
            "albedo:Camera B",
            "normal:Camera B",
            80,
            45,
        ),
    ]
    assert [path.relative_to(output_dir).as_posix() for _image, path in saved] == [
        "000-camera-a/beauty_linear.exr",
        "000-camera-a/beauty.png",
        "000-camera-a/beauty_linear_denoised.exr",
        "000-camera-a/beauty_denoised.png",
        "000-camera-a/diffuse_albedo.exr",
        "000-camera-a/normals.exr",
        "000-camera-a/depth.exr",
        "001-camera-b/beauty_linear.exr",
        "001-camera-b/beauty.png",
        "001-camera-b/beauty_linear_denoised.exr",
        "001-camera-b/beauty_denoised.png",
        "001-camera-b/diffuse_albedo.exr",
        "001-camera-b/normals.exr",
        "001-camera-b/depth.exr",
    ]
    assert saved[0][0] == "linear:Camera A"
    assert saved[1][0] == "tonemapped:linear:Camera A"
    assert saved[2][0] == "denoised:linear:Camera A"
    assert saved[3][0] == "tonemapped:denoised:linear:Camera A"


def test_main_loads_scene_on_cuda_and_uses_default_output(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    scene_path = tmp_path / "example.usda"
    scene_path.write_text("#usda 1.0\n", encoding="utf-8")
    device = object()
    scene = object()
    captured: dict[str, Any] = {}

    monkeypatch.setattr(
        render_scene,
        "create_device",
        lambda *, device_type: captured.update(device_type=device_type) or device,
    )
    monkeypatch.setattr(
        render_scene,
        "load_scene",
        lambda actual_device, actual_path: captured.update(
            load_device=actual_device,
            scene_path=actual_path,
        )
        or scene,
    )

    def fake_render_scene(
        actual_device: Any,
        actual_scene: Any,
        actual_scene_path: Path,
        output_dir: Path,
        **kwargs: Any,
    ) -> None:
        captured.update(
            render_device=actual_device,
            scene=actual_scene,
            render_scene_path=actual_scene_path,
            output_dir=output_dir,
            **kwargs,
        )

    monkeypatch.setattr(render_scene, "render_scene", fake_render_scene)

    render_scene.main(
        [
            str(scene_path),
            "--spp",
            "32",
            "--bounces",
            "5",
            "--width",
            "1920",
            "--height",
            "1080",
            "--no-denoise",
        ]
    )

    assert captured["device_type"] == spy.DeviceType.cuda
    assert captured["load_device"] is device
    assert captured["scene_path"] == scene_path.resolve()
    assert captured["render_device"] is device
    assert captured["scene"] is scene
    assert captured["render_scene_path"] == scene_path.resolve()
    assert captured["output_dir"] == tmp_path / "example_render"
    assert captured["spp"] == 32
    assert captured["bounces"] == 5
    assert captured["width"] == 1920
    assert captured["height"] == 1080
    assert captured["denoise"] is False
