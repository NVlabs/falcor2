# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import inspect
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from os import PathLike
from pathlib import Path

import falcor2 as f2
import slangpy as spy
from falcor2.editor.utils import create_device, save_image
from falcor2.rendergraph import ContainerSpec
from falcor2.rendernodes import PathTracerPipeline


DEFAULT_DEVICE_TYPE = "automatic"
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
DEFAULT_TITLE = "Falcor2 PyScene Preview"
DEFAULT_INTERACTIVE_SPP = 1
DEFAULT_HEADLESS_SPP = 128
MAX_HEADLESS_SPP_PER_BATCH = 32
DEVICE_TYPE_CHOICES = ("automatic", "d3d12", "vulkan", "cuda")


def _positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return result


def create_preview_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=DEFAULT_TITLE)
    parser.add_argument(
        "--device-type",
        choices=DEVICE_TYPE_CHOICES,
        default=DEFAULT_DEVICE_TYPE,
    )
    parser.add_argument("--width", type=_positive_int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=_positive_int, default=DEFAULT_HEIGHT)
    parser.add_argument(
        "--spp",
        type=_positive_int,
        help=(
            f"Samples per pixel. Defaults to {DEFAULT_INTERACTIVE_SPP} interactively and "
            f"{DEFAULT_HEADLESS_SPP} when --out is set."
        ),
    )
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--no-tone-map",
        dest="tone_map",
        action="store_false",
        help="Disable artistic tone mapping; PNG output still uses its standard sRGB encoding.",
    )
    return parser


def parse_preview_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    args = create_preview_parser().parse_args(argv)
    if args.spp is None:
        args.spp = DEFAULT_HEADLESS_SPP if args.out is not None else DEFAULT_INTERACTIVE_SPP
    return args


def create_preview_pipeline(
    device: spy.Device,
    *,
    tone_map: bool = True,
) -> PathTracerPipeline:
    pipeline = PathTracerPipeline.create(device)
    pipeline.path_tracer.max_depth = 3
    pipeline.path_tracer.enable_nee = True
    pipeline.path_tracer.enable_mis = True
    pipeline.path_tracer.use_background_color = False
    pipeline.tone_map = tone_map
    return pipeline


def _find_external_caller_source() -> str | PathLike[str] | None:
    source_path: str | PathLike[str] | None = None
    frame = inspect.currentframe()
    try:
        caller = frame.f_back if frame is not None else None
        while caller is not None:
            module_name = caller.f_globals.get("__name__", "")
            candidate = caller.f_globals.get("__file__")
            if isinstance(candidate, (str, PathLike)) and not (
                module_name == "falcor2" or module_name.startswith("falcor2.")
            ):
                source_path = candidate
                break
            caller = caller.f_back
    finally:
        del frame

    if source_path is None:
        main_module = sys.modules.get("__main__")
        source_path = getattr(main_module, "__file__", None)
    return source_path


def _set_importer_source_path(
    importer: f2.Importer, source_path: str | PathLike[str] | None
) -> None:
    if importer.source_path != Path() or source_path is None:
        return
    importer.source_path = Path(source_path).resolve()


@dataclass(frozen=True)
class CameraRenderResult:
    camera_name: str
    camera_path: str
    path: Path
    width: int
    height: int


def _validate_render_size(width: int, height: int, spp: int) -> None:
    if width <= 0:
        raise ValueError("width must be positive")
    if height <= 0:
        raise ValueError("height must be positive")
    if spp <= 0:
        raise ValueError("spp must be positive")


def _camera_filename(index: int, name: str) -> str:
    slug = re.sub(r"[^a-z0-9_-]+", "-", name.lower(), flags=re.ASCII).strip("-_")
    return f"{index:03d}-{slug or 'camera'}.png"


def render_scene_cameras(
    device: spy.Device,
    scene: f2.Scene,
    output_dir: Path,
    *,
    width: int,
    height: int,
    spp: int,
    tone_map: bool = True,
) -> list[CameraRenderResult]:
    _validate_render_size(width, height, spp)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cameras = sorted(
        scene.components.find_all(type=f2.Camera),
        key=lambda camera: (str(camera.entity.name), str(camera.name)),
    )
    if not cameras:
        raise ValueError("preview scene does not contain a camera")

    pipeline = create_preview_pipeline(
        device,
        tone_map=tone_map,
    )
    pipeline.output_spec = ContainerSpec.texture2d(
        spy.Format.rgba32_float,
        (height, width),
    )
    results: list[CameraRenderResult] = []
    for index, camera in enumerate(cameras):
        camera_name = str(camera.name)
        camera_path = str(camera.entity.name)
        print(f"Rendering camera {index + 1}/{len(cameras)}: {camera_name}", flush=True)
        pipeline.reset()
        image = None
        remaining_spp = spp
        while remaining_spp > 0:
            pipeline.spp = min(remaining_spp, MAX_HEADLESS_SPP_PER_BATCH)
            image = pipeline(scene, camera=camera)
            device.wait()
            remaining_spp -= pipeline.spp
        assert image is not None
        path = output_dir / _camera_filename(index, camera_name)
        save_image(image, path)
        print(f"Saved: {path}", flush=True)
        results.append(
            CameraRenderResult(
                camera_name=camera_name,
                camera_path=camera_path,
                path=path,
                width=width,
                height=height,
            )
        )
    return results


def preview(argv: Sequence[str] | None = None) -> None:
    """Build and preview the edits recorded by the current importer."""
    args = parse_preview_args(argv)
    importer = f2.Importer.get()
    _set_importer_source_path(importer, _find_external_caller_source())

    device_type = getattr(spy.DeviceType, args.device_type)
    device = create_device(
        device_type=device_type,
        enable_cuda_interop=args.out is None,
    )
    importer_scene = importer.build_importer_scene()
    if not importer_scene.cameras:
        importer_scene.add_default_camera_best_view(50.0, args.width / args.height)
    scene = f2.Scene.from_importer_scene(device, importer_scene)
    importer.run_scene_loaded_callbacks(scene)
    if args.out is not None:
        render_scene_cameras(
            device,
            scene,
            args.out,
            width=args.width,
            height=args.height,
            spp=args.spp,
            tone_map=args.tone_map,
        )
        return

    pipeline = create_preview_pipeline(
        device,
        tone_map=args.tone_map,
    )
    pipeline.spp = args.spp

    from falcor2.editor import Editor, EditorConfig

    editor = Editor.create(
        device,
        config=EditorConfig(
            width=args.width,
            height=args.height,
            title=DEFAULT_TITLE,
            vsync=False,
        ),
        scene=scene,
    )
    while editor.update():
        if editor.needs_render:
            image = pipeline(scene, delta_time=editor.dt)
            editor.present(image)
