# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render every camera in a scene to beauty and guide images."""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path
from typing import Sequence

import falcor2 as f2
import slangpy as spy
from falcor2.editor import create_device, load_scene, save_image
from falcor2.rendergraph import ContainerSpec
from falcor2.rendernodes import PathTracerPipeline


DEFAULT_SPP = 1024
DEFAULT_BOUNCES = 8
DEFAULT_WIDTH = 1920
DEFAULT_HEIGHT = 1080


def _positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return result


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render beauty and guide images from every camera in a scene."
    )
    parser.add_argument("scene", type=Path, help="Scene file to render.")
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        help="Output directory. Defaults to <scene>_render next to the scene.",
    )
    parser.add_argument("--spp", type=_positive_int, default=DEFAULT_SPP)
    parser.add_argument("--bounces", type=_positive_int, default=DEFAULT_BOUNCES)
    parser.add_argument("--width", type=_positive_int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=_positive_int, default=DEFAULT_HEIGHT)
    parser.add_argument(
        "--denoise",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also write CUDA OptiX-denoised beauty images (default: enabled).",
    )
    return parser


def _default_output_directory(scene_path: Path) -> Path:
    return scene_path.with_name(f"{scene_path.stem}_render")


def _prepare_output_directory(output_dir: Path, scene_path: Path) -> None:
    if output_dir.is_symlink():
        raise ValueError(f"output directory must not be a symbolic link: {output_dir}")

    resolved_output = output_dir.resolve()
    resolved_scene = scene_path.resolve()
    if resolved_scene == resolved_output or resolved_output in resolved_scene.parents:
        raise ValueError("output directory must not contain the input scene")
    if resolved_output == Path(resolved_output.anchor):
        raise ValueError("output directory must not be a filesystem root")

    if output_dir.exists():
        if not output_dir.is_dir():
            raise ValueError(f"output path exists and is not a directory: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)


def _camera_directory_name(index: int, camera: f2.Camera) -> str:
    name = str(camera.name) or str(camera.entity.name).replace("\\", "/").rsplit("/", 1)[-1]
    slug = re.sub(r"[^a-z0-9_-]+", "-", name.lower(), flags=re.ASCII).strip("-_")
    return f"{index:03d}-{slug or 'camera'}"


def _optix_image(image: spy.Tensor, width: int, height: int) -> f2.OptixImage2D:
    result = f2.OptixImage2D()
    result.buffer = image.storage
    result.width = width
    result.height = height
    result.row_stride_in_bytes = width * 4 * 4
    result.pixel_stride_in_bytes = 4 * 4
    result.format = f2.OptixPixelFormat.float4
    return result


def _create_denoiser(device: spy.Device, width: int, height: int) -> f2.OptixDenoiser:
    desc = f2.OptixDenoiserDesc()
    desc.model_kind = f2.OptixModelKind.aov
    desc.alpha_mode = f2.OptixAlphaMode.copy
    desc.albedo_guide_layer = True
    desc.normal_guide_layer = True
    desc.max_width = width
    desc.max_height = height
    return f2.OptixDenoiser(device, desc)


def _denoise(
    denoiser: f2.OptixDenoiser,
    image: spy.Tensor,
    albedo: spy.Tensor,
    normal: spy.Tensor,
    width: int,
    height: int,
) -> spy.Tensor:
    output = spy.Tensor.empty_like(image)

    layer = f2.OptixDenoiserLayer()
    layer.input = _optix_image(image, width, height)
    layer.output = _optix_image(output, width, height)
    layer.type = f2.OptixDenoiserAOVType.beauty

    guides = f2.OptixDenoiserGuideLayer()
    guides.albedo = _optix_image(albedo, width, height)
    guides.normal = _optix_image(normal, width, height)

    denoiser.denoise(f2.OptixDenoiserParams(), guides, [layer])
    return output


def render_scene(
    device: spy.Device,
    scene: f2.Scene,
    scene_path: Path,
    output_dir: Path,
    *,
    spp: int = DEFAULT_SPP,
    bounces: int = DEFAULT_BOUNCES,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    denoise: bool = True,
) -> None:
    """Render beauty and all reflected guide buffers from every scene camera."""
    cameras = sorted(
        scene.components.find_all(type=f2.Camera),
        key=lambda camera: (str(camera.entity.name), str(camera.name)),
    )
    if not cameras:
        raise ValueError("scene does not contain a camera")

    _prepare_output_directory(output_dir, scene_path)

    pipeline = PathTracerPipeline.create(device)
    pipeline.tone_map = False
    pipeline.spp = spp
    pipeline.path_tracer.max_depth = bounces
    pipeline.path_tracer.enable_nee = True
    pipeline.guide_output_specs = {
        name: ContainerSpec.tensor() for name in pipeline.guide_output_specs
    }
    pipeline.guide_output_specs["diffuse_albedo"] = ContainerSpec.tensor(spy.Format.rgba32_float)
    pipeline.guide_output_specs["normals"] = ContainerSpec.tensor(spy.Format.rgba32_float)

    optix_denoiser = _create_denoiser(device, width, height) if denoise else None

    for index, camera in enumerate(cameras):
        camera_dir = output_dir / _camera_directory_name(index, camera)
        camera_dir.mkdir()
        print(
            f"Rendering camera {index + 1}/{len(cameras)}: {camera.entity.name}",
            flush=True,
        )

        pipeline.reset()
        pipeline.output_spec = ContainerSpec.tensor(
            spy.Format.rgba32_float,
            (height, width),
        )
        linear_beauty, guides = pipeline(scene, camera=camera, output_guides=True)
        device.wait()

        save_image(linear_beauty, camera_dir / "beauty_linear.exr")
        beauty = pipeline.tonemapper(linear_beauty)
        device.wait()
        save_image(beauty, camera_dir / "beauty.png")

        if optix_denoiser is not None:
            albedo = guides["diffuse_albedo"]
            normal = guides["normals"]
            if albedo is None or normal is None:
                raise RuntimeError("path tracer did not produce OptiX guide buffers")
            linear_beauty_denoised = _denoise(
                optix_denoiser,
                linear_beauty,
                albedo,
                normal,
                width,
                height,
            )
            device.wait()
            save_image(
                linear_beauty_denoised,
                camera_dir / "beauty_linear_denoised.exr",
            )
            beauty_denoised = pipeline.tonemapper(linear_beauty_denoised)
            device.wait()
            save_image(beauty_denoised, camera_dir / "beauty_denoised.png")

        for name, guide in guides.items():
            if guide is not None:
                save_image(guide, camera_dir / f"{name}.exr")

        print(f"Saved: {camera_dir}", flush=True)


def main(argv: Sequence[str] | None = None) -> None:
    args = create_parser().parse_args(argv)
    scene_path = args.scene.resolve()
    if not scene_path.is_file():
        raise ValueError(f"scene does not exist or is not a file: {scene_path}")
    output_dir = (
        args.output.absolute() if args.output is not None else _default_output_directory(scene_path)
    )

    device = create_device(device_type=spy.DeviceType.cuda)
    scene = load_scene(device, scene_path)
    render_scene(
        device,
        scene,
        scene_path,
        output_dir,
        spp=args.spp,
        bounces=args.bounces,
        width=args.width,
        height=args.height,
        denoise=args.denoise,
    )


if __name__ == "__main__":
    main()
