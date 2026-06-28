# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
from pathlib import Path

import falcor2 as f2
import slangpy as spy
from falcor2.editor import Editor, EditorConfig, create_device
from falcor2.rendernodes import PathTracerPipeline

DESCRIPTION = "Example PathTracer"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCENE_PATH = PROJECT_ROOT / "data/assets/cornell-box/usdpreviewsurface/cornell-box.usda"
DEFAULT_WIDTH = 1920
DEFAULT_HEIGHT = 1080


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument(
        "--scene-path",
        type=Path,
        default=DEFAULT_SCENE_PATH,
        help="Path to a scene file.",
    )
    parser.add_argument(
        "--device-type",
        choices=("automatic", "d3d12", "vulkan", "cuda"),
        default="automatic",
        help="Device type used for renderer.",
    )
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device_type = getattr(spy.DeviceType, args.device_type)
    device = create_device(device_type=device_type)

    scene = f2.Scene.create(device, args.scene_path)

    pipeline = PathTracerPipeline.create(device)
    pipeline.path_tracer.max_depth = 3
    pipeline.path_tracer.enable_nee = True
    pipeline.path_tracer.enable_mis = True
    pipeline.path_tracer.enable_analytic_lights = True
    pipeline.path_tracer.enable_environment_light = False
    pipeline.path_tracer.enable_emissive_triangles = True
    pipeline.path_tracer.env_map_as_background = False
    pipeline.tone_map = True

    editor = Editor.create(
        device,
        config=EditorConfig(
            width=args.width,
            height=args.height,
            title=DESCRIPTION,
            vsync=False,
        ),
        scene=scene,
    )

    while editor.update():
        if editor.needs_render:
            image = pipeline(scene)
            editor.present(image)


if __name__ == "__main__":
    main()
