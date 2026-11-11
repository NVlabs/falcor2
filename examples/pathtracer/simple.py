# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
from pathlib import Path

import falcor2 as f2
import slangpy as spy
from falcor2.editor import (
    Editor,
    EditorConfig,
    EditorArgumentParser,
    create_device,
)
from falcor2.rendernodes import PathTracerPipeline

DESCRIPTION = "Example PathTracer"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCENE_PATH = PROJECT_ROOT / "data/scenes/cornell-box.py"
DEFAULT_WIDTH = 1920
DEFAULT_HEIGHT = 1080


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = EditorArgumentParser(
        description=DESCRIPTION,
        editor_defaults=EditorConfig(
            width=DEFAULT_WIDTH,
            height=DEFAULT_HEIGHT,
            title=DESCRIPTION,
            vsync=False,
        ),
    )
    parser.add_argument(
        "--scene-path",
        type=Path,
        default=DEFAULT_SCENE_PATH,
        help="Path to a scene file.",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    device_type = getattr(spy.DeviceType, args.device_type)
    device = create_device(device_type=device_type, enable_cuda_interop=True)

    scene = f2.Scene.load(device, args.scene_path)

    pipeline = PathTracerPipeline.create(device)
    pipeline.path_tracer.max_depth = 3
    pipeline.path_tracer.enable_nee = True
    pipeline.path_tracer.enable_mis = True
    pipeline.path_tracer.use_background_color = False
    pipeline.tone_map = True

    editor = Editor.create(
        device,
        config=args.editor_config,
        scene=scene,
    )
    editor.run(lambda scene: pipeline(scene, delta_time=editor.dt))


if __name__ == "__main__":
    main()
