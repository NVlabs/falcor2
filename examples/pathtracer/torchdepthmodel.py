# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

import falcor2
import slangpy as spy
from falcor2.editor import Editor, EditorConfig, create_torch_device, load_scene
from falcor2.rendergraph import ContainerSpec
from falcor2.rendernodes import PathTracerPipeline


SCENE_PATH = Path("data/scenes/cornell-box-env.py")
MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
MODEL_INPUT_HEIGHT = 448
VIEWPORT_WIDTH = 1280
VIEWPORT_HEIGHT = 720
DEPTH_CONTOUR_REPEAT_COUNT = 8
DEPTH_OVERLAY_OPACITY = 0.2
SPLIT_VIEW = "--split" in sys.argv[1:]


def compute_model_input_size(
    width: int,
    height: int,
    max_height: int,
    alignment: int = 14,
) -> tuple[int, int]:
    if max_height <= 0:
        raise ValueError("max_height must be greater than zero.")
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be greater than zero.")
    if alignment <= 0:
        raise ValueError("alignment must be greater than zero.")

    if height <= max_height:
        return width, height

    scale = max_height / float(height)
    scaled_width = max(alignment, int(round((width * scale) / alignment)) * alignment)
    scaled_height = max(alignment, int(round((height * scale) / alignment)) * alignment)
    return scaled_width, scaled_height


def robust_normalize_depth(
    depth: torch.Tensor,
    low_quantile: float = 0.02,
    high_quantile: float = 0.98,
) -> torch.Tensor:
    depth = torch.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    flat = depth.reshape(-1)
    near_value = torch.quantile(flat, low_quantile)
    far_value = torch.quantile(flat, high_quantile)

    if torch.isclose(near_value, far_value):
        return torch.zeros_like(depth)

    normalized = (depth - near_value) / (far_value - near_value)
    return normalized.clamp(0.0, 1.0)


def resize_chw_image(image: torch.Tensor, width: int, height: int) -> torch.Tensor:
    resized = F.interpolate(
        image.unsqueeze(0),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )
    return resized.squeeze(0)


def compose_split_view(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    _, _, width = left.shape
    split_column = width // 2
    output = left.clone()
    output[:, :, split_column:] = right[:, :, split_column:]
    return output


def contour_signal(
    depth: torch.Tensor, repeat_count: int = DEPTH_CONTOUR_REPEAT_COUNT
) -> torch.Tensor:
    if repeat_count <= 0:
        raise ValueError("repeat_count must be greater than zero.")

    depth = depth.clamp(0.0, 1.0)
    contour_phase = torch.frac(depth * float(repeat_count))
    return 1.0 - torch.abs(contour_phase * 2.0 - 1.0)


def colorize_depth(depth: torch.Tensor) -> torch.Tensor:
    contour = contour_signal(depth)
    red_channel = contour
    green_channel = torch.zeros_like(contour)
    blue_channel = 1.0 - contour
    return torch.cat((red_channel, green_channel, blue_channel), dim=1).clamp(0.0, 1.0)


def blend_depth_overlay(
    rgb: torch.Tensor,
    depth_rgb: torch.Tensor,
    opacity: float = DEPTH_OVERLAY_OPACITY,
) -> torch.Tensor:
    if not 0.0 <= opacity <= 1.0:
        raise ValueError("opacity must be in the range [0, 1].")

    return torch.lerp(rgb, depth_rgb, opacity).clamp(0.0, 1.0)


class DepthAnythingViewer:
    def __init__(self, split_view: bool):
        super().__init__()
        self.m_processor = AutoImageProcessor.from_pretrained(MODEL_ID)
        self.m_model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.m_model = AutoModelForDepthEstimation.from_pretrained(MODEL_ID).to(self.m_model_device)
        self.m_model.eval()
        self.m_split_view = split_view

    def _prepare_model_image(self, rgb: torch.Tensor):
        width = int(rgb.shape[2])
        height = int(rgb.shape[1])
        resized_width, resized_height = compute_model_input_size(width, height, MODEL_INPUT_HEIGHT)
        if (resized_width, resized_height) != (width, height):
            rgb = resize_chw_image(rgb, resized_width, resized_height)

        return rgb.detach().clamp(0.0, 1.0).permute(1, 2, 0).mul(255.0).round().to(torch.uint8)

    def _infer_depth(self, rgb: torch.Tensor) -> torch.Tensor:
        inputs = self.m_processor(images=self._prepare_model_image(rgb), return_tensors="pt")
        inputs = {key: value.to(self.m_model_device) for key, value in inputs.items()}

        with torch.inference_mode():
            predicted_depth = self.m_model(**inputs).predicted_depth

        predicted_depth = (
            F.interpolate(
                predicted_depth.unsqueeze(1),
                size=(rgb.shape[1], rgb.shape[2]),
                mode="bicubic",
                align_corners=False,
            )
            .squeeze(1)
            .squeeze(0)
        )
        predicted_depth = predicted_depth.to(device=rgb.device, dtype=rgb.dtype)
        return robust_normalize_depth(predicted_depth)

    def process(self, image: torch.Tensor) -> torch.Tensor:
        rgb = image[0:3]
        alpha = image[3:4]
        depth = self._infer_depth(rgb).view(1, 1, rgb.shape[1], rgb.shape[2])
        depth_rgb = colorize_depth(depth).squeeze(0)
        overlay_rgb = blend_depth_overlay(rgb, depth_rgb)
        overlay_rgba = torch.cat((overlay_rgb, alpha), dim=0)

        if self.m_split_view:
            beauty_rgba = torch.cat((rgb, alpha), dim=0)
            return compose_split_view(beauty_rgba, overlay_rgba).clamp(0.0, 1.0)

        return overlay_rgba.clamp(0.0, 1.0)


def create_scene_and_camera(
    device: spy.Device,
    scene_path: Path,
) -> tuple[falcor2.Scene, falcor2.Camera]:
    scene = load_scene(device, scene_path)
    camera = scene.active_camera
    if camera is None:
        raise RuntimeError(f"Scene file does not contain an active camera: {scene_path}")

    return scene, camera


def main() -> None:
    device = create_torch_device()
    scene, camera = create_scene_and_camera(
        device=device,
        scene_path=SCENE_PATH,
    )

    pipeline = PathTracerPipeline.create(device)
    pipeline.path_tracer.max_depth = 3
    pipeline.path_tracer.enable_nee = True
    pipeline.path_tracer.enable_mis = True
    pipeline.path_tracer.env_map_as_background = False
    pipeline.tone_map = True
    pipeline.output_spec = ContainerSpec.torch(format=spy.Format.rgba32_float)
    pipeline.spp = 16

    viewer = DepthAnythingViewer(split_view=SPLIT_VIEW)

    editor = Editor.create(
        device,
        config=EditorConfig(
            width=VIEWPORT_WIDTH,
            height=VIEWPORT_HEIGHT,
            title="PyTorch Depth Model Viewer",
            vsync=False,
        ),
    )

    print(f"Loading depth model '{MODEL_ID}' complete.")
    if SPLIT_VIEW:
        print("Presenting beauty on the left and a contour depth overlay on the right.")
    else:
        print(
            "Presenting a contour depth overlay. Pass --split to compare against the beauty pass."
        )

    while editor.update(scene, camera):
        if editor.needs_render:
            image = pipeline(scene, camera)
            editor.present(viewer.process(image))


if __name__ == "__main__":
    main()
