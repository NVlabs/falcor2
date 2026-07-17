# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F

import slangpy as spy
from falcor2.editor import Editor, EditorConfig, create_torch_device, load_scene
from falcor2.rendergraph import ContainerSpec
from falcor2.rendernodes import PathTracerPipeline

SCENE_PATH = "data/scenes/cornell-box-env.py"
VIEWPORT_WIDTH = 1280
VIEWPORT_HEIGHT = 720


def gaussian_kernel_1d(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    kernel = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0], device=device, dtype=dtype)
    return kernel / kernel.sum()


def blur_rgb(image: torch.Tensor) -> torch.Tensor:
    kernel = gaussian_kernel_1d(image.device, image.dtype)
    kernel_x = kernel.view(1, 1, 1, 5).repeat(3, 1, 1, 1)
    kernel_y = kernel.view(1, 1, 5, 1).repeat(3, 1, 1, 1)

    image = F.pad(image, (2, 2, 0, 0), mode="reflect")
    image = F.conv2d(image, kernel_x, groups=3)
    image = F.pad(image, (0, 0, 2, 2), mode="reflect")
    return F.conv2d(image, kernel_y, groups=3)


class NeonPostFx:
    """
    Fun silly post effect that applies a bloom and edge detection to the image, and blends it back with the original.
    This is just to demonstrate how you can easily use PyTorch to do custom post processing on path traced images.
    """

    def __init__(self):
        super().__init__()
        self.m_history: torch.Tensor | None = None

    def process(self, image: torch.Tensor) -> torch.Tensor:
        rgb = image[0:3].unsqueeze(0)
        luminance = (
            0.2126 * rgb[:, 0:1, :, :] + 0.7152 * rgb[:, 1:2, :, :] + 0.0722 * rgb[:, 2:3, :, :]
        )

        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            device=rgb.device,
            dtype=rgb.dtype,
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
            device=rgb.device,
            dtype=rgb.dtype,
        ).view(1, 1, 3, 3)

        edge_x = F.conv2d(F.pad(luminance, (1, 1, 1, 1), mode="reflect"), sobel_x)
        edge_y = F.conv2d(F.pad(luminance, (1, 1, 1, 1), mode="reflect"), sobel_y)
        edges = torch.sqrt(edge_x * edge_x + edge_y * edge_y)
        edges = torch.pow(edges.clamp(0.0, 2.5) / 2.5, 0.45)

        bright = torch.relu(rgb - 0.45) * 2.2
        bloom = blur_rgb(blur_rgb(bright))
        bloom = torch.pow(bloom.clamp_min(0.0), 0.85)

        edge_color = torch.cat(
            (
                edges * 0.20,
                edges * 1.10,
                edges * 1.80,
            ),
            dim=1,
        )

        stylized_rgb = torch.stack(
            (
                torch.pow(rgb[:, 0, :, :].clamp(0.0, 1.0), 0.75) * 1.05,
                torch.pow(rgb[:, 1, :, :].clamp(0.0, 1.0), 0.85) * 0.95,
                torch.pow(rgb[:, 2, :, :].clamp(0.0, 1.0), 0.65) * 1.25,
            ),
            dim=1,
        )

        output = (stylized_rgb * 0.72) + (bloom * 2.8) + (edge_color * 1.35)
        output = output + torch.cat(
            (
                bloom[:, 2:3, :, :] * 0.10,
                bloom[:, 1:2, :, :] * 0.06,
                bloom[:, 0:1, :, :] * 0.22,
            ),
            dim=1,
        )

        if self.m_history is None or self.m_history.shape != output.shape:
            self.m_history = output.detach().clone()
        else:
            self.m_history.mul_(0.72).add_(output.detach(), alpha=0.28)
            output = output * 0.65 + self.m_history * 0.35

        output = output.clamp(0.0, 1.0)
        alpha = image[3:4]
        return torch.cat((output.squeeze(0), alpha), dim=0)


device = create_torch_device()
scene = load_scene(device, SCENE_PATH)
camera = scene.active_camera
if camera is None:
    raise RuntimeError(f"Scene file does not contain an active camera: {SCENE_PATH}")

pipeline = PathTracerPipeline.create(device)
pipeline.path_tracer.max_depth = 3
pipeline.path_tracer.enable_nee = True
pipeline.path_tracer.enable_mis = True
pipeline.path_tracer.env_map_as_background = False
pipeline.tone_map = True
pipeline.output_spec = ContainerSpec.torch(format=spy.Format.rgba32_float)

post_fx = NeonPostFx()

editor = Editor.create(
    device,
    config=EditorConfig(
        width=VIEWPORT_WIDTH,
        height=VIEWPORT_HEIGHT,
        title="Falcor2 Cornell Box Path Tracer Viewer",
        vsync=False,
    ),
)

while editor.update(scene, camera):
    if editor.needs_render:
        image = pipeline(scene, camera)
        image = post_fx.process(image)
        editor.present(image)
