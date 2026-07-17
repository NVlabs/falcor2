# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
from pathlib import Path

import falcor2 as f2
import falcor2.ngx as fngx
import slangpy as spy
from falcor2.editor import Editor, EditorConfig
from falcor2.editor.utils import get_slang_include_paths
from falcor2.rendernodes import PathTracerPipeline

DESCRIPTION = "Example PathTracer with DLSS-RR"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCENE_PATH = PROJECT_ROOT / "data/scenes/cornell-box.py"
SLANG_SOURCE_PATH = Path(__file__).with_name("dlss_rr_viewer.slang")
DEFAULT_WIDTH = 1920
DEFAULT_HEIGHT = 1080

VIEW_INDICES = {
    "color": 0,
    "dlss_output": 1,
    "diffuse_albedo": 2,
    "specular_albedo": 3,
    "normals": 4,
    "roughness": 5,
    "depth": 6,
    "motion_vectors": 7,
    "specular_hit_distance": 8,
}

DEFAULT_VIEW_NAME = "dlss_output"


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


def create_ngx_ready_device(device_type: spy.DeviceType) -> spy.Device:
    """Create a device with any extra hooks NGX needs before device creation."""
    compiler_options = {"include_paths": get_slang_include_paths()}
    kwargs = {
        "type": device_type,
        "enable_debug_layers": True,
        "compiler_options": compiler_options,
    }

    if device_type == spy.DeviceType.vulkan:
        # NGX Vulkan support needs extensions enabled before the SlangPy device exists.
        info = fngx.get_vulkan_pre_device_info()
        kwargs["additional_vulkan_instance_extensions"] = info.required_vulkan_instance_extensions
        kwargs["additional_vulkan_device_extensions"] = info.required_vulkan_device_extensions

    return spy.Device(**kwargs)


def make_texture(
    device: spy.Device,
    width: int,
    height: int,
    format: spy.Format,
    label: str,
) -> spy.Texture:
    return device.create_texture(
        format=format,
        width=width,
        height=height,
        mip_count=1,
        usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
        label=label,
    )


class DlssRrViewer:
    """DLSS-RR viewer with selectable pipeline buffers."""

    def __init__(self, device: spy.Device) -> None:
        self._device = device
        self._pipeline = PathTracerPipeline.create(device)
        self._pipeline.enable_dlss_rr = True
        self._pipeline.path_tracer.max_depth = 3
        self._pipeline.path_tracer.enable_nee = True
        self._pipeline.path_tracer.enable_mis = True
        self._pipeline.path_tracer.enable_analytic_lights = True
        self._pipeline.path_tracer.enable_environment_light = True
        self._pipeline.path_tracer.enable_emissive_triangles = True
        self._pipeline.path_tracer.env_map_as_background = False

        self._display_output: spy.Texture | None = None
        self._view_module = None
        self._view_name = DEFAULT_VIEW_NAME

    @property
    def view_name(self) -> str:
        return self._view_name

    def cycle_view(self) -> None:
        names = self._view_names()
        index = names.index(self._view_name)
        self._view_name = names[(index + 1) % len(names)]
        print(f"DLSS-RR view: {self._view_name}")

    def render(self, scene: f2.Scene) -> spy.Texture | None:
        camera = scene.active_camera
        if camera is None:
            return None

        dlss_output, guides = self._pipeline(scene, camera, output_guides=True)
        if self._view_name == "dlss_output":
            return dlss_output
        buffers = dict(guides)
        buffers["dlss_output"] = dlss_output
        return self._visualize_buffers(buffers, dlss_output.width, dlss_output.height)

    def _view_names(self) -> list[str]:
        return list(VIEW_INDICES)

    def _ensure_display_output(self, width: int, height: int) -> spy.Texture:
        if (
            self._display_output is None
            or self._display_output.width != width
            or self._display_output.height != height
        ):
            self._display_output = make_texture(
                self._device,
                width,
                height,
                spy.Format.rgba32_float,
                "cornellbox_dlss_rr_display",
            )
        return self._display_output

    def _get_view_module(self):
        if self._view_module is None:
            self._view_module = spy.Module.load_from_file(self._device, str(SLANG_SOURCE_PATH))
        return self._view_module

    def _visualize_buffers(
        self,
        buffers: dict[str, spy.Texture | None],
        width: int,
        height: int,
    ) -> spy.Texture:
        output = self._ensure_display_output(width, height)
        self._get_view_module().visualize_dlss_rr_view(
            pixel=spy.grid((height, width)),
            view_index=VIEW_INDICES[self._view_name],
            color=buffers["color"],
            dlss_output=buffers["dlss_output"],
            diffuse_albedo=buffers["diffuse_albedo"],
            specular_albedo=buffers["specular_albedo"],
            normals=buffers["normals"],
            roughness=buffers["roughness"],
            depth=buffers["depth"],
            specular_hit_distance=buffers["specular_hit_distance"],
            motion_vectors=buffers["motion_vectors"],
            output=output,
        )
        return output


def main() -> None:
    args = parse_args()
    device_type = getattr(spy.DeviceType, args.device_type)
    device = create_ngx_ready_device(device_type)

    scene = f2.Scene.create(device, args.scene_path)
    viewer = DlssRrViewer(device)

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
    editor._cycle_render_mode = viewer.cycle_view
    print(f"DLSS-RR view: {viewer.view_name}")

    while editor.update():
        if editor.needs_render:
            image = viewer.render(scene)
            editor.present(image)


if __name__ == "__main__":
    main()
