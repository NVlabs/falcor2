# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import falcor2 as f2
import falcor2.ngx as fngx
import slangpy as spy
from falcor2.editor import Editor, EditorConfig
from falcor2.editor.utils import get_slang_include_paths
from falcor2.rendergraph import ContainerSpec
from falcor2.rendernodes import (
    AccumulatorNode,
    DLSSFrameGenNode,
    DLSSSuperResNode,
    ReferencePathTracerNode,
)
from falcor2.utils.jitter import frame_jitter

DESCRIPTION = "Example PathTracer with DLSS Super Resolution and Frame Generation"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCENE_PATH = PROJECT_ROOT / "data/assets/cornell-box/usdpreviewsurface/cornell-box.usda"
SLANG_SOURCE_PATH = Path(__file__).with_name("dlss_viewer.slang")
DEFAULT_WIDTH = 1920
DEFAULT_HEIGHT = 1088

VIEW_INDICES = {
    "path_color": 0,
    "dlss_sr": 1,
    "dlss_fg": 2,
    "hardware_depth": 3,
    "motion_vectors": 4,
}


@dataclass(frozen=True)
class DlssSupport:
    """Capabilities used by the DLSS demo."""

    super_resolution: bool
    frame_generation: bool


def enabled_view_names(support: DlssSupport) -> list[str]:
    """Return the demo views enabled for the detected NGX capabilities."""
    views = ["path_color"]
    if support.super_resolution:
        views.append("dlss_sr")
    if support.frame_generation:
        views.append("dlss_fg")
    views.extend(["hardware_depth", "motion_vectors"])
    return views


def default_view_name(support: DlssSupport) -> str:
    """Return the most useful initial view for the detected capabilities."""
    if support.frame_generation:
        return "dlss_fg"
    if support.super_resolution:
        return "dlss_sr"
    return "path_color"


def validate_samples_per_frame(samples_per_frame: int) -> int:
    """Return a valid positive samples-per-frame count."""
    if samples_per_frame <= 0:
        raise ValueError("DLSS demo samples per frame must be positive.")
    return samples_per_frame


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
    parser.add_argument(
        "--quality",
        choices=("dlaa", "quality", "balanced", "performance", "ultra_performance"),
        default="ultra_performance",
        help="DLSS-SR quality mode.",
    )
    parser.add_argument(
        "--disable-sr",
        action="store_true",
        help="Render without DLSS Super Resolution even if it is supported.",
    )
    parser.add_argument(
        "--disable-fg",
        action="store_true",
        help="Render without DLSS Frame Generation even if it is supported.",
    )
    parser.add_argument(
        "--require-supported",
        action="store_true",
        help="Exit with an error if requested DLSS features are unavailable.",
    )
    parser.add_argument(
        "--samples-per-frame",
        type=int,
        default=32,
        help="Path-tracer samples accumulated before each DLSS evaluate.",
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
    """Create a shader-readable and writable 2D texture."""
    return device.create_texture(
        format=format,
        width=width,
        height=height,
        mip_count=1,
        usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
        label=label,
    )


def _default_color_format(device: spy.Device) -> spy.Format:
    """Return a writable HDR color format for the selected backend."""
    if device.desc.type == spy.DeviceType.cuda:
        return spy.Format.rgba32_float
    return spy.Format.rgba16_float


class DlssViewer:
    """DLSS-SR and DLSS-G viewer with selectable intermediate buffers."""

    def __init__(
        self,
        device: spy.Device,
        support: DlssSupport,
        quality: fngx.QualityMode,
        samples_per_frame: int = 32,
    ) -> None:
        self._device = device
        self._support = support
        self._samples_per_frame = validate_samples_per_frame(samples_per_frame)
        self._path_tracer = self._create_path_tracer()
        self._accumulation_path_tracer = self._create_path_tracer()
        self._accumulator = AccumulatorNode.create(device)
        self._prev_camera_uniforms = None

        self._dlss_sr = DLSSSuperResNode.create(device)
        self._dlss_sr.quality = quality
        self._dlss_fg = DLSSFrameGenNode.create(device)

        self._display_output: spy.Texture | None = None
        self._view_module = None
        self._view_name = default_view_name(support)
        self._view_names = enabled_view_names(support)
        self._iteration = 0
        self._last_render_key: (
            tuple[int, int, int, int, fngx.QualityMode, bool, bool, int] | None
        ) = None

    @property
    def view_name(self) -> str:
        """Currently selected display view."""
        return self._view_name

    @property
    def support(self) -> DlssSupport:
        """DLSS capabilities enabled for this viewer."""
        return self._support

    def cycle_view(self) -> None:
        """Cycle through the available color and guide views."""
        index = self._view_names.index(self._view_name)
        self._view_name = self._view_names[(index + 1) % len(self._view_names)]
        print(f"DLSS view: {self._view_name}")

    def render(self, scene: f2.Scene) -> spy.Texture | None:
        """Render one frame and return the currently selected view texture."""
        camera = scene.active_camera
        if camera is None:
            return None

        target_width = int(camera.width)
        target_height = int(camera.height)
        color_spec, guide_specs = self._get_path_tracer_specs(target_width, target_height)
        if not isinstance(color_spec.dims, tuple) or len(color_spec.dims) != 2:
            raise ValueError("DLSS demo color dimensions must be two-dimensional.")
        render_height, render_width = color_spec.dims

        render_key = (
            int(render_width),
            int(render_height),
            target_width,
            target_height,
            self._dlss_sr.quality,
            self._support.super_resolution,
            self._support.frame_generation,
            self._samples_per_frame,
        )
        if render_key != self._last_render_key:
            self._iteration = 0
            self._path_tracer.reset()
            self._accumulation_path_tracer.reset()
            self._accumulator.reset()
            self._dlss_sr.reset()
            self._dlss_fg.reset()
            self._last_render_key = render_key

        jitter_x, jitter_y = frame_jitter(self._iteration)
        color, guides = self._render_accumulated_color(
            scene,
            camera,
            color_spec,
            guide_specs,
            spy.float2(jitter_x, jitter_y),
        )

        dlss_sr_output = color
        if self._support.super_resolution:
            self._dlss_sr.output_spec = ContainerSpec.texture2d(
                _default_color_format(self._device),
                (target_height, target_width),
            )
            dlss_sr_output = self._dlss_sr(
                color,
                guides,
                target_width,
                target_height,
                jitter_offset=spy.float2(-jitter_x, -jitter_y),
            )

        dlss_fg_output: spy.Texture | None = None
        if self._support.frame_generation:
            dlss_fg_output = self._dlss_fg(
                dlss_sr_output,
                guides,
                camera,
                jitter_offset=spy.float2(-jitter_x, -jitter_y),
            )

        self._iteration += 1

        buffers = {
            "path_color": color,
            "dlss_sr": dlss_sr_output,
            "dlss_fg": dlss_fg_output,
            "hardware_depth": guides["hardware_depth"],
            "motion_vectors": guides["motion_vectors"],
        }
        selected = buffers[self._view_name]
        if selected is not None and self._view_name in {"dlss_sr", "dlss_fg"}:
            return selected
        return self._visualize_buffers(buffers, target_width, target_height)

    def _create_path_tracer(self) -> ReferencePathTracerNode:
        """Create one path tracer node with the demo's render settings."""
        path_tracer = ReferencePathTracerNode.create(self._device)
        path_tracer.max_depth = 3
        path_tracer.enable_nee = True
        path_tracer.enable_mis = True
        path_tracer.enable_analytic_lights = True
        path_tracer.enable_environment_light = True
        path_tracer.enable_emissive_triangles = True
        path_tracer.env_map_as_background = False
        return path_tracer

    def _render_accumulated_color(
        self,
        scene: f2.Scene,
        camera: f2.Camera,
        color_spec: ContainerSpec,
        guide_specs: dict[str, ContainerSpec],
        subpixel_offset: spy.float2,
    ) -> tuple[spy.Texture, dict[str, spy.Texture | None]]:
        """Render guides once and accumulate several color samples for this frame."""
        base_iteration = self._iteration * self._samples_per_frame
        # self._accumulator.reset()

        if camera.get_uniforms() != self._prev_camera_uniforms:
            self._accumulator.reset()
            self._prev_camera_uniforms = camera.get_uniforms()

        self._path_tracer.output_spec = color_spec
        self._path_tracer.guide_output_specs = guide_specs
        first_color, guides = self._path_tracer(
            scene,
            camera,
            iteration=base_iteration,
            subpixel_offset=subpixel_offset,
            subpixel_random_jitter=0.0,
        )
        accumulated_color = self._accumulator(first_color)

        self._accumulation_path_tracer.output_spec = color_spec
        self._accumulation_path_tracer.guide_output_specs = {}
        for sample_index in range(1, self._samples_per_frame):
            sample_color, _ = self._accumulation_path_tracer(
                scene,
                camera,
                iteration=base_iteration + sample_index,
                subpixel_offset=subpixel_offset,
                subpixel_random_jitter=0.01,
            )
            accumulated_color = self._accumulator(sample_color)

        return accumulated_color, guides

    def _get_path_tracer_specs(
        self,
        target_width: int,
        target_height: int,
    ) -> tuple[ContainerSpec, dict[str, ContainerSpec]]:
        """Return color and guide specs for the current SR mode."""
        if self._support.super_resolution:
            return self._dlss_sr.get_optimal_specs(target_width, target_height)
        dims = (target_height, target_width)
        return (
            ContainerSpec.texture2d(_default_color_format(self._device), dims),
            {
                "hardware_depth": ContainerSpec.texture2d(spy.Format.r32_float, dims),
                "motion_vectors": ContainerSpec.texture2d(self._motion_vector_format(), dims),
            },
        )

    def _motion_vector_format(self) -> spy.Format:
        """Return the motion-vector format for generated guide textures."""
        if self._device.desc.type == spy.DeviceType.cuda:
            return spy.Format.rg32_float
        return spy.Format.rg16_float

    def _ensure_display_output(self, width: int, height: int) -> spy.Texture:
        """Create or reuse the display texture used by guide visualizations."""
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
                "cornellbox_dlss_display",
            )
        return self._display_output

    def _get_view_module(self):
        """Load the display shader lazily."""
        if self._view_module is None:
            self._view_module = spy.Module.load_from_file(self._device, str(SLANG_SOURCE_PATH))
        return self._view_module

    def _visualize_buffers(
        self,
        buffers: dict[str, spy.Texture | None],
        width: int,
        height: int,
    ) -> spy.Texture:
        """Render a displayable view of a DLSS input, output, or guide texture."""
        output = self._ensure_display_output(width, height)
        self._get_view_module().visualize_dlss_view(
            pixel=spy.grid((height, width)),
            view_index=VIEW_INDICES[self._view_name],
            path_color=buffers["path_color"],
            dlss_sr=buffers["dlss_sr"],
            dlss_fg=buffers["dlss_fg"] or buffers["dlss_sr"],
            hardware_depth=buffers["hardware_depth"],
            motion_vectors=buffers["motion_vectors"],
            output=output,
        )
        return output


def get_enabled_support(
    ngx: fngx.NGX,
    enable_sr: bool,
    enable_fg: bool,
    require_supported: bool,
) -> DlssSupport:
    """Resolve requested feature toggles against runtime NGX capabilities."""
    sr_supported = bool(ngx.info.dlss_supported)
    fg_supported = bool(ngx.info.frame_generation_supported)
    support = DlssSupport(
        super_resolution=enable_sr and sr_supported,
        frame_generation=enable_fg and fg_supported,
    )

    missing = []
    if enable_sr and not sr_supported:
        missing.append("DLSS Super Resolution")
    if enable_fg and not fg_supported:
        missing.append("DLSS Frame Generation")
    if require_supported and missing:
        raise RuntimeError("Requested DLSS features are unavailable: " + ", ".join(missing))
    return support


def print_support(ngx: fngx.NGX, support: DlssSupport) -> None:
    """Print a concise NGX capability summary for the demo."""
    print(f"DLSS Super Resolution: {'enabled' if support.super_resolution else 'disabled'}")
    print(f"DLSS Frame Generation: {'enabled' if support.frame_generation else 'disabled'}")
    if ngx.info.dlss_needs_updated_driver:
        print(
            "DLSS-SR requires a newer driver: "
            f"{ngx.info.dlss_minimum_driver_version_major}."
            f"{ngx.info.dlss_minimum_driver_version_minor}"
        )
    if ngx.info.frame_generation_needs_updated_driver:
        print(
            "DLSS-FG requires a newer driver: "
            f"{ngx.info.frame_generation_minimum_driver_version_major}."
            f"{ngx.info.frame_generation_minimum_driver_version_minor}"
        )


def main() -> None:
    args = parse_args()
    device_type = getattr(spy.DeviceType, args.device_type)
    quality = getattr(fngx.QualityMode, args.quality)
    device = create_ngx_ready_device(device_type)
    ngx = fngx.NGX.get(device)
    support = get_enabled_support(
        ngx,
        enable_sr=not args.disable_sr,
        enable_fg=not args.disable_fg,
        require_supported=args.require_supported,
    )
    print_support(ngx, support)

    scene = f2.Scene.create(device, args.scene_path)

    viewer = DlssViewer(
        device,
        support,
        quality,
        samples_per_frame=args.samples_per_frame,
    )

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
    print(f"DLSS view: {viewer.view_name}")

    while editor.update():
        if editor.needs_render:
            image = viewer.render(scene)
            editor.present(image)


if __name__ == "__main__":
    main()
