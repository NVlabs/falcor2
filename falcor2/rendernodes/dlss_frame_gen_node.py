# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DLSS Frame Generation render node for backbuffer, depth, and motion inputs."""

from __future__ import annotations

from math import radians
from typing import Any, ClassVar

import slangpy as spy

import falcor2 as f2
import falcor2.ngx as fngx
from falcor2.rendergraph import ContainerSpec, RenderNode, TextureBridge, TextureOutput


class DLSSFrameGenNode(RenderNode):
    """Evaluate NVIDIA DLSS Frame Generation from backbuffer and guide textures."""

    GUIDE_NAMES: ClassVar[tuple[str, ...]] = ("hardware_depth", "motion_vectors")

    def __init__(self, device: spy.Device) -> None:
        """Create a DLSS-G node for ``device`` without allocating runtime resources."""
        super().__init__()
        self._device = device
        self.output_spec = ContainerSpec.texture2d()
        self._ngx: fngx.NGX | None = None
        self._feature: fngx.DLSSGFeature | None = None
        self._feature_key: tuple[int, int, int, int, spy.Format] | None = None
        self._previous_clip_from_world: spy.float4x4 | None = None
        self._reset = True
        self._texture_bridge = TextureBridge(device)

    @classmethod
    def create(cls, device: spy.Device) -> "DLSSFrameGenNode":
        """Create a DLSS-G node for ``device``."""
        return cls(device)

    def reset(self) -> None:
        """Reset DLSS-G history on the next evaluate call."""
        self._reset = True
        self._previous_clip_from_world = None

    def _get_ngx(self) -> fngx.NGX:
        """Return the lazily created NGX context."""
        if self._ngx is None:
            self._ngx = fngx.NGX.get(self._device)
        return self._ngx

    def _get_output(self, backbuffer: spy.Texture) -> TextureOutput:
        """Create the texture target used by NGX and final output conversion."""
        return self._texture_bridge.output_texture(
            "output",
            self.output_spec,
            default_spec=ContainerSpec.texture2d(
                backbuffer.format,
                (backbuffer.height, backbuffer.width),
            ),
            required_channels=4,
            allowed_formats={spy.Format.rgba16_float, spy.Format.rgba32_float},
            error_prefix="DLSSG output",
        )

    def _make_feature_key(
        self,
        backbuffer: spy.Texture,
        depth: spy.Texture,
    ) -> tuple[int, int, int, int, spy.Format]:
        """Return the dimensions and format that define a DLSS-G feature."""
        return (
            backbuffer.width,
            backbuffer.height,
            depth.width,
            depth.height,
            backbuffer.format,
        )

    def _ensure_feature(self, backbuffer: spy.Texture, depth: spy.Texture) -> None:
        """Create or recreate the DLSS-G feature if the required key changed."""
        key = self._make_feature_key(backbuffer, depth)
        if key == self._feature_key:
            return

        desc = fngx.DLSSGFeatureDesc()
        desc.width = backbuffer.width
        desc.height = backbuffer.height
        desc.render_width = depth.width
        desc.render_height = depth.height
        desc.backbuffer_format = backbuffer.format

        self._feature = self._get_ngx().create_dlss_g_feature(desc)
        self._feature_key = key
        self._reset = True
        self._previous_clip_from_world = None

    def _input_texture(
        self,
        slot: str,
        name: str,
        value: Any,
        expected_dims: tuple[int, int] | None = None,
        cmd: spy.CommandEncoder | None = None,
    ) -> spy.Texture:
        """Validate one texture input used by DLSS-G."""
        return self._texture_bridge.input_texture(
            slot,
            value,
            required_dims=expected_dims,
            error_prefix=f"DLSSG {name}",
            cmd=cmd,
        )

    def _validate_inputs(
        self,
        backbuffer: Any,
        guides: dict[str, Any],
        camera: f2.Camera,
        output: spy.Texture,
        cmd: spy.CommandEncoder | None,
    ) -> tuple[spy.Texture, spy.Texture, spy.Texture]:
        """Validate DLSS-G-specific texture, guide, and camera constraints."""
        backbuffer_texture = self._input_texture("backbuffer", "backbuffer", backbuffer, cmd=cmd)
        if camera is None:
            raise ValueError("DLSSG camera must not be None.")
        self._input_texture(
            "output",
            "output_interpolated_frame",
            output,
            (backbuffer_texture.height, backbuffer_texture.width),
            cmd=cmd,
        )

        for name in self.GUIDE_NAMES:
            if guides.get(name) is None:
                raise ValueError(f"Missing DLSSG guide '{name}'.")

        depth = self._input_texture(
            "guide:hardware_depth",
            "guide 'hardware_depth'",
            guides["hardware_depth"],
            cmd=cmd,
        )
        motion_vectors = self._input_texture(
            "guide:motion_vectors",
            "guide 'motion_vectors'",
            guides["motion_vectors"],
            (depth.height, depth.width),
            cmd=cmd,
        )
        return backbuffer_texture, depth, motion_vectors

    def _make_camera_desc(
        self,
        camera: f2.Camera,
        render_width: int,
        render_height: int,
        jitter_offset: spy.float2,
    ) -> fngx.DLSSGCameraDesc:
        """Build the DLSS-G camera descriptor from Falcor camera state."""
        camera_desc = fngx.DLSSGCameraDesc()
        view_from_world = camera.calc_view_from_world()
        clip_from_view = camera.calc_clip_from_view(render_width, render_height)
        clip_from_world = spy.math.mul(clip_from_view, view_from_world)
        previous_clip_from_world = (
            self._previous_clip_from_world
            if self._previous_clip_from_world is not None
            else clip_from_world
        )

        camera_desc.camera_view_to_clip = clip_from_view
        camera_desc.clip_to_camera_view = spy.math.inverse(clip_from_view)
        camera_desc.clip_to_lens_clip = spy.float4x4.identity()
        camera_desc.clip_to_prev_clip = spy.math.mul(
            previous_clip_from_world,
            spy.math.inverse(clip_from_world),
        )
        camera_desc.prev_clip_to_clip = spy.math.mul(
            clip_from_world,
            spy.math.inverse(previous_clip_from_world),
        )
        camera_desc.jitter_offset = jitter_offset
        camera_desc.motion_vector_scale = spy.float2(
            1.0 / float(render_width),
            1.0 / float(render_height),
        )
        camera_desc.camera_pinhole_offset = spy.float2(0.0, 0.0)

        uniforms = camera.calc_uniforms(render_width, render_height)
        camera_desc.camera_pos = uniforms.position
        camera_desc.camera_up = spy.math.normalize(uniforms.image_v)
        camera_desc.camera_right = spy.math.normalize(uniforms.image_u)
        camera_desc.camera_fwd = spy.math.normalize(uniforms.image_w)
        camera_desc.camera_near = 0.1
        camera_desc.camera_far = 1000.0
        camera_desc.camera_fov = radians(float(camera.fov_y))
        camera_desc.camera_aspect_ratio = float(render_width) / float(render_height)
        camera_desc.color_buffers_hdr = True
        camera_desc.depth_inverted = False
        camera_desc.camera_motion_included = True
        camera_desc.reset = self._reset
        camera_desc.multi_frame_count = 1
        camera_desc.multi_frame_index = 1
        self._previous_clip_from_world = clip_from_world
        return camera_desc

    def forward(
        self,
        backbuffer: Any,
        guides: dict[str, Any],
        camera: f2.Camera,
        jitter_offset: spy.float2 = spy.float2(0.0, 0.0),
        cmd: spy.CommandEncoder | None = None,
    ) -> Any:
        """Evaluate DLSS-G and return the requested final output container."""
        backbuffer_texture = self._input_texture("backbuffer", "backbuffer", backbuffer, cmd=cmd)
        output = self._get_output(backbuffer_texture)
        backbuffer_texture, depth, motion_vectors = self._validate_inputs(
            backbuffer_texture,
            guides,
            camera,
            output.texture,
            cmd,
        )
        self._ensure_feature(backbuffer_texture, depth)

        desc = fngx.DLSSGEvaluateDesc()
        desc.backbuffer = backbuffer_texture
        desc.depth = depth
        desc.motion_vectors = motion_vectors
        desc.output_interpolated_frame = output.texture
        desc.camera = self._make_camera_desc(
            camera,
            depth.width,
            depth.height,
            jitter_offset,
        )

        assert self._feature is not None
        if cmd is not None:
            self._feature.evaluate(cmd, desc)
        else:
            self._feature.evaluate(desc)

        self._reset = False
        return output.finish(cmd)
