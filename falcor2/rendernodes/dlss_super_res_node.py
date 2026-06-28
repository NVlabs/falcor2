# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DLSS Super Resolution render node for color, depth, and motion inputs."""

from __future__ import annotations

from typing import Any, ClassVar

import slangpy as spy

import falcor2.ngx as fngx
from falcor2.rendergraph import ContainerSpec, RenderNode, TextureBridge, TextureOutput


class DLSSSuperResNode(RenderNode):
    """Evaluate NVIDIA DLSS Super Resolution from color and guide textures."""

    GUIDE_NAMES: ClassVar[tuple[str, ...]] = ("hardware_depth", "motion_vectors")

    def __init__(self, device: spy.Device) -> None:
        """Create a DLSS-SR node for ``device`` without allocating runtime resources."""
        super().__init__()
        self._device = device
        self.quality = fngx.QualityMode.quality
        self.output_spec = ContainerSpec.texture2d()
        self._ngx: fngx.NGX | None = None
        self._feature: fngx.DLSSSRFeature | None = None
        self._feature_key: tuple[int, int, int, int, fngx.QualityMode] | None = None
        self._optimal_specs_cache: dict[
            tuple[int, int, fngx.QualityMode],
            tuple[ContainerSpec, dict[str, ContainerSpec]],
        ] = {}
        self._reset = True
        self._texture_bridge = TextureBridge(device)

    @classmethod
    def create(cls, device: spy.Device) -> "DLSSSuperResNode":
        """Create a DLSS-SR node for ``device``."""
        return cls(device)

    def reset(self) -> None:
        """Reset DLSS-SR history on the next evaluate call."""
        self._reset = True

    def _get_ngx(self) -> fngx.NGX:
        """Return the lazily created NGX context."""
        if self._ngx is None:
            self._ngx = fngx.NGX.get(self._device)
        return self._ngx

    def _default_color_format(self) -> spy.Format:
        """Return the default color target format for the current backend."""
        if self._device.desc.type == spy.DeviceType.cuda:
            return spy.Format.rgba32_float
        return spy.Format.rgba16_float

    def _motion_vector_format(self) -> spy.Format:
        """Return the motion-vector format for generated guide specs."""
        if self._device.desc.type == spy.DeviceType.cuda:
            return spy.Format.rg32_float
        return spy.Format.rg16_float

    def get_optimal_settings(
        self,
        target_width: int,
        target_height: int,
    ) -> fngx.DLSSOptimalSettings:
        """Query NGX for the render size recommended for the current quality mode."""
        return self._get_ngx().get_dlss_optimal_settings(
            target_width,
            target_height,
            self.quality,
        )

    def get_optimal_specs(
        self,
        target_width: int,
        target_height: int,
    ) -> tuple[ContainerSpec, dict[str, ContainerSpec]]:
        """Return the path-tracer color and guide specs required by DLSS-SR."""
        if target_width <= 0 or target_height <= 0:
            raise ValueError("DLSSSR target dimensions must be positive.")

        key = (target_width, target_height, self.quality)
        cached = self._optimal_specs_cache.get(key)
        if cached is not None:
            return cached

        settings = self.get_optimal_settings(target_width, target_height)
        render_dims = (settings.render_height, settings.render_width)
        color_spec = ContainerSpec.texture2d(self._default_color_format(), render_dims)
        guide_specs = {
            "hardware_depth": ContainerSpec.texture2d(spy.Format.r32_float, render_dims),
            "motion_vectors": ContainerSpec.texture2d(self._motion_vector_format(), render_dims),
        }
        result = (color_spec, guide_specs)
        self._optimal_specs_cache[key] = result
        return result

    def _get_output(self, target_width: int, target_height: int) -> TextureOutput:
        """Create the texture target used by NGX and final output conversion."""
        return self._texture_bridge.output_texture(
            "output",
            self.output_spec,
            default_spec=ContainerSpec.texture2d(
                self._default_color_format(),
                (target_height, target_width),
            ),
            required_channels=4,
            allowed_formats={spy.Format.rgba16_float, spy.Format.rgba32_float},
            error_prefix="DLSSSR output",
        )

    def _make_feature_key(
        self,
        color: spy.Texture,
        output: spy.Texture,
    ) -> tuple[int, int, int, int, fngx.QualityMode]:
        """Return the dimensions and quality that define a DLSS-SR feature."""
        return (
            color.width,
            color.height,
            output.width,
            output.height,
            self.quality,
        )

    def _ensure_feature(self, color: spy.Texture, output: spy.Texture) -> None:
        """Create or recreate the DLSS-SR feature if the required key changed."""
        key = self._make_feature_key(color, output)
        if key == self._feature_key:
            return

        desc = fngx.DLSSSRFeatureDesc()
        desc.quality = self.quality
        desc.render_width = color.width
        desc.render_height = color.height
        desc.target_width = output.width
        desc.target_height = output.height
        desc.is_hdr = True
        desc.depth_inverted = False
        desc.motion_vectors_jittered = False
        desc.auto_exposure = True

        self._feature = self._get_ngx().create_dlss_sr_feature(desc)
        self._feature_key = key
        self._reset = True

    def _input_texture(
        self,
        slot: str,
        name: str,
        value: Any,
        expected_dims: tuple[int, int] | None = None,
        cmd: spy.CommandEncoder | None = None,
    ) -> spy.Texture:
        """Validate one texture input used by DLSS-SR."""
        return self._texture_bridge.input_texture(
            slot,
            value,
            required_dims=expected_dims,
            error_prefix=f"DLSSSR {name}",
            cmd=cmd,
        )

    def _validate_inputs(
        self,
        color: Any,
        guides: dict[str, Any],
        output: spy.Texture,
        target_width: int,
        target_height: int,
        cmd: spy.CommandEncoder | None,
    ) -> tuple[spy.Texture, dict[str, spy.Texture]]:
        """Validate DLSS-SR-specific texture and guide constraints."""
        color_texture = self._input_texture("color", "color", color, cmd=cmd)
        color_dims = (color_texture.height, color_texture.width)

        if output.width != target_width or output.height != target_height:
            raise ValueError("DLSSSR output dimensions must match the target size.")

        guide_textures: dict[str, spy.Texture] = {}
        for name in self.GUIDE_NAMES:
            guide = guides.get(name)
            if guide is None:
                raise ValueError(f"Missing DLSSSR guide '{name}'.")
            guide_textures[name] = self._input_texture(
                f"guide:{name}",
                f"guide '{name}'",
                guide,
                color_dims,
                cmd=cmd,
            )

        return color_texture, guide_textures

    def forward(
        self,
        color: Any,
        guides: dict[str, Any],
        target_width: int,
        target_height: int,
        jitter_offset: spy.float2 = spy.float2(0.0, 0.0),
        cmd: spy.CommandEncoder | None = None,
    ) -> Any:
        """Evaluate DLSS-SR and return the requested final output container."""
        output = self._get_output(target_width, target_height)
        color_texture, guide_textures = self._validate_inputs(
            color,
            guides,
            output.texture,
            target_width,
            target_height,
            cmd,
        )
        self._ensure_feature(color_texture, output.texture)

        desc = fngx.DLSSSREvaluateDesc()
        desc.color = color_texture
        desc.depth = guide_textures["hardware_depth"]
        desc.motion_vectors = guide_textures["motion_vectors"]
        desc.output = output.texture
        desc.jitter_offset_x = float(jitter_offset.x)
        desc.jitter_offset_y = float(jitter_offset.y)
        desc.reset = self._reset
        desc.motion_vector_scale_x = 1.0
        desc.motion_vector_scale_y = 1.0
        desc.pre_exposure = 1.0
        desc.exposure_scale = 1.0

        assert self._feature is not None
        if cmd is not None:
            self._feature.evaluate(cmd, desc)
        else:
            self._feature.evaluate(desc)

        self._reset = False
        return output.finish(cmd)
