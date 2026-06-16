# SPDX-License-Identifier: Apache-2.0
"""Adapt render containers to texture-only APIs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import slangpy as spy

from falcor2.rendergraph.container import Container
from falcor2.rendergraph.container_spec import AUTO, ContainerSpec
from falcor2.rendergraph.image_format import HALF_FORMATS, SINT_FORMATS, UINT_FORMATS, channel_count


@dataclass
class TextureOutput:
    """A texture target for an operation that can be finished as the requested container."""

    texture: spy.Texture
    spec: ContainerSpec
    _bridge: "TextureBridge"
    _slot: str

    def finish(self, cmd: spy.CommandEncoder | None = None) -> Any:
        """Return the requested output container, copying from ``texture`` if needed."""
        return self._bridge.finish_output(self._slot, self.texture, self.spec, cmd=cmd)


class TextureBridge:
    """Scratch-resource bridge for APIs that consume or produce textures only."""

    def __init__(self, device: spy.Device) -> None:
        self._device = device
        self._input_textures: dict[str, spy.Texture] = {}
        self._output_textures: dict[str, spy.Texture] = {}
        self._outputs: dict[str, Any] = {}
        self._utils_module: spy.Module | None = None

    def input_texture(
        self,
        slot: str,
        value: Any,
        *,
        required_dims: tuple[int, int] | None = None,
        allowed_formats: set[spy.Format] | None = None,
        error_prefix: str = "Texture input",
        cmd: spy.CommandEncoder | None = None,
    ) -> spy.Texture:
        """Return ``value`` as a texture, copying supported non-texture containers."""
        self._validate_device(value, error_prefix)
        spec = ContainerSpec.from_container(value)
        self._validate_texture_contract(
            spec,
            required_dims=required_dims,
            required_channels=None,
            allowed_formats=allowed_formats,
            error_prefix=error_prefix,
        )

        if isinstance(value, spy.Texture):
            return value

        texture_spec = ContainerSpec.texture2d(spec.format, spec.dims)
        texture = Container.create_temp(
            self._device,
            texture_spec,
            current=self._input_textures.get(slot),
        )
        self._input_textures[slot] = texture
        self._copy(value, texture, spec.format, cmd=cmd)
        return texture

    def output_texture(
        self,
        slot: str,
        requested: ContainerSpec,
        *,
        default_spec: ContainerSpec,
        required_channels: int | None = None,
        allowed_formats: set[spy.Format] | None = None,
        error_prefix: str = "Texture output",
    ) -> TextureOutput:
        """Create a texture target for an output resolved against ``default_spec``."""
        self._validate_default_spec(default_spec, error_prefix)
        spec = requested.resolved(default_spec)
        assert isinstance(default_spec.dims, tuple)
        self._validate_texture_contract(
            spec,
            required_dims=default_spec.dims,
            required_channels=required_channels,
            allowed_formats=allowed_formats,
            error_prefix=error_prefix,
        )

        texture_spec = ContainerSpec.texture2d(spec.format, spec.dims)
        if spec.container_type is spy.Texture:
            texture = Container.create_temp(
                self._device,
                texture_spec,
                current=self._outputs.get(slot),
            )
            self._outputs[slot] = texture
        else:
            texture = Container.create_temp(
                self._device,
                texture_spec,
                current=self._output_textures.get(slot),
            )
            self._output_textures[slot] = texture

        return TextureOutput(texture=texture, spec=spec, _bridge=self, _slot=slot)

    def finish_output(
        self,
        slot: str,
        texture: spy.Texture,
        spec: ContainerSpec,
        *,
        cmd: spy.CommandEncoder | None = None,
    ) -> Any:
        """Return the final output container, copying from ``texture`` if needed."""
        if spec.container_type is spy.Texture:
            return texture

        output = Container.create_temp(self._device, spec, current=self._outputs.get(slot))
        self._outputs[slot] = output
        self._copy(texture, Container.to_render_layout(output), spec.format, cmd=cmd)
        return output

    def _get_utils_module(self) -> spy.Module:
        if self._utils_module is None:
            self._utils_module = spy.Module.load_from_file(self._device, "falcor2/utils.slang")
        return self._utils_module

    def _copy(
        self,
        source: Any,
        output: Any,
        format_value: spy.Format,
        *,
        cmd: spy.CommandEncoder | None,
    ) -> None:
        scalar_type = self._copy_scalar_type(format_value)
        kwargs = {
            "color": Container.to_render_layout(source),
            "output": output,
        }
        if cmd is not None:
            kwargs["_append_to"] = cmd
        self._get_utils_module()[f"fill_color<{scalar_type},{channel_count(format_value)}>"](
            **kwargs
        )

    def _copy_scalar_type(self, format_value: spy.Format) -> str:
        if format_value in UINT_FORMATS:
            return "uint"
        if format_value in SINT_FORMATS:
            return "int"
        if format_value in HALF_FORMATS:
            return "half"
        if format_value.name.endswith("_float"):
            return "float"
        raise ValueError(f"Texture bridge copy does not support format {format_value}.")

    def _validate_default_spec(self, default_spec: ContainerSpec, error_prefix: str) -> None:
        if default_spec.container_type is not spy.Texture:
            raise ValueError(f"{error_prefix} default_spec must describe a 2D texture.")
        if default_spec.format is spy.Format.undefined or default_spec.dims is AUTO:
            raise ValueError(f"{error_prefix} default_spec must be fully resolved.")
        assert isinstance(default_spec.dims, tuple)
        if len(default_spec.dims) != 2:
            raise ValueError(f"{error_prefix} default_spec dimensions must be two-dimensional.")

    def _validate_texture_contract(
        self,
        spec: ContainerSpec,
        *,
        required_dims: tuple[int, int] | None,
        required_channels: int | None,
        allowed_formats: set[spy.Format] | None,
        error_prefix: str,
    ) -> None:
        if spec.container_type is AUTO or spec.format is spy.Format.undefined or spec.dims is AUTO:
            raise ValueError(f"{error_prefix} spec must be fully resolved.")
        if not isinstance(spec.dims, tuple) or len(spec.dims) != 2:
            raise ValueError(f"{error_prefix} dimensions must be two-dimensional.")
        if required_dims is not None and spec.dims != required_dims:
            raise ValueError(f"{error_prefix} dimensions do not match.")
        if required_channels is not None and channel_count(spec.format) != required_channels:
            raise ValueError(f"{error_prefix} format must have {required_channels} channels.")
        if allowed_formats is not None and spec.format not in allowed_formats:
            allowed_names = ", ".join(
                format_value.name for format_value in sorted(allowed_formats, key=lambda f: f.name)
            )
            raise ValueError(f"{error_prefix} format must be one of: {allowed_names}.")

    def _validate_device(self, value: Any, error_prefix: str) -> None:
        if isinstance(value, (spy.Texture, spy.Tensor)) and value.device != self._device:
            raise ValueError(f"{error_prefix} must be on the bridge device.")
