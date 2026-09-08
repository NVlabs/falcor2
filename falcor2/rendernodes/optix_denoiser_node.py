# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OptiX denoiser render node for floating-point RGBA images."""

from __future__ import annotations

from typing import Any, ClassVar, Mapping

import slangpy as spy

import falcor2 as f2
from falcor2.reflection import reflected, reflected_property
from falcor2.rendergraph import Container, ContainerSpec, RenderNode


_FLOAT_RGBA_FORMATS = frozenset((spy.Format.rgba16_float, spy.Format.rgba32_float))
_OPTIX_FORMAT_INFO = {
    spy.Format.rgba16_float: (f2.OptixPixelFormat.half4, 8),
    spy.Format.rgba32_float: (f2.OptixPixelFormat.float4, 16),
}


@reflected
class OptixDenoiserNode(RenderNode):
    """Denoise linear HDR color while preserving its render-container contract."""

    GUIDE_NAMES: ClassVar[tuple[str, ...]] = ("diffuse_albedo", "normals")

    def __init__(self, device: spy.Device) -> None:
        super().__init__()
        self._device = device
        self._copy_func: Any | None = None
        self._denoiser: f2.OptixDenoiser | None = None
        self._denoiser_key: tuple[tuple[int, int], bool, bool] | None = None
        self._input: spy.Tensor | None = None
        self._output: spy.Tensor | None = None
        self._albedo: spy.Tensor | None = None
        self._normal: spy.Tensor | None = None
        self._result: Any = None
        self._blend_factor = 0.0

    @classmethod
    def create(cls, device: spy.Device) -> "OptixDenoiserNode":
        return cls(device)

    @property
    def is_supported(self) -> bool:
        """Whether the device can execute CUDA or CUDA-interoperable OptiX work."""
        return bool(
            self._device.desc.type == spy.DeviceType.cuda or self._device.supports_cuda_interop
        )

    @reflected_property(value_range=(0.0, 1.0), ui_label="Blend factor")
    def blend_factor(self) -> float:
        """Blend between fully denoised (zero) and unmodified input (one)."""
        return self._blend_factor

    @blend_factor.setter
    def blend_factor(self, value: float) -> None:
        blend_factor = float(value)
        if not 0.0 <= blend_factor <= 1.0:
            raise ValueError("blend_factor must be between 0 and 1.")
        self._blend_factor = blend_factor

    def _validate_input(self, value: Any, label: str) -> ContainerSpec:
        if isinstance(value, (spy.Tensor, spy.Texture)) and value.device != self._device:
            raise ValueError(f"{label} must use the node's device.")
        spec = ContainerSpec.from_container(value)
        if not isinstance(spec.dims, tuple) or len(spec.dims) != 2:
            raise ValueError(f"{label} must be two-dimensional.")
        if spec.format not in _FLOAT_RGBA_FORMATS:
            raise ValueError(f"{label} must be floating-point RGBA.")
        return spec

    def _ensure_resources(
        self,
        dims: tuple[int, int],
        use_albedo: bool,
        use_normal: bool,
        stage_color: bool,
        stage_albedo: bool,
        stage_normal: bool,
    ) -> None:
        key = (dims, use_albedo, use_normal)
        if key != self._denoiser_key:
            height, width = dims
            desc = f2.OptixDenoiserDesc()
            desc.model_kind = f2.OptixModelKind.aov
            desc.alpha_mode = f2.OptixAlphaMode.copy
            desc.albedo_guide_layer = use_albedo
            desc.normal_guide_layer = use_normal
            desc.max_width = width
            desc.max_height = height
            self._denoiser = f2.OptixDenoiser(self._device, desc)
            self._denoiser_key = key

        self._input = self._ensure_staging_buffer(self._input, dims, stage_color)
        self._output = self._ensure_staging_buffer(self._output, dims, stage_color)
        self._albedo = self._ensure_staging_buffer(self._albedo, dims, stage_albedo)
        self._normal = self._ensure_staging_buffer(self._normal, dims, stage_normal)

    def _ensure_staging_buffer(
        self,
        current: spy.Tensor | None,
        dims: tuple[int, int],
        required: bool,
    ) -> spy.Tensor | None:
        if not required:
            return None
        if current is None or current.shape.as_tuple() != dims:
            return spy.Tensor.empty(self._device, shape=dims, dtype="float4")
        return current

    @staticmethod
    def _can_use_directly(value: Any) -> bool:
        """Whether ``value`` is a tightly packed buffer that OptiX can access directly."""
        return (
            isinstance(value, spy.Tensor)
            and value.offset == 0
            and value.is_contiguous()
            and Container.format(value) in _FLOAT_RGBA_FORMATS
        )

    def _copy(self, source: Any, output: Any, cmd: spy.CommandEncoder | None) -> None:
        if self._copy_func is None:
            self._copy_func = spy.Module.load_from_file(self._device, "falcor2/utils.slang")[
                "fill_color<float,4>"
            ]
        self._copy_func(color=Container.to_render_layout(source), output=output, _append_to=cmd)

    @staticmethod
    def _optix_image(value: spy.Tensor, width: int, height: int) -> f2.OptixImage2D:
        format_value = Container.format(value)
        optix_format, pixel_size = _OPTIX_FORMAT_INFO[format_value]
        image = f2.OptixImage2D()
        image.buffer = value.storage
        image.width = width
        image.height = height
        image.row_stride_in_bytes = width * pixel_size
        image.pixel_stride_in_bytes = pixel_size
        image.format = optix_format
        return image

    def _exec(
        self,
        input: Any,
        guides: Mapping[str, Any] | None = None,
        cmd: spy.CommandEncoder | None = None,
    ) -> Any:
        """Denoise ``input`` with available albedo/normal guides and preserve its contract."""
        if not self.is_supported:
            raise RuntimeError("OptiX denoising requires a CUDA or CUDA-interoperable device.")
        if cmd is not None and self._device.desc.type != spy.DeviceType.cuda:
            # Graphics-backend callbacks expose a native graphics command buffer, not a CUstream,
            # so OptiX cannot be ordered within a caller-owned D3D12/Vulkan command buffer.
            raise ValueError(
                "Recording OptiX denoising into a caller-owned command encoder requires a CUDA "
                "device."
            )

        spec = self._validate_input(input, "OptiX denoiser input")
        assert isinstance(spec.dims, tuple)
        guides = guides or {}
        albedo = guides.get("diffuse_albedo")
        normal = guides.get("normals")
        for name, value in (("diffuse albedo guide", albedo), ("normal guide", normal)):
            if value is None:
                continue
            guide_spec = self._validate_input(value, f"OptiX {name}")
            if guide_spec.dims != spec.dims:
                raise ValueError(f"OptiX {name} dimensions must match the denoiser input.")

        self._result = Container.create_temp(self._device, spec, current=self._result)
        direct_color = self._can_use_directly(input) and self._can_use_directly(self._result)
        direct_albedo = albedo is not None and self._can_use_directly(albedo)
        direct_normal = normal is not None and self._can_use_directly(normal)
        self._ensure_resources(
            spec.dims,
            albedo is not None,
            normal is not None,
            stage_color=not direct_color,
            stage_albedo=albedo is not None and not direct_albedo,
            stage_normal=normal is not None and not direct_normal,
        )
        assert self._denoiser is not None

        height, width = spec.dims
        if direct_color:
            assert isinstance(input, spy.Tensor)
            assert isinstance(self._result, spy.Tensor)
            optix_input = input
            optix_output = self._result
        else:
            assert self._input is not None
            assert self._output is not None
            self._copy(input, self._input, cmd)
            optix_input = self._input
            optix_output = self._output

        guide_layer = f2.OptixDenoiserGuideLayer()
        if albedo is not None:
            if direct_albedo:
                assert isinstance(albedo, spy.Tensor)
                optix_albedo = albedo
            else:
                assert self._albedo is not None
                self._copy(albedo, self._albedo, cmd)
                optix_albedo = self._albedo
            guide_layer.albedo = self._optix_image(optix_albedo, width, height)
        if normal is not None:
            if direct_normal:
                assert isinstance(normal, spy.Tensor)
                optix_normal = normal
            else:
                assert self._normal is not None
                self._copy(normal, self._normal, cmd)
                optix_normal = self._normal
            guide_layer.normal = self._optix_image(optix_normal, width, height)

        layer = f2.OptixDenoiserLayer()
        layer.input = self._optix_image(optix_input, width, height)
        layer.output = self._optix_image(optix_output, width, height)
        layer.type = f2.OptixDenoiserAOVType.beauty
        params = f2.OptixDenoiserParams()
        params.blend_factor = self.blend_factor
        if cmd is None:
            self._denoiser.denoise(params, guide_layer, [layer])
        else:
            self._denoiser.denoise(params, guide_layer, [layer], cmd)

        if not direct_color:
            assert self._output is not None
            self._copy(self._output, Container.to_render_layout(self._result), cmd)
        return self._result
