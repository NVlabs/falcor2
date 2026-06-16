# SPDX-License-Identifier: Apache-2.0
"""
Container utilities for the Python render-graph layer.

This module is the runtime-facing counterpart to ``ContainerSpec``. It knows how to:
- inspect supported container backends and recover their ``format + dims`` contract
- allocate new containers from either an existing container or a resolved spec
- adapt containers into the layout expected by runtime shader calls

The current render-oriented conventions are:
- textures use ``dims=(H, W)``
- render-like ``spy.Tensor`` values use ``dims`` plus a dtype implied by ``format``
- torch tensors expose a public ``CHW`` layout and are adapted to render layout on demand
"""

from typing import Any

import slangpy as spy

from . import container_torch
from .container_spec import AUTO, ContainerSpec
from .image_format import (
    HALF_FORMATS,
    SCALAR_TYPE_TO_FORMAT,
    SINT_FORMATS,
    UINT_FORMATS,
    channel_count,
    format_to_typename,
)

ClearValue = spy.float4 | spy.int4 | spy.uint4

UTILS_MODULE_PATH = "falcor2/utils.slang"
_UTILS_MODULES: dict[int, spy.Module] = {}


class Container:
    """Backend-agnostic helpers for render containers."""

    @staticmethod
    def format_and_dims(container: Any) -> tuple[spy.Format, tuple[int, ...]]:
        """Return the render-oriented ``(format, dims)`` pair for a supported container."""
        if isinstance(container, spy.Tensor):
            dims = container.shape.as_tuple()
            st = spy.TypeReflection.ScalarType.none
            if isinstance(container.dtype, spy.reflection.ScalarType):
                elems = 1
                st = container.dtype.slang_scalar_type
            if isinstance(container.dtype, spy.reflection.VectorType):
                elems = container.dtype.num_elements
                st = container.dtype.slang_scalar_type
            return (
                SCALAR_TYPE_TO_FORMAT[elems][st],
                dims,
            )
        if isinstance(container, spy.Texture):
            if container.type == spy.TextureType.texture_2d:
                return container.format, (int(container.height), int(container.width))
            raise ValueError(f"Unsupported texture type: {container.type!r}")

        if container_torch.is_torch_tensor(container):
            return container_torch.infer_format_and_dims(container)  # type: ignore

        raise TypeError(f"Unsupported container type: {type(container)!r}")

    @staticmethod
    def format(container: Any) -> spy.Format:
        """Return only the inferred render format for ``container``."""
        format_value, _ = Container.format_and_dims(container)
        return format_value

    @staticmethod
    def dims(container: Any) -> tuple[int, ...]:
        """Return only the inferred render dims for ``container``."""
        _, dims = Container.format_and_dims(container)
        return dims

    @staticmethod
    def empty_like(
        container: Any,
        override_dims: tuple[int, ...] | None = None,
        override_format: spy.Format | None = None,
        label: str = "",
    ) -> Any:
        """
        Allocate a new container matching the backend of ``container``.

        ``override_dims`` and ``override_format`` adjust the render contract while preserving
        the container backend.
        """
        if isinstance(container, spy.Tensor):
            return spy.Tensor.empty(
                container.device,
                shape=override_dims or container.shape,
                dtype=format_to_typename(override_format) if override_format else container.dtype,
            )
        if isinstance(container, spy.Texture):
            if container.type == spy.TextureType.texture_2d:
                format_value = override_format or container.format
                width = override_dims[1] if override_dims else container.width
                height = override_dims[0] if override_dims else container.height
                return container.device.create_texture(
                    type=spy.TextureType.texture_2d,
                    format=format_value,
                    width=width,
                    height=height,
                    mip_count=1,
                    usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
                    label=label,
                )
        if container_torch.is_torch_tensor(container):
            return container_torch.empty_like(
                container,  # type: ignore
                override_dims=override_dims,
                override_format=override_format,
            )  # type: ignore
        raise TypeError(f"Unsupported container type: {type(container)!r}")

    @staticmethod
    def empty(
        device: spy.Device,
        spec: ContainerSpec,
        label: str = "",
    ) -> Any:
        """Allocate a container from a fully resolved ``ContainerSpec``."""
        if spec.container_type is AUTO or spec.format is AUTO or spec.dims is AUTO:
            raise ValueError("ContainerSpec must be fully resolved before allocation.")
        assert isinstance(spec.dims, tuple)
        assert isinstance(spec.format, spy.Format)
        assert isinstance(spec.container_type, type)

        if spec.container_type is spy.Tensor:
            return spy.Tensor.empty(device, shape=spec.dims, dtype=format_to_typename(spec.format))

        if spec.container_type is spy.Texture:
            if len(spec.dims) != 2:
                raise ValueError("Texture containers currently require 2D dims.")
            return device.create_texture(
                type=spy.TextureType.texture_2d,
                format=spec.format,
                width=int(spec.dims[1]),
                height=int(spec.dims[0]),
                mip_count=1,
                usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
                label=label,
            )

        if container_torch.is_torch_tensor_type(spec.container_type):
            return container_torch.allocate(dims=spec.dims, format_value=spec.format)

        raise TypeError(f"Unsupported container type: {spec.container_type!r}")

    @staticmethod
    def is_like(a: Any, b: Any) -> bool:
        """Return ``True`` when two concrete containers share the same backend-specific storage shape."""
        if type(a) != type(b):
            return False
        if isinstance(a, spy.Tensor):
            return a.shape == b.shape and a.dtype == b.dtype and a.device == b.device
        if isinstance(a, spy.Texture):
            return (
                a.type == b.type
                and a.format == b.format
                and a.width == b.width
                and a.height == b.height
                and a.device == b.device
            )
        if container_torch.is_torch_tensor(a):
            return a.shape == b.shape and a.dtype == b.dtype and a.device == b.device
        return False

    @staticmethod
    def is_spec(container: Any, spec: ContainerSpec) -> bool:
        """Return ``True`` when ``container`` matches the resolved render contract in ``spec``."""
        if container is None:
            return False
        if type(container) != spec.container_type:
            return False
        format_value, dims = Container.format_and_dims(container)
        return format_value == spec.format and dims == spec.dims

    @staticmethod
    def to_render_layout(container: Any) -> Any:
        """
        Adapt ``container`` into the layout expected by runtime shader calls.

        ``spy.Texture`` and ``spy.Tensor`` already use the runtime layout and pass through
        unchanged. Torch tensors are permuted in place to the layout expected by SlangPy.
        """
        if isinstance(container, spy.Tensor):
            return container
        if isinstance(container, spy.Texture):
            return container
        if container_torch.is_torch_tensor(container):
            return container_torch.to_render_layout(container)  # type: ignore
        raise TypeError(f"Unsupported container type: {type(container)!r}")

    @staticmethod
    def clear(
        container: Any,
        clear_value: ClearValue | None = None,
        command_encoder: spy.CommandEncoder | None = None,
        device: spy.Device | None = None,
    ) -> None:
        """Clear a supported render container to ``clear_value``."""
        device = _resolve_device(container, command_encoder, device)
        if command_encoder is None:
            if device is None:
                raise ValueError("Clearing this container requires a device or command encoder.")
            command_encoder = device.create_command_encoder()
            Container.clear(
                container,
                clear_value=clear_value,
                command_encoder=command_encoder,
                device=device,
            )
            device.submit_command_buffer(command_encoder.finish())
            return

        if isinstance(container, spy.Texture):
            if _is_uint_format(Container.format(container)):
                assert clear_value is None or isinstance(clear_value, spy.uint4)
                command_encoder.clear_texture_uint(
                    container,
                    clear_value=clear_value or spy.uint4(0, 0, 0, 0),
                )
            elif _is_sint_format(Container.format(container)):
                assert clear_value is None or isinstance(clear_value, spy.int4)
                command_encoder.clear_texture_sint(
                    container,
                    clear_value=clear_value or spy.int4(0, 0, 0, 0),
                )
            else:
                assert clear_value is None or isinstance(clear_value, spy.float4)
                command_encoder.clear_texture_float(
                    container,
                    clear_value=clear_value or spy.float4(0.0, 0.0, 0.0, 0.0),
                )
            return

        if isinstance(container, spy.Tensor):
            if clear_value is None or (
                isinstance(clear_value, (spy.float4, spy.int4, spy.uint4))
                and clear_value.x == 0
                and clear_value.y == 0
                and clear_value.z == 0
                and clear_value.w == 0
            ):
                container.clear(command_encoder)
            else:
                _dispatch_fill(container, clear_value, command_encoder, device)
            return

        if container_torch.is_torch_tensor(container):
            _dispatch_fill(container, clear_value, command_encoder, device)
            return

        raise TypeError(f"Unsupported container type: {type(container)!r}")

    @staticmethod
    def create_temp(device: spy.Device, spec: ContainerSpec, current: Any = None) -> Any:
        """
        Allocate a temporary container matching ``spec``, reusing ``current`` when possible
        for none-torch containers.
        """
        if (
            current == None
            or container_torch.is_torch_tensor(current)
            or not Container.is_spec(current, spec)
        ):
            return Container.empty(device, spec)
        else:
            return current

    @staticmethod
    def create_temp_like(device: spy.Device, target: Any, current: Any = None) -> Any:
        """
        Allocate a temporary container matching the backend and render contract of ``target``, reusing
        ``current`` when possible for none-torch containers.
        """
        if (
            current == None
            or container_torch.is_torch_tensor(current)
            or not Container.is_like(current, target)
        ):
            return Container.empty_like(target)
        else:
            return current


def _is_uint_format(format_value: spy.Format) -> bool:
    return format_value in UINT_FORMATS


def _is_sint_format(format_value: spy.Format) -> bool:
    return format_value in SINT_FORMATS


def _is_half_format(format_value: spy.Format) -> bool:
    return format_value in HALF_FORMATS


def _resolve_device(
    container: Any,
    command_encoder: spy.CommandEncoder | None = None,
    device: spy.Device | None = None,
) -> spy.Device | None:
    if device is not None:
        return device
    if command_encoder is not None:
        return command_encoder.device
    if isinstance(container, (spy.Tensor, spy.Texture)):
        return container.device
    return None


def _utils_module(device: spy.Device) -> spy.Module:
    key = id(device)
    module = _UTILS_MODULES.get(key)
    if module is None:
        module = spy.Module.load_from_file(device, UTILS_MODULE_PATH)
        _UTILS_MODULES[key] = module
    return module


def _dispatch_fill(
    container: Any,
    clear_value: ClearValue | None,
    command_encoder: spy.CommandEncoder,
    device: spy.Device | None,
) -> None:
    if device is None:
        raise ValueError("Clearing this container requires a device or command encoder.")

    format_value = Container.format(container)
    channels = channel_count(format_value)
    target = Container.to_render_layout(container)
    module = _utils_module(device)

    if _is_uint_format(format_value):
        clear_value = clear_value or spy.uint4(0, 0, 0, 0)
        if not isinstance(clear_value, spy.uint4):
            raise TypeError("Unsigned integer clears require a uint4 clear value.")
        if channels == 1:
            cv = spy.uint1(clear_value.x)
        elif channels == 2:
            cv = spy.uint2(clear_value.x, clear_value.y)
        elif channels == 3:
            cv = spy.uint3(clear_value.x, clear_value.y, clear_value.z)
        elif channels == 4:
            cv = clear_value
        else:
            raise ValueError(f"Unsupported channel count: {channels}")
        module[f"fill_color<uint,{channels}>"](cv, output=target, _append_to=command_encoder)
    elif _is_sint_format(format_value):
        clear_value = clear_value or spy.int4(0, 0, 0, 0)
        if not isinstance(clear_value, spy.int4):
            raise TypeError("Signed integer clears require an int4 clear value.")
        if channels == 1:
            cv = spy.int1(clear_value.x)
        elif channels == 2:
            cv = spy.int2(clear_value.x, clear_value.y)
        elif channels == 3:
            cv = spy.int3(clear_value.x, clear_value.y, clear_value.z)
        elif channels == 4:
            cv = clear_value
        else:
            raise ValueError(f"Unsupported channel count: {channels}")
        module[f"fill_color<int,{channels}>"](cv, output=target, _append_to=command_encoder)
    elif _is_half_format(format_value):
        clear_value = clear_value or spy.float4(0.0, 0.0, 0.0, 0.0)
        if not isinstance(clear_value, spy.float4):
            raise TypeError("Floating-point clears require a float4 clear value.")
        if channels == 1:
            cv = spy.float16_t1(spy.float16_t(clear_value.x))
        elif channels == 2:
            cv = spy.float16_t2(spy.float16_t(clear_value.x), spy.float16_t(clear_value.y))
        elif channels == 4:
            cv = spy.float16_t4(
                spy.float16_t(clear_value.x),
                spy.float16_t(clear_value.y),
                spy.float16_t(clear_value.z),
                spy.float16_t(clear_value.w),
            )
        else:
            raise ValueError(f"Unsupported channel count: {channels}")
        module[f"fill_color<half,{channels}>"](cv, output=target, _append_to=command_encoder)
    else:
        clear_value = clear_value or spy.float4(0.0, 0.0, 0.0, 0.0)
        if not isinstance(clear_value, spy.float4):
            raise TypeError("Floating-point clears require a float4 clear value.")
        if channels == 1:
            cv = spy.float1(clear_value.x)
        elif channels == 2:
            cv = spy.float2(clear_value.x, clear_value.y)
        elif channels == 3:
            cv = spy.float3(clear_value.x, clear_value.y, clear_value.z)
        elif channels == 4:
            cv = clear_value
        else:
            raise ValueError(f"Unsupported channel count: {channels}")
        module[f"fill_color<float,{channels}>"](cv, output=target, _append_to=command_encoder)
