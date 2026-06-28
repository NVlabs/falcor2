# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Torch-specific helpers for the Python graph layer.

This module defines the torch-facing render contract:
- public render tensors use ``CHW`` layout
- render dims are interpreted as ``(H, W)``
- runtime shader calls receive a temporary render layout view when needed
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import slangpy as spy

from .image_format import channel_count, format_to_typename

if TYPE_CHECKING:
    import torch


_TORCH_MODULE: Any | None = None
_TORCH_ATTEMPTED = False
_FORMAT_TO_TORCH: dict[spy.Format, Any] | None = None
_TORCH_TO_FORMAT: tuple[dict[Any, spy.Format], ...] | None = None


def _try_get_torch() -> Any | None:
    """Import torch once on demand and return the module when available."""
    global _TORCH_MODULE, _TORCH_ATTEMPTED
    if _TORCH_ATTEMPTED:
        return _TORCH_MODULE

    _TORCH_ATTEMPTED = True
    try:
        import torch
    except ImportError:
        _TORCH_MODULE = None
    else:
        _TORCH_MODULE = torch

    return _TORCH_MODULE


def _get_torch() -> Any:
    """Return the torch module, raising when torch support is unavailable."""
    torch = _try_get_torch()
    if torch is None:
        raise RuntimeError("PyTorch is not available.")
    return torch


def _get_format_to_torch() -> dict[spy.Format, Any]:
    """Build and cache the format-to-torch dtype mapping."""
    global _FORMAT_TO_TORCH
    if _FORMAT_TO_TORCH is None:
        torch = _get_torch()
        _FORMAT_TO_TORCH = {
            spy.Format.r16_float: torch.float16,
            spy.Format.rg16_float: torch.float16,
            spy.Format.rgba16_float: torch.float16,
            spy.Format.r32_float: torch.float32,
            spy.Format.rg32_float: torch.float32,
            spy.Format.rgb32_float: torch.float32,
            spy.Format.rgba32_float: torch.float32,
            spy.Format.r32_sint: torch.int32,
            spy.Format.rg32_sint: torch.int32,
            spy.Format.rgb32_sint: torch.int32,
            spy.Format.rgba32_sint: torch.int32,
        }
    return _FORMAT_TO_TORCH


def _get_torch_to_format() -> tuple[dict[Any, spy.Format], ...]:
    """Build and cache the torch dtype/channel-count to format mapping."""
    global _TORCH_TO_FORMAT
    if _TORCH_TO_FORMAT is None:
        torch = _get_torch()
        _TORCH_TO_FORMAT = (
            {},
            {
                torch.float16: spy.Format.r16_float,
                torch.float32: spy.Format.r32_float,
                torch.int32: spy.Format.r32_sint,
            },
            {
                torch.float16: spy.Format.rg16_float,
                torch.float32: spy.Format.rg32_float,
                torch.int32: spy.Format.rg32_sint,
            },
            {
                torch.float32: spy.Format.rgb32_float,
                torch.int32: spy.Format.rgb32_sint,
            },
            {
                torch.float16: spy.Format.rgba16_float,
                torch.float32: spy.Format.rgba32_float,
                torch.int32: spy.Format.rgba32_sint,
            },
        )
    return _TORCH_TO_FORMAT


def tensor_type() -> type[Any]:
    """Return the runtime torch tensor type."""
    return _get_torch().Tensor


def is_torch_tensor_type(value: type[Any] | None) -> bool:
    """Return ``True`` when ``value`` is the runtime ``torch.Tensor`` type."""
    if value is None or value is spy.Tensor or value is spy.Texture:
        return False
    torch = _try_get_torch()
    return torch is not None and value is torch.Tensor


def is_torch_tensor(value: Any) -> bool:
    """Return ``True`` if ``value`` is a torch tensor in the current environment."""
    if value is None or isinstance(value, (spy.Tensor, spy.Texture)):
        return False
    torch = _try_get_torch()
    if torch is not None:
        return isinstance(value, torch.Tensor)
    return False


def require_torch_tensor(value: Any) -> Any:
    """Validate that ``value`` is a torch tensor and return it."""
    if not is_torch_tensor(value):
        raise TypeError("Expected a torch.Tensor")
    return value


def format_to_torch(format_value: spy.Format) -> "torch.dtype":
    """Map a Falcor/Slang image format to the matching torch dtype."""
    return _get_format_to_torch()[format_value]


def torch_to_format(dtype: "torch.dtype", channels: int) -> spy.Format:
    """Map a torch dtype and channel count to the matching Falcor/Slang image format."""
    return _get_torch_to_format()[channels][dtype]


def to_render_layout(value: Any) -> Any:
    """
    Return the layout expected by runtime shader calls.

    Torch tensors are adapted from public ``CHW`` layout to render layout. Other supported
    container types pass through unchanged.
    """
    if isinstance(value, (spy.Tensor, spy.Texture)):
        return value

    tensor = require_torch_tensor(value)
    if tensor.ndim == 3:
        return tensor.permute(1, 2, 0)
    return tensor


def allocate(dims: tuple[int, ...], format_value: spy.Format) -> Any:
    """Allocate a torch render tensor in public ``CHW`` layout."""
    torch = _get_torch()

    if len(dims) != 2:
        raise ValueError("Torch render tensors currently require 2D dims.")

    channels = channel_count(format_value)
    dtype = format_to_torch(format_value)
    return torch.empty((channels, dims[0], dims[1]), dtype=dtype, device="cuda").contiguous()


def torch_to_slangpy(device: spy.Device, value: Any, dtype: Any = "float4") -> spy.Tensor:
    """Wrap a torch tensor as a ``spy.Tensor`` using render layout."""
    if isinstance(dtype, spy.Format):
        dtype = format_to_typename(dtype)
    rl = to_render_layout(value)
    rl = rl.contiguous()
    return spy.Tensor.from_torch(device, rl, dtype)


def infer_format_and_dims(tensor: "torch.Tensor") -> tuple[spy.Format, tuple[int, ...]]:
    """Infer the render ``(format, dims)`` pair from a torch tensor in ``CHW`` layout."""
    if tensor.ndim != 3:
        raise ValueError("Torch render tensors must have shape (C, H, W).")

    channels, height, width = (int(dim) for dim in tensor.shape)
    format_value = torch_to_format(tensor.dtype, channels)
    return format_value, (height, width)


def empty_like(
    container: "torch.Tensor",
    override_dims: tuple[int, ...] | None = None,
    override_format: spy.Format | None = None,
) -> Any:
    """Allocate a new torch tensor matching the render contract of ``container``."""
    torch = _get_torch()
    if override_dims is None:
        _, dims = infer_format_and_dims(container)
    else:
        dims = override_dims

    if override_format is None:
        format_value, _ = infer_format_and_dims(container)
    else:
        format_value = override_format

    channels = channel_count(format_value)
    dtype = format_to_torch(format_value)
    return torch.empty(
        (channels, dims[0], dims[1]), dtype=dtype, device=container.device
    ).contiguous()


def tensors_matches_spec(
    tensor: "torch.Tensor", dims: tuple[int, ...], format_value: spy.Format
) -> bool:
    """Return ``True`` when a torch tensor matches a render contract in public ``CHW`` layout."""
    _get_torch()

    if len(dims) != 2:
        return False

    channels = channel_count(format_value)
    dtype = format_to_torch(format_value)
    height, width = dims
    return (
        tuple(int(dim) for dim in tensor.shape) == (channels, height, width)
        and tensor.dtype == dtype
    )
