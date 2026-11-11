# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from os import PathLike
from pathlib import Path
import sys
from typing import TYPE_CHECKING, Any, Union, overload

import numpy as np
import slangpy as spy

import falcor2 as f2
from falcor2._package_paths import falcor_slang_include_paths

if TYPE_CHECKING:
    from slangpy import DeviceDescParam


def get_slang_include_paths() -> list[str]:
    """Returns include paths needed to compile pathtracer shaders."""
    return [str(path) for path in (*falcor_slang_include_paths(), spy.SHADER_PATH)]


def get_module_and_shader_cache_root() -> Path:
    """Return a user-writable root for the separate module and shader caches."""
    if os.name == "nt":
        cache_home = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    elif sys.platform == "darwin":
        cache_home = Path.home() / "Library" / "Caches"
    else:
        cache_home = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return cache_home / "falcor2" / "app"


@overload
def create_device(
    device_type: spy.DeviceType = spy.DeviceType.automatic,
    enable_debug_layers: bool = False,
    enable_cuda_interop: bool = False,
    enable_module_cache: bool = False,
    enable_shader_cache: bool = False,
) -> spy.Device: ...


@overload
def create_device(desc: "DeviceDescParam") -> spy.Device: ...


def create_device(*args: Any, **kwargs: Any) -> spy.Device:
    """Creates a device with Falcor2's Slang include paths and fast floating-point mode.

    Accepts legacy arguments or a DeviceDesc/dict, which is copied before use.
    Dictionary cache paths default when absent; explicit paths and None are preserved.
    An unset or default floating-point mode becomes fast; explicit precise is preserved.
    """
    if "desc" in kwargs or (args and isinstance(args[0], (spy.DeviceDesc, dict))):
        return _create_device_from_desc(*args, **kwargs)
    return _create_device_legacy(*args, **kwargs)


def _prepare_device_desc(desc: "DeviceDescParam") -> spy.DeviceDesc:
    if not isinstance(desc, (spy.DeviceDesc, dict)):
        raise TypeError("desc must be a DeviceDesc or a dictionary")
    if isinstance(desc, dict):
        cache_root = get_module_and_shader_cache_root()
        desc = dict(desc)
        if "module_cache_path" not in desc:
            desc["module_cache_path"] = cache_root / "module"
        elif desc["module_cache_path"] is None:
            # DeviceDesc rejects None; omission leaves the cache disabled.
            del desc["module_cache_path"]

        if "shader_cache_path" not in desc:
            desc["shader_cache_path"] = cache_root / "shader"
        elif desc["shader_cache_path"] is None:
            del desc["shader_cache_path"]
    desc = spy.DeviceDesc(desc)
    desc.compiler_options.include_paths = get_slang_include_paths() + list(
        desc.compiler_options.include_paths
    )
    if desc.compiler_options.floating_point_mode == spy.SlangFloatingPointMode.default:
        desc.compiler_options.floating_point_mode = spy.SlangFloatingPointMode.fast
    return desc


def _create_device_from_desc(desc: "DeviceDescParam") -> spy.Device:
    return spy.Device(_prepare_device_desc(desc))


def _create_device_legacy(
    device_type: spy.DeviceType = spy.DeviceType.automatic,
    enable_debug_layers: bool = False,
    enable_cuda_interop: bool = False,
    enable_module_cache: bool = False,
    enable_shader_cache: bool = False,
) -> spy.Device:
    cache_root = get_module_and_shader_cache_root()
    desc = spy.DeviceDesc(
        {
            "type": device_type,
            "enable_debug_layers": enable_debug_layers,
            "enable_cuda_interop": enable_cuda_interop,
        }
    )
    if enable_module_cache:
        desc.module_cache_path = cache_root / "module"
    if enable_shader_cache:
        desc.shader_cache_path = cache_root / "shader"
    return _create_device_from_desc(desc)


@overload
def create_torch_device(
    device_type: spy.DeviceType = spy.DeviceType.automatic,
    enable_debug_layers: bool = False,
    *,
    torch_device: Any = None,
) -> spy.Device: ...


@overload
def create_torch_device(desc: "DeviceDescParam", torch_device: Any = None) -> spy.Device: ...


def create_torch_device(*args: Any, **kwargs: Any) -> spy.Device:
    """Creates a PyTorch-compatible device with Falcor2's Slang defaults.

    Accepts legacy arguments or a DeviceDesc/dict, which is copied before use.
    Dictionary cache paths default when absent; explicit paths and None are preserved.
    An unset or default floating-point mode becomes fast; explicit precise is preserved.
    """
    if "desc" in kwargs or (args and isinstance(args[0], (spy.DeviceDesc, dict))):
        return _create_torch_device_from_desc(*args, **kwargs)
    return _create_torch_device_legacy(*args, **kwargs)


def _create_torch_device_from_desc(desc: "DeviceDescParam", torch_device: Any = None) -> spy.Device:
    desc = _prepare_device_desc(desc)
    if desc.type == spy.DeviceType.automatic:
        desc.type = spy.DeviceType.cuda
    return spy.create_torch_device(desc, torch_device=torch_device)


def _create_torch_device_legacy(
    device_type: spy.DeviceType = spy.DeviceType.automatic,
    enable_debug_layers: bool = False,
    *,
    torch_device: Any = None,
) -> spy.Device:
    return _create_torch_device_from_desc(
        spy.DeviceDesc({"type": device_type, "enable_debug_layers": enable_debug_layers}),
        torch_device,
    )


def load_scene(
    device: spy.Device,
    path: Union[str, PathLike[str]],
    recompute_normals: bool = False,
) -> f2.Scene:
    """Load a scene from a file using the C++ scene system."""
    import_options = f2.ImportOptions()
    import_options.recompute_normals = recompute_normals
    return f2.Scene.load(device, path, import_options)


def save_image(tensor: spy.Tensor, path: Union[str, PathLike[str]]) -> None:
    """Save a tensor as an image file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = tensor.to_numpy()

    if len(data.shape) == 2:
        data = np.stack([data] * 3, axis=-1)
        pixel_format = spy.Bitmap.PixelFormat.rgb
    elif len(data.shape) == 3:
        if data.shape[2] == 1:
            data = np.stack([data[:, :, 0]] * 3, axis=-1)
            pixel_format = spy.Bitmap.PixelFormat.rgb
        elif data.shape[2] == 2:
            data = np.concatenate(
                [data, np.zeros((data.shape[0], data.shape[1], 1), dtype=data.dtype)], axis=-1
            )
            pixel_format = spy.Bitmap.PixelFormat.rgb
        elif data.shape[2] == 3:
            pixel_format = spy.Bitmap.PixelFormat.rgb
        elif data.shape[2] == 4:
            pixel_format = spy.Bitmap.PixelFormat.rgba
        else:
            raise ValueError(f"Unsupported number of channels: {data.shape[2]}")
    else:
        raise ValueError(f"Unsupported tensor shape: {data.shape}")

    bitmap = spy.Bitmap(data=data, pixel_format=pixel_format, srgb_gamma=False)
    if path.suffix not in (".exr", ".hdr"):
        bitmap = bitmap.convert(component_type=spy.Bitmap.ComponentType.uint8, srgb_gamma=True)
    bitmap.write(path)
