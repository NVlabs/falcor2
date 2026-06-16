# SPDX-License-Identifier: Apache-2.0

from os import PathLike
from pathlib import Path
from typing import Union

import numpy as np
import slangpy as spy

import falcor2 as f2


def get_slang_include_paths() -> list[Union[Path, str]]:
    """Returns include paths needed to compile pathtracer shaders."""
    return [
        Path(__file__).parent.parent.parent / "slang",
        spy.SHADER_PATH,
    ]


def create_device(
    device_type: spy.DeviceType = spy.DeviceType.automatic,
    enable_debug_layers: bool = False,
) -> spy.Device:
    """Creates a device with correct Slang include paths configured."""
    return spy.create_device(
        type=device_type,
        include_paths=get_slang_include_paths(),
        enable_debug_layers=enable_debug_layers,
    )


def create_torch_device(
    device_type: spy.DeviceType = spy.DeviceType.automatic,
    enable_debug_layers: bool = False,
) -> spy.Device:
    if device_type == spy.DeviceType.automatic:
        device_type = spy.DeviceType.cuda
    return spy.create_torch_device(
        type=device_type,
        include_paths=get_slang_include_paths(),
        enable_debug_layers=enable_debug_layers,
    )


def load_scene(
    device: spy.Device,
    path: Union[str, PathLike[str]],
    recompute_normals: bool = False,
) -> f2.Scene:
    """Load a scene from a file using the C++ scene system."""
    return f2.Scene(device, path, recompute_normals)


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
