# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backend selection before fixture setup and xdist scheduling."""

from collections.abc import Sequence

import pytest
from slangpy import DeviceType


def parse_device_types(value: str) -> list[DeviceType]:
    """Parse an ordered list of backend names, ignoring duplicate entries."""
    names = [name.strip() for name in value.split(",")]
    available = ("d3d12", "vulkan", "cuda", "metal")
    for name in names:
        if name not in available:
            raise ValueError(
                f"Invalid device type {name!r}; expected a comma-separated list of "
                + ", ".join(available)
            )
    return list(dict.fromkeys(DeviceType[name] for name in names))


def deselect_device_items(
    config: pytest.Config,
    items: list[pytest.Item],
    *,
    device_type: DeviceType | Sequence[DeviceType] | None,
) -> None:
    """Filter DeviceType parameters (including indirect fixtures) and shader items."""
    if device_type is None:
        return
    device_types = (device_type,) if isinstance(device_type, DeviceType) else device_type
    selected = []
    deselected = []
    for item in items:
        parameters = getattr(getattr(item, "callspec", None), "params", {})
        backends = [value for value in parameters.values() if isinstance(value, DeviceType)]
        shader_backend = getattr(item, "device_type", None)
        if isinstance(shader_backend, DeviceType):
            backends.append(shader_backend)
        if any(backend not in device_types for backend in backends):
            deselected.append(item)
        else:
            selected.append(item)
    if deselected:
        items[:] = selected
        config.hook.pytest_deselected(items=deselected)
