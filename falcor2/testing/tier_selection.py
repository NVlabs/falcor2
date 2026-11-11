# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Source-local nightly parameter selection, applied before xdist schedules tests."""

from collections.abc import Mapping, Sequence
from typing import Any

import pytest
from _pytest.compat import NOTSET
from slangpy import DeviceType


def _matches_parameters(
    item: pytest.Item,
    parameters: Mapping[str, Any],
    marker_name: str,
    conditions: Mapping[str, Any],
) -> bool:
    matches = True
    for name, values in conditions.items():
        if name not in parameters:
            raise pytest.UsageError(
                f"{item.nodeid}: {marker_name} refers to unknown parameter {name!r}"
            )
        if not isinstance(values, (list, tuple)):
            raise pytest.UsageError(
                f"{item.nodeid}: {marker_name} parameter {name!r} requires a list or tuple"
            )
        matches = matches and any(parameters[name] == value for value in values)
    return matches


def mark_nightly_parameters(items: Sequence[pytest.Item]) -> None:
    """Resolve conditional nightly marks before fixture setup and scheduling.

    Conditions within a marker are ANDed; separate markers are ORed.
    nightly_if uses explicit excluded values. nightly_if_not_vk defers every
    non-Vulkan backend, optionally limited by additional parameter conditions.
    """
    for item in items:
        parameters = getattr(getattr(item, "callspec", None), "params", {})
        for marker in item.iter_markers("nightly_if"):
            if marker.args or not marker.kwargs:
                raise pytest.UsageError(
                    f"{item.nodeid}: nightly_if requires named parameter value lists"
                )
            if _matches_parameters(item, parameters, marker.name, marker.kwargs):
                item.add_marker(pytest.mark.nightly)

        for marker in item.iter_markers("nightly_if_not_vk"):
            if marker.args:
                raise pytest.UsageError(
                    f"{item.nodeid}: nightly_if_not_vk accepts only keyword arguments"
                )
            conditions = dict(marker.kwargs)
            device_param = conditions.pop("device_param", "device_type")
            if not isinstance(device_param, str):
                raise pytest.UsageError(
                    f"{item.nodeid}: nightly_if_not_vk device_param requires a parameter name"
                )
            if device_param not in parameters:
                raise pytest.UsageError(
                    f"{item.nodeid}: nightly_if_not_vk refers to unknown parameter {device_param!r}"
                )
            device_type = parameters[device_param]
            matches = _matches_parameters(item, parameters, marker.name, conditions)
            # Pytest inserts NOTSET for an empty backend matrix and supplies its
            # own skip/xfail mark. Preserve that capability result without a device.
            if device_type is NOTSET:
                continue
            if not isinstance(device_type, DeviceType):
                raise pytest.UsageError(
                    f"{item.nodeid}: nightly_if_not_vk parameter {device_param!r} requires a DeviceType value"
                )
            if device_type != DeviceType.vulkan and matches:
                item.add_marker(pytest.mark.nightly)


def deselect_nightly_items(
    config: pytest.Config, items: list[pytest.Item], *, nightly: bool
) -> None:
    """Report deferred items as deselected, never as successful or skipped work."""
    mark_nightly_parameters(items)
    if nightly:
        return
    deferred = [item for item in items if item.get_closest_marker("nightly") is not None]
    if deferred:
        items[:] = [item for item in items if item.get_closest_marker("nightly") is None]
        config.hook.pytest_deselected(items=deferred)
