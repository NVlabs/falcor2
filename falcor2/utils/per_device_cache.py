# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cache values for the lifetime of the device that owns them."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar
import weakref

import slangpy as spy


T = TypeVar("T")


@dataclass(frozen=True)
class _Entry(Generic[T]):
    device: weakref.ReferenceType[spy.Device]
    value: T


class PerDeviceCache(Generic[T]):
    """Store at most one value per device and evict it when the device closes."""

    def __init__(self) -> None:
        super().__init__()
        self._entries: dict[int, _Entry[T]] = {}

    def get_or_create(self, device: spy.Device, factory: Callable[[spy.Device], T]) -> T:
        """Return the value for ``device``, creating it with ``factory`` if needed."""
        key = id(device)
        entry = self._entries.get(key)
        if entry is not None and entry.device() is device:
            return entry.value

        cache_ref = weakref.ref(self)

        def remove_on_close(closed_device: spy.Device) -> None:
            cache = cache_ref()
            if cache is not None:
                cache.discard(closed_device)

        value = factory(device)
        self._entries[key] = _Entry(device=weakref.ref(device), value=value)
        try:
            device.register_device_close_callback(remove_on_close)
        except Exception:
            self.discard(device)
            raise
        return value

    def discard(self, device: spy.Device) -> None:
        """Remove the value for ``device`` if it is present."""
        key = id(device)
        entry = self._entries.get(key)
        if entry is not None and entry.device() is device:
            self._entries.pop(key)

    def clear(self) -> None:
        """Remove every cached value."""
        self._entries.clear()

    def __contains__(self, device: spy.Device) -> bool:
        entry = self._entries.get(id(device))
        return entry is not None and entry.device() is device

    def __len__(self) -> int:
        return len(self._entries)
