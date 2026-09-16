# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import slangpy as spy

from falcor2.utils.per_device_cache import PerDeviceCache


class _FakeDevice:
    def __init__(self) -> None:
        super().__init__()
        self._callbacks: list[Callable[[_FakeDevice], None]] = []

    def register_device_close_callback(self, callback: Callable[[_FakeDevice], None]) -> int:
        self._callbacks.append(callback)
        return len(self._callbacks)

    def close(self) -> None:
        for callback in self._callbacks:
            callback(self)
        self._callbacks.clear()


def _as_device(device: _FakeDevice) -> spy.Device:
    return cast(spy.Device, device)


def test_per_device_cache_creates_one_value_per_device() -> None:
    cache = PerDeviceCache[object]()
    device = _as_device(_FakeDevice())
    created: list[object] = []

    def factory(_: spy.Device) -> object:
        value = object()
        created.append(value)
        return value

    first = cache.get_or_create(device, factory)
    second = cache.get_or_create(device, factory)

    assert first is second
    assert created == [first]
    assert device in cache


def test_per_device_cache_evicts_value_when_device_closes() -> None:
    cache = PerDeviceCache[object]()
    fake_device = _FakeDevice()
    device = _as_device(fake_device)

    cache.get_or_create(device, lambda _: object())
    fake_device.close()

    assert device not in cache
    assert len(cache) == 0


def test_per_device_cache_supports_explicit_discard_and_clear() -> None:
    cache = PerDeviceCache[object]()
    first = _as_device(_FakeDevice())
    second = _as_device(_FakeDevice())
    cache.get_or_create(first, lambda _: object())
    cache.get_or_create(second, lambda _: object())

    cache.discard(first)

    assert first not in cache
    assert second in cache

    cache.clear()

    assert len(cache) == 0
