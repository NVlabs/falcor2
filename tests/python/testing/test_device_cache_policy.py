# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from falcor2.testing import helpers


def test_device_cache_policy_boundaries() -> None:
    current = Path("tests/python/render/test_scene.py")
    same_file = Path("tests/python/render/test_scene.py")
    next_file = Path("tests/python/render/test_material.py")

    assert not helpers._should_recycle_device_cache("session", current, None)
    assert not helpers._should_recycle_device_cache("file", current, same_file)
    assert helpers._should_recycle_device_cache("file", current, next_file)
    assert helpers._should_recycle_device_cache("file", current, None)
    assert helpers._should_recycle_device_cache("test", current, same_file)


def test_close_leaked_devices_reports_only_open_uncached_devices(monkeypatch: Any) -> None:
    class FakeDevice:
        def __init__(self, label: str | None, *, is_closed: bool = False):
            super().__init__()
            self.desc = SimpleNamespace(label=label)
            self.is_closed = is_closed
            self.close_count = 0

        def close(self) -> None:
            self.close_count += 1
            self.is_closed = True

    cached = FakeDevice("cached-by-identity")
    cached_by_label = FakeDevice("cached-by-label")
    already_closed = FakeDevice("uncached-closed", is_closed=True)
    leaked = FakeDevice("uncached-open")
    unlabeled = FakeDevice(None)
    devices = [cached, cached_by_label, already_closed, leaked, unlabeled]
    monkeypatch.setattr(helpers, "Device", SimpleNamespace(get_created_devices=lambda: devices))
    monkeypatch.setattr(helpers, "DEVICE_CACHE", {"cached": cached})

    labels = helpers.close_leaked_devices()

    assert labels == ["uncached-open", "<unlabeled>"]
    assert leaked.close_count == 1
    assert unlabeled.close_count == 1
    assert cached.close_count == 0
    assert cached_by_label.close_count == 0
    assert already_closed.close_count == 0


def test_get_device_enables_compilation_reports_for_telemetry(monkeypatch: Any) -> None:
    created: list[dict[str, Any]] = []

    def create_device(**kwargs: Any) -> SimpleNamespace:
        created.append(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(helpers, "Device", create_device)
    helpers.configure_test_device_options(compilation_reports_enabled=True)
    try:
        helpers.get_device(helpers.DeviceType.d3d12, use_cache=False)
    finally:
        helpers.configure_test_device_options()

    assert created[0]["enable_compilation_reports"] is True
