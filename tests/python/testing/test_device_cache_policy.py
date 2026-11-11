# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

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


@pytest.mark.parametrize("cache_enabled", [False, True])
@pytest.mark.parametrize(
    "floating_point_mode",
    [helpers.SlangFloatingPointMode.fast, helpers.SlangFloatingPointMode.precise],
)
def test_get_device_enables_compilation_reports_for_telemetry(
    monkeypatch: Any,
    tmp_path: Path,
    cache_enabled: bool,
    floating_point_mode: helpers.SlangFloatingPointMode,
) -> None:
    created: list[Any] = []

    def create_device(desc: Any) -> SimpleNamespace:
        created.append(desc)
        return SimpleNamespace()

    monkeypatch.setattr(helpers, "MODULE_CACHE_ENABLED", cache_enabled)
    monkeypatch.setattr(helpers, "SHADER_CACHE_ENABLED", cache_enabled)
    monkeypatch.setattr(helpers, "MODULE_AND_SHADER_CACHE_ROOT", tmp_path)
    monkeypatch.setattr(helpers, "COMPILATION_REPORTS_ENABLED", False)
    monkeypatch.delenv("PYTEST_XDIST_WORKER", raising=False)
    with monkeypatch.context() as patch:
        patch.setattr(helpers, "create_device", create_device)
        patch.setattr(helpers, "COMPILATION_REPORTS_ENABLED", True)
        helpers.get_device(
            helpers.DeviceType.d3d12,
            use_cache=False,
            floating_point_mode=floating_point_mode,
        )

    desc = created[0]
    assert isinstance(desc, dict)
    assert desc["enable_compilation_reports"] is True
    assert desc["compiler_options"].floating_point_mode == floating_point_mode
    assert helpers.MODULE_CACHE_ENABLED is cache_enabled
    assert helpers.SHADER_CACHE_ENABLED is cache_enabled
    assert helpers.MODULE_AND_SHADER_CACHE_ROOT == tmp_path
    assert helpers.COMPILATION_REPORTS_ENABLED is False
    assert desc["module_cache_path"] == (tmp_path / "python" / "module" if cache_enabled else None)
    assert desc["shader_cache_path"] == (tmp_path / "python" / "shader" if cache_enabled else None)
