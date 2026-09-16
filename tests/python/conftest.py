# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import gc
import hashlib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
import re
import uuid
import argparse
from typing import Any

import pytest
import slangpy as spy
import falcor2 as f2
import falcor2.testing.helpers as helpers
from falcor2.testing.image_test_plugin import ImageTestPlugin
from falcor2.testing.shader_test_plugin import SlangTestFile, clear_dispatcher_cache
from falcor2.testing.telemetry_plugin import (
    DEFAULT_TELEMETRY_OUTPUT_PATH,
    DEVICE_LEAKS_STASH_KEY,
    TelemetryPlugin,
)
from falcor2.utils.per_device_cache import PerDeviceCache
from tools import crashpad

DEVICE_CACHE_POLICIES = ("session", "file", "test")

CRASHPAD_KIND = "python"
CRASHPAD_SUPPORT = spy.crashpad.is_supported()


@dataclass(frozen=True)
class _HelmetSceneEntry:
    scene: f2.Scene
    generation: int


def pytest_addoption(parser: pytest.Parser):
    """Add command line options for testing."""
    parser.addoption(
        "--image-tests-generate",
        action="store_true",
        default=False,
        help="Regenerate image test data rather than comparing it",
    )
    parser.addoption(
        "--image-tests",
        action="store_true",
        default=False,
        help="Run image tests (defaults to true)",
    )
    parser.addoption(
        "--image-tests-only",
        action="store_true",
        default=False,
        help="Run only tests that use the image_test fixture",
    )
    parser.addoption(
        "--slow",
        action="store_true",
        default=False,
        help="Run slow tests (skipped by default)",
    )
    parser.addoption(
        "--all",
        action="store_true",
        default=False,
        help="Run all tests, including slow and image tests (equivalent to --slow --image-tests)",
    )
    parser.addoption(
        "--module-cache",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable the persistent module cache for test devices",
    )
    parser.addoption(
        "--shader-cache",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable the persistent shader cache for test devices",
    )
    parser.addoption(
        "--module-and-shader-cache-dir",
        type=Path,
        default=None,
        help="Root directory for persistent test caches",
    )
    parser.addoption(
        "--telemetry",
        action="store_true",
        default=False,
        help="Write per-test timing, memory, resource-leak, and shader-compilation telemetry",
    )
    parser.addoption(
        "--telemetry-output",
        type=Path,
        default=DEFAULT_TELEMETRY_OUTPUT_PATH,
        help="Path for the merged test telemetry JSON sidecar",
    )
    parser.addoption(
        "--device-cache-policy",
        choices=DEVICE_CACHE_POLICIES,
        default="file",
        help="Recycle worker GPU state at session, test-file, or test boundaries",
    )


def pytest_configure(config: pytest.Config):
    """Configure the image test plugin."""
    helpers.configure_test_device_options(
        module_cache_enabled=bool(config.getoption("--module-cache")),
        shader_cache_enabled=bool(config.getoption("--shader-cache")),
        compilation_reports_enabled=bool(config.getoption("--telemetry")),
        cache_dir=config.getoption("--module-and-shader-cache-dir"),
    )
    config.pluginmanager.register(ImageTestPlugin(config), "image_test_plugin")
    if config.getoption("--telemetry"):
        telemetry_output = config.getoption("--telemetry-output")
        assert isinstance(telemetry_output, Path)
        config.pluginmanager.register(
            TelemetryPlugin(config, telemetry_output),
            "telemetry_plugin",
        )
    if CRASHPAD_SUPPORT and not os.environ.get("PYTEST_XDIST_WORKER"):
        crashpad.setup(CRASHPAD_KIND)


@pytest.hookimpl(tryfirst=True)
def pytest_sessionstart(session: pytest.Session):
    # Pytest's stdout/stderr capture can invalidate the FILE handles used by
    # SGL's console logger while native code is still emitting diagnostics.
    spy.ConsoleLoggerOutput.IGNORE_PRINT_EXCEPTION = True

    if not CRASHPAD_SUPPORT:
        return

    print("Starting Crashpad handler for Python tests ...")
    try:
        spy.crashpad.start_handler(database=crashpad.database_dir(CRASHPAD_KIND))
    except (RuntimeError, OSError) as exc:
        print(f"Failed to start Crashpad handler ({exc})")


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session: pytest.Session, exitstatus: int):
    clear_dispatcher_cache()
    helpers.close_all_devices()


@pytest.hookimpl(hookwrapper=True, trylast=True)
def pytest_runtest_protocol(item: pytest.Item, nextitem: pytest.Item | None) -> Iterator[None]:
    try:
        yield
    finally:
        # pytest clears item.funcargs only after the teardown report is complete.
        # Recycle devices after the full protocol so fixtures no longer retain
        # device-owned objects while garbage collection runs.
        item.stash[DEVICE_LEAKS_STASH_KEY] = helpers.close_leaked_devices()
        policy = str(item.config.getoption("--device-cache-policy"))
        next_path = nextitem.path if nextitem is not None else None
        if helpers._should_recycle_device_cache(policy, item.path, next_path):
            _recycle_worker_gpu_state()


def _recycle_worker_gpu_state() -> None:
    """Release caches that retain device-owned state, then close all devices."""
    clear_dispatcher_cache()
    gc.collect()
    helpers.close_all_devices(reason="cache boundary")
    gc.collect()


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item: pytest.Item) -> None:
    if CRASHPAD_SUPPORT:
        crashpad.notify_current_test(CRASHPAD_KIND, item.nodeid)


@pytest.hookimpl(tryfirst=True)
def pytest_terminal_summary(terminalreporter: Any, exitstatus: int) -> None:
    if CRASHPAD_SUPPORT and os.environ.get("FALCOR_CRASHPAD_DEFER_REPORT") != "1":
        crashpad.report(CRASHPAD_KIND, terminalreporter.config.get_terminal_writer())


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]):
    """Handles skipping of tests based on image test flags and slow test flags"""
    gen_images = config.getoption("--image-tests-generate")
    test_images = config.getoption("--image-tests")
    test_images_only = config.getoption("--image-tests-only")
    slow = config.getoption("--slow")
    if config.getoption("--all"):
        slow = True
        test_images = True
        test_images_only = False
        gen_images = False

    for item in items:
        uses_image_test = "image_test" in getattr(item, "fixturenames", [])
        is_slow_test = "slow" in [mark.name for mark in item.iter_markers()]

        # Skip slow tests if --slow is not specified
        if is_slow_test and not slow:
            item.add_marker(pytest.mark.skip(reason="Slow test skipped (use --slow to run)"))

        if not uses_image_test:
            if gen_images or test_images_only:
                item.add_marker(pytest.mark.skip(reason="Skipping non-image test"))
        else:
            if not gen_images and not test_images:
                item.add_marker(pytest.mark.skip(reason="Skipping image test"))


@pytest.fixture
def image_test(request: pytest.FixtureRequest):
    """Fixture that provides ImageTest functionality to tests."""
    plugin = request.config.pluginmanager.get_plugin("image_test_plugin")
    assert plugin is not None, "ImageTestPlugin not found"
    return plugin.create_image_test(request)


def pytest_collect_file(parent: pytest.Collector, file_path: Path):
    if file_path.suffix == ".slang" and file_path.stem.startswith("test_"):
        return SlangTestFile.from_parent(parent, path=file_path)


@pytest.fixture
def device(device_type: spy.DeviceType) -> spy.Device:
    """Cached device for the given device_type."""
    return helpers.get_device(device_type)


@pytest.fixture
def empty_scene(device: spy.Device) -> f2.Scene:
    """A fresh empty scene, already updated."""
    scene = f2.Scene.create(device)
    scene.update()
    return scene


@pytest.fixture(scope="session")
def _helmet_scene_cache() -> Iterator[PerDeviceCache[_HelmetSceneEntry]]:
    cache = PerDeviceCache[_HelmetSceneEntry]()
    yield cache
    cache.clear()


@pytest.fixture
def helmet_scene(
    device: spy.Device,
    _helmet_scene_cache: PerDeviceCache[_HelmetSceneEntry],
) -> Iterator[f2.Scene]:
    """Cached Avocado scene, recreated if a test mutates it."""
    entry = _helmet_scene_cache.get_or_create(device, _load_helmet_scene_entry)

    yield entry.scene

    update_flags = entry.scene.update()
    if (
        update_flags != f2.SceneUpdateFlags.none
        or entry.scene.update_generation != entry.generation
    ):
        _helmet_scene_cache.discard(device)


@pytest.fixture
def workspace_tmp_path(request: pytest.FixtureRequest) -> Path:
    """Temporary workspace path with a readable test-derived name."""
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", request.node.nodeid).strip("_")
    slug = slug[-96:] or "test"
    digest = hashlib.sha256(request.node.nodeid.encode()).hexdigest()[:12]
    path = Path(".tmp") / "pytest" / f"{slug}-{digest}-{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def _load_helmet_scene(device: spy.Device) -> f2.Scene:
    scene = f2.Scene.load(device, "data/assets/kronos/Avocado/glTF-Binary/Avocado.glb")
    scene.update()
    scene.update()
    return scene


def _load_helmet_scene_entry(device: spy.Device) -> _HelmetSceneEntry:
    scene = _load_helmet_scene(device)
    return _HelmetSceneEntry(scene=scene, generation=scene.update_generation)
