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
from falcor2.testing.tier_selection import deselect_nightly_items
from falcor2.testing.shader_test_plugin import SlangTestFile, clear_dispatcher_cache
from falcor2.testing.telemetry_plugin import (
    DEFAULT_TELEMETRY_OUTPUT_PATH,
    DEVICE_LEAKS_STASH_KEY,
    TelemetryPlugin,
)
from falcor2.testing.benchmark import (  # noqa: F401
    add_benchmark_options,
    benchmark_image_test,
    BenchmarkPlugin,
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
        "--nightly",
        action="store_true",
        default=False,
        help="Run nightly and slow tests (nightly tests are deselected by default)",
    )
    parser.addoption(
        "--all",
        action="store_true",
        default=False,
        help=(
            "Run all tests, including slow, nightly, and image tests "
            "(equivalent to --slow --nightly --image-tests)"
        ),
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
    # The benchmark options belong to the plugin. They have to be declared here
    # because a plugin registered during pytest_configure is too late for option
    # parsing; the plugin instance itself is registered below.
    add_benchmark_options(parser)


def _benchmark_requested(config: pytest.Config) -> bool:
    """Return whether this pytest invocation asked for benchmark mode."""
    return any(
        bool(config.getoption(option, default=False))
        for option in (
            "--benchmark",
            "--benchmark-only",
            "--benchmark-submit",
            "--benchmark-export-submissions",
        )
    )


def pytest_configure(config: pytest.Config):
    """Configure the image test plugin."""
    if _benchmark_requested(config):
        config.option.benchmark = True

    # Enable compilation reports when benchmarking (needed for compile time metrics)
    compilation_reports_needed = bool(config.getoption("--telemetry")) or bool(
        config.getoption("--benchmark")
    )

    helpers.configure_test_device_options(
        module_cache_enabled=bool(config.getoption("--module-cache")),
        shader_cache_enabled=bool(config.getoption("--shader-cache")),
        compilation_reports_enabled=compilation_reports_needed,
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

    # Registered unconditionally: its hooks are inert unless --benchmark is on.
    # Keep the implication above in this conftest so test devices already have
    # the compilation reports benchmarks need before the plugin is registered.
    config.pluginmanager.register(BenchmarkPlugin(config), "falcor2_benchmark")

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


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]):
    """Select the nightly tier before applying the existing slow/image gates."""
    gen_images = config.getoption("--image-tests-generate")
    test_images = config.getoption("--image-tests")
    test_images_only = config.getoption("--image-tests-only")
    slow = config.getoption("--slow")
    nightly = config.getoption("--nightly")
    benchmark = config.getoption("--benchmark")
    benchmark_only = config.getoption("--benchmark-only")
    # The nightly tier is a superset of the slow tier.
    if nightly:
        slow = True
    if config.getoption("--all"):
        slow = True
        nightly = True
        test_images = True
        test_images_only = False
        gen_images = False

    deselect_nightly_items(config, items, nightly=nightly)

    for item in items:
        uses_image_test = "image_test" in getattr(item, "fixturenames", [])
        measures_only = benchmark_only and "benchmark_image_test" in getattr(
            item, "fixturenames", []
        )
        marker_names = {mark.name for mark in item.iter_markers()}
        is_slow_test = "slow" in marker_names
        is_benchmark_test = "benchmark" in marker_names

        # Skip slow tests if --slow is not specified
        if is_slow_test and not slow:
            item.add_marker(pytest.mark.skip(reason="Slow test skipped (use --slow to run)"))

        if is_benchmark_test and not benchmark:
            item.add_marker(
                pytest.mark.skip(reason="Benchmark test skipped (use --benchmark to run)")
            )
            continue

        if not uses_image_test:
            if gen_images or test_images_only:
                item.add_marker(pytest.mark.skip(reason="Skipping non-image test"))
        else:
            # Benchmark-only fixtures still depend on image_test, but do not
            # compare images. Ordinary image tests retain their usual gate.
            if not gen_images and not test_images and not measures_only:
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
