# SPDX-License-Identifier: Apache-2.0

import os
import hashlib
from collections.abc import Iterator
from pathlib import Path
import re
import uuid
from typing import Any

import pytest
import slangpy as spy
import falcor2 as f2
import falcor2.testing.helpers as helpers
from falcor2.testing.image_test_plugin import ImageTestPlugin
from falcor2.testing.shader_test_plugin import SlangTestFile, clear_dispatcher_cache
from tools import crashpad

_HELMET_SCENE_CACHE: dict[int, tuple[f2.Scene, int]] = {}

CRASHPAD_KIND = "python"
CRASHPAD_SUPPORT = spy.crashpad.is_supported()


def pytest_addoption(parser: pytest.Parser):
    """Add command line options for image testing."""
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


def pytest_configure(config: pytest.Config):
    """Configure the image test plugin."""
    config.pluginmanager.register(ImageTestPlugin(config), "image_test_plugin")
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


@pytest.hookimpl(trylast=True)
def pytest_runtest_teardown(item: pytest.Item, nextitem: pytest.Item | None):
    helpers.close_leaked_devices()


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
    scene = f2.Scene(device)
    scene.update()
    return scene


@pytest.fixture
def helmet_scene(device: spy.Device) -> Iterator[f2.Scene]:
    """Cached DamagedHelmet scene, recreated if a test mutates it."""
    key = id(device)
    cached = _HELMET_SCENE_CACHE.get(key)
    if cached is None:
        scene = _load_helmet_scene(device)
        generation = scene.update_generation
        _HELMET_SCENE_CACHE[key] = (scene, generation)
    else:
        scene, generation = cached

    yield scene

    update_flags = scene.update()
    if update_flags != f2.SceneUpdateFlags.none or scene.update_generation != generation:
        _HELMET_SCENE_CACHE.pop(key, None)


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
    scene = f2.Scene(device, "data/assets/kronos/DamagedHelmet/glTF/DamagedHelmet.gltf")
    scene.update()
    scene.update()
    return scene
