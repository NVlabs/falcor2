# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark selection and measurement failures, without rendering scene assets."""

from collections.abc import Iterator
import json
from pathlib import Path
import runpy
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, sentinel

import pytest
import slangpy as spy

from falcor2.testing import helpers
from falcor2.testing.benchmark import fixtures
from falcor2.testing.benchmark import gpu_timing
from tests.python.conftest import pytest_collection_modifyitems


class SelectionItem:
    """Minimal collected item for exercising the real selection hook."""

    def __init__(self, fixtures: list[str], marks: list[Any]):
        self.fixturenames = fixtures
        self.marks = list(marks)

    def iter_markers(self, name: str | None = None) -> Iterator[Any]:
        return (mark for mark in self.marks if name is None or mark.name == name)

    def get_closest_marker(self, name: str) -> Any:
        return next(self.iter_markers(name), None)

    def add_marker(self, mark: Any) -> None:
        self.marks.append(mark)


def select(items: list[SelectionItem], **options: bool) -> None:
    """Apply selection with all unspecified command-line flags disabled."""
    config = Mock()
    config.getoption.side_effect = lambda name: options.get(name, False)
    pytest_collection_modifyitems(config, items)


@pytest.mark.parametrize("benchmark_only", [False, True])
@pytest.mark.parametrize("image_tests", [False, True])
def test_benchmark_only_bypasses_only_its_image_gate(
    benchmark_only: bool, image_tests: bool
) -> None:
    benchmark = SelectionItem(["benchmark_image_test", "image_test"], [pytest.mark.benchmark])
    image = SelectionItem(["image_test"], [])
    select(
        [benchmark, image],
        **{
            "--benchmark": True,
            "--benchmark-only": benchmark_only,
            "--image-tests": image_tests,
        },
    )
    assert (benchmark.get_closest_marker("skip") is None) == (benchmark_only or image_tests)
    assert (image.get_closest_marker("skip") is None) == image_tests


def test_benchmark_remains_opt_in() -> None:
    item = SelectionItem(["benchmark_image_test", "image_test"], [pytest.mark.benchmark])
    select([item], **{"--image-tests": True})
    assert item.get_closest_marker("skip") is not None


@pytest.mark.parametrize("nightly", [False, True])
def test_dpel_keeps_ordinary_nightly_image_coverage(nightly: bool) -> None:
    path = helpers.PROJECT_ROOT / "tests/python/benchmarks/internal/test_dpel_materials.py"
    if not path.exists():
        pytest.skip("DPEL workload is absent from the public source export")
    workload = runpy.run_path(str(path))["test_dpel_material"]
    item = SelectionItem(["benchmark_image_test", "image_test"], workload.pytestmark)
    items = [item]
    select(items, **{"--nightly": nightly, "--image-tests": True})
    assert bool(items) == nightly
    assert item.get_closest_marker("skip") is None


@pytest.mark.parametrize(
    "devices",
    [
        [spy.DeviceType.d3d12, spy.DeviceType.vulkan, spy.DeviceType.cuda],
        [spy.DeviceType.vulkan],
        [spy.DeviceType.d3d12],
    ],
)
def test_render_backends_have_references(
    monkeypatch: pytest.MonkeyPatch, devices: list[spy.DeviceType]
) -> None:
    monkeypatch.setattr(helpers, "DEFAULT_DEVICE_TYPES", devices)
    path = helpers.PROJECT_ROOT / "tests/python/benchmarks/test_render_benchmarks.py"
    namespace = runpy.run_path(str(path))
    workload = namespace["test_benchmark_pathtracer"]
    backend_mark = next(
        mark
        for mark in workload.pytestmark
        if mark.name == "parametrize" and mark.args[0] == "device_type"
    )
    selected = backend_mark.args[1]
    assert selected == [device for device in devices if device != spy.DeviceType.d3d12]
    references = json.loads(path.with_suffix(".images.json").read_text())
    for case in namespace["CASES"]:
        for device in selected:
            assert f"test_benchmark_pathtracer[{case}-{device}]" in references


def measure(frequency: float, timestamps: list[float]) -> float:
    """Exercise the real timing function with controlled query results."""
    device = Mock()
    device.info = SimpleNamespace(timestamp_frequency=frequency)
    device.create_query_pool.return_value.get_results.return_value = timestamps
    gpu_time, _, _, image = fixtures._measure_render_time(device, lambda cmd: "rendered")
    assert image == "rendered"
    device.submit_command_buffer.assert_called_once()
    device.wait.assert_called_once()
    return gpu_time


@pytest.mark.parametrize("frequency", [0, -1, float("nan"), float("inf")])
def test_invalid_frequency_fails_measurement(frequency: float) -> None:
    with pytest.raises(RuntimeError, match="timestamp frequency"):
        measure(frequency, [100, 200])


@pytest.mark.parametrize("timestamps", [[], [100]])
def test_missing_timestamps_fail_measurement(timestamps: list[float]) -> None:
    with pytest.raises(RuntimeError, match="Expected two GPU timestamps"):
        measure(1000, timestamps)


@pytest.mark.parametrize("timestamps", [[200, 100], [100, float("nan")], [100, float("inf")]])
def test_invalid_duration_fails_measurement(timestamps: list[float]) -> None:
    with pytest.raises(RuntimeError, match="timestamp duration"):
        measure(1000, timestamps)


def test_valid_timestamps_convert_to_milliseconds() -> None:
    assert measure(1_000_000, [1000, 2500]) == pytest.approx(1.5)
    assert measure(1_000_000, [1000, 1000]) == 0


def test_gpu_phase_samples_sum_to_their_own_frame_total() -> None:
    frames = [
        gpu_timing.decode_timestamps(ticks, 1_000_000)
        for ticks in ([0, 1, 5, 7, 10, 11, 13, 15], [0, 1, 9, 10, 11, 12, 13, 16])
    ]
    assert frames[0]["components"] == pytest.approx(
        {"path_tracing": 0.004, "accumulation": 0.003, "tone_mapping": 0.002, "other": 0.006}
    )
    metric = gpu_timing.build_gpu_metric(frames)
    assert metric["id"] == "gpu_time_diagnostic"
    assert metric["unit"] == "ms"
    for index, total in enumerate(metric["distribution"]["samples"]):
        assert sum(
            component["distribution"]["samples"][index]
            for component in metric["breakdown"]["components"]
        ) == pytest.approx(total)


@pytest.mark.parametrize(
    "ticks,frequency",
    [
        (list(range(8)), 0),
        (list(range(8)), float("nan")),
        (list(range(7)), 1000),
        ([0, 1, 5, 4, 6, 7, 8, 9], 1000),
        ([0, 1, 2, 3, 4, 5, 6, float("inf")], 1000),
    ],
)
def test_invalid_gpu_phase_timestamps_fail(ticks: list[float], frequency: float) -> None:
    with pytest.raises(ValueError):
        gpu_timing.decode_timestamps(ticks, frequency)


def test_gpu_phase_nodes_are_restored_after_failure() -> None:
    class Node:
        def forward(self, **kwargs: Any) -> int:
            return 7

    node = Node()
    encoder = Mock()
    with pytest.raises(RuntimeError, match="render failed"):
        with gpu_timing._time_node(node, encoder, Mock(), 1, [0]):
            assert node.forward(cmd=encoder) == 7
            with pytest.raises(ValueError, match="exactly once"):
                node.forward(cmd=encoder)
            raise RuntimeError("render failed")
    assert "forward" not in vars(node)
    assert node.forward() == 7


def test_nonbenchmark_texture_is_converted_before_image_comparison() -> None:
    texture = Mock(spec=spy.Texture)
    texture.to_numpy.return_value = sentinel.image_array
    render = Mock(return_value=texture)
    image_test = Mock()
    config = Mock()
    config.getoption.side_effect = lambda _name, default=False: default
    fixture = fixtures.BenchmarkImageTest(Mock(), image_test, config)

    fixture(Mock(), render, tolerance=0.01, additional_name="texture")

    render.assert_called_once_with(cmd=None)
    texture.to_numpy.assert_called_once_with()
    image_test.assert_called_once_with(
        sentinel.image_array,
        tolerance=0.01,
        additional_name="texture",
        debug_output_normalize=False,
        debug_output_stratify=False,
    )


@pytest.mark.parametrize("intensity_type", [float, spy.float3])
def test_environment_intensity_preserves_baseline_api_type(intensity_type: Any) -> None:
    """Keep equal RGB intensity on historical and current light APIs."""
    workload = runpy.run_path(
        str(helpers.PROJECT_ROOT / "tests/python/benchmarks/test_render_benchmarks.py")
    )

    class Light(dict):
        intensity = intensity_type(1.0)

    light = Light()
    scene = Mock()
    scene.create_entity.return_value.create_component.return_value = light
    workload["_add_env_map"](scene, Path("environment.exr"), 0.5)
    assert type(light.intensity) is intensity_type
    if intensity_type is float:
        assert light.intensity == 0.5
    else:
        assert list(light.intensity) == [0.5, 0.5, 0.5]
    assert light["env_map_path"] == "environment.exr"
