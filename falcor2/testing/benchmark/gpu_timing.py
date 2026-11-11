# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU intervals for the benchmark's single-sample, non-denoising pipeline."""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
import math
from typing import TYPE_CHECKING, Any, Iterator, NamedTuple, Sequence, TypedDict

if TYPE_CHECKING:
    import slangpy as spy
    import falcor2 as f2
    from falcor2.rendergraph import RenderNode
    from falcor2.rendernodes.pathtracer_pipeline_node import PathTracerPipeline
    from slangpy.testing.benchmark import BenchViewMetric


class GPUPhase(NamedTuple):
    """A named, sequential pipeline node measured by a start/end query pair."""

    id: str
    label: str
    node_attribute: str


DEFAULT_PHASES = (
    GPUPhase("path_tracing", "Path tracing", "path_tracer"),
    GPUPhase("accumulation", "Accumulation", "accumulator"),
    GPUPhase("tone_mapping", "Tone mapping", "tonemapper"),
)
OTHER_LABEL = "Other frame work and timing gaps"


class FrameTotal(TypedDict):
    total: float


class FrameTiming(FrameTotal, total=False):
    """Frame duration and optional disjoint phase durations, all in milliseconds."""

    components: dict[str, float]


def _validate_phases(phases: Sequence[GPUPhase]) -> None:
    """Reject ambiguous identities before allocating queries or exporting components."""
    ids = [phase.id for phase in phases]
    nodes = [phase.node_attribute for phase in phases]
    if len(set(ids)) != len(ids) or len(set(nodes)) != len(nodes) or "other" in ids:
        raise ValueError("GPU phases need unique IDs and nodes; 'other' is reserved")
    if any(not phase.id or not phase.label or not phase.node_attribute for phase in phases):
        raise ValueError("GPU phases need an ID, label and node attribute")


def decode_timestamps(
    ticks: list[float], frequency: float, phases: Sequence[GPUPhase] = DEFAULT_PHASES
) -> FrameTiming:
    """Convert ordered timestamps to disjoint intervals in milliseconds.

    The outer timestamps enclose every stage. For example, indices 1 and 2
    enclose the first phase, while the first and last enclose the whole frame. Gaps between
    named stages remain visible as ``other`` rather than being attributed to a
    shader. Reject invalid ordering instead of hiding over-accounting.
    """
    _validate_phases(phases)
    expected = 2 * (len(phases) + 1)
    if len(ticks) != expected or not math.isfinite(frequency) or frequency <= 0:
        raise ValueError("Invalid GPU timestamp count or frequency")
    if not all(math.isfinite(tick) for tick in ticks) or any(
        b < a for a, b in zip(ticks, ticks[1:])
    ):
        raise ValueError("GPU timestamps are non-finite or out of order")
    scale = 1000 / frequency
    result: FrameTiming = {"total": (ticks[-1] - ticks[0]) * scale}
    if phases:
        components = {
            phase.id: (ticks[2 * index + 2] - ticks[2 * index + 1]) * scale
            for index, phase in enumerate(phases)
        }
        # Sum the gaps directly to avoid cancellation yielding a negative residual.
        components["other"] = (
            sum(ticks[i + 1] - ticks[i] for i in range(0, expected - 1, 2)) * scale
        )
        result["components"] = components
    return result


@contextmanager
def _time_node(
    node: RenderNode,
    encoder: spy.CommandEncoder,
    query_pool: spy.QueryPool,
    query_index: int,
    calls: list[int],
) -> Iterator[None]:
    """Temporarily bracket one node, preserving any existing instance override."""
    original = node.forward
    had_override = "forward" in vars(node)
    override = vars(node).get("forward")

    def timed(*args: Any, **kwargs: Any) -> Any:
        # RenderNode.forward deliberately accepts heterogeneous node arguments/results.
        if kwargs.get("cmd") is not encoder:
            raise ValueError("Timed node must use the frame's command encoder")
        if calls[0]:
            raise ValueError("Each benchmark phase must execute exactly once per frame")
        calls[0] += 1
        encoder.write_timestamp(query_pool, query_index)
        value = original(*args, **kwargs)
        encoder.write_timestamp(query_pool, query_index + 1)
        return value

    node.forward = timed
    try:
        yield
    finally:
        if had_override:
            node.forward = override
        else:
            del node.forward


def measure_frame(
    device: spy.Device,
    pipeline: PathTracerPipeline,
    scene: f2.Scene,
    camera: f2.Camera,
    *,
    phases: Sequence[GPUPhase] = DEFAULT_PHASES,
) -> FrameTiming:
    """Time one frame on one encoder, resolving all queries after its completion.

    No wait is inserted between stages. Each named node must execute once, in the
    specified order on this encoder. Pass an empty sequence for whole-frame timing.
    Missing, repeated or overlapping phases fail rather than mislabeling results.
    """
    import slangpy as spy

    _validate_phases(phases)
    count = 2 * (len(phases) + 1)
    query_pool = device.create_query_pool(type=spy.QueryType.timestamp, count=count)
    encoder = device.create_command_encoder()
    calls = [[0] for _ in phases]
    with ExitStack() as stack:
        if phases:
            for index, phase in enumerate(phases):
                node = getattr(pipeline, phase.node_attribute)
                stack.enter_context(
                    _time_node(node, encoder, query_pool, 1 + 2 * index, calls[index])
                )
        encoder.write_timestamp(query_pool, 0)
        pipeline(scene, camera, cmd=encoder)
        encoder.write_timestamp(query_pool, count - 1)
        if phases and any(call[0] != 1 for call in calls):
            raise ValueError("A required benchmark GPU phase did not execute")
    device.submit_command_buffer(encoder.finish())
    device.wait()
    return decode_timestamps(
        list(query_pool.get_results(0, count)), float(device.info.timestamp_frequency), phases
    )


def build_gpu_metric(
    frames: Sequence[FrameTiming], phases: Sequence[GPUPhase] = DEFAULT_PHASES
) -> BenchViewMetric:
    """Preserve aligned frame/component samples for BenchView's mean breakdown.

    Each component sample refers to the same frame as the parent sample at that
    index. Their means add to the parent mean; independent medians need not add.
    """
    if not frames:
        raise ValueError("No GPU frames measured")
    _validate_phases(phases)
    metric: BenchViewMetric = {
        "id": "gpu_time_diagnostic",
        "name": "Instrumented frame GPU time",
        "unit": "ms",
        "direction": "lower",
        "distribution": {"samples": [frame["total"] for frame in frames]},
    }
    if all("components" in frame for frame in frames):
        names = {phase.id: phase.label for phase in phases}
        names["other"] = OTHER_LABEL
        if any(set(frame["components"]) != set(names) for frame in frames):
            raise ValueError("GPU frame components do not match the phase definitions")
        metric["breakdown"] = {
            "mode": "complete",
            "components": [
                {
                    "id": name,
                    "name": label,
                    "distribution": {"samples": [frame["components"][name] for frame in frames]},
                }
                for name, label in names.items()
            ],
        }
    elif any("components" in frame for frame in frames):
        raise ValueError("Cannot mix instrumented and uninstrumented GPU frames")
    return metric
