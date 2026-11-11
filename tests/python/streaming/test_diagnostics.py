# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import NamedTuple

import pytest
from falcor2.streaming.session import _EncoderDiagnostics


class _Sample(NamedTuple):
    byte_size: int
    gpu_prepare_ms: float
    encode_wall_ms: float
    submit_to_output_ms: float


def test_measured_rate_window_idle_and_session_reset() -> None:
    diagnostics = _EncoderDiagnostics(10.0)
    assert diagnostics.snapshot(10.0)["encoded_bitrate"] == 0
    for timestamp in [10.1, 10.4]:
        diagnostics.add(
            _Sample(
                byte_size=1000, gpu_prepare_ms=0.5, encode_wall_ms=2.0, submit_to_output_ms=5.0
            ),
            timestamp,
        )
    stats = diagnostics.snapshot(10.5)
    assert stats["encoded_bitrate"] == 32000
    assert stats["encoded_fps"] == 4
    assert stats["encoded_bytes"] == 2000
    assert stats["encode_wall_ms"] == 2
    assert diagnostics.snapshot(11.2)["encoded_bitrate"] == 8000
    idle = diagnostics.snapshot(11.5)
    assert idle["encoded_bitrate"] == idle["encoded_fps"] == idle["encode_wall_ms"] == 0
    assert idle["encoded_bytes"] == 2000
    assert _EncoderDiagnostics(12).snapshot(12)["encoded_bytes"] == 0


def test_timings_percentile_and_bounded_history() -> None:
    diagnostics = _EncoderDiagnostics(0)
    for value in range(1, 101):
        diagnostics.add(
            _Sample(
                byte_size=value,
                gpu_prepare_ms=value / 2,
                encode_wall_ms=value,
                submit_to_output_ms=value * 2,
            ),
            value / 200,
        )
    stats = diagnostics.snapshot(1)
    assert stats["encode_wall_ms"] == pytest.approx(50.5)
    assert stats["encode_wall_ms_p95"] == 95
    for _ in range(1000):
        diagnostics.add(
            _Sample(byte_size=1, gpu_prepare_ms=1, encode_wall_ms=1, submit_to_output_ms=2),
            1,
        )
    assert diagnostics.snapshot(1)["encoder_window_frames"] == 512
