# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from falcor2.streaming.session import _BitratePolicy


def _stats() -> dict[str, int]:
    return dict(
        receiver_reports=0,
        receiver_bitrate_reports=0,
        receiver_bitrate=100000,
        encoded_bitrate=60000,
        recoveries=0,
        queue_age_ms=0,
        loss_fraction=0,
    )


def test_sparse_output_does_not_reduce_quality_target() -> None:
    policy = _BitratePolicy(1000000, 8000000)
    stats = _stats()
    for index in range(1, 31):
        stats.update(receiver_reports=index, receiver_bitrate_reports=index)
        stats["receiver_bitrate"] = 100000 if index < 15 else 50000
        stats["encoded_bitrate"] = 60000
        assert policy.update(stats) == 8000000
    # A content change can demand more bitrate without a new capacity estimate.
    stats["encoded_bitrate"] = 7000000
    for index in range(31, 36):
        stats["receiver_reports"] = index
        assert policy.update(stats) == 8000000


def test_fresh_receiver_decrease_under_load_is_honored_once() -> None:
    policy = _BitratePolicy(1000000, 8000000)
    stats = _stats()
    stats.update(receiver_bitrate=8000000, receiver_bitrate_reports=1, encoded_bitrate=6000000)
    policy.update(stats)
    stats.update(receiver_bitrate=2000000, receiver_bitrate_reports=2)
    assert policy.update(stats) == 2000000
    assert policy.reason == "receiver_decrease"
    for _ in range(10):
        assert policy.update(stats) == 2000000  # Stale feedback cannot repeatedly cut/probe.


@pytest.mark.parametrize("signal", ["loss", "queue", "recovery"])
def test_congestion_without_remb_still_reduces_and_recovers(signal: str) -> None:
    policy = _BitratePolicy(1000000, 8000000)
    stats = _stats()
    stats.update(receiver_bitrate=0, encoded_bitrate=7000000)
    for index in range(1, 3):
        stats["receiver_reports"] = index
        stats["loss_fraction"] = 26 if signal == "loss" else 0
        stats["queue_age_ms"] = 150 if signal == "queue" else 0
        stats["recoveries"] = index if signal == "recovery" else 0
        policy.update(stats)
    assert policy.target == 6000000
    stats.update(loss_fraction=0, queue_age_ms=0)
    for index in range(3, 63):
        stats["receiver_reports"] = index
        policy.update(stats)
    assert policy.target == 8000000


def test_silence_and_empty_encoder_windows_never_probe() -> None:
    policy = _BitratePolicy(1000000, 8000000)
    stats = _stats()
    stats.update(loss_fraction=30, encoded_bitrate=4000000)
    for index in range(1, 30):
        stats["receiver_reports"] = index
        policy.update(stats)
    assert policy.target == 1000000
    stats.update(loss_fraction=0, encoded_bitrate=0)
    for index in range(30, 50):
        stats["receiver_reports"] = index
        assert policy.update(stats) == 1000000
    stats["encoded_bitrate"] = 1000000
    for _ in range(20):
        assert policy.update(stats) == 1000000
    for index in range(50, 140):
        stats["receiver_reports"] = index
        policy.update(stats)
    assert policy.target == 8000000


def test_cumulative_loss_catches_congestion_between_clean_receiver_reports() -> None:
    policy = _BitratePolicy(1000000, 8000000)
    stats = _stats()
    stats.update(
        receiver_reports=1, receiver_sequence=100, receiver_lost=0, encoded_bitrate=4000000
    )
    policy.update(stats)
    # Each latest RR says zero; thirty packets were lost earlier in each tick.
    stats.update(receiver_reports=2, receiver_sequence=200, receiver_lost=30)
    policy.update(stats)
    assert policy.target == 6000000
    assert policy.loss_fraction == 76
    assert policy.reason == "congestion"
    assert policy.update(stats) == 6000000  # Stale counters cannot cut again.
    stats.update(receiver_reports=3, receiver_sequence=300, receiver_lost=60)
    assert policy.update(stats) == 4500000


def test_cumulative_loss_credits_repairs_and_handles_extended_sequence_wrap() -> None:
    policy = _BitratePolicy(1000000, 8000000)
    stats = _stats()
    stats.update(
        receiver_reports=1, receiver_sequence=0xFFFFFFFE, receiver_lost=10, encoded_bitrate=1000000
    )
    policy.update(stats)
    stats.update(receiver_reports=2, receiver_sequence=5, receiver_lost=0, loss_fraction=200)
    assert policy.update(stats) == 8000000  # Late repairs make cumulative loss fall.
    assert policy.loss_fraction == 0
    policy.target = 1000000
    for index in range(3, 10):
        stats["receiver_reports"] = index
        assert policy.update(stats) == 1000000  # No received-packet progress, no probe.
    for index in range(10, 13):
        stats.update(receiver_reports=index, receiver_sequence=index * 100)
        policy.update(stats)
    assert policy.target == 1250000
