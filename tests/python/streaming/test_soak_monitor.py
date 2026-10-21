# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from types import SimpleNamespace

import pytest

pytest.importorskip("psutil")
import process_sample


def test_process_sampling_filters_gpu_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    pid = os.getpid()
    monkeypatch.setattr(
        process_sample.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout=f"{pid}, 12\n{pid}, 8\n99999999, 999\n"),
    )
    stats = process_sample.sample(pid)
    assert stats["gpu_memory_mib"] == 20
    assert stats["rss_bytes"] > 0
    assert stats["threads"] > 0
    assert stats["os_resources"] > 0


def test_unavailable_gpu_accounting_is_not_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        process_sample.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout=f"{os.getpid()}, [N/A]\n"),
    )
    assert process_sample.sample(os.getpid())["gpu_memory_mib"] is None
