# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU benchmarking for the Falcor2 Python test suite.

``BenchmarkPlugin`` owns the ``--benchmark*`` options and the BenchView
submission, so a conftest only has to register it. The payload itself is built
by ``slangpy.testing.benchmark``; this package supplies Falcor2's project
identity and the metric builders describing what a Falcor2 benchmark measures.
"""

from .fixtures import (
    benchmark_image_test,
    BenchmarkImageTest,
    RenderResult,
    SampleTimer,
)
from .plugin import (
    add_benchmark_options,
    BenchmarkPlugin,
    FALCOR2_PROJECT,
    build_compile_time_breakdown_metric,
    build_compile_time_by_program_metric,
)
