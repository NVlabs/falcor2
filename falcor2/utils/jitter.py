# SPDX-License-Identifier: Apache-2.0
"""Sample-jitter helpers shared by path-tracing examples and render nodes."""

from __future__ import annotations


def halton(index: int, base: int) -> float:
    """Return one value from a Halton sequence."""
    result = 0.0
    fraction = 1.0 / float(base)
    while index > 0:
        result += fraction * float(index % base)
        index //= base
        fraction /= float(base)
    return result


def frame_jitter(frame_index: int) -> tuple[float, float]:
    """Return a center-relative 2D jitter sample for ``frame_index``."""
    sample_index = frame_index + 1
    return halton(sample_index, 2) - 0.5, halton(sample_index, 3) - 0.5
