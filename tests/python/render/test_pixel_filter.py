# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from math import erf
from statistics import NormalDist

import numpy as np
import pytest
import slangpy as spy

import falcor2 as f2
import falcor2.testing.helpers as helpers


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES[:1])
def test_pixel_filter_sampling_and_evaluation(
    device_type: spy.DeviceType, device: spy.Device
) -> None:
    # A few deterministic inputs exercise inverse-CDF sampling, weights, and evaluation against FP64 formulas.
    points = np.array([[0, 0], [0.125, 0.25], [0.5, 0.5], [0.875, 0.75]], dtype=np.float32)
    gaussian = f2.GaussianReconstructionFilter()
    box = f2.BoxReconstructionFilter()

    module = spy.Module.load_from_file(device, "render/test_pixel_filter.slang")
    func = module.sample_filter.return_type(spy.Tensor)
    inputs = spy.Tensor.from_numpy(device, points)
    # The same callable must resolve and upload each concrete filter independently.
    np.testing.assert_allclose(
        func(pixel_filter=box, u=inputs, offset=inputs).to_numpy()[:, 0, :2], points - 0.5
    )
    # The kernel has PBRT/Mitsuba's shape, with a truncated Gaussian sampling proposal.
    normal = NormalDist()
    for sigma, radius, width in ((0.5, 1.5, 2.0), (0.75, 0.01, 0.0)):
        box.width = width
        gaussian.sigma = sigma
        gaussian.radius = radius
        sigma, radius = gaussian.sigma, gaussian.radius
        offsets = np.array(
            [[0, 0], [radius / 2, 0], [radius, 0], [2 * radius, 0]], dtype=np.float32
        )
        offset_inputs = spy.Tensor.from_numpy(device, offsets)
        result = func(pixel_filter=gaussian, u=inputs, offset=offset_inputs).to_numpy()
        cdf_limit = erf(radius / (np.sqrt(2) * sigma))
        expected = sigma * np.array(
            [[normal.inv_cdf(0.5 + (float(u) - 0.5) * cdf_limit) for u in pair] for pair in points]
        )
        np.testing.assert_allclose(result[:, 0, :2], expected, atol=2e-6 * sigma)
        a = 0.5 * (radius / sigma) ** 2
        integral = np.sqrt(2 * np.pi) * sigma * cdf_limit - 2 * radius * np.exp(-a)

        def evaluate(offset: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            t = offset.astype(np.float64) / sigma
            exponent = 0.5 * t**2
            # exp(-exponent)-exp(-a), retaining precision for tiny cutoffs.
            kernel = np.exp(-exponent) * -np.expm1(np.minimum(exponent - a, 0)) / integral
            value = kernel.prod(axis=1)
            pdf = np.exp(-exponent.sum(axis=1)) / (2 * np.pi * sigma**2 * cdf_limit**2)
            return value, np.where(np.all(np.abs(offset) <= radius, axis=1), pdf, 0)

        value, pdf = evaluate(result[:, 0, :2])
        np.testing.assert_allclose(result[:, 0, 2], value / pdf, rtol=2e-4, atol=2e-5)
        np.testing.assert_allclose(result[:, 1, 0], value, rtol=2e-4, atol=2e-5)
        value, _ = evaluate(offsets)
        np.testing.assert_allclose(result[:, 1, 1], value, rtol=2e-4, atol=2e-5)
        result = func(pixel_filter=box, u=inputs, offset=offset_inputs).to_numpy()
        np.testing.assert_allclose(result[:, 0, :2], width * (points - 0.5))
        box_value = 1 / width**2 if width else 0
        np.testing.assert_equal(result[:, 1, 0], box_value)
        box_eval = np.where(np.all(np.abs(offsets) <= width / 2, axis=1), box_value, 0)
        np.testing.assert_equal(result[:, 1, 1], box_eval)
        np.testing.assert_equal(result[:, 0, 2], 1)

    box.width = -1
    with pytest.raises(RuntimeError, match="width must be finite and nonnegative"):
        func(pixel_filter=box, u=inputs, offset=inputs)
    box.width = 0
    np.testing.assert_equal(func(pixel_filter=box, u=inputs, offset=inputs).to_numpy()[:, 0, :2], 0)
