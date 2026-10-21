# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Narrow isotropic GGX retains transverse half-vector information."""

import numpy as np
import pytest
import slangpy as spy

import falcor2.testing.helpers as helpers


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_isotropic_ggx_near_normal_density(device_type: spy.DeviceType, device: spy.Device) -> None:
    module = spy.Module(device.load_module("render/test_isotropic_ggx_precision.slang"))
    x = np.array([0, 1e-5, 1e-4, 2e-4, 2.5e-4, 8.8e-4, 0.01, 0.1, 0.5])
    h = np.column_stack([x, np.zeros_like(x), np.sqrt(1 - x * x)]).astype(np.float32)
    directions = spy.Tensor.from_numpy(device, h)
    hd = h.astype(np.float64)
    for alpha in (0.0009, 0.0025, float(np.float16(0.1)) ** 2, 0.16, 1.0):
        alpha = float(np.float32(alpha))
        actual = module.density(alpha, directions).to_numpy()
        # Vector and scalar APIs have different available input precision. Check
        # each against its own exact input, including Z rounding to one.
        denom = np.sum(hd[:, :2] ** 2, axis=1) + alpha**2 * hd[:, 2] ** 2
        expected = alpha**2 / (np.pi * denom**2)
        scalar_denom = 1 - hd[:, 2] ** 2 + alpha**2 * hd[:, 2] ** 2
        scalar_expected = alpha**2 / (np.pi * scalar_denom**2)
        np.testing.assert_allclose(actual[:, 0], expected, rtol=1e-6)
        np.testing.assert_allclose(actual[:, 1], scalar_expected, rtol=1e-6)
        np.testing.assert_allclose(actual[:, 2], expected * hd[:, 2], rtol=1e-6)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_isotropic_ggx_reflection_density(device_type: spy.DeviceType, device: spy.Device) -> None:
    module = spy.Module(device.load_module("render/test_isotropic_ggx_precision.slang"))
    x = np.linspace(0, 0.04, 513)
    wo = np.column_stack([x, np.zeros_like(x), np.sqrt(1 - x * x)]).astype(np.float32)
    directions = spy.Tensor.from_numpy(device, wo)
    h = module.half_vector(directions).to_numpy().astype(np.float64)
    for alpha in (0.001, float(np.float16(0.1)) ** 2, 0.04):
        alpha = float(np.float32(alpha))
        actual = module.reflection(alpha, directions).to_numpy()
        denom = np.sum(h[:, :2] ** 2, axis=1) + alpha**2 * h[:, 2] ** 2
        d = alpha**2 / (np.pi * denom**2)
        cosine = wo[:, 2].astype(np.float64)
        masking = 2 / (1 + np.sqrt(1 + alpha**2 * (1 - cosine**2) / cosine**2))
        np.testing.assert_allclose(
            actual[:, :3], np.repeat((d * masking / 4)[:, None], 3, axis=1), rtol=2e-6
        )
        np.testing.assert_allclose(actual[:, 3], d / 4, rtol=2e-6)
