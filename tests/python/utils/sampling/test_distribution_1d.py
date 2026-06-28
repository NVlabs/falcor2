# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import slangpy as spy
import numpy as np
import pytest
import falcor2.testing.helpers as helpers
from falcor2.utils.chi2 import ChiSquareTest
from falcor2 import DiscreteDistribution1D, AliasTable1D


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize(
    "weights",
    [
        np.array([0.5, 0.5]),
        np.array([0.1, 0.2, 0.3, 0.4]),
        np.exp(-np.linspace(-2, 2, 32) ** 2),
    ],
)
def test_discrete_distribution_1d(device_type: spy.DeviceType, weights: np.ndarray):
    device = helpers.get_device(device_type)
    module = spy.Module(device.load_module("utils/sampling/test_distribution_1d.slang"))
    dist = DiscreteDistribution1D(device, weights)

    test = ChiSquareTest(
        device=device,
        module=module,
        target="DiscreteDistribution1DTarget",
        target_bindings={"dist": dist},
        res=spy.uint2(dist.size, 1),
    )
    assert test.run(), test.messages


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize(
    "weights",
    [
        np.array([0.5, 0.5]),
        np.array([0.1, 0.2, 0.3, 0.4]),
        np.exp(-np.linspace(-2, 2, 32) ** 2),
    ],
)
def test_alias_table_1d(device_type: spy.DeviceType, weights: np.ndarray):
    device = helpers.get_device(device_type)
    module = spy.Module(device.load_module("utils/sampling/test_distribution_1d.slang"))
    dist = AliasTable1D(device, weights)

    test = ChiSquareTest(
        device=device,
        module=module,
        target="AliasTable1DTarget",
        target_bindings={"dist": dist},
        res=spy.uint2(dist.size, 1),
    )
    assert test.run(), test.messages


if __name__ == "__main__":
    pytest.main([__file__, "-vs"])
