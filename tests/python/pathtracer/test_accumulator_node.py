# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest
import slangpy as spy
import falcor2.testing.helpers as helpers


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_accumulator_update_and_output(device_type: spy.DeviceType, device: spy.Device) -> None:
    """Accumulator averages successive inputs without reallocating."""
    from falcor2.rendernodes import AccumulatorNode

    acc = AccumulatorNode.create(device)
    input_a = spy.Tensor.from_numpy(device, np.ones((4, 4, 4), dtype=np.float32))
    input_b = spy.Tensor.from_numpy(device, np.full((4, 4, 4), 3.0, dtype=np.float32))

    output = acc(input_a)
    np.testing.assert_allclose(output.to_numpy()[..., :3], 1.0, atol=1e-6)
    np.testing.assert_allclose(output.to_numpy()[..., 3], 1.0, atol=1e-6)

    output = acc(input_b)
    np.testing.assert_allclose(output.to_numpy()[..., :3], 2.0, atol=1e-6)
    np.testing.assert_allclose(output.to_numpy()[..., 3], 1.0, atol=1e-6)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_accumulator_reset(device_type: spy.DeviceType, device: spy.Device) -> None:
    """Accumulator reset starts a fresh accumulation sequence."""
    from falcor2.rendernodes import AccumulatorNode

    acc = AccumulatorNode.create(device)
    input_tensor = spy.Tensor.from_numpy(device, np.ones((2, 2, 4), dtype=np.float32))

    acc(input_tensor)
    acc.reset()
    output = acc(input_tensor)

    np.testing.assert_allclose(output.to_numpy()[..., :3], 1.0, atol=1e-6)
    np.testing.assert_allclose(output.to_numpy()[..., 3], 1.0, atol=1e-6)
