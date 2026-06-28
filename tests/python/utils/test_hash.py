# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import slangpy as spy
import pytest
import falcor2.testing.helpers as helpers


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_jenkins_hash(device_type: spy.DeviceType):
    device = helpers.get_device(device_type)
    module = spy.Module(device.load_module("falcor2/utils.slang"))
    inputs = [0x00000000, 0x00000001, 0x80000000, 0xFFFFFFFE, 0xFFFFFFFF]
    expected = [0x6B4ED927, 0xB48681B6, 0x7E7B3C12, 0x7E1DCAC2, 0xFE64C182]
    for input, expected in zip(inputs, expected):
        assert module.jenkins_hash(input) == expected


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
