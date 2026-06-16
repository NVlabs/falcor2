# SPDX-License-Identifier: Apache-2.0
import pytest
import slangpy as spy
import falcor2.testing.helpers as helpers


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_enumerate_adapters(device_type: spy.DeviceType):
    spy.Device.enumerate_adapters(type=device_type)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
