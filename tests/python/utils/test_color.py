# SPDX-License-Identifier: Apache-2.0

import slangpy as spy
import pytest
import falcor2.testing.helpers as helpers


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_colormap_gray(device_type: spy.DeviceType):
    device = helpers.get_device(device_type)
    module = spy.Module.load_from_file(device, "falcor2/utils.slang")

    res = module.colormap_gray(0.5)
    assert res == spy.float3(0.5)

    res = module.colormap_gray(2)
    assert res == spy.float3(1)

    res = module.colormap_gray(-1)
    assert res == spy.float3(0)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_colormap_jet(device_type: spy.DeviceType):
    device = helpers.get_device(device_type)
    module = spy.Module.load_from_file(device, "falcor2/utils.slang")

    # Test x = 0.5
    res = module.colormap_jet(0.5)
    helpers.assert_float3_equal(res, spy.float3(0.5, 1.0, 0.5))

    # Test x = 0.0
    res = module.colormap_jet(0.0)
    helpers.assert_float3_equal(res, spy.float3(0.0, 0.0, 0.5))

    # Test x = 1.0
    res = module.colormap_jet(1.0)
    helpers.assert_float3_equal(res, spy.float3(0.5, 0.0, 0.0))

    # Test x > 1.0 (should clamp to 1.0)
    res = module.colormap_jet(2.0)
    helpers.assert_float3_equal(res, spy.float3(0.5, 0.0, 0.0))

    # Test x < 0.0 (should clamp to 0.0)
    res = module.colormap_jet(-1.0)
    helpers.assert_float3_equal(res, spy.float3(0.0, 0.0, 0.5))


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_colormap_viridis(device_type: spy.DeviceType):
    device = helpers.get_device(device_type)
    module = spy.Module.load_from_file(device, "falcor2/utils.slang")

    # Test x = 0.5
    res = module.colormap_viridis(0.5)
    helpers.assert_float3_equal(res, spy.float3(0.12187948, 0.56583987, 0.54562263))

    # Test x = 0.0
    res = module.colormap_viridis(0.0)
    helpers.assert_float3_equal(res, spy.float3(0.27772733, 0.00540734, 0.33409981))

    # Test x = 1.0
    res = module.colormap_viridis(1.0)
    helpers.assert_float3_equal(res, spy.float3(0.98682980, 0.90633600, 0.13102000))

    # Test x > 1.0
    res = module.colormap_viridis(2.0)
    helpers.assert_float3_equal(res, spy.float3(0.98682980, 0.90633600, 0.13102000))

    # Test x < 0.0
    res = module.colormap_viridis(-1.0)
    helpers.assert_float3_equal(res, spy.float3(0.27772733, 0.00540734, 0.33409981))


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_colormap_plasma(device_type: spy.DeviceType):
    device = helpers.get_device(device_type)
    module = spy.Module.load_from_file(device, "falcor2/utils.slang")

    # Test x = 0.5
    res = module.colormap_plasma(0.5)
    helpers.assert_float3_equal(res, spy.float3(0.80275672, 0.27856503, 0.48639677))

    # Test x = 0.0
    res = module.colormap_plasma(0.0)
    helpers.assert_float3_equal(res, spy.float3(0.05873234, 0.02333671, 0.54334018))

    # Test x = 1.0
    res = module.colormap_plasma(1.0)
    helpers.assert_float3_equal(res, spy.float3(0.93304839, 0.96770000, 0.14876719))

    # Test x > 1.0
    res = module.colormap_plasma(2.0)
    helpers.assert_float3_equal(res, spy.float3(0.93304839, 0.96770000, 0.14876719))

    # Test x < 0.0
    res = module.colormap_plasma(-1.0)
    helpers.assert_float3_equal(res, spy.float3(0.05873234, 0.02333671, 0.54334018))


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_colormap_magma(device_type: spy.DeviceType):
    device = helpers.get_device(device_type)
    module = spy.Module.load_from_file(device, "falcor2/utils.slang")

    # Test x = 0.5
    res = module.colormap_magma(0.5)
    helpers.assert_float3_equal(res, spy.float3(0.71951918, 0.20837606, 0.46788972))

    # Test x = 0.0
    res = module.colormap_magma(0.0)
    # Values are slightly negative, saturate to 0
    helpers.assert_float3_equal(res, spy.float3(0.0, 0.0, 0.0))

    # Test x = 1.0
    res = module.colormap_magma(1.0)
    helpers.assert_float3_equal(res, spy.float3(0.99781816, 0.97700759, 0.73023212))

    # Test x > 1.0
    res = module.colormap_magma(2.0)
    helpers.assert_float3_equal(res, spy.float3(0.99781816, 0.97700759, 0.73023212))

    # Test x < 0.0
    res = module.colormap_magma(-1.0)
    helpers.assert_float3_equal(res, spy.float3(0.0, 0.0, 0.0))


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_colormap_inferno(device_type: spy.DeviceType):
    device = helpers.get_device(device_type)
    module = spy.Module.load_from_file(device, "falcor2/utils.slang")

    # Test x = 0.5
    res = module.colormap_inferno(0.5)
    helpers.assert_float3_equal(res, spy.float3(0.72772460, 0.21059096, 0.31991176))

    # Test x = 0.0
    res = module.colormap_inferno(0.0)
    # B component is slightly negative, saturates to 0
    helpers.assert_float3_equal(res, spy.float3(0.00021894, 0.00165100, 0.0))

    # Test x = 1.0
    res = module.colormap_inferno(1.0)
    # G component is slightly > 1, B is < 0
    helpers.assert_float3_equal(res, spy.float3(0.97984900, 1.0, 0.6568603))

    # Test x > 1.0
    res = module.colormap_inferno(2.0)
    helpers.assert_float3_equal(res, spy.float3(0.97984900, 1.0, 0.6568603))

    # Test x < 0.0
    res = module.colormap_inferno(-1.0)
    helpers.assert_float3_equal(res, spy.float3(0.00021894, 0.00165100, 0.0))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
