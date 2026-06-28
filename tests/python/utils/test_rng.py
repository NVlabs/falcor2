# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import slangpy as spy
import pytest
import falcor2.testing.helpers as helpers


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_lcg(device_type: spy.DeviceType):
    device = helpers.get_device(device_type)
    module = spy.Module(device.load_module("utils/test_rng.slang"))
    seed = 0xDEADBEEF
    assert module["test_rng<LCG>"](index=0, seed=seed) == 1789648770
    assert module["test_rng<LCG>"](index=1000, seed=seed) == 1401344458
    assert module["test_rng<LCG>"](index=1000000, seed=seed) == 1407034562


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_xorshift32(device_type: spy.DeviceType):
    device = helpers.get_device(device_type)
    module = spy.Module(device.load_module("utils/test_rng.slang"))
    seed = 0xDEADBEEF
    assert module["test_rng<Xorshift32>"](index=0, seed=seed) == 352763474
    assert module["test_rng<Xorshift32>"](index=1000, seed=seed) == 3588811896
    assert module["test_rng<Xorshift32>"](index=1000000, seed=seed) == 1243982098


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_splitmix64(device_type: spy.DeviceType):
    device = helpers.get_device(device_type)
    module = spy.Module(device.load_module("utils/test_rng.slang"))
    seed = spy.uint2(0xDEADBEEF, 0xBEEFDEAD)
    assert module["test_rng<SplitMix64>"](index=0, seed=seed) == 2715894472
    assert module["test_rng<SplitMix64>"](index=1000, seed=seed) == 941800462
    assert module["test_rng<SplitMix64>"](index=1000000, seed=seed) == 622993350


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_xoshiro128starstar(device_type: spy.DeviceType):
    device = helpers.get_device(device_type)
    module = spy.Module(device.load_module("utils/test_rng.slang"))
    seed = spy.uint4(0x01234567, 0x89ABCDEF, 0x76543210, 0xFEDCBA98)
    assert module["test_rng<Xoshiro128StarStar>"](index=0, seed=seed) == 2576977298
    assert module["test_rng<Xoshiro128StarStar>"](index=1000, seed=seed) == 1143679330
    assert module["test_rng<Xoshiro128StarStar>"](index=1000000, seed=seed) == 3091139770


if __name__ == "__main__":
    pytest.main([__file__, "-vs"])
