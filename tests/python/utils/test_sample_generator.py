# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public-source Sobol reference, stratification and PRNG continuation checks."""

import numpy as np
import pytest
import slangpy as spy

from falcor2.testing import helpers
from falcor2.utils.sample_generators import SampleGeneratorType, _load_tables, bind_sample_generator

MASK = (1 << 32) - 1
SAMPLER = SampleGeneratorType.sobol_pbrt


def _hash(x: int) -> int:
    x = (x + 0x7ED55D16 + (x << 12)) & MASK
    x = (x ^ 0xC761C23C ^ (x >> 19)) & MASK
    x = (x + 0x165667B1 + (x << 5)) & MASK
    x = ((x + 0xD3A2646C) ^ (x << 9)) & MASK
    x = (x + 0xFD7046C5 + (x << 3)) & MASK
    return (x ^ 0xB55A4F09 ^ (x >> 16)) & MASK


def _sobol(index: int, dimension: int, seed: int) -> int:
    value = 0
    for bit, direction in enumerate(_load_tables()["sobol_directions"][dimension]):
        if index & (1 << bit):
            value ^= direction
    return value ^ _hash((seed + dimension) & MASK)


def _overflow(index: int, seed: int, count: int) -> int:
    state = _hash(index ^ _hash(seed)) | 1
    for _ in range(count):
        state ^= (state << 13) & MASK
        state ^= state >> 7
        state ^= (state << 17) & MASK
    return state


def _function(device_type: spy.DeviceType, name: str):
    device = helpers.get_device(device_type)
    module = spy.Module(device.load_module("utils/test_sample_generator.slang"))
    function = module[f"{name}<{SAMPLER.slang_type_name}>"].write(
        lambda cursor: bind_sample_generator(cursor, device, SAMPLER)
    )
    return device, module, function


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_sobol_reference(device_type: spy.DeviceType):
    device, module, sample = _function(device_type, "sample")
    assert module[f"sampler_size<{SAMPLER.slang_type_name}>"]() == 12
    assert np.asarray(_load_tables()["sobol_directions"]).shape == (1024, 32)
    indices = np.array([0, 1, 2, 3, 511, 65535, 16777217, 0xFFFFFFFF], np.uint32)
    for dimension in [0, 1, 2, 15, 127, 128, 511, 1022, 1023]:
        for seed in [0, 1987, MASK]:
            result = sample(
                index=spy.Tensor.from_numpy(device, indices), seed=seed, dimension=dimension
            ).to_numpy()
            expected = [_sobol(int(i), dimension, seed) for i in indices]
            np.testing.assert_array_equal(
                result, expected, err_msg=f"dimension={dimension}, seed={seed}"
            )
    # The first Sobol coordinate is bit reversal, independent of the packaged table.
    for index in indices:
        expected = int(f"{int(index):032b}"[::-1], 2) ^ _hash(0)
        assert sample(index=int(index), seed=0, dimension=0) == expected


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_sobol_stratification_and_joint_quality(device_type: spy.DeviceType):
    device, _, sample = _function(device_type, "sample")
    indices = spy.Tensor.from_numpy(device, np.arange(4096, dtype=np.uint32))
    for seed in [0, 193]:
        for dimension in [0, 127, 128, 1023]:
            x = sample(index=indices, seed=seed, dimension=dimension).to_numpy() / 2**32
            for count in [16, 256, 4096]:
                assert len(np.unique((x[:count] * count).astype(int))) == count
        for d0, d1 in [(0, 1), (128, 129), (1022, 1023)]:
            x = sample(index=indices, seed=seed, dimension=d0).to_numpy() / 2**32
            y = sample(index=indices, seed=seed, dimension=d1).to_numpy() / 2**32
            assert abs(np.mean(x * y) - 0.25) < 0.005
            assert abs(np.mean(x + y < 1) - 0.5) < 0.015


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_sobol_pixel_and_vectors(device_type: spy.DeviceType):
    device, module, sample = _function(device_type, "sample")
    pixel_sample = module[f"sample_pixel<{SAMPLER.slang_type_name}>"].write(
        lambda cursor: bind_sample_generator(cursor, device, SAMPLER)
    )
    for pixel in [spy.uint2(0, 0), spy.uint2(137, 29), spy.uint2(1531, 719)]:
        seed = module.get_pixel_scrambling_seed(pixel=pixel)
        for index in [0, 1, 31]:
            for dimension in [0, 128, 1023, 1024, 1100]:
                assert pixel_sample(index=index, pixel=pixel, dimension=dimension) == sample(
                    index=index, seed=seed, dimension=dimension
                )
    vector = module[f"sample_vector<{SAMPLER.slang_type_name}>"].write(
        lambda cursor: bind_sample_generator(cursor, device, SAMPLER)
    )
    for dimension in [0, 1021, 1023, 1024]:
        values = vector(index=37, seed=921, dimension=dimension)
        for i in range(4):
            expected = int(sample(index=37, seed=921, dimension=dimension + i))
            assert float(values[i]) == (expected >> 8) / 2**24


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_sobol_exact_overflow(device_type: spy.DeviceType):
    device, module, sample = _function(device_type, "sample")
    for index, seed in [(0, 0), (37, 1987), (MASK, MASK)]:
        for dimension in [1024, 1025, 1100, 1280, 2048]:
            assert sample(index=index, seed=seed, dimension=dimension) == _overflow(
                index, seed, dimension - 1024 + 1
            )
    copy_matches = module[f"copy_matches<{SAMPLER.slang_type_name}>"].write(
        lambda cursor: bind_sample_generator(cursor, device, SAMPLER)
    )
    for dimension in [0, 1023, 1024, 1025, 2048]:
        assert copy_matches(index=37, seed=1987, dimension=dimension)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_legacy_streams(device_type: spy.DeviceType):
    device = helpers.get_device(device_type)
    module = spy.Module(device.load_module("utils/test_sample_generator.slang"))
    # Captured on main 306fa416af8054d7ed8a49c3396f8e2d362319cc.
    for kind, rows in [
        (
            "UniformSampleGenerator",
            [
                (0, False, (3227769423, 160584428, 761560138, 1808070401)),
                (0, True, (3651158080, 2397978495, 3716372436, 1815701769)),
                (1, False, (4225498831, 801179096, 3733110893, 385066240)),
                (1, True, (3274046624, 2756168274, 1638569707, 1666542660)),
                (12345, False, (2231106165, 3628503512, 3813197174, 937923973)),
                (12345, True, (2881243295, 1621496094, 2082269086, 2101172096)),
                (4294967295, False, (2230040006, 3761999289, 1662413296, 862746441)),
                (4294967295, True, (2993221886, 1604445515, 220094813, 2805089505)),
            ],
        ),
        (
            "TinyUniformSampleGenerator",
            [
                (0, False, (1013904223, 1196435762, 3519870697, 2868466484)),
                (0, True, (3180588591, 275760578, 3940108857, 4238044740)),
                (1, False, (1015568748, 1586005467, 2165703038, 3027450565)),
                (1, True, (4208510495, 2976406770, 3882227625, 4186733812)),
                (12345, False, (87628868, 71072467, 2332836374, 2726892157)),
                (12345, True, (1898778244, 1516587027, 942026326, 3543896509)),
                (4294967295, False, (1012239698, 806866057, 579071060, 2709482403)),
                (4294967295, True, (2792870490, 4179525105, 1437752988, 3849054539)),
            ],
        ),
    ]:
        for seed, pixel_mode, expected in rows:
            result = module[f"legacy_stream<{kind}>"](
                seed=seed, pixel=spy.uint2(1531, 719), pixel_mode=pixel_mode
            )
            assert tuple(int(result[i]) for i in range(4)) == expected
