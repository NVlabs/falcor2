# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Select and bind sequential sample generators, including standalone Slang callers."""

from enum import IntEnum
from functools import lru_cache
import json
from pathlib import Path
from typing import Any

import numpy as np
import slangpy as spy

from falcor2.utils.per_device_cache import PerDeviceCache


class SampleGeneratorType(IntEnum):
    """Concrete Slang generator types; Uniform preserves the renderer's default stream."""

    uniform = 0
    sobol_pbrt = 1

    @property
    def slang_type_name(self) -> str:
        return (
            "UniformSampleGenerator",
            "PBRTSobolSampleGenerator",
        )[self]


@lru_cache(maxsize=1)
def _load_tables() -> dict[str, Any]:
    return json.loads(Path(__file__).with_name("qmc_tables.json").read_text())


_sobol_cache: PerDeviceCache[spy.Buffer] = PerDeviceCache()


def _sobol_buffer(device: spy.Device) -> spy.Buffer:
    directions = np.asarray(_load_tables()["sobol_directions"], dtype=np.uint32)
    return device.create_buffer(
        data=directions, struct_size=4, usage=spy.BufferUsage.shader_resource
    )


def bind_sample_generator(cursor: Any, device: spy.Device, sampler: SampleGeneratorType) -> None:
    """Bind the selected generator's shared tables to a SlangPy dispatch cursor.

    Standalone callers specialize their generic function with ``sampler.slang_type_name``
    and attach this callback with ``Function.write``. Uniform needs no resources.
    Tables are uploaded once per device and released from the cache on close.
    """
    if sampler == SampleGeneratorType.sobol_pbrt:
        cursor["qmc_sobol_table"] = _sobol_cache.get_or_create(device, _sobol_buffer)
