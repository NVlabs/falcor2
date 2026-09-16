# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Distribution and sample-contract tests for phase functions."""

from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
import pytest
import slangpy as spy

import falcor2.testing.helpers as helpers
from falcor2.utils.chi2 import ChiSquareTest


SAMPLES_PER_THREAD = 1024
WI = spy.float3(0.267261, -0.534522, 0.801784)


PHASE_FUNCTION_CONFIGS = [
    pytest.param(
        "phase_function_tests::TestIsotropicPhaseFunction", {"dummy": 0.0}, id="isotropic"
    ),
    pytest.param("HenyeyGreensteinPhaseFunction", {"g": 0.65}, id="henyey-greenstein-forward"),
    pytest.param("HenyeyGreensteinPhaseFunction", {"g": -0.65}, id="henyey-greenstein-backward"),
    pytest.param(
        "HenyeyGreensteinPhaseFunction", {"g": 1e-4}, id="henyey-greenstein-near-isotropic"
    ),
    pytest.param(
        "DualHenyeyGreensteinPhaseFunction",
        {
            "mean_cosine": [[0.7, 0.7, 0.7], [-0.3, -0.3, -0.3]],
            "w0": [0.35, 0.35, 0.35],
        },
        id="dual-henyey-greenstein",
    ),
    pytest.param(
        "DualHenyeyGreensteinPhaseFunction",
        {
            "mean_cosine": [[0.75, 0.55, 0.35], [-0.45, -0.25, -0.05]],
            "w0": [0.2, 0.5, 0.8],
        },
        id="dual-henyey-greenstein-spectral",
    ),
    pytest.param(
        "DualHenyeyGreensteinPhaseFunction",
        {
            "mean_cosine": [[1e-4, 1e-4, 1e-4], [-2e-4, -2e-4, -2e-4]],
            "w0": [0.5, 0.5, 0.5],
        },
        id="dual-henyey-greenstein-near-isotropic",
    ),
]


def _resource_scope(fixture_name: str, config: pytest.Config) -> Literal["function", "module"]:
    """Select fixture lifetime from the configured device cache policy."""
    return "function" if config.getoption("--device-cache-policy") == "test" else "module"


@dataclass(frozen=True)
class _Resources:
    device: spy.Device
    module: spy.Module


@pytest.fixture(scope=_resource_scope, params=helpers.DEFAULT_DEVICE_TYPES)
def _resources(request: pytest.FixtureRequest) -> _Resources:
    """Create shader test resources for each configured GPU backend."""
    device_type = cast(spy.DeviceType, request.param)
    device = helpers.get_device(device_type)
    module = spy.Module(device.load_module("render/phase_function_tests.slang"))
    return _Resources(device=device, module=module)


def test_null_phase_function_sample_fails(_resources: _Resources) -> None:
    """The invalid sentinel is not a continuous sampling distribution."""
    result = _resources.module["phase_function_tests::test_null_phase_function_sample"](wi=WI)
    assert np.allclose(result, 0.0)


@pytest.mark.parametrize("phase_function_type,phase_function_bindings", PHASE_FUNCTION_CONFIGS)
def test_phase_function_chi2(
    _resources: _Resources,
    phase_function_type: str,
    phase_function_bindings: dict[str, Any],
) -> None:
    """Verify sample() directions follow eval_pdf() over the sphere."""
    test = ChiSquareTest(
        device=_resources.device,
        module=_resources.module,
        target=f"phase_function_tests::PhaseFunctionTarget<{phase_function_type}>",
        target_bindings={"phase_function": phase_function_bindings, "wi": WI},
        res=spy.uint2(100, 51),
        sample_count=500000,
        ires=16,
    )
    assert test.run(test_count=len(PHASE_FUNCTION_CONFIGS)), test.messages


@pytest.mark.parametrize("phase_function_type,phase_function_bindings", PHASE_FUNCTION_CONFIGS)
def test_phase_function_eval_sample_consistency(
    _resources: _Resources,
    phase_function_type: str,
    phase_function_bindings: dict[str, Any],
) -> None:
    """Verify PhaseFunctionSample reports normalized directions and eval/pdf weights."""
    sample_count = 100000
    thread_count = (sample_count + SAMPLES_PER_THREAD - 1) // SAMPLES_PER_THREAD
    weight_errors = spy.Tensor.zeros(_resources.device, (thread_count,), dtype=float)
    pdf_errors = spy.Tensor.zeros(_resources.device, (thread_count,), dtype=float)
    direction_errors = spy.Tensor.zeros(_resources.device, (thread_count,), dtype=float)
    counters = spy.Tensor.zeros(_resources.device, (3,), dtype="uint")

    kernel = _resources.module[
        f"phase_function_tests::test_eval_sample_consistency<{phase_function_type}>"
    ]
    kernel(
        tid=spy.grid((thread_count,)),
        sample_count=sample_count,
        samples_per_thread=SAMPLES_PER_THREAD,
        seed=5678,
        phase_function=phase_function_bindings,
        wi=WI,
        weight_errors=weight_errors,
        pdf_errors=pdf_errors,
        direction_errors=direction_errors,
        counters=counters,
    )

    counts = counters.to_numpy()
    assert int(counts[0]) == sample_count
    assert int(counts[1]) == 0
    assert int(counts[2]) == 0
    assert float(np.max(weight_errors.to_numpy())) < 1e-4
    assert float(np.max(pdf_errors.to_numpy())) < 1e-4
    assert float(np.max(direction_errors.to_numpy())) < 1e-5
