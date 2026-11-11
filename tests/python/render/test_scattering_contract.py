# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Witnesses that the shared checker catches invalid samples and evaluations."""

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import slangpy as spy

import falcor2.testing.helpers as helpers
from falcor2.utils.chi2 import ChiSquareTest

from .scattering_tests import EstimatorMeanTest, SampleContractTest


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize(
    "full_response,changes,passes",
    [
        pytest.param(False, {}, True, id="selected-component"),
        pytest.param(True, {}, True, id="full-response"),
        pytest.param(False, {"component_xor": 1}, False, id="wrong-component"),
        pytest.param(False, {"component_xor": 8}, False, id="unknown-component"),
        pytest.param(False, {"joint_pdf_scale": 2.0}, False, id="wrong-joint-density"),
        pytest.param(False, {"marginal_pdf_scale": 2.0}, False, id="wrong-marginal-density"),
        pytest.param(True, {"marginal_pdf_scale": 2.0}, False, id="full-wrong-marginal-density"),
        pytest.param(False, {"claim_full_response": True}, False, id="wrong-full-response-marker"),
        pytest.param(False, {"weight_scale": 2.0}, False, id="wrong-component-weight"),
        pytest.param(True, {"weight_scale": 2.0}, False, id="wrong-full-response-weight"),
    ],
)
def test_default_contract_checks_returned_component(
    device: spy.Device, full_response: bool, changes: dict[str, Any], passes: bool
) -> None:
    module = spy.Module(device.load_module("render/scattering_test_fixtures.slang"))
    bindings = {
        "full_response": full_response,
        "claim_full_response": False,
        "component_xor": 0,
        "joint_pdf_scale": 1.0,
        "marginal_pdf_scale": 1.0,
        "weight_scale": 1.0,
        **changes,
    }
    test = SampleContractTest(device, module, "ComponentTarget", bindings, sample_count=4096)
    assert test.run() == passes, test.messages


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize(
    "sample_changes,evaluation_changes,category,weight_passes,pdf_passes",
    [
        pytest.param({}, {}, 1, True, True, id="continuous"),
        pytest.param({"pdf": 0.0, "flags": 4}, {}, 2, True, True, id="delta"),
        pytest.param({"flags": 4}, {}, 0, True, True, id="delta-positive-pdf"),
        pytest.param({"pdf": 0.0}, {}, 0, True, True, id="continuous-zero-pdf"),
        pytest.param({"pdf": -1.0}, {}, 0, True, True, id="negative-pdf"),
        pytest.param({"pdf": np.nan}, {}, 0, True, True, id="nan-pdf"),
        pytest.param({"weight": [1.0, np.inf, 1.0]}, {}, 0, True, True, id="infinite-weight"),
        pytest.param({"weight": [-1.0, 1.0, 1.0]}, {}, 0, True, True, id="negative-weight"),
        pytest.param({"wo": [0.0, 0.0, 2.0]}, {}, 0, True, True, id="nonunit-direction"),
        pytest.param({"wo": [0.0, np.nan, 1.0]}, {}, 0, True, True, id="nan-direction"),
        pytest.param({}, {"value": [np.nan, 1.0, 1.0]}, 0, True, True, id="nan-evaluation"),
        pytest.param({}, {"value": [-1.0, 1.0, 1.0]}, 0, True, True, id="negative-evaluation"),
        pytest.param({}, {"pdf": np.inf}, 0, True, True, id="infinite-evaluated-pdf"),
        pytest.param({}, {"pdf": 0.0}, 0, True, True, id="zero-evaluated-pdf"),
        pytest.param({}, {"pdf": 0.25}, 1, True, False, id="wrong-pdf"),
        pytest.param({"weight": [2.0, 2.0, 2.0]}, {}, 1, False, True, id="wrong-weight"),
        pytest.param(
            {"pdf": 1e-12, "weight": [1e6, 1e6, 1e6]},
            {"pdf": 1e-12, "value": [1e-6, 1e-6, 1e-6]},
            1,
            True,
            True,
            id="tiny-density",
        ),
        pytest.param(
            {"pdf": 1e-12, "weight": [1e6, 1e6, 1e6]},
            {"pdf": 1e-12, "value": [1e-5, 1e-5, 1e-5]},
            1,
            False,
            True,
            id="tiny-density-wrong-weight",
        ),
    ],
)
def test_sample_contract_checker(
    device: spy.Device,
    sample_changes: dict[str, Any],
    evaluation_changes: dict[str, Any],
    category: int,
    weight_passes: bool,
    pdf_passes: bool,
) -> None:
    module = spy.Module(device.load_module("render/scattering_tests.slang"))
    sample = {"wo": [0.0, 0.0, 1.0], "weight": [1.0, 1.0, 1.0], "pdf": 0.5, "flags": 1}
    evaluated = {"value": [0.5, 0.5, 0.5], "pdf": 0.5}
    sample.update(deepcopy(sample_changes))
    evaluated.update(deepcopy(evaluation_changes))
    result = module["scattering_tests::check_sample"](
        sample,
        evaluated,
        1e-3,
        1e-3,
        1e-8,
    )
    assert result["category"] == category
    assert (result["weight_error"] <= 1) == weight_passes
    assert (result["pdf_error"] <= 1) == pdf_passes


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("pdf_scale", [1.0, 2.0], ids=["correct", "wrong-density"])
def test_independent_checks_detect_wrong_sampling_density(
    device: spy.Device,
    pdf_scale: float,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The chi-square utility writes diagnostics on the intentional failure.
    monkeypatch.chdir(tmp_path)
    module = spy.Module(device.load_module("render/scattering_test_fixtures.slang"))
    bindings = {"pdf_scale": pdf_scale}
    contract = SampleContractTest(device, module, "RejectionTarget", bindings, sample_count=8192)
    assert contract.run(), contract.messages
    mean = EstimatorMeanTest(device, module, "RejectionTarget", bindings, sample_count=65536)
    assert mean.run() == (pdf_scale == 1.0), mean.messages
    distribution = ChiSquareTest(
        device=device,
        module=module,
        target="RejectionTarget",
        target_bindings=bindings,
        res=spy.uint2(16, 9),
        sample_count=65536,
        ires=16,
    )
    assert distribution.run(significance_level=0.001) == (pdf_scale == 1.0), distribution.messages
