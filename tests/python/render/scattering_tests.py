# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared numerical contracts for direct BSDFs and scene-backed materials.

Consistency is an algebraic check, not evidence of an unbiased estimator. The
independent mean check integrates evaluation using uniform-sphere directions and
counts failed sampling attempts as zero. Distribution tests remain separate.
"""

from collections.abc import Callable
from enum import IntFlag
from typing import Any

import numpy as np
import slangpy as spy


SAMPLES_PER_THREAD = 1024
GlobalBindings = dict[str, Any] | Callable[..., Any] | None


class BSDFFlags(IntFlag):
    """Python enum matching Slang's BSDFFlags definition."""

    none = 0x00
    diffuse_reflection = 0x01
    glossy_reflection = 0x02
    delta_reflection = 0x04
    diffuse_transmission = 0x10
    glossy_transmission = 0x20
    delta_transmission = 0x40
    glossy_curve = 0x80
    diffuse = 0x11
    glossy = 0x22
    delta = 0x44
    curve = 0x80
    reflection = 0x07
    transmission = 0x70
    non_delta = 0xB3
    non_delta_reflection = 0x03
    non_delta_transmission = 0x30
    all = 0xF7


class ScatteringTest:
    def __init__(
        self,
        device: spy.Device,
        module: spy.Module,
        target: str,
        target_bindings: dict[str, Any],
        global_bindings: GlobalBindings = None,
        sample_count: int = 65536,
        type_conformances: list[spy.TypeConformance] | None = None,
        prelude: str | None = None,
    ) -> None:
        self.device = device
        self.module = module
        self.target = target
        self.target_bindings = target_bindings
        self.global_bindings = global_bindings
        self.sample_count = sample_count
        self.type_conformances = type_conformances
        self.prelude = prelude
        self.messages = ""

    def function(self, name: str) -> Any:
        function = self.module[f"scattering_tests::{name}<{self.target}>"].as_func()
        if self.prelude is not None:
            function = function.prelude(self.prelude)
        if self.type_conformances is not None:
            function = function.type_conformances(self.type_conformances)
        if callable(self.global_bindings):
            return function.write(self.global_bindings)
        if self.global_bindings is not None:
            return function.set(self.global_bindings)
        return function


class SampleContractTest(ScatteringTest):
    def run(
        self,
        weight_tol: float = 1e-3,
        pdf_tol: float = 1e-3,
        quiet: bool = True,
        *,
        check_weight: bool = True,
        atol: float = 1e-8,
        require_continuous: bool = True,
    ) -> bool:
        """Check payload, marginal PDF and the represented component's throughput weight.

        ``check_weight=False`` retains finite evaluation, PDF agreement, unit
        directions and delta validation for documented numerical defects. Both full
        response and selected-component weights use the returned component identity;
        their joint-density check is independent of the marginal directional PDF check.
        """
        if not all(np.isfinite(x) and x > 0 for x in (weight_tol, pdf_tol, atol)):
            raise ValueError("Contract tolerances must be finite and positive")
        thread_count = (self.sample_count + SAMPLES_PER_THREAD - 1) // SAMPLES_PER_THREAD
        weight_errors = spy.Tensor.zeros(self.device, (thread_count,), dtype=float)
        pdf_errors = spy.Tensor.zeros(self.device, (thread_count,), dtype=float)
        counters = spy.Tensor.zeros(self.device, (4,), dtype="uint")
        self.function("test_sample_contract")(
            tid=spy.grid((thread_count,)),
            sample_count=self.sample_count,
            samples_per_thread=SAMPLES_PER_THREAD,
            seed=5678,
            target={"_type": self.target, **self.target_bindings},
            weight_rtol=weight_tol,
            pdf_rtol=pdf_tol,
            atol=atol,
            weight_errors=weight_errors,
            pdf_errors=pdf_errors,
            counters=counters,
        )
        counts = counters.to_numpy()
        weight_error = float(np.max(weight_errors.to_numpy()))
        pdf_error = float(np.max(pdf_errors.to_numpy()))
        accepted = counts[0] if require_continuous else counts[0] + counts[2]
        passed = bool(
            counts.sum() == self.sample_count
            and counts[3] == 0
            and accepted > 0
            and np.isfinite(pdf_error)
            and pdf_error <= 1
            and (not check_weight or (np.isfinite(weight_error) and weight_error <= 1))
        )
        self.messages = (
            f"Sample contract ({self.target}): {counts[0]} continuous, {counts[1]} rejected, "
            f"{counts[2]} delta, {counts[3]} invalid\n"
            f"  PDF error / tolerance: {pdf_error:.6g}\n"
            f"  Component weight error / tolerance: {weight_error:.6g} "
            f"({'checked' if check_weight else 'not required'})\n"
            f"  {'PASSED' if passed else 'FAILED'}"
        )
        if not passed or not quiet:
            print(self.messages)
        return passed


class EstimatorMeanTest(ScatteringTest):
    def run(self, *, seed: int = 20261105, quiet: bool = True) -> bool:
        """Compare independent means using six standard errors plus 2e-5.

        Use broad continuous lobes: uniform-sphere integration cannot reliably
        resolve arbitrarily narrow peaks. Delta expectations need analytic tests.
        Fixed independent streams make failures reproducible. Six standard errors
        is a conservative normal-approximation threshold, not a proof of equality.
        """
        sample_rng, integration_rng = (
            np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(2)
        )
        seeds = sample_rng.integers(0, 2**32, (self.sample_count, 3), dtype=np.uint32)
        z = integration_rng.uniform(-1, 1, self.sample_count)
        phi = integration_rng.uniform(0, 2 * np.pi, self.sample_count)
        radius = np.sqrt(1 - z * z)
        directions = np.stack((radius * np.cos(phi), radius * np.sin(phi), z), -1).astype(
            np.float32
        )
        target = {"_type": self.target, **self.target_bindings}
        weights = (
            self.function("sample_continuous_weight")(
                target=target, seed=spy.Tensor.from_numpy(self.device, seeds)
            )
            .to_numpy()
            .astype(np.float64)
        )
        values = self.function("eval_value")(
            target=target, wo=spy.Tensor.from_numpy(self.device, directions)
        ).to_numpy().astype(np.float64) * (4 * np.pi)
        error = np.abs(weights.mean(0) - values.mean(0))
        standard_error = np.sqrt(
            (weights.var(0, ddof=1) + values.var(0, ddof=1)) / self.sample_count
        )
        passed = bool(
            np.isfinite(weights).all()
            and np.isfinite(values).all()
            and np.all(error <= 6 * standard_error + 2e-5)
        )
        self.messages = (
            f"Estimator mean ({self.target}): sample={weights.mean(0)}, integral={values.mean(0)}, "
            f"SE={standard_error}; {'PASSED' if passed else 'FAILED'}"
        )
        if not passed or not quiet:
            print(self.messages)
        return passed
