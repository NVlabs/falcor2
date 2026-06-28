# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
BSDF statistical test runners.

Provides test classes for validating BSDF implementations:
- Chi-square goodness-of-fit test (sample distribution matches eval_pdf)
- Eval/sample consistency (eval/eval_pdf matches weight from sample)
- Helmholtz reciprocity (f(wi,wo) = f(wo,wi) for non-delta reflection)
"""

import numpy as np
import slangpy as spy
from typing import Any, Optional

from falcor2.utils.chi2 import ChiSquareTest

SAMPLES_PER_THREAD: int = 1024

# Keep in sync with Slang LobeTypes::all. SlangPy zero-fills omitted struct
# fields in Python dict bindings, so tests must supply this nonzero default.
LOBE_TYPES_ALL: int = 0xF7


def _make_bsdf_context(bc: Optional[dict[str, Any]]) -> dict[str, Any]:
    context: dict[str, Any] = {
        "ior_i": 1.0,
        "ior_t": 1.0,
        "sample_lobe_types_hint": LOBE_TYPES_ALL,
    }
    if bc is not None:
        context.update(bc)
    return context


class BSDFChiSquareTest:
    """Chi-square goodness-of-fit test for BSDF sampling.

    Tests that bsdf.sample() produces directions distributed
    according to bsdf.eval_pdf() over the full sphere.

    Only tests continuous (non-delta) lobes. BSDFs with purely delta
    distributions (e.g., perfect mirrors, smooth dielectrics) cannot be
    tested this way.
    """

    def __init__(
        self,
        device: spy.Device,
        module: spy.Module,
        bsdf_type: str,
        bsdf_bindings: dict[str, Any],
        wi: spy.float3,
        bc: Optional[dict[str, Any]] = None,
        res: spy.uint2 = spy.uint2(200, 101),
        sample_count: int = 1000000,
        ires: int = 16,
        global_bindings: dict[str, Any] | None = None,
    ):
        """
        Initialize a BSDF chi-square test.

        Args:
            device: Device for running kernels.
            module: Module containing the BSDF type and test infrastructure.
            bsdf_type: Name of the Slang type implementing IBSDF.
            bsdf_bindings: Dictionary of field values for the BSDF struct.
            wi: Incident direction (must be in positive hemisphere).
            bc: BSDF context with IOR values. Defaults to vacuum (1.0/1.0).
            res: Resolution of the 2D histogram grid.
            sample_count: Total number of samples to draw.
            ires: Integration resolution per bin for PDF tabulation. Higher values
                give more accurate integration for peaked distributions (e.g., GGX).
        """
        super().__init__()

        self.test = ChiSquareTest(
            device=device,
            module=module,
            target=f"bsdf_tests::BSDFTarget<{bsdf_type}>",
            target_bindings={
                "bsdf": bsdf_bindings,
                "wi": wi,
                "bc": _make_bsdf_context(bc),
            },
            res=res,
            sample_count=sample_count,
            ires=ires,
            global_bindings=global_bindings,
        )

        self.messages = ""

    def run(self, significance_level: float = 0.01, **kwargs: Any) -> bool:
        """
        Run the chi-square test.

        Args:
            significance_level: Threshold for rejecting the null hypothesis.
            **kwargs: Additional arguments passed to ChiSquareTest.run().

        Returns:
            True if the null hypothesis is accepted (samples match PDF).
        """
        result = self.test.run(significance_level=significance_level, **kwargs)
        self.messages = self.test.messages
        return result


class BSDFEvalSampleConsistencyTest:
    """Test eval/sample consistency of a BSDF.

    For each non-delta sample, verifies:
    1. eval(wi, wo) / eval_pdf(wi, wo) ~= weight (weight consistency)
    2. eval_pdf(wi, wo) ~= pdf from sample() (pdf consistency)
    """

    def __init__(
        self,
        device: spy.Device,
        module: spy.Module,
        bsdf_type: str,
        bsdf_bindings: dict[str, Any],
        wi: spy.float3,
        bc: Optional[dict[str, Any]] = None,
        sample_count: int = 100000,
        global_bindings: dict[str, Any] | None = None,
    ):
        """
        Initialize a BSDF eval/sample consistency test.

        Args:
            device: Device for running kernels.
            module: Module containing the BSDF type and test infrastructure.
            bsdf_type: Name of the Slang type implementing IBSDF.
            bsdf_bindings: Dictionary of field values for the BSDF struct.
            wi: Incident direction (must be in positive hemisphere).
            bc: BSDF context with IOR values. Defaults to vacuum (1.0/1.0).
            sample_count: Total number of samples to draw.
        """
        super().__init__()
        self.device = device
        self.module = module
        self.bsdf_type = bsdf_type
        self.bsdf_bindings = bsdf_bindings
        self.wi = wi
        self.bc = _make_bsdf_context(bc)
        self.sample_count = sample_count
        self.global_bindings = global_bindings
        self.messages = ""

    def run(
        self,
        weight_tol: float = 1e-3,
        pdf_tol: float = 1e-3,
        quiet: bool = True,
    ) -> bool:
        """
        Run the eval/sample consistency test.

        Args:
            weight_tol: Maximum allowed relative error for weight consistency.
            pdf_tol: Maximum allowed relative error for PDF consistency.
            quiet: If True, suppress console output if test passes.

        Returns:
            True if all consistency checks pass.
        """
        thread_count = (self.sample_count + SAMPLES_PER_THREAD - 1) // SAMPLES_PER_THREAD

        weight_errors = spy.Tensor.zeros(self.device, (thread_count,), dtype=float)
        pdf_errors = spy.Tensor.zeros(self.device, (thread_count,), dtype=float)
        counters = spy.Tensor.zeros(self.device, (4,), dtype="uint")

        kernel = self.module[f"bsdf_tests::test_eval_sample_consistency<{self.bsdf_type}>"]
        if self.global_bindings is not None:
            kernel = kernel.as_func().set(self.global_bindings)
        kernel(
            tid=spy.grid((thread_count,)),
            sample_count=self.sample_count,
            samples_per_thread=SAMPLES_PER_THREAD,
            seed=5678,
            bsdf=self.bsdf_bindings,
            wi=self.wi,
            bc=self.bc,
            weight_errors=weight_errors,
            pdf_errors=pdf_errors,
            counters=counters,
        )

        weight_errs = weight_errors.to_numpy()
        pdf_errs = pdf_errors.to_numpy()
        counts = counters.to_numpy()

        max_weight_err = float(np.max(weight_errs))
        max_pdf_err = float(np.max(pdf_errs))
        n_valid = int(counts[0])
        n_failed = int(counts[1])
        n_delta = int(counts[2])
        n_negative = int(counts[3])

        passed = True
        self.messages = f"Eval/sample consistency ({self.bsdf_type}):\n"
        self.messages += (
            f"  Samples: {n_valid} valid, {n_failed} failed, "
            f"{n_delta} delta, {n_negative} negative\n"
        )
        self.messages += f"  Max weight error: {max_weight_err:.6e} (tolerance: {weight_tol})\n"
        self.messages += f"  Max PDF error: {max_pdf_err:.6e} (tolerance: {pdf_tol})\n"

        if n_negative > 0:
            self.messages += f"  FAIL: {n_negative} samples with negative pdf or weight\n"
            passed = False
        if max_weight_err > weight_tol:
            self.messages += (
                f"  FAIL: Weight error {max_weight_err:.6e} exceeds " f"tolerance {weight_tol}\n"
            )
            passed = False
        if max_pdf_err > pdf_tol:
            self.messages += (
                f"  FAIL: PDF error {max_pdf_err:.6e} exceeds " f"tolerance {pdf_tol}\n"
            )
            passed = False
        if n_valid == 0:
            self.messages += "  FAIL: No valid (non-delta) samples to test\n"
            passed = False

        if passed:
            self.messages += "  PASSED\n"

        if not passed or not quiet:
            print(self.messages)

        return passed


class BSDFReciprocityTest:
    """Test Helmholtz reciprocity for a BSDF.

    For non-delta reflection lobes, verifies:
      eval(wi, wo) * |wi.z| ~= eval(wo, wi) * |wo.z|

    This follows from f(wi, wo) = f(wo, wi) and the convention
    that eval returns f * |cos(theta_o)|.

    Only tests directions where both wi and wo are in the positive hemisphere.
    """

    def __init__(
        self,
        device: spy.Device,
        module: spy.Module,
        bsdf_type: str,
        bsdf_bindings: dict[str, Any],
        wi: spy.float3,
        bc: Optional[dict[str, Any]] = None,
        sample_count: int = 100000,
        global_bindings: dict[str, Any] | None = None,
    ):
        """
        Initialize a BSDF reciprocity test.

        Args:
            device: Device for running kernels.
            module: Module containing the BSDF type and test infrastructure.
            bsdf_type: Name of the Slang type implementing IBSDF.
            bsdf_bindings: Dictionary of field values for the BSDF struct.
            wi: Incident direction (must be in positive hemisphere).
            bc: BSDF context with IOR values. Defaults to vacuum (1.0/1.0).
            sample_count: Total number of samples to draw.
        """
        super().__init__()
        self.device = device
        self.module = module
        self.bsdf_type = bsdf_type
        self.bsdf_bindings = bsdf_bindings
        self.wi = wi
        self.bc = _make_bsdf_context(bc)
        self.sample_count = sample_count
        self.global_bindings = global_bindings
        self.messages = ""

    def run(self, tolerance: float = 1e-3, quiet: bool = True) -> bool:
        """
        Run the reciprocity test.

        Args:
            tolerance: Maximum allowed relative error for reciprocity.
            quiet: If True, suppress console output if test passes.

        Returns:
            True if reciprocity holds within tolerance.
        """
        thread_count = (self.sample_count + SAMPLES_PER_THREAD - 1) // SAMPLES_PER_THREAD

        errors = spy.Tensor.zeros(self.device, (thread_count,), dtype=float)
        counters = spy.Tensor.zeros(self.device, (2,), dtype="uint")

        kernel = self.module[f"bsdf_tests::test_reciprocity<{self.bsdf_type}>"]
        if self.global_bindings is not None:
            kernel = kernel.as_func().set(self.global_bindings)
        kernel(
            tid=spy.grid((thread_count,)),
            sample_count=self.sample_count,
            samples_per_thread=SAMPLES_PER_THREAD,
            seed=9012,
            bsdf=self.bsdf_bindings,
            wi=self.wi,
            bc=self.bc,
            errors=errors,
            counters=counters,
        )

        errs = errors.to_numpy()
        counts = counters.to_numpy()

        max_err = float(np.max(errs))
        n_valid = int(counts[0])
        n_skipped = int(counts[1])

        passed = True
        self.messages = f"Reciprocity ({self.bsdf_type}):\n"
        self.messages += f"  Samples: {n_valid} valid, {n_skipped} skipped\n"
        self.messages += f"  Max reciprocity error: {max_err:.6e} (tolerance: {tolerance})\n"

        if max_err > tolerance:
            self.messages += (
                f"  FAIL: Reciprocity error {max_err:.6e} exceeds " f"tolerance {tolerance}\n"
            )
            passed = False
        if n_valid == 0:
            # Not a failure -- could be pure transmission or pure delta BSDF.
            self.messages += "  WARN: No valid (non-delta reflection) samples to test\n"

        if passed:
            self.messages += "  PASSED\n"

        if not passed or not quiet:
            print(self.messages)

        return passed
