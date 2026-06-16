# SPDX-License-Identifier: Apache-2.0

# Portions of this file are derived from Mitsuba 3.
# Copyright (c) Mitsuba contributors.
# Licensed under the BSD 3-Clause License.
# See LICENSES/mitsuba3.txt for the full license text.

"""
Chi-square goodness-of-fit test for validating sampling distributions.

This module provides tools for statistically testing whether a sampling method
produces samples distributed according to its claimed PDF. This is essential
for validating importance sampling implementations in rendering.

The test works by:
1. Drawing many samples and binning them into a histogram
2. Integrating the PDF over the same bins to get expected frequencies
3. Computing a chi-squared statistic to measure the discrepancy
4. Using the regularized gamma function to compute a p-value

If the p-value is below the significance level, we reject the null hypothesis
that the samples follow the claimed distribution.
"""

import math
import numpy as np
import slangpy as spy
from typing import Optional, Any


def rlgamma(a: float, x: float) -> float:
    """
    Regularized lower incomplete gamma function based on CEPHES.

    Computes P(a, x) = (1/Gamma(a)) * integral from 0 to x of t^(a-1) * e^(-t) dt

    This is used to compute the CDF of the chi-squared distribution, which is
    needed to convert the chi-squared statistic to a p-value.

    Args:
        a: Shape parameter (must be positive).
        x: Upper integration limit (must be non-negative).

    Returns:
        The regularized lower incomplete gamma function value P(a, x).
    """
    eps = 1e-15
    big = 4.503599627370496e15
    biginv = 2.22044604925031308085e-16

    if a < 0 or x < 0:
        raise ValueError("out of range")

    if x == 0:
        return 0

    ax = (a * math.log(x)) - x - math.lgamma(a)

    if ax < -709.78271289338399:
        return 1.0 if a < x else 0.0

    if x <= 1 or x <= a:
        r2 = a
        c2 = 1
        ans2 = 1

        while True:
            r2 = r2 + 1
            c2 = c2 * x / r2
            ans2 += c2

            if not (c2 / ans2 > eps):
                break

        return math.exp(ax) * ans2 / a

    c = 0
    y = 1 - a
    z = x + y + 1
    p3 = 1
    q3 = x
    p2 = x + 1
    q2 = z * x
    ans = p2 / q2

    while True:
        c += 1
        y += 1
        z += 2
        yc = y * c
        p = (p2 * z) - (p3 * yc)
        q = (q2 * z) - (q3 * yc)

        if q != 0:
            nextans = p / q
            error = math.fabs((ans - nextans) / nextans)
            ans = nextans
        else:
            error = 1

        p3 = p2
        p2 = p
        q3 = q2
        q2 = q

        if math.fabs(p) > big:
            p3 *= biginv
            p2 *= biginv
            q3 *= biginv
            q2 *= biginv

        if not (error > eps):
            break

    return 1 - math.exp(ax) * ans


def chi2(obs: np.ndarray, exp: np.ndarray, pool_threshold: float) -> tuple[float, int, int, int]:
    """
    Compute the chi-squared statistic with cell pooling for low expected frequencies.

    This function pools cells with low expected frequencies to ensure statistical
    validity of the chi-squared test.

    Args:
        obs: Array of observed frequencies (will be flattened).
        exp: Array of expected frequencies (will be flattened).
        pool_threshold: Minimum expected frequency threshold. Cells below this
            threshold are pooled together.

    Returns:
        A tuple containing:
        - chsq: The chi-squared statistic.
        - dof: Degrees of freedom (number of cells used minus 1).
        - n_pooled_in: Number of cells that were pooled.
        - n_pooled_out: Number of pooled groups created.
    """
    obs = obs.ravel()
    exp = exp.ravel()
    assert len(obs) == len(exp)
    n = len(obs)

    chsq = 0.0
    pooled_obs = 0.0
    pooled_exp = 0.0
    dof = 0
    n_pooled_in = 0
    n_pooled_out = 0

    for i in range(n):
        if exp[i] == 0 and obs[i] == 0:
            continue

        if exp[i] < pool_threshold:
            pooled_obs += obs[i]
            pooled_exp += exp[i]
            n_pooled_in += 1

            if pooled_exp > pool_threshold:
                diff = pooled_obs - pooled_exp
                chsq += (diff * diff) / pooled_exp
                pooled_obs = 0.0
                pooled_exp = 0.0
                n_pooled_out += 1
                dof += 1
        else:
            diff = obs[i] - exp[i]
            chsq += (diff * diff) / exp[i]
            dof += 1

    return chsq, dof - 1, n_pooled_in, n_pooled_out


class ChiSquareTest:
    """
    Chi-square goodness-of-fit test runner for GPU-based sampling distributions.

    This class orchestrates the chi-square test by:
    1. Running GPU kernels to build a histogram from samples
    2. Running GPU kernels to integrate the PDF over histogram bins
    3. Computing the chi-squared statistic with cell pooling
    4. Determining whether to accept or reject the null hypothesis

    The test uses a 2D grid of bins in the parameter domain. For domains that
    map to higher dimensions (e.g., spherical), the domain interface handles
    the forward/backward mapping.

    Example usage:
        test = ChiSquareTest(device, module, "MyDistribution", spy.uint2(32, 32))
        passed = test.run()
    """

    # Number of samples each GPU thread processes (for efficiency)
    SAMPLES_PER_THREAD: int = 1024

    def __init__(
        self,
        device: spy.Device,
        module: spy.Module,
        target: str,
        res: spy.uint2,
        ires: int = 4,
        sample_count: int = 1000000,
        target_bindings: Optional[dict[str, Any]] = None,
        global_bindings: Optional[dict[str, Any]] = None,
    ):
        """
        Initialize a chi-square test.

        Args:
            device: Device for running kernels.
            module: Module containing the target distribution.
            target: Name of the Slang type implementing ITarget interface.
            res: Resolution of the 2D histogram grid (width, height).
            ires: Integration resolution per bin for PDF tabulation.
            sample_count: Total number of samples to draw for the histogram.
            target_bindings: Optional dictionary of bindings to pass to the target type.
            global_bindings: Optional module global resource bindings.
        """
        super().__init__()

        if ires < 2:
            raise ValueError(f"ires must be >= 2 for the trapezoidal rule, got {ires}")

        self.device = device
        self.module = module
        self.target = target
        self.target_bindings = target_bindings if target_bindings else {}
        self.global_bindings = global_bindings
        self.sample_count = sample_count
        self.res = res
        self.ires = ires
        self.shape = (self.res.y, self.res.x)

        self.histogram: Optional[np.ndarray] = None
        self.pdf: Optional[np.ndarray] = None

        self.messages: str = f"{self.target}:\n"
        self.fail = False

    def tabulate_histogram(self):
        """
        Build a histogram by drawing samples from the target distribution.
        """
        # Create histogram tensor.
        histogram = spy.Tensor.zeros(self.device, (self.res.y, self.res.x), dtype="uint")
        # Create outside counter tensor.
        outside_count = spy.Tensor.zeros(self.device, (1,), dtype="uint")
        # Populate histogram.
        tabulate_histogram = self.module[f"chi2::tabulate_histogram<{self.target}>"]
        if self.global_bindings is not None:
            tabulate_histogram = tabulate_histogram.as_func().set(self.global_bindings)
        thread_count = (
            self.sample_count + ChiSquareTest.SAMPLES_PER_THREAD - 1
        ) // ChiSquareTest.SAMPLES_PER_THREAD
        tabulate_histogram(
            tid=spy.grid((thread_count,)),
            sample_count=self.sample_count,
            samples_per_thread=ChiSquareTest.SAMPLES_PER_THREAD,
            seed=1234,
            target={"_type": self.target, **self.target_bindings},
            histogram=histogram,
            outside_count=outside_count,
        )

        self.histogram = histogram.to_numpy()

        # Sanity checks to detect obvious sampling bugs.
        outside_count = outside_count.to_numpy()[0]
        if outside_count > 0:
            self._log(f"Encountered {outside_count} out-of-bounds samples")
            self.fail = True

        histogram_min = np.min(self.histogram)
        if histogram_min < 0:
            self._log(f"Encountered a cell with negative sample weights: {histogram_min}")
            self.fail = True

        # Check normalization (should sum to ~sample_count for a valid PDF)
        self.histogram_sum = np.sum(self.histogram) / self.sample_count
        if self.histogram_sum > 1.1:
            self._log(f"Sample weights add up to a value greater than 1.0: {self.histogram_sum}")
            self.fail = True

    def tabulate_pdf(self):
        """
        Integrate the target PDF over each histogram bin.
        """
        pdf = spy.Tensor.zeros(self.device, (self.res.y, self.res.x), dtype=float)
        tabulate_pdf = self.module[f"chi2::tabulate_pdf<{self.target}>"]
        if self.global_bindings is not None:
            tabulate_pdf = tabulate_pdf.as_func().set(self.global_bindings)
        tabulate_pdf(
            tid=spy.grid((self.res.y, self.res.x)),
            res=self.res,
            ires=self.ires,
            sample_count=self.sample_count,
            target={"_type": self.target, **self.target_bindings},
            pdf=pdf,
        )

        self.pdf = pdf.to_numpy()

        # Sanity checks to detect obvious PDF bugs.
        pdf_min = np.min(self.pdf)
        if pdf_min < 0:
            self._log(f"Failure: Encountered a cell with a negative PDF value: {pdf_min}")
            self.fail = True

        self.pdf_sum = np.sum(self.pdf) / self.sample_count
        if self.pdf_sum > 1.1:
            self._log(f"Failure: PDF integrates to a value greater than 1.0: {self.pdf_sum}")
            self.fail = True

    def run(
        self,
        significance_level: float = 0.01,
        test_count: int = 1,
        quiet: bool = True,
    ) -> bool:
        """
        Run the chi-square goodness-of-fit test.

        Args:
            significance_level: Threshold for rejecting the null hypothesis.
                Lower values require stronger evidence to reject (default: 0.01 = 1%).
            test_count: Number of independent tests being run. Used for Sidak
                correction to control family-wise error rate.
            quiet: If True, suppress console output and data file generation if test passes.

        Returns:
            True if the null hypothesis is accepted (samples match PDF),
            False if rejected or if sanity checks failed.
        """
        # Lazily compute histogram and PDF if not already done
        if self.histogram is None:
            self.tabulate_histogram()
        if self.pdf is None:
            self.tabulate_pdf()

        assert self.histogram is not None
        assert self.pdf is not None

        # Sort entries by expected frequency (increasing) to enable cell pooling.
        # Pooling starts with low-frequency cells to ensure each pooled group
        # has sufficient expected count for the chi-square approximation to be valid.
        histogram = self.histogram.ravel()
        pdf = self.pdf.ravel()
        index = np.array([i[0] for i in sorted(enumerate(pdf), key=lambda x: x[1])])
        histogram = histogram[index]
        pdf = pdf[index]

        # Compute chi^2 statistic with cell pooling (threshold=5 for expected frequency)
        chi2val, dof, pooled_in, pooled_out = chi2(histogram, pdf, 5)

        if dof < 1:
            self._log("Failure: The number of degrees of freedom is too low!")
            self.fail = True

        if np.any((pdf == 0) & (histogram != 0)):
            self._log(
                "Failure: Found samples in a cell with expected "
                "frequency 0. Rejecting the null hypothesis!"
            )
            self.fail = True

        if pooled_in > 0:
            self._log(
                f"Pooled {pooled_in} low-valued cells into {pooled_out} cells to ensure sufficiently high expected cell frequencies"
            )

        self._log(f"Histogram sum = {self.histogram_sum}, PDF sum = {self.pdf_sum}")

        self._log(f"Chi^2 statistic = {chi2val} (d.o.f = {dof})")

        # Compute p-value: probability of observing a test statistic at least
        # as extreme as chi2val, assuming the null hypothesis is true.
        # The chi-square distribution CDF is a regularized gamma function.
        if not self.fail:
            self.p_value = 1 - rlgamma(dof / 2, chi2val / 2)
        else:
            self.p_value = 0.0

        # Apply Sidak correction for multiple hypothesis testing.
        significance_level = 1.0 - (1.0 - significance_level) ** (1.0 / test_count)

        result = False

        if self.fail:
            self._log(
                "Not running the test for reasons listed above. Target "
                'density and histogram were written to "chi2_data.py'
            )
        elif self.p_value < significance_level or not math.isfinite(self.p_value):
            self._log(
                f'***** Rejected ***** the null hypothesis (p-value = {self.p_value}, significance level = {significance_level}). Target density and histogram were written to "chi2_data.py".'
            )
        else:
            self._log(
                f"Accepted the null hypothesis (p-value = {self.p_value}, significance level = {significance_level})"
            )
            result = True

        if not result or not quiet:
            print(self.messages)
            if not result:
                self._dump_tables()

        return result

    def _dump_tables(self):
        """
        Write histogram and PDF data to a Python file for debugging.

        The generated file can be run directly to visualize the PDF,
        histogram, and their difference using matplotlib.
        """
        assert self.histogram is not None
        assert self.pdf is not None

        def table_to_str(table: np.ndarray) -> str:
            def fmt(v: float) -> str:
                if np.isnan(v):
                    return "float('nan')"
                if np.isposinf(v):
                    return "float('inf')"
                if np.isneginf(v):
                    return "float('-inf')"
                return repr(float(v))

            return (
                "["
                + ", ".join(
                    "[" + ", ".join(fmt(table[y][x]) for x in range(self.res.x)) + "]"
                    for y in range(self.res.y)
                )
                + "]"
            )

        with open("chi2_data.py", "w") as f:
            f.write(f"pdf={table_to_str(self.pdf)}\n")
            f.write(f"histogram={table_to_str(self.histogram)}\n\n")
            f.write("if __name__ == '__main__':\n")
            f.write("    import matplotlib.pyplot as plt\n")
            f.write("    import numpy as np\n\n")
            f.write("    fig, axs = plt.subplots(1,3, figsize=(15, 5))\n")
            f.write("    pdf = np.array(pdf)\n")
            f.write("    histogram = np.array(histogram)\n")
            f.write("    diff=histogram - pdf\n")
            f.write("    absdiff=np.abs(diff).max()\n")
            f.write(
                "    pdf_plot = axs[0].imshow(pdf, vmin=0, aspect='equal', interpolation='nearest')\n"
            )
            f.write(
                "    hist_plot = axs[1].imshow(histogram, vmin=0, aspect='equal', interpolation='nearest')\n"
            )
            f.write(
                "    diff_plot = axs[2].imshow(diff, aspect='equal', vmin=-absdiff, vmax=absdiff, interpolation='nearest', cmap='coolwarm')\n"
            )
            f.write("    axs[0].title.set_text('PDF')\n")
            f.write("    axs[1].title.set_text('Histogram')\n")
            f.write("    axs[2].title.set_text('Difference')\n")
            f.write("    props = dict(fraction=0.046, pad=0.04)\n")
            f.write("    fig.colorbar(pdf_plot, ax=axs[0], **props)\n")
            f.write("    fig.colorbar(hist_plot, ax=axs[1], **props)\n")
            f.write("    fig.colorbar(diff_plot, ax=axs[2], **props)\n")
            f.write("    plt.tight_layout()\n")
            f.write("    plt.show()\n")

    def _log(self, message: str):
        """Append a message to the internal log."""
        self.messages += message + "\n"
