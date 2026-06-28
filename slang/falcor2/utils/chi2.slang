// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing utils;

/// Chi-square goodness-of-fit test utilities for validating sampling distributions.
///
/// This namespace provides GPU kernels and interfaces for testing whether a sampling
/// method produces samples distributed according to its claimed PDF. The test works by:
/// 1. Drawing many samples and binning them into a 2D histogram
/// 2. Integrating the PDF over the same bins to get expected frequencies
/// 3. Computing a chi-squared statistic (done on CPU in Python)
///
/// To add a new testable distribution, implement the ITarget interface.
/// The domain interface handles mapping between the 2D parameter space used
/// for binning and the actual target domain (e.g., unit sphere, hemisphere).
namespace chi2 {

/// Interface for mapping between a 2D parameter domain and the target domain.
///
/// The chi-square test always uses a 2D histogram grid. This interface defines
/// how to map from the 2D parameter space to the actual domain of the distribution
/// (which could be 1D, 2D, 3D, etc.) and back.
///
/// - `forward()`: Maps a 2D parameter point to the target domain (for PDF evaluation)
/// - `backward()`: Maps a target domain sample back to 2D (for histogram binning)
public interface IDomain
{
    /// Dimensionality of the target domain (1, 2, or 3).
    public static const int Dim;

    /// Whether the domain is discrete (1) or continuous (0).
    /// Discrete domains skip integration and evaluate PDF at bin centers.
    public static const int Discrete;

    /// Returns the bounds of the 2D parameter domain as [min, max].
    public float2[2] get_bounds();

    /// Map a 2D parameter point to the target domain.
    public vector<float, Dim> forward(float2 p);

    /// Map a target domain point back to 2D parameter space.
    public float2 backward(vector<float, Dim> p);
};

// ============================================================================
// Builtin domain implementations
// ============================================================================

/// Identity map on the line (1D continuous domain).
/// Used for testing 1D distributions.
public struct LineDomain : IDomain
{
    public static const int Dim = 1;
    public static const int Discrete = 0;

    public float2[2] bounds;

    public __init(float min = -1.0, float max = 1.0) { bounds = { float2(min, -0.5), float2(max, 0.5) }; }

    public float2[2] get_bounds() { return bounds; }

    public float1 forward(float2 p) { return p.x; }

    public float2 backward(float1 p) { return float2(p, 0.0); }
};

/// Discrete line domain (1D discrete domain).
/// Used for testing discrete 1D distributions.
public struct DiscreteLineDomain : IDomain
{
    public static const int Dim = 1;
    public static const int Discrete = 1;

    public float2[2] bounds;

    public __init(uint min, uint max) { bounds = { float2(min, -0.5), float2(max, 0.5) }; }

    public float2[2] get_bounds() { return bounds; }

    public float1 forward(float2 p) { return p.x; }

    public float2 backward(float1 p) { return float2(p, 0.0); }
};

/// Identity map on the 2D plane (2D continuous domain).
/// Used for testing 2D distributions.
public struct PlanarDomain : IDomain
{
    public static const int Dim = 2;
    public static const int Discrete = 0;

    public float2[2] bounds;

    public __init(float2 min = float2(-1.0), float2 max = float2(1.0)) { bounds = { min, max }; }

    public float2[2] get_bounds() { return bounds; }

    public float2 forward(float2 p) { return p; }

    public float2 backward(float2 p) { return p; }
};

/// Spherical domain mapping from (phi, cos_theta) to unit sphere directions.
/// Used for testing spherical distributions like cosine-weighted hemisphere sampling.
///
/// Parameter space: (phi, cos_theta) where phi in [-pi, pi], cos_theta in [-1, 1]
/// Target space: Unit vectors on S^2
public struct SphericalDomain : IDomain
{
    public static const int Dim = 3;
    public static const int Discrete = 0;

    public float2[2] bounds;

    public __init(float2 min = float2(-float.PI, -1.0), float2 max = float2(float.PI, 1.0)) { bounds = { min, max }; }

    public float2[2] get_bounds() { return bounds; }

    public float3 forward(float2 p)
    {
        float cos_theta = -p.y;
        float sin_theta = safe_sqrt(fma(-cos_theta, cos_theta, 1.0));
        float sin_phi, cos_phi;
        sincos(p.x, sin_phi, cos_phi);
        return float3(cos_phi * sin_theta, sin_phi * sin_theta, cos_theta);
    }

    public float2 backward(float3 p) { return float2(atan2(p.y, p.x), -p.z); }
};

// ============================================================================
// ITarget Interface
// ============================================================================

/// Interface for distributions that can be tested with chi-square.
///
/// To test a distribution, implement this interface with:
/// - A domain type that defines the parameter-to-target mapping
/// - A sample() method that draws samples from the distribution
/// - A pdf() method that evaluates the probability density function
///
/// The test will verify that samples drawn from sample() are distributed
/// according to pdf() by comparing observed vs expected frequencies.
public interface ITarget
{
    /// The domain type that defines the parameter mapping.
    associatedtype Domain : IDomain;

    /// Returns an instance of the domain with appropriate bounds.
    public Domain get_domain();

    /// Draw a random sample from the distribution.
    public bool sample<SG : ISampleGenerator>(inout SG sg, out vector<float, Domain.Dim> p);

    /// Evaluate the probability density at the given point.
    /// For discrete distributions, this is the probability mass function.
    public float pdf(vector<float, Domain.Dim> p);
};

// ============================================================================
// GPU Kernels
// ============================================================================

/// GPU kernel to build a histogram by sampling from the target distribution.
///
/// Each thread draws `samples_per_thread` samples, maps them back to the
/// 2D parameter domain, and atomically increments the appropriate histogram bin.
///
/// @param tid Thread index (1D dispatch)
/// @param sample_count Total number of samples to draw
/// @param samples_per_thread Number of samples each thread processes
/// @param seed Random seed for reproducibility
/// @param target The distribution to sample from
/// @param histogram Output tensor holding the histogram (atomic, 2D)
/// @param outside_count Output tensor for counting samples that fall outside bounds
public void tabulate_histogram<Target : ITarget>(
    uint tid,
    uint sample_count,
    uint samples_per_thread,
    uint seed,
    Target target,
    AtomicTensor<uint, 2> histogram,
    AtomicTensor<uint, 1> outside_count
)
{
    let domain = target.get_domain();
    float2[2] bounds = domain.get_bounds();
    float2 extents = bounds[1] - bounds[0];

    // Initialize per-thread RNG using TEA block cipher for decorrelation.
    uint2 v = block_cipher_tea(seed, tid);
    UniformSampleGenerator sg = UniformSampleGenerator(v.x);

    for (uint i = 0; i < samples_per_thread; ++i) {
        uint sample_index = tid * samples_per_thread + i;

        if (sample_index >= sample_count)
            break;

        // Sample from the distribution.
        vector<float, Target.Domain.Dim> sample_out = 0;
        if (!target.sample(sg, sample_out))
            continue;

        // Map sample back to 2D parameter domain for binning.
        float2 xy = domain.backward(sample_out);

        // Check if sample is within expected bounds.
        float2 eps = extents * 1e-4;
        if (any((xy < bounds[0] - eps) | (xy > bounds[1] + eps)))
            outside_count.add(0, 1);

        // Normalize to [0, 1]^2 for bin index computation.
        xy = clamp((xy - bounds[0]) / extents, 0.0, 1.0);

        // Compute histogram bin index (note: shape is [rows, cols] = [y, x]).
        int2 shape = int2(histogram.shape[1], histogram.shape[0]);
        int2 index = int2(xy * shape);
        index = min(index, shape - 1);

        histogram.add(index, 1);
    }
}

/// GPU kernel to integrate the PDF over each histogram bin.
///
/// For continuous distributions: Uses 2D trapezoidal rule with `ires` samples
/// per dimension to integrate the PDF over each bin.
///
/// For discrete distributions: Simply evaluates the PDF at bin centers.
///
/// @param tid Thread index (2D dispatch, one thread per bin)
/// @param res Histogram resolution (width, height)
/// @param ires Integration resolution per bin (samples per dimension)
/// @param sample_count Total sample count (used to scale PDF to expected frequency)
/// @param target The distribution whose PDF to evaluate
/// @param pdf Output tensor for expected frequencies
public void
tabulate_pdf<Target : ITarget>(uint2 tid, uint2 res, uint ires, uint sample_count, Target target, WTensor<float, 2> pdf)
{
    let domain = target.get_domain();

    if (Target.Domain.Discrete == 0) {
        // Continuous distribution: integrate PDF over the bin using trapezoidal rule.
        float2 lo = domain.get_bounds()[0];
        float2 hi = domain.get_bounds()[1];
        float2 bin_size = (hi - lo) / float2(res);
        float2 step = bin_size / float(ires - 1);
        float2 bin_lo = lo + bin_size * float2(tid);

        float sum = 0.0;
        for (uint j = 0; j < ires; j++) {
            float wy = (j == 0 || j == ires - 1) ? 0.5 : 1.0;
            for (uint i = 0; i < ires; i++) {
                float wx = (i == 0 || i == ires - 1) ? 0.5 : 1.0;
                float2 p = bin_lo + step * float2(i, j);
                let v = domain.forward(p);
                sum += wx * wy * target.pdf(v);
            }
        }

        sum *= step.x * step.y * sample_count;

        pdf.store(tid, sum);
    } else {
        // Discrete distribution: evaluate PDF at bin center.
        float2 p = float2(tid) + 0.5;
        let v = domain.forward(p);
        pdf.store(tid, target.pdf(v) * sample_count);
    }
}

} // namespace chi2
