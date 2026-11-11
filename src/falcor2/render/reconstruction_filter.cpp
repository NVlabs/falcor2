// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "reconstruction_filter.h"

#include "falcor2/core/error.h"

#include <sgl/core/signature_buffer.h>
#include <sgl/device/cursor_utils.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>

namespace falcor {

namespace {

uint64_t next_reconstruction_filter_generation()
{
    static std::atomic<uint64_t> id{0};
    return ++id;
}

float checked_density(double density)
{
    FALCOR_CHECK(
        density >= double(std::numeric_limits<float>::min()) && density <= double(std::numeric_limits<float>::max()),
        "Filter parameters produce unrepresentable FP32 density."
    );
    return float(density);
}

} // namespace

ReconstructionFilter::ReconstructionFilter()
    : m_generation(next_reconstruction_filter_generation())
{
}

void ReconstructionFilter::notify_changed()
{
    m_generation = next_reconstruction_filter_generation();
}

void ReconstructionFilter::write_slangpy_signature(sgl::SignatureBuffer& signature, const ReconstructionFilter* value)
{
    signature.add("falcor2.ReconstructionFilter:");
    signature.add(value ? value->slang_type_name() : "<null>");
}

BoxReconstructionFilter::BoxReconstructionFilter(float width)
    : width(width)
{
}

void BoxReconstructionFilter::write_to_cursor(sgl::ShaderCursor cursor) const
{
    FALCOR_CHECK(std::isfinite(width) && width >= 0.f, "width must be finite and nonnegative.");
    if (width > 0.f)
        checked_density(1.0 / (double(width) * double(width)));
    cursor["width"] = width;
}

GaussianReconstructionFilter::GaussianReconstructionFilter(float sigma, float radius)
    : sigma(sigma)
    , radius(radius)
{
}

void GaussianReconstructionFilter::write_to_cursor(sgl::ShaderCursor cursor) const
{
    FALCOR_CHECK(std::isfinite(sigma) && sigma > 0.f, "sigma must be finite and positive.");
    FALCOR_CHECK(std::isfinite(radius) && radius > 0.f, "radius must be finite and positive.");
    const double normalized_radius = double(radius) / double(sigma);
    const double a = 0.5 * normalized_radius * normalized_radius;
    const double peak = -std::expm1(-a);
    const double min_float = std::numeric_limits<float>::min();
    // Leave FP32 headroom for evaluating the shifted kernel near its support boundary.
    FALCOR_CHECK(peak >= min_float * 16777216.0, "radius is too small relative to sigma for FP32 Gaussian evaluation.");
    FALCOR_CHECK(
        std::min(double(sigma), double(radius)) >= min_float * 8192.0,
        "sigma and radius produce subnormal sampling intermediates."
    );
    FALCOR_CHECK(
        double(radius) <= double(std::numeric_limits<float>::max()) / 4.0,
        "radius produces unrepresentable filter support."
    );
    const double cdf_limit = std::erf(normalized_radius / std::sqrt(2.0));
    double integral;
    if (a < 0.5) {
        // Integrate exp(-a*y*y)-exp(-a) on y in [-1,1] without cancellation.
        // Its half-integral divided by a is sum((-a)^(n-1)/n! * 2n/(2n+1)).
        double term = 1.0;
        double sum = 0.0;
        for (int n = 1; n <= 12; ++n) {
            sum += term * (2.0 * n) / (2.0 * n + 1.0);
            term *= -a / double(n + 1);
        }
        integral = 2.0 * double(radius) * a * sum;
    } else {
        integral = std::sqrt(2.0 * 3.141592653589793) * double(sigma) * cdf_limit - 2.0 * double(radius) * std::exp(-a);
    }
    // Normalize the peak-scaled 1D kernel. Keeping this scale finite also handles
    // tight cutoffs, where the shifted Gaussian tends to a parabolic kernel.
    const double normalization = peak / integral;
    checked_density(normalization * normalization);
    cursor["sigma"] = sigma;
    cursor["radius"] = radius;
    cursor["cdf_limit"] = float(cdf_limit);
    cursor["peak"] = float(peak);
    cursor["normalization"] = float(normalization);
}

FALCOR_STATIC_ONCE(reflection::register_type<ReconstructionFilter>());
FALCOR_STATIC_ONCE(sgl::cursor_utils::register_cursor_writer<ReconstructionFilter>());
FALCOR_STATIC_ONCE(reflection::register_type<BoxReconstructionFilter>());
FALCOR_STATIC_ONCE(reflection::register_type<GaussianReconstructionFilter>());

} // namespace falcor
