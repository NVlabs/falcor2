// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "usd_importer_curve_tesselator.h"
#include "usd_importer_utils.h"

#include "falcor2/core/types.h"

BEGIN_DISABLE_USD_WARNINGS
#include <pxr/base/gf/vec3f.h>
#include <pxr/base/vt/array.h>
END_DISABLE_USD_WARNINGS

#include <cstdint>

namespace falcor {
namespace usd_importer {

namespace {

/// Basis matrices for cubic curve evaluation.
/// P(t) = [t^3, t^2, t, 1] * M * [P0, P1, P2, P3]^T
// clang-format off
static constexpr float BEZIER_BASIS[4][4] = {
    {-1.f,  3.f, -3.f, 1.f},
    { 3.f, -6.f,  3.f, 0.f},
    {-3.f,  3.f,  0.f, 0.f},
    { 1.f,  0.f,  0.f, 0.f},
};

static constexpr float BSPLINE_BASIS[4][4] = {
    {-1.f / 6.f,  3.f / 6.f, -3.f / 6.f, 1.f / 6.f},
    { 3.f / 6.f, -6.f / 6.f,  3.f / 6.f, 0.f},
    {-3.f / 6.f,  0.f,        3.f / 6.f,  0.f},
    { 1.f / 6.f,  4.f / 6.f,  1.f / 6.f,  0.f},
};

static constexpr float CATMULL_ROM_BASIS[4][4] = {
    {-0.5f,  1.5f, -1.5f,  0.5f},
    { 1.0f, -2.5f,  2.0f, -0.5f},
    {-0.5f,  0.0f,  0.5f,  0.0f},
    { 0.0f,  1.0f,  0.0f,  0.0f},
};
// clang-format on

/// Evaluate a cubic curve at parameter t using a 4x4 basis matrix and 4 control values.
template<typename T>
static T evaluate_cubic(const float basis[4][4], const T& p0, const T& p1, const T& p2, const T& p3, float t)
{
    float t2 = t * t;
    float t3 = t2 * t;
    float c0 = basis[0][0] * t3 + basis[1][0] * t2 + basis[2][0] * t + basis[3][0];
    float c1 = basis[0][1] * t3 + basis[1][1] * t2 + basis[2][1] * t + basis[3][1];
    float c2 = basis[0][2] * t3 + basis[1][2] * t2 + basis[2][2] * t + basis[3][2];
    float c3 = basis[0][3] * t3 + basis[1][3] * t2 + basis[2][3] * t + basis[3][3];
    return p0 * c0 + p1 * c1 + p2 * c2 + p3 * c3;
}

/// Get per-vertex radius from USD widths array, handling varying interpolation modes.
static float get_vertex_radius(const pxr::VtArray<float>& usd_widths, size_t index, float default_radius)
{
    if (usd_widths.empty())
        return default_radius;
    if (usd_widths.size() == 1)
        return usd_widths[0] * 0.5f;
    if (index < usd_widths.size())
        return usd_widths[index] * 0.5f;
    return default_radius;
}

/// Tessellate cubic curves into linear segments and append to ImporterCurve.
static void tessellate_cubic_curves(
    ImporterCurve& curve,
    const pxr::VtArray<pxr::GfVec3f>& usd_points,
    const pxr::VtArray<float>& usd_widths,
    const pxr::VtArray<int>& usd_vertex_counts,
    const float basis[4][4],
    int span_stride,
    int segments_per_span,
    float default_radius
)
{
    uint32_t vertex_offset = 0;

    for (int curve_idx = 0; curve_idx < static_cast<int>(usd_vertex_counts.size()); ++curve_idx) {
        int vertex_count = usd_vertex_counts[curve_idx];
        // Number of cubic spans in this curve.
        // Bezier (stride=3): spans at CVs 0,3,6,... => (N-4)/3 + 1 = (N-1)/3
        // B-spline/CatmullRom (stride=1): spans at CVs 0,1,2,... => (N-4)/1 + 1 = N-3
        int span_count = (vertex_count - 4) / span_stride + 1;

        if (span_count <= 0) {
            vertex_offset += vertex_count;
            continue;
        }

        // For each span, evaluate positions and radii at uniform parameter values.
        uint32_t first_tessellated_vertex = static_cast<uint32_t>(curve.positions.size());

        for (int span = 0; span < span_count; ++span) {
            // Control point indices for this span.
            int i0 = vertex_offset + span * span_stride;
            int i1 = i0 + 1;
            int i2 = i0 + 2;
            int i3 = i0 + 3;

            float3 p0 = to_falcor(usd_points[i0]);
            float3 p1 = to_falcor(usd_points[i1]);
            float3 p2 = to_falcor(usd_points[i2]);
            float3 p3 = to_falcor(usd_points[i3]);

            float r0 = get_vertex_radius(usd_widths, i0, default_radius);
            float r1 = get_vertex_radius(usd_widths, i1, default_radius);
            float r2 = get_vertex_radius(usd_widths, i2, default_radius);
            float r3 = get_vertex_radius(usd_widths, i3, default_radius);

            // Evaluate at uniform t values.
            // First span starts at t=0, subsequent spans skip t=0 to avoid duplicate vertices.
            int t_start = (span == 0) ? 0 : 1;
            for (int seg = t_start; seg <= segments_per_span; ++seg) {
                float t = static_cast<float>(seg) / static_cast<float>(segments_per_span);
                curve.positions.push_back(evaluate_cubic(basis, p0, p1, p2, p3, t));
                curve.radii.push_back(evaluate_cubic(basis, r0, r1, r2, r3, t));
            }
        }

        // Generate line segment indices for this tessellated curve.
        uint32_t tessellated_vertex_count = static_cast<uint32_t>(curve.positions.size()) - first_tessellated_vertex;
        for (uint32_t j = 0; j < tessellated_vertex_count - 1; ++j) {
            curve.indices.push_back(first_tessellated_vertex + j);
            curve.indices.push_back(first_tessellated_vertex + j + 1);
        }

        vertex_offset += vertex_count;
    }
}

} // namespace

ImporterCurve tessellate_curves(const TesselatorInputCurve& input)
{
    ImporterCurve curve;
    curve.name = input.name;

    const auto& usd_points = input.points;
    const auto& usd_widths = input.widths;
    const auto& usd_vertex_counts = input.vertex_counts;

    if (input.type == TesselatorInputCurve::Type::linear) {
        // Copy positions.
        curve.positions.resize(usd_points.size());
        for (size_t i = 0; i < usd_points.size(); ++i) {
            curve.positions[i] = to_falcor(usd_points[i]);
        }

        // Copy radii (width / 2).
        curve.radii.resize(usd_points.size());
        for (size_t i = 0; i < usd_points.size(); ++i) {
            curve.radii[i] = get_vertex_radius(usd_widths, i, input.default_radius);
        }

        // Generate line segment indices for linear curves.
        uint32_t vertex_offset = 0;
        for (int vertex_count : usd_vertex_counts) {
            for (int j = 0; j < vertex_count - 1; ++j) {
                curve.indices.push_back(vertex_offset + j);
                curve.indices.push_back(vertex_offset + j + 1);
            }
            vertex_offset += vertex_count;
        }
    } else {
        // Cubic curves: select basis matrix and stride.
        const float (*basis)[4] = nullptr;
        int span_stride = 0;

        switch (input.basis) {
        case TesselatorInputCurve::Basis::bezier:
            basis = BEZIER_BASIS;
            span_stride = 3;
            break;
        case TesselatorInputCurve::Basis::bspline:
            basis = BSPLINE_BASIS;
            span_stride = 1;
            break;
        case TesselatorInputCurve::Basis::catmull_rom:
            basis = CATMULL_ROM_BASIS;
            span_stride = 1;
            break;
        }

        tessellate_cubic_curves(
            curve,
            usd_points,
            usd_widths,
            usd_vertex_counts,
            basis,
            span_stride,
            input.segments_per_span,
            input.default_radius
        );
    }

    return curve;
}

} // namespace usd_importer
} // namespace falcor
