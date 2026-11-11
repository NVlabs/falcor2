// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "curve_tessellator.h"
#include <algorithm>
#include <array>

namespace falcor::curve_tessellator {

/// Preserve the round-curve importer's legacy sampling without changing ribbon tessellation.
struct SamplingOptions {
    bool refine_linear = true;
    bool close_periodic = true;
};

struct Strand {
    size_t point_offset;
    size_t varying_offset;
    int count;
    int spans;
    int varying_count;
    size_t samples;
};


/// Basis matrices for cubic curve evaluation.
/// P(t) = [t^3, t^2, t, 1] * M * [P0, P1, P2, P3]^T
// clang-format off
inline constexpr float BEZIER_BASIS[4][4] = {
    {-1.f,  3.f, -3.f, 1.f},
    { 3.f, -6.f,  3.f, 0.f},
    {-3.f,  3.f,  0.f, 0.f},
    { 1.f,  0.f,  0.f, 0.f},
};

inline constexpr float BSPLINE_BASIS[4][4] = {
    {-1.f / 6.f,  3.f / 6.f, -3.f / 6.f, 1.f / 6.f},
    { 3.f / 6.f, -6.f / 6.f,  3.f / 6.f, 0.f},
    {-3.f / 6.f,  0.f,        3.f / 6.f,  0.f},
    { 1.f / 6.f,  4.f / 6.f,  1.f / 6.f,  0.f},
};

inline constexpr float CATMULL_ROM_BASIS[4][4] = {
    {-0.5f,  1.5f, -1.5f,  0.5f},
    { 1.0f, -2.5f,  2.0f, -0.5f},
    {-0.5f,  0.0f,  0.5f,  0.0f},
    { 0.0f,  1.0f,  0.0f,  0.0f},
};
// clang-format on

/// Count cubic spans, including reflected endpoint spans for pinned splines.
/// @param vertex_count Number of authored control points in the strand.
/// @param span_stride Control-point advance: 3 for Bezier, 1 for B-spline/Catmull-Rom.
/// @param pinned Whether reflected endpoint controls are required for this spline.
/// @param periodic Whether controls wrap around the strand instead of using open endpoints.
inline int get_cubic_span_count(int vertex_count, int span_stride, bool pinned, bool periodic = false)
{
    if (periodic)
        return vertex_count / span_stride;
    if (pinned)
        return vertex_count < 2 ? 0 : vertex_count - 1;
    // Number of cubic spans in this curve.
    // Bezier (stride=3): spans at CVs 0,3,6,... => (N-4)/3 + 1 = (N-1)/3
    // B-spline/CatmullRom (stride=1): spans at CVs 0,1,2,... => (N-4)/1 + 1 = N-3
    return vertex_count < 4 ? 0 : (vertex_count - 4) / span_stride + 1;
}

/// Read a cubic control, reflecting the neighboring authored value at either pinned endpoint.
/// Open curves request authored indices or the phantom indices -1 and values.size(); periodic controls wrap.
/// Vertex attributes use the same reflection as positions; varying attributes never use this helper.
template<typename T>
T get_cubic_control(std::span<const T> values, int index, bool periodic = false)
{
    const int count = static_cast<int>(values.size());
    if (periodic)
        return values[(index % count + count) % count];
    FALCOR_ASSERT(count > 0 && index >= -1 && index <= count);
    FALCOR_ASSERT((index >= 0 && index < count) || count >= 2);
    if (index == -1)
        return values[0] * 2.f - values[1];
    if (index == count)
        return values[count - 1] * 2.f - values[count - 2];
    return values[index];
}

/// Evaluate a fractional boundary index; periodic values wrap, open values clamp.
/// Callers provide a nonempty strand and a nonnegative parameter.
template<typename T>
T sample_linear(std::span<const T> values, float parameter, bool periodic)
{
    const int count = static_cast<int>(values.size());
    FALCOR_ASSERT(count > 0 && parameter >= 0.f);
    const int first = static_cast<int>(parameter);
    const int a = periodic ? first % count : std::min(first, count - 1);
    const int b = periodic ? (first + 1) % count : std::min(first + 1, count - 1);
    const float t = parameter - first;
    return values[a] * (1.f - t) + values[b] * t;
}

/// The same polynomial weights drive positions and vertex attributes in both output representations.
inline std::array<float, 4> cubic_weights(const float basis[4][4], float t, bool derivative = false)
{
    std::array<float, 4> result;
    const float t2 = t * t;
    const float t3 = t2 * t;
    for (size_t i = 0; i < result.size(); ++i)
        result[i] = derivative ? 3.f * basis[0][i] * t2 + 2.f * basis[1][i] * t + basis[2][i]
                               : basis[0][i] * t3 + basis[1][i] * t2 + basis[2][i] * t + basis[3][i];
    return result;
}

/// Count authored spans, including pinned or periodic endpoint semantics.
inline int span_count(const CurveInput& input, int count, bool periodic)
{
    if (input.type == Type::linear)
        return std::max(0, count - (periodic ? 0 : 1));
    const int stride = input.basis == Basis::bezier ? 3 : 1;
    return get_cubic_span_count(count, stride, input.wrap == Wrap::pinned && stride == 1, periodic);
}

/// Build strand ranges once per sample, before either output representation allocates or writes vertices.
inline std::vector<Strand> prepare_strands(const CurveInput& input, SamplingOptions options = {})
{
    detail::validate_refinement_level(input.refinement_level);
    FALCOR_CHECK(
        input.varying_counts.empty() || input.varying_counts.size() == input.vertex_counts.size(),
        "Curve '{}' has invalid varying counts",
        input.name
    );
    const bool periodic = input.wrap == Wrap::periodic;
    const bool closed = periodic && options.close_periodic;
    const bool cage = input.type == Type::linear || input.refinement_level == 0;
    const int steps = input.type == Type::linear && !options.refine_linear ? 1 : input.refinement_level + 1;
    std::vector<Strand> strands;
    size_t point_offset = 0, varying_offset = 0;
    for (size_t i = 0; i < input.vertex_counts.size(); ++i) {
        const int count = input.vertex_counts[i];
        FALCOR_CHECK(count >= 0, "Curve '{}' has negative vertex count", input.name);
        const int spans = span_count(input, count, closed);
        const int varying_count = input.varying_counts.empty() ? span_count(input, count, periodic) + (periodic ? 0 : 1)
                                                               : input.varying_counts[i];
        FALCOR_CHECK(varying_count >= 0, "Curve '{}' has negative varying count", input.name);
        const int intervals = cage ? std::max(0, count - (closed ? 0 : 1)) : spans;
        const size_t samples = (cage ? count > 0 : spans > 0) ? size_t(intervals) * steps + 1 : 0;
        strands.push_back({point_offset, varying_offset, count, spans, varying_count, samples});
        point_offset += count;
        varying_offset += varying_count;
    }
    FALCOR_CHECK(
        point_offset == input.sample.points.size(),
        "Curve '{}' point count does not match strands",
        input.name
    );
    return strands;
}

struct Evaluation {
    std::array<int, 4> indices;
    std::array<float, 4> weights;
    std::array<float, 4> derivatives;
    float varying_parameter;
};

inline Evaluation schedule(const CurveInput& input, const Strand& strand, size_t sample, SamplingOptions options)
{
    const bool cage = input.type == Type::linear || input.refinement_level == 0;
    const bool periodic = input.wrap == Wrap::periodic;
    const bool closed = periodic && options.close_periodic;
    const int intervals = cage ? strand.count - (closed ? 0 : 1) : strand.spans;
    const int steps = input.type == Type::linear && !options.refine_linear ? 1 : input.refinement_level + 1;
    Evaluation e{};
    if (strand.count == 1) {
        e.weights[0] = 1.f;
        return e;
    }
    const int span = std::min(int(sample / steps), intervals - 1);
    const float t = float(sample - size_t(span) * steps) / steps;
    e.varying_parameter = cage
        ? (span + t) * float(strand.varying_count - (periodic ? 0 : 1)) / (strand.count - (periodic ? 0 : 1))
        : span + t;
    if (cage) {
        e.indices = {span, span + 1, span, span};
        e.weights = {1.f - t, t, 0.f, 0.f};
        e.derivatives = {-1.f, 1.f, 0.f, 0.f};
    } else {
        const int first = input.basis == Basis::bezier ? span * 3 : span - int(input.wrap == Wrap::pinned);
        e.indices = {first, first + 1, first + 2, first + 3};
        const auto basis = input.basis == Basis::bezier ? BEZIER_BASIS
            : input.basis == Basis::bspline             ? BSPLINE_BASIS
                                                        : CATMULL_ROM_BASIS;
        e.weights = cubic_weights(basis, t);
        e.derivatives = cubic_weights(basis, t, true);
    }
    return e;
}

template<typename T>
T evaluate(std::span<const T> values, const Evaluation& e, bool periodic, bool derivative = false)
{
    T value(0.f);
    for (size_t k = 0; k < 4; ++k)
        value += get_cubic_control(values, e.indices[k], periodic) * (derivative ? e.derivatives[k] : e.weights[k]);
    return value;
}

template<typename T>
T attribute(
    const Attribute<T>& attr,
    const Strand& strand,
    size_t strand_index,
    const Evaluation& e,
    bool periodic,
    bool closed,
    T fallback
)
{
    if (attr.values.empty())
        return fallback;
    // Round curves historically permit singleton widths regardless of interpolation.
    // Ribbon validation still enforces the authored cardinality before reaching this sampler.
    if (attr.values.size() == 1 || attr.interpolation == Interpolation::constant)
        return attr.values[0];
    if (attr.interpolation == Interpolation::uniform)
        return attr.values[strand_index];
    if (attr.interpolation == Interpolation::vertex)
        return evaluate(attr.values.subspan(strand.point_offset, strand.count), e, closed);
    return sample_linear(
        attr.values.subspan(strand.varying_offset, strand.varying_count),
        e.varying_parameter,
        periodic
    );
}

struct Sample {
    float3 position;
    float3 tangent;
    float3 normal;
    float width;
    float2 uv;
};

/// Emit samples in strand order without allocating an intermediate curve or mesh.
/// The oriented specialization also evaluates derivatives/normals for the ribbon's frame and joint pass.
/// Static attributes are skipped when filling later deformation streams.
template<bool oriented, typename Emit>
void evaluate_samples(
    const CurveInput& input,
    const Strand& strand,
    size_t strand_index,
    SamplingOptions options,
    bool static_attributes,
    Emit&& emit
)
{
    const bool periodic = input.wrap == Wrap::periodic;
    const bool closed = periodic && options.close_periodic;
    const auto points = input.sample.points.subspan(strand.point_offset, strand.count);
    for (size_t i = 0; i < strand.samples; ++i) {
        const auto e = schedule(input, strand, i, options);
        Sample s{};
        s.position = evaluate(points, e, closed);
        if constexpr (oriented) {
            s.tangent = evaluate(points, e, closed, true);
            s.normal = attribute(input.sample.normals, strand, strand_index, e, periodic, closed, float3(0.f));
        }
        s.width = attribute(input.sample.widths, strand, strand_index, e, periodic, closed, 2.f * input.default_radius);
        if (static_attributes) {
            const float u = strand.samples > 1 ? float(i) / (strand.samples - 1) : 0.f;
            s.uv = attribute(input.texcoords, strand, strand_index, e, periodic, closed, float2(u, 0.5f));
        }
        emit(i, s);
    }
}

} // namespace falcor::curve_tessellator
