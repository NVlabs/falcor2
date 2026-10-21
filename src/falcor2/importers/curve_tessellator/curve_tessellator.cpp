// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "curve_tessellator.h"
#include "curve_tessellator_internal.h"
#include <sgl/core/logger.h>
#include <limits>

namespace falcor::curve_tessellator {
namespace {

// Preserve existing round geometry: linear refinement is ignored and periodic strands remain open.
constexpr SamplingOptions ROUND_SAMPLING{false, false};

void tessellate_curves_impl(
    const CurveInput& input,
    ImporterVertexStream<float3> positions,
    ImporterVertexStream<float> radii,
    ImporterCurve* curve,
    size_t sample_count = 1
)
{
    const auto strands = prepare_strands(input, ROUND_SAMPLING);
    size_t vertex_count = 0, segment_count = 0;
    for (const auto& strand : strands) {
        vertex_count += strand.samples;
        segment_count += strand.samples ? strand.samples - 1 : 0;
    }
    FALCOR_CHECK(vertex_count <= std::numeric_limits<uint32_t>::max(), "Curve '{}' exceeds index limits", input.name);
    if (curve) {
        curve->resize_vertices(vertex_count, sample_count);
        curve->texcoords.resize(vertex_count);
        curve->indices.reserve(2 * segment_count);
        positions = curve->position_stream();
        radii = curve->radius_stream();
    }
    FALCOR_CHECK(
        positions.size == vertex_count && radii.size == vertex_count,
        "Curve '{}' changes sample layout",
        input.name
    );
    uint32_t start = 0;
    size_t skipped_edges = 0;
    for (size_t strand_index = 0; strand_index < strands.size(); ++strand_index) {
        const auto& strand = strands[strand_index];
        const auto& widths = input.sample.widths;
        if (widths.values.size() > 1) {
            size_t end = 1;
            switch (widths.interpolation) {
            case Interpolation::constant:
                break;
            case Interpolation::uniform:
                end = strand_index + 1;
                break;
            case Interpolation::vertex:
                end = strand.point_offset + strand.count;
                break;
            case Interpolation::varying:
                end = strand.varying_offset + strand.varying_count;
                break;
            }
            FALCOR_CHECK(
                widths.values.size() >= end,
                "Curve '{}' width count does not match its interpolation",
                input.name
            );
        }
        evaluate_samples<false>(
            input,
            strand,
            strand_index,
            ROUND_SAMPLING,
            curve != nullptr,
            [&](size_t i, const Sample& sample)
            {
                positions[start + i] = sample.position;
                radii[start + i] = 0.5f * sample.width;
                if (curve)
                    curve->texcoords[start + i] = sample.uv;
            }
        );
        if (curve) {
            for (uint32_t i = 0; i + 1 < strand.samples; ++i) {
                if (input.type == Type::cubic && input.refinement_level == 0 && input.remove_coincident_edges
                    && positions[start + i] == positions[start + i + 1]) {
                    ++skipped_edges;
                    continue;
                }
                curve->indices.push_back(start + i);
                curve->indices.push_back(start + i + 1);
            }
        }
        start += static_cast<uint32_t>(strand.samples);
    }
    if (skipped_edges)
        sgl::log_warn("Curve '{}' skipped {} coincident-center control-point edges", input.name, skipped_edges);
}
} // namespace

ImporterCurve tessellate_curves(const CurveInput& input, size_t sample_count)
{
    ImporterCurve curve;
    curve.properties = input.properties;
    curve.name = input.name;
    tessellate_curves_impl(input, {}, {}, &curve, sample_count);
    return curve;
}

void tessellate_curves(
    const CurveInput& input,
    ImporterVertexStream<float3> positions,
    ImporterVertexStream<float> radii
)
{
    tessellate_curves_impl(input, positions, radii, nullptr);
}
} // namespace falcor::curve_tessellator
