// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "curve_tessellator.h"
#include "curve_tessellator_internal.h"
#include "falcor2/core/error.h"
#include "sgl/core/logger.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <utility>

namespace falcor::curve_tessellator {
namespace {
using namespace sgl::math;

bool finite(float v)
{
    return std::isfinite(v);
}
bool finite(float2 v)
{
    return finite(v.x) && finite(v.y);
}
bool finite(float3 v)
{
    return finite(v.x) && finite(v.y) && finite(v.z);
}

float3 unit(float3 v)
{
    const float scale = std::max({std::abs(v.x), std::abs(v.y), std::abs(v.z)});
    if (!(scale > 0.f) || !finite(scale))
        return float3(0.f);
    v /= scale;
    return v / std::sqrt(dot(v, v));
}

void validate_strand_topology(const CurveInput& input, int count)
{
    const bool periodic = input.wrap == Wrap::periodic;
    FALCOR_CHECK(count >= 2, "Ribbon '{}' requires at least two points per strand", input.name);
    // Pinning changes only B-spline/Catmull-Rom endpoints; linear and Bezier stay nonperiodic.
    if (input.type == Type::linear)
        return;
    if (input.basis == Basis::bezier) {
        FALCOR_CHECK(
            count >= (periodic ? 3 : 4) && (count - (periodic ? 0 : 1)) % 3 == 0,
            "Ribbon '{}' has invalid Bezier topology",
            input.name
        );
        return;
    }
    if (input.wrap == Wrap::pinned)
        return;
    FALCOR_CHECK(count >= (periodic ? 3 : 4), "Ribbon '{}' has too few spline points", input.name);
}

template<typename T>
void validate_attribute(
    const CurveInput& input,
    const Attribute<T>& attr,
    size_t point_count,
    size_t varying_count,
    const char* name,
    bool required = false
)
{
    if (attr.values.empty()) {
        FALCOR_CHECK(!required, "Ribbon '{}' requires {}", input.name, name);
        return;
    }
    size_t expected = 0;
    switch (attr.interpolation) {
    case Interpolation::constant:
        expected = 1;
        break;
    case Interpolation::uniform:
        expected = input.vertex_counts.size();
        break;
    case Interpolation::vertex:
        expected = point_count;
        break;
    case Interpolation::varying:
        expected = varying_count;
        break;
    }
    FALCOR_CHECK(attr.values.size() == expected, "Ribbon '{}' {} count does not match interpolation", input.name, name);
    for (const T& value : attr.values)
        FALCOR_CHECK(finite(value), "Ribbon '{}' has nonfinite {}", input.name, name);
}

// Dimensionless tolerances: classify nearly parallel frames without depending on authored normal magnitude.
constexpr float FRAME_SINE_SQUARED_MIN = 1e-12f;
constexpr float ANTIPODAL_DOT_TOLERANCE = 1e-6f;

float3 transport(float3 normal, float3 from, float3 to)
{
    const float cosine = dot(from, to);
    if (cosine <= -1.f + ANTIPODAL_DOT_TOLERANCE)
        return float3(0.f); // An antiparallel pair has no unique minimal-rotation axis.
    const float3 axis = cross(from, to);
    const float3 rotated = normal + cross(axis, normal) + cross(axis, cross(axis, normal)) / (1.f + cosine);
    return unit(rotated - to * dot(rotated, to));
}

size_t repair_frames(std::vector<Sample>& samples, bool periodic, std::string_view name, size_t strand)
{
    // A periodic endpoint is a UV duplicate, not an independent repair anchor.
    const size_t count = samples.size() - (periodic ? 1 : 0);
    std::vector<float3> orientations(count, float3(0.f));
    size_t valid_count = 0;
    for (size_t i = 0; i < count; ++i) {
        auto& s = samples[i];
        FALCOR_CHECK(finite(s.normal), "Ribbon '{}' strand {} sample {} has nonfinite normals", name, strand, i);
        s.normal = unit(s.normal);
        const float3 side = cross(s.normal, s.tangent);
        if (dot(side, side) > FRAME_SINE_SQUARED_MIN) {
            orientations[i] = unit(cross(s.tangent, side));
            ++valid_count;
        }
    }
    FALCOR_CHECK(valid_count, "Ribbon '{}' strand {} has no valid neighboring orientation frames", name, strand);
    if (valid_count != count) {
        std::vector<size_t> before(count, count), after(count, count);
        size_t anchor = count;
        const size_t passes = periodic ? 2 * count : count;
        for (size_t step = 0; step < passes; ++step) {
            const size_t i = step % count;
            if (dot(orientations[i], orientations[i]) > 0.f)
                anchor = i;
            before[i] = anchor;
        }
        anchor = count;
        for (size_t step = 0; step < passes; ++step) {
            const size_t i = count - 1 - step % count;
            if (dot(orientations[i], orientations[i]) > 0.f)
                anchor = i;
            after[i] = anchor;
        }
        for (size_t i = 0; i < count; ++i) {
            if (dot(orientations[i], orientations[i]) > 0.f)
                continue;
            const size_t a = before[i], b = after[i];
            float3 n(0.f);
            float weight_sum = 0.f;
            for (size_t neighbor : {a, b}) {
                if (neighbor == count)
                    continue;
                const float3 transported
                    = transport(orientations[neighbor], samples[neighbor].tangent, samples[i].tangent);
                FALCOR_CHECK(
                    dot(transported, transported) > 0.f,
                    "Ribbon '{}' strand {} sample {} has ambiguous antiparallel tangent transport",
                    name,
                    strand,
                    i
                );
                // Inverse parameter distance is equivalent to linear blending between the two anchors.
                const size_t distance = neighbor == a ? (i + count - a) % count : (b + count - i) % count;
                const float weight = 1.f / float(distance);
                n += transported * weight;
                weight_sum += weight;
            }
            n /= weight_sum;
            FALCOR_CHECK(
                dot(n, n) > FRAME_SINE_SQUARED_MIN,
                "Ribbon '{}' strand {} sample {} has canceling neighboring orientation frames",
                name,
                strand,
                i
            );
            samples[i].normal = unit(n);
        }
    }
    // Diagnose a sampled half-turn even when the sampling schedule misses its singular midpoint.
    // This deliberately does not attempt continuous cubic root isolation.
    for (size_t i = 0; i < count - (periodic ? 0 : 1); ++i) {
        const size_t j = (i + 1) % count;
        const float3 a = unit(cross(samples[i].tangent, cross(samples[i].normal, samples[i].tangent)));
        const float3 b = unit(cross(samples[j].tangent, cross(samples[j].normal, samples[j].tangent)));
        const float3 transported = transport(a, samples[i].tangent, samples[j].tangent);
        FALCOR_CHECK(
            dot(transported, transported) > 0.f && dot(transported, b) > -1.f + ANTIPODAL_DOT_TOLERANCE,
            "Ribbon '{}' strand {} span {} has ambiguous opposing orientation frames",
            name,
            strand,
            i
        );
    }
    if (periodic) {
        samples.back().normal = samples.front().normal;
        samples.back().tangent = samples.front().tangent;
    }
    return count - valid_count;
}

std::vector<Sample>
evaluate_strand(const CurveInput& input, const Strand& strand, size_t strand_index, bool static_attributes)
{
    const bool periodic = input.wrap == Wrap::periodic;
    std::vector<Sample> samples(strand.samples);
    evaluate_samples<true>(
        input,
        strand,
        strand_index,
        {},
        static_attributes,
        [&](size_t i, Sample s)
        {
            FALCOR_CHECK(finite(s.position), "Ribbon '{}' produced nonfinite centerline samples", input.name);
            s.tangent = unit(s.tangent);
            s.width = std::max(0.f, s.width);
            samples[i] = s;
        }
    );
    // Control polygon joins use the adjacent secants, including both sides of a closed seam.
    if (input.type == Type::linear || input.refinement_level == 0) {
        for (size_t i = 0; i < samples.size(); ++i) {
            const size_t before = i ? i - 1 : (periodic ? samples.size() - 2 : 0);
            const size_t after = i + 1 < samples.size() ? i + 1 : (periodic ? 1 : i);
            samples[i].tangent = unit(
                unit(samples[i].position - samples[before].position)
                + unit(samples[after].position - samples[i].position)
            );
        }
    }
    // Two linear passes fill zero derivatives, including long coincident runs.
    float3 previous(0.f);
    for (auto& s : samples) {
        if (dot(s.tangent, s.tangent) > 0.f)
            previous = s.tangent;
        else
            s.tangent = previous;
    }
    previous = float3(0.f);
    for (auto it = samples.rbegin(); it != samples.rend(); ++it) {
        if (dot(it->tangent, it->tangent) > 0.f)
            previous = it->tangent;
        else
            it->tangent = previous;
    }
    return samples;
}
std::vector<Strand> prepare_ribbon_strands(const CurveInput& input)
{
    const auto strands = prepare_strands(input);
    size_t point_count = 0, varying_count = 0, vertex_count = 0;
    for (const auto& strand : strands) {
        validate_strand_topology(input, strand.count);
        FALCOR_CHECK(
            strand.samples <= (std::numeric_limits<uint32_t>::max() - vertex_count) / 2,
            "Ribbon '{}' exceeds mesh index limits",
            input.name
        );
        point_count += strand.count;
        varying_count += strand.varying_count;
        vertex_count += 2 * strand.samples;
    }
    const auto& source = input.sample;
    for (float3 p : source.points)
        FALCOR_CHECK(finite(p), "Ribbon '{}' has nonfinite points", input.name);
    validate_attribute(input, source.normals, point_count, varying_count, "normals", true);
    validate_attribute(input, source.widths, point_count, varying_count, "widths");
    for (float w : source.widths.values)
        FALCOR_CHECK(w >= 0.f, "Ribbon '{}' has negative width", input.name);
    validate_attribute(input, input.texcoords, point_count, varying_count, "st");
    return strands;
}

size_t vertex_count(std::span<const Strand> strands)
{
    size_t count = 0;
    for (const auto& strand : strands)
        count += 2 * strand.samples;
    return count;
}

// Only one strand's sampled frames are temporary. Later motion samples write directly to output streams.
void evaluate_sample(
    const CurveInput& input,
    std::span<const Strand> strands,
    ImporterVertexStream<float3> positions,
    ImporterVertexStream<float3> normals,
    ImporterMesh* mesh,
    bool motion
)
{
    FALCOR_CHECK(
        positions.size == vertex_count(strands) && normals.size == positions.size,
        "Ribbon '{}' changes sample layout",
        input.name
    );
    const bool periodic = input.wrap == Wrap::periodic;
    size_t repaired_frames = 0;
    uint32_t start = 0;
    for (size_t strand_index = 0; strand_index < strands.size(); ++strand_index) {
        const auto& strand = strands[strand_index];
        auto samples = evaluate_strand(input, strand, strand_index, mesh != nullptr);
        if (dot(samples.front().tangent, samples.front().tangent) == 0.f) {
            FALCOR_CHECK(!motion, "Ribbon '{}' motion sample has no usable centerline tangent", input.name);
            start += static_cast<uint32_t>(2 * strand.samples);
            continue;
        }
        repaired_frames += repair_frames(samples, periodic, input.name, strand_index);
        for (size_t i = 0; i < samples.size(); ++i) {
            const auto& sample = samples[i];
            const float3 side = unit(cross(unit(sample.normal), sample.tangent));
            for (uint32_t edge = 0; edge < 2; ++edge) {
                const uint32_t index = start + static_cast<uint32_t>(2 * i) + edge;
                const float3 p = sample.position + side * (sample.width * (edge ? 0.5f : -0.5f));
                FALCOR_CHECK(finite(p) && finite(sample.uv), "Ribbon '{}' produced nonfinite geometry", input.name);
                positions[index] = p;
                normals[index] = sample.normal;
                if (mesh)
                    mesh->texcoord_stream()[index]
                        = input.texcoords.values.empty() ? float2(float(edge), sample.uv.x) : sample.uv;
            }
        }
        const uint32_t end = start + static_cast<uint32_t>(2 * strand.samples);
        if (periodic) {
            for (uint32_t edge = 0; edge < 2; ++edge) {
                positions[end - 2 + edge] = positions[start + edge];
                normals[end - 2 + edge] = normals[start + edge];
            }
        }
        // Validate each sample, but generate indices only once. Degeneracies may open later in motion.
        auto emit = [&](uint3 tri)
        {
            if (input.left_handed)
                std::swap(tri.y, tri.z);
            const float3 normal = cross(positions[tri.y] - positions[tri.x], positions[tri.z] - positions[tri.x]);
            FALCOR_CHECK(finite(normal), "Ribbon '{}' has overflowing surface derivatives", input.name);
            if (mesh && (motion || dot(normal, normal) != 0.f))
                mesh->subgeometries[0].indices.push_back(tri);
        };
        for (uint32_t a = start; a + 2 < end; a += 2) {
            emit(uint3(a, a + 2, a + 1));
            emit(uint3(a + 1, a + 2, a + 3));
        }
        start = end;
    }
    if (repaired_frames)
        sgl::log_warn(
            "Ribbon '{}' repaired {} orientation samples from neighboring frames",
            input.name,
            repaired_frames
        );
}

// Static degeneracies can leave unreferenced edges. Motion never performs this compaction.
ImporterMesh compact_static_mesh(ImporterMesh mesh)
{
    auto& triangles = mesh.subgeometries[0].indices;
    std::vector<uint32_t> remap(mesh.vertex_count(), std::numeric_limits<uint32_t>::max());
    uint32_t used = 0;
    for (const auto tri : triangles)
        for (uint32_t i : {tri.x, tri.y, tri.z})
            if (remap[i] == std::numeric_limits<uint32_t>::max())
                remap[i] = used++;
    if (used == mesh.vertex_count())
        return mesh;
    ImporterMesh result;
    result.name = mesh.name;
    result.properties = mesh.properties;
    if (used == 0)
        return result;
    result.ensure_attributes(
        {{ImporterSemantic::position}, {ImporterSemantic::normal}, {ImporterSemantic::tex_coord, 0, 2}}
    );
    result.allocate_vertices(used);
    for (size_t i = 0; i < remap.size(); ++i) {
        if (remap[i] == std::numeric_limits<uint32_t>::max())
            continue;
        result.position_stream()[remap[i]] = mesh.position_stream()[i];
        result.normal_stream()[remap[i]] = mesh.normal_stream()[i];
        result.texcoord_stream()[remap[i]] = mesh.texcoord_stream()[i];
    }
    for (auto& tri : triangles)
        tri = uint3(remap[tri.x], remap[tri.y], remap[tri.z]);
    result.subgeometries = std::move(mesh.subgeometries);
    return result;
}
} // namespace

ImporterMesh tessellate_ribbons(const CurveInput& input, size_t sample_count)
{
    FALCOR_CHECK(sample_count > 0, "Ribbon '{}' requires at least one sample", input.name);
    const auto strands = prepare_ribbon_strands(input);
    ImporterMesh result;
    result.name = input.name;
    result.properties = input.properties;
    std::vector<ImporterMeshDefaultAttribute> attributes{{ImporterSemantic::tex_coord, 0, 2}};
    for (uint32_t sample = 0; sample < sample_count; ++sample) {
        attributes.push_back({ImporterSemantic::position, sample});
        attributes.push_back({ImporterSemantic::normal, sample});
    }
    result.ensure_attributes(attributes);
    result.allocate_vertices(vertex_count(strands));
    result.subgeometries.push_back({input.name, {}, input.material_index});
    evaluate_sample(input, strands, result.position_stream(), result.normal_stream(), &result, sample_count > 1);
    return sample_count > 1 ? std::move(result) : compact_static_mesh(std::move(result));
}

void tessellate_ribbons(
    const CurveInput& input,
    ImporterVertexStream<float3> positions,
    ImporterVertexStream<float3> normals
)
{
    const auto strands = prepare_ribbon_strands(input);
    evaluate_sample(input, strands, positions, normals, nullptr, true);
}
} // namespace falcor::curve_tessellator
