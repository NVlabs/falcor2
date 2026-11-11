// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "tangent_transfer.h"
#include "falcor2/core/error.h"
#include <cmath>
#include <limits>

namespace falcor {
namespace {

// Relative-frame transport: T_target = F_target * transpose(F_reference) * T_reference.
// Blender's Crazyspace uses the first incident face to estimate each vertex's rotation:
// https://github.com/blender/blender/blob/v4.5.0/source/blender/blenkernel/intern/crazyspace.cc#L170-L205
// Its Deform Curves on Surface node also uses relative orthonormal frames:
// https://github.com/blender/blender/blob/v4.5.0/source/blender/nodes/geometry/nodes/node_geo_deform_curves_on_surface.cc#L121-L154
// Here the frame comes from triangle edges, with shading-normal projection after transfer.
// This captures rotation about an unchanged normal, but does not reproduce stretch/shear or regenerated MikkTSpace.
struct Frame {
    float3 x, y, z;
};

bool make_frame(std::span<const float3> positions, size_t vertex, uint2 neighbors, Frame& frame)
{
    if (neighbors.x == std::numeric_limits<uint32_t>::max())
        return false;
    const float3 edge = positions[neighbors.x] - positions[vertex];
    const float3 other = positions[neighbors.y] - positions[vertex];
    const float3 normal = cross(edge, other);
    const float length_squared = dot(edge, edge);
    const float area_squared = dot(normal, normal);
    // Relative area keeps the degeneracy decision independent of scene units.
    if (!std::isfinite(area_squared) || area_squared <= 1e-12f * length_squared * dot(other, other))
        return false;
    frame.x = edge / std::sqrt(length_squared);
    frame.z = normal / std::sqrt(area_squared);
    frame.y = cross(frame.z, frame.x);
    return true;
}

} // namespace

TangentTransfer::TangentTransfer(
    std::span<const uint3> triangles,
    std::span<const float3> positions,
    std::span<const float3> normals,
    std::span<const float3> tangents
)
    : m_normals(normals)
    , m_tangents(tangents)
    , m_neighbors(positions.size(), uint2(std::numeric_limits<uint32_t>::max()))
    , m_local_tangents(positions.size(), float3(0))
{
    FALCOR_CHECK(
        normals.size() == positions.size() && tangents.size() == positions.size(),
        "Tangent transfer requires matching vertex arrays"
    );
    // Keep the first incident triangle fixed across poses; changing neighbors could introduce orientation jumps.
    for (const uint3 triangle : triangles) {
        for (int corner = 0; corner < 3; ++corner) {
            FALCOR_CHECK(triangle[corner] < positions.size(), "Invalid tangent transfer triangle");
            auto& neighbors = m_neighbors[triangle[corner]];
            if (neighbors.x == std::numeric_limits<uint32_t>::max())
                neighbors = uint2(triangle[(corner + 1) % 3], triangle[(corner + 2) % 3]);
        }
    }
    // Store F_reference^T * T_reference once. Zero marks an unusable frame for the normal-alignment fallback.
    for (size_t i = 0; i < positions.size(); ++i) {
        Frame frame;
        if (make_frame(positions, i, m_neighbors[i], frame))
            m_local_tangents[i]
                = float3(dot(tangents[i], frame.x), dot(tangents[i], frame.y), dot(tangents[i], frame.z));
    }
}

void TangentTransfer::evaluate(
    std::span<const float3> positions,
    std::span<const float3> normals,
    std::span<float3> tangents
) const
{
    FALCOR_CHECK(
        positions.size() == m_normals.size() && normals.size() == positions.size()
            && tangents.size() == positions.size(),
        "Tangent transfer target count mismatch"
    );
    for (size_t i = 0; i < positions.size(); ++i) {
        const float3 local = m_local_tangents[i];
        const float3 n = normals[i];
        float3 tangent;
        Frame frame;
        if (dot(local, local) > 0 && make_frame(positions, i, m_neighbors[i], frame)) {
            tangent = frame.x * local.x + frame.y * local.y + frame.z * local.z;
        } else {
            // Without a geometric frame, use Rodrigues' shortest-arc rotation between the shading normals.
            const float3 t = m_tangents[i];
            const float cosine = dot(m_normals[i], n);
            const float3 axis = cross(m_normals[i], n);
            // A half turn about t handles antiparallel normals deterministically.
            tangent = cosine > -1.f + 1e-6f ? t + cross(axis, t) + cross(axis, cross(axis, t)) / (1.f + cosine) : t;
        }
        // The geometric frame need not agree with the shading normal. Project and repair a collapsed projection.
        tangent -= n * dot(tangent, n);
        if (dot(tangent, tangent) <= 1e-20f)
            tangent = cross(std::abs(n.z) < 0.9f ? float3(0, 0, 1) : float3(0, 1, 0), n);
        tangents[i] = normalize(tangent);
    }
}

} // namespace falcor
