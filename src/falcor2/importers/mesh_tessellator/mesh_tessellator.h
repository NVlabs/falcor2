// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/importers/importer_types.h"

#include <span>
#include <string>
#include <vector>

namespace falcor {
namespace mesh_tessellator {

enum class MeshInterpolation {
    constant,
    uniform,
    vertex,
    varying,
    face_varying,
};

enum class SubdivisionScheme {
    none,
    catmull_clark,
    bilinear,
    loop,
};

enum class MeshOrientation {
    right_handed,
    left_handed,
};

enum class FaceVaryingLinearInterpolation {
    none,
    corners_only,
    corners_plus_1,
    corners_plus_2,
    boundaries,
    all,
};

enum class VertexBoundaryInterpolation {
    none,
    edge_only,
    edge_and_corner,
};

/// A borrowed authored float32 attribute stored as a tightly packed value table.
/// Indices select values for interpolation elements: the mesh, coarse faces, mesh vertices, or face vertices,
/// depending on interpolation. Every authored index must address an existing value. The caller owns both spans for
/// the duration of tessellate().
struct MeshAttributeStream {
    ImporterMeshDefaultAttribute attribute;
    MeshInterpolation interpolation = MeshInterpolation::vertex;
    // Tightly packed scalar components for the authored value table.
    std::span<const float> values;
    // Optional map from interpolation elements to entries in the value table.
    std::span<const int> indices;

    size_t component_count() const { return attribute.components; }
    size_t value_count() const { return component_count() == 0 ? 0 : values.size() / component_count(); }
};

/// Borrowed bulk mesh data plus small owned metadata. All spans must remain valid until tessellate() returns.
///
/// Following the PxOsd/HdMesh input contract, tessellate() assumes borrowed topology and attribute arrays have
/// already been validated instead of scanning them solely for validation. Face counts and all topology, attribute,
/// hole, and subgeometry indices must be mutually consistent and in range. Positions must be an unindexed float3
/// vertex stream. Only faces assigned to a subgeometry are emitted; holes remain in the subdivision topology but are
/// not emitted. A refinement level of zero, or a subdivision scheme of none, selects ordinary polygon triangulation.
/// Primary normal streams are used by ordinary polygon triangulation. Refined subdivision surfaces generate normals
/// from position derivatives instead.
struct TessellatorInputMesh {
    std::string name;
    SubdivisionScheme subdivision_scheme = SubdivisionScheme::catmull_clark;
    MeshOrientation orientation = MeshOrientation::right_handed;
    FaceVaryingLinearInterpolation face_varying_linear_interpolation = FaceVaryingLinearInterpolation::corners_plus_1;
    VertexBoundaryInterpolation vertex_boundary_interpolation = VertexBoundaryInterpolation::edge_and_corner;

    std::span<const int> face_vertex_counts;
    std::span<const int> face_vertex_indices;
    std::span<const int> hole_indices;
    /// Import-processing properties, initialized with ImporterMesh::make_properties().
    Properties properties;

    /// Ordered samples with shared topology and attribute layouts. Normals are empty or match the position sample
    /// count. Each output position/normal attribute uses its sample index; zero is the reference geometry.
    std::span<const MeshAttributeStream> positions;
    std::span<const MeshAttributeStream> normals;
    /// Static attributes other than primary positions and normals.
    std::vector<MeshAttributeStream> streams;

    struct Subgeometry {
        std::string name;
        int material_index = -1;
        std::span<const int> face_indices;
    };

    std::vector<Subgeometry> subgeometries;
};

/// Generate smooth normals for one sample by accumulating incident face-corner cross products over the coarse
/// topology. normal_values must be empty, separate storage; the result has one float3 per position and is suitable
/// for a varying normal stream to shade the control cage smoothly.
FALCOR_API void
calculate_smooth_normals(const TessellatorInputMesh& input, size_t sample, std::vector<float>& normal_values);

FALCOR_API ImporterMesh tessellate(const TessellatorInputMesh& input);

} // namespace mesh_tessellator
} // namespace falcor
