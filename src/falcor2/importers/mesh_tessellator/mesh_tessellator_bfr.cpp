// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "mesh_tessellator_internal.h"

#include "falcor2/core/types.h"

#include <sgl/core/short_vector.h>

#include <opensubdiv/bfr/refinerSurfaceFactory.h>
#include <opensubdiv/bfr/tessellation.h>
#include <opensubdiv/far/topologyDescriptor.h>
#include <opensubdiv/far/topologyRefinerFactory.h>
#include <opensubdiv/sdc/types.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <numeric>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace falcor {
namespace mesh_tessellator {
namespace detail {

using namespace OpenSubdiv;

namespace {

constexpr int MAX_REFINEMENT_LEVEL = 8;
constexpr int CONTINUOUS_REGION = 0;

using Surface = Bfr::Surface<float>;

Sdc::SchemeType get_subdivision_scheme(const TessellatorInputMesh& input)
{
    switch (input.subdivision_scheme) {
    case SubdivisionScheme::catmull_clark:
        return Sdc::SCHEME_CATMARK;
    case SubdivisionScheme::bilinear:
        return Sdc::SCHEME_BILINEAR;
    case SubdivisionScheme::loop:
        return Sdc::SCHEME_LOOP;
    case SubdivisionScheme::none:
        SGL_UNREACHABLE();
    }
    SGL_UNREACHABLE();
}

void set_fvar_linear_interpolation(Sdc::Options& options, FaceVaryingLinearInterpolation interpolation)
{
    switch (interpolation) {
    case FaceVaryingLinearInterpolation::none:
        options.SetFVarLinearInterpolation(Sdc::Options::FVAR_LINEAR_NONE);
        break;
    case FaceVaryingLinearInterpolation::corners_only:
        options.SetFVarLinearInterpolation(Sdc::Options::FVAR_LINEAR_CORNERS_ONLY);
        break;
    case FaceVaryingLinearInterpolation::corners_plus_1:
        options.SetFVarLinearInterpolation(Sdc::Options::FVAR_LINEAR_CORNERS_PLUS1);
        break;
    case FaceVaryingLinearInterpolation::corners_plus_2:
        options.SetFVarLinearInterpolation(Sdc::Options::FVAR_LINEAR_CORNERS_PLUS2);
        break;
    case FaceVaryingLinearInterpolation::boundaries:
        options.SetFVarLinearInterpolation(Sdc::Options::FVAR_LINEAR_BOUNDARIES);
        break;
    case FaceVaryingLinearInterpolation::all:
        options.SetFVarLinearInterpolation(Sdc::Options::FVAR_LINEAR_ALL);
        break;
    }
}

void set_vtx_boundary_interpolation(Sdc::Options& options, VertexBoundaryInterpolation interpolation)
{
    switch (interpolation) {
    case VertexBoundaryInterpolation::none:
        options.SetVtxBoundaryInterpolation(Sdc::Options::VTX_BOUNDARY_NONE);
        break;
    case VertexBoundaryInterpolation::edge_only:
        options.SetVtxBoundaryInterpolation(Sdc::Options::VTX_BOUNDARY_EDGE_ONLY);
        break;
    case VertexBoundaryInterpolation::edge_and_corner:
        options.SetVtxBoundaryInterpolation(Sdc::Options::VTX_BOUNDARY_EDGE_AND_CORNER);
        break;
    }
}


/// Append one coarse face's transformed facets with one destination-vector size change.
void append_triangles(std::vector<uint3>& destination, std::span<const int3> source)
{
    const size_t destination_offset = destination.size();
    destination.resize(destination_offset + source.size());
    for (size_t triangle_index = 0; triangle_index < source.size(); ++triangle_index) {
        destination[destination_offset + triangle_index] = uint3(source[triangle_index]);
    }
}

/// One continuity identifier per output stream at a topological boundary point.
using BoundaryRegions = sgl::short_vector<int, 4>;

struct BoundaryVertexVariant {
    BoundaryRegions regions;
    int vertex_index;
};

template<size_t InlineVariantCount>
struct SharedBoundaryPoint {
    // A single topological point can require several renderer vertices when attribute continuity differs by face.
    sgl::short_vector<BoundaryVertexVariant, InlineVariantCount> variants;
};

using SharedVertex = SharedBoundaryPoint<6>;
using SharedEdgeSample = SharedBoundaryPoint<2>;

class SharedBoundary {
public:
    SharedBoundary(int vertex_count, int edge_count, int samples_per_edge)
        : m_vertices(vertex_count)
        , m_edge_samples(edge_count * samples_per_edge)
        , m_samples_per_edge(samples_per_edge)
    {
    }

    SharedVertex& vertex(int vertex_index) { return m_vertices[vertex_index]; }

    SharedEdgeSample& edge_sample(int edge_index, int sample_index)
    {
        return m_edge_samples[edge_index * m_samples_per_edge + sample_index];
    }

private:
    std::vector<SharedVertex> m_vertices;
    std::vector<SharedEdgeSample> m_edge_samples;
    int m_samples_per_edge;
};

/// What the evaluation phase must do for one stream at one face coordinate.
/// Copy actions carry the older canonical output vertex whose bit-identical value must be reused.
class StreamAction {
public:
    static StreamAction evaluate() { return StreamAction(EVALUATE); }
    static StreamAction skip() { return StreamAction(SKIP); }
    static StreamAction copy_from(int vertex_index)
    {
        FALCOR_ASSERT(vertex_index >= 0);
        return StreamAction(vertex_index);
    }

    bool should_evaluate() const { return m_value == EVALUATE; }
    bool should_skip() const { return m_value == SKIP; }
    bool should_copy() const { return m_value >= 0; }
    int source_vertex() const
    {
        FALCOR_ASSERT(should_copy());
        return m_value;
    }

private:
    explicit StreamAction(int value)
        : m_value(value)
    {
    }

    static constexpr int SKIP = -2;
    static constexpr int EVALUATE = -1;
    int m_value;
};

static_assert(sizeof(StreamAction) == sizeof(int));

StreamAction&
stream_action(std::vector<StreamAction>& actions, int coordinate_count, size_t stream_index, int coordinate_index)
{
    return actions[stream_index * coordinate_count + coordinate_index];
}

const StreamAction&
stream_action(std::span<const StreamAction> actions, int coordinate_count, size_t stream_index, int coordinate_index)
{
    return actions[stream_index * coordinate_count + coordinate_index];
}

/// Map one face-local boundary coordinate to a final renderer vertex. An exact region-signature match reuses an
/// existing vertex; otherwise a new variant records which streams must be evaluated or copied from older variants.
template<size_t InlineVariantCount>
int map_output_vertex(
    SharedBoundaryPoint<InlineVariantCount>& shared_point,
    const BoundaryRegions& regions,
    int coordinate_index,
    int coordinate_count,
    std::vector<StreamAction>& actions,
    int& next_vertex_index
)
{
    FALCOR_ASSERT_EQ(actions.size(), regions.size() * size_t(coordinate_count));

    // Reuse a renderer vertex when every output stream has the same continuity region.
    for (const BoundaryVertexVariant& variant : shared_point.variants) {
        if (variant.regions == regions) {
            for (size_t stream_index = 0; stream_index < regions.size(); ++stream_index)
                stream_action(actions, coordinate_count, stream_index, coordinate_index) = StreamAction::skip();
            return variant.vertex_index;
        }
    }

    const int vertex_index = next_vertex_index++;
    // A new combination still copies each continuous stream from its oldest matching variant. Deferring this copy
    // until evaluation keeps the topology pass independent of whether that source belongs to this or an earlier face.
    for (size_t stream_index = 0; stream_index < regions.size(); ++stream_index) {
        StreamAction action = StreamAction::evaluate();
        for (const BoundaryVertexVariant& variant : shared_point.variants) {
            if (variant.regions[stream_index] == regions[stream_index]) {
                action = StreamAction::copy_from(variant.vertex_index);
                break;
            }
        }
        stream_action(actions, coordinate_count, stream_index, coordinate_index) = action;
    }

    shared_point.variants.push_back({regions, vertex_index});
    return vertex_index;
}

void evaluate_positions_and_normals(
    OutputStreams& output,
    size_t position_stream_index,
    size_t normal_stream_index,
    const Surface& surface,
    std::span<const float2> coordinates,
    std::span<const int> coordinate_vertex_indices,
    std::span<const StreamAction> actions,
    std::vector<float>& patch_points
)
{
    const int coordinate_count = int(coordinate_vertex_indices.size());
    const Surface::PointDescriptor descriptor(3);
    patch_points.resize(surface.GetNumPatchPoints() * 3);
    surface
        .PreparePatchPoints(output[position_stream_index].mesh_values(), descriptor, patch_points.data(), descriptor);

    for (int coordinate_index = 0; coordinate_index < coordinate_count; ++coordinate_index) {
        const int vertex_index = coordinate_vertex_indices[coordinate_index];
        const StreamAction& position_action
            = stream_action(actions, coordinate_count, position_stream_index, coordinate_index);
        const StreamAction& normal_action
            = stream_action(actions, coordinate_count, normal_stream_index, coordinate_index);

        if (position_action.should_copy())
            output.copy_value(position_stream_index, position_action.source_vertex(), vertex_index);
        if (normal_action.should_copy())
            output.copy_value(normal_stream_index, normal_action.source_vertex(), vertex_index);
        if (!position_action.should_evaluate() && !normal_action.should_evaluate())
            continue;

        float3 unused_position;
        float3& position = position_action.should_evaluate() ? output.value<float3>(position_stream_index, vertex_index)
                                                             : unused_position;
        const float2& coordinate = coordinates[coordinate_index];
        if (normal_action.should_evaluate()) {
            float3 du;
            float3 dv;
            surface.Evaluate(&coordinate.x, patch_points.data(), descriptor, &position.x, &du.x, &dv.x);
            output.value<float3>(normal_stream_index, vertex_index) = sgl::math::normalize(sgl::math::cross(du, dv));
        } else {
            surface.Evaluate(&coordinate.x, patch_points.data(), descriptor, &position.x);
        }
    }
}

void evaluate_authored_stream(
    OutputStreams& output,
    size_t stream_index,
    int face_index,
    const Surface& vertex_surface,
    const Surface& varying_surface,
    std::span<const Surface> fvar_surfaces,
    std::span<const float2> coordinates,
    std::span<const int> coordinate_vertex_indices,
    std::span<const StreamAction> actions,
    std::vector<float>& patch_points
)
{
    OutputStream& stream = output[stream_index];
    const int coordinate_count = int(coordinate_vertex_indices.size());
    bool needs_evaluation = false;
    for (int coordinate_index = 0; coordinate_index < coordinate_count; ++coordinate_index) {
        needs_evaluation |= stream_action(actions, coordinate_count, stream_index, coordinate_index).should_evaluate();
    }

    const bool uses_face_value = stream.input->interpolation == MeshInterpolation::constant
        || stream.input->interpolation == MeshInterpolation::uniform;
    std::span<const float> face_value;
    const Surface* surface = nullptr;
    const Surface::PointDescriptor descriptor{int(stream.component_count())};
    if (needs_evaluation) {
        if (uses_face_value) {
            const int element_index = stream.input->interpolation == MeshInterpolation::constant ? 0 : face_index;
            face_value = stream_element(*stream.input, element_index);
        } else {
            switch (stream.input->interpolation) {
            case MeshInterpolation::vertex:
                surface = &vertex_surface;
                break;
            case MeshInterpolation::varying:
                surface = varying_surface.IsValid() ? &varying_surface : &vertex_surface;
                break;
            case MeshInterpolation::face_varying:
                surface = fvar_surfaces[stream.fvar_channel].IsValid() ? &fvar_surfaces[stream.fvar_channel]
                                                                       : &vertex_surface;
                break;
            case MeshInterpolation::constant:
            case MeshInterpolation::uniform:
                SGL_UNREACHABLE();
            }
            patch_points.resize(surface->GetNumPatchPoints() * stream.component_count());
            surface->PreparePatchPoints(stream.mesh_values(), descriptor, patch_points.data(), descriptor);
        }
    }

    for (int coordinate_index = 0; coordinate_index < coordinate_count; ++coordinate_index) {
        const StreamAction& action = stream_action(actions, coordinate_count, stream_index, coordinate_index);
        if (action.should_skip())
            continue;

        const int vertex_index = coordinate_vertex_indices[coordinate_index];
        if (action.should_copy()) {
            output.copy_value(stream_index, action.source_vertex(), vertex_index);
        } else if (uses_face_value) {
            output.set_value(stream_index, vertex_index, face_value);
        } else {
            const float2& coordinate = coordinates[coordinate_index];
            surface->Evaluate(&coordinate.x, patch_points.data(), descriptor, output.value(stream_index, vertex_index));
        }
    }
}

/// Estimate a useful initial capacity for the unified output vertex buffer. A fully shared regular surface
/// approaches V * rate^2 vertices, while discontinuous streams can split boundary samples into multiple renderer
/// vertices. Allow 20% for continuous topology and 50% when splitting is possible. This remains only an allocation
/// hint: OutputStreams grows safely if needed, and the cap avoids exceeding the conservative coarse-grid estimate.
size_t estimate_output_vertex_capacity(
    std::span<const OutputStream> streams,
    size_t coarse_vertex_count,
    size_t tessellation_rate
)
{
    const bool may_split_boundaries = std::any_of(
        streams.begin(),
        streams.end(),
        [](const OutputStream& stream)
        {
            return stream.region_mode != RegionMode::continuous;
        }
    );

    const size_t fully_shared_estimate = coarse_vertex_count * tessellation_rate * tessellation_rate;
    const size_t margin = fully_shared_estimate / (may_split_boundaries ? 2 : 5);
    const size_t coarse_grid_estimate = coarse_vertex_count * (tessellation_rate + 1) * (tessellation_rate + 1);
    return std::min(fully_shared_estimate + margin, coarse_grid_estimate);
}

} // namespace

ImporterMesh tessellate_subdivision_surface(const TessellatorInputMesh& input)
{
    FALCOR_CHECK(
        input.refinement_level >= 0 && input.refinement_level <= MAX_REFINEMENT_LEVEL,
        "Subdivision refinement level must be between 0 and {}",
        MAX_REFINEMENT_LEVEL
    );

    const size_t vertex_count = input.positions.value_count();
    check_stream_layout(
        input.positions,
        input.face_vertex_counts.size(),
        vertex_count,
        input.face_vertex_indices.size()
    );
    FALCOR_CHECK(input.positions.component_count() == 3, "Subdivision positions must have three components");
    FALCOR_CHECK(input.positions.indices.empty(), "Subdivision positions cannot be indexed separately");

    MeshAttributeStream default_uv;
    const std::vector<const MeshAttributeStream*> source_streams = collect_streams(input, default_uv);
    for (const MeshAttributeStream* stream : source_streams) {
        // Subdivision normals are generated from position derivatives, matching the original tessellator.
        if (stream->attribute.semantic == ImporterSemantic::normal && stream->attribute.index == 0)
            continue;
        check_stream_layout(*stream, input.face_vertex_counts.size(), vertex_count, input.face_vertex_indices.size());
    }

    const Sdc::SchemeType scheme = get_subdivision_scheme(input);
    const bool left_handed = is_left_handed(input.orientation);

    // Position and generated normal use the same storage and boundary handling as all authored attributes.
    OutputStreams output;
    output.reserve(source_streams.size() + 2);
    const size_t position_stream_index = output.add(input.positions.attribute, &input.positions);
    const size_t normal_stream_index = output.add({ImporterSemantic::normal, 0, 3});
    output[normal_stream_index].region_mode
        = input.subdivision_scheme == SubdivisionScheme::bilinear ? RegionMode::per_face : RegionMode::continuous;

    for (const MeshAttributeStream* stream : source_streams) {
        if (stream->attribute.semantic == ImporterSemantic::normal && stream->attribute.index == 0)
            continue;

        const size_t output_stream_index = output.add(stream->attribute, stream);
        OutputStream& output_stream = output[output_stream_index];
        switch (stream->interpolation) {
        case MeshInterpolation::constant:
        case MeshInterpolation::vertex:
        case MeshInterpolation::varying:
            output_stream.region_mode = RegionMode::continuous;
            break;
        case MeshInterpolation::uniform:
            output_stream.region_mode = RegionMode::per_face;
            break;
        case MeshInterpolation::face_varying:
            output_stream.region_mode = RegionMode::face_varying;
            break;
        }
    }

    std::vector<Far::TopologyDescriptor::FVarChannel> fvar_channels;
    fvar_channels.reserve(output.size() - 2);
    // OpenSubdiv requires one value index per face vertex for every fvar channel. Unindexed USD primvars use this
    // shared identity mapping; it remains empty when every fvar channel already provides indices.
    std::vector<Far::Index> unindexed_fvar_indices;
    bool has_varying_stream = false;
    // Bfr accepts authored indices directly for face-varying channels. Prepare dense control values for the
    // vertex and varying surfaces, which have no independent topology channel.
    for (OutputStream& stream : output.streams().subspan(2)) {
        if (stream.input->interpolation == MeshInterpolation::face_varying) {
            stream.fvar_channel = int(fvar_channels.size());
            Far::TopologyDescriptor::FVarChannel channel;
            channel.numValues = int(stream.input->value_count());
            if (stream.input->indices.empty()) {
                if (unindexed_fvar_indices.empty()) {
                    unindexed_fvar_indices.resize(input.face_vertex_indices.size());
                    std::iota(unindexed_fvar_indices.begin(), unindexed_fvar_indices.end(), 0);
                }
                channel.valueIndices = unindexed_fvar_indices.data();
            } else {
                channel.valueIndices = stream.input->indices.data();
            }
            fvar_channels.push_back(channel);
        } else if (stream.input->interpolation == MeshInterpolation::vertex
                   || stream.input->interpolation == MeshInterpolation::varying) {
            stream.prepare_control_values(vertex_count);
            has_varying_stream |= stream.input->interpolation == MeshInterpolation::varying;
        }
    }

    Sdc::Options options;
    set_vtx_boundary_interpolation(options, input.vertex_boundary_interpolation);
    set_fvar_linear_interpolation(options, input.face_varying_linear_interpolation);
    Far::TopologyRefinerFactory<Far::TopologyDescriptor>::Options refiner_options(scheme, options);

    Far::TopologyDescriptor descriptor = {};
    descriptor.numVertices = int(vertex_count);
    descriptor.numFaces = int(input.face_vertex_counts.size());
    descriptor.numVertsPerFace = input.face_vertex_counts.data();
    descriptor.vertIndicesPerFace = input.face_vertex_indices.data();
    descriptor.numHoles = int(input.hole_indices.size());
    descriptor.holeIndices = input.hole_indices.data();
    descriptor.isLeftHanded = left_handed;
    descriptor.numFVarChannels = int(fvar_channels.size());
    descriptor.fvarChannels = fvar_channels.empty() ? nullptr : fvar_channels.data();

    std::unique_ptr<Far::TopologyRefiner> refiner(
        Far::TopologyRefinerFactory<Far::TopologyDescriptor>::Create(descriptor, refiner_options)
    );
    FALCOR_CHECK(refiner, "Failed to create OpenSubdiv topology refiner");

    using SurfaceFactory = Bfr::RefinerSurfaceFactory<>;
    SurfaceFactory::Options surface_options;
    surface_options.SetApproxLevelSmooth(std::max(2, input.refinement_level));
    surface_options.SetApproxLevelSharp(std::max(6, input.refinement_level));
    SurfaceFactory surface_factory(*refiner, surface_options);
    const Far::TopologyLevel& base_level = refiner->GetLevel(0);

    Surface vertex_surface;
    Surface varying_surface;
    std::vector<Surface> fvar_surfaces(fvar_channels.size());
    std::vector<Bfr::SurfaceFactoryMeshAdapter::FVarID> fvar_ids(fvar_channels.size());
    std::iota(fvar_ids.begin(), fvar_ids.end(), 0);

    ImporterMesh result;
    std::vector<int> subgeometry_mapping;
    initialize_result(input, result, subgeometry_mapping);
    exclude_holes_from_output(input, subgeometry_mapping);

    const int tessellation_rate = 1 << input.refinement_level;
    const int samples_per_edge = tessellation_rate - 1;
    output.reserve_vertices(estimate_output_vertex_capacity(output.streams(), vertex_count, size_t(tessellation_rate)));
    SharedBoundary shared_boundary(base_level.GetNumVertices(), base_level.GetNumEdges(), samples_per_edge);

    std::vector<float2> coordinates;
    std::vector<int3> triangles;
    // Final output vertex for each Bfr face coordinate. The boundary prefix also remaps Bfr's facet indices.
    std::vector<int> coordinate_vertex_indices;
    std::vector<StreamAction> stream_actions;
    // Reused by positions and every authored stream; no face retains patch points for multiple streams at once.
    std::vector<float> patch_points;
    // The continuity region of each output stream at every topological boundary point. Each face coordinate's
    // region vector is compared to the existing variants at that point to determine whether a new renderer vertex is
    // required.
    BoundaryRegions vertex_regions;
    BoundaryRegions edge_regions;
    vertex_regions.reserve(output.size());
    edge_regions.reserve(output.size());

    Bfr::Tessellation::Options tessellation_options;
    tessellation_options.SetFacetSize(3);
    const int face_count = surface_factory.GetNumFaces();
    for (int face_index = 0; face_index < face_count; ++face_index) {
        if (subgeometry_mapping[face_index] < 0)
            continue;

        surface_factory.InitSurfaces(
            face_index,
            &vertex_surface,
            fvar_surfaces.empty() ? nullptr : fvar_surfaces.data(),
            fvar_ids.empty() ? nullptr : fvar_ids.data(),
            int(fvar_surfaces.size()),
            has_varying_stream ? &varying_surface : nullptr
        );
        if (!vertex_surface.IsValid())
            continue;

        Bfr::Tessellation tessellation(vertex_surface.GetParameterization(), tessellation_rate, tessellation_options);
        const int coordinate_count = tessellation.GetNumCoords();
        coordinates.resize(coordinate_count);
        tessellation.GetCoords(&coordinates.front().x);
        coordinate_vertex_indices.resize(coordinate_count);
        stream_actions.assign(output.size() * coordinate_count, StreamAction::evaluate());

        triangles.resize(tessellation.GetNumFacets());
        tessellation.GetFacets(reinterpret_cast<int*>(triangles.data()));

        const Far::ConstIndexArray face_vertices = base_level.GetFaceVertices(face_index);
        const Far::ConstIndexArray face_edges = base_level.GetFaceEdges(face_index);
        const int boundary_count = tessellation.GetNumBoundaryCoords();

        // Phase 1: map every face coordinate to a final vertex index without changing output storage.
        int next_vertex_index = int(output.vertex_count());
        // Bfr lists each face vertex followed by the samples along its outgoing edge. Follow OpenSubdiv's
        // bfr_tutorial_2_2 and remap that boundary ring before appending the face-local interior.
        int boundary_index = 0;
        for (int local_edge = 0; local_edge < face_vertices.size(); ++local_edge) {
            const int vertex_index = face_vertices[local_edge];
            vertex_regions.clear();
            for (const OutputStream& stream : output.streams()) {
                int region = CONTINUOUS_REGION;
                if (stream.region_mode == RegionMode::per_face) {
                    region = face_index + 1;
                } else if (stream.region_mode == RegionMode::face_varying
                           && !base_level.DoesVertexFVarTopologyMatch(vertex_index, stream.fvar_channel)) {
                    region = base_level.GetFaceFVarValues(face_index, stream.fvar_channel)[local_edge] + 1;
                }
                vertex_regions.push_back(region);
            }

            coordinate_vertex_indices[boundary_index] = map_output_vertex(
                shared_boundary.vertex(vertex_index),
                vertex_regions,
                boundary_index,
                coordinate_count,
                stream_actions,
                next_vertex_index
            );
            ++boundary_index;

            const int edge_index = face_edges[local_edge];
            const Far::ConstIndexArray edge_vertices = base_level.GetEdgeVertices(edge_index);
            const bool is_canonical_orientation = edge_vertices[0] == vertex_index;
            edge_regions.clear();
            for (const OutputStream& stream : output.streams()) {
                int region = CONTINUOUS_REGION;
                if (stream.region_mode == RegionMode::per_face
                    || (stream.region_mode == RegionMode::face_varying
                        && !base_level.DoesEdgeFVarTopologyMatch(edge_index, stream.fvar_channel))) {
                    region = face_index + 1;
                }
                edge_regions.push_back(region);
            }

            for (int local_sample = 0; local_sample < samples_per_edge; ++local_sample) {
                const int canonical_sample
                    = is_canonical_orientation ? local_sample : samples_per_edge - 1 - local_sample;
                coordinate_vertex_indices[boundary_index] = map_output_vertex(
                    shared_boundary.edge_sample(edge_index, canonical_sample),
                    edge_regions,
                    boundary_index,
                    coordinate_count,
                    stream_actions,
                    next_vertex_index
                );
                ++boundary_index;
            }
        }
        FALCOR_ASSERT_EQ(boundary_index, boundary_count);

        const int interior_vertex_count = coordinate_count - boundary_count;
        const int interior_start = next_vertex_index;
        std::iota(coordinate_vertex_indices.begin() + boundary_count, coordinate_vertex_indices.end(), interior_start);
        next_vertex_index += interior_vertex_count;

        // Phase 2: materialize the newly allocated indices with one vector mutation.
        output.resize_vertices(size_t(next_vertex_index));

        // Phase 3: with final indices fixed, evaluate each stream over the face coordinates in one tight loop.
        evaluate_positions_and_normals(
            output,
            position_stream_index,
            normal_stream_index,
            vertex_surface,
            coordinates,
            coordinate_vertex_indices,
            stream_actions,
            patch_points
        );
        for (size_t stream_index = 2; stream_index < output.size(); ++stream_index) {
            evaluate_authored_stream(
                output,
                stream_index,
                face_index,
                vertex_surface,
                varying_surface,
                fvar_surfaces,
                coordinates,
                coordinate_vertex_indices,
                stream_actions,
                patch_points
            );
        }

        // Phase 4: remap Bfr's face-local facets and append them as one batch.
        const int interior_offset = interior_start - boundary_count;
        tessellation.TransformFacetCoordIndices(
            reinterpret_cast<int*>(triangles.data()),
            coordinate_vertex_indices.data(),
            interior_offset
        );

        ImporterMesh::Subgeometry& subgeometry = result.subgeometries[subgeometry_mapping[face_index]];
        append_triangles(subgeometry.indices, triangles);
    }

    // Direct ImporterMesh output is possible, but shared subdivision vertices make the final count unknown until
    // all faces are processed. A practical implementation needs heuristic reservation and growth-safe stream
    // access, and retains excess capacity. It showed no measurable speedup, so copy into an exact final allocation.
    output.write(result);

    return result;
}

} // namespace detail
} // namespace mesh_tessellator
} // namespace falcor
