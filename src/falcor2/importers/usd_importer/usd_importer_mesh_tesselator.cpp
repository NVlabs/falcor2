// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "usd_importer_mesh_tesselator.h"
#include "usd_importer_utils.h"
#include "usd_importer_macros.h"

#include "falcor2/importers/usd_importer/usd_importer.h"
#include "falcor2/core/types.h"
#include "falcor2/utils/indexed_vector.h"

BEGIN_DISABLE_USD_WARNINGS
#include <pxr/base/gf/vec2f.h>
#include <pxr/base/gf/vec3f.h>
#include <pxr/base/tf/token.h>
#include <pxr/base/vt/array.h>
#include <pxr/base/vt/types.h>
#include <pxr/usd/usdGeom/tokens.h>
END_DISABLE_USD_WARNINGS

#include <opensubdiv/bfr/refinerSurfaceFactory.h>
#include <opensubdiv/bfr/tessellation.h>
#include <opensubdiv/far/topologyDescriptor.h>
#include <opensubdiv/far/topologyRefinerFactory.h>
#include <opensubdiv/sdc/types.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>
#include <algorithm>
#include <span>

namespace falcor {
namespace usd_importer {

using namespace OpenSubdiv;

namespace {

template<typename T>
const T&
extract(const pxr::VtArray<T>& value, const pxr::TfToken& interpolation, int uniform, int varying, int fvarying)
{
    if (interpolation == pxr::UsdGeomTokens->constant)
        return value[0];
    if (interpolation == pxr::UsdGeomTokens->uniform)
        return value[uniform];
    if (interpolation == pxr::UsdGeomTokens->varying)
        return value[varying];
    if (interpolation == pxr::UsdGeomTokens->vertex)
        return value[varying];
    if (interpolation == pxr::UsdGeomTokens->faceVarying)
        return value[fvarying];
    return value[0];
}


Sdc::Options::FVarLinearInterpolation get_fvar_linear_interpolation(const pxr::TfToken& fvar_interpolation)
{
    if (fvar_interpolation == pxr::UsdGeomTokens->cornersOnly)
        return Sdc::Options::FVAR_LINEAR_CORNERS_ONLY;
    if (fvar_interpolation == pxr::UsdGeomTokens->cornersPlus1)
        return Sdc::Options::FVAR_LINEAR_CORNERS_PLUS1;
    if (fvar_interpolation == pxr::UsdGeomTokens->cornersPlus2)
        return Sdc::Options::FVAR_LINEAR_CORNERS_PLUS2;
    if (fvar_interpolation == pxr::UsdGeomTokens->boundaries)
        return Sdc::Options::FVAR_LINEAR_BOUNDARIES;
    if (fvar_interpolation == pxr::UsdGeomTokens->all)
        return Sdc::Options::FVAR_LINEAR_ALL;
    return Sdc::Options::FVAR_LINEAR_NONE;
}

Sdc::Options::VtxBoundaryInterpolation get_vtx_boundary_interpolation(const pxr::TfToken vtx_boundary_interpolation)
{
    if (vtx_boundary_interpolation == pxr::UsdGeomTokens->none)
        return Sdc::Options::VTX_BOUNDARY_NONE;
    if (vtx_boundary_interpolation == pxr::UsdGeomTokens->edgeOnly)
        return Sdc::Options::VTX_BOUNDARY_EDGE_ONLY;
    return Sdc::Options::VTX_BOUNDARY_EDGE_AND_CORNER;
}

ImporterMesh triangulate(const TesselatorInputMesh& input)
{
    bool generate_normals = input.normals.empty();

    std::vector<int> sorted_holes(input.hole_indices.begin(), input.hole_indices.end());
    std::sort(sorted_holes.begin(), sorted_holes.end());
    sorted_holes.erase(std::unique(sorted_holes.begin(), sorted_holes.end()), sorted_holes.end());

    int next[2] = {1, 2};
    if (input.orientation == pxr::UsdGeomTokens->leftHanded)
        std::swap(next[0], next[1]);

    ImporterMesh result;
    result.name = input.name;
    result.subgeometries.resize(input.subgeometries.size());
    std::vector<int> subgeometry_mapping(input.face_vertex_counts.size(), -1);
    for (size_t i = 0; i < input.subgeometries.size(); ++i) {
        result.subgeometries[i].material_name = input.subgeometries[i].material_name;
        result.subgeometries[i].name = input.subgeometries[i].name;

        for (auto& face_index : input.subgeometries[i].face_indices)
            subgeometry_mapping[face_index] = (int)i;
    }

    result.ensure_attributes({
        {ImporterSemantic::position},
        {ImporterSemantic::normal},
        {ImporterSemantic::tex_coord},
    });

    size_t hole_index = 0;
    for (size_t face_index = 0, fvar_index = 0; face_index < input.face_vertex_counts.size();
         fvar_index += input.face_vertex_counts[face_index], ++face_index) {

        int vertex_count = input.face_vertex_counts[face_index];
        if (vertex_count < 3 || subgeometry_mapping[face_index] < 0)
            continue;
        if (hole_index < sorted_holes.size() && sorted_holes[hole_index] == (int)face_index) {
            ++hole_index;
            continue;
        }

        float3 flat_normal(0.f);
        if (generate_normals) {
            const pxr::GfVec3f& v0 = input.positions[input.face_indices[fvar_index]];
            for (int i = 0; i < vertex_count - 2; ++i) {
                const pxr::GfVec3f& v1 = input.positions[input.face_indices[fvar_index + i + next[0]]];
                const pxr::GfVec3f& v2 = input.positions[input.face_indices[fvar_index + i + next[1]]];
                flat_normal += sgl::math::cross(to_falcor(v1 - v0), to_falcor(v2 - v0));
            }
            flat_normal = sgl::math::normalize(flat_normal);
        }

        // Allocate vertices in final mesh then get streams to write to them
        int vertex_offset = (int)result.allocate_vertices(vertex_count);
        auto positions = result.position_stream();
        auto normals = result.normal_stream();
        auto uvs = result.texcoord_stream();

        for (int v = 0; v < vertex_count; ++v) {
            int primvar_uniform_index = (int)face_index;
            int primvar_varying_index = input.face_indices[fvar_index + v];
            int primvar_fvar_index = (int)fvar_index + v;

            positions[vertex_offset + v] = to_falcor(input.positions[primvar_varying_index]);
            normals[vertex_offset + v] = flat_normal;
            uvs[vertex_offset + v] = float2(0.f);

            if (!input.normals.empty()) {
                normals[vertex_offset + v] = to_falcor(extract(
                    input.normals,
                    input.normals_interpolation,
                    primvar_uniform_index,
                    primvar_varying_index,
                    primvar_fvar_index
                ));
            }

            if (!input.uvs.empty()) {
                uvs[vertex_offset + v] = to_falcor(extract(
                    input.uvs,
                    input.uv_interpolation,
                    primvar_uniform_index,
                    primvar_varying_index,
                    primvar_fvar_index
                ));
            }
        }

        for (int v = 0; v < vertex_count - 2; ++v) {
            uint3 triangle_indices;
            triangle_indices[0] = vertex_offset + 0;
            triangle_indices[1] = vertex_offset + v + next[0];
            triangle_indices[2] = vertex_offset + v + next[1];

            result.subgeometries[subgeometry_mapping[face_index]].indices.push_back(triangle_indices);
        }
    }

    return result;
}

} // namespace


ImporterMesh tessellate(const TesselatorInputMesh& input)
{
    if (input.positions.empty() || input.face_vertex_counts.empty() || input.face_indices.empty())
        return {};

    if (input.subdiv_scheme == pxr::UsdGeomTokens->none || input.refinement_level == 0)
        return triangulate(input);

    using SurfaceFactory = Bfr::RefinerSurfaceFactory<>;
    using Surface = Bfr::Surface<float>;
    using FVarID = Bfr::SurfaceFactoryMeshAdapter::FVarID;

    Sdc::SchemeType scheme = Sdc::SCHEME_CATMARK;
    if (input.subdiv_scheme == pxr::UsdGeomTokens->bilinear)
        scheme = Sdc::SCHEME_BILINEAR;
    if (input.subdiv_scheme == pxr::UsdGeomTokens->loop) {
        auto it = std::find_if(
            input.face_vertex_counts.begin(),
            input.face_vertex_counts.end(),
            [](int i)
            {
                return i != 3;
            }
        );

        if (it != input.face_vertex_counts.end())
            throw std::runtime_error("Loop subdivision supports only triangular meshes");

        scheme = Sdc::SCHEME_LOOP;
    }

    Sdc::Options options;
    options.SetVtxBoundaryInterpolation(get_vtx_boundary_interpolation(input.vertex_boundary_interpolation));
    options.SetFVarLinearInterpolation(get_fvar_linear_interpolation(input.facevarying_interpolation));
    Far::TopologyRefinerFactory<Far::TopologyDescriptor>::Options refined_options(scheme, options);

    Far::TopologyDescriptor::FVarChannel channels;
    Far::TopologyDescriptor desc = {};
    desc.numVertices = (int)input.face_indices.size();
    desc.numFaces = (int)input.face_vertex_counts.size();
    desc.numVertsPerFace = input.face_vertex_counts.data();
    desc.vertIndicesPerFace = input.face_indices.data();
    desc.numHoles = (int)input.hole_indices.size();
    desc.holeIndices = input.hole_indices.data();
    desc.isLeftHanded = (input.orientation == pxr::UsdGeomTokens->leftHanded);
    desc.fvarChannels = &channels;

    const pxr::GfVec2f* uv_data = input.uvs.empty() ? nullptr : input.uvs.data();

    IndexedVector<pxr::GfVec2f, Far::Index, pxr::TfHash> indexed_uvs;
    /// Facevarying interpolation in USD is position based, but OSD expects indexed values,
    /// because it recognizes value boundaries based on indices, not values
    if (!input.uvs.empty() && input.uv_interpolation == pxr::UsdGeomTokens->faceVarying) {
        for (auto& uv : input.uvs)
            indexed_uvs.append(uv);

        uv_data = indexed_uvs.get_values().data();
        channels.numValues = (int)indexed_uvs.get_values().size();
        channels.valueIndices = indexed_uvs.get_indices().data();
        desc.numFVarChannels++;
    }

    std::unique_ptr<Far::TopologyRefiner> refiner(
        Far::TopologyRefinerFactory<Far::TopologyDescriptor>::Create(desc, refined_options)
    );

    SurfaceFactory::Options surface_options;
    SurfaceFactory surface_factory(*refiner, surface_options);

    Surface vertex_surface;
    Surface varying_surface;
    Surface fvarying_surface;

    std::vector<float> face_patch_points;
    std::vector<float2> tess_coords;
    std::vector<float3> out_positions;
    std::vector<float3> out_normals;
    std::vector<float2> out_uvs;
    std::vector<int3> out_triangles;

    Bfr::Tessellation::Options tess_options;
    tess_options.SetFacetSize(3);

    bool left_handed = input.orientation == pxr::UsdGeomTokens->leftHanded;

    const uint32_t face_count = surface_factory.GetNumFaces();
    Surface* uv_surface = nullptr;
    if (input.uv_interpolation == pxr::UsdGeomTokens->faceVarying)
        uv_surface = &fvarying_surface;
    if (input.uv_interpolation == pxr::UsdGeomTokens->vertex)
        uv_surface = &vertex_surface;
    if (input.uv_interpolation == pxr::UsdGeomTokens->varying)
        uv_surface = &varying_surface;
    if (input.uvs.empty())
        uv_surface = nullptr;

    ImporterMesh result;
    result.name = input.name;
    result.subgeometries.resize(input.subgeometries.size());
    std::vector<int> subgeometry_mapping(input.face_vertex_counts.size(), -1);
    for (size_t i = 0; i < input.subgeometries.size(); ++i) {
        result.subgeometries[i].material_name = input.subgeometries[i].material_name;
        result.subgeometries[i].name = input.subgeometries[i].name;

        for (auto& face_index : input.subgeometries[i].face_indices)
            subgeometry_mapping[face_index] = (int)i;
    }

    result.ensure_attributes({
        {ImporterSemantic::position},
        {ImporterSemantic::normal},
        {ImporterSemantic::tex_coord},
    });

    // It is safe to create face-varying surface even if uvs aren't provided.
    FVarID fvar_id = 0;
    for (uint32_t face_index = 0; face_index < face_count; ++face_index) {
        // vertex surface is mandatory
        // facevarying surface will be initialized only if desc.numFVarChannels == 1
        // varying surface will be initialized only if uv_surface is using it.
        surface_factory.InitSurfaces(
            face_index,
            &vertex_surface,
            &fvarying_surface,
            &fvar_id,
            desc.numFVarChannels,
            (uv_surface == &varying_surface) ? &varying_surface : nullptr
        );

        if (!vertex_surface.IsValid())
            continue;

        // Fall back to vertex_surface is the uv interpolation failed.
        if (uv_surface && !uv_surface->IsValid())
            uv_surface = &vertex_surface;

        Bfr::Tessellation tess_pattern(vertex_surface.GetParameterization(), input.refinement_level + 1, tess_options);
        int coord_count = tess_pattern.GetNumCoords();
        tess_coords.resize(coord_count);
        tess_pattern.GetCoords((float*)tess_coords.data());

        out_normals.resize(coord_count);
        out_positions.resize(coord_count);

        /// Handles vertex/varying/facevarying interpolation
        if (uv_surface) {
            Surface::PointDescriptor point_desc(2); // # of floats per element, float2 -> 2
            face_patch_points.resize(uv_surface->GetNumPatchPoints() * point_desc.size);
            out_uvs.resize(coord_count);
            uv_surface->PreparePatchPoints((const float*)uv_data, point_desc, face_patch_points.data(), point_desc);
            for (int i = 0; i < coord_count; ++i) {
                uv_surface->Evaluate(
                    (const float*)&tess_coords[i],
                    face_patch_points.data(),
                    point_desc,
                    (float*)&out_uvs[i]
                );
            }

        } else if (input.uv_interpolation == pxr::UsdGeomTokens->uniform) {
            out_uvs.assign(coord_count, to_falcor(input.uvs[face_index]));
        } else {
            out_uvs.assign(coord_count, float2(0.f));
        }

        Surface::PointDescriptor point_desc(3);
        face_patch_points.resize(vertex_surface.GetNumPatchPoints() * point_desc.size);
        vertex_surface
            .PreparePatchPoints((float*)input.positions.data(), point_desc, face_patch_points.data(), point_desc);
        // Calculate positions and normals (using derivative of position)
        {
            float3 du, dv;
            for (int i = 0; i < coord_count; ++i) {
                vertex_surface.Evaluate(
                    (const float*)&tess_coords[i],
                    face_patch_points.data(),
                    point_desc,
                    (float*)&out_positions[i],
                    (float*)&du,
                    (float*)&dv
                );

                out_normals[i] = sgl::math::normalize(sgl::math::cross(du, dv));
                if (left_handed)
                    out_normals[i] = -out_normals[i];
            }
        }

        out_triangles.resize(tess_pattern.GetNumFacets());
        tess_pattern.GetFacets((int*)out_triangles.data());

        // Allocate vertices in final mesh then get streams to write to them
        size_t vertex_offset = result.allocate_vertices(coord_count);
        auto positions = result.position_stream();
        auto normals = result.normal_stream();
        auto uvs = result.texcoord_stream();

        for (int i = 0; i < coord_count; ++i) {
            positions[vertex_offset + i] = out_positions[i];
            normals[vertex_offset + i] = out_normals[i];
            uvs[vertex_offset + i] = out_uvs[i];
        }

        auto& active_subgeometry = result.subgeometries[subgeometry_mapping[face_index]];
        for (int i = 0; i < tess_pattern.GetNumFacets(); ++i) {
            active_subgeometry.indices.push_back(out_triangles[i] + (int)vertex_offset);
        }
    }

    return result;
}

} // namespace usd_importer
} // namespace falcor
