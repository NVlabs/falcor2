// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "usd_importer_mesh_adapter.h"
#include "usd_importer_utils.h"
#include "falcor2/core/error.h"
#include "falcor2/importers/mesh_tessellator/mesh_tessellator.h"

BEGIN_DISABLE_USD_WARNINGS
#include <pxr/base/gf/vec2f.h>
#include <pxr/base/gf/vec3f.h>
#include <pxr/base/tf/token.h>
#include <pxr/base/vt/array.h>
#include <pxr/usd/usd/attribute.h>
#include <pxr/usd/usdGeom/primvarsAPI.h>
#include <pxr/usd/usdGeom/subset.h>
#include <pxr/usd/usdGeom/tokens.h>
END_DISABLE_USD_WARNINGS

#include <algorithm>
#include <cstddef>
#include <span>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace falcor {
namespace usd_importer {
namespace {

using namespace mesh_tessellator;

MeshInterpolation get_mesh_interpolation(const pxr::TfToken& interpolation)
{
    if (interpolation == pxr::UsdGeomTokens->constant)
        return MeshInterpolation::constant;
    if (interpolation == pxr::UsdGeomTokens->uniform)
        return MeshInterpolation::uniform;
    if (interpolation == pxr::UsdGeomTokens->vertex)
        return MeshInterpolation::vertex;
    if (interpolation == pxr::UsdGeomTokens->varying)
        return MeshInterpolation::varying;
    if (interpolation == pxr::UsdGeomTokens->faceVarying)
        return MeshInterpolation::face_varying;
    FALCOR_THROW("Unsupported mesh stream interpolation mode '{}'", interpolation.GetString());
}

SubdivisionScheme get_subdivision_scheme(const pxr::TfToken& scheme, std::string_view mesh_name)
{
    if (scheme == pxr::UsdGeomTokens->none)
        return SubdivisionScheme::none;
    if (scheme == pxr::UsdGeomTokens->catmullClark)
        return SubdivisionScheme::catmull_clark;
    if (scheme == pxr::UsdGeomTokens->bilinear)
        return SubdivisionScheme::bilinear;
    if (scheme == pxr::UsdGeomTokens->loop)
        return SubdivisionScheme::loop;

    sgl::log_warn(
        "Unsupported subdivision scheme '{}' on mesh '{}'; using Catmull-Clark.",
        scheme.GetString(),
        mesh_name
    );
    return SubdivisionScheme::catmull_clark;
}

MeshOrientation get_mesh_orientation(const pxr::TfToken& orientation)
{
    // Match PxOsd: with optional validation disabled, every value other than rightHanded is treated as flipped.
    return orientation == pxr::UsdGeomTokens->rightHanded ? MeshOrientation::right_handed
                                                          : MeshOrientation::left_handed;
}

FaceVaryingLinearInterpolation
get_face_varying_linear_interpolation(const pxr::TfToken& interpolation, std::string_view mesh_name)
{
    if (interpolation == pxr::UsdGeomTokens->none || interpolation.IsEmpty())
        return FaceVaryingLinearInterpolation::none;
    if (interpolation == pxr::UsdGeomTokens->cornersOnly)
        return FaceVaryingLinearInterpolation::corners_only;
    if (interpolation == pxr::UsdGeomTokens->cornersPlus1)
        return FaceVaryingLinearInterpolation::corners_plus_1;
    if (interpolation == pxr::UsdGeomTokens->cornersPlus2)
        return FaceVaryingLinearInterpolation::corners_plus_2;
    if (interpolation == pxr::UsdGeomTokens->boundaries)
        return FaceVaryingLinearInterpolation::boundaries;
    if (interpolation == pxr::UsdGeomTokens->all)
        return FaceVaryingLinearInterpolation::all;

    sgl::log_warn(
        "Unknown face-varying interpolation rule '{}' on mesh '{}'; using the OpenSubdiv default.",
        interpolation.GetString(),
        mesh_name
    );
    return FaceVaryingLinearInterpolation::all;
}

VertexBoundaryInterpolation
get_vertex_boundary_interpolation(const pxr::TfToken& interpolation, std::string_view mesh_name)
{
    if (interpolation == pxr::UsdGeomTokens->none)
        return VertexBoundaryInterpolation::none;
    if (interpolation == pxr::UsdGeomTokens->edgeOnly)
        return VertexBoundaryInterpolation::edge_only;
    if (interpolation == pxr::UsdGeomTokens->edgeAndCorner || interpolation.IsEmpty())
        return VertexBoundaryInterpolation::edge_and_corner;

    sgl::log_warn(
        "Unknown vertex boundary interpolation rule '{}' on mesh '{}'; using the OpenSubdiv default.",
        interpolation.GetString(),
        mesh_name
    );
    return VertexBoundaryInterpolation::none;
}

template<typename T>
MeshAttributeStream make_mesh_attribute_stream(
    ImporterMeshDefaultAttribute attribute,
    const pxr::TfToken& interpolation,
    const pxr::VtArray<T>& values,
    const pxr::VtArray<int>* indices = nullptr
)
{
    static_assert(std::is_trivially_copyable_v<T>);
    static_assert(sizeof(T) % sizeof(float) == 0);
    static_assert(alignof(T) == alignof(float));
    constexpr size_t component_count = sizeof(T) / sizeof(float);
    FALCOR_CHECK(
        component_count == attribute.components,
        "Mesh attribute element has {} tightly packed float components but the stream declares {}",
        component_count,
        attribute.components
    );

    return {
        .attribute = attribute,
        .interpolation = get_mesh_interpolation(interpolation),
        // Gf float vectors are contiguous float aggregates on all supported platforms; the checks above make the
        // layout assumption explicit so the tessellator can borrow the USD value table without copying it.
        .values
        = std::span<const float>(reinterpret_cast<const float*>(values.cdata()), values.size() * component_count),
        .indices = indices ? std::span<const int>(indices->cdata(), indices->size()) : std::span<const int>(),
    };
}

} // namespace

ImporterMesh convert_usd_mesh(const pxr::UsdGeomMesh& usd_mesh, const ResolveMeshMaterial& resolve_material)
{
    using namespace pxr;
    using namespace mesh_tessellator;

    const std::string mesh_name = usd_mesh.GetPath().GetString();
    UsdGeomPrimvarsAPI primvar_API(usd_mesh);

    TessellatorInputMesh input;
    input.name = mesh_name;
    input.subdivision_scheme = get_subdivision_scheme(
        get_attribute(usd_mesh.GetSubdivisionSchemeAttr(), UsdGeomTokens->catmullClark),
        mesh_name
    );
    input.orientation = get_mesh_orientation(get_attribute(usd_mesh.GetOrientationAttr(), UsdGeomTokens->rightHanded));
    input.face_varying_linear_interpolation = get_face_varying_linear_interpolation(
        get_attribute(usd_mesh.GetFaceVaryingLinearInterpolationAttr(), UsdGeomTokens->cornersPlus1),
        mesh_name
    );
    input.vertex_boundary_interpolation = get_vertex_boundary_interpolation(
        get_attribute(usd_mesh.GetInterpolateBoundaryAttr(), UsdGeomTokens->edgeAndCorner),
        mesh_name
    );

    VtArray<GfVec3f> positions;
    VtArray<int> face_vertex_counts;
    VtArray<int> face_vertex_indices;
    VtArray<int> hole_indices;
    usd_mesh.GetPointsAttr().Get(&positions);
    input.positions = make_mesh_attribute_stream({ImporterSemantic::position, 0, 3}, UsdGeomTokens->vertex, positions);
    usd_mesh.GetFaceVertexCountsAttr().Get(&face_vertex_counts);
    usd_mesh.GetFaceVertexIndicesAttr().Get(&face_vertex_indices);
    input.face_vertex_counts = std::span<const int>(face_vertex_counts.cdata(), face_vertex_counts.size());
    input.face_vertex_indices = std::span<const int>(face_vertex_indices.cdata(), face_vertex_indices.size());

    if (auto attr = usd_mesh.GetHoleIndicesAttr(); attr)
        attr.Get(&hole_indices);
    input.hole_indices = std::span<const int>(hole_indices.cdata(), hole_indices.size());

    bool apply_override = true;
    if (auto attr = usd_mesh.GetPrim().GetAttribute(TfToken("refinementEnableOverride")); attr)
        attr.Get<bool>(&apply_override);
    if (apply_override) {
        if (auto attr = usd_mesh.GetPrim().GetAttribute(TfToken("refinementLevel")); attr)
            attr.Get<int>(&input.refinement_level);
    }

    VtArray<GfVec2f> uvs;
    VtArray<int> uv_indices;
    if (UsdGeomPrimvar uv_primvar = get_uv_primvar(usd_mesh); uv_primvar && uv_primvar.HasValue()) {
        const int element_size = uv_primvar.GetElementSize();
        FALCOR_CHECK(
            element_size == 1,
            "UV primvar '{}' on mesh '{}' has unsupported elementSize {}; expected 1.",
            uv_primvar.GetName().GetText(),
            usd_mesh.GetPath().GetText(),
            element_size
        );

        uv_primvar.Get(&uvs);
        uv_primvar.GetIndices(&uv_indices);
        if (!uvs.empty()) {
            input.streams.push_back(make_mesh_attribute_stream(
                {ImporterSemantic::tex_coord, 0, 2},
                uv_primvar.GetInterpolation(),
                uvs,
                &uv_indices
            ));
        }
    }

    VtArray<GfVec3f> normals;
    VtArray<int> normal_indices;
    if (UsdGeomPrimvar normals_primvar = primvar_API.GetPrimvar(TfToken("primvars:normals"));
        normals_primvar && normals_primvar.HasValue()) {
        const int element_size = normals_primvar.GetElementSize();
        FALCOR_CHECK(
            element_size == 1,
            "Normal primvar '{}' on mesh '{}' has unsupported elementSize {}; expected 1.",
            normals_primvar.GetName().GetText(),
            usd_mesh.GetPath().GetText(),
            element_size
        );

        normals_primvar.Get(&normals);
        const bool indexed = normals_primvar.GetIndices(&normal_indices);
        if (!normals.empty()) {
            input.streams.push_back(make_mesh_attribute_stream(
                {ImporterSemantic::normal, 0, 3},
                normals_primvar.GetInterpolation(),
                normals,
                indexed ? &normal_indices : nullptr
            ));
        }
    } else if (usd_mesh.GetNormalsAttr().IsAuthored()) {
        // Normals specified via the attribute cannot be indexed, so there is no need to flatten them.
        usd_mesh.GetNormalsAttr().Get(&normals);
        if (!normals.empty()) {
            input.streams.push_back(make_mesh_attribute_stream(
                {ImporterSemantic::normal, 0, 3},
                usd_mesh.GetNormalsInterpolation(),
                normals
            ));
        }
    }

    std::vector<UsdGeomSubset> geom_subsets;
    for (const UsdGeomSubset& geom_subset : UsdShadeMaterialBindingAPI(usd_mesh).GetMaterialBindSubsets()) {
        TfToken element_type;
        if (geom_subset.GetElementTypeAttr().Get(&element_type) && element_type == UsdGeomTokens->face)
            geom_subsets.push_back(geom_subset);
    }
    struct SubgeometryStorage {
        std::string name;
        std::string material_name;
        VtArray<int> face_indices;
    };
    std::vector<SubgeometryStorage> subgeometry_storage;
    subgeometry_storage.reserve(geom_subsets.size() + 1);

    for (auto& geom_subset : geom_subsets) {
        UsdAttribute indices_attr = geom_subset.GetIndicesAttr();
        if (indices_attr) {
            SubgeometryStorage& subgeometry = subgeometry_storage.emplace_back();
            indices_attr.Get(&subgeometry.face_indices);
            subgeometry.name = geom_subset.GetPath().GetString();
            subgeometry.material_name = resolve_material(UsdShadeMaterialBindingAPI(geom_subset), mesh_name);
        }
    }

    VtArray<int> unassigned_indices
        = UsdGeomSubset::GetUnassignedIndices(geom_subsets, input.face_vertex_counts.size());
    if (!unassigned_indices.empty()) {
        SubgeometryStorage& subgeometry = subgeometry_storage.emplace_back();
        subgeometry.face_indices = std::move(unassigned_indices);
        subgeometry.name = mesh_name;
        subgeometry.material_name = resolve_material(UsdShadeMaterialBindingAPI(usd_mesh), mesh_name);
    }

    input.subgeometries.reserve(subgeometry_storage.size());
    for (SubgeometryStorage& source : subgeometry_storage) {
        input.subgeometries.push_back({
            .name = std::move(source.name),
            .material_name = std::move(source.material_name),
            .face_indices = std::span<const int>(source.face_indices.cdata(), source.face_indices.size()),
        });
    }

    const bool has_authored_texcoords = std::any_of(
        input.streams.begin(),
        input.streams.end(),
        [](const MeshAttributeStream& stream)
        {
            return stream.attribute.semantic == ImporterSemantic::tex_coord && stream.attribute.index == 0;
        }
    );

    ImporterMesh result = tessellate(input);
    if (!result.position_stream().valid())
        return result;

    // Compact the tessellated streams before generating tangents. MikkTSpace may then split shared vertices where its
    // per-corner tangent frames differ, but it deliberately does not merge vertices itself.
    result.deduplicate_vertices();
    if (has_authored_texcoords) {
        result.add_tangents_from_uvs();
    } else {
        result.add_tangents_from_normals();
    }

    return result;
}

} // namespace usd_importer
} // namespace falcor
