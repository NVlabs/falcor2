// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "usd_importer_mesh_adapter.h"
#include "usd_importer_utils.h"
#include "falcor2/core/error.h"
#include "falcor2/importers/importer.h"
#include "falcor2/importers/mesh_tessellator/mesh_tessellator.h"

BEGIN_DISABLE_USD_WARNINGS
#include <pxr/base/gf/vec2f.h>
#include <pxr/base/gf/vec3f.h>
#include <pxr/base/tf/token.h>
#include <pxr/base/vt/array.h>
#include <pxr/usd/usd/attribute.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usdGeom/primvarsAPI.h>
#include <pxr/usd/usdGeom/subset.h>
#include <pxr/usd/usdGeom/tokens.h>
END_DISABLE_USD_WARNINGS

#include <algorithm>
#include <array>
#include <cstddef>
#include <span>
#include <string>
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

MeshAttributeStream make_mesh_attribute_stream(
    ImporterMeshDefaultAttribute attribute,
    const pxr::TfToken& interpolation,
    const std::vector<float>& values,
    const pxr::VtArray<int>* indices = nullptr
)
{
    return {
        .attribute = std::move(attribute),
        .interpolation = get_mesh_interpolation(interpolation),
        .values = values,
        .indices = indices ? std::span<const int>(indices->cdata(), indices->size()) : std::span<const int>(),
    };
}

/// Look up a GetPrimvarName-style name (without "primvars:"); unlike USD's GetBaseName, preserve sub-namespaces.
pxr::UsdGeomPrimvar get_primvar_by_name(const pxr::UsdGeomMesh& mesh, std::string_view primvar_name)
{
    return pxr::UsdGeomPrimvarsAPI(mesh).FindPrimvarWithInheritance(pxr::TfToken(std::string(primvar_name)));
}

/// Component count for supported named float32 primvars, including USD role aliases; zero means unsupported.
uint32_t get_named_float_component_count(const pxr::SdfValueTypeName& type_name)
{
    using namespace pxr;

    if (type_name == SdfValueTypeNames->FloatArray)
        return 1;
    if (type_name == SdfValueTypeNames->Float2Array || type_name == SdfValueTypeNames->TexCoord2fArray)
        return 2;
    if (type_name == SdfValueTypeNames->Float3Array || type_name == SdfValueTypeNames->Point3fArray
        || type_name == SdfValueTypeNames->Normal3fArray || type_name == SdfValueTypeNames->Vector3fArray
        || type_name == SdfValueTypeNames->Color3fArray || type_name == SdfValueTypeNames->TexCoord3fArray)
        return 3;
    if (type_name == SdfValueTypeNames->Float4Array || type_name == SdfValueTypeNames->Color4fArray)
        return 4;
    return 0;
}

/// Pack vector elements contiguously for a Falcor MeshAttributeStream without expanding USD primvar indices.
template<typename T>
bool get_float_values(
    const pxr::UsdGeomPrimvar& primvar,
    uint32_t component_count,
    std::vector<float>& values,
    pxr::UsdTimeCode time
)
{
    pxr::VtArray<T> source;
    if (!primvar.Get(&source, time))
        return false;
    values.resize(source.size() * component_count);
    for (size_t value_index = 0; value_index < source.size(); ++value_index) {
        for (uint32_t component = 0; component < component_count; ++component)
            values[value_index * component_count + component] = source[value_index][component];
    }
    return true;
}

/// Read a supported named float32 primvar into the tessellator's scalar-packed value array.
bool get_named_float_values(
    const pxr::UsdGeomPrimvar& primvar,
    uint32_t component_count,
    std::vector<float>& values,
    pxr::UsdTimeCode time
)
{
    if (component_count == 1) {
        pxr::VtArray<float> source;
        if (!primvar.Get(&source, time))
            return false;
        values.assign(source.begin(), source.end());
        return true;
    }
    if (component_count == 2)
        return get_float_values<pxr::GfVec2f>(primvar, component_count, values, time);
    if (component_count == 3)
        return get_float_values<pxr::GfVec3f>(primvar, component_count, values, time);
    return get_float_values<pxr::GfVec4f>(primvar, component_count, values, time);
}

} // namespace

ImporterMesh convert_usd_mesh(
    const pxr::UsdGeomMesh& usd_mesh,
    const ResolveMaterial& resolve_material,
    const ImportOptions& import_options
)
{
    using namespace pxr;
    using namespace mesh_tessellator;

    const std::string mesh_name = usd_mesh.GetPath().GetString();
    UsdGeomPrimvarsAPI primvar_API(usd_mesh);

    TessellatorInputMesh input;
    input.properties = ImporterMesh::make_properties(import_options);
    read_import_property<int>(usd_mesh.GetPrim(), "refinementLevel", input.properties, "refinement_level");
    read_import_property<bool>(usd_mesh.GetPrim(), "falcor:recomputeNormals", input.properties, "recompute_normals");
    const int refinement_level = input.properties.get<int>("refinement_level");
    const auto& texcoord_primvar_names = import_options.usd_texcoord_primvar_names;
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

    UsdGeomPrimvar normals_primvar = primvar_API.FindPrimvarWithInheritance(TfToken("primvars:normals"));
    // USD-authored normals apply only to polygonal meshes. Subdivision schemes define their own normals, including
    // when refinement zero selects the control-cage representation below.
    const UsdAttribute normal_attribute = input.subdivision_scheme == SubdivisionScheme::none
        ? (normals_primvar && normals_primvar.HasValue() ? normals_primvar.GetAttr() : usd_mesh.GetNormalsAttr())
        : UsdAttribute();
    std::vector<UsdAttribute> sampled_attributes = {usd_mesh.GetPointsAttr()};
    if (normal_attribute)
        sampled_attributes.push_back(normal_attribute);
    const GeometrySampleTimes times(usd_mesh.GetPrim(), sampled_attributes, import_options.import_mesh_motion);
    const auto& sample_times = times.sample_times;
    // Only position and normal values vary across motion samples.
    const UsdTimeCode initial_time = sample_times.front();

    std::vector<VtArray<GfVec3f>> sampled_positions(sample_times.size());
    std::vector<MeshAttributeStream> position_samples(sampled_positions.size());
    for (size_t sample = 0; sample < sampled_positions.size(); ++sample) {
        usd_mesh.GetPointsAttr().Get(&sampled_positions[sample], sample_times[sample]);
        FALCOR_CHECK(
            sampled_positions[sample].size() == sampled_positions[0].size(),
            "Mesh motion '{}' changes point count",
            mesh_name
        );
        position_samples[sample] = make_mesh_attribute_stream(
            {ImporterSemantic::position, 0, 3},
            UsdGeomTokens->vertex,
            sampled_positions[sample]
        );
    }
    input.positions = position_samples;
    VtArray<int> face_vertex_counts;
    VtArray<int> face_vertex_indices;
    VtArray<int> hole_indices;
    read_fixed_indices(
        usd_mesh.GetFaceVertexCountsAttr(),
        face_vertex_counts,
        initial_time,
        import_options.import_mesh_motion
    );
    read_fixed_indices(
        usd_mesh.GetFaceVertexIndicesAttr(),
        face_vertex_indices,
        initial_time,
        import_options.import_mesh_motion
    );
    input.face_vertex_counts = std::span<const int>(face_vertex_counts.cdata(), face_vertex_counts.size());
    input.face_vertex_indices = std::span<const int>(face_vertex_indices.cdata(), face_vertex_indices.size());

    if (auto attr = usd_mesh.GetHoleIndicesAttr(); attr)
        read_fixed_indices(attr, hole_indices, initial_time, import_options.import_mesh_motion);
    input.hole_indices = std::span<const int>(hole_indices.cdata(), hole_indices.size());

    std::vector<std::string> secondary_primvar_names;
    secondary_primvar_names.reserve(texcoord_primvar_names.size());
    for (const std::string& configured_name : texcoord_primvar_names) {
        if (!configured_name.empty())
            secondary_primvar_names.emplace_back(strip_primvars_name(configured_name));
    }

    std::array<UsdGeomPrimvar, 8> uv_primvars;
    uv_primvars[0] = get_uv_primvar(usd_mesh, secondary_primvar_names);
    for (size_t slot = 0; slot < texcoord_primvar_names.size(); ++slot) {
        const std::string& configured_name = texcoord_primvar_names[slot];
        if (!configured_name.empty())
            uv_primvars[slot + 1] = get_primvar_by_name(usd_mesh, strip_primvars_name(configured_name));
    }

    // Primvars handled by the UV/normal paths must not also become named additional attributes below.
    std::vector<std::string> claimed_primvar_names;
    auto claim_primvar = [&](const UsdGeomPrimvar& primvar)
    {
        const std::string name = primvar.GetPrimvarName().GetString();
        if (std::find(claimed_primvar_names.begin(), claimed_primvar_names.end(), name) == claimed_primvar_names.end())
            claimed_primvar_names.push_back(name);
    };

    std::array<VtArray<GfVec2f>, 8> uv_values;
    std::array<VtArray<int>, 8> uv_indices;
    for (size_t slot = 0; slot < uv_primvars.size(); ++slot) {
        const UsdGeomPrimvar& uv_primvar = uv_primvars[slot];
        if (!uv_primvar || !uv_primvar.HasValue())
            continue;

        const int element_size = uv_primvar.GetElementSize();
        FALCOR_CHECK(
            element_size == 1,
            "UV{} primvar '{}' on mesh '{}' has unsupported elementSize {}; expected 1.",
            slot,
            uv_primvar.GetName().GetText(),
            usd_mesh.GetPath().GetText(),
            element_size
        );

        FALCOR_CHECK(
            has_uv_primvar_value(uv_primvar),
            "UV{} primvar '{}' on mesh '{}' has unsupported type '{}'.",
            slot,
            uv_primvar.GetName().GetText(),
            usd_mesh.GetPath().GetText(),
            uv_primvar.GetTypeName().GetAsToken().GetText()
        );

        claim_primvar(uv_primvar);
        get_uv_values(uv_primvar, uv_values[slot], initial_time);
        read_fixed_indices(
            uv_primvar.GetIndicesAttr(),
            uv_indices[slot],
            initial_time,
            import_options.import_mesh_motion
        );
        if (!uv_values[slot].empty()) {
            input.streams.push_back(make_mesh_attribute_stream(
                {ImporterSemantic::tex_coord, static_cast<uint32_t>(slot), 2},
                uv_primvar.GetInterpolation(),
                uv_values[slot],
                &uv_indices[slot]
            ));
        }
    }

    std::vector<VtArray<GfVec3f>> sampled_normals;
    std::vector<MeshAttributeStream> normal_samples;
    VtArray<int> normal_indices;
    if (normals_primvar && normals_primvar.HasValue())
        claim_primvar(normals_primvar);

    if (normal_attribute && normal_attribute.HasValue()) {
        TfToken normal_interpolation = usd_mesh.GetNormalsInterpolation();
        bool indexed = false;
        if (normals_primvar && normals_primvar.HasValue()) {
            const int element_size = normals_primvar.GetElementSize();
            FALCOR_CHECK(
                element_size == 1,
                "Normal primvar '{}' on mesh '{}' has unsupported elementSize {}; expected 1.",
                normals_primvar.GetName().GetText(),
                usd_mesh.GetPath().GetText(),
                element_size
            );

            normal_interpolation = normals_primvar.GetInterpolation();
            indexed = read_fixed_indices(
                normals_primvar.GetIndicesAttr(),
                normal_indices,
                initial_time,
                import_options.import_mesh_motion
            );
        }

        sampled_normals.resize(input.positions.size());
        normal_samples.reserve(input.positions.size());
        for (size_t sample = 0; sample < sampled_normals.size(); ++sample) {
            const bool has_normals = normal_attribute.Get(&sampled_normals[sample], sample_times[sample]);
            if (sample == 0 && sampled_normals[sample].empty())
                break;
            FALCOR_CHECK(
                has_normals && sampled_normals[sample].size() == sampled_normals[0].size(),
                "Mesh motion '{}' changes normal count",
                mesh_name
            );
            normal_samples.push_back(make_mesh_attribute_stream(
                {ImporterSemantic::normal, 0, 3},
                normal_interpolation,
                sampled_normals[sample],
                indexed ? &normal_indices : nullptr
            ));
        }
    }

    // Own the converted values/indices referenced by MeshAttributeStream spans until tessellation completes.
    struct NamedAttributeStorage {
        std::vector<float> values;
        VtArray<int> indices;
    };
    const std::vector<UsdGeomPrimvar> all_primvars = primvar_API.FindPrimvarsWithInheritance();
    std::vector<NamedAttributeStorage> named_attribute_storage;
    named_attribute_storage.reserve(all_primvars.size());
    for (const UsdGeomPrimvar& primvar : all_primvars) {
        const std::string name = primvar.GetPrimvarName().GetString();
        if (!primvar.HasValue()
            || std::find(claimed_primvar_names.begin(), claimed_primvar_names.end(), name)
                != claimed_primvar_names.end()
            || primvar.GetElementSize() != 1)
            continue;

        const uint32_t component_count = get_named_float_component_count(primvar.GetTypeName());
        if (component_count == 0)
            continue;

        NamedAttributeStorage& storage = named_attribute_storage.emplace_back();
        if (!get_named_float_values(primvar, component_count, storage.values, initial_time) || storage.values.empty()) {
            named_attribute_storage.pop_back();
            continue;
        }
        const bool indexed = read_fixed_indices(
            primvar.GetIndicesAttr(),
            storage.indices,
            initial_time,
            import_options.import_mesh_motion
        );
        input.streams.push_back(make_mesh_attribute_stream(
            {ImporterSemantic::unknown, 0, component_count, name},
            primvar.GetInterpolation(),
            storage.values,
            indexed ? &storage.indices : nullptr
        ));
    }

    std::vector<UsdGeomSubset> geom_subsets;
    for (const UsdGeomSubset& geom_subset : UsdShadeMaterialBindingAPI(usd_mesh).GetMaterialBindSubsets()) {
        TfToken element_type;
        if (geom_subset.GetElementTypeAttr().Get(&element_type) && element_type == UsdGeomTokens->face)
            geom_subsets.push_back(geom_subset);
    }
    struct SubgeometryStorage {
        std::string name;
        int material_index = -1;
        VtArray<int> face_indices;
    };
    std::vector<SubgeometryStorage> subgeometry_storage;
    subgeometry_storage.reserve(geom_subsets.size() + 1);

    for (auto& geom_subset : geom_subsets) {
        UsdAttribute indices_attr = geom_subset.GetIndicesAttr();
        if (indices_attr) {
            SubgeometryStorage& subgeometry = subgeometry_storage.emplace_back();
            read_fixed_indices(indices_attr, subgeometry.face_indices, initial_time, import_options.import_mesh_motion);
            subgeometry.name = geom_subset.GetPath().GetString();
            subgeometry.material_index = resolve_material(UsdShadeMaterialBindingAPI(geom_subset), mesh_name);
        }
    }

    VtArray<int> unassigned_indices
        = UsdGeomSubset::GetUnassignedIndices(geom_subsets, input.face_vertex_counts.size(), initial_time);
    if (!unassigned_indices.empty()) {
        SubgeometryStorage& subgeometry = subgeometry_storage.emplace_back();
        subgeometry.face_indices = std::move(unassigned_indices);
        subgeometry.name = mesh_name;
        subgeometry.material_index = resolve_material(UsdShadeMaterialBindingAPI(usd_mesh), mesh_name);
    }

    input.subgeometries.reserve(subgeometry_storage.size());
    for (SubgeometryStorage& source : subgeometry_storage) {
        input.subgeometries.push_back({
            .name = std::move(source.name),
            .material_index = source.material_index,
            .face_indices = std::span<const int>(source.face_indices.cdata(), source.face_indices.size()),
        });
    }

    // Refinement zero triangulates the control cage. For smooth subdivision schemes, match Storm's smooth-hull
    // shading instead of Falcor1's faceted missing-normal fallback. Bilinear and polygonal control cages remain
    // faceted.
    const bool generate_smooth_normals = refinement_level == 0
        && (input.subdivision_scheme == SubdivisionScheme::catmull_clark
            || input.subdivision_scheme == SubdivisionScheme::loop);
    std::vector<std::vector<float>> generated_normal_values;
    if (generate_smooth_normals) {
        generated_normal_values.resize(input.positions.size());
        normal_samples.resize(input.positions.size());
        for (size_t sample = 0; sample < input.positions.size(); ++sample) {
            calculate_smooth_normals(input, sample, generated_normal_values[sample]);
            normal_samples[sample] = {
                .attribute = {ImporterSemantic::normal, 0, 3},
                .interpolation = MeshInterpolation::varying,
                .values = generated_normal_values[sample],
            };
        }
    }
    input.normals = normal_samples;

    const bool has_authored_texcoords = std::any_of(
        input.streams.begin(),
        input.streams.end(),
        [](const MeshAttributeStream& stream)
        {
            return stream.attribute.semantic == ImporterSemantic::tex_coord && stream.attribute.index == 0;
        }
    );

    ImporterMesh result = tessellate(input);
    // Sample payloads are no longer needed during deduplication and tangent generation.
    sampled_positions.clear();
    sampled_normals.clear();
    generated_normal_values.clear();
    if (!result.position_stream().valid())
        return result;

    result.motion_times = times.motion_times;
    result.motion_interpolation_mode = times.interpolation;

    // Compact the tessellated streams before generating tangents. MikkTSpace may then split shared vertices where its
    // per-corner tangent frames differ, but it deliberately does not merge vertices itself.
    result.deduplicate_vertices();
    // The shared options pass prepares both base and samples before transferring tangents.
    if (!result.motion_times.empty() && result.properties.get<bool>("recompute_normals"))
        return result;
    if (has_authored_texcoords) {
        result.add_tangents_from_uvs();
    } else {
        result.add_tangents_from_normals();
    }

    return result;
}

} // namespace usd_importer
} // namespace falcor
