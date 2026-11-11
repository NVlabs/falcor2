// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "usd_importer_curve_adapter.h"
#include "usd_importer_utils.h"
#include "falcor2/importers/importer.h"
#include "falcor2/importers/curve_tessellator/curve_tessellator.h"
#include <type_traits>
#include <sgl/core/logger.h>

namespace falcor::usd_importer {
namespace {
using namespace pxr;
using namespace curve_tessellator;

// Resolve precedence once so ignored builtin attributes cannot add times to the sample schedule.
struct CurveAttributeSource {
    pxr::UsdAttribute attribute;
    pxr::UsdGeomPrimvar primvar;
    pxr::TfToken mode;
    std::optional<size_t> value_count;

    CurveAttributeSource(
        const pxr::UsdGeomBasisCurves& curves,
        const char* name,
        const pxr::UsdAttribute& builtin,
        const pxr::TfToken& builtin_mode
    )
        : attribute(builtin)
        , mode(builtin_mode)
    {
        const auto candidate = pxr::UsdGeomPrimvarsAPI(curves).FindPrimvarWithInheritance(pxr::TfToken(name));
        if (candidate && candidate.HasValue()) {
            primvar = candidate;
            attribute = primvar.GetAttr();
            mode = primvar.GetInterpolation();
        }
    }

    void check_indices(pxr::UsdTimeCode time, bool import_motion) const
    {
        if (primvar) {
            FALCOR_CHECK(
                primvar.GetElementSize() == 1,
                "Curve attribute '{}' has invalid element size",
                attribute.GetPath().GetString()
            );
            pxr::VtIntArray indices;
            read_fixed_indices(primvar.GetIndicesAttr(), indices, time, import_motion);
        }
    }

    // Legacy round curves allow missing builtin widths to select their default radius.
    template<typename T>
    pxr::VtArray<T> read(pxr::UsdTimeCode time, bool allow_missing = false)
    {
        pxr::VtArray<T> values;
        if (attribute.HasValue())
            FALCOR_CHECK(
                attribute.Get(&values, time) || allow_missing,
                "Cannot read curve attribute '{}'",
                attribute.GetPath().GetString()
            );
        FALCOR_CHECK(
            !value_count || *value_count == values.size(),
            "Curve attribute '{}' changes value count",
            attribute.GetPath().GetString()
        );
        value_count = values.size();
        if (primvar)
            FALCOR_CHECK(
                primvar.ComputeFlattened(&values, time),
                "Curve attribute '{}' has invalid values or indices",
                attribute.GetPath().GetString()
            );
        return values;
    }
};

// Resolve inherited st and validate fixed indices before flattening its values.
pxr::UsdGeomPrimvar read_curve_texcoords(
    const pxr::UsdGeomBasisCurves& curves,
    pxr::UsdTimeCode time,
    bool import_motion,
    pxr::VtArray<pxr::GfVec2f>& values
)
{
    const auto st = pxr::UsdGeomPrimvarsAPI(curves).FindPrimvarWithInheritance(pxr::TfToken("st"));
    if (!st || !st.HasValue())
        return {};
    pxr::VtIntArray indices;
    read_fixed_indices(st.GetIndicesAttr(), indices, time, import_motion);
    FALCOR_CHECK(
        st.GetElementSize() == 1 && get_uv_values(st, values, time, true) && !values.empty(),
        "Curve '{}' has invalid st values or indices",
        curves.GetPath().GetString()
    );
    return st;
}

Interpolation interpolation(const TfToken& token, const std::string& name)
{
    if (token == UsdGeomTokens->constant)
        return Interpolation::constant;
    if (token == UsdGeomTokens->uniform)
        return Interpolation::uniform;
    if (token == UsdGeomTokens->vertex)
        return Interpolation::vertex;
    if (token == UsdGeomTokens->varying)
        return Interpolation::varying;
    FALCOR_THROW("Curve '{}' has unsupported interpolation '{}'", name, token.GetString());
}

template<typename T>
void copy_values(const pxr::VtArray<T>& source, auto& destination)
{
    destination.clear();
    destination.reserve(source.size());
    for (const auto& value : source)
        destination.push_back(to_falcor(value));
}
} // namespace

std::optional<CurveGeometry> convert_usd_curves(
    const pxr::UsdGeomBasisCurves& curves,
    const ResolveMaterial& resolve_material,
    const ImportOptions& options
)
{
    using namespace pxr;
    using namespace curve_tessellator;
    const auto prim = curves.GetPrim();
    CurveAttributeSource normals_source(curves, "normals", curves.GetNormalsAttr(), curves.GetNormalsInterpolation());
    CurveAttributeSource widths_source(curves, "widths", curves.GetWidthsAttr(), curves.GetWidthsInterpolation());
    const GeometrySampleTimes times(
        prim,
        {curves.GetPointsAttr(), normals_source.attribute, widths_source.attribute},
        options.import_curve_motion,
        UsdTimeCode::EarliestTime()
    );
    const auto& sample_times = times.sample_times;
    const UsdTimeCode initial_time = sample_times.front();
    normals_source.check_indices(initial_time, options.import_curve_motion);
    widths_source.check_indices(initial_time, options.import_curve_motion);
    auto first_normals = normals_source.read<GfVec3f>(initial_time);
    // Decide the output once. Later normal reads still enforce fixed attribute cardinality.
    const bool oriented = !first_normals.empty();
    CurveInput input;
    input.name = curves.GetPath().GetString();
    TfToken type, basis, wrap, orientation;
    curves.GetTypeAttr().Get(&type);
    curves.GetBasisAttr().Get(&basis);
    curves.GetWrapAttr().Get(&wrap);
    curves.GetOrientationAttr().Get(&orientation);
    if (type != UsdGeomTokens->linear && type != UsdGeomTokens->cubic) {
        FALCOR_CHECK(!oriented, "Ribbon '{}' has unsupported type", input.name);
        sgl::log_warn("Skipping BasisCurves prim '{}' with unsupported type '{}'", input.name, type.GetString());
        return std::nullopt;
    }
    input.type = type == UsdGeomTokens->linear ? Type::linear : Type::cubic;
    if (input.type == Type::cubic) {
        if (basis == UsdGeomTokens->bezier)
            input.basis = Basis::bezier;
        else if (basis == UsdGeomTokens->bspline)
            input.basis = Basis::bspline;
        else if (basis == UsdGeomTokens->catmullRom)
            input.basis = Basis::catmull_rom;
        else {
            FALCOR_CHECK(!oriented, "Ribbon '{}' has unsupported basis '{}'", input.name, basis.GetString());
            sgl::log_warn(
                "Skipping cubic BasisCurves prim '{}' with unsupported basis '{}'",
                input.name,
                basis.GetString()
            );
            return std::nullopt;
        }
    }
    if (wrap == UsdGeomTokens->periodic)
        input.wrap = Wrap::periodic;
    else if (wrap == UsdGeomTokens->pinned)
        input.wrap = Wrap::pinned;
    else
        FALCOR_CHECK(
            !oriented || wrap == UsdGeomTokens->nonperiodic,
            "Ribbon '{}' has unsupported wrap '{}'",
            input.name,
            wrap.GetString()
        );
    if (!oriented && input.type == Type::cubic && input.wrap == Wrap::periodic)
        sgl::log_warn(
            "BasisCurves prim '{}' uses periodic wrap mode which is not yet supported, treating as nonperiodic",
            input.name
        );
    input.left_handed = orientation == UsdGeomTokens->leftHanded;
    auto curve_properties = ImporterCurve::make_properties(options);
    read_import_property<int>(prim, "refinementLevel", curve_properties, "refinement_level");
    input.refinement_level = curve_properties.get<int>("refinement_level");
    input.properties = oriented ? ImporterMesh::make_properties(options) : curve_properties;
    if (oriented) {
        input.properties.set("refinement_level", input.refinement_level);
        read_import_property<bool>(prim, "falcor:recomputeNormals", input.properties, "recompute_normals");
    }
    input.remove_coincident_edges = sample_times.size() == 1;
    input.material_index = resolve_material(UsdShadeMaterialBindingAPI(curves), input.name);

    VtIntArray counts;
    const bool has_counts
        = read_fixed_indices(curves.GetCurveVertexCountsAttr(), counts, initial_time, options.import_curve_motion);
    FALCOR_CHECK(!oriented || has_counts, "Ribbon '{}' is missing vertex counts", input.name);
    if (!oriented && counts.empty()) {
        sgl::log_warn("Skipping BasisCurves prim '{}' with no vertex counts", input.name);
        return std::nullopt;
    }
    input.vertex_counts = {counts.cdata(), counts.size()};
    input.sample.widths.interpolation = interpolation(widths_source.mode, input.name);
    if (oriented)
        input.sample.normals.interpolation = interpolation(normals_source.mode, input.name);
    VtVec2fArray usd_uvs;
    std::vector<float2> uvs;
    const auto st = read_curve_texcoords(curves, initial_time, options.import_curve_motion, usd_uvs);
    if (st) {
        copy_values(usd_uvs, uvs);
        input.texcoords = {uvs, interpolation(st.GetInterpolation(), input.name)};
    }
    VtIntArray varying_counts;
    if (input.sample.widths.interpolation == Interpolation::varying
        || (oriented && input.sample.normals.interpolation == Interpolation::varying)
        || (st && input.texcoords.interpolation == Interpolation::varying)) {
        varying_counts = compute_curve_varying_counts(curves, initial_time);
        input.varying_counts = {varying_counts.cdata(), varying_counts.size()};
    }

    // One sample's ownership backs the borrowed input for both output writers.
    std::vector<float3> points, normals;
    VtFloatArray widths;
    std::optional<size_t> point_count;
    auto read_sample = [&](size_t sample)
    {
        const auto time = sample_times[sample];
        VtVec3fArray usd_points;
        const bool has_points = curves.GetPointsAttr().Get(&usd_points, time);
        FALCOR_CHECK(
            (!oriented && !options.import_curve_motion) || has_points,
            "Curve '{}' is missing points",
            input.name
        );
        FALCOR_CHECK(!point_count || *point_count == usd_points.size(), "Curve '{}' changes point count", input.name);
        point_count = usd_points.size();
        auto usd_normals = sample == 0 ? std::move(first_normals) : normals_source.read<GfVec3f>(time);
        copy_values(usd_points, points);
        copy_values(usd_normals, normals);
        widths = widths_source.read<float>(time, !oriented && !widths_source.primvar);
        FALCOR_CHECK(
            oriented || !widths_source.primvar || !widths.empty(),
            "Curve '{}' has invalid widths values or indices",
            input.name
        );
        input.sample.points = points;
        input.sample.normals.values = normals;
        input.sample.widths.values = {widths.cdata(), widths.size()};
    };

    auto convert = [&]<typename Output>(Output result) -> std::optional<CurveGeometry>
    {
        constexpr bool ribbon = std::is_same_v<Output, ImporterMesh>;
        for (uint32_t sample = 0; sample < sample_times.size(); ++sample) {
            read_sample(sample);
            if (sample == 0) {
                if constexpr (!ribbon) {
                    if (points.empty()) {
                        sgl::log_warn("Skipping BasisCurves prim '{}' with no points", input.name);
                        return std::nullopt;
                    }
                    if (st) {
                        size_t expected = 0;
                        switch (input.texcoords.interpolation) {
                        case Interpolation::constant:
                            expected = 1;
                            break;
                        case Interpolation::uniform:
                            expected = counts.size();
                            break;
                        case Interpolation::vertex:
                            expected = points.size();
                            break;
                        case Interpolation::varying:
                            for (int count : varying_counts) {
                                FALCOR_CHECK(count >= 0, "Curve '{}' has a negative varying sample count", input.name);
                                expected += count;
                            }
                            break;
                        }
                        FALCOR_CHECK(
                            uvs.size() == expected,
                            "Curve '{}' st count does not match its interpolation",
                            input.name
                        );
                    }
                    result = tessellate_curves(input, sample_times.size());
                    if (result.indices.empty())
                        return std::nullopt;
                } else {
                    result = tessellate_ribbons(input, sample_times.size());
                }
            } else {
                if constexpr (ribbon)
                    tessellate_ribbons(input, result.position_stream(sample), result.normal_stream(sample));
                else
                    tessellate_curves(input, result.position_stream(sample), result.radius_stream(sample));
            }
        }
        result.motion_times = times.motion_times;
        result.motion_sample_indices = times.motion_sample_indices;
        result.motion_interpolation_mode = times.interpolation;
        if constexpr (ribbon) {
            if (result.position_stream().valid()) {
                result.deduplicate_vertices();
                // Match the mesh adapter: recompute all sample normals before tangent transfer.
                if (!result.motion_times.empty() && result.properties.template get<bool>("recompute_normals"))
                    return result;
                if (uvs.empty())
                    result.add_tangents_from_uvs();
                else
                    result.add_tangents_from_normals();
            }
        } else {
            result.material_index = input.material_index;
        }
        return result;
    };
    return oriented ? convert(ImporterMesh{}) : convert(ImporterCurve{});
}
} // namespace falcor::usd_importer
