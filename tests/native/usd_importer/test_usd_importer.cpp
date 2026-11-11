// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#include "testing.h"
#include "falcor2/render/material/mtlx/codegen/codegen.h"
#include "falcor2/core/properties.h"
#include "falcor2/importers/importer.h"
#include "falcor2/importers/importer_types.h"
#include "falcor2/importers/material_conversions.h"
#include "falcor2/importers/usd_importer/usd_importer.h"
#include "falcor2/importers/usd_importer/usd_importer_utils.h"
#include "falcor2/core/object.h"

BEGIN_DISABLE_USD_WARNINGS
#include <pxr/base/vt/array.h>
#include <pxr/base/vt/value.h>
#include <pxr/base/tf/stringUtils.h>
#include <pxr/usd/sdf/assetPath.h>
#include <pxr/usd/sdf/layer.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usdGeom/basisCurves.h>
#include <pxr/usd/usdGeom/mesh.h>
END_DISABLE_USD_WARNINGS

#include <sgl/core/platform.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string_view>
#include <utility>

using namespace falcor;

TEST_SUITE_BEGIN("usd_importer");

TEST_CASE("USD refinement properties resolve defaults and authored levels")
{
    using namespace pxr;
    auto stage = UsdStage::CreateInMemory();
    auto mesh = UsdGeomMesh::Define(stage, SdfPath("/Mesh"));
    mesh.CreatePointsAttr().Set(VtVec3fArray{{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}});
    mesh.CreateFaceVertexCountsAttr().Set(VtIntArray{4});
    mesh.CreateFaceVertexIndicesAttr().Set(VtIntArray{0, 1, 2, 3});
    auto curves = UsdGeomBasisCurves::Define(stage, SdfPath("/Curves"));
    curves.CreateTypeAttr().Set(UsdGeomTokens->cubic);
    curves.CreateBasisAttr().Set(UsdGeomTokens->bspline);
    curves.CreateWrapAttr().Set(UsdGeomTokens->nonperiodic);
    curves.CreateCurveVertexCountsAttr().Set(VtIntArray{4});
    curves.CreatePointsAttr().Set(VtVec3fArray{{0, 0, 0}, {0, 1, 0}, {1, 1, 0}, {1, 0, 0}});
    curves.CreateWidthsAttr().Set(VtFloatArray{0.1f});
    curves.SetWidthsInterpolation(UsdGeomTokens->constant);

    auto check = [&](const ImportOptions& options, int mesh_level, int curve_level)
    {
        const auto scene = UsdImporter().import_stage(*stage, options);
        REQUIRE_EQ(scene->meshes.size(), 1);
        REQUIRE_EQ(scene->curves.size(), 1);
        CHECK_EQ(scene->meshes[0].properties.get<int>("refinement_level"), mesh_level);
        CHECK_EQ(scene->meshes[0].subgeometries[0].indices.size(), 2 * (mesh_level + 1) * (mesh_level + 1));
        const auto& curve = scene->curves[0];
        CHECK_EQ(curve.properties.get<int>("refinement_level"), curve_level);
        CHECK_EQ(curve.indices.size(), 2 * (curve_level == 0 ? 3 : curve_level + 1));
        CHECK(curve.position_stream()[0].y == doctest::Approx(curve_level == 0 ? 0.f : 5.f / 6.f));
    };
    check({}, 0, 0);
    ImportOptions options;
    options.mesh_refinement_level = 2;
    options.curve_refinement_level = 7;
    check(options, 2, 7);
    auto mesh_level = mesh.GetPrim().CreateAttribute(TfToken("refinementLevel"), SdfValueTypeNames->Int);
    auto curve_level = curves.GetPrim().CreateAttribute(TfToken("refinementLevel"), SdfValueTypeNames->Int);
    mesh.GetPrim().CreateAttribute(TfToken("refinementEnableOverride"), SdfValueTypeNames->Bool).Set(false);
    check(options, 2, 7); // Declarations without values retain independent defaults.
    for (int level : {0, 1, 2, 3, 4, 63, 64, 255, 256}) {
        CAPTURE(level);
        mesh_level.Set(level);
        curve_level.Set(level);
        check(options, level, level);
    }
    mesh_level.Block();
    curve_level.Block();
    check(options, 2, 7);

    // Linear curves retain authored coincident edges; polygon meshes stay polygons.
    mesh.CreateSubdivisionSchemeAttr().Set(UsdGeomTokens->none);
    curves.CreateTypeAttr().Set(UsdGeomTokens->linear);
    curves.CreatePointsAttr().Set(VtVec3fArray{{0, 0, 0}, {0, 0, 0}, {1, 1, 0}, {1, 0, 0}});
    const auto scene = UsdImporter().import_stage(*stage, options);
    CHECK_EQ(scene->meshes[0].subgeometries[0].indices.size(), 2);
    CHECK_EQ(scene->curves[0].indices.size(), 6);

    for (int invalid : {-1, 257}) {
        CAPTURE(invalid);
        mesh_level.Set(invalid);
        CHECK_THROWS(UsdImporter().import_stage(*stage, options));
        mesh_level.Set(0);
        curve_level.Set(invalid);
        CHECK_THROWS(UsdImporter().import_stage(*stage, options));
        curve_level.Set(0);
        CHECK_THROWS(UsdImporter().import_stage(*stage, {.mesh_refinement_level = invalid}));
        CHECK_THROWS(UsdImporter().import_stage(*stage, {.curve_refinement_level = invalid}));
    }
    mesh.GetPrim().RemoveProperty(TfToken("refinementLevel"));
    mesh.GetPrim().CreateAttribute(TfToken("refinementLevel"), SdfValueTypeNames->Float).Set(2.f);
    CHECK_THROWS_WITH(UsdImporter().import_stage(*stage), doctest::Contains("refinementLevel"));
}

TEST_CASE("USD per-object processing properties override importer defaults")
{
    using namespace pxr;
    auto stage = UsdStage::CreateInMemory();
    REQUIRE(stage->GetRootLayer()->ImportFromString(R"usd(#usda 1.0
def Mesh "Mesh" {
    int[] faceVertexCounts = [3]
    int[] faceVertexIndices = [0, 1, 2]
    point3f[] points = [(0,0,0), (1,0,0), (0,1,0)]
    normal3f[] normals = [(1,0,0), (1,0,0), (1,0,0)] (interpolation = "vertex")
    uniform token subdivisionScheme = "none"
}
def RectLight "Rect" {}
def CylinderLight "Cylinder" {
    custom bool falcor:convertToMesh = true
}
def SphereLight "Point" {
    bool treatAsPoint = true
    custom bool falcor:convertToMesh = true
}
def DiskLight "Shaped" (prepend apiSchemas = ["ShapingAPI"]) {
    float inputs:shaping:cone:angle = 30
    custom bool falcor:convertToMesh = true
}
)usd"));
    auto normals = stage->GetPrimAtPath(SdfPath("/Mesh"))
                       .CreateAttribute(TfToken("falcor:recomputeNormals"), SdfValueTypeNames->Bool);
    auto convert = stage->GetPrimAtPath(SdfPath("/Rect"))
                       .CreateAttribute(TfToken("falcor:convertToMesh"), SdfValueTypeNames->Bool);
    for (bool authored : {false, true}) {
        CAPTURE(authored);
        normals.Set(authored);
        convert.Set(authored);
        ImportOptions options;
        options.recompute_normals = !authored;
        options.force_rectangle_lights_to_geometry = !authored;
        const auto scene = UsdImporter().import_stage(*stage, options);
        CHECK_EQ(scene->meshes.size(), authored ? 2 : 1);
        CHECK_EQ(scene->lights.size(), authored ? 3 : 4);
        const auto& mesh = scene->meshes[0];
        CHECK_EQ(mesh.properties.get<bool>("recompute_normals"), authored);
        CHECK(mesh.normal_stream()[0].z == doctest::Approx(authored ? 1.f : 0.f));
        if (authored)
            CHECK_FALSE(scene->meshes[1].properties.get<bool>("recompute_normals"));
    }
    normals.Block();
    convert.Block();
    const auto scene
        = UsdImporter().import_stage(*stage, {.recompute_normals = true, .force_rectangle_lights_to_geometry = true});
    REQUIRE_EQ(scene->meshes.size(), 2);
    CHECK(scene->meshes[1].properties.get<bool>("recompute_normals"));
    CHECK(scene->meshes[0].normal_stream()[0].z == doctest::Approx(1.f));
    for (const auto& path : {SdfPath("/Mesh"), SdfPath("/Rect")}) {
        auto prim = stage->GetPrimAtPath(path);
        const TfToken name(path == SdfPath("/Mesh") ? "falcor:recomputeNormals" : "falcor:convertToMesh");
        prim.RemoveProperty(name);
        prim.CreateAttribute(name, SdfValueTypeNames->Int).Set(1);
        CHECK_THROWS_WITH(UsdImporter().import_stage(*stage), doctest::Contains(name.GetText()));
        prim.RemoveProperty(name);
    }
}

TEST_CASE("curves_primary_uv_interpolation_and_direct_control_points")
{
    using namespace pxr;
    for (bool direct : {false, true}) {
        for (const TfToken& basis : {UsdGeomTokens->bezier, UsdGeomTokens->bspline, UsdGeomTokens->catmullRom}) {
            for (const TfToken& interpolation :
                 {UsdGeomTokens->constant, UsdGeomTokens->uniform, UsdGeomTokens->vertex, UsdGeomTokens->varying}) {
                CAPTURE(direct);
                CAPTURE(basis.GetString());
                CAPTURE(interpolation.GetString());
                auto stage = UsdStage::CreateInMemory();
                auto curves = UsdGeomBasisCurves::Define(stage, SdfPath("/Curves"));
                curves.CreateTypeAttr().Set(UsdGeomTokens->cubic);
                curves.CreateBasisAttr().Set(basis);
                curves.CreateWrapAttr().Set(UsdGeomTokens->nonperiodic);
                curves.CreateCurveVertexCountsAttr().Set(VtIntArray{4, 4});
                curves.CreatePointsAttr().Set(
                    VtVec3fArray{{0, 0, 0}, {1, 0, 0}, {2, 0, 0}, {3, 0, 0}, {0, 1, 0}, {1, 1, 0}, {2, 1, 0}, {3, 1, 0}}
                );
                curves.CreateWidthsAttr().Set(VtFloatArray{0.2f, 0.2f, 0.2f, 0.2f, 0.4f, 0.4f, 0.4f, 0.4f});
                curves.SetWidthsInterpolation(UsdGeomTokens->vertex);
                auto st = UsdGeomPrimvarsAPI(curves)
                              .CreatePrimvar(TfToken("st"), SdfValueTypeNames->TexCoord2dArray, interpolation);
                VtVec2dArray values;
                if (interpolation == UsdGeomTokens->constant)
                    values = {{0.3, 0.7}};
                else if (interpolation == UsdGeomTokens->uniform)
                    values = {{0.2, 0.8}, {0.6, 0.4}};
                else if (interpolation == UsdGeomTokens->varying)
                    values = {{0, 0.1}, {1, 0.9}, {2, 0.2}, {3, 0.8}};
                else
                    values = {{0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {1, 0}, {0, 0}};
                st.Set(values);
                VtIntArray indices(values.size());
                for (size_t i = 0; i < indices.size(); ++i)
                    indices[i] = int(indices.size() - 1 - i);
                st.SetIndices(indices);
                const auto scene = UsdImporter().import_stage(*stage, {.curve_refinement_level = direct ? 0 : 7});
                REQUIRE(scene->curves.size() == 1);
                const auto& curve = scene->curves[0];
                const size_t strand_size = direct ? 4 : 9;
                REQUIRE(curve.texcoords.size() == 2 * strand_size);
                CHECK(curve.indices.size() == 4 * (strand_size - 1));
                CHECK(curve.radius_stream()[0] == doctest::Approx(0.1f));
                CHECK(curve.radius_stream()[strand_size] == doctest::Approx(0.2f));
                if (interpolation == UsdGeomTokens->constant)
                    CHECK(curve.texcoords.back().x == doctest::Approx(0.3));
                else if (interpolation == UsdGeomTokens->uniform) {
                    CHECK(curve.texcoords.front().x == doctest::Approx(0.6));
                    CHECK(curve.texcoords.back().x == doctest::Approx(0.2));
                } else if (interpolation == UsdGeomTokens->varying) {
                    CHECK(curve.texcoords.front().x == doctest::Approx(3));
                    CHECK(curve.texcoords[strand_size - 1].x == doctest::Approx(2));
                    CHECK(curve.texcoords[strand_size].x == doctest::Approx(1));
                    CHECK(curve.texcoords.back().x == doctest::Approx(0));
                    CHECK(curve.texcoords[1].x == doctest::Approx(direct ? 3.f - 1.f / 3 : 3.f - 1.f / 8));
                } else {
                    const bool bezier = basis == UsdGeomTokens->bezier;
                    const bool bspline = basis == UsdGeomTokens->bspline;
                    CHECK(
                        curve.texcoords.front().x
                        == doctest::Approx(
                            direct || bezier ? 0.f
                                : bspline    ? 2.f / 3
                                             : 1.f
                        )
                    );
                    CHECK(curve.texcoords.back().x == 0.f);
                    // Indexed first-strand values are [0, 1, 0, 0]. At t=1/2 the
                    // second control point has these basis weights, not a linear endpoint blend.
                    const float midpoint = bezier ? 3.f / 8 : bspline ? 23.f / 48 : 9.f / 16;
                    CHECK(curve.texcoords[direct ? 1 : 4].x == doctest::Approx(direct ? 1.f : midpoint));
                }
            }
        }
    }
}

TEST_CASE("curves_varying_counts_follow_schema_topology")
{
    using namespace pxr;
    auto stage = UsdStage::CreateInMemory();
    auto curves = UsdGeomBasisCurves::Define(stage, SdfPath("/Curves"));
    auto check = [&](const TfToken& type,
                     const TfToken& basis,
                     const TfToken& wrap,
                     const VtIntArray& vertices,
                     const VtIntArray& expected)
    {
        CAPTURE(type.GetString());
        CAPTURE(basis.GetString());
        CAPTURE(wrap.GetString());
        curves.CreateTypeAttr().Set(type);
        curves.CreateBasisAttr().Set(basis);
        curves.CreateWrapAttr().Set(wrap);
        curves.CreateCurveVertexCountsAttr().Set(vertices);
        CHECK(usd_importer::compute_curve_varying_counts(curves, UsdTimeCode::Default()) == expected);
    };
    for (const auto& wrap : {UsdGeomTokens->nonperiodic, UsdGeomTokens->periodic, UsdGeomTokens->pinned}) {
        check(UsdGeomTokens->linear, UsdGeomTokens->bezier, wrap, {2, 3, 7}, {2, 3, 7});
        const bool periodic = wrap == UsdGeomTokens->periodic;
        check(
            UsdGeomTokens->cubic,
            UsdGeomTokens->bezier,
            wrap,
            periodic ? VtIntArray{3, 6, 9} : VtIntArray{4, 7, 10},
            {2 - int(periodic), 3 - int(periodic), 4 - int(periodic)}
        );
        for (const auto& basis : {UsdGeomTokens->bspline, UsdGeomTokens->catmullRom}) {
            if (wrap == UsdGeomTokens->pinned)
                check(UsdGeomTokens->cubic, basis, wrap, {2, 3, 4, 7}, {2, 3, 4, 7});
            else
                check(
                    UsdGeomTokens->cubic,
                    basis,
                    wrap,
                    {4, 5, 7},
                    periodic ? VtIntArray{4, 5, 7} : VtIntArray{2, 3, 5}
                );
        }
    }
    curves.GetCurveVertexCountsAttr().Set(VtIntArray{3, 7}, UsdTimeCode(1));
    CHECK(usd_importer::compute_curve_varying_counts(curves, UsdTimeCode(1)) == VtIntArray{3, 7});
    curves.GetCurveVertexCountsAttr().Set(VtIntArray{});
    CHECK(usd_importer::compute_curve_varying_counts(curves, UsdTimeCode::Default()).empty());
}

TEST_CASE("curves_periodic_varying_offsets_and_defaults")
{
    using namespace pxr;
    for (const TfToken& wrap : {UsdGeomTokens->periodic, UsdGeomTokens->pinned}) {
        auto stage = UsdStage::CreateInMemory();
        auto curves = UsdGeomBasisCurves::Define(stage, SdfPath("/Curves"));
        curves.CreateTypeAttr().Set(UsdGeomTokens->cubic);
        curves.CreateBasisAttr().Set(UsdGeomTokens->bspline);
        curves.CreateWrapAttr().Set(wrap);
        curves.CreateCurveVertexCountsAttr().Set(VtIntArray{4, 5});
        curves.CreatePointsAttr().Set(
            VtVec3fArray{
                {0, 0, 0},
                {1, 0, 0},
                {2, 0, 0},
                {3, 0, 0},
                {0, 1, 0},
                {1, 1, 0},
                {2, 1, 0},
                {3, 1, 0},
                {4, 1, 0}
            }
        );
        const bool periodic = wrap == UsdGeomTokens->periodic;
        const int first_count = 4;
        const int second_count = 5;
        VtVec2hArray values;
        for (int i = 0; i < first_count; ++i)
            values.push_back(GfVec2h(float(i), 0.5f));
        for (int i = 0; i < second_count; ++i)
            values.push_back(GfVec2h(float(10 + i), 0.5f));
        auto st = UsdGeomPrimvarsAPI(curves)
                      .CreatePrimvar(TfToken("st"), SdfValueTypeNames->TexCoord2hArray, UsdGeomTokens->varying);
        st.Set(values);
        for (bool direct : {false, true}) {
            const auto scene = UsdImporter().import_stage(*stage, {.curve_refinement_level = direct ? 0 : 7});
            REQUIRE(scene->curves.size() == 1);
            const auto& curve = scene->curves[0];
            CHECK(curve.texcoords[direct ? 4 : (periodic ? 9 : 25)].x == doctest::Approx(10));
            CHECK(curve.indices.size() == (direct ? 14 : periodic ? 48 : 112));
            CHECK(curve.texcoords[1].x == doctest::Approx(direct ? 1.f : 1.f / 8));
        }
        st.GetAttr().Clear();
        const auto generated = UsdImporter().import_stage(*stage, {.curve_refinement_level = 7});
        CHECK(generated->curves[0].texcoords[0].x == 0.f);
        CHECK(generated->curves[0].texcoords[periodic ? 8 : 24].x == 1.f);
        CHECK(generated->curves[0].texcoords[periodic ? 9 : 25].x == 0.f);
        CHECK(generated->curves[0].texcoords.back().x == 1.f);
    }
}

TEST_CASE("curves_pinned_varying_widths_select_each_strand")
{
    using namespace pxr;
    for (const TfToken& basis : {UsdGeomTokens->bspline, UsdGeomTokens->catmullRom}) {
        auto stage = UsdStage::CreateInMemory();
        auto curves = UsdGeomBasisCurves::Define(stage, SdfPath("/Curves"));
        curves.CreateTypeAttr().Set(UsdGeomTokens->cubic);
        curves.CreateBasisAttr().Set(basis);
        curves.CreateWrapAttr().Set(UsdGeomTokens->pinned);
        curves.CreateCurveVertexCountsAttr().Set(VtIntArray{4, 5});
        curves.CreatePointsAttr().Set(
            VtVec3fArray{
                {0, 0, 0},
                {1, 0, 0},
                {2, 0, 0},
                {3, 0, 0},
                {0, 1, 0},
                {1, 1, 0},
                {2, 1, 0},
                {3, 1, 0},
                {4, 1, 0}
            }
        );
        curves.CreateWidthsAttr().Set(VtFloatArray{0.2f, 0.4f, 0.6f, 0.8f, 2.f, 2.2f, 2.4f, 2.6f, 2.8f});
        curves.SetWidthsInterpolation(UsdGeomTokens->varying);
        for (int level : {0, 3}) {
            CAPTURE(basis.GetString());
            CAPTURE(level);
            const auto scene = UsdImporter().import_stage(*stage, {.curve_refinement_level = level});
            REQUIRE(scene->curves.size() == 1);
            const auto& curve = scene->curves[0];
            const size_t first_count = level == 0 ? 4 : 13;
            REQUIRE(curve.radius_stream().size == (level == 0 ? 9 : 30));
            for (size_t i = 0; i < curve.radius_stream().size; ++i) {
                const size_t j = i < first_count ? i : i - first_count;
                const float sample = level == 0 ? float(j) : float(j) / (level + 1);
                const float base = i < first_count ? 0.1f : 1.f;
                CHECK(curve.radius_stream()[i] == doctest::Approx(base + 0.1f * sample));
            }
        }
    }
}

TEST_CASE("curves_pinned_endpoints_match_reflected_controls")
{
    using namespace pxr;
    for (const auto& basis : {UsdGeomTokens->bspline, UsdGeomTokens->catmullRom}) {
        for (int count : {2, 3, 4, 7}) {
            for (const auto& interpolation :
                 {UsdGeomTokens->constant, UsdGeomTokens->uniform, UsdGeomTokens->vertex, UsdGeomTokens->varying}) {
                for (int level : {1, 3}) {
                    CAPTURE(basis.GetString());
                    CAPTURE(count);
                    CAPTURE(interpolation.GetString());
                    CAPTURE(level);
                    auto stage = UsdStage::CreateInMemory();
                    auto curves = UsdGeomBasisCurves::Define(stage, SdfPath("/Curves"));
                    curves.CreateTypeAttr().Set(UsdGeomTokens->cubic);
                    curves.CreateBasisAttr().Set(basis);
                    curves.CreateWrapAttr().Set(UsdGeomTokens->pinned);
                    curves.CreateCurveVertexCountsAttr().Set(VtIntArray{count});
                    VtVec3fArray points;
                    VtFloatArray widths;
                    VtVec2fArray uvs;
                    for (int i = 0; i < count; ++i) {
                        points.push_back(GfVec3f(float(i), i % 2 ? 2.f : -1.f, 0.25f * i * i));
                        widths.push_back(2.f + 0.2f * (i % 3));
                        uvs.push_back(GfVec2f(0.3f * (i % 3), 0.2f * i));
                    }
                    if (interpolation == UsdGeomTokens->constant || interpolation == UsdGeomTokens->uniform) {
                        widths.resize(1);
                        uvs.resize(1);
                    }
                    curves.CreatePointsAttr().Set(points);
                    curves.CreateWidthsAttr().Set(widths);
                    curves.SetWidthsInterpolation(interpolation);
                    auto st = UsdGeomPrimvarsAPI(curves)
                                  .CreatePrimvar(TfToken("st"), SdfValueTypeNames->TexCoord2fArray, interpolation);
                    st.Set(uvs);
                    const auto scene = UsdImporter().import_stage(*stage, {.curve_refinement_level = level});
                    REQUIRE_EQ(scene->curves.size(), 1);
                    const auto& curve = scene->curves[0];
                    const size_t output_count = (count - 1) * (level + 1) + 1;
                    REQUIRE_EQ(curve.position_stream().size, output_count);
                    REQUIRE_EQ(curve.indices.size(), 2 * (output_count - 1));
                    // Independent endpoint and first-span midpoint formulas catch missing or duplicated ends.
                    const GfVec3f next = count > 2 ? points[2] : 2.f * points[1] - points[0];
                    const GfVec3f midpoint = basis == UsdGeomTokens->bspline
                        ? (25.f * points[0] + 22.f * points[1] + next) / 48.f
                        : (7.f * points[0] + 10.f * points[1] - next) / 16.f;
                    for (int c = 0; c < 3; ++c) {
                        CHECK(curve.position_stream()[0][c] == doctest::Approx(points.front()[c]));
                        CHECK(curve.position_stream()[output_count - 1][c] == doctest::Approx(points.back()[c]));
                        CHECK(curve.position_stream()[(level + 1) / 2][c] == doctest::Approx(midpoint[c]));
                    }
                    CHECK(curve.radius_stream()[0] == doctest::Approx(0.5f * widths.front()));
                    CHECK(curve.radius_stream()[output_count - 1] == doctest::Approx(0.5f * widths.back()));
                    for (int c = 0; c < 2; ++c) {
                        CHECK(curve.texcoords.front()[c] == doctest::Approx(uvs.front()[c]));
                        CHECK(curve.texcoords.back()[c] == doctest::Approx(uvs.back()[c]));
                    }
                    if (interpolation == UsdGeomTokens->varying) {
                        for (size_t i = 0; i < output_count; ++i) {
                            const size_t a = std::min(i / (level + 1), size_t(count - 2));
                            const float t = float(i) / (level + 1) - float(a);
                            CHECK(
                                curve.radius_stream()[i]
                                == doctest::Approx(0.5f * ((1.f - t) * widths[a] + t * widths[a + 1]))
                            );
                            CHECK(curve.texcoords[i].x == doctest::Approx((1.f - t) * uvs[a][0] + t * uvs[a + 1][0]));
                        }
                    }
                    // Explicit reflected nonperiodic data is an equivalent representation.
                    VtVec3fArray expanded_points{2.f * points[0] - points[1]};
                    expanded_points.insert(expanded_points.end(), points.begin(), points.end());
                    expanded_points.push_back(2.f * points[count - 1] - points[count - 2]);
                    curves.GetPointsAttr().Set(expanded_points);
                    curves.GetCurveVertexCountsAttr().Set(VtIntArray{count + 2});
                    curves.GetWrapAttr().Set(UsdGeomTokens->nonperiodic);
                    if (interpolation == UsdGeomTokens->vertex) {
                        VtFloatArray expanded_widths{2.f * widths[0] - widths[1]};
                        expanded_widths.insert(expanded_widths.end(), widths.begin(), widths.end());
                        expanded_widths.push_back(2.f * widths[count - 1] - widths[count - 2]);
                        curves.GetWidthsAttr().Set(expanded_widths);
                        VtVec2fArray expanded_uvs{2.f * uvs[0] - uvs[1]};
                        expanded_uvs.insert(expanded_uvs.end(), uvs.begin(), uvs.end());
                        expanded_uvs.push_back(2.f * uvs[count - 1] - uvs[count - 2]);
                        st.Set(expanded_uvs);
                    }
                    const auto expanded = UsdImporter().import_stage(*stage, {.curve_refinement_level = level});
                    REQUIRE_EQ(expanded->curves.size(), 1);
                    const auto& reference = expanded->curves[0];
                    REQUIRE_EQ(reference.position_stream().size, output_count);
                    CHECK(curve.indices == reference.indices);
                    for (size_t i = 0; i < output_count; ++i) {
                        for (int c = 0; c < 3; ++c)
                            CHECK(curve.position_stream()[i][c] == doctest::Approx(reference.position_stream()[i][c]));
                        CHECK(curve.radius_stream()[i] == doctest::Approx(reference.radius_stream()[i]));
                        for (int c = 0; c < 2; ++c)
                            CHECK(curve.texcoords[i][c] == doctest::Approx(reference.texcoords[i][c]));
                    }
                }
            }
        }
    }
}

TEST_CASE("curves_pinned_short_strands_and_indexed_varying_motion")
{
    using namespace pxr;
    for (const auto& basis : {UsdGeomTokens->bspline, UsdGeomTokens->catmullRom}) {
        CAPTURE(basis.GetString());
        auto stage = UsdStage::CreateInMemory();
        auto curves = UsdGeomBasisCurves::Define(stage, SdfPath("/Curves"));
        curves.CreateTypeAttr().Set(UsdGeomTokens->cubic);
        curves.CreateBasisAttr().Set(basis);
        curves.CreateWrapAttr().Set(UsdGeomTokens->pinned);
        const VtIntArray counts{2, 3, 4, 7};
        curves.CreateCurveVertexCountsAttr().Set(counts);
        VtVec3fArray points;
        VtFloatArray values;
        VtVec2fArray uvs;
        VtIntArray indices;
        for (size_t strand = 0; strand < counts.size(); ++strand) {
            for (int i = 0; i < counts[strand]; ++i) {
                points.push_back(GfVec3f(float(i), float(strand), 0.f));
                values.push_back(0.2f * float(values.size() + 1));
                uvs.push_back(GfVec2f(float(uvs.size()), 0.5f));
                indices.push_back(int(indices.size()));
            }
        }
        std::reverse(indices.begin(), indices.end());
        curves.CreatePointsAttr().Set(points, UsdTimeCode(1));
        for (auto& point : points)
            point[2] = 0.25f * point[0] * point[0];
        curves.GetPointsAttr().Set(points, UsdTimeCode(2));
        auto widths = UsdGeomPrimvarsAPI(curves)
                          .CreatePrimvar(TfToken("widths"), SdfValueTypeNames->FloatArray, UsdGeomTokens->varying);
        widths.SetIndices(indices);
        widths.Set(values, UsdTimeCode(1));
        for (float& value : values)
            value *= 2.f;
        widths.Set(values, UsdTimeCode(2));
        auto st = UsdGeomPrimvarsAPI(curves)
                      .CreatePrimvar(TfToken("st"), SdfValueTypeNames->TexCoord2fArray, UsdGeomTokens->varying);
        st.Set(uvs);
        st.SetIndices(indices);
        for (int level : {0, 3}) {
            CAPTURE(level);
            const auto scene
                = UsdImporter().import_stage(*stage, {.import_curve_motion = true, .curve_refinement_level = level});
            REQUIRE_EQ(scene->curves.size(), 1);
            const auto& curve = scene->curves[0];
            REQUIRE_EQ(curve.sample_count(), 2);
            REQUIRE_EQ(curve.radius_stream().size, level == 0 ? 16 : 52);
            size_t input_offset = 0;
            size_t output_offset = 0;
            for (int count : counts) {
                // Every authored span, including the two endpoint spans, is retained.
                const int output_count = level == 0 ? count : (count - 1) * 4 + 1;
                for (int j = 0; j < output_count; ++j) {
                    const float boundary = float(input_offset) + (level == 0 ? float(j) : float(j) / 4.f);
                    const float value = 15.f - boundary; // Reversed primvar indices.
                    CHECK(curve.radius_stream(0)[output_offset] == doctest::Approx(0.1f * (value + 1)));
                    CHECK(curve.radius_stream(1)[output_offset] == doctest::Approx(0.2f * (value + 1)));
                    CHECK(curve.texcoords[output_offset].x == doctest::Approx(value));
                    if (j == 0 || j == output_count - 1) {
                        const auto& endpoint = points[input_offset + (j == 0 ? 0 : count - 1)];
                        for (int c = 0; c < 3; ++c) {
                            CHECK(
                                curve.position_stream(0)[output_offset][c]
                                == doctest::Approx(c == 2 ? 0.f : endpoint[c])
                            );
                            CHECK(curve.position_stream(1)[output_offset][c] == doctest::Approx(endpoint[c]));
                        }
                    }
                    ++output_offset;
                }
                input_offset += count;
            }
        }
        // A truncated legacy N-2 array must not shift the following strand's attributes.
        widths.SetIndices(VtIntArray{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
        CHECK_THROWS_WITH(
            UsdImporter().import_stage(*stage, {.import_curve_motion = true}),
            doctest::Contains("width count does not match")
        );
        widths.SetIndices(indices);
        st.SetIndices(VtIntArray{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
        CHECK_THROWS_WITH(
            UsdImporter().import_stage(*stage, {.import_curve_motion = true}),
            doctest::Contains("st count does not match")
        );
    }
}

TEST_CASE("curves_width_primvar_precedence_and_indices")
{
    using namespace pxr;
    for (const TfToken& interpolation :
         {UsdGeomTokens->constant, UsdGeomTokens->uniform, UsdGeomTokens->vertex, UsdGeomTokens->varying}) {
        for (bool indexed : {false, true}) {
            auto stage = UsdStage::CreateInMemory();
            auto curves = UsdGeomBasisCurves::Define(stage, SdfPath("/Curves"));
            curves.CreateTypeAttr().Set(UsdGeomTokens->cubic);
            curves.CreateBasisAttr().Set(UsdGeomTokens->bezier);
            curves.CreateCurveVertexCountsAttr().Set(VtIntArray{7, 4});
            curves.CreatePointsAttr().Set(
                VtVec3fArray{
                    {0, 0, 0},
                    {1, 0, 0},
                    {2, 0, 0},
                    {3, 0, 0},
                    {4, 0, 0},
                    {5, 0, 0},
                    {6, 0, 0},
                    {0, 3, 0},
                    {1, 3, 0},
                    {2, 3, 0},
                    {3, 3, 0}
                }
            );
            curves.CreateWidthsAttr().Set(VtFloatArray{0.02f});
            curves.SetWidthsInterpolation(UsdGeomTokens->constant);
            auto widths = UsdGeomPrimvarsAPI(curves)
                              .CreatePrimvar(TfToken("widths"), SdfValueTypeNames->FloatArray, interpolation);
            VtFloatArray values;
            if (interpolation == UsdGeomTokens->constant)
                values = {0.6f};
            else if (interpolation == UsdGeomTokens->uniform)
                values = {0.4f, 0.8f};
            else if (interpolation == UsdGeomTokens->vertex)
                values = {0.4f, 0.4f, 0.4f, 0.4f, 0.4f, 0.4f, 0.4f, 0.8f, 0.8f, 0.8f, 0.8f};
            else
                values = {0.f, 0.8f, 0.f, 0.4f, 0.8f};
            if (indexed) {
                std::reverse(values.begin(), values.end());
                VtIntArray indices(values.size());
                for (size_t i = 0; i < indices.size(); ++i)
                    indices[i] = int(indices.size() - 1 - i);
                widths.SetIndices(indices);
            }
            widths.Set(values);
            for (int level : {0, 3}) {
                CAPTURE(interpolation.GetString());
                CAPTURE(indexed);
                CAPTURE(level);
                const auto scene = UsdImporter().import_stage(*stage, {.curve_refinement_level = level});
                REQUIRE(scene->curves.size() == 1);
                const auto& curve = scene->curves[0];
                const size_t first_count = level == 0 ? 7 : 9;
                REQUIRE(curve.radius_stream().size == (level == 0 ? 11 : 14));
                for (size_t i = 0; i < curve.radius_stream().size; ++i) {
                    float expected = interpolation == UsdGeomTokens->constant ? 0.3f : i < first_count ? 0.2f : 0.4f;
                    if (interpolation == UsdGeomTokens->varying) {
                        const float x = curve.position_stream()[i].x;
                        expected = i < first_count ? 0.4f * (1.f - std::abs(x - 3.f) / 3.f) : 0.2f + 0.2f * x / 3.f;
                    }
                    CHECK(curve.radius_stream()[i] == doctest::Approx(expected));
                }
            }
        }
    }
}

TEST_CASE("curves_width_primvar_inheritance_fallback_and_validation")
{
    using namespace pxr;
    auto stage = UsdStage::CreateInMemory();
    auto parent = stage->DefinePrim(SdfPath("/Root"));
    auto curves = UsdGeomBasisCurves::Define(stage, SdfPath("/Root/Curves"));
    curves.CreateTypeAttr().Set(UsdGeomTokens->linear);
    curves.CreateCurveVertexCountsAttr().Set(VtIntArray{2});
    curves.CreatePointsAttr().Set(VtVec3fArray{{0, 0, 0}, {1, 0, 0}});
    curves.CreateWidthsAttr().Set(VtFloatArray{0.2f});
    curves.SetWidthsInterpolation(UsdGeomTokens->constant);
    auto check_radius = [&](float expected)
    {
        const auto scene = UsdImporter().import_stage(*stage);
        REQUIRE(scene->curves.size() == 1);
        REQUIRE(scene->curves[0].radius_stream().size == 2);
        for (float radius : scene->curves[0].radius_stream())
            CHECK(radius == doctest::Approx(expected));
    };
    check_radius(0.1f);
    auto inherited = UsdGeomPrimvarsAPI(parent)
                         .CreatePrimvar(TfToken("widths"), SdfValueTypeNames->FloatArray, UsdGeomTokens->constant);
    inherited.Set(VtFloatArray{0.6f});
    check_radius(0.3f);
    curves.GetWidthsAttr().Clear();
    check_radius(0.3f);
    parent.RemoveProperty(TfToken("primvars:widths"));
    check_radius(0.01f);
    curves.CreateWidthsAttr().Set(VtFloatArray{0.2f});
    auto widths = UsdGeomPrimvarsAPI(curves)
                      .CreatePrimvar(TfToken("widths"), SdfValueTypeNames->FloatArray, UsdGeomTokens->vertex);
    widths.Set(VtFloatArray{0.8f, 0.8f});
    check_radius(0.4f);
    widths.GetAttr().Block();
    check_radius(0.1f);
    widths.Set(VtFloatArray{});
    CHECK_THROWS_WITH(UsdImporter().import_stage(*stage), doctest::Contains("values or indices"));
    widths.Set(VtFloatArray{0.8f});
    widths.SetIndices(VtIntArray{0, 1});
    CHECK_THROWS_WITH(UsdImporter().import_stage(*stage), doctest::Contains("values or indices"));
    widths.SetIndices(VtIntArray{0, 0});
    widths.SetElementSize(2);
    CHECK_THROWS_WITH(UsdImporter().import_stage(*stage), doctest::Contains("element size"));
}

TEST_CASE("curves_inherited_uvs_and_degenerate_control_edges")
{
    using namespace pxr;
    auto stage = UsdStage::CreateInMemory();
    const auto parent = stage->DefinePrim(SdfPath("/Root"));
    auto st = UsdGeomPrimvarsAPI(parent)
                  .CreatePrimvar(TfToken("st"), SdfValueTypeNames->TexCoord2fArray, UsdGeomTokens->constant);
    st.Set(VtVec2fArray{{0.25f, 0.75f}});
    auto curves = UsdGeomBasisCurves::Define(stage, SdfPath("/Root/Curves"));
    curves.CreateTypeAttr().Set(UsdGeomTokens->cubic);
    curves.CreateBasisAttr().Set(UsdGeomTokens->bezier);
    curves.CreateCurveVertexCountsAttr().Set(VtIntArray{4});
    curves.CreatePointsAttr().Set(VtVec3fArray{{0, 0, 0}, {0, 0, 0}, {1, 0, 0}, {2, 0, 0}});
    const auto scene = UsdImporter().import_stage(*stage, {.curve_refinement_level = 0});
    REQUIRE(scene->curves.size() == 1);
    CHECK((scene->curves[0].indices == std::vector<uint32_t>{1, 2, 2, 3}));
    CHECK(scene->curves[0].texcoords.back().y == 0.75f);
    st.Set(VtVec2fArray{{0, 0}, {1, 1}});
    CHECK_THROWS(UsdImporter().import_stage(*stage, {.curve_refinement_level = 0}));
    st.Set(VtVec2fArray{});
    CHECK_THROWS(UsdImporter().import_stage(*stage, {.curve_refinement_level = 0}));
}

TEST_CASE("usd_value_property_conversion")
{
    SUBCASE("bool")
    {
        Properties props;
        REQUIRE(usd_importer::set_from_value(props, "flag", pxr::VtValue(true)));
        CHECK_EQ(props.type("flag"), PropertyType::bool_);
        CHECK_EQ(props.get<bool>("flag"), true);
    }

    SUBCASE("int")
    {
        Properties props;
        REQUIRE(usd_importer::set_from_value(props, "count", pxr::VtValue(uint32_t(7))));
        CHECK_EQ(props.type("count"), PropertyType::int_);
        CHECK_EQ(props.get<int64_t>("count"), 7);
    }

    SUBCASE("float")
    {
        Properties props;
        REQUIRE(usd_importer::set_from_value(props, "roughness", pxr::VtValue(3.5)));
        CHECK_EQ(props.type("roughness"), PropertyType::float_);
        CHECK(props.get<double>("roughness") == doctest::Approx(3.5));

        props = Properties{};
        REQUIRE(usd_importer::set_from_value(props, "roughness", pxr::VtValue(3.5f)));
        CHECK_EQ(props.type("roughness"), PropertyType::float_);
        CHECK(props.get<double>("roughness") == doctest::Approx(3.5f));
    }

    SUBCASE("int2")
    {
        Properties props;
        REQUIRE(usd_importer::set_from_value(props, "tile", pxr::VtValue(pxr::GfVec2i(1, 2))));
        CHECK_EQ(props.type("tile"), PropertyType::int2);
        CHECK_EQ(props.get<int2>("tile"), int2(1, 2));
    }

    SUBCASE("int3")
    {
        Properties props;
        REQUIRE(usd_importer::set_from_value(props, "indices", pxr::VtValue(pxr::GfVec3i(1, 2, 3))));
        CHECK_EQ(props.type("indices"), PropertyType::int3);
        CHECK_EQ(props.get<int3>("indices"), int3(1, 2, 3));
    }

    SUBCASE("int4")
    {
        Properties props;
        REQUIRE(usd_importer::set_from_value(props, "plane", pxr::VtValue(pxr::GfVec4i(1, 2, 3, 4))));
        CHECK_EQ(props.type("plane"), PropertyType::int4);
        CHECK_EQ(props.get<int4>("plane"), int4(1, 2, 3, 4));
    }

    SUBCASE("uint2")
    {
        Properties props;
        REQUIRE(usd_importer::set_from_value(props, "resolution", pxr::VtValue(pxr::GfSize2(640, 480))));
        CHECK_EQ(props.type("resolution"), PropertyType::uint2);
        CHECK_EQ(props.get<uint2>("resolution"), uint2(640, 480));
    }

    SUBCASE("uint3")
    {
        Properties props;
        REQUIRE(usd_importer::set_from_value(props, "grid", pxr::VtValue(pxr::GfSize3(4, 5, 6))));
        CHECK_EQ(props.type("grid"), PropertyType::uint3);
        CHECK_EQ(props.get<uint3>("grid"), uint3(4, 5, 6));
    }

    SUBCASE("float2")
    {
        Properties props;
        REQUIRE(usd_importer::set_from_value(props, "uv", pxr::VtValue(pxr::GfVec2d(0.25, 0.5))));
        CHECK_EQ(props.type("uv"), PropertyType::float2);
        CHECK_EQ(props.get<float2>("uv"), float2(0.25f, 0.5f));

        props = Properties{};
        REQUIRE(usd_importer::set_from_value(props, "uv", pxr::VtValue(pxr::GfVec2f(0.25f, 0.5f))));
        CHECK_EQ(props.type("uv"), PropertyType::float2);
        CHECK_EQ(props.get<float2>("uv"), float2(0.25f, 0.5f));
    }

    SUBCASE("float3")
    {
        Properties props;
        REQUIRE(usd_importer::set_from_value(props, "color", pxr::VtValue(pxr::GfVec3d(0.25, 0.5, 0.75))));
        CHECK_EQ(props.type("color"), PropertyType::float3);
        CHECK_EQ(props.get<float3>("color"), float3(0.25f, 0.5f, 0.75f));

        props = Properties{};
        REQUIRE(usd_importer::set_from_value(props, "color", pxr::VtValue(pxr::GfVec3f(0.25f, 0.5f, 0.75f))));
        CHECK_EQ(props.type("color"), PropertyType::float3);
        CHECK_EQ(props.get<float3>("color"), float3(0.25f, 0.5f, 0.75f));
    }

    SUBCASE("float4")
    {
        Properties props;
        REQUIRE(usd_importer::set_from_value(props, "rgba", pxr::VtValue(pxr::GfVec4d(0.25, 0.5, 0.75, 1.0))));
        CHECK_EQ(props.type("rgba"), PropertyType::float4);
        CHECK_EQ(props.get<float4>("rgba"), float4(0.25f, 0.5f, 0.75f, 1.0f));

        props = Properties{};
        REQUIRE(usd_importer::set_from_value(props, "rgba", pxr::VtValue(pxr::GfVec4f(0.25f, 0.5f, 0.75f, 1.0f))));
        CHECK_EQ(props.type("rgba"), PropertyType::float4);
        CHECK_EQ(props.get<float4>("rgba"), float4(0.25f, 0.5f, 0.75f, 1.0f));
    }

    SUBCASE("string token")
    {
        Properties props;
        REQUIRE(usd_importer::set_from_value(props, "mode", pxr::VtValue(pxr::TfToken("repeat"))));
        CHECK_EQ(props.type("mode"), PropertyType::string);
        CHECK_EQ(props.get<std::string>("mode"), std::string("repeat"));
    }

    SUBCASE("string asset path")
    {
        Properties props;
        REQUIRE(usd_importer::set_from_value(props, "path", pxr::VtValue(pxr::SdfAssetPath("textures/albedo.png"))));
        CHECK_EQ(props.type("path"), PropertyType::string);
        CHECK_EQ(props.get<std::string>("path"), std::string("textures/albedo.png"));
    }

    SUBCASE("unsupported array stringify fallback")
    {
        Properties props;
        pxr::VtArray<pxr::VtArray<bool>> values;
        pxr::VtArray<bool> nested;
        nested.push_back(true);
        nested.push_back(false);
        values.push_back(nested);
        pxr::VtValue value(values);
        REQUIRE(usd_importer::set_from_value(props, "flags", value));
        CHECK_EQ(props.type("flags"), PropertyType::string);
        CHECK_EQ(props.get<std::string>("flags"), pxr::TfStringify(value));
    }

    SUBCASE("bool array")
    {
        Properties props;
        pxr::VtArray<bool> values;
        values.push_back(true);
        values.push_back(false);
        values.push_back(true);
        REQUIRE(usd_importer::set_from_value(props, "flags", pxr::VtValue(values)));
        CHECK_EQ(props.type("flags"), PropertyType::list);
        CHECK_EQ(props.get_list<bool>("flags"), std::vector<bool>{true, false, true});
    }

    SUBCASE("int array")
    {
        Properties props;
        pxr::VtArray<uint32_t> values;
        values.push_back(7u);
        values.push_back(11u);
        REQUIRE(usd_importer::set_from_value(props, "counts", pxr::VtValue(values)));
        CHECK_EQ(props.type("counts"), PropertyType::list);
        CHECK_EQ(props.get_list<int64_t>("counts"), std::vector<int64_t>{7, 11});
    }

    SUBCASE("float array")
    {
        Properties props;
        pxr::VtArray<float> values;
        values.push_back(0.5f);
        values.push_back(1.25f);
        REQUIRE(usd_importer::set_from_value(props, "weights", pxr::VtValue(values)));
        CHECK_EQ(props.type("weights"), PropertyType::list);
        CHECK_EQ(props.get_list<double>("weights"), std::vector<double>{0.5, 1.25});
    }

    SUBCASE("float3 array")
    {
        Properties props;
        pxr::VtArray<pxr::GfVec3d> values;
        values.push_back(pxr::GfVec3d(0.25, 0.5, 0.75));
        values.push_back(pxr::GfVec3d(1.0, 2.0, 3.0));
        REQUIRE(usd_importer::set_from_value(props, "colors", pxr::VtValue(values)));
        CHECK_EQ(props.type("colors"), PropertyType::list);
        CHECK_EQ(
            props.get_list<float3>("colors"),
            std::vector<float3>{float3(0.25f, 0.5f, 0.75f), float3(1.0f, 2.0f, 3.0f)}
        );
    }

    SUBCASE("string token array")
    {
        Properties props;
        pxr::VtArray<pxr::TfToken> values;
        values.push_back(pxr::TfToken("repeat"));
        values.push_back(pxr::TfToken("clamp"));
        REQUIRE(usd_importer::set_from_value(props, "modes", pxr::VtValue(values)));
        CHECK_EQ(props.type("modes"), PropertyType::list);
        CHECK_EQ(
            props.get_list<std::string>("modes"),
            std::vector<std::string>{std::string("repeat"), std::string("clamp")}
        );
    }

    SUBCASE("string asset path array")
    {
        Properties props;
        pxr::VtArray<pxr::SdfAssetPath> values;
        values.push_back(pxr::SdfAssetPath("textures/albedo.png"));
        values.push_back(pxr::SdfAssetPath("textures/normal.png"));
        REQUIRE(usd_importer::set_from_value(props, "paths", pxr::VtValue(values)));
        CHECK_EQ(props.type("paths"), PropertyType::list);
        CHECK_EQ(
            props.get_list<std::string>("paths"),
            std::vector<std::string>{std::string("textures/albedo.png"), std::string("textures/normal.png")}
        );
    }
}

// Helper function to validate cornell-box geometry
static void validate_cornell_box_geometry(const ref<falcor::ImporterScene>& scene)
{
    CHECK_EQ(scene->meshes.size(), 16);

    auto find_mesh = [&](const std::string& name) -> const ImporterMesh*
    {
        auto it = std::find_if(
            scene->meshes.begin(),
            scene->meshes.end(),
            [&](const ImporterMesh& m)
            {
                return m.name == name;
            }
        );
        return it != scene->meshes.end() ? &(*it) : nullptr;
    };

    auto find_subgeometry = [](const ImporterMesh& mesh, const std::string& name) -> const ImporterMesh::Subgeometry*
    {
        auto it = std::find_if(
            mesh.subgeometries.begin(),
            mesh.subgeometries.end(),
            [&](const ImporterMesh::Subgeometry& s)
            {
                return s.name == name;
            }
        );
        return it != mesh.subgeometries.end() ? &(*it) : nullptr;
    };

    struct MeshRef {
        std::string name;
        size_t vertices_count;
    };

    std::vector<MeshRef> mesh_refs = {
        {"/cornell_box/tall_box_front/tall_box_front", 4},
        {"/cornell_box/tall_box_right/tall_box_right", 4},
        {"/cornell_box/tall_box_back/tall_box_back", 4},
        {"/cornell_box/tall_box_left/tall_box_left", 4},
        {"/cornell_box/tall_box_top/tall_box_top", 4},
        {"/cornell_box/short_box_back/short_box_back", 4},
        {"/cornell_box/short_box_right/short_box_right", 4},
        {"/cornell_box/short_box_front/short_box_front", 4},
        {"/cornell_box/short_box_left/short_box_left", 4},
        {"/cornell_box/short_box_top/short_box_top", 4},
        {"/cornell_box/left_wall/left_wall", 4},
        {"/cornell_box/right_wall/right_wall", 4},
        {"/cornell_box/back_wall/back_wall", 4},
        {"/cornell_box/light/light", 4},
        {"/cornell_box/ceiling/ceiling", 4},
        {"/cornell_box/floor/floor", 4},
    };

    for (const auto& ref : mesh_refs) {
        const ImporterMesh* mesh = find_mesh(ref.name);
        CHECK(mesh);
        if (mesh) {
            CHECK_EQ(mesh->vertex_count(), ref.vertices_count);
            CHECK_EQ(mesh->subgeometries.size(), 1);
            const ImporterMesh::Subgeometry* subgeo = find_subgeometry(*mesh, ref.name);
            CHECK(subgeo);
            if (subgeo) {
                CHECK_EQ(subgeo->indices.size(), 2);
            }
        }
    }

    CHECK_EQ(scene->cameras.size(), 1);
    if (scene->cameras.size() > 0) {
        CHECK_EQ(scene->cameras[0].name, "camera");
    }
}

TEST_CASE("UsdImporter - camera optics and node association")
{
    const std::filesystem::path camera_path
        = testing::project_directory() / "data" / "assets" / "test_usd_importer" / "test_camera.usda";
    REQUIRE(std::filesystem::exists(camera_path));

    auto importer = make_ref<UsdImporter>();
    ref<ImporterScene> scene = importer->load_scene(camera_path);
    REQUIRE(scene);
    REQUIRE_EQ(scene->cameras.size(), 2);

    auto find_camera_index = [&](std::string_view name) -> int
    {
        for (size_t index = 0; index < scene->cameras.size(); ++index) {
            if (scene->cameras[index].name == name)
                return static_cast<int>(index);
        }
        return -1;
    };

    const int camera_index = find_camera_index("CameraComponent");
    REQUIRE_GE(camera_index, 0);
    const ImporterCamera& camera = scene->cameras[camera_index];
    CHECK_EQ(camera.name, "CameraComponent");
    CHECK_EQ(camera.focal_length, doctest::Approx(100.f));
    CHECK_EQ(camera.fstop, doctest::Approx(2.8f));
    CHECK_EQ(camera.sensor_size_mm, doctest::Approx(19.f));
    CHECK_EQ(camera.vertical_sensor_size_mm(), doctest::Approx(19.f));
    CHECK(camera.enable_depth_of_field);
    CHECK_EQ(camera.focus_distance, doctest::Approx(2.5f));
    CHECK_EQ(camera.depth_range.x, doctest::Approx(0.1f));
    CHECK_EQ(camera.depth_range.y, doctest::Approx(1000.f));
    CHECK_EQ(camera.projection, ImporterCamera::Projection::perspective);
    CHECK_EQ(camera.fov_direction, ImporterCamera::FOVDirection::vertical);
    CHECK_EQ(
        camera.vertical_fov_degrees(),
        doctest::Approx(ImporterCamera::fov_degrees_from_focal_length(19.f, 100.f))
    );

    int camera_node_index = -1;
    for (size_t index = 0; index < scene->nodes.size(); ++index) {
        if (scene->nodes[index].camera_index == camera_index) {
            camera_node_index = static_cast<int>(index);
            break;
        }
    }
    REQUIRE_GE(camera_node_index, 0);

    const ImporterNode& camera_node = scene->nodes[camera_node_index];
    CHECK_EQ(camera_node.name, "/CameraActor/CameraComponent");
    CHECK_EQ(camera_node.transform.get_col(3).xyz(), float3(4.f, 5.f, 6.f));
    REQUIRE_GE(camera_node.parent, 0);
    CHECK_EQ(scene->nodes[camera_node.parent].name, "/CameraActor");
    CHECK_EQ(
        std::count_if(
            scene->nodes.begin(),
            scene->nodes.end(),
            [](const ImporterNode& node)
            {
                return node.name == "/CameraActor/CameraComponent";
            }
        ),
        1
    );

    REQUIRE(camera_node.source_asset_ref);
    CHECK_EQ(camera_node.source_asset_ref->format, "usd");
    CHECK_EQ(camera_node.source_asset_ref->asset_path, camera_path.lexically_normal().string());
    CHECK_EQ(camera_node.source_asset_ref->object_path, "/CameraActor/CameraComponent");
    CHECK_EQ(camera.source_asset_ref, camera_node.source_asset_ref);

    const int fallback_camera_index = find_camera_index("HorizontalFallbackCamera");
    REQUIRE_GE(fallback_camera_index, 0);
    const ImporterCamera& fallback_camera = scene->cameras[fallback_camera_index];
    CHECK_EQ(fallback_camera.fov_direction, ImporterCamera::FOVDirection::horizontal);
    CHECK_EQ(fallback_camera.sensor_size_mm, doctest::Approx(32.f));
    CHECK_EQ(fallback_camera.vertical_sensor_size_mm(), doctest::Approx(24.f));
}

TEST_CASE("UsdImporter - invisible hierarchy preserves camera but not renderable scene objects")
{
    const std::filesystem::path scene_path = testing::get_case_temp_directory() / "invisible_parent_camera.usda";
    {
        std::ofstream file(scene_path);
        file << R"usd(#usda 1.0

def Xform "HiddenGroup"
{
    token visibility = "invisible"

    def Camera "ValidationCamera"
    {
        float focalLength = 50
        float horizontalAperture = 36
        float verticalAperture = 24
    }

    def Mesh "HiddenMesh"
    {
        int[] faceVertexCounts = [3]
        int[] faceVertexIndices = [0, 1, 2]
        point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
    }

    def DistantLight "HiddenLight"
    {
        float inputs:intensity = 1
    }
}
)usd";
    }

    ref<ImporterScene> scene = make_ref<UsdImporter>()->load_scene(scene_path);
    REQUIRE(scene);
    REQUIRE_EQ(scene->cameras.size(), 1);
    CHECK_EQ(scene->cameras[0].name, "ValidationCamera");
    CHECK(scene->meshes.empty());
    CHECK(scene->lights.empty());
}

TEST_CASE("UsdImporter - native instance reuses prim transform node")
{
    const std::filesystem::path scene_path = testing::get_case_temp_directory() / "native_instance.usda";
    {
        std::ofstream file(scene_path);
        REQUIRE(file);
        file << R"usd(#usda 1.0

def Xform "Prototype"
{
    def Mesh "Triangle"
    {
        uniform token subdivisionScheme = "none"
        int[] faceVertexCounts = [3]
        int[] faceVertexIndices = [0, 1, 2]
        point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
    }
}

def Xform "Instance" (
    instanceable = true
    prepend references = </Prototype>
)
{
    double3 xformOp:translate = (1, 2, 3)
    uniform token[] xformOpOrder = ["xformOp:translate"]
}
)usd";
    }

    const ref<ImporterScene> scene = make_ref<UsdImporter>()->load_scene(scene_path);
    REQUIRE(scene);

    const auto instance_node = std::find_if(
        scene->nodes.begin(),
        scene->nodes.end(),
        [](const ImporterNode& node)
        {
            return node.name == "/Instance";
        }
    );
    REQUIRE(instance_node != scene->nodes.end());
    CHECK_GE(instance_node->prototype_index, 0);
    CHECK_EQ(instance_node->transform.get_col(3).xyz(), float3(1.f, 2.f, 3.f));
    CHECK_EQ(
        std::count_if(
            scene->nodes.begin(),
            scene->nodes.end(),
            [](const ImporterNode& node)
            {
                return node.name == "/Instance";
            }
        ),
        1
    );
}

TEST_CASE("UsdImporter - non-transformable native instances retain separate prototype nodes")
{
    const std::filesystem::path scene_path = testing::get_case_temp_directory() / "scope_instances.usda";
    {
        std::ofstream file(scene_path);
        REQUIRE(file);
        file << R"usd(#usda 1.0

def Scope "PrototypeA"
{
    def Mesh "TriangleA"
    {
        uniform token subdivisionScheme = "none"
        int[] faceVertexCounts = [3]
        int[] faceVertexIndices = [0, 1, 2]
        point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
    }
}

def Scope "PrototypeB"
{
    def Mesh "TriangleB"
    {
        uniform token subdivisionScheme = "none"
        int[] faceVertexCounts = [3]
        int[] faceVertexIndices = [0, 1, 2]
        point3f[] points = [(0, 0, 0), (2, 0, 0), (0, 2, 0)]
    }
}

def Xform "Parent"
{
    double3 xformOp:translate = (1, 2, 3)
    uniform token[] xformOpOrder = ["xformOp:translate"]

    def Scope "InstanceA" (
        instanceable = true
        prepend references = </PrototypeA>
    ) {}

    def Scope "InstanceB" (
        instanceable = true
        prepend references = </PrototypeB>
    ) {}

    def Scope "HiddenInstance" (
        instanceable = true
        prepend references = </PrototypeA>
    )
    {
        token visibility = "invisible"
    }
}

def BasisCurves "AfterParent"
{
    uniform token type = "linear"
    int[] curveVertexCounts = [2]
    point3f[] points = [(0, 0, 0), (0, 1, 0)]
    double3 xformOp:translate = (4, 5, 6)
    uniform token[] xformOpOrder = ["xformOp:translate"]
}
)usd";
    }

    const ref<ImporterScene> scene = make_ref<UsdImporter>()->load_scene(scene_path);
    REQUIRE(scene);
    const auto parent_node = std::find_if(
        scene->nodes.begin(),
        scene->nodes.end(),
        [](const ImporterNode& node)
        {
            return node.name == "/Parent";
        }
    );
    REQUIRE(parent_node != scene->nodes.end());
    CHECK_LT(parent_node->prototype_index, 0);
    CHECK_EQ(parent_node->transform.get_col(3).xyz(), float3(1.f, 2.f, 3.f));
    REQUIRE_EQ(parent_node->children.size(), 3);

    std::array<int, 2> prototype_indices;
    for (size_t i = 0; i < prototype_indices.size(); ++i) {
        const ImporterNode& instance_node = scene->nodes[parent_node->children[i]];
        const std::string suffix = i == 0 ? "A" : "B";
        CHECK_EQ(instance_node.name, "/Parent/Instance" + suffix);
        CHECK_EQ(instance_node.parent, std::distance(scene->nodes.begin(), parent_node));
        CHECK_EQ(instance_node.transform, float4x4::identity());
        REQUIRE_GE(instance_node.prototype_index, 0);
        REQUIRE_LT(instance_node.prototype_index, scene->prototypes.size());
        prototype_indices[i] = instance_node.prototype_index;

        const ImporterPrototype& prototype = scene->prototypes[instance_node.prototype_index];
        REQUIRE_GE(prototype.root_node, 0);
        REQUIRE_LT(prototype.root_node, scene->nodes.size());
        const ImporterNode& prototype_root = scene->nodes[prototype.root_node];
        REQUIRE_EQ(prototype_root.children.size(), 1);
        const ImporterNode& mesh_node = scene->nodes[prototype_root.children.front()];
        REQUIRE_GE(mesh_node.mesh_index, 0);
        REQUIRE_LT(mesh_node.mesh_index, scene->meshes.size());
        CHECK(scene->meshes[mesh_node.mesh_index].name.ends_with("/Triangle" + suffix));
    }
    CHECK_NE(prototype_indices[0], prototype_indices[1]);

    const ImporterNode& hidden_node = scene->nodes[parent_node->children.back()];
    CHECK_EQ(hidden_node.name, "/Parent/HiddenInstance");
    CHECK_EQ(hidden_node.parent, std::distance(scene->nodes.begin(), parent_node));
    CHECK_LT(hidden_node.prototype_index, 0);

    REQUIRE_EQ(scene->curves.size(), 1);
    const auto curve_node = std::find_if(
        scene->nodes.begin(),
        scene->nodes.end(),
        [](const ImporterNode& node)
        {
            return node.name == "/AfterParent";
        }
    );
    REQUIRE(curve_node != scene->nodes.end());
    CHECK_EQ(curve_node->curve_index, 0);
    CHECK_EQ(curve_node->parent, parent_node->parent);
    CHECK_EQ(curve_node->transform.get_col(3).xyz(), float3(4.f, 5.f, 6.f));
    CHECK_EQ(
        std::count_if(
            scene->nodes.begin(),
            scene->nodes.end(),
            [](const ImporterNode& node)
            {
                return node.name == "/AfterParent";
            }
        ),
        1
    );
}

TEST_CASE("UsdImporter - scopes preserve grouping and transform inheritance")
{
    const std::filesystem::path scene_path = testing::get_case_temp_directory() / "scope_groups.usda";
    {
        std::ofstream file(scene_path);
        REQUIRE(file);
        file << R"usd(#usda 1.0
(
    metersPerUnit = 1
    upAxis = "Y"
)

def Xform "Parent"
{
    double3 xformOp:translate = (1, 2, 3)
    uniform token[] xformOpOrder = ["xformOp:translate"]

    def Scope "Materials"
    {
        def Material "Unused"
        {
            def Shader "Surface" {}
        }
    }

    def Scope "Lights"
    {
        def Scope "Nested"
        {
            def SphereLight "Point"
            {
                bool treatAsPoint = 1
                double3 xformOp:translate = (4, 5, 6)
                uniform token[] xformOpOrder = ["xformOp:translate"]
            }
        }
    }

    def Camera "AfterGroup" {}
}
)usd";
    }

    const ref<ImporterScene> scene = make_ref<UsdImporter>()->load_scene(scene_path);
    REQUIRE(scene);
    REQUIRE_EQ(scene->lights.size(), 1);
    REQUIRE_EQ(scene->cameras.size(), 1);

    struct ExpectedNode {
        std::string_view name;
        std::string_view parent_name;
        float3 translation;
    };
    const std::array<ExpectedNode, 6> expected_nodes = {{
        {"/Parent", "Root", float3(1.f, 2.f, 3.f)},
        {"/Parent/Materials", "/Parent", float3(0.f)},
        {"/Parent/Lights", "/Parent", float3(0.f)},
        {"/Parent/Lights/Nested", "/Parent/Lights", float3(0.f)},
        {"/Parent/Lights/Nested/Point", "/Parent/Lights/Nested", float3(4.f, 5.f, 6.f)},
        {"/Parent/AfterGroup", "/Parent", float3(0.f)},
    }};
    for (const ExpectedNode& expected : expected_nodes) {
        CAPTURE(expected.name);
        const auto matches_name = [&](const ImporterNode& node)
        {
            return node.name == expected.name;
        };
        REQUIRE_EQ(std::count_if(scene->nodes.begin(), scene->nodes.end(), matches_name), 1);
        const auto node = std::find_if(scene->nodes.begin(), scene->nodes.end(), matches_name);
        REQUIRE_GE(node->parent, 0);
        REQUIRE_LT(node->parent, scene->nodes.size());
        const ImporterNode& parent = scene->nodes[node->parent];
        CHECK_EQ(parent.name, expected.parent_name);
        CHECK_EQ(node->transform, sgl::math::matrix_from_translation(expected.translation));
        CHECK_EQ(
            std::count(parent.children.begin(), parent.children.end(), std::distance(scene->nodes.begin(), node)),
            1
        );
        if (expected.name == "/Parent/Materials")
            CHECK(node->children.empty());
        if (expected.name == "/Parent/AfterGroup")
            CHECK_EQ(node->camera_index, 0);
        if (expected.name == "/Parent/Lights/Nested/Point") {
            CHECK_EQ(node->light_index, 0);
            float4x4 world_from_node = node->transform;
            for (int parent_idx = node->parent; parent_idx >= 0; parent_idx = scene->nodes[parent_idx].parent)
                world_from_node = mul(scene->nodes[parent_idx].transform, world_from_node);
            CHECK_EQ(world_from_node.get_col(3).xyz(), float3(5.f, 7.f, 9.f));
        }
    }
}

TEST_CASE("UsdImporter - indexed normal primvar")
{
    const std::filesystem::path scene_path = testing::get_case_temp_directory() / "indexed_normals.usda";
    {
        std::ofstream file(scene_path);
        REQUIRE(file);
        file << R"usd(#usda 1.0

def Mesh "IndexedNormals"
{
    uniform token subdivisionScheme = "none"
    int[] faceVertexCounts = [4]
    int[] faceVertexIndices = [0, 1, 2, 3]
    point3f[] points = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
    normal3f[] primvars:normals = [(0, 0, 1), (0, 1, 0)] (
        interpolation = "faceVarying"
    )
    int[] primvars:normals:indices = [0, 1, 0, 1]
}
)usd";
    }

    const ref<ImporterScene> scene = make_ref<UsdImporter>()->load_scene(scene_path);
    REQUIRE(scene);
    REQUIRE_EQ(scene->meshes.size(), 1);

    const ImporterMesh& mesh = scene->meshes.front();
    const auto mesh_node = std::find_if(
        scene->nodes.begin(),
        scene->nodes.end(),
        [](const ImporterNode& node)
        {
            return node.name == "/IndexedNormals";
        }
    );
    REQUIRE(mesh_node != scene->nodes.end());
    CHECK_EQ(mesh_node->mesh_index, 0);
    CHECK_EQ(
        std::count_if(
            scene->nodes.begin(),
            scene->nodes.end(),
            [](const ImporterNode& node)
            {
                return node.name == "/IndexedNormals";
            }
        ),
        1
    );
    REQUIRE_EQ(mesh.vertex_count(), 4);
    const auto positions = mesh.position_stream();
    const auto normals = mesh.normal_stream();
    REQUIRE(positions.valid());
    REQUIRE(normals.valid());

    const std::array expected{
        std::pair{float3(0.f, 0.f, 0.f), float3(0.f, 0.f, 1.f)},
        std::pair{float3(1.f, 0.f, 0.f), float3(0.f, 1.f, 0.f)},
        std::pair{float3(1.f, 1.f, 0.f), float3(0.f, 0.f, 1.f)},
        std::pair{float3(0.f, 1.f, 0.f), float3(0.f, 1.f, 0.f)},
    };
    for (size_t vertex_index = 0; vertex_index < mesh.vertex_count(); ++vertex_index) {
        CHECK_EQ(positions[vertex_index], expected[vertex_index].first);
        CHECK_EQ(normals[vertex_index], expected[vertex_index].second);
    }
}

TEST_CASE("UsdImporter - texture coordinate primvar numeric types")
{
    const std::array type_names{
        "half2",
        "float2",
        "double2",
        "texCoord2h",
        "texCoord2f",
        "texCoord2d",
    };

    for (std::string_view type_name : type_names) {
        CAPTURE(type_name);
        const std::filesystem::path scene_path
            = testing::get_case_temp_directory() / fmt::format("{}_texcoords.usda", type_name);
        {
            std::ofstream file(scene_path);
            REQUIRE(file);
            file << R"usd(#usda 1.0

def Mesh "Float2Texcoords"
{
    uniform token subdivisionScheme = "none"
    int[] faceVertexCounts = [3]
    int[] faceVertexIndices = [0, 1, 2]
    point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
    )usd" << type_name
                 << R"usd([] primvars:st = [(0.125, 0.25), (0.75, 0.25), (0.125, 0.875)] (
        interpolation = "vertex"
    )
}
)usd";
        }

        const ref<ImporterScene> scene = make_ref<UsdImporter>()->load_scene(scene_path);
        REQUIRE(scene);
        REQUIRE_EQ(scene->meshes.size(), 1);

        const ImporterMesh& mesh = scene->meshes.front();
        REQUIRE_EQ(mesh.vertex_count(), 3);
        const auto texcoords = mesh.texcoord_stream();
        REQUIRE(texcoords.valid());
        CHECK_EQ(texcoords[0], float2(0.125f, 0.25f));
        CHECK_EQ(texcoords[1], float2(0.75f, 0.25f));
        CHECK_EQ(texcoords[2], float2(0.125f, 0.875f));
    }
}

TEST_CASE("UsdImporter - configurable texture coordinate primvar mapping")
{
    const std::filesystem::path checked_asset = testing::project_directory() / "data" / "assets" / "test_usd_importer"
        / "test_additional_geometry_properties.usda";
    ref<ImporterScene> scene = make_ref<UsdImporter>()->load_scene(checked_asset);
    REQUIRE(scene);
    REQUIRE_EQ(scene->meshes.size(), 1);

    ImporterMesh& mesh = scene->meshes.front();
    const std::array<std::array<float, 6>, 8> expected_x{
        std::array{0.f, 1.f, 0.f, 2.f, 3.f, 2.f},
        std::array{10.f, 11.f, 10.f, 11.f, 11.f, 10.f},
        std::array{20.f, 21.f, 20.f, 22.f, 23.f, 22.f},
        std::array{30.f, 31.f, 30.f, 32.f, 33.f, 32.f},
        std::array{40.f, 41.f, 40.f, 42.f, 43.f, 42.f},
        std::array{50.f, 51.f, 50.f, 52.f, 53.f, 52.f},
        std::array{60.f, 61.f, 60.f, 62.f, 63.f, 62.f},
        std::array{70.f, 71.f, 70.f, 72.f, 73.f, 72.f},
    };
    REQUIRE_EQ(mesh.vertex_count(), 6);
    for (uint32_t slot = 0; slot < expected_x.size(); ++slot) {
        const auto texcoords = mesh.find_stream<float2>(ImporterSemantic::tex_coord, slot);
        REQUIRE(texcoords.valid());
        for (size_t vertex = 0; vertex < mesh.vertex_count(); ++vertex)
            CHECK_EQ(texcoords[vertex].x, expected_x[slot][vertex]);
    }

    const std::filesystem::path mapping_asset = testing::get_case_temp_directory() / "mapped_texcoords.usda";
    {
        std::ofstream file(mapping_asset);
        REQUIRE(file);
        file << R"usd(#usda 1.0

def Mesh "MappedTexcoords"
{
    uniform token subdivisionScheme = "none"
    int[] faceVertexCounts = [3]
    int[] faceVertexIndices = [0, 1, 2]
    point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
    normal3f[] normals = [(1, 0, 0), (1, 0, 0), (1, 0, 0)] (
        interpolation = "vertex"
    )
    texCoord2f[] primvars:st = [(1, 0), (1, 0), (1, 0)] (interpolation = "vertex")
    texCoord2f[] primvars:st_0 = [(9, 0), (9, 0), (9, 0)] (interpolation = "vertex")
    texCoord2f[] primvars:st1 = [(11, 0), (11, 0), (11, 0)] (interpolation = "vertex")
    texCoord2f[] primvars:lightmap = [(21, 0), (21, 0), (21, 0)] (interpolation = "vertex")
    texCoord2f[] primvars:LightMap = [(31, 0), (31, 0), (31, 0)] (interpolation = "vertex")
    half2[] primvars:customHalf = [(61, 0), (61, 0), (61, 0)] (interpolation = "vertex")
    double2[] primvars:customDouble = [(71, 0), (71, 0), (71, 0)] (interpolation = "vertex")
}
)usd";
    }

    // Keep the original five-Boolean aggregate initialization source-compatible.
    ImportOptions options{false, false, false, false, false};
    options.usd_texcoord_primvar_names
        = {"lightmap", "primvars:lightmap", "LightMap", "", "missing", "customHalf", "customDouble"};
    options.recompute_normals = true;
    scene = make_ref<UsdImporter>()->load_scene(mapping_asset, options);
    REQUIRE(scene);
    REQUIRE_EQ(scene->meshes.size(), 1);
    ImporterMesh& mapped_mesh = scene->meshes.front();
    const std::array<std::pair<uint32_t, float>, 6> mapped_values{
        std::pair{0u, 1.f},
        std::pair{1u, 21.f},
        std::pair{2u, 21.f},
        std::pair{3u, 31.f},
        std::pair{6u, 61.f},
        std::pair{7u, 71.f},
    };
    for (const auto& [slot, expected] : mapped_values) {
        const auto texcoords = mapped_mesh.find_stream<float2>(ImporterSemantic::tex_coord, slot);
        REQUIRE(texcoords.valid());
        CHECK_EQ(texcoords[0].x, expected);
    }
    CHECK_FALSE(mapped_mesh.find_stream<float2>(ImporterSemantic::tex_coord, 4).valid());
    CHECK_FALSE(mapped_mesh.find_stream<float2>(ImporterSemantic::tex_coord, 5).valid());
    const auto recomputed_normals = mapped_mesh.normal_stream();
    REQUIRE(recomputed_normals.valid());
    CHECK_EQ(recomputed_normals[0].x, doctest::Approx(0.f));
    CHECK_EQ(std::abs(recomputed_normals[0].z), doctest::Approx(1.f));

    const std::filesystem::path secondary_only_asset
        = testing::get_case_temp_directory() / "secondary_only_texcoords.usda";
    {
        std::ofstream file(secondary_only_asset);
        REQUIRE(file);
        file << R"usd(#usda 1.0

def Mesh "SecondaryOnly"
{
    uniform token subdivisionScheme = "none"
    int[] faceVertexCounts = [3]
    int[] faceVertexIndices = [0, 1, 2]
    point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
    texCoord2f[] primvars:st1 = [(4, 0), (4, 0), (4, 0)] (interpolation = "vertex")
}
)usd";
    }
    scene = make_ref<UsdImporter>()->load_scene(secondary_only_asset);
    REQUIRE(scene);
    ImporterMesh& secondary_only_mesh = scene->meshes.front();
    const auto generated_uv0 = secondary_only_mesh.find_stream<float2>(ImporterSemantic::tex_coord, 0);
    const auto authored_uv1 = secondary_only_mesh.find_stream<float2>(ImporterSemantic::tex_coord, 1);
    REQUIRE(generated_uv0.valid());
    REQUIRE(authored_uv1.valid());
    CHECK_EQ(generated_uv0[0], float2(0.f));
    CHECK_EQ(authored_uv1[0].x, 4.f);
}

TEST_CASE("UsdImporter - named float primvars")
{
    const std::filesystem::path checked_asset = testing::project_directory() / "data" / "assets" / "test_usd_importer"
        / "test_additional_geometry_properties.usda";
    ref<ImporterScene> scene = make_ref<UsdImporter>()->load_scene(checked_asset);
    REQUIRE(scene);
    ImporterMesh& mesh = scene->meshes.front();
    ImporterMeshAttribute* pref_attribute = mesh.find_attribute("Pref");
    REQUIRE(pref_attribute);
    CHECK_EQ(pref_attribute->semantic, ImporterSemantic::unknown);
    CHECK_EQ(pref_attribute->type, sgl::DataType::float32);
    CHECK_EQ(pref_attribute->num_components, 3);
    CHECK(mesh.find_attribute("pref") == nullptr);
    const auto pref = mesh.get_stream<float3>(*pref_attribute);
    const auto positions = mesh.position_stream();
    REQUIRE(pref.valid());
    for (size_t vertex = 0; vertex < mesh.vertex_count(); ++vertex) {
        CHECK_EQ(pref[vertex].z, 10.f);
        CHECK_NE(pref[vertex], positions[vertex]);
    }

    const std::filesystem::path named_asset = testing::get_case_temp_directory() / "named_float_primvars.usda";
    {
        std::ofstream file(named_asset);
        REQUIRE(file);
        file << R"usd(#usda 1.0

def Mesh "NamedFloatPrimvars"
{
    uniform token subdivisionScheme = "none"
    int[] faceVertexCounts = [3, 3]
    int[] faceVertexIndices = [0, 1, 2, 3, 4, 5]
    point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (2, 0, 0), (3, 0, 0), (2, 1, 0)]
    texCoord2f[] primvars:st = [(0, 0), (1, 0), (0, 1), (2, 0), (3, 0), (2, 1)] (interpolation = "vertex")
    point3f[] primvars:Pref = [(0, 0, 2), (1, 0, 2), (0, 1, 2), (2, 0, 2), (3, 0, 2), (2, 1, 2)] (
        interpolation = "vertex"
    )
    float[] primvars:shaderContract = [1, 2, 3, 4, 5, 6] (interpolation = "vertex")
    float2[] primvars:width2 = [(1, 2), (1, 2), (1, 2), (1, 2), (1, 2), (1, 2)] (interpolation = "vertex")
    color3f[] primvars:role3 = [(3, 4, 5)] (interpolation = "constant")
    color4f[] primvars:role4 = [(6, 7, 8, 9)] (interpolation = "constant")
    float[] primvars:faceTag = [3, 7] (interpolation = "uniform")
    double[] primvars:unsupportedDouble = [1, 2] (interpolation = "uniform")
    int[] primvars:unsupportedInt = [1, 2] (interpolation = "uniform")
}
)usd";
    }

    scene = make_ref<UsdImporter>()->load_scene(named_asset);
    REQUIRE(scene);
    ImporterMesh& named_mesh = scene->meshes.front();
    const std::array<std::pair<std::string_view, uint32_t>, 6> supported{
        std::pair{"Pref", 3u},
        std::pair{"shaderContract", 1u},
        std::pair{"width2", 2u},
        std::pair{"role3", 3u},
        std::pair{"role4", 4u},
        std::pair{"faceTag", 1u},
    };
    for (const auto& [name, components] : supported) {
        const ImporterMeshAttribute* attribute = named_mesh.find_attribute(name);
        REQUIRE(attribute);
        CHECK_EQ(attribute->num_components, components);
    }
    CHECK(named_mesh.find_attribute("unsupportedDouble") == nullptr);
    CHECK(named_mesh.find_attribute("unsupportedInt") == nullptr);

    const ImporterMeshAttribute* face_tag_attribute = named_mesh.find_attribute("faceTag");
    REQUIRE(face_tag_attribute);
    const auto face_tag = named_mesh.get_stream<float>(*face_tag_attribute);
    REQUIRE_EQ(named_mesh.vertex_count(), 6);
    for (size_t vertex = 0; vertex < named_mesh.vertex_count(); ++vertex)
        CHECK_EQ(face_tag[vertex], vertex < 3 ? 3.f : 7.f);
}

TEST_CASE("UsdImporter - light radiometry attributes")
{
    const std::filesystem::path lights_path
        = testing::project_directory() / "data" / "assets" / "test_usd_importer" / "test_lights.usda";
    REQUIRE(std::filesystem::exists(lights_path));

    auto importer = make_ref<UsdImporter>();
    ref<ImporterScene> scene = importer->load_scene(lights_path);
    REQUIRE(scene);
    REQUIRE_EQ(scene->lights.size(), 3);

    auto find_light = [&](std::string_view name) -> const ImporterLight*
    {
        auto it = std::find_if(
            scene->lights.begin(),
            scene->lights.end(),
            [&](const ImporterLight& light)
            {
                return light.name == name;
            }
        );
        return it != scene->lights.end() ? &(*it) : nullptr;
    };

    const ImporterLight* sun = find_light("/Sun");
    REQUIRE(sun);
    CHECK_EQ(sun->type, ImporterLight::Type::distant);
    CHECK_EQ(sun->degree_angular_diameter, doctest::Approx(0.5357f));
    CHECK_EQ(sun->color.x, doctest::Approx(0.25f));
    CHECK_EQ(sun->color.y, doctest::Approx(0.5f));
    CHECK_EQ(sun->color.z, doctest::Approx(0.75f));
    CHECK_EQ(sun->intensity, doctest::Approx(3.f));
    CHECK_EQ(sun->exposure, doctest::Approx(2.f));
    CHECK_FALSE(sun->normalize);

    const ImporterLight* normalized_sun = find_light("/NormalizedSun");
    REQUIRE(normalized_sun);
    CHECK_EQ(normalized_sun->type, ImporterLight::Type::distant);
    CHECK_EQ(normalized_sun->degree_angular_diameter, doctest::Approx(60.f));
    CHECK_EQ(normalized_sun->color.x, doctest::Approx(0.25f));
    CHECK_EQ(normalized_sun->color.y, doctest::Approx(0.5f));
    CHECK_EQ(normalized_sun->color.z, doctest::Approx(0.75f));
    CHECK_EQ(normalized_sun->intensity, doctest::Approx(3.f));
    CHECK_EQ(normalized_sun->exposure, doctest::Approx(2.f));
    CHECK(normalized_sun->normalize);

    const ImporterLight* sky = find_light("/TexturelessSky");
    REQUIRE(sky);
    CHECK_EQ(sky->type, ImporterLight::Type::dome);
    CHECK_EQ(sky->pole_axis, EnvMapPoleAxis::y);
    CHECK(sky->env_map_path.empty());
    CHECK_EQ(sky->color.x, doctest::Approx(0.5f));
    CHECK_EQ(sky->color.y, doctest::Approx(0.25f));
    CHECK_EQ(sky->color.z, doctest::Approx(0.125f));
    CHECK_EQ(sky->intensity, doctest::Approx(4.f));
    CHECK_EQ(sky->exposure, doctest::Approx(1.f));
}

TEST_CASE("UsdImporter - versioned dome light")
{
    std::string stage_up_axis = "Z";
    SUBCASE("Y-up stage")
    {
        stage_up_axis = "Y";
    }
    SUBCASE("Z-up stage") { }
    const std::filesystem::path light_path = testing::get_case_temp_directory() / "versioned_dome_light.usda";
    {
        std::ofstream file(light_path);
        REQUIRE(file);
        file << "#usda 1.0\n(\n    upAxis = \"" << stage_up_axis << "\"\n)\n";
        file << R"usd(
def DomeLight_1 "VersionedTexturelessSky"
{
    color3f inputs:color = (0.25, 0.5, 1)
    bool inputs:enableColorTemperature = 1
    float inputs:colorTemperature = 3000
    float inputs:exposure = 2
    float inputs:intensity = 2
    asset inputs:texture:file
    uniform token poleAxis = "scene"
    double3 xformOp:translate = (7, 8, 9)
    uniform token[] xformOpOrder = ["xformOp:translate"]

    def Xform "Child"
    {
        double3 xformOp:translate = (1, 2, 3)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }
}

def DomeLight_1 "ExplicitYPole"
{
    asset inputs:texture:file
    uniform token poleAxis = "Y"
}

def DomeLight_1 "ExplicitZPole"
{
    asset inputs:texture:file
    uniform token poleAxis = "Z"
}

def DomeLight "Legacy" {}
)usd";
    }

    ref<ImporterScene> scene = make_ref<UsdImporter>()->load_scene(light_path);
    REQUIRE(scene);
    REQUIRE_EQ(scene->lights.size(), 4);
    auto light_it = std::find_if(
        scene->lights.begin(),
        scene->lights.end(),
        [](const ImporterLight& light)
        {
            return light.name == "/VersionedTexturelessSky";
        }
    );
    REQUIRE(light_it != scene->lights.end());
    const ImporterLight& light = *light_it;
    CHECK_EQ(light.name, "/VersionedTexturelessSky");
    CHECK_EQ(light.type, ImporterLight::Type::dome);
    CHECK(light.env_map_path.empty());
    CHECK_EQ(light.color.x, doctest::Approx(0.25f));
    CHECK_EQ(light.color.y, doctest::Approx(0.5f));
    CHECK_EQ(light.color.z, doctest::Approx(1.f));
    CHECK_EQ(light.intensity, doctest::Approx(2.f));
    CHECK_EQ(light.exposure, doctest::Approx(2.f));
    CHECK(light.enable_color_temperature);
    CHECK_EQ(light.color_temperature, doctest::Approx(3000.f));

    auto find_light_node = [&](std::string_view name) -> const ImporterNode*
    {
        auto it = std::find_if(
            scene->nodes.begin(),
            scene->nodes.end(),
            [&](const ImporterNode& node)
            {
                return node.light_index >= 0 && scene->lights[node.light_index].name == name;
            }
        );
        return it == scene->nodes.end() ? nullptr : &(*it);
    };

    const ImporterNode* scene_pole = find_light_node("/VersionedTexturelessSky");
    REQUIRE(scene_pole);
    CHECK_EQ(scene_pole->name, "/VersionedTexturelessSky");
    CHECK_EQ(scene_pole->transform, sgl::math::matrix_from_translation(float3(7.f, 8.f, 9.f)));
    CHECK_EQ(
        scene->lights[scene_pole->light_index].pole_axis,
        stage_up_axis == "Y" ? EnvMapPoleAxis::y : EnvMapPoleAxis::z
    );
    REQUIRE_GE(scene_pole->parent, 0);
    CHECK_EQ(scene->nodes[scene_pole->parent].name, "Root");
    CHECK_LT(scene->nodes[scene_pole->parent].light_index, 0);

    auto child_it = std::find_if(
        scene->nodes.begin(),
        scene->nodes.end(),
        [](const ImporterNode& node)
        {
            return node.name == "/VersionedTexturelessSky/Child";
        }
    );
    REQUIRE(child_it != scene->nodes.end());
    REQUIRE_GE(child_it->parent, 0);
    CHECK_EQ(&scene->nodes[child_it->parent], scene_pole);
    CHECK_EQ(child_it->transform, sgl::math::matrix_from_translation(float3(1.f, 2.f, 3.f)));

    const ImporterNode* y_pole = find_light_node("/ExplicitYPole");
    REQUIRE(y_pole);
    CHECK_EQ(y_pole->name, "/ExplicitYPole");
    CHECK_EQ(y_pole->transform, float4x4::identity());
    CHECK_EQ(scene->lights[y_pole->light_index].pole_axis, EnvMapPoleAxis::y);
    CHECK_EQ(
        std::count_if(
            scene->nodes.begin(),
            scene->nodes.end(),
            [](const ImporterNode& node)
            {
                return node.name == "/ExplicitYPole";
            }
        ),
        1
    );

    const ImporterNode* z_pole = find_light_node("/ExplicitZPole");
    REQUIRE(z_pole);
    CHECK_EQ(z_pole->name, "/ExplicitZPole");
    CHECK_EQ(z_pole->transform, float4x4::identity());
    CHECK_EQ(scene->lights[z_pole->light_index].pole_axis, EnvMapPoleAxis::z);
    const ImporterNode* legacy = find_light_node("/Legacy");
    REQUIRE(legacy);
    CHECK_EQ(legacy->transform, float4x4::identity());
    CHECK_EQ(scene->lights[legacy->light_index].pole_axis, EnvMapPoleAxis::y);
    CHECK_EQ(scene->nodes.size(), 6);
}

TEST_CASE("UsdImporter - normalized area light radiance")
{
    const std::filesystem::path lights_path = testing::get_case_temp_directory() / "normalized_area_lights.usda";
    {
        std::ofstream file(lights_path);
        REQUIRE(file);
        file << R"usd(#usda 1.0
(
    metersPerUnit = 1
)

def Xform "ScaledPlanarLights"
{
    double3 xformOp:scale = (2, 3, 4)
    uniform token[] xformOpOrder = ["xformOp:scale"]

    def RectLight "Rect"
    {
        float inputs:width = 2
        float inputs:height = 5
        float inputs:exposure = 1
        float inputs:intensity = 60
        bool inputs:normalize = 1
    }

    def DiskLight "Disk"
    {
        float inputs:radius = 2
        float inputs:intensity = 24
        bool inputs:normalize = 1
    }
}

def Xform "ScaledSphereLight"
{
    double3 xformOp:scale = (2, 2, 2)
    uniform token[] xformOpOrder = ["xformOp:scale"]

    def SphereLight "Sphere"
    {
        float inputs:radius = 0.5
        float inputs:intensity = 8
        bool inputs:normalize = 1
    }
}

def Xform "NonUniformSphereLight"
{
    double3 xformOp:scale = (2, 1, 1)
    uniform token[] xformOpOrder = ["xformOp:scale"]

    def SphereLight "Sphere"
    {
        float inputs:radius = 1
        float inputs:intensity = 1
        bool inputs:normalize = 1
    }
}

def DomeLight "Dome"
{
    float inputs:intensity = 7
    bool inputs:normalize = 1
}
)usd";
    }

    auto importer = make_ref<UsdImporter>();
    ref<ImporterScene> scene = importer->load_scene(lights_path);
    REQUIRE(scene);
    REQUIRE_EQ(scene->lights.size(), 5);

    auto find_light = [&](std::string_view name) -> const ImporterLight*
    {
        auto it = std::find_if(
            scene->lights.begin(),
            scene->lights.end(),
            [&](const ImporterLight& light)
            {
                return light.name == name;
            }
        );
        return it != scene->lights.end() ? &(*it) : nullptr;
    };

    const ImporterLight* rect = find_light("/ScaledPlanarLights/Rect");
    REQUIRE(rect);
    CHECK_EQ(rect->type, ImporterLight::Type::rectangular);
    CHECK_EQ(rect->intensity, doctest::Approx(60.f));
    CHECK(rect->normalize);
    CHECK_EQ(rect->exposure, doctest::Approx(1.f));
    CHECK_EQ(rect->width, doctest::Approx(2.f));
    CHECK_EQ(rect->height, doctest::Approx(5.f));
    CHECK(scene->meshes.empty());
    CHECK(scene->materials.empty());
    auto rect_node = std::find_if(
        scene->nodes.begin(),
        scene->nodes.end(),
        [](const ImporterNode& node)
        {
            return node.name == "/ScaledPlanarLights/Rect" && node.light_index >= 0;
        }
    );
    REQUIRE(rect_node != scene->nodes.end());
    CHECK_LT(rect_node->mesh_index, 0);
    CHECK_GE(rect_node->light_index, 0);
    CHECK_EQ(rect_node->transform, float4x4::identity());
    CHECK_EQ(
        std::count_if(
            scene->nodes.begin(),
            scene->nodes.end(),
            [](const ImporterNode& node)
            {
                return node.name == "/ScaledPlanarLights/Rect";
            }
        ),
        1
    );

    const ImporterLight* disk = find_light("/ScaledPlanarLights/Disk");
    REQUIRE(disk);
    CHECK_EQ(disk->type, ImporterLight::Type::disk);
    CHECK_EQ(disk->intensity, doctest::Approx(24.f));
    CHECK(disk->normalize);
    CHECK_EQ(
        std::count_if(
            scene->nodes.begin(),
            scene->nodes.end(),
            [](const ImporterNode& node)
            {
                return node.name == "/ScaledPlanarLights/Disk";
            }
        ),
        1
    );

    const ImporterLight* sphere = find_light("/ScaledSphereLight/Sphere");
    REQUIRE(sphere);
    CHECK_EQ(sphere->type, ImporterLight::Type::sphere);
    CHECK_EQ(sphere->intensity, doctest::Approx(8.f));
    CHECK(sphere->normalize);

    const ImporterLight* non_uniform_sphere = find_light("/NonUniformSphereLight/Sphere");
    REQUIRE(non_uniform_sphere);
    CHECK_EQ(non_uniform_sphere->type, ImporterLight::Type::sphere);
    CHECK_EQ(non_uniform_sphere->intensity, doctest::Approx(1.f));
    CHECK(non_uniform_sphere->normalize);

    const ImporterLight* dome = find_light("/Dome");
    REQUIRE(dome);
    CHECK_EQ(dome->type, ImporterLight::Type::dome);
    CHECK_EQ(dome->intensity, doctest::Approx(7.f));
    CHECK(dome->normalize);
}

TEST_CASE("UsdImporter - light dimensions use USD schema fallbacks")
{
    const std::filesystem::path lights_path = testing::get_case_temp_directory() / "default_light_dimensions.usda";
    {
        std::ofstream file(lights_path);
        REQUIRE(file);
        file << R"usd(#usda 1.0

def SphereLight "Sphere"
{
}

def DiskLight "Disk"
{
}

def CylinderLight "Cylinder"
{
}
)usd";
    }

    const ref<ImporterScene> scene = make_ref<UsdImporter>()->load_scene(lights_path);
    REQUIRE(scene);
    REQUIRE_EQ(scene->lights.size(), 3);

    auto find_light = [&](std::string_view name) -> const ImporterLight*
    {
        const auto it = std::find_if(
            scene->lights.begin(),
            scene->lights.end(),
            [&](const ImporterLight& light)
            {
                return light.name == name;
            }
        );
        return it != scene->lights.end() ? &*it : nullptr;
    };

    const ImporterLight* sphere = find_light("/Sphere");
    REQUIRE(sphere);
    CHECK_EQ(sphere->radius, doctest::Approx(0.5f));

    const ImporterLight* disk = find_light("/Disk");
    REQUIRE(disk);
    CHECK_EQ(disk->radius, doctest::Approx(0.5f));

    const ImporterLight* cylinder = find_light("/Cylinder");
    REQUIRE(cylinder);
    CHECK_EQ(cylinder->type, ImporterLight::Type::cylinder);
    CHECK_EQ(cylinder->radius, doctest::Approx(0.5f));
    CHECK_EQ(cylinder->length, doctest::Approx(1.f));
    CHECK_FALSE(cylinder->treat_as_line);
}

TEST_CASE("UsdImporter - cylinder light properties")
{
    const std::filesystem::path lights_path = testing::get_case_temp_directory() / "cylinder_light.usda";
    {
        std::ofstream file(lights_path);
        REQUIRE(file);
        file << R"usd(#usda 1.0

def CylinderLight "Cylinder" (
    prepend apiSchemas = ["ShapingAPI"]
)
{
    color3f inputs:color = (0.25, 0.5, 1)
    custom bool inputs:enableInPrimaryRay = 1
    float inputs:exposure = 2
    float inputs:intensity = 3
    float inputs:length = 4
    bool inputs:normalize = 1
    float inputs:radius = 2
    float inputs:shaping:cone:angle = 35
    float inputs:shaping:cone:softness = 0.25
    float inputs:shaping:focus = 4
    bool treatAsLine = 1
}
)usd";
    }

    const ref<ImporterScene> scene = make_ref<UsdImporter>()->load_scene(lights_path);
    REQUIRE(scene);
    REQUIRE_EQ(scene->lights.size(), 1);
    const ImporterLight& cylinder = scene->lights.front();
    CHECK_EQ(cylinder.type, ImporterLight::Type::cylinder);
    CHECK_EQ(cylinder.color, float3(0.25f, 0.5f, 1.f));
    CHECK_EQ(cylinder.intensity, doctest::Approx(3.f));
    CHECK_EQ(cylinder.exposure, doctest::Approx(2.f));
    CHECK(cylinder.normalize);
    CHECK(cylinder.geometry_visible);
    CHECK_EQ(cylinder.radius, doctest::Approx(2.f));
    CHECK_EQ(cylinder.length, doctest::Approx(4.f));
    CHECK(cylinder.treat_as_line);
    CHECK(cylinder.enable_shaping);
    CHECK_EQ(cylinder.shaping_cone_angle, doctest::Approx(35.f));
    CHECK_EQ(cylinder.shaping_cone_softness, doctest::Approx(0.25f));
    CHECK_EQ(cylinder.shaping_focus, doctest::Approx(4.f));
}

TEST_CASE("UsdImporter - finite sphere classification and point hint preserve USD light energy")
{
    const std::filesystem::path lights_path = testing::get_case_temp_directory() / "point_and_spot_lights.usda";
    {
        std::ofstream file(lights_path);
        REQUIRE(file);
        file << R"usd(#usda 1.0
(
    metersPerUnit = 0.01
)

def SphereLight "Point"
{
    float inputs:exposure = 1
    float inputs:intensity = 79577.47
    float inputs:radius = 1
    bool treatAsPoint = 1
}

def SphereLight "Spot" (
    prepend apiSchemas = ["ShapingAPI"]
)
{
    float inputs:exposure = 2
    float inputs:intensity = 79577.47
    float inputs:radius = 1
    float inputs:shaping:cone:angle = 35
    float inputs:shaping:cone:softness = 0.42857143
    bool treatAsPoint = 1
}

def SphereLight "SpotFocused" (
    prepend apiSchemas = ["ShapingAPI"]
)
{
    float inputs:intensity = 79577.47
    float inputs:radius = 1
    float inputs:shaping:cone:angle = 35
    float inputs:shaping:cone:softness = 0.42857143
    float inputs:shaping:focus = 2
    bool treatAsPoint = 1
}

def SphereLight "FiniteSphere"
{
    float inputs:intensity = 1
    bool inputs:normalize = 1
    float inputs:radius = 2
}

def SphereLight "FiniteShapedSphere" (
    prepend apiSchemas = ["ShapingAPI"]
)
{
    float inputs:intensity = 1
    bool inputs:normalize = 1
    float inputs:radius = 2
    float inputs:shaping:cone:angle = 40
    float inputs:shaping:cone:softness = 0.25
    float inputs:shaping:focus = 4
}

def DiskLight "AuthoredDisk"
{
    float inputs:intensity = 1
    bool inputs:normalize = 1
    float inputs:radius = 2
}

def DiskLight "AuthoredShapedDisk" (
    prepend apiSchemas = ["ShapingAPI"]
)
{
    float inputs:intensity = 1
    bool inputs:normalize = 1
    float inputs:radius = 2
    float inputs:shaping:cone:angle = 30
    float inputs:shaping:cone:softness = 0.5
    float inputs:shaping:focus = 2
}

def RectLight "AuthoredShapedRect" (
    prepend apiSchemas = ["ShapingAPI"]
)
{
    float inputs:height = 3
    float inputs:intensity = 1
    bool inputs:normalize = 1
    float inputs:shaping:cone:angle = 50
    float inputs:shaping:cone:softness = 0.2
    float inputs:shaping:focus = 1.5
    float inputs:width = 2
}
)usd";
    }

    ref<ImporterScene> scene = make_ref<UsdImporter>()->load_scene(lights_path);
    REQUIRE(scene);
    REQUIRE_EQ(scene->lights.size(), 8);

    auto find_light = [&](const std::string& name) -> const ImporterLight*
    {
        auto it = std::find_if(
            scene->lights.begin(),
            scene->lights.end(),
            [&](const ImporterLight& light)
            {
                return light.name == name;
            }
        );
        return it == scene->lights.end() ? nullptr : &*it;
    };

    const ImporterLight* point = find_light("/Point");
    REQUIRE(point);
    CHECK_EQ(point->type, ImporterLight::Type::sphere);
    CHECK(point->treat_as_point);
    CHECK_EQ(point->radius, doctest::Approx(1.f));
    CHECK_FALSE(point->enable_virtual_sphere_shrinking);
    CHECK_EQ(point->intensity, doctest::Approx(79577.47f));
    CHECK_EQ(point->exposure, doctest::Approx(1.f));

    const ImporterLight* spot = find_light("/Spot");
    REQUIRE(spot);
    CHECK_EQ(spot->type, ImporterLight::Type::sphere);
    CHECK(spot->treat_as_point);
    CHECK_EQ(spot->intensity, doctest::Approx(79577.47f));
    CHECK_EQ(spot->exposure, doctest::Approx(2.f));
    CHECK(spot->enable_shaping);
    CHECK_EQ(spot->shaping_cone_angle, doctest::Approx(35.f));
    CHECK_EQ(spot->shaping_cone_softness, doctest::Approx(3.f / 7.f));
    CHECK_EQ(spot->shaping_focus, doctest::Approx(0.f));
    const int spot_index = static_cast<int>(spot - scene->lights.data());

    auto spot_node = std::find_if(
        scene->nodes.begin(),
        scene->nodes.end(),
        [&](const ImporterNode& node)
        {
            return node.light_index == spot_index;
        }
    );
    REQUIRE(spot_node != scene->nodes.end());
    CHECK_EQ(spot_node->transform, float4x4::identity());

    const ImporterLight* focused_spot = find_light("/SpotFocused");
    REQUIRE(focused_spot);
    CHECK_EQ(focused_spot->type, ImporterLight::Type::sphere);
    CHECK(focused_spot->treat_as_point);
    CHECK(focused_spot->enable_shaping);
    CHECK_EQ(focused_spot->shaping_focus, doctest::Approx(2.f));

    const ImporterLight* finite_sphere = find_light("/FiniteSphere");
    REQUIRE(finite_sphere);
    CHECK_EQ(finite_sphere->type, ImporterLight::Type::sphere);
    CHECK_FALSE(finite_sphere->enable_virtual_sphere_shrinking);
    CHECK_FALSE(finite_sphere->enable_shaping);
    CHECK_EQ(finite_sphere->radius, doctest::Approx(2.f));
    CHECK_EQ(finite_sphere->intensity, doctest::Approx(1.f));
    CHECK(finite_sphere->normalize);

    const ImporterLight* finite_shaped_sphere = find_light("/FiniteShapedSphere");
    REQUIRE(finite_shaped_sphere);
    CHECK_EQ(finite_shaped_sphere->type, ImporterLight::Type::sphere);
    CHECK_FALSE(finite_shaped_sphere->enable_virtual_sphere_shrinking);
    CHECK(finite_shaped_sphere->enable_shaping);
    CHECK_EQ(finite_shaped_sphere->radius, doctest::Approx(2.f));
    CHECK_EQ(finite_shaped_sphere->intensity, doctest::Approx(finite_sphere->intensity));
    CHECK_EQ(finite_shaped_sphere->shaping_cone_angle, doctest::Approx(40.f));
    CHECK_EQ(finite_shaped_sphere->shaping_cone_softness, doctest::Approx(0.25f));
    CHECK_EQ(finite_shaped_sphere->shaping_focus, doctest::Approx(4.f));

    const ImporterLight* authored_disk = find_light("/AuthoredDisk");
    REQUIRE(authored_disk);
    CHECK_EQ(authored_disk->type, ImporterLight::Type::disk);
    CHECK_FALSE(authored_disk->enable_shaping);
    CHECK_EQ(authored_disk->radius, doctest::Approx(2.f));
    CHECK_EQ(authored_disk->intensity, doctest::Approx(1.f));
    CHECK(authored_disk->normalize);

    const ImporterLight* authored_shaped_disk = find_light("/AuthoredShapedDisk");
    REQUIRE(authored_shaped_disk);
    CHECK_EQ(authored_shaped_disk->type, ImporterLight::Type::disk);
    CHECK(authored_shaped_disk->enable_shaping);
    CHECK_EQ(authored_shaped_disk->radius, doctest::Approx(2.f));
    CHECK_EQ(authored_shaped_disk->intensity, doctest::Approx(authored_disk->intensity));
    CHECK_EQ(authored_shaped_disk->shaping_cone_angle, doctest::Approx(30.f));
    CHECK_EQ(authored_shaped_disk->shaping_cone_softness, doctest::Approx(0.5f));
    CHECK_EQ(authored_shaped_disk->shaping_focus, doctest::Approx(2.f));

    const ImporterLight* authored_shaped_rect = find_light("/AuthoredShapedRect");
    REQUIRE(authored_shaped_rect);
    CHECK_EQ(authored_shaped_rect->type, ImporterLight::Type::rectangular);
    CHECK(authored_shaped_rect->enable_shaping);
    CHECK_EQ(authored_shaped_rect->width, doctest::Approx(2.f));
    CHECK_EQ(authored_shaped_rect->height, doctest::Approx(3.f));
    CHECK_EQ(authored_shaped_rect->intensity, doctest::Approx(1.f));
    CHECK_EQ(authored_shaped_rect->shaping_cone_angle, doctest::Approx(50.f));
    CHECK_EQ(authored_shaped_rect->shaping_cone_softness, doctest::Approx(0.2f));
    CHECK_EQ(authored_shaped_rect->shaping_focus, doctest::Approx(1.5f));

    const float4x4 expected_shaped_transform = float4x4::identity();
    for (const ImporterLight* shaped_light : {finite_shaped_sphere, authored_disk, authored_shaped_disk}) {
        const int light_index = static_cast<int>(shaped_light - scene->lights.data());
        auto light_node = std::find_if(
            scene->nodes.begin(),
            scene->nodes.end(),
            [&](const ImporterNode& node)
            {
                return node.light_index == light_index;
            }
        );
        REQUIRE(light_node != scene->nodes.end());
        CHECK_EQ(light_node->transform, expected_shaped_transform);
    }

    const int rect_index = static_cast<int>(authored_shaped_rect - scene->lights.data());
    auto rect_node = std::find_if(
        scene->nodes.begin(),
        scene->nodes.end(),
        [&](const ImporterNode& node)
        {
            return node.light_index == rect_index;
        }
    );
    REQUIRE(rect_node != scene->nodes.end());
    CHECK_EQ(rect_node->transform, float4x4::identity());
}

TEST_CASE("UsdImporter - Unreal exporter provenance enables virtual shrinking only for finite spheres")
{
    auto check_marker = [](std::string_view file_name, std::string_view marker)
    {
        const std::filesystem::path lights_path = testing::get_case_temp_directory() / file_name;
        {
            std::ofstream file(lights_path);
            REQUIRE(file);
            file << "#usda 1.0\n\n"
                 << marker << R"usd(

def SphereLight "FiniteSphere"
{
    float inputs:radius = 2
}

def SphereLight "Point"
{
    float inputs:radius = 2
    bool treatAsPoint = 1
}
)usd";
        }

        ref<ImporterScene> scene = make_ref<UsdImporter>()->load_scene(lights_path);
        REQUIRE(scene);
        REQUIRE_EQ(scene->lights.size(), 2);

        auto find_light = [&](std::string_view name) -> const ImporterLight*
        {
            auto it = std::find_if(
                scene->lights.begin(),
                scene->lights.end(),
                [&](const ImporterLight& light)
                {
                    return light.name == name;
                }
            );
            return it == scene->lights.end() ? nullptr : &*it;
        };

        const ImporterLight* finite_sphere = find_light("/FiniteSphere");
        REQUIRE(finite_sphere);
        CHECK_EQ(finite_sphere->type, ImporterLight::Type::sphere);
        CHECK(finite_sphere->enable_virtual_sphere_shrinking);

        const ImporterLight* point = find_light("/Point");
        REQUIRE(point);
        CHECK_EQ(point->type, ImporterLight::Type::sphere);
        CHECK(point->treat_as_point);
        CHECK_FALSE(point->enable_virtual_sphere_shrinking);
    };

    SUBCASE("legacy unrealMaterial attribute")
    {
        check_marker(
            "unreal_legacy_light_export.usda",
            R"usd(def Scope "ExporterMetadata"
{
    custom string unrealMaterial = "/Game/Materials/M_Lamp"
})usd"
        );
    }

    SUBCASE("unreal material render context")
    {
        check_marker(
            "unreal_render_context_light_export.usda",
            R"usd(def Material "UnrealMaterial"
{
    token outputs:unreal:surface = "unreal"
})usd"
        );
    }
}

// Helper function to print all materials in a scene
static void print_materials(const ref<falcor::ImporterScene>& scene, const std::string& test_name)
{
    MESSAGE("\n=== Materials for " << test_name << " ===");
    MESSAGE("Total materials: " << scene->materials.size());
    for (const auto& mat : scene->materials) {
        MESSAGE("\nMaterial: " << mat.name);
        auto converted = convert_material(mat);
        if (converted.has_value()) {
            MESSAGE(converted->to_string());
        } else {
            MESSAGE("<no conversion result>");
        }
    }
}

TEST_CASE("load_test_usdpreviewsurface")
{
    std::filesystem::path cornell_box_path
        = testing::project_directory() / "data" / "assets" / "cornell-box" / "usdpreviewsurface" / "cornell-box.usda";

    auto importer = falcor::make_ref<falcor::UsdImporter>();
    ref<falcor::ImporterScene> scene = importer->load_scene(cornell_box_path);

    CHECK_EQ(scene->materials.size(), 4);

    validate_cornell_box_geometry(scene);

    struct ExpectedMaterial {
        std::string name;
        float3 base_color_factor;
        float3 emissive_factor;
        float metallic_factor;
        float roughness_factor;
    };

    const std::vector<ExpectedMaterial> expected_materials = {
        {"/cornell_box/materials/white", float3(0.725f, 0.71f, 0.68f), float3(0.f, 0.f, 0.f), 0.f, 1.f},
        {"/cornell_box/materials/green", float3(0.14f, 0.45f, 0.091f), float3(0.f, 0.f, 0.f), 0.f, 1.f},
        {"/cornell_box/materials/light", float3(0.18f, 0.18f, 0.18f), float3(17.f, 12.f, 4.f), 0.f, 0.5f},
        {"/cornell_box/materials/red", float3(0.63f, 0.065f, 0.05f), float3(0.f, 0.f, 0.f), 0.f, 1.f},
    };

    for (const auto& expected : expected_materials) {
        auto it = std::find_if(
            scene->materials.begin(),
            scene->materials.end(),
            [&](const ImporterMaterial& m)
            {
                return m.name == expected.name;
            }
        );
        REQUIRE(it != scene->materials.end());

        auto converted = convert_material(*it, {"UsdPreviewSurface"}, {"StandardMaterial"});
        REQUIRE(converted.has_value());

        CHECK_EQ(converted->get<std::string_view>("_scene_material_type", ""), "StandardMaterial");
        CHECK_EQ(converted->get<float3>("base_color_factor", float3(0.f)), expected.base_color_factor);
        CHECK_EQ(converted->get<float3>("emissive_factor", float3(0.f)), expected.emissive_factor);
        CHECK_EQ(converted->get<float>("metallic_factor", -1.f), expected.metallic_factor);
        CHECK_EQ(converted->get<float>("roughness_factor", -1.f), expected.roughness_factor);
    }
}

TEST_CASE("load_test_mtlx_ref")
{
    std::filesystem::path cornell_box_path
        = testing::project_directory() / "data" / "assets" / "cornell-box" / "mtlx-ref" / "cornell-box.usda";

    auto importer = falcor::make_ref<falcor::UsdImporter>();
    ref<falcor::ImporterScene> scene = importer->load_scene(cornell_box_path);

    CHECK_EQ(scene->materials.size(), 4);

    validate_cornell_box_geometry(scene);

    const std::filesystem::path expected_mtlx_basepath
        = testing::project_directory() / "data" / "assets" / "cornell-box" / "mtlx-ref";

    struct ExpectedMaterial {
        std::string name;
        std::string node_name;
    };

    const std::vector<ExpectedMaterial> expected_materials = {
        {"/cornell_box/MaterialX/Materials/white", "white"},
        {"/cornell_box/MaterialX/Materials/red", "red"},
        {"/cornell_box/MaterialX/Materials/light", "light"},
        {"/cornell_box/MaterialX/Materials/green", "green"},
    };

    for (const auto& expected : expected_materials) {
        auto it = std::find_if(
            scene->materials.begin(),
            scene->materials.end(),
            [&](const ImporterMaterial& m)
            {
                return m.name == expected.name;
            }
        );
        REQUIRE(it != scene->materials.end());

        auto converted = convert_material(*it);
        REQUIRE(converted.has_value());

        CHECK_EQ(converted->get<std::string_view>("_scene_material_type", ""), "MaterialXMaterial");
        CHECK_EQ(converted->get<std::string_view>("mtlx_node_name", ""), expected.node_name);
        CHECK_EQ(converted->get<std::string_view>("mtlx_path", ""), "cornell-box.mtlx");

        const std::filesystem::path actual_basepath = converted->get<std::filesystem::path>("mtlx_basepath", {});
        CHECK_EQ(actual_basepath.lexically_normal(), expected_mtlx_basepath.lexically_normal());
    }
}

TEST_CASE("load_test_mtlx_usdshade")
{
    std::filesystem::path cornell_box_path
        = testing::project_directory() / "data" / "assets" / "cornell-box" / "mtlx-usdshade" / "cornell-box.usda";

    auto importer = falcor::make_ref<falcor::UsdImporter>();
    ref<falcor::ImporterScene> scene = importer->load_scene(cornell_box_path);

    CHECK_EQ(scene->materials.size(), 4);

    validate_cornell_box_geometry(scene);

    struct ExpectedMaterial {
        std::string name;
        std::string node_name;
        std::string diffuse_color_value;
        std::string roughness_value;
        std::string emissive_color_value;
    };

    const std::vector<ExpectedMaterial> expected_materials = {
        {"/cornell_box/materials/white", "white", "0.725, 0.71, 0.68", "1", ""},
        {"/cornell_box/materials/red", "red", "0.63, 0.065, 0.05", "1", ""},
        {"/cornell_box/materials/light", "light", "", "", "17, 12, 4"},
        {"/cornell_box/materials/green", "green", "0.14, 0.45, 0.091", "1", ""},
    };

    for (const auto& expected : expected_materials) {
        auto it = std::find_if(
            scene->materials.begin(),
            scene->materials.end(),
            [&](const ImporterMaterial& m)
            {
                return m.name == expected.name;
            }
        );
        REQUIRE(it != scene->materials.end());

        auto converted = convert_material(*it, {"MaterialX"}, {"MaterialXMaterial"});
        REQUIRE(converted.has_value());

        CHECK_EQ(converted->get<std::string_view>("_scene_material_type", ""), "MaterialXMaterial");
        CHECK_EQ(converted->get<std::string_view>("mtlx_node_name", ""), expected.node_name);

        const std::string mtlx_buffer = converted->get<std::string>("mtlx_buffer", "");
        CHECK_FALSE(mtlx_buffer.empty());
        CHECK(mtlx_buffer.find("<?xml") != std::string::npos);
        CHECK(mtlx_buffer.find("<materialx") != std::string::npos);
        CHECK(mtlx_buffer.find("<surfacematerial name=\"" + expected.node_name + "\"") != std::string::npos);

        if (!expected.diffuse_color_value.empty()) {
            CHECK(
                mtlx_buffer.find(
                    "<input name=\"diffuseColor\" type=\"color3\" value=\"" + expected.diffuse_color_value + "\" />"
                )
                != std::string::npos
            );
        }
        if (!expected.roughness_value.empty()) {
            CHECK(
                mtlx_buffer.find(
                    "<input name=\"roughness\" type=\"float\" value=\"" + expected.roughness_value + "\" />"
                )
                != std::string::npos
            );
        }

        if (!expected.emissive_color_value.empty()) {
            CHECK(
                mtlx_buffer.find(
                    "<input name=\"emissiveColor\" type=\"color3\" value=\"" + expected.emissive_color_value + "\" />"
                )
                != std::string::npos
            );
        }
    }
}

TEST_CASE("load_test_mdl")
{
    std::filesystem::path cornell_box_path
        = testing::project_directory() / "data" / "assets" / "cornell-box" / "mdl" / "cornell-box.usda";

    auto importer = falcor::make_ref<falcor::UsdImporter>();
    ref<falcor::ImporterScene> scene = importer->load_scene(cornell_box_path);

    CHECK_EQ(scene->materials.size(), 4);

    validate_cornell_box_geometry(scene);

    struct ExpectedMaterial {
        std::string name;
        std::string mdl_name_suffix;
    };

    const std::vector<ExpectedMaterial> expected_materials = {
        {"/cornell_box/materials/white", "::white"},
        {"/cornell_box/materials/red", "::red"},
        {"/cornell_box/materials/light", "::light"},
        {"/cornell_box/materials/green", "::green"},
    };

    const std::string expected_mdl_library_path
        = cornell_box_path.root_path().string() + ";" + cornell_box_path.parent_path().string();

    for (const auto& expected : expected_materials) {
        auto it = std::find_if(
            scene->materials.begin(),
            scene->materials.end(),
            [&](const ImporterMaterial& m)
            {
                return m.name == expected.name;
            }
        );
        REQUIRE(it != scene->materials.end());

        auto converted = convert_material(*it, {"MDL"}, {"MDLMaterial"});
        REQUIRE(converted.has_value());

        CHECK_EQ(converted->get<std::string_view>("_scene_material_type", ""), "MDLMaterial");
        CHECK_EQ(converted->get<bool>("mdl_class_compilation", false), true);
        CHECK_EQ(converted->get<std::string>("mdl_library_path", ""), expected_mdl_library_path);

        const std::string mdl_material_name = converted->get<std::string>("mdl_material_name", "");
        CHECK_FALSE(mdl_material_name.empty());
        CHECK(mdl_material_name.ends_with(expected.mdl_name_suffix));
    }

    auto unresolved_material = scene->materials.front();
    auto& terminal = unresolved_material.output_to_material_network.at("_terminal:mdl:surface").back();
    terminal.set("info:mdl:sourceAsset", std::filesystem::path("./missing.mdl"));
    terminal.set("info:mdl:sourceAsset:is_resolved", false);
    CHECK_FALSE(convert_material(unresolved_material, {"MDL"}, {"MDLMaterial"}).has_value());
}

TEST_CASE("load_test_display_color")
{
    std::filesystem::path cornell_box_path
        = testing::project_directory() / "data" / "assets" / "cornell-box" / "display-color" / "cornell-box.usda";

    auto importer = falcor::make_ref<falcor::UsdImporter>();
    ref<falcor::ImporterScene> scene = importer->load_scene(cornell_box_path);

    // display-color uses primitive display colors, not materials
    CHECK_EQ(scene->materials.size(), 0);

    validate_cornell_box_geometry(scene);

    // Print any display color information from the meshes
    std::cout << "\n=== Mesh Display Colors for display-color ===" << std::endl;
    std::cout << "Total meshes: " << scene->meshes.size() << std::endl;
    for (size_t i = 0; i < scene->meshes.size() && i < 5; ++i) {
        const auto& mesh = scene->meshes[i];
        std::cout << "Mesh " << i << ": " << mesh.name << std::endl;
    }
}

TEST_CASE("load_scene_usdpreviewsurface_material_input")
{
    const std::filesystem::path test_scene_path = testing::project_directory() / "data" / "assets" / "test_usd_importer"
        / "test_usdpreviewsurface_material_input.usda";
    REQUIRE(std::filesystem::exists(test_scene_path));

    auto importer = falcor::make_ref<falcor::UsdImporter>();
    ref<falcor::ImporterScene> scene = importer->load_scene(test_scene_path.string());

    REQUIRE(scene);
    REQUIRE_EQ(scene->materials.size(), 1);

    const ImporterMaterial& material = scene->materials[0];
    CHECK_EQ(material.name, "/Root/Looks/Mat");
    CHECK_EQ(material.params.get<std::string_view>("_type", ""), "usd_UsdPreviewSurface");
    CHECK_EQ(material.params.get<float3>("diffuseColor", float3(0)), float3(0.2f, 0.7f, 0.3f));
}

TEST_CASE("convert_material_usdpreviewsurface_to_standardmaterial")
{
    const std::filesystem::path test_scene_path = testing::project_directory() / "data" / "assets" / "test_usd_importer"
        / "test_usdpreviewsurface_material_input.usda";
    REQUIRE(std::filesystem::exists(test_scene_path));

    auto importer = falcor::make_ref<falcor::UsdImporter>();
    ref<falcor::ImporterScene> scene = importer->load_scene(test_scene_path);
    REQUIRE(scene);
    REQUIRE_EQ(scene->materials.size(), 1);

    auto result = convert_material(scene->materials[0], {"UsdPreviewSurface"}, {"StandardMaterial"});
    REQUIRE(result.has_value());

    Properties expected;
    expected.set("_scene_material_type", std::string("StandardMaterial"));
    expected.set("base_color_factor", float3(0.2f, 0.7f, 0.3f));
    expected.set("emissive_factor", float3(0.0f, 0.0f, 0.0f));
    expected.set("metallic_factor", 0.0f);
    expected.set("roughness_factor", 0.4f);
    CHECK_EQ(*result, expected);
}

TEST_CASE("convert_material_usdpreviewsurface_emissive_texture_uses_unit_factor")
{
    const std::filesystem::path test_scene_path = testing::project_directory() / "tests" / "native" / "usd_importer"
        / "test_usdpreviewsurface_emissive_texture.usda";
    REQUIRE(std::filesystem::exists(test_scene_path));

    auto importer = falcor::make_ref<falcor::UsdImporter>();
    ref<falcor::ImporterScene> scene = importer->load_scene(test_scene_path);
    REQUIRE(scene);
    REQUIRE_EQ(scene->materials.size(), 1);

    auto result = convert_material(scene->materials[0], {"UsdPreviewSurface"}, {"StandardMaterial"});
    REQUIRE(result.has_value());

    CHECK(result->find("emissive_texture_path") != result->end());
    CHECK_EQ(result->get<float3>("emissive_factor", float3(0.f)), float3(1.f));
    CHECK(result->get<bool>("emissive_texture_srgb"));
}

TEST_CASE("convert_falcor_usd_materials")
{
    const std::filesystem::path test_scene_path
        = testing::project_directory() / "tests" / "native" / "usd_importer" / "test_falcor_materials.usda";
    REQUIRE(std::filesystem::exists(test_scene_path));

    auto importer = falcor::make_ref<falcor::UsdImporter>();
    ref<falcor::ImporterScene> scene = importer->load_scene(test_scene_path);
    REQUIRE(scene);
    REQUIRE_EQ(scene->materials.size(), 5);

    auto find_material = [&](std::string_view name) -> const ImporterMaterial*
    {
        auto it = std::find_if(
            scene->materials.begin(),
            scene->materials.end(),
            [&](const ImporterMaterial& material)
            {
                return material.name == name;
            }
        );
        return it != scene->materials.end() ? &*it : nullptr;
    };

    const std::filesystem::path expected_texture_path = std::filesystem::weakly_canonical(test_scene_path);

    const ImporterMaterial* standard_ptr = find_material("/Root/Looks/Standard");
    REQUIRE(standard_ptr);
    const ImporterMaterial& standard = *standard_ptr;
    REQUIRE(standard.output_to_material_network.contains("_terminal:falcor:surface"));
    auto standard_result = convert_material(standard);
    REQUIRE(standard_result);
    CHECK_EQ(standard_result->get<std::string>("_scene_material_type"), "StandardMaterial");
    CHECK_EQ(standard_result->get<AlphaMode>("alpha_mode"), AlphaMode::mask);
    CHECK_EQ(standard_result->get<float>("alpha_factor"), doctest::Approx(0.75f));
    CHECK_EQ(standard_result->get<float>("alpha_cutoff"), doctest::Approx(0.25f));
    CHECK_EQ(standard_result->get<float3>("base_color_factor"), float3(0.2f, 0.4f, 0.6f));
    CHECK(standard_result->get<bool>("double_sided"));
    CHECK_EQ(standard_result->get<float3>("emissive_factor"), float3(0.1f, 0.2f, 0.3f));
    CHECK_EQ(standard_result->get<float>("ior"), doctest::Approx(1.7f));
    CHECK_EQ(standard_result->get<float>("metallic_factor"), doctest::Approx(0.8f));
    CHECK_EQ(standard_result->get<float>("roughness_factor"), doctest::Approx(0.35f));
    CHECK_EQ(standard_result->get<float>("diffuse_transmission_factor"), doctest::Approx(0.2f));
    CHECK_EQ(standard_result->get<float>("specular_transmission_factor"), doctest::Approx(0.9f));
    CHECK(standard_result->get<bool>("thin_walled"));
    CHECK_EQ(standard_result->get<float3>("transmission_factor"), float3(0.15f, 0.75f, 0.3f));
    CHECK_EQ(standard_result->get<uint>("metallic_texture_channel"), 2u);
    CHECK_EQ(standard_result->get<uint>("roughness_texture_channel"), 1u);
    CHECK_EQ(
        standard_result->get<std::filesystem::path>("base_color_texture_path").lexically_normal(),
        expected_texture_path.lexically_normal()
    );
    CHECK_EQ(
        standard_result->get<std::filesystem::path>("emissive_texture_path").lexically_normal(),
        expected_texture_path.lexically_normal()
    );
    CHECK_EQ(
        standard_result->get<std::filesystem::path>("metallic_roughness_texture_path").lexically_normal(),
        expected_texture_path.lexically_normal()
    );
    CHECK_EQ(
        standard_result->get<std::filesystem::path>("normal_texture_path").lexically_normal(),
        expected_texture_path.lexically_normal()
    );
    CHECK_EQ(
        standard_result->get<std::filesystem::path>("transmission_texture_path").lexically_normal(),
        expected_texture_path.lexically_normal()
    );
    CHECK(standard_result->get<bool>("base_color_texture_srgb"));
    CHECK(standard_result->get<bool>("emissive_texture_srgb"));
    CHECK_FALSE(standard_result->has_property("metallic_roughness_texture_srgb"));
    CHECK_FALSE(standard_result->has_property("normal_texture_srgb"));
    CHECK(standard_result->get<bool>("transmission_texture_srgb"));
    CHECK(importer_material::is_texture_srgb(*standard_result, "base_color_texture_path"));
    CHECK_FALSE(importer_material::is_texture_srgb(*standard_result, "normal_texture_path"));

    const ImporterMaterial* unlit_ptr = find_material("/Root/Looks/Unlit");
    REQUIRE(unlit_ptr);
    const ImporterMaterial& unlit = *unlit_ptr;
    auto unlit_result = convert_material(unlit);
    REQUIRE(unlit_result);
    CHECK_EQ(unlit_result->get<std::string>("_scene_material_type"), "UnlitMaterial");
    CHECK_EQ(unlit_result->get<AlphaMode>("alpha_mode"), AlphaMode::blend);
    CHECK_EQ(unlit_result->get<float>("alpha_factor"), doctest::Approx(0.6f));
    CHECK_EQ(unlit_result->get<float>("alpha_cutoff"), doctest::Approx(0.4f));
    CHECK_EQ(unlit_result->get<float3>("base_color_factor"), float3(1.5f, 2.f, 4.f));
    CHECK_FALSE(unlit_result->get<bool>("double_sided"));
    CHECK_EQ(
        unlit_result->get<std::filesystem::path>("base_color_texture_path").lexically_normal(),
        expected_texture_path.lexically_normal()
    );
    CHECK(unlit_result->get<bool>("base_color_texture_srgb"));

    const ImporterMaterial* standard_defaults_ptr = find_material("/Root/Looks/StandardDefaults");
    REQUIRE(standard_defaults_ptr);
    auto standard_defaults_result = convert_material(*standard_defaults_ptr);
    REQUIRE(standard_defaults_result);
    CHECK_EQ(standard_defaults_result->size(), 1);
    CHECK_EQ(standard_defaults_result->get<std::string>("_scene_material_type"), "StandardMaterial");

    const ImporterMaterial* unlit_defaults_ptr = find_material("/Root/Looks/UnlitDefaults");
    REQUIRE(unlit_defaults_ptr);
    auto unlit_defaults_result = convert_material(*unlit_defaults_ptr);
    REQUIRE(unlit_defaults_result);
    CHECK_EQ(unlit_defaults_result->size(), 1);
    CHECK_EQ(unlit_defaults_result->get<std::string>("_scene_material_type"), "UnlitMaterial");

    const ImporterMaterial* openpbr_ptr = find_material("/Root/Looks/OpenPBR");
    REQUIRE(openpbr_ptr);
    const ImporterMaterial& openpbr = *openpbr_ptr;
    auto openpbr_result = convert_material(openpbr);
    REQUIRE(openpbr_result);
    CHECK_EQ(openpbr_result->get<std::string>("_scene_material_type"), "OpenPBRMaterial");
    CHECK_EQ(openpbr_result->get<float>("base_weight"), doctest::Approx(0.9f));
    CHECK_EQ(openpbr_result->get<float3>("base_color"), float3(0.25f, 0.5f, 0.75f));
    CHECK_EQ(openpbr_result->get<float>("specular_roughness"), doctest::Approx(0.35f));
    CHECK_EQ(openpbr_result->get<uint>("specular_roughness_texture_channel"), 1u);
    CHECK_EQ(openpbr_result->get<float3>("transmission_scatter"), float3(0.1f, 0.2f, 0.3f));
    CHECK_EQ(openpbr_result->get<float>("thin_film_ior"), doctest::Approx(1.45f));
    CHECK_EQ(openpbr_result->get<float>("emission_luminance"), doctest::Approx(2.5f));
    CHECK_EQ(openpbr_result->get<float>("opacity_factor"), doctest::Approx(0.8f));
    CHECK_EQ(openpbr_result->get<float>("opacity_threshold"), doctest::Approx(0.2f));
    CHECK_EQ(openpbr_result->get<uint>("opacity_texture_channel"), 2u);
    CHECK_EQ(openpbr_result->get<float>("normal_texture_scale"), doctest::Approx(0.75f));
    CHECK(openpbr_result->get<bool>("geometry_thin_walled"));
    CHECK_EQ(
        openpbr_result->get<std::filesystem::path>("base_color_texture_path").lexically_normal(),
        expected_texture_path.lexically_normal()
    );
    CHECK_EQ(
        openpbr_result->get<std::filesystem::path>("specular_roughness_texture_path").lexically_normal(),
        expected_texture_path.lexically_normal()
    );
    CHECK_EQ(
        openpbr_result->get<std::filesystem::path>("opacity_texture_path").lexically_normal(),
        expected_texture_path.lexically_normal()
    );
    CHECK_EQ(
        openpbr_result->get<std::filesystem::path>("normal_texture_path").lexically_normal(),
        expected_texture_path.lexically_normal()
    );
    CHECK(openpbr_result->get<bool>("base_color_texture_srgb"));
    CHECK_FALSE(openpbr_result->has_property("specular_roughness_texture_srgb"));
    CHECK_FALSE(openpbr_result->has_property("opacity_texture_srgb"));
    CHECK_FALSE(openpbr_result->has_property("normal_texture_srgb"));

    {
        ImporterMaterial extended = unlit;
        auto& terminal = extended.output_to_material_network.at("_terminal:falcor:surface").back();
        terminal.set("unexpected", 1.f);
        terminal.set("unexpected:typename", "float");
        auto result = convert_material(extended);
        REQUIRE(result);
        CHECK_FALSE(result->has_property("unexpected"));
    }

    {
        ImporterMaterial extended = openpbr;
        auto& terminal = extended.output_to_material_network.at("_terminal:falcor:surface").back();
        terminal.set("unexpected", 1.f);
        terminal.set("unexpected:typename", "float");
        auto result = convert_material(extended);
        REQUIRE(result);
        CHECK_FALSE(result->has_property("unexpected"));
    }
}

TEST_CASE_NIGHTLY("load shader ball Falcor overlays")
{
    struct OverlayCase {
        std::filesystem::path filename;
        std::string material_name;
        std::string material_type;
    };

    const std::array cases = {
        OverlayCase{
            "shader-ball/standard.usda",
            "/standard_shader_ball_scene/materials/falcor_standard",
            "StandardMaterial"
        },
        OverlayCase{
            "shader-ball/openpbr.usda",
            "/standard_shader_ball_scene/materials/falcor_openpbr",
            "OpenPBRMaterial"
        },
    };

    for (const OverlayCase& overlay : cases) {
        CAPTURE(overlay.filename);
        const std::filesystem::path scene_path = testing::project_directory() / "data" / "scenes" / overlay.filename;
        REQUIRE(std::filesystem::exists(scene_path));

        auto importer = falcor::make_ref<falcor::UsdImporter>();
        ref<falcor::ImporterScene> scene = importer->load_scene(scene_path);
        REQUIRE(scene);
        REQUIRE_EQ(scene->lights.size(), 5);
        for (const ImporterLight& light : scene->lights) {
            CHECK_EQ(light.type, ImporterLight::Type::rectangular);
            CHECK(light.geometry_visible);
        }

        const auto material_it = std::find_if(
            scene->materials.begin(),
            scene->materials.end(),
            [&](const ImporterMaterial& material)
            {
                return material.name == overlay.material_name;
            }
        );
        REQUIRE(material_it != scene->materials.end());
        auto converted = convert_material(*material_it);
        REQUIRE(converted);
        CHECK_EQ(converted->get<std::string>("_scene_material_type"), overlay.material_type);

        if (overlay.material_type == "StandardMaterial") {
            CHECK_EQ(converted->get<float3>("base_color_factor"), float3(0.18f));
            CHECK_EQ(converted->get<float>("roughness_factor"), doctest::Approx(0.25f));
            CHECK_EQ(converted->get<float>("ior"), doctest::Approx(1.5f));
        } else {
            CHECK_EQ(converted->get<float3>("base_color"), float3(0.18f));
            CHECK_EQ(converted->get<float>("specular_roughness"), doctest::Approx(0.25f));
            CHECK_EQ(converted->get<float>("specular_ior"), doctest::Approx(1.5f));
        }

        const std::string surface_name = "/standard_shader_ball_scene/shader_ball/material_surface";
        const auto surface_it = std::find_if(
            scene->meshes.begin(),
            scene->meshes.end(),
            [&](const ImporterMesh& mesh)
            {
                return mesh.name == surface_name;
            }
        );
        REQUIRE(surface_it != scene->meshes.end());
        REQUIRE_FALSE(surface_it->subgeometries.empty());
        for (const ImporterMesh::Subgeometry& subgeometry : surface_it->subgeometries)
            CHECK_EQ(subgeometry.material_index, static_cast<int>(material_it - scene->materials.begin()));

        ref<ImporterScene> default_scene = import_scene(scene_path);
        REQUIRE(default_scene);
        CHECK_EQ(default_scene->lights.size(), scene->lights.size());
        CHECK_EQ(default_scene->meshes.size(), scene->meshes.size());
        CHECK_EQ(default_scene->materials.size(), scene->materials.size());
    }
}

TEST_CASE("convert_material_materialx_to_materialxmaterial_mtlx_reference")
{
    const std::filesystem::path test_scene_path
        = testing::project_directory() / "tests" / "native" / "usd_importer" / "test_scope_materials_mtlx_openpbr.usda";
    REQUIRE(std::filesystem::exists(test_scene_path));

    auto importer = falcor::make_ref<falcor::UsdImporter>();
    ref<falcor::ImporterScene> scene = importer->load_scene(test_scene_path);
    REQUIRE(scene);
    REQUIRE_EQ(scene->materials.size(), 1);

    const std::filesystem::path expected_mtlx_basepath
        = testing::project_directory() / "external" / "MaterialX" / "resources" / "Materials" / "Examples" / "OpenPbr";

    const std::string path = "/Root/MaterialX/Materials/Default";
    auto it = std::find_if(
        scene->materials.begin(),
        scene->materials.end(),
        [&](const ImporterMaterial& m)
        {
            return m.name == path;
        }
    );
    REQUIRE(it != scene->materials.end());

    auto result = convert_material(*it, {"MaterialX"}, {"MaterialXMaterial"});
    REQUIRE(result.has_value());

    CHECK_EQ(result->get<std::string_view>("_scene_material_type", ""), "MaterialXMaterial");
    CHECK_EQ(result->get<std::filesystem::path>("mtlx_path", {}), std::filesystem::path("open_pbr_default.mtlx"));
    CHECK_EQ(result->get<std::string_view>("mtlx_node_name", ""), "Default");

    const std::filesystem::path actual_basepath = result->get<std::filesystem::path>("mtlx_basepath", {});
    CHECK_EQ(actual_basepath.lexically_normal(), expected_mtlx_basepath.lexically_normal());

    mtlx::CodeGenDesc desc;
    desc.document = actual_basepath / result->get<std::filesystem::path>("mtlx_path", {});
    desc.node_name = result->get<std::string>("mtlx_node_name", "");
    auto generated = mtlx::CodeGen::generate(desc);
    REQUIRE(generated);
    CHECK_FALSE(generated->module_source.empty());
}

TEST_CASE("convert_material_materialx_id_network_to_materialxmaterial")
{
    ImporterMaterial material;
    material.name = "/Root/Looks/GeneratedMtlx";

    Properties terminal;
    terminal.set("_path", "/Root/Looks/GeneratedMtlx/Surface");
    terminal.set("_type", "Shader");
    terminal.set("info:id", "ND_standard_surface_surfaceshader");
    terminal.set("info:implementationSource", "id");
    terminal.set("base_color", float3(0.2f, 0.7f, 0.3f));
    terminal.set("base_color:typename", "color3f");
    terminal.set("roughness", 0.4f);
    terminal.set("roughness:typename", "float");

    material.output_to_material_network["_terminal:mtlx:surface"] = {terminal};

    auto result = convert_material(material, {"MaterialX"}, {"MaterialXMaterial"});
    REQUIRE(result.has_value());

    CHECK_EQ(result->get<std::string_view>("_scene_material_type"), "MaterialXMaterial");
    CHECK_FALSE(result->get<std::string>("mtlx_node_name").empty());

    const std::string mtlx_buffer = result->get<std::string>("mtlx_buffer");
    CHECK_FALSE(mtlx_buffer.empty());
    // Verify that the buffer contains MaterialX XML structure
    CHECK(mtlx_buffer.find("<?xml") != std::string::npos);
    CHECK(mtlx_buffer.find("<materialx") != std::string::npos);
    CHECK(mtlx_buffer.find("<implementation") == std::string::npos);
    CHECK(mtlx_buffer.find("<nodedef") == std::string::npos);
    CHECK(mtlx_buffer.find("xi:include") == std::string::npos);
}

TEST_CASE("convert_material_mtlx_openpbr_id_network_to_materialxmaterial")
{
    ImporterMaterial material;
    material.name = "/Root/Looks/GeneratedOpenPbr";

    Properties terminal;
    terminal.set("_path", "/Root/Looks/GeneratedOpenPbr/Surface");
    terminal.set("_type", "Shader");
    terminal.set("info:id", "ND_open_pbr_surface_surfaceshader");
    terminal.set("info:implementationSource", "id");
    terminal.set("base_weight", 1.0f);
    terminal.set("base_weight:typename", "float");
    terminal.set("base_color", float3(0.2f, 0.7f, 0.3f));
    terminal.set("base_color:typename", "color3f");
    terminal.set("specular_roughness", 0.4f);
    terminal.set("specular_roughness:typename", "float");

    material.output_to_material_network["_terminal:mtlx:surface"] = {terminal};

    auto result = convert_material(material, {"MaterialX"}, {"MaterialXMaterial"});
    REQUIRE(result.has_value());

    CHECK_EQ(result->get<std::string_view>("_scene_material_type"), "MaterialXMaterial");
    CHECK_EQ(result->get<std::string_view>("mtlx_node_name", ""), "GeneratedOpenPbr");

    const std::string mtlx_buffer = result->get<std::string>("mtlx_buffer");
    CHECK_FALSE(mtlx_buffer.empty());
    CHECK(mtlx_buffer.find("<?xml") != std::string::npos);
    CHECK(mtlx_buffer.find("<materialx") != std::string::npos);
    CHECK(mtlx_buffer.find("<open_pbr_surface") != std::string::npos);
    CHECK(mtlx_buffer.find("<implementation") == std::string::npos);
    CHECK(mtlx_buffer.find("<nodedef") == std::string::npos);
    CHECK(mtlx_buffer.find("xi:include") == std::string::npos);

    mtlx::CodeGenDesc desc;
    desc.document = mtlx_buffer;
    desc.node_name = result->get<std::string>("mtlx_node_name", "");
    auto generated = mtlx::CodeGen::generate(desc);
    REQUIRE(generated);
    CHECK_FALSE(generated->module_source.empty());
}

TEST_CASE("convert_material_no_match_returns_empty")
{
    const std::filesystem::path test_scene_path = testing::project_directory() / "data" / "assets" / "test_usd_importer"
        / "test_usdpreviewsurface_material_input.usda";
    REQUIRE(std::filesystem::exists(test_scene_path));

    auto importer = falcor::make_ref<falcor::UsdImporter>();
    ref<falcor::ImporterScene> scene = importer->load_scene(test_scene_path);
    REQUIRE(scene);
    REQUIRE_EQ(scene->materials.size(), 1);

    auto result = convert_material(scene->materials[0], {"MDL"}, {"MDLMaterial"});
    CHECK_FALSE(result.has_value());
}

TEST_CASE("load_linear_curves")
{
    const std::filesystem::path curves_path
        = testing::project_directory() / "data" / "assets" / "test_usd_importer" / "test_curves.usda";
    REQUIRE(std::filesystem::exists(curves_path));

    auto importer = falcor::make_ref<falcor::UsdImporter>();
    ref<falcor::ImporterScene> scene = importer->load_scene(curves_path);
    REQUIRE(scene);

    // We expect 6 curve prims (2 linear + 4 cubic).
    REQUIRE_EQ(scene->curves.size(), 6);

    // Find curves by exact trailing name match.
    auto find_curve = [&](const std::string& name) -> const ImporterCurve*
    {
        for (const auto& c : scene->curves) {
            if (c.name.size() >= name.size() && c.name.substr(c.name.size() - name.size()) == name)
                return &c;
        }
        return nullptr;
    };

    // First curve: 3 vertices, 2 segments.
    {
        const ImporterCurve* curve = find_curve("linear_curves");
        REQUIRE(curve);
        CHECK_EQ(curve->position_stream().size, 5);
        CHECK_EQ(curve->radius_stream().size, 5);
        // First curve has 3 verts (2 segments), second has 2 verts (1 segment) = 3 segments * 2 indices = 6.
        CHECK_EQ(curve->indices.size(), 6);

        // Check radii are half of widths.
        CHECK(curve->radius_stream()[0] == doctest::Approx(0.05f));
        CHECK(curve->radius_stream()[1] == doctest::Approx(0.04f));
        CHECK(curve->radius_stream()[2] == doctest::Approx(0.03f));

        // Check indices form line segments: (0,1), (1,2), (3,4).
        CHECK_EQ(curve->indices[0], 0);
        CHECK_EQ(curve->indices[1], 1);
        CHECK_EQ(curve->indices[2], 1);
        CHECK_EQ(curve->indices[3], 2);
        CHECK_EQ(curve->indices[4], 3);
        CHECK_EQ(curve->indices[5], 4);
    }

    // Second curve: no widths, should use default radius.
    {
        const ImporterCurve* curve = find_curve("linear_curves_no_widths");
        REQUIRE(curve);
        CHECK_EQ(curve->position_stream().size, 2);
        CHECK_EQ(curve->radius_stream().size, 2);
        CHECK_EQ(curve->indices.size(), 2);

        // Default radius should be 0.01.
        CHECK(curve->radius_stream()[0] == doctest::Approx(0.01f));
        CHECK(curve->radius_stream()[1] == doctest::Approx(0.01f));
    }

    // Check nodes reference curves.
    int curve_node_count = 0;
    for (const auto& node : scene->nodes) {
        if (node.curve_index >= 0)
            curve_node_count++;
    }
    CHECK_EQ(curve_node_count, 6);
}

TEST_CASE("load_cubic_bezier_curves")
{
    const std::filesystem::path curves_path
        = testing::project_directory() / "data" / "assets" / "test_usd_importer" / "test_curves.usda";

    auto importer = falcor::make_ref<falcor::UsdImporter>();
    ref<falcor::ImporterScene> scene = importer->load_scene(curves_path, {.curve_refinement_level = 7});
    REQUIRE(scene);

    // Find curves by exact trailing name match.
    auto find_curve = [&](const std::string& name) -> const ImporterCurve*
    {
        for (const auto& c : scene->curves) {
            if (c.name.size() >= name.size() && c.name.substr(c.name.size() - name.size()) == name)
                return &c;
        }
        return nullptr;
    };

    // Bezier curve: 4 CVs = 1 span, 8 segments per span = 9 tessellated vertices.
    {
        const ImporterCurve* curve = find_curve("bezier_curve");
        REQUIRE(curve);
        CHECK_EQ(curve->position_stream().size, 9); // 8 segments + 1 = 9 vertices
        CHECK_EQ(curve->radius_stream().size, 9);
        CHECK_EQ(curve->indices.size(), 16); // 8 segments * 2 indices

        // Bezier curve endpoints should match first and last control points.
        CHECK(curve->position_stream()[0].x == doctest::Approx(0.0f));
        CHECK(curve->position_stream()[0].y == doctest::Approx(0.0f));
        CHECK(curve->position_stream()[curve->vertex_count() - 1].x == doctest::Approx(1.0f));
        CHECK(curve->position_stream()[curve->vertex_count() - 1].y == doctest::Approx(0.0f));

        // Radii at endpoints: widths are [0.2, 0.2, 0.1, 0.1], radii = width/2.
        // Bezier interpolates endpoints, so first radius = 0.1, last radius = 0.05.
        CHECK(curve->radius_stream()[0] == doctest::Approx(0.1f));
        CHECK(curve->radius_stream()[curve->vertex_count() - 1] == doctest::Approx(0.05f));

        // Midpoint (t=0.5) of bezier with CVs (0,0,0),(0,1,0),(1,1,0),(1,0,0):
        // P(0.5) = 0.125*(0,0,0) + 0.375*(0,1,0) + 0.375*(1,1,0) + 0.125*(1,0,0)
        //        = (0.5, 0.75, 0)
        CHECK(curve->position_stream()[4].x == doctest::Approx(0.5f));
        CHECK(curve->position_stream()[4].y == doctest::Approx(0.75f));
    }

    // Multi-span Bezier: 7 CVs = 2 spans (stride 3), 2*8=16 segments, 17 tessellated vertices.
    {
        const ImporterCurve* curve = find_curve("bezier_multi_span");
        REQUIRE(curve);
        CHECK_EQ(curve->position_stream().size, 17); // 2 spans * 8 + 1 = 17 vertices
        CHECK_EQ(curve->radius_stream().size, 17);
        CHECK_EQ(curve->indices.size(), 32); // 16 segments * 2 indices

        // First endpoint = first CV (0,0,0).
        CHECK(curve->position_stream()[0].x == doctest::Approx(0.0f));
        CHECK(curve->position_stream()[0].y == doctest::Approx(0.0f));
        // Last endpoint = last CV (3,1,0).
        CHECK(curve->position_stream()[curve->vertex_count() - 1].x == doctest::Approx(3.0f));
        CHECK(curve->position_stream()[curve->vertex_count() - 1].y == doctest::Approx(1.0f));

        // Junction between spans (vertex 8) should be CV[3] = (1,0,0).
        CHECK(curve->position_stream()[8].x == doctest::Approx(1.0f));
        CHECK(curve->position_stream()[8].y == doctest::Approx(0.0f));
    }

    // B-spline curve: 4 CVs = 1 span, 9 tessellated vertices.
    {
        const ImporterCurve* curve = find_curve("bspline_curve");
        REQUIRE(curve);
        CHECK_EQ(curve->position_stream().size, 9);
        CHECK_EQ(curve->indices.size(), 16);

        // B-spline does NOT interpolate endpoints.
        // P(0) with bspline basis and CVs (0,0,0),(0,1,0),(1,1,0),(1,0,0):
        // = (1/6)*(0,0,0) + (4/6)*(0,1,0) + (1/6)*(1,1,0) + 0*(1,0,0)
        // = (1/6, 5/6, 0)
        CHECK(curve->position_stream()[0].x == doctest::Approx(1.f / 6.f));
        CHECK(curve->position_stream()[0].y == doctest::Approx(5.f / 6.f));
    }

    // CatmullRom curve: 4 CVs = 1 span, 9 tessellated vertices.
    {
        const ImporterCurve* curve = find_curve("catmullrom_curve");
        REQUIRE(curve);
        CHECK_EQ(curve->position_stream().size, 9);
        CHECK_EQ(curve->indices.size(), 16);

        // CatmullRom interpolates through P1 and P2.
        // P(0) = P1 = (0,1,0)
        CHECK(curve->position_stream()[0].x == doctest::Approx(0.0f));
        CHECK(curve->position_stream()[0].y == doctest::Approx(1.0f));
        // P(1) = P2 = (1,1,0)
        CHECK(curve->position_stream()[curve->vertex_count() - 1].x == doctest::Approx(1.0f));
        CHECK(curve->position_stream()[curve->vertex_count() - 1].y == doctest::Approx(1.0f));
    }
}

// ---------------------------------------------------------------------------
// Animation import tests
// ---------------------------------------------------------------------------

/// Helper to find a node by name in an ImporterScene.
static const ImporterNode* find_node(const ref<ImporterScene>& scene, const std::string& name)
{
    for (const auto& node : scene->nodes) {
        if (node.name == name)
            return &node;
    }
    return nullptr;
}

/// Helper to get the index of a node by name.
static int find_node_index(const ref<ImporterScene>& scene, const std::string& name)
{
    for (size_t i = 0; i < scene->nodes.size(); ++i) {
        if (scene->nodes[i].name == name)
            return static_cast<int>(i);
    }
    return -1;
}

TEST_CASE("UsdImporter - Linear Translation Animation")
{
    auto importer = make_ref<UsdImporter>();
    auto file
        = testing::project_directory() / "data" / "assets" / "animation_test" / "usd_anim_linear_translation.usda";
    REQUIRE(std::filesystem::exists(file));

    auto scene = importer->load_scene(file);
    REQUIRE(scene != nullptr);

    // Should have animation data.
    CHECK_FALSE(scene->animation.channels.empty());
    CHECK_EQ(scene->animation.name, "UsdAnimation");

    // Find the animated node.
    const ImporterNode* cube_node = find_node(scene, "/Cube");
    REQUIRE(cube_node != nullptr);

    // Should have a translation animation channel.
    REQUIRE_NE(cube_node->animation_translation, -1);
    CHECK_EQ(cube_node->animation_rotation, -1);
    CHECK_EQ(cube_node->animation_scale, -1);

    const auto& ch = scene->animation.channels[cube_node->animation_translation];
    CHECK_EQ(ch.value_type, AnimationValueType::float3);
    CHECK_EQ(ch.interpolation_mode, AnimationInterpolationMode::linear);
    REQUIRE_EQ(ch.keyframe_count(), 3);

    // Time codes 0, 24, 48 at 24 fps => 0s, 1s, 2s.
    CHECK_EQ(ch.times[0], doctest::Approx(0.f));
    CHECK_EQ(ch.times[1], doctest::Approx(1.f));
    CHECK_EQ(ch.times[2], doctest::Approx(2.f));

    // Values: (0,0,0), (2.5,0,0), (5,0,0).
    REQUIRE_EQ(ch.values.size(), 9);
    CHECK_EQ(ch.values[0], doctest::Approx(0.f).epsilon(1e-4));
    CHECK_EQ(ch.values[1], doctest::Approx(0.f).epsilon(1e-4));
    CHECK_EQ(ch.values[2], doctest::Approx(0.f).epsilon(1e-4));
    CHECK_EQ(ch.values[3], doctest::Approx(2.5f).epsilon(1e-4));
    CHECK_EQ(ch.values[4], doctest::Approx(0.f).epsilon(1e-4));
    CHECK_EQ(ch.values[5], doctest::Approx(0.f).epsilon(1e-4));
    CHECK_EQ(ch.values[6], doctest::Approx(5.f).epsilon(1e-4));
    CHECK_EQ(ch.values[7], doctest::Approx(0.f).epsilon(1e-4));
    CHECK_EQ(ch.values[8], doctest::Approx(0.f).epsilon(1e-4));

    // No tangent data for linear.
    CHECK(ch.in_tangents.empty());
    CHECK(ch.out_tangents.empty());

    CHECK_EQ(scene->animation.key_range().end, doctest::Approx(2.f));
}

TEST_CASE("UsdImporter - Rotation Animation")
{
    auto importer = make_ref<UsdImporter>();
    auto file = testing::project_directory() / "data" / "assets" / "animation_test" / "usd_anim_rotation.usda";
    REQUIRE(std::filesystem::exists(file));

    auto scene = importer->load_scene(file);
    REQUIRE(scene != nullptr);

    const ImporterNode* cube_node = find_node(scene, "/Cube");
    REQUIRE(cube_node != nullptr);

    // Should have a rotation animation channel (from matrix decomposition).
    REQUIRE_NE(cube_node->animation_rotation, -1);

    const auto& ch = scene->animation.channels[cube_node->animation_rotation];
    CHECK_EQ(ch.value_type, AnimationValueType::quatf);
    CHECK_EQ(ch.interpolation_mode, AnimationInterpolationMode::linear);
    REQUIRE_EQ(ch.keyframe_count(), 2);

    // Time codes 0, 24 at 24 fps => 0s, 1s.
    CHECK_EQ(ch.times[0], doctest::Approx(0.f));
    CHECK_EQ(ch.times[1], doctest::Approx(1.f));

    // First keyframe: identity rotation (0, 0, 0, 1) in xyzw order.
    REQUIRE_EQ(ch.values.size(), 8);
    CHECK_EQ(ch.values[0], doctest::Approx(0.f).epsilon(1e-3));
    CHECK_EQ(ch.values[1], doctest::Approx(0.f).epsilon(1e-3));
    CHECK_EQ(ch.values[2], doctest::Approx(0.f).epsilon(1e-3));
    CHECK_EQ(ch.values[3], doctest::Approx(1.f).epsilon(1e-3));

    // Second keyframe: 90 degrees around Y => (0, sin(45), 0, cos(45)).
    CHECK_EQ(ch.values[4], doctest::Approx(0.f).epsilon(1e-3));
    CHECK_EQ(ch.values[5], doctest::Approx(0.7071068f).epsilon(1e-3));
    CHECK_EQ(ch.values[6], doctest::Approx(0.f).epsilon(1e-3));
    CHECK_EQ(ch.values[7], doctest::Approx(0.7071068f).epsilon(1e-3));
}

TEST_CASE("UsdImporter - Multi Channel Animation")
{
    auto importer = make_ref<UsdImporter>();
    auto file = testing::project_directory() / "data" / "assets" / "animation_test" / "usd_anim_multi_channel.usda";
    REQUIRE(std::filesystem::exists(file));

    auto scene = importer->load_scene(file);
    REQUIRE(scene != nullptr);

    const ImporterNode* cube_node = find_node(scene, "/Cube");
    REQUIRE(cube_node != nullptr);

    // Should have all three animation channels.
    CHECK_NE(cube_node->animation_translation, -1);
    CHECK_NE(cube_node->animation_rotation, -1);
    CHECK_NE(cube_node->animation_scale, -1);

    // The union of time samples is {0, 48, 72} => {0s, 2s, 3s}.
    const auto& trans_ch = scene->animation.channels[cube_node->animation_translation];
    CHECK_EQ(trans_ch.value_type, AnimationValueType::float3);
    REQUIRE_EQ(trans_ch.keyframe_count(), 3);
    CHECK_EQ(trans_ch.times[0], doctest::Approx(0.f));
    CHECK_EQ(trans_ch.times[1], doctest::Approx(2.f));
    CHECK_EQ(trans_ch.times[2], doctest::Approx(3.f));

    // Translation at t=2s should be (1,2,3), at t=3s held at (1,2,3).
    CHECK_EQ(trans_ch.values[3], doctest::Approx(1.f).epsilon(1e-3));
    CHECK_EQ(trans_ch.values[4], doctest::Approx(2.f).epsilon(1e-3));
    CHECK_EQ(trans_ch.values[5], doctest::Approx(3.f).epsilon(1e-3));

    const auto& scale_ch = scene->animation.channels[cube_node->animation_scale];
    CHECK_EQ(scale_ch.value_type, AnimationValueType::float3);
    REQUIRE_EQ(scale_ch.keyframe_count(), 3);

    // Scale at t=3s should be (2,2,2).
    CHECK_EQ(scale_ch.values[6], doctest::Approx(2.f).epsilon(1e-3));
    CHECK_EQ(scale_ch.values[7], doctest::Approx(2.f).epsilon(1e-3));
    CHECK_EQ(scale_ch.values[8], doctest::Approx(2.f).epsilon(1e-3));

    // Duration should be 3s (max time across channels).
    CHECK_EQ(scene->animation.key_range().end, doctest::Approx(3.f));
}

TEST_CASE("UsdImporter - Multi Node Animation")
{
    auto importer = make_ref<UsdImporter>();
    auto file = testing::project_directory() / "data" / "assets" / "animation_test" / "usd_anim_multi_node.usda";
    REQUIRE(std::filesystem::exists(file));

    auto scene = importer->load_scene(file);
    REQUIRE(scene != nullptr);

    REQUIRE_GE(scene->animation.channels.size(), 2);

    const ImporterNode* parent_node = find_node(scene, "/Parent");
    REQUIRE(parent_node != nullptr);
    const ImporterNode* child_node = find_node(scene, "/Parent/Child");
    REQUIRE(child_node != nullptr);

    // Parent has translation animation.
    CHECK_NE(parent_node->animation_translation, -1);

    // Child has rotation animation.
    CHECK_NE(child_node->animation_rotation, -1);

    // Verify parent-child relationship.
    int parent_idx = find_node_index(scene, "/Parent");
    CHECK_EQ(child_node->parent, parent_idx);

    // Parent translation: (0,0,0) -> (10,0,0).
    const auto& parent_ch = scene->animation.channels[parent_node->animation_translation];
    CHECK_EQ(parent_ch.value_type, AnimationValueType::float3);
    REQUIRE_EQ(parent_ch.keyframe_count(), 2);
    CHECK_EQ(parent_ch.values[3], doctest::Approx(10.f).epsilon(1e-3));

    // Child rotation channel.
    const auto& child_ch = scene->animation.channels[child_node->animation_rotation];
    CHECK_EQ(child_ch.value_type, AnimationValueType::quatf);
    REQUIRE_EQ(child_ch.keyframe_count(), 2);
}

TEST_CASE("UsdImporter - No Animation")
{
    auto importer = make_ref<UsdImporter>();
    auto file = testing::project_directory() / "data" / "assets" / "animation_test" / "usd_anim_static.usda";
    REQUIRE(std::filesystem::exists(file));

    auto scene = importer->load_scene(file);
    REQUIRE(scene != nullptr);

    // Scene without animation should have empty animation data.
    CHECK(scene->animation.channels.empty());
    CHECK_FALSE(scene->animation.key_range().is_valid());

    // No node should reference any animation channel.
    for (const auto& node : scene->nodes) {
        CHECK_EQ(node.animation_translation, -1);
        CHECK_EQ(node.animation_rotation, -1);
        CHECK_EQ(node.animation_scale, -1);
    }
}

TEST_CASE("UsdImporter - Time Code Normalization (30fps)")
{
    auto importer = make_ref<UsdImporter>();
    auto file = testing::project_directory() / "data" / "assets" / "animation_test" / "usd_anim_30fps.usda";
    REQUIRE(std::filesystem::exists(file));

    auto scene = importer->load_scene(file);
    REQUIRE(scene != nullptr);

    const ImporterNode* cube_node = find_node(scene, "/Cube");
    REQUIRE(cube_node != nullptr);
    REQUIRE_NE(cube_node->animation_translation, -1);

    const auto& ch = scene->animation.channels[cube_node->animation_translation];
    CHECK_EQ(ch.value_type, AnimationValueType::float3);
    REQUIRE_EQ(ch.keyframe_count(), 3);

    // Time codes 0, 30, 60 at 30 fps => 0s, 1s, 2s.
    CHECK_EQ(ch.times[0], doctest::Approx(0.f));
    CHECK_EQ(ch.times[1], doctest::Approx(1.f));
    CHECK_EQ(ch.times[2], doctest::Approx(2.f));

    // Values: (0,0,0), (3,0,0), (6,0,0).
    CHECK_EQ(ch.values[0], doctest::Approx(0.f).epsilon(1e-4));
    CHECK_EQ(ch.values[3], doctest::Approx(3.f).epsilon(1e-4));
    CHECK_EQ(ch.values[6], doctest::Approx(6.f).epsilon(1e-4));

    CHECK_EQ(scene->animation.key_range().end, doctest::Approx(2.f));
}

TEST_CASE("UsdImporter - stage import preserves source asset identity")
{
    auto stage = pxr::UsdStage::CreateInMemory();
    std::string source_asset_path;
    SUBCASE("anonymous stage") { }
    SUBCASE("file-backed stage")
    {
        const auto path = testing::get_case_temp_directory() / "source.usda";
        stage = pxr::UsdStage::CreateNew(path.string());
        REQUIRE(stage);
        source_asset_path = stage->GetRootLayer()->GetRealPath();
        REQUIRE_FALSE(source_asset_path.empty());
    }
    REQUIRE(stage);
    REQUIRE(stage->DefinePrim(pxr::SdfPath("/Root"), pxr::TfToken("Xform")));
    REQUIRE(stage->DefinePrim(pxr::SdfPath("/Root/Camera"), pxr::TfToken("Camera")));
    REQUIRE(stage->DefinePrim(pxr::SdfPath("/Root/Light"), pxr::TfToken("SphereLight")));

    const auto scene = UsdImporter().import_stage(*stage);
    REQUIRE(scene);
    REQUIRE_EQ(scene->nodes.size(), 4);
    REQUIRE_EQ(scene->cameras.size(), 1);
    REQUIRE_EQ(scene->lights.size(), 1);
    REQUIRE(scene->cameras[0].source_asset_ref);
    CHECK_FALSE(scene->nodes[0].source_asset_ref.has_value());
    for (size_t index = 1; index < scene->nodes.size(); ++index) {
        const auto& source = scene->nodes[index].source_asset_ref;
        REQUIRE(source);
        CHECK_EQ(source->format, "usd");
        CHECK_EQ(source->asset_path, source_asset_path);
        CHECK(source->asset_path.data() == scene->cameras[0].source_asset_ref->asset_path.data());
        CHECK_FALSE(source->object_path.empty());
    }
    CHECK(scene->cameras[0].source_asset_ref == SourceAssetRef("usd", source_asset_path, "/Root/Camera"));
    CHECK(scene->lights[0].source_asset_ref == SourceAssetRef("usd", source_asset_path, "/Root/Light"));
}


TEST_CASE("materialx_direct_material_reference")
{
    using namespace pxr;
    const auto directory = testing::get_case_temp_directory();
    std::filesystem::create_directories(directory / "looks");
    const auto material_path = directory / "looks/materials.mtlx";
    std::ofstream(material_path) << R"(<?xml version="1.0"?>
<materialx version="1.38">
  <standard_surface name="surface" type="surfaceshader" />
  <surfacematerial name="red" type="material">
    <input name="surfaceshader" type="surfaceshader" nodename="surface" />
  </surfacematerial>
  <surfacematerial name="paint:blue" type="material">
    <input name="surfaceshader" type="surfaceshader" nodename="surface" />
  </surfacematerial>
</materialx>)";
    const auto scene_path = directory / "scene.usda";
    const auto weak_path = directory / "looks/asset.usda";
    std::ofstream(weak_path) << R"(#usda 1.0
 def Xform "World" {
    def Material "Paint" (
        prepend references = @materials.mtlx@</MaterialX/Materials/paint__blue>
    ) {}
    def Mesh "Triangle" (prepend apiSchemas = ["MaterialBindingAPI"]) {
        int[] faceVertexCounts = [3]
        int[] faceVertexIndices = [0, 1, 2]
        point3f[] points = [(0,0,0), (1,0,0), (0,1,0)]
        uniform token subdivisionScheme = "none"
        rel material:binding = </World/Paint>
    }
})";
    std::string opinion;
    std::string expected_node = "paint:blue";
    bool nested_reference = false;
    SUBCASE("Select one namespaced material through a nested USD reference")
    {
        nested_reference = true;
    }
    SUBCASE("Explicit USD reference on the material retains its nested MaterialX reference")
    {
        opinion = "references = @looks/asset.usda@</World/Paint>";
    }
    SUBCASE("Stronger prim opinions do not change the reference's source directory")
    {
        opinion = "documentation = \"stronger opinion\"";
    }
    SUBCASE("Stronger deletion does not resurrect the weak reference")
    {
        opinion = "delete references = @looks/materials.mtlx@</MaterialX/Materials/paint__blue>";
        expected_node.clear();
    }
    SUBCASE("Explicit replacement selects a different material")
    {
        opinion = "references = @looks/materials.mtlx@</MaterialX/Materials/red>";
        expected_node = "red";
    }
    SUBCASE("Missing target does not select another material from the document")
    {
        opinion = "references = @looks/materials.mtlx@</MaterialX/Materials/missing>";
        expected_node.clear();
    }
    SUBCASE("The strongest direct material reference wins")
    {
        opinion = "references = [@looks/materials.mtlx@</MaterialX/Materials/red>, "
                  "@looks/materials.mtlx@</MaterialX/Materials/paint__blue>]";
        expected_node = "red";
    }
    std::ofstream(scene_path) << fmt::format(
        R"(#usda 1.0
(defaultPrim = "World" {})
def Xform "World" ({}) {{
    over "Paint" ({}) {{}}
}})",
        nested_reference ? "" : "\nsubLayers = [@looks/asset.usda@]",
        nested_reference ? "prepend references = @looks/asset.usda@</World>" : "",
        opinion
    );
    auto stage = UsdStage::Open(scene_path.string());
    REQUIRE(stage);
    std::string original_root;
    stage->GetRootLayer()->ExportToString(&original_root);
    const auto scene = UsdImporter().import_stage(*stage);
    REQUIRE_EQ(scene->meshes.size(), 1);
    REQUIRE_EQ(scene->materials.size(), 1);
    CHECK_EQ(scene->materials[0].name, "/World/Paint");
    if (expected_node.empty()) {
        CHECK(scene->materials[0].output_to_material_network.empty());
    } else {
        auto converted = convert_material(scene->materials[0]);
        REQUIRE(converted);
        CHECK_EQ(converted->get<std::string>("_scene_material_type"), "MaterialXMaterial");
        CHECK_EQ(converted->get<std::string>("mtlx_node_name"), expected_node);
        CHECK_EQ(converted->get<std::filesystem::path>("mtlx_path"), material_path.filename());
        CHECK_EQ(
            converted->get<std::filesystem::path>("mtlx_basepath").lexically_normal(),
            material_path.parent_path()
        );
    }
    std::string root_after_import;
    stage->GetRootLayer()->ExportToString(&root_after_import);
    CHECK_EQ(root_after_import, original_root);
    CHECK_FALSE(stage->GetPrimAtPath(SdfPath("/World/Paint/Shader")));
}

TEST_CASE("Round varying widths interpolate span boundaries across strands")
{
    const char* source = R"(#usda 1.0
def BasisCurves "Round" {
 uniform token type = "cubic"
 uniform token basis = "bezier"
 int[] curveVertexCounts = [7, 4]
 point3f[] points = [(0,0,0),(1,0,0),(2,0,0),(3,0,0),(4,0,0),(5,0,0),(6,0,0),
                     (0,3,0),(1,3,0),(2,3,0),(3,3,0)]
 float[] widths = [0,0.8,0,0.4,0.8] (interpolation = "varying")
}
)";
    for (int level : {0, 3}) {
        auto stage = pxr::UsdStage::CreateInMemory();
        REQUIRE(stage->GetRootLayer()->ImportFromString(source));
        auto scene = UsdImporter().import_stage(*stage, {.curve_refinement_level = level});
        REQUIRE(scene->curves.size() == 1);
        const auto& curve = scene->curves.front();
        const size_t first_count = level == 0 ? 7 : 9;
        REQUIRE(curve.radius_stream().size == (level == 0 ? 11 : 14));
        for (size_t i = 0; i < first_count; ++i) {
            const float x = curve.position_stream()[i].x;
            CHECK(curve.radius_stream()[i] == doctest::Approx(.4f * (1.f - std::abs(x - 3.f) / 3.f)));
        }
        for (size_t i = first_count; i < curve.radius_stream().size; ++i)
            CHECK(curve.radius_stream()[i] == doctest::Approx(.2f + .2f * curve.position_stream()[i].x / 3.f));
    }
}

TEST_CASE("Round uniform widths select each strand")
{
    const char* source = R"(#usda 1.0
def BasisCurves "Round" {
 uniform token type = "cubic"
 uniform token basis = "bezier"
 int[] curveVertexCounts = [4,4]
 point3f[] points = [(0,0,0),(1,0,0),(2,0,0),(3,0,0), (0,3,0),(1,3,0),(2,3,0),(3,3,0)]
 float[] widths = [0.4,0.8] (interpolation = "uniform")
}
)";
    for (int level : {0, 3}) {
        auto stage = pxr::UsdStage::CreateInMemory();
        REQUIRE(stage->GetRootLayer()->ImportFromString(source));
        auto scene = UsdImporter().import_stage(*stage, {.curve_refinement_level = level});
        REQUIRE(scene->curves.size() == 1);
        const auto& curve = scene->curves.front();
        REQUIRE(curve.radius_stream().size == (level == 0 ? 8 : 10));
        for (size_t i = 0; i < curve.radius_stream().size; ++i)
            CHECK(curve.radius_stream()[i] == doctest::Approx(i < curve.radius_stream().size / 2 ? .2f : .4f));
    }
}


TEST_SUITE_END();
