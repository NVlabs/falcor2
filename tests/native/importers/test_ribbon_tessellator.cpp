// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "testing.h"
#include "falcor2/importers/curve_tessellator/curve_tessellator.h"
#include "falcor2/importers/importer.h"
#include "falcor2/importers/usd_importer/usd_importer.h"
#include <array>
#include <cmath>
#include <fstream>
#include <limits>

using namespace falcor;
using namespace falcor::curve_tessellator;
TEST_SUITE_BEGIN("ribbon_tessellator");

namespace {
struct Input {
    std::vector<float3> points{{0, 0, 0}, {2, 0, 0}};
    std::vector<int> counts{2};
    std::vector<float3> normals{{0, 0, 1}};
    std::vector<float> widths{2.f};
    std::vector<float2> uvs;
    CurveInput input;
    CurveSample sample;
    Input()
    {
        input.name = "test ribbon";
        input.properties = ImporterMesh::make_properties({});
        sample.normals.interpolation = Interpolation::constant;
        sample.widths.interpolation = Interpolation::constant;
    }
    ImporterMesh run()
    {
        sample.points = points;
        input.vertex_counts = counts;
        sample.normals.values = normals;
        sample.widths.values = widths;
        input.texcoords.values = uvs;
        input.sample = sample;
        return tessellate_ribbons(input);
    }
};
void finite_mesh(const ImporterMesh& mesh)
{
    for (auto p : mesh.position_stream())
        CHECK(std::isfinite(p.x + p.y + p.z));
    for (auto n : mesh.normal_stream()) {
        CHECK(std::isfinite(n.x + n.y + n.z));
        CHECK(sgl::math::dot(n, n) == doctest::Approx(1.f));
    }
}
ref<ImporterScene> load_source(std::string_view source, const ImportOptions& options = {})
{
    const auto path = testing::get_case_temp_directory() / "ribbon.usda";
    {
        std::ofstream out(path);
        out << source;
    }
    return UsdImporter().load_scene(path, options);
}
const char* STRIP = R"(#usda 1.0
def BasisCurves "Strip" {
 uniform token type = "linear"
 int[] curveVertexCounts = [2]
 point3f[] points = [(0,0,0), (2,0,0)]
 normal3f[] normals = [(0,0,1)] (interpolation = "constant")
 float[] widths = [2] (interpolation = "constant")
}
)";
} // namespace

TEST_CASE("Straight ribbon full width and winding")
{
    Input t;
    for (bool left : {false, true}) {
        t.input.left_handed = left;
        auto m = t.run();
        REQUIRE(m.vertex_count() == 4);
        REQUIRE(m.subgeometries[0].indices.size() == 2);
        finite_mesh(m);
        const auto triangle = m.subgeometries[0].indices[0];
        const auto positions = m.position_stream();
        const auto geometric = sgl::math::cross(
            positions[triangle.y] - positions[triangle.x],
            positions[triangle.z] - positions[triangle.x]
        );
        CHECK((geometric.z < 0.f) == left);
        for (auto p : m.position_stream()) {
            CHECK(std::abs(p.y) == doctest::Approx(1.f));
            CHECK(p.z == 0.f);
        }
        for (auto n : m.normal_stream())
            CHECK(n.z == doctest::Approx(1.f));
    }
    t.widths.clear();
    auto m = t.run();
    for (auto p : m.position_stream())
        CHECK(std::abs(p.y) == doctest::Approx(.01f));
}

TEST_CASE("Pinned two-point splines interpolate endpoints")
{
    for (auto basis : {Basis::bspline, Basis::catmull_rom}) {
        Input t;
        t.input.type = Type::cubic;
        t.input.basis = basis;
        t.input.wrap = Wrap::pinned;
        t.input.refinement_level = 3;
        t.widths = {2.f, 4.f};
        t.sample.widths.interpolation = Interpolation::varying;
        auto m = t.run();
        REQUIRE(m.vertex_count() == 10);
        finite_mesh(m);
        float min_x = 100.f, max_x = -100.f;
        for (auto p : m.position_stream()) {
            min_x = std::min(min_x, p.x);
            max_x = std::max(max_x, p.x);
            CHECK(std::abs(p.y) == doctest::Approx(1.f + .5f * p.x));
        }
        CHECK(min_x == doctest::Approx(0.f));
        CHECK(max_x == doctest::Approx(2.f));
    }
}

TEST_CASE("Cubic bases sample a straight line and distinguish varying widths")
{
    for (auto basis : {Basis::bezier, Basis::bspline, Basis::catmull_rom}) {
        Input t;
        t.points = {{0, 0, 0}, {1, 0, 0}, {2, 0, 0}, {3, 0, 0}};
        t.counts = {4};
        t.input.type = Type::cubic;
        t.input.basis = basis;
        t.input.refinement_level = 3;
        t.widths = {2, 4};
        t.sample.widths.interpolation = Interpolation::varying;
        auto m = t.run();
        REQUIRE(m.vertex_count() == 10);
        finite_mesh(m);
        const float start = basis == Basis::bezier ? 0.f : 1.f;
        const float length = basis == Basis::bezier ? 3.f : 1.f;
        for (auto p : m.position_stream())
            CHECK(std::abs(p.y) == doctest::Approx(1.f + (p.x - start) / length));
        t.sample.widths.interpolation = Interpolation::vertex;
        t.widths = {2, 2, 2, 2};
        m = t.run();
        for (auto p : m.position_stream())
            CHECK(std::abs(p.y) == doctest::Approx(1.f));
        t.input.refinement_level = 0;
        m = t.run();
        CHECK(m.vertex_count() == 8);
    }
}

TEST_CASE("Uniform values do not connect independent strands")
{
    Input t;
    t.points = {{0, 0, 0}, {2, 0, 0}, {0, 5, 0}, {2, 5, 0}};
    t.counts = {2, 2};
    t.widths = {2, 4};
    t.sample.widths.interpolation = Interpolation::uniform;
    t.normals = {{0, 0, 1}, {0, 0, 1}};
    t.sample.normals.interpolation = Interpolation::uniform;
    auto m = t.run();
    REQUIRE(m.subgeometries[0].indices.size() == 4);
    for (auto tri : m.subgeometries[0].indices) {
        const auto p = m.position_stream();
        CHECK(std::abs(p[tri.x].y - p[tri.y].y) <= 4.f);
    }
}

TEST_CASE("Linear twist and authored centerline UVs")
{
    Input t;
    t.normals = {{0, 0, 1}, {0, 1, 0}};
    t.sample.normals.interpolation = Interpolation::vertex;
    t.input.refinement_level = 3;
    t.uvs = {{.2f, .3f}, {.8f, .7f}};
    auto m = t.run();
    finite_mesh(m);
    for (size_t i = 0; i < m.vertex_count(); ++i) {
        const auto p = m.position_stream()[i];
        const auto uv = m.texcoord_stream()[i];
        CHECK(uv.x == doctest::Approx(.2f + .3f * p.x));
        CHECK(uv.y == doctest::Approx(.3f + .2f * p.x));
        CHECK(std::sqrt(p.y * p.y + p.z * p.z) == doctest::Approx(1.f));
        const float u = p.x * .5f;
        const float3 expected_side = sgl::math::normalize(float3(0, 1.f - u, -u));
        CHECK(std::abs(sgl::math::dot(float3(0, p.y, p.z), expected_side)) == doctest::Approx(1.f));
    }
}


TEST_CASE("Generated UVs use transverse U and equal V ranges for unequal Bezier spans")
{
    Input t;
    t.points = {{0, 0, 0}, {1.f / 3, 0, 0}, {2.f / 3, 0, 0}, {1, 0, 0}, {8.f / 3, 0, 0}, {13.f / 3, 0, 0}, {6, 0, 0}};
    t.counts = {7};
    t.input.type = Type::cubic;
    t.input.basis = Basis::bezier;
    for (int level : {0, 1, 7, 31}) {
        t.input.refinement_level = level;
        auto mesh = t.run();
        for (size_t i = 0; i < mesh.vertex_count(); ++i) {
            const auto p = mesh.position_stream()[i];
            const auto uv = mesh.texcoord_stream()[i];
            const float v = p.x <= 1.f ? p.x * .5f : .5f + (p.x - 1.f) / 10.f;
            CHECK(uv.x == doctest::Approx(.5f + p.y / 2.f));
            CHECK(uv.y == doctest::Approx(v));
        }
    }
}

TEST_CASE("Periodic seams have matching normals and distinct UVs")
{
    for (Type type : {Type::linear, Type::cubic}) {
        Input t;
        t.points = {{0, 0, 0}, {2, 0, 0}, {2, 2, 1}, {0, 2, 0}};
        t.counts = {4};
        t.widths = {.2f};
        t.input.wrap = Wrap::periodic;
        t.input.type = type;
        t.input.basis = Basis::bspline;
        t.input.refinement_level = 3;
        auto m = t.run();
        finite_mesh(m);
        int matches = 0;
        for (size_t a = 0; a < m.vertex_count(); ++a)
            for (size_t b = a + 1; b < m.vertex_count(); ++b) {
                if (sgl::math::length(m.position_stream()[a] - m.position_stream()[b]) < 1e-6f
                    && std::abs(m.texcoord_stream()[a].y - m.texcoord_stream()[b].y) > .9f) {
                    CHECK(sgl::math::length(m.normal_stream()[a] - m.normal_stream()[b]) < 1e-6f);
                    ++matches;
                }
            }
        CHECK(matches == 2);
    }
}

TEST_CASE("Tapered tips omit unreferenced vertices before normal recomputation")
{
    Input t;
    t.widths = {0, 2};
    t.sample.widths.interpolation = Interpolation::vertex;
    auto m = t.run();
    REQUIRE(m.vertex_count() == 3);
    REQUIRE(m.subgeometries[0].indices.size() == 1);
    m.add_normals_from_faces();
    m.add_tangents_from_normals();
    finite_mesh(m);
    t.widths = {0, 0};
    m = t.run();
    CHECK(!m.position_stream().valid());
}

TEST_CASE("Malformed and degenerate ribbon input is diagnosed")
{
    Input t;
    t.counts = {3};
    CHECK_THROWS(t.run());
    t.counts = {2};
    t.widths = {-1};
    CHECK_THROWS(t.run());
    t.widths = {2};
    t.normals = {{1, 0, 0}};
    CHECK_THROWS(t.run());
    t.normals = {{0, 0, 0}};
    CHECK_THROWS(t.run());
    t.normals = {{0, 0, 1}};
    t.points[1] = t.points[0];
    CHECK(!t.run().position_stream().valid());
    t.points[1] = {2, 0, 0};
    t.widths = {1, 2};
    CHECK_THROWS(t.run());
    t.widths = {2};
    t.input.refinement_level = 257;
    CHECK_THROWS(t.run());
}

TEST_CASE("Many strands and coincident runs remain finite")
{
    Input t;
    t.points.clear();
    t.counts.clear();
    for (int i = 0; i < 1000; ++i) {
        t.points.push_back({float(i) * 3, 0, 0});
        t.points.push_back({float(i) * 3 + 2, 0, 0});
        t.counts.push_back(2);
    }
    auto m = t.run();
    CHECK(m.subgeometries[0].indices.size() == 2000);
    finite_mesh(m);
    t.points.assign(10000, float3(0.f));
    t.points.push_back({2, 0, 0});
    t.counts = {10001};
    m = t.run();
    finite_mesh(m);
    CHECK(m.subgeometries[0].indices.size() == 2);
}

TEST_CASE("USD ribbon dispatch, interpolation and empty normals")
{
    auto scene = load_source(STRIP);
    REQUIRE(scene->meshes.size() == 1);
    CHECK(scene->curves.empty());
    const auto& m = scene->meshes[0];
    finite_mesh(m);
    CHECK(m.tangent_stream().valid());
    CHECK(m.source_asset_ref->object_path == "/Strip");
    for (size_t i = 0; i < m.vertex_count(); ++i) {
        const auto p = m.position_stream()[i];
        const auto uv = m.texcoord_stream()[i];
        CHECK(uv.x == doctest::Approx(.5f + p.y / 2.f));
        CHECK(uv.y == doctest::Approx(p.x / 2.f));
    }
    std::string source = STRIP;
    const auto at = source.find("normal3f[] normals");
    const auto end = source.find('\n', at);
    source.erase(at, end - at);
    scene = load_source(source);
    CHECK(scene->meshes.empty());
    CHECK(scene->curves.size() == 1);
}

TEST_CASE("USD indexed primvar precedence and malformed indices")
{
    std::string source = STRIP;
    const auto pos = source.find('}');
    source.insert(pos, R"(
 normal3f[] primvars:normals = [(0,1,0)] (interpolation = "constant")
 int[] primvars:normals:indices = [0]
 float[] primvars:widths = [4] (interpolation = "constant")
 int[] primvars:widths:indices = [0]
 texCoord2f[] primvars:st = [(0.2,0.3),(0.8,0.7)] (interpolation = "vertex")
 int[] primvars:st:indices = [1,0]
)");
    auto scene = load_source(source);
    REQUIRE(scene->meshes.size() == 1);
    const auto& m = scene->meshes[0];
    finite_mesh(m);
    for (auto p : m.position_stream()) {
        CHECK(p.y == 0.f);
        CHECK(std::abs(p.z) == doctest::Approx(2.f));
    }
    for (size_t i = 0; i < m.vertex_count(); ++i)
        CHECK(m.texcoord_stream()[i].x == doctest::Approx(.8f - .3f * m.position_stream()[i].x));
    const auto index = source.find("normals:indices = [0]");
    source.replace(index, 20, "normals:indices = [9]");
    CHECK_THROWS(load_source(source));
}

TEST_CASE("OpenUSD example imports five ribbon meshes and five round curves")
{
    ImportOptions options;
    options.curve_refinement_level = 7;
    const auto path = std::filesystem::path(FALCOR_PROJECT_DIR) / "data/assets/test_usd_ribbons/basisCurves.usda";
    auto scene = UsdImporter().load_scene(path, options);
    REQUIRE(scene->meshes.size() == 5);
    CHECK(scene->curves.size() == 5);
    for (const auto& m : scene->meshes) {
        finite_mesh(m);
        CHECK(m.tangent_stream().valid());
        CHECK(!m.subgeometries[0].indices.empty());
    }
    options.recompute_normals = true;
    scene = UsdImporter().load_scene(path, options);
    for (const auto& m : scene->meshes)
        finite_mesh(m);
}

TEST_CASE("Periodic Bezier closes its last span and constant UVs stay constant")
{
    Input t;
    t.points = {{0, 0, 0}, {1, 0, 0}, {2, 1, 0}, {2, 2, 0}, {1, 2, 0}, {0, 1, 0}};
    t.counts = {6};
    t.widths = {.1f};
    t.input.type = Type::cubic;
    t.input.wrap = Wrap::periodic;
    t.input.refinement_level = 3;
    t.uvs = {{.2f, .7f}};
    t.input.texcoords.interpolation = Interpolation::constant;
    auto mesh = t.run();
    REQUIRE(mesh.subgeometries[0].indices.size() == 16);
    finite_mesh(mesh);
    for (auto uv : mesh.texcoord_stream()) {
        CHECK(uv.x == doctest::Approx(.2f));
        CHECK(uv.y == doctest::Approx(.7f));
    }
    t.counts = {5};
    CHECK_THROWS(t.run());
}

TEST_CASE("USD left handed orientation and malformed normals")
{
    std::string source = STRIP;
    source.insert(source.find('}'), "uniform token orientation = \"leftHanded\"\n");
    auto scene = load_source(source);
    for (auto normal : scene->meshes[0].normal_stream())
        CHECK(normal.z == doctest::Approx(1.f));
    Input t;
    t.normals = {{0, 0, 1}, {0, 0, 1}};
    CHECK_THROWS(t.run());
    t.normals = {{0, 0, 1}};
    t.points[0].x = std::numeric_limits<float>::infinity();
    CHECK_THROWS(t.run());
}

TEST_CASE("Authored shading normals preserve their tangent component")
{
    Input t;
    t.normals = {{1, 0, 1}, {0, 1, 1}};
    t.sample.normals.interpolation = Interpolation::vertex;
    t.input.refinement_level = 3;
    const auto mesh = t.run();
    for (size_t i = 0; i < mesh.vertex_count(); ++i) {
        const float u = mesh.position_stream()[i].x / 2.f;
        const float3 expected = sgl::math::normalize(float3(1.f - u, u, 1.f));
        CHECK(sgl::math::length(mesh.normal_stream()[i] - expected) < 1e-6f);
    }
}

TEST_CASE("Neighbor repair handles isolated singularities runs and one-sided anchors")
{
    for (float scale : {1e-20f, 1.f, 1e20f}) {
        Input t;
        t.points = {{0, 0, 0}, {1, 0, 0}, {2, 0, 0}, {3, 0, 0}, {4, 0, 0}};
        t.counts = {5};
        t.sample.normals.interpolation = Interpolation::vertex;
        t.normals = {{0, 0, scale}, {scale, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, scale, 0}};
        auto mesh = t.run();
        REQUIRE(mesh.vertex_count() == 10);
        finite_mesh(mesh);
        for (size_t i = 0; i < mesh.vertex_count(); ++i) {
            const float u = mesh.position_stream()[i].x / 4.f;
            const float3 expected = sgl::math::normalize(float3(0, u, 1.f - u));
            CHECK(sgl::math::length(mesh.normal_stream()[i] - expected) < 1e-6f);
        }
        t.normals = {{0, 0, 0}, {0, 0, 0}, {0, scale, scale}, {0, 0, 0}, {0, 0, 0}};
        mesh = t.run();
        for (auto n : mesh.normal_stream())
            CHECK(sgl::math::length(n - sgl::math::normalize(float3(0, 1, 1))) < 1e-6f);
    }
}

TEST_CASE("Neighbor transport follows changing tangents and wraps the periodic seam")
{
    Input t;
    t.points = {{0, 0, 0}, {2, 0, 0}, {2, 1, 0}};
    t.counts = {3};
    t.widths = {.2f};
    t.sample.normals.interpolation = Interpolation::vertex;
    t.normals = {{0, 1, 0}, {0, 0, 0}, {0, 0, 0}};
    auto mesh = t.run();
    for (size_t i = 0; i < mesh.vertex_count(); ++i) {
        const float u = mesh.texcoord_stream()[i].y;
        const float3 expected = u == 0.f ? float3(0, 1, 0)
            : u == 1.f                   ? float3(-1, 0, 0)
                                         : sgl::math::normalize(float3(-1, 1, 0));
        CHECK(sgl::math::length(mesh.normal_stream()[i] - expected) < 1e-6f);
    }
    t.points = {{0, 0, 0}, {2, 0, 0}, {2, 2, 0}, {0, 2, 0}};
    t.counts = {4};
    t.input.wrap = Wrap::periodic;
    t.normals = {{0, 0, 0}, {0, 0, 1}, {0, 0, 0}, {0, 0, 1}};
    mesh = t.run();
    REQUIRE(mesh.vertex_count() == 10);
    for (auto n : mesh.normal_stream())
        CHECK(sgl::math::length(n - float3(0, 0, 1)) < 1e-6f);
    int seam_vertices = 0;
    for (auto uv : mesh.texcoord_stream())
        seam_vertices += uv.y == 0.f || uv.y == 1.f;
    CHECK(seam_vertices == 4);
}

TEST_CASE("Repair never borrows across strands or invents an ambiguous half-turn")
{
    Input t;
    t.sample.normals.interpolation = Interpolation::vertex;
    t.normals = {{0, 0, 1}, {0, 0, -1}};
    for (int refinement : {0, 1, 2}) {
        t.input.refinement_level = refinement;
        CHECK_THROWS_WITH(t.run(), doctest::Contains("test ribbon"));
    }
    t.points = {{0, 0, 0}, {1, 0, 0}, {0, 0, 0}};
    t.counts = {3};
    t.normals = {{0, 0, 1}, {0, 0, 0}, {0, 0, 0}};
    t.input.refinement_level = 0;
    CHECK_THROWS_WITH(t.run(), doctest::Contains("antiparallel"));
    t.points = {{0, 0, 0}, {2, 0, 0}, {0, 3, 0}, {2, 3, 0}};
    t.counts = {2, 2};
    t.normals = {{0, 0, 1}, {0, 0, 1}, {0, 0, 0}, {0, 0, 0}};
    CHECK_THROWS_WITH(t.run(), doctest::Contains("strand 1"));
}

TEST_CASE("Unequal length corner and twisted joint share cross-sections")
{
    Input t;
    t.points = {{0, 0, 0}, {4, 0, 0}, {4, 1, 0}};
    t.counts = {3};
    t.widths = {.2f};
    t.sample.normals.interpolation = Interpolation::vertex;
    t.normals = {{0, 0, 1}, {0, 0, 1}, {0, 1, 1}};
    for (int refinement : {0, 3}) {
        t.input.refinement_level = refinement;
        const auto mesh = t.run();
        const size_t intervals = 2 * (refinement + 1);
        REQUIRE(mesh.vertex_count() == 2 * (intervals + 1));
        REQUIRE(mesh.subgeometries[0].indices.size() == 2 * intervals);
        std::vector<uint32_t> joint;
        for (uint32_t i = 0; i < mesh.vertex_count(); ++i) {
            if (mesh.texcoord_stream()[i].y != .5f)
                continue;
            joint.push_back(i);
            const float3 offset = mesh.position_stream()[i] - float3(4, 0, 0);
            CHECK(sgl::math::length(offset) == doctest::Approx(.1f));
            CHECK(std::abs(offset.x) == doctest::Approx(.1f / std::sqrt(2.f)));
            CHECK(offset.x == doctest::Approx(-offset.y));
        }
        REQUIRE(joint.size() == 2);
        for (uint32_t index : joint) {
            int uses = 0;
            for (auto tri : mesh.subgeometries[0].indices)
                uses += tri.x == index || tri.y == index || tri.z == index;
            CHECK(uses == 3);
        }
    }
}

TEST_CASE("USD repair diagnostics and explicit normal recomputation")
{
    std::string source = STRIP;
    const auto at = source.find("[(0,0,1)] (interpolation = \"constant\")");
    REQUIRE(at != std::string::npos);
    source.replace(
        at,
        std::string("[(0,0,1)] (interpolation = \"constant\")").size(),
        "[(1,0,1),(0,0,0)] (interpolation = \"vertex\")"
    );
    ImportOptions options;
    options.curve_refinement_level = 0;
    auto scene = load_source(source, options);
    const auto mesh = scene->meshes[0];
    for (size_t i = 0; i < mesh.vertex_count(); ++i) {
        const float3 expected
            = mesh.position_stream()[i].x == 0.f ? sgl::math::normalize(float3(1, 0, 1)) : float3(0, 0, 1);
        CHECK(sgl::math::length(mesh.normal_stream()[i] - expected) < 1e-6f);
    }
    options.recompute_normals = true;
    scene = load_source(source, options);
    for (auto n : scene->meshes[0].normal_stream())
        CHECK(sgl::math::length(n - float3(0, 0, 1)) < 1e-6f);
}

TEST_CASE("Motion shares topology across collapsed widths and validates sample layouts")
{
    Input t;
    t.widths = {0.f};
    CHECK(!t.run().position_stream().valid());
    const std::vector<float> open_widths{2.f};
    auto mesh = tessellate_ribbons(t.input, 2);
    t.input.sample.widths.values = open_widths;
    tessellate_ribbons(t.input, mesh.position_stream(1), mesh.normal_stream(1));
    REQUIRE(mesh.vertex_count() == 4);
    REQUIRE(mesh.subgeometries[0].indices.size() == 2);
    for (auto p : mesh.position_stream(0))
        CHECK(p.y == 0.f);
    for (auto p : mesh.position_stream(1))
        CHECK(std::abs(p.y) == 1.f);
    t.input.sample.normals.interpolation = Interpolation::vertex;
    CHECK_THROWS_WITH(
        tessellate_ribbons(t.input, mesh.position_stream(1), mesh.normal_stream(1)),
        doctest::Contains("count")
    );
    t.input.sample = t.sample;
    t.input.sample.points = t.input.sample.points.first(1);
    CHECK_THROWS_WITH(
        tessellate_ribbons(t.input, mesh.position_stream(1), mesh.normal_stream(1)),
        doctest::Contains("point count")
    );
    t.input.sample = t.sample;
    const std::vector<float3> collapsed{{0, 0, 0}, {0, 0, 0}};
    t.input.sample.points = collapsed;
    CHECK_THROWS_WITH(
        tessellate_ribbons(t.input, mesh.position_stream(1), mesh.normal_stream(1)),
        doctest::Contains("no usable centerline tangent")
    );
    CHECK_THROWS_WITH(tessellate_ribbons(t.input, 0), doctest::Contains("at least one sample"));
}

TEST_SUITE_END();
