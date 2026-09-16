// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "testing.h"
#include "falcor2/importers/importer_types.h"
#include "falcor2/importers/mesh_tessellator/mesh_tessellator.h"
#include "falcor2/importers/usd_importer/usd_importer.h"

#include <sgl/core/platform.h>

#include <algorithm>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace falcor;

TEST_SUITE_BEGIN("mesh_tessellator");

static ref<ImporterScene> load_test_usda(std::string_view file_name, std::string_view source)
{
    const std::filesystem::path scene_path = testing::get_case_temp_directory() / std::filesystem::path(file_name);
    {
        std::ofstream file(scene_path);
        file << source;
    }

    return make_ref<UsdImporter>()->load_scene(scene_path);
}

static std::unordered_set<uint32_t> referenced_vertices(const ImporterMesh::Subgeometry& subgeometry)
{
    std::unordered_set<uint32_t> result;
    for (const uint3& triangle : subgeometry.indices) {
        result.insert(triangle.x);
        result.insert(triangle.y);
        result.insert(triangle.z);
    }
    return result;
}

class TestTessellatorInput {
public:
    mesh_tessellator::TessellatorInputMesh input;

    TestTessellatorInput() = default;
    TestTessellatorInput(const TestTessellatorInput&) = delete;
    TestTessellatorInput& operator=(const TestTessellatorInput&) = delete;
    TestTessellatorInput(TestTessellatorInput&&) noexcept = default;
    TestTessellatorInput& operator=(TestTessellatorInput&&) noexcept = default;

    void set_topology(
        std::initializer_list<int> face_vertex_counts,
        std::initializer_list<int> face_vertex_indices,
        std::initializer_list<int> hole_indices = {}
    )
    {
        m_face_vertex_counts = face_vertex_counts;
        m_face_vertex_indices = face_vertex_indices;
        m_hole_indices = hole_indices;
        input.face_vertex_counts = m_face_vertex_counts;
        input.face_vertex_indices = m_face_vertex_indices;
        input.hole_indices = m_hole_indices;
    }

    void set_positions(std::initializer_list<float> values)
    {
        m_positions = values;
        input.positions.values = m_positions;
    }

    void set_positions(std::initializer_list<float3> values)
    {
        m_positions.clear();
        m_positions.reserve(values.size() * 3);
        for (const float3& value : values) {
            m_positions.push_back(value.x);
            m_positions.push_back(value.y);
            m_positions.push_back(value.z);
        }
        input.positions.values = m_positions;
    }

    void add_stream(
        ImporterMeshDefaultAttribute attribute,
        mesh_tessellator::MeshInterpolation interpolation,
        std::initializer_list<float> values,
        std::initializer_list<int> indices = {}
    )
    {
        AttributeStorage& storage = m_attributes.emplace_back();
        storage.values = values;
        storage.indices = indices;
        input.streams.push_back({
            .attribute = attribute,
            .interpolation = interpolation,
            .values = storage.values,
            .indices = storage.indices,
        });
    }

    void add_subgeometry(std::string name, std::initializer_list<int> face_indices)
    {
        std::vector<int>& storage = m_subgeometry_faces.emplace_back(face_indices);
        input.subgeometries.push_back({.name = std::move(name), .face_indices = storage});
    }

private:
    struct AttributeStorage {
        std::vector<float> values;
        std::vector<int> indices;
    };

    std::vector<int> m_face_vertex_counts;
    std::vector<int> m_face_vertex_indices;
    std::vector<int> m_hole_indices;
    std::vector<float> m_positions;
    std::deque<AttributeStorage> m_attributes;
    std::deque<std::vector<int>> m_subgeometry_faces;
};

TEST_CASE("subdivision patches share topological boundary vertices")
{
    using namespace mesh_tessellator;

    TestTessellatorInput storage;
    TessellatorInputMesh& input = storage.input;
    input.name = "Tetrahedron";
    input.subdivision_scheme = SubdivisionScheme::loop;
    input.refinement_level = 2;
    storage.set_topology({3, 3, 3, 3}, {0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3});
    storage.set_positions({{1.f, 1.f, 1.f}, {-1.f, -1.f, 1.f}, {-1.f, 1.f, -1.f}, {1.f, -1.f, -1.f}});
    storage.add_subgeometry(input.name, {0, 1, 2, 3});

    const ImporterMesh mesh = tessellate(input);
    auto positions = mesh.position_stream();
    auto normals = mesh.normal_stream();
    REQUIRE(positions.valid());
    REQUIRE(normals.valid());
    CHECK_EQ(mesh.vertex_count(), 34);

    std::unordered_map<uint64_t, int> edge_use_counts;
    size_t triangle_count = 0;
    for (const ImporterMesh::Subgeometry& subgeometry : mesh.subgeometries) {
        triangle_count += subgeometry.indices.size();
        for (const uint3& triangle : subgeometry.indices) {
            CHECK_LT(triangle.x, mesh.vertex_count());
            CHECK_LT(triangle.y, mesh.vertex_count());
            CHECK_LT(triangle.z, mesh.vertex_count());
            for (int edge = 0; edge < 3; ++edge) {
                const uint32_t v0 = triangle[edge];
                const uint32_t v1 = triangle[(edge + 1) % 3];
                const uint64_t key = (uint64_t(std::min(v0, v1)) << 32) | std::max(v0, v1);
                ++edge_use_counts[key];
            }
        }
    }
    CHECK_EQ(triangle_count, 64);
    for (const auto& [edge, use_count] : edge_use_counts) {
        CAPTURE(edge);
        CHECK_EQ(use_count, 2);
    }

    for (size_t i = 0; i < mesh.vertex_count(); ++i)
        CHECK(sgl::math::length(normals[i]) == doctest::Approx(1.f));
}

TEST_CASE("subdivision patches share boundaries across Bfr parameterizations")
{
    using namespace mesh_tessellator;

    TestTessellatorInput storage;
    TessellatorInputMesh& input = storage.input;
    input.name = "TriangleAndQuad";
    input.subdivision_scheme = SubdivisionScheme::catmull_clark;
    input.refinement_level = 2;
    storage.set_topology({3, 4}, {0, 1, 2, 1, 0, 3, 4});
    storage.set_positions({
        {0.f, 0.f, 0.f},
        {1.f, 0.f, 0.f},
        {0.5f, 0.f, 1.f},
        {0.f, 0.f, -1.f},
        {1.f, 0.f, -1.f},
    });
    storage.add_subgeometry("Triangle", {0});
    storage.add_subgeometry("Quad", {1});

    const ImporterMesh mesh = tessellate(input);
    REQUIRE_EQ(mesh.subgeometries.size(), 2);

    const auto triangle_vertices = referenced_vertices(mesh.subgeometries[0]);
    const auto quad_vertices = referenced_vertices(mesh.subgeometries[1]);
    size_t shared_vertex_count = 0;
    for (uint32_t vertex : triangle_vertices)
        shared_vertex_count += quad_vertices.contains(vertex);

    CHECK_EQ(shared_vertex_count, 5);
}

TEST_CASE("subdivision patch stitching preserves face-varying UV seams")
{
    using namespace mesh_tessellator;

    TestTessellatorInput storage;
    TessellatorInputMesh& input = storage.input;
    input.name = "Plane";
    input.subdivision_scheme = SubdivisionScheme::loop;
    input.refinement_level = 1;
    storage.set_topology({3, 3}, {0, 1, 2, 0, 2, 3});
    storage.set_positions({
        {-1.f, 0.f, -1.f},
        {1.f, 0.f, -1.f},
        {1.f, 0.f, 1.f},
        {-1.f, 0.f, 1.f},
    });
    storage.add_stream(
        {ImporterSemantic::tex_coord, 0, 2},
        MeshInterpolation::face_varying,
        {0.f, 0.f, 1.f, 0.f, 1.f, 1.f, 0.f, 0.f, 1.f, 1.f, 0.f, 1.f},
        {0, 1, 2, 3, 4, 5}
    );
    storage.add_subgeometry("FirstFace", {0});
    storage.add_subgeometry("SecondFace", {1});

    const ImporterMesh mesh = tessellate(input);
    REQUIRE_EQ(mesh.subgeometries.size(), 2);
    auto positions = mesh.position_stream();
    auto normals = mesh.normal_stream();
    auto uvs = mesh.texcoord_stream();
    REQUIRE(positions.valid());
    REQUIRE(normals.valid());
    REQUIRE(uvs.valid());

    size_t uv_seam_vertex_pairs = 0;
    for (size_t i = 0; i < mesh.vertex_count(); ++i) {
        for (size_t j = i + 1; j < mesh.vertex_count(); ++j) {
            if (positions[i] != positions[j] || uvs[i] == uvs[j])
                continue;

            ++uv_seam_vertex_pairs;
            CHECK_EQ(normals[i], normals[j]);
        }
    }
    CHECK_EQ(uv_seam_vertex_pairs, 1);

    const auto first_vertices = referenced_vertices(mesh.subgeometries[0]);
    const auto second_vertices = referenced_vertices(mesh.subgeometries[1]);
    for (uint32_t vertex : first_vertices)
        CHECK_FALSE(second_vertices.contains(vertex));
}

TEST_CASE("stream tessellator handles multiple runtime-sized attribute streams")
{
    using namespace mesh_tessellator;

    TestTessellatorInput storage;
    TessellatorInputMesh& input = storage.input;
    input.name = "StreamPlane";
    input.subdivision_scheme = SubdivisionScheme::catmull_clark;
    input.orientation = MeshOrientation::right_handed;
    input.face_varying_linear_interpolation = FaceVaryingLinearInterpolation::corners_plus_1;
    input.vertex_boundary_interpolation = VertexBoundaryInterpolation::edge_and_corner;
    input.refinement_level = 2;
    storage.set_topology({4, 4}, {0, 1, 4, 3, 1, 2, 5, 4});
    storage.set_positions({
        -1.f,
        0.f,
        -1.f,
        0.f,
        0.f,
        -1.f,
        1.f,
        0.f,
        -1.f,
        -1.f,
        0.f,
        1.f,
        0.f,
        0.f,
        1.f,
        1.f,
        0.f,
        1.f,
    });

    storage.add_stream(
        {ImporterSemantic::tex_coord, 0, 2},
        MeshInterpolation::face_varying,
        {0.f, 0.f, 0.5f, 0.f, 1.f, 0.f, 0.f, 1.f, 0.5f, 1.f, 1.f, 1.f},
        {0, 1, 4, 3, 1, 2, 5, 4}
    );
    storage.add_stream(
        {ImporterSemantic::tex_coord, 1, 2},
        MeshInterpolation::face_varying,
        {
            0.f,
            0.f,
            1.f,
            0.f,
            1.f,
            1.f,
            0.f,
            1.f,
            0.f,
            0.f,
            1.f,
            0.f,
            1.f,
            1.f,
            0.f,
            1.f,
        },
        {0, 1, 2, 3, 4, 5, 6, 7}
    );
    storage.add_stream({ImporterSemantic::color, 0, 4}, MeshInterpolation::constant, {0.25f, 0.5f, 0.75f, 1.f});
    storage.add_stream(
        {ImporterSemantic::color, 1, 1},
        MeshInterpolation::varying,
        {10.f, 20.f, 30.f},
        {2, 0, 1, 2, 0, 1}
    );
    storage
        .add_stream({ImporterSemantic::color, 2, 1}, MeshInterpolation::varying, {30.f, 10.f, 20.f, 30.f, 10.f, 20.f});
    storage.add_subgeometry(input.name, {0, 1});

    ImporterMesh mesh = tessellate(input);
    REQUIRE_EQ(mesh.vertex_count(), 50);
    REQUIRE_EQ(mesh.subgeometries.size(), 1);
    REQUIRE_EQ(mesh.subgeometries[0].indices.size(), 64);

    auto positions = mesh.position_stream();
    auto uv0 = mesh.find_stream<float2>(ImporterSemantic::tex_coord, 0);
    auto uv1 = mesh.find_stream<float2>(ImporterSemantic::tex_coord, 1);
    const ImporterMeshAttribute* color_attribute = mesh.find_attribute(ImporterSemantic::color, 0);
    REQUIRE(positions.valid());
    REQUIRE(uv0.valid());
    REQUIRE(uv1.valid());
    REQUIRE(color_attribute);
    REQUIRE_EQ(color_attribute->num_components, 4);
    auto colors = mesh.get_stream<float4>(*color_attribute);
    auto indexed_varying = mesh.find_stream<float>(ImporterSemantic::color, 1);
    auto dense_varying = mesh.find_stream<float>(ImporterSemantic::color, 2);
    REQUIRE(indexed_varying.valid());
    REQUIRE(dense_varying.valid());

    size_t split_uv1_pairs = 0;
    for (size_t first = 0; first < mesh.vertex_count(); ++first) {
        CHECK_EQ(colors[first], float4(0.25f, 0.5f, 0.75f, 1.f));
        CHECK_EQ(indexed_varying[first], dense_varying[first]);
        for (size_t second = first + 1; second < mesh.vertex_count(); ++second) {
            if (positions[first] != positions[second] || uv1[first] == uv1[second])
                continue;

            ++split_uv1_pairs;
            CHECK_EQ(uv0[first], uv0[second]);
        }
    }
    CHECK_EQ(split_uv1_pairs, 5);
}

TEST_CASE("USD tessellator adapter follows PxOsd token fallbacks")
{
    const auto make_source = [](std::string_view subdivision_scheme,
                                std::string_view orientation,
                                std::string_view face_varying_interpolation,
                                std::string_view boundary_interpolation)
    {
        std::string source = R"usd(#usda 1.0

def Mesh "TokenFallbacks"
{
)usd";
        const auto add_token = [&](std::string_view name, std::string_view value)
        {
            if (!value.empty()) {
                source += "    uniform token ";
                source += name;
                source += " = \"";
                source += value;
                source += "\"\n";
            }
        };
        add_token("subdivisionScheme", subdivision_scheme);
        add_token("orientation", orientation);
        add_token("faceVaryingLinearInterpolation", face_varying_interpolation);
        add_token("interpolateBoundary", boundary_interpolation);
        source += R"usd(    int refinementLevel = 2
    int[] faceVertexCounts = [4, 4]
    int[] faceVertexIndices = [0, 1, 4, 3, 1, 2, 5, 4]
    point3f[] points = [(-1, 0, -1), (0, 0, -1), (1, 0, -1), (-1, 0, 1), (0, 0, 1), (1, 0, 1)]
    texCoord2f[] primvars:st = [(0, 0), (0.5, 0), (0.5, 1), (0, 1), (0.5, 0), (1, 0), (1, 1), (0.5, 1)] (
        interpolation = "faceVarying"
    )
}
)usd";
        return source;
    };
    const auto check_equivalent = [](const ImporterMesh& candidate, const ImporterMesh& reference)
    {
        REQUIRE_EQ(candidate.vertex_count(), reference.vertex_count());
        REQUIRE_EQ(candidate.subgeometries.size(), reference.subgeometries.size());
        for (size_t i = 0; i < candidate.subgeometries.size(); ++i)
            CHECK(candidate.subgeometries[i].indices == reference.subgeometries[i].indices);
        REQUIRE_EQ(candidate.buffers().size(), reference.buffers().size());
        for (size_t i = 0; i < candidate.buffers().size(); ++i) {
            CHECK_EQ(candidate.buffers()[i].stride, reference.buffers()[i].stride);
            CHECK(candidate.buffers()[i].data == reference.buffers()[i].data);
        }
    };

    SUBCASE("missing tokens use USD mesh defaults")
    {
        const ref<ImporterScene> candidate
            = load_test_usda("missing_tessellator_tokens.usda", make_source({}, {}, {}, {}));
        const ref<ImporterScene> reference = load_test_usda(
            "explicit_usd_mesh_defaults.usda",
            make_source("catmullClark", "rightHanded", "cornersPlus1", "edgeAndCorner")
        );
        REQUIRE(candidate);
        REQUIRE(reference);
        REQUIRE_EQ(candidate->meshes.size(), 1);
        REQUIRE_EQ(reference->meshes.size(), 1);
        check_equivalent(candidate->meshes.front(), reference->meshes.front());
    }
}

TEST_CASE("USD tessellator adapter treats empty face-varying UVs as unauthored")
{
    const ref<ImporterScene> scene = load_test_usda(
        "subdivision_empty_facevarying_uv.usda",
        R"usd(#usda 1.0

def Mesh "Triangle"
{
    uniform token subdivisionScheme = "loop"
    int refinementLevel = 1
    int[] faceVertexCounts = [3]
    int[] faceVertexIndices = [0, 1, 2]
    point3f[] points = [(-1, 0, -1), (1, 0, -1), (0, 0, 1)]
    texCoord2f[] primvars:st = [] (
        interpolation = "faceVarying"
    )
}
)usd"
    );
    REQUIRE(scene);
    REQUIRE_EQ(scene->meshes.size(), 1);

    const ImporterMesh& mesh = scene->meshes.front();
    const auto uvs = mesh.texcoord_stream();
    REQUIRE(uvs.valid());
    for (size_t i = 0; i < mesh.vertex_count(); ++i)
        CHECK_EQ(uvs[i], float2(0.f));
}

TEST_CASE("USD tessellator adapter imports only face material-binding subsets")
{
    const ref<ImporterScene> scene = load_test_usda(
        "mixed_geom_subset_families.usda",
        R"usd(#usda 1.0

def Mesh "MixedSubsets"
{
    uniform token subdivisionScheme = "none"
    int[] faceVertexCounts = [3, 3]
    int[] faceVertexIndices = [0, 1, 2, 0, 2, 3]
    point3f[] points = [(-1, 0, -1), (1, 0, -1), (1, 0, 1), (-1, 0, 1)]

    def GeomSubset "materialFaces"
    {
        uniform token elementType = "face"
        uniform token familyName = "materialBind"
        int[] indices = [0]
    }

    def GeomSubset "selectionFaces"
    {
        uniform token elementType = "face"
        uniform token familyName = "selection"
        int[] indices = [1]
    }

    def GeomSubset "materialPoints"
    {
        uniform token elementType = "point"
        uniform token familyName = "materialBind"
        int[] indices = [0]
    }
}
)usd"
    );
    REQUIRE(scene);
    REQUIRE_EQ(scene->meshes.size(), 1);

    const ImporterMesh& mesh = scene->meshes.front();
    REQUIRE_EQ(mesh.subgeometries.size(), 2);
    CHECK_EQ(mesh.subgeometries[0].name, std::string("/MixedSubsets/materialFaces"));
    CHECK_EQ(mesh.subgeometries[1].name, std::string("/MixedSubsets"));
    CHECK_EQ(mesh.subgeometries[0].indices.size(), 1);
    CHECK_EQ(mesh.subgeometries[1].indices.size(), 1);
}

TEST_CASE("stream tessellator preserves ordinary streams and subgeometry topology")
{
    using namespace mesh_tessellator;

    TestTessellatorInput storage;
    TessellatorInputMesh& input = storage.input;
    input.name = "ordinary triangulation";
    input.subdivision_scheme = SubdivisionScheme::none;
    input.orientation = MeshOrientation::left_handed;
    storage.set_topology({3, 4, 5}, {0, 1, 2, 0, 2, 3, 4, 0, 4, 5, 6, 1}, {1});
    storage.set_positions({
        0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 1.f, 1.f, 0.f, 0.f, 1.f, 0.f, -1.f, 1.f, 0.f, -1.f, 0.f, 0.f, 0.f, -1.f, 0.f,
    });
    storage.add_stream(
        {ImporterSemantic::normal, 0, 3},
        MeshInterpolation::uniform,
        {0.f, 0.f, 1.f, 0.f, 0.f, 1.f, 0.f, 0.f, 1.f}
    );
    storage.add_stream(
        {ImporterSemantic::tex_coord, 0, 2},
        MeshInterpolation::face_varying,
        {
            0.f,  0.f,  1.f, 0.f, 1.f, 1.f, 0.f,  0.f, 1.f, 1.f,  0.f, 1.f,
            0.5f, 0.5f, 0.f, 0.f, 0.f, 1.f, 0.5f, 1.f, 1.f, 0.5f, 1.f, 0.f,
        }
    );
    storage.add_subgeometry("first", {0, 1});
    storage.add_subgeometry("second", {2});

    const ImporterMesh mesh = tessellate(input);
    REQUIRE_EQ(mesh.vertex_count(), 8);
    REQUIRE_EQ(mesh.subgeometries.size(), 2);
    CHECK(mesh.subgeometries[0].indices == std::vector<uint3>{{0, 2, 1}});
    CHECK(mesh.subgeometries[1].indices == std::vector<uint3>{{3, 5, 4}, {3, 6, 5}, {3, 7, 6}});

    const auto positions = mesh.position_stream();
    const auto normals = mesh.normal_stream();
    const auto texcoords = mesh.texcoord_stream();
    REQUIRE(positions.valid());
    REQUIRE(normals.valid());
    REQUIRE(texcoords.valid());
    const std::vector<float3> expected_positions{
        {0.f, 0.f, 0.f},
        {1.f, 0.f, 0.f},
        {1.f, 1.f, 0.f},
        {0.f, 0.f, 0.f},
        {-1.f, 1.f, 0.f},
        {-1.f, 0.f, 0.f},
        {0.f, -1.f, 0.f},
        {1.f, 0.f, 0.f},
    };
    const std::vector<float2> expected_texcoords{
        {0.f, 0.f},
        {1.f, 0.f},
        {1.f, 1.f},
        {0.f, 0.f},
        {0.f, 1.f},
        {0.5f, 1.f},
        {1.f, 0.5f},
        {1.f, 0.f},
    };
    for (size_t vertex_index = 0; vertex_index < mesh.vertex_count(); ++vertex_index) {
        CHECK_EQ(positions[vertex_index], expected_positions[vertex_index]);
        CHECK_EQ(normals[vertex_index], float3(0.f, 0.f, 1.f));
        CHECK_EQ(texcoords[vertex_index], expected_texcoords[vertex_index]);
    }
}

TEST_CASE("stream tessellator validates face metadata while constructing the output mapping")
{
    using namespace mesh_tessellator;

    const auto make_input = []
    {
        TestTessellatorInput storage;
        storage.input.name = "invalid face metadata";
        storage.input.subdivision_scheme = SubdivisionScheme::none;
        storage.set_topology({3}, {0, 1, 2});
        storage.set_positions({0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f, 0.f});
        return storage;
    };

    SUBCASE("out-of-range subgeometry face")
    {
        TestTessellatorInput storage = make_input();
        storage.add_subgeometry("invalid", {1});
        CHECK_THROWS_WITH(tessellate(storage.input), doctest::Contains("references face 1 but the mesh has 1 faces"));
    }

    SUBCASE("duplicate subgeometry face")
    {
        TestTessellatorInput storage = make_input();
        storage.add_subgeometry("first", {0});
        storage.add_subgeometry("second", {0});
        CHECK_THROWS_WITH(tessellate(storage.input), doctest::Contains("face 0 is assigned to multiple subgeometries"));
    }

    SUBCASE("out-of-range hole face")
    {
        TestTessellatorInput storage = make_input();
        storage.set_topology({3}, {0, 1, 2}, {-1});
        storage.add_subgeometry("valid", {0});
        CHECK_THROWS_WITH(
            tessellate(storage.input),
            doctest::Contains("hole references face -1 but the mesh has 1 faces")
        );
    }
}

TEST_CASE("stream tessellator preserves orientation in winding and generated normals")
{
    using namespace mesh_tessellator;

    const auto make_input = [](SubdivisionScheme subdivision_scheme, bool left_handed)
    {
        TestTessellatorInput storage;
        TessellatorInputMesh& input = storage.input;
        input.name = "oriented quad";
        input.subdivision_scheme = subdivision_scheme;
        input.orientation = left_handed ? MeshOrientation::left_handed : MeshOrientation::right_handed;
        input.refinement_level = 1;
        if (left_handed)
            storage.set_topology({4}, {0, 3, 2, 1});
        else
            storage.set_topology({4}, {0, 1, 2, 3});
        storage.set_positions({
            0.f,
            0.f,
            0.f,
            1.f,
            0.f,
            0.f,
            1.f,
            1.f,
            0.f,
            0.f,
            1.f,
            0.f,
        });
        storage.add_subgeometry(input.name, {0});
        return storage;
    };

    const auto check_orientation = [](const ImporterMesh& mesh)
    {
        const auto positions = mesh.position_stream();
        const auto normals = mesh.normal_stream();
        REQUIRE(positions.valid());
        REQUIRE(normals.valid());
        REQUIRE_EQ(mesh.subgeometries.size(), 1);
        REQUIRE_FALSE(mesh.subgeometries[0].indices.empty());

        const float3 expected_normal(0.f, 0.f, 1.f);
        float minimum_winding_alignment = 1.f;
        float minimum_normal_alignment = 1.f;
        float minimum_normal_winding_alignment = 1.f;
        for (const uint3& triangle : mesh.subgeometries[0].indices) {
            REQUIRE_LT(triangle.x, mesh.vertex_count());
            REQUIRE_LT(triangle.y, mesh.vertex_count());
            REQUIRE_LT(triangle.z, mesh.vertex_count());

            const float3 geometric_normal = sgl::math::normalize(
                sgl::math::cross(
                    positions[triangle.y] - positions[triangle.x],
                    positions[triangle.z] - positions[triangle.x]
                )
            );

            minimum_winding_alignment
                = std::min(minimum_winding_alignment, sgl::math::dot(geometric_normal, expected_normal));
            for (uint32_t vertex : {triangle.x, triangle.y, triangle.z}) {
                minimum_normal_alignment
                    = std::min(minimum_normal_alignment, sgl::math::dot(normals[vertex], expected_normal));
                minimum_normal_winding_alignment
                    = std::min(minimum_normal_winding_alignment, sgl::math::dot(normals[vertex], geometric_normal));
            }
        }

        // Check winding and shading independently so two compensating orientation errors cannot make the test pass.
        CHECK_GT(minimum_winding_alignment, 0.999f);
        CHECK_GT(minimum_normal_alignment, 0.999f);
        CHECK_GT(minimum_normal_winding_alignment, 0.999f);
    };

    SUBCASE("ordinary right-handed")
    {
        const TestTessellatorInput input = make_input(SubdivisionScheme::none, false);
        check_orientation(tessellate(input.input));
    }
    SUBCASE("ordinary left-handed")
    {
        const TestTessellatorInput input = make_input(SubdivisionScheme::none, true);
        check_orientation(tessellate(input.input));
    }
    SUBCASE("Catmull-Clark right-handed")
    {
        const TestTessellatorInput input = make_input(SubdivisionScheme::catmull_clark, false);
        check_orientation(tessellate(input.input));
    }
    SUBCASE("Catmull-Clark left-handed")
    {
        const TestTessellatorInput input = make_input(SubdivisionScheme::catmull_clark, true);
        check_orientation(tessellate(input.input));
    }
}

TEST_CASE("ordinary triangulation skips holes independently of face eligibility")
{
    using namespace mesh_tessellator;

    TestTessellatorInput storage;
    TessellatorInputMesh& input = storage.input;
    input.name = "ordinary holes";
    input.subdivision_scheme = SubdivisionScheme::none;
    input.orientation = MeshOrientation::right_handed;
    storage.set_topology({2, 3, 3, 4}, {0, 1, 0, 1, 2, 0, 2, 3, 0, 1, 2, 3}, {1, 2});
    storage.set_positions({0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 1.f, 1.f, 0.f, 0.f, 1.f, 0.f});
    storage.add_subgeometry(input.name, {0, 1, 3});

    const ImporterMesh mesh = tessellate(input);
    REQUIRE_EQ(mesh.vertex_count(), 4);
    REQUIRE_EQ(mesh.subgeometries.size(), 1);
    CHECK(mesh.subgeometries[0].indices == std::vector<uint3>{{0, 1, 2}, {0, 2, 3}});
}

TEST_CASE("subdivision patch stitching follows normal continuity")
{
    using namespace mesh_tessellator;

    const auto tessellate_two_quads = [](SubdivisionScheme subdivision_scheme)
    {
        TestTessellatorInput storage;
        TessellatorInputMesh& input = storage.input;
        input.name = "Fold";
        input.subdivision_scheme = subdivision_scheme;
        input.refinement_level = 1;
        storage.set_topology({4, 4}, {0, 1, 2, 3, 1, 4, 5, 2});
        storage.set_positions({
            {-1.f, 0.f, -1.f},
            {0.f, 0.f, -1.f},
            {0.f, 0.f, 1.f},
            {-1.f, 0.f, 1.f},
            {1.f, 1.f, -1.f},
            {1.f, 1.f, 1.f},
        });
        storage.add_stream(
            {ImporterSemantic::tex_coord, 0, 2},
            MeshInterpolation::vertex,
            {0.f, 0.f, 0.5f, 0.f, 0.5f, 1.f, 0.f, 1.f, 1.f, 0.f, 1.f, 1.f}
        );
        storage.add_subgeometry("FirstFace", {0});
        storage.add_subgeometry("SecondFace", {1});
        return tessellate(input);
    };

    SUBCASE("Catmull-Clark shares smooth shading vertices")
    {
        const ImporterMesh mesh = tessellate_two_quads(SubdivisionScheme::catmull_clark);
        REQUIRE_EQ(mesh.subgeometries.size(), 2);

        const auto first_vertices = referenced_vertices(mesh.subgeometries[0]);
        const auto second_vertices = referenced_vertices(mesh.subgeometries[1]);
        size_t shared_vertex_count = 0;
        for (uint32_t vertex : first_vertices)
            shared_vertex_count += second_vertices.contains(vertex);
        CHECK_EQ(shared_vertex_count, 3);
    }

    SUBCASE("bilinear preserves discontinuous face-side normals")
    {
        const ImporterMesh mesh = tessellate_two_quads(SubdivisionScheme::bilinear);
        REQUIRE_EQ(mesh.subgeometries.size(), 2);
        auto positions = mesh.position_stream();
        auto normals = mesh.normal_stream();
        auto uvs = mesh.texcoord_stream();
        REQUIRE(positions.valid());
        REQUIRE(normals.valid());
        REQUIRE(uvs.valid());

        const auto first_vertices = referenced_vertices(mesh.subgeometries[0]);
        const auto second_vertices = referenced_vertices(mesh.subgeometries[1]);
        size_t shared_vertex_count = 0;
        size_t split_normal_count = 0;
        for (uint32_t first : first_vertices) {
            shared_vertex_count += second_vertices.contains(first);
            for (uint32_t second : second_vertices) {
                if (positions[first] == positions[second] && sgl::math::dot(normals[first], normals[second]) < 0.99f) {
                    ++split_normal_count;
                    CHECK_EQ(uvs[first], uvs[second]);
                }
            }
        }
        CHECK_EQ(shared_vertex_count, 0);
        CHECK_EQ(split_normal_count, 3);
    }
}

TEST_CASE("subdivision patch stitching follows UV interpolation")
{
    using namespace mesh_tessellator;

    SUBCASE("constant")
    {
        TestTessellatorInput storage;
        TessellatorInputMesh& input = storage.input;
        input.name = "Plane";
        input.subdivision_scheme = SubdivisionScheme::loop;
        input.refinement_level = 1;
        storage.set_topology({3, 3}, {0, 1, 2, 0, 2, 3});
        storage.set_positions({
            {-1.f, 0.f, -1.f},
            {1.f, 0.f, -1.f},
            {1.f, 0.f, 1.f},
            {-1.f, 0.f, 1.f},
        });
        storage.add_stream({ImporterSemantic::tex_coord, 0, 2}, MeshInterpolation::constant, {0.25f, 0.75f});
        storage.add_subgeometry(input.name, {0, 1});

        const ImporterMesh mesh = tessellate(input);
        auto uvs = mesh.texcoord_stream();
        REQUIRE(uvs.valid());
        for (size_t i = 0; i < mesh.vertex_count(); ++i)
            CHECK_EQ(uvs[i], float2(0.25f, 0.75f));
    }

    SUBCASE("uniform")
    {
        TestTessellatorInput storage;
        TessellatorInputMesh& input = storage.input;
        input.name = "Plane";
        input.subdivision_scheme = SubdivisionScheme::loop;
        input.refinement_level = 1;
        storage.set_topology({3, 3}, {0, 1, 2, 0, 2, 3});
        storage.set_positions({
            {-1.f, 0.f, -1.f},
            {1.f, 0.f, -1.f},
            {1.f, 0.f, 1.f},
            {-1.f, 0.f, 1.f},
        });
        storage.add_stream(
            {ImporterSemantic::tex_coord, 0, 2},
            MeshInterpolation::uniform,
            {0.8f, 0.9f, 0.1f, 0.2f},
            {1, 0}
        );
        storage.add_subgeometry("FirstFace", {0});
        storage.add_subgeometry("SecondFace", {1});

        const ImporterMesh mesh = tessellate(input);
        REQUIRE_EQ(mesh.subgeometries.size(), 2);
        auto uvs = mesh.texcoord_stream();
        REQUIRE(uvs.valid());

        const auto first_vertices = referenced_vertices(mesh.subgeometries[0]);
        const auto second_vertices = referenced_vertices(mesh.subgeometries[1]);
        for (uint32_t vertex : first_vertices)
            CHECK_EQ(uvs[vertex], float2(0.1f, 0.2f));
        for (uint32_t vertex : second_vertices)
            CHECK_EQ(uvs[vertex], float2(0.8f, 0.9f));
    }
}

TEST_SUITE_END();
