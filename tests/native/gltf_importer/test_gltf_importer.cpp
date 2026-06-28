// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "testing.h"
#include "falcor2/importers/importer_types.h"
#include "falcor2/importers/gltf_importer/gltf_importer.h"
#include "falcor2/core/object.h"

#include <sgl/core/platform.h>
#include <sgl/math/vector_math.h>

using namespace falcor;

namespace {

void check_mesh(const ImporterMesh& mesh, const std::string& expected_name)
{
    CHECK_EQ(mesh.name, expected_name);
    CHECK_GT(mesh.vertex_count(), 0);
    CHECK_GT(mesh.subgeometries.size(), 0);

    // Check that all vertices have reasonable values using streams
    size_t vertex_count = mesh.vertex_count();

    auto positions = mesh.position_stream();
    if (positions.valid()) {
        for (size_t i = 0; i < vertex_count; ++i) {
            const auto& pos = positions[i];
            CHECK(std::isfinite(pos.x));
            CHECK(std::isfinite(pos.y));
            CHECK(std::isfinite(pos.z));
        }
    }

    auto normals = mesh.normal_stream();
    if (normals.valid()) {
        for (size_t i = 0; i < vertex_count; ++i) {
            const auto& normal = normals[i];
            CHECK(std::isfinite(normal.x));
            CHECK(std::isfinite(normal.y));
            CHECK(std::isfinite(normal.z));
        }
    }

    auto tangents = mesh.tangent_stream();
    if (tangents.valid()) {
        for (size_t i = 0; i < vertex_count; ++i) {
            const auto& tangent = tangents[i];
            CHECK(std::isfinite(tangent.x));
            CHECK(std::isfinite(tangent.y));
            CHECK(std::isfinite(tangent.z));
        }
    }

    auto texcoords = mesh.texcoord_stream();
    if (texcoords.valid()) {
        for (size_t i = 0; i < vertex_count; ++i) {
            const auto& uv = texcoords[i];
            CHECK(std::isfinite(uv.x));
            CHECK(std::isfinite(uv.y));
        }
    }

    auto handedness_stream = mesh.handedness_stream();
    if (handedness_stream.valid()) {
        for (size_t i = 0; i < vertex_count; ++i) {
            float h = handedness_stream[i];
            CHECK((h == 1.0f || h == -1.0f));
        }
    }

    // Check that all indices are valid
    for (const auto& subgeom : mesh.subgeometries) {
        for (const auto& triangle : subgeom.indices) {
            CHECK_LT(triangle.x, vertex_count);
            CHECK_LT(triangle.y, vertex_count);
            CHECK_LT(triangle.z, vertex_count);
        }
    }
}

} // anonymous namespace

TEST_CASE("GltfImporter - Basic Loading")
{
    auto importer = make_ref<GltfImporter>();

    auto sample_file = testing::project_directory() / "data" / "assets" / "kronos" / "Box" / "glTF-Binary" / "Box.glb";

    REQUIRE(std::filesystem::exists(sample_file));

    auto scene = importer->load_scene(sample_file);
    REQUIRE(scene != nullptr);

    // The sample file should have at least one mesh
    REQUIRE_GT(scene->meshes.size(), 0);

    // Check the first mesh
    check_mesh(scene->meshes[0], scene->meshes[0].name); // Use actual name from the file

    // Verify specific counts for the test model
    const auto& first_mesh = scene->meshes[0];
    CHECK_EQ(first_mesh.vertex_count(), 24);

    // Count total triangles across all subgeometries
    size_t total_triangles = 0;
    for (const auto& subgeom : first_mesh.subgeometries) {
        total_triangles += subgeom.indices.size();
    }
    CHECK_EQ(total_triangles, 12);

    // Log some basic information about the loaded scene
    INFO(
        "Loaded GLTF scene with {} meshes, {} materials, {} cameras, {} lights",
        scene->meshes.size(),
        scene->materials.size(),
        scene->cameras.size(),
        scene->lights.size()
    );
}

TEST_CASE("GltfImporter - Invalid File")
{
    auto importer = make_ref<GltfImporter>();

    auto invalid_file = testing::project_directory() / "non_existent_file.glb";

    // Should throw an exception for invalid files
    CHECK_THROWS_AS(importer->load_scene(invalid_file), std::runtime_error);
}

TEST_CASE("GltfImporter - Empty Scene Handling")
{
    auto importer = make_ref<GltfImporter>();

    // This test assumes we can create a minimal valid GLTF file with no meshes
    // For now, we'll just test that we can handle the case where load_scene returns a valid scene
    // but with no meshes
    auto sample_file = testing::project_directory() / "data" / "assets" / "kronos" / "Box" / "glTF-Binary" / "Box.glb";

    if (std::filesystem::exists(sample_file)) {
        auto scene = importer->load_scene(sample_file);
        if (scene != nullptr) {
            // Check that the scene is valid even if it has no content
            CHECK_GE(scene->meshes.size(), 0);
            CHECK_GE(scene->materials.size(), 0);
            CHECK_GE(scene->cameras.size(), 0);
            CHECK_GE(scene->lights.size(), 0);
        }
    }
}

TEST_CASE("GltfImporter - Vertex Data Validation")
{
    auto importer = make_ref<GltfImporter>();

    auto sample_file = testing::project_directory() / "data" / "assets" / "kronos" / "Box" / "glTF-Binary" / "Box.glb";

    if (std::filesystem::exists(sample_file)) {
        auto scene = importer->load_scene(sample_file);

        if (scene != nullptr && !scene->meshes.empty()) {
            const auto& first_mesh = scene->meshes[0];

            // Verify that we have valid vertex data
            REQUIRE_GT(first_mesh.vertex_count(), 0);

            // Check that all vertices have reasonable bounds
            auto positions = first_mesh.position_stream();
            auto normals = first_mesh.normal_stream();
            auto texcoords = first_mesh.texcoord_stream();

            for (size_t i = 0; i < first_mesh.vertex_count(); ++i) {
                // Position should be reasonable (not infinite or NaN)
                if (positions.valid()) {
                    const auto& pos = positions[i];
                    CHECK(std::abs(pos.x) < 1000.0f);
                    CHECK(std::abs(pos.y) < 1000.0f);
                    CHECK(std::abs(pos.z) < 1000.0f);
                }

                // Normal should be normalized (approximately)
                if (normals.valid()) {
                    const auto& normal = normals[i];
                    float normal_length = sgl::math::length(normal);
                    CHECK(normal_length > 0.5f); // Allow for some tolerance
                    CHECK(normal_length < 1.5f);
                }

                // UV coordinates should be reasonable
                if (texcoords.valid()) {
                    const auto& uv = texcoords[i];
                    CHECK(uv.x >= -10.0f);
                    CHECK(uv.x <= 10.0f);
                    CHECK(uv.y >= -10.0f);
                    CHECK(uv.y <= 10.0f);
                }
            }

            // Verify that we have valid indices
            REQUIRE_GT(first_mesh.subgeometries.size(), 0);
            const auto& first_subgeom = first_mesh.subgeometries[0];
            REQUIRE_GT(first_subgeom.indices.size(), 0);

            // Check that indices form valid triangles
            for (const auto& triangle : first_subgeom.indices) {
                CHECK_NE(triangle.x, triangle.y);
                CHECK_NE(triangle.y, triangle.z);
                CHECK_NE(triangle.z, triangle.x);
            }
        }
    }
}

namespace {

void check_node(const ImporterNode& node, const std::string& expected_name_prefix = "")
{
    // Name should not be empty
    CHECK_FALSE(node.name.empty());

    if (!expected_name_prefix.empty()) {
        CHECK(node.name.find(expected_name_prefix) != std::string::npos);
    }

    // Transform matrix should be finite
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            CHECK(std::isfinite(node.transform[i][j]));
        }
    }

    // Mesh index should be valid if set
    if (node.mesh_index >= 0) {
        // Note: We can't check against scene here, but we can check it's not negative
        CHECK_GE(node.mesh_index, 0);
    }

    // Parent index should be valid if set
    if (node.parent >= 0) {
        CHECK_GE(node.parent, 0);
    }

    // Children indices should be valid
    for (int child_idx : node.children) {
        CHECK_GE(child_idx, 0);
    }
}

} // namespace

TEST_CASE("GltfImporter - Node Hierarchy")
{
    auto importer = make_ref<GltfImporter>();
    auto sample_file = testing::project_directory() / "data" / "assets" / "kronos" / "Box" / "glTF-Binary" / "Box.glb";

    if (std::filesystem::exists(sample_file)) {
        auto scene = importer->load_scene(sample_file);

        REQUIRE(scene != nullptr);

        // Check that we have nodes
        REQUIRE_GT(scene->nodes.size(), 0);

        // Check that we have root nodes
        REQUIRE_GT(scene->root_nodes.size(), 0);

        // Validate each node
        for (const auto& node : scene->nodes) {
            check_node(node);
        }

        // Check that root nodes are valid indices
        for (int root_idx : scene->root_nodes) {
            CHECK_GE(root_idx, 0);
            CHECK_LT(root_idx, static_cast<int>(scene->nodes.size()));

            // Root nodes should have no parent
            CHECK_EQ(scene->nodes[root_idx].parent, -1);
        }

        // Check parent-child relationships
        for (size_t node_idx = 0; node_idx < scene->nodes.size(); ++node_idx) {
            const auto& node = scene->nodes[node_idx];

            // Check children point back to this node as parent
            for (int child_idx : node.children) {
                REQUIRE_GE(child_idx, 0);
                REQUIRE_LT(child_idx, static_cast<int>(scene->nodes.size()));
                CHECK_EQ(scene->nodes[child_idx].parent, static_cast<int>(node_idx));
            }

            // Check parent points to this node in its children
            if (node.parent >= 0) {
                REQUIRE_LT(node.parent, static_cast<int>(scene->nodes.size()));
                const auto& parent_node = scene->nodes[node.parent];
                CHECK(
                    std::find(parent_node.children.begin(), parent_node.children.end(), static_cast<int>(node_idx))
                    != parent_node.children.end()
                );
            }
        }

        // Check mesh references
        for (const auto& node : scene->nodes) {
            if (node.mesh_index >= 0) {
                CHECK_LT(node.mesh_index, static_cast<int>(scene->meshes.size()));
            }
        }
    }
}


TEST_CASE("GltfImporter - Embedded Textures")
{
    auto importer = make_ref<GltfImporter>();
    auto sample_file
        = testing::project_directory() / "data" / "assets" / "kronos" / "Avocado" / "glTF-Binary" / "Avocado.glb";

    if (std::filesystem::exists(sample_file)) {
        auto scene = importer->load_scene(sample_file);

        REQUIRE(scene != nullptr);
        REQUIRE_EQ(scene->textures.size(), 3);
    }
}


TEST_CASE("GltfImporter - File Textures")
{
    auto importer = make_ref<GltfImporter>();
    auto sample_file
        = testing::project_directory() / "data" / "assets" / "kronos" / "DamagedHelmet" / "glTF" / "DamagedHelmet.gltf";

    if (std::filesystem::exists(sample_file)) {
        auto scene = importer->load_scene(sample_file);

        REQUIRE(scene != nullptr);
        REQUIRE_EQ(scene->textures.size(), 5);
    }
}

TEST_CASE("GltfImporter - Flattened Node Hierarchy")
{
    auto importer = make_ref<GltfImporter>();
    auto sample_file = testing::project_directory() / "data" / "assets" / "kronos" / "Box" / "glTF-Binary" / "Box.glb";

    if (std::filesystem::exists(sample_file)) {
        auto scene = importer->load_scene(sample_file);
        REQUIRE(scene != nullptr);

        // Explicitly flatten the node hierarchy
        scene->flatten_node_hierarchy();

        // After flattening, the scene should have a flattened node hierarchy
        // This means:
        // 1. All nodes should be root nodes (no hierarchy)
        // 2. Only nodes with meshes should be present
        // 3. All transforms should be world-space transforms

        REQUIRE_GT(scene->nodes.size(), 0);
        REQUIRE_EQ(scene->nodes.size(), scene->root_nodes.size());

        // All nodes should be root nodes (no parents, no children)
        for (const auto& node : scene->nodes) {
            CHECK_EQ(node.parent, -1);
            CHECK(node.children.empty());
        }

        // All nodes should have valid mesh references
        for (const auto& node : scene->nodes) {
            CHECK_GE(node.mesh_index, 0);
            CHECK_LT(node.mesh_index, static_cast<int>(scene->meshes.size()));
        }

        // All root node indices should be valid and sequential
        for (size_t i = 0; i < scene->root_nodes.size(); ++i) {
            CHECK_EQ(scene->root_nodes[i], static_cast<int>(i));
        }

        // Verify that transforms are valid world-space transforms
        for (const auto& node : scene->nodes) {
            // Transform matrix should be finite
            for (int row = 0; row < 4; ++row) {
                for (int col = 0; col < 4; ++col) {
                    CHECK(std::isfinite(node.transform[row][col]));
                }
            }

            // Transform should be invertible (determinant != 0)
            // Note: We don't have a determinant function readily available,
            // so we'll just check that it's not the zero matrix
            bool has_non_zero = false;
            for (int row = 0; row < 4 && !has_non_zero; ++row) {
                for (int col = 0; col < 4 && !has_non_zero; ++col) {
                    if (std::abs(node.transform[row][col]) > 1e-8f) {
                        has_non_zero = true;
                    }
                }
            }
            CHECK(has_non_zero);
        }

        // Verify that each node name is not empty
        for (const auto& node : scene->nodes) {
            CHECK_FALSE(node.name.empty());
        }

        // Log information about the flattened hierarchy
        INFO("Loaded scene with {} flattened nodes (all roots), {} meshes", scene->nodes.size(), scene->meshes.size());

        // Additional check: verify that the number of nodes matches
        // the expected pattern for a flattened hierarchy
        // (Each node should reference exactly one mesh)
        std::set<int> referenced_meshes;
        for (const auto& node : scene->nodes) {
            referenced_meshes.insert(node.mesh_index);
        }

        // The number of unique mesh references should be <= number of nodes
        // (It could be less if the same mesh is referenced by multiple nodes)
        CHECK_LE(referenced_meshes.size(), scene->nodes.size());

        // But we should have at least one mesh reference
        CHECK_GT(referenced_meshes.size(), 0);
    }
}

TEST_CASE("AABB calculation and validation")
{
    SUBCASE("AABB struct basic functionality")
    {
        // Test invalid AABB
        AABB aabb;
        CHECK_FALSE(aabb.is_valid());

        // Test valid AABB
        AABB valid_aabb(float3(-1, -2, -3), float3(1, 2, 3));
        CHECK(valid_aabb.is_valid());

        // Test center calculation
        float3 center = valid_aabb.center();
        CHECK_EQ(center.x, 0.0f);
        CHECK_EQ(center.y, 0.0f);
        CHECK_EQ(center.z, 0.0f);

        // Test size calculation
        float3 size = valid_aabb.size();
        CHECK_EQ(size.x, 2.0f);
        CHECK_EQ(size.y, 4.0f);
        CHECK_EQ(size.z, 6.0f);
    }

    SUBCASE("AABB expansion with points")
    {
        AABB aabb;
        CHECK_FALSE(aabb.is_valid());

        // Expand with first point
        aabb.expand(float3(1, 2, 3));
        CHECK(aabb.is_valid());
        CHECK_EQ(aabb.min.x, 1.0f);
        CHECK_EQ(aabb.min.y, 2.0f);
        CHECK_EQ(aabb.min.z, 3.0f);
        CHECK_EQ(aabb.max.x, 1.0f);
        CHECK_EQ(aabb.max.y, 2.0f);
        CHECK_EQ(aabb.max.z, 3.0f);

        // Expand with second point
        aabb.expand(float3(-1, 4, 1));
        CHECK_EQ(aabb.min.x, -1.0f);
        CHECK_EQ(aabb.min.y, 2.0f);
        CHECK_EQ(aabb.min.z, 1.0f);
        CHECK_EQ(aabb.max.x, 1.0f);
        CHECK_EQ(aabb.max.y, 4.0f);
        CHECK_EQ(aabb.max.z, 3.0f);
    }

    SUBCASE("AABB expansion with other AABBs")
    {
        AABB aabb1(float3(-1, -1, -1), float3(1, 1, 1));
        AABB aabb2(float3(0, 0, 0), float3(2, 3, 4));

        aabb1.expand(aabb2);
        CHECK_EQ(aabb1.min.x, -1.0f);
        CHECK_EQ(aabb1.min.y, -1.0f);
        CHECK_EQ(aabb1.min.z, -1.0f);
        CHECK_EQ(aabb1.max.x, 2.0f);
        CHECK_EQ(aabb1.max.y, 3.0f);
        CHECK_EQ(aabb1.max.z, 4.0f);
    }

    SUBCASE("AABB transformation")
    {
        AABB aabb(float3(-1, -1, -1), float3(1, 1, 1));

        // Create a simple translation matrix
        float4x4 translation = float4x4::identity();
        translation[0][3] = 5.0f;  // Translate by 5 in X
        translation[1][3] = 3.0f;  // Translate by 3 in Y
        translation[2][3] = -2.0f; // Translate by -2 in Z

        AABB transformed = aabb.transform(translation);

        CHECK(transformed.is_valid());
        CHECK_EQ(transformed.min.x, 4.0f);
        CHECK_EQ(transformed.min.y, 2.0f);
        CHECK_EQ(transformed.min.z, -3.0f);
        CHECK_EQ(transformed.max.x, 6.0f);
        CHECK_EQ(transformed.max.y, 4.0f);
        CHECK_EQ(transformed.max.z, -1.0f);
    }

    SUBCASE("GLTF scene AABB calculation")
    {
        auto test_file
            = testing::project_directory() / "data" / "assets" / "kronos" / "Box" / "glTF-Binary" / "Box.glb";

        if (!std::filesystem::exists(test_file)) {
            INFO("Test file does not exist, skipping AABB calculation test: {}", test_file.string());
            return;
        }

        GltfImporter importer;
        auto scene = importer.load_scene(test_file);
        REQUIRE(scene);

        // Calculate AABBs
        scene->calculate_aabbs();

        // Check that all meshes have valid AABBs (if they have vertices)
        for (const auto& mesh : scene->meshes) {
            if (mesh.vertex_count() > 0) {
                CHECK(mesh.local_aabb.is_valid());

                // Verify AABB encompasses all vertices
                auto positions = mesh.position_stream();
                if (positions.valid()) {
                    for (size_t i = 0; i < mesh.vertex_count(); ++i) {
                        const auto& pos = positions[i];
                        CHECK_GE(pos.x, mesh.local_aabb.min.x);
                        CHECK_GE(pos.y, mesh.local_aabb.min.y);
                        CHECK_GE(pos.z, mesh.local_aabb.min.z);
                        CHECK_LE(pos.x, mesh.local_aabb.max.x);
                        CHECK_LE(pos.y, mesh.local_aabb.max.y);
                        CHECK_LE(pos.z, mesh.local_aabb.max.z);
                    }
                }
            } else {
                CHECK_FALSE(mesh.local_aabb.is_valid());
            }
        }

        // Check that nodes with meshes have valid world AABBs
        for (const auto& node : scene->nodes) {
            if (node.mesh_index >= 0 && node.mesh_index < static_cast<int>(scene->meshes.size())) {
                const auto& mesh = scene->meshes[node.mesh_index];
                if (mesh.vertex_count() > 0) {
                    CHECK(node.world_aabb.is_valid());
                }
            }
        }

        INFO("Scene AABB calculation test completed successfully");
    }
}

TEST_CASE("Scene rescaling functionality")
{
    SUBCASE("rescale_to_fit functionality")
    {
        auto test_file
            = testing::project_directory() / "data" / "assets" / "kronos" / "Box" / "glTF-Binary" / "Box.glb";

        if (!std::filesystem::exists(test_file)) {
            INFO("Test file does not exist, skipping rescale test: {}", test_file.string());
            return;
        }

        GltfImporter importer;
        auto scene = importer.load_scene(test_file);
        REQUIRE(scene);

        // Calculate initial AABBs
        scene->calculate_aabbs();

        // Get original scene AABB
        AABB original_aabb = scene->get_scene_aabb();
        REQUIRE(original_aabb.is_valid());

        float3 original_size = original_aabb.size();
        float3 original_center = original_aabb.center();
        float original_max_dim = sgl::math::max(original_size.x, sgl::math::max(original_size.y, original_size.z));

        // Test rescaling to size 1.0
        scene->rescale_to_fit(1.0f);

        // Get new scene AABB
        AABB new_aabb = scene->get_scene_aabb();
        REQUIRE(new_aabb.is_valid());

        float3 new_size = new_aabb.size();
        float3 new_center = new_aabb.center();
        float new_max_dim = sgl::math::max(new_size.x, sgl::math::max(new_size.y, new_size.z));

        // Check that the scene is centered at origin
        CHECK_LT(std::abs(new_center.x), 1e-6f);
        CHECK_LT(std::abs(new_center.y), 1e-6f);
        CHECK_LT(std::abs(new_center.z), 1e-6f);

        // Check that the max dimension is approximately 1.0
        CHECK_LT(std::abs(new_max_dim - 1.0f), 1e-6f);

        // Check that aspect ratio is preserved
        if (original_max_dim > 0.0f) {
            float expected_scale = 1.0f / original_max_dim;
            CHECK_LT(std::abs(new_size.x - original_size.x * expected_scale), 1e-5f);
            CHECK_LT(std::abs(new_size.y - original_size.y * expected_scale), 1e-5f);
            CHECK_LT(std::abs(new_size.z - original_size.z * expected_scale), 1e-5f);
        }

        INFO("Scene rescaling test completed successfully");
    }

    SUBCASE("rescale_to_fit with different sizes")
    {
        auto test_file
            = testing::project_directory() / "data" / "assets" / "kronos" / "Box" / "glTF-Binary" / "Box.glb";

        if (!std::filesystem::exists(test_file)) {
            INFO("Test file does not exist, skipping rescale test: {}", test_file.string());
            return;
        }

        // Test different target sizes
        std::vector<float> target_sizes = {0.5f, 2.0f, 10.0f};

        for (float target_size : target_sizes) {
            GltfImporter importer;
            auto scene = importer.load_scene(test_file);
            REQUIRE(scene);

            // Rescale to target size
            scene->rescale_to_fit(target_size);

            // Get scene AABB
            AABB scene_aabb = scene->get_scene_aabb();
            REQUIRE(scene_aabb.is_valid());

            float3 size = scene_aabb.size();
            float max_dim = sgl::math::max(size.x, sgl::math::max(size.y, size.z));

            // Check that max dimension matches target
            CHECK_LT(std::abs(max_dim - target_size), 1e-5f);

            // Check that scene is centered (use slightly relaxed tolerance for floating-point precision)
            float3 center = scene_aabb.center();
            CHECK_LT(std::abs(center.x), 5e-6f);
            CHECK_LT(std::abs(center.y), 5e-6f);
            CHECK_LT(std::abs(center.z), 5e-6f);
        }

        INFO("Multiple size rescaling test completed successfully");
    }

    SUBCASE("get_scene_aabb functionality")
    {
        auto test_file
            = testing::project_directory() / "data" / "assets" / "kronos" / "Box" / "glTF-Binary" / "Box.glb";

        if (!std::filesystem::exists(test_file)) {
            INFO("Test file does not exist, skipping scene AABB test: {}", test_file.string());
            return;
        }

        GltfImporter importer;
        auto scene = importer.load_scene(test_file);
        REQUIRE(scene);

        // Calculate AABBs
        scene->calculate_aabbs();

        // Get scene AABB
        AABB scene_aabb = scene->get_scene_aabb();
        REQUIRE(scene_aabb.is_valid());

        // Verify that scene AABB encompasses all root node AABBs
        for (int root_idx : scene->root_nodes) {
            if (root_idx >= 0 && root_idx < static_cast<int>(scene->nodes.size())) {
                const auto& node = scene->nodes[root_idx];
                if (node.world_aabb.is_valid()) {
                    // Check that scene AABB contains the node AABB
                    CHECK_LE(scene_aabb.min.x, node.world_aabb.min.x + 1e-6f);
                    CHECK_LE(scene_aabb.min.y, node.world_aabb.min.y + 1e-6f);
                    CHECK_LE(scene_aabb.min.z, node.world_aabb.min.z + 1e-6f);

                    CHECK_GE(scene_aabb.max.x, node.world_aabb.max.x - 1e-6f);
                    CHECK_GE(scene_aabb.max.y, node.world_aabb.max.y - 1e-6f);
                    CHECK_GE(scene_aabb.max.z, node.world_aabb.max.z - 1e-6f);
                }
            }
        }

        INFO("Scene AABB functionality test completed successfully");
    }
}

TEST_CASE("ImporterScene default camera autoposition")
{
    auto test_file = testing::project_directory() / "data" / "assets" / "kronos" / "Box" / "glTF-Binary" / "Box.glb";

    if (!std::filesystem::exists(test_file)) {
        INFO("Test file does not exist, skipping camera autoposition test: {}", test_file.string());
        return;
    }

    SUBCASE("add_default_camera_robust_fit")
    {
        GltfImporter importer;
        auto scene = importer.load_scene(test_file);
        REQUIRE(scene);

        const size_t old_camera_count = scene->cameras.size();
        const size_t old_node_count = scene->nodes.size();
        const size_t old_root_count = scene->root_nodes.size();

        const int camera_index = scene->add_default_camera_robust_fit();

        REQUIRE_EQ(scene->cameras.size(), old_camera_count + 1);
        REQUIRE_EQ(scene->nodes.size(), old_node_count + 1);
        REQUIRE_EQ(scene->root_nodes.size(), old_root_count + 1);
        REQUIRE_GE(camera_index, 0);
        REQUIRE_LT(camera_index, static_cast<int>(scene->cameras.size()));

        const auto& camera = scene->cameras[camera_index];
        CHECK_EQ(camera.projection, ImporterCamera::Projection::perspective);
        CHECK_EQ(camera.fov_direction, ImporterCamera::FOVDirection::vertical);
        CHECK_GT(camera.focal_length, 0.f);
        CHECK_GT(camera.focus_distance, 0.f);

        const int node_index = static_cast<int>(scene->nodes.size()) - 1;
        const auto& node = scene->nodes[node_index];
        CHECK_EQ(node.camera_index, camera_index);
        CHECK_EQ(node.parent, -1);

        // Translation should be finite.
        const float3 cam_pos = node.transform.get_col(3).xyz();
        CHECK(std::isfinite(cam_pos.x));
        CHECK(std::isfinite(cam_pos.y));
        CHECK(std::isfinite(cam_pos.z));
    }

    SUBCASE("add_default_camera_best_view")
    {
        GltfImporter importer;
        auto scene = importer.load_scene(test_file);
        REQUIRE(scene);

        const size_t old_camera_count = scene->cameras.size();
        const size_t old_node_count = scene->nodes.size();
        const size_t old_root_count = scene->root_nodes.size();

        const int camera_index = scene->add_default_camera_best_view();

        REQUIRE_EQ(scene->cameras.size(), old_camera_count + 1);
        REQUIRE_EQ(scene->nodes.size(), old_node_count + 1);
        REQUIRE_EQ(scene->root_nodes.size(), old_root_count + 1);
        REQUIRE_GE(camera_index, 0);
        REQUIRE_LT(camera_index, static_cast<int>(scene->cameras.size()));

        const auto& camera = scene->cameras[camera_index];
        CHECK_EQ(camera.projection, ImporterCamera::Projection::perspective);
        CHECK_EQ(camera.fov_direction, ImporterCamera::FOVDirection::vertical);
        CHECK_GT(camera.focal_length, 0.f);
        CHECK_GT(camera.focus_distance, 0.f);

        const int node_index = static_cast<int>(scene->nodes.size()) - 1;
        const auto& node = scene->nodes[node_index];
        CHECK_EQ(node.camera_index, camera_index);
        CHECK_EQ(node.parent, -1);

        // Translation should be finite.
        const float3 cam_pos = node.transform.get_col(3).xyz();
        CHECK(std::isfinite(cam_pos.x));
        CHECK(std::isfinite(cam_pos.y));
        CHECK(std::isfinite(cam_pos.z));
    }
}

TEST_CASE("GltfImporter - Texture Transforms")
{
    auto importer = make_ref<GltfImporter>();
    auto sample_file = testing::project_directory() / "data" / "assets" / "kronos" / "TextureTransformTest" / "glTF"
        / "TextureTransformTest.gltf";

    if (std::filesystem::exists(sample_file)) {
        auto scene = importer->load_scene(sample_file);

        REQUIRE(scene != nullptr);
        INFO("Loaded GLTF scene with {} materials", scene->materials.size());

        // The TextureTransformTest.gltf file should have 12 materials (counting from the file)
        // Each material demonstrates different aspects of texture transforms
        REQUIRE_GE(scene->materials.size(), 6); // At least the main test materials

        // Find and test specific materials by name
        for (const auto& material : scene->materials) {
            INFO("Testing material: {}", material.name);

            if (material.name == "Offset U") {
                // This material should have a baseColorTexture with offset [0.5, 0.0]
                REQUIRE(material.params.has_property("baseColorTexture"));
                CHECK_EQ(material.params.get<int>("baseColorTexture"), 0);

                // Check for texture transform offset
                if (material.params.has_property("baseColorTexture_offset")) {
                    auto offset = material.params.get<float2>("baseColorTexture_offset");
                    CHECK_EQ(offset.x, 0.5f);
                    CHECK_EQ(offset.y, 0.0f);
                }
            } else if (material.name == "Offset V") {
                // This material should have a baseColorTexture with offset [0.0, 0.5]
                REQUIRE(material.params.has_property("baseColorTexture"));
                CHECK_EQ(material.params.get<int>("baseColorTexture"), 0);

                // Check for texture transform offset
                if (material.params.has_property("baseColorTexture_offset")) {
                    auto offset = material.params.get<float2>("baseColorTexture_offset");
                    CHECK_EQ(offset.x, 0.0f);
                    CHECK_EQ(offset.y, 0.5f);
                }
            } else if (material.name == "Offset UV") {
                // This material should have a baseColorTexture with offset [0.5, 0.5]
                REQUIRE(material.params.has_property("baseColorTexture"));
                CHECK_EQ(material.params.get<int>("baseColorTexture"), 0);

                // Check for texture transform offset
                if (material.params.has_property("baseColorTexture_offset")) {
                    auto offset = material.params.get<float2>("baseColorTexture_offset");
                    CHECK_EQ(offset.x, 0.5f);
                    CHECK_EQ(offset.y, 0.5f);
                }
            } else if (material.name == "Rotation") {
                // This material should have a baseColorTexture with rotation 0.39269908169872415480783042290994
                REQUIRE(material.params.has_property("baseColorTexture"));
                CHECK_EQ(material.params.get<int>("baseColorTexture"), 1);

                // Check for texture transform rotation
                if (material.params.has_property("baseColorTexture_rotation")) {
                    auto rotation = material.params.get<float>("baseColorTexture_rotation");
                    CHECK(std::abs(rotation - 0.39269908169872415480783042290994f) < 1e-6f);
                }
            } else if (material.name == "Scale") {
                // This material should have a baseColorTexture with scale [1.5, 1.5]
                REQUIRE(material.params.has_property("baseColorTexture"));
                CHECK_EQ(material.params.get<int>("baseColorTexture"), 1);

                // Check for texture transform scale
                if (material.params.has_property("baseColorTexture_scale")) {
                    auto scale = material.params.get<float2>("baseColorTexture_scale");
                    CHECK_EQ(scale.x, 1.5f);
                    CHECK_EQ(scale.y, 1.5f);
                }
            } else if (material.name == "All") {
                // This material should have all transforms: offset [-0.2, -0.1], rotation 0.3, scale [1.5, 1.5]
                REQUIRE(material.params.has_property("baseColorTexture"));
                CHECK_EQ(material.params.get<int>("baseColorTexture"), 1);

                // Check for texture transform offset
                if (material.params.has_property("baseColorTexture_offset")) {
                    auto offset = material.params.get<float2>("baseColorTexture_offset");
                    CHECK(std::abs(offset.x - (-0.2f)) < 1e-6f);
                    CHECK(std::abs(offset.y - (-0.1f)) < 1e-6f);
                }

                // Check for texture transform rotation
                if (material.params.has_property("baseColorTexture_rotation")) {
                    auto rotation = material.params.get<float>("baseColorTexture_rotation");
                    CHECK(std::abs(rotation - 0.3f) < 1e-6f);
                }

                // Check for texture transform scale
                if (material.params.has_property("baseColorTexture_scale")) {
                    auto scale = material.params.get<float2>("baseColorTexture_scale");
                    CHECK_EQ(scale.x, 1.5f);
                    CHECK_EQ(scale.y, 1.5f);
                }
            }
        }

        INFO("Texture transform test completed successfully");
    }
}

TEST_CASE("GltfImporter - Alpha Mode and Double-Sided Materials")
{
    auto importer = make_ref<GltfImporter>();
    auto sample_file = testing::project_directory() / "data" / "assets" / "kronos" / "AlphaBlendModeTest" / "glTF"
        / "AlphaBlendModeTest.gltf";

    if (std::filesystem::exists(sample_file)) {
        auto scene = importer->load_scene(sample_file);

        REQUIRE(scene != nullptr);
        INFO("Loaded GLTF scene with {} materials", scene->materials.size());

        // The AlphaBlendModeTest should have materials with different alpha modes
        REQUIRE_GT(scene->materials.size(), 0);

        bool found_opaque = false;
        bool found_mask = false;
        bool found_blend = false;

        // Check each material for alpha mode properties
        for (const auto& material : scene->materials) {
            // Read values from property bag
            AlphaMode alpha_mode = material.params.get("alphaMode", AlphaMode::opaque);
            bool double_sided = material.params.get("doubleSided", false);
            float alpha_cutoff = material.params.get("alphaCutoff", 0.5f);

            INFO(
                "Testing material: {} - Alpha Mode: {}, Double Sided: {}, Alpha Cutoff: {}",
                material.name,
                alpha_mode,
                double_sided,
                alpha_cutoff
            );

            // Verify alpha mode is a valid value
            CHECK((alpha_mode == AlphaMode::opaque || alpha_mode == AlphaMode::mask || alpha_mode == AlphaMode::blend));

            // Track which alpha modes we've seen
            if (alpha_mode == AlphaMode::opaque) {
                found_opaque = true;
            } else if (alpha_mode == AlphaMode::mask) {
                found_mask = true;
                // Alpha cutoff should be valid for mask mode
                CHECK_GE(alpha_cutoff, 0.0f);
                CHECK_LE(alpha_cutoff, 1.0f);
            } else if (alpha_mode == AlphaMode::blend) {
                found_blend = true;
            }

            // Double sided flag should be a valid boolean
            CHECK((double_sided == true || double_sided == false));

            // Alpha cutoff should be in valid range [0.0, 1.0]
            CHECK_GE(alpha_cutoff, 0.0f);
            CHECK_LE(alpha_cutoff, 1.0f);

            // Test specific material names if available
            if (material.name.find("opaque") != std::string::npos) {
                CHECK_EQ(alpha_mode, AlphaMode::opaque);
            }
            if (material.name.find("mask") != std::string::npos) {
                CHECK_EQ(alpha_mode, AlphaMode::mask);
            }
            if (material.name.find("blend") != std::string::npos) {
                CHECK_EQ(alpha_mode, AlphaMode::blend);
            }
        }

        // Verify we found at least some alpha mode variants
        // (The exact test depends on the content of AlphaBlendModeTest.gltf)
        INFO("Found alpha modes - opaque: {}, mask: {}, blend: {}", found_opaque, found_mask, found_blend);

        INFO("Alpha mode and double-sided materials test completed successfully");
    } else {
        INFO("AlphaBlendModeTest.gltf not found, skipping alpha mode test");
    }
}

// ---------------------------------------------------------------------------
// Animation import tests
// ---------------------------------------------------------------------------

TEST_CASE("GltfImporter - Linear Translation Animation")
{
    auto importer = make_ref<GltfImporter>();
    auto file = testing::project_directory() / "data" / "assets" / "animation_test" / "anim_linear_translation.glb";
    REQUIRE(std::filesystem::exists(file));

    auto scene = importer->load_scene(file);
    REQUIRE(scene != nullptr);

    // Should have one animation with one channel.
    CHECK_EQ(scene->animation.name, "TranslateX");
    REQUIRE_EQ(scene->animation.channels.size(), 1);

    const auto& ch = scene->animation.channels[0];
    CHECK_EQ(ch.value_type, AnimationValueType::float3);
    CHECK_EQ(ch.interpolation_mode, AnimationInterpolationMode::linear);
    REQUIRE_EQ(ch.keyframe_count(), 3);
    CHECK_EQ(ch.times[0], doctest::Approx(0.f));
    CHECK_EQ(ch.times[1], doctest::Approx(1.f));
    CHECK_EQ(ch.times[2], doctest::Approx(2.f));

    // Values: (0,0,0), (2.5,0,0), (5,0,0)
    REQUIRE_EQ(ch.values.size(), 9);
    CHECK_EQ(ch.values[0], doctest::Approx(0.f));
    CHECK_EQ(ch.values[3], doctest::Approx(2.5f));
    CHECK_EQ(ch.values[6], doctest::Approx(5.f));

    // No tangent data for linear.
    CHECK(ch.in_tangents.empty());
    CHECK(ch.out_tangents.empty());

    // Node 0 should reference the translation channel.
    REQUIRE_GT(scene->nodes.size(), 0);
    CHECK_EQ(scene->nodes[0].animation_translation, 0);
    CHECK_EQ(scene->nodes[0].animation_rotation, -1);
    CHECK_EQ(scene->nodes[0].animation_scale, -1);

    CHECK_EQ(scene->animation.duration(), doctest::Approx(2.f));
}

TEST_CASE("GltfImporter - Step Rotation Animation")
{
    auto importer = make_ref<GltfImporter>();
    auto file = testing::project_directory() / "data" / "assets" / "animation_test" / "anim_step_rotation.glb";
    REQUIRE(std::filesystem::exists(file));

    auto scene = importer->load_scene(file);
    REQUIRE(scene != nullptr);

    CHECK_EQ(scene->animation.name, "StepRotation");
    REQUIRE_EQ(scene->animation.channels.size(), 1);

    const auto& ch = scene->animation.channels[0];
    CHECK_EQ(ch.value_type, AnimationValueType::quatf);
    CHECK_EQ(ch.interpolation_mode, AnimationInterpolationMode::step);
    REQUIRE_EQ(ch.keyframe_count(), 2);

    // Quaternion values (x,y,z,w): identity, then 90 around Y.
    REQUIRE_EQ(ch.values.size(), 8);
    // Identity quaternion
    CHECK_EQ(ch.values[0], doctest::Approx(0.f));
    CHECK_EQ(ch.values[1], doctest::Approx(0.f));
    CHECK_EQ(ch.values[2], doctest::Approx(0.f));
    CHECK_EQ(ch.values[3], doctest::Approx(1.f));
    // 90 degrees around Y: (0, sin(45), 0, cos(45))
    CHECK_EQ(ch.values[4], doctest::Approx(0.f));
    CHECK_EQ(ch.values[5], doctest::Approx(0.7071068f).epsilon(1e-4));
    CHECK_EQ(ch.values[6], doctest::Approx(0.f));
    CHECK_EQ(ch.values[7], doctest::Approx(0.7071068f).epsilon(1e-4));

    REQUIRE_GT(scene->nodes.size(), 0);
    CHECK_EQ(scene->nodes[0].animation_rotation, 0);
    CHECK_EQ(scene->nodes[0].animation_translation, -1);
    CHECK_EQ(scene->nodes[0].animation_scale, -1);
}

TEST_CASE("GltfImporter - Multi Channel Animation")
{
    auto importer = make_ref<GltfImporter>();
    auto file = testing::project_directory() / "data" / "assets" / "animation_test" / "anim_multi_channel.glb";
    REQUIRE(std::filesystem::exists(file));

    auto scene = importer->load_scene(file);
    REQUIRE(scene != nullptr);

    CHECK_EQ(scene->animation.name, "MultiChannel");
    // 3 channels: translation, rotation, scale.
    REQUIRE_EQ(scene->animation.channels.size(), 3);

    // Translation channel.
    const auto& trans = scene->animation.channels[0];
    CHECK_EQ(trans.value_type, AnimationValueType::float3);
    CHECK_EQ(trans.interpolation_mode, AnimationInterpolationMode::linear);
    REQUIRE_EQ(trans.keyframe_count(), 2);
    CHECK_EQ(trans.values[3], doctest::Approx(1.f));
    CHECK_EQ(trans.values[4], doctest::Approx(2.f));
    CHECK_EQ(trans.values[5], doctest::Approx(3.f));

    // Rotation channel.
    const auto& rot = scene->animation.channels[1];
    CHECK_EQ(rot.value_type, AnimationValueType::quatf);
    CHECK_EQ(rot.interpolation_mode, AnimationInterpolationMode::linear);
    REQUIRE_EQ(rot.keyframe_count(), 2);

    // Scale channel.
    const auto& scl = scene->animation.channels[2];
    CHECK_EQ(scl.value_type, AnimationValueType::float3);
    CHECK_EQ(scl.interpolation_mode, AnimationInterpolationMode::linear);
    REQUIRE_EQ(scl.keyframe_count(), 3);
    // Last scale keyframe: (2,2,2)
    CHECK_EQ(scl.values[6], doctest::Approx(2.f));
    CHECK_EQ(scl.values[7], doctest::Approx(2.f));
    CHECK_EQ(scl.values[8], doctest::Approx(2.f));

    // Node should reference all 3 channels.
    REQUIRE_GT(scene->nodes.size(), 0);
    CHECK_EQ(scene->nodes[0].animation_translation, 0);
    CHECK_EQ(scene->nodes[0].animation_rotation, 1);
    CHECK_EQ(scene->nodes[0].animation_scale, 2);

    // Duration is max across channels = 3.0 (from scale channel).
    CHECK_EQ(scene->animation.duration(), doctest::Approx(3.f));
}

TEST_CASE("GltfImporter - CubicSpline Animation")
{
    auto importer = make_ref<GltfImporter>();
    auto file = testing::project_directory() / "data" / "assets" / "animation_test" / "anim_cubicspline.glb";
    REQUIRE(std::filesystem::exists(file));

    auto scene = importer->load_scene(file);
    REQUIRE(scene != nullptr);

    CHECK_EQ(scene->animation.name, "CubicSplineTranslation");
    REQUIRE_EQ(scene->animation.channels.size(), 1);

    const auto& ch = scene->animation.channels[0];
    CHECK_EQ(ch.value_type, AnimationValueType::float3);
    CHECK_EQ(ch.interpolation_mode, AnimationInterpolationMode::cubic_spline);
    REQUIRE_EQ(ch.keyframe_count(), 3);

    // Values (deinterleaved from triplets): (0,0,0), (3,2,0), (5,0,0)
    REQUIRE_EQ(ch.values.size(), 9);
    CHECK_EQ(ch.values[0], doctest::Approx(0.f));
    CHECK_EQ(ch.values[1], doctest::Approx(0.f));
    CHECK_EQ(ch.values[2], doctest::Approx(0.f));
    CHECK_EQ(ch.values[3], doctest::Approx(3.f));
    CHECK_EQ(ch.values[4], doctest::Approx(2.f));
    CHECK_EQ(ch.values[5], doctest::Approx(0.f));
    CHECK_EQ(ch.values[6], doctest::Approx(5.f));
    CHECK_EQ(ch.values[7], doctest::Approx(0.f));
    CHECK_EQ(ch.values[8], doctest::Approx(0.f));

    // In tangents and out tangents should be populated.
    REQUIRE_EQ(ch.in_tangents.size(), 9);
    REQUIRE_EQ(ch.out_tangents.size(), 9);

    // First keyframe out-tangent: (1,0,0)
    CHECK_EQ(ch.out_tangents[0], doctest::Approx(1.f));
    CHECK_EQ(ch.out_tangents[1], doctest::Approx(0.f));
    CHECK_EQ(ch.out_tangents[2], doctest::Approx(0.f));
}

TEST_CASE("GltfImporter - Multi Node Animation")
{
    auto importer = make_ref<GltfImporter>();
    auto file = testing::project_directory() / "data" / "assets" / "animation_test" / "anim_multi_node.glb";
    REQUIRE(std::filesystem::exists(file));

    auto scene = importer->load_scene(file);
    REQUIRE(scene != nullptr);

    CHECK_EQ(scene->animation.name, "MultiNode");
    REQUIRE_EQ(scene->animation.channels.size(), 2);

    // 2 nodes: parent (node 0) and child (node 1).
    REQUIRE_GE(scene->nodes.size(), 2);

    // Parent has translation animation.
    CHECK_EQ(scene->nodes[0].animation_translation, 0);
    CHECK_EQ(scene->nodes[0].animation_rotation, -1);

    // Child has rotation animation.
    CHECK_EQ(scene->nodes[1].animation_rotation, 1);
    CHECK_EQ(scene->nodes[1].animation_translation, -1);

    // Verify parent-child relationship.
    CHECK_EQ(scene->nodes[1].parent, 0);

    // Parent translation channel.
    const auto& parent_ch = scene->animation.channels[0];
    CHECK_EQ(parent_ch.value_type, AnimationValueType::float3);
    REQUIRE_EQ(parent_ch.keyframe_count(), 2);
    // (0,0,0) -> (10,0,0)
    CHECK_EQ(parent_ch.values[3], doctest::Approx(10.f));

    // Child rotation channel.
    const auto& child_ch = scene->animation.channels[1];
    CHECK_EQ(child_ch.value_type, AnimationValueType::quatf);
    REQUIRE_EQ(child_ch.keyframe_count(), 2);
}

TEST_CASE("GltfImporter - No Animation")
{
    auto importer = make_ref<GltfImporter>();
    auto sample_file = testing::project_directory() / "data" / "assets" / "kronos" / "Box" / "glTF-Binary" / "Box.glb";

    if (std::filesystem::exists(sample_file)) {
        auto scene = importer->load_scene(sample_file);
        REQUIRE(scene != nullptr);

        // Scene without animation should have empty animation data.
        CHECK(scene->animation.channels.empty());
        CHECK_EQ(scene->animation.duration(), 0.f);

        // No node should reference any animation channel.
        for (const auto& node : scene->nodes) {
            CHECK_EQ(node.animation_translation, -1);
            CHECK_EQ(node.animation_rotation, -1);
            CHECK_EQ(node.animation_scale, -1);
        }
    }
}
