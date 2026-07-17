// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "testing.h"
#include "falcor2/importers/importer.h"
#include "falcor2/importers/importer_edits.h"

#include <sgl/math/vector_math.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <string_view>

using namespace falcor;

namespace {

std::filesystem::path cornell_box_path()
{
    return testing::project_directory() / "data" / "assets" / "cornell-box" / "usdpreviewsurface" / "cornell-box.usda";
}

std::filesystem::path animation_test_path(std::string_view filename)
{
    return testing::project_directory() / "data" / "assets" / "animation_test" / std::filesystem::path(filename);
}

std::filesystem::path environment_map_path()
{
    return testing::project_directory() / "data" / "assets" / "envmaps" / "aerodynamics_workshop_512.hdr";
}

std::filesystem::path box_glb_path()
{
    return testing::project_directory() / "data" / "assets" / "kronos" / "Box" / "glTF-Binary" / "Box.glb";
}

void touch_file(const std::filesystem::path& path)
{
    std::filesystem::create_directories(path.parent_path());
    std::ofstream(path).put('\0');
}

class CurrentPathGuard {
public:
    explicit CurrentPathGuard(const std::filesystem::path& path)
        : m_original_path(std::filesystem::current_path())
    {
        std::filesystem::current_path(path);
    }

    ~CurrentPathGuard() { std::filesystem::current_path(m_original_path); }

private:
    std::filesystem::path m_original_path;
};

bool any_mesh_normal_differs(const ref<ImporterScene>& lhs, const ref<ImporterScene>& rhs)
{
    const size_t mesh_count = std::min(lhs->meshes.size(), rhs->meshes.size());
    for (size_t mesh_index = 0; mesh_index < mesh_count; ++mesh_index) {
        const ImporterMesh& lhs_mesh = lhs->meshes[mesh_index];
        const ImporterMesh& rhs_mesh = rhs->meshes[mesh_index];

        if (!lhs_mesh.normal_attribute().valid() || !rhs_mesh.normal_attribute().valid()) {
            continue;
        }

        auto lhs_normals = lhs_mesh.normal_stream();
        auto rhs_normals = rhs_mesh.normal_stream();
        const size_t vertex_count = std::min(lhs_mesh.vertex_count(), rhs_mesh.vertex_count());
        for (size_t vertex_index = 0; vertex_index < vertex_count; ++vertex_index) {
            if (sgl::math::length(lhs_normals[vertex_index] - rhs_normals[vertex_index]) > 1e-4f) {
                return true;
            }
        }
    }

    return false;
}

void check_all_meshes_have_recomputed_frames(const ref<ImporterScene>& scene)
{
    for (const ImporterMesh& mesh : scene->meshes) {
        if (mesh.vertex_count() == 0) {
            continue;
        }

        CHECK(mesh.normal_attribute().valid());
        CHECK(mesh.tangent_attribute().valid());
        CHECK(mesh.handedness_attribute().valid());
    }
}

} // namespace

TEST_CASE("Importer edit stream - empty build creates empty importer scene")
{
    ref<Importer> importer = Importer::create();
    ref<ImporterScene> scene = importer->build_importer_scene();

    REQUIRE(scene);
    CHECK(scene->cameras.empty());
    CHECK(scene->nodes.empty());
    CHECK(scene->root_nodes.empty());
}

TEST_CASE("Importer edit stream - set environment creates dome light node")
{
    ref<Importer> importer = Importer::create();
    importer->env().set(environment_map_path(), 5.f, "TestEnv");

    ref<ImporterScene> scene = importer->build_importer_scene();

    REQUIRE(scene);
    REQUIRE_EQ(scene->lights.size(), 1);
    REQUIRE_EQ(scene->nodes.size(), 1);
    REQUIRE_EQ(scene->root_nodes.size(), 1);

    const ImporterLight& light = scene->lights[0];
    CHECK_EQ(light.name, "TestEnv");
    CHECK_EQ(light.type, ImporterLight::Type::dome);
    CHECK_EQ(light.env_map_path, std::filesystem::canonical(environment_map_path()).string());
    CHECK_EQ(light.exposure, 5.f);

    const ImporterNode& node = scene->nodes[0];
    CHECK_EQ(node.name, "TestEnv");
    CHECK_EQ(node.light_index, 0);
    CHECK_EQ(node.parent, -1);
    CHECK_EQ(scene->root_nodes[0], 0);
}

TEST_CASE("Importer edit stream - set environment overwrites first dome light")
{
    const auto root = testing::get_case_temp_directory();
    const auto first_path = root / "first.hdr";
    const auto second_path = root / "second.hdr";
    touch_file(first_path);
    touch_file(second_path);

    ref<Importer> importer = Importer::create();
    importer->env().set(first_path, 1.f, "FirstEnv");
    importer->env().set(second_path, 5.f, "SecondEnv");

    ref<ImporterScene> scene = importer->build_importer_scene();

    REQUIRE(scene);
    REQUIRE_EQ(scene->lights.size(), 1);
    REQUIRE_EQ(scene->nodes.size(), 1);
    REQUIRE_EQ(scene->root_nodes.size(), 1);
    CHECK_EQ(scene->lights[0].name, "SecondEnv");
    CHECK_EQ(scene->lights[0].type, ImporterLight::Type::dome);
    CHECK_EQ(scene->lights[0].env_map_path, std::filesystem::canonical(second_path).string());
    CHECK_EQ(scene->lights[0].exposure, 5.f);
    CHECK_EQ(scene->nodes[0].name, "SecondEnv");
    CHECK_EQ(scene->nodes[0].light_index, 0);
}

TEST_CASE("Importer edit stream - set environment overwrites imported dome light")
{
    const auto env_path = testing::get_case_temp_directory() / "new.hdr";
    touch_file(env_path);

    ImporterBuildContext context;
    context.scene = make_ref<ImporterScene>();

    ImporterLight old_light;
    old_light.name = "OldEnv";
    old_light.type = ImporterLight::Type::dome;
    old_light.env_map_path = "old.hdr";
    old_light.exposure = -1.f;
    context.scene->lights.push_back(old_light);

    ImporterNode old_node;
    old_node.name = "OldEnv";
    old_node.light_index = 0;
    context.scene->nodes.push_back(old_node);
    context.scene->root_nodes.push_back(0);

    SetEnvironmentEdit edit(env_path, 3.f, "NewEnv");
    edit.apply(context);

    REQUIRE_EQ(context.scene->lights.size(), 1);
    REQUIRE_EQ(context.scene->nodes.size(), 1);
    REQUIRE_EQ(context.scene->root_nodes.size(), 1);
    CHECK_EQ(context.scene->lights[0].name, "NewEnv");
    CHECK_EQ(context.scene->lights[0].type, ImporterLight::Type::dome);
    CHECK_EQ(context.scene->lights[0].env_map_path, std::filesystem::canonical(env_path).string());
    CHECK_EQ(context.scene->lights[0].exposure, 3.f);
    CHECK_EQ(context.scene->nodes[0].name, "NewEnv");
    CHECK_EQ(context.scene->nodes[0].light_index, 0);
}

TEST_CASE("Importer edit stream - source path is normalized when assigned")
{
    const auto root = testing::get_case_temp_directory();
    const auto other = root / "other";
    std::filesystem::create_directories(other);

    CurrentPathGuard guard(root);
    ref<Importer> importer = Importer::create();
    importer->set_source_path("scenes/test.py");
    const auto expected = (root / "scenes/test.py").lexically_normal();
    CHECK(importer->source_path() == expected);

    std::filesystem::current_path(other);
    CHECK(importer->source_path() == expected);
    importer->set_source_path({});
    CHECK(importer->source_path().empty());
}

TEST_CASE("Importer edit stream - source and explicit roots have deterministic priority")
{
    const auto root = testing::get_case_temp_directory();
    const auto source = root / "source";
    const auto first = root / "first";
    const auto second = root / "second";
    touch_file(source / "environment.hdr");
    touch_file(first / "environment.hdr");
    touch_file(second / "environment.hdr");

    ref<Importer> source_importer = Importer::create();
    source_importer->set_source_path(source / "scene.py");
    const std::vector<std::filesystem::path> roots = {first, second};
    source_importer->add_search_paths(roots);
    source_importer->env().set("environment.hdr");
    auto source_scene = source_importer->build_importer_scene();
    CHECK(source_scene->lights[0].env_map_path == std::filesystem::canonical(source / "environment.hdr").string());

    std::filesystem::remove(source / "environment.hdr");
    auto explicit_scene = source_importer->build_importer_scene();
    CHECK(explicit_scene->lights[0].env_map_path == std::filesystem::canonical(first / "environment.hdr").string());
}

TEST_CASE("Importer edit stream - replacing source path changes source-relative resolution")
{
    const auto root = testing::get_case_temp_directory();
    const auto first = root / "first";
    const auto second = root / "second";
    touch_file(first / "environment.hdr");
    touch_file(second / "environment.hdr");

    ref<Importer> importer = Importer::create();
    importer->set_source_path(first / "scene.py");
    importer->set_source_path(second / "scene.py");
    importer->env().set("environment.hdr");

    auto scene = importer->build_importer_scene();
    CHECK(scene->lights[0].env_map_path == std::filesystem::canonical(second / "environment.hdr").string());
}

TEST_CASE("Importer edit stream - relative explicit roots use the source directory")
{
    const auto source = testing::get_case_temp_directory() / "source";
    touch_file(source / "assets/environment.hdr");

    ref<Importer> importer = Importer::create();
    importer->set_source_path(source / "scene.py");
    importer->add_search_path("assets");
    importer->env().set("environment.hdr");

    auto scene = importer->build_importer_scene();
    CHECK(scene->lights[0].env_map_path == std::filesystem::canonical(source / "assets/environment.hdr").string());
}

TEST_CASE("Importer edit stream - unset source falls back to the build working directory")
{
    const auto working = testing::get_case_temp_directory() / "working";
    touch_file(working / "environment.hdr");
    CurrentPathGuard guard(working);

    ref<Importer> importer = Importer::create();
    importer->env().set("environment.hdr");

    auto scene = importer->build_importer_scene();
    CHECK(scene->lights[0].env_map_path == std::filesystem::canonical(working / "environment.hdr").string());
}

TEST_CASE("Importer edit stream - source-relative assets resolve during each build")
{
    const auto source = testing::get_case_temp_directory() / "source";
    std::filesystem::create_directories(source);
    std::filesystem::copy_file(box_glb_path(), source / "Box.glb");

    ref<Importer> importer = Importer::create();
    importer->set_source_path(source / "scene.py");
    importer->import_asset("Box.glb");

    auto first = importer->build_importer_scene();
    auto second = importer->build_importer_scene();
    CHECK_FALSE(first->meshes.empty());
    CHECK(first->meshes.size() == second->meshes.size());
}

TEST_CASE("Importer edit stream - missing assets report source and searched roots")
{
    const auto root = testing::get_case_temp_directory();
    ref<Importer> importer = Importer::create();
    importer->set_source_path(root / "source/scene.py");
    importer->add_search_path(root / "additional");
    importer->env().set("missing.hdr");

    try {
        importer->build_importer_scene();
        FAIL("Expected missing environment map to fail during build");
    } catch (const std::exception& error) {
        const std::string message = error.what();
        CHECK(message.find("environment map") != std::string::npos);
        CHECK(message.find("missing.hdr") != std::string::npos);
        CHECK(message.find("source") != std::string::npos);
        CHECK(message.find("additional") != std::string::npos);
        CHECK(message.find("AssetResolver(search_paths=[") != std::string::npos);
    }
}

TEST_CASE("Importer edit stream - camera selector resolves into camera node")
{
    ref<Importer> importer = Importer::create();

    ImporterCameraSelector camera_selector = importer->cameras().create(
        "Camera",
        2.f,
        35.f,
        4.f,
        float2(0.1f, 500.f),
        ImporterCamera::Projection::orthographic,
        ImporterCamera::FOVDirection::horizontal,
        36.f
    );
    ImporterNodeSelector node_selector = importer->nodes().create("CameraNode", float4x4::identity(), camera_selector);

    CHECK(camera_selector.id != 0);
    CHECK(node_selector.id != 0);
    CHECK(camera_selector.id != node_selector.id);

    ref<ImporterScene> scene = importer->build_importer_scene();

    REQUIRE(scene);
    REQUIRE_EQ(scene->cameras.size(), 1);
    REQUIRE_EQ(scene->nodes.size(), 1);
    REQUIRE_EQ(scene->root_nodes.size(), 1);

    const ImporterCamera& camera = scene->cameras[0];
    CHECK_EQ(camera.name, "Camera");
    CHECK_EQ(camera.focus_distance, 2.f);
    CHECK_EQ(camera.focal_length, 35.f);
    CHECK_EQ(camera.fstop, 4.f);
    CHECK_EQ(camera.depth_range, float2(0.1f, 500.f));
    CHECK_EQ(camera.projection, ImporterCamera::Projection::orthographic);
    CHECK_EQ(camera.fov_direction, ImporterCamera::FOVDirection::horizontal);
    CHECK_EQ(camera.sensor_size_mm, 36.f);

    const ImporterNode& node = scene->nodes[0];
    CHECK_EQ(node.name, "CameraNode");
    CHECK_EQ(node.camera_index, 0);
    CHECK_EQ(node.parent, -1);
    CHECK(node.children.empty());
    CHECK_EQ(scene->root_nodes[0], 0);

    ref<ImporterScene> rebuilt_scene = importer->build_importer_scene();
    REQUIRE(rebuilt_scene);
    CHECK(rebuilt_scene.get() != scene.get());
    REQUIRE_EQ(rebuilt_scene->cameras.size(), 1);
    REQUIRE_EQ(rebuilt_scene->nodes.size(), 1);
    CHECK_EQ(rebuilt_scene->nodes[0].camera_index, 0);
}

TEST_CASE("Importer edit stream - camera create_fov derives focal length")
{
    ref<Importer> importer = Importer::create();

    ImporterCameraSelector camera_selector = importer->cameras().create_fov(
        "FovCamera",
        45.f,
        3.f,
        5.6f,
        float2(0.25f, 250.f),
        ImporterCamera::Projection::perspective,
        ImporterCamera::FOVDirection::vertical,
        24.f
    );
    importer->nodes().create("FovCameraNode", float4x4::identity(), camera_selector);

    ref<ImporterScene> scene = importer->build_importer_scene();
    REQUIRE(scene);
    REQUIRE_EQ(scene->cameras.size(), 1);

    const ImporterCamera& camera = scene->cameras[0];
    CHECK_EQ(camera.name, "FovCamera");
    CHECK_EQ(camera.focus_distance, 3.f);
    CHECK_EQ(camera.fstop, 5.6f);
    CHECK_EQ(camera.depth_range, float2(0.25f, 250.f));
    CHECK_EQ(camera.fov_direction, ImporterCamera::FOVDirection::vertical);
    CHECK_EQ(camera.sensor_size_mm, 24.f);
    CHECK(camera.focal_length == doctest::Approx(ImporterCamera::focal_length_from_fov_degrees(45.f, 24.f)));
}

TEST_CASE("Importer edit stream - node create_camera creates camera and node")
{
    ref<Importer> importer = Importer::create();
    const float4x4 transform = sgl::math::matrix_from_translation(float3(1.f, 2.f, 3.f));

    ImporterNodeSelector node_selector = importer->nodes().create_camera(
        "View",
        transform,
        2.f,
        35.f,
        4.f,
        float2(0.1f, 500.f),
        ImporterCamera::Projection::orthographic,
        ImporterCamera::FOVDirection::horizontal,
        36.f
    );

    CHECK(node_selector.id != 0);
    ref<ImporterScene> scene = importer->build_importer_scene();
    REQUIRE_EQ(scene->cameras.size(), 1);
    REQUIRE_EQ(scene->nodes.size(), 1);
    CHECK_EQ(scene->cameras[0].name, "View");
    CHECK_EQ(scene->cameras[0].focal_length, 35.f);
    CHECK_EQ(scene->nodes[0].name, "View");
    CHECK_EQ(scene->nodes[0].camera_index, 0);
    CHECK_EQ(scene->nodes[0].transform, transform);
}

TEST_CASE("Importer edit stream - node create_camera_fov creates camera and node")
{
    ref<Importer> importer = Importer::create();

    ImporterNodeSelector node_selector = importer->nodes().create_camera_fov(
        "Camera",
        float4x4::identity(),
        45.f,
        1.f,
        8.f,
        float2(0.01f, 10000.f),
        ImporterCamera::Projection::perspective,
        ImporterCamera::FOVDirection::vertical,
        24.f
    );

    CHECK(node_selector.id != 0);
    ref<ImporterScene> scene = importer->build_importer_scene();
    REQUIRE_EQ(scene->cameras.size(), 1);
    REQUIRE_EQ(scene->nodes.size(), 1);
    CHECK_EQ(scene->cameras[0].name, "Camera");
    CHECK(scene->cameras[0].focal_length == doctest::Approx(ImporterCamera::focal_length_from_fov_degrees(45.f, 24.f)));
    CHECK_EQ(scene->nodes[0].name, "Camera");
    CHECK_EQ(scene->nodes[0].camera_index, 0);
}

TEST_CASE("Importer edit stream - node selectors resolve parent child hierarchy")
{
    ref<Importer> importer = Importer::create();

    ImporterNodeSelector parent_selector = importer->nodes().create("Parent");
    ImporterNodeSelector child_selector = importer->nodes().create("Child", float4x4::identity(), {}, parent_selector);

    CHECK(parent_selector.id != 0);
    CHECK(child_selector.id != 0);
    CHECK(parent_selector.id != child_selector.id);

    ref<ImporterScene> scene = importer->build_importer_scene();

    REQUIRE(scene);
    REQUIRE_EQ(scene->cameras.size(), 0);
    REQUIRE_EQ(scene->nodes.size(), 2);
    REQUIRE_EQ(scene->root_nodes.size(), 1);

    CHECK_EQ(scene->root_nodes[0], 0);
    CHECK_EQ(scene->nodes[0].name, "Parent");
    CHECK_EQ(scene->nodes[0].parent, -1);
    REQUIRE_EQ(scene->nodes[0].children.size(), 1);
    CHECK_EQ(scene->nodes[0].children[0], 1);

    CHECK_EQ(scene->nodes[1].name, "Child");
    CHECK_EQ(scene->nodes[1].parent, 0);
    CHECK_EQ(scene->nodes[1].camera_index, -1);
    CHECK(scene->nodes[1].children.empty());
}

TEST_CASE("Importer edit stream - unresolved selector fails during build")
{
    ref<Importer> importer = Importer::create();

    ImporterCameraSelector missing_camera;
    missing_camera.id = 999;
    importer->nodes().create("CameraNode", float4x4::identity(), missing_camera);

    CHECK_THROWS(importer->build_importer_scene());
}

TEST_CASE("Importer edit stream - current importer auto creates and scoped restore works")
{
    Importer::clear_current();

    ref<Importer> initial = Importer::get();
    REQUIRE(initial);
    CHECK_EQ(Importer::get().get(), initial.get());

    ref<Importer> scoped = Importer::create();
    {
        ScopedCurrentImporter current(scoped);
        CHECK_EQ(Importer::get().get(), scoped.get());
    }

    CHECK_EQ(Importer::get().get(), initial.get());

    Importer::clear_current();
    ref<Importer> after_clear = Importer::get();
    REQUIRE(after_clear);
    CHECK(after_clear.get() != initial.get());

    Importer::clear_current();
}

TEST_CASE("Importer edit stream - import_asset defers loading until build")
{
    ref<Importer> importer = Importer::create();

    importer->import_asset(testing::project_directory() / "data" / "assets" / "missing-scene.usda");

    CHECK_THROWS(importer->build_importer_scene());
}

TEST_CASE("Importer edit stream - import_asset imports cornell box at build time")
{
    const std::filesystem::path cornell_box = cornell_box_path();
    REQUIRE(std::filesystem::exists(cornell_box));

    ImportOptions options;
    options.recompute_normals = true;
    ref<Importer> importer = Importer::create(options);
    importer->import_asset(cornell_box);

    ref<ImporterScene> direct_scene = import_scene(cornell_box);
    REQUIRE(direct_scene);

    ref<ImporterScene> scene = importer->build_importer_scene();

    REQUIRE(scene);
    CHECK_EQ(scene->materials.size(), direct_scene->materials.size());
    CHECK_EQ(scene->textures.size(), direct_scene->textures.size());
    CHECK_EQ(scene->meshes.size(), direct_scene->meshes.size());
    CHECK_EQ(scene->curves.size(), direct_scene->curves.size());
    CHECK_EQ(scene->nodes.size(), direct_scene->nodes.size());
    CHECK_EQ(scene->cameras.size(), direct_scene->cameras.size());
    CHECK_EQ(scene->lights.size(), direct_scene->lights.size());
    CHECK_EQ(scene->prototypes.size(), direct_scene->prototypes.size());
    CHECK_EQ(scene->root_nodes.size(), direct_scene->root_nodes.size());
    CHECK_GT(scene->meshes.size(), 0);
    CHECK_GT(scene->nodes.size(), 0);
    CHECK_GT(scene->root_nodes.size(), 0);
    check_all_meshes_have_recomputed_frames(scene);
}

TEST_CASE("Importer edit stream - material selector replaces every match")
{
    ImporterBuildContext context;
    context.scene = make_ref<ImporterScene>();

    ImporterMaterial first;
    first.name = "Shared";
    first.params.set("old", 1);
    first.constructor = [](Scene&, const ImporterMaterial&) -> Material*
    {
        return nullptr;
    };
    first.output_to_material_network["surface"].push_back(Properties());
    context.scene->materials.push_back(std::move(first));

    ImporterMaterial untouched;
    untouched.name = "Other";
    untouched.params.set("old", 2);
    context.scene->materials.push_back(std::move(untouched));

    ImporterMaterial second;
    second.name = "Shared";
    second.params.set("old", 3);
    second.constructor = [](Scene&, const ImporterMaterial&) -> Material*
    {
        return nullptr;
    };
    second.output_to_material_network["surface"].push_back(Properties());
    context.scene->materials.push_back(std::move(second));

    Properties props;
    props.set("replacement", 7);
    ReplaceMaterialEdit edit("Shared", "MaterialXMaterial", props);
    edit.apply(context);

    for (size_t index : {size_t(0), size_t(2)}) {
        const ImporterMaterial& material = context.scene->materials[index];
        CHECK_EQ(material.name, "Shared");
        CHECK_EQ(material.params.get<std::string>("_scene_material_type"), "MaterialXMaterial");
        CHECK_EQ(material.params.get<int>("replacement"), 7);
        CHECK_FALSE(material.params.has_property("old"));
        CHECK_FALSE(static_cast<bool>(material.constructor));
        CHECK(material.output_to_material_network.empty());
    }
    CHECK_EQ(context.scene->materials[1].params.get<int>("old"), 2);
    CHECK_FALSE(context.scene->materials[1].params.has_property("_scene_material_type"));
}

TEST_CASE("Importer edit stream - empty material selection is a no-op")
{
    ref<Importer> importer = Importer::create();
    importer->materials().find("Missing").replace("StandardMaterial");

    ref<ImporterScene> scene = importer->build_importer_scene();
    CHECK(scene->materials.empty());
}

TEST_CASE("Importer edit stream - material selector validates input")
{
    ref<Importer> importer = Importer::create();
    CHECK_THROWS(importer->materials().find(""));
    CHECK_THROWS(importer->materials().find("Material").replace(""));
    CHECK_THROWS(importer->materials().find("Material").replace(ImporterMaterial::Constructor{}));
}

TEST_CASE("Importer edit stream - import_asset preserves imported animation name")
{
    const std::filesystem::path animated_asset = animation_test_path("anim_linear_translation.glb");
    REQUIRE(std::filesystem::exists(animated_asset));

    ref<ImporterScene> direct_scene = import_scene(animated_asset);
    REQUIRE(direct_scene);
    REQUIRE_EQ(direct_scene->animation.name, "TranslateX");
    REQUIRE_EQ(direct_scene->animation.channels.size(), 1);

    ref<Importer> importer = Importer::create();
    importer->import_asset(animated_asset);

    ref<ImporterScene> scene = importer->build_importer_scene();
    REQUIRE(scene);

    CHECK_EQ(scene->animation.name, direct_scene->animation.name);
    REQUIRE_EQ(scene->animation.channels.size(), direct_scene->animation.channels.size());
    REQUIRE_FALSE(scene->nodes.empty());
    CHECK_EQ(scene->nodes[0].animation_translation, 0);
}

TEST_CASE("Importer edit stream - imported content offsets before created camera node")
{
    const std::filesystem::path cornell_box = cornell_box_path();
    REQUIRE(std::filesystem::exists(cornell_box));

    ref<ImporterScene> direct_scene = import_scene(cornell_box);
    REQUIRE(direct_scene);

    ref<Importer> importer = Importer::create();
    importer->import_asset(cornell_box);
    ImporterCameraSelector camera_selector = importer->cameras().create("ExtraCamera");
    ImporterNodeSelector node_selector
        = importer->nodes().create("ExtraCameraNode", float4x4::identity(), camera_selector);

    CHECK(camera_selector.id != 0);
    CHECK(node_selector.id != 0);
    CHECK(camera_selector.id != node_selector.id);

    ref<ImporterScene> scene = importer->build_importer_scene();
    REQUIRE(scene);

    const int expected_camera_index = static_cast<int>(direct_scene->cameras.size());
    const int expected_node_index = static_cast<int>(direct_scene->nodes.size());

    REQUIRE_EQ(scene->materials.size(), direct_scene->materials.size());
    REQUIRE_EQ(scene->meshes.size(), direct_scene->meshes.size());
    REQUIRE_EQ(scene->cameras.size(), direct_scene->cameras.size() + 1);
    REQUIRE_EQ(scene->nodes.size(), direct_scene->nodes.size() + 1);
    REQUIRE_EQ(scene->root_nodes.size(), direct_scene->root_nodes.size() + 1);

    const ImporterCamera& extra_camera = scene->cameras[expected_camera_index];
    CHECK_EQ(extra_camera.name, "ExtraCamera");

    const ImporterNode& extra_node = scene->nodes[expected_node_index];
    CHECK_EQ(extra_node.name, "ExtraCameraNode");
    CHECK_EQ(extra_node.camera_index, expected_camera_index);
    CHECK_EQ(extra_node.parent, -1);
    CHECK(
        std::find(scene->root_nodes.begin(), scene->root_nodes.end(), expected_node_index) != scene->root_nodes.end()
    );
}
