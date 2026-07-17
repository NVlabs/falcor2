// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "testing.h"

#include "falcor2/render/scene.h"
#include "falcor2/render/scene_import.h"
#include "falcor2/render/ray_tracing_setup.h"
#include "falcor2/render/geometry/static_mesh_geometry.h"
#include "falcor2/render/geometry/geometry_group.h"
#include "falcor2/render/component/geometry_instance.h"
#include "falcor2/render/component/camera.h"
#include "falcor2/render/component/light.h"
#include "falcor2/render/material/standard_material.h"
#include "falcor2/importers/importer_types.h"
#include "falcor2/core/python_interpreter.h"

#include <sgl/math/vector.h>

#include <array>

#include <filesystem>

using namespace falcor;

namespace {

class TestSceneGlobals : public SceneGlobals {
public:
    explicit TestSceneGlobals(uint32_t key)
        : m_key(key)
    {
    }

    uint32_t key() const { return m_key; }

    void bind(sgl::ShaderCursor) const override { }

private:
    uint32_t m_key;
};

} // namespace

// ---------------------------------------------------------------------------
// 1. Object add/remove/compact cycle
// ---------------------------------------------------------------------------

TEST_SUITE_BEGIN("Scene");

TEST_CASE_GPU("ray tracing setup pads three ray type slots")
{
    auto scene = Scene::create(ref(ctx.device));

    SceneRayTracingSetup::RayDesc primary{
        .name = "primary",
        .has_miss = true,
        .has_closest_hit = true,
    };
    SceneRayTracingSetup::RayDesc secondary{
        .name = "secondary",
        .has_miss = true,
        .has_closest_hit = true,
    };
    SceneRayTracingSetup::Options options{.skip_unused_geometry_types = false};
    SceneRayTracingSetup setup = SceneRayTracingSetup::create(scene.get(), {primary, secondary}, options);

    CHECK_EQ(setup.sbt_miss_entry_points.size(), 3);
    CHECK_EQ(setup.sbt_miss_entry_points[0], "_scene_primary_miss");
    CHECK_EQ(setup.sbt_miss_entry_points[1], "_scene_secondary_miss");
    CHECK(setup.sbt_miss_entry_points[2].empty());

    REQUIRE_EQ(setup.sbt_hit_group_names.size(), 6);
    CHECK_EQ(setup.sbt_hit_group_names[0], "_scene_primary_triangle_hit_group");
    CHECK_EQ(setup.sbt_hit_group_names[1], "_scene_secondary_triangle_hit_group");
    CHECK_EQ(setup.sbt_hit_group_names[2], "__dummy_hit_group");
    CHECK_EQ(setup.sbt_hit_group_names[3], "_scene_primary_lss_hit_group");
    CHECK_EQ(setup.sbt_hit_group_names[4], "_scene_secondary_lss_hit_group");
    CHECK_EQ(setup.sbt_hit_group_names[5], "__dummy_hit_group");
}

TEST_CASE_GPU("ray tracing setup rejects more than three ray types")
{
    auto scene = Scene::create(ref(ctx.device));
    std::array<SceneRayTracingSetup::RayDesc, 4> ray_descs;
    CHECK_THROWS(SceneRayTracingSetup::create(scene.get(), ray_descs));
}

TEST_CASE_GPU("named refcounted scene globals garbage collection")
{
    auto scene = Scene::create(ref(ctx.device));
    uint32_t factory_calls = 0;
    auto get_scene_globals = [&](const char* name, uint32_t key) -> ref<SceneGlobals>
    {
        return scene->get_or_create_scene_globals(
            name,
            [&]() -> ref<SceneGlobals>
            {
                ++factory_calls;
                return make_ref<TestSceneGlobals>(key);
            }
        );
    };

    {
        ref<SceneGlobals> material_globals = get_scene_globals("test_scene_globals_1", 1);
        SceneGlobals* raw_globals = material_globals.get();
        ref<SceneGlobals> external_globals = scene->scene_globals()[0];

        REQUIRE_EQ(scene->scene_globals().size(), 1);
        CHECK_EQ(scene->scene_globals()[0].get(), raw_globals);
        CHECK_EQ(raw_globals->ref_count(), 3);
        CHECK_EQ(factory_calls, 1);

        material_globals.reset();
        CHECK_EQ(scene->scene_globals()[0].get(), raw_globals);
        CHECK_EQ(raw_globals->ref_count(), 2);
        scene->update();
        REQUIRE_EQ(scene->scene_globals().size(), 1);
        CHECK_EQ(scene->scene_globals()[0].get(), raw_globals);

        external_globals.reset();
        CHECK_EQ(scene->scene_globals()[0].get(), raw_globals);
        CHECK_EQ(raw_globals->ref_count(), 1);
        scene->update();
        CHECK_EQ(scene->scene_globals().size(), 0);
    }

    {
        ref<SceneGlobals> material_globals = get_scene_globals("test_scene_globals_recreated", 1);
        CHECK_EQ(factory_calls, 2);

        material_globals.reset();
        scene->update();
        CHECK_EQ(scene->scene_globals().size(), 0);

        ref<SceneGlobals> new_material_globals = get_scene_globals("test_scene_globals_recreated", 7);
        CHECK_EQ(static_cast<TestSceneGlobals*>(new_material_globals.get())->key(), 7);
        CHECK_EQ(factory_calls, 3);
        CHECK_EQ(scene->scene_globals().size(), 1);

        new_material_globals.reset();
        scene->update();
        CHECK_EQ(scene->scene_globals().size(), 0);
    }

    ref<SceneGlobals> material_globals = get_scene_globals("test_scene_globals_shared", 2);
    SceneGlobals* raw_globals = material_globals.get();
    {
        ref<SceneGlobals> other_material_globals = get_scene_globals("test_scene_globals_shared", 99);
        CHECK_EQ(other_material_globals.get(), raw_globals);
        CHECK_EQ(static_cast<TestSceneGlobals*>(other_material_globals.get())->key(), 2);
        CHECK_EQ(raw_globals->ref_count(), 3);
        CHECK_EQ(factory_calls, 4);

        other_material_globals.reset();
        scene->update();
        REQUIRE_EQ(scene->scene_globals().size(), 1);
        CHECK_EQ(scene->scene_globals()[0].get(), raw_globals);
        CHECK_EQ(raw_globals->ref_count(), 2);
    }

    ref<SceneGlobals> other_globals = get_scene_globals("test_scene_globals_other", 3);
    CHECK_NE(other_globals.get(), raw_globals);
    CHECK_EQ(scene->scene_globals().size(), 2);

    material_globals.reset();
    scene->update();
    REQUIRE_EQ(scene->scene_globals().size(), 1);
    CHECK_EQ(scene->scene_globals()[0].get(), other_globals.get());

    other_globals.reset();
    scene->update();
    CHECK_EQ(scene->scene_globals().size(), 0);
}

TEST_CASE_GPU("importer scene uv origin conversion and adoption")
{
    auto importer_scene = make_ref<ImporterScene>();
    importer_scene->uv_origin = UVOrigin::lower_left;
    importer_scene->materials.emplace_back();
    importer_scene->materials.back().name = "uv_origin_test_material";

    ImporterMesh mesh;
    mesh.name = "uv_origin_test_mesh";
    mesh.uv_origin = UVOrigin::lower_left;
    mesh.ensure_attributes({
        {ImporterSemantic::position},
        {ImporterSemantic::normal},
        {ImporterSemantic::tangent},
        {ImporterSemantic::handedness},
        {ImporterSemantic::tex_coord},
    });
    mesh.allocate_vertices(3, true);

    auto positions = mesh.position_stream();
    positions[0] = float3(-1.f, 0.f, 0.f);
    positions[1] = float3(1.f, 0.f, 0.f);
    positions[2] = float3(0.f, 1.f, 0.f);

    auto normals = mesh.normal_stream();
    auto tangents = mesh.tangent_stream();
    auto handedness = mesh.handedness_stream();
    for (size_t i = 0; i < mesh.vertex_count(); ++i) {
        normals[i] = float3(0.f, 0.f, 1.f);
        tangents[i] = float3(1.f, 0.f, 0.f);
        handedness[i] = 1.f;
    }

    auto texcoords = mesh.texcoord_stream();
    texcoords[0] = float2(0.25f, -0.25f);
    texcoords[1] = float2(0.5f, 0.5f);
    texcoords[2] = float2(0.75f, 1.25f);

    mesh.subgeometries.push_back(
        ImporterMesh::Subgeometry{
            .name = "uv_origin_test_submesh",
            .indices = {uint3(0, 1, 2)},
            .material_name = "uv_origin_test_material",
        }
    );
    mesh.calculate_local_aabb();
    importer_scene->meshes.push_back(mesh);

    auto check_imported_uvs = [](Scene* scene, float2 uv0, float2 uv1, float2 uv2)
    {
        REQUIRE_EQ(scene->geometries().size(), 1);
        auto* geometry = dynamic_cast<StaticMeshGeometry*>(scene->geometries()[0]);
        REQUIRE(geometry != nullptr);
        REQUIRE_EQ(geometry->sub_mesh_count(), 1);
        REQUIRE_EQ(geometry->vertex_count(0), 3);

        float2 expected_uvs[] = {uv0, uv1, uv2};
        for (size_t i = 0; i < 3; ++i) {
            auto actual_uv = shared::detail::unpack_triangle_vertex(geometry->vertices(0)[i]).uv[0];
            CHECK(actual_uv.x == doctest::Approx(expected_uvs[i].x));
            CHECK(actual_uv.y == doctest::Approx(expected_uvs[i].y));
        }
    };

    {
        auto scene = Scene::create(ref(ctx.device));
        load_importer_scene(scene.get(), *importer_scene);

        CHECK(scene->options().uv_origin == UVOrigin::upper_left);
        check_imported_uvs(scene.get(), float2(0.25f, 1.25f), float2(0.5f, 0.5f), float2(0.75f, -0.25f));
    }

    {
        auto scene = Scene::create(ref(ctx.device), *importer_scene);
        CHECK(scene->options().uv_origin == UVOrigin::lower_left);
        check_imported_uvs(scene.get(), float2(0.25f, -0.25f), float2(0.5f, 0.5f), float2(0.75f, 1.25f));
    }

    {
        auto scene = Scene::create(ref(ctx.device), *importer_scene);
        CHECK(scene->options().uv_origin == UVOrigin::lower_left);
        check_imported_uvs(scene.get(), float2(0.25f, -0.25f), float2(0.5f, 0.5f), float2(0.75f, 1.25f));
    }

    {
        auto scene = Scene::create(ref(ctx.device), *importer_scene, UVOrigin::upper_left);
        CHECK(scene->options().uv_origin == UVOrigin::upper_left);
        check_imported_uvs(scene.get(), float2(0.25f, 1.25f), float2(0.5f, 0.5f), float2(0.75f, -0.25f));
    }
}

TEST_CASE_GPU("importer scene dome light preserves environment map path")
{
    auto importer_scene = make_ref<ImporterScene>();

    ImporterLight dome_light;
    dome_light.name = "Test Dome Light";
    dome_light.type = ImporterLight::Type::dome;
    dome_light.env_map_path = "data/assets/envmaps/aerodynamics_workshop_512.hdr";
    dome_light.exposure = 5.f;
    importer_scene->lights.push_back(dome_light);

    ImporterNode dome_node;
    dome_node.name = "Test Dome Light";
    dome_node.light_index = 0;
    importer_scene->nodes.push_back(dome_node);
    importer_scene->root_nodes.push_back(0);

    auto scene = Scene::create(ref(ctx.device));
    load_importer_scene(scene.get(), *importer_scene);

    REQUIRE_EQ(scene->components().size(), 1);
    const auto* env_map_light = dynamic_cast<const EnvMapLight*>(scene->components()[0]);
    REQUIRE(env_map_light != nullptr);
    CHECK_EQ(env_map_light->env_map_path(), std::filesystem::path(dome_light.env_map_path));
    CHECK_EQ(env_map_light->exposure(), doctest::Approx(dome_light.exposure));
}

TEST_CASE_GPU("importer scene textureless dome light becomes constant environment")
{
    auto importer_scene = make_ref<ImporterScene>();

    ImporterLight dome_light;
    dome_light.name = "Textureless Dome Light";
    dome_light.type = ImporterLight::Type::dome;
    dome_light.intensity = float3(0.25f, 0.5f, 0.75f);
    dome_light.exposure = 2.f;
    importer_scene->lights.push_back(dome_light);

    ImporterNode dome_node;
    dome_node.name = dome_light.name;
    dome_node.light_index = 0;
    importer_scene->nodes.push_back(dome_node);
    importer_scene->root_nodes.push_back(0);

    auto scene = Scene::create(ref(ctx.device));
    load_importer_scene(scene.get(), *importer_scene);

    REQUIRE_EQ(scene->components().size(), 1);
    const auto* constant_light = dynamic_cast<const ConstantLight*>(scene->components()[0]);
    REQUIRE(constant_light != nullptr);
    CHECK_EQ(constant_light->radiance(), dome_light.intensity);
    CHECK_EQ(constant_light->exposure(), doctest::Approx(dome_light.exposure));
}

TEST_CASE_GPU("importer scene converts distant angular diameter to cone half angle")
{
    auto importer_scene = make_ref<ImporterScene>();

    ImporterLight importer_light;
    importer_light.name = "Distant Light";
    importer_light.type = ImporterLight::Type::distant;
    importer_light.intensity = float3(2.f, 3.f, 4.f);
    importer_light.degree_angular_diameter = 12.f;
    importer_scene->lights.push_back(importer_light);

    ImporterNode light_node;
    light_node.name = importer_light.name;
    light_node.light_index = 0;
    importer_scene->nodes.push_back(light_node);
    importer_scene->root_nodes.push_back(0);

    auto scene = Scene::create(ref(ctx.device));
    load_importer_scene(scene.get(), *importer_scene);

    REQUIRE_EQ(scene->components().size(), 1);
    const auto* distant_light = dynamic_cast<const DistantLight*>(scene->components()[0]);
    REQUIRE(distant_light != nullptr);
    CHECK_EQ(distant_light->radiance(), importer_light.intensity);
    CHECK_EQ(distant_light->cutoff_angle(), doctest::Approx(6.f));
}

TEST_CASE_GPU("importer scene uses default material for missing assignments")
{
    auto importer_scene = make_ref<ImporterScene>();

    ImporterMesh mesh;
    mesh.name = "materialless_mesh";
    mesh.ensure_attributes({{ImporterSemantic::position}});
    mesh.allocate_vertices(3, true);
    auto positions = mesh.position_stream();
    positions[0] = float3(-1.f, 0.f, 0.f);
    positions[1] = float3(1.f, 0.f, 0.f);
    positions[2] = float3(0.f, 1.f, 0.f);
    mesh.subgeometries.push_back(
        ImporterMesh::Subgeometry{
            .name = "unbound_submesh",
            .indices = {uint3(0, 1, 2)},
        }
    );
    mesh.subgeometries.push_back(
        ImporterMesh::Subgeometry{
            .name = "missing_material_submesh",
            .indices = {uint3(0, 1, 2)},
            .material_name = "MissingMaterial",
        }
    );
    mesh.calculate_local_aabb();
    importer_scene->meshes.push_back(std::move(mesh));

    ImporterNode node;
    node.name = "materialless_mesh";
    node.mesh_index = 0;
    importer_scene->nodes.push_back(node);
    importer_scene->root_nodes.push_back(0);

    auto scene = Scene::create(ref(ctx.device), *importer_scene);

    REQUIRE_EQ(scene->materials().size(), 1);
    CHECK(dynamic_cast<StandardMaterial*>(scene->materials()[0]) != nullptr);
    CHECK_EQ(scene->materials()[0]->name(), "DefaultMaterial");

    REQUIRE_EQ(scene->components().size(), 1);
    auto* instance = dynamic_cast<GeometryInstance*>(scene->components()[0]);
    REQUIRE(instance != nullptr);
    REQUIRE_EQ(instance->materials().size(), 2);
    CHECK_EQ(instance->materials()[0], scene->materials()[0]);
    CHECK_EQ(instance->materials()[1], scene->materials()[0]);
}

TEST_CASE_GPU("importer material constructor creates and names live material")
{
    auto importer_scene = make_ref<ImporterScene>();
    bool constructor_called = false;

    ImporterMaterial importer_material;
    importer_material.name = "Constructed Material";
    importer_material.params.set("roughness_factor", 0.25f);
    importer_material.constructor = [&](Scene& destination, const ImporterMaterial& source)
    {
        constructor_called = true;
        CHECK_EQ(source.name, "Constructed Material");
        CHECK_EQ(source.params.get<float>("roughness_factor"), 0.25f);
        return destination.create_material<StandardMaterial>(source.params);
    };
    importer_scene->materials.push_back(std::move(importer_material));

    auto scene = Scene::create(ref(ctx.device), *importer_scene);

    CHECK(constructor_called);
    REQUIRE_EQ(scene->materials().size(), 1);
    CHECK(dynamic_cast<StandardMaterial*>(scene->materials()[0]) != nullptr);
    CHECK_EQ(scene->materials()[0]->name(), "Constructed Material");
}

TEST_CASE_GPU("importer material constructor rejects null material")
{
    auto importer_scene = make_ref<ImporterScene>();
    ImporterMaterial importer_material;
    importer_material.name = "Null Material";
    importer_material.constructor = [](Scene&, const ImporterMaterial&) -> Material*
    {
        return nullptr;
    };
    importer_scene->materials.push_back(std::move(importer_material));

    try {
        Scene::create(ref(ctx.device), *importer_scene);
        FAIL("Expected a null importer material constructor result to throw");
    } catch (const std::exception& e) {
        CHECK(
            std::string(e.what()).find("Constructor for importer material 'Null Material' returned null")
            != std::string::npos
        );
    }
}

TEST_CASE_GPU("python importer material constructor runs from native scene load")
{
    auto python = PythonInterpreter::get().create_context();
    const std::string project_path = testing::project_directory().generic_string();
    python.execute_string(fmt::format("import sys\nsys.path.insert(0, r'{}')", project_path));

    const std::filesystem::path scene_path = testing::project_directory() / "data" / "scenes" / "checker-material.py";
    auto scene = Scene::create(ref(ctx.device), scene_path);

    Material* material = scene->materials().find("Material_MR");
    REQUIRE(material != nullptr);
    CHECK_EQ(material->slang_type_name(), "CheckerMaterial");
}

TEST_CASE_GPU("importer scene camera becomes active and preserves vertical fov")
{
    auto importer_scene = make_ref<ImporterScene>();

    ImporterCamera importer_camera;
    importer_camera.name = "Test Camera";
    importer_camera.focus_distance = 1.f;
    importer_camera.focal_length = 50.f;
    importer_camera.fstop = 8.f;
    importer_camera.depth_range = float2(0.01f, 10000.f);
    importer_camera.projection = ImporterCamera::Projection::perspective;
    importer_camera.fov_direction = ImporterCamera::FOVDirection::vertical;
    importer_camera.sensor_size_mm = 24.f;
    importer_camera.focal_length = ImporterCamera::focal_length_from_fov_degrees(42.f, importer_camera.sensor_size_mm);
    importer_scene->cameras.push_back(importer_camera);

    ImporterNode camera_node;
    camera_node.name = "Test Camera Node";
    camera_node.camera_index = 0;
    importer_scene->nodes.push_back(camera_node);
    importer_scene->root_nodes.push_back(0);

    auto scene = Scene::create(ref(ctx.device));
    load_importer_scene(scene.get(), *importer_scene);

    Camera* active_camera = scene->active_camera();
    REQUIRE(active_camera != nullptr);
    CHECK_EQ(active_camera->fov_y(), doctest::Approx(42.f));
}

TEST_CASE_GPU("importer scene camera converts horizontal fov with four by three sensor assumption")
{
    auto importer_scene = make_ref<ImporterScene>();

    ImporterCamera importer_camera;
    importer_camera.name = "Horizontal FOV Camera";
    importer_camera.fov_direction = ImporterCamera::FOVDirection::horizontal;
    importer_camera.sensor_size_mm = 32.f;
    importer_camera.focal_length = ImporterCamera::focal_length_from_fov_degrees(60.f, importer_camera.sensor_size_mm);
    importer_scene->cameras.push_back(importer_camera);

    ImporterNode camera_node;
    camera_node.name = "Horizontal FOV Camera Node";
    camera_node.camera_index = 0;
    importer_scene->nodes.push_back(camera_node);
    importer_scene->root_nodes.push_back(0);

    auto scene = Scene::create(ref(ctx.device));
    load_importer_scene(scene.get(), *importer_scene);

    const float expected_vertical_fov
        = ImporterCamera::fov_degrees_from_focal_length(24.f, importer_camera.focal_length);

    Camera* active_camera = scene->active_camera();
    REQUIRE(active_camera != nullptr);
    CHECK_EQ(active_camera->fov_y(), doctest::Approx(expected_vertical_fov));
}

TEST_CASE_GPU("add remove compact materials")
{
    auto scene = Scene::create(ref(ctx.device));

    auto* m0 = scene->create_material("StandardMaterial");
    auto* m1 = scene->create_material("StandardMaterial");
    auto* m2 = scene->create_material("StandardMaterial");
    m0->set_name("m0");
    m1->set_name("m1");
    m2->set_name("m2");

    CHECK_EQ(scene->materials().size(), 3);
    CHECK_EQ(m0->collection_index(), 0);
    CHECK_EQ(m1->collection_index(), 1);
    CHECK_EQ(m2->collection_index(), 2);

    // Remove the middle material.
    m1->remove();
    // Before update, collection still has 3 entries.
    CHECK_EQ(scene->materials().size(), 3);

    scene->update();

    // After update, removed material is gone; remaining are compacted.
    CHECK_EQ(scene->materials().size(), 2);
    CHECK_EQ(scene->materials()[0]->name(), "m0");
    CHECK_EQ(scene->materials()[1]->name(), "m2");
    // Indices should be updated after compaction.
    CHECK_EQ(scene->materials()[0]->collection_index(), 0);
    CHECK_EQ(scene->materials()[1]->collection_index(), 1);
}

TEST_CASE_GPU("add remove compact entities")
{
    auto scene = Scene::create(ref(ctx.device));

    auto* e0 = scene->create_entity();
    auto* e1 = scene->create_entity();
    auto* e2 = scene->create_entity();
    e0->set_name("e0");
    e1->set_name("e1");
    e2->set_name("e2");

    CHECK_EQ(scene->entities().size(), 3);

    // Remove first and last.
    e0->remove();
    e2->remove();

    scene->update();

    CHECK_EQ(scene->entities().size(), 1);
    CHECK_EQ(scene->entities()[0]->name(), "e1");
    CHECK_EQ(scene->entities()[0]->collection_index(), 0);
}

TEST_CASE_GPU("add remove compact geometries")
{
    auto scene = Scene::create(ref(ctx.device));

    auto* g0 = scene->create_geometry("StaticMeshGeometry");
    auto* g1 = scene->create_geometry("StaticMeshGeometry");
    auto* g2 = scene->create_geometry("StaticMeshGeometry");
    g0->set_name("g0");
    g1->set_name("g1");
    g2->set_name("g2");

    CHECK_EQ(scene->geometries().size(), 3);

    g1->remove();
    scene->update();

    CHECK_EQ(scene->geometries().size(), 2);
    CHECK_EQ(scene->geometries()[0]->name(), "g0");
    CHECK_EQ(scene->geometries()[1]->name(), "g2");
}

// ---------------------------------------------------------------------------
// 2. Dirty flag propagation
// ---------------------------------------------------------------------------

TEST_CASE_GPU("dirty flags on add")
{
    auto scene = Scene::create(ref(ctx.device));

    // Initially no dirty flags.
    CHECK_FALSE(scene->material_collection().is_dirty());

    auto* m0 = scene->create_material("StandardMaterial");
    REQUIRE(m0 != nullptr);

    // After adding, collection should be dirty with 'added' flag.
    CHECK(scene->material_collection().is_dirty());
    CHECK(scene->material_collection().has_dirty(Material::DirtyFlags::added));

    // After update, dirty flags should be reset.
    scene->update();
    CHECK_FALSE(scene->material_collection().is_dirty());
}

TEST_CASE_GPU("dirty flags on entity transform")
{
    auto scene = Scene::create(ref(ctx.device));

    auto* entity = scene->create_entity();
    // 'added' flag is set on creation.
    CHECK(scene->entity_collection().has_dirty(Entity::DirtyFlags::added));

    scene->update();
    CHECK_FALSE(scene->entity_collection().is_dirty());

    // Modify transform.
    Transform t;
    t.set_translation(float3(1.f, 2.f, 3.f));
    entity->set_transform(t);

    CHECK(scene->entity_collection().is_dirty());
    CHECK(scene->entity_collection().has_dirty(Entity::DirtyFlags::transform));

    scene->update();
    CHECK_FALSE(scene->entity_collection().is_dirty());
}

TEST_CASE_GPU("dirty flags on remove")
{
    auto scene = Scene::create(ref(ctx.device));

    scene->create_material("StandardMaterial");
    scene->update();
    CHECK_FALSE(scene->material_collection().is_dirty());

    scene->materials()[0]->remove();
    CHECK(scene->material_collection().has_dirty(Material::DirtyFlags::removed));

    scene->update();
    CHECK_FALSE(scene->material_collection().is_dirty());
    CHECK_EQ(scene->materials().size(), 0);
}

TEST_CASE_GPU("combined dirty flags")
{
    auto scene = Scene::create(ref(ctx.device));

    auto* e0 = scene->create_entity();
    auto* e1 = scene->create_entity();

    scene->update();
    CHECK_FALSE(scene->entity_collection().is_dirty());

    // Modify transforms on both entities.
    Transform t;
    t.set_translation(float3(1.f, 0.f, 0.f));
    e0->set_transform(t);

    t.set_translation(float3(0.f, 1.f, 0.f));
    e1->set_transform(t);

    // Combined dirty flags should reflect both modifications.
    auto combined = scene->entity_collection().combined_dirty_flags();
    CHECK(is_set(combined, Entity::DirtyFlags::transform));

    scene->update();
    CHECK_FALSE(scene->entity_collection().is_dirty());
}

// ---------------------------------------------------------------------------
// 3. Entity hierarchy
// ---------------------------------------------------------------------------

TEST_CASE_GPU("entity parent child hierarchy")
{
    auto scene = Scene::create(ref(ctx.device));

    auto* parent = scene->create_entity();
    auto* child = scene->create_entity();

    child->set_parent(parent);

    CHECK_EQ(child->parent(), parent);
    CHECK_EQ(parent->children().size(), 1);
    CHECK_EQ(parent->children()[0], child);
}

TEST_CASE_GPU("world from object matrix propagation")
{
    auto scene = Scene::create(ref(ctx.device));

    auto* parent = scene->create_entity();
    auto* child = scene->create_entity();

    // Set parent translation to (10, 0, 0).
    Transform parent_t;
    parent_t.set_translation(float3(10.f, 0.f, 0.f));
    parent->set_transform(parent_t);

    // Set child translation to (0, 5, 0).
    Transform child_t;
    child_t.set_translation(float3(0.f, 5.f, 0.f));
    child->set_transform(child_t);

    child->set_parent(parent);

    // Child world matrix should combine both translations.
    float4x4 child_world = child->world_from_object_matrix();
    // Translation is at [row][3] (row-major): [0][3]=x, [1][3]=y, [2][3]=z.
    CHECK(child_world[0][3] == doctest::Approx(10.f));
    CHECK(child_world[1][3] == doctest::Approx(5.f));
    CHECK(child_world[2][3] == doctest::Approx(0.f));

    // Parent world matrix should just be its own translation.
    float4x4 parent_world = parent->world_from_object_matrix();
    CHECK(parent_world[0][3] == doctest::Approx(10.f));
    CHECK(parent_world[1][3] == doctest::Approx(0.f));
}

TEST_CASE_GPU("reparent entity updates world matrix")
{
    auto scene = Scene::create(ref(ctx.device));

    auto* parent_a = scene->create_entity();
    auto* parent_b = scene->create_entity();
    auto* child = scene->create_entity();

    Transform ta;
    ta.set_translation(float3(10.f, 0.f, 0.f));
    parent_a->set_transform(ta);

    Transform tb;
    tb.set_translation(float3(0.f, 20.f, 0.f));
    parent_b->set_transform(tb);

    Transform tc;
    tc.set_translation(float3(1.f, 1.f, 1.f));
    child->set_transform(tc);

    // Parent under A.
    child->set_parent(parent_a);
    float4x4 m = child->world_from_object_matrix();
    CHECK(m[0][3] == doctest::Approx(11.f));
    CHECK(m[1][3] == doctest::Approx(1.f));

    // Re-parent under B.
    child->set_parent(parent_b);
    m = child->world_from_object_matrix();
    CHECK(m[0][3] == doctest::Approx(1.f));
    CHECK(m[1][3] == doctest::Approx(21.f));

    // Verify A no longer has the child.
    CHECK(parent_a->children().empty());
    CHECK_EQ(parent_b->children().size(), 1);
}

TEST_CASE_GPU("deep hierarchy propagation")
{
    auto scene = Scene::create(ref(ctx.device));

    auto* root = scene->create_entity();
    auto* mid = scene->create_entity();
    auto* leaf = scene->create_entity();

    mid->set_parent(root);
    leaf->set_parent(mid);

    // Each level translates by (1, 0, 0).
    Transform t;
    t.set_translation(float3(1.f, 0.f, 0.f));
    root->set_transform(t);
    mid->set_transform(t);
    leaf->set_transform(t);

    float4x4 m = leaf->world_from_object_matrix();
    CHECK(m[0][3] == doctest::Approx(3.f));
}

TEST_CASE_GPU("set world transform with parent hierarchy")
{
    auto scene = Scene::create(ref(ctx.device));

    auto* parent = scene->create_entity();
    auto* child = scene->create_entity();

    // Parent has a uniform scale of 2x and translation of (10, 0, 0).
    Transform parent_t;
    parent_t.set_translation(float3(10.f, 0.f, 0.f));
    parent_t.set_scale(float3(2.f, 2.f, 2.f));
    parent->set_transform(parent_t);

    child->set_parent(parent);

    // Set a desired world position of (20, 10, 0) on the child.
    Transform world_t;
    world_t.set_translation(float3(20.f, 10.f, 0.f));
    child->set_world_transform(world_t);

    // Verify the child's world matrix has the expected world translation.
    float4x4 child_world = child->world_from_object_matrix();
    CHECK(child_world[0][3] == doctest::Approx(20.f));
    CHECK(child_world[1][3] == doctest::Approx(10.f));
    CHECK(child_world[2][3] == doctest::Approx(0.f));

    // Verify the local transform was adjusted (should be (5, 5, 0) due to 2x scale).
    float3 local_pos = child->transform().translation();
    CHECK(local_pos.x == doctest::Approx(5.f));
    CHECK(local_pos.y == doctest::Approx(5.f));
    CHECK(local_pos.z == doctest::Approx(0.f));
}

TEST_CASE_GPU("set world transform without parent")
{
    auto scene = Scene::create(ref(ctx.device));

    auto* entity = scene->create_entity();

    // Without a parent, set_world_transform should behave like set_transform.
    Transform world_t;
    world_t.set_translation(float3(5.f, 10.f, 15.f));
    entity->set_world_transform(world_t);

    float4x4 m = entity->world_from_object_matrix();
    CHECK(m[0][3] == doctest::Approx(5.f));
    CHECK(m[1][3] == doctest::Approx(10.f));
    CHECK(m[2][3] == doctest::Approx(15.f));
}

// ---------------------------------------------------------------------------
// 4. Component lifecycle
// ---------------------------------------------------------------------------

TEST_CASE_GPU("component attached to entity")
{
    auto scene = Scene::create(ref(ctx.device));

    auto* entity = scene->create_entity();
    auto* camera = entity->create_component<Camera>();

    CHECK_EQ(scene->components().size(), 1);
    CHECK_EQ(entity->components().size(), 1);
    CHECK_EQ(entity->components()[0], camera);
    CHECK_EQ(camera->entity(), entity);
}

TEST_CASE_GPU("removing entity removes components")
{
    auto scene = Scene::create(ref(ctx.device));

    auto* entity = scene->create_entity();
    entity->create_component<Camera>();
    entity->create_component<ConstantLight>();

    CHECK_EQ(scene->components().size(), 2);
    CHECK_EQ(scene->entities().size(), 1);

    entity->remove();
    scene->update();

    CHECK_EQ(scene->entities().size(), 0);
    CHECK_EQ(scene->components().size(), 0);
}

TEST_CASE_GPU("removing entity removes children and their components")
{
    auto scene = Scene::create(ref(ctx.device));

    auto* parent = scene->create_entity();
    auto* child = scene->create_entity();
    child->set_parent(parent);
    child->create_component<Camera>();

    // Initial update to allocate render resources for all entities.
    scene->update();

    CHECK_EQ(scene->entities().size(), 2);
    CHECK_EQ(scene->components().size(), 1);

    // Removing parent should also remove child and its components.
    parent->remove();
    scene->update();

    CHECK_EQ(scene->entities().size(), 0);
    CHECK_EQ(scene->components().size(), 0);
}

// ---------------------------------------------------------------------------
// Helper: create a StaticMeshGeometry with a simple triangle.
// ---------------------------------------------------------------------------

static StaticMeshGeometry* create_triangle_geometry(Scene* scene, const float3& v0, const float3& v1, const float3& v2)
{
    static float3 positions[3];
    positions[0] = v0;
    positions[1] = v1;
    positions[2] = v2;

    static float3 normals[3] = {{0, 1, 0}, {0, 1, 0}, {0, 1, 0}};
    static float3 tangents[3] = {{1, 0, 0}, {1, 0, 0}, {1, 0, 0}};
    static float handedness[3] = {1.f, 1.f, 1.f};
    static float2 texcoords[3] = {{0, 0}, {1, 0}, {0, 1}};
    static uint32_t indices[3] = {0, 1, 2};

    StaticMeshGeometryDataDesc desc = {};
    desc.name = "triangle";
    desc.vertex_count = 3;
    desc.position_stream = {positions, sizeof(float3)};
    desc.normal_stream = {normals, sizeof(float3)};
    desc.tangent_stream = {tangents, sizeof(float3)};
    desc.handedness_stream = {handedness, sizeof(float)};
    desc.texcoord_stream = {texcoords, sizeof(float2)};
    desc.sub_meshes.push_back({
        .name = "sub0",
        .index_count = 3,
        .index_stream = {indices, sizeof(uint32_t)},
    });

    StaticMeshGeometry* geom = scene->create_geometry<StaticMeshGeometry>();
    geom->set_mesh_data(desc);
    return geom;
}

// ---------------------------------------------------------------------------
// Geometry local AABB
// ---------------------------------------------------------------------------

TEST_CASE_GPU("geometry local aabb")
{
    auto scene = Scene::create(ref(ctx.device));

    auto* geom
        = create_triangle_geometry(scene, float3(-1.f, 0.f, -1.f), float3(1.f, 0.f, -1.f), float3(0.f, 2.f, 1.f));

    CHECK(geom->local_aabb().is_valid());
    CHECK(geom->local_aabb().min.x == doctest::Approx(-1.f));
    CHECK(geom->local_aabb().max.x == doctest::Approx(1.f));
    CHECK(geom->local_aabb().min.y == doctest::Approx(0.f));
    CHECK(geom->local_aabb().max.y == doctest::Approx(2.f));
    CHECK(geom->local_aabb().min.z == doctest::Approx(-1.f));
    CHECK(geom->local_aabb().max.z == doctest::Approx(1.f));
}

// ---------------------------------------------------------------------------
// Entity world AABB
// ---------------------------------------------------------------------------

TEST_CASE_GPU("entity world aabb no geometry")
{
    auto scene = Scene::create(ref(ctx.device));

    auto* entity = scene->create_entity();
    entity->set_name("empty");

    AABB aabb = entity->world_aabb();
    CHECK_FALSE(aabb.is_valid());
}

TEST_CASE_GPU("entity world aabb identity transform")
{
    auto scene = Scene::create(ref(ctx.device));

    auto* geom = create_triangle_geometry(scene, float3(0.f, 0.f, 0.f), float3(2.f, 0.f, 0.f), float3(0.f, 3.f, 0.f));

    auto* entity = scene->create_entity();
    auto* instance = entity->create_component<GeometryInstance>();
    instance->set_geometry(geom);

    AABB aabb = entity->world_aabb();
    CHECK(aabb.is_valid());
    CHECK(aabb.min.x == doctest::Approx(0.f));
    CHECK(aabb.max.x == doctest::Approx(2.f));
    CHECK(aabb.min.y == doctest::Approx(0.f));
    CHECK(aabb.max.y == doctest::Approx(3.f));
}

TEST_CASE_GPU("entity world aabb with translation")
{
    auto scene = Scene::create(ref(ctx.device));

    auto* geom = create_triangle_geometry(scene, float3(0.f, 0.f, 0.f), float3(1.f, 0.f, 0.f), float3(0.f, 1.f, 0.f));

    auto* entity = scene->create_entity();
    Transform t;
    t.set_translation(float3(10.f, 20.f, 30.f));
    entity->set_transform(t);

    auto* instance = entity->create_component<GeometryInstance>();
    instance->set_geometry(geom);

    AABB aabb = entity->world_aabb();
    CHECK(aabb.is_valid());
    CHECK(aabb.min.x == doctest::Approx(10.f));
    CHECK(aabb.max.x == doctest::Approx(11.f));
    CHECK(aabb.min.y == doctest::Approx(20.f));
    CHECK(aabb.max.y == doctest::Approx(21.f));
    CHECK(aabb.min.z == doctest::Approx(30.f));
    CHECK(aabb.max.z == doctest::Approx(30.f));

    float3 center = aabb.center();
    CHECK(center.x == doctest::Approx(10.5f));
    CHECK(center.y == doctest::Approx(20.5f));
    CHECK(center.z == doctest::Approx(30.f));
}

TEST_SUITE_END();
