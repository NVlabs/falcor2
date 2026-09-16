// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "testing.h"

#include "falcor2/render/scene.h"
#include "falcor2/render/ray_tracing_setup.h"
#include "falcor2/render/emissive_geometry_system.h"
#include "falcor2/render/light_system.h"
#include "falcor2/render/geometry/static_mesh_geometry.h"
#include "falcor2/render/geometry/geometry_group.h"
#include "falcor2/render/component/geometry_instance.h"
#include "falcor2/render/component/camera.h"
#include "falcor2/render/component/light.h"
#include "falcor2/render/material/emission_only_material.h"
#include "falcor2/render/material/standard_material.h"
#include "falcor2/importers/importer.h"
#include "falcor2/importers/importer_types.h"
#include "falcor2/core/python_interpreter.h"

#include <sgl/math/vector.h>

#include <array>

#include <filesystem>
#include <fstream>

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
        scene->append(*importer_scene);

        CHECK(scene->config().uv_origin == UVOrigin::upper_left);
        check_imported_uvs(scene.get(), float2(0.25f, 1.25f), float2(0.5f, 0.5f), float2(0.75f, -0.25f));
    }

    {
        auto scene = Scene::from_importer_scene(ref(ctx.device), *importer_scene);
        CHECK(scene->config().uv_origin == UVOrigin::lower_left);
        check_imported_uvs(scene.get(), float2(0.25f, -0.25f), float2(0.5f, 0.5f), float2(0.75f, 1.25f));
    }

    {
        auto scene = Scene::from_importer_scene(ref(ctx.device), *importer_scene);
        CHECK(scene->config().uv_origin == UVOrigin::lower_left);
        check_imported_uvs(scene.get(), float2(0.25f, -0.25f), float2(0.5f, 0.5f), float2(0.75f, 1.25f));
    }

    {
        auto scene = Scene::from_importer_scene(ref(ctx.device), *importer_scene, SceneConfig(UVOrigin::upper_left));
        CHECK(scene->config().uv_origin == UVOrigin::upper_left);
        check_imported_uvs(scene.get(), float2(0.25f, 1.25f), float2(0.5f, 0.5f), float2(0.75f, -0.25f));
    }
}

TEST_CASE_GPU("importer scene textured dome light preserves radiometry")
{
    auto importer_scene = make_ref<ImporterScene>();

    ImporterLight dome_light;
    dome_light.name = "Test Dome Light";
    dome_light.type = ImporterLight::Type::dome;
    dome_light.env_map_path = "data/assets/envmaps/aerodynamics_workshop_512.hdr";
    dome_light.intensity = float3(0.25f, 0.5f, 0.75f);
    dome_light.exposure = 5.f;
    importer_scene->lights.push_back(dome_light);

    ImporterNode dome_node;
    dome_node.name = "Test Dome Light";
    dome_node.light_index = 0;
    importer_scene->nodes.push_back(dome_node);
    importer_scene->root_nodes.push_back(0);

    auto scene = Scene::from_importer_scene(ref(ctx.device), *importer_scene);

    REQUIRE_EQ(scene->components().size(), 1);
    const auto* env_map_light = dynamic_cast<const EnvMapLight*>(scene->components()[0]);
    REQUIRE(env_map_light != nullptr);
    CHECK_EQ(env_map_light->env_map_path(), std::filesystem::path(dome_light.env_map_path));
    CHECK_EQ(env_map_light->intensity(), dome_light.intensity);
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

    auto scene = Scene::from_importer_scene(ref(ctx.device), *importer_scene);

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
    importer_light.exposure = 2.f;
    importer_light.degree_angular_diameter = 12.f;
    importer_scene->lights.push_back(importer_light);

    ImporterNode light_node;
    light_node.name = importer_light.name;
    light_node.light_index = 0;
    importer_scene->nodes.push_back(light_node);
    importer_scene->root_nodes.push_back(0);

    auto scene = Scene::from_importer_scene(ref(ctx.device), *importer_scene);

    REQUIRE_EQ(scene->components().size(), 1);
    const auto* distant_light = dynamic_cast<const DistantLight*>(scene->components()[0]);
    REQUIRE(distant_light != nullptr);
    CHECK_EQ(distant_light->radiance(), importer_light.intensity);
    CHECK_EQ(distant_light->exposure(), doctest::Approx(importer_light.exposure));
    CHECK_EQ(distant_light->cutoff_angle(), doctest::Approx(6.f));
}

TEST_CASE_GPU("importer scene creates shaped point light")
{
    auto importer_scene = make_ref<ImporterScene>();

    ImporterLight importer_light;
    importer_light.name = "Spot Light";
    importer_light.type = ImporterLight::Type::point;
    importer_light.intensity = float3(2.f, 3.f, 4.f);
    importer_light.exposure = 2.f;
    importer_light.enable_shaping = true;
    importer_light.shaping_cone_angle = 35.f;
    importer_light.shaping_cone_softness = 3.f / 7.f;
    importer_light.shaping_focus = 2.f;
    importer_scene->lights.push_back(importer_light);

    ImporterNode light_node;
    light_node.name = importer_light.name;
    light_node.light_index = 0;
    importer_scene->nodes.push_back(light_node);
    importer_scene->root_nodes.push_back(0);

    auto scene = Scene::from_importer_scene(ref(ctx.device), *importer_scene);

    REQUIRE_EQ(scene->components().size(), 1);
    const auto* point_light = dynamic_cast<const PointLight*>(scene->components()[0]);
    REQUIRE(point_light != nullptr);
    REQUIRE(point_light->class_descriptor().base());
    CHECK_EQ(point_light->class_descriptor().base()->name(), "Light");
    CHECK_EQ(point_light->intensity(), importer_light.intensity);
    CHECK_EQ(point_light->exposure(), doctest::Approx(importer_light.exposure));
    CHECK(point_light->enable_shaping());
    CHECK_EQ(point_light->shaping_cone_angle(), doctest::Approx(35.f));
    CHECK_EQ(point_light->shaping_cone_softness(), doctest::Approx(3.f / 7.f));
    CHECK_EQ(point_light->shaping_focus(), doctest::Approx(importer_light.shaping_focus));
}

TEST_CASE_GPU("importer scene preserves point light radiometry")
{
    auto importer_scene = make_ref<ImporterScene>();

    ImporterLight importer_light;
    importer_light.name = "Point Light";
    importer_light.type = ImporterLight::Type::point;
    importer_light.intensity = float3(2.f, 3.f, 4.f);
    importer_light.exposure = 2.f;
    importer_light.enable_color_temperature = true;
    importer_light.color_temperature = 3000.f;
    importer_scene->lights.push_back(importer_light);

    ImporterNode light_node;
    light_node.name = importer_light.name;
    light_node.light_index = 0;
    importer_scene->nodes.push_back(light_node);
    importer_scene->root_nodes.push_back(0);

    auto scene = Scene::from_importer_scene(ref(ctx.device), *importer_scene);

    REQUIRE_EQ(scene->components().size(), 1);
    const auto* point_light = dynamic_cast<const PointLight*>(scene->components()[0]);
    REQUIRE(point_light != nullptr);
    CHECK_EQ(point_light->intensity(), importer_light.intensity);
    CHECK_EQ(point_light->exposure(), doctest::Approx(importer_light.exposure));
    CHECK(point_light->enable_color_temperature());
    CHECK_EQ(point_light->color_temperature(), doctest::Approx(importer_light.color_temperature));
}

TEST_CASE_GPU("importer scene creates shaped analytic sphere, disk, and rectangle lights")
{
    auto importer_scene = make_ref<ImporterScene>();

    ImporterLight sphere;
    sphere.name = "Sphere Light";
    sphere.type = ImporterLight::Type::sphere;
    sphere.intensity = float3(2.f, 3.f, 4.f);
    sphere.exposure = 1.f;
    sphere.radius = 0.75f;
    sphere.enable_shaping = true;
    sphere.shaping_cone_angle = 35.f;
    sphere.shaping_cone_softness = 0.2f;
    sphere.shaping_focus = 2.f;
    sphere.enable_virtual_sphere_shrinking = true;
    importer_scene->lights.push_back(sphere);

    ImporterLight disk;
    disk.name = "Shaped Disk Light";
    disk.type = ImporterLight::Type::disk;
    disk.intensity = float3(5.f, 6.f, 7.f);
    disk.exposure = 2.f;
    disk.radius = 1.25f;
    disk.enable_shaping = true;
    disk.shaping_cone_angle = 40.f;
    disk.shaping_cone_softness = 0.25f;
    disk.shaping_focus = 4.f;
    importer_scene->lights.push_back(disk);

    ImporterLight rect;
    rect.name = "Rectangle Light";
    rect.type = ImporterLight::Type::rectangular;
    rect.intensity = float3(8.f, 9.f, 10.f);
    rect.exposure = 3.f;
    rect.width = 2.5f;
    rect.height = 1.5f;
    rect.enable_shaping = true;
    rect.shaping_cone_angle = 50.f;
    rect.shaping_cone_softness = 0.2f;
    rect.shaping_focus = 1.5f;
    importer_scene->lights.push_back(rect);

    for (int light_index = 0; light_index < 3; ++light_index) {
        ImporterNode light_node;
        light_node.name = importer_scene->lights[light_index].name;
        light_node.light_index = light_index;
        importer_scene->nodes.push_back(light_node);
        importer_scene->root_nodes.push_back(light_index);
    }

    auto scene = Scene::from_importer_scene(ref(ctx.device), *importer_scene);

    REQUIRE_EQ(scene->components().size(), 3);
    const SphereLight* sphere_light = scene->components().find<SphereLight>();
    const DiskLight* disk_light = scene->components().find<DiskLight>();
    const RectLight* rect_light = scene->components().find<RectLight>();

    REQUIRE(sphere_light);
    REQUIRE(sphere_light->class_descriptor().base());
    CHECK_EQ(sphere_light->class_descriptor().base()->name(), "Light");
    CHECK_EQ(sphere_light->radiance(), sphere.intensity);
    CHECK_EQ(sphere_light->exposure(), doctest::Approx(sphere.exposure));
    CHECK_EQ(sphere_light->radius(), doctest::Approx(sphere.radius));
    CHECK(sphere_light->enable_virtual_sphere_shrinking());
    CHECK(sphere_light->enable_shaping());
    CHECK_EQ(sphere_light->shaping_cone_angle(), doctest::Approx(sphere.shaping_cone_angle));
    CHECK_EQ(sphere_light->shaping_cone_softness(), doctest::Approx(sphere.shaping_cone_softness));
    CHECK_EQ(sphere_light->shaping_focus(), doctest::Approx(sphere.shaping_focus));

    REQUIRE(disk_light);
    REQUIRE(disk_light->class_descriptor().base());
    CHECK_EQ(disk_light->class_descriptor().base()->name(), "Light");
    CHECK_EQ(disk_light->radiance(), disk.intensity);
    CHECK_EQ(disk_light->exposure(), doctest::Approx(disk.exposure));
    CHECK_EQ(disk_light->radius(), doctest::Approx(disk.radius));
    CHECK(disk_light->enable_shaping());
    CHECK_EQ(disk_light->shaping_cone_angle(), doctest::Approx(disk.shaping_cone_angle));
    CHECK_EQ(disk_light->shaping_cone_softness(), doctest::Approx(disk.shaping_cone_softness));
    CHECK_EQ(disk_light->shaping_focus(), doctest::Approx(disk.shaping_focus));

    REQUIRE(rect_light);
    REQUIRE(rect_light->class_descriptor().base());
    CHECK_EQ(rect_light->class_descriptor().base()->name(), "Light");
    CHECK_EQ(rect_light->radiance(), rect.intensity);
    CHECK_EQ(rect_light->exposure(), doctest::Approx(rect.exposure));
    CHECK_EQ(rect_light->width(), doctest::Approx(rect.width));
    CHECK_EQ(rect_light->height(), doctest::Approx(rect.height));
    CHECK(rect_light->enable_shaping());
    CHECK_EQ(rect_light->shaping_cone_angle(), doctest::Approx(rect.shaping_cone_angle));
    CHECK_EQ(rect_light->shaping_cone_softness(), doctest::Approx(rect.shaping_cone_softness));
    CHECK_EQ(rect_light->shaping_focus(), doctest::Approx(rect.shaping_focus));

    Properties properties = rect_light->properties();
    CHECK_EQ(properties.get<float3>("radiance"), rect.intensity);
    CHECK(properties.get<bool>("enable_shaping"));
    CHECK_EQ(properties.get<float>("shaping_cone_angle"), doctest::Approx(rect.shaping_cone_angle));
    CHECK_EQ(properties.get<float>("shaping_focus"), doctest::Approx(rect.shaping_focus));
    CHECK_EQ(properties.get<float>("width"), doctest::Approx(rect.width));

    Properties sphere_properties = sphere_light->properties();
    CHECK(sphere_properties.get<bool>("enable_virtual_sphere_shrinking"));
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

    auto scene = Scene::from_importer_scene(ref(ctx.device), *importer_scene);

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

    auto scene = Scene::from_importer_scene(ref(ctx.device), *importer_scene);

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
        Scene::from_importer_scene(ref(ctx.device), *importer_scene);
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
    const std::filesystem::path scene_path = testing::project_directory() / "data" / "scenes" / "checker-material.py";
    auto scene = Scene::load(ref(ctx.device), scene_path);

    Material* material = scene->materials().find("Red");
    REQUIRE(material != nullptr);
    CHECK_EQ(material->slang_type_name(), "CheckerMaterial");
}

TEST_CASE_GPU("native pyscene load resolves adjacent helper relative asset and callback")
{
    const std::filesystem::path scene_path
        = testing::project_directory() / "tests" / "native" / "render" / "pyscene_native_loading.py";
    ref<Importer> previous_importer = Importer::create();
    ScopedCurrentImporter current_importer(previous_importer);

    PythonContext observer = PythonInterpreter::get().create_context();
    observer.execute_string("import sys\nsaved_sys_path = list(sys.path)");

    ref<Scene> scene = Scene::load(ref(ctx.device), scene_path);

    CHECK_EQ(Importer::get().get(), previous_importer.get());
    observer.execute_string("assert sys.path == saved_sys_path");
    REQUIRE(scene->active_camera() != nullptr);
    CHECK_EQ(scene->active_camera()->name(), "PyScene Camera");
    CHECK(scene->materials().find("/Root/Looks/Mat") != nullptr);
    CHECK(scene->materials().find("PyScene Callback Material") != nullptr);
}

TEST_CASE_GPU("native pyscene append loads edits and callback observes existing scene")
{
    const std::filesystem::path scene_path
        = testing::project_directory() / "tests" / "native" / "render" / "pyscene_native_loading.py";
    ref<Scene> scene = Scene::create(ref(ctx.device));
    Material* existing_material = scene->create_material<StandardMaterial>();
    existing_material->set_name("Existing Material");

    scene->append(scene_path);

    CHECK_EQ(scene->materials().find("Existing Material"), existing_material);
    REQUIRE(scene->active_camera() != nullptr);
    CHECK_EQ(scene->active_camera()->name(), "PyScene Camera");
    CHECK(scene->materials().find("/Root/Looks/Mat") != nullptr);
    CHECK(scene->materials().find("PyScene Append Callback Observed Existing Material") != nullptr);
}

TEST_CASE_GPU("native pyscene load failure restores importer and Python search path")
{
    const std::filesystem::path scene_path
        = testing::project_directory() / "tests" / "native" / "render" / "pyscene_native_loading_failure.py";
    ref<Importer> previous_importer = Importer::create();
    ScopedCurrentImporter current_importer(previous_importer);

    PythonContext observer = PythonInterpreter::get().create_context();
    observer.execute_string("import sys\nsaved_sys_path = list(sys.path)");

    try {
        Scene::load(ref(ctx.device), scene_path);
        FAIL("Expected the failing PyScene to throw");
    } catch (const PythonException& e) {
        CHECK(std::string(e.what()).find("intentional native PyScene loading failure") != std::string::npos);
    }

    CHECK_EQ(Importer::get().get(), previous_importer.get());
    observer.execute_string("assert sys.path == saved_sys_path");
}

TEST_CASE_GPU("scene from importer and append run loaded callbacks once")
{
    ref<Importer> importer = Importer::create();
    importer->nodes().create("Recorded Node");
    int callback_count = 0;
    importer->on_scene_loaded(
        [&](ref<Scene> scene)
        {
            ++callback_count;
            CHECK(scene->entities().find("Recorded Node") != nullptr);
        }
    );

    ref<Scene> loaded_scene = Scene::from_importer(ref(ctx.device), *importer);
    CHECK_EQ(callback_count, 1);

    ref<Scene> appended_scene = Scene::create(ref(ctx.device));
    appended_scene->append(*importer);
    CHECK_EQ(callback_count, 2);
}

TEST_CASE_GPU("importer scene camera becomes active and preserves projection properties")
{
    auto importer_scene = make_ref<ImporterScene>();

    ImporterCamera importer_camera;
    importer_camera.name = "Test Camera";
    importer_camera.focal_length = 50.f;
    importer_camera.fstop = 8.f;
    importer_camera.sensor_size_mm = 24.f;
    importer_camera.enable_depth_of_field = true;
    importer_camera.focus_distance = 0.f;
    importer_camera.depth_range = float2(0.25f, 250.f);
    importer_camera.projection = ImporterCamera::Projection::perspective;
    importer_camera.fov_direction = ImporterCamera::FOVDirection::vertical;
    importer_camera.focal_length = ImporterCamera::focal_length_from_fov_degrees(42.f, importer_camera.sensor_size_mm);
    importer_scene->cameras.push_back(importer_camera);

    ImporterNode camera_node;
    camera_node.name = "Test Camera Node";
    camera_node.camera_index = 0;
    importer_scene->nodes.push_back(camera_node);
    importer_scene->root_nodes.push_back(0);

    auto scene = Scene::from_importer_scene(ref(ctx.device), *importer_scene);

    Camera* active_camera = scene->active_camera();
    REQUIRE(active_camera != nullptr);
    CHECK_EQ(active_camera->focal_length(), importer_camera.focal_length);
    CHECK_EQ(active_camera->fstop(), 8.f);
    CHECK_EQ(active_camera->sensor_height(), importer_camera.sensor_size_mm);
    CHECK(active_camera->enable_depth_of_field());
    CHECK_EQ(active_camera->focus_distance(), 0.f);
    CHECK_EQ(active_camera->fov_y(), doctest::Approx(42.f));
    CHECK_EQ(active_camera->depth_range(), importer_camera.depth_range);
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

    auto scene = Scene::from_importer_scene(ref(ctx.device), *importer_scene);

    const float expected_vertical_fov
        = ImporterCamera::fov_degrees_from_focal_length(24.f, importer_camera.focal_length);

    Camera* active_camera = scene->active_camera();
    REQUIRE(active_camera != nullptr);
    CHECK_EQ(active_camera->sensor_height(), 24.f);
    CHECK_EQ(active_camera->focal_length(), importer_camera.focal_length);
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

TEST_CASE_GPU("geometry instance material slots can be assigned and cleared")
{
    auto scene = Scene::create(ref(ctx.device));

    Entity* entity = scene->create_entity();
    GeometryInstance* instance = entity->create_component<GeometryInstance>();
    Geometry* geometry = scene->create_geometry<GeometryGroup>();
    Material* first_material = scene->create_material<StandardMaterial>();
    Material* third_material = scene->create_material<StandardMaterial>();

    CHECK_EQ(instance->material_slot_count(), 0);

    instance->set_geometry(geometry);
    CHECK_EQ(instance->material_slot_count(), 1);

    CHECK(instance->set_material(0, first_material));
    REQUIRE_EQ(instance->materials().size(), 1);
    CHECK_EQ(instance->materials()[0], first_material);
    CHECK_FALSE(instance->set_material(0, first_material));

    CHECK(instance->set_material(2, third_material));
    REQUIRE_EQ(instance->materials().size(), 3);
    CHECK_EQ(instance->materials()[0], first_material);
    CHECK_EQ(instance->materials()[1], nullptr);
    CHECK_EQ(instance->materials()[2], third_material);
    CHECK_EQ(instance->material_slot_count(), 3);

    CHECK(instance->set_material(0, nullptr));
    CHECK_EQ(instance->materials()[0], nullptr);

    CHECK_FALSE(instance->set_material(4, nullptr));
    CHECK_EQ(instance->materials().size(), 3);
}

TEST_CASE_GPU("emissive triangle sampling spans sample batches")
{
    auto scene = Scene::create(ref(ctx.device));

    Properties properties;
    properties.set("emissive_factor", float3(1.f));
    auto* material = scene->create_material<StandardMaterial>(properties);
    auto* geometry
        = create_triangle_geometry(scene, float3(0.f, 0.f, 0.f), float3(1.f, 0.f, 0.f), float3(0.f, 1.f, 0.f));

    // StandardMaterial requests a padded 10,016 samples per triangle. The final
    // triangle starts beyond the 1,048,576-sample batch boundary.
    static constexpr uint32_t INSTANCE_COUNT = 106;
    for (uint32_t i = 0; i < INSTANCE_COUNT; ++i) {
        auto* instance = scene->create_entity()->create_component<GeometryInstance>();
        instance->set_geometry(geometry);
        instance->set_materials({material});
    }

    scene->update();

    const auto* emissive_geometry_system = scene->_emissive_geometry_system();
    CHECK_EQ(emissive_geometry_system->triangle_count(), INSTANCE_COUNT);
    CHECK_EQ(emissive_geometry_system->active_triangle_count(), INSTANCE_COUNT);

    const auto triangles = emissive_geometry_system->triangles();
    const auto triangle_flux = emissive_geometry_system->triangle_flux();
    const auto active_triangle_ids = emissive_geometry_system->active_triangle_ids();
    REQUIRE_EQ(triangles.size(), INSTANCE_COUNT);
    REQUIRE_EQ(triangle_flux.size(), INSTANCE_COUNT);
    REQUIRE_EQ(active_triangle_ids.size(), INSTANCE_COUNT);
    CHECK_GT(triangle_flux.front(), 0.f);
    CHECK_EQ(static_cast<uint32_t>(active_triangle_ids.front()), 0u);

    CHECK_EQ(emissive_geometry_system->triangles().data(), triangles.data());
    CHECK_EQ(emissive_geometry_system->triangle_flux().data(), triangle_flux.data());
    CHECK_EQ(emissive_geometry_system->active_triangle_ids().data(), active_triangle_ids.data());
}

TEST_CASE_GPU("imported planar area light geometry faces local negative Z")
{
    const std::filesystem::path path = testing::get_case_temp_directory() / "visible_planar_lights.usda";
    {
        std::ofstream file(path);
        file << R"usd(#usda 1.0

def RectLight "VisibleRect"
{
    bool inputs:enableColorTemperature = 1
    float inputs:colorTemperature = 3000
    float inputs:height = 2
    float inputs:intensity = 4
    float inputs:width = 3
}

def DiskLight "VisibleDisk"
{
    float inputs:intensity = 2
    float inputs:radius = 1
}
)usd";
    }

    ImportOptions options;
    options.force_rectangle_lights_to_geometry = true;
    options.force_disk_lights_to_geometry = true;
    const ref<ImporterScene> imported_scene = import_scene(path, options);
    REQUIRE(imported_scene);
    REQUIRE(imported_scene->lights.empty());
    REQUIRE_EQ(imported_scene->meshes.size(), 2);

    for (const ImporterMesh& mesh : imported_scene->meshes) {
        const auto positions = mesh.position_stream();
        const auto normals = mesh.normal_stream();
        const auto tangents = mesh.tangent_stream();
        const auto handedness = mesh.handedness_stream();
        REQUIRE(positions.valid());
        REQUIRE(normals.valid());
        REQUIRE(tangents.valid());
        REQUIRE(handedness.valid());
        REQUIRE_EQ(positions.size, normals.size);
        REQUIRE_EQ(positions.size, tangents.size);
        REQUIRE_EQ(positions.size, handedness.size);
        for (size_t vertex_index = 0; vertex_index < positions.size; ++vertex_index) {
            CHECK_EQ(normals[vertex_index], float3(0.f, 0.f, -1.f));
            CHECK_EQ(tangents[vertex_index], float3(1.f, 0.f, 0.f));
            CHECK_EQ(handedness[vertex_index], -1.f);
        }

        REQUIRE_EQ(mesh.subgeometries.size(), 1);
        for (const uint3 indices : mesh.subgeometries[0].indices) {
            const float3 geometric_normal = sgl::math::normalize(
                sgl::math::cross(
                    positions[indices.y] - positions[indices.x],
                    positions[indices.z] - positions[indices.x]
                )
            );
            CHECK_EQ(geometric_normal.x, doctest::Approx(0.f));
            CHECK_EQ(geometric_normal.y, doctest::Approx(0.f));
            CHECK_EQ(geometric_normal.z, doctest::Approx(-1.f));
        }
    }

    const ref<Scene> scene = Scene::from_importer_scene(ref(ctx.device), *imported_scene);
    scene->update();

    REQUIRE_EQ(scene->materials().size(), 2);
    const EmissionOnlyMaterial* color_temperature_material = nullptr;
    for (const Material* scene_material : scene->materials()) {
        const auto* material = dynamic_cast<const EmissionOnlyMaterial*>(scene_material);
        REQUIRE(material != nullptr);
        if (material->enable_color_temperature())
            color_temperature_material = material;
    }
    REQUIRE(color_temperature_material != nullptr);
    CHECK_EQ(color_temperature_material->color_temperature(), doctest::Approx(3000.f));

    REQUIRE(scene->_light_system());
    CHECK_EQ(scene->_light_system()->light_count(), 0);
    CHECK_EQ(scene->_light_system()->analytic_light_count(), 0);

    const EmissiveGeometrySystem* emissive_geometry = scene->_emissive_geometry_system();
    REQUIRE(emissive_geometry);
    CHECK_EQ(emissive_geometry->triangle_count(), 2 + 64);
    CHECK_EQ(emissive_geometry->active_triangle_count(), 2 + 64);
    REQUIRE_EQ(emissive_geometry->triangles().size(), 2 + 64);
    REQUIRE_EQ(emissive_geometry->triangle_flux().size(), 2 + 64);
    for (size_t triangle_index = 0; triangle_index < emissive_geometry->triangles().size(); ++triangle_index) {
        CHECK_EQ(emissive_geometry->triangles()[triangle_index].normal_ws, float3(0.f, 0.f, -1.f));
        CHECK_GT(emissive_geometry->triangle_flux()[triangle_index], 0.f);
    }
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
