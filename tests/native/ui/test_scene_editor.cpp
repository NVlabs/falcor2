// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "testing.h"

#include "falcor2/render/component/camera.h"
#include "falcor2/render/component/geometry_instance.h"
#include "falcor2/render/component/light.h"
#include "falcor2/render/geometry/geometry_group.h"
#include "falcor2/render/geometry/static_mesh_geometry.h"
#include "falcor2/render/material/standard_material.h"
#include "falcor2/render/scene.h"
#include "falcor2/ui/camera_controller.h"
#include "falcor2/ui/scene_editor.h"
#include "falcor2/ui/scene_editor_gizmos.h"
#include "falcor2/ui/scene_editor_outliner.h"

#include <cmath>

using namespace falcor;
using namespace falcor::ui::detail;

TEST_SUITE_BEGIN("SceneEditor");

TEST_CASE_GPU("outliner query filters names and component types")
{
    auto scene = Scene::create(ref(ctx.device));

    Entity* key_light = scene->create_entity();
    key_light->set_name("Key Light");
    key_light->create_component<ConstantLight>();

    Entity* camera = scene->create_entity();
    camera->set_name("Main Camera");
    camera->create_component<Camera>();

    Geometry* geometry = scene->create_geometry<GeometryGroup>();
    geometry->set_name("Key Mesh");
    Material* material = scene->create_material<StandardMaterial>();
    material->set_name("Key Material");

    CHECK(OutlinerQuery("key").matches_entity(key_light));
    CHECK(OutlinerQuery("KEY LIGHT").matches_entity(key_light));
    CHECK(OutlinerQuery("t:light").matches_entity(key_light));
    CHECK(OutlinerQuery("key t:Light").matches_entity(key_light));
    CHECK_FALSE(OutlinerQuery("fill t:Light").matches_entity(key_light));
    CHECK_FALSE(OutlinerQuery("t:Camera").matches_entity(key_light));
    CHECK(OutlinerQuery("main t:camera").matches_entity(camera));

    CHECK(OutlinerQuery("t:").empty());
    CHECK(OutlinerQuery("key mesh").matches_object(geometry));
    CHECK(OutlinerQuery("KEY MATERIAL").matches_object(material));
    CHECK_FALSE(OutlinerQuery("camera").matches_object(geometry));
    CHECK_FALSE(OutlinerQuery("t:Light").matches_object(geometry));
}

TEST_CASE_GPU("outliner query keeps ancestors of matching entities visible")
{
    auto scene = Scene::create(ref(ctx.device));

    Entity* root = scene->create_entity();
    root->set_name("Root");
    Entity* light = scene->create_entity();
    light->set_name("Sun");
    light->set_parent(root);
    light->create_component<DistantLight>();
    Entity* sibling = scene->create_entity();
    sibling->set_name("Camera");
    sibling->set_parent(root);
    sibling->create_component<Camera>();

    std::unordered_set<const Entity*> visible_entities;
    CHECK(OutlinerQuery("t:Light").collect_visible_entities(root, visible_entities));
    CHECK(visible_entities.contains(root));
    CHECK(visible_entities.contains(light));
    CHECK_FALSE(visible_entities.contains(sibling));

    Entity* fill = scene->create_entity();
    fill->set_name("Fill");
    fill->set_parent(light);
    CHECK(is_entity_ancestor(root, fill));
    CHECK(is_entity_ancestor(light, fill));
    CHECK_FALSE(is_entity_ancestor(fill, fill));
    CHECK_FALSE(is_entity_ancestor(sibling, fill));
    CHECK_FALSE(is_entity_ancestor(nullptr, fill));
    CHECK_FALSE(is_entity_ancestor(root, nullptr));
}

TEST_CASE_GPU("outliner name sorting is stable")
{
    auto scene = Scene::create(ref(ctx.device));

    Entity* zulu = scene->create_entity();
    zulu->set_name("Zulu");
    Entity* alpha_first = scene->create_entity();
    alpha_first->set_name("Alpha");
    Entity* alpha_second = scene->create_entity();
    alpha_second->set_name("Alpha");

    std::vector<Entity*> entities{zulu, alpha_first, alpha_second};
    sort_outliner_objects(entities);

    CHECK_EQ(entities, std::vector<Entity*>{alpha_first, alpha_second, zulu});
}

TEST_CASE("outliner filtering preserves tree expansion state")
{
    OutlinerTreeOpenState collapsed = outliner_tree_open_state(false, true);
    CHECK(collapsed.draw_open);
    CHECK_FALSE(collapsed.stored_open);

    OutlinerTreeOpenState expanded = outliner_tree_open_state(true, true);
    CHECK(expanded.draw_open);
    CHECK(expanded.stored_open);

    OutlinerTreeOpenState unfiltered = outliner_tree_open_state(false, false);
    CHECK_FALSE(unfiltered.draw_open);
    CHECK_FALSE(unfiltered.stored_open);

    OutlinerTreeOpenState revealed = outliner_tree_open_state(false, false, true);
    CHECK(revealed.draw_open);
    CHECK(revealed.stored_open);

    OutlinerTreeOpenState filtered_and_revealed = outliner_tree_open_state(false, true, true);
    CHECK(filtered_and_revealed.draw_open);
    CHECK(filtered_and_revealed.stored_open);
}

TEST_CASE_GPU("editor gizmo projection maps world points to viewport coordinates")
{
    auto scene = Scene::create(ref(ctx.device));
    Entity* entity = scene->create_entity();

    const float4x4 identity = float4x4::identity();
    auto center = project_gizmo(entity, float3(0.f), identity, float2(10.f, 20.f), float2(200.f, 100.f));
    REQUIRE(center.has_value());
    CHECK(center->entity == entity);
    CHECK(center->screen_position.x == doctest::Approx(110.f));
    CHECK(center->screen_position.y == doctest::Approx(70.f));
    CHECK(center->depth == doctest::Approx(0.f));

    auto upper_right = project_gizmo(entity, float3(1.f, 1.f, 0.f), identity, float2(10.f, 20.f), float2(200.f, 100.f));
    REQUIRE(upper_right.has_value());
    CHECK(upper_right->screen_position.x == doctest::Approx(210.f));
    CHECK(upper_right->screen_position.y == doctest::Approx(20.f));

    auto just_outside = project_gizmo(entity, float3(1.01f, 0.f, 0.f), identity, float2(0.f), float2(100.f));
    REQUIRE(just_outside.has_value());
    CHECK(just_outside->screen_position.x == doctest::Approx(100.5f));
    CHECK_FALSE(project_gizmo(entity, float3(0.f, 0.f, 1.01f), identity, float2(0.f), float2(100.f)));

    float4x4 negative_w = float4x4::identity();
    negative_w.set_row(3, float4(0.f, 0.f, 0.f, -1.f));
    CHECK_FALSE(project_gizmo(entity, float3(0.f), negative_w, float2(0.f), float2(100.f)));
}

TEST_CASE("editor gizmo line segments are clipped instead of discarded at viewport edges")
{
    const float4x4 identity = float4x4::identity();
    auto horizontal = project_gizmo_segment(
        float3(-2.f, 0.f, 0.5f),
        float3(2.f, 0.f, 0.5f),
        identity,
        float2(10.f, 20.f),
        float2(200.f, 100.f)
    );
    REQUIRE(horizontal.has_value());
    CHECK((*horizontal)[0].x == doctest::Approx(10.f));
    CHECK((*horizontal)[0].y == doctest::Approx(70.f));
    CHECK((*horizontal)[1].x == doctest::Approx(210.f));
    CHECK((*horizontal)[1].y == doctest::Approx(70.f));

    auto through_near_plane = project_gizmo_segment(
        float3(-0.5f, 0.f, -1.f),
        float3(0.5f, 0.f, 0.5f),
        identity,
        float2(0.f),
        float2(100.f)
    );
    REQUIRE(through_near_plane.has_value());
    CHECK((*through_near_plane)[0].x == doctest::Approx(58.3333f));
    CHECK((*through_near_plane)[0].y == doctest::Approx(50.f));
    CHECK((*through_near_plane)[1].x == doctest::Approx(75.f));
    CHECK((*through_near_plane)[1].y == doctest::Approx(50.f));

    CHECK_FALSE(
        project_gizmo_segment(float3(-2.f, 2.f, 0.5f), float3(2.f, 2.f, 0.5f), identity, float2(0.f), float2(100.f))
    );
    CHECK_FALSE(
        project_gizmo_segment(float3(-0.5f, 0.f, -1.f), float3(0.5f, 0.f, -0.5f), identity, float2(0.f), float2(100.f))
    );
}

TEST_CASE("light shaping gizmo supports the full cone angle range")
{
    auto check_profile = [](float angle, float expected_axial_offset, float expected_radius)
    {
        const LightGizmoConeProfile profile = make_light_gizmo_cone_profile(2.f, angle);
        CHECK(profile.axial_offset == doctest::Approx(expected_axial_offset).epsilon(1e-5));
        CHECK(profile.radius == doctest::Approx(expected_radius).epsilon(1e-5));
    };

    check_profile(0.f, 2.f, 0.f);
    check_profile(90.f, 0.f, 2.f);
    check_profile(120.f, -1.f, std::sqrt(3.f));
    check_profile(180.f, -2.f, 0.f);
}

TEST_CASE_GPU("editor gizmo hit testing prefers screen distance then depth")
{
    auto scene = Scene::create(ref(ctx.device));
    Entity* near_entity = scene->create_entity();
    Entity* far_entity = scene->create_entity();
    Entity* offset_entity = scene->create_entity();

    std::vector<ProjectedGizmo> gizmos{
        {.entity = far_entity, .screen_position = float2(50.f), .depth = 0.8f, .hit_radius = 12.f},
        {.entity = near_entity, .screen_position = float2(50.f), .depth = 0.2f, .hit_radius = 12.f},
        {.entity = offset_entity, .screen_position = float2(55.f, 50.f), .depth = 0.1f, .hit_radius = 12.f},
    };

    CHECK(pick_gizmo(gizmos, float2(50.f)) == near_entity);
    CHECK(pick_gizmo(gizmos, float2(52.f, 50.f)) == near_entity);
    CHECK(pick_gizmo(gizmos, float2(54.f, 50.f)) == offset_entity);
    CHECK(pick_gizmo(gizmos, float2(67.f, 50.f)) == offset_entity);
    CHECK(pick_gizmo(gizmos, float2(68.f, 50.f)) == nullptr);
}

TEST_CASE_GPU("camera gizmo frustum follows camera field of view and aspect ratio")
{
    auto scene = Scene::create(ref(ctx.device));
    Entity* entity = scene->create_entity();
    Camera* camera = entity->create_component<Camera>();
    camera->set_width(200);
    camera->set_height(100);
    camera->set_fov_y(90.f);

    const CameraGizmoFrustum frustum = make_camera_gizmo_frustum(camera, 1.f);
    CHECK_EQ(frustum.origin, float3(0.f));
    const std::array<float3, 4> expected{
        float3(-2.f, -1.f, -1.f),
        float3(2.f, -1.f, -1.f),
        float3(2.f, 1.f, -1.f),
        float3(-2.f, 1.f, -1.f),
    };
    for (size_t i = 0; i < expected.size(); ++i) {
        CHECK(frustum.corners[i].x == doctest::Approx(expected[i].x));
        CHECK(frustum.corners[i].y == doctest::Approx(expected[i].y));
        CHECK(frustum.corners[i].z == doctest::Approx(expected[i].z));
    }
}

TEST_CASE_GPU("frame selected command updates the active camera")
{
    auto scene = Scene::create(ref(ctx.device));

    Entity* camera_entity = scene->create_entity();
    Camera* camera = camera_entity->create_component<Camera>();
    scene->set_active_camera(camera);

    const float3 positions[]{{-1.f, -1.f, 0.f}, {1.f, -1.f, 0.f}, {0.f, 1.f, 0.f}};
    const uint32_t indices[]{0, 1, 2};
    StaticMeshGeometryDataDesc mesh_desc{
        .name = "triangle",
        .vertex_count = 3,
        .position_stream = {positions, sizeof(float3)},
        .sub_meshes = {
            {
                .name = "triangle",
                .index_count = 3,
                .index_stream = {indices, sizeof(uint32_t)},
            },
        },
    };
    StaticMeshGeometry* geometry = scene->create_geometry<StaticMeshGeometry>();
    geometry->set_mesh_data(mesh_desc);
    StandardMaterial* material = scene->create_material<StandardMaterial>();
    Entity* selected_entity = scene->create_entity();
    GeometryInstance* instance = selected_entity->create_component<GeometryInstance>();
    instance->set_geometry(geometry);
    instance->set_material(0, material);

    auto camera_controller = make_ref<ui::CameraController>();
    auto editor = make_ref<ui::SceneEditor>();
    editor->set_scene(scene);
    editor->set_camera_controller(camera_controller);
    editor->set_selected_object(selected_entity);

    // Clear initial scene creation dirtiness so the next update only reflects
    // the camera change made by framing the selection.
    scene->update();

    CHECK(editor->handle_keyboard_shortcut({sgl::KeyboardEventType::key_press, sgl::KeyCode::f}));
    CHECK_EQ(camera_controller->pivot(), float3(0.f));
    CHECK(camera_entity->transform().translation().x == doctest::Approx(0.f));
    CHECK(camera_entity->transform().translation().y == doctest::Approx(0.f));
    CHECK(camera_entity->transform().translation().z == doctest::Approx(std::sqrt(8.f) * 1.5f));
    CHECK_EQ(scene->active_camera(), camera);

    const SceneUpdateFlags update_flags = scene->update();
    CHECK(is_set(update_flags, SceneUpdateFlags::transforms));
    CHECK(is_set(update_flags, SceneUpdateFlags::render_state));
}

TEST_SUITE_END();
