// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "testing.h"

#include "falcor2/render/component/camera.h"
#include "falcor2/render/component/geometry_instance.h"
#include "falcor2/render/component/light.h"
#include "falcor2/render/component/transform_animation.h"
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

TEST_CASE_GPU("editor selection getter ignores a child removed externally")
{
    auto scene = Scene::create(ref(ctx.device));
    Entity* parent = scene->create_entity();
    Entity* child = scene->create_entity();
    child->set_parent(parent);
    scene->update();

    auto editor = make_ref<ui::SceneEditor>();
    editor->set_scene(scene);
    editor->set_selected_object(child);
    CHECK_EQ(editor->selected_object(), child);

    parent->remove();
    CHECK_EQ(editor->selected_object(), nullptr);
    scene->update();

    CHECK_EQ(editor->selected_object(), nullptr);
}

TEST_CASE_GPU("escape clears selection without removing the object")
{
    auto scene = Scene::create(ref(ctx.device));
    Entity* entity = scene->create_entity();
    auto controller = make_ref<ui::CameraController>();
    auto editor = make_ref<ui::SceneEditor>();
    editor->set_scene(scene);
    editor->set_camera_controller(controller);
    editor->set_selected_object(entity);

    sgl::MouseEvent event{};
    event.type = sgl::MouseEventType::button_down;
    event.button = sgl::MouseButton::right;
    REQUIRE(controller->handle_mouse_event(event));
    REQUIRE(controller->is_interacting());
    CHECK_FALSE(editor->handle_keyboard_shortcut({sgl::KeyboardEventType::key_press, sgl::KeyCode::escape}));
    CHECK_EQ(editor->selected_object(), entity);

    controller->cancel_interaction();
    CHECK(editor->handle_keyboard_shortcut({sgl::KeyboardEventType::key_press, sgl::KeyCode::escape}));
    CHECK_EQ(editor->selected_object(), nullptr);
    CHECK(entity->is_valid());
    CHECK_EQ(entity->scene(), scene.get());
    CHECK_FALSE(editor->handle_keyboard_shortcut({sgl::KeyboardEventType::key_press, sgl::KeyCode::escape}));
}

TEST_CASE_GPU("frame selected uses entity position when geometry bounds are unavailable")
{
    auto scene = Scene::create(ref(ctx.device));
    Entity* view_entity = scene->create_entity();
    Camera* view_camera = view_entity->create_component<Camera>();
    view_camera->set_focus_distance(5.f);
    scene->set_active_camera(view_camera);

    Entity* empty_entity = scene->create_entity();
    Entity* light_entity = scene->create_entity();
    light_entity->create_component<ConstantLight>();
    Entity* camera_entity = scene->create_entity();
    camera_entity->create_component<Camera>();

    const std::array<Entity*, 3> targets{empty_entity, light_entity, camera_entity};
    const std::array<float3, 3> positions{
        float3(1.f, 2.f, 3.f),
        float3(-4.f, 5.f, 6.f),
        float3(7.f, -8.f, 9.f),
    };
    for (size_t i = 0; i < targets.size(); ++i) {
        Transform transform;
        transform.set_translation(positions[i]);
        targets[i]->set_transform(transform);
    }

    auto camera_controller = make_ref<ui::CameraController>();
    auto editor = make_ref<ui::SceneEditor>();
    editor->set_scene(scene);
    editor->set_camera_controller(camera_controller);

    for (size_t i = 0; i < targets.size(); ++i) {
        editor->set_selected_object(targets[i]);
        CHECK(editor->frame_selected(view_camera));
        CHECK_EQ(camera_controller->pivot(), positions[i]);
        CHECK(camera_controller->orbit_distance() == doctest::Approx(5.f));
    }
}

TEST_CASE_GPU("editor object templates create named empty entities and materials")
{
    auto scene = Scene::create(ref(ctx.device));
    auto editor = make_ref<ui::SceneEditor>();
    editor->set_scene(scene);

    const uint64_t initial_version = editor->selection_version();
    Entity* entity = editor->create_object("entity")->as<Entity>();
    REQUIRE(entity);
    CHECK_EQ(entity->name(), "Empty Entity");
    CHECK(entity->components().empty());
    CHECK_EQ(entity->parent(), nullptr);
    CHECK_EQ(entity->transform().translation(), float3(0.f));
    CHECK_EQ(editor->selected_object(), entity);
    CHECK_EQ(editor->selection_version(), initial_version + 1);

    struct MaterialCase {
        std::string_view type;
        std::string_view name;
    };
    constexpr std::array materials{
        MaterialCase{"EmissionOnlyMaterial", "Emission Only Material"},
        MaterialCase{"OpenPBRMaterial", "OpenPBR Material"},
        MaterialCase{"StandardMaterial", "Standard Material"},
        MaterialCase{"StandardSpecGlossMaterial", "Standard Specular-Glossiness Material"},
        MaterialCase{"UnlitMaterial", "Unlit Material"},
    };
    for (const auto& entry : materials) {
        CAPTURE(entry.type);
        const uint64_t version = editor->selection_version();
        Material* material = editor->create_object(std::string("material.").append(entry.type))->as<Material>();
        REQUIRE(material);
        CHECK_EQ(editor->selected_object(), material);
        CHECK_EQ(editor->selection_version(), version + 1);
        CHECK_EQ(std::string_view(material->class_name()), entry.type);
        CHECK_EQ(material->name(), entry.name);
        CHECK_EQ(material->scene(), scene.get());
    }
    CHECK_EQ(editor->selection_version(), initial_version + 1 + materials.size());

    SceneObject* selected = editor->selected_object();
    CHECK_THROWS(editor->create_object("material.MDLMaterial"));
    CHECK_THROWS(editor->create_object("material.MaterialXMaterial"));
    CHECK_EQ(editor->selected_object(), selected);
    CHECK_EQ(editor->selection_version(), initial_version + 1 + materials.size());
    CHECK_EQ(scene->materials().size(), materials.size());
    CHECK_EQ(scene->entities().size(), 2);
    CHECK_EQ(scene->components().size(), 1);
}

TEST_CASE_GPU("editor entity templates create and select complete objects")
{
    struct TemplateCase {
        std::string_view key;
        std::string_view name;
        std::string_view component_type;
    };
    constexpr std::array cases{
        TemplateCase{"camera", "Camera", "Camera"},
        TemplateCase{"point_light", "Point Light", "PointLight"},
        TemplateCase{"spot_light", "Spot Light", "PointLight"},
        TemplateCase{"distant_light", "Distant Light", "DistantLight"},
        TemplateCase{"rect_light", "Rect Light", "RectLight"},
        TemplateCase{"disk_light", "Disk Light", "DiskLight"},
        TemplateCase{"sphere_light", "Sphere Light", "SphereLight"},
        TemplateCase{"cylinder_light", "Cylinder Light", "CylinderLight"},
        TemplateCase{"constant_light", "Constant Light", "ConstantLight"},
        TemplateCase{"environment_light", "Environment Map", "EnvMapLight"},
    };

    auto scene = Scene::create(ref(ctx.device));
    Entity* view_entity = scene->create_entity();
    Camera* view_camera = view_entity->create_component<Camera>();
    scene->set_active_camera(view_camera);
    auto editor = make_ref<ui::SceneEditor>();
    editor->set_scene(scene);

    for (const auto& entry : cases) {
        CAPTURE(entry.name);
        const uint64_t version = editor->selection_version();
        Entity* entity = editor->create_object(entry.key)->as<Entity>();
        REQUIRE(entity);
        CHECK_EQ(editor->selected_object(), entity);
        CHECK_EQ(editor->selection_version(), version + 1);
        CHECK_EQ(entity->name(), entry.name);
        CHECK_EQ(entity->scene(), scene.get());
        CHECK_EQ(entity->parent(), nullptr);
        CHECK_EQ(entity->transform().translation(), float3(0.f));
        REQUIRE_EQ(entity->components().size(), 1);
        Component* component = entity->components().front();
        CHECK_EQ(component->entity(), entity);
        CHECK_EQ(std::string_view(component->class_name()), entry.component_type);
        CHECK_EQ(scene->active_camera(), view_camera);
        if (auto* point = component->as<PointLight>()) {
            CHECK_EQ(point->enable_shaping(), entry.key == "spot_light");
            if (entry.key == "spot_light") {
                CHECK_EQ(point->shaping_cone_angle(), 30.f);
                CHECK_EQ(point->shaping_cone_softness(), 0.2f);
            }
        }
    }
    CHECK_EQ(scene->entities().size(), cases.size() + 2);
    scene->update();
    CHECK_EQ(scene->active_camera(), view_camera);
}

TEST_CASE_GPU("editor camera template participates in automatic camera activation")
{
    auto scene = Scene::create(ref(ctx.device));
    auto editor = make_ref<ui::SceneEditor>();
    editor->set_scene(scene);
    Entity* entity = editor->create_object("camera")->as<Entity>();
    REQUIRE(entity);
    Camera* camera = entity->components().front()->as<Camera>();
    REQUIRE(camera);
    scene->update();
    CHECK_EQ(scene->active_camera(), camera);
}

TEST_CASE_GPU("editor environment template updates without an assigned texture")
{
    auto scene = Scene::create(ref(ctx.device));
    auto editor = make_ref<ui::SceneEditor>();
    editor->set_scene(scene);
    const uint64_t version = editor->selection_version();
    Entity* entity = editor->create_object("environment_light")->as<Entity>();
    REQUIRE(entity);
    CHECK_EQ(entity->name(), "Environment Map");
    CHECK_EQ(editor->selected_object(), entity);
    CHECK_EQ(editor->selection_version(), version + 1);
    REQUIRE_EQ(entity->components().size(), 1);
    auto* light = entity->components().front()->as<EnvMapLight>();
    REQUIRE(light);
    CHECK_EQ(light->entity(), entity);
    CHECK(light->env_map_path().empty());
    CHECK(light->active());
    scene->update();
    scene->update();
    CHECK_EQ(editor->selected_object(), entity);
}

TEST_CASE_GPU("editor rejects invalid creation requests without changing selection")
{
    auto scene = Scene::create(ref(ctx.device));
    auto editor = make_ref<ui::SceneEditor>();
    CHECK_THROWS(editor->create_object("entity"));
    CHECK_THROWS(editor->create_object("camera"));
    CHECK_THROWS(editor->create_object("environment_light"));
    CHECK_THROWS(editor->create_object("material.StandardMaterial"));
    CHECK_EQ(editor->selection_version(), 0);
    editor->set_scene(scene);
    SceneObject* selected = editor->create_object("entity");
    const uint64_t version = editor->selection_version();
    CHECK_THROWS(editor->create_object("unknown"));
    CHECK_THROWS(editor->create_object(""));
    CHECK_THROWS(editor->create_object("Camera"));
    CHECK_THROWS(editor->create_object("material.Unknown"));
    CHECK_EQ(scene->entities().size(), 2);
    CHECK_EQ(scene->components().size(), 1);
    CHECK_EQ(scene->materials().size(), 0);
    CHECK_EQ(editor->selected_object(), selected);
    CHECK_EQ(editor->selection_version(), version);
}

TEST_CASE_GPU("frame selected command updates the active camera")
{
    auto scene = Scene::create(ref(ctx.device));

    Entity* camera_entity = scene->create_entity();
    Camera* camera = camera_entity->create_component<Camera>();
    scene->set_active_camera(camera);

    const float3 positions[]{{-1.f, -1.f, 0.f}, {1.f, -1.f, 0.f}, {0.f, 1.f, 0.f}};
    const uint32_t indices[]{0, 1, 2};
    const MeshVertexAttributeDesc vertex_attributes[]{
        {"POSITION",
         MeshAttributeSemantic::position,
         0,
         sgl::DataType::float32,
         3,
         positions,
         sizeof(float3),
         sizeof(positions)},
    };
    StaticMeshGeometryDataDesc mesh_desc{
        .name = "triangle",
        .vertex_count = 3,
        .vertex_attributes = vertex_attributes,
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

    SUBCASE("zero focus distance uses scene scale for initial navigation")
    {
        camera->set_focus_distance(0.f);
        editor->update_camera(0.1f, 320, 240);
        CHECK(camera_controller->orbit_distance() == doctest::Approx(std::sqrt(8.f)));
        CHECK_EQ(camera->focus_distance(), 0.f);
    }
    SUBCASE("positive focus distance") { }

    CHECK(editor->handle_keyboard_shortcut({sgl::KeyboardEventType::key_press, sgl::KeyCode::f}));
    CHECK_EQ(camera_controller->pivot(), float3(0.f));
    CHECK(camera_entity->transform().translation().x == doctest::Approx(0.f));
    CHECK(camera_entity->transform().translation().y == doctest::Approx(0.f));
    CHECK(camera_entity->transform().translation().z == doctest::Approx(std::sqrt(8.f) * 1.5f));
    CHECK_EQ(scene->active_camera(), camera);

    editor->update_camera(0.1f, 320, 240);
    CHECK_EQ(camera_controller->pivot(), float3(0.f));
    CHECK(camera_controller->orbit_distance() == doctest::Approx(std::sqrt(8.f) * 1.5f));

    const SceneUpdateFlags update_flags = scene->update();
    CHECK(is_set(update_flags, SceneUpdateFlags::transforms));
    CHECK(is_set(update_flags, SceneUpdateFlags::render_state));
}

TEST_CASE_GPU("camera preparation frames once and preserves later camera edits")
{
    auto scene = Scene::create(ref(ctx.device));
    auto editor = make_ref<ui::SceneEditor>();
    auto controller = make_ref<ui::CameraController>();
    SUBCASE("with navigation")
    {
        editor->set_camera_controller(controller);
    }
    SUBCASE("without navigation") { }
    editor->set_scene(scene);
    Transform transform;
    transform.set_translation(float3(10.f, 20.f, 30.f));
    scene->free_camera()->entity()->set_transform(transform);
    editor->request_initial_camera_framing();
    scene->update();

    auto* camera = editor->update_camera(0.1f, 320, 240);
    REQUIRE_EQ(camera, scene->free_camera());
    CHECK_EQ(camera->entity()->transform().translation(), float3(0.f, 0.f, 5.f));
    CHECK_EQ(camera->width(), 320);
    CHECK_EQ(camera->height(), 240);
    if (editor->camera_controller())
        CHECK_EQ(controller->pivot(), float3(0.f));

    // Initial framing commits the free-camera selection across later camera creation.
    scene->create_entity()->create_component<Camera>();
    camera->entity()->set_transform(transform);
    scene->update();
    CHECK_EQ(editor->update_camera(0.1f, 240, 640), camera);
    CHECK_EQ(camera->entity()->transform().translation(), transform.translation());
    CHECK_EQ(camera->width(), 240);
    CHECK_EQ(camera->height(), 640);

    // A pending request belongs only to the attached scene; new attachments opt in again.
    editor->request_initial_camera_framing();
    auto next_scene = Scene::create(ref(ctx.device));
    next_scene->free_camera()->entity()->set_transform(transform);
    editor->set_scene(next_scene);
    next_scene->update();
    CHECK_EQ(editor->update_camera(0.1f, 320, 240), next_scene->free_camera());
    CHECK_EQ(next_scene->free_camera()->entity()->transform().translation(), transform.translation());
    editor->request_initial_camera_framing();
    editor->update_camera(0.1f, 320, 240);
    CHECK_EQ(next_scene->free_camera()->entity()->transform().translation(), float3(0.f, 0.f, 5.f));

    sgl::MouseEvent event{};
    event.type = sgl::MouseEventType::button_down;
    event.button = sgl::MouseButton::right;
    controller->handle_mouse_event(event);
    editor->set_scene(nullptr);
    editor->request_initial_camera_framing();
    CHECK_EQ(editor->update_camera(0.1f, 320, 240), nullptr);
    if (editor->camera_controller())
        CHECK_FALSE(controller->is_interacting());
}

TEST_CASE_GPU("camera preparation preserves authored cameras and does not reframe on deletion")
{
    auto scene = Scene::create(ref(ctx.device));
    auto* camera = scene->create_entity()->create_component<Camera>();
    Transform transform;
    transform.set_translation(float3(10.f, 20.f, 30.f));
    camera->entity()->set_transform(transform);
    scene->free_camera()->entity()->set_transform(transform);
    auto editor = make_ref<ui::SceneEditor>();
    editor->set_scene(scene);
    editor->request_initial_camera_framing();
    scene->update();

    CHECK_EQ(editor->update_camera(0.1f, 640, 480), camera);
    CHECK_EQ(camera->entity()->transform().translation(), transform.translation());
    CHECK_EQ(camera->width(), 640);
    CHECK_EQ(camera->height(), 480);
    editor->set_view_camera(scene->free_camera());
    CHECK_EQ(scene->active_camera(), camera);
    editor->set_view_camera(nullptr);
    CHECK_EQ(scene->active_camera(), camera);

    camera->entity()->remove();
    scene->update();
    CHECK_EQ(editor->update_camera(0.1f, 240, 320), scene->free_camera());
    CHECK_EQ(scene->free_camera()->entity()->transform().translation(), transform.translation());
    CHECK_EQ(scene->free_camera()->width(), 240);
    CHECK_EQ(scene->free_camera()->height(), 320);
}

TEST_CASE_GPU("camera navigation follows explicit reframing without restoring old inertia")
{
    auto scene = Scene::create(ref(ctx.device));
    auto* camera = scene->free_camera();
    Transform transform;
    transform.set_translation(float3(10.f, 0.f, 5.f));
    camera->entity()->set_transform(transform);
    scene->update();
    auto controller = make_ref<ui::CameraController>();
    auto editor = make_ref<ui::SceneEditor>();
    editor->set_scene(scene);
    editor->set_camera_controller(controller);
    editor->update_camera(0.1f, 320, 240);

    sgl::MouseEvent scroll{};
    scroll.type = sgl::MouseEventType::scroll;
    scroll.scroll = float2(0.f, 1.f);
    controller->handle_mouse_event(scroll);
    editor->update_camera(0.1f, 320, 240);
    editor->request_initial_camera_framing();
    controller->handle_mouse_event(scroll);
    editor->update_camera(0.1f, 320, 240);
    const float3 position = camera->entity()->transform().translation();
    CHECK_EQ(position.x, 0.f);
    CHECK_EQ(position.z, doctest::Approx(5.f * std::exp(-0.3f)));
    CHECK_EQ(controller->pivot(), float3(0.f));

    // Ordinary navigation must retain its pivot even though orbit and focus distances differ.
    editor->update_camera(0.01f, 320, 240);
    CHECK_EQ(controller->pivot(), float3(0.f));
    const float distance = controller->orbit_distance();
    camera->set_focus_distance(12.f);
    editor->update_camera(0.01f, 320, 240);
    // An optical focus edit must neither move the pivot nor stop the existing dolly inertia.
    CHECK_EQ(controller->pivot(), float3(0.f));
    CHECK_LT(controller->orbit_distance(), distance);

    controller->reset_inertia();
    const float orbit_distance = controller->orbit_distance();
    const float3 moved_position = camera->entity()->transform().translation();
    camera->set_focus_distance(0.f);
    editor->update_camera(0.1f, 320, 240);
    CHECK_EQ(controller->orbit_distance(), orbit_distance);
    CHECK_EQ(controller->pivot(), float3(0.f));
    CHECK_EQ(camera->entity()->transform().translation(), moved_position);

    // External transforms move the pivot with the camera without restoring optical focus as the orbit distance.
    transform.set_translation(float3(10.f, 20.f, 30.f));
    camera->entity()->set_transform(transform);
    scene->update();
    editor->update_camera(0.1f, 320, 240);
    CHECK_EQ(controller->transform().translation(), transform.translation());
    CHECK_EQ(controller->orbit_distance(), orbit_distance);
    CHECK_EQ(controller->pivot(), float3(10.f, 20.f, 30.f - orbit_distance));
}

TEST_CASE_GPU("camera switching clears inertia but preserves pending input")
{
    auto scene = Scene::create(ref(ctx.device));
    auto* first = scene->create_entity()->create_component<Camera>();
    auto* second = scene->create_entity()->create_component<Camera>();
    first->set_focus_distance(0.f);
    second->set_focus_distance(12.f);
    scene->update();
    auto controller = make_ref<ui::CameraController>();
    auto editor = make_ref<ui::SceneEditor>();
    editor->set_scene(scene);
    editor->set_camera_controller(controller);
    editor->update_camera(0.1f, 320, 240);

    // An empty scene with zero optical focus still has a useful navigation distance.
    CHECK_EQ(controller->orbit_distance(), 5.f);
    CHECK_EQ(first->focus_distance(), 0.f);

    sgl::MouseEvent scroll{};
    scroll.type = sgl::MouseEventType::scroll;
    scroll.scroll = float2(0.f, 1.f);
    controller->handle_mouse_event(scroll);
    editor->update_camera(0.1f, 320, 240);
    CHECK_NE(first->entity()->transform().translation().z, 0.f);
    scene->set_active_camera(second);
    editor->update_camera(0.1f, 320, 240);
    CHECK_EQ(second->entity()->transform().translation(), float3(0.f));
    CHECK_EQ(controller->orbit_distance(), 12.f);

    // A gesture received before the first update of a newly selected camera still applies.
    scene->set_active_camera(scene->free_camera());
    controller->handle_mouse_event(scroll);
    editor->update_camera(0.1f, 320, 240);
    CHECK_LT(scene->free_camera()->entity()->transform().translation().z, 5.f);
    const float3 position = scene->free_camera()->entity()->transform().translation();
    scene->update();
    CHECK_EQ(scene->active_camera(), scene->free_camera());
    CHECK_EQ(scene->free_camera()->entity()->transform().translation(), position);
}

TEST_CASE_GPU("camera navigation and framing preserve animated cameras and edit static cameras")
{
    for (const char* animation_target : {"none", "self", "ancestor", "disconnected"}) {
        SUBCASE(animation_target)
        {
            auto scene = Scene::create(ref(ctx.device));
            auto* parent = scene->create_entity();
            auto* middle = scene->create_entity();
            middle->set_parent(parent);
            auto* entity = scene->create_entity();
            entity->set_parent(middle);
            Transform local;
            local.set_translation(float3(1.f, 2.f, 5.f));
            entity->set_transform(local);
            auto* camera = entity->create_component<Camera>();
            camera->set_focal_length(35.f);
            camera->set_sensor_height(30.f);
            camera->set_fstop(2.f);
            camera->set_exposure(1.5f);
            camera->set_focus_distance(9.f);
            camera->set_enable_depth_of_field(true);
            camera->set_depth_range(float2(0.02f, 300.f));
            camera->set_width(320);
            camera->set_height(240);
            const std::string_view target = animation_target;
            const bool animated = target == "self" || target == "ancestor";
            if (target != "none") {
                auto channel = make_ref<AnimationChannel>();
                channel->set_times({0.f, 1.f});
                channel->set_values({0.f, 0.f, 0.f, 10.f, 0.f, 0.f});
                auto* animation = scene->create_animation();
                animation->set_channels({channel});
                auto* binding = (target == "self" ? entity : parent)->create_component<TransformAnimation>();
                binding->set_animation(animation);
                binding->set_translation_channel(0);
                if (target == "disconnected")
                    binding->set_animation(nullptr);
            }
            auto controller = make_ref<ui::CameraController>();
            controller->set_smoothing(false);
            auto editor = make_ref<ui::SceneEditor>();
            editor->set_scene(scene);
            editor->set_camera_controller(controller);
            scene->set_time(0.25);
            scene->update();
            CHECK_EQ(editor->update_camera(0.1f, 320, 240), camera);
            scene->set_time(0.5);
            scene->update();
            CHECK_EQ(editor->update_camera(0.1f, 320, 240), camera);
            float3 world_position = local.translation();
            if (target == "self")
                world_position = float3(5.f, 0.f, 0.f);
            else if (target == "ancestor")
                world_position = float3(6.f, 2.f, 5.f);
            CHECK_EQ(controller->transform().translation(), world_position);
            const float3 local_position = entity->transform().translation();
            Camera* expected_camera = animated ? scene->free_camera() : camera;

            SUBCASE("scroll while paused starts from the evaluated view")
            {
                sgl::MouseEvent scroll{};
                scroll.type = sgl::MouseEventType::scroll;
                scroll.scroll = float2(0.f, 1.f);
                controller->handle_mouse_event(scroll);
                CHECK_EQ(editor->update_camera(0.1f, 320, 240), expected_camera);
                CHECK_EQ(expected_camera->entity()->transform().translation().x, world_position.x);
                CHECK_LT(expected_camera->entity()->transform().translation().z, world_position.z);
            }
            SUBCASE("frame selected shortcut while paused shares the navigation policy")
            {
                auto* selected_entity = scene->create_entity();
                editor->set_selected_object(selected_entity);
                CHECK(editor->handle_keyboard_shortcut({sgl::KeyboardEventType::key_press, sgl::KeyCode::f}));
                CHECK_EQ(editor->update_camera(0.1f, 320, 240), expected_camera);
            }
            CHECK_EQ(scene->active_camera(), expected_camera);
            if (animated) {
                auto* free_camera = scene->free_camera();
                CHECK_EQ(entity->transform().translation(), local_position);
                CHECK_EQ(free_camera->entity()->parent(), nullptr);
                CHECK_EQ(free_camera->focal_length(), camera->focal_length());
                CHECK_EQ(free_camera->sensor_height(), camera->sensor_height());
                CHECK_EQ(free_camera->fstop(), camera->fstop());
                CHECK_EQ(free_camera->exposure(), camera->exposure());
                CHECK_EQ(free_camera->focus_distance(), camera->focus_distance());
                CHECK_EQ(free_camera->enable_depth_of_field(), camera->enable_depth_of_field());
                CHECK_EQ(free_camera->depth_range(), camera->depth_range());
                CHECK_EQ(free_camera->width(), camera->width());
                CHECK_EQ(free_camera->height(), camera->height());
                const float3 free_position = free_camera->entity()->transform().translation();
                scene->set_time(0.75);
                scene->set_active_camera(camera);
                scene->update();
                CHECK_EQ(editor->update_camera(0.1f, 320, 240), camera);
                CHECK_EQ(
                    controller->transform().translation(),
                    target == "self" ? float3(7.5f, 0.f, 0.f) : float3(8.5f, 2.f, 5.f)
                );
                CHECK_EQ(entity->transform().translation(), target == "self" ? float3(7.5f, 0.f, 0.f) : local_position);
                scene->set_active_camera(free_camera);
                CHECK_EQ(editor->update_camera(0.1f, 320, 240), free_camera);
                CHECK_EQ(free_camera->entity()->transform().translation(), free_position);
            } else {
                CHECK_NE(entity->transform().translation(), local_position);
            }
        }
    }
}

TEST_SUITE_END();
