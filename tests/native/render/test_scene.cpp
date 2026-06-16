// SPDX-License-Identifier: Apache-2.0

#include "testing.h"

#include "falcor2/render/scene.h"
#include "falcor2/render/geometry/static_mesh_geometry.h"
#include "falcor2/render/geometry/geometry_group.h"
#include "falcor2/render/component/geometry_instance.h"
#include "falcor2/render/component/camera.h"
#include "falcor2/render/component/light.h"

#include <sgl/math/vector.h>

using namespace falcor;

// ---------------------------------------------------------------------------
// 1. Object add/remove/compact cycle
// ---------------------------------------------------------------------------

TEST_SUITE_BEGIN("Scene");

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
