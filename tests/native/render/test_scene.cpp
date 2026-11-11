// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "testing.h"

#include "falcor2/render/scene.h"
#include "falcor2/render/render_scene.h"
#include "falcor2/render/ray_tracing_setup.h"
#include "falcor2/render/emissive_geometry_system.h"
#include "falcor2/render/light_system.h"
#include "falcor2/render/geometry/static_curve_geometry.h"
#include "falcor2/render/geometry/static_mesh_geometry.h"
#include "falcor2/render/geometry/geometry_group.h"
#include "falcor2/render/component/geometry_instance.h"
#include "falcor2/render/component/camera.h"
#include "falcor2/render/component/light.h"
#include "falcor2/render/material/emission_only_material.h"
#include "falcor2/render/material/standard_material.h"
#include "falcor2/render/material/unlit_material.h"
#include "falcor2/importers/importer.h"
#include "falcor2/importers/importer_types.h"
#include "falcor2/core/python_interpreter.h"

#include <sgl/math/vector.h>
#include <sgl/device/cursor_utils.h>
#include <sgl/device/kernel.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>

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

TEST_CASE("GeometryMotionFlags packing")
{
    for (auto type :
         {shared::GeometryType::triangle,
          shared::GeometryType::lss,
          shared::GeometryType::area_light,
          shared::GeometryType::invalid}) {
        for (auto flags : {shared::GeometryInstanceFlags::none, shared::GeometryInstanceFlags::vertex_motion}) {
            shared::GeometryInstanceDescHeader header = {};
            header.type_and_flags = shared::detail::pack_geometry_instance_type_and_flags(type, flags);
            CHECK_EQ(shared::detail::get_geometry_instance_type(header), type);
            CHECK_EQ(shared::detail::get_geometry_instance_flags(header), flags);
        }
    }
    CHECK_EQ(sizeof(shared::GeometryInstanceDescHeader), 16);
    CHECK_EQ(sizeof(shared::GeometryInstanceDesc), 32);
    CHECK_EQ(sizeof(shared::TriangleGeometryInstanceDesc), sizeof(shared::GeometryInstanceDesc));
    CHECK_EQ(sizeof(shared::LSSGeometryInstanceDesc), sizeof(shared::GeometryInstanceDesc));
    CHECK_EQ(sizeof(shared::AreaLightGeometryInstanceDesc), sizeof(shared::GeometryInstanceDesc));
}

TEST_CASE("scene pipeline equality covers every setup input independently of shaders")
{
    const SceneRequirements baseline;
    CHECK(baseline.matches_shader_inputs(&baseline));
    CHECK(baseline.matches(&baseline));
    CHECK_FALSE(baseline.matches_shader_inputs(nullptr));
    CHECK_FALSE(baseline.matches(nullptr));
    const auto equivalent = baseline;
    CHECK(baseline.matches(&equivalent));
    CHECK(baseline == equivalent);
    auto check_pipeline_change = [&](const SceneRequirements& changed)
    {
        CHECK(changed.matches_shader_inputs(&baseline));
        CHECK_FALSE(changed.matches(&baseline));
        CHECK(changed != baseline);
    };
    auto changed = baseline;
    ++changed.geometry_type_mask;
    check_pipeline_change(changed);
    changed = baseline;
    ++changed.hit_group_policy.ray_type_count;
    check_pipeline_change(changed);
    changed = baseline;
    ++changed.hit_group_policy.geometry_type_count;
    check_pipeline_change(changed);
    changed = baseline;
    changed.hit_group_policy.mode = HitGroupPolicy::Mode::per_geometry_instance_id;
    check_pipeline_change(changed);
    changed = baseline;
    changed.lss_intersection_mode = LSSIntersectionMode::procedural;
    check_pipeline_change(changed);
    changed = baseline;
    changed.ray_tracing_pipeline_flags = sgl::RayTracingPipelineFlags::enable_linear_swept_spheres;
    check_pipeline_change(changed);
}

TEST_CASE("scene shader comparison includes conformances and capabilities")
{
    SceneRequirements baseline;
    baseline.type_conformances = {{"IMaterial", "First", 1}, {"IMaterial", "Second", 2}};
    auto changed = baseline;
    auto check_shader_change = [&]()
    {
        CHECK_FALSE(changed.matches_shader_inputs(&baseline));
        CHECK(changed != baseline);
    };
    changed.type_conformances[0].id = 3;
    check_shader_change();
    changed = baseline;
    changed.type_conformances[0].type_name = "Third";
    check_shader_change();
    changed = baseline;
    changed.type_conformances[0].interface_name = "IOther";
    check_shader_change();
    changed = baseline;
    std::swap(changed.type_conformances[0], changed.type_conformances[1]);
    check_shader_change();
    for (auto flag :
         {SceneCapabilityFlags::geometry_properties,
          SceneCapabilityFlags::opacity_evaluation,
          SceneCapabilityFlags::material_aovs}) {
        changed = baseline;
        changed.capabilities = {.flags = flag};
        check_shader_change();
    }
}

TEST_CASE_GPU("geometry capability requirements follow imported material usage")
{
    auto importer_scene = make_ref<ImporterScene>();
    importer_scene->materials.emplace_back();
    ImporterMesh mesh;
    mesh.ensure_attributes({
        {ImporterSemantic::position},
        {ImporterSemantic::normal},
        {ImporterSemantic::tangent},
        {ImporterSemantic::handedness},
        {ImporterSemantic::tex_coord},
    });
    mesh.allocate_vertices(3, true);
    mesh.position_stream()[0] = float3(-1.f, 0.f, 0.f);
    mesh.position_stream()[1] = float3(1.f, 0.f, 0.f);
    mesh.position_stream()[2] = float3(0.f, 1.f, 0.f);
    for (size_t i = 0; i < 3; ++i) {
        mesh.normal_stream()[i] = float3(0.f, 0.f, 1.f);
        mesh.tangent_stream()[i] = float3(1.f, 0.f, 0.f);
        mesh.handedness_stream()[i] = 1.f;
    }
    mesh.subgeometries.push_back({.indices = {uint3(0, 1, 2)}, .material_index = 0});
    mesh.calculate_local_aabb();
    importer_scene->meshes.push_back(mesh);

    auto basic = Scene::from_importer_scene(ref(ctx.device), *importer_scene);
    basic->update();
    CHECK_FALSE(basic->_render_scene()->has_additional_attributes());
    CHECK_FALSE(basic->resolve_requirements({}, SceneCapabilityPolicy::required)
                    ->capabilities.has(SceneCapabilityFlags::geometry_properties));
    CHECK_FALSE(basic->requirements()->capabilities.has(SceneCapabilityFlags::geometry_properties));

    // Imported vertex colors enable material color use; unused streams do not require geometry access.
    const ImporterMeshDefaultAttribute extra_streams[] = {
        {.semantic = ImporterSemantic::tex_coord, .index = 1},
        {.semantic = ImporterSemantic::tex_coord, .components = 3},
        {.semantic = ImporterSemantic::color},
        {.semantic = ImporterSemantic::unknown, .components = 1, .name = "custom_weight"},
    };
    for (const auto& extra : extra_streams) {
        ImporterMesh with_extra;
        with_extra.ensure_attributes({{ImporterSemantic::position}, extra});
        with_extra.allocate_vertices(3, true);
        for (size_t i = 0; i < 3; ++i)
            with_extra.position_stream()[i] = mesh.position_stream()[i];
        with_extra.subgeometries = mesh.subgeometries;
        with_extra.calculate_local_aabb();
        importer_scene->meshes = {mesh, with_extra};
        auto detected = Scene::from_importer_scene(ref(ctx.device), *importer_scene);
        detected->update();
        CHECK(detected->_render_scene()->has_additional_attributes());
        const bool needs_geometry = extra.semantic == ImporterSemantic::color;
        CHECK(detected->requirements()->capabilities.has(SceneCapabilityFlags::geometry_properties) == needs_geometry);
        // Settings module names hash their source; separate scenes can have distinct module objects.
        CHECK(
            (detected->requirements()->modules[0]->name() == basic->requirements()->modules[0]->name())
            == !needs_geometry
        );
        CHECK(detected->required_capabilities().has(SceneCapabilityFlags::geometry_properties) == needs_geometry);
        CHECK(
            detected->resolve_requirements({}, SceneCapabilityPolicy::required)
                ->capabilities.has(SceneCapabilityFlags::geometry_properties)
            == needs_geometry
        );
    }
}

TEST_CASE_GPU("BLAS rebuild and refit")
{
    bool is_dynamic = false;
    SUBCASE("rebuild") { }
    SUBCASE("refit")
    {
        is_dynamic = true;
    }
    auto scene = Scene::create(ref(ctx.device));
    auto& render = *scene->_render_scene();
    const auto mesh = render.create_triangle_geometry({.vertex_count = 6, .index_count = 3, .is_dynamic = is_dynamic});
    RenderHandle geometries[] = {mesh};
    const auto group = render.create_geometry_group({.geometries = geometries});
    const auto transform = render.create_transform({.world_from_object = float4x4::identity()});
    render.create_geometry_instance({.geometry_group = group, .transform = transform});
    shared::PackedTriangleVertex vertices[6] = {};
    vertices[1].position = {1, 0, 0};
    vertices[2].position = {0, 1, 0};
    vertices[3].position = {3, 0, 0};
    vertices[4].position = {4, 0, 0};
    vertices[5].position = {3, 1, 0};
    const uint32_t indices[] = {0, 1, 2};
    render.update_triangle_geometry(mesh, {.vertices = vertices, .indices = indices});
    render.update({}, false);
    const auto blas = render.get_geometry_group(group).blas;
    REQUIRE(blas);

    auto module = ctx.device->load_module_from_source("blas_update_probe", R"(
import falcor2.render;
[shader("compute")]
[numthreads(1, 1, 1)]
void main(RWStructuredBuffer<float> result, uint output_index)
{
    float3 origin = output_index == 3 ? float3(3.2, 0.2, -1.0) : float3(0.2, 0.2, -1.0);
    RayDesc ray = {origin, 0.0, float3(0.0, 0.0, 1.0), 10.0};
    RayQuery<RAY_FLAG_FORCE_OPAQUE> query;
    query.TraceRayInline(g_scene.tlas, 0, 0xff, ray);
    while (query.Proceed()) {}
    result[output_index] = query.CommittedStatus() == COMMITTED_TRIANGLE_HIT ? query.CommittedRayT() : -1.0;
}
)");
    auto modules = scene->requirements()->modules;
    modules.push_back(module);
    auto kernel = ctx.device->create_compute_kernel(
        {.program = ctx.device->link_program(modules, {module->entry_point("main")})}
    );
    auto result = ctx.device->create_buffer({.size = 4 * sizeof(float), .usage = sgl::BufferUsage::unordered_access});
    auto trace = [&](uint32_t output_index)
    {
        kernel->dispatch(
            uint3(1),
            [&](sgl::ShaderCursor cursor)
            {
                render.bind_to_scene(cursor["g_scene"]);
                cursor = cursor.find_entry_point(0);
                cursor["result"] = result;
                cursor["output_index"] = output_index;
            }
        );
    };
    trace(0);
    for (auto& vertex : vertices)
        vertex.position.z = 1.f;
    render.update_triangle_geometry(mesh, {.vertices = vertices, .indices = indices}, RenderGeometryDirtyFlags::refit);
    render.update({}, false);
    CHECK(render.get_geometry_group(group).blas == blas);
    trace(1);

    // A rebuild request must win over refit even when vertex/index counts stay the same.
    const uint32_t replaced[] = {3, 4, 5};
    render.update_triangle_geometry(mesh, {.vertices = vertices, .indices = replaced});
    render.update_triangle_geometry(mesh, {.vertices = vertices, .indices = replaced}, RenderGeometryDirtyFlags::refit);
    render.update({}, false);
    CHECK(render.get_geometry_group(group).blas == blas);
    trace(2);
    trace(3);
    ctx.device->wait();
    const auto hits = result->get_elements<float>();
    CHECK(hits[0] == doctest::Approx(1.f));
    CHECK(hits[1] == doctest::Approx(2.f));
    CHECK(hits[2] == -1.f);
    CHECK(hits[3] == doctest::Approx(2.f));
}

TEST_CASE_GPU("triangle additional attribute enablement tracks nonempty geometries")
{
    ref<RenderScene> render_scene = RenderScene::create(ctx.device);

    const RenderTriangleGeometryDesc desc{
        .vertex_count = 3,
        .index_count = 3,
    };
    const RenderTriangleGeometryHandle first = render_scene->create_triangle_geometry(desc);
    const RenderTriangleGeometryHandle second = render_scene->create_triangle_geometry(desc);

    const std::array<shared::PackedTriangleVertex, 3> vertices{};
    const std::array<uint32_t, 3> indices{0, 1, 2};
    const std::array<float, 6> texcoords{};
    const RenderTriangleGeometryAdditionalAttributeData texcoord{
        .mask_bit = 1,
        .type = shared::AdditionalAttributeType::float32,
        .component_count = 2,
        .data = std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(texcoords.data()), sizeof(texcoords)),
    };

    auto update = [&](RenderTriangleGeometryHandle handle, bool with_additional_attribute)
    {
        RenderTriangleGeometryData data{
            .vertices = vertices,
            .indices = indices,
        };
        if (with_additional_attribute)
            data.additional_attributes = std::span<const RenderTriangleGeometryAdditionalAttributeData>(&texcoord, 1);
        render_scene->update_triangle_geometry(handle, data);
    };

    CHECK_EQ(render_scene->triangle_additional_attribute_geometry_count(), 0);

    update(first, true);
    CHECK_EQ(render_scene->triangle_additional_attribute_geometry_count(), 1);

    update(second, true);
    CHECK_EQ(render_scene->triangle_additional_attribute_geometry_count(), 2);

    update(first, false);
    CHECK_EQ(render_scene->triangle_additional_attribute_geometry_count(), 1);

    render_scene->destroy_triangle_geometry(second);
    CHECK_EQ(render_scene->triangle_additional_attribute_geometry_count(), 0);

    render_scene->destroy_triangle_geometry(first);
    CHECK_EQ(render_scene->triangle_additional_attribute_geometry_count(), 0);
}

TEST_CASE_GPU("scene geometry attribute registry")
{
    auto scene = Scene::create(ref(ctx.device));
    scene->update();
    const uint64_t initial_requirements_generation = scene->requirements_generation();
    const auto get_or_create_named
        = [&](std::string_view name, shared::AdditionalAttributeType type, uint32_t component_count)
    {
        return scene
            ->_get_or_create_geometry_attribute_id(MeshAttributeSemantic::unknown, 0, name, type, component_count);
    };

    CHECK_EQ(
        scene->_get_or_create_geometry_attribute_id(
            MeshAttributeSemantic::tex_coord,
            0,
            "",
            shared::AdditionalAttributeType::float32,
            2
        ),
        0u
    );
    CHECK_EQ(
        scene->_get_or_create_geometry_attribute_id(
            MeshAttributeSemantic::tex_coord,
            7,
            "",
            shared::AdditionalAttributeType::float16,
            3
        ),
        7u
    );
    CHECK_EQ(
        scene->_get_or_create_geometry_attribute_id(
            MeshAttributeSemantic::color,
            0,
            "",
            shared::AdditionalAttributeType::float32,
            4
        ),
        8u
    );
    CHECK_FALSE(scene->_get_geometry_attribute_id(MeshAttributeSemantic::tex_coord, 8, ""));
    CHECK_FALSE(scene->_get_geometry_attribute_id(MeshAttributeSemantic::unknown, 0, "missing"));

    CHECK_EQ(scene->_reserve_geometry_attribute_id("Pref"), 16u);
    CHECK_EQ(scene->_reserve_geometry_attribute_id("Pref"), 16u);
    CHECK_EQ(scene->_get_geometry_attribute_id(MeshAttributeSemantic::unknown, 0, "Pref"), 16u);
    CHECK_EQ(get_or_create_named("Pref", shared::AdditionalAttributeType::float32, 3), 16u);
    CHECK_EQ(get_or_create_named("Pref", shared::AdditionalAttributeType::float32, 3), 16u);
    CHECK_THROWS_WITH(
        get_or_create_named("Pref", shared::AdditionalAttributeType::float16, 3),
        doctest::Contains("inconsistent source type or component count")
    );
    CHECK_THROWS_WITH(
        get_or_create_named("Pref", shared::AdditionalAttributeType::float32, 2),
        doctest::Contains("inconsistent source type or component count")
    );

    CHECK_EQ(get_or_create_named("shared", shared::AdditionalAttributeType::float32, 2), 17u);
    CHECK_EQ(scene->_reserve_geometry_attribute_id("shared"), 17u);
    CHECK_EQ(
        scene->_get_or_create_geometry_attribute_id(
            MeshAttributeSemantic::weights,
            0,
            "WEIGHTS_0",
            shared::AdditionalAttributeType::float32,
            4
        ),
        18u
    );
    CHECK_EQ(scene->_get_geometry_attribute_id(MeshAttributeSemantic::tex_coord, 0, ""), 0u);
    CHECK_EQ(scene->_get_geometry_attribute_id(MeshAttributeSemantic::unknown, 0, "shared"), 17u);
    CHECK_EQ(scene->_get_geometry_attribute_id(MeshAttributeSemantic::weights, 0, "WEIGHTS_0"), 18u);
    CHECK_THROWS_WITH(
        get_or_create_named("", shared::AdditionalAttributeType::float32, 1),
        doctest::Contains("must have a name")
    );
    CHECK_THROWS_WITH(scene->_reserve_geometry_attribute_id(""), doctest::Contains("must have a name"));

    const SceneUpdateFlags update_flags = scene->update();
    CHECK_FALSE(is_set(update_flags, SceneUpdateFlags::requirements));
    CHECK_EQ(scene->requirements_generation(), initial_requirements_generation);

    CHECK_EQ(get_or_create_named("geometry_owned", shared::AdditionalAttributeType::int32, 1), 19u);
    Geometry* removed_geometry = scene->create_geometry("StaticMeshGeometry");
    removed_geometry->remove();
    scene->update();
    CHECK_EQ(scene->_reserve_geometry_attribute_id("geometry_owned"), 19u);
    CHECK_EQ(scene->_reserve_geometry_attribute_id("after_removal"), 20u);

    auto full_scene = Scene::create(ref(ctx.device));
    const auto get_or_create_full_named = [&](std::string_view name)
    {
        return full_scene->_get_or_create_geometry_attribute_id(
            MeshAttributeSemantic::unknown,
            0,
            name,
            shared::AdditionalAttributeType::float32,
            1
        );
    };
    // MDL candidate reservation is host-side work: exercise it once, without updating/compiling the Scene shader.
#if SGL_WINDOWS
    const bool check_mdl_candidates = ctx.device->type() == sgl::DeviceType::d3d12;
#else
    const bool check_mdl_candidates = ctx.device->type() == sgl::DeviceType::vulkan;
#endif
    const uint32_t reserved_count
        = Scene::NAMED_ATTRIBUTE_ID_END - Scene::NAMED_ATTRIBUTE_ID_BEGIN - uint32_t(check_mdl_candidates);
    for (uint32_t index = 0; index < reserved_count; ++index) {
        const std::string name = index == 0 ? "Pref" : fmt::format("custom_{}", index);
        const uint32_t id = index % 2 == 0 ? full_scene->_reserve_geometry_attribute_id(name)
                                           : get_or_create_full_named(name).value();
        CHECK_EQ(id, 16 + index);
        CHECK_EQ(full_scene->_reserve_geometry_attribute_id(name), id);
    }
    if (check_mdl_candidates) {
        Properties props;
        props.set(
            "mdl_library_path",
            (testing::project_directory() / "tests/python/render/data/mdl_generated_material_pipeline").string()
        );
        props.set("mdl_material_name", "scene_data_metadata::Conservative");
        full_scene->create_material("MDLMaterial", props);
        // Pref reuses an ID; the unused SDK candidate consumes the final slot. The checks below then overflow.
        CHECK_EQ(full_scene->_reserve_geometry_attribute_id("unrelated_in_helper"), Scene::NAMED_ATTRIBUTE_ID_END - 1);
    }
    CHECK_THROWS_WITH(
        get_or_create_full_named("overflow"),
        doctest::Contains("more than 16 distinct named geometry attributes")
    );
    CHECK_THROWS_WITH(
        full_scene->_reserve_geometry_attribute_id("reserved_overflow"),
        doctest::Contains("more than 16 distinct named geometry attributes")
    );
}

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
    SceneRayTracingSetup setup = SceneRayTracingSetup::create(scene.get(), {primary, secondary}, nullptr, options);

    CHECK_EQ(setup.sbt_miss_entry_points.size(), 3);
    CHECK_EQ(setup.sbt_miss_entry_points[0], "_scene_primary_miss");
    CHECK_EQ(setup.sbt_miss_entry_points[1], "_scene_secondary_miss");
    CHECK(setup.sbt_miss_entry_points[2].empty());

    REQUIRE_EQ(setup.sbt_hit_group_names.size(), 9);
    CHECK_EQ(setup.sbt_hit_group_names[0], "_scene_primary_triangle_hit_group");
    CHECK_EQ(setup.sbt_hit_group_names[1], "_scene_secondary_triangle_hit_group");
    CHECK_EQ(setup.sbt_hit_group_names[2], "__dummy_hit_group");
    CHECK_EQ(setup.sbt_hit_group_names[3], "_scene_primary_lss_hit_group");
    CHECK_EQ(setup.sbt_hit_group_names[4], "_scene_secondary_lss_hit_group");
    CHECK_EQ(setup.sbt_hit_group_names[5], "__dummy_hit_group");
    CHECK_EQ(setup.sbt_hit_group_names[6], "_scene_primary_area_light_hit_group");
    CHECK_EQ(setup.sbt_hit_group_names[7], "_scene_secondary_area_light_hit_group");
    CHECK_EQ(setup.sbt_hit_group_names[8], "__dummy_hit_group");
}

TEST_CASE_GPU("ray tracing setup rejects unsupported ray and geometry counts")
{
    auto scene = Scene::create(ref(ctx.device));
    std::array<SceneRayTracingSetup::RayDesc, 4> ray_descs;
    CHECK_THROWS(SceneRayTracingSetup::create(scene.get(), ray_descs));
    auto requirements = make_ref<SceneRequirements>(*scene->requirements());
    requirements->hit_group_policy.geometry_type_count = 32;
    CHECK_THROWS_WITH(
        SceneRayTracingSetup::create(scene.get(), {ray_descs[0]}, requirements.get()),
        doctest::Contains("unsupported geometry types")
    );
}

TEST_CASE_GPU("scene requirements track geometry type changes")
{
    auto scene = Scene::create(ref(ctx.device));
    scene->update();

    const uint64_t initial_generation = scene->requirements_generation();
    const auto initial_requirements = scene->requirements();
    const auto initial_selected = scene->resolve_requirements({}, SceneCapabilityPolicy::required);
    CHECK_FALSE(scene->requirements()->has_geometry_type(shared::GeometryType::lss));

    const float3 positions[] = {{0.f, 0.f, 0.f}, {0.f, 1.f, 0.f}};
    const float radii[] = {0.1f, 0.1f};
    const uint32_t indices[] = {0, 1};
    StaticCurveGeometryDataDesc desc{
        .name = "requirements-generation-test",
        .vertex_count = 2,
        .position_stream = {positions, sizeof(float3)},
        .radius_stream = {radii, sizeof(float)},
        .index_count = 2,
        .index_stream = {indices, sizeof(uint32_t)},
    };
    StaticCurveGeometry* curve = scene->create_geometry<StaticCurveGeometry>();
    curve->set_curve_data(desc);

    const SceneUpdateFlags added_flags = scene->update();
    CHECK(is_set(added_flags, SceneUpdateFlags::requirements));
    CHECK_GT(scene->requirements_generation(), initial_generation);
    CHECK(scene->requirements()->has_geometry_type(shared::GeometryType::lss));
    CHECK(scene->requirements() != initial_requirements);
    CHECK(scene->requirements()->matches_shader_inputs(initial_requirements.get()));
    const auto selected = scene->resolve_requirements({}, SceneCapabilityPolicy::required);
    CHECK(selected != initial_selected);
    CHECK(selected->matches_shader_inputs(initial_selected.get()));
    CHECK_FALSE(initial_requirements->has_geometry_type(shared::GeometryType::lss));

    const uint64_t populated_generation = scene->requirements_generation();
    curve->set_curve_data({});

    const SceneUpdateFlags removed_flags = scene->update();
    CHECK(is_set(removed_flags, SceneUpdateFlags::requirements));
    CHECK_GT(scene->requirements_generation(), populated_generation);
    CHECK_FALSE(scene->requirements()->has_geometry_type(shared::GeometryType::lss));
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
            .material_index = 0,
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
    dome_light.pole_axis = EnvMapPoleAxis::z;
    dome_light.color = float3(0.25f, 0.5f, 0.75f);
    dome_light.exposure = 5.f;
    importer_scene->lights.push_back(dome_light);

    ImporterNode dome_node;
    dome_node.name = "Test Dome Light";
    dome_node.light_index = 0;
    importer_scene->nodes.push_back(dome_node);
    importer_scene->root_nodes.push_back(0);

    auto scene = Scene::from_importer_scene(ref(ctx.device), *importer_scene);

    REQUIRE_EQ(scene->components().size(), 2);
    const auto* env_map_light = dynamic_cast<const EnvMapLight*>(scene->components()[1]);
    REQUIRE(env_map_light != nullptr);
    CHECK_EQ(env_map_light->env_map_path(), std::filesystem::path(dome_light.env_map_path));
    CHECK_EQ(env_map_light->color(), dome_light.color);
    CHECK_EQ(env_map_light->intensity(), doctest::Approx(dome_light.intensity));
    CHECK_EQ(env_map_light->exposure(), doctest::Approx(dome_light.exposure));
    CHECK_EQ(env_map_light->pole_axis(), EnvMapPoleAxis::z);
}

TEST_CASE_GPU("importer scene textureless dome light becomes constant environment")
{
    auto importer_scene = make_ref<ImporterScene>();

    ImporterLight dome_light;
    dome_light.name = "Textureless Dome Light";
    dome_light.type = ImporterLight::Type::dome;
    dome_light.color = float3(0.25f, 0.5f, 0.75f);
    dome_light.exposure = 2.f;
    importer_scene->lights.push_back(dome_light);

    ImporterNode dome_node;
    dome_node.name = dome_light.name;
    dome_node.light_index = 0;
    importer_scene->nodes.push_back(dome_node);
    importer_scene->root_nodes.push_back(0);

    auto scene = Scene::from_importer_scene(ref(ctx.device), *importer_scene);

    REQUIRE_EQ(scene->components().size(), 2);
    const auto* constant_light = dynamic_cast<const ConstantLight*>(scene->components()[1]);
    REQUIRE(constant_light != nullptr);
    CHECK_EQ(constant_light->color(), dome_light.color);
    CHECK_EQ(constant_light->intensity(), doctest::Approx(dome_light.intensity));
    CHECK_EQ(constant_light->exposure(), doctest::Approx(dome_light.exposure));
}

TEST_CASE_GPU("importer scene converts distant angular diameter to cone half angle")
{
    auto importer_scene = make_ref<ImporterScene>();

    ImporterLight importer_light;
    importer_light.name = "Distant Light";
    importer_light.type = ImporterLight::Type::distant;
    importer_light.color = float3(2.f, 3.f, 4.f);
    importer_light.exposure = 2.f;
    importer_light.degree_angular_diameter = 12.f;
    importer_scene->lights.push_back(importer_light);

    ImporterNode light_node;
    light_node.name = importer_light.name;
    light_node.light_index = 0;
    importer_scene->nodes.push_back(light_node);
    importer_scene->root_nodes.push_back(0);

    auto scene = Scene::from_importer_scene(ref(ctx.device), *importer_scene);

    REQUIRE_EQ(scene->components().size(), 2);
    const auto* distant_light = dynamic_cast<const DistantLight*>(scene->components()[1]);
    REQUIRE(distant_light != nullptr);
    CHECK_EQ(distant_light->color(), importer_light.color);
    CHECK_EQ(distant_light->intensity(), doctest::Approx(importer_light.intensity));
    CHECK_EQ(distant_light->exposure(), doctest::Approx(importer_light.exposure));
    CHECK_EQ(distant_light->cutoff_angle(), doctest::Approx(6.f));
}

TEST_CASE_GPU("importer scene creates shaped point light")
{
    auto importer_scene = make_ref<ImporterScene>();

    ImporterLight importer_light;
    importer_light.name = "Spot Light";
    importer_light.type = ImporterLight::Type::point;
    importer_light.color = float3(2.f, 3.f, 4.f);
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

    REQUIRE_EQ(scene->components().size(), 2);
    const auto* point_light = dynamic_cast<const PointLight*>(scene->components()[1]);
    REQUIRE(point_light != nullptr);
    REQUIRE(point_light->class_descriptor().base());
    CHECK_EQ(point_light->class_descriptor().base()->name(), "Light");
    CHECK_EQ(point_light->color(), importer_light.color);
    CHECK_EQ(point_light->intensity(), doctest::Approx(importer_light.intensity));
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
    importer_light.color = float3(2.f, 3.f, 4.f);
    importer_light.intensity = 5.f;
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

    REQUIRE_EQ(scene->components().size(), 2);
    const auto* point_light = dynamic_cast<const PointLight*>(scene->components()[1]);
    REQUIRE(point_light != nullptr);
    CHECK_EQ(point_light->color(), importer_light.color);
    CHECK_EQ(point_light->intensity(), doctest::Approx(importer_light.intensity));
    CHECK_EQ(point_light->exposure(), doctest::Approx(importer_light.exposure));
    CHECK(point_light->enable_color_temperature());
    CHECK_EQ(point_light->color_temperature(), doctest::Approx(importer_light.color_temperature));
}

TEST_CASE_GPU("importer scene preserves treated sphere power when converting to a point light")
{
    auto importer_scene = make_ref<ImporterScene>();

    ImporterLight importer_light;
    importer_light.name = "Treated Point";
    importer_light.type = ImporterLight::Type::sphere;
    importer_light.intensity = 3.f;
    importer_light.radius = 0.5f;
    importer_light.treat_as_point = true;
    float expected_point_intensity = 0.f;
    SUBCASE("unnormalized")
    {
        importer_light.normalize = false;
        expected_point_intensity = 3.f * static_cast<float>(M_PI);
    }
    SUBCASE("normalized")
    {
        importer_light.normalize = true;
        expected_point_intensity = 0.75f;
    }
    importer_scene->lights.push_back(importer_light);

    ImporterNode light_node;
    light_node.name = importer_light.name;
    light_node.light_index = 0;
    light_node.transform = float4x4::identity();
    light_node.transform[0][0] = 2.f;
    light_node.transform[1][1] = 2.f;
    light_node.transform[2][2] = 2.f;
    importer_scene->nodes.push_back(light_node);
    importer_scene->root_nodes.push_back(0);

    auto scene = Scene::from_importer_scene(ref(ctx.device), *importer_scene);

    REQUIRE_EQ(scene->components().size(), 2);
    const auto* point_light = dynamic_cast<const PointLight*>(scene->components()[1]);
    REQUIRE(point_light != nullptr);
    CHECK_EQ(point_light->intensity(), doctest::Approx(expected_point_intensity));
    CHECK_FALSE(point_light->normalize());
}

TEST_CASE_GPU("importer scene creates shaped analytic area lights")
{
    auto importer_scene = make_ref<ImporterScene>();

    ImporterLight sphere;
    sphere.name = "Sphere Light";
    sphere.type = ImporterLight::Type::sphere;
    sphere.color = float3(2.f, 3.f, 4.f);
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
    disk.color = float3(5.f, 6.f, 7.f);
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
    rect.color = float3(8.f, 9.f, 10.f);
    rect.normalize = true;
    rect.exposure = 3.f;
    rect.width = 2.5f;
    rect.height = 1.5f;
    rect.enable_shaping = true;
    rect.shaping_cone_angle = 50.f;
    rect.shaping_cone_softness = 0.2f;
    rect.shaping_focus = 1.5f;
    importer_scene->lights.push_back(rect);

    ImporterLight cylinder;
    cylinder.name = "Cylinder Light";
    cylinder.type = ImporterLight::Type::cylinder;
    cylinder.color = float3(4.f, 3.f, 2.f);
    cylinder.normalize = true;
    cylinder.exposure = 0.5f;
    cylinder.radius = 0.625f;
    cylinder.length = 3.5f;
    cylinder.enable_shaping = true;
    cylinder.shaping_cone_angle = 45.f;
    cylinder.shaping_cone_softness = 0.3f;
    cylinder.shaping_focus = 2.5f;
    importer_scene->lights.push_back(cylinder);

    for (int light_index = 0; light_index < 4; ++light_index) {
        ImporterNode light_node;
        light_node.name = importer_scene->lights[light_index].name;
        light_node.light_index = light_index;
        importer_scene->nodes.push_back(light_node);
        importer_scene->root_nodes.push_back(light_index);
    }

    auto scene = Scene::from_importer_scene(ref(ctx.device), *importer_scene);

    REQUIRE_EQ(scene->components().size(), 5);
    const SphereLight* sphere_light = scene->components().find<SphereLight>();
    const DiskLight* disk_light = scene->components().find<DiskLight>();
    const RectLight* rect_light = scene->components().find<RectLight>();
    const CylinderLight* cylinder_light = scene->components().find<CylinderLight>();

    REQUIRE(sphere_light);
    REQUIRE(sphere_light->class_descriptor().base());
    CHECK_EQ(sphere_light->class_descriptor().base()->name(), "Light");
    CHECK_EQ(sphere_light->color(), sphere.color);
    CHECK_EQ(sphere_light->intensity(), doctest::Approx(sphere.intensity));
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
    CHECK_EQ(disk_light->color(), disk.color);
    CHECK_EQ(disk_light->intensity(), doctest::Approx(disk.intensity));
    CHECK_EQ(disk_light->exposure(), doctest::Approx(disk.exposure));
    CHECK_EQ(disk_light->radius(), doctest::Approx(disk.radius));
    CHECK(disk_light->enable_shaping());
    CHECK_EQ(disk_light->shaping_cone_angle(), doctest::Approx(disk.shaping_cone_angle));
    CHECK_EQ(disk_light->shaping_cone_softness(), doctest::Approx(disk.shaping_cone_softness));
    CHECK_EQ(disk_light->shaping_focus(), doctest::Approx(disk.shaping_focus));

    REQUIRE(rect_light);
    REQUIRE(rect_light->class_descriptor().base());
    CHECK_EQ(rect_light->class_descriptor().base()->name(), "Light");
    CHECK_EQ(rect_light->color(), rect.color);
    CHECK_EQ(rect_light->intensity(), doctest::Approx(rect.intensity));
    CHECK_EQ(rect_light->normalize(), rect.normalize);
    CHECK_EQ(rect_light->exposure(), doctest::Approx(rect.exposure));
    CHECK_EQ(rect_light->width(), doctest::Approx(rect.width));
    CHECK_EQ(rect_light->height(), doctest::Approx(rect.height));
    CHECK(rect_light->enable_shaping());
    CHECK_EQ(rect_light->shaping_cone_angle(), doctest::Approx(rect.shaping_cone_angle));
    CHECK_EQ(rect_light->shaping_cone_softness(), doctest::Approx(rect.shaping_cone_softness));
    CHECK_EQ(rect_light->shaping_focus(), doctest::Approx(rect.shaping_focus));

    REQUIRE(cylinder_light);
    REQUIRE(cylinder_light->class_descriptor().base());
    CHECK_EQ(cylinder_light->class_descriptor().base()->name(), "Light");
    CHECK_EQ(cylinder_light->color(), cylinder.color);
    CHECK_EQ(cylinder_light->normalize(), cylinder.normalize);
    CHECK_EQ(cylinder_light->exposure(), doctest::Approx(cylinder.exposure));
    CHECK_EQ(cylinder_light->radius(), doctest::Approx(cylinder.radius));
    CHECK_EQ(cylinder_light->length(), doctest::Approx(cylinder.length));
    CHECK(cylinder_light->enable_shaping());
    CHECK_EQ(cylinder_light->shaping_cone_angle(), doctest::Approx(cylinder.shaping_cone_angle));
    CHECK_EQ(cylinder_light->shaping_cone_softness(), doctest::Approx(cylinder.shaping_cone_softness));
    CHECK_EQ(cylinder_light->shaping_focus(), doctest::Approx(cylinder.shaping_focus));

    Properties properties = rect_light->properties();
    CHECK_EQ(properties.get<float3>("color"), rect.color);
    CHECK_EQ(properties.get<float>("intensity"), doctest::Approx(rect.intensity));
    CHECK_EQ(properties.get<bool>("normalize"), rect.normalize);
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
    bool use_vertex_color = false;
    SUBCASE("without vertex colors") { }
    SUBCASE("with vertex colors")
    {
        mesh.ensure_attributes({{ImporterSemantic::color}});
        use_vertex_color = true;
    }
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
            .material_index = 0,
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
    const Any default_vertex_color = scene->materials()[0]->get_property("use_vertex_color");
    CHECK_EQ(*any_cast<bool>(&default_vertex_color), use_vertex_color);

    REQUIRE_EQ(scene->components().size(), 2);
    auto* instance = dynamic_cast<GeometryInstance*>(scene->components()[1]);
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

TEST_CASE_GPU("glTF unlit importer material creates direction-independent material")
{
    auto importer_scene = make_ref<ImporterScene>();
    ImporterMaterial importer_material;
    importer_material.name = "Unlit";
    importer_material.params.set("_type", "gltf_unlit");
    importer_material.params.set("baseColorFactor", float4(0.25f, 0.5f, 0.75f, 0.6f));
    importer_material.params.set("metallicFactor", 0.2f);
    importer_material.params.set("roughnessFactor", 0.3f);
    importer_material.params.set("emissiveFactor", float3(8.f));
    importer_material.params.set("alphaMode", AlphaMode::mask);
    importer_material.params.set("alphaCutoff", 0.4f);
    importer_material.params.set("doubleSided", true);
    importer_scene->materials.push_back(std::move(importer_material));

    auto scene = Scene::from_importer_scene(ref(ctx.device), *importer_scene);
    REQUIRE_EQ(scene->materials().size(), 1);

    auto* material = dynamic_cast<UnlitMaterial*>(scene->materials()[0]);
    REQUIRE(material != nullptr);
    CHECK_EQ(material->name(), "Unlit");
    CHECK_EQ(material->_base_color_factor(), float3(0.25f, 0.5f, 0.75f));
    CHECK_EQ(material->flags(), shared::MaterialFlags::unlit | shared::MaterialFlags::two_sided);
    // This fixture contains no mesh or COLOR_0 stream to evaluate.
    CHECK_FALSE(material->get_property<bool>("use_vertex_color"));

    const Material::OpacityDesc opacity = material->opacity_desc();
    CHECK_EQ(opacity.flags, shared::OpacityFlags::enabled | shared::OpacityFlags::use_threshold);
    CHECK_EQ(opacity.factor, doctest::Approx(0.6f));
    CHECK_EQ(opacity.threshold, doctest::Approx(0.4f));

    // UV0-only opacity keeps the shared texture fast path. Other UV sets or
    // vertex alpha require the material's own opacity evaluator.
    material->set_property("base_color_texture_texcoord", 1u);
    CHECK(is_set(material->opacity_desc().flags, shared::OpacityFlags::evaluate_material));
    material->set_property("base_color_texture_texcoord", 0u);
    CHECK_FALSE(is_set(material->opacity_desc().flags, shared::OpacityFlags::evaluate_material));
    material->set_property("use_vertex_color", true);
    CHECK(is_set(material->opacity_desc().flags, shared::OpacityFlags::evaluate_material));
    material->set_property("alpha_mode", AlphaMode::opaque);
    CHECK_EQ(material->opacity_desc().flags, shared::OpacityFlags::none);
    material->set_property("alpha_mode", AlphaMode::blend);
    CHECK_EQ(material->opacity_desc().flags, shared::OpacityFlags::enabled | shared::OpacityFlags::evaluate_material);
    material->set_property("use_vertex_color", false);
    CHECK_EQ(material->opacity_desc().flags, shared::OpacityFlags::enabled);
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

TEST_CASE_GPU("scene retains imported USD additional geometry properties")
{
    const std::filesystem::path scene_path = testing::project_directory() / "data" / "assets" / "test_usd_importer"
        / "test_additional_geometry_properties.usda";
    SceneConfig config{
        .uv_origin = UVOrigin::lower_left,
    };
    ref<Scene> scene = Scene::load(ref(ctx.device), scene_path, {}, config);

    REQUIRE_EQ(scene->geometries().size(), 1);
    auto* geometry = dynamic_cast<StaticMeshGeometry*>(scene->geometries()[0]);
    REQUIRE(geometry != nullptr);
    REQUIRE_EQ(geometry->sub_mesh_count(), 1);
    REQUIRE_EQ(geometry->vertex_count(0), 6);
    const std::array<uint32_t, 6> expected_indices{0, 1, 2, 3, 4, 5};
    CHECK(std::equal(geometry->indices(0).begin(), geometry->indices(0).end(), expected_indices.begin()));

    const std::array<std::array<float, 6>, 8> expected_u{
        std::array{0.f, 1.f, 0.f, 2.f, 3.f, 2.f},
        std::array{10.f, 11.f, 10.f, 11.f, 11.f, 10.f},
        std::array{20.f, 21.f, 20.f, 22.f, 23.f, 22.f},
        std::array{30.f, 31.f, 30.f, 32.f, 33.f, 32.f},
        std::array{40.f, 41.f, 40.f, 42.f, 43.f, 42.f},
        std::array{50.f, 51.f, 50.f, 52.f, 53.f, 52.f},
        std::array{60.f, 61.f, 60.f, 62.f, 63.f, 62.f},
        std::array{70.f, 71.f, 70.f, 72.f, 73.f, 72.f},
    };
    for (uint32_t slot = 0; slot < expected_u.size(); ++slot) {
        const MeshVertexAttribute* attribute
            = geometry->find_vertex_attribute(0, MeshAttributeSemantic::tex_coord, slot);
        REQUIRE(attribute != nullptr);
        CHECK_EQ(attribute->type, sgl::DataType::float32);
        CHECK_EQ(attribute->component_count, 2);
        REQUIRE_EQ(attribute->data.size(), geometry->vertex_count(0) * sizeof(float2));
        for (size_t vertex = 0; vertex < geometry->vertex_count(0); ++vertex) {
            float2 value;
            std::memcpy(&value, attribute->data.data() + vertex * sizeof(float2), sizeof(value));
            CHECK_EQ(value.x, expected_u[slot][vertex]);
        }
    }

    const auto attributes = geometry->vertex_attributes(0);
    const auto pref_it = std::find_if(
        attributes.begin(),
        attributes.end(),
        [](const MeshVertexAttribute& attribute)
        {
            return attribute.semantic == MeshAttributeSemantic::unknown && attribute.name == "Pref";
        }
    );
    REQUIRE(pref_it != attributes.end());
    CHECK_EQ(pref_it->type, sgl::DataType::float32);
    CHECK_EQ(pref_it->component_count, 3);
    REQUIRE_EQ(pref_it->data.size(), geometry->vertex_count(0) * sizeof(float3));
    CHECK(
        std::none_of(
            attributes.begin(),
            attributes.end(),
            [](const MeshVertexAttribute& attribute)
            {
                return attribute.name == "pref";
            }
        )
    );
    for (size_t vertex = 0; vertex < geometry->vertex_count(0); ++vertex) {
        float3 value;
        std::memcpy(&value, pref_it->data.data() + vertex * sizeof(float3), sizeof(value));
        CHECK_EQ(value.x, expected_u[0][vertex]);
        CHECK_EQ(value.z, 10.f);
    }

    scene->update();
    for (uint32_t slot = 1; slot <= 7; ++slot)
        CHECK_EQ(scene->_get_geometry_attribute_id(MeshAttributeSemantic::tex_coord, slot, ""), slot);
    CHECK_EQ(scene->_get_geometry_attribute_id(MeshAttributeSemantic::unknown, 0, "Pref"), 16u);
    CHECK_FALSE(scene->_get_geometry_attribute_id(MeshAttributeSemantic::unknown, 0, "pref"));

    const RenderScene* render_scene = scene->_render_scene();
    const RenderGeometryGroup& group = render_scene->get_geometry_group(geometry->_render_geometry_group());
    REQUIRE_EQ(group.geometries.size(), 1);
    const RenderTriangleGeometry& render_geometry
        = render_scene->get_triangle_geometry(RenderTriangleGeometryHandle(group.geometries.front()));
    REQUIRE_EQ(render_geometry.additional_attributes.size(), 8);
    uint32_t presence_mask = 0;
    for (const RenderGeometryAdditionalAttribute& attribute : render_geometry.additional_attributes) {
        presence_mask |= 1u << attribute.mask_bit;
        CHECK_EQ(attribute.type, shared::AdditionalAttributeType::float32);
        CHECK_EQ(attribute.component_count, attribute.mask_bit == 16 ? 3u : 2u);
    }
    CHECK_EQ(presence_mask, 0x100feu);
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

TEST_CASE_GPU("collection queries exclude built-in objects unless requested")
{
    auto scene = Scene::create(ref(ctx.device));
    auto& components = scene->components();
    Camera* builtin = scene->free_camera();
    CHECK_EQ(components.find<Camera>(), nullptr);
    CHECK(components.find_all<Camera>().empty());
    CHECK_EQ(components.find("free_camera"), nullptr);
    CHECK_EQ(components.find<Camera>(true), builtin);
    CHECK_EQ(components.find("free_camera", true), builtin);

    Camera* first = scene->create_entity()->create_component<Camera>();
    Camera* named = scene->create_entity()->create_component<Camera>();
    first->set_name("first");
    named->set_name("free_camera");
    CHECK_EQ(components.find<Camera>(), first);
    CHECK_EQ(components.find<Camera>(false), first);
    CHECK_EQ(components.find("free_camera"), named);
    CHECK_EQ(components.find<Camera>("free_camera"), named);
    CHECK_EQ(components.find<Camera>("free_camera", true), builtin);
    CHECK_EQ(components.find<Camera>("missing"), nullptr);
    CHECK(components.find_all("free_camera") == std::vector<Component*>{named});
    CHECK(components.find_all("free_camera", true) == std::vector<Component*>{builtin, named});
    CHECK(components.find_all<Camera>() == std::vector<Camera*>{first, named});
    CHECK(components.find_all<Camera>(true) == std::vector<Camera*>{builtin, first, named});
    CHECK(components.find_all<Camera>("free_camera") == std::vector<Camera*>{named});
    CHECK(components.find_all<Camera>("free_camera", true) == std::vector<Camera*>{builtin, named});
    CHECK(components.find_all<Camera>("missing").empty());

    const char* name_ptr = "free_camera";
    const std::string name_string = name_ptr;
    const std::string_view name_view = name_ptr;
    CHECK_EQ(components.find<Camera>(name_ptr), named);
    CHECK_EQ(components.find<Camera>(name_ptr, true), builtin);
    CHECK_EQ(components.find<Camera>(name_string), named);
    CHECK_EQ(components.find<Camera>(name_view), named);
    CHECK(components.find_all<Camera>(name_ptr) == std::vector<Camera*>{named});
    CHECK(components.find_all<Camera>(name_ptr, true) == std::vector<Camera*>{builtin, named});
    CHECK(components.find_all<Camera>(name_string) == std::vector<Camera*>{named});
    CHECK(components.find_all<Camera>(name_view) == std::vector<Camera*>{named});
    CHECK(components.find_all<Camera>(false) == std::vector<Camera*>{first, named});
}

TEST_CASE_GPU("filtered ranges preserve raw collection and dirty traversal")
{
    auto scene = Scene::create(ref(ctx.device));
    auto& entities = scene->entities();
    CHECK(entities.iter().begin() == entities.iter().end());
    CHECK(scene->materials().iter().begin() == scene->materials().iter().end());
    Entity* authored = scene->create_entity();
    Entity* builtin = scene->free_camera()->entity();
    std::vector<Entity*> filtered;
    for (Entity* entity : entities.iter())
        filtered.push_back(entity);
    CHECK(filtered == std::vector<Entity*>{authored});
    std::vector<Entity*> raw;
    for (Entity* entity : entities)
        raw.push_back(entity);
    CHECK(raw == std::vector<Entity*>{builtin, authored});
    CHECK_EQ(entities.size(), raw.size());
    for (Entity* entity : raw)
        CHECK_EQ(entities[entity->collection_index()], entity);
    filtered.clear();
    for (Entity* entity : entities.iter(true))
        filtered.push_back(entity);
    CHECK(filtered == raw);

    // An iterator obtained from a temporary range still borrows the persistent view.
    auto iterator = entities.iter().begin();
    CHECK_EQ(*iterator, authored);
    ++iterator;
    CHECK(iterator == entities.iter().end());
    const auto& const_entities = entities;
    std::vector<const Entity*> const_filtered;
    for (auto* entity : const_entities.iter()) {
        static_assert(std::is_same_v<decltype(entity), const Entity*>);
        const_filtered.push_back(entity);
    }
    CHECK(const_filtered == std::vector<const Entity*>{authored});

    std::vector<Entity*> dirty;
    scene->entity_collection().for_each(
        [&](Entity* entity, Entity::DirtyFlags)
        {
            dirty.push_back(entity);
        },
        Entity::DirtyFlags::added
    );
    CHECK(dirty == raw);
}

TEST_CASE_GPU("free camera structural protection is atomic")
{
    auto scene = Scene::create(ref(ctx.device));
    Camera* camera = scene->free_camera();
    REQUIRE(camera);
    Entity* entity = camera->entity();
    CHECK(camera->is_builtin());
    CHECK(entity->is_builtin());
    CHECK_EQ(scene->entities().size(), 1);
    CHECK_EQ(scene->components().size(), 1);
    CHECK_EQ(entity->transform().translation(), float3(0.f, 0.f, 5.f));
    Entity* child = scene->create_entity();
    child->set_parent(entity);
    CHECK_THROWS_WITH(entity->remove(), doctest::Contains("Cannot remove a built-in object."));
    CHECK(child->is_valid());
    CHECK(camera->is_valid());
    CHECK_THROWS_WITH(camera->remove(), doctest::Contains("Cannot remove a built-in object."));
    CHECK_THROWS_WITH(entity->set_parent(child), doctest::Contains("Cannot parent a built-in object."));
    CHECK_THROWS_WITH(entity->set_name("renamed"), doctest::Contains("Cannot rename a built-in object."));
    CHECK_THROWS_WITH(camera->set_name("renamed"), doctest::Contains("Cannot rename a built-in object."));
    CHECK_THROWS_WITH(
        entity->create_component<Camera>(),
        doctest::Contains("Cannot add components to a built-in object.")
    );
    CHECK_THROWS_WITH(
        entity->create_component("TransformAnimation"),
        doctest::Contains("Cannot add components to a built-in object.")
    );
    CHECK_THROWS_WITH(
        entity->create_component(typeid(Camera)),
        doctest::Contains("Cannot add components to a built-in object.")
    );
    scene->update();
    CHECK_EQ(entity->parent(), nullptr);
    CHECK_EQ(entity->name(), "free_camera");
    CHECK_EQ(entity->components().size(), 1);
    CHECK_EQ(scene->components().size(), 1);
    CHECK_EQ(scene->entities().size(), 2);
    auto* ordinary = scene->create_entity();
    ordinary->set_name("free_camera");
    CHECK_FALSE(ordinary->is_builtin());
    CHECK_NOTHROW(ordinary->remove());
}

TEST_CASE_GPU("camera selection survives updates imports and deletion")
{
    auto scene = Scene::create(ref(ctx.device));
    Camera* free_camera = scene->free_camera();
    CHECK_EQ(scene->active_camera(), free_camera);
    Transform saved_view;
    saved_view.set_translation(float3(10.f, 20.f, 30.f));
    free_camera->entity()->set_transform(saved_view);
    scene->update();
    CHECK_EQ(scene->active_camera(), free_camera);
    CHECK_EQ(free_camera->entity()->transform().translation(), saved_view.translation());

    bool explicit_free_camera = false;
    SUBCASE("automatic initial selection") { }
    SUBCASE("null selects the free camera")
    {
        scene->set_active_camera(nullptr);
        explicit_free_camera = true;
    }
    SUBCASE("explicit free camera")
    {
        scene->set_active_camera(free_camera);
        explicit_free_camera = true;
    }

    Entity* parent = scene->create_entity();
    Entity* entity = scene->create_entity();
    entity->set_parent(parent);
    Camera* first = entity->create_component<Camera>();
    Camera* expected = explicit_free_camera ? free_camera : first;
    CHECK_EQ(scene->active_camera(), free_camera);
    scene->update();
    CHECK_EQ(scene->active_camera(), expected);

    Camera* second = scene->create_entity()->create_component<Camera>();
    scene->update();
    CHECK_EQ(scene->active_camera(), expected);

    auto importer_scene = make_ref<ImporterScene>();
    importer_scene->cameras.push_back(ImporterCamera{});
    ImporterNode camera_node;
    camera_node.camera_index = 0;
    importer_scene->nodes.push_back(camera_node);
    importer_scene->root_nodes.push_back(0);
    scene->append(*importer_scene);
    CHECK_EQ(scene->active_camera(), expected);
    scene->update();
    CHECK_EQ(scene->active_camera(), expected);

    parent->remove();
    CHECK_EQ(scene->active_camera(), free_camera);
    scene->append(*importer_scene);
    CHECK_EQ(scene->active_camera(), free_camera);
    scene->update();
    CHECK_EQ(scene->active_camera(), free_camera);

    scene->create_entity()->create_component<Camera>();
    scene->update();
    CHECK_EQ(scene->active_camera(), free_camera);
    CHECK_EQ(free_camera->entity()->transform().translation(), saved_view.translation());

    scene->set_active_camera(second);
    scene->append(*importer_scene);
    CHECK_EQ(scene->active_camera(), second);
    scene->update();
    CHECK_EQ(scene->active_camera(), second);
}

TEST_CASE_GPU("copying a camera preserves the free view after source deletion")
{
    auto scene = Scene::create(ref(ctx.device));
    Camera* camera = scene->create_entity()->create_component<Camera>();
    Transform transform;
    transform.set_translation(float3(4.f, 5.f, 6.f));
    camera->entity()->set_transform(transform);
    camera->set_focal_length(35.f);
    scene->update();
    Camera* free_camera = scene->activate_free_camera_from(camera);
    CHECK_EQ(free_camera, scene->free_camera());
    CHECK_EQ(scene->active_camera(), free_camera);
    scene->set_active_camera(camera);
    auto* other = scene->create_entity()->create_component<Camera>();
    camera->remove();
    CHECK_EQ(scene->active_camera(), free_camera);
    scene->update();
    CHECK_EQ(scene->active_camera(), free_camera);
    CHECK_EQ(free_camera->entity()->transform().translation(), transform.translation());
    CHECK_EQ(free_camera->focal_length(), 35.f);
    CHECK(other->is_valid());
}

TEST_CASE_GPU("framing the free camera does not change selection or authored views")
{
    auto scene = Scene::create(ref(ctx.device));
    Camera* camera = scene->create_entity()->create_component<Camera>();
    Transform transform;
    transform.set_translation(float3(1.f, 2.f, 3.f));
    camera->entity()->set_transform(transform);
    camera->set_focal_length(35.f);
    camera->set_width(123);
    camera->set_height(456);
    scene->update();

    Camera* free_camera = scene->free_camera();
    free_camera->entity()->set_transform(transform);
    free_camera->set_focal_length(70.f);
    free_camera->set_depth_range(float2(2.f, 3.f));
    free_camera->set_focus_distance(42.f);
    free_camera->set_width(320);
    free_camera->set_height(240);
    scene->frame_free_camera();
    CHECK_EQ(scene->active_camera(), camera);
    CHECK_EQ(camera->entity()->transform().translation(), transform.translation());
    CHECK_EQ(camera->focal_length(), 35.f);
    CHECK_EQ(camera->width(), 123);
    CHECK_EQ(camera->height(), 456);
    CHECK_EQ(free_camera->entity()->transform().translation(), float3(0.f, 0.f, 5.f));
    CHECK_EQ(free_camera->focal_length(), 70.f);
    CHECK_EQ(free_camera->focus_distance(), 5.f);
    CHECK_EQ(free_camera->depth_range(), float2(0.01f, 1000.f));
    CHECK_EQ(free_camera->width(), 320);
    CHECK_EQ(free_camera->height(), 240);
    scene->frame_free_camera(2.f, float2(0.1f, 200.f));
    CHECK_EQ(free_camera->width(), 320);
    CHECK_EQ(free_camera->height(), 240);
    CHECK_EQ(free_camera->depth_range(), float2(0.1f, 200.f));
    CHECK_THROWS(scene->frame_free_camera(0.f));
}

TEST_CASE_GPU("importer scene camera becomes active and preserves projection properties")
{
    auto importer_scene = make_ref<ImporterScene>();

    ImporterCamera importer_camera;
    importer_camera.name = "Test Camera";
    importer_camera.focal_length = 50.f;
    importer_camera.fstop = 8.f;
    importer_camera.exposure = -2.f;
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
    CHECK_EQ(active_camera->exposure(), -2.f);
    CHECK_EQ(active_camera->calc_uniforms().exposure_scale, 0.25f);
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
    ref<Entity> e2(scene->create_entity());
    e0->set_name("e0");
    e1->set_name("e1");
    e2->set_name("e2");

    CHECK_EQ(scene->entities().size(), 4);

    // Remove first and last.
    e0->remove();
    e2->remove();

    scene->update();

    CHECK_EQ(scene->entities().size(), 2);
    CHECK_EQ(scene->entities()[1]->name(), "e1");
    CHECK_EQ(scene->entities()[1]->collection_index(), 1);
    REQUIRE_EQ(e2->scene(), nullptr);
    CHECK_EQ(e2->collection_index(), INVALID_SCENE_OBJECT_COLLECTION_INDEX);
    e2->remove();
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
    scene->update(); // Synchronize the built-in camera before testing entity-only changes.

    auto* entity = scene->create_entity();
    // 'added' flag is set on creation.
    CHECK(scene->entity_collection().has_dirty(Entity::DirtyFlags::added));

    const SceneUpdateFlags added_flags = scene->update();
    CHECK_FALSE(is_set(added_flags, SceneUpdateFlags::render_state));
    CHECK_FALSE(scene->entity_collection().is_dirty());

    // Modify transform.
    Transform t;
    t.set_translation(float3(1.f, 2.f, 3.f));
    entity->set_transform(t);

    CHECK(scene->entity_collection().is_dirty());
    CHECK(scene->entity_collection().has_dirty(Entity::DirtyFlags::transform));

    const SceneUpdateFlags transform_flags = scene->update();
    CHECK_FALSE(is_set(transform_flags, SceneUpdateFlags::render_state));
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

    CHECK_EQ(scene->components().size(), 2);
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

    CHECK_EQ(scene->components().size(), 3);
    CHECK_EQ(scene->entities().size(), 2);

    entity->remove();
    scene->update();

    CHECK_EQ(scene->entities().size(), 1);
    CHECK_EQ(scene->components().size(), 1);
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

    CHECK_EQ(scene->entities().size(), 3);
    CHECK_EQ(scene->components().size(), 2);

    // Removing parent should also remove child and its components.
    parent->remove();
    scene->update();

    CHECK_EQ(scene->entities().size(), 1);
    CHECK_EQ(scene->components().size(), 1);
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
    const MeshVertexAttributeDesc vertex_attributes[]{
        {"POSITION",
         MeshAttributeSemantic::position,
         0,
         sgl::DataType::float32,
         3,
         positions,
         sizeof(float3),
         sizeof(positions)},
        {"NORMAL",
         MeshAttributeSemantic::normal,
         0,
         sgl::DataType::float32,
         3,
         normals,
         sizeof(float3),
         sizeof(normals)},
        {"TANGENT",
         MeshAttributeSemantic::tangent,
         0,
         sgl::DataType::float32,
         3,
         tangents,
         sizeof(float3),
         sizeof(tangents)},
        {"HANDEDNESS",
         MeshAttributeSemantic::handedness,
         0,
         sgl::DataType::float32,
         1,
         handedness,
         sizeof(float),
         sizeof(handedness)},
        {"TEXCOORD_0",
         MeshAttributeSemantic::tex_coord,
         0,
         sgl::DataType::float32,
         2,
         texcoords,
         sizeof(float2),
         sizeof(texcoords)},
    };

    StaticMeshGeometryDataDesc desc = {};
    desc.name = "triangle";
    desc.vertex_count = 3;
    desc.vertex_attributes = vertex_attributes;
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

TEST_CASE_GPU("material backface culling setter updates existing geometry instances")
{
    auto scene = Scene::create(ref(ctx.device));
    auto* geometry = create_triangle_geometry(scene.get(), {0, 0, 0}, {1, 0, 0}, {0, 1, 0});
    auto* material = scene->create_material<StandardMaterial>();
    auto* instance = scene->create_entity()->create_component<GeometryInstance>();
    instance->set_geometry(geometry);
    instance->set_materials({material});
    scene->update();
    const auto instance_id = instance->geometry_instance_id();

    for (bool enabled : {true, false}) {
        material->set_enable_backface_culling(enabled);
        CHECK(scene->material_collection().has_dirty(Material::DirtyFlags::backface_culling));
        CHECK(is_set(scene->update(), SceneUpdateFlags::render_state));
        CHECK_EQ(instance->geometry_instance_id(), instance_id);

        material->set_enable_backface_culling(enabled);
        CHECK_FALSE(scene->material_collection().has_dirty(Material::DirtyFlags::backface_culling));
        CHECK_FALSE(is_set(scene->update(), SceneUpdateFlags::render_state));
    }
}

TEST_CASE_GPU("removed dirty geometry instances do not recreate render state")
{
    auto scene = Scene::create(ref(ctx.device));
    auto geometry = create_triangle_geometry(scene, float3(0.f), float3(1.f, 0.f, 0.f), float3(0.f, 1.f, 0.f));
    auto entity = scene->create_entity();
    ref<GeometryInstance> instance(entity->create_component<GeometryInstance>());
    instance->set_geometry(geometry);
    instance->set_material(0, scene->create_material<StandardMaterial>());
    scene->update();
    REQUIRE_NE(instance->geometry_instance_id(), shared::GeometryInstanceID::invalid);

    instance->set_material(0, nullptr);
    SUBCASE("remove component")
    {
        instance->remove();
    }
    SUBCASE("remove entity")
    {
        entity->remove();
    }
    scene->update();

    CHECK_EQ(instance->geometry_instance_id(), shared::GeometryInstanceID::invalid);
    CHECK_EQ(scene->components().size(), 1u);
}

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

static GeometryInstance* create_emissive_triangle(Scene* scene, Entity* entity)
{
    Properties properties;
    properties.set("emissive_factor", float3(1.f));
    auto* material = scene->create_material<StandardMaterial>(properties);
    auto* geometry = create_triangle_geometry(scene, float3(0.f), float3(1.f, 0.f, 0.f), float3(0.f, 1.f, 0.f));
    auto* instance = entity->create_component<GeometryInstance>();
    instance->set_geometry(geometry);
    instance->set_materials({material});
    return instance;
}

TEST_CASE_GPU("emissive transform stretch classification")
{
    auto scene = Scene::create(ref(ctx.device));
    auto* parent = scene->create_entity();
    auto* entity = scene->create_entity();
    entity->set_parent(parent);
    create_emissive_triangle(scene, entity);
    scene->update();

    const auto* emissive = scene->_emissive_geometry_system();
    REQUIRE_EQ(emissive->active_triangle_count(), 1u);
    const float initial_flux = emissive->active_triangle_flux()[0];
    REQUIRE_GT(initial_flux, 0.f);
    auto generations = emissive->generations();
    const auto synchronize = [&](bool expected_stretch_change)
    {
        const float previous_flux = emissive->active_triangle_flux()[0];
        // Reading the lazy matrix must not consume the classification's previous state.
        (void)entity->world_from_object_matrix();
        scene->update();
        CHECK_EQ(emissive->generations().topology, generations.topology);
        CHECK_EQ(emissive->generations().geometry, generations.geometry + 1);
        CHECK_EQ(emissive->generations().flux, generations.flux + uint64_t(expected_stretch_change));
        if (!expected_stretch_change)
            CHECK_EQ(emissive->active_triangle_flux()[0], previous_flux);
        generations = emissive->generations();
        CHECK_FALSE(scene->component_collection().is_dirty());
    };
    const auto check_area_and_flux = [&](float area_ratio)
    {
        REQUIRE_EQ(emissive->active_triangle_count(), 1u);
        CHECK(emissive->triangles()[0].area_ws == doctest::Approx(0.5f * area_ratio));
        CHECK(emissive->active_triangle_flux()[0] == doctest::Approx(initial_flux * area_ratio));
        CHECK_EQ(emissive->triangle_flux()[0], emissive->active_triangle_flux()[0]);
    };

    SUBCASE("rigid motion preserves nonuniform stretch")
    {
        Transform transform;
        transform.set_translation(float3(1.f, 2.f, 3.f));
        entity->set_transform(transform);
        synchronize(false);
        check_area_and_flux(1.f);
        CHECK_EQ(emissive->triangles()[0].pos_ws[0], transform.translation());
        transform.set_scale(float3(2.f, 3.f, 4.f));
        entity->set_transform(transform);
        synchronize(true);
        check_area_and_flux(6.f);
        transform.set_rotation_xyz(float3(0.3f, 0.6f, 0.9f));
        entity->set_transform(transform);
        synchronize(false);
        check_area_and_flux(6.f);

        Transform parent_transform;
        parent_transform.set_rotation_xyz(float3(0.7f, -0.3f, 0.5f));
        parent_transform.set_translation(float3(-3.f, 4.f, 1.f));
        parent->set_transform(parent_transform);
        synchronize(false);
        check_area_and_flux(6.f);
        entity->set_parent(nullptr);
        synchronize(false);
        check_area_and_flux(6.f);

        // A world reflection preserves area but still requires geometry/normal updates.
        parent_transform.set_scale(float3(-1.f, 1.f, 1.f));
        parent->set_transform(parent_transform);
        entity->set_parent(parent);
        synchronize(false);
        check_area_and_flux(6.f);
    }

    SUBCASE("child rotation beneath nonuniform scale changes stretch")
    {
        Transform transform;
        transform.set_scale(float3(2.f, 3.f, 4.f));
        entity->set_transform(transform);
        synchronize(true);
        // Moving the same scale from child to parent preserves world stretch.
        entity->set_transform(Transform());
        Transform parent_transform;
        parent_transform.set_scale(float3(2.f, 3.f, 4.f));
        parent->set_transform(parent_transform);
        synchronize(false);
        check_area_and_flux(6.f);
        transform = Transform();
        // Rotating the XY triangle into XZ changes area from 2*3 to 2*4.
        transform.set_rotation_xyz(float3(float(M_PI) / 2.f, 0.f, 0.f));
        entity->set_transform(transform);
        synchronize(true);
        check_area_and_flux(8.f);
        entity->set_parent(nullptr);
        synchronize(true);
        check_area_and_flux(1.f);
    }

    SUBCASE("rotation before scale changes stretch")
    {
        Transform transform;
        transform.set_composition_order(Transform::CompositionOrder::rst);
        transform.set_scale(float3(2.f, 3.f, 4.f));
        entity->set_transform(transform);
        synchronize(true);
        check_area_and_flux(6.f);
        transform.set_rotation_xyz(float3(float(M_PI) / 2.f, 0.f, 0.f));
        entity->set_transform(transform);
        synchronize(true);
        check_area_and_flux(8.f);
    }

    SUBCASE("small scale changes accumulate against a retained reference")
    {
        bool detected = false;
        for (uint32_t i = 1; i <= 32; ++i) {
            Transform transform;
            transform.set_scale(float3(1.f + float(i) * std::numeric_limits<float>::epsilon()));
            entity->set_transform(transform);
            (void)entity->world_from_object_matrix();
            scene->update();
            detected |= emissive->generations().flux != generations.flux;
        }
        CHECK(detected);
    }

    SUBCASE("small and anisotropic scales retain relative precision")
    {
        for (const float3 scale : {float3(1e-10f), float3(1e-10f, 1e10f, 1.f)}) {
            Transform transform;
            transform.set_scale(scale);
            entity->set_transform(transform);
            synchronize(true);
            transform.set_rotation_xyz(float3(0.3f, 0.6f, 0.9f));
            entity->set_transform(transform);
            synchronize(false);
            transform.set_scale(scale * 1.01f);
            entity->set_transform(transform);
            synchronize(true);
        }
    }

    SUBCASE("singular transforms conservatively invalidate stretch")
    {
        Transform transform;
        transform.set_scale(float3(0.f, 1.f, 1.f));
        entity->set_transform(transform);
        synchronize(true);
        check_area_and_flux(0.f);
        transform.set_translation(float3(1.f));
        entity->set_transform(transform);
        synchronize(true);
        check_area_and_flux(0.f);
        transform.set_scale(float3(1.f));
        entity->set_transform(transform);
        synchronize(true);
        check_area_and_flux(1.f);
    }
}

TEST_CASE_GPU("emissive material edits and motion share one update")
{
    auto scene = Scene::create(ref(ctx.device));
    auto* entity = scene->create_entity();
    auto* instance = create_emissive_triangle(scene, entity);
    auto* material = instance->materials()[0];
    scene->update();

    const auto* emissive = scene->_emissive_geometry_system();
    REQUIRE_EQ(emissive->active_triangle_count(), 1u);
    const float initial_flux = emissive->active_triangle_flux()[0];
    REQUIRE_GT(initial_flux, 0.f);
    auto generations = emissive->generations();

    // Material-only rebuilds must not report a geometry change.
    material->set_property("emissive_factor", float3(2.f));
    scene->update();
    CHECK_EQ(emissive->generations().topology, generations.topology + 1);
    CHECK_EQ(emissive->generations().geometry, generations.geometry);
    CHECK_EQ(emissive->generations().flux, generations.flux + 1);
    CHECK(emissive->active_triangle_flux()[0] == doctest::Approx(2.f * initial_flux));
    generations = emissive->generations();

    Transform transform;
    transform.set_translation(float3(3.f, 2.f, 1.f));
    transform.set_scale(float3(2.f, 3.f, 4.f));
    const float area_ratio = transform.scale().x * transform.scale().y;
    material->set_property("emissive_factor", float3(3.f));
    entity->set_transform(transform);
    (void)entity->world_from_object_matrix();
    const auto flags = scene->update();
    CHECK(is_set(flags, SceneUpdateFlags::emissive_geometry));
    CHECK_EQ(emissive->generations().topology, generations.topology + 1);
    CHECK_EQ(emissive->generations().geometry, generations.geometry + 1);
    CHECK_EQ(emissive->generations().flux, generations.flux + 1);
    REQUIRE_EQ(emissive->triangle_count(), 1u);
    REQUIRE_EQ(emissive->active_triangle_count(), 1u);
    CHECK_EQ(emissive->triangles()[0].pos_ws[0], transform.translation());
    CHECK(emissive->triangles()[0].area_ws == doctest::Approx(0.5f * area_ratio));
    CHECK(emissive->active_triangle_flux()[0] == doctest::Approx(3.f * area_ratio * initial_flux));
    CHECK_EQ(emissive->triangle_flux()[0], emissive->active_triangle_flux()[0]);
    generations = emissive->generations();
    const float flux = emissive->active_triangle_flux()[0];

    // Rebuilding membership must anchor the reference at the new scale.
    transform.set_translation(float3(4.f, 2.f, 1.f));
    entity->set_transform(transform);
    scene->update();
    CHECK_EQ(emissive->generations().topology, generations.topology);
    CHECK_EQ(emissive->generations().geometry, generations.geometry + 1);
    CHECK_EQ(emissive->generations().flux, generations.flux);
    CHECK_EQ(emissive->triangles()[0].pos_ws[0], transform.translation());
    CHECK_EQ(emissive->active_triangle_flux()[0], flux);
    generations = emissive->generations();
    CHECK_FALSE(is_set(scene->update(), SceneUpdateFlags::emissive_geometry));
    CHECK(emissive->generations() == generations);
}

TEST_CASE_GPU("emissive flux detects area changes in a nearly parallel world basis")
{
    auto scene = Scene::create(ref(ctx.device));
    auto* parent = scene->create_entity();
    auto* entity = scene->create_entity();
    entity->set_parent(parent);
    create_emissive_triangle(scene, entity);
    Transform transform;
    transform.set_rotation_xyz(float3(0.f, 0.f, float(M_PI) / 4.f));
    entity->set_transform(transform);

    // Exercise both conservative fallback and a skewed but still comparable metric.
    for (const auto [scale, multiplier] : {std::array{1e-4f, 2.f}, std::array{1e-2f, 1.001f}}) {
        CAPTURE(scale);
        CAPTURE(multiplier);
        Transform parent_transform;
        parent_transform.set_scale(float3(1.f, scale, 1.f));
        parent->set_transform(parent_transform);
        scene->update();
        const auto* emissive = scene->_emissive_geometry_system();
        REQUIRE_EQ(emissive->active_triangle_count(), 1u);
        const auto generations = emissive->generations();
        const float flux = emissive->active_triangle_flux()[0];
        REQUIRE_GT(flux, 0.f);
        parent_transform.set_scale(float3(1.f, multiplier * scale, 1.f));
        parent->set_transform(parent_transform);
        (void)entity->world_from_object_matrix();
        scene->update();
        CHECK_EQ(emissive->generations().topology, generations.topology);
        CHECK_EQ(emissive->generations().geometry, generations.geometry + 1);
        CHECK_EQ(emissive->generations().flux, generations.flux + 1);
        CHECK(emissive->triangles()[0].area_ws == doctest::Approx(0.5f * multiplier * scale));
        CHECK(emissive->active_triangle_flux()[0] == doctest::Approx(multiplier * flux));
        CHECK_EQ(emissive->triangle_flux()[0], emissive->active_triangle_flux()[0]);
    }
}

TEST_CASE_GPU("emissive flux detects changed basis angles with unchanged basis lengths")
{
    for (uint32_t axis = 0; axis < 3; ++axis) {
        CAPTURE(axis);
        auto scene = Scene::create(ref(ctx.device));
        Properties properties;
        properties.set("emissive_factor", float3(1.f));
        auto* material = scene->create_material<StandardMaterial>(properties);
        const uint32_t i = (axis + 1) % 3;
        const uint32_t j = (axis + 2) % 3;
        float3 edge0(0.f);
        edge0[i] = edge0[j] = 1.f;
        float3 edge1(0.f);
        edge1[axis] = 1.f;
        auto* geometry = create_triangle_geometry(scene, float3(0.f), edge0, edge1);
        auto* parent = scene->create_entity();
        const float3 scale(2.f, 3.f, 4.f);
        Transform parent_transform;
        parent_transform.set_scale(scale);
        parent->set_transform(parent_transform);
        auto* entity = scene->create_entity();
        entity->set_parent(parent);
        auto* instance = entity->create_component<GeometryInstance>();
        instance->set_geometry(geometry);
        instance->set_materials({material});
        float3 angles(0.f);
        angles[axis] = float(M_PI) / 4.f;
        Transform transform;
        transform.set_rotation_xyz(angles);
        entity->set_transform(transform);
        scene->update();
        const auto* emissive = scene->_emissive_geometry_system();
        REQUIRE_EQ(emissive->active_triangle_count(), 1u);
        const float flux = emissive->active_triangle_flux()[0];
        REQUIRE_GT(flux, 0.f);
        const auto generations = emissive->generations();

        // Opposite 45-degree rotations preserve the metric diagonal but reverse its off-diagonal entry.
        // The triangle's diagonal edge moves between the two differently scaled parent axes.
        transform.set_rotation_xyz(-angles);
        entity->set_transform(transform);
        scene->update();
        CHECK_EQ(emissive->generations().topology, generations.topology);
        CHECK_EQ(emissive->generations().geometry, generations.geometry + 1);
        CHECK_EQ(emissive->generations().flux, generations.flux + 1);
        CHECK(emissive->active_triangle_flux()[0] == doctest::Approx(flux * scale[i] / scale[j]));
        CHECK_EQ(emissive->triangle_flux()[0], emissive->active_triangle_flux()[0]);
    }
}

TEST_CASE_GPU("emissive transform cache follows contributing entities")
{
    auto scene = Scene::create(ref(ctx.device));
    Properties properties;
    properties.set("emissive_factor", float3(1.f));
    auto* material = scene->create_material<StandardMaterial>(properties);
    auto* dark_material = scene->create_material<StandardMaterial>();
    auto* geometry = create_triangle_geometry(scene, float3(0.f), float3(1.f, 0.f, 0.f), float3(0.f, 1.f, 0.f));
    auto* dark_entity = scene->create_entity();
    auto* dark_instance = dark_entity->create_component<GeometryInstance>();
    dark_instance->set_geometry(geometry);
    dark_instance->set_materials({dark_material});
    auto* entity = scene->create_entity();
    // Membership must search past non-geometry, empty and non-emissive components on the same entity.
    entity->create_component<Camera>();
    entity->create_component<GeometryInstance>();
    auto* non_emissive_instance = entity->create_component<GeometryInstance>();
    non_emissive_instance->set_geometry(geometry);
    non_emissive_instance->set_materials({dark_material});
    auto* instance = entity->create_component<GeometryInstance>();
    instance->set_geometry(geometry);
    instance->set_materials({material});
    scene->update();
    const auto* emissive = scene->_emissive_geometry_system();

    SUBCASE("non-emissive motion leaves all emissive generations unchanged")
    {
        const auto generations = emissive->generations();
        Transform transform;
        transform.set_scale(float3(2.f, 3.f, 4.f));
        dark_entity->set_transform(transform);
        const auto flags = scene->update();
        CHECK(is_set(flags, SceneUpdateFlags::transforms));
        CHECK_FALSE(is_set(flags, SceneUpdateFlags::emissive_geometry));
        CHECK(emissive->generations() == generations);
    }

    SUBCASE("material reassignment refreshes membership")
    {
        instance->set_materials({dark_material});
        dark_instance->set_materials({material});
        scene->update();
        auto generations = emissive->generations();
        Transform transform;
        transform.set_scale(float3(2.f));
        entity->set_transform(transform);
        scene->update();
        CHECK(emissive->generations() == generations);
        dark_entity->set_transform(transform);
        scene->update();
        CHECK_EQ(emissive->generations().flux, generations.flux + 1);
        CHECK(emissive->triangles()[0].area_ws == doctest::Approx(2.f));
    }

    SUBCASE("multiple components retain membership until the last emitter is removed")
    {
        auto* second_instance = entity->create_component<GeometryInstance>();
        second_instance->set_geometry(geometry);
        second_instance->set_materials({material});
        scene->update();
        REQUIRE_EQ(emissive->triangle_count(), 2u);
        instance->remove();
        scene->update();
        REQUIRE_EQ(emissive->triangle_count(), 1u);
        auto generations = emissive->generations();
        Transform transform;
        transform.set_scale(float3(2.f));
        entity->set_transform(transform);
        scene->update();
        CHECK_EQ(emissive->generations().flux, generations.flux + 1);
        second_instance->remove();
        scene->update();
        REQUIRE_EQ(emissive->triangle_count(), 0u);
        CHECK(emissive->triangles().empty());
        CHECK(emissive->triangle_flux().empty());
        CHECK(emissive->active_triangle_ids().empty());
        CHECK(emissive->active_triangle_flux().empty());
        generations = emissive->generations();
        transform.set_scale(float3(3.f));
        entity->set_transform(transform);
        scene->update();
        CHECK(emissive->generations() == generations);
    }

    SUBCASE("entity removal and collection compaction preserve surviving membership")
    {
        dark_instance->set_materials({material});
        scene->update();
        dark_entity->remove();
        scene->update();
        REQUIRE_EQ(emissive->triangle_count(), 1u);
        const auto generations = emissive->generations();
        Transform transform;
        transform.set_scale(float3(2.f));
        entity->set_transform(transform);
        scene->update();
        CHECK_EQ(emissive->generations().flux, generations.flux + 1);
        entity->remove();
        scene->update();
        REQUIRE_EQ(emissive->triangle_count(), 0u);
        scene->create_entity();
        scene->update();
    }

    SUBCASE("initially zero-area emitters recover after scaling")
    {
        Transform transform;
        transform.set_scale(float3(0.f, 1.f, 1.f));
        entity->set_transform(transform);
        // Force emission setup while the emitter is degenerate.
        instance->set_materials({dark_material});
        instance->set_materials({material});
        scene->update();
        REQUIRE_EQ(emissive->triangle_count(), 1u);
        REQUIRE_EQ(emissive->active_triangle_count(), 1u);
        CHECK_EQ(emissive->active_triangle_flux()[0], 0.f);
        const auto generations = emissive->generations();
        entity->set_transform(Transform());
        scene->update();
        CHECK_EQ(emissive->generations().flux, generations.flux + 1);
        CHECK_GT(emissive->active_triangle_flux()[0], 0.f);
    }
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
(
    metersPerUnit = 1
)

def RectLight "VisibleRect"
{
    bool inputs:enableColorTemperature = 1
    float inputs:colorTemperature = 3000
    float inputs:height = 2
    float inputs:intensity = 4
    bool inputs:normalize = 1
    float inputs:width = 3
    float3 xformOp:scale = (2, 3, 1)
    uniform token[] xformOpOrder = ["xformOp:scale"]
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
    REQUIRE_EQ(imported_scene->materials.size(), 2);
    const auto rect_material = std::find_if(
        imported_scene->materials.begin(),
        imported_scene->materials.end(),
        [](const ImporterMaterial& material)
        {
            return material.name.find("VisibleRect") != std::string::npos;
        }
    );
    REQUIRE(rect_material != imported_scene->materials.end());
    CHECK_EQ(rect_material->params.get<float>("emission_luminance"), doctest::Approx(1.f / 9.f));

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

// ---------------------------------------------------------------------------
// Scene source asset reference lookup tests
// ---------------------------------------------------------------------------

TEST_CASE_GPU("scene source lookup survives renaming and removal")
{
    auto scene = Scene::create(ref(ctx.device));
    auto* first = scene->create_material<StandardMaterial>();
    auto* second = scene->create_material<StandardMaterial>();
    const SourceAssetRef source("usd", "asset.usda", "/Paint");
    first->_set_source_asset_ref(source);
    second->_set_source_asset_ref(source);
    CHECK_EQ(scene->find_objects_by_source_asset_ref(source).size(), 2);
    first->set_name("Renamed");
    CHECK_EQ(scene->find_objects_by_source_asset_ref(source).size(), 2);
    first->remove();
    CHECK(scene->find_objects_by_source_asset_ref(source) == std::vector<SceneObject*>{second});
    scene->update();
    CHECK(scene->find_objects_by_source_asset_ref(source) == std::vector<SceneObject*>{second});
    second->remove();
    CHECK(scene->find_objects_by_source_asset_ref(source).empty());
}

TEST_CASE_GPU("scene source lookup scans all object kinds and reflects source changes")
{
    auto scene = Scene::create(ref(ctx.device));
    auto* material = scene->create_material<StandardMaterial>();
    auto* geometry = scene->create_geometry<StaticMeshGeometry>();
    auto* animation = scene->create_animation();
    auto* entity = scene->create_entity();
    auto* camera = entity->create_component<Camera>();
    const std::vector<SceneObject*> objects{material, geometry, animation, entity, camera};
    const SourceAssetRef source("usd", "asset.usda", "/Shared");
    const SourceAssetRef other_source("usd", "other.usda", "/Shared");
    CHECK(scene->find_objects_by_source_asset_ref({}).empty());
    for (auto* object : objects) {
        CHECK(object->source_asset_ref() == nullptr);
        object->_set_source_asset_ref(source);
    }
    const auto found = scene->find_objects_by_source_asset_ref(source);
    CHECK_EQ(found.size(), objects.size());
    for (auto* object : objects)
        CHECK(std::find(found.begin(), found.end(), object) != found.end());
    CHECK(scene->find_objects_by_source_asset_ref(other_source).empty());

    material->_set_source_asset_ref(other_source);
    CHECK(scene->find_objects_by_source_asset_ref(other_source) == std::vector<SceneObject*>{material});
    CHECK_EQ(scene->find_objects_by_source_asset_ref(source).size(), 4);
    material->_set_source_asset_ref(std::nullopt);
    CHECK(material->source_asset_ref() == nullptr);
    CHECK(scene->find_objects_by_source_asset_ref(other_source).empty());
    geometry->_set_source_asset_ref(SourceAssetRef{});
    REQUIRE(geometry->source_asset_ref());
    CHECK(*geometry->source_asset_ref() == SourceAssetRef{});
    CHECK(scene->find_objects_by_source_asset_ref({}) == std::vector<SceneObject*>{geometry});
    geometry->_set_source_asset_ref(std::nullopt);
    CHECK(geometry->source_asset_ref() == nullptr);
    CHECK(scene->find_objects_by_source_asset_ref({}).empty());

    // Removing an entity immediately excludes both it and its components, before update/compaction.
    entity->remove();
    CHECK(scene->find_objects_by_source_asset_ref(source) == std::vector<SceneObject*>{animation});
}

TEST_SUITE_END();
