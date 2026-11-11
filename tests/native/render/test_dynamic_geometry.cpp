// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "falcor2/render/geometry/vertex_cache_deformer.h"
#include "testing.h"
#include "falcor2/render/geometry/geometry_animation.h"
#include "falcor2/render/utils/geometry_dispatch.h"
#include "falcor2/render/geometry/dynamic_mesh_geometry.h"
#include "falcor2/render/geometry/dynamic_curve_geometry.h"
#include "falcor2/render/scene.h"
#include "falcor2/render/component/geometry_instance.h"
#include "falcor2/render/material/emission_only_material.h"
#include "falcor2/render/emissive_geometry_system.h"
using namespace falcor;

TEST_SUITE_BEGIN("DynamicGeometry");

TEST_CASE("dispatch_limits")
{
    const auto small = geometry_dispatch_max_tile_size({32, 1, 1}, {3, 2, 1});
    CHECK(small.x == 96);
    CHECK(small.y == 2);
    const auto d3d = geometry_dispatch_max_tile_size({64, 1, 1}, {65535, 65535, 65535});
    CHECK(d3d.x == 4194240);
    CHECK(d3d.y == 65535);
    const auto large = geometry_dispatch_max_tile_size({64, 1, 1}, {UINT32_MAX, UINT32_MAX, UINT32_MAX});
    CHECK(large.x % 64 == 0);
    CHECK(uint64_t(large.x) + 63 <= UINT32_MAX);
    CHECK(large.y == UINT32_MAX);
}

TEST_CASE("selection_and_history")
{
    const float times[] = {-2, 1, 3};
    auto select = [&](double time, AnimationInterpolationMode mode = AnimationInterpolationMode::linear)
    {
        return select_geometry_samples(time, times, mode);
    };
    CHECK(select(-3).source0 == 0);
    CHECK(select(4).source0 == 2);
    CHECK(select(1).source0 == 1);
    CHECK(select(1).source1 == 1);
    CHECK(select(-0.5).source1_weight == 0.5f);
    CHECK(select(0, AnimationInterpolationMode::step).source0 == 0);
    CHECK(select(1, AnimationInterpolationMode::step).source0 == 1);
    GeometryAnimationHistory history;
    using HistoryUpdate = shared::GeometryHistoryUpdate;
    CHECK(history.update(false) == HistoryUpdate::initialize);
    CHECK(history.update(false) == HistoryUpdate::none);
    CHECK(history.update(true) == HistoryUpdate::capture_current);
    CHECK(history.update(false) == HistoryUpdate::catch_up);
    CHECK(history.update(false) == HistoryUpdate::none);
    CHECK(history.update(true) == HistoryUpdate::capture_current);
    CHECK(history.update(true) == HistoryUpdate::capture_current);
    CHECK(history.update(false) == HistoryUpdate::catch_up);
    CHECK(history.update(false) == HistoryUpdate::none);
    history = {};
    CHECK(history.update(false) == HistoryUpdate::initialize);
}
TEST_CASE("mapped_selection_and_history")
{
    const float times[] = {-2, 0, 0, 2, 4};
    const uint32_t samples[] = {2, 1, 0, 2, 2};
    auto select = [&](double time, AnimationInterpolationMode mode = AnimationInterpolationMode::linear)
    {
        return select_geometry_samples(time, times, mode, samples);
    };
    CHECK(select(-3).source0 == 2);
    CHECK(select(-1).source0 == 2);
    CHECK(select(-1).source1 == 1);
    CHECK(select(-1).source1_weight == 0.5f);
    CHECK(select(0).source0 == 0);
    CHECK(select(0).source1 == 0);
    CHECK(select(1).source0 == 0);
    CHECK(select(1).source1 == 2);
    CHECK(select(1).source1_weight == 0.5f);
    CHECK(select(5).source0 == 2);
    CHECK(select(-1, AnimationInterpolationMode::step).source0 == 2);
    CHECK(select(0, AnimationInterpolationMode::step).source0 == 0);
    // Different timeline keys can refer to the same evaluated source selection.
    CHECK(select(-2) == select(2));
    CHECK(select(2) == select(3));
    CHECK(select(3) == select(4));
}
// An external input selects the pose independently of Scene.time, using the existing cache kernel.
class ControlledCacheDeformer : public VertexCacheDeformer<shared::PackedTriangleVertex> {
public:
    using VertexCacheDeformer::VertexCacheDeformer;
    double input_time = -1;
    size_t update_count = 0;
    void prepare(sgl::CommandEncoder* encoder, double, bool initialize) override
    {
        ++update_count;
        VertexCacheDeformer::prepare(encoder, input_time, initialize);
    }
    AnimationTimeRange animation_range() const override { return {}; }
    bool has_animation() const override { return false; }
};

TEST_CASE_GPU_NIGHTLY("remapping_history_and_emissive_deformation")
{
    auto set_mesh_samples = [&](DynamicMeshGeometry& geometry,
                                std::span<const StaticMeshGeometryDataDesc> samples,
                                std::span<const float> times,
                                AnimationInterpolationMode interpolation,
                                std::span<const uint32_t> sample_indices = {})
    {
        std::vector<MeshGeometryData> sub_meshes;
        auto deformer = create_vertex_cache(ctx.device, samples, times, interpolation, sub_meshes, sample_indices);
        geometry.set_mesh_data(std::move(sub_meshes), std::move(deformer));
    };
    auto set_curve_samples = [&](DynamicCurveGeometry& geometry,
                                 std::span<const StaticCurveGeometryDataDesc> samples,
                                 std::span<const float> times,
                                 AnimationInterpolationMode interpolation,
                                 std::span<const uint32_t> sample_indices = {})
    {
        std::vector<shared::PackedLSSVertex> vertices;
        std::vector<uint32_t> indices;
        std::vector<GeometryVertexAttribute> attributes;
        auto deformer = create_vertex_cache(
            ctx.device,
            samples,
            times,
            interpolation,
            vertices,
            indices,
            attributes,
            sample_indices
        );
        geometry.set_curve_data(std::move(vertices), std::move(indices), std::move(attributes), std::move(deformer));
    };
    auto scene = Scene::create(ref(ctx.device));
    auto* render_scene = scene->_render_scene();
    using MeshUpdater = VertexCacheUpdater<shared::PackedTriangleVertex>;
    using CurveUpdater = VertexCacheUpdater<shared::PackedLSSVertex>;
    auto& mesh_updater = render_scene->get_or_create_geometry_updater<MeshUpdater>();
    CHECK(&mesh_updater == &render_scene->get_or_create_geometry_updater<MeshUpdater>());
    auto& curve_updater = render_scene->get_or_create_geometry_updater<CurveUpdater>();
    CHECK(static_cast<GeometryUpdater*>(&mesh_updater) != static_cast<GeometryUpdater*>(&curve_updater));
    auto* mesh = scene->create_geometry<DynamicMeshGeometry>();
    const float times[] = {-1, 1};
    const float3 points[][3] = {{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}}, {{0, 0, 2}, {2, 0, 2}, {0, 2, 2}}};
    const uint32_t indices[][3] = {{2, 0, 1}, {1, 2, 0}};
    const float3 normals[][3] = {{{0, 0, 1}, {0, 0, 1}, {0, 0, 1}}, {{0, 1, 0}, {0, 1, 0}, {0, 1, 0}}};
    const float3 tangents[] = {{1, 0, 0}, {1, 0, 0}, {1, 0, 0}};
    const float2 uvs[] = {{0.1f, 0.2f}, {0.3f, 0.4f}, {0.5f, 0.6f}};
    MeshVertexAttributeDesc attrs[2][4];
    StaticMeshGeometryDataDesc samples[2];
    for (size_t i = 0; i < 2; ++i) {
        attrs[i][0]
            = {.semantic = MeshAttributeSemantic::position,
               .type = sgl::DataType::float32,
               .component_count = 3,
               .data = points[i],
               .stride = sizeof(float3),
               .data_size = sizeof(points[i])};
        samples[i].vertex_count = 3;
        attrs[i][1]
            = {.semantic = MeshAttributeSemantic::normal,
               .type = sgl::DataType::float32,
               .component_count = 3,
               .data = normals[i],
               .stride = sizeof(float3),
               .data_size = sizeof(normals[i])};
        attrs[i][2]
            = {.semantic = MeshAttributeSemantic::tangent,
               .type = sgl::DataType::float32,
               .component_count = 3,
               .data = tangents,
               .stride = sizeof(float3),
               .data_size = sizeof(tangents)};
        attrs[i][3]
            = {.semantic = MeshAttributeSemantic::tex_coord,
               .type = sgl::DataType::float32,
               .component_count = 2,
               .data = uvs,
               .stride = sizeof(float2),
               .data_size = sizeof(uvs)};
        samples[i].vertex_attributes = attrs[i];
        for (const auto& part : indices)
            samples[i].sub_meshes.push_back({.index_count = 3, .index_stream = {part, sizeof(uint32_t)}});
    }
    set_mesh_samples(*mesh, samples, times, AnimationInterpolationMode::linear);
    CHECK(!scene->has_animation());
    CHECK(!scene->animation_range().is_valid());
    auto* instance = scene->create_entity()->create_component<GeometryInstance>();
    instance->set_geometry(mesh);
    // Queries reflect instance edits immediately, before Scene::update().
    CHECK(scene->has_animation());
    CHECK(scene->animation_range().start == -1);
    CHECK(scene->animation_range().end == 1);
    instance->set_geometry(nullptr);
    CHECK(!scene->animation_range().is_valid());
    CHECK(!scene->has_animation());
    instance->set_geometry(mesh);
    CHECK(scene->has_animation());
    auto* material = scene->create_material<EmissionOnlyMaterial>();
    material->set_property("emission_luminance", 1.f);
    instance->set_materials({material, material});
    auto* curve = scene->create_geometry<DynamicCurveGeometry>();
    const float radii[][3] = {{0.1f, 0.2f, 0.3f}, {0.3f, 0.4f, 0.5f}};
    const uint32_t endpoints[] = {2, 0, 1, 2};
    StaticCurveGeometryDataDesc curve_samples[2];
    for (size_t i = 0; i < 2; ++i) {
        curve_samples[i].vertex_count = 3;
        curve_samples[i].position_stream = {points[i], sizeof(float3)};
        curve_samples[i].radius_stream = {radii[i], sizeof(float)};
        curve_samples[i].index_count = 4;
        curve_samples[i].index_stream = {endpoints, sizeof(uint32_t)};
    }
    set_curve_samples(*curve, curve_samples, times, AnimationInterpolationMode::step);
    // Both lifecycle callbacks run for geometry removed before its first update.
    scene->create_geometry<DynamicMeshGeometry>()->remove();
    scene->create_geometry<DynamicCurveGeometry>()->remove();
    scene->set_time(0);
    scene->update();
    const RenderScene& render = *scene->_render_scene();
    auto mesh_handles = render.get_geometry_group(mesh->_render_geometry_group()).geometries;
    REQUIRE(mesh_handles.size() == 2);
    auto curve_handle
        = RenderLSSGeometryHandle{render.get_geometry_group(curve->_render_geometry_group()).geometries.front()};
    auto check_mesh = [&](float current_z, float previous_z)
    {
        for (size_t part = 0; part < mesh_handles.size(); ++part) {
            auto handle = RenderTriangleGeometryHandle{mesh_handles[part]};
            const auto& geometry = render.get_triangle_geometry(handle);
            for (uint16_t member = 0; member < 2; ++member) {
                auto vertices
                    = render.triangle_vertex_buffer(handle, member)->get_elements<shared::PackedTriangleVertex>();
                for (size_t i = 0; i < 3; ++i)
                    CHECK(
                        vertices[geometry.vertex_offset + i].position.z
                        == doctest::Approx(member ? previous_z : current_z)
                    );
                const auto vertex = shared::detail::unpack_triangle_vertex(vertices[geometry.vertex_offset]);
                const float weight = (member ? previous_z : current_z) / 2;
                const float3 normal = sgl::math::normalize(float3(0, weight, 1 - weight));
                CHECK(vertex.normal.y == doctest::Approx(normal.y).epsilon(0.002f));
                CHECK(vertex.normal.z == doctest::Approx(normal.z).epsilon(0.002f));
                CHECK(vertex.tangent.x == doctest::Approx(1.f).epsilon(0.002f));
                CHECK(vertex.uv[0].x == uvs[indices[part][0]].x);
                const float scale = 1 + (member ? previous_z : current_z) / 2;
                CHECK(
                    vertices[geometry.vertex_offset].position.x
                    == doctest::Approx(points[0][indices[part][0]].x * scale)
                );
                CHECK(
                    vertices[geometry.vertex_offset].position.y
                    == doctest::Approx(points[0][indices[part][0]].y * scale)
                );
            }
        }
    };
    check_mesh(1, 1);
    scene->set_time(-1);
    scene->update();
    check_mesh(0, 1);
    scene->update();
    check_mesh(0, 0);
    REQUIRE(scene->_emissive_geometry_system()->active_triangle_count() == 2);
    const float initial_flux = scene->_emissive_geometry_system()->active_triangle_flux()[0];
    auto before = scene->_emissive_geometry_system()->generations();
    scene->set_time(0);
    scene->update();
    check_mesh(1, 0);
    auto after = scene->_emissive_geometry_system()->generations();
    CHECK(after.topology == before.topology);
    CHECK(after.geometry > before.geometry);
    CHECK(after.flux > before.flux);
    CHECK(scene->_emissive_geometry_system()->active_triangle_flux()[0] > initial_flux);
    const auto& curve_output = render.get_lss_geometry(curve_handle);
    auto vertices = render.lss_vertex_buffer(curve_handle, 0)->get_elements<shared::PackedLSSVertex>();
    CHECK(vertices[curve_output.vertex_offset].radius == doctest::Approx(0.3f));
    CHECK(vertices[curve_output.vertex_offset].position.z == 0);
    const auto blas = render.get_geometry_group(mesh->_render_geometry_group()).blas;
    scene->update();
    check_mesh(1, 1);
    CHECK(render.get_geometry_group(mesh->_render_geometry_group()).blas == blas);
    CHECK(scene->_emissive_geometry_system()->generations() == after);
    scene->update();
    check_mesh(1, 1);
    scene->set_time(1);
    scene->update();
    check_mesh(2, 1);
    vertices = render.lss_vertex_buffer(curve_handle, 0)->get_elements<shared::PackedLSSVertex>();
    CHECK(vertices[curve_output.vertex_offset].radius == doctest::Approx(0.5f));
    scene->set_time(-1);
    scene->update();
    check_mesh(0, 2);
    CHECK(scene->animation_range().end == 1);
    // Non-emissive motion must not invalidate stationary lighting, even on an emitter's entity.
    const float later_times[] = {1, 3};
    auto* non_emissive_mesh = scene->create_geometry<DynamicMeshGeometry>();
    set_mesh_samples(*non_emissive_mesh, samples, later_times, AnimationInterpolationMode::linear);
    auto* non_emissive_instance = instance->entity()->create_component<GeometryInstance>();
    non_emissive_instance->set_geometry(non_emissive_mesh);
    set_curve_samples(*curve, curve_samples, later_times, AnimationInterpolationMode::linear);
    auto* curve_instance = instance->entity()->create_component<GeometryInstance>();
    curve_instance->set_geometry(curve);
    CHECK(scene->animation_range().end == 3);
    scene->set_time(1);
    scene->update();
    before = scene->_emissive_geometry_system()->generations();
    scene->set_time(2);
    scene->update();
    CHECK(scene->_emissive_geometry_system()->generations() == before);

    // Membership changes must also track a second emissive instance on the same entity.
    non_emissive_instance->set_materials({material, material});
    scene->update();
    before = scene->_emissive_geometry_system()->generations();
    scene->set_time(3);
    scene->update();
    after = scene->_emissive_geometry_system()->generations();
    CHECK(after.topology == before.topology);
    CHECK(after.geometry > before.geometry);
    CHECK(after.flux > before.flux);
    // Growing both pools must initialize existing geometry at the new time, including static vertex fields.
    const size_t mesh_buffer_size
        = render.triangle_vertex_buffer(RenderTriangleGeometryHandle{mesh_handles[0]}, 0)->size();
    const auto current_curve_handle
        = RenderLSSGeometryHandle{render.get_geometry_group(curve->_render_geometry_group()).geometries.front()};
    const size_t curve_buffer_size = render.lss_vertex_buffer(current_curve_handle, 0)->size();
    auto* added_mesh = scene->create_geometry<DynamicMeshGeometry>();
    set_mesh_samples(*added_mesh, samples, times, AnimationInterpolationMode::linear);
    auto* added_curve = scene->create_geometry<DynamicCurveGeometry>();
    set_curve_samples(*added_curve, curve_samples, times, AnimationInterpolationMode::linear);
    scene->set_time(0);
    scene->update();
    REQUIRE(render.triangle_vertex_buffer(RenderTriangleGeometryHandle{mesh_handles[0]}, 0)->size() > mesh_buffer_size);
    REQUIRE(render.lss_vertex_buffer(current_curve_handle, 0)->size() > curve_buffer_size);
    check_mesh(1, 1);
    const auto& grown_curve = render.get_lss_geometry(current_curve_handle);
    for (uint16_t member = 0; member < 2; ++member) {
        const auto data
            = render.lss_vertex_buffer(current_curve_handle, member)->get_elements<shared::PackedLSSVertex>();
        CHECK(data[grown_curve.vertex_offset].position.z == 0);
        CHECK(data[grown_curve.vertex_offset].radius == doctest::Approx(0.3f));
    }
    scene->update();
    check_mesh(1, 1);
    added_mesh->remove();
    added_curve->remove();
    scene->set_time(-1);
    scene->update();
    check_mesh(0, 0);
    set_mesh_samples(*mesh, samples, times, AnimationInterpolationMode::linear);
    scene->set_time(0);
    scene->update();
    mesh_handles = render.get_geometry_group(mesh->_render_geometry_group()).geometries;
    check_mesh(1, 1);
    // Repeated keys and a jump use the same two resident samples for both geometry types.
    const float mapped_times[] = {-1, 1, 1, 3};
    const uint32_t sample_indices[] = {0, 1, 0, 1};
    set_mesh_samples(*mesh, samples, mapped_times, AnimationInterpolationMode::linear, sample_indices);
    set_curve_samples(*curve, curve_samples, mapped_times, AnimationInterpolationMode::linear, sample_indices);
    CHECK(scene->animation_range().end == 3);
    scene->set_time(0);
    scene->update();
    mesh_handles = render.get_geometry_group(mesh->_render_geometry_group()).geometries;
    check_mesh(1, 1);
    scene->set_time(1);
    scene->update();
    check_mesh(0, 1);
    scene->set_time(2);
    scene->update();
    check_mesh(1, 0);
    auto mapped_curve_handle
        = RenderLSSGeometryHandle{render.get_geometry_group(curve->_render_geometry_group()).geometries.front()};
    const auto& mapped_curve = render.get_lss_geometry(mapped_curve_handle);
    auto mapped_vertices = render.lss_vertex_buffer(mapped_curve_handle, 0)->get_elements<shared::PackedLSSVertex>();
    CHECK(mapped_vertices[mapped_curve.vertex_offset].position.z == 1);
    CHECK(mapped_vertices[mapped_curve.vertex_offset].radius == doctest::Approx(0.4f));
    const uint32_t invalid_indices[] = {0, 2, 0, 1};
    CHECK_THROWS(
        set_curve_samples(*curve, curve_samples, mapped_times, AnimationInterpolationMode::linear, invalid_indices)
    );
    non_emissive_mesh->remove();
    mesh->remove();
    curve_instance->remove();
    CHECK(!scene->has_animation());
    CHECK(!scene->animation_range().is_valid());
    auto* curve_entity = scene->create_entity();
    curve_instance = curve_entity->create_component<GeometryInstance>();
    curve_instance->set_geometry(curve);
    CHECK(scene->has_animation());
    CHECK(scene->animation_range().end == 3);
    set_curve_samples(*curve, {curve_samples, 1}, {times, 1}, AnimationInterpolationMode::linear);
    CHECK(!scene->has_animation());
    CHECK(scene->animation_range().start == -1);
    CHECK(scene->animation_range().end == -1);
    set_curve_samples(*curve, curve_samples, mapped_times, AnimationInterpolationMode::linear, sample_indices);
    CHECK(scene->has_animation());
    CHECK(scene->animation_range().end == 3);
    curve_entity->remove();
    CHECK(!scene->has_animation());
    CHECK(!scene->animation_range().is_valid());
    curve->remove();
    scene->update();
    CHECK(!scene->has_animation());

    // A changed deformer input must evaluate at paused scene time and preserve final-output history.
    std::vector<shared::PackedTriangleVertex> packed_samples;
    AABB bounds;
    for (const auto& pose : points) {
        for (const auto& position : pose) {
            shared::TriangleVertex vertex = {};
            vertex.position = position;
            vertex.normal = float3(0, 0, 1);
            vertex.tangent = float3(1, 0, 0);
            packed_samples.push_back(shared::detail::pack_triangle_vertex(vertex));
            bounds.expand(position);
        }
    }
    std::vector<MeshGeometryData> topology(1);
    topology[0].vertices.assign(packed_samples.begin(), packed_samples.begin() + 3);
    topology[0].indices = {0, 1, 2};
    sgl::short_vector<GeometrySampleStorage<shared::PackedTriangleVertex>, 1> sources;
    sources.emplace_back().reset(ctx.device, std::move(packed_samples), 3);
    auto deformer = std::make_unique<ControlledCacheDeformer>(
        std::move(sources),
        times,
        AnimationInterpolationMode::linear,
        std::span<const uint32_t>{},
        bounds
    );
    auto* control = deformer.get();
    auto* controlled_mesh = scene->create_geometry<DynamicMeshGeometry>();
    controlled_mesh->set_mesh_data(std::move(topology), std::move(deformer));
    scene->create_entity()->create_component<GeometryInstance>()->set_geometry(controlled_mesh);
    CHECK(!scene->has_animation());
    CHECK(!scene->animation_range().is_valid());
    scene->set_time(0);
    scene->update();
    const auto controlled_handle = RenderTriangleGeometryHandle{
        render.get_geometry_group(controlled_mesh->_render_geometry_group()).geometries.front()
    };
    auto check_controlled = [&](float current_z, float previous_z)
    {
        CHECK(scene->time() == 0);
        const auto& output = render.get_triangle_geometry(controlled_handle);
        for (uint16_t member = 0; member < 2; ++member) {
            const auto data = render.triangle_vertex_buffer(controlled_handle, member)
                                  ->get_elements<shared::PackedTriangleVertex>();
            for (size_t i = 0; i < 3; ++i)
                CHECK(data[output.vertex_offset + i].position.z == (member ? previous_z : current_z));
        }
    };
    CHECK(control->update_count == 1);
    check_controlled(0, 0);
    control->input_time = 1;
    scene->update();
    check_controlled(2, 0);
    scene->update();
    check_controlled(2, 2);
    scene->update();
    check_controlled(2, 2);
    CHECK(control->update_count == 4);

    ref<DynamicMeshGeometry> removed_mesh(controlled_mesh);
    controlled_mesh->remove();
    scene->update();
    CHECK(control->update_count == 4);
    removed_mesh.reset();

    // Registered deformers can outlive their scene, but may not evaluate after it is destroyed.
    ref<DynamicMeshGeometry> retained_mesh(scene->create_geometry<DynamicMeshGeometry>());
    ref<DynamicCurveGeometry> retained_curve(scene->create_geometry<DynamicCurveGeometry>());
    set_mesh_samples(*retained_mesh, samples, times, AnimationInterpolationMode::linear);
    set_curve_samples(*retained_curve, curve_samples, times, AnimationInterpolationMode::linear);
    scene->update();
    scene.reset();
    retained_mesh.reset();
    retained_curve.reset();
}

TEST_SUITE_END();
