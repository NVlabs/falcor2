// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "emissive_geometry_system.h"

#include "falcor2/render/scene.h"
#include "falcor2/render/entity.h"
#include "falcor2/render/component/geometry_instance.h"

#include "falcor2/utils/algorithm/prefix_sum.h"
#include "falcor2/utils/io/mesh_writer.h"

#include <sgl/device/device.h>
#include <sgl/device/kernel.h>
#include <sgl/device/command.h>
#include <sgl/device/shader.h>
#include <sgl/device/shader_cursor.h>

#include <algorithm>
#include <cmath>
#include <limits>

namespace falcor {

namespace {
constexpr bool ENABLE_DEBUG = false;
} // namespace

EmissiveGeometrySystem::TransformStretch::TransformStretch(const float4x4& world_matrix)
{
    for (uint32_t i = 0; i < 4; ++i) {
        const float4 column = world_matrix.get_col(i);
        for (uint32_t k = 0; k < 4; ++k) {
            if (!std::isfinite(column[k]))
                return;
        }
    }
    // A^T A is invariant under world-space rigid motion, including reflections.
    // Double precision accommodates extreme scales without overflowing the products.
    for (uint32_t i = 0; i < 3; ++i) {
        for (uint32_t k = 0; k < 3; ++k) {
            const double value = world_matrix.get_col(i)[k];
            diagonal[i] += value * value;
            off_diagonal[i] += value * double(world_matrix.get_col((i + 1) % 3)[k]);
        }
        if (diagonal[i] <= 0.0)
            return;
    }

    const double xy = off_diagonal[0] / std::sqrt(diagonal[0] * diagonal[1]);
    const double yz = off_diagonal[1] / std::sqrt(diagonal[1] * diagonal[2]);
    const double xz = off_diagonal[2] / std::sqrt(diagonal[2] * diagonal[0]);
    const double determinant = 1.0 + 2.0 * xy * xz * yz - xy * xy - xz * xz - yz * yz;
    const double minors = 3.0 - xy * xy - xz * xz - yz * yz;
    if (determinant <= 32.0 * std::numeric_limits<float>::epsilon() || minors <= 0.0)
        return;

    // For the normalized metric C, det(C) / sum(principal minors) = 1 / trace(inverse(C))
    // bounds its smallest eigenvalue from below. Tighten the entrywise tolerance accordingly:
    // otherwise tiny metric differences can hide large area changes in a nearly parallel basis.
    tolerance_scale = determinant / minors;
}

bool EmissiveGeometrySystem::TransformStretch::preserves_area(const TransformStretch& reference) const
{
    if (tolerance_scale == 0.0 || reference.tolerance_scale == 0.0)
        return false;
    // Quaternion interpolation and composing float rotations accumulate roundoff in A^T A.
    // Allow a few parts per million for an orthogonal basis (tolerance_scale = 1/3),
    // while retaining the conditioning-dependent bound for nearly parallel basis vectors.
    constexpr double TOLERANCE = 64.0 * std::numeric_limits<float>::epsilon();
    const double tolerance = TOLERANCE * reference.tolerance_scale;
    const double tolerance_squared = tolerance * tolerance;
    // Squaring avoids square roots per comparison. Products of finite float matrix entries,
    // including these squared metric differences, remain within double precision's range.
    for (uint32_t i = 0; i < 3; ++i) {
        const double diagonal_difference = diagonal[i] - reference.diagonal[i];
        if (diagonal_difference * diagonal_difference
            > tolerance_squared * reference.diagonal[i] * reference.diagonal[i])
            return false;
        const double off_diagonal_difference = off_diagonal[i] - reference.off_diagonal[i];
        if (off_diagonal_difference * off_diagonal_difference
            > tolerance_squared * reference.diagonal[i] * reference.diagonal[(i + 1) % 3])
            return false;
    }
    return true;
}

EmissiveGeometrySystem::EmissiveGeometrySystem(Scene* scene)
    : SceneSystem{scene}
    , m_device(scene->device())
    , m_geometries(scene->geometry_collection())
    , m_materials(scene->material_collection())
    , m_entities(scene->entity_collection())
    , m_components(scene->component_collection())
{
}

void EmissiveGeometrySystem::clear_resources()
{
    m_transform_stretch.clear();
    m_triangle_count = 0;
    m_active_triangle_count = 0;
    m_triangles.clear();
    m_triangle_flux.clear();
    m_active_triangle_ids.clear();
    m_active_triangle_flux.clear();
    m_geometry_instance_to_triangle_id_buffer = nullptr;
    m_triangles_buffer = nullptr;
    m_triangle_emission_buffer = nullptr;
    m_triangle_flux_buffer = nullptr;
    m_triangle_id_to_active_triangle_id_buffer = nullptr;
    m_active_triangle_id_to_triangle_id_buffer = nullptr;
    m_active_triangle_flux_buffer = nullptr;
}

void EmissiveGeometrySystem::create_kernels()
{
    const auto& requirements = m_scene->requirements();
    if (m_gather_triangle_count_kernel && requirements->matches_shader_inputs(m_scene_requirements.get())) {
        m_scene_requirements = requirements;
        return;
    }

    std::vector<ref<sgl::SlangModule>> modules;
    modules.push_back(m_device->load_module("falcor2/render/kernels/emissive_geometry_kernels.slang"));
    modules.insert(modules.end(), requirements->modules.begin(), requirements->modules.end());
    ref<sgl::SlangModule> module
        = m_device->compose_modules("emissive_geometry_kernels", modules, requirements->type_conformances);

    auto create_kernel = [&](std::string_view entry_point_name)
    {
        return m_device->create_compute_kernel({
            .program = m_device->link_program({module}, {module->entry_point(entry_point_name)}),
        });
    };

    m_gather_triangle_count_kernel = create_kernel("gather_triangle_count");
    m_setup_geometry_instance_to_triangle_id_kernel = create_kernel("setup_geometry_instance_to_triangle_id");
    m_setup_triangle_kernel = create_kernel("setup_triangle");
    m_setup_triangle_sampling_kernel = create_kernel("setup_triangle_sampling");
    m_compute_triangle_max_emission_kernel = create_kernel("compute_triangle_max_emission");
    m_compute_triangle_max_emission_factor_kernel = create_kernel("compute_triangle_max_emission_factor");
    m_accumulate_triangle_emission_kernel = create_kernel("accumulate_triangle_emission");
    m_finalize_triangle_emission_kernel = create_kernel("finalize_triangle_emission");
    m_gather_active_triangles_kernel = create_kernel("gather_active_triangles");
    m_setup_active_triangle_mapping_kernel = create_kernel("setup_active_triangle_mapping");
    m_update_triangle_kernel = create_kernel("update_triangle");

    m_scene_requirements = requirements;
}

SceneUpdateFlags EmissiveGeometrySystem::update(SceneUpdateContext& ctx)
{
    create_kernels();

    const bool materials_changed = m_materials.has_dirty(
        Material::DirtyFlags::emission | Material::DirtyFlags::added | Material::DirtyFlags::removed
    );
    const bool geometries_changed = m_geometries.is_dirty();

    bool instances_changed = false;
    if (m_components.has_dirty(~Component::DirtyFlags::entity_transform)) {
        for (size_t i = 0; i < m_components.size(); ++i) {
            if (is_set(m_components.dirty_flags()[i], ~Component::DirtyFlags::entity_transform)
                && m_components[i]->as<GeometryInstance>()) {
                instances_changed = true;
                break;
            }
        }
    }

    const bool update_emission = materials_changed || geometries_changed || instances_changed;
    TransformChanges transform_changes;
    if (!update_emission)
        transform_changes = classify_transform_changes();

    if (!update_emission && !transform_changes.geometry_changed)
        return SceneUpdateFlags::none;

    const bool update_flux = update_emission || transform_changes.flux_changed;

    RenderScene* render_scene = m_scene->_render_scene();

    uint32_t geometry_instance_count = sgl::narrow_cast<uint32_t>(render_scene->geometry_instance_desc_count());
    if (geometry_instance_count == 0) {
        m_transform_stretch.clear();
        if (m_triangle_count == 0)
            return SceneUpdateFlags::none;
        clear_resources();
        ++m_generations.topology;
        ++m_generations.geometry;
        ++m_generations.flux;
        return SceneUpdateFlags::emissive_geometry;
    }

    if (update_emission)
        transform_changes.geometry_changed = rebuild_emission(ctx, geometry_instance_count);

    update_triangles(ctx, update_flux);

    if (ENABLE_DEBUG) {
        if (m_triangle_count > 0) {
            MeshWriter writer;
            writer.begin_submesh("emissive_triangles");
            ctx.submit();
            auto triangles = m_triangles_buffer->get_elements<shared::PackedEmissiveTriangle>();
            auto triangle_flux = m_triangle_flux_buffer->get_elements<float>();
            for (size_t i = 0; i < m_triangle_count; ++i) {
                fmt::println("triangle {}: flux={}", i, triangle_flux[i]);
                shared::EmissiveTriangle triangle = shared::detail::unpack_emissive_triangle(triangles[i]);
                writer.triangle(triangle.pos_ws[0], triangle.pos_ws[1], triangle.pos_ws[2]);
            }
            writer.end_submesh();
            writer.write("emissive_triangles.gltf");
        }
    }

    const bool topology_changed = update_emission;
    const bool geometry_changed = geometries_changed || instances_changed || transform_changes.geometry_changed;
    const bool flux_changed = update_flux;
    if (topology_changed)
        ++m_generations.topology;
    if (geometry_changed)
        ++m_generations.geometry;
    if (flux_changed)
        ++m_generations.flux;

    return SceneUpdateFlags::emissive_geometry;
}

EmissiveGeometrySystem::TransformChanges EmissiveGeometrySystem::classify_transform_changes()
{
    TransformChanges changes;
    if (!m_entities.has_dirty(Entity::DirtyFlags::transform))
        return changes;

    for (auto& [entity, reference] : m_transform_stretch) {
        if (!is_set(m_entities.dirty_flags()[entity->collection_index()], Entity::DirtyFlags::transform))
            continue;
        changes.geometry_changed = true;
        const TransformStretch stretch(entity->world_from_object_matrix());
        if (!stretch.preserves_area(reference)) {
            changes.flux_changed = true;
            // Keep the reference anchored so small scale edits cannot accumulate unseen.
            reference = stretch;
        }
    }

    return changes;
}

bool EmissiveGeometrySystem::rebuild_transform_stretch(std::span<const uint32_t> triangle_counts)
{
    bool transforms_changed = false;
    m_transform_stretch.clear();
    const auto contributes_triangles = [&triangle_counts](Component* component)
    {
        const auto* instance = component->as<GeometryInstance>();
        if (!instance || !instance->is_valid())
            return false;
        const uint32_t first = uint32_t(instance->geometry_instance_id());
        for (uint32_t i = 0; i < instance->geometry_instance_count(); ++i) {
            if (triangle_counts[first + i] > 0)
                return true;
        }
        return false;
    };
    for (const Entity* entity : m_entities.objects()) {
        if (!entity->is_valid())
            continue;
        const auto& components = entity->components();
        if (!std::any_of(components.begin(), components.end(), contributes_triangles))
            continue;
        transforms_changed
            |= is_set(m_entities.dirty_flags()[entity->collection_index()], Entity::DirtyFlags::transform);
        m_transform_stretch.push_back({entity, TransformStretch(entity->world_from_object_matrix())});
    }
    return transforms_changed;
}

bool EmissiveGeometrySystem::rebuild_emission(SceneUpdateContext& ctx, uint32_t geometry_instance_count)
{
    // ------------------------------------------------------------------------
    // Gather potentially emissive geometry instances and count triangles.
    // ------------------------------------------------------------------------

    // is_emissive[geometry_instance_id] = 0 (not emissive) / 1 (potentially emissive)
    // triangle_count[geometry_instance_id] = 0 (not emissive) / triangle count (potentially emissive)

    ref<sgl::Buffer> is_emissive_buffer = m_device->create_buffer({
        .size = geometry_instance_count * sizeof(uint32_t),
        .usage = sgl::BufferUsage::unordered_access | sgl::BufferUsage::shader_resource,
        .label = "is_emissive",
    });
    ref<sgl::Buffer> triangle_count_buffer = m_device->create_buffer({
        .size = geometry_instance_count * sizeof(uint32_t),
        .usage = sgl::BufferUsage::unordered_access | sgl::BufferUsage::shader_resource,
        .label = "triangle_count",
    });
    m_gather_triangle_count_kernel->dispatch(
        uint3(geometry_instance_count, 1, 1),
        [&](sgl::ShaderCursor cursor)
        {
            m_scene->bind(cursor, SceneBindFlags::all & ~SceneBindFlags::emissive_geometry);
            cursor = cursor.find_entry_point(0);
            cursor["is_emissive"] = is_emissive_buffer;
            cursor["triangle_count"] = triangle_count_buffer;
            cursor["geometry_instance_count"] = geometry_instance_count;
        },
        ctx.command_encoder()
    );
    if (ENABLE_DEBUG) {
        ctx.submit();
        auto is_emissive = is_emissive_buffer->get_elements<uint32_t>();
        auto triangle_count = triangle_count_buffer->get_elements<uint32_t>();
        SGL_PRINT(is_emissive);
        SGL_PRINT(triangle_count);
    }

    // ------------------------------------------------------------------------
    // Compute triangle offsets and total triangle count using prefix sum.
    // ------------------------------------------------------------------------

    ref<sgl::Buffer> triangle_offset_buffer = m_device->create_buffer({
        .size = geometry_instance_count * sizeof(uint32_t),
        .usage = sgl::BufferUsage::unordered_access | sgl::BufferUsage::shader_resource,
        .label = "triangle_offset",
    });
    ref<sgl::Buffer> total_triangle_count_buffer = m_device->create_buffer({
        .size = sizeof(uint32_t),
        .usage = sgl::BufferUsage::unordered_access | sgl::BufferUsage::shader_resource,
        .label = "total_triangle_count",
    });
    // Copy triangle_count_buffer to triangle_offset_buffer to use it as input for prefix sum.
    ctx.command_encoder()
        ->copy_buffer(triangle_offset_buffer, 0, triangle_count_buffer, 0, geometry_instance_count * sizeof(uint32_t));
    prefix_sum(
        ctx.command_encoder(),
        sgl::DataType::uint32,
        triangle_offset_buffer,
        geometry_instance_count,
        total_triangle_count_buffer
    );
    // Read back total triangle count (stalls the host).
    ctx.submit();
    m_triangle_count = total_triangle_count_buffer->get_element<uint32_t>(0);

    if (ENABLE_DEBUG) {
        ctx.submit();
        auto triangle_offset = triangle_offset_buffer->get_elements<uint32_t>();
        SGL_PRINT(m_triangle_count);
        SGL_PRINT(triangle_offset);
    }

    if (m_triangle_count == 0) {
        clear_resources();
        return false;
    }

    // Rebuild membership from the same GPU eligibility decision that creates emissive triangles.
    // Counts include zero-area and currently inactive triangles, which may gain flux after scaling.
    const bool transforms_changed
        = rebuild_transform_stretch(triangle_count_buffer->get_elements<uint32_t>(0, geometry_instance_count));

    // ------------------------------------------------------------------------
    // Prepare geometry_instance_to_triangle_id lookup table
    // ------------------------------------------------------------------------

    if (!m_geometry_instance_to_triangle_id_buffer
        || m_geometry_instance_to_triangle_id_buffer->size()
            < geometry_instance_count * sizeof(shared::EmissiveTriangleID)) {
        m_geometry_instance_to_triangle_id_buffer = m_device->create_buffer({
            .size = geometry_instance_count * sizeof(shared::EmissiveTriangleID),
            .usage = sgl::BufferUsage::unordered_access | sgl::BufferUsage::shader_resource,
            .label = "EmissiveGeometry::geometry_instance_to_triangle_id",
        });
    }
    m_setup_geometry_instance_to_triangle_id_kernel->dispatch(
        uint3(geometry_instance_count, 1, 1),
        [&](sgl::ShaderCursor cursor)
        {
            cursor = cursor.find_entry_point(0);
            cursor["is_emissive"] = is_emissive_buffer;
            cursor["triangle_offset"] = triangle_offset_buffer;
            cursor["geometry_instance_to_triangle_id"] = m_geometry_instance_to_triangle_id_buffer;
            cursor["geometry_instance_count"] = geometry_instance_count;
        },
        ctx.command_encoder()
    );

    if (ENABLE_DEBUG) {
        ctx.submit();
        auto geometry_instance_to_triangle_id = m_geometry_instance_to_triangle_id_buffer->get_elements<uint32_t>();
        SGL_PRINT(geometry_instance_to_triangle_id);
    }

    // ------------------------------------------------------------------------
    // Prepare triangles buffer
    // ------------------------------------------------------------------------

    if (!m_triangles_buffer || m_triangles_buffer->size() < m_triangle_count * sizeof(shared::PackedEmissiveTriangle)) {
        m_triangles_buffer = m_device->create_buffer({
            .size = m_triangle_count * sizeof(shared::PackedEmissiveTriangle),
            .usage = sgl::BufferUsage::unordered_access | sgl::BufferUsage::shader_resource,
            .label = "EmissiveGeometry::triangles",
        });
    }
    m_setup_triangle_kernel->dispatch(
        uint3(m_triangle_count, 1, 1),
        [&](sgl::ShaderCursor cursor)
        {
            m_scene->bind(cursor, SceneBindFlags::all & ~SceneBindFlags::emissive_geometry);
            cursor = cursor.find_entry_point(0);
            cursor["triangles"] = m_triangles_buffer;
            cursor["triangle_offset"] = triangle_offset_buffer;
            cursor["triangle_count"] = m_triangle_count;
            cursor["geometry_instance_count"] = geometry_instance_count;
        },
        ctx.command_encoder()
    );

    if (ENABLE_DEBUG) {
        ctx.submit();
        auto triangles = m_triangles_buffer->get_elements<shared::PackedEmissiveTriangle>();
        for (size_t i = 0; i < triangles.size(); ++i) {
            shared::EmissiveTriangle triangle = shared::detail::unpack_emissive_triangle(triangles[i]);
            fmt::println(
                "triangle[{}] = EmissiveTriangle(geometry_instance_id={}, material_id={}, uv[0]={}, "
                "uv[1]={}, "
                "uv[2]={})",
                i,
                uint32_t(triangle.geometry_instance_id),
                uint32_t(triangle.material_id),
                triangle.uv[0],
                triangle.uv[1],
                triangle.uv[2]
            );
        }
    }

    sample_triangle_emission(ctx);
    rebuild_active_triangles(ctx);
    return transforms_changed;
}

void EmissiveGeometrySystem::sample_triangle_emission(SceneUpdateContext& ctx)
{
    // ------------------------------------------------------------------------
    // Setup per-triangle emission sampling parameters
    // ------------------------------------------------------------------------

    ref<sgl::Buffer> sample_grid_buffer = m_device->create_buffer({
        .size = m_triangle_count * sizeof(uint2),
        .usage = sgl::BufferUsage::unordered_access | sgl::BufferUsage::shader_resource,
        .label = "sample_grid",
    });
    ref<sgl::Buffer> sample_count_buffer = m_device->create_buffer({
        .size = m_triangle_count * sizeof(uint64_t),
        .usage = sgl::BufferUsage::unordered_access | sgl::BufferUsage::shader_resource,
        .label = "sample_count",
    });
    m_setup_triangle_sampling_kernel->dispatch(
        uint3(m_triangle_count, 1, 1),
        [&](sgl::ShaderCursor cursor)
        {
            m_scene->bind(cursor, SceneBindFlags::all & ~SceneBindFlags::emissive_geometry);
            cursor = cursor.find_entry_point(0);
            cursor["triangles"] = m_triangles_buffer;
            cursor["sample_count"] = sample_count_buffer;
            cursor["sample_grid"] = sample_grid_buffer;
            cursor["triangle_count"] = m_triangle_count;
        },
        ctx.command_encoder()
    );

    if (ENABLE_DEBUG) {
        ctx.submit();
        auto sample_grid = sample_grid_buffer->get_elements<sgl::uint2>();
        auto sample_count = sample_count_buffer->get_elements<uint64_t>();
        for (size_t i = 0; i < m_triangle_count; ++i) {
            fmt::println("triangle {}: sample_grid={}, sample_count={}", i, sample_grid[i], sample_count[i]);
        }
    }

    // ------------------------------------------------------------------------
    // Compute emission sample offsets and total sample count
    // ------------------------------------------------------------------------

    ref<sgl::Buffer> sample_offset_buffer = m_device->create_buffer({
        .size = m_triangle_count * sizeof(uint64_t),
        .usage = sgl::BufferUsage::unordered_access | sgl::BufferUsage::shader_resource,
        .label = "sample_offset",
    });
    ref<sgl::Buffer> total_sample_count_buffer = m_device->create_buffer({
        .size = sizeof(uint64_t),
        .usage = sgl::BufferUsage::unordered_access | sgl::BufferUsage::shader_resource,
        .label = "total_sample_count",
    });
    ctx.command_encoder()
        ->copy_buffer(sample_offset_buffer, 0, sample_count_buffer, 0, m_triangle_count * sizeof(uint64_t));
    prefix_sum(
        ctx.command_encoder(),
        sgl::DataType::uint64,
        sample_offset_buffer,
        m_triangle_count,
        total_sample_count_buffer
    );

    ctx.submit();
    uint64_t total_sample_count = total_sample_count_buffer->get_element<uint64_t>(0);
    if (ENABLE_DEBUG) {
        SGL_PRINT(total_sample_count);
    }

    // ------------------------------------------------------------------------
    // Compute max emission value for each triangle
    // ------------------------------------------------------------------------

    ref<sgl::Buffer> max_emission_buffer = m_device->create_buffer({
        .size = m_triangle_count * sizeof(float),
        .usage = sgl::BufferUsage::unordered_access | sgl::BufferUsage::shader_resource,
        .label = "max_emission",
    });
    ctx.command_encoder()->clear_buffer(max_emission_buffer);

    static constexpr uint64_t SAMPLE_BATCH_SIZE = 1024 * 1024;
    static constexpr uint32_t MAX_PERSISTENT_THREAD_COUNT = 64 * 1024;
    for (uint64_t first_sample = 0; first_sample < total_sample_count; first_sample += SAMPLE_BATCH_SIZE) {
        uint32_t batch_sample_count
            = sgl::narrow_cast<uint32_t>(std::min(SAMPLE_BATCH_SIZE, total_sample_count - first_sample));
        uint32_t thread_count = std::min(MAX_PERSISTENT_THREAD_COUNT, batch_sample_count);
        m_compute_triangle_max_emission_kernel->dispatch(
            uint3(thread_count, 1, 1),
            [&](sgl::ShaderCursor cursor)
            {
                m_scene->bind(cursor, SceneBindFlags::all);
                cursor = cursor.find_entry_point(0);
                cursor["triangles"] = m_triangles_buffer;
                cursor["sample_offset"] = sample_offset_buffer;
                cursor["sample_count"] = sample_count_buffer;
                cursor["sample_grid"] = sample_grid_buffer;
                cursor["triangle_count"] = m_triangle_count;
                cursor["thread_count"] = thread_count;
                cursor["first_sample"] = first_sample;
                cursor["batch_sample_count"] = batch_sample_count;
                cursor["max_emission"] = max_emission_buffer;
            },
            ctx.command_encoder()
        );
        // Submitting each batch separately is required. Pooling all batches into one submission can cause
        // device loss when scenes contain many triangles.
        ctx.submit();
    }

    if (ENABLE_DEBUG) {
        ctx.submit();
        auto max_emission = max_emission_buffer->get_elements<float>();
        for (size_t i = 0; i < m_triangle_count; ++i) {
            fmt::println("triangle {}: max_emission={}", i, max_emission[i]);
        }
    }

    // ------------------------------------------------------------------------
    // Compute emission scaling factor to allow accumulation in fixed-point format
    // ------------------------------------------------------------------------

    ref<sgl::Buffer> emission_factor_buffer = m_device->create_buffer({
        .size = m_triangle_count * sizeof(float),
        .usage = sgl::BufferUsage::unordered_access | sgl::BufferUsage::shader_resource,
        .label = "emission_factor",
    });

    m_compute_triangle_max_emission_factor_kernel->dispatch(
        uint3(m_triangle_count, 1, 1),
        [&](sgl::ShaderCursor cursor)
        {
            cursor = cursor.find_entry_point(0);
            cursor["sample_count"] = sample_count_buffer;
            cursor["max_emission"] = max_emission_buffer;
            cursor["emission_factor"] = emission_factor_buffer;
            cursor["triangle_count"] = m_triangle_count;
        },
        ctx.command_encoder()
    );

    if (ENABLE_DEBUG) {
        ctx.submit();
        auto emission_factor = emission_factor_buffer->get_elements<float>();
        for (size_t i = 0; i < m_triangle_count; ++i) {
            fmt::println("triangle {}: emission_factor={}", i, emission_factor[i]);
        }
    }

    // ------------------------------------------------------------------------
    // Sample triangle emission and accumulate
    // ------------------------------------------------------------------------

    ref<sgl::Buffer> accumulated_emission_buffer = m_device->create_buffer({
        .size = m_triangle_count * sizeof(uint64_t) * 3,
        .usage = sgl::BufferUsage::unordered_access | sgl::BufferUsage::shader_resource,
        .label = "accumulated_emission",
    });
    ctx.command_encoder()->clear_buffer(accumulated_emission_buffer);

    for (uint64_t first_sample = 0; first_sample < total_sample_count; first_sample += SAMPLE_BATCH_SIZE) {
        uint32_t batch_sample_count
            = sgl::narrow_cast<uint32_t>(std::min(SAMPLE_BATCH_SIZE, total_sample_count - first_sample));
        uint32_t thread_count = std::min(MAX_PERSISTENT_THREAD_COUNT, batch_sample_count);
        m_accumulate_triangle_emission_kernel->dispatch(
            uint3(thread_count, 1, 1),
            [&](sgl::ShaderCursor cursor)
            {
                m_scene->bind(cursor, SceneBindFlags::all);
                cursor = cursor.find_entry_point(0);
                cursor["triangles"] = m_triangles_buffer;
                cursor["sample_offset"] = sample_offset_buffer;
                cursor["sample_count"] = sample_count_buffer;
                cursor["sample_grid"] = sample_grid_buffer;
                cursor["emission_factor"] = emission_factor_buffer;
                cursor["triangle_count"] = m_triangle_count;
                cursor["thread_count"] = thread_count;
                cursor["first_sample"] = first_sample;
                cursor["batch_sample_count"] = batch_sample_count;
                cursor["accumulated_emission"] = accumulated_emission_buffer;
            },
            ctx.command_encoder()
        );
        // Submitting each batch separately is required. Pooling all batches into one submission can cause
        // device loss when scenes contain many triangles.
        ctx.submit();
    }

    if (ENABLE_DEBUG) {
        ctx.submit();
        auto accumulated_emission = accumulated_emission_buffer->get_elements<uint64_t>();
        for (size_t i = 0; i < m_triangle_count; ++i) {
            uint64_t r = accumulated_emission[i * 3];
            uint64_t g = accumulated_emission[i * 3 + 1];
            uint64_t b = accumulated_emission[i * 3 + 2];
            fmt::println("triangle {}: accumulated_emission=({}, {}, {})", i, r, g, b);
        }
    }

    // ------------------------------------------------------------------------
    // Finalize average triangle emission
    // ------------------------------------------------------------------------

    if (!m_triangle_emission_buffer || m_triangle_emission_buffer->size() < m_triangle_count * sizeof(float4)) {
        m_triangle_emission_buffer = m_device->create_buffer({
            .size = m_triangle_count * sizeof(float4),
            .usage = sgl::BufferUsage::unordered_access | sgl::BufferUsage::shader_resource,
            .label = "EmissiveGeometry::triangle_emission",
        });
    }

    m_finalize_triangle_emission_kernel->dispatch(
        uint3(m_triangle_count, 1, 1),
        [&](sgl::ShaderCursor cursor)
        {
            cursor = cursor.find_entry_point(0);
            cursor["sample_grid"] = sample_grid_buffer;
            cursor["accumulated_emission"] = accumulated_emission_buffer;
            cursor["emission_factor"] = emission_factor_buffer;
            cursor["triangle_emission"] = m_triangle_emission_buffer;
            cursor["triangle_count"] = m_triangle_count;
        },
        ctx.command_encoder()
    );

    if (ENABLE_DEBUG) {
        ctx.submit();
        auto triangle_emission = m_triangle_emission_buffer->get_elements<float4>();
        for (size_t i = 0; i < m_triangle_count; ++i) {
            fmt::println("triangle {}: emission={}", i, triangle_emission[i]);
        }
    }
}

void EmissiveGeometrySystem::rebuild_active_triangles(SceneUpdateContext& ctx)
{
    // ------------------------------------------------------------------------
    // Update active triangles
    // ------------------------------------------------------------------------

    ref<sgl::Buffer> is_active_buffer = m_device->create_buffer({
        .size = m_triangle_count * sizeof(uint32_t),
        .usage = sgl::BufferUsage::unordered_access | sgl::BufferUsage::shader_resource,
        .label = "is_active",
    });
    ref<sgl::Buffer> active_offset_buffer = m_device->create_buffer({
        .size = m_triangle_count * sizeof(uint32_t),
        .usage = sgl::BufferUsage::unordered_access | sgl::BufferUsage::shader_resource,
        .label = "active_offset",
    });
    ref<sgl::Buffer> active_count_buffer = m_device->create_buffer({
        .size = sizeof(uint32_t),
        .usage = sgl::BufferUsage::unordered_access | sgl::BufferUsage::shader_resource,
        .label = "active_count_buffer",
    });

    m_gather_active_triangles_kernel->dispatch(
        uint3(m_triangle_count, 1, 1),
        [&](sgl::ShaderCursor cursor)
        {
            cursor = cursor.find_entry_point(0);
            cursor["triangle_emission"] = m_triangle_emission_buffer;
            cursor["is_active"] = is_active_buffer;
            cursor["triangle_count"] = m_triangle_count;
        },
        ctx.command_encoder()
    );

    ctx.command_encoder()
        ->copy_buffer(active_offset_buffer, 0, is_active_buffer, 0, m_triangle_count * sizeof(uint32_t));

    // Compute active offsets.
    prefix_sum(
        ctx.command_encoder(),
        sgl::DataType::uint32,
        active_offset_buffer,
        m_triangle_count,
        active_count_buffer
    );

    // Read back total active triangle count (stalls the host).
    ctx.submit();
    m_active_triangle_count = active_count_buffer->get_element<uint32_t>(0);

    if (ENABLE_DEBUG) {
        SGL_PRINT(m_active_triangle_count);
    }

    if (!m_triangle_id_to_active_triangle_id_buffer
        || m_triangle_id_to_active_triangle_id_buffer->size() < m_triangle_count * sizeof(uint32_t)) {
        m_triangle_id_to_active_triangle_id_buffer = m_device->create_buffer({
            .size = m_triangle_count * sizeof(uint32_t),
            .usage = sgl::BufferUsage::unordered_access | sgl::BufferUsage::shader_resource,
            .label = "EmissiveGeometry::triangle_id_to_active_triangle_id",
        });
    }

    // Active triangle to triangle mapping is only needed if we have active triangles.
    if (m_active_triangle_count > 0) {
        if (!m_active_triangle_id_to_triangle_id_buffer
            || m_active_triangle_id_to_triangle_id_buffer->size() < m_active_triangle_count * sizeof(uint32_t)) {
            m_active_triangle_id_to_triangle_id_buffer = m_device->create_buffer({
                .size = m_active_triangle_count * sizeof(uint32_t),
                .usage = sgl::BufferUsage::unordered_access | sgl::BufferUsage::shader_resource,
                .label = "EmissiveGeometry::active_triangle_id_to_triangle_id",
            });
        }
    } else {
        m_active_triangle_id_to_triangle_id_buffer = nullptr;
    }

    m_setup_active_triangle_mapping_kernel->dispatch(
        uint3(m_triangle_count, 1, 1),
        [&](sgl::ShaderCursor cursor)
        {
            cursor = cursor.find_entry_point(0);
            cursor["is_active"] = is_active_buffer;
            cursor["active_offset"] = active_offset_buffer;
            cursor["triangle_id_to_active_triangle_id"] = m_triangle_id_to_active_triangle_id_buffer;
            cursor["active_triangle_id_to_triangle_id"] = m_active_triangle_id_to_triangle_id_buffer;
            cursor["triangle_count"] = m_triangle_count;
        },
        ctx.command_encoder()
    );

    if (ENABLE_DEBUG) {
        ctx.submit();
        auto is_active = is_active_buffer->get_elements<uint32_t>();
        auto active_offset = active_offset_buffer->get_elements<uint32_t>();
        auto triangle_id_to_active_triangle_id = m_triangle_id_to_active_triangle_id_buffer->get_elements<uint32_t>();
        SGL_PRINT(is_active);
        SGL_PRINT(active_offset);
        SGL_PRINT(triangle_id_to_active_triangle_id);
        if (m_active_triangle_id_to_triangle_id_buffer) {
            auto active_triangle_id_to_triangle_id
                = m_active_triangle_id_to_triangle_id_buffer->get_elements<uint32_t>();
            SGL_PRINT(active_triangle_id_to_triangle_id);
        }
    }
}

void EmissiveGeometrySystem::update_triangles(SceneUpdateContext& ctx, bool update_flux)
{
    if (m_triangle_count == 0)
        return;

    // ------------------------------------------------------------------------
    // Update triangles
    // ------------------------------------------------------------------------

    if (!m_triangle_flux_buffer || m_triangle_flux_buffer->size() < m_triangle_count * sizeof(float)) {
        m_triangle_flux_buffer = m_device->create_buffer({
            .size = m_triangle_count * sizeof(float),
            .usage = sgl::BufferUsage::unordered_access | sgl::BufferUsage::shader_resource,
            .label = "EmissiveGeometry::triangle_flux",
        });
    }

    // Active triangle flux buffer is only needed if we have active triangles.
    if (m_active_triangle_count > 0) {
        if (!m_active_triangle_flux_buffer
            || m_active_triangle_flux_buffer->size() < m_active_triangle_count * sizeof(float)) {
            m_active_triangle_flux_buffer = m_device->create_buffer({
                .size = m_active_triangle_count * sizeof(float),
                .usage = sgl::BufferUsage::unordered_access | sgl::BufferUsage::shader_resource,
                .label = "EmissiveGeometry::active_triangle_flux",
            });
        }
    } else {
        m_active_triangle_flux_buffer = nullptr;
    }

    m_update_triangle_kernel->dispatch(
        uint3(m_triangle_count, 1, 1),
        [&](sgl::ShaderCursor cursor)
        {
            m_scene->bind(cursor, SceneBindFlags::all & ~SceneBindFlags::emissive_geometry);
            cursor = cursor.find_entry_point(0);
            cursor["triangles"] = m_triangles_buffer;
            cursor["triangle_emission"] = m_triangle_emission_buffer;
            cursor["triangle_flux"] = m_triangle_flux_buffer;
            cursor["active_triangle_flux"] = m_active_triangle_flux_buffer;
            cursor["triangle_id_to_active_triangle_id"] = m_triangle_id_to_active_triangle_id_buffer;
            cursor["geometry_instance_to_triangle_id"] = m_geometry_instance_to_triangle_id_buffer;
            cursor["triangle_count"] = m_triangle_count;
            cursor["update_flux"] = update_flux;
        },
        ctx.command_encoder()
    );

    ctx.submit();
    if (update_flux) {
        if (m_active_triangle_flux_buffer) {
            m_active_triangle_flux = m_active_triangle_flux_buffer->get_elements<float>(0, m_active_triangle_count);
        } else {
            m_active_triangle_flux.clear();
        }
    }
    if (ENABLE_DEBUG) {
        auto triangle_id_to_active_triangle_id = m_triangle_id_to_active_triangle_id_buffer->get_elements<uint32_t>();
        SGL_PRINT(m_active_triangle_flux);
        SGL_PRINT(triangle_id_to_active_triangle_id);
        if (m_active_triangle_id_to_triangle_id_buffer) {
            auto active_triangle_id_to_triangle_id
                = m_active_triangle_id_to_triangle_id_buffer->get_elements<uint32_t>();
            SGL_PRINT(active_triangle_id_to_triangle_id);
        }
    }
}

void EmissiveGeometrySystem::bind_to_scene(const sgl::ShaderCursor& cursor) const
{
    sgl::ShaderCursor emissive_geometry = cursor["emissive_geometry"];
    emissive_geometry["triangle_count"] = m_triangle_count;
    emissive_geometry["active_triangle_count"] = m_active_triangle_count;
    emissive_geometry["triangles"] = m_triangles_buffer;
    emissive_geometry["triangle_emission"] = m_triangle_emission_buffer;
    emissive_geometry["triangle_flux"] = m_triangle_flux_buffer;
    emissive_geometry["geometry_instance_to_triangle_id"] = m_geometry_instance_to_triangle_id_buffer;
    emissive_geometry["triangle_id_to_active_triangle_id"] = m_triangle_id_to_active_triangle_id_buffer;
    emissive_geometry["active_triangle_id_to_triangle_id"] = m_active_triangle_id_to_triangle_id_buffer;
}

std::span<const shared::EmissiveTriangle> EmissiveGeometrySystem::triangles() const
{
    if (m_triangles_generations.topology != m_generations.topology
        || m_triangles_generations.geometry != m_generations.geometry) {
        m_triangles.clear();
        if (m_triangles_buffer && m_triangle_count > 0) {
            auto packed_triangles
                = m_triangles_buffer->get_elements<shared::PackedEmissiveTriangle>(0, m_triangle_count);
            m_triangles.resize(m_triangle_count);
            std::transform(
                packed_triangles.begin(),
                packed_triangles.end(),
                m_triangles.begin(),
                shared::detail::unpack_emissive_triangle
            );
        }
        m_triangles_generations = m_generations;
    }
    return m_triangles;
}

std::span<const float> EmissiveGeometrySystem::triangle_flux() const
{
    if (m_triangle_flux_generations.topology != m_generations.topology
        || m_triangle_flux_generations.flux != m_generations.flux) {
        m_triangle_flux.clear();
        if (m_triangle_flux_buffer && m_triangle_count > 0)
            m_triangle_flux = m_triangle_flux_buffer->get_elements<float>(0, m_triangle_count);
        m_triangle_flux_generations = m_generations;
    }
    return m_triangle_flux;
}

std::span<const shared::EmissiveTriangleID> EmissiveGeometrySystem::active_triangle_ids() const
{
    if (m_active_triangle_ids_generations.topology != m_generations.topology) {
        m_active_triangle_ids.clear();
        if (m_active_triangle_id_to_triangle_id_buffer && m_active_triangle_count > 0) {
            m_active_triangle_ids = m_active_triangle_id_to_triangle_id_buffer
                                        ->get_elements<shared::EmissiveTriangleID>(0, m_active_triangle_count);
        }
        m_active_triangle_ids_generations = m_generations;
    }
    return m_active_triangle_ids;
}

} // namespace falcor
