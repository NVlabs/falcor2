// SPDX-License-Identifier: Apache-2.0

#include "emissive_geometry_system.h"

#include "falcor2/render/scene.h"
#include "falcor2/render/entity.h"
#include "falcor2/render/component/light.h"
#include "falcor2/render/component/geometry_instance.h"

#include "falcor2/utils/algorithm/prefix_sum.h"
#include "falcor2/utils/io/mesh_writer.h"

#include <sgl/core/hash.h>

#include <sgl/device/device.h>
#include <sgl/device/kernel.h>
#include <sgl/device/command.h>
#include <sgl/device/shader.h>
#include <sgl/device/shader_cursor.h>

#include <algorithm>

namespace falcor {

EmissiveGeometrySystem::EmissiveGeometrySystem(Scene* scene)
    : SceneSystem{scene}
    , m_device(scene->device())
    , m_geometries(scene->geometry_collection())
    , m_materials(scene->material_collection())
    , m_entities(scene->entity_collection())
    , m_components(scene->component_collection())
{
}

void EmissiveGeometrySystem::create_kernels()
{
    // Early out if kernels are already created and up-to-date with the scene requirements.
    if (m_gather_triangle_count_kernel && m_requirements_generation == m_scene->requirements_generation())
        return;

    m_requirements_generation = m_scene->requirements_generation();

    const SceneRequirements& requirements = m_scene->requirements();

    std::vector<ref<sgl::SlangModule>> modules;
    modules.push_back(m_device->load_module("falcor2/render/kernels/emissive_geometry_kernels.slang"));
    modules.insert(modules.end(), requirements.modules.begin(), requirements.modules.end());
    ref<sgl::SlangModule> module
        = m_device->compose_modules("emissive_geometry_kernels", modules, requirements.type_conformances);

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
}

SceneUpdateFlags EmissiveGeometrySystem::update(SceneUpdateContext& ctx)
{
    create_kernels();

    static constexpr bool ENABLE_DEBUG = false;

    const bool materials_changed = m_materials.is_dirty();
    const bool geometries_changed = m_geometries.is_dirty();

    // TODO: We need better ways to track changes in components.
    // TODO: We should separate different transform changes: scale changes need to update emission distributions,
    // non-scale changes only need to update triangle data.
    bool transforms_changed = false;
    bool instances_changed = false;
    if (m_entities.is_dirty() || m_components.is_dirty()) {
        for (size_t i = 0; i < m_components.size(); ++i) {
            Component::DirtyFlags dirty_flags = m_components.dirty_flags()[i];
            if (dirty_flags == Component::DirtyFlags::none)
                continue;
            if (GeometryInstance* instance = m_components[i]->as<GeometryInstance>()) {
                instances_changed |= is_set(dirty_flags, ~Component::DirtyFlags::entity_transform);
                transforms_changed |= is_set(dirty_flags, Component::DirtyFlags::entity_transform);
            }
        }
    }

    if (!materials_changed && !geometries_changed && !transforms_changed && !instances_changed)
        return SceneUpdateFlags::none;

    const bool update_emission = materials_changed || geometries_changed || instances_changed;
    const bool update_triangles = update_emission || transforms_changed;

    RenderScene* render_scene = m_scene->_render_scene();

    uint32_t geometry_instance_count = sgl::narrow_cast<uint32_t>(render_scene->geometry_instance_desc_count());
    if (geometry_instance_count == 0)
        return SceneUpdateFlags::none;

    if (update_emission) {

        // ------------------------------------------------------------------------
        // Gather potentially emissive geometry instances and count triangles.
        // ------------------------------------------------------------------------

        // Gather emissive triangle counts.
        // is_emissive[geometry_instance_id] = 0 (not emissive) / 1 (potentially emissive)
        // triangle_count[geometry_instance_id] = 0 (not emissive) / triangle count (potentially emissive)

        // Create temporary buffer for is_emissive.
        ref<sgl::Buffer> is_emissive_buffer = m_device->create_buffer({
            .size = geometry_instance_count * sizeof(uint32_t),
            .usage = sgl::BufferUsage::unordered_access | sgl::BufferUsage::shader_resource,
            .label = "is_emissive",
        });
        // Create temporary buffer for triangle_count.
        ref<sgl::Buffer> triangle_count_buffer = m_device->create_buffer({
            .size = geometry_instance_count * sizeof(uint32_t),
            .usage = sgl::BufferUsage::unordered_access | sgl::BufferUsage::shader_resource,
            .label = "triangle_count",
        });
        // Dispatch kernel to gather emissive triangle counts.
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
        // Debug output.
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

        // Create temporary buffer for triangle_offset.
        ref<sgl::Buffer> triangle_offset_buffer = m_device->create_buffer({
            .size = geometry_instance_count * sizeof(uint32_t),
            .usage = sgl::BufferUsage::unordered_access | sgl::BufferUsage::shader_resource,
            .label = "triangle_offset",
        });
        // Create temporary buffer for total_triangle_count.
        ref<sgl::Buffer> total_triangle_count_buffer = m_device->create_buffer({
            .size = sizeof(uint32_t),
            .usage = sgl::BufferUsage::unordered_access | sgl::BufferUsage::shader_resource,
            .label = "total_triangle_count",
        });
        // Copy triangle_count_buffer to triangle_offset_buffer to use it as input for prefix sum.
        ctx.command_encoder()->copy_buffer(
            triangle_offset_buffer,
            0,
            triangle_count_buffer,
            0,
            geometry_instance_count * sizeof(uint32_t)
        );
        // Compute triangle offsets and total triangle count using prefix sum.
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
        // Debug output.
        if (ENABLE_DEBUG) {
            ctx.submit();
            auto triangle_offset = triangle_offset_buffer->get_elements<uint32_t>();
            SGL_PRINT(m_triangle_count);
            SGL_PRINT(triangle_offset);
        }

        if (m_triangle_count > 0) {
            // ------------------------------------------------------------------------
            // Prepare geometry_instance_to_triangle_id lookup table
            // ------------------------------------------------------------------------

            // Write geometry_instance_to_triangle_id lookup table.
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
                auto geometry_instance_to_triangle_id
                    = m_geometry_instance_to_triangle_id_buffer->get_elements<uint32_t>();
                SGL_PRINT(geometry_instance_to_triangle_id);
            }

            // ------------------------------------------------------------------------
            // Prepare triangles buffer
            // ------------------------------------------------------------------------

            if (!m_triangles_buffer
                || m_triangles_buffer->size() < m_triangle_count * sizeof(shared::PackedEmissiveTriangle)) {
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

            // ------------------------------------------------------------------------
            // Setup per-triangle emission sampling parameters
            // ------------------------------------------------------------------------

            // Setup triangle sampling parameters.
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

            uint32_t thread_count = static_cast<uint32_t>(std::min(total_sample_count, uint64_t(1024 * 1024)));
            m_compute_triangle_max_emission_kernel->dispatch(
                uint3(thread_count, 1, 1),
                [&](sgl::ShaderCursor cursor)
                {
                    m_scene->bind(cursor, SceneBindFlags::all & ~SceneBindFlags::emissive_geometry);
                    cursor = cursor.find_entry_point(0);
                    cursor["triangles"] = m_triangles_buffer;
                    cursor["sample_offset"] = sample_offset_buffer;
                    cursor["sample_count"] = sample_count_buffer;
                    cursor["sample_grid"] = sample_grid_buffer;
                    cursor["triangle_count"] = m_triangle_count;
                    cursor["thread_count"] = thread_count;
                    cursor["total_sample_count"] = total_sample_count;
                    cursor["max_emission"] = max_emission_buffer;
                },
                ctx.command_encoder()
            );

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

            m_accumulate_triangle_emission_kernel->dispatch(
                uint3(thread_count, 1, 1),
                [&](sgl::ShaderCursor cursor)
                {
                    m_scene->bind(cursor, SceneBindFlags::all & ~SceneBindFlags::emissive_geometry);
                    cursor = cursor.find_entry_point(0);
                    cursor["triangles"] = m_triangles_buffer;
                    cursor["sample_offset"] = sample_offset_buffer;
                    cursor["sample_count"] = sample_count_buffer;
                    cursor["sample_grid"] = sample_grid_buffer;
                    cursor["emission_factor"] = emission_factor_buffer;
                    cursor["triangle_count"] = m_triangle_count;
                    cursor["thread_count"] = thread_count;
                    cursor["total_sample_count"] = total_sample_count;
                    cursor["accumulated_emission"] = accumulated_emission_buffer;
                },
                ctx.command_encoder()
            );

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

            // ------------------------------------------------------------------------
            // Update active triangles
            // ------------------------------------------------------------------------

            // Create temporary buffer for is_active.
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
                    || m_active_triangle_id_to_triangle_id_buffer->size()
                        < m_active_triangle_count * sizeof(uint32_t)) {
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
                auto triangle_id_to_active_triangle_id
                    = m_triangle_id_to_active_triangle_id_buffer->get_elements<uint32_t>();
                auto active_triangle_id_to_triangle_id
                    = m_active_triangle_id_to_triangle_id_buffer->get_elements<uint32_t>();
                SGL_PRINT(is_active);
                SGL_PRINT(active_offset);
                SGL_PRINT(triangle_id_to_active_triangle_id);
                SGL_PRINT(active_triangle_id_to_triangle_id);
            }
        } else {
            m_active_triangle_count = 0;
            m_geometry_instance_to_triangle_id_buffer = nullptr;
            m_triangles_buffer = nullptr;
            m_triangle_emission_buffer = nullptr;
            m_triangle_id_to_active_triangle_id_buffer = nullptr;
            m_active_triangle_id_to_triangle_id_buffer = nullptr;
        }
    }

    if (update_triangles) {

        if (m_triangle_count > 0) {

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
                },
                ctx.command_encoder()
            );

            ctx.submit();
            if (m_active_triangle_flux_buffer) {
                m_active_triangle_flux = m_active_triangle_flux_buffer->get_elements<float>(0, m_active_triangle_count);
            } else {
                m_active_triangle_flux.clear();
            }
            if (ENABLE_DEBUG) {
                auto triangle_id_to_active_triangle_id
                    = m_triangle_id_to_active_triangle_id_buffer->get_elements<uint32_t>();
                auto active_triangle_id_to_triangle_id
                    = m_active_triangle_id_to_triangle_id_buffer->get_elements<uint32_t>();
                SGL_PRINT(m_active_triangle_flux);
                SGL_PRINT(triangle_id_to_active_triangle_id);
                SGL_PRINT(active_triangle_id_to_triangle_id);
            }

            // Build alias table.

            m_active_triangle_alias_table.reset(
                new AliasTable1D(m_device, m_active_triangle_flux, "EmissiveTriangle::active_triangle_alias_table")
            );

        } else {
            m_triangle_flux_buffer = nullptr;
            m_active_triangle_flux_buffer = nullptr;
            m_active_triangle_alias_table = nullptr;
        }
    }

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

    return SceneUpdateFlags::none;
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
    if (m_active_triangle_alias_table)
        emissive_geometry["active_triangle_alias_table"] = *m_active_triangle_alias_table;
}

} // namespace falcor
