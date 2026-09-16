// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/render/scene_system.h"
#include "falcor2/render/shared_emissive_geometry_types.h"

#include <sgl/device/fwd.h>
#include <sgl/device/shader.h>

#include <cstdint>
#include <span>
#include <vector>

namespace falcor {

/// Independent monotonic generations for sampler-relevant emissive geometry data.
/// Each generation advances when data in its domain may have changed.
struct EmissiveGeometryGenerations {
    /// Triangle counts, active set, or identifier mappings may have changed.
    uint64_t topology{0};
    /// World-space triangle positions, normals, or areas may have changed.
    uint64_t geometry{0};
    /// Active triangle emitted radiant flux may have changed.
    uint64_t flux{0};

    bool operator==(const EmissiveGeometryGenerations&) const = default;
};

/// Scene system responsible for managing emissive geometry.
class EmissiveGeometrySystem : public SceneSystem {
    FALCOR_OBJECT(EmissiveGeometrySystem)
public:
    /// Constructor.
    /// @param scene The scene this system belongs to.
    EmissiveGeometrySystem(Scene* scene);

    // SceneSystem interface

    virtual SceneUpdateFlags update(SceneUpdateContext& ctx) override;
    virtual void bind_to_scene(const sgl::ShaderCursor& cursor) const override;

    /// Current sampler-relevant data generations.
    const EmissiveGeometryGenerations& generations() const { return m_generations; }

    /// Total number of potentially emissive triangles.
    uint32_t triangle_count() const { return m_triangle_count; }

    /// Number of active triangles with non-zero emission.
    uint32_t active_triangle_count() const { return m_active_triangle_count; }

    /// Luminance-weighted one-sided emitted radiant flux in active triangle order.
    std::span<const float> active_triangle_flux() const { return m_active_triangle_flux; }

    /// CPU snapshot of all potentially emissive triangles in global triangle ID order.
    FALCOR_API std::span<const shared::EmissiveTriangle> triangles() const;

    /// Luminance-weighted one-sided emitted radiant flux in global triangle ID order.
    FALCOR_API std::span<const float> triangle_flux() const;

    /// Global triangle IDs in active triangle order.
    FALCOR_API std::span<const shared::EmissiveTriangleID> active_triangle_ids() const;

private:
    void create_kernels();
    void clear_resources();

private:
    sgl::Device* m_device;
    GeometryCollection& m_geometries;
    MaterialCollection& m_materials;
    EntityCollection& m_entities;
    ComponentCollection& m_components;

    ref<sgl::ComputeKernel> m_gather_triangle_count_kernel;
    ref<sgl::ComputeKernel> m_setup_geometry_instance_to_triangle_id_kernel;
    ref<sgl::ComputeKernel> m_setup_triangle_kernel;
    ref<sgl::ComputeKernel> m_setup_triangle_sampling_kernel;
    ref<sgl::ComputeKernel> m_compute_triangle_max_emission_kernel;
    ref<sgl::ComputeKernel> m_compute_triangle_max_emission_factor_kernel;
    ref<sgl::ComputeKernel> m_accumulate_triangle_emission_kernel;
    ref<sgl::ComputeKernel> m_finalize_triangle_emission_kernel;
    ref<sgl::ComputeKernel> m_gather_active_triangles_kernel;
    ref<sgl::ComputeKernel> m_setup_active_triangle_mapping_kernel;
    ref<sgl::ComputeKernel> m_update_triangle_kernel;

    uint64_t m_requirements_generation{0};
    EmissiveGeometryGenerations m_generations;

    uint32_t m_triangle_count{0};
    uint32_t m_active_triangle_count{0};

    /// List of (potentially) emissive triangles.
    mutable std::vector<shared::EmissiveTriangle> m_triangles;
    mutable EmissiveGeometryGenerations m_triangles_generations;
    /// Luminance-weighted one-sided flux of each potentially emissive triangle.
    mutable std::vector<float> m_triangle_flux;
    mutable EmissiveGeometryGenerations m_triangle_flux_generations;
    /// List of active emissive triangles that have non-zero emission.
    mutable std::vector<shared::EmissiveTriangleID> m_active_triangle_ids;
    mutable EmissiveGeometryGenerations m_active_triangle_ids_generations;
    /// Luminance-weighted one-sided flux of each active emissive triangle.
    std::vector<float> m_active_triangle_flux;

    ref<sgl::Buffer> m_geometry_instance_to_triangle_id_buffer;
    ref<sgl::Buffer> m_triangles_buffer;
    ref<sgl::Buffer> m_triangle_emission_buffer;
    ref<sgl::Buffer> m_triangle_flux_buffer;
    ref<sgl::Buffer> m_triangle_id_to_active_triangle_id_buffer;
    ref<sgl::Buffer> m_active_triangle_id_to_triangle_id_buffer;
    ref<sgl::Buffer> m_active_triangle_flux_buffer;
};

} // namespace falcor
