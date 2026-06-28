// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/render/scene_system.h"
#include "falcor2/render/shared_scene_types.h"

#include "falcor2/utils/sampling/distribution_1d.h"

#include <sgl/device/fwd.h>
#include <sgl/device/shader.h>


namespace falcor {

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

private:
    void create_kernels();

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

    uint32_t m_triangle_count{0};
    uint32_t m_active_triangle_count{0};

    /// List of (potentially) emissive triangles.
    std::vector<shared::EmissiveTriangle> m_triangles;
    /// List of (potentially) emissive triangle fluxes.
    std::vector<float> m_triangle_flux;
    /// List of active emissive triangles that have non-zero emission.
    std::vector<shared::EmissiveTriangleID> m_active_triangle_ids;
    /// List of active emissive triangle fluxes.
    std::vector<float> m_active_triangle_flux;

    ref<sgl::Buffer> m_geometry_instance_to_triangle_id_buffer;
    ref<sgl::Buffer> m_triangles_buffer;
    ref<sgl::Buffer> m_triangle_emission_buffer;
    ref<sgl::Buffer> m_triangle_flux_buffer;
    ref<sgl::Buffer> m_triangle_id_to_active_triangle_id_buffer;
    ref<sgl::Buffer> m_active_triangle_id_to_triangle_id_buffer;
    ref<sgl::Buffer> m_active_triangle_flux_buffer;

    /// TODO(scene): We might wanna move this to a light sampler object instead.
    /// Alias table over all active emissive triangles (for sampling).
    ref<AliasTable1D> m_active_triangle_alias_table;
};

} // namespace falcor
