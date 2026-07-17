// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/render/scene_system.h"
#include "falcor2/render/shared_scene_types.h"

#include "falcor2/utils/managed_vector.h"
#include "falcor2/utils/sampling/distribution_1d.h"

#include <sgl/device/fwd.h>

namespace falcor {

/// Scene system responsible for managing lights.
class LightSystem : public SceneSystem {
    FALCOR_OBJECT(LightSystem)
public:
    /// Constructor.
    /// @param scene The scene this system belongs to.
    LightSystem(Scene* scene);

    // SceneSystem interface

    virtual SceneUpdateFlags update(SceneUpdateContext& ctx) override;
    virtual void bind_to_scene(const sgl::ShaderCursor& cursor) const override;

    // LightSystem interface

    /// Get the type conformances for all light types.
    std::span<const sgl::TypeConformance> required_type_conformances() const { return m_type_conformances; }

private:
    void create_kernels();

    ComponentCollection& m_components;

    ref<sgl::ComputeKernel> m_compute_environment_light_powers_kernel;
    ref<sgl::Buffer> m_environment_light_power_buffer;

    std::vector<sgl::TypeConformance> m_type_conformances;
    ManagedVector<shared::LightData> m_light_data;
    uint32_t m_light_count{0};
    uint32_t m_analytic_light_count{0};
    shared::LightID m_environment_light_id{shared::LightID::invalid};
    uint32_t m_environment_light_count{0};
    ref<AliasTable1D> m_analytic_light_selection_distribution;
    ref<AliasTable1D> m_environment_light_selection_distribution;
};

} // namespace falcor
