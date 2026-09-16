// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/render/scene_system.h"
#include "falcor2/render/shared_scene_types.h"

#include "falcor2/utils/managed_vector.h"
#include <sgl/device/fwd.h>

#include <cstdint>
#include <span>
#include <vector>

namespace falcor {

/// Independent monotonic generations for sampler-relevant light data.
/// Each generation advances when data in its domain may have changed.
struct LightGenerations {
    /// Light counts, active state, type, or identifier mappings may have changed.
    uint64_t topology{0};
    /// Light positions, directions, shapes, or other spatial data may have changed.
    uint64_t geometry{0};
    /// Scalar estimates used to construct light-selection distributions may have changed.
    uint64_t selection_weights{0};

    bool operator==(const LightGenerations&) const = default;
};

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

    /// Current sampler-relevant data generations.
    const LightGenerations& generations() const { return m_generations; }

    /// Total number of active component lights.
    uint32_t light_count() const { return m_light_count; }

    /// Number of active analytic lights.
    uint32_t analytic_light_count() const { return m_analytic_light_count; }

    /// Number of active environment lights.
    uint32_t environment_light_count() const { return m_environment_light_count; }

    /// Non-negative luminance-weighted selection estimates in LightID order.
    /// Finite lights report emitted radiant flux; infinite lights report incident radiance integrated over solid angle.
    std::span<const float> light_powers() const;

private:
    void create_kernels();

    ComponentCollection& m_components;

    ref<sgl::ComputeKernel> m_compute_light_powers_kernel;
    ref<sgl::Buffer> m_light_power_buffer;

    std::vector<sgl::TypeConformance> m_type_conformances;
    ManagedVector<shared::LightData> m_light_data;
    uint32_t m_light_count{0};
    uint32_t m_analytic_light_count{0};
    shared::LightID m_environment_light_id{shared::LightID::invalid};
    uint32_t m_environment_light_count{0};
    mutable std::vector<float> m_light_powers;
    mutable LightGenerations m_light_power_generations;
    LightGenerations m_generations;
};

} // namespace falcor
