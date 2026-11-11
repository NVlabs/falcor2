// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/core/object.h"
#include "falcor2/render/ray_tracing_policy.h"
#include "falcor2/render/scene_capabilities.h"
#include "falcor2/render/shared_scene_types.h"

#include <sgl/device/shader.h>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace falcor {

/// Scene shader and ray-tracing configuration. Scene publishes immutable snapshots as ref<const SceneRequirements>.
/// Retained snapshots survive scene updates. Construct or copy before publication; do not retain mutable aliases.
struct FALCOR_API SceneRequirements : Object {
    FALCOR_OBJECT(SceneRequirements)
public:
    /// Scene settings and material modules to link with the consumer shader.
    std::vector<ref<sgl::SlangModule>> modules;

    /// Material and light type conformances, including dispatch IDs, required when linking scene shaders.
    std::vector<sgl::TypeConformance> type_conformances;

    /// Capabilities supported by this shader configuration, independent of current scene usage.
    SceneCapabilities capabilities;

    /// Policy defining the shader binding table layout.
    HitGroupPolicy hit_group_policy;

    /// Bit mask of shared::GeometryType values represented in the scene.
    uint32_t geometry_type_mask{0};

    /// Intersection implementation used for linear swept sphere geometry.
    LSSIntersectionMode lss_intersection_mode{LSSIntersectionMode::hardware};

    /// Required ray-tracing pipeline flags.
    sgl::RayTracingPipelineFlags ray_tracing_pipeline_flags = sgl::RayTracingPipelineFlags::none;

    bool has_geometry_type(shared::GeometryType type) const
    {
        const uint32_t index = static_cast<uint32_t>(type);
        return index < 32 && (geometry_type_mask & (1u << index)) != 0;
    }

    /// Compare modules, type conformances, and capabilities, ignoring ray-tracing setup fields.
    /// Modules compare by underlying Slang component identity, independently of their SGL wrappers.
    /// Returns false for nullptr.
    bool matches_shader_inputs(const SceneRequirements* other) const
    {
        if (this == other)
            return true;
        if (!other || capabilities != other->capabilities || type_conformances != other->type_conformances)
            return false;
        return std::ranges::equal(
            modules,
            other->modules,
            [](const auto& lhs, const auto& rhs)
            {
                return lhs == rhs || (lhs && rhs && lhs->slang_component_type() == rhs->slang_component_type());
            }
        );
    }

    /// Compare all shader and ray-tracing setup fields. Returns false for nullptr.
    bool matches(const SceneRequirements* other) const
    {
        return this == other
            || (matches_shader_inputs(other) && hit_group_policy == other->hit_group_policy
                && geometry_type_mask == other->geometry_type_mask
                && lss_intersection_mode == other->lss_intersection_mode
                && ray_tracing_pipeline_flags == other->ray_tracing_pipeline_flags);
    }

    bool operator==(const SceneRequirements& other) const { return matches(&other); }
};

} // namespace falcor
