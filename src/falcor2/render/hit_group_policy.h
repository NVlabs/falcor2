// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/core/enum.h"
#include "falcor2/core/error.h"

#include <functional>

namespace falcor {

/// Policy to determine how hit groups are generated for the scene.
struct HitGroupPolicy {
    enum class Mode {
        per_geometry_type,
        per_geometry_instance_id,
    };
    SGL_ENUM_INFO(
        Mode,
        {
            {Mode::per_geometry_type, "per_geometry_type"},
            {Mode::per_geometry_instance_id, "per_geometry_instance_id"},
        }
    );

    /// Policy mode.
    Mode mode{Mode::per_geometry_type};
    /// Number of ray types.
    uint32_t ray_type_count{1};
    /// Number of geometry types (used for per_geometry_type mode).
    uint32_t geometry_type_count{2};

    uint32_t instance_contribution_to_hit_group_index(uint32_t geometry_type, uint32_t instance_id) const
    {
        switch (mode) {
        case Mode::per_geometry_type:
            return geometry_type * ray_type_count;
        case Mode::per_geometry_instance_id:
            return instance_id * ray_type_count;
        default:
            FALCOR_THROW("Invalid hit group policy mode.");
        }
    }

    uint32_t
    hit_group_index(uint32_t ray_type, uint32_t geometry_type, uint32_t instance_id, uint32_t geometry_index) const
    {
        uint32_t base_index = instance_contribution_to_hit_group_index(geometry_type, instance_id);
        uint32_t geometry_offset = geometry_index * multiplier_for_geometry_contribution_to_hit_group_index();
        return base_index + geometry_offset + ray_type;
    }

    uint32_t multiplier_for_geometry_contribution_to_hit_group_index() const
    {
        switch (mode) {
        case Mode::per_geometry_type:
            return 0;
        case Mode::per_geometry_instance_id:
            return ray_type_count;
        default:
            FALCOR_THROW("Invalid hit group policy mode.");
        }
    }
};

SGL_ENUM_REGISTER(HitGroupPolicy::Mode);

} // namespace falcor
