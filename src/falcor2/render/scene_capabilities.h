// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/core/enum.h"
#include "falcor2/core/macros.h"

#include <compare>
#include <cstdint>

namespace falcor {

/// Capabilities that may be required to render a scene.
enum class SceneCapabilityFlags : uint32_t {
    none = 0,
    /// Access to additional geometry attributes and object-space transforms.
    geometry_properties = 1u << 0,
    /// Support material-authored AOV writes; consumers still select which outputs to request and bind.
    material_aovs = 1u << 1,
    /// Evaluate material opacity when accepting or rejecting ray intersections.
    opacity_evaluation = 1u << 2,
    /// Track interiors when a material can transmit through a non-thin surface.
    solid_transmission = 1u << 3,
    /// Enable all supported shader capabilities, including AOV support without an output request.
    all = geometry_properties | material_aovs | opacity_evaluation | solid_transmission,
};
FALCOR_ENUM_CLASS_OPERATORS(SceneCapabilityFlags);
SGL_ENUM_FLAGS_INFO(
    SceneCapabilityFlags,
    {
        {SceneCapabilityFlags::none, "none"},
        {SceneCapabilityFlags::geometry_properties, "geometry_properties"},
        {SceneCapabilityFlags::material_aovs, "material_aovs"},
        {SceneCapabilityFlags::opacity_evaluation, "opacity_evaluation"},
        {SceneCapabilityFlags::solid_transmission, "solid_transmission"},
        {SceneCapabilityFlags::all, "all"},
    }
);
SGL_ENUM_REGISTER(SceneCapabilityFlags);

/// A set of capabilities required or selected for rendering a scene.
struct FALCOR_API SceneCapabilities {
    SceneCapabilityFlags flags{SceneCapabilityFlags::none};

    static SceneCapabilities all() { return {.flags = SceneCapabilityFlags::all}; }
    bool has(SceneCapabilityFlags requested) const { return (flags & requested) == requested; }
    bool covers(const SceneCapabilities& required) const { return has(required.flags); }
    SceneCapabilities merged(const SceneCapabilities& other) const { return {.flags = flags | other.flags}; }
    auto operator<=>(const SceneCapabilities&) const = default;
};

/// Selects the capabilities supported by a rendering configuration.
enum class SceneCapabilityPolicy {
    /// Enable every supported capability, independently of scene and consumer requirements.
    all,
    /// Enable the combined conservative requirements of the scene and the consumer.
    required,
    /// Use caller-specified capabilities, rejecting selections that omit scene or consumer requirements.
    specified,
};
SGL_ENUM_INFO(
    SceneCapabilityPolicy,
    {
        {SceneCapabilityPolicy::all, "all"},
        {SceneCapabilityPolicy::required, "required"},
        {SceneCapabilityPolicy::specified, "specified"},
    }
);
SGL_ENUM_REGISTER(SceneCapabilityPolicy);

} // namespace falcor
