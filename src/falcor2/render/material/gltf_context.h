// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "gltf_specialization.h"
#include "falcor2/render/fwd.h"

#include <set>

namespace falcor {

/// Selects and tracks the active shader specializations for native glTF materials.
class GLTFContext {
public:
    GLTFContext(GLTFSpecializationMode mode, GLTFFeatureFlags allowed_features);

    /// Recompute the live masks and assign compile masks before type conformances are collected.
    void update(MaterialCollection& materials);

    const std::set<GLTFMasks>& metallic_roughness_specializations() const { return m_metallic_roughness.active; }
    const std::set<GLTFMasks>& specular_glossiness_specializations() const { return m_specular_glossiness.active; }

private:
    struct WorkflowState {
        GLTFMasks effective_union;
        GLTFMasks nontransmissive_union;
        GLTFMasks transmissive_union;
        std::set<GLTFMasks> active;
    };

    GLTFSpecializationMode m_mode;
    uint32_t m_allowed_features;
    WorkflowState m_metallic_roughness;
    WorkflowState m_specular_glossiness;
};

} // namespace falcor
