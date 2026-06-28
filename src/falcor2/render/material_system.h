// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "scene_system.h"

#include "falcor2/render/shared_scene_types.h"

#include "falcor2/utils/managed_vector.h"

#include <sgl/device/fwd.h>
#include <sgl/device/shader.h>

#include <span>
#include <vector>

namespace falcor {

/// Scene system responsible for managing materials.
class MaterialSystem : public SceneSystem {
    FALCOR_OBJECT(MaterialSystem)
public:
    /// Constructor.
    /// @param scene The scene this system belongs to.
    MaterialSystem(Scene* scene);

    // SceneSystem interface

    virtual SceneUpdateFlags update(SceneUpdateContext& ctx) override;
    virtual void bind_to_scene(const sgl::ShaderCursor& cursor) const override;

    /// Get the type conformances for all active materials.
    std::span<const sgl::TypeConformance> required_type_conformances() const { return m_type_conformances; }

    /// Get the additional modules required by all active materials.
    std::span<const ref<sgl::SlangModule>> required_modules() const { return m_required_modules; }

private:
    /// Allocate a new material ID.
    shared::MaterialID allocate_material_id();
    /// Free a material ID.
    void free_material_id(shared::MaterialID material_id);

    MaterialCollection& m_materials;

    ManagedVector<shared::MaterialData> m_material_data;

    std::vector<sgl::TypeConformance> m_type_conformances;
    std::vector<ref<sgl::SlangModule>> m_required_modules;

    // Next material ID to allocate. Starts at 1 since 0 is reserved for InvalidMaterial.
    uint32_t m_next_material_id{1};
    std::vector<uint32_t> m_free_material_ids;
};

} // namespace falcor
