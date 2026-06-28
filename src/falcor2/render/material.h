// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/render/scene_object.h"
#include "falcor2/render/shared_scene_types.h"

#include "falcor2/core/cursor_writer.h"
#include "falcor2/core/properties.h"
#include "falcor2/core/enum.h"


#include <sgl/device/buffer_cursor.h>
#include <sgl/device/shader_cursor.h>

namespace sgl {
class SignatureBuffer;
} // namespace sgl

namespace falcor {

class Material;

/// Base class for scene materials.
class FALCOR_API Material : public SceneObject {
    FALCOR_SCENE_OBJECT(Material, SceneObject)
public:
    FALCOR_STATIC_WRITE_TO_CURSOR(Material);

    /// Dirty flags for Material.
    enum class DirtyFlags : uint32_t {
        none = 0,
        properties = (1u << 0),
        // Shared SceneObject::DirtyFlags
        resources = (1u << 29),
        added = (1u << 30),
        removed = (1u << 31),
    };
    SGL_ENUM_FLAGS_INFO(
        DirtyFlags,
        {
            {DirtyFlags::none, "none"},
            {DirtyFlags::properties, "properties"},
            {DirtyFlags::resources, "resources"},
            {DirtyFlags::added, "added"},
            {DirtyFlags::removed, "removed"},
        }
    );

    /// Constructor.
    Material();

    /// Destructor.
    virtual ~Material();

    // SceneObject interface

    virtual SceneObjectKind kind() const override { return SceneObjectKind::material; }

    virtual void remove() override;

    // Material interface

    /// Slang type name of the struct representing this material type.
    /// Specifies the type used for write_to_cursor().
    const std::string& slang_type_name() const { return m_slang_type_name; }
    void set_slang_type_name(std::string_view slang_type_name) { m_slang_type_name = slang_type_name; }

    /// Material ID.
    shared::MaterialID material_id() const { return m_material_id; }
    void set_material_id(shared::MaterialID material_id) { m_material_id = material_id; }

    /// Called during Scene::update() to update this material.
    /// @param ctx Update context.
    virtual void update(SceneUpdateContext& ctx) { FALCOR_UNUSED(ctx); }

    /// Called during Scene::update() to write this material to the corresponding slang struct (BufferElementCursor).
    virtual void write_to_cursor(sgl::BufferElementCursor cursor) const { FALCOR_UNUSED(cursor); }

    /// Called during Scene::update() to write this material to the corresponding slang struct (ShaderCursor).
    virtual void write_to_cursor(sgl::ShaderCursor cursor) const { FALCOR_UNUSED(cursor); }

    /// Write the SlangPy cache signature for cursor-writer binding.
    static void write_slangpy_signature(sgl::SignatureBuffer& signature, const Material* value);

    /// Returns material flags used by the scene material header.
    virtual shared::MaterialFlags flags() const { return shared::MaterialFlags::none; }

    /// Returns an additional Slang module required by this material.
    virtual ref<sgl::SlangModule> required_module() const { return {}; }

    /// Reflect this class.
    template<reflection::ClassReflector R>
    static void reflect(R& r)
    {
        FALCOR_UNUSED(r);
    }

protected:
    void mark_dirty(DirtyFlags flags);

    std::string m_slang_type_name;
    shared::MaterialID m_material_id{shared::MaterialID::invalid};
};

FALCOR_ENUM_CLASS_OPERATORS(Material::DirtyFlags);
SGL_ENUM_REGISTER(Material::DirtyFlags);

#define FALCOR_SCENE_REGISTER_MATERIAL(name) FALCOR_SCENE_REGISTER_CLASS(name, Material)

} // namespace falcor
