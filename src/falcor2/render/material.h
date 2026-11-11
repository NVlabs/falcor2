// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/render/scene_object.h"
#include "falcor2/render/scene_capabilities.h"
#include "falcor2/render/shared_scene_types.h"
#include "falcor2/render/texture_manager.h"

#include "falcor2/core/cursor_writer.h"
#include "falcor2/core/properties.h"
#include "falcor2/core/enum.h"
#include "falcor2/utils/buffer_offset_pair.h"
#include "falcor2/utils/offset_allocator.h"


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
        backface_culling = (1u << 1),
        // An edit may change emissive eligibility, emission sampling, or emitted power.
        emission = (1u << 2),
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
            {DirtyFlags::backface_culling, "backface_culling"},
            {DirtyFlags::emission, "emission"},
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
    /// - two_sided indicates the material can interact with rays on either side of the geometric surface.
    /// - thin_walled is definitive but independent of whether the material can transmit.
    virtual shared::MaterialFlags flags() const { return shared::MaterialFlags::two_sided; }

    /// Nested dielectric priority. Zero denotes the highest effective priority.
    uint32_t nested_priority() const { return m_nested_priority; }

    /// Whether hardware triangle backface culling is enabled for this material.
    bool enable_backface_culling() const { return m_enable_backface_culling; }
    void set_enable_backface_culling(bool value)
    {
        if (m_enable_backface_culling == value)
            return;
        m_enable_backface_culling = value;
        mark_dirty_backface_culling();
    }

    /// Complete description of opacity evaluation for this material.
    struct OpacityDesc {
        shared::OpacityFlags flags{shared::OpacityFlags::none};
        TextureHandle texture_handle;
        uint32_t texture_channel{3};
        float factor{1.f};
        float threshold{0.f};
    };

    virtual OpacityDesc opacity_desc() const { return {}; }

    /// Returns the capabilities required by this material after its last update.
    /// Includes all generated material outputs and may conservatively include unused capabilities.
    /// The default implementation reports opacity requirements.
    /// Materials that access geometry properties must declare that requirement.
    virtual SceneCapabilities required_capabilities() const { return material_capabilities(false); }

    /// Returns an additional Slang module required by this material.
    virtual ref<sgl::SlangModule> required_module() const { return {}; }

    /// Reflect this class.
    template<reflection::ClassReflector R>
    static void reflect(R& r)
    {
        r.def_rw(
            "nested_priority",
            &Material::m_nested_priority,
            "Nested dielectric priority. Zero denotes the highest effective priority.",
            reflection::default_value(0u),
            reflection::value_range(0, 255),
            reflection::on_change(&Material::mark_dirty_properties)
        );
        r.def_rw(
            "enable_backface_culling",
            &Material::m_enable_backface_culling,
            "Whether hardware triangle backface culling is enabled for this material.",
            reflection::default_value(false),
            reflection::on_change(&Material::mark_dirty_backface_culling)
        );
    }

protected:
    /// Combines geometry-property requirements with the material's opacity requirements.
    SceneCapabilities material_capabilities(bool uses_geometry_properties) const
    {
        SceneCapabilities result;
        if (uses_geometry_properties)
            result.flags |= SceneCapabilityFlags::geometry_properties;
        if (is_set(opacity_desc().flags, shared::OpacityFlags::enabled))
            result.flags |= SceneCapabilityFlags::opacity_evaluation;
        return result;
    }

    /// Ensure at least byte_size bytes, preserving contents and zeroing new capacity. Zero releases storage.
    void ensure_data_size(size_t byte_size);
    /// Includes allocation padding and marks data dirty. Pool growth invalidates the borrowed span.
    std::span<uint8_t> mutable_data();
    std::span<const uint8_t> data() const;
    /// Current buffer and offset after Scene::update; invalid for empty or removed data.
    BufferOffsetPair data_ref() const;

    /// Mark material changes for the next scene update. Include emission for edits that may change
    /// is_emissive(), compute_emission_sampling_grid(), or eval_emission(), even when currently dark.
    /// Custom materials with unknown emission dependencies should include emission for all authored edits.
    void mark_dirty(DirtyFlags flags);
    void mark_dirty_properties();
    void mark_dirty_resources();
    void mark_dirty_emission_properties();
    void mark_dirty_emission_resources();
    void mark_dirty_backface_culling();

    std::string m_slang_type_name;
    shared::MaterialID m_material_id{shared::MaterialID::invalid};
    uint32_t m_nested_priority{0};

private:
    friend class MaterialSystem;
    OffsetAllocator::Allocation m_data_allocation;
    bool m_enable_backface_culling{false};
};

FALCOR_ENUM_CLASS_OPERATORS(Material::DirtyFlags);
SGL_ENUM_REGISTER(Material::DirtyFlags);

inline void Material::mark_dirty_properties()
{
    mark_dirty(DirtyFlags::properties);
}

inline void Material::mark_dirty_resources()
{
    mark_dirty(DirtyFlags::resources);
}

inline void Material::mark_dirty_emission_properties()
{
    mark_dirty(DirtyFlags::properties | DirtyFlags::emission);
}

inline void Material::mark_dirty_emission_resources()
{
    mark_dirty(DirtyFlags::resources | DirtyFlags::emission);
}

inline void Material::mark_dirty_backface_culling()
{
    mark_dirty(DirtyFlags::properties | DirtyFlags::backface_culling);
}

#define FALCOR_SCENE_REGISTER_MATERIAL(name) FALCOR_SCENE_REGISTER_CLASS(name, Material)

} // namespace falcor
