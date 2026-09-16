// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "material_system.h"

#include "falcor2/render/scene.h"
#include "falcor2/render/texture_manager.h"

#include <sgl/device/device.h>
#include <sgl/device/command.h>
#include <sgl/device/shader.h>
#include <sgl/device/shader_cursor.h>

#include <unordered_map>
#include <unordered_set>

namespace falcor {

MaterialSystem::MaterialSystem(Scene* scene)
    : SceneSystem{scene}
    , m_materials(scene->material_collection())
    , m_material_data({
          .label = "MaterialSystem::material_data",
      })
{
}

SceneUpdateFlags MaterialSystem::update(SceneUpdateContext& ctx)
{
    if (!m_materials.is_dirty())
        return SceneUpdateFlags::none;

    // Update materials.
    for (size_t i = 0; i < m_materials.size(); ++i) {
        Material* material = m_materials[i];
        if (!material->is_valid())
            continue;
        material->update(ctx);
    }

    // Assign material ids.
    for (size_t i = 0; i < m_materials.size(); ++i) {
        Material* material = m_materials[i];
        Material::DirtyFlags dirty_flags = m_materials.dirty_flags()[i];
        if (is_set(dirty_flags, Material::DirtyFlags::added)) {
            material->set_material_id(allocate_material_id());
        } else if (is_set(dirty_flags, Material::DirtyFlags::removed)) {
            free_material_id(material->material_id());
            material->set_material_id(shared::MaterialID::invalid);
        }
    }

    // Check if any valid material uses opacity evaluation.
    m_requires_opacity_evaluation = false;
    for (size_t i = 0; i < m_materials.size(); ++i) {
        const Material* material = m_materials[i];
        if (!material->is_valid())
            continue;
        if (is_set(material->opacity_desc().flags, shared::OpacityFlags::enabled))
            m_requires_opacity_evaluation = true;
    }

    // Build type conformances.
    // TODO(scene): We should (optionally) only build conformances for materials actually used in the scene.
    std::set<std::string> slang_type_names;
    for (size_t i = 0; i < m_materials.size(); ++i) {
        Material* material = m_materials[i];
        if (!material->is_valid())
            continue;
        slang_type_names.emplace(material->slang_type_name());
    }
    m_type_conformances.clear();
    m_type_conformances.push_back(sgl::TypeConformance{"IMaterial", "InvalidMaterial", 0});
    slang_type_names.erase("InvalidMaterial");
    std::unordered_map<std::string, int32_t> slang_type_name_to_type_id = {{"InvalidMaterial", 0}};
    // Start at 1 since 0 is reserved for InvalidMaterial.
    int32_t type_id = 1;
    for (const std::string& slang_type_name : slang_type_names) {
        m_type_conformances.push_back(sgl::TypeConformance{"IMaterial", slang_type_name, type_id});
        slang_type_name_to_type_id.emplace(slang_type_name, type_id);
        type_id++;
    }

    // Independent loads can produce distinct SlangModule wrappers for the same underlying Slang
    // component. Deduplicate by the component identity that SGL passes to Slang during composition,
    // then order the unique wrappers by name to keep scene requirements deterministic.
    std::unordered_set<slang::IComponentType*> required_component_types;
    m_required_modules.clear();
    for (size_t i = 0; i < m_materials.size(); ++i) {
        Material* material = m_materials[i];
        if (!material->is_valid())
            continue;
        if (const ref<sgl::SlangModule>& module = material->required_module()) {
            if (required_component_types.insert(module->slang_component_type()).second)
                m_required_modules.push_back(module);
        }
    }

    std::sort(
        m_required_modules.begin(),
        m_required_modules.end(),
        [](const ref<sgl::SlangModule>& lhs, const ref<sgl::SlangModule>& rhs)
        {
            return lhs->name() < rhs->name();
        }
    );

    // Resize material data buffer to max required size (some slots might be unused).
    m_material_data.resize(m_next_material_id);

    // Write InvalidMaterial data to slot 0.
    m_material_data[0] = {};
    m_material_data[0].header.type_id = 0; // InvalidMaterial
    m_material_data[0].header.material_data = shared::detail::pack_material_data(shared::MaterialFlags::none, 0);
    m_material_data[0].header.opacity_texture = 0xffffffff;
    m_material_data[0].header.opacity_data
        = shared::detail::pack_opacity_data(shared::OpacityFlags::none, 3, 0.5f, 1.f);

    // Write material data to corresponding slots based on material ID.
    for (size_t i = 0; i < m_materials.size(); ++i) {
        Material* material = m_materials[i];
        if (!material->is_valid())
            continue;

        uint32_t material_index = static_cast<uint32_t>(material->material_id());
        Material::OpacityDesc opacity_desc = material->opacity_desc();
        m_material_data[material_index].header.type_id = slang_type_name_to_type_id[material->slang_type_name()];
        m_material_data[material_index].header.material_data
            = shared::detail::pack_material_data(material->flags(), material->nested_priority());
        m_material_data[material_index].header.opacity_texture = opacity_desc.texture_handle.data();
        m_material_data[material_index].header.opacity_data = shared::detail::pack_opacity_data(
            opacity_desc.flags,
            opacity_desc.texture_channel,
            opacity_desc.threshold,
            opacity_desc.factor
        );

        std::string buffer_type_name = fmt::format("StructuredBuffer<{}>", material->slang_type_name());

        // Search render_module and all required modules for the type layout.
        auto layout = m_scene->render_module()->layout();
        auto buffer_type = layout->find_type_by_name(buffer_type_name.c_str());
        if (!buffer_type) {
            if (const ref<sgl::SlangModule>& module = material->required_module()) {
                layout = module->layout();
                buffer_type = layout->find_type_by_name(buffer_type_name.c_str());
            }
        }
        FALCOR_CHECK(buffer_type, "Failed to find type '{}' in any module.", buffer_type_name);

        auto buffer_layout = layout->get_type_layout(buffer_type);
        auto element_type_layout = buffer_layout->element_type_layout();
        auto buffer_cursor = make_ref<sgl::BufferCursor>(
            m_scene->device()->type(),
            element_type_layout,
            &m_material_data[material_index],
            sizeof(shared::MaterialData)
        );
        material->write_to_cursor((*buffer_cursor)[0]);
    }

    m_material_data.mark_all_dirty();
    m_material_data.update_buffer(ctx.command_encoder());

    return SceneUpdateFlags::materials;
}

void MaterialSystem::bind_to_scene(const sgl::ShaderCursor& cursor) const
{
    auto material_system = cursor["material_system"];
    material_system["material_data"] = m_material_data.buffer();
}

shared::MaterialID MaterialSystem::allocate_material_id()
{
    shared::MaterialID material_id{shared::MaterialID::invalid};
    if (m_free_material_ids.empty()) {
        material_id = shared::MaterialID{m_next_material_id++};
    } else {
        material_id = shared::MaterialID{m_free_material_ids.back()};
        m_free_material_ids.pop_back();
    }
    return material_id;
}

void MaterialSystem::free_material_id(shared::MaterialID material_id)
{
    m_free_material_ids.push_back(static_cast<uint32_t>(material_id));
}

} // namespace falcor
