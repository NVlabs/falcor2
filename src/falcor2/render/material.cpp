// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "material.h"

#include "falcor2/render/scene.h"
#include "falcor2/render/material_system.h"

#include <sgl/core/signature_buffer.h>
#include <sgl/device/cursor_utils.h>

#include <algorithm>

namespace falcor {

Material::Material() { }

Material::~Material() { }

void Material::ensure_data_size(size_t byte_size)
{
    FALCOR_CHECK(m_scene && is_valid(), "Material data requires a live scene material");
    auto& storage = m_scene->_material_system()->m_data_storage;
    if (byte_size == 0) {
        if (!m_data_allocation.is_valid())
            return;
        storage.free(m_data_allocation);
        m_data_allocation = {};
    } else if (byte_size > storage.allocation_size(m_data_allocation)) {
        auto allocation = storage.allocate(byte_size);
        // Allocation may move the CPU buffer, so acquire both views afterwards.
        auto source = storage.data(m_data_allocation);
        auto target = storage.mutable_data(allocation);
        if (!source.empty())
            std::copy(source.begin(), source.end(), target.begin());
        std::fill(target.begin() + source.size(), target.end(), uint8_t(0));
        storage.free(m_data_allocation);
        m_data_allocation = allocation;
    } else {
        return;
    }
    mark_dirty_properties();
}

std::span<uint8_t> Material::mutable_data()
{
    FALCOR_CHECK(m_scene && is_valid(), "Material data requires a live scene material");
    mark_dirty_properties();
    return m_scene->_material_system()->m_data_storage.mutable_data(m_data_allocation);
}

std::span<const uint8_t> Material::data() const
{
    FALCOR_CHECK(m_scene && is_valid(), "Material data requires a live scene material");
    return m_scene->_material_system()->m_data_storage.data(m_data_allocation);
}

BufferOffsetPair Material::data_ref() const
{
    if (!is_valid())
        return {};
    return m_scene->_material_system()->data_ref(m_data_allocation);
}

void Material::remove()
{
    SceneObject::remove();
    m_scene->_mark_material_removed(this);
}

void Material::mark_dirty(DirtyFlags flags)
{
    m_scene->_mark_material_dirty(this, flags);
}

void Material::write_slangpy_signature(sgl::SignatureBuffer& signature, const Material* value)
{
    signature.add("falcor2.Material:");
    signature.add(value ? value->slang_type_name() : "<null>");
}

FALCOR_STATIC_ONCE(reflection::register_type<Material>());
FALCOR_STATIC_ONCE(sgl::cursor_utils::register_cursor_writer<Material>());

} // namespace falcor
