// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "material.h"

#include "falcor2/render/scene.h"

#include <sgl/core/signature_buffer.h>
#include <sgl/device/cursor_utils.h>

namespace falcor {

Material::Material() { }

Material::~Material() { }

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
