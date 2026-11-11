// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "gltf_context.h"

#include "gltf_material.h"
#include "gltf_specgloss_material.h"
#include "falcor2/core/error.h"
#include "falcor2/render/scene_object.h"

#include <cstddef>

namespace falcor {

namespace {

namespace F = shared::gltfmat::FeatureBits;
namespace T = shared::gltfmat::TextureBits;
namespace S = shared::gltfmat::SpecGlossTextureBits;

bool is_transmissive(const GLTFMasks& masks)
{
    return (masks.features & (F::TRANSMISSION | F::DIFFUSE_TRANSMISSION)) != 0;
}

uint32_t metallic_roughness_texture_capabilities(uint32_t features)
{
    uint32_t textures = T::BASE_COLOR | T::METALLIC_ROUGHNESS;
    if (features & F::NORMAL)
        textures |= T::NORMAL;
    if (features & F::EMISSION)
        textures |= T::EMISSIVE;
    if (features & F::SPECULAR)
        textures |= T::SPECULAR | T::SPECULAR_COLOR;
    if (features & F::TRANSMISSION)
        textures |= T::TRANSMISSION;
    if (features & F::CLEARCOAT)
        textures |= T::CLEARCOAT | T::CLEARCOAT_ROUGHNESS | T::CLEARCOAT_NORMAL;
    if (features & F::SHEEN)
        textures |= T::SHEEN_COLOR | T::SHEEN_ROUGHNESS;
    if (features & F::IRIDESCENCE)
        textures |= T::IRIDESCENCE | T::IRIDESCENCE_THICKNESS;
    if (features & F::ANISOTROPY)
        textures |= T::ANISOTROPY;
    if (features & F::DIFFUSE_TRANSMISSION)
        textures |= T::DIFFUSE_TRANSMISSION | T::DIFFUSE_TRANSMISSION_COLOR;
    return textures;
}

uint32_t specular_glossiness_texture_capabilities(uint32_t features)
{
    uint32_t textures = S::DIFFUSE | S::SPECULAR_GLOSSINESS;
    if (features & F::NORMAL)
        textures |= S::NORMAL;
    if (features & F::EMISSION)
        textures |= S::EMISSIVE;
    return textures;
}

GLTFMasks all_masks(uint32_t allowed_features, bool specular_glossiness)
{
    const uint32_t features = allowed_features & (specular_glossiness ? F::ALL_SPEC_GLOSS : F::ALL_MR);
    const uint32_t textures = specular_glossiness ? specular_glossiness_texture_capabilities(features)
                                                  : metallic_roughness_texture_capabilities(features);
    return {features, textures, textures, textures};
}

} // namespace

GLTFContext::GLTFContext(GLTFSpecializationMode mode, GLTFFeatureFlags allowed_features)
    : m_mode(mode)
    , m_allowed_features(static_cast<uint32_t>(allowed_features))
{
    FALCOR_CHECK(
        (m_allowed_features & ~F::ALL_MR) == 0,
        "gltf_allowed_features contains unknown bits: 0x{:x}",
        m_allowed_features & ~F::ALL_MR
    );
    FALCOR_CHECK(
        mode == GLTFSpecializationMode::required || mode == GLTFSpecializationMode::single
            || mode == GLTFSpecializationMode::all || mode == GLTFSpecializationMode::balanced,
        "Invalid gltf_specialization_mode"
    );
}

void GLTFContext::update(MaterialCollection& materials)
{
    m_metallic_roughness = {};
    m_specular_glossiness = {};

    const auto include_effective = [](WorkflowState& workflow, const GLTFMasks& masks)
    {
        workflow.effective_union = workflow.effective_union.merged(masks);
        GLTFMasks& balanced_union
            = is_transmissive(masks) ? workflow.transmissive_union : workflow.nontransmissive_union;
        balanced_union = balanced_union.merged(masks);
    };

    // Derive each workflow's live unions before choosing any material's compile masks.
    for (size_t index = 0; index < materials.size(); ++index) {
        Material* material = materials[index];
        if (!material->is_valid())
            continue;
        if (const auto* gltf = dynamic_cast<const GLTFMaterial*>(material)) {
            const GLTFMasks masks = gltf->effective_masks();
            FALCOR_ASSERT((masks.features & ~m_allowed_features) == 0);
            include_effective(m_metallic_roughness, masks);
        } else if (const auto* specular_glossiness = dynamic_cast<const GLTFSpecGlossMaterial*>(material)) {
            const GLTFMasks masks = specular_glossiness->effective_masks();
            FALCOR_ASSERT((masks.features & ~m_allowed_features) == 0);
            include_effective(m_specular_glossiness, masks);
        }
    }

    const GLTFMasks metallic_roughness_all = all_masks(m_allowed_features, false);
    const GLTFMasks specular_glossiness_all = all_masks(m_allowed_features, true);
    const auto select_compile = [&](const WorkflowState& workflow, const GLTFMasks& effective, const GLTFMasks& all)
    {
        switch (m_mode) {
        case GLTFSpecializationMode::required:
            return effective;
        case GLTFSpecializationMode::single:
            return workflow.effective_union;
        case GLTFSpecializationMode::all:
            return all;
        case GLTFSpecializationMode::balanced:
            return is_transmissive(effective) ? workflow.transmissive_union : workflow.nontransmissive_union;
        }
        FALCOR_UNREACHABLE();
    };
    for (size_t index = 0; index < materials.size(); ++index) {
        Material* material = materials[index];
        if (!material->is_valid())
            continue;
        if (auto* gltf = dynamic_cast<GLTFMaterial*>(material)) {
            const GLTFMasks effective = gltf->effective_masks();
            const GLTFMasks compile = select_compile(m_metallic_roughness, effective, metallic_roughness_all);
            FALCOR_ASSERT(compile.covers(effective));
            gltf->set_compile_masks(compile);
            m_metallic_roughness.active.insert(compile);
        } else if (auto* specular_glossiness = dynamic_cast<GLTFSpecGlossMaterial*>(material)) {
            const GLTFMasks effective = specular_glossiness->effective_masks();
            const GLTFMasks compile = select_compile(m_specular_glossiness, effective, specular_glossiness_all);
            FALCOR_ASSERT(compile.covers(effective));
            specular_glossiness->set_compile_masks(compile);
            m_specular_glossiness.active.insert(compile);
        }
    }
}

} // namespace falcor
