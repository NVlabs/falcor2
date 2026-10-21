// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "transform_animation.h"

#include <algorithm>

namespace falcor {

TransformAnimation::~TransformAnimation()
{
    unregister_animation();
}

void TransformAnimation::unregister_animation()
{
    if (m_animation)
        std::erase(m_animation->m_transform_animations, this);
}

void TransformAnimation::remove()
{
    if (!is_valid())
        return;
    // Removal unregisters the consumer but preserves its retained animation reference.
    unregister_animation();
    mark_dirty(DirtyFlags::animation_state);
    Component::remove();
}

void TransformAnimation::clear_invalid_references()
{
    if (m_animation && !m_animation->is_valid()) {
        unregister_animation();
        m_animation = nullptr;
        mark_dirty(DirtyFlags::animation_state);
    }
}

void TransformAnimation::set_animation(Animation* animation)
{
    FALCOR_CHECK(is_valid() && scene(), "TransformAnimation must be live");
    if (animation) {
        FALCOR_CHECK(
            animation->is_valid() && animation->scene() == scene(),
            "Animation must be live in the same Scene"
        );
    }
    if (animation != m_animation.get()) {
        if (animation)
            animation->m_transform_animations.push_back(this);
        unregister_animation();
        m_animation = ref(animation);
        mark_dirty(DirtyFlags::animation_state);
    }
}

void TransformAnimation::set_translation_channel(AnimationChannelIndex index)
{
    if (index != m_translation) {
        m_translation = index;
        mark_dirty(DirtyFlags::animation_state);
    }
}

void TransformAnimation::set_rotation_channel(AnimationChannelIndex index)
{
    if (index != m_rotation) {
        m_rotation = index;
        mark_dirty(DirtyFlags::animation_state);
    }
}

void TransformAnimation::set_scale_channel(AnimationChannelIndex index)
{
    if (index != m_scale) {
        m_scale = index;
        mark_dirty(DirtyFlags::animation_state);
    }
}

FALCOR_SCENE_REGISTER_COMPONENT(TransformAnimation);

} // namespace falcor
