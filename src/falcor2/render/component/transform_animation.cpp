// SPDX-License-Identifier: Apache-2.0

#include "transform_animation.h"

namespace falcor {

void TransformAnimation::clear_invalid_references()
{
    if (m_animation && !m_animation->is_valid()) {
        m_animation = nullptr;
        mark_dirty(DirtyFlags::animation_state);
    }
}

void TransformAnimation::set_animation(Animation* animation)
{
    if (animation != m_animation) {
        m_animation = animation;
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
