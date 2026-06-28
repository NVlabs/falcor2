// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/render/component.h"
#include "falcor2/render/animation.h"

namespace falcor {

/// Transform animation component.
/// Binds animation channels to an entity's transform (translation, rotation, scale).
class FALCOR_API TransformAnimation : public Component {
    FALCOR_SCENE_OBJECT(TransformAnimation, Component)
public:
    /// The animation object containing the channels.
    Animation* animation() const { return m_animation; }
    void set_animation(Animation* animation);

    /// Index into Animation::channels for translation (expects float3), -1 if none.
    AnimationChannelIndex translation_channel() const { return m_translation; }
    void set_translation_channel(AnimationChannelIndex index);

    /// Index into Animation::channels for rotation (expects quatf), -1 if none.
    AnimationChannelIndex rotation_channel() const { return m_rotation; }
    void set_rotation_channel(AnimationChannelIndex index);

    /// Index into Animation::channels for scale (expects float3), -1 if none.
    AnimationChannelIndex scale_channel() const { return m_scale; }
    void set_scale_channel(AnimationChannelIndex index);

    /// Returns true if any channel is bound.
    bool has_channels() const { return m_translation != -1 || m_rotation != -1 || m_scale != -1; }

    // SceneObject interface

    virtual void clear_invalid_references() override;

    /// Reflect this class.
    template<reflection::ClassReflector R>
    static void reflect(R& r)
    {
        FALCOR_UNUSED(r);
    }

private:
    Animation* m_animation{nullptr};
    AnimationChannelIndex m_translation{-1};
    AnimationChannelIndex m_rotation{-1};
    AnimationChannelIndex m_scale{-1};
};

} // namespace falcor
