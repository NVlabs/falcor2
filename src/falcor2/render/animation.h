// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/render/scene_object.h"
#include "falcor2/render/scene_types.h"
#include "falcor2/render/animation_timing.h"
#include "falcor2/core/macros.h"
#include "falcor2/core/types.h"

#include <vector>
#include <span>
#include <string>
#include <cstdint>

namespace falcor {

/// Index type for referencing animation channels. -1 means no channel.
using AnimationChannelIndex = int;

/// A single animation channel storing typed keyframe data.
/// A channel may belong to at most one Animation. Setters request reevaluation on
/// the next Scene::update(), even at unchanged time. Scene-dependent edits follow
/// the owning Animation's scene lifetime contract. Detached channels remain editable.
class FALCOR_API AnimationChannel : public Object {
    FALCOR_OBJECT(AnimationChannel)
public:
    AnimationChannel() = default;

    /// Accessor caches the last queried keyframe index for sequential playback optimization.
    /// Use the default (nullptr) for single-threaded evaluation, or provide an external
    /// Accessor for thread-safe concurrent evaluation of the same channel.
    struct Accessor {
        size_t last_index = 0;
    };

    /// Value type of keyframe data.
    AnimationValueType value_type() const { return m_value_type; }
    void set_value_type(AnimationValueType value_type);
    /// Interpolation mode.
    AnimationInterpolationMode interpolation_mode() const { return m_interpolation_mode; }
    void set_interpolation_mode(AnimationInterpolationMode interpolation_mode);
    /// Keyframe timestamps in seconds.
    const std::vector<float>& times() const { return m_times; }
    void set_times(std::vector<float> times);
    /// Keyframe values (flat array: 1/3/4 floats per key depending on value_type).
    const std::vector<float>& values() const { return m_values; }
    void set_values(std::vector<float> values);
    /// In-tangent per keyframe (same layout as values), only populated for cubic_spline.
    const std::vector<float>& in_tangents() const { return m_in_tangents; }
    void set_in_tangents(std::vector<float> in_tangents);
    /// Out-tangent per keyframe (same layout as values), only populated for cubic_spline.
    const std::vector<float>& out_tangents() const { return m_out_tangents; }
    void set_out_tangents(std::vector<float> out_tangents);

    /// Returns the number of components per keyframe value.
    size_t components_per_value() const;

    /// Returns the number of keyframes.
    size_t keyframe_count() const;

    float evaluate_float(float t, Accessor* accessor = nullptr) const;
    float3 evaluate_float3(float t, Accessor* accessor = nullptr) const;
    quatf evaluate_quatf(float t, Accessor* accessor = nullptr) const;

private:
    FALCOR_NON_COPYABLE_AND_MOVABLE(AnimationChannel);

    friend class Animation;

    Animation* m_animation{nullptr};
    AnimationValueType m_value_type = AnimationValueType::float3;
    AnimationInterpolationMode m_interpolation_mode = AnimationInterpolationMode::linear;
    std::vector<float> m_times;
    std::vector<float> m_values;
    std::vector<float> m_in_tangents;
    std::vector<float> m_out_tangents;

    void notify_changed();

    /// Default accessor used when no external one is provided.
    mutable Accessor m_accessor;

    /// Find the keyframe segment index for time t, using the accessor as a starting hint.
    size_t find_keyframe_index(float t, Accessor& accessor) const;

    /// Evaluate N-component channel at time t.
    /// IsQuat enables quaternion-specific behavior (slerp, hemisphere alignment, normalization).
    template<size_t N, bool IsQuat>
    void evaluate_impl(float t, float* out, Accessor& accessor) const;
};

/// A collection of animation channels.
class FALCOR_API Animation : public SceneObject {
    FALCOR_SCENE_OBJECT(Animation, SceneObject)
public:
    /// Dirty flags for Animation.
    enum class DirtyFlags : uint32_t {
        none = 0,
        data = (1u << 0),
        // Shared SceneObject::DirtyFlags
        added = (1u << 30),
        removed = (1u << 31),
    };

    virtual SceneObjectKind kind() const override { return SceneObjectKind::animation; }

    void remove() override;

    ~Animation() override;

    /// Channels owned by this animation. References retain the channel objects
    /// without copying keyframe arrays, and can outlive this animation.
    /// Replacing the collection attaches the incoming objects and detaches removed
    /// channels. Null channels or channels owned by another Animation are rejected
    /// without changing the collection or ownership. Reassigning this collection is valid.
    /// Replacement requests reevaluation on the next Scene::update(), even at unchanged time.
    const std::vector<ref<AnimationChannel>>& channels() const { return m_channels; }
    void set_channels(std::vector<ref<AnimationChannel>> channels);

    /// Get channel by index. Returns nullptr if index is out of range.
    const AnimationChannel* channel(AnimationChannelIndex index) const
    {
        if (index >= 0 && static_cast<size_t>(index) < m_channels.size())
            return m_channels[index].get();
        return nullptr;
    }

    /// The keyframe interval across all channels in seconds; invalid when there are no keys.
    AnimationTimeRange key_range() const;

    /// Non-owning bindings referencing this animation. Updated by TransformAnimation::set_animation()
    /// and component removal. The view is invalidated when bindings change.
    std::span<TransformAnimation* const> transform_animations() const { return m_transform_animations; }

    /// Reflect this class.
    template<reflection::ClassReflector R>
    static void reflect(R& r)
    {
        FALCOR_UNUSED(r);
    }

private:
    friend class AnimationChannel;
    friend class TransformAnimation;

    void mark_dirty_data();

    std::vector<ref<AnimationChannel>> m_channels;
    std::vector<TransformAnimation*> m_transform_animations;
};

FALCOR_ENUM_CLASS_OPERATORS(Animation::DirtyFlags);

} // namespace falcor
