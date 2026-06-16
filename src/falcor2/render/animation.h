// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/render/scene_object.h"
#include "falcor2/importers/importer_types.h"
#include "falcor2/core/macros.h"
#include "falcor2/core/types.h"

#include <vector>
#include <string>
#include <cstdint>

namespace falcor {

/// Index type for referencing animation channels. -1 means no channel.
using AnimationChannelIndex = int;

/// A single animation channel storing typed keyframe data.
class FALCOR_API AnimationChannel {
public:
    /// Accessor caches the last queried keyframe index for sequential playback optimization.
    /// Use the default (nullptr) for single-threaded evaluation, or provide an external
    /// Accessor for thread-safe concurrent evaluation of the same channel.
    struct Accessor {
        size_t last_index = 0;
    };

    /// Value type of keyframe data.
    AnimationValueType value_type = AnimationValueType::float3;
    /// Interpolation mode.
    AnimationInterpolationMode interpolation_mode = AnimationInterpolationMode::linear;
    /// Keyframe timestamps in seconds.
    std::vector<float> times;
    /// Keyframe values (flat array: 1/3/4 floats per key depending on value_type).
    std::vector<float> values;
    /// In-tangent per keyframe (same layout as values), only populated for cubic_spline.
    std::vector<float> in_tangents;
    /// Out-tangent per keyframe (same layout as values), only populated for cubic_spline.
    std::vector<float> out_tangents;

    /// Returns the number of components per keyframe value.
    size_t components_per_value() const;

    /// Returns the number of keyframes.
    size_t keyframe_count() const;

    float evaluate_float(float t, Accessor* accessor = nullptr) const;
    float3 evaluate_float3(float t, Accessor* accessor = nullptr) const;
    quatf evaluate_quatf(float t, Accessor* accessor = nullptr) const;

private:
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
        // Shared SceneObject::DirtyFlags
        added = (1u << 30),
        removed = (1u << 31),
    };

    virtual SceneObjectKind kind() const override { return SceneObjectKind::animation; }

    /// Flat storage of all animation channels.
    std::vector<AnimationChannel> channels;

    /// Get channel by index. Returns nullptr if index is out of range.
    const AnimationChannel* channel(AnimationChannelIndex index) const
    {
        if (index >= 0 && static_cast<size_t>(index) < channels.size())
            return &channels[index];
        return nullptr;
    }

    /// Returns the duration (max keyframe time across all channels).
    float duration() const;

    /// Reflect this class.
    template<reflection::ClassReflector R>
    static void reflect(R& r)
    {
        FALCOR_UNUSED(r);
    }
};

FALCOR_ENUM_CLASS_OPERATORS(Animation::DirtyFlags);

} // namespace falcor
