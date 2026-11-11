// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "animation.h"

#include "falcor2/render/scene.h"

#include <sgl/math/quaternion_math.h>
#include <sgl/math/vector_math.h>

#include <algorithm>
#include <cmath>

namespace falcor {

// ----------------------------------------------------------------------------
// AnimationChannel
// ----------------------------------------------------------------------------

void AnimationChannel::set_value_type(AnimationValueType value_type)
{
    m_value_type = value_type;
    notify_changed();
}

void AnimationChannel::set_interpolation_mode(AnimationInterpolationMode interpolation_mode)
{
    m_interpolation_mode = interpolation_mode;
    notify_changed();
}

void AnimationChannel::set_times(std::vector<float> times)
{
    m_times = std::move(times);
    m_accessor = {};
    notify_changed();
}

void AnimationChannel::set_values(std::vector<float> values)
{
    m_values = std::move(values);
    notify_changed();
}

void AnimationChannel::set_in_tangents(std::vector<float> in_tangents)
{
    m_in_tangents = std::move(in_tangents);
    notify_changed();
}

void AnimationChannel::set_out_tangents(std::vector<float> out_tangents)
{
    m_out_tangents = std::move(out_tangents);
    notify_changed();
}

void AnimationChannel::notify_changed()
{
    if (m_animation)
        m_animation->mark_dirty_data();
}

size_t AnimationChannel::components_per_value() const
{
    switch (m_value_type) {
    case AnimationValueType::float_:
        return 1;
    case AnimationValueType::float3:
        return 3;
    case AnimationValueType::quatf:
        return 4;
    }
    return 0;
}

size_t AnimationChannel::keyframe_count() const
{
    return m_times.size();
}

size_t AnimationChannel::find_keyframe_index(float t, Accessor& accessor) const
{
    const size_t count = m_times.size();
    if (count <= 1)
        return 0;

    const size_t max_idx = count - 2;

    // Probe cached index first (common case: sequential playback).
    if (accessor.last_index <= max_idx) {
        if (t >= m_times[accessor.last_index]) {
            // Check if still in the same segment.
            if (accessor.last_index == max_idx || t < m_times[accessor.last_index + 1])
                return accessor.last_index;
            // Check next segment (sequential advance).
            if (accessor.last_index + 1 <= max_idx && t >= m_times[accessor.last_index + 1]
                && (accessor.last_index + 1 == max_idx || t < m_times[accessor.last_index + 2])) {
                accessor.last_index = accessor.last_index + 1;
                return accessor.last_index;
            }
        }
    }

    // Fallback: binary search.
    auto it = std::upper_bound(m_times.begin(), m_times.end(), t);
    if (it == m_times.begin()) {
        accessor.last_index = 0;
    } else if (it == m_times.end()) {
        accessor.last_index = max_idx;
    } else {
        accessor.last_index = static_cast<size_t>(it - m_times.begin()) - 1;
    }
    return accessor.last_index;
}

/// Hermite spline evaluation for N-component float arrays.
template<size_t N>
static void hermite(const float* p0, const float* m0, const float* p1, const float* m1, float t, float* out)
{
    float t2 = t * t;
    float t3 = t2 * t;
    float h00 = 2.f * t3 - 3.f * t2 + 1.f;
    float h10 = t3 - 2.f * t2 + t;
    float h01 = -2.f * t3 + 3.f * t2;
    float h11 = t3 - t2;
    for (size_t i = 0; i < N; ++i)
        out[i] = h00 * p0[i] + h10 * m0[i] + h01 * p1[i] + h11 * m1[i];
}

// ----------------------------------------------------------------------------
// Unified channel evaluation (N=1 float, N=3 float3, N=4+IsQuat quatf)
// ----------------------------------------------------------------------------

/// Reinterpret a float array as a quaternion reference.
static const quatf& as_quat(const float* p)
{
    return reinterpret_cast<const quatf&>(*p);
}
static quatf& as_quat(float* p)
{
    return reinterpret_cast<quatf&>(*p);
}

/// Hemisphere-align quaternion src relative to ref. Returns src or -src.
static quatf align_quat(const quatf& ref, const quatf& src)
{
    return (dot(ref, src) < 0.f) ? -src : src;
}

template<size_t N, bool IsQuat>
void AnimationChannel::evaluate_impl(float t, float* out, Accessor& accessor) const
{
    static_assert(!IsQuat || N == 4, "IsQuat requires N == 4");

    const size_t count = keyframe_count();

    if (count == 0)
        return;

    // Clamp to range.
    if (count == 1 || t <= m_times.front()) {
        for (size_t i = 0; i < N; ++i)
            out[i] = m_values[i];
        return;
    }
    if (t >= m_times.back()) {
        const size_t last = (count - 1) * N;
        for (size_t i = 0; i < N; ++i)
            out[i] = m_values[last + i];
        return;
    }

    const size_t idx = find_keyframe_index(t, accessor);
    const float t0 = m_times[idx];
    const float t1 = m_times[idx + 1];
    const float dt = t1 - t0;
    const float alpha = (dt > 0.f) ? (t - t0) / dt : 0.f;

    const float* v0 = &m_values[idx * N];
    const float* v1 = &m_values[(idx + 1) * N];

    // For quaternions, hemisphere-align v1 relative to v0 for all interpolation modes.
    [[maybe_unused]] quatf q1a;
    if constexpr (IsQuat)
        q1a = align_quat(as_quat(v0), as_quat(v1));
    // Pointer to the (possibly aligned) second keyframe value.
    const float* w1 = IsQuat ? &q1a.x : v1;

    switch (m_interpolation_mode) {
    case AnimationInterpolationMode::step:
        for (size_t i = 0; i < N; ++i)
            out[i] = v0[i];
        break;

    case AnimationInterpolationMode::linear:
        if constexpr (IsQuat) {
            as_quat(out) = sgl::math::slerp(as_quat(v0), as_quat(v1), alpha);
        } else {
            for (size_t i = 0; i < N; ++i)
                out[i] = v0[i] + (v1[i] - v0[i]) * alpha;
        }
        break;

    case AnimationInterpolationMode::cubic_hermite: {
        float m0[N], m1[N];

        if (idx == 0) {
            for (size_t i = 0; i < N; ++i)
                m0[i] = w1[i] - v0[i];
        } else {
            const float dt_span = m_times[idx + 1] - m_times[idx - 1];
            const float* v_prev = &m_values[(idx - 1) * N];
            if constexpr (IsQuat) {
                quatf q_prev = align_quat(as_quat(v0), as_quat(v_prev));
                for (size_t i = 0; i < N; ++i)
                    m0[i] = (w1[i] - (&q_prev.x)[i]) / dt_span * dt;
            } else {
                for (size_t i = 0; i < N; ++i)
                    m0[i] = (w1[i] - v_prev[i]) / dt_span * dt;
            }
        }

        if (idx + 2 >= count) {
            for (size_t i = 0; i < N; ++i)
                m1[i] = w1[i] - v0[i];
        } else {
            const float dt_span = m_times[idx + 2] - m_times[idx];
            const float* v_next = &m_values[(idx + 2) * N];
            if constexpr (IsQuat) {
                quatf q_next = align_quat(as_quat(v0), as_quat(v_next));
                for (size_t i = 0; i < N; ++i)
                    m1[i] = ((&q_next.x)[i] - v0[i]) / dt_span * dt;
            } else {
                for (size_t i = 0; i < N; ++i)
                    m1[i] = (v_next[i] - v0[i]) / dt_span * dt;
            }
        }

        hermite<N>(v0, m0, w1, m1, alpha, out);
        if constexpr (IsQuat)
            as_quat(out) = sgl::math::normalize(as_quat(out));
        break;
    }

    case AnimationInterpolationMode::cubic_spline: {
        const float* out_tan_0 = &m_out_tangents[idx * N];
        const float* in_tan_1 = &m_in_tangents[(idx + 1) * N];

        float m0[N], m1[N];
        for (size_t i = 0; i < N; ++i) {
            m0[i] = out_tan_0[i] * dt;
            m1[i] = in_tan_1[i] * dt;
        }

        hermite<N>(v0, m0, v1, m1, alpha, out);
        if constexpr (IsQuat)
            as_quat(out) = sgl::math::normalize(as_quat(out));
        break;
    }
    }
}

// Explicit instantiations.
template void AnimationChannel::evaluate_impl<1, false>(float t, float* out, Accessor& accessor) const;
template void AnimationChannel::evaluate_impl<3, false>(float t, float* out, Accessor& accessor) const;
template void AnimationChannel::evaluate_impl<4, true>(float t, float* out, Accessor& accessor) const;

// ----------------------------------------------------------------------------
// Typed public evaluation methods
// ----------------------------------------------------------------------------

float AnimationChannel::evaluate_float(float t, Accessor* accessor) const
{
    FALCOR_CHECK(m_value_type == AnimationValueType::float_, "Channel value type is not float");
    float value = 0.f;
    evaluate_impl<1, false>(t, &value, accessor ? *accessor : m_accessor);
    return value;
}

float3 AnimationChannel::evaluate_float3(float t, Accessor* accessor) const
{
    FALCOR_CHECK(m_value_type == AnimationValueType::float3, "Channel value type is not float3");
    float3 value = {};
    evaluate_impl<3, false>(t, &value.x, accessor ? *accessor : m_accessor);
    return value;
}

quatf AnimationChannel::evaluate_quatf(float t, Accessor* accessor) const
{
    FALCOR_CHECK(m_value_type == AnimationValueType::quatf, "Channel value type is not quatf");
    quatf value = {};
    evaluate_impl<4, true>(t, &value.x, accessor ? *accessor : m_accessor);
    return value;
}

// ----------------------------------------------------------------------------
// Animation
// ----------------------------------------------------------------------------

void Animation::remove()
{
    if (!is_valid())
        return;
    SceneObject::remove();
    if (scene())
        scene()->_mark_animation_removed(this);
}

Animation::~Animation()
{
    for (const auto& channel : m_channels)
        channel->m_animation = nullptr;
}

void Animation::set_channels(std::vector<ref<AnimationChannel>> channels)
{
    for (const auto& channel : channels) {
        FALCOR_CHECK(channel, "Animation channel must not be null");
        FALCOR_CHECK(
            !channel->m_animation || channel->m_animation == this,
            "Animation channel already belongs to another Animation"
        );
    }

    for (const auto& channel : m_channels)
        channel->m_animation = nullptr;
    m_channels = std::move(channels);
    for (const auto& channel : m_channels)
        channel->m_animation = this;
    mark_dirty_data();
}

void Animation::mark_dirty_data()
{
    if (m_scene)
        m_scene->_mark_animation_dirty(this, DirtyFlags::data);
}

AnimationTimeRange Animation::key_range() const
{
    AnimationTimeRange range;
    for (const auto& channel : m_channels) {
        if (!channel->times().empty())
            range.expand({channel->times().front(), channel->times().back()});
    }
    return range;
}

FALCOR_SCENE_REGISTER_CLASS(Animation, Animation);

} // namespace falcor
