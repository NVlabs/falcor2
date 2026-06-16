// SPDX-License-Identifier: Apache-2.0

#include "animation.h"

#include <sgl/math/quaternion_math.h>
#include <sgl/math/vector_math.h>

#include <algorithm>
#include <cmath>

namespace falcor {

// ----------------------------------------------------------------------------
// AnimationChannel
// ----------------------------------------------------------------------------

size_t AnimationChannel::components_per_value() const
{
    switch (value_type) {
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
    return times.size();
}

size_t AnimationChannel::find_keyframe_index(float t, Accessor& accessor) const
{
    const size_t count = times.size();
    if (count <= 1)
        return 0;

    const size_t max_idx = count - 2;

    // Probe cached index first (common case: sequential playback).
    if (accessor.last_index <= max_idx) {
        if (t >= times[accessor.last_index]) {
            // Check if still in the same segment.
            if (accessor.last_index == max_idx || t < times[accessor.last_index + 1])
                return accessor.last_index;
            // Check next segment (sequential advance).
            if (accessor.last_index + 1 <= max_idx && t >= times[accessor.last_index + 1]
                && (accessor.last_index + 1 == max_idx || t < times[accessor.last_index + 2])) {
                accessor.last_index = accessor.last_index + 1;
                return accessor.last_index;
            }
        }
    }

    // Fallback: binary search.
    auto it = std::upper_bound(times.begin(), times.end(), t);
    if (it == times.begin()) {
        accessor.last_index = 0;
    } else if (it == times.end()) {
        accessor.last_index = max_idx;
    } else {
        accessor.last_index = static_cast<size_t>(it - times.begin()) - 1;
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
    if (count == 1 || t <= times.front()) {
        for (size_t i = 0; i < N; ++i)
            out[i] = values[i];
        return;
    }
    if (t >= times.back()) {
        const size_t last = (count - 1) * N;
        for (size_t i = 0; i < N; ++i)
            out[i] = values[last + i];
        return;
    }

    const size_t idx = find_keyframe_index(t, accessor);
    const float t0 = times[idx];
    const float t1 = times[idx + 1];
    const float dt = t1 - t0;
    const float alpha = (dt > 0.f) ? (t - t0) / dt : 0.f;

    const float* v0 = &values[idx * N];
    const float* v1 = &values[(idx + 1) * N];

    // For quaternions, hemisphere-align v1 relative to v0 for all interpolation modes.
    [[maybe_unused]] quatf q1a;
    if constexpr (IsQuat)
        q1a = align_quat(as_quat(v0), as_quat(v1));
    // Pointer to the (possibly aligned) second keyframe value.
    const float* w1 = IsQuat ? &q1a.x : v1;

    switch (interpolation_mode) {
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
            const float dt_span = times[idx + 1] - times[idx - 1];
            const float* v_prev = &values[(idx - 1) * N];
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
            const float dt_span = times[idx + 2] - times[idx];
            const float* v_next = &values[(idx + 2) * N];
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
        const float* out_tan_0 = &out_tangents[idx * N];
        const float* in_tan_1 = &in_tangents[(idx + 1) * N];

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
    FALCOR_CHECK(value_type == AnimationValueType::float_, "Channel value type is not float");
    float value = 0.f;
    evaluate_impl<1, false>(t, &value, accessor ? *accessor : m_accessor);
    return value;
}

float3 AnimationChannel::evaluate_float3(float t, Accessor* accessor) const
{
    FALCOR_CHECK(value_type == AnimationValueType::float3, "Channel value type is not float3");
    float3 value = {};
    evaluate_impl<3, false>(t, &value.x, accessor ? *accessor : m_accessor);
    return value;
}

quatf AnimationChannel::evaluate_quatf(float t, Accessor* accessor) const
{
    FALCOR_CHECK(value_type == AnimationValueType::quatf, "Channel value type is not quatf");
    quatf value = {};
    evaluate_impl<4, true>(t, &value.x, accessor ? *accessor : m_accessor);
    return value;
}

// ----------------------------------------------------------------------------
// Animation
// ----------------------------------------------------------------------------

float Animation::duration() const
{
    float max_time = 0.f;
    for (const auto& channel : channels) {
        if (!channel.times.empty())
            max_time = std::max(max_time, channel.times.back());
    }
    return max_time;
}

FALCOR_SCENE_REGISTER_CLASS(Animation, Animation);

} // namespace falcor
