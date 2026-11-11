// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/core/enum.h"

namespace falcor {

/// Texture coordinate origin convention for authored UVs.
enum class UVOrigin {
    upper_left,
    lower_left,
};
SGL_ENUM_INFO(
    UVOrigin,
    {
        {UVOrigin::upper_left, "upper_left"},
        {UVOrigin::lower_left, "lower_left"},
    }
);
SGL_ENUM_REGISTER(UVOrigin);

/// Indexing mode for curve line segments.
enum class CurveIndexingMode {
    /// Each segment is defined by a pair of indices: indices[2k], indices[2k+1].
    list,
    /// Each segment is defined by a single index: positions[indices[k]] and positions[indices[k]+1].
    successive,
};
SGL_ENUM_INFO(
    CurveIndexingMode,
    {
        {CurveIndexingMode::list, "list"},
        {CurveIndexingMode::successive, "successive"},
    }
);
SGL_ENUM_REGISTER(CurveIndexingMode);

/// Determines component count and interpolation method for animation keyframes.
enum class AnimationValueType {
    /// Scalar float (lerp)
    float_,
    /// 3-component float vector(lerp)
    float3,
    /// 4-component float quaternion (slerp)
    quatf,
};
SGL_ENUM_INFO(
    AnimationValueType,
    {
        {AnimationValueType::float_, "float"},
        {AnimationValueType::float3, "float3"},
        {AnimationValueType::quatf, "quatf"},
    }
);
SGL_ENUM_REGISTER(AnimationValueType);

/// Interpolation mode for animation keyframes.
enum class AnimationInterpolationMode {
    /// No interpolation, use left keyframe value
    step,
    /// Linear interpolation (lerp for float/float3, slerp for quatf)
    linear,
    /// Catmull-Rom auto-tangents
    cubic_hermite,
    /// Explicit in/out tangents per keyframe
    cubic_spline,
};
SGL_ENUM_INFO(
    AnimationInterpolationMode,
    {
        {AnimationInterpolationMode::step, "step"},
        {AnimationInterpolationMode::linear, "linear"},
        {AnimationInterpolationMode::cubic_hermite, "cubic_hermite"},
        {AnimationInterpolationMode::cubic_spline, "cubic_spline"},
    }
);
SGL_ENUM_REGISTER(AnimationInterpolationMode);

/// Alpha modes for glTF-style materials.
enum class AlphaMode {
    /// The alpha value is ignored and the rendered output is fully opaque.
    opaque = 0,
    /// Alpha values below the cutoff are discarded.
    mask = 1,
    /// Alpha values are evaluated stochastically during intersection traversal.
    blend = 2,
};
SGL_ENUM_INFO(
    AlphaMode,
    {
        {AlphaMode::opaque, "opaque"},
        {AlphaMode::mask, "mask"},
        {AlphaMode::blend, "blend"},
    }
);
SGL_ENUM_REGISTER(AlphaMode);

/// Alignment of an environment map's top pole in the owning entity's local space.
enum class EnvMapPoleAxis {
    y,
    z,
};
SGL_ENUM_INFO(
    EnvMapPoleAxis,
    {
        {EnvMapPoleAxis::y, "y"},
        {EnvMapPoleAxis::z, "z"},
    }
);
SGL_ENUM_REGISTER(EnvMapPoleAxis);

} // namespace falcor
