// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//
// Scatter-mode policies for Mx139 dielectric-like MaterialX microfacet BSDFs.
//
// MaterialX exposes scatter_mode as R, T, or RT. The runtime policy preserves
// the current generated behavior, where the enum is folded into reflection and
// transmission tint values. Static policies are selected by codegen only when
// the enum value is proven constant; they remove the enum-dependent branch while
// still respecting zero tint values.

implementing mx139;

namespace mx139 {

public struct MxScatterTints
{
    public float3 reflection;
    public float3 transmission;
};

public MxScatterTints split_scatter_tints(int scatter_mode, float3 tint)
{
    MxScatterTints result;
    result.reflection = (scatter_mode == 1) ? float3(0.0) : tint;
    result.transmission = (scatter_mode == 0) ? float3(0.0) : tint;
    return result;
}

public interface IMxScatterModePolicy
{
    [ForceInline]
    public static bool dorefl(float3 reflection_tint);
    [ForceInline]
    public static bool dorefr(float3 transmission_tint);
}

public struct MxScatterRuntimeMode : IMxScatterModePolicy
{
    [ForceInline]
    public static bool dorefl(float3 reflection_tint) { return mx_hmax(reflection_tint) > 0.0; }

    [ForceInline]
    public static bool dorefr(float3 transmission_tint) { return mx_hmax(transmission_tint) > 0.0; }
}

public struct MxScatterStaticRMode : IMxScatterModePolicy
{
    [ForceInline]
    public static bool dorefl(float3 reflection_tint) { return MxScatterRuntimeMode::dorefl(reflection_tint); }

    [ForceInline]
    public static bool dorefr(float3 transmission_tint) { return false; }
}

public struct MxScatterStaticTMode : IMxScatterModePolicy
{
    [ForceInline]
    public static bool dorefl(float3 reflection_tint) { return false; }

    [ForceInline]
    public static bool dorefr(float3 transmission_tint) { return MxScatterRuntimeMode::dorefr(transmission_tint); }
}

public struct MxScatterStaticRTMode : IMxScatterModePolicy
{
    [ForceInline]
    public static bool dorefl(float3 reflection_tint) { return MxScatterRuntimeMode::dorefl(reflection_tint); }

    [ForceInline]
    public static bool dorefr(float3 transmission_tint) { return MxScatterRuntimeMode::dorefr(transmission_tint); }
}

} // namespace mx139
