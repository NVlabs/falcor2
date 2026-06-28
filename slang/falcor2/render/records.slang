// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing render;

/// Base interface for sample records.
public interface IBaseSample
{
    /// Probability density of the sample.
    public property float pdf;
};

/// Interface for position sample records.
public interface IPositionSample : IBaseSample
{
    /// Sampled position in world-space.
    public property float3 pos_ws;
    /// Surface normal at the sampled position in world-space.
    public property float3 normal_ws;
    /// UV coordinates at the sampled position.
    public property float2 uv;
    /// True if sample was drawn from a dirac delta distribution.
    public property bool delta;
};

/// Emitter types a direction sample can be associated with.
public enum class EmitterType : uint {
    unknown,
    analytic,
    environment,
    emissive_triangle,
};

/// Interface for direction sample records.
public interface IDirectionSample : IPositionSample
{
    /// Sampled direction in world-space (normalized).
    public property float3 dir_ws;
    /// Distance to the sampled position.
    public property float distance;
    /// Emitter type the sample is associated with.
    public property EmitterType emitter_type;
    /// Primitive ID of the sampled emitter.
    public property uint primitive_id;
};

/// Sample record base struct.
/// Sample record for position samples (w.r.t. area measure).
public struct PositionSample : IPositionSample
{
    /// Probability density w.r.t. area measure at the sampled position.
    public float pdf = {};
    /// Sampled position in world-space.
    public float3 pos_ws = {};
    /// Surface normal at the sampled position in world-space.
    public float3 normal_ws = {};
    /// UV coordinates at the sampled position.
    public float2 uv = {};
    /// True if sample was drawn from a dirac delta distribution.
    public bool delta = {};
};

/// Sample record for direction samples (w.r.t. solid angle measure from a shading point).
public struct DirectionSample : IDirectionSample
{
    /// Probability density w.r.t. solid angle measure from the shading point.
    public float pdf = {};
    /// Sampled position in world-space.
    public float3 pos_ws = {};
    /// Surface normal at the sampled position in world-space.
    public float3 normal_ws = {};
    /// UV coordinates at the sampled position.
    public float2 uv = {};
    /// True if sample was drawn from a dirac delta distribution.
    public bool delta = {};
    /// Sampled direction in world-space (normalized).
    public float3 dir_ws = {};
    /// Distance to the sampled position.
    public float distance = {};
    /// Emitter type the sample is associated with.
    public EmitterType emitter_type = {};
    /// Primitive ID of the sampled emitter.
    public uint primitive_id = {};
};
