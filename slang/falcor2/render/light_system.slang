// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing render;

public struct LightSystem
{
    /// Buffer containing all lights.
    public StructuredBuffer<LightData> light_data;
    /// Total number of lights.
    public uint light_count;
    /// Total number of analytic lights.
    public uint analytic_light_count;
    /// Light ID of environment light (or LightID::invalid if none is available.)
    public LightID environment_light_id;
    /// Number of environment lights, stored contiguously starting at environment_light_id.
    public uint environment_light_count;
    /// Alias table used for selecting analytic lights.
    public AliasTable1D analytic_light_selection_distribution;
    /// Alias table used for selecting environment lights by emitted power.
    public AliasTable1D environment_light_selection_distribution;

    /// Returns true if the scene has analytic lights.
    public bool has_any_analytic_lights() { return analytic_light_count > 0; }

    /// Returns true if the scene has one or more environment lights.
    public bool has_environment_lights() { return environment_light_count > 0; }

    /// Get a light.
    public ILight get_light(LightID id)
    {
        LightData data = light_data[uint(id)];
        return createDynamicObject<ILight>(data.header.type, data);
    }

    /// Get an environment light by its zero-based environment-light index.
    public ILight get_environment_light(uint index) { return get_light(LightID(uint(environment_light_id) + index)); }

    /// Evaluate the sum of all environment lights along a direction.
    public float3
    eval_environment_direction<Interaction : IInteraction>(const Interaction ref, const DirectionSample sample)
    {
        float3 radiance = {};
        for (uint index = 0; index < environment_light_count; ++index)
            radiance += get_environment_light(index).eval_direction(ref, sample);
        return radiance;
    }
};
