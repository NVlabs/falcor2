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
    /// Alias table used for selecting analytic lights.
    public AliasTable1D analytic_light_selection_distribution;

    /// Returns true if the scene has analytic lights.
    public bool has_any_analytic_lights() { return analytic_light_count > 0; }

    /// Returns true if the scene has an environment light.
    public bool has_environment_light() { return environment_light_id != LightID::invalid; }

    /// Get a light.
    public ILight get_light(LightID id)
    {
        LightData data = light_data[uint(id)];
        return createDynamicObject<ILight>(data.header.type, data);
    }
};
