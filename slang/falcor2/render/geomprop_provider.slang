// SPDX-License-Identifier: Apache-2.0

implementing render;

// Spaces used by MaterialX transform nodes. The default provider treats them as
// identical, but scene integrations can override the extern provider to supply
// object/world/model transforms without changing generated material code.
public enum class GeomPropSpace : uint {
    model = 0,
    object = 1,
    world = 2,
};

static const float4x4 k_geomprop_identity4x4 = { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 };

public float4x4 geomprop_identity4x4()
{
    return k_geomprop_identity4x4;
}

public interface IGeomPropProvider
{
    public static int get_int(SurfaceInteraction si, uint id);
    public static bool get_bool(SurfaceInteraction si, uint id);
    public static float get_float(SurfaceInteraction si, uint id);
    public static float2 get_float2(SurfaceInteraction si, uint id);
    public static float3 get_float3(SurfaceInteraction si, uint id);
    public static float4 get_float4(SurfaceInteraction si, uint id);
    public static float4x4 get_space_transform(SurfaceInteraction si, GeomPropSpace to_space, GeomPropSpace from_space);
};

public struct DummyGeomPropProvider : IGeomPropProvider
{
    public static int get_int(SurfaceInteraction si, uint id) { return 0; }
    public static bool get_bool(SurfaceInteraction si, uint id) { return false; }
    public static float get_float(SurfaceInteraction si, uint id) { return 0.0; }
    public static float2 get_float2(SurfaceInteraction si, uint id) { return si.uv; }
    public static float3 get_float3(SurfaceInteraction si, uint id) { return float3(si.uv, 0.0); }
    public static float4 get_float4(SurfaceInteraction si, uint id) { return float4(si.uv, 0.0, 0.0); }
    public static float4x4 get_space_transform(SurfaceInteraction si, GeomPropSpace to_space, GeomPropSpace from_space)
    {
        return geomprop_identity4x4();
    }
};

public extern struct GeomPropProvider : IGeomPropProvider = DummyGeomPropProvider;
