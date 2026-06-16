// SPDX-License-Identifier: Apache-2.0

implementing scene;

public enum class EmissiveTriangleID : uint { invalid = 0xffffffff };

public struct EmissiveGeometryInstance
{
    public uint triangle_start;
    public uint triangle_count;
}

public struct EmissiveTriangle
{
    /// Position in world-space.
    public float3 pos_ws[3];
    /// Texture coordinates.
    public float2 uv[3];
    /// Normal in world-space.
    public float3 normal_ws;
    /// Area in world-space.
    public float area_ws;
    /// Geometry instance this triangle belongs to.
    public GeometryInstanceID geometry_instance;
    /// Material this triangle belongs to.
    public MaterialID material;

    /// Returns the interpolated position.
    /// @param bary Barycentric coordinates.
    /// @return Interpolated position in world space.
    public float3 get_pos_ws(float3 bary) { return pos_ws[0] * bary[0] + pos_ws[1] * bary[1] + pos_ws[2] * bary[2]; }

    /// Returns the interpolated texture coordinate.
    /// @param bary Barycentric coordinates.
    /// @return Interpolated texture coordinate.
    public float2 get_uv(float3 bary) { return uv[0] * bary[0] + uv[1] * bary[1] + uv[2] * bary[2]; }
}

public struct EmissiveFlux
{
    public float flux;
    // public float3 average_radiance;
}

public struct EmissiveTriangles
{
    /// Total number of emissive triangles.
    public uint triangle_count;

    public StructuredBuffer<EmissiveGeometryInstance> geometry_instances;
    public StructuredBuffer<EmissiveTriangle> triangles;
    public StructuredBuffer<EmissiveFlux> fluxes;

    public EmissiveGeometryInstance get_geometry_instance(GeometryInstanceID id)
    {
        return geometry_instances[uint(id)];
    }

    public EmissiveTriangleID get_triangle_id(GeometryInstanceID geometry_instance, uint primitive_index)
    {
        return EmissiveTriangleID(get_geometry_instance(geometry_instance).triangle_start + primitive_index);
    }

    public EmissiveTriangle get_triangle(EmissiveTriangleID id) { return triangles[uint(id)]; }

    public EmissiveFlux get_flux(EmissiveTriangleID id) { return fluxes[uint(id)]; }
}

[shader("compute")]
public void setup_emissive_geometry_triangle_count(
    uint tid: SV_DispatchThreadID,
    uniform uint geometry_instance_count,
    uniform RWStructuredBuffer<EmissiveGeometryInstance> emissive_geometry_instances
)
{
    if (tid >= geometry_instance_count)
        return;
    EmissiveGeometryInstance instance = {};
    const GeometryInstance geometry_instance = g_scene.get_geometry_instance(GeometryInstanceID(tid));

    float2 uvs[UV_COUNT];
    for (int i = 0; i < UV_COUNT; i++)
        uvs[i] = float2(0.0, 0.0);

    const Material material = g_scene.get_material(geometry_instance.material, uvs);
    if (any(material.emission > 0)) {
        instance.triangle_count = geometry_instance.index_count / 3;
    }
    emissive_geometry_instances[tid] = instance;
}

/// Setup the emissive triangles.
/// Fills in both the geometry_instance and material fields.
[shader("compute")]
public void setup_emissive_triangles(
    uint2 tid: SV_DispatchThreadID,
    uniform uint geometry_instance_count,
    uniform StructuredBuffer<EmissiveGeometryInstance> emissive_geometry_instances,
    uniform RWStructuredBuffer<EmissiveTriangle> emissive_triangles
)
{
    if (tid.x >= geometry_instance_count)
        return;
    EmissiveGeometryInstance instance = emissive_geometry_instances[tid.x];
    uint triangle_index = tid.y;
    if (instance.triangle_count == 0 || triangle_index >= instance.triangle_count)
        return;
    EmissiveTriangle triangle = {};
    triangle.geometry_instance = GeometryInstanceID(tid.x);
    triangle.material = g_scene.get_geometry_instance(triangle.geometry_instance).material;
    emissive_triangles[instance.triangle_start + triangle_index] = triangle;
}

[shader("compute")]
public void update_emissive_triangles(
    uint tid: SV_DispatchThreadID,
    uniform uint triangle_count,
    uniform StructuredBuffer<EmissiveGeometryInstance> emissive_geometry_instances,
    uniform RWStructuredBuffer<EmissiveTriangle> emissive_triangles,
    uniform RWStructuredBuffer<EmissiveFlux> emissive_fluxes
)
{
    if (tid >= triangle_count)
        return;
    EmissiveTriangle triangle = emissive_triangles[tid];
    EmissiveGeometryInstance instance = emissive_geometry_instances[uint(triangle.geometry_instance)];

    uint triangle_index = tid - instance.triangle_start;
    Vertex vertices[3] = g_scene.get_triangle_vertices(triangle.geometry_instance, triangle_index);
    float4x4 transform = g_scene.get_transform(triangle.geometry_instance);

    triangle.pos_ws[0] = mul(transform, float4(vertices[0].position, 1.0)).xyz;
    triangle.pos_ws[1] = mul(transform, float4(vertices[1].position, 1.0)).xyz;
    triangle.pos_ws[2] = mul(transform, float4(vertices[2].position, 1.0)).xyz;

    float3 edge0 = triangle.pos_ws[1] - triangle.pos_ws[0];
    float3 edge1 = triangle.pos_ws[2] - triangle.pos_ws[0];
    triangle.normal_ws = normalize(cross(edge0, edge1));

    float flux;
    if (any(isnan(triangle.normal_ws))) {
        triangle.area_ws = 0.0;
        triangle.normal_ws = float3(0.0, 0.0, 0.0);
        flux = 0;
    } else {
        triangle.area_ws = length(cross(edge0, edge1)) / 2;
        float2 uvs[UV_COUNT];
        for (int i = 0; i < UV_COUNT; i++)
            uvs[i] = float2(0.0, 0.0);
        float3 emission = g_scene.get_material(triangle.geometry_instance, uvs).emission;
        flux = triangle.area_ws * dot(emission, float3(0.2126, 0.7152, 0.0722));
    }
    emissive_triangles[tid] = triangle;
    emissive_fluxes[tid].flux = flux;
}
