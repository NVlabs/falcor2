// SPDX-License-Identifier: Apache-2.0

implementing scene;

static const uint MAX_TEXTURE_COUNT = 256;

public ParameterBlock<Scene> g_scene;
public uniform TextureID g_debug_gradient_texture;

public static float4 s_debug_gradient;

public struct HitInfo
{
    public bool valid;
    public float2 bary;
    public float t;
    public GeometryInstanceID geometry_instance_id;
    public uint primitive_index;
};

public struct Scene
{
    // TLAS build from instances.
    public RaytracingAccelerationStructure tlas;

    // Geometry instances (combines mesh info and material).
    public StructuredBuffer<GeometryInstance> geometry_instances;

    // Materials
    public StructuredBuffer<MaterialData> materials;

    // 2D float4 textures.
    // Note: the 3 represents the 3rd dimension of the texture.
    public Tensor<float, 3> textures[MAX_TEXTURE_COUNT];
    public AtomicTensor<float, 3> texture_grads[MAX_TEXTURE_COUNT];
    public uint texture_grads_enabled[MAX_TEXTURE_COUNT];

    // Shared vertex and index buffers.
    public StructuredBuffer<Vertex> vertices;
    public StructuredBuffer<uint> indices;

    /// Instance transforms (indexed by GeometryInstanceID).
    public StructuredBuffer<float4x4> transforms;

    /// Inverse transpose instance transforms (indexed by GeometryInstanceID).
    public StructuredBuffer<float4x4> inverse_transpose_transforms;

    // Emissive triangle data used for NEE.
    public EmissiveTriangles emissive_triangles;

    // Environment map.
    public EnvMap env_map;

    public bool has_emissive_triangles() { return emissive_triangles.triangle_count > 0; }
    public bool has_env_map() { return env_map.importance_map_dims.z > 0; }

    public GeometryInstance get_geometry_instance(GeometryInstanceID id) { return geometry_instances[uint(id)]; }

    public float4x4 get_transform(GeometryInstanceID id) { return transforms[uint(id)]; }

    public float4x4 get_inverse_transpose_transform(GeometryInstanceID id)
    {
        return inverse_transpose_transforms[uint(id)];
    }

    public uint3 get_triangle_indices(GeometryInstanceID id, uint primitive_index)
    {
        GeometryInstance instance = get_geometry_instance(id);
        return uint3(
            indices[instance.index_start + primitive_index * 3 + 0],
            indices[instance.index_start + primitive_index * 3 + 1],
            indices[instance.index_start + primitive_index * 3 + 2]
        );
    }

    public Vertex[3] get_triangle_vertices(GeometryInstanceID id, uint primitive_index)
    {
        GeometryInstance instance = get_geometry_instance(id);
        uint3 indices = get_triangle_indices(id, primitive_index);
        return {
            vertices[instance.vertex_start + indices.x],
            vertices[instance.vertex_start + indices.y],
            vertices[instance.vertex_start + indices.z]
        };
    }

    public VertexData get_vertex_data(HitInfo hit_info)
    {
        return get_vertex_data(hit_info.geometry_instance_id, hit_info.primitive_index, hit_info.bary);
    }

    // Takes primitive index + barycentric coordinates and performs
    // interpolation to get the vertex data.
    public VertexData get_vertex_data(GeometryInstanceID geometry_instance_id, uint primitive_index, float2 bary)
    {
        GeometryInstance instance = get_geometry_instance(geometry_instance_id);

        float4x4 transform = get_transform(geometry_instance_id);
        float4x4 inverse_transpose_transform = get_inverse_transpose_transform(geometry_instance_id);

        uint i0 = indices[instance.index_start + primitive_index * 3 + 0];
        uint i1 = indices[instance.index_start + primitive_index * 3 + 1];
        uint i2 = indices[instance.index_start + primitive_index * 3 + 2];
        Vertex v0 = vertices[instance.vertex_start + i0];
        Vertex v1 = vertices[instance.vertex_start + i1];
        Vertex v2 = vertices[instance.vertex_start + i2];

        uint uvchannel = materials[uint(instance.material)].albedo.tex_coord;

        float3 b = float3(1.0 - bary.x - bary.y, bary.x, bary.y);
        float3 pos_ls = b.x * v0.position + b.y * v1.position + b.z * v2.position;
        float3 normal_ls = b.x * v0.normal + b.y * v1.normal + b.z * v2.normal;
        float3 tangent_ls = b.x * v0.tangent + b.y * v1.tangent + b.z * v2.tangent;
        float3 bitangent_ls = b.x * v0.bitangent + b.y * v1.bitangent + b.z * v2.bitangent;
        float3 face_normal_ls = normalize(cross(v1.position - v0.position, v2.position - v0.position));
        float3 color = b.x * v0.color + b.y * v1.color + b.z * v2.color;

        float2 uv[UV_COUNT];
        [ForceUnroll]
        for (int i = 0; i < UV_COUNT; i++)
            uv[i] = b.x * v0.uv[i] + b.y * v1.uv[i] + b.z * v2.uv[i];

        VertexData vertex = {};
        vertex.pos_ws = mul(transform, float4(pos_ls, 1.0)).xyz;
        vertex.normal_ws = normalize(mul(inverse_transpose_transform, float4(normal_ls, 0.0)).xyz);
        vertex.tangent_ws = normalize(mul(transform, float4(tangent_ls, 0.0)).xyz);
        vertex.bitangent_ws = normalize(mul(transform, float4(bitangent_ls, 0.0)).xyz);
        vertex.face_normal_ws = normalize(mul(inverse_transpose_transform, float4(face_normal_ls, 0.0)).xyz);
        vertex.uv = uv;
        vertex.color = color;
        return vertex;
    }

    // Wrap an integer coordinate according to the given wrap mode.
    // max is the size of the dimension being wrapped - i.e. it is an EXCLUSIVE upper bound.
    [Differentiable]
    int wrap(int u, int max, TextureWrapMode wrap_mode)
    {
        if (max <= 0)
            u = 0;
        else if (wrap_mode == TextureWrapMode::ClampToEdge)
            u = clamp(u, 0, max - 1);
        else if (wrap_mode == TextureWrapMode::MirroredRepeat) {
            int period = max * 2;
            u = u % period;
            if (u < 0)
                u += period;
            if (u >= max)
                u = period - u - 1;
        } else { // Repeat
            u = u % max;
            if (u < 0)
                u += max;
        }
        return u;
    }

    // Fetch a single pixel from a texture by texel index.
    [Differentiable, BackwardDerivative(fetch_texture_bwd)]
    float4 fetch_texture(
        TextureID texture,
        int2 texel,
        TextureWrapMode wrap_mode_s = TextureWrapMode::Repeat,
        TextureWrapMode wrap_mode_t = TextureWrapMode::Repeat
    )
    {
        texel.x = wrap(texel.x, textures[uint(texture)].shape[1], wrap_mode_s);
        texel.y = wrap(texel.y, textures[uint(texture)].shape[0], wrap_mode_t);

        float4 val = 0;
        int dim = textures[uint(texture)].shape[2];
        if (dim >= 1)
            val.x = textures[uint(texture)][texel.y, texel.x, 0];
        if (dim >= 2)
            val.y = textures[uint(texture)][texel.y, texel.x, 1];
        if (dim >= 3)
            val.z = textures[uint(texture)][texel.y, texel.x, 2];
        if (dim >= 4)
            val.w = textures[uint(texture)][texel.y, texel.x, 3];

        return val;
    }

    void fetch_texture_bwd(
        TextureID texture,
        int2 texel,
        TextureWrapMode wrap_mode_s,
        TextureWrapMode wrap_mode_t,
        float4 grad
    )
    {
        texel.x = wrap(texel.x, textures[uint(texture)].shape[1], wrap_mode_s);
        texel.y = wrap(texel.y, textures[uint(texture)].shape[0], wrap_mode_t);

        if (texture_grads_enabled[uint(texture)] != 0) {
            int dim = textures[uint(texture)].shape[2];
            if (dim >= 1)
                texture_grads[uint(texture)].add(texel.y, texel.x, 0, grad.x);
            if (dim >= 2)
                texture_grads[uint(texture)].add(texel.y, texel.x, 1, grad.y);
            if (dim >= 3)
                texture_grads[uint(texture)].add(texel.y, texel.x, 2, grad.z);
            if (dim >= 4)
                texture_grads[uint(texture)].add(texel.y, texel.x, 3, grad.w);
        }
        if (uint(texture) == g_debug_gradient_texture) {
            s_debug_gradient.xyz += grad.xyz;
        }
    }

    // Sample a texture using point sampling at the given UV coordinates.
    [Differentiable, PreferRecompute, __NoSideEffect]
    float4 sample_texture_point(
        TextureID texture,
        no_diff float2 uv,
        TextureWrapMode wrap_mode_s = TextureWrapMode::Repeat,
        TextureWrapMode wrap_mode_t = TextureWrapMode::Repeat
    )
    {
        float2 shape = float2(textures[uint(texture)].shape[1], textures[uint(texture)].shape[0]);
        int2 texel = int2(uv * shape);
        return fetch_texture(texture, texel, wrap_mode_s, wrap_mode_t);
    }

    // Sample a texture using bilinear interpolation at the given UV coordinates.
    [Differentiable, PreferRecompute, __NoSideEffect]
    float4 sample_texture_linear(
        TextureID texture,
        no_diff float2 uv,
        TextureWrapMode wrap_mode_s = TextureWrapMode::Repeat,
        TextureWrapMode wrap_mode_t = TextureWrapMode::Repeat
    )
    {
        float2 shape = float2(textures[uint(texture)].shape[1], textures[uint(texture)].shape[0]);
        int2 texel = int2(uv * shape);
        float2 frac = frac(uv * shape);

        float4 c00 = fetch_texture(texture, texel, wrap_mode_s, wrap_mode_t);
        float4 c10 = fetch_texture(texture, texel + int2(1, 0), wrap_mode_s, wrap_mode_t);
        float4 c01 = fetch_texture(texture, texel + int2(0, 1), wrap_mode_s, wrap_mode_t);
        float4 c11 = fetch_texture(texture, texel + int2(1, 1), wrap_mode_s, wrap_mode_t);

        return lerp(lerp(c00, c10, frac.x), lerp(c01, c11, frac.x), frac.y);
    }

    // Sample a texture using bilinear interpolation at the given UV coordinates.
    [Differentiable, PreferRecompute, __NoSideEffect]
    float4 sample_texture(MaterialTexture texture_info, no_diff float2 uv)
    {
        TextureID texture = texture_info.texture;
        float3x3 transform = texture_info.transform;
        uv = mul(transform, float3(uv, 1.0)).xy;

        if (texture_info.filter == TextureFilterMode::Nearest) {
            return sample_texture_point(texture, uv, texture_info.wrap_s, texture_info.wrap_t);
        } else {
            return sample_texture_linear(texture, uv, texture_info.wrap_s, texture_info.wrap_t);
        }
    }

    [Differentiable]
    public Material get_material(MaterialID id, no_diff float2 uvs[UV_COUNT])
    {
        MaterialData data = materials[uint(id)];

        Material res;
        float4 albedoalpha = sample_texture(data.albedo, uvs[data.albedo.tex_coord]);
        res.albedo = albedoalpha.xyz;
        res.alpha = albedoalpha.w;
        res.specular = sample_texture(data.specular, uvs[data.specular.tex_coord]).xyz;
        res.emission = sample_texture(data.emission, uvs[data.emission.tex_coord]).xyz * data.emission_strength;
        res.roughness = sample_texture(data.roughness, uvs[data.roughness.tex_coord]).x;
        res.metallic = sample_texture(data.metallic, uvs[data.metallic.tex_coord]).x;
        res.normal = sample_texture(data.normal, uvs[data.normal.tex_coord]).xyz;
        res.normal = normalize(lerp(float3(0.0, 0.0, 1.0), res.normal, data.normal_map_strength));

        // If this is a spec/gloss material, convert to metal/rough. It's not hugely
        // efficient to do this on the fly, but it simplifies dealing texture transforms
        // and filtering modes significantly.
        if (data.type == 1) {
            dsg_to_amr(res.albedo, res.specular, res.roughness, res.albedo, res.metallic, res.roughness);
            res.specular = 0.5;
        }

        return res;
    }

    [Differentiable]
    public Material get_material(HitInfo hit_info, no_diff float2 uvs[UV_COUNT])
    {
        return get_material(hit_info.geometry_instance_id, uvs);
    }

    [Differentiable]
    public Material get_material(GeometryInstanceID geometry_instance_id, no_diff float2 uvs[UV_COUNT])
    {
        GeometryInstance instance = get_geometry_instance(geometry_instance_id);
        return get_material(instance.material, uvs);
    }

    [Differentiable]
    public float3 eval_emitted_radiance(MaterialID material, no_diff float2 uv)
    {
        MaterialData data = materials[uint(material)];
        return sample_texture(data.emission, uv).xyz * data.emission_strength;
    }

    public bool is_valid_hit(HitInfo hit, float3 ray_dir, float alpha_threshold)
    {
        var material_id = get_geometry_instance(hit.geometry_instance_id).material;
        var double_sided = materials[uint(material_id)].double_sided;
        var alpha_mode = materials[uint(material_id)].alpha_mode;

        if (!double_sided || alpha_mode != AlphaMode::Opaque) {
            var verts = get_triangle_vertices(hit.geometry_instance_id, hit.primitive_index);
            var face_normal_ls = cross(verts[1].position - verts[0].position, verts[2].position - verts[0].position);
            var inverse_transpose_transform = get_inverse_transpose_transform(hit.geometry_instance_id);
            var face_normal = mul(inverse_transpose_transform, float4(face_normal_ls, 0.0)).xyz;
            if (dot(ray_dir, face_normal) > 0)
                return false;

            if (alpha_mode != AlphaMode::Opaque) {
                var uvchannel = materials[uint(material_id)].albedo.tex_coord;
                var b = float3(1.0 - hit.bary.x - hit.bary.y, hit.bary.x, hit.bary.y);
                var uv = b.x * verts[0].uv[uvchannel] + b.y * verts[1].uv[uvchannel] + b.z * verts[2].uv[uvchannel];
                var alpha = sample_texture(materials[uint(material_id)].albedo, uv).w;
                if (alpha_mode == AlphaMode::Mask) {
                    return alpha >= materials[uint(material_id)].alpha_cutoff;
                } else { // Blend
                    return alpha > alpha_threshold;
                }
            }
        }

        return true;
    }
}
