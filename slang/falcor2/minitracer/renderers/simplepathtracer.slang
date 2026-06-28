// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

module simplepathtracer;

import slangpy;
import falcor2.minitracer.scene;
import falcor2.utils;
import falcor2.minitracer.rendering;

// Configuration

/// Enable next event estimation.
public extern static const bool ENABLE_NEE = true;
/// Enable multiple importance sampling.
public extern static const bool ENABLE_MIS = true;
/// Enable emissive triangles.
public extern static const bool ENABLE_EMISSIVE_TRIANGLES = true;
/// Enable environment map.
public extern static const bool ENABLE_ENV_MAP = true;
/// Maximum path depth.
// (1 = directly visible emission, 2 = direct lighting, 3 = one indirect bounce, etc.)
public extern static const uint MAX_DEPTH = 3;
/// If true, env map is used as background
public extern static const bool ENV_MAP_AS_BACKGROUND = true;
/// If not using env map as background, this is the color used for missed rays
public extern static const float3 BACKGROUND_COLOR = float3(1, 1, 1);


static SceneLightSampler<ENABLE_EMISSIVE_TRIANGLES, ENABLE_ENV_MAP> scene_light_sampler;

// Constants

public extension Camera
{
    public void __slangpy_load(Context2D context, out Ray value)
    {
        float2 pix_size = 1.0 / this.dims;
        float2 uv = float2(context.call_id_vec) / this.dims;
        uv.y = 1 - uv.y;
        value = get_ray(uv);
    }
    public void __slangpy_store(Context2D context, in Ray value) {};

    public void __slangpy_load(Context2D context, out RaySampler value)
    {
        value.camera = this;
        value.id = context.call_id_vec;

        float2 pix_size = 1.0 / this.dims;
        float2 uv = float2(context.call_id_vec) / this.dims;
        uv.y = 1 - uv.y;
        value.uv = uv;
        value.distribution = pix_size;
    }
    public void __slangpy_store(Context2D context, in RaySampler value) {};
}

public struct RaySampler
{
    public Camera camera;
    public uint2 id;
    public float2 uv;
    public float2 distribution;

    public Ray sample_ray(inout UniformSampleGenerator sg)
    {
        float2 offset = next_float2(sg) * float2(1, -1);
        float2 uv = this.uv + offset * this.distribution;
        return camera.get_ray(uv);
    }
}

public struct PathState : IDifferentiable
{
    public no_diff UniformSampleGenerator sg;

    public float3 scatter_pos_ws;
    public float3 scatter_dir_ws;
    public float3 scatter_normal_ws;
    public float scatter_pdf;

    public float3 throughput;
    public float3 radiance;
};

// Generate a ray from interaction point to randomly sampled light
Ray create_visibility_ray(const Interaction interaction, const DirectionSample sample)
{
    Ray ray;
    ray.origin = offset_ray_origin(interaction.pos_ws, interaction.normal_ws);
    ray.dir = sample.dir_ws;
    ray.t_min = 0.0;
    ray.t_max = float.MAX;
    if (sample.emitter_type == EmitterType::emissive_triangle) {
        float3 target = offset_ray_origin(sample.pos_ws, sample.normal_ws);
        float3 to_light = target - ray.origin;
        ray.t_max = length(to_light);
        ray.dir = to_light / ray.t_max;
    }
    return ray;
}

// Basic bent normal adjustment to stop normal maps causing outgoing ray to
// be behind the geometric normal
[Differentiable]
public float3 adjust_normal(float3 wi, float3 geom_normal, float3 shading_normal)
{
    const float kCosThetaThreshold = 0.01f;
    float cosTheta = dot(-wi, shading_normal);
    if (cosTheta <= kCosThetaThreshold) {
        float t = saturate(cosTheta * (1.f / kCosThetaThreshold));
        return normalize(lerp(geom_normal, shading_normal, t));
    } else {
        return shading_normal;
    }
}

[Differentiable]
public void trace_path(
    no_diff ISceneIntersector intersector,
    no_diff Ray ray,
    no_diff uint2 id,
    no_diff UniformSampleGenerator sg,
    out float3 color
)
{
    PathState path;
    path.sg = sg;
    path.throughput = float3(1, 1, 1);
    path.radiance = float3(0, 0, 0);
    path.scatter_pos_ws = {};
    path.scatter_dir_ws = {};
    path.scatter_normal_ws = {};
    path.scatter_pdf = 0;

    // Random alpha threshold for stocastic transparency
    float alpha_threshold = next_float(path.sg);

    uint depth = 1;
    [MaxIters(MAX_DEPTH)]
    while (depth <= MAX_DEPTH) {
        HitInfo hit = intersector.intersect(ray, alpha_threshold);
        bool is_primary_hit = depth == 1;

        // Handle missed rays.
        if (!hit.valid) {
            // Add environment map contribution (if enabled):
            // - Environment map is always evaluated for primary rays
            // - If NEE is disabled, it is evaluated for secondary rays
            // - If NEE and MIS are enabled, it is evaluated for secondary rays and weighted by MIS
            bool handle_env_map
                = ENABLE_ENV_MAP && g_scene.has_env_map() && (is_primary_hit || !ENABLE_NEE || ENABLE_MIS);
            if (handle_env_map) {
                float mis_weight = 1.0;
                if (!is_primary_hit && ENABLE_NEE && ENABLE_MIS) {
                    Interaction interaction = {};
                    interaction.pos_ws = path.scatter_pos_ws;
                    interaction.normal_ws = path.scatter_normal_ws;
                    DirectionSample sample = {};
                    sample.dir_ws = ray.dir;
                    sample.emitter_type = EmitterType::env_map;
                    float light_pdf = scene_light_sampler.eval_pdf(interaction, sample);
                    mis_weight = no_diff mis_heuristic(1, path.scatter_pdf, 1, light_pdf);
                }
                path.radiance += path.throughput * g_scene.env_map.eval(ray.dir) * mis_weight;

                if (!is_primary_hit || ENV_MAP_AS_BACKGROUND) {
                    path.radiance += path.throughput * g_scene.env_map.eval(ray.dir) * mis_weight;
                } else {
                    path.radiance = BACKGROUND_COLOR;
                }
                break;
            }
            break;
        }

        VertexData vertex = g_scene.get_vertex_data(hit);

        // Invert lighting for back face (single sides faces are culled in the query)
        if (dot(ray.dir, vertex.face_normal_ws) >= 0) {
            vertex.normal_ws = -vertex.normal_ws;
            vertex.tangent_ws = -vertex.tangent_ws;
            vertex.bitangent_ws = -vertex.bitangent_ws;
            vertex.face_normal_ws = -vertex.face_normal_ws;
        }

        // Load material and create a BRDF.
        Material material = g_scene.get_material(hit, vertex.uv);
        let brdf = DiffuseSpecularBRDF(
            material.albedo * vertex.color,
            material.specular,
            material.metallic,
            material.roughness
        );

        // Calculate a shading frame, apply normal map adjustment, and re-calculate shading frame.
        Frame frame = Frame(vertex.normal_ws, vertex.tangent_ws);
        float3 adj_normal = normalize(frame.to_global(material.normal));
        adj_normal = adjust_normal(ray.dir, vertex.face_normal_ws, adj_normal);
        frame = Frame(adj_normal, vertex.tangent_ws);

        // Add emissive triangle contribution (if enabled):
        // - Emissive triangles are always evaluated for primary rays
        // - If NEE is disabled, they are evaluated for secondary rays
        // - If NEE and MIS are enabled, they are evaluated for secondary rays and weighted by MIS
        bool handle_emission = ENABLE_EMISSIVE_TRIANGLES && (is_primary_hit || !ENABLE_NEE || ENABLE_MIS);
        if (handle_emission && any(material.emission > 0)) {
            float mis_weight = 1.0;
            if (!is_primary_hit && g_scene.has_emissive_triangles() && ENABLE_NEE && ENABLE_MIS) {
                Interaction interaction = {};
                interaction.pos_ws = path.scatter_pos_ws;
                interaction.normal_ws = path.scatter_normal_ws;
                DirectionSample sample = {};
                sample.pos_ws = vertex.pos_ws;
                sample.normal_ws = vertex.normal_ws;
                sample.emitter_type = EmitterType::emissive_triangle;
                sample.primitive_id
                    = uint(g_scene.emissive_triangles.get_triangle_id(hit.geometry_instance_id, hit.primitive_index));
                float light_pdf = scene_light_sampler.eval_pdf(interaction, sample);
                mis_weight = no_diff mis_heuristic(1, path.scatter_pdf, 1, light_pdf);
            }
            path.radiance += path.throughput * material.emission * mis_weight;
        }

        // Add NEE (next event estimation) contribution (if enabled).
        float3 wi = frame.to_local(-ray.dir);
        if (ENABLE_NEE && depth < MAX_DEPTH) {
            Interaction interaction = {};
            interaction.pos_ws = vertex.pos_ws;
            interaction.normal_ws = vertex.normal_ws;
            DirectionSample sample = {};
            float3 weight = {};
            if (scene_light_sampler.sample(interaction, sg, sample, weight)) {
                if (!intersector.is_occluded(create_visibility_ray(interaction, sample), alpha_threshold)) {
                    float3 wo = frame.to_local(sample.dir_ws);
                    float mis_weight = 1.0;
                    if (ENABLE_MIS) {
                        float scatter_pdf = brdf.eval_pdf(wi, wo);
                        mis_weight = no_diff mis_heuristic(1, sample.pdf, 1, scatter_pdf);
                    }
                    path.radiance += path.throughput * brdf.eval(wi, wo) * weight * mis_weight;
                }
            }
        }

        // Sample BRDF to generate next path segment.
        float3 wo;
        float3 weight;
        float pdf;
        if (brdf.sample(wi, wo, pdf, weight, sg)) {
            path.throughput *= weight;
        } else {
            break;
        }
        path.scatter_pos_ws = offset_ray_origin(vertex.pos_ws, vertex.face_normal_ws);
        path.scatter_dir_ws = frame.to_global(wo);
        path.scatter_normal_ws = vertex.normal_ws;
        path.scatter_pdf = pdf;

        // Generate next ray.
        ray = Ray(path.scatter_pos_ws, path.scatter_dir_ws);
        depth++;
    }

    color = path.radiance;
}

// Perform N samples of forward path tracer
public void render<Intersector : ISceneIntersector>(
    RaySampler ray_sampler,
    out float4 output,
    uniform uint spp,
    uniform uint iteration
)
{
    Intersector intersector;
    float3 radiance = 0;
    for (uint i = 0; i < spp; i++) {
        UniformSampleGenerator sg = UniformSampleGenerator(ray_sampler.id, iteration * spp + i);
        Ray ray = ray_sampler.sample_ray(sg);
        float3 c = 0;
        trace_path(intersector, ray, ray_sampler.id, sg, c);
        radiance += c;
    }
    output = float4(radiance / float(spp), 1);
}

// Perform N samples of backward path tracer
public void render_bwd<Intersector : ISceneIntersector>(
    RaySampler ray_sampler,
    float4 output_grad,
    uniform uint spp,
    uniform uint iteration
)
{
    Intersector intersector;
    // s_debug_gradient = float4(0, 0, 0, 0);
    output_grad.xyz /= float(spp);
    for (uint i = 0; i < spp; i++) {
        UniformSampleGenerator sg = UniformSampleGenerator(ray_sampler.id, iteration * spp + i);
        Ray ray = ray_sampler.sample_ray(sg);
        bwd_diff(trace_path)(intersector, ray, ray_sampler.id, sg, output_grad.xyz);
    }
    // debug_gradient = s_debug_gradient;
    // debug_gradient.xyz /= sample_count;
    // debug_gradient.w = 1.0;
}

// Render single guide ray
public bool trace_guide(
    ISceneIntersector intersector,
    Ray ray,
    uint spp,
    out float depth,
    out float3 position,
    out float3 normal,
    out float3 tangent,
    out float3 shading_normal,
    out float2 uv,
    out float3 albedo,
    out float3 specular,
    out float metallic,
    out float roughness
)
{
    depth = -1;
    position = float3(0, 0, 0);
    normal = float3(0, 0, 0);
    tangent = float3(0, 0, 0);
    shading_normal = float3(0, 0, 0);
    uv = float2(0, 0);
    albedo = float3(0, 0, 0);
    specular = float3(0, 0, 0);
    metallic = 0;
    roughness = 0;

    HitInfo hit = intersector.intersect(ray, 1);
    if (!hit.valid) {
        return false;
    }

    VertexData vertex = g_scene.get_vertex_data(hit);
    depth = hit.t;
    position = vertex.pos_ws;
    normal = vertex.normal_ws;
    tangent = vertex.tangent_ws;
    uv = vertex.uv[0]; // just output channel 0 for now

    Material material = g_scene.get_material(hit, vertex.uv);
    albedo = material.albedo * vertex.color;
    specular = material.specular;
    metallic = material.metallic;
    roughness = material.roughness;

    Frame frame = Frame(vertex.normal_ws, vertex.tangent_ws);
    float3 adj_normal = normalize(frame.to_global(material.normal));
    shading_normal = adjust_normal(ray.dir, vertex.face_normal_ws, adj_normal);

    return true;
}
// Render N samples of guide information (eg normals, depth, etc)
public void render_guides<Intersector : ISceneIntersector>(
    RaySampler ray_sampler,
    uint spp,
    out float depth,
    out float3 position,
    out float3 normal,
    out float3 tangent,
    out float3 shading_normal,
    out float2 uv,
    out float3 albedo,
    out float3 specular,
    out float metallic,
    out float roughness
)
{
    Intersector intersector;
    depth = -1;
    position = float3(0, 0, 0);
    normal = float3(0, 0, 0);
    tangent = float3(0, 0, 0);
    shading_normal = float3(0, 0, 0);
    albedo = float3(0, 0, 0);
    uv = float2(0, 0);
    specular = float3(0, 0, 0);
    metallic = 0;
    roughness = 0;
    if (true) {
        Ray ray = ray_sampler.camera.get_ray(ray_sampler.uv);
        trace_guide(
            intersector,
            ray,
            spp,
            depth,
            position,
            normal,
            tangent,
            shading_normal,
            uv,
            albedo,
            specular,
            metallic,
            roughness
        );
    } else {
        float tot = 0;
        for (uint i = 0; i < spp; i++) {
            UniformSampleGenerator sg = UniformSampleGenerator(ray_sampler.id, i);
            Ray ray = ray_sampler.sample_ray(sg);
            float d;
            float3 p;
            float3 n;
            float3 t;
            float3 sn;
            float2 u;
            float3 a;
            float3 s;
            float m;
            float r;
            if (trace_guide(intersector, ray, spp, d, p, n, t, sn, u, a, s, m, r)) {
                depth += d;
                position += p;
                normal += n;
                tangent += t;
                shading_normal += sn;
                uv += u;
                albedo += a;
                specular += s;
                metallic += m;
                roughness += r;
                tot += 1.0;
            }
        }
        if (tot > 0) {
            depth /= tot;
            position /= tot;
            normal = normalize(normal / tot);
            tangent = normalize(tangent / tot);
            shading_normal = normalize(shading_normal / tot);
            uv /= tot;
            albedo /= tot;
            specular /= tot;
            metallic /= tot;
            roughness /= tot;
        }
    }
}

public void clamp_value(inout float value, float min, float max)
{
    if (value < min) {
        value = min;
    } else if (value > max) {
        value = max;
    }
}

public void random_value(float random, inout float value)
{
    value = random;
}

// Copy Tensor of colours to a texture with optional
// scaling, accumulartion, abs, and tone mapping
public void resample(
    uint2 dispatchThreadID,
    Tensor<float4, 2> input,
    RWTexture2D<float4> output,
    uniform uint2 output_pos,
    uniform uint2 output_size,
    uniform float scale,
    uniform bool accumulator,
    uniform bool set_abs,
    uniform bool tone_map
)
{
    uint2 output_dimensions;
    output.GetDimensions(output_dimensions.x, output_dimensions.y);

    uint2 input_dimensions = uint2(input.shape[1], input.shape[0]);

    if (any(dispatchThreadID >= output_dimensions)) {
        return;
    }

    float2 uv = (float2(dispatchThreadID) + 0.5) / float2(output_size);

    uint2 coord = uint2(uv * float2(input_dimensions));

    float4 col = input[coord];

    if (set_abs) {
        col = abs(col);
    }

    if (accumulator) {
        col /= col.w;
    }

    if (tone_map) {
        col.xyz = tonemap_aces_film(col.xyz);
    }

    output[dispatchThreadID.xy] = col * scale;
}

public void resample_rg(
    uint2 dispatchThreadID,
    ITensor<float2, 2> input,
    RWTexture2D<float4> output,
    uniform uint2 output_pos,
    uniform uint2 output_size,
    uniform float scale,
    uniform bool set_abs,
    uniform bool tone_map
)
{
    uint2 output_dimensions;
    output.GetDimensions(output_dimensions.x, output_dimensions.y);

    uint2 input_dimensions = uint2(input.shape[1], input.shape[0]);

    if (any(dispatchThreadID >= output_dimensions)) {
        return;
    }

    float2 uv = (float2(dispatchThreadID) + 0.5) / float2(output_size);

    uint2 coord = uint2(uv * float2(input_dimensions));

    float4 col = float4(input[coord], 0.0, 1.0);

    if (set_abs) {
        col = abs(col);
    }

    if (tone_map) {
        col.xyz = tonemap_aces_film(col.xyz);
    }

    output[dispatchThreadID.xy] = col * scale;
}

public void resample_rgb(
    uint2 dispatchThreadID,
    ITensor<float3, 2> input,
    RWTexture2D<float4> output,
    uniform uint2 output_pos,
    uniform uint2 output_size,
    uniform float scale,
    uniform bool set_abs,
    uniform bool tone_map
)
{
    uint2 output_dimensions;
    output.GetDimensions(output_dimensions.x, output_dimensions.y);

    uint2 input_dimensions = uint2(input.shape[1], input.shape[0]);

    if (any(dispatchThreadID >= output_dimensions)) {
        return;
    }

    float2 uv = (float2(dispatchThreadID) + 0.5) / float2(output_size);

    uint2 coord = uint2(uv * float2(input_dimensions));

    float4 col = float4(input[coord], 1.0);

    if (set_abs) {
        col = abs(col);
    }

    if (tone_map) {
        col.xyz = tonemap_aces_film(col.xyz);
    }

    output[dispatchThreadID.xy] = col * scale;
}

public void resample_grayscale(
    uint2 dispatchThreadID,
    ITensor<float, 2> input,
    RWTexture2D<float4> output,
    uniform uint2 output_pos,
    uniform uint2 output_size,
    uniform float scale,
    uniform bool set_abs,
    uniform bool tone_map
)
{
    uint2 output_dimensions;
    output.GetDimensions(output_dimensions.x, output_dimensions.y);

    uint2 input_dimensions = uint2(input.shape[1], input.shape[0]);

    if (any(dispatchThreadID >= output_dimensions)) {
        return;
    }

    float2 uv = (float2(dispatchThreadID) + 0.5) / float2(output_size);

    uint2 coord = uint2(uv * float2(input_dimensions));

    float4 col = float4(input[coord], input[coord], input[coord], 1.0);

    if (set_abs) {
        col = abs(col);
    }

    if (tone_map) {
        col.xyz = tonemap_aces_film(col.xyz);
    }

    output[dispatchThreadID.xy] = col * scale;
}

public void apply_tonemap(float4 input, out float4 output)
{
    output.xyz = tonemap_aces_film(input.xyz);
    output.w = input.w;
}


public struct RayTracingSceneIntersector : ISceneIntersector
{
    public HitInfo intersect(Ray ray, float alpha_threshold)
    {
        HitInfo hit = {};
        TraceRay(
            g_scene.tlas,
            0,
            0xff,
            0 /* RayContributionToHitGroupIndex */,
            0 /* MultiplierForGeometryContributionHitGroupIndex */,
            0 /* MissShaderIndex */,
            ray.to_ray_desc(),
            hit
        );
        return hit;
    }

    public bool is_occluded(Ray ray, float alpha_threshold)
    {
        HitInfo hit = intersect(ray, alpha_threshold);
        return hit.valid;
    }
};

[shader("miss")]
public void rt_miss(inout HitInfo hit)
{
    // hit.valid = false;
}

[shader("closesthit")]
void rt_closest_hit(inout HitInfo hit, BuiltInTriangleIntersectionAttributes attribs)
{
    hit.valid = true;
    hit.bary = attribs.barycentrics;
    hit.t = RayTCurrent();
#ifndef __TARGET_CUDA__
    hit.geometry_instance_id = GeometryInstanceID(InstanceID() + GeometryIndex());
#else
    hit.geometry_instance_id = GeometryInstanceID(InstanceID());
#endif
    hit.primitive_index = PrimitiveIndex();
}

[shader("anyhit")]
void rt_any_hit(inout HitInfo /*hit*/, BuiltInTriangleIntersectionAttributes attribs)
{
    HitInfo temp_hit = {};
    temp_hit.valid = true;
    temp_hit.bary = attribs.barycentrics;
    temp_hit.t = RayTCurrent();
#ifndef __TARGET_CUDA__
    temp_hit.geometry_instance_id = GeometryInstanceID(InstanceID() + GeometryIndex());
#else
    temp_hit.geometry_instance_id = GeometryInstanceID(InstanceID());
#endif
    temp_hit.primitive_index = PrimitiveIndex();

    if (!g_scene.is_valid_hit(temp_hit, WorldRayDirection(), 1.0))
        IgnoreHit();
}
