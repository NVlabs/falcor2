// SPDX-License-Identifier: Apache-2.0

import slangpy;
module material_sphere_tracer;

__exported import falcor2.render;
__exported import falcor2.utils;

import falcor2.render.materials.materialx.materialx_test_geomprop_provider;
import sgl.device.print;

static LightSampler<false, true, false> light_sampler;
static const int MAX_BOUNCES = 32;
static const int RUSSIAN_ROULETTE_START_BOUNCE = 4;
static const float RUSSIAN_ROULETTE_MIN_SURVIVAL = 0.05;
static const float RUSSIAN_ROULETTE_MAX_SURVIVAL = 0.95;
static const uint DIRECT_LIGHTING_MODE_NEE_ONLY = 0u;
static const uint DIRECT_LIGHTING_MODE_DIRECT_HITS_ONLY = 1u;
static const uint DIRECT_LIGHTING_MODE_BOTH_50_50 = 2u;
static const uint DIRECT_LIGHTING_MODE_BOTH_MIS = 3u;

bool direct_lighting_uses_nee(const uint mode)
{
    return mode == DIRECT_LIGHTING_MODE_NEE_ONLY || mode == DIRECT_LIGHTING_MODE_BOTH_50_50
        || mode == DIRECT_LIGHTING_MODE_BOTH_MIS;
}

bool direct_lighting_uses_direct_hits(const uint mode)
{
    return mode == DIRECT_LIGHTING_MODE_DIRECT_HITS_ONLY || mode == DIRECT_LIGHTING_MODE_BOTH_50_50
        || mode == DIRECT_LIGHTING_MODE_BOTH_MIS;
}

bool direct_lighting_uses_mis(const uint mode)
{
    return mode == DIRECT_LIGHTING_MODE_BOTH_MIS;
}

float direct_lighting_nee_scale(const uint mode)
{
    return mode == DIRECT_LIGHTING_MODE_BOTH_50_50 ? 0.5 : 1.0;
}

float direct_lighting_direct_hit_scale(const uint mode)
{
    return mode == DIRECT_LIGHTING_MODE_BOTH_50_50 ? 0.5 : 1.0;
}

float3 visualizer_background_color()
{
    return float3(0.0);
}

bool intersect_sphere(
    const Ray ray,
    const float3 sphere_center,
    const float sphere_radius,
    out float t,
    const bool backfaceCull = false
)
{
    t = {};

    // Implementation is taken from Chapter 7 of Ray-Tracing Gems.
    float3 f = ray.origin - sphere_center;
    float a = dot(ray.dir, ray.dir);
    float b = dot(-f, ray.dir);
    float discriminant = sphere_radius * sphere_radius - dot(f + b / a * ray.dir, f + b / a * ray.dir);

    if (discriminant < 0.f)
        return false;

    if (b == 0 && discriminant == 0 && ray.t_min <= 0) {
        t = 0.f;
        return (t <= ray.t_max);
    }

    float c = dot(f, f) - sphere_radius * sphere_radius;
    float signB = (b < 0) ? -1 : 1;
    float q = b + signB * sqrt(a * discriminant);
    float t0 = c / q;
    float t1 = q / a;

    float tc = (t0 < ray.t_min && !backfaceCull) ? t1 : t0;
    if (tc < ray.t_min)
        return false;

    t = tc;
    return (t <= ray.t_max);
}

struct SphereTraceCamera
{
    float3 position;
    float3 image_u;
    float3 image_v;
    float3 image_w;

    __init()
    {
        float fov = 45;
        float img_size = tan(radians(fov / 2));

        position = float3(0, 0, 3);
        image_u = float3(img_size, 0, 0);
        image_v = float3(0, -img_size, 0);
        image_w = float3(0, 0, -1);
    }

    Ray get_ray(float2 uv)
    {
        uv = uv * 2 - 1;
        float3 dir = normalize(uv.x * image_u + uv.y * image_v + image_w);
        return Ray(position, dir);
    }
};

float2 map_to_sphere_uv(float3 N_ws)
{
    // Match OSL-style tessellated sphere UVs. For a point on the sphere:
    //   x = -sin(u * 2pi) * sin(v * pi)
    //   y =  cos(v * pi)
    //   z = -cos(u * 2pi) * sin(v * pi)
    var phi = atan2(-N_ws.x, -N_ws.z);
    while (phi >= float.TWO_PI)
        phi -= float.TWO_PI;
    while (phi < 0)
        phi += float.TWO_PI;

    let v = clamp(acos(clamp(N_ws.y, -1.0, 1.0)) / float.PI, 0, 1);
    let u = clamp(phi / float.TWO_PI, 0, 1);

    return { u, v };
}

bool is_finite_float(const float value)
{
    return value == value && abs(value) < 1e20;
}

bool is_finite_float2(const float2 value)
{
    return is_finite_float(value.x) && is_finite_float(value.y);
}

bool is_finite_float3(const float3 value)
{
    return is_finite_float(value.x) && is_finite_float(value.y) && is_finite_float(value.z);
}

float max_component(const float3 value)
{
    return max(max(value.x, value.y), value.z);
}

bool is_nonzero_finite_float3(const float3 value)
{
    return is_finite_float3(value) && max_component(abs(value)) > 0.0;
}

float dense_direct_probe_weight(const int light_count)
{
    return 12.566370614359172 / float(light_count);
}

float3 dense_direct_probe_direction(const int index, const int light_count)
{
    let t = (float(index) + 0.5) / float(light_count);
    let y = 1.0 - 2.0 * t;
    let r = sqrt(max(0.0, 1.0 - y * y));
    let phi = 2.39996322972865332 * float(index);
    return normalize(float3(cos(phi) * r, y, sin(phi) * r));
}

float dense_env_grid_probe_weight(const int index, const int grid_size)
{
    return 4.0 * float.PI / float(grid_size * grid_size);
}

float3 dense_env_grid_probe_direction(const int index, const int grid_size)
{
    let x_index = index % grid_size;
    let y_index = index / grid_size;
    let u = (float(x_index) + 0.5) / float(grid_size);
    let v = (float(y_index) + 0.5) / float(grid_size);
    let y = 1.0 - 2.0 * v;
    let radius = sqrt(max(0.0, 1.0 - y * y));
    let phi = float.TWO_PI * (u - 0.5);
    return normalize(float3(radius * sin(phi), y, -radius * cos(phi)));
}

float dense_env_texel_probe_weight(const int index, const int env_width, const int env_height)
{
    let y_index = index / env_width;
    let theta = float.PI * (float(y_index) + 0.5) / float(env_height);
    return (float.TWO_PI / float(env_width)) * (float.PI / float(env_height)) * sin(theta);
}

float3 dense_env_texel_probe_direction(const int index, const int env_width, const int env_height)
{
    let x_index = index % env_width;
    let y_index = index / env_width;
    let u = (float(x_index) + 0.5) / float(env_width);
    let v = (float(y_index) + 0.5) / float(env_height);
    let theta = float.PI * v;
    let phi = float.TWO_PI * (u - 0.5);
    let sin_theta = sin(theta);
    return normalize(float3(sin_theta * sin(phi), cos(theta), -sin_theta * cos(phi)));
}

public struct AnalyticSphereLodSampler : ILodSampler
{
    bool valid;
    float2 duv_dx;
    float2 duv_dy;

    public __generic<T : ITexelElement> T sample(Texture2D<T> texture, SamplerState sampler, float2 uv)
    {
        if (valid)
            return texture.SampleGrad(sampler, uv, duv_dx, duv_dy);
        return texture.SampleLevel(sampler, uv, 0);
    }

    public __generic<T : ITexelElement> T sample(Texture3D<T> texture, SamplerState sampler, float3 uvw)
    {
        if (valid)
            return texture.SampleGrad(sampler, uvw, float3(duv_dx, 0), float3(duv_dy, 0));
        return texture.SampleLevel(sampler, uvw, 0);
    }
};

AnalyticSphereLodSampler make_lod0_sphere_lod_sampler()
{
    AnalyticSphereLodSampler sampler;
    sampler.valid = false;
    sampler.duv_dx = {};
    sampler.duv_dy = {};
    return sampler;
}

float3 camera_image_vector(const SphereTraceCamera camera, const float2 uv)
{
    let screen_uv = uv * 2 - 1;
    return screen_uv.x * camera.image_u + screen_uv.y * camera.image_v + camera.image_w;
}

float3 normalized_direction_derivative(const float3 q, const float3 direction, const float3 dq)
{
    return (dq - direction * dot(direction, dq)) / length(q);
}

bool sphere_normal_to_uv_derivative(const float3 normal_ws, const float3 dnormal_ws, out float2 duv)
{
    duv = {};

    // Keep this in lockstep with map_to_sphere_uv(). In that mapping, u is
    // derived from atan2(-x, -z), and v is derived from acos(y).
    let xz_length2 = normal_ws.x * normal_ws.x + normal_ws.z * normal_ws.z;
    if (xz_length2 < 1e-8)
        return false;

    let xz_length = sqrt(xz_length2);
    if (xz_length < 1e-4)
        return false;

    let du = (normal_ws.z * dnormal_ws.x - normal_ws.x * dnormal_ws.z) / (xz_length2 * float.TWO_PI);
    let dv = -dnormal_ws.y / (xz_length * float.PI);
    duv = float2(du, dv);
    return is_finite_float2(duv);
}

bool sphere_uv_derivative(const Ray ray, const float t, const float3 direction_derivative, out float2 duv)
{
    duv = {};

    let normal_ws = normalize(ray.origin + ray.dir * t);
    let root = -(dot(ray.origin, ray.dir) + t);
    if (abs(root) < 1e-4)
        return false;

    let db = dot(ray.origin, direction_derivative);
    let dt = -(1 + dot(ray.origin, ray.dir) / root) * db;
    let position_derivative = direction_derivative * t + ray.dir * dt;
    let normal_derivative = position_derivative - normal_ws * dot(normal_ws, position_derivative);

    return sphere_normal_to_uv_derivative(normal_ws, normal_derivative, duv);
}

AnalyticSphereLodSampler make_analytic_sphere_lod_sampler(
    const SphereTraceCamera camera,
    const float2 uv,
    const float2 image_size,
    const Ray ray,
    const float t,
    const bool use_analytic_texture_filtering,
    const float texture_filter_width_scale
)
{
    AnalyticSphereLodSampler sampler;
    sampler.valid = false;
    sampler.duv_dx = {};
    sampler.duv_dy = {};

    if (!use_analytic_texture_filtering)
        return sampler;

    let q = camera_image_vector(camera, uv);
    let dq_dx = camera.image_u * (2.0 / image_size.x);
    let dq_dy = camera.image_v * (2.0 / image_size.y);
    let direction_dx = normalized_direction_derivative(q, ray.dir, dq_dx);
    let direction_dy = normalized_direction_derivative(q, ray.dir, dq_dy);

    float2 duv_dx;
    float2 duv_dy;
    if (!sphere_uv_derivative(ray, t, direction_dx, duv_dx) || !sphere_uv_derivative(ray, t, direction_dy, duv_dy))
        return sampler;

    duv_dx *= texture_filter_width_scale;
    duv_dy *= texture_filter_width_scale;

    // Reject singular or obviously explosive footprints. Falling back to LOD 0
    // is preferable to feeding invalid gradients into backend texture sampling.
    if (length(duv_dx) > 16.0 || length(duv_dy) > 16.0)
        return sampler;

    sampler.valid = is_finite_float2(duv_dx) && is_finite_float2(duv_dy);
    sampler.duv_dx = duv_dx;
    sampler.duv_dy = duv_dy;
    return sampler;
}

SurfaceInteraction setup_sphere_surface_interaction(const Ray ray, const float t)
{
    let P_ws = ray.origin + ray.dir * t;
    let Ng_ws = normalize(P_ws);

    SurfaceInteraction si;
    si.position_ws = P_ws;
    si.wi_ws = -ray.dir;
    // Compute UVs from the geometric normal before flipping N/Ng for
    // backfacing hits, so transmitted paths see the expected local hemisphere.
    let front_facing = dot(-ray.dir, Ng_ws) >= 0.0;
    let shading_N_ws = front_facing ? Ng_ws : -Ng_ws;
    si.uv = map_to_sphere_uv(Ng_ws);
    si.normal_ws = shading_N_ws;
    si.shading_frame_ws = Frame(shading_N_ws);
    si.front_facing = front_facing;
    si.ior = 1.0;
    return si;
}

SurfaceInteraction setup_eval_plot_surface_interaction(const float3 view_direction_ws)
{
    let Ng_ws = float3(0.0, 0.0, 1.0);

    SurfaceInteraction si;
    si.position_ws = Ng_ws;
    si.wi_ws = normalize(view_direction_ws);
    si.uv = map_to_sphere_uv(Ng_ws);
    si.normal_ws = Ng_ws;
    si.shading_frame_ws = Frame(Ng_ws);
    si.front_facing = dot(si.wi_ws, Ng_ws) >= 0.0;
    si.ior = 1.0;
    return si;
}

float3 offset_sphere_scatter_origin(const SurfaceInteraction si, const float3 wo_ws)
{
    let orientation = (dot(wo_ws, si.normal_ws) >= 0.0) ? 1.0 : -1.0;
    return offset_ray_origin(si.position_ws, si.normal_ws * orientation);
}

bool analytic_sphere_environment_visible(const SurfaceInteraction si, const float3 dir_ws)
{
    let dir = normalize(dir_ws);
    Ray shadow_ray = Ray(offset_ray_origin(si.position_ws, dir), dir);
    float t;
    return !intersect_sphere(shadow_ray, float3(0.0), 1.0, t, false);
}

float3 eval_environment_miss(
    const SurfaceInteraction previous_si,
    const float3 dir_ws,
    const float previous_pdf,
    const bool previous_delta,
    const uint direct_lighting_mode
)
{
    DirectionSample light_sample = {};
    light_sample.dir_ws = dir_ws;
    light_sample.distance = float.MAX;
    light_sample.emitter_type = EmitterType::environment;

    let light = g_scene.light_system.get_light(g_scene.light_system.environment_light_id);
    float3 env_radiance = light.eval_direction(previous_si, light_sample);
    float env_pdf = light_sampler.eval_direction_pdf(previous_si, light_sample);
    let use_sample_mis = direct_lighting_uses_mis(direct_lighting_mode) && !previous_delta && previous_pdf > 0.0;
    float w = use_sample_mis ? mis_heuristic(1, previous_pdf, 1, env_pdf) : 1.0;
    return env_radiance * w * direct_lighting_direct_hit_scale(direct_lighting_mode);
}

extern struct Material : IMaterial;
extern struct SampleGenerator : ISampleGenerator;

struct KahanSummer
{
    __init() { }

    [mutating]
    void add(float3 value)
    {
        float3 y = value - c;
        float3 t = total + y;
        c = (t - total) - y;
        total = t;
    }

    float3 total = 0;
    float3 c = 0;
};

float neumaier_correction(const float total, const float value, const float next_total)
{
    return abs(total) >= abs(value) ? (total - next_total) + value : (value - next_total) + total;
}

void compensated_accumulate(inout float3 total, inout float3 residual, const float3 value)
{
    const float3 next_total = total + value;
    residual += float3(
        neumaier_correction(total.x, value.x, next_total.x),
        neumaier_correction(total.y, value.y, next_total.y),
        neumaier_correction(total.z, value.z, next_total.z)
    );
    total = next_total;
}

float3 resolved_accumulation(const float3 total, const float3 residual, const uint sample_count)
{
    return sample_count > 0 ? (total + residual) / float(sample_count) : visualizer_background_color();
}

void material_visualizer_store_path_log4(uniform WTensor<float, 1> path_log, uint slot, float4 value)
{
    const uint base_index = slot * 4u;
    path_log.store(base_index + 0u, value.x);
    path_log.store(base_index + 1u, value.y);
    path_log.store(base_index + 2u, value.z);
    path_log.store(base_index + 3u, value.w);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void sphere_tracing_kernel(
    uint3 tid: SV_DispatchThreadID,
    uniform Material material,
    uniform uint2 camera_size,
    uniform uint2 pixel_offset,
    uniform uint sample_index_offset,
    uniform uint sample_count,
    uniform uint completed_sample_count,
    uniform uint direct_lighting_mode,
    uniform bool use_analytic_texture_filtering,
    uniform float texture_filter_width_scale,
    uniform bool clear_result,
    uniform RWTexture2D<float4> accumulation_sum,
    uniform RWTexture2D<float4> accumulation_residual,
    uniform RWTexture2D<float4> result
)
{
    uint width, height;
    result.GetDimensions(width, height);

    if (tid.x >= width || tid.y >= height)
        return;

    set_print_enabled(false);

    let camera = SphereTraceCamera();
    const uint2 pixel = pixel_offset + tid.xy;
    if (pixel.x >= camera_size.x || pixel.y >= camera_size.y) {
        if (clear_result) {
            accumulation_sum[tid.xy] = float4(0.0);
            accumulation_residual[tid.xy] = float4(0.0);
        }
        result[tid.xy] = float4(visualizer_background_color(), 1.0);
        return;
    }

    const bool use_nee = direct_lighting_uses_nee(direct_lighting_mode);
    const bool use_direct_hits = direct_lighting_uses_direct_hits(direct_lighting_mode);
    const bool use_mis = direct_lighting_uses_mis(direct_lighting_mode);
    const int batch_sample_count = max(1, int(sample_count));
    KahanSummer color = {};
    for (int i = 0; i < batch_sample_count; ++i) {
        const uint global_sample_index = sample_index_offset + uint(i);
        var sg = SampleGenerator(pixel, global_sample_index);
        let uv = (float2(pixel) + next_float2(sg)) / float2(float(camera_size.x), float(camera_size.y));
        var ray = camera.get_ray(uv);
        var throughput = float3(1.0);
        SurfaceInteraction previous_si = {};
        var previous_pdf = 0.0;
        var previous_delta = false;
        var has_previous_scatter = false;

        for (int bounce = 0; bounce < MAX_BOUNCES; ++bounce) {
            float t;
            bool hit = intersect_sphere(ray, float3(0), 1, t, false);
            if (!hit) {
                if (bounce == 0 || !has_previous_scatter) {
                    color.add(visualizer_background_color());
                } else if (use_direct_hits) {
                    float3 contribution = throughput
                        * eval_environment_miss(
                                              previous_si,
                                              ray.dir,
                                              previous_pdf,
                                              previous_delta,
                                              direct_lighting_mode
                        );
                    if (is_finite_float3(contribution))
                        color.add(contribution);
                }
                break;
            }

            let si = setup_sphere_surface_interaction(ray, t);
            var lod_sampler = make_lod0_sphere_lod_sampler();
            if (bounce == 0) {
                lod_sampler = make_analytic_sphere_lod_sampler(
                    camera,
                    uv,
                    float2(float(camera_size.x), float(camera_size.y)),
                    ray,
                    t,
                    use_analytic_texture_filtering,
                    texture_filter_width_scale
                );
            }
            let material_instance = material.setup_material_instance(si, lod_sampler, MaterialInstanceHints::none);

            let emitted_radiance = material_instance.eval_emissive_radiance();
            if (is_finite_float3(emitted_radiance)) {
                let contribution = throughput * max(emitted_radiance, float3(0.0));
                if (is_finite_float3(contribution))
                    color.add(contribution);
            }

            if (use_nee) {
                DirectionSample light_sample = {};
                float3 light_weight = {};
                if (light_sampler.sample_direction(si, sg, light_sample, light_weight)
                    && analytic_sphere_environment_visible(si, light_sample.dir_ws)) {
                    float bsdf_pdf = material_instance.eval_pdf(si, light_sample.dir_ws);
                    float3 bsdf_eval = material_instance.eval(si, light_sample.dir_ws, sg);
                    if (bsdf_pdf > 0.0 && is_nonzero_finite_float3(bsdf_eval)) {
                        float w = use_mis ? mis_heuristic(1, light_sample.pdf, 1, bsdf_pdf) : 1;
                        float3 contribution = throughput * light_weight * bsdf_eval * w
                            * direct_lighting_nee_scale(direct_lighting_mode);
                        if (is_finite_float3(contribution))
                            color.add(contribution);
                    }
                }
            }

            if (!use_direct_hits || bounce + 1 >= MAX_BOUNCES)
                break;

            BSDFSample sample = {};
            if (!material_instance.sample(si, sg, sample))
                break;
            if (!is_nonzero_finite_float3(sample.weight) || !is_finite_float3(sample.wo_ws))
                break;

            throughput *= sample.weight;
            if (!is_nonzero_finite_float3(throughput))
                break;

            if (bounce >= RUSSIAN_ROULETTE_START_BOUNCE) {
                let survival_probability = clamp(
                    max_component(abs(throughput)),
                    RUSSIAN_ROULETTE_MIN_SURVIVAL,
                    RUSSIAN_ROULETTE_MAX_SURVIVAL
                );
                if (next_float(sg) > survival_probability)
                    break;
                throughput /= survival_probability;
            }

            previous_si = si;
            previous_pdf = sample.pdf;
            previous_delta = sample.has_lobe(LobeTypes::delta);
            has_previous_scatter = true;
            ray = Ray(offset_sphere_scatter_origin(si, sample.wo_ws), normalize(sample.wo_ws));
        }
    }

    float3 total = clear_result ? float3(0.0) : accumulation_sum[tid.xy].xyz;
    float3 residual = clear_result ? float3(0.0) : accumulation_residual[tid.xy].xyz;
    compensated_accumulate(total, residual, color.total);
    const uint new_sample_count = completed_sample_count + sample_count;
    accumulation_sum[tid.xy] = float4(total, 1.0);
    accumulation_residual[tid.xy] = float4(residual, 1.0);

    float3 color_result = resolved_accumulation(total, residual, new_sample_count);
    if (!is_finite_float3(color_result))
        color_result = visualizer_background_color();
    result[tid.xy] = float4(color_result, 1);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void material_visualizer_path_log_kernel(
    uint3 tid: SV_DispatchThreadID,
    uniform Material material,
    uniform int max_bounces,
    uniform uint2 camera_size,
    uniform uint2 pixel,
    uniform uint sample_index,
    uniform uint direct_lighting_mode,
    uniform bool no_jitter,
    uniform bool use_analytic_texture_filtering,
    uniform float texture_filter_width_scale,
    uniform bool enable_debug_print,
    uniform WTensor<float, 1> path_log
)
{
    if (tid.x != 0 || tid.y != 0 || tid.z != 0)
        return;

    set_print_enabled(enable_debug_print);

    const int bounce_count = clamp(max_bounces, 0, MAX_BOUNCES);
    const bool use_nee = direct_lighting_uses_nee(direct_lighting_mode);
    const bool use_direct_hits = direct_lighting_uses_direct_hits(direct_lighting_mode);
    const bool use_mis = direct_lighting_uses_mis(direct_lighting_mode);
    const SphereTraceCamera camera = SphereTraceCamera();
    const float2 image_size = float2(float(camera_size.x), float(camera_size.y));
    var sg = SampleGenerator(pixel, sample_index);

    const float2 jitter = no_jitter ? float2(0.5) : next_float2(sg);
    const float2 uv = (float2(float(pixel.x), float(pixel.y)) + jitter) / image_size;
    var ray = camera.get_ray(uv);
    var radiance = float3(0.0);
    var throughput = float3(1.0);

    material_visualizer_store_path_log4(
        path_log,
        0u,
        float4(float(pixel.x), float(pixel.y), float(sample_index), float(bounce_count))
    );
    material_visualizer_store_path_log4(path_log, 1u, float4(jitter, 0.0, 0.0));
    material_visualizer_store_path_log4(
        path_log,
        2u,
        float4(jitter, float(pixel.x) + jitter.x, float(pixel.y) + jitter.y)
    );
    material_visualizer_store_path_log4(path_log, 3u, float4(ray.origin, 0.0));
    material_visualizer_store_path_log4(path_log, 4u, float4(ray.dir, 0.0));

    float t;
    bool hit = intersect_sphere(ray, float3(0.0), 1.0, t, false);
    material_visualizer_store_path_log4(path_log, 5u, float4(hit ? 1.0 : 0.0, t, 0.0, 0.0));

    if (!hit || bounce_count <= 0) {
        material_visualizer_store_path_log4(path_log, 22u, float4(radiance, 0.0));
        return;
    }

    const SurfaceInteraction si = setup_sphere_surface_interaction(ray, t);
    var lod_sampler = make_lod0_sphere_lod_sampler();
    lod_sampler = make_analytic_sphere_lod_sampler(
        camera,
        uv,
        image_size,
        ray,
        t,
        use_analytic_texture_filtering,
        texture_filter_width_scale
    );
    let material_instance = material.setup_material_instance(si, lod_sampler, MaterialInstanceHints::none);

    material_visualizer_store_path_log4(path_log, 6u, float4(0.0, 0.0, si.front_facing ? 1.0 : 0.0, 0.0));
    material_visualizer_store_path_log4(path_log, 7u, float4(si.position_ws, 0.0));
    material_visualizer_store_path_log4(path_log, 8u, float4(si.normal_ws, 0.0));
    material_visualizer_store_path_log4(path_log, 9u, float4(si.shading_frame_ws.normal, 0.0));
    material_visualizer_store_path_log4(path_log, 10u, float4(si.wi_ws, 0.0));
    material_visualizer_store_path_log4(path_log, 11u, float4(si.uv, si.ior, 0.0));

    const float3 emitted_radiance = material_instance.eval_emissive_radiance();
    if (is_finite_float3(emitted_radiance))
        radiance += throughput * max(emitted_radiance, float3(0.0));

    if (use_nee) {
        DirectionSample light_sample = {};
        float3 light_weight = {};
        const bool sampled_light = light_sampler.sample_direction(si, sg, light_sample, light_weight);
        const bool visible = sampled_light && analytic_sphere_environment_visible(si, light_sample.dir_ws);
        var bsdf_pdf = 0.0;
        var bsdf_eval = float3(0.0);
        if (sampled_light) {
            bsdf_pdf = material_instance.eval_pdf(si, light_sample.dir_ws);
            bsdf_eval = material_instance.eval(si, light_sample.dir_ws, sg);
        }
        const float mis_weight = sampled_light && use_mis ? mis_heuristic(1, light_sample.pdf, 1, bsdf_pdf) : 1.0;
        const float direct_weight = mis_weight * direct_lighting_nee_scale(direct_lighting_mode);
        var direct_contribution = float3(0.0);
        if (visible && bsdf_pdf > 0.0 && is_nonzero_finite_float3(bsdf_eval)) {
            direct_contribution = throughput * light_weight * bsdf_eval * direct_weight;
            if (is_finite_float3(direct_contribution))
                radiance += direct_contribution;
        }

        material_visualizer_store_path_log4(path_log, 13u, float4(light_sample.dir_ws, sampled_light ? 1.0 : 0.0));
        material_visualizer_store_path_log4(path_log, 14u, float4(light_weight, 0.0));
        material_visualizer_store_path_log4(
            path_log,
            15u,
            float4(light_sample.pdf, bsdf_pdf, visible ? 1.0 : 0.0, direct_weight)
        );
        material_visualizer_store_path_log4(path_log, 16u, float4(bsdf_eval, 0.0));
        material_visualizer_store_path_log4(path_log, 17u, float4(direct_contribution, 0.0));
        print(
            "material_env sampled={} vis={} dir={} weight={} lpdf={} beval={} bpdf={} mis={} contrib={}",
            sampled_light,
            visible,
            light_sample.dir_ws,
            light_weight,
            light_sample.pdf,
            bsdf_eval,
            bsdf_pdf,
            direct_weight,
            direct_contribution
        );
    }

    if (use_direct_hits) {
        BSDFSample sample = {};
        const bool sampled_bsdf = material_instance.sample(si, sg, sample);
        material_visualizer_store_path_log4(path_log, 18u, float4(sample.wo_ws, sampled_bsdf ? 1.0 : 0.0));
        material_visualizer_store_path_log4(path_log, 19u, float4(sample.weight, 0.0));
        material_visualizer_store_path_log4(
            path_log,
            20u,
            float4(
                sample.pdf,
                sample.has_lobe(LobeTypes::delta) ? 1.0 : 0.0,
                sample.has_lobe(LobeTypes::reflection) ? 1.0 : 0.0,
                sample.has_lobe(LobeTypes::transmission) ? 1.0 : 0.0
            )
        );
        print(
            "material_bsdf_sample sampled={} wo={} weight={} pdf={} delta={} reflection={} transmission={}",
            sampled_bsdf,
            sample.wo_ws,
            sample.weight,
            sample.pdf,
            sample.has_lobe(LobeTypes::delta),
            sample.has_lobe(LobeTypes::reflection),
            sample.has_lobe(LobeTypes::transmission)
        );

        if (sampled_bsdf && is_nonzero_finite_float3(sample.weight) && is_finite_float3(sample.wo_ws)) {
            throughput *= sample.weight;
            Ray next_ray = Ray(offset_sphere_scatter_origin(si, sample.wo_ws), normalize(sample.wo_ws));
            material_visualizer_store_path_log4(path_log, 21u, float4(next_ray.dir, 0.0));
            if (bounce_count > 1) {
                float next_t;
                const bool next_hit = intersect_sphere(next_ray, float3(0.0), 1.0, next_t, false);
                if (!next_hit) {
                    const float3 miss_contribution = throughput
                        * eval_environment_miss(si,
                                                next_ray.dir,
                                                sample.pdf,
                                                sample.has_lobe(LobeTypes::delta),
                                                direct_lighting_mode);
                    if (is_finite_float3(miss_contribution))
                        radiance += miss_contribution;
                }
            }
        }
    }

    material_visualizer_store_path_log4(path_log, 22u, float4(radiance, 0.0));
    print("material_radiance {}", radiance);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void sphere_tracing_scene_material_kernel(
    uint3 tid: SV_DispatchThreadID,
    uniform uint material_id,
    uniform uint2 camera_size,
    uniform uint2 pixel_offset,
    uniform uint sample_index_offset,
    uniform uint sample_count,
    uniform uint completed_sample_count,
    uniform uint direct_lighting_mode,
    uniform bool use_analytic_texture_filtering,
    uniform float texture_filter_width_scale,
    uniform bool clear_result,
    uniform RWTexture2D<float4> accumulation_sum,
    uniform RWTexture2D<float4> accumulation_residual,
    uniform RWTexture2D<float4> result
)
{
    uint width, height;
    result.GetDimensions(width, height);

    if (tid.x >= width || tid.y >= height)
        return;

    set_print_enabled(false);

    let material = g_scene.material_system.get_material(MaterialID(material_id));
    let material_header = g_scene.material_system.get_material_data(MaterialID(material_id)).header;
    let cull_backfaces = (material_header.flags & MaterialFlags::thin_walled) == MaterialFlags::none;
    let camera = SphereTraceCamera();
    const uint2 pixel = pixel_offset + tid.xy;
    if (pixel.x >= camera_size.x || pixel.y >= camera_size.y) {
        if (clear_result) {
            accumulation_sum[tid.xy] = float4(0.0);
            accumulation_residual[tid.xy] = float4(0.0);
        }
        result[tid.xy] = float4(visualizer_background_color(), 1.0);
        return;
    }

    const bool use_nee = direct_lighting_uses_nee(direct_lighting_mode);
    const bool use_direct_hits = direct_lighting_uses_direct_hits(direct_lighting_mode);
    const bool use_mis = direct_lighting_uses_mis(direct_lighting_mode);
    const int batch_sample_count = max(1, int(sample_count));
    KahanSummer color = {};
    for (int i = 0; i < batch_sample_count; ++i) {
        const uint global_sample_index = sample_index_offset + uint(i);
        var sg = SampleGenerator(pixel, global_sample_index);
        let uv = (float2(pixel) + next_float2(sg)) / float2(float(camera_size.x), float(camera_size.y));
        var ray = camera.get_ray(uv);
        var throughput = float3(1.0);
        SurfaceInteraction previous_si = {};
        var previous_pdf = 0.0;
        var previous_delta = false;
        var has_previous_scatter = false;

        for (int bounce = 0; bounce < MAX_BOUNCES; ++bounce) {
            float t;
            bool hit = intersect_sphere(ray, float3(0), 1, t, cull_backfaces);
            if (!hit) {
                if (bounce == 0 || !has_previous_scatter) {
                    color.add(visualizer_background_color());
                } else if (use_direct_hits) {
                    float3 contribution = throughput
                        * eval_environment_miss(
                                              previous_si,
                                              ray.dir,
                                              previous_pdf,
                                              previous_delta,
                                              direct_lighting_mode
                        );
                    if (is_finite_float3(contribution))
                        color.add(contribution);
                }
                break;
            }

            let si = setup_sphere_surface_interaction(ray, t);
            var lod_sampler = make_lod0_sphere_lod_sampler();
            if (bounce == 0) {
                lod_sampler = make_analytic_sphere_lod_sampler(
                    camera,
                    uv,
                    float2(float(camera_size.x), float(camera_size.y)),
                    ray,
                    t,
                    use_analytic_texture_filtering,
                    texture_filter_width_scale
                );
            }
            let material_instance = material.setup_material_instance(si, lod_sampler, MaterialInstanceHints::none);

            let emitted_radiance = material_instance.eval_emissive_radiance();
            if (is_finite_float3(emitted_radiance)) {
                let contribution = throughput * max(emitted_radiance, float3(0.0));
                if (is_finite_float3(contribution))
                    color.add(contribution);
            }

            if (use_nee) {
                DirectionSample light_sample = {};
                float3 light_weight = {};
                if (light_sampler.sample_direction(si, sg, light_sample, light_weight)
                    && analytic_sphere_environment_visible(si, light_sample.dir_ws)) {
                    float bsdf_pdf = material_instance.eval_pdf(si, light_sample.dir_ws);
                    float3 bsdf_eval = material_instance.eval(si, light_sample.dir_ws, sg);
                    if (bsdf_pdf > 0.0 && is_nonzero_finite_float3(bsdf_eval)) {
                        float w = use_mis ? mis_heuristic(1, light_sample.pdf, 1, bsdf_pdf) : 1;
                        float3 contribution = throughput * light_weight * bsdf_eval * w
                            * direct_lighting_nee_scale(direct_lighting_mode);
                        if (is_finite_float3(contribution))
                            color.add(contribution);
                    }
                }
            }

            if (!use_direct_hits || bounce + 1 >= MAX_BOUNCES)
                break;

            BSDFSample sample = {};
            if (!material_instance.sample(si, sg, sample))
                break;
            if (!is_nonzero_finite_float3(sample.weight) || !is_finite_float3(sample.wo_ws))
                break;

            throughput *= sample.weight;
            if (!is_nonzero_finite_float3(throughput))
                break;

            if (bounce >= RUSSIAN_ROULETTE_START_BOUNCE) {
                let survival_probability = clamp(
                    max_component(abs(throughput)),
                    RUSSIAN_ROULETTE_MIN_SURVIVAL,
                    RUSSIAN_ROULETTE_MAX_SURVIVAL
                );
                if (next_float(sg) > survival_probability)
                    break;
                throughput /= survival_probability;
            }

            previous_si = si;
            previous_pdf = sample.pdf;
            previous_delta = sample.has_lobe(LobeTypes::delta);
            has_previous_scatter = true;
            ray = Ray(offset_sphere_scatter_origin(si, sample.wo_ws), normalize(sample.wo_ws));
        }
    }

    float3 total = clear_result ? float3(0.0) : accumulation_sum[tid.xy].xyz;
    float3 residual = clear_result ? float3(0.0) : accumulation_residual[tid.xy].xyz;
    compensated_accumulate(total, residual, color.total);
    const uint new_sample_count = completed_sample_count + sample_count;
    accumulation_sum[tid.xy] = float4(total, 1.0);
    accumulation_residual[tid.xy] = float4(residual, 1.0);

    float3 color_result = resolved_accumulation(total, residual, new_sample_count);
    if (!is_finite_float3(color_result))
        color_result = visualizer_background_color();
    result[tid.xy] = float4(color_result, 1);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void sphere_tracing_first_hit_aov_kernel(
    uint3 tid: SV_DispatchThreadID,
    uniform Material material,
    uniform uint2 camera_size,
    uniform uint2 pixel_offset,
    uniform uint sample_index_offset,
    uniform uint sample_count,
    uniform uint completed_sample_count,
    uniform uint direct_lighting_mode,
    uniform bool use_analytic_texture_filtering,
    uniform float texture_filter_width_scale,
    uniform bool clear_result,
    uniform RWTexture2D<float4> beauty_sum,
    uniform RWTexture2D<float4> beauty_residual,
    uniform RWTexture2D<float4> first_hit_non_delta_ibl_sum,
    uniform RWTexture2D<float4> first_hit_non_delta_ibl_residual,
    uniform RWTexture2D<float4> secondary_transport_sum,
    uniform RWTexture2D<float4> secondary_transport_residual,
    uniform RWTexture2D<float4> result,
    uniform RWTexture2D<float4> first_hit_non_delta_ibl,
    uniform RWTexture2D<float4> secondary_transport
)
{
    uint width, height;
    result.GetDimensions(width, height);

    if (tid.x >= width || tid.y >= height)
        return;

    set_print_enabled(false);

    let camera = SphereTraceCamera();
    const uint2 pixel = pixel_offset + tid.xy;
    if (pixel.x >= camera_size.x || pixel.y >= camera_size.y) {
        if (clear_result) {
            beauty_sum[tid.xy] = float4(0.0);
            beauty_residual[tid.xy] = float4(0.0);
            first_hit_non_delta_ibl_sum[tid.xy] = float4(0.0);
            first_hit_non_delta_ibl_residual[tid.xy] = float4(0.0);
            secondary_transport_sum[tid.xy] = float4(0.0);
            secondary_transport_residual[tid.xy] = float4(0.0);
        }
        const float4 background = float4(visualizer_background_color(), 1.0);
        result[tid.xy] = background;
        first_hit_non_delta_ibl[tid.xy] = background;
        secondary_transport[tid.xy] = background;
        return;
    }

    const bool use_nee = direct_lighting_uses_nee(direct_lighting_mode);
    const bool use_direct_hits = direct_lighting_uses_direct_hits(direct_lighting_mode);
    const bool use_mis = direct_lighting_uses_mis(direct_lighting_mode);
    const int batch_sample_count = max(1, int(sample_count));
    KahanSummer beauty = {};
    KahanSummer first_hit_non_delta_ibl_batch = {};
    KahanSummer secondary_transport_batch = {};
    for (int i = 0; i < batch_sample_count; ++i) {
        const uint global_sample_index = sample_index_offset + uint(i);
        var sg = SampleGenerator(pixel, global_sample_index);
        let uv = (float2(pixel) + next_float2(sg)) / float2(float(camera_size.x), float(camera_size.y));
        var ray = camera.get_ray(uv);
        var throughput = float3(1.0);
        SurfaceInteraction previous_si = {};
        var previous_pdf = 0.0;
        var previous_delta = false;
        var has_previous_scatter = false;
        var first_sample_may_hit_ibl = false;
        var first_sample_non_delta_reflection = false;

        for (int bounce = 0; bounce < MAX_BOUNCES; ++bounce) {
            float t;
            bool hit = intersect_sphere(ray, float3(0), 1, t, false);
            if (!hit) {
                if (bounce == 0 || !has_previous_scatter) {
                    let contribution = visualizer_background_color();
                    beauty.add(contribution);
                } else if (use_direct_hits) {
                    float3 contribution = throughput
                        * eval_environment_miss(
                                              previous_si,
                                              ray.dir,
                                              previous_pdf,
                                              previous_delta,
                                              direct_lighting_mode
                        );
                    if (is_finite_float3(contribution)) {
                        beauty.add(contribution);
                        if (bounce == 1 && first_sample_may_hit_ibl) {
                            if (first_sample_non_delta_reflection)
                                first_hit_non_delta_ibl_batch.add(contribution);
                            else
                                secondary_transport_batch.add(contribution);
                        } else {
                            secondary_transport_batch.add(contribution);
                        }
                    }
                }
                break;
            }

            if (bounce > 0)
                first_sample_may_hit_ibl = false;

            let si = setup_sphere_surface_interaction(ray, t);
            var lod_sampler = make_lod0_sphere_lod_sampler();
            if (bounce == 0) {
                lod_sampler = make_analytic_sphere_lod_sampler(
                    camera,
                    uv,
                    float2(float(camera_size.x), float(camera_size.y)),
                    ray,
                    t,
                    use_analytic_texture_filtering,
                    texture_filter_width_scale
                );
            }
            let material_instance = material.setup_material_instance(si, lod_sampler, MaterialInstanceHints::none);

            let emitted_radiance = material_instance.eval_emissive_radiance();
            if (is_finite_float3(emitted_radiance)) {
                let contribution = throughput * max(emitted_radiance, float3(0.0));
                if (is_finite_float3(contribution)) {
                    beauty.add(contribution);
                    if (bounce == 0)
                        first_hit_non_delta_ibl_batch.add(contribution);
                    else
                        secondary_transport_batch.add(contribution);
                }
            }

            if (use_nee) {
                DirectionSample light_sample = {};
                float3 light_weight = {};
                if (light_sampler.sample_direction(si, sg, light_sample, light_weight)
                    && analytic_sphere_environment_visible(si, light_sample.dir_ws)) {
                    float bsdf_pdf = material_instance.eval_pdf(si, light_sample.dir_ws);
                    float3 bsdf_eval = material_instance.eval(si, light_sample.dir_ws, sg);
                    if (bsdf_pdf > 0.0 && is_nonzero_finite_float3(bsdf_eval)) {
                        float w = use_mis ? mis_heuristic(1, light_sample.pdf, 1, bsdf_pdf) : 1;
                        float3 contribution = throughput * light_weight * bsdf_eval * w
                            * direct_lighting_nee_scale(direct_lighting_mode);
                        if (is_finite_float3(contribution)) {
                            beauty.add(contribution);
                            if (bounce == 0)
                                first_hit_non_delta_ibl_batch.add(contribution);
                            else
                                secondary_transport_batch.add(contribution);
                        }
                    }
                }
            }

            if (!use_direct_hits || bounce + 1 >= MAX_BOUNCES)
                break;

            BSDFSample sample = {};
            if (!material_instance.sample(si, sg, sample))
                break;
            if (!is_nonzero_finite_float3(sample.weight) || !is_finite_float3(sample.wo_ws))
                break;

            throughput *= sample.weight;
            if (!is_nonzero_finite_float3(throughput))
                break;

            if (bounce >= RUSSIAN_ROULETTE_START_BOUNCE) {
                let survival_probability = clamp(
                    max_component(abs(throughput)),
                    RUSSIAN_ROULETTE_MIN_SURVIVAL,
                    RUSSIAN_ROULETTE_MAX_SURVIVAL
                );
                if (next_float(sg) > survival_probability)
                    break;
                throughput /= survival_probability;
            }

            if (bounce == 0) {
                first_sample_may_hit_ibl = true;
                let sample_is_reflection = sample.has_lobe(LobeTypes::reflection);
                let sample_is_transmission = sample.has_lobe(LobeTypes::transmission);
                let sample_is_delta = sample.has_lobe(LobeTypes::delta);
                first_sample_non_delta_reflection = sample_is_reflection && !sample_is_delta && !sample_is_transmission;
            }

            previous_si = si;
            previous_pdf = sample.pdf;
            previous_delta = sample.has_lobe(LobeTypes::delta);
            has_previous_scatter = true;
            ray = Ray(offset_sphere_scatter_origin(si, sample.wo_ws), normalize(sample.wo_ws));
        }
    }

    float3 beauty_total = clear_result ? float3(0.0) : beauty_sum[tid.xy].xyz;
    float3 beauty_residual_value = clear_result ? float3(0.0) : beauty_residual[tid.xy].xyz;
    float3 first_hit_total = clear_result ? float3(0.0) : first_hit_non_delta_ibl_sum[tid.xy].xyz;
    float3 first_hit_residual_value = clear_result ? float3(0.0) : first_hit_non_delta_ibl_residual[tid.xy].xyz;
    float3 secondary_total = clear_result ? float3(0.0) : secondary_transport_sum[tid.xy].xyz;
    float3 secondary_residual_value = clear_result ? float3(0.0) : secondary_transport_residual[tid.xy].xyz;

    compensated_accumulate(beauty_total, beauty_residual_value, beauty.total);
    compensated_accumulate(first_hit_total, first_hit_residual_value, first_hit_non_delta_ibl_batch.total);
    compensated_accumulate(secondary_total, secondary_residual_value, secondary_transport_batch.total);

    const uint new_sample_count = completed_sample_count + sample_count;
    beauty_sum[tid.xy] = float4(beauty_total, 1.0);
    beauty_residual[tid.xy] = float4(beauty_residual_value, 1.0);
    first_hit_non_delta_ibl_sum[tid.xy] = float4(first_hit_total, 1.0);
    first_hit_non_delta_ibl_residual[tid.xy] = float4(first_hit_residual_value, 1.0);
    secondary_transport_sum[tid.xy] = float4(secondary_total, 1.0);
    secondary_transport_residual[tid.xy] = float4(secondary_residual_value, 1.0);

    float3 beauty_result = resolved_accumulation(beauty_total, beauty_residual_value, new_sample_count);
    if (!is_finite_float3(beauty_result))
        beauty_result = visualizer_background_color();
    result[tid.xy] = float4(beauty_result, 1);
    first_hit_non_delta_ibl[tid.xy]
        = float4(resolved_accumulation(first_hit_total, first_hit_residual_value, new_sample_count), 1);
    secondary_transport[tid.xy]
        = float4(resolved_accumulation(secondary_total, secondary_residual_value, new_sample_count), 1);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void dense_direct_kernel(
    uint3 tid: SV_DispatchThreadID,
    uniform Material material,
    uniform int direct_light_count,
    uniform bool use_analytic_texture_filtering,
    uniform float texture_filter_width_scale,
    uniform RWTexture2D<float4> result
)
{
    uint width, height;
    result.GetDimensions(width, height);

    if (tid.x >= width || tid.y >= height)
        return;

    set_print_enabled(false);

    let camera = SphereTraceCamera();
    let raster_aligned_uv = (float2(tid.xy) + 0.5) / float2(width, height);
    let uv = float2(raster_aligned_uv.x, 1.0 - raster_aligned_uv.y);
    let ray = camera.get_ray(uv);

    float t;
    bool hit = intersect_sphere(ray, float3(0), 1, t, true);
    if (!hit) {
        result[tid.xy] = float4(visualizer_background_color(), 1);
        return;
    }

    let si = setup_sphere_surface_interaction(ray, t);
    let lod_sampler = make_analytic_sphere_lod_sampler(
        camera,
        uv,
        float2(width, height),
        ray,
        t,
        use_analytic_texture_filtering,
        texture_filter_width_scale
    );
    let material_instance = material.setup_material_instance(si, lod_sampler, MaterialInstanceHints::none);

    var color = float3(0.0);
    let emitted_radiance = material_instance.eval_emissive_radiance();
    if (is_finite_float3(emitted_radiance))
        color += max(emitted_radiance, float3(0.0));

    let clamped_light_count = clamp(direct_light_count, 1, 65536);
    let weight = dense_direct_probe_weight(clamped_light_count);
    for (int i = 0; i < clamped_light_count; ++i) {
        var sg = SampleGenerator(tid.xy, uint(i));
        let wo_ws = dense_direct_probe_direction(i, clamped_light_count);
        let contribution = material_instance.eval(si, wo_ws, sg) * weight;
        if (is_finite_float3(contribution))
            color += contribution;
    }

    result[tid.xy] = float4(color, 1);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void dense_env_direct_kernel(
    uint3 tid: SV_DispatchThreadID,
    uniform Material material,
    uniform int direct_light_count,
    uniform bool use_env_grid,
    uniform int env_grid_size,
    uniform bool use_env_importance,
    uniform bool use_env_tile,
    uniform int env_width,
    uniform int env_height,
    uniform int env_tile_start,
    uniform int env_tile_count,
    uniform bool flip_camera_y,
    uniform bool flip_env_x,
    uniform bool flip_env_y,
    uniform bool flip_env_z,
    uniform bool use_analytic_texture_filtering,
    uniform float texture_filter_width_scale,
    uniform RWTexture2D<float4> result
)
{
    uint width, height;
    result.GetDimensions(width, height);

    if (tid.x >= width || tid.y >= height)
        return;

    set_print_enabled(false);

    let camera = SphereTraceCamera();
    let raster_aligned_uv = (float2(tid.xy) + 0.5) / float2(width, height);
    let uv = flip_camera_y ? float2(raster_aligned_uv.x, 1.0 - raster_aligned_uv.y) : raster_aligned_uv;
    let ray = camera.get_ray(uv);

    float t;
    bool hit = intersect_sphere(ray, float3(0), 1, t, true);
    if (!hit) {
        result[tid.xy] = float4(visualizer_background_color(), 1);
        return;
    }

    let si = setup_sphere_surface_interaction(ray, t);
    let lod_sampler = make_analytic_sphere_lod_sampler(
        camera,
        uv,
        float2(width, height),
        ray,
        t,
        use_analytic_texture_filtering,
        texture_filter_width_scale
    );
    let material_instance = material.setup_material_instance(si, lod_sampler, MaterialInstanceHints::none);

    var color = float3(0.0);
    let emitted_radiance = material_instance.eval_emissive_radiance();
    if (is_finite_float3(emitted_radiance))
        color += max(emitted_radiance, float3(0.0));

    let clamped_grid_size = clamp(env_grid_size, 1, 512);
    let clamped_env_width = max(env_width, 1);
    let clamped_env_height = max(env_height, 1);
    let env_texel_count = clamped_env_width * clamped_env_height;
    let clamped_tile_start = clamp(env_tile_start, 0, env_texel_count - 1);
    let clamped_tile_count = clamp(env_tile_count, 1, env_texel_count - clamped_tile_start);
    let clamped_light_count = use_env_tile ? clamped_tile_count
        : use_env_grid                     ? clamped_grid_size * clamped_grid_size
                                           : clamp(direct_light_count, 1, 65536);
    let fibonacci_weight = dense_direct_probe_weight(clamped_light_count);
    for (int i = 0; i < clamped_light_count; ++i) {
        var sg = SampleGenerator(use_env_importance ? uint2(uint(i), 0) : tid.xy, uint(i));
        float3 wo_ws = float3(0.0);
        float3 weighted_env_radiance = float3(0.0);
        if (use_env_importance) {
            DirectionSample light_sample = {};
            float3 light_weight = {};
            let light = g_scene.light_system.get_light(g_scene.light_system.environment_light_id);
            if (!light.sample_direction(si, sg, light_sample, light_weight))
                continue;
            wo_ws = light_sample.dir_ws;
            weighted_env_radiance = light_weight / float(clamped_light_count);
        } else {
            let env_index = clamped_tile_start + i;
            wo_ws = use_env_tile ? dense_env_texel_probe_direction(env_index, clamped_env_width, clamped_env_height)
                : use_env_grid   ? dense_env_grid_probe_direction(i, clamped_grid_size)
                                 : dense_direct_probe_direction(i, clamped_light_count);
            let env_weight = use_env_tile
                ? dense_env_texel_probe_weight(env_index, clamped_env_width, clamped_env_height)
                : use_env_grid ? dense_env_grid_probe_weight(i, clamped_grid_size)
                               : fibonacci_weight;
            let env_dir_ws = float3(
                flip_env_x ? -wo_ws.x : wo_ws.x,
                flip_env_y ? -wo_ws.y : wo_ws.y,
                flip_env_z ? -wo_ws.z : wo_ws.z
            );
            weighted_env_radiance
                = eval_environment_miss(si, env_dir_ws, 0.0, true, DIRECT_LIGHTING_MODE_DIRECT_HITS_ONLY) * env_weight;
        }
        let contribution = material_instance.eval(si, wo_ws, sg) * weighted_env_radiance;
        if (is_finite_float3(contribution))
            color += contribution;
    }

    result[tid.xy] = float4(color, 1);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void explicit_direct_kernel(
    uint3 tid: SV_DispatchThreadID,
    uniform Material material,
    StructuredBuffer<float4> direct_lights,
    uniform int direct_light_count,
    uniform bool flip_camera_y,
    uniform bool flip_light_x,
    uniform bool flip_light_y,
    uniform bool flip_light_z,
    uniform bool use_analytic_texture_filtering,
    uniform float texture_filter_width_scale,
    uniform RWTexture2D<float4> result
)
{
    uint width, height;
    result.GetDimensions(width, height);

    if (tid.x >= width || tid.y >= height)
        return;

    set_print_enabled(false);

    let camera = SphereTraceCamera();
    let raster_aligned_uv = (float2(tid.xy) + 0.5) / float2(width, height);
    let uv = flip_camera_y ? float2(raster_aligned_uv.x, 1.0 - raster_aligned_uv.y) : raster_aligned_uv;
    let ray = camera.get_ray(uv);

    float t;
    bool hit = intersect_sphere(ray, float3(0), 1, t, true);
    if (!hit) {
        result[tid.xy] = float4(visualizer_background_color(), 1);
        return;
    }

    let si = setup_sphere_surface_interaction(ray, t);
    let lod_sampler = make_analytic_sphere_lod_sampler(
        camera,
        uv,
        float2(width, height),
        ray,
        t,
        use_analytic_texture_filtering,
        texture_filter_width_scale
    );
    let material_instance = material.setup_material_instance(si, lod_sampler, MaterialInstanceHints::none);

    var color = float3(0.0);
    let emitted_radiance = material_instance.eval_emissive_radiance();
    if (is_finite_float3(emitted_radiance))
        color += max(emitted_radiance, float3(0.0));

    let clamped_light_count = clamp(direct_light_count, 1, 1048576);
    for (int i = 0; i < clamped_light_count; ++i) {
        var sg = SampleGenerator(tid.xy, uint(i));
        let light_direction_and_pad = direct_lights[i * 2];
        let light_color_and_pad = direct_lights[i * 2 + 1];
        let base_wo_ws = light_direction_and_pad.xyz;
        let wo_ws = normalize(float3(
            flip_light_x ? -base_wo_ws.x : base_wo_ws.x,
            flip_light_y ? -base_wo_ws.y : base_wo_ws.y,
            flip_light_z ? -base_wo_ws.z : base_wo_ws.z
        ));
        let weighted_radiance = max(light_color_and_pad.xyz, float3(0.0));
        let contribution = material_instance.eval(si, wo_ws, sg) * weighted_radiance;
        if (is_finite_float3(contribution))
            color += contribution;
    }

    result[tid.xy] = float4(color, 1);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void eval_plot_kernel(
    uint3 tid: SV_DispatchThreadID,
    uniform Material material,
    uniform float3 view_direction_ws,
    uniform bool use_analytic_texture_filtering,
    uniform float texture_filter_width_scale,
    uniform RWTexture2D<float4> result
)
{
    uint width, height;
    result.GetDimensions(width, height);

    if (tid.x >= width || tid.y >= height)
        return;

    set_print_enabled(false);

    let plot_uv = (float2(tid.xy) + 0.5) / float2(width, height);
    let disk = plot_uv * 2.0 - 1.0;
    let radius_squared = dot(disk, disk);
    if (radius_squared > 1.0) {
        result[tid.xy] = float4(visualizer_background_color(), 1);
        return;
    }

    let camera = SphereTraceCamera();
    let center_uv = float2(0.5, 0.5);
    let ray = camera.get_ray(center_uv);

    float t;
    bool hit = intersect_sphere(ray, float3(0), 1, t, true);
    if (!hit) {
        result[tid.xy] = float4(visualizer_background_color(), 1);
        return;
    }

    let si = setup_eval_plot_surface_interaction(view_direction_ws);
    let lod_sampler = make_analytic_sphere_lod_sampler(
        camera,
        center_uv,
        float2(width, height),
        ray,
        t,
        use_analytic_texture_filtering,
        texture_filter_width_scale
    );
    let material_instance = material.setup_material_instance(si, lod_sampler, MaterialInstanceHints::none);

    var sg = SampleGenerator(tid.xy, 0);
    let wo_ws = normalize(float3(disk.x, disk.y, sqrt(max(0.0, 1.0 - radius_squared))));
    var value = material_instance.eval(si, wo_ws, sg);
    if (!is_finite_float3(value))
        value = float3(0.0);
    let emitted_radiance = material_instance.eval_emissive_radiance();
    if (is_finite_float3(emitted_radiance))
        value += max(emitted_radiance, float3(0.0));
    result[tid.xy] = float4(max(value, float3(0.0)), 1);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void albedo_kernel(
    uint3 tid: SV_DispatchThreadID,
    uniform Material material,
    uniform bool use_analytic_texture_filtering,
    uniform float texture_filter_width_scale,
    uniform RWTexture2D<float4> result
)
{
    uint width, height;
    result.GetDimensions(width, height);

    if (tid.x >= width || tid.y >= height)
        return;

    set_print_enabled(false);

    let camera = SphereTraceCamera();
    let uv = (float2(tid.xy) + 0.5) / float2(width, height);
    let ray = camera.get_ray(uv);

    float t;
    bool hit = intersect_sphere(ray, float3(0), 1, t, true);
    if (!hit) {
        result[tid.xy] = float4(visualizer_background_color(), 1);
        return;
    }

    let si = setup_sphere_surface_interaction(ray, t);
    let lod_sampler = make_analytic_sphere_lod_sampler(
        camera,
        uv,
        float2(width, height),
        ray,
        t,
        use_analytic_texture_filtering,
        texture_filter_width_scale
    );
    let material_instance = material.setup_material_instance(si, lod_sampler, MaterialInstanceHints::none);
    let properties = material_instance.collect_properties(si);
    let albedo = properties.diffuse_reflection_albedo + properties.specular_reflection_albedo;
    result[tid.xy] = float4(albedo, 1);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void preview_properties_kernel(
    uint3 tid: SV_DispatchThreadID,
    uniform Material material,
    uniform bool use_analytic_texture_filtering,
    uniform float texture_filter_width_scale,
    uniform RWTexture2D<float4> albedo_result,
    uniform RWTexture2D<float4> emission_result
)
{
    uint width, height;
    albedo_result.GetDimensions(width, height);

    if (tid.x >= width || tid.y >= height)
        return;

    set_print_enabled(false);

    let camera = SphereTraceCamera();
    let uv = (float2(tid.xy) + 0.5) / float2(width, height);
    let ray = camera.get_ray(uv);

    float t;
    bool hit = intersect_sphere(ray, float3(0), 1, t, true);
    if (!hit) {
        let background = float4(visualizer_background_color(), 1);
        albedo_result[tid.xy] = background;
        emission_result[tid.xy] = background;
        return;
    }

    let si = setup_sphere_surface_interaction(ray, t);
    let lod_sampler = make_analytic_sphere_lod_sampler(
        camera,
        uv,
        float2(width, height),
        ray,
        t,
        use_analytic_texture_filtering,
        texture_filter_width_scale
    );
    let material_instance = material.setup_material_instance(si, lod_sampler, MaterialInstanceHints::none);
    let properties = material_instance.collect_properties(si);

    let albedo = properties.diffuse_reflection_albedo + properties.specular_reflection_albedo;
    albedo_result[tid.xy] = float4(albedo, 1);
    emission_result[tid.xy] = float4(properties.emission, 1);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void delta_only_kernel(
    uint3 tid: SV_DispatchThreadID,
    uniform Material material,
    uniform int spp,
    uniform bool use_analytic_texture_filtering,
    uniform float texture_filter_width_scale,
    uniform RWTexture2D<float4> result
)
{
    uint width, height;
    result.GetDimensions(width, height);

    if (tid.x >= width || tid.y >= height)
        return;

    set_print_enabled(false);

    let camera = SphereTraceCamera();
    KahanSummer color = {};
    for (int i = 0; i < spp; ++i) {
        var sg = SampleGenerator(tid.xy, i);
        let uv = (float2(tid.xy) + next_float2(sg)) / float2(width, height);
        var ray = camera.get_ray(uv);
        var throughput = float3(1.0);
        SurfaceInteraction previous_si = {};
        var previous_pdf = 0.0;
        var has_delta_scatter = false;

        for (int bounce = 0; bounce < MAX_BOUNCES; ++bounce) {
            float t;
            bool hit = intersect_sphere(ray, float3(0), 1, t, false);
            if (!hit) {
                if (has_delta_scatter) {
                    float3 contribution = throughput
                        * eval_environment_miss(
                                              previous_si,
                                              ray.dir,
                                              previous_pdf,
                                              true,
                                              DIRECT_LIGHTING_MODE_DIRECT_HITS_ONLY
                        );
                    if (is_finite_float3(contribution))
                        color.add(contribution);
                }
                break;
            }

            let si = setup_sphere_surface_interaction(ray, t);
            var lod_sampler = make_lod0_sphere_lod_sampler();
            if (bounce == 0) {
                lod_sampler = make_analytic_sphere_lod_sampler(
                    camera,
                    uv,
                    float2(width, height),
                    ray,
                    t,
                    use_analytic_texture_filtering,
                    texture_filter_width_scale
                );
            }
            let material_instance = material.setup_material_instance(si, lod_sampler, MaterialInstanceHints::none);

            BSDFSample sample = {};
            if (!material_instance.sample(si, sg, sample))
                break;
            if (!sample.has_lobe(LobeTypes::delta))
                break;
            if (!is_nonzero_finite_float3(sample.weight) || !is_finite_float3(sample.wo_ws))
                break;

            throughput *= sample.weight;
            if (!is_nonzero_finite_float3(throughput))
                break;

            if (bounce >= RUSSIAN_ROULETTE_START_BOUNCE) {
                let survival_probability = clamp(
                    max_component(abs(throughput)),
                    RUSSIAN_ROULETTE_MIN_SURVIVAL,
                    RUSSIAN_ROULETTE_MAX_SURVIVAL
                );
                if (next_float(sg) > survival_probability)
                    break;
                throughput /= survival_probability;
            }

            previous_si = si;
            previous_pdf = sample.pdf;
            has_delta_scatter = true;
            ray = Ray(offset_sphere_scatter_origin(si, sample.wo_ws), normalize(sample.wo_ws));
        }
    }

    float3 color_result = color.total / spp;
    if (!is_finite_float3(color_result))
        color_result = float3(0.0);
    result[tid.xy] = float4(color_result, 1);
}

[shader("compute")]
[numthreads(128, 1, 1)]
void material_instance_eval_sample_consistency_kernel(
    uint3 tid: SV_DispatchThreadID,
    uniform Material material,
    uniform uint sample_count,
    uniform uint samples_per_thread,
    uniform uint seed,
    uniform WTensor<float, 1> weight_max_errors,
    uniform WTensor<float, 1> pdf_max_errors,
    uniform WTensor<float, 1> weight_sum_errors,
    uniform WTensor<float, 1> pdf_sum_errors,
    uniform WTensor<float, 1> delta_weight_sum_luma,
    uniform WTensor<float, 1> delta_weight_max_luma,
    uniform WTensor<float, 1> worst_values,
    uniform AtomicTensor<uint, 1> counters
)
{
    set_print_enabled(false);

    let camera = SphereTraceCamera();
    let ray = camera.get_ray(float2(0.5, 0.5));
    float t;
    if (!intersect_sphere(ray, float3(0), 1, t, true)) {
        counters.add(1, samples_per_thread);
        return;
    }

    let si = setup_sphere_surface_interaction(ray, t);
    let lod_sampler = make_lod0_sphere_lod_sampler();
    let material_instance = material.setup_material_instance(si, lod_sampler, MaterialInstanceHints::none);

    uint2 v = block_cipher_tea(seed, tid.x);
    var sg = SampleGenerator(v.xy, seed);

    float max_weight_err = 0.0;
    float max_pdf_err = 0.0;
    float sum_weight_err = 0.0;
    float sum_pdf_err = 0.0;
    float sum_delta_weight_luma = 0.0;
    float max_delta_weight_luma = 0.0;

    float worst_weight_sample_pdf = 0.0;
    float worst_weight_eval_pdf = 0.0;
    float3 worst_weight_sample_weight = float3(0.0);
    float3 worst_weight_computed_weight = float3(0.0);
    float3 worst_weight_wo = float3(0.0);

    float worst_pdf_sample_pdf = 0.0;
    float worst_pdf_eval_pdf = 0.0;
    float3 worst_pdf_wo = float3(0.0);

    for (uint i = 0; i < samples_per_thread; ++i) {
        uint sample_index = tid.x * samples_per_thread + i;
        if (sample_index >= sample_count)
            break;

        BSDFSample sample = {};
        if (!material_instance.sample(si, sg, sample)) {
            counters.add(1, 1);
            continue;
        }

        if (sample.has_lobe(LobeTypes::delta)) {
            if (is_finite_float3(sample.weight)) {
                let delta_luma = dot(max(sample.weight, float3(0.0)), float3(0.2126, 0.7152, 0.0722));
                sum_delta_weight_luma += delta_luma;
                max_delta_weight_luma = max(max_delta_weight_luma, delta_luma);
            }
            counters.add(2, 1);
            continue;
        }

        if (sample.pdf < 0.0 || any(sample.weight < 0.0)) {
            counters.add(3, 1);
            continue;
        }

        if (!is_finite_float(sample.pdf) || !is_finite_float3(sample.weight) || !is_finite_float3(sample.wo_ws)) {
            counters.add(5, 1);
            continue;
        }

        float eval_pdf_value = material_instance.eval_pdf(si, sample.wo_ws);
        if (!is_finite_float(eval_pdf_value)) {
            counters.add(5, 1);
            continue;
        }

        if (sample.pdf <= 1e-8) {
            counters.add(6, 1);
            continue;
        }

        let pdf_err = abs(eval_pdf_value - sample.pdf) / max(abs(sample.pdf), 1e-8);
        sum_pdf_err += pdf_err;
        if (pdf_err > max_pdf_err) {
            max_pdf_err = pdf_err;
            worst_pdf_sample_pdf = sample.pdf;
            worst_pdf_eval_pdf = eval_pdf_value;
            worst_pdf_wo = sample.wo_ws;
        }

        if (eval_pdf_value <= 1e-8) {
            counters.add(4, 1);
            continue;
        }

        float3 eval_value = material_instance.eval(si, sample.wo_ws, sg);
        if (!is_finite_float3(eval_value)) {
            counters.add(5, 1);
            continue;
        }

        float3 computed_weight = eval_value / eval_pdf_value;
        if (!is_finite_float3(computed_weight)) {
            counters.add(5, 1);
            continue;
        }

        float weight_err = 0.0;
        for (int c = 0; c < 3; ++c) {
            let denom = max(abs(sample.weight[c]), 1e-8);
            weight_err = max(weight_err, abs(computed_weight[c] - sample.weight[c]) / denom);
        }
        sum_weight_err += weight_err;
        if (weight_err > max_weight_err) {
            max_weight_err = weight_err;
            worst_weight_sample_pdf = sample.pdf;
            worst_weight_eval_pdf = eval_pdf_value;
            worst_weight_sample_weight = sample.weight;
            worst_weight_computed_weight = computed_weight;
            worst_weight_wo = sample.wo_ws;
        }

        counters.add(0, 1);
    }

    weight_max_errors.store(tid.x, max_weight_err);
    pdf_max_errors.store(tid.x, max_pdf_err);
    weight_sum_errors.store(tid.x, sum_weight_err);
    pdf_sum_errors.store(tid.x, sum_pdf_err);
    delta_weight_sum_luma.store(tid.x, sum_delta_weight_luma);
    delta_weight_max_luma.store(tid.x, max_delta_weight_luma);

    let base_index = tid.x * 18;
    worst_values.store(base_index + 0, worst_weight_sample_pdf);
    worst_values.store(base_index + 1, worst_weight_eval_pdf);
    worst_values.store(base_index + 2, worst_weight_sample_weight.x);
    worst_values.store(base_index + 3, worst_weight_sample_weight.y);
    worst_values.store(base_index + 4, worst_weight_sample_weight.z);
    worst_values.store(base_index + 5, worst_weight_computed_weight.x);
    worst_values.store(base_index + 6, worst_weight_computed_weight.y);
    worst_values.store(base_index + 7, worst_weight_computed_weight.z);
    worst_values.store(base_index + 8, worst_weight_wo.x);
    worst_values.store(base_index + 9, worst_weight_wo.y);
    worst_values.store(base_index + 10, worst_weight_wo.z);
    worst_values.store(base_index + 11, worst_pdf_sample_pdf);
    worst_values.store(base_index + 12, worst_pdf_eval_pdf);
    worst_values.store(base_index + 13, worst_pdf_wo.x);
    worst_values.store(base_index + 14, worst_pdf_wo.y);
    worst_values.store(base_index + 15, worst_pdf_wo.z);
    worst_values.store(base_index + 16, si.uv.x);
    worst_values.store(base_index + 17, si.uv.y);
}

float3 material_instance_spherical_domain_forward(float2 p)
{
    let cos_theta = -p.y;
    let sin_theta = safe_sqrt(max(0.0, 1.0 - cos_theta * cos_theta));
    float sin_phi;
    float cos_phi;
    sincos(p.x, sin_phi, cos_phi);
    return float3(cos_phi * sin_theta, sin_phi * sin_theta, cos_theta);
}

float2 material_instance_spherical_domain_backward(float3 p)
{
    return float2(atan2(p.y, p.x), -p.z);
}

[shader("compute")]
[numthreads(128, 1, 1)]
void material_instance_sample_histogram_kernel(
    uint3 tid: SV_DispatchThreadID,
    uniform Material material,
    uniform uint sample_count,
    uniform uint samples_per_thread,
    uniform uint seed,
    uniform AtomicTensor<uint, 2> histogram,
    uniform AtomicTensor<uint, 1> counters
)
{
    set_print_enabled(false);

    let camera = SphereTraceCamera();
    let ray = camera.get_ray(float2(0.5, 0.5));
    float t;
    if (!intersect_sphere(ray, float3(0), 1, t, true)) {
        counters.add(1, samples_per_thread);
        return;
    }

    let si = setup_sphere_surface_interaction(ray, t);
    let lod_sampler = make_lod0_sphere_lod_sampler();
    let material_instance = material.setup_material_instance(si, lod_sampler, MaterialInstanceHints::none);

    uint2 v = block_cipher_tea(seed, tid.x);
    var sg = SampleGenerator(v.xy, seed);

    let bounds_min = float2(-float.PI, -1.0);
    let bounds_max = float2(float.PI, 1.0);
    let extents = bounds_max - bounds_min;

    for (uint i = 0; i < samples_per_thread; ++i) {
        uint sample_index = tid.x * samples_per_thread + i;
        if (sample_index >= sample_count)
            break;

        BSDFSample sample = {};
        if (!material_instance.sample(si, sg, sample)) {
            counters.add(1, 1);
            continue;
        }

        if (sample.has_lobe(LobeTypes::delta)) {
            counters.add(2, 1);
            continue;
        }

        if (!is_finite_float3(sample.wo_ws)) {
            counters.add(4, 1);
            continue;
        }

        let xy = material_instance_spherical_domain_backward(normalize(sample.wo_ws));
        let eps = extents * 1e-4;
        if (any((xy < bounds_min - eps) | (xy > bounds_max + eps))) {
            counters.add(3, 1);
            continue;
        }

        let normalized_xy = clamp((xy - bounds_min) / extents, 0.0, 1.0);
        let shape = int2(histogram.shape[1], histogram.shape[0]);
        int2 index = int2(normalized_xy * float2(shape));
        index = min(index, shape - 1);
        histogram.add(index, 1);
        counters.add(0, 1);
    }
}

[shader("compute")]
[numthreads(8, 8, 1)]
void material_instance_pdf_grid_kernel(
    uint3 tid: SV_DispatchThreadID,
    uniform Material material,
    uniform uint2 res,
    uniform uint integration_res,
    uniform WTensor<float, 2> pdf
)
{
    if (tid.x >= res.x || tid.y >= res.y)
        return;

    set_print_enabled(false);

    let camera = SphereTraceCamera();
    let ray = camera.get_ray(float2(0.5, 0.5));
    float t;
    if (!intersect_sphere(ray, float3(0), 1, t, true)) {
        pdf.store(tid.xy, 0.0);
        return;
    }

    let si = setup_sphere_surface_interaction(ray, t);
    let lod_sampler = make_lod0_sphere_lod_sampler();
    let material_instance = material.setup_material_instance(si, lod_sampler, MaterialInstanceHints::none);

    let ires = max(integration_res, 2);
    let bounds_min = float2(-float.PI, -1.0);
    let bounds_max = float2(float.PI, 1.0);
    let bin_size = (bounds_max - bounds_min) / float2(res);
    let step = bin_size / float(ires - 1);
    let bin_lo = bounds_min + bin_size * float2(tid.xy);

    float sum = 0.0;
    for (uint j = 0; j < ires; ++j) {
        let wy = (j == 0 || j == ires - 1) ? 0.5 : 1.0;
        for (uint i = 0; i < ires; ++i) {
            let wx = (i == 0 || i == ires - 1) ? 0.5 : 1.0;
            let p = bin_lo + step * float2(i, j);
            let wo = material_instance_spherical_domain_forward(p);
            let value = material_instance.eval_pdf(si, wo);
            if (is_finite_float(value) && value > 0.0)
                sum += wx * wy * value;
        }
    }

    pdf.store(tid.xy, sum * step.x * step.y);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void full_sphere_eval_pdf_kernel(
    uint3 tid: SV_DispatchThreadID,
    uniform Material material,
    uniform float3 view_direction_ws,
    uniform bool use_analytic_texture_filtering,
    uniform float texture_filter_width_scale,
    uniform uint pdf_integration_res,
    uniform RWTexture2D<float4> eval_result,
    uniform RWTexture2D<float4> pdf_result
)
{
    uint width, height;
    eval_result.GetDimensions(width, height);

    if (tid.x >= width || tid.y >= height)
        return;

    set_print_enabled(false);

    let camera = SphereTraceCamera();
    let center_uv = float2(0.5, 0.5);
    let ray = camera.get_ray(center_uv);

    float t;
    bool hit = intersect_sphere(ray, float3(0), 1, t, true);
    if (!hit) {
        eval_result[tid.xy] = float4(0.0);
        pdf_result[tid.xy] = float4(0.0);
        return;
    }

    let si = setup_eval_plot_surface_interaction(view_direction_ws);
    let lod_sampler = make_analytic_sphere_lod_sampler(
        camera,
        center_uv,
        float2(width, height),
        ray,
        t,
        use_analytic_texture_filtering,
        texture_filter_width_scale
    );
    let material_instance = material.setup_material_instance(si, lod_sampler, MaterialInstanceHints::none);

    let uv = (float2(tid.xy) + 0.5) / float2(width, height);
    let p = float2(-float.PI, -1.0) + uv * float2(float.TWO_PI, 2.0);
    let wo_ws = material_instance_spherical_domain_forward(p);

    var sg = SampleGenerator(tid.xy, 0);
    let eval_value = material_instance.eval(si, wo_ws, sg);
    let integration_res = max(pdf_integration_res, uint(1));
    float pdf_sum = 0.0;
    uint pdf_count = 0;
    for (uint j = 0; j < integration_res; ++j) {
        for (uint i = 0; i < integration_res; ++i) {
            let sub_uv = (float2(tid.xy) + (float2(i, j) + 0.5) / float(integration_res)) / float2(width, height);
            let sub_p = float2(-float.PI, -1.0) + sub_uv * float2(float.TWO_PI, 2.0);
            let sub_wo_ws = material_instance_spherical_domain_forward(sub_p);
            let sub_pdf = material_instance.eval_pdf(si, sub_wo_ws);
            if (is_finite_float(sub_pdf) && sub_pdf >= 0.0) {
                pdf_sum += sub_pdf;
                ++pdf_count;
            }
        }
    }
    let pdf_value = pdf_count > 0 ? pdf_sum / float(pdf_count) : 0.0;
    eval_result[tid.xy] = float4(is_finite_float3(eval_value) ? max(eval_value, float3(0.0)) : float3(0.0), 1.0);
    pdf_result[tid.xy] = float4(pdf_value, 0.0, 0.0, 1.0);
}

[shader("compute")]
[numthreads(128, 1, 1)]
void full_sphere_sample_histogram_kernel(
    uint3 tid: SV_DispatchThreadID,
    uniform Material material,
    uniform uint sample_count,
    uniform uint samples_per_thread,
    uniform uint seed,
    uniform float3 view_direction_ws,
    uniform bool use_analytic_texture_filtering,
    uniform float texture_filter_width_scale,
    uniform AtomicTensor<uint, 2> histogram,
    uniform AtomicTensor<uint, 1> counters,
    uniform WTensor<float, 1> delta_weight_sum_luma,
    uniform WTensor<float, 1> delta_weight_max_luma
)
{
    set_print_enabled(false);

    let camera = SphereTraceCamera();
    let center_uv = float2(0.5, 0.5);
    let ray = camera.get_ray(center_uv);

    float t;
    bool hit = intersect_sphere(ray, float3(0), 1, t, true);
    if (!hit) {
        counters.add(1, samples_per_thread);
        delta_weight_sum_luma.store(tid.x, 0.0);
        delta_weight_max_luma.store(tid.x, 0.0);
        return;
    }

    let si = setup_eval_plot_surface_interaction(view_direction_ws);
    let lod_sampler = make_analytic_sphere_lod_sampler(
        camera,
        center_uv,
        float2(histogram.shape[1], histogram.shape[0]),
        ray,
        t,
        use_analytic_texture_filtering,
        texture_filter_width_scale
    );
    let material_instance = material.setup_material_instance(si, lod_sampler, MaterialInstanceHints::none);

    let bounds_min = float2(-float.PI, -1.0);
    let bounds_max = float2(float.PI, 1.0);
    let extents = bounds_max - bounds_min;
    float delta_sum = 0.0;
    float delta_max = 0.0;

    for (uint i = 0; i < samples_per_thread; ++i) {
        uint sample_index = tid.x * samples_per_thread + i;
        if (sample_index >= sample_count)
            break;

        // Keep the TEA8 + Numerical Recipes LCG stream stable for full-sphere
        // MX139 histogram probes.
        var sg = SampleGenerator(block_cipher_tea(seed, sample_index, 8).x);

        BSDFSample sample = {};
        if (!material_instance.sample(si, sg, sample)) {
            counters.add(1, 1);
            continue;
        }

        if (!is_finite_float(sample.pdf) || !is_finite_float3(sample.weight) || !is_finite_float3(sample.wo_ws)) {
            counters.add(4, 1);
            continue;
        }

        if (sample.pdf < 0.0 || any(sample.weight < 0.0)) {
            counters.add(3, 1);
            continue;
        }

        if (sample.has_lobe(LobeTypes::delta)) {
            let delta_luma = dot(max(sample.weight, float3(0.0)), float3(0.2126, 0.7152, 0.0722));
            delta_sum += delta_luma;
            delta_max = max(delta_max, delta_luma);
            counters.add(2, 1);
            continue;
        }

        let xy = material_instance_spherical_domain_backward(normalize(sample.wo_ws));
        let eps = extents * 1e-4;
        if (any((xy < bounds_min - eps) | (xy > bounds_max + eps))) {
            counters.add(5, 1);
            continue;
        }

        let normalized_xy = clamp((xy - bounds_min) / extents, 0.0, 1.0);
        let shape = int2(histogram.shape[1], histogram.shape[0]);
        int2 index = int2(normalized_xy * float2(shape));
        index = min(index, shape - 1);
        histogram.add(index, 1);
        counters.add(0, 1);
    }

    delta_weight_sum_luma.store(tid.x, delta_sum);
    delta_weight_max_luma.store(tid.x, delta_max);
}
