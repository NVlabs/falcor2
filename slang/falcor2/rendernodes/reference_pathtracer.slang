// SPDX-License-Identifier: Apache-2.0

module reference;

#include "falcor2/render/scene_ray_tracing.slangh"

// TODO(scene) needed to get access to render types for type conformances (i.e. ILight)
__exported import falcor2.render;
import falcor2.utils;

import slangpy;

// Configuration externs

/// Enable next event estimation.
extern static const bool ENABLE_NEE = false;
/// Enable multiple importance sampling.
extern static const bool ENABLE_MIS = true;
/// Enable analytic lights.
extern static const bool ENABLE_ANALYTIC_LIGHTS = true;
/// Enable environment light.
extern static const bool ENABLE_ENVIRONMENT_LIGHT = true;
/// Enable emissive triangles.
extern static const bool ENABLE_EMISSIVE_TRIANGLES = true;
/// Use the environment map for primary-ray background visibility.
extern static const bool ENV_MAP_AS_BACKGROUND = true;
/// Background color when the environment map is hidden from the background.
extern static const float3 BACKGROUND_COLOR = float3(0.0, 0.0, 0.0);
/// Maximum path depth.
// (1 = directly visible emission, 2 = direct lighting, 3 = one indirect bounce, etc.)
extern static const uint MAX_DEPTH = 3;
static LightSampler<ENABLE_ANALYTIC_LIGHTS, ENABLE_ENVIRONMENT_LIGHT, ENABLE_EMISSIVE_TRIANGLES> light_sampler;

// Path state

struct PathState : IDifferentiable
{
    no_diff UniformSampleGenerator sg;

    float3 scatter_pos_ws;
    float3 scatter_dir_ws;
    float3 scatter_normal_ws;
    float scatter_pdf;

    float3 throughput;
    float3 radiance;
};

// Generate a ray from interaction point to randomly sampled light
Ray create_visibility_ray(const SurfaceInteraction ref, const DirectionSample sample)
{
    Ray ray;
    ray.origin = offset_ray_origin(ref.position_ws, ref.normal_ws);
    ray.dir = sample.dir_ws;
    ray.t_min = 0.0;
    ray.t_max = sample.distance;
    if (sample.emitter_type == EmitterType::emissive_triangle) {
        float3 target = offset_ray_origin(sample.pos_ws, sample.normal_ws);
        float3 to_light = target - ray.origin;
        ray.t_max = length(to_light);
        ray.dir = to_light / ray.t_max;
    }
    return ray;
}

//------------------------------------------
// Guide output

// The reference path tracer writes optional guide outputs through this interface.
// Python injects a WriteGuide implementation in the generated prelude.
[__AttributeUsage(_AttributeTargets.Function)]
struct OutputSpecAttribute
{
    string format;
    string clear_value;
}

// This interface defines what outputs can be made, and provides additional metadata to describe
// how to initialize them. The generated prelude contains an implementation of this interface
// that writes to the actual guide outputs.
public interface IWriteGuide
{
    [OutputSpec("rgba16_float", "0,0,0,1")]
    public static void diffuse_albedo(uint2 coord, float4 value);
    public static bool diffuse_albedo_enabled();
    [OutputSpec("rgba16_float", "0,0,0,1")]
    public static void material_color(uint2 coord, float4 value);
    public static bool material_color_enabled();
    [OutputSpec("rgba16_float", "0,0,0,1")]
    public static void specular_albedo(uint2 coord, float4 value);
    public static bool specular_albedo_enabled();
    [OutputSpec("rgba16_float", "0,0,1,0")]
    public static void normals(uint2 coord, float4 value);
    public static bool normals_enabled();
    [OutputSpec("r16_float", "1")]
    public static void roughness(uint2 coord, float value);
    public static bool roughness_enabled();
    [OutputSpec("r32_float", "0")]
    public static void depth(uint2 coord, float value);
    public static bool depth_enabled();
    [OutputSpec("r32_float", "1")]
    public static void hardware_depth(uint2 coord, float value);
    public static bool hardware_depth_enabled();
    [OutputSpec("r16_float", "0")]
    public static void specular_hit_distance(uint2 coord, float value);
    public static bool specular_hit_distance_enabled();
    [OutputSpec("rg16_float", "0,0")]
    public static void motion_vectors(uint2 coord, float2 value);
    public static bool motion_vectors_enabled();
    [OutputSpec("rgba16_float", "0,0,0,0")]
    public static void emission(uint2 coord, float4 value);
    public static bool emission_enabled();
    [OutputSpec("rgba32_uint", "0xffffffff,0xffffffff,0xffffffff,0")]
    public static void geometry_id(uint2 coord, uint4 value);
    public static bool geometry_id_enabled();
}

// The prelude also exports this WriteGuide type to be the implementation it generates.
extern struct WriteGuide : IWriteGuide;

// These link-time constants are also generated in the prelude. They let the
// shader specialize away guide work when a corresponding output is not bound.
// They don't have to be exposed by the user, but if they are, they are configured
// correctly.
extern static const bool WriteGuide_diffuse_albedo = false;
extern static const bool WriteGuide_material_color = false;
extern static const bool WriteGuide_specular_albedo = false;
extern static const bool WriteGuide_normals = false;
extern static const bool WriteGuide_roughness = false;
extern static const bool WriteGuide_depth = false;
extern static const bool WriteGuide_hardware_depth = false;
extern static const bool WriteGuide_specular_hit_distance = false;
extern static const bool WriteGuide_motion_vectors = false;
extern static const bool WriteGuide_emission = false;
extern static const bool WriteGuide_geometry_id = false;

// Path tracing function (adapted from scene_test path_tracer.slang)

float3 trace_path<SceneIntersector : ISceneIntersector>(
    SceneIntersector intersector,
    no_diff HitInfo primary_hit,
    no_diff Ray ray,
    no_diff uint2 id,
    no_diff UniformSampleGenerator sg,
    Camera current_camera,
    Camera previous_camera
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

    HitInfo hit = primary_hit;
    float3 primary_specular_origin_ws = {};
    bool primary_specular_hit = false;

    uint depth = 1;
    [MaxIters(MAX_DEPTH)]
    while (depth <= MAX_DEPTH) {
        bool is_primary_hit = depth == 1;

        // Handle missed rays.
        if (!hit.valid) {
            bool has_environment_light = g_scene.has_environment_light();
            bool show_environment_background = has_environment_light && is_primary_hit && ENV_MAP_AS_BACKGROUND;
            bool handle_environment_light
                = has_environment_light && !is_primary_hit && ENABLE_ENVIRONMENT_LIGHT && (!ENABLE_NEE || ENABLE_MIS);
            if (show_environment_background || handle_environment_light) {
                SurfaceInteraction scatter_si = {};
                scatter_si.position_ws = path.scatter_pos_ws;
                scatter_si.normal_ws = path.scatter_normal_ws;
                DirectionSample sample = {};
                sample.dir_ws = ray.dir;
                sample.emitter_type = EmitterType::environment;
                float mis_weight = 1.0;
                if (handle_environment_light && ENABLE_NEE && ENABLE_MIS) {
                    float light_pdf = light_sampler.eval_direction_pdf(scatter_si, sample);
                    mis_weight = mis_heuristic(1, path.scatter_pdf, 1, light_pdf);
                }
                let light = g_scene.light_system.get_light(g_scene.light_system.environment_light_id);
                let eval = light.eval_direction(scatter_si, sample);
                path.radiance += path.throughput * eval * mis_weight;
            } else if (is_primary_hit) {
                path.radiance += BACKGROUND_COLOR;
            }
            break;
        }

        // Make surface interaction at hit point.
        SurfaceInteraction si = g_scene.make_surface_interaction(hit, -ray.dir);

        // Ignore backface.
        if (!si.front_facing)
            break;
        if (is_primary_hit)
            write_primary_surface_interaction(id, hit, si, current_camera, previous_camera);

        // If the previous hit was the primary specular hit, record the distance between that and
        // this next hit, to get the 'specular hit distance' used for DLSS-RR
        if (WriteGuide::specular_hit_distance_enabled() && primary_specular_hit) {
            write_specular_hit_distance(id, length(si.position_ws - primary_specular_origin_ws));
            primary_specular_hit = false;
        }

        // Make material instance.
        let lod_sampler = ExplicitLodSampler(0);
        let material_instance = g_scene.get_material_instance(hit, si, lod_sampler);
        if (is_primary_hit)
            write_primary_material(id, si, material_instance);

        // Add emissive triangle contribution.
        bool handle_emission = ENABLE_EMISSIVE_TRIANGLES && g_scene.has_emissive_triangles()
            && (is_primary_hit || !ENABLE_NEE || ENABLE_MIS);
        if (handle_emission && any(material_instance.is_emissive())) {
            float mis_weight = 1.0;
            if (!is_primary_hit && ENABLE_NEE && ENABLE_MIS) {
                SurfaceInteraction scatter_si = {};
                scatter_si.position_ws = path.scatter_pos_ws;
                scatter_si.normal_ws = path.scatter_normal_ws;
                DirectionSample sample = {};
                sample.pos_ws = si.position_ws;
                sample.normal_ws = si.normal_ws;
                sample.dir_ws = ray.dir;
                sample.emitter_type = EmitterType::emissive_triangle;
                sample.primitive_id
                    = uint(g_scene.emissive_geometry.get_triangle_id(hit.geometry_instance_id, hit.primitive_index));
                float light_pdf = light_sampler.eval_direction_pdf(scatter_si, sample);
                mis_weight = mis_heuristic(1, path.scatter_pdf, 1, light_pdf);
            }
            path.radiance += path.throughput * material_instance.eval_emissive_radiance() * mis_weight;
        }

        // Add NEE (next event estimation) contribution.
        if (ENABLE_NEE && depth < MAX_DEPTH) {
            DirectionSample sample = {};
            float3 weight = {};
            if (light_sampler.sample_direction(si, sg, sample, weight)) {
                if (!intersector.is_occluded(create_visibility_ray(si, sample))) {
                    float mis_weight = 1.0;
                    if (ENABLE_MIS && !sample.delta) {
                        float scatter_pdf = material_instance.eval_pdf(si, sample.dir_ws);
                        mis_weight = mis_heuristic(1, sample.pdf, 1, scatter_pdf);
                    }
                    path.radiance
                        += path.throughput * material_instance.eval(si, sample.dir_ws, sg) * weight * mis_weight;
                }
            }
        }

        // Sample BRDF to generate next path segment.
        BSDFSample sample = {};
        if (material_instance.sample(si, sg, sample)) {
            path.throughput *= sample.weight;
        } else {
            break;
        }

        // Track whether the primary ray hit a specular surface and sampled a specular lobe,
        // for use in capturing specular hit distance.
        if (WriteGuide::specular_hit_distance_enabled() && is_primary_hit) {
            bool sampled_primary_specular_reflection
                = sample.has_lobe(LobeTypes::glossy_reflection) || sample.has_lobe(LobeTypes::delta_reflection);
            if (sampled_primary_specular_reflection) {
                primary_specular_origin_ws = si.position_ws;
                primary_specular_hit = true;
            }
        }

        path.scatter_pos_ws = offset_ray_origin(si.position_ws, si.normal_ws);
        path.scatter_dir_ws = sample.wo_ws;
        path.scatter_normal_ws = si.normal_ws;
        path.scatter_pdf = sample.pdf;

        // Generate next ray.
        ray = Ray(path.scatter_pos_ws, path.scatter_dir_ws);
        depth++;

        hit = intersector.intersect(ray);
    }

    return path.radiance;
}

// Ray tracing intersector (adapted from scene_test.slang)

[raypayload]
public struct IntersectorPayload
{
    HitInfo hit : read(caller)
        : write(closesthit);

    __init() { hit = {}; }
};

public struct SceneIntersector : ISceneIntersector
{
    public HitInfo intersect(Ray ray)
    {
        IntersectorPayload payload = {};
        TraceRay(
            g_scene.tlas,
            0,
            0xff,
            0 /* RayContributionToHitGroupIndex */,
            0 /* MultiplierForGeometryContributionHitGroupIndex */,
            0 /* MissShaderIndex */,
            ray.to_ray_desc(),
            payload
        );
        return payload.hit;
    }

    public bool is_occluded(Ray ray)
    {
        HitInfo hit = intersect(ray);
        return hit.valid;
    }
};

// RT entry points

SCENE_RAY_TYPE_MISS(intersect, intersect_miss, IntersectorPayload);
void intersect_miss(inout IntersectorPayload payload) { }

SCENE_RAY_TYPE_CLOSEST_HIT(intersect, intersect_closest_hit, IntersectorPayload);
void intersect_closest_hit<D : IDeferredHitInfo>(inout IntersectorPayload payload, D deferred_hit)
{
    payload.hit = deferred_hit.create();
}

// Calculate primary ray UV coordinates.
//
// The base sample is pixel center. Random jitter and the explicit offset are
// center-relative, then clamped so the final sample remains inside the pixel.
float2 get_primary_uv(
    RaySampler ray_sampler,
    inout UniformSampleGenerator sg,
    float2 subpixel_offset,
    float subpixel_random_jitter
)
{
    float2 pixel_offset = float2(0.5, 0.5);
    if (subpixel_random_jitter != 0.0) {
        pixel_offset += (next_float2(sg) - 0.5) * subpixel_random_jitter;
    }
    pixel_offset = saturate(pixel_offset + subpixel_offset);

    float2 uv = (float2(ray_sampler.id) + pixel_offset) / float2(ray_sampler.camera.dims);
    uv.y = 1 - uv.y;
    return uv;
}

float4 read_diffuse_albedo(MaterialProperties properties)
{
    return float4(saturate(properties.diffuse_reflection_albedo), 1.0);
}

float4 read_specular_albedo(MaterialProperties properties)
{
    return float4(saturate(properties.specular_reflection_albedo), 1.0);
}

float4 read_normals(SurfaceInteraction si, MaterialProperties properties)
{
    float3 guide_normal = dot(properties.guide_normal, properties.guide_normal) > 0.0
        ? normalize(properties.guide_normal)
        : si.front_normal_ws();
    return float4(guide_normal, 0.0);
}

float read_roughness(MaterialProperties properties)
{
    return saturate(properties.roughness);
}

float read_depth(SurfaceInteraction si, Camera camera)
{
    return dot(si.position_ws - camera.position, camera.image_w);
}

float read_hardware_depth(SurfaceInteraction si, Camera camera)
{
    const float z_near = 0.1;
    const float z_far = 1000.0;
    float linear_depth = max(read_depth(si, camera), z_near);
    return saturate((z_far / (z_far - z_near)) * (1.0 - (z_near / linear_depth)));
}

void write_specular_hit_distance(uint2 coord, float distance)
{
    if (WriteGuide::specular_hit_distance_enabled())
        WriteGuide::specular_hit_distance(coord, distance);
}

void write_primary_material(uint2 coord, SurfaceInteraction si, IMaterialInstance material_instance)
{
    const bool needs_material_properties = WriteGuide::diffuse_albedo_enabled() || WriteGuide::specular_albedo_enabled()
        || WriteGuide::normals_enabled() || WriteGuide::roughness_enabled();
    if (needs_material_properties) {
        MaterialProperties properties = material_instance.collect_properties(si);
        if (WriteGuide::diffuse_albedo_enabled())
            WriteGuide::diffuse_albedo(coord, read_diffuse_albedo(properties));
        if (WriteGuide::specular_albedo_enabled())
            WriteGuide::specular_albedo(coord, read_specular_albedo(properties));
        if (WriteGuide::normals_enabled())
            WriteGuide::normals(coord, read_normals(si, properties));
        if (WriteGuide::roughness_enabled())
            WriteGuide::roughness(coord, read_roughness(properties));
    }
    if (WriteGuide::material_color_enabled())
        WriteGuide::material_color(coord, float4(saturate(material_instance.eval_material_color(si)), 1.0));
    if (WriteGuide::emission_enabled())
        WriteGuide::emission(coord, float4(material_instance.eval_emissive_radiance(), 1.0));
}

void write_primary_surface_interaction(
    uint2 coord,
    no_diff HitInfo hit,
    SurfaceInteraction si,
    Camera camera,
    Camera previous_camera
)
{
    if (WriteGuide::depth_enabled())
        WriteGuide::depth(coord, read_depth(si, camera));
    if (WriteGuide::hardware_depth_enabled())
        WriteGuide::hardware_depth(coord, read_hardware_depth(si, camera));
    if (WriteGuide::motion_vectors_enabled())
        WriteGuide::motion_vectors(
            coord,
            g_scene.get_motion_vector(camera, previous_camera, hit.geometry_instance_id, si.position_ws)
        );
    if (WriteGuide::geometry_id_enabled()) {
        let geometry_instance = g_scene.get_geometry_instance_desc(hit.geometry_instance_id);
        WriteGuide::geometry_id(
            coord,
            uint4(
                uint(hit.geometry_type),
                uint(hit.geometry_instance_id),
                hit.primitive_index,
                uint(geometry_instance.header.material_id)
            )
        );
    }
}

public void render(
    RaySampler ray_sampler,
    Camera previous_camera,
    out float4 output,
    uint iteration,
    float2 subpixel_offset,
    float subpixel_random_jitter
)
{
    uint2 id = ray_sampler.id;
    UniformSampleGenerator sg = UniformSampleGenerator(id, iteration);

    float2 uv = get_primary_uv(ray_sampler, sg, subpixel_offset, subpixel_random_jitter);
    Ray ray = ray_sampler.camera.get_ray(uv);

    SceneIntersector intersector;
    HitInfo hit = intersector.intersect(ray);
    float3 radiance = trace_path(intersector, hit, ray, id, sg, ray_sampler.camera, previous_camera);

    output = float4(radiance, 1.0);
}
