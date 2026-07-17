// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
/// Path scheduling implementation.
extern static const SchedulingMode SCHEDULING_MODE = SchedulingMode::simple;
/// Visibility traversal implementation.
extern static const VisibilityRayMode VISIBILITY_RAY_MODE = VisibilityRayMode::ray_query;

static LightSampler<ENABLE_ANALYTIC_LIGHTS, ENABLE_ENVIRONMENT_LIGHT, ENABLE_EMISSIVE_TRIANGLES> light_sampler;

public enum class SchedulingMode : uint {
    simple = 0,
    ser = 1,
};

public enum class VisibilityRayMode : uint {
    ray_query = 0,
    trace_ray = 1,
};

// Path state

struct PathVertex : IDifferentiable
{
    float3 position_ws = {};
    no_diff float3 normal_ws = {};
    no_diff bool front_facing = {};

    __init(const SurfaceInteraction si)
    {
        position_ws = si.position_ws;
        normal_ws = si.normal_ws;
        front_facing = si.front_facing;
    }

    float3 ray_origin_for_direction(const float3 dir_ws)
    {
        return offset_ray_origin(position_ws, dot(dir_ws, normal_ws) >= 0.0 ? normal_ws : -normal_ws);
    }

    float3 oriented_normal() { return front_facing ? normal_ws : -normal_ws; }
};

enum class HemisphereFlags {
    none = 0,
    upper = (1 << 0),
    lower = (1 << 1),
};

enum class PathFlags {
    none = 0,
    /// Path is active (not terminated by Russian roulette or max depth).
    active = (1 << 0),
    /// Last vertex sampled a delta reflection or transmission.
    delta = (1 << 1),
    /// Last vertex sampled a light using NEE (upper hemisphere).
    light_sampled_upper = (1 << 2),
    /// Last vertex sampled a light using NEE (lower hemisphere).
    light_sampled_lower = (1 << 3),
    /// Primary hit sampled a glossy/delta reflection lobe.
    primary_specular_reflection = (1 << 4),
};

struct PathState : IDifferentiable
{
    static const uint VERTEX_INDEX_BIT_COUNT = 16;
    static const uint VERTEX_INDEX_MASK = (1 << VERTEX_INDEX_BIT_COUNT) - 1;

    no_diff UniformSampleGenerator sg;
    no_diff uint path_id = {};
    no_diff uint flags_and_vertex_index = {};

    float3 ray_origin_ws = {};
    float3 ray_dir_ws = {};

    float3 scatter_normal_ws = {};
    float scatter_pdf = {};
    float3 throughput = float3(1.0);
    float3 radiance = {};

    __init(UniformSampleGenerator sg, uint path_id, Ray ray)
    {
        this.sg = sg;
        this.path_id = path_id;
        this.ray_origin_ws = ray.origin;
        this.ray_dir_ws = ray.dir;
        this.set_active();
    }

    /// Path flags.
    property PathFlags flags
    {
        get {
            return PathFlags(flags_and_vertex_index >> VERTEX_INDEX_BIT_COUNT);
        }
        set {
            flags_and_vertex_index
                = (flags_and_vertex_index & VERTEX_INDEX_MASK) | (uint(newValue) << VERTEX_INDEX_BIT_COUNT);
        }
    }

    /// Vertex index.
    /// 0 = camera, 1 = primary hit, 2 = first bounce, etc.
    property uint vertex_index
    {
        get {
            return flags_and_vertex_index & VERTEX_INDEX_MASK;
        }
        set {
            flags_and_vertex_index = (flags_and_vertex_index & ~VERTEX_INDEX_MASK) | (newValue & VERTEX_INDEX_MASK);
        }
    }

    // Flags

    /// Check if a specific flag is set.
    bool has_flag(const PathFlags flag)
    {
        const uint bit = uint(flag) << VERTEX_INDEX_BIT_COUNT;
        return (flags_and_vertex_index & bit) != 0;
    }

    /// Set or clear a specific flag.
    [mutating]
    void set_flag(const PathFlags flag, const bool value = true)
    {
        const uint bit = uint(flag) << VERTEX_INDEX_BIT_COUNT;
        if (value)
            flags_and_vertex_index |= bit;
        else
            flags_and_vertex_index &= ~bit;
    }

    bool is_active() { return has_flag(PathFlags::active); }
    bool is_delta() { return has_flag(PathFlags::delta); }
    bool is_light_sampled_upper() { return has_flag(PathFlags::light_sampled_upper); }
    bool is_light_sampled_lower() { return has_flag(PathFlags::light_sampled_lower); }
    bool is_light_sampled() { return has_flag(PathFlags::light_sampled_upper | PathFlags::light_sampled_lower); }
    bool is_primary_specular_reflection() { return has_flag(PathFlags::primary_specular_reflection); }

    [mutating]
    void set_active(bool value = true) { set_flag(PathFlags::active, value); }

    [mutating]
    void terminate() { set_active(false); }

    [mutating]
    void set_delta(bool value = true) { set_flag(PathFlags::delta, value); }

    [mutating]
    void set_light_sampled_upper(bool value = true) { set_flag(PathFlags::light_sampled_upper, value); }

    [mutating]
    void set_light_sampled_lower(bool value = true) { set_flag(PathFlags::light_sampled_lower, value); }

    [mutating]
    void set_primary_specular_reflection(bool value = true) { set_flag(PathFlags::primary_specular_reflection, value); }

    [mutating]
    void set_light_sampled(bool upper, bool lower)
    {
        set_light_sampled_upper(upper);
        set_light_sampled_lower(lower);
    }

    /// Helpers

    bool is_light_samplable() { return !is_delta(); }

    HemisphereFlags light_sampled_hemispheres()
    {
        HemisphereFlags result = HemisphereFlags::none;
        if (is_light_sampled_upper())
            result = result | HemisphereFlags::upper;
        if (is_light_sampled_lower())
            result = result | HemisphereFlags::lower;
        return result;
    }

    [mutating]
    void set_light_sampled_hemispheres(HemisphereFlags hemispheres)
    {
        set_light_sampled(hemispheres & HemisphereFlags::upper, hemispheres & HemisphereFlags::lower);
    }

    uint2 pixel_id(const uint2 dims) { return uint2(path_id % dims.x, path_id / dims.x); }
};

public struct TracePathContext
{
    Camera current_camera = {};
    Camera previous_camera = {};

    uint2 pixel_id(const PathState path) { return path.pixel_id(current_camera.dims); }
};

public ParameterBlock<TracePathContext> g_trace_path_context;

public interface IVisibilityQuery
{
    public bool is_occluded(Ray ray);
}

bool has_material_flag(const HitInfo hit, MaterialFlags flag)
{
    return (g_scene.get_material_header(hit).flags & flag) != MaterialFlags::none;
}

bool is_surface_visible(const HitInfo hit, const SurfaceInteraction si)
{
    if (si.front_facing)
        return true;
    return has_material_flag(hit, MaterialFlags::double_sided);
}

HemisphereFlags make_light_sampling_decision(const BSDFFlags lobe_types)
{
    HemisphereFlags result = HemisphereFlags::none;
    if (is_set(lobe_types, BSDFFlags::non_delta_reflection))
        result = result | HemisphereFlags::upper;
    if (is_set(lobe_types, BSDFFlags::non_delta_transmission))
        result = result | HemisphereFlags::lower;
    if (is_set(lobe_types, BSDFFlags::curve))
        result = result | HemisphereFlags::upper | HemisphereFlags::lower;
    return result;
}

bool is_direction_in_sampled_hemispheres(const float3 normal_ws, const HemisphereFlags hemispheres, const float3 dir_ws)
{
    bool upper = dot(dir_ws, normal_ws) >= 0.0;
    bool can_sample_upper = (hemispheres & HemisphereFlags::upper) != 0;
    bool can_sample_lower = (hemispheres & HemisphereFlags::lower) != 0;
    return (upper && can_sample_upper) || (!upper && can_sample_lower);
}

bool is_direction_in_sampled_hemispheres(
    const PathVertex vertex,
    const HemisphereFlags hemispheres,
    const float3 dir_ws
)
{
    return is_direction_in_sampled_hemispheres(vertex.oriented_normal(), hemispheres, dir_ws);
}

bool can_apply_light_mis(const PathState path, const float3 dir_ws)
{
    return path.is_light_samplable() && path.is_light_sampled()
        && is_direction_in_sampled_hemispheres(path.scatter_normal_ws, path.light_sampled_hemispheres(), dir_ws);
}

SurfaceInteraction make_scatter_interaction(const PathState path)
{
    SurfaceInteraction si = {};
    si.position_ws = path.ray_origin_ws;
    si.normal_ws = path.scatter_normal_ws;
    si.front_facing = true;
    si.wi_ws = -path.ray_dir_ws;
    return si;
}

// Generate a ray from interaction point to randomly sampled light
Ray create_visibility_ray(const PathVertex ref, const DirectionSample sample)
{
    Ray ray;
    ray.origin = ref.ray_origin_for_direction(sample.dir_ws);
    ray.dir = sample.dir_ws;
    ray.t_min = 0.0;
    ray.t_max = sample.distance;
    if (sample.emitter_type == EmitterType::emissive_triangle) {
        PathVertex target_vertex;
        target_vertex.position_ws = sample.pos_ws;
        target_vertex.normal_ws = sample.normal_ws;
        target_vertex.front_facing = true;
        float3 target = target_vertex.ray_origin_for_direction(-sample.dir_ws);
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

// Native callers do not bind guide outputs. Python can override this default with
// the generated prelude implementation.
public struct NoWriteGuide : IWriteGuide
{
    public static override void diffuse_albedo(uint2 coord, float4 value) { }
    public static bool diffuse_albedo_enabled() { return false; }

    public static override void material_color(uint2 coord, float4 value) { }
    public static bool material_color_enabled() { return false; }

    public static override void specular_albedo(uint2 coord, float4 value) { }
    public static bool specular_albedo_enabled() { return false; }

    public static override void normals(uint2 coord, float4 value) { }
    public static bool normals_enabled() { return false; }

    public static override void roughness(uint2 coord, float value) { }
    public static bool roughness_enabled() { return false; }

    public static override void depth(uint2 coord, float value) { }
    public static bool depth_enabled() { return false; }

    public static override void hardware_depth(uint2 coord, float value) { }
    public static bool hardware_depth_enabled() { return false; }

    public static override void specular_hit_distance(uint2 coord, float value) { }
    public static bool specular_hit_distance_enabled() { return false; }

    public static override void motion_vectors(uint2 coord, float2 value) { }
    public static bool motion_vectors_enabled() { return false; }

    public static override void emission(uint2 coord, float4 value) { }
    public static bool emission_enabled() { return false; }

    public static override void geometry_id(uint2 coord, uint4 value) { }
    public static bool geometry_id_enabled() { return false; }
};

// The prelude also exports this WriteGuide type to be the implementation it generates.
extern struct WriteGuide : IWriteGuide = NoWriteGuide;

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

void sample_direct_lighting<VisibilityQuery : IVisibilityQuery>(
    VisibilityQuery visibility_query,
    const SurfaceInteraction si,
    IMaterialInstance material_instance,
    inout PathState path
)
{
    path.set_light_sampled_hemispheres(HemisphereFlags::none);
    if (!ENABLE_NEE || path.vertex_index >= MAX_DEPTH)
        return;

    BSDFFlags lobe_types = material_instance.get_lobe_types(si);
    HemisphereFlags sampled_hemispheres = make_light_sampling_decision(lobe_types);
    path.set_light_sampled_hemispheres(sampled_hemispheres);
    if (sampled_hemispheres == HemisphereFlags::none)
        return;

    DirectionSample sample = {};
    float3 weight = {};
    PathVertex vertex = { si };
    if (light_sampler.sample_direction(si, path.sg, sample, weight)
        && is_direction_in_sampled_hemispheres(vertex, sampled_hemispheres, sample.dir_ws)) {
        if (!visibility_query.is_occluded(create_visibility_ray(vertex, sample))) {
            float mis_weight = 1.0;
            if (ENABLE_MIS && !sample.delta) {
                float scatter_pdf = material_instance.eval_pdf(si, sample.dir_ws);
                mis_weight = mis_heuristic(1, sample.pdf, 1, scatter_pdf);
            }
            path.radiance += path.throughput * material_instance.eval(si, sample.dir_ws, path.sg) * weight * mis_weight;
        }
    }
}

bool generate_scatter_ray(const SurfaceInteraction si, const MaterialSample sample, inout PathState path)
{
    path.throughput *= sample.weight;
    path.scatter_pdf = sample.pdf;
    path.scatter_normal_ws = si.front_normal_ws();
    path.set_delta(sample.has_flag(BSDFFlags::delta));
    const float3 offset_normal_ws = dot(sample.wo_ws, si.normal_ws) >= 0.0 ? si.normal_ws : -si.normal_ws;
    path.ray_origin_ws = offset_ray_origin(si.position_ws, offset_normal_ws);
    path.ray_dir_ws = sample.wo_ws;
    path.vertex_index += 1;

    return any(path.throughput > 0.0);
}

void handle_miss(inout PathState path)
{
    bool is_primary_miss = path.vertex_index == 0;
    bool has_environment_lights = g_scene.has_environment_lights();
    bool show_environment_background = has_environment_lights && is_primary_miss && ENV_MAP_AS_BACKGROUND;
    bool direct_light_sampled = can_apply_light_mis(path, path.ray_dir_ws);
    bool handle_environment_light = has_environment_lights && !is_primary_miss && ENABLE_ENVIRONMENT_LIGHT
        && (!ENABLE_NEE || ENABLE_MIS || !direct_light_sampled);

    if (show_environment_background || handle_environment_light) {
        SurfaceInteraction scatter_si = make_scatter_interaction(path);
        DirectionSample sample = {};
        sample.dir_ws = path.ray_dir_ws;
        sample.emitter_type = EmitterType::environment;
        float mis_weight = 1.0;
        if (handle_environment_light && ENABLE_NEE && ENABLE_MIS && direct_light_sampled) {
            float light_pdf = light_sampler.eval_direction_pdf(scatter_si, sample);
            mis_weight = mis_heuristic(1, path.scatter_pdf, 1, light_pdf);
        }
        let eval = g_scene.light_system.eval_environment_direction(scatter_si, sample);
        path.radiance += path.throughput * eval * mis_weight;
    } else if (is_primary_miss) {
        path.radiance += BACKGROUND_COLOR;
    }

    path.terminate();
}

void handle_hit<VisibilityQuery : IVisibilityQuery>(
    VisibilityQuery visibility_query,
    no_diff HitInfo hit,
    inout PathState path,
    const TracePathContext context
)
{
    bool is_primary_hit = path.vertex_index == 1;
    uint2 id = context.pixel_id(path);

    // Make surface interaction at hit point.
    SurfaceInteraction si = g_scene.make_surface_interaction(hit, -path.ray_dir_ws);

    // Ignore backface.
    if (!is_surface_visible(hit, si)) {
        path.terminate();
        return;
    }

    if (is_primary_hit)
        write_primary_surface_interaction(id, hit, si, context.current_camera, context.previous_camera);

    // If the primary hit sampled a specular reflection, record the scatter path length to this hit.
    if (WriteGuide::specular_hit_distance_enabled() && path.is_primary_specular_reflection()) {
        write_specular_hit_distance(id, length(si.position_ws - path.ray_origin_ws));
        path.set_primary_specular_reflection(false);
    }

    // Make material instance.
    let lod_sampler = ExplicitLodSampler(0);
    let material_instance
        = g_scene.get_material_instance(hit, si, lod_sampler, MaterialInstanceHints::adjust_shading_normal);
    if (is_primary_hit)
        write_primary_material(id, si, material_instance);

    // Add emissive triangle contribution.
    bool direct_light_sampled = can_apply_light_mis(path, path.ray_dir_ws);
    bool handle_emission = ENABLE_EMISSIVE_TRIANGLES && g_scene.has_emissive_triangles()
        && (is_primary_hit || !ENABLE_NEE || ENABLE_MIS || !direct_light_sampled);
    if (handle_emission && any(material_instance.is_emissive())) {
        float mis_weight = 1.0;
        if (!is_primary_hit && ENABLE_NEE && ENABLE_MIS && direct_light_sampled) {
            SurfaceInteraction scatter_si = make_scatter_interaction(path);
            DirectionSample sample = {};
            sample.pos_ws = si.position_ws;
            sample.normal_ws = si.normal_ws;
            sample.dir_ws = path.ray_dir_ws;
            sample.emitter_type = EmitterType::emissive_triangle;
            sample.primitive_id
                = uint(g_scene.emissive_geometry.get_triangle_id(hit.geometry_instance_id, hit.primitive_index));
            float light_pdf = light_sampler.eval_direction_pdf(scatter_si, sample);
            mis_weight = mis_heuristic(1, path.scatter_pdf, 1, light_pdf);
        }
        path.radiance += path.throughput * material_instance.eval_emissive_radiance() * mis_weight;
    }

    if (path.vertex_index >= MAX_DEPTH) {
        path.terminate();
        return;
    }

    // Add NEE (next event estimation) contribution.
    sample_direct_lighting(visibility_query, si, material_instance, path);

    // Sample BRDF to generate next path segment.
    MaterialSample sample = {};
    if (!material_instance.sample(si, path.sg, sample)) {
        path.terminate();
        return;
    }

    bool valid_scatter = generate_scatter_ray(si, sample, path);

    // Track whether the primary ray hit a specular surface and sampled a specular reflection lobe.
    if (is_primary_hit) {
        bool sampled_primary_specular_reflection
            = sample.has_flag(BSDFFlags::glossy_reflection) || sample.has_flag(BSDFFlags::delta_reflection);
        path.set_primary_specular_reflection(sampled_primary_specular_reflection);
    }

    if (!valid_scatter)
        path.terminate();
}

[raypayload]
public struct PathPayload
{
    PathState path : read(caller, closesthit, miss)
        : write(caller, closesthit, miss);
};

[raypayload]
public struct VisibilityPayload
{
    bool visible : read(caller)
        : write(anyhit, miss);

    __init() { visible = false; }
};

static const uint SCATTER_RAY_INDEX = 0;
static const uint VISIBILITY_RAY_INDEX = 1;

public struct InlineVisibilityQuery : IVisibilityQuery
{
    public bool is_occluded(Ray ray)
    {
        RayQuerySceneIntersector intersector;
        return intersector.is_occluded(ray);
    }
};

public struct TraceRayVisibilityQuery : IVisibilityQuery
{
    public bool is_occluded(Ray ray)
    {
        VisibilityPayload payload = {};
        TraceRay(
            g_scene.tlas,
            RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER,
            0xff,
            VISIBILITY_RAY_INDEX,
            0 /* MultiplierForGeometryContributionHitGroupIndex */,
            VISIBILITY_RAY_INDEX,
            ray.to_ray_desc(),
            payload
        );
        return !payload.visible;
    }
};

void handle_scatter_hit<D : IDeferredHitInfo>(inout PathPayload payload, D deferred_hit)
{
    if (payload.path.vertex_index == 0)
        payload.path.vertex_index = 1;

    if (VISIBILITY_RAY_MODE == VisibilityRayMode::ray_query) {
        InlineVisibilityQuery visibility_query;
        handle_hit(visibility_query, deferred_hit.create(), payload.path, g_trace_path_context);
    } else if (VISIBILITY_RAY_MODE == VisibilityRayMode::trace_ray) {
        TraceRayVisibilityQuery visibility_query;
        handle_hit(visibility_query, deferred_hit.create(), payload.path, g_trace_path_context);
    }
}

SCENE_RAY_TYPE_MISS(scatter, scatter_miss, PathPayload);
void scatter_miss(inout PathPayload payload)
{
    handle_miss(payload.path);
}

SCENE_RAY_TYPE_CLOSEST_HIT(scatter, scatter_closest_hit, PathPayload);
void scatter_closest_hit<D : IDeferredHitInfo>(inout PathPayload payload, D deferred_hit)
{
    handle_scatter_hit(payload, deferred_hit);
}

SCENE_RAY_TYPE_MISS(visibility, visibility_miss, VisibilityPayload);
void visibility_miss(inout VisibilityPayload payload)
{
    payload.visible = true;
}

SCENE_RAY_TYPE_ANY_HIT(visibility, visibility_any_hit, VisibilityPayload);
void visibility_any_hit<D : IDeferredHitInfo>(inout VisibilityPayload payload, D deferred_hit)
{
    Ray ray = Ray(WorldRayOrigin(), WorldRayDirection(), RayTMin(), RayTCurrent());
    HitInfo hit = deferred_hit.create();
    if (OpacityEvaluator::is_valid_hit(ray, hit))
        AcceptHitAndEndSearch();
    else
        IgnoreHit();
}

public struct SimpleScheduler
{
    public float3 trace_path(no_diff Ray ray, no_diff uint path_id, no_diff UniformSampleGenerator sg)
    {
        PathPayload payload;
        payload.path = PathState(sg, path_id, ray);

        [MaxIters(MAX_DEPTH + 1)]
        while (payload.path.is_active()) {
            ray = Ray(payload.path.ray_origin_ws, payload.path.ray_dir_ws);
            TraceRay(
                g_scene.tlas,
                0,
                0xff,
                SCATTER_RAY_INDEX,
                0 /* MultiplierForGeometryContributionHitGroupIndex */,
                SCATTER_RAY_INDEX,
                ray.to_ray_desc(),
                payload
            );
        }
        return payload.path.radiance;
    }
};

public struct ReorderingScheduler
{
    public float3 trace_path(no_diff Ray ray, no_diff uint path_id, no_diff UniformSampleGenerator sg)
    {
        PathPayload payload;
        payload.path = PathState(sg, path_id, ray);

        [MaxIters(MAX_DEPTH + 1)]
        while (payload.path.is_active()) {
            ray = Ray(payload.path.ray_origin_ws, payload.path.ray_dir_ws);
            HitObject hit_object = HitObject::TraceRay(
                g_scene.tlas,
                0,
                0xff,
                SCATTER_RAY_INDEX,
                0 /* MultiplierForGeometryContributionHitGroupIndex */,
                SCATTER_RAY_INDEX,
                ray.to_ray_desc(),
                payload
            );

            uint coherence_hint = 0;
            if (hit_object.IsHit()) {
                GeometryInstanceID geometry_instance_id
                    = GeometryInstanceID(hit_object.GetInstanceID() + hit_object.GetGeometryIndex());
                coherence_hint = uint(g_scene.get_geometry_instance_desc(geometry_instance_id).header.material_id);
            }
            ReorderThread(hit_object, coherence_hint, 16);
            HitObject::Invoke(g_scene.tlas, hit_object, payload);
        }
        return payload.path.radiance;
    }
};

public float3 trace_path(no_diff Ray ray, no_diff uint path_id, no_diff UniformSampleGenerator sg)
{
    if (SCHEDULING_MODE == SchedulingMode::simple) {
        SimpleScheduler scheduler;
        return scheduler.trace_path(ray, path_id, sg);
    } else if (SCHEDULING_MODE == SchedulingMode::ser) {
        ReorderingScheduler scheduler;
        return scheduler.trace_path(ray, path_id, sg);
    } else {
        return float3(0.0, 0.0, 0.0);
    }
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

    uint path_id = id.y * ray_sampler.camera.dims.x + id.x;
    float3 radiance = trace_path(ray, path_id, sg);

    output = float4(radiance, 1.0);
}
