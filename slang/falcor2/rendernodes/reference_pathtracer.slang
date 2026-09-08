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
/// Track closed-surface interiors for solid dielectric transmission.
extern static const bool ENABLE_INTERIOR_TRACKING = true;
/// Handle priority-based interfaces between nested interiors.
extern static const bool ENABLE_NESTED_INTERIORS = false;
/// Handle homogeneous absorption and scattering in tracked interiors.
extern static const bool ENABLE_HOMOGENEOUS_MEDIA = true;
/// Terminate low-throughput paths using Russian roulette.
extern static const bool ENABLE_RUSSIAN_ROULETTE = true;
/// Enable stochastic depth of field for cameras with a nonzero aperture.
extern static const bool ENABLE_DEPTH_OF_FIELD = true;
/// Path scheduling implementation.
extern static const SchedulingMode SCHEDULING_MODE = SchedulingMode::simple;
/// Visibility traversal implementation.
extern static const VisibilityRayMode VISIBILITY_RAY_MODE = VisibilityRayMode::ray_query;

public extern struct LightSampler : ILightSampler = PowerLightSampler;

static const bool HANDLE_HOMOGENEOUS_MEDIA = ENABLE_INTERIOR_TRACKING && ENABLE_HOMOGENEOUS_MEDIA;
static const float RUSSIAN_ROULETTE_MAX_SURVIVAL = 0.95;

public ParameterBlock<LightSampler> light_sampler;

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
    float3 pos_ws = {};
    no_diff float3 normal_ws = {};
    no_diff bool front_facing = {};

    __init(const SurfaceInteraction si)
    {
        pos_ws = si.position_ws;
        normal_ws = si.normal_ws;
        front_facing = si.front_facing;
    }

    float3 ray_origin_for_direction(const float3 dir_ws)
    {
        return offset_ray_origin(pos_ws, dot(dir_ws, normal_ws) >= 0.0 ? normal_ws : -normal_ws);
    }

    float3 oriented_normal() { return front_facing ? normal_ws : -normal_ws; }
};

enum class HemisphereFlags {
    none = 0,
    upper = (1 << 0),
    lower = (1 << 1),
    both = upper | lower,
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
    /// Current ray ends at a finite sampled homogeneous-medium event.
    volume_event = (1 << 5),
};

struct PathState : IDifferentiable
{
    static const uint DEPTH_BIT_COUNT = 8;
    static const uint DEPTH_MASK = (1 << DEPTH_BIT_COUNT) - 1;
    static const uint PATH_FLAGS_BIT_COUNT = 8;
    static const uint PATH_FLAGS_MASK = ((1 << PATH_FLAGS_BIT_COUNT) - 1) << DEPTH_BIT_COUNT;
    static const uint SKIPPED_INTERFACES_BIT_OFFSET = DEPTH_BIT_COUNT + PATH_FLAGS_BIT_COUNT;
    static const uint SKIPPED_INTERFACES_MASK = 0xff << SKIPPED_INTERFACES_BIT_OFFSET;
    static const uint MAX_SKIPPED_INTERFACES = (ENABLE_INTERIOR_TRACKING && ENABLE_NESTED_INTERIORS) ? 8 : 0;

    no_diff UniformSampleGenerator sg;
    no_diff uint path_id = {};
    no_diff uint flags_and_depth = {};
    no_diff InteriorList interior_list = {};

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
            return PathFlags((flags_and_depth & PATH_FLAGS_MASK) >> DEPTH_BIT_COUNT);
        }
        set {
            flags_and_depth = (flags_and_depth & ~PATH_FLAGS_MASK) | (uint(newValue) << DEPTH_BIT_COUNT);
        }
    }

    /// Path depth.
    /// 0 = camera, 1 = primary hit, 2 = first bounce, etc.
    property uint depth
    {
        get {
            return flags_and_depth & DEPTH_MASK;
        }
        set {
            flags_and_depth = (flags_and_depth & ~DEPTH_MASK) | (newValue & DEPTH_MASK);
        }
    }

    /// Number of consecutive nested interfaces skipped without scattering.
    property uint skipped_interface_count
    {
        get {
            return (flags_and_depth & SKIPPED_INTERFACES_MASK) >> SKIPPED_INTERFACES_BIT_OFFSET;
        }
        set {
            flags_and_depth = (flags_and_depth & ~SKIPPED_INTERFACES_MASK)
                | ((newValue << SKIPPED_INTERFACES_BIT_OFFSET) & SKIPPED_INTERFACES_MASK);
        }
    }

    // Flags

    /// Check if a specific flag is set.
    bool has_flag(const PathFlags flag)
    {
        const uint bit = uint(flag) << DEPTH_BIT_COUNT;
        return (flags_and_depth & bit) != 0;
    }

    /// Set or clear a specific flag.
    [mutating]
    void set_flag(const PathFlags flag, const bool value = true)
    {
        const uint bit = uint(flag) << DEPTH_BIT_COUNT;
        if (value)
            flags_and_depth |= bit;
        else
            flags_and_depth &= ~bit;
    }

    bool is_active() { return has_flag(PathFlags::active); }
    bool is_delta() { return has_flag(PathFlags::delta); }
    bool is_light_sampled_upper() { return has_flag(PathFlags::light_sampled_upper); }
    bool is_light_sampled_lower() { return has_flag(PathFlags::light_sampled_lower); }
    bool is_light_sampled() { return has_flag(PathFlags::light_sampled_upper | PathFlags::light_sampled_lower); }
    bool is_primary_specular_reflection() { return has_flag(PathFlags::primary_specular_reflection); }
    bool is_volume_event() { return has_flag(PathFlags::volume_event); }

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
    void set_volume_event(bool value = true) { set_flag(PathFlags::volume_event, value); }

    [mutating]
    void set_light_sampled(bool upper, bool lower)
    {
        set_light_sampled_upper(upper);
        set_light_sampled_lower(lower);
    }

    /// Helpers

    /// Advance to the next accepted scattering vertex.
    [mutating]
    void advance_depth()
    {
        depth += 1;
        skipped_interface_count = 0;
    }

    /// Increment the skipped interface count, returning false if the maximum is exceeded.
    [mutating]
    bool try_increment_skipped_interfaces()
    {
        uint count = skipped_interface_count;
        if (count >= MAX_SKIPPED_INTERFACES)
            return false;
        skipped_interface_count = count + 1;
        return true;
    }

    bool is_light_samplable() { return !is_delta(); }

    bool is_inside_tracked_interior() { return ENABLE_INTERIOR_TRACKING && !interior_list.is_empty(); }
    bool is_inside_homogeneous_medium() { return HANDLE_HOMOGENEOUS_MEDIA && !interior_list.is_empty(); }

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

Ray prepare_path_segment(inout PathState path)
{
    Ray ray = Ray(path.ray_origin_ws, path.ray_dir_ws);

    // If the path is inside a homogeneous medium, sample a distance to the next scattering event.
    if (HANDLE_HOMOGENEOUS_MEDIA) {
        path.set_volume_event(false);
        if (path.is_inside_homogeneous_medium()) {
            const HomogeneousMediumProperties medium = path.interior_list.active_medium();
            const HomogeneousVolumeSampler sampler = HomogeneousVolumeSampler(medium);
            float distance;
            if (!sampler.sample_distance(path.throughput, path.sg, distance))
                return ray;

            if (isfinite(distance) && distance < ray.t_max) {
                ray.t_max = distance;
                path.set_volume_event();
            }
        }
    }

    return ray;
}

public struct TracePathContext
{
    Camera current_camera = {};
    Camera previous_camera = {};

    uint2 pixel_id(const PathState path) { return path.pixel_id(current_camera.dims); }
};

public ParameterBlock<TracePathContext> g_trace_path_context;

public interface IVisibilityQuery
{
    public bool is_occluded(Ray ray, OpacityEvaluator opacity_evaluator);
}

HemisphereFlags get_material_nee_hemispheres(const BSDFFlags lobe_types)
{
    HemisphereFlags result = HemisphereFlags::none;
    if (is_set(lobe_types, BSDFFlags::non_delta_reflection))
        result = result | HemisphereFlags::upper;
    if (is_set(lobe_types, BSDFFlags::non_delta_transmission))
        result = result | HemisphereFlags::lower;
    if (is_set(lobe_types, BSDFFlags::curve))
        result = result | HemisphereFlags::both;
    return result;
}

/// Returns the hemispheres for which the NEE visibility estimator can produce a valid result.
/// This is a capability mask that is intersected with the hemispheres supported by the material.
/// Without homogeneous media, binary surface visibility is valid in both hemispheres and imposes no restriction.
HemisphereFlags get_nee_visibility_supported_hemispheres(
    const bool is_entering,
    const MaterialID material_id,
    const bool is_thin_walled,
    const InteriorList interiors
)
{
    if (!HANDLE_HOMOGENEOUS_MEDIA)
        return HemisphereFlags::both;

    if (is_thin_walled)
        return interiors.is_empty() ? HemisphereFlags::both : HemisphereFlags::none;

    if (is_entering)
        return interiors.is_empty() ? HemisphereFlags::upper : HemisphereFlags::none;

    // A two-sided backface can be an open surface rather than the boundary of a tracked interior.
    // In vacuum, allow NEE for reflection back into the incident hemisphere.
    if (interiors.is_empty())
        return HemisphereFlags::upper;

    return interiors.contains_only(material_id) ? HemisphereFlags::lower : HemisphereFlags::none;
}

bool is_direction_in_sampled_hemispheres(const float3 normal_ws, const HemisphereFlags hemispheres, const float3 dir_ws)
{
    bool upper = dot(dir_ws, normal_ws) >= 0.0;
    bool can_sample_upper = (hemispheres & HemisphereFlags::upper) != 0;
    bool can_sample_lower = (hemispheres & HemisphereFlags::lower) != 0;
    return upper ? can_sample_upper : can_sample_lower;
}

bool is_direction_in_sampled_hemispheres(
    const PathVertex vertex,
    const HemisphereFlags hemispheres,
    const float3 dir_ws
)
{
    return is_direction_in_sampled_hemispheres(vertex.oriented_normal(), hemispheres, dir_ws);
}

bool was_direction_sampled_by_nee(const PathState path, const float3 dir_ws)
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

/// Creates a context whose receiver normal covers exactly one sampled hemisphere.
/// A zero normal requests unrestricted sampling when both (or neither) are enabled.
LightSampleContext make_light_sample_context(const SurfaceInteraction si, const HemisphereFlags sampled_hemispheres)
{
    LightSampleContext ctx = LightSampleContext(si);
    const float3 upper_normal_ws = si.front_normal_ws();
    if (sampled_hemispheres == HemisphereFlags::upper)
        ctx.normal_ws = upper_normal_ws;
    else if (sampled_hemispheres == HemisphereFlags::lower)
        ctx.normal_ws = -upper_normal_ws;
    else
        ctx.normal_ws = {};
    return ctx;
}

// Generate a ray from the shading point to a randomly sampled light.
Ray create_visibility_ray(const PathVertex shading_vertex, const LightSample sample)
{
    Ray ray;
    ray.origin = shading_vertex.ray_origin_for_direction(sample.dir_ws);
    ray.dir = sample.dir_ws;
    ray.t_min = 0.0;
    ray.t_max = sample.distance;
    if (sample.emitter_type == EmitterType::emissive_triangle) {
        PathVertex target_vertex;
        target_vertex.pos_ws = sample.pos_ws;
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
    string operation;
}

// This interface defines what outputs can be made, and provides additional metadata to describe
// how to initialize them. The generated prelude contains an implementation of this interface
// that writes to the actual guide outputs.
public interface IWriteGuide
{
    [OutputSpec("rgba16_float", "0,0,0,1", "write")]
    public static void diffuse_albedo(uint2 coord, float4 value);
    public static bool diffuse_albedo_enabled();
    [OutputSpec("rgba16_float", "0,0,0,1", "write")]
    public static void material_color(uint2 coord, float4 value);
    public static bool material_color_enabled();
    [OutputSpec("rgba16_float", "0,0,0,1", "write")]
    public static void specular_albedo(uint2 coord, float4 value);
    public static bool specular_albedo_enabled();
    [OutputSpec("rgba16_float", "0,0,1,0", "write")]
    public static void normals(uint2 coord, float4 value);
    public static bool normals_enabled();
    [OutputSpec("r16_float", "1", "write")]
    public static void roughness(uint2 coord, float value);
    public static bool roughness_enabled();
    [OutputSpec("r16_float", "0", "write")]
    public static void metallic(uint2 coord, float value);
    public static bool metallic_enabled();
    [OutputSpec("r32_float", "0", "write")]
    public static void depth(uint2 coord, float value);
    public static bool depth_enabled();
    [OutputSpec("r32_float", "1", "write")]
    public static void hardware_depth(uint2 coord, float value);
    public static bool hardware_depth_enabled();
    [OutputSpec("r16_float", "0", "write")]
    public static void specular_hit_distance(uint2 coord, float value);
    public static bool specular_hit_distance_enabled();
    [OutputSpec("rg16_float", "0,0", "write")]
    public static void motion_vectors(uint2 coord, float2 value);
    public static bool motion_vectors_enabled();
    [OutputSpec("rgba16_float", "0,0,0,0", "write")]
    public static void emission(uint2 coord, float4 value);
    public static bool emission_enabled();
    [OutputSpec("rgba32_uint", "0xffffffff,0xffffffff,0xffffffff,0", "write")]
    public static void geometry_id(uint2 coord, uint4 value);
    public static bool geometry_id_enabled();
    [OutputSpec("r32_uint", "0", "increment")]
    public static void nan_count(uint2 coord, uint value);
    public static bool nan_count_enabled();
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

    public static override void metallic(uint2 coord, float value) { }
    public static bool metallic_enabled() { return false; }

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

    public static override void nan_count(uint2 coord, uint value) { }
    public static bool nan_count_enabled() { return false; }
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
extern static const bool WriteGuide_metallic = false;
extern static const bool WriteGuide_depth = false;
extern static const bool WriteGuide_hardware_depth = false;
extern static const bool WriteGuide_specular_hit_distance = false;
extern static const bool WriteGuide_motion_vectors = false;
extern static const bool WriteGuide_emission = false;
extern static const bool WriteGuide_geometry_id = false;
extern static const bool WriteGuide_nan_count = false;

void sample_direct_lighting<VisibilityQuery : IVisibilityQuery>(
    VisibilityQuery visibility_query,
    const SurfaceInteraction si,
    const MaterialID material_id,
    const bool is_thin_walled,
    IMaterialInstance material_instance,
    inout PathState path
)
{
    path.set_light_sampled_hemispheres(HemisphereFlags::none);
    if (!ENABLE_NEE)
        return;

    BSDFFlags lobe_types = material_instance.get_lobe_types(si);
    HemisphereFlags sampled_hemispheres = get_material_nee_hemispheres(lobe_types)
        & get_nee_visibility_supported_hemispheres(si.front_facing, material_id, is_thin_walled, path.interior_list);
    path.set_light_sampled_hemispheres(sampled_hemispheres);
    if (sampled_hemispheres == HemisphereFlags::none)
        return;

    LightSample sample = {};
    PathVertex vertex = { si };
    LightSampleContext light_ctx = make_light_sample_context(si, sampled_hemispheres);
    if (!light_sampler.sample(light_ctx, path.sg, sample))
        return;
    if (!is_direction_in_sampled_hemispheres(vertex, sampled_hemispheres, sample.dir_ws))
        return;

    OpacityEvaluator opacity_evaluator = OpacityEvaluator(path.sg.next_uint());
    if (visibility_query.is_occluded(create_visibility_ray(vertex, sample), opacity_evaluator))
        return;

    float mis_weight = 1.0;
    float3 scatter_eval;
    if (ENABLE_MIS && !sample.nee_only) {
        const BSDFEval scatter = material_instance.eval_with_pdf(si, sample.dir_ws, path.sg);
        scatter_eval = scatter.value;
        mis_weight = mis_heuristic(1, sample.pdf, 1, scatter.pdf);
    } else {
        scatter_eval = material_instance.eval(si, sample.dir_ws, path.sg);
    }
    path.radiance += path.throughput * scatter_eval * sample.weight * mis_weight;
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

    return any(path.throughput > 0.0);
}

public struct PathTracerSettings
{
    /// Maximum path depth.
    /// 1 = directly visible emission, 2 = direct lighting, 3 = one indirect bounce, etc.
    uint max_depth = 3;
    /// Path depth at which Russian roulette starts.
    uint rr_depth = 3;
    /// Use a constant color instead of the environment map for primary-ray background visibility.
    bool use_background_color = false;
    /// Constant color used for the primary-ray background.
    float3 background_color = float3(0.0, 0.0, 0.0);
};

bool should_skip_nested_interface(const PathState path, const MaterialHeader material_header)
{
    return ENABLE_INTERIOR_TRACKING && ENABLE_NESTED_INTERIORS
        && !is_set(material_header.flags, MaterialFlags::thin_walled)
        && !path.interior_list.is_true_interface(material_header.nested_priority);
}

public struct PathTracer
{
    PathTracerSettings settings;

    /// Run Russian roulette and return true if the path should terminate.
    /// Compensate the throughput when the path survives.
    bool should_terminate_by_russian_roulette(inout PathState path)
    {
        if (!ENABLE_RUSSIAN_ROULETTE || path.depth < settings.rr_depth)
            return false;

        const float3 throughput = detach(path.throughput);
        const float survival_probability
            = min(max(throughput.x, max(throughput.y, throughput.z)), RUSSIAN_ROULETTE_MAX_SURVIVAL);
        if (next_float(path.sg) > survival_probability)
            return true;

        path.throughput /= survival_probability;
        return false;
    }

    void handle_volume_event(inout PathState path, const float distance)
    {
        const HomogeneousMediumProperties medium = path.interior_list.active_medium();
        const HomogeneousVolumeSampler sampler = HomogeneousVolumeSampler(medium);
        const float3 event_weight = sampler.eval_event_weight(path.throughput, distance);
        path.throughput *= event_weight;
        path.ray_origin_ws += path.ray_dir_ws * distance;
        path.set_volume_event(false);
        path.set_light_sampled(false, false);
        path.set_delta(false);
        path.set_primary_specular_reflection(false);
        path.scatter_normal_ws = {};
        path.advance_depth();

        if (!any(path.throughput > 0.0) || path.depth >= settings.max_depth) {
            path.terminate();
            return;
        }

        const HenyeyGreensteinPhaseFunction phase_function
            = HenyeyGreensteinPhaseFunction(clamp(medium.anisotropy, -0.999, 0.999));
        PhaseFunctionSample phase_sample = {};
        if (!phase_function.sample(-path.ray_dir_ws, path.sg, phase_sample)) {
            path.terminate();
            return;
        }

        path.throughput *= phase_sample.weight;
        path.scatter_pdf = phase_sample.pdf;
        path.ray_dir_ws = phase_sample.wo_ws;
        if (!any(path.throughput > 0.0) || should_terminate_by_russian_roulette(path))
            path.terminate();
    }

    void handle_miss(inout PathState path, const float ray_t_current)
    {
        if (HANDLE_HOMOGENEOUS_MEDIA) {
            if (path.is_volume_event()) {
                handle_volume_event(path, ray_t_current);
                return;
            }
            if (path.is_inside_homogeneous_medium()) {
                const HomogeneousVolumeSampler sampler = HomogeneousVolumeSampler(path.interior_list.active_medium());
                path.throughput *= sampler.eval_surface_weight_to_infinity(path.throughput);
            }
            path.set_volume_event(false);
        }

        bool is_primary_miss = path.depth == 0;
        bool has_environment_lights = g_scene.has_environment_lights();
        bool show_environment_background = has_environment_lights && is_primary_miss && !settings.use_background_color;
        bool nee_sampled_direction = was_direction_sampled_by_nee(path, path.ray_dir_ws);
        bool nee_replaces_environment_hit = ENABLE_NEE && !ENABLE_MIS && nee_sampled_direction;
        bool apply_environment_mis = ENABLE_NEE && ENABLE_MIS && nee_sampled_direction;
        bool handle_environment_light = has_environment_lights && !is_primary_miss && !nee_replaces_environment_hit;

        if (show_environment_background || handle_environment_light) {
            SurfaceInteraction scatter_si = make_scatter_interaction(path);
            LightSampleContext light_ctx = make_light_sample_context(scatter_si, path.light_sampled_hemispheres());
            float mis_weight = 1.0;
            if (handle_environment_light && apply_environment_mis) {
                float light_pdf = light_sampler.eval_environment_pdf(light_ctx, path.ray_dir_ws);
                mis_weight = mis_heuristic(1, path.scatter_pdf, 1, light_pdf);
            }
            let eval = g_scene.light_system.eval_environment_radiance(light_ctx, path.ray_dir_ws);
            path.radiance += path.throughput * eval * mis_weight;
        } else if (is_primary_miss && settings.use_background_color) {
            path.radiance += settings.background_color;
        }

        path.terminate();
    }

    void handle_hit<VisibilityQuery : IVisibilityQuery>(
        VisibilityQuery visibility_query,
        no_diff IHitInfo hit,
        inout PathState path,
        const TracePathContext context
    )
    {
        uint2 id = context.pixel_id(path);

        // Make surface interaction at hit point.
        SurfaceInteraction si = g_scene.make_surface_interaction(hit, -path.ray_dir_ws);
        if (HANDLE_HOMOGENEOUS_MEDIA) {
            float segment_length = length(si.position_ws - path.ray_origin_ws);
            const HomogeneousMediumProperties medium = path.interior_list.active_medium();
            const HomogeneousVolumeSampler sampler = HomogeneousVolumeSampler(medium);
            if (sampler.has_scattering())
                path.throughput *= sampler.eval_surface_weight(path.throughput, segment_length);
            else
                path.throughput *= sampler.eval_transmittance(segment_length);
        }
        path.set_volume_event(false);
        MaterialID material_id = g_scene.get_geometry_instance_desc(hit.geometry_instance_id).header.material_id;
        MaterialHeader material_header = g_scene.material_system.get_material_data(material_id).header;

        // Reject backfaces of materials that cannot interact with both sides of the surface.
        bool surface_visible = si.front_facing || is_set(material_header.flags, MaterialFlags::two_sided);
        if (!surface_visible) {
            path.terminate();
            return;
        }

        if (ENABLE_INTERIOR_TRACKING)
            si.exterior_ior = path.interior_list.exterior_ior(material_id, si.front_facing);

        // Make material instance.
        let lod_sampler = ExplicitLodSampler(0);
        let material_instance
            = g_scene.get_material_instance(hit, si, lod_sampler, MaterialInstanceHints::adjust_shading_normal);
        bool is_thin_walled = is_set(material_header.flags, MaterialFlags::thin_walled);
        bool is_unlit = is_set(material_header.flags, MaterialFlags::unlit);

        if (should_skip_nested_interface(path, material_header)) {
            if (!path.try_increment_skipped_interfaces()) {
                path.terminate();
                return;
            }
            InteriorListUpdateResult result = cross_interior_boundary(
                si.front_facing,
                material_id,
                material_header.nested_priority,
                material_instance,
                path.interior_list
            );
            if (result != InteriorListUpdateResult::success) {
                path.terminate();
                return;
            }

            float3 offset_normal_ws = dot(path.ray_dir_ws, si.normal_ws) >= 0.f ? si.normal_ws : -si.normal_ws;
            path.ray_origin_ws = offset_ray_origin(si.position_ws, offset_normal_ws);
            return;
        }

        path.advance_depth();
        bool is_primary_hit = path.depth == 1;

        if (is_primary_hit)
            write_primary_surface_interaction(id, hit, si, context.current_camera, context.previous_camera);

        // If the primary hit sampled a specular reflection, record the scatter path length to this hit.
        if (WriteGuide::specular_hit_distance_enabled() && path.is_primary_specular_reflection()) {
            write_specular_hit_distance(id, length(si.position_ws - path.ray_origin_ws));
            path.set_primary_specular_reflection(false);
        }

        if (is_primary_hit)
            write_primary_material(id, si, material_instance);

        // Unlit materials are not lit by any future events, so terminate path immediately.
        if (is_unlit) {
            path.radiance += path.throughput * material_instance.eval_emissive_radiance();
            path.terminate();
            return;
        }

        // Add emissive triangle contribution.
        bool nee_sampled_direction = was_direction_sampled_by_nee(path, path.ray_dir_ws);
        bool nee_replaces_emitter_hit = ENABLE_NEE && !ENABLE_MIS && nee_sampled_direction;
        bool apply_emitter_mis = !is_primary_hit && ENABLE_NEE && ENABLE_MIS && nee_sampled_direction;
        bool handle_emission = g_scene.has_emissive_triangles() && (is_primary_hit || !nee_replaces_emitter_hit);
        if (handle_emission && any(material_instance.is_emissive())) {
            float mis_weight = 1.0;
            if (apply_emitter_mis) {
                SurfaceInteraction scatter_si = make_scatter_interaction(path);
                LightSampleContext light_ctx = make_light_sample_context(scatter_si, path.light_sampled_hemispheres());
                EmissiveTriangleID triangle_id
                    = g_scene.emissive_geometry.get_triangle_id(hit.geometry_instance_id, hit.primitive_index);
                float light_pdf = light_sampler.eval_emissive_pdf(light_ctx, triangle_id, si.position_ws, si.normal_ws);
                mis_weight = mis_heuristic(1, path.scatter_pdf, 1, light_pdf);
            }
            path.radiance += path.throughput * material_instance.eval_emissive_radiance() * mis_weight;
        }

        if (path.depth >= settings.max_depth) {
            path.terminate();
            return;
        }

        // Add NEE (next event estimation) contribution.
        sample_direct_lighting(visibility_query, si, material_id, is_thin_walled, material_instance, path);

        // Sample BRDF to generate next path segment.
        MaterialSample sample = {};
        if (!material_instance.sample(si, path.sg, sample)) {
            path.terminate();
            return;
        }

        bool crossed_solid_interface
            = ENABLE_INTERIOR_TRACKING && !is_thin_walled && sample.has_flag(BSDFFlags::transmission);
        if (crossed_solid_interface) {
            InteriorListUpdateResult result = cross_interior_boundary(
                si.front_facing,
                material_id,
                material_header.nested_priority,
                material_instance,
                path.interior_list
            );
            if (result != InteriorListUpdateResult::success) {
                path.terminate();
                return;
            }
        }

        bool valid_scatter = generate_scatter_ray(si, sample, path);

        // Track whether the primary ray hit a specular surface and sampled a specular reflection lobe.
        if (is_primary_hit) {
            bool sampled_primary_specular_reflection
                = sample.has_flag(BSDFFlags::glossy_reflection) || sample.has_flag(BSDFFlags::delta_reflection);
            path.set_primary_specular_reflection(sampled_primary_specular_reflection);
        }

        if (!valid_scatter || should_terminate_by_russian_roulette(path))
            path.terminate();
    }
};

public ParameterBlock<PathTracer> g_path_tracer;

InteriorListUpdateResult cross_interior_boundary(
    const bool is_entering,
    const MaterialID material_id,
    const uint nested_priority,
    IMaterialInstance material_instance,
    inout InteriorList interior_list
)
{
    if (is_entering) {
        HomogeneousMediumProperties interior_medium = {};
        if (HANDLE_HOMOGENEOUS_MEDIA)
            interior_medium = material_instance.get_interior_medium();
        let interior_entry
            = interior_list
                  .make_entry(material_id, nested_priority, material_instance.get_interior_ior(), interior_medium);
        return interior_list.enter(interior_entry);
    } else {
        return interior_list.exit(material_id);
    }
}

[raypayload]
public struct PathPayload
{
    PathState path : read(caller, closesthit, miss)
        : write(caller, closesthit, miss);

    OpacityEvaluator opacity_evaluator : read(anyhit)
        : write(caller);
};

[raypayload]
public struct VisibilityPayload
{
    bool visible : read(caller)
        : write(anyhit, miss);

    OpacityEvaluator opacity_evaluator : read(anyhit)
        : write(caller);

    __init()
    {
        visible = false;
        // TODO(slang): Enabling this crashes slang compiler
        // opacity_evaluator = {};
    }
};

static const uint SCATTER_RAY_INDEX = 0;
static const uint VISIBILITY_RAY_INDEX = 1;

public struct InlineVisibilityQuery : IVisibilityQuery
{
    public bool is_occluded(Ray ray, OpacityEvaluator opacity_evaluator)
    {
        RayQuerySceneIntersector intersector = RayQuerySceneIntersector(opacity_evaluator);
        return intersector.is_occluded(ray, MaterialInstanceHints::adjust_shading_normal);
    }
};

public struct TraceRayVisibilityQuery : IVisibilityQuery
{
    public bool is_occluded(Ray ray, OpacityEvaluator opacity_evaluator)
    {
        VisibilityPayload payload = {};
        payload.opacity_evaluator = opacity_evaluator;
        TraceRay(
            g_scene.tlas,
            RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER | OpacityEvaluator.RAY_FLAGS,
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
    if (VISIBILITY_RAY_MODE == VisibilityRayMode::ray_query) {
        InlineVisibilityQuery visibility_query;
        g_path_tracer.handle_hit(visibility_query, deferred_hit.create(), payload.path, g_trace_path_context);
    } else if (VISIBILITY_RAY_MODE == VisibilityRayMode::trace_ray) {
        TraceRayVisibilityQuery visibility_query;
        g_path_tracer.handle_hit(visibility_query, deferred_hit.create(), payload.path, g_trace_path_context);
    }
}

SCENE_RAY_TYPE_MISS(scatter, scatter_miss, PathPayload);
void scatter_miss(inout PathPayload payload)
{
    g_path_tracer.handle_miss(payload.path, RayTCurrent());
}

SCENE_RAY_TYPE_CLOSEST_HIT(scatter, scatter_closest_hit, PathPayload);
void scatter_closest_hit<D : IDeferredHitInfo>(inout PathPayload payload, D deferred_hit)
{
    handle_scatter_hit(payload, deferred_hit);
}

SCENE_RAY_TYPE_ANY_HIT(scatter, scatter_any_hit, PathPayload);
void scatter_any_hit<D : IDeferredHitInfo>(inout PathPayload payload, D deferred_hit)
{
    Ray ray = Ray(WorldRayOrigin(), WorldRayDirection(), RayTMin(), RayTCurrent());
    if (!payload.opacity_evaluator
             .is_valid_hit(ray, deferred_hit.create(), MaterialInstanceHints::adjust_shading_normal))
        IgnoreHit();
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
    IHitInfo hit = deferred_hit.create();
    if (payload.opacity_evaluator.is_valid_hit(ray, hit, MaterialInstanceHints::adjust_shading_normal))
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

        while (payload.path.is_active()) {
            ray = prepare_path_segment(payload.path);
            payload.opacity_evaluator = OpacityEvaluator(payload.path.sg.next_uint());
            TraceRay(
                g_scene.tlas,
                OpacityEvaluator.RAY_FLAGS,
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

enum class SERWorkClass : uint {
    environment_miss = 0b000,
    unlit_terminal = 0b001,
    depth_terminal = 0b010,
    surface_scatter = 0b100,
    volume_scatter = 0b101,
    nested_passthrough = 0b110,
};

uint get_ser_coherence_hint(const HitObject hit_object, const PathState path)
{
    static const uint WORK_CLASS_BIT_OFFSET = 13;
    static const uint CAMERA_SEGMENT_BIT = 1 << 12;
    static const uint ACTIVE_MEDIUM_BIT = 1 << 11;
    static const uint EMITTER_MIS_PATH_BIT = 1 << 10;
    static const uint MATERIAL_TYPE_MASK = 0xff;

    SERWorkClass work_class;
    uint material_type = 0;

    if (!hit_object.IsHit()) {
        work_class = path.is_volume_event() ? SERWorkClass::volume_scatter : SERWorkClass::environment_miss;
    } else {
        GeometryInstanceID geometry_instance_id
            = GeometryInstanceID(hit_object.GetInstanceID() + hit_object.GetGeometryIndex());
        MaterialID material_id = g_scene.get_geometry_instance_desc(geometry_instance_id).header.material_id;
        MaterialHeader material_header = g_scene.material_system.get_material_data(material_id).header;

        material_type = material_header.type_id & MATERIAL_TYPE_MASK;
        if (should_skip_nested_interface(path, material_header))
            work_class = SERWorkClass::nested_passthrough;
        else if (is_set(material_header.flags, MaterialFlags::unlit))
            work_class = SERWorkClass::unlit_terminal;
        else if (path.depth + 1 >= g_path_tracer.settings.max_depth)
            work_class = SERWorkClass::depth_terminal;
        else
            work_class = SERWorkClass::surface_scatter;
    }

    // The high byte groups paths by expected shader work. The low byte provides material-type
    // coherence until material types use distinct closest-hit entry points.
    uint coherence_hint = uint(work_class) << WORK_CLASS_BIT_OFFSET;
    if (path.depth == 0)
        coherence_hint |= CAMERA_SEGMENT_BIT;
    if (path.is_inside_homogeneous_medium())
        coherence_hint |= ACTIVE_MEDIUM_BIT;
    if (was_direction_sampled_by_nee(path, path.ray_dir_ws))
        coherence_hint |= EMITTER_MIS_PATH_BIT;
    coherence_hint |= material_type;

    return coherence_hint;
}

public struct ReorderingScheduler
{
    public float3 trace_path(no_diff Ray ray, no_diff uint path_id, no_diff UniformSampleGenerator sg)
    {
        PathPayload payload;
        payload.path = PathState(sg, path_id, ray);

        while (payload.path.is_active()) {
            ray = prepare_path_segment(payload.path);
            payload.opacity_evaluator = OpacityEvaluator(payload.path.sg.next_uint());
            HitObject hit_object = HitObject::TraceRay(
                g_scene.tlas,
                OpacityEvaluator.RAY_FLAGS,
                0xff,
                SCATTER_RAY_INDEX,
                0 /* MultiplierForGeometryContributionHitGroupIndex */,
                SCATTER_RAY_INDEX,
                ray.to_ray_desc(),
                payload
            );

            uint coherence_hint = get_ser_coherence_hint(hit_object, payload.path);
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

void write_specular_hit_distance(uint2 coord, float distance)
{
    if (WriteGuide::specular_hit_distance_enabled())
        WriteGuide::specular_hit_distance(coord, distance);
}

void write_primary_material(uint2 coord, SurfaceInteraction si, IMaterialInstance material_instance)
{
    if (WriteGuide::diffuse_albedo_enabled() || WriteGuide::material_color_enabled()
        || WriteGuide::specular_albedo_enabled() || WriteGuide::normals_enabled() || WriteGuide::roughness_enabled()
        || WriteGuide::metallic_enabled() || WriteGuide::emission_enabled()) {
        MaterialProperties properties = material_instance.collect_properties(si);
        if (WriteGuide::diffuse_albedo_enabled())
            WriteGuide::diffuse_albedo(coord, float4(saturate(properties.diffuse_reflection_albedo), 1.0));
        if (WriteGuide::specular_albedo_enabled())
            WriteGuide::specular_albedo(coord, float4(saturate(properties.specular_reflection_albedo), 1.0));
        if (WriteGuide::normals_enabled()) {
            float3 guide_normal = dot(properties.guide_normal, properties.guide_normal) > 0.0
                ? normalize(properties.guide_normal)
                : si.front_normal_ws();
            WriteGuide::normals(coord, float4(guide_normal, 0.0));
        }
        if (WriteGuide::roughness_enabled())
            WriteGuide::roughness(coord, saturate(properties.roughness));
        if (WriteGuide::material_color_enabled())
            WriteGuide::material_color(coord, float4(saturate(properties.material_color), 1.0));
        if (WriteGuide::metallic_enabled())
            WriteGuide::metallic(coord, saturate(properties.metallic));
        if (WriteGuide::emission_enabled())
            WriteGuide::emission(coord, float4(properties.emission, 1.0));
    }
}

void write_primary_surface_interaction(
    uint2 coord,
    no_diff IHitInfo hit,
    SurfaceInteraction si,
    Camera camera,
    Camera previous_camera
)
{
    if (WriteGuide::depth_enabled() || WriteGuide::hardware_depth_enabled()) {
        float depth = dot(si.position_ws - camera.position, camera.image_w);
        if (WriteGuide::depth_enabled())
            WriteGuide::depth(coord, depth);
        if (WriteGuide::hardware_depth_enabled()) {
            const float z_near = 0.1;
            const float z_far = 1000.0;
            float linear_depth = max(depth, z_near);
            float hardware_depth = saturate((z_far / (z_far - z_near)) * (1.0 - (z_near / linear_depth)));
            WriteGuide::hardware_depth(coord, hardware_depth);
        }
    }
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
    if (ENABLE_DEPTH_OF_FIELD && ray_sampler.camera.aperture_radius > 0.0 && ray_sampler.camera.focus_distance > 0.0)
        ray = ray_sampler.camera.get_ray(uv, next_float2(sg));

    uint path_id = id.y * ray_sampler.camera.dims.x + id.x;
    float3 radiance = trace_path(ray, path_id, sg);

    if (WriteGuide::nan_count_enabled() && any(isnan(radiance)))
        WriteGuide::nan_count(id, 1u);

    output = float4(radiance, 1.0);
}
