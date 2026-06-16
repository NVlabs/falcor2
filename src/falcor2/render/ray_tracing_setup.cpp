// SPDX-License-Identifier: Apache-2.0

#include "falcor2/render/ray_tracing_setup.h"

#include "falcor2/render/scene.h"

#include "falcor2/core/error.h"

#include <sgl/device/device.h>

namespace falcor {

static constexpr const char* DUMMY_HIT_GROUP_NAME = "__dummy_hit_group";

SceneRayTracingSetup
SceneRayTracingSetup::create(const Scene* scene, std::span<const RayDesc> ray_descs, std::optional<Options> options)
{
    // Convert RayDesc to PerGeometryTypeRayDesc with empty hit groups, then call the main create function.
    std::vector<PerGeometryTypeRayDesc> per_geom_descs;
    per_geom_descs.reserve(ray_descs.size());

    auto entry_point_name = [](std::string_view ray_type_name, std::string_view suffix)
    {
        return fmt::format("_scene_{}_{}", ray_type_name, suffix);
    };

    for (const RayDesc& ray_desc : ray_descs) {
        PerGeometryTypeRayDesc per_geom_desc;

        // Miss shader.
        per_geom_desc.miss_entry_point = ray_desc.has_miss ? entry_point_name(ray_desc.name, "miss") : "";

        // Triangle hit group.
        {
            sgl::HitGroupDesc hit_group_desc{
                .hit_group_name = entry_point_name(ray_desc.name, "triangle_hit_group"),
                .closest_hit_entry_point
                = ray_desc.has_closest_hit ? entry_point_name(ray_desc.name, "triangle_closest_hit") : "",
                .any_hit_entry_point = ray_desc.has_any_hit ? entry_point_name(ray_desc.name, "triangle_any_hit") : "",
            };
            per_geom_desc.hit_groups[shared::GeometryType::triangle] = hit_group_desc;
        }

        // LSS hit group.
        switch (scene->_render_scene()->lss_mode()) {
        case RenderLSSMode::hardware: {
            sgl::HitGroupDesc lss_hit_group_desc{
                .hit_group_name = entry_point_name(ray_desc.name, "lss_hit_group"),
                .closest_hit_entry_point
                = ray_desc.has_closest_hit ? entry_point_name(ray_desc.name, "lss_closest_hit") : "",
                .any_hit_entry_point = ray_desc.has_any_hit ? entry_point_name(ray_desc.name, "lss_any_hit") : "",
            };
            per_geom_desc.hit_groups[shared::GeometryType::lss] = lss_hit_group_desc;
            break;
        }
        case RenderLSSMode::procedural: {
            sgl::HitGroupDesc lss_hit_group_desc{
                .hit_group_name = entry_point_name(ray_desc.name, "lss_hit_group"),
                .closest_hit_entry_point
                = ray_desc.has_closest_hit ? entry_point_name(ray_desc.name, "procedural_lss_closest_hit") : "",
                .any_hit_entry_point
                = ray_desc.has_any_hit ? entry_point_name(ray_desc.name, "procedural_lss_any_hit") : "",
                .intersection_entry_point = "procedural_lss_intersection",
            };
            per_geom_desc.hit_groups[shared::GeometryType::lss] = lss_hit_group_desc;
            break;
        }
        }
        per_geom_descs.push_back(std::move(per_geom_desc));
    }

    return create(
        scene,
        std::span<const PerGeometryTypeRayDesc>(per_geom_descs.data(), per_geom_descs.size()),
        options
    );
}

SceneRayTracingSetup SceneRayTracingSetup::create(
    const Scene* scene,
    std::span<const PerGeometryTypeRayDesc> ray_descs,
    std::optional<Options> options_
)
{
    Options options = options_.value_or(Options{});
    const HitGroupPolicy& policy = scene->hit_group_policy();

    FALCOR_CHECK(policy.mode == HitGroupPolicy::Mode::per_geometry_type, "Only per_geometry_type mode is supported.");
    FALCOR_CHECK(!ray_descs.empty(), "ray_descs must not be empty.");

    SceneRayTracingSetup result;

    // Determine which geometry types are active in the scene for pipeline flags.
    bool has_lss = scene->_render_scene()->has_geometry_type(shared::GeometryType::lss);
    RenderLSSMode lss_mode = scene->_render_scene()->lss_mode();
    result.pipeline_flags = (has_lss && lss_mode == RenderLSSMode::hardware)
        ? sgl::RayTracingPipelineFlags::enable_linear_swept_spheres
        : sgl::RayTracingPipelineFlags::none;

    if (is_set(result.pipeline_flags, sgl::RayTracingPipelineFlags::enable_linear_swept_spheres)
        && !scene->device()->has_feature(sgl::Feature::acceleration_structure_linear_swept_spheres))
        FALCOR_THROW("Scene requires hardware LSS but device lacks support.");

    // Collect unique entry point names (avoid duplicates).
    auto add_entry_point = [&](const std::string& name)
    {
        if (name.empty())
            return;
        for (const auto& ep : result.entry_points) {
            if (ep == name)
                return;
        }
        result.entry_points.push_back(name);
    };

    // Geometry types in enum order to match policy expectations.
    static constexpr shared::GeometryType geometry_types[] = {
        shared::GeometryType::triangle,
        shared::GeometryType::lss,
    };

    // Process each ray type required by the policy.
    // Ray types beyond what the user provided get dummy (empty) entries.
    // Within each ray type, geometry types not in the hit_groups map also get dummy entries.
    // SBT layout: [geometry_type * ray_type_count + ray_type].
    // We build hit_groups and sbt_hit_group_names in parallel.

    // Track whether we need a dummy hit group descriptor.
    bool needs_dummy_hit_group = false;

    // Temporary per-ray-type, per-geometry-type hit group names for SBT building.
    // sbt_names[rt][gt] = hit group name.
    std::vector<std::vector<std::string>> sbt_names(policy.ray_type_count);

    for (uint32_t rt = 0; rt < policy.ray_type_count; ++rt) {
        sbt_names[rt].resize(policy.geometry_type_count);

        if (rt < ray_descs.size()) {
            const PerGeometryTypeRayDesc& ray_desc = ray_descs[rt];

            // Add miss entry point.
            result.sbt_miss_entry_points.push_back(ray_desc.miss_entry_point);
            add_entry_point(ray_desc.miss_entry_point);

            // Process each geometry type.
            for (uint32_t gt = 0; gt < policy.geometry_type_count; ++gt) {
                shared::GeometryType geom_type = geometry_types[gt];

                // Skip geometry types not present in the scene if requested.
                if (options.skip_unused_geometry_types && !scene->has_geometry_type(geom_type)) {
                    sbt_names[rt][gt] = DUMMY_HIT_GROUP_NAME;
                    needs_dummy_hit_group = true;
                    continue;
                }

                auto it = ray_desc.hit_groups.find(geom_type);

                if (it != ray_desc.hit_groups.end()) {
                    const sgl::HitGroupDesc& found = it->second;

                    // Patch intersection entry point for LSS if using hardware LSS on CUDA.
                    if (geom_type == shared::GeometryType::lss && lss_mode == RenderLSSMode::hardware
                        && scene->device()->type() == sgl::DeviceType::cuda) {
                        sgl::HitGroupDesc patched = found;
                        patched.intersection_entry_point = "__builtin_intersection__linear_swept_spheres";
                        result.hit_groups.push_back(patched);
                    } else {
                        result.hit_groups.push_back(found);
                    }

                    // Collect entry point names (skip empty strings).
                    add_entry_point(found.closest_hit_entry_point);
                    add_entry_point(found.any_hit_entry_point);
                    add_entry_point(found.intersection_entry_point);

                    sbt_names[rt][gt] = found.hit_group_name;
                } else {
                    // Geometry type not provided by the user, use dummy hit group.
                    sbt_names[rt][gt] = DUMMY_HIT_GROUP_NAME;
                    needs_dummy_hit_group = true;
                }
            }
        } else {
            // Dummy ray type not provided by the user.
            result.sbt_miss_entry_points.push_back("");

            for (uint32_t gt = 0; gt < policy.geometry_type_count; ++gt) {
                sbt_names[rt][gt] = DUMMY_HIT_GROUP_NAME;
            }
            needs_dummy_hit_group = true;
        }
    }

    // Add the dummy hit group descriptor if any SBT slots reference it.
    if (needs_dummy_hit_group)
        result.hit_groups.push_back({.hit_group_name = DUMMY_HIT_GROUP_NAME});

    // Flatten SBT hit group names in [geometry_type * ray_type_count + ray_type] order.
    result.sbt_hit_group_names.reserve(policy.geometry_type_count * policy.ray_type_count);
    for (uint32_t gt = 0; gt < policy.geometry_type_count; ++gt) {
        for (uint32_t rt = 0; rt < policy.ray_type_count; ++rt)
            result.sbt_hit_group_names.push_back(std::move(sbt_names[rt][gt]));
    }

    return result;
}

sgl::ref<sgl::ShaderProgram> SceneRayTracingSetup::link_program(
    sgl::ref<sgl::SlangModule> module,
    std::vector<std::string> additional_entry_points
) const
{
    std::vector<sgl::ref<sgl::SlangEntryPoint>> resolved;
    resolved.reserve(additional_entry_points.size() + entry_points.size());
    for (const auto& name : additional_entry_points)
        resolved.push_back(module->entry_point(name));
    for (const auto& name : entry_points)
        resolved.push_back(module->entry_point(name));
    return module->session()->device()->link_program({module}, std::move(resolved));
}

sgl::ref<sgl::RayTracingPipeline> SceneRayTracingSetup::create_pipeline(sgl::RayTracingPipelineDesc desc) const
{
    desc.hit_groups = hit_groups;
    desc.flags = desc.flags | pipeline_flags;
    return desc.program->device()->create_ray_tracing_pipeline(std::move(desc));
}

sgl::ref<sgl::ShaderTable> SceneRayTracingSetup::create_shader_table(
    sgl::ref<sgl::ShaderProgram> program,
    std::vector<std::string> ray_gen_entry_points
) const
{
    return program->device()->create_shader_table({
        .program = std::move(program),
        .ray_gen_entry_points = std::move(ray_gen_entry_points),
        .miss_entry_points = sbt_miss_entry_points,
        .hit_group_names = sbt_hit_group_names,
    });
}

} // namespace falcor
