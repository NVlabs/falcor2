// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "falcor2/ui/scene_editor_gizmos.h"

#include "falcor2/render/component/camera.h"
#include "falcor2/render/component/light.h"
#include "falcor2/render/entity.h"
#include "falcor2/render/scene.h"

#include <imgui.h>

#include <sgl/math/matrix_math.h>
#include <sgl/math/vector_math.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <utility>

namespace falcor::ui::detail {
namespace {

constexpr float HIT_PADDING = 2.f;
constexpr float HALO_PADDING = 6.f;
constexpr float GUIDE_THICKNESS = 2.5f;

bool is_finite(float4 value)
{
    return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z) && std::isfinite(value.w);
}

float2 ndc_to_screen(float2 ndc, float2 viewport_position, float2 viewport_size)
{
    return viewport_position + float2((0.5f * ndc.x + 0.5f) * viewport_size.x, (0.5f - 0.5f * ndc.y) * viewport_size.y);
}

bool overlaps_viewport(float2 screen_position, float half_size, float2 viewport_position, float2 viewport_size)
{
    const float2 viewport_max = viewport_position + viewport_size;
    return screen_position.x + half_size >= viewport_position.x && screen_position.x - half_size <= viewport_max.x
        && screen_position.y + half_size >= viewport_position.y && screen_position.y - half_size <= viewport_max.y;
}

float3 normalized_vector(float3 vector, float3 fallback)
{
    const float length = sgl::math::length(vector);
    return length > 1e-6f ? vector / length : fallback;
}

float3 normalized_axis(const float4x4& world_from_object, uint32_t column, float3 fallback)
{
    return normalized_vector(world_from_object.get_col(column).xyz(), fallback);
}

float3 normalized_normal_axis(const float4x4& world_from_object, uint32_t column, float3 fallback)
{
    return normalized_axis(sgl::math::transpose(sgl::math::inverse(world_from_object)), column, fallback);
}

ImU32 light_shaping_guide_color(float alpha)
{
    return ImGui::ColorConvertFloat4ToU32(ImVec4(0.35f, 0.7f, 1.f, alpha));
}

Icon light_icon(const Light* light)
{
    switch (light->light_type()) {
    case shared::LightType::point_light:
        return light->as<const PointLight>()->enable_shaping() ? Icon::light_spot : Icon::light_point;
    case shared::LightType::distant_light:
        return Icon::light_distant;
    case shared::LightType::sphere_light:
        return Icon::light_sphere;
    case shared::LightType::disk_light:
        return Icon::light_disk;
    case shared::LightType::rect_light:
        return Icon::light_rect;
    case shared::LightType::env_map_light:
        return Icon::light_env_map;
    case shared::LightType::constant_light:
    default:
        return Icon::light_constant;
    }
}

class GizmoDrawContext {
public:
    GizmoDrawContext(
        ImDrawList* draw_list,
        const float4x4& view_projection,
        float2 viewport_position,
        float2 viewport_size
    )
        : m_draw_list(draw_list)
        , m_view_projection(view_projection)
        , m_viewport_position(viewport_position)
        , m_viewport_size(viewport_size)
    {
    }

    void draw_line(float3 start, float3 end, ImU32 color, float thickness = GUIDE_THICKNESS) const
    {
        const auto segment = project_gizmo_segment(start, end, m_view_projection, m_viewport_position, m_viewport_size);
        if (!segment)
            return;

        m_draw_list->AddLine(
            ImVec2((*segment)[0].x, (*segment)[0].y),
            ImVec2((*segment)[1].x, (*segment)[1].y),
            color,
            thickness
        );
    }

    void
    draw_polyline(std::span<const float3> positions, bool closed, ImU32 color, float thickness = GUIDE_THICKNESS) const
    {
        if (positions.size() < 2)
            return;

        const size_t segment_count = closed ? positions.size() : positions.size() - 1;
        for (size_t i = 0; i < segment_count; ++i)
            draw_line(positions[i], positions[(i + 1) % positions.size()], color, thickness);
    }

    void draw_polyline(
        std::initializer_list<float3> positions,
        bool closed,
        ImU32 color,
        float thickness = GUIDE_THICKNESS
    ) const
    {
        draw_polyline(std::span<const float3>(positions.begin(), positions.size()), closed, color, thickness);
    }

    void draw_ring(float3 center, float3 axis_u, float3 axis_v, ImU32 color, float thickness = GUIDE_THICKNESS) const
    {
        constexpr uint32_t SEGMENT_COUNT = 32;
        constexpr float TWO_PI = 6.28318530718f;
        float3 start = center + axis_u;
        for (uint32_t i = 1; i <= SEGMENT_COUNT; ++i) {
            const float angle = TWO_PI * static_cast<float>(i % SEGMENT_COUNT) / static_cast<float>(SEGMENT_COUNT);
            const float3 end = center + std::cos(angle) * axis_u + std::sin(angle) * axis_v;
            draw_line(start, end, color, thickness);
            start = end;
        }
    }

    void draw_cone(
        float3 apex,
        float3 base_center,
        float3 base_axis_u,
        float3 base_axis_v,
        ImU32 color,
        float thickness = GUIDE_THICKNESS
    ) const
    {
        draw_ring(base_center, base_axis_u, base_axis_v, color, thickness);
        draw_line(apex, base_center + base_axis_u, color, thickness);
        draw_line(apex, base_center - base_axis_u, color, thickness);
        draw_line(apex, base_center + base_axis_v, color, thickness);
        draw_line(apex, base_center - base_axis_v, color, thickness);
    }

    void draw_arc(
        float3 center,
        float3 axis_u,
        float3 axis_v,
        float angle,
        ImU32 color,
        float thickness = GUIDE_THICKNESS
    ) const
    {
        constexpr uint32_t MAX_SEGMENT_COUNT = 16;
        constexpr float PI = 3.14159265359f;
        const uint32_t segment_count
            = std::max(1u, static_cast<uint32_t>(std::ceil(MAX_SEGMENT_COUNT * std::clamp(angle, 0.f, PI) / PI)));
        float3 start = center + axis_u;
        for (uint32_t i = 1; i <= segment_count; ++i) {
            const float t = angle * static_cast<float>(i) / static_cast<float>(segment_count);
            const float3 end = center + std::cos(t) * axis_u + std::sin(t) * axis_v;
            draw_line(start, end, color, thickness);
            start = end;
        }
    }

private:
    ImDrawList* m_draw_list;
    float4x4 m_view_projection;
    float2 m_viewport_position;
    float2 m_viewport_size;
};

struct LightGizmoTransform {
    float3 center;
    float3 axis_x;
    float3 axis_y;
    float3 axis_z;
    float direction_length;
};

void make_light_shaping_basis(float3 direction, const LightGizmoTransform& transform, float3& basis_u, float3& basis_v)
{
    basis_u = transform.axis_x - direction * sgl::math::dot(transform.axis_x, direction);
    float basis_u_length = sgl::math::length(basis_u);
    if (basis_u_length <= 1e-6f) {
        basis_u = transform.axis_y - direction * sgl::math::dot(transform.axis_y, direction);
        basis_u_length = sgl::math::length(basis_u);
    }
    if (basis_u_length <= 1e-6f) {
        const float3 reference = std::abs(direction.z) < 0.9f ? float3(0.f, 0.f, 1.f) : float3(0.f, 1.f, 0.f);
        basis_u = sgl::math::cross(reference, direction);
        basis_u_length = sgl::math::length(basis_u);
    }
    basis_u /= basis_u_length;
    basis_v = sgl::math::normalize(sgl::math::cross(direction, basis_u));
}

void draw_light_shaping_boundary(
    const GizmoDrawContext& context,
    const LightGizmoTransform& transform,
    float3 direction,
    float3 basis_u,
    float3 basis_v,
    float half_angle,
    bool draw_angle_arcs,
    ImU32 color,
    float thickness
)
{
    const LightGizmoConeProfile profile = make_light_gizmo_cone_profile(transform.direction_length, half_angle);
    const float3 ring_center = transform.center + profile.axial_offset * direction;
    if (profile.radius > 1e-6f) {
        context.draw_cone(
            transform.center,
            ring_center,
            profile.radius * basis_u,
            profile.radius * basis_v,
            color,
            thickness
        );
    } else {
        context.draw_line(transform.center, ring_center, color, thickness);
    }

    if (!draw_angle_arcs || half_angle <= 0.f)
        return;

    const float angle = sgl::math::radians(std::clamp(half_angle, 0.f, 180.f));
    const float3 direction_axis = transform.direction_length * direction;
    const float3 basis_axis_u = transform.direction_length * basis_u;
    const float3 basis_axis_v = transform.direction_length * basis_v;
    context.draw_arc(transform.center, direction_axis, basis_axis_u, angle, color, thickness);
    context.draw_arc(transform.center, direction_axis, -basis_axis_u, angle, color, thickness);
    context.draw_arc(transform.center, direction_axis, basis_axis_v, angle, color, thickness);
    context.draw_arc(transform.center, direction_axis, -basis_axis_v, angle, color, thickness);
}

void draw_centered_light_shaping_gizmo(
    const GizmoDrawContext& context,
    const LightGizmoTransform& transform,
    float3 direction,
    float cone_angle,
    float cone_softness,
    ImU32 color
)
{
    direction = sgl::math::normalize(direction);
    float3 basis_u;
    float3 basis_v;
    make_light_shaping_basis(direction, transform, basis_u, basis_v);

    const float cutoff_angle = std::clamp(cone_angle, 0.f, 180.f);
    const float falloff_angle = std::clamp(cone_angle * (1.f - std::clamp(cone_softness, 0.f, 1.f)), 0.f, 180.f);

    const ImU32 guide_color = light_shaping_guide_color(0.8f);
    const ImU32 falloff_color = light_shaping_guide_color(0.35f);
    if (falloff_angle < cutoff_angle - 1e-4f) {
        draw_light_shaping_boundary(
            context,
            transform,
            direction,
            basis_u,
            basis_v,
            falloff_angle,
            false,
            falloff_color,
            1.5f
        );
    }
    draw_light_shaping_boundary(
        context,
        transform,
        direction,
        basis_u,
        basis_v,
        cutoff_angle,
        cutoff_angle > 90.f,
        guide_color,
        GUIDE_THICKNESS
    );
    context.draw_line(transform.center, transform.center + transform.direction_length * direction, color);
}

void draw_area_light_shaping_rays_at_angle(
    const GizmoDrawContext& context,
    std::span<const float3> start_points,
    std::span<const float3> outward_directions,
    float3 direction,
    float guide_length,
    float half_angle,
    ImU32 color,
    float thickness
)
{
    FALCOR_ASSERT(start_points.size() == outward_directions.size());
    const LightGizmoConeProfile profile = make_light_gizmo_cone_profile(1.f, std::clamp(half_angle, 0.f, 90.f));
    for (size_t i = 0; i < start_points.size(); ++i) {
        const float3 ray_direction = profile.axial_offset * direction + profile.radius * outward_directions[i];
        context.draw_line(start_points[i], start_points[i] + guide_length * ray_direction, color, thickness);
    }
}

void draw_area_light_shaping_gizmo(
    const GizmoDrawContext& context,
    const LightGizmoTransform& transform,
    std::span<const float3> start_points,
    std::span<const float3> outward_directions,
    float3 direction,
    float cone_angle,
    float cone_softness
)
{
    direction = sgl::math::normalize(direction);
    const float cutoff_angle = std::clamp(cone_angle, 0.f, 90.f);
    const float falloff_angle = std::clamp(cone_angle * (1.f - std::clamp(cone_softness, 0.f, 1.f)), 0.f, 90.f);

    const ImU32 guide_color = light_shaping_guide_color(0.8f);
    const ImU32 falloff_color = light_shaping_guide_color(0.35f);
    if (falloff_angle < cutoff_angle - 1e-4f) {
        draw_area_light_shaping_rays_at_angle(
            context,
            start_points,
            outward_directions,
            direction,
            transform.direction_length,
            falloff_angle,
            falloff_color,
            1.5f
        );
    }
    draw_area_light_shaping_rays_at_angle(
        context,
        start_points,
        outward_directions,
        direction,
        transform.direction_length,
        cutoff_angle,
        guide_color,
        GUIDE_THICKNESS
    );
}

void draw_light_direction_gizmo(
    const GizmoDrawContext& context,
    const LightGizmoTransform& transform,
    float3 direction,
    ImU32 color
)
{
    context.draw_polyline({transform.center, transform.center + transform.direction_length * direction}, false, color);
}

void draw_sphere_light_gizmo(
    const GizmoDrawContext& context,
    const SphereLight* light,
    const float4x4& world_from_object,
    const LightGizmoTransform& transform,
    ImU32 color
)
{
    const float radius = light->radius() * sgl::math::length(world_from_object.get_col(0).xyz());
    if (radius > 0.f) {
        context.draw_ring(transform.center, radius * float3(1.f, 0.f, 0.f), radius * float3(0.f, 1.f, 0.f), color);
        context.draw_ring(transform.center, radius * float3(1.f, 0.f, 0.f), radius * float3(0.f, 0.f, 1.f), color);
        context.draw_ring(transform.center, radius * float3(0.f, 1.f, 0.f), radius * float3(0.f, 0.f, 1.f), color);
    }
    if (light->enable_shaping()) {
        draw_centered_light_shaping_gizmo(
            context,
            transform,
            -transform.axis_z,
            light->shaping_cone_angle(),
            light->shaping_cone_softness(),
            color
        );
    }
}

void draw_disk_light_gizmo(
    const GizmoDrawContext& context,
    const DiskLight* light,
    const float4x4& world_from_object,
    const LightGizmoTransform& transform,
    ImU32 color
)
{
    const float3 radius_u = light->radius() * world_from_object.get_col(0).xyz();
    const float3 radius_v = light->radius() * world_from_object.get_col(1).xyz();
    context.draw_ring(transform.center, radius_u, radius_v, color);
    const float3 direction = -normalized_normal_axis(world_from_object, 2, transform.axis_z);
    if (light->enable_shaping()) {
        const std::array<float3, 4> start_points{
            transform.center + radius_u,
            transform.center - radius_u,
            transform.center + radius_v,
            transform.center - radius_v,
        };
        const std::array<float3, 4> outward_directions{
            transform.axis_x,
            -transform.axis_x,
            transform.axis_y,
            -transform.axis_y,
        };
        draw_area_light_shaping_gizmo(
            context,
            transform,
            start_points,
            outward_directions,
            direction,
            light->shaping_cone_angle(),
            light->shaping_cone_softness()
        );
    }
    draw_light_direction_gizmo(context, transform, direction, color);
}

void draw_rect_light_gizmo(
    const GizmoDrawContext& context,
    const RectLight* light,
    const float4x4& world_from_object,
    const LightGizmoTransform& transform,
    ImU32 color
)
{
    const float3 extent_u = 0.5f * light->width() * world_from_object.get_col(0).xyz();
    const float3 extent_v = 0.5f * light->height() * world_from_object.get_col(1).xyz();
    const std::array<float3, 4> corner_offsets{
        -extent_u - extent_v,
        extent_u - extent_v,
        extent_u + extent_v,
        -extent_u + extent_v,
    };
    context.draw_polyline(
        {
            transform.center + corner_offsets[0],
            transform.center + corner_offsets[1],
            transform.center + corner_offsets[2],
            transform.center + corner_offsets[3],
        },
        true,
        color
    );
    const float3 direction = -normalized_normal_axis(world_from_object, 2, transform.axis_z);
    if (light->enable_shaping()) {
        const std::array<float3, 4> start_points{
            transform.center + corner_offsets[0],
            transform.center + corner_offsets[1],
            transform.center + corner_offsets[2],
            transform.center + corner_offsets[3],
        };
        const std::array<float3, 4> outward_directions{
            normalized_vector(
                corner_offsets[0],
                normalized_vector(-transform.axis_x - transform.axis_y, -transform.axis_x)
            ),
            normalized_vector(
                corner_offsets[1],
                normalized_vector(transform.axis_x - transform.axis_y, transform.axis_x)
            ),
            normalized_vector(
                corner_offsets[2],
                normalized_vector(transform.axis_x + transform.axis_y, transform.axis_x)
            ),
            normalized_vector(
                corner_offsets[3],
                normalized_vector(-transform.axis_x + transform.axis_y, -transform.axis_x)
            ),
        };
        draw_area_light_shaping_gizmo(
            context,
            transform,
            start_points,
            outward_directions,
            direction,
            light->shaping_cone_angle(),
            light->shaping_cone_softness()
        );
    }
    draw_light_direction_gizmo(context, transform, direction, color);
}

void draw_selected_light_gizmo(
    const GizmoDrawContext& context,
    Light* light,
    const float4x4& world_from_object,
    float3 center,
    float3 camera_position
)
{
    const LightGizmoTransform transform{
        .center = center,
        .axis_x = normalized_axis(world_from_object, 0, float3(1.f, 0.f, 0.f)),
        .axis_y = normalized_axis(world_from_object, 1, float3(0.f, 1.f, 0.f)),
        .axis_z = normalized_axis(world_from_object, 2, float3(0.f, 0.f, 1.f)),
        .direction_length = std::max(0.25f, 0.15f * sgl::math::length(center - camera_position)),
    };
    const ImU32 color = ImGui::ColorConvertFloat4ToU32(ImVec4(0.95f, 0.85f, 0.2f, 1.f));

    if (const PointLight* point = light->as<PointLight>(); point && point->enable_shaping())
        draw_centered_light_shaping_gizmo(
            context,
            transform,
            -transform.axis_z,
            point->shaping_cone_angle(),
            point->shaping_cone_softness(),
            color
        );
    else if (light->is<DistantLight>())
        draw_light_direction_gizmo(context, transform, transform.axis_z, color);
    else if (const SphereLight* sphere = light->as<SphereLight>())
        draw_sphere_light_gizmo(context, sphere, world_from_object, transform, color);
    else if (const DiskLight* disk = light->as<DiskLight>())
        draw_disk_light_gizmo(context, disk, world_from_object, transform, color);
    else if (const RectLight* rect = light->as<RectLight>())
        draw_rect_light_gizmo(context, rect, world_from_object, transform, color);
}

void draw_camera_frustum(const GizmoDrawContext& context, const CameraGizmoFrustum& frustum)
{
    const ImU32 color = ImGui::ColorConvertFloat4ToU32(ImVec4(0.95f, 0.85f, 0.2f, 1.f));
    for (const float3& corner : frustum.corners)
        context.draw_line(frustum.origin, corner, color);
    context.draw_polyline(frustum.corners, true, color);
}

} // namespace

std::optional<ProjectedGizmo> project_gizmo(
    Entity* entity,
    float3 world_position,
    const float4x4& view_projection,
    float2 viewport_position,
    float2 viewport_size
)
{
    if (!entity || viewport_size.x <= 0.f || viewport_size.y <= 0.f)
        return std::nullopt;

    const float4 clip_position = sgl::math::mul(view_projection, float4(world_position, 1.f));
    if (!is_finite(clip_position) || clip_position.w <= 1e-6f)
        return std::nullopt;

    const float3 ndc = clip_position.xyz() / clip_position.w;
    if (ndc.z < 0.f || ndc.z > 1.f)
        return std::nullopt;

    return ProjectedGizmo{
        .entity = entity,
        .screen_position = ndc_to_screen(ndc.xy(), viewport_position, viewport_size),
        .depth = ndc.z,
    };
}

std::optional<std::array<float2, 2>> project_gizmo_segment(
    float3 start,
    float3 end,
    const float4x4& view_projection,
    float2 viewport_position,
    float2 viewport_size
)
{
    if (viewport_size.x <= 0.f || viewport_size.y <= 0.f)
        return std::nullopt;

    const float4 clip_start = sgl::math::mul(view_projection, float4(start, 1.f));
    const float4 clip_end = sgl::math::mul(view_projection, float4(end, 1.f));
    if (!is_finite(clip_start) || !is_finite(clip_end))
        return std::nullopt;

    float t_min = 0.f;
    float t_max = 1.f;
    auto clip_plane = [&](float start_distance, float end_distance)
    {
        if (start_distance < 0.f && end_distance < 0.f)
            return false;
        if (start_distance < 0.f || end_distance < 0.f) {
            const float t = start_distance / (start_distance - end_distance);
            if (start_distance < 0.f)
                t_min = std::max(t_min, t);
            else
                t_max = std::min(t_max, t);
        }
        return t_min <= t_max;
    };

    constexpr float MIN_CLIP_W = 1e-6f;
    const std::array<std::array<float, 2>, 7> plane_distances{
        std::array{clip_start.x + clip_start.w, clip_end.x + clip_end.w},
        std::array{clip_start.w - clip_start.x, clip_end.w - clip_end.x},
        std::array{clip_start.y + clip_start.w, clip_end.y + clip_end.w},
        std::array{clip_start.w - clip_start.y, clip_end.w - clip_end.y},
        std::array{clip_start.z, clip_end.z},
        std::array{clip_start.w - clip_start.z, clip_end.w - clip_end.z},
        std::array{clip_start.w - MIN_CLIP_W, clip_end.w - MIN_CLIP_W},
    };
    for (const auto& distances : plane_distances) {
        if (!clip_plane(distances[0], distances[1]))
            return std::nullopt;
    }

    const float4 clipped_start = clip_start + t_min * (clip_end - clip_start);
    const float4 clipped_end = clip_start + t_max * (clip_end - clip_start);
    const float2 start_ndc = clipped_start.xy() / clipped_start.w;
    const float2 end_ndc = clipped_end.xy() / clipped_end.w;
    return std::array{
        ndc_to_screen(start_ndc, viewport_position, viewport_size),
        ndc_to_screen(end_ndc, viewport_position, viewport_size),
    };
}

Entity* pick_gizmo(std::span<const ProjectedGizmo> gizmos, float2 screen_position)
{
    float best_distance_squared = std::numeric_limits<float>::infinity();
    float best_depth = std::numeric_limits<float>::infinity();
    Entity* best_entity = nullptr;

    for (const ProjectedGizmo& gizmo : gizmos) {
        if (!gizmo.entity || gizmo.hit_radius < 0.f)
            continue;

        const float2 offset = gizmo.screen_position - screen_position;
        const float distance_squared = offset.x * offset.x + offset.y * offset.y;
        if (distance_squared > gizmo.hit_radius * gizmo.hit_radius)
            continue;

        if (distance_squared < best_distance_squared
            || (distance_squared == best_distance_squared && gizmo.depth < best_depth)) {
            best_distance_squared = distance_squared;
            best_depth = gizmo.depth;
            best_entity = gizmo.entity;
        }
    }

    return best_entity;
}

CameraGizmoFrustum make_camera_gizmo_frustum(const Camera* camera, float display_length)
{
    FALCOR_CHECK(camera && camera->entity(), "Cannot construct a camera gizmo frustum without a camera entity.");

    const float4x4 world_from_object = camera->entity()->world_from_object_matrix();
    const float3 origin = world_from_object.get_col(3).xyz();
    const float3 right = normalized_axis(world_from_object, 0, float3(1.f, 0.f, 0.f));
    const float3 up = normalized_axis(world_from_object, 1, float3(0.f, 1.f, 0.f));
    const float3 forward = -normalized_axis(world_from_object, 2, float3(0.f, 0.f, 1.f));

    display_length = std::max(display_length, 0.f);
    const float aspect = camera->height() > 0 ? std::max(float(camera->width()) / float(camera->height()), 0.f) : 1.f;
    const float half_height
        = display_length * std::tan(0.5f * sgl::math::radians(std::clamp(camera->fov_y(), 1.f, 179.f)));
    const float half_width = aspect * half_height;
    const float3 center = origin + display_length * forward;
    const float3 extent_x = half_width * right;
    const float3 extent_y = half_height * up;

    return {
        .origin = origin,
        .corners = {
            center - extent_x - extent_y,
            center + extent_x - extent_y,
            center + extent_x + extent_y,
            center - extent_x + extent_y,
        },
    };
}

LightGizmoConeProfile make_light_gizmo_cone_profile(float guide_length, float half_angle)
{
    const float length = std::max(guide_length, 0.f);
    const float angle = sgl::math::radians(std::clamp(half_angle, 0.f, 180.f));
    return {
        .axial_offset = length * std::cos(angle),
        .radius = length * std::max(std::sin(angle), 0.f),
    };
}

SceneGizmoRenderer::SceneGizmoRenderer(ref<IconLibrary> icons)
    : m_icons(std::move(icons))
{
    FALCOR_CHECK(m_icons, "Cannot create a scene gizmo renderer without UI icons.");
}

Entity* SceneGizmoRenderer::pick(
    Scene* scene,
    const float4x4& view,
    const float4x4& projection,
    float2 viewport_position,
    float2 viewport_size,
    float2 screen_position,
    bool show_cameras,
    bool show_lights
) const
{
    if (!scene || !m_icons->texture() || viewport_size.x <= 0.f || viewport_size.y <= 0.f)
        return nullptr;

    const float4x4 view_projection = sgl::math::mul(projection, view);
    std::vector<ProjectedGizmo> gizmos;
    auto append_gizmo = [&](Entity* entity, float3 position, Icon icon)
    {
        std::optional<ProjectedGizmo> gizmo
            = project_gizmo(entity, position, view_projection, viewport_position, viewport_size);
        if (!gizmo)
            return;
        const float icon_half_size = 0.5f * m_icons->icon(icon).preferred_size;
        if (!overlaps_viewport(gizmo->screen_position, icon_half_size, viewport_position, viewport_size))
            return;
        gizmo->hit_radius = icon_half_size + HIT_PADDING;
        gizmos.push_back(*gizmo);
    };

    for (Component* component : scene->components()) {
        if (show_cameras) {
            Camera* camera = component->as<Camera>();
            if (camera && camera->is_valid() && camera->entity() && camera->entity()->is_valid()) {
                Entity* entity = camera->entity();
                append_gizmo(entity, entity->world_from_object_matrix().get_col(3).xyz(), Icon::camera);
            }
        }
        if (show_lights) {
            Light* light = component->as<Light>();
            if (light && light->is_valid() && light->entity() && light->entity()->is_valid()) {
                Entity* entity = light->entity();
                append_gizmo(entity, entity->world_from_object_matrix().get_col(3).xyz(), light_icon(light));
            }
        }
    }

    return pick_gizmo(gizmos, screen_position);
}

void SceneGizmoRenderer::draw(
    Scene* scene,
    SceneObject* selected_object,
    const float4x4& view,
    const float4x4& projection,
    float2 viewport_position,
    float2 viewport_size,
    bool show_cameras,
    bool show_lights
)
{
    m_visible_gizmos.clear();
    if (!scene || !m_icons->texture() || viewport_size.x <= 0.f || viewport_size.y <= 0.f)
        return;

    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    const ImVec2 clip_min(viewport_position.x, viewport_position.y);
    const ImVec2 clip_max(viewport_position.x + viewport_size.x, viewport_position.y + viewport_size.y);
    draw_list->PushClipRect(clip_min, clip_max, true);

    const float4x4 view_projection = sgl::math::mul(projection, view);
    const GizmoDrawContext context(draw_list, view_projection, viewport_position, viewport_size);
    const float3 viewport_camera_position = sgl::math::inverse(view).get_col(3).xyz();

    auto append_visible_gizmo
        = [&](const ProjectedGizmo& projected, IconImage image, float4 color, bool selected, bool active_accent)
    {
        const float icon_half_size = 0.5f * image.preferred_size;
        if (overlaps_viewport(projected.screen_position, icon_half_size, viewport_position, viewport_size)) {
            m_visible_gizmos.push_back({
                .projected = projected,
                .image = image,
                .color = color,
                .selected = selected,
                .active_accent = active_accent,
            });
        }
    };

    for (Component* component : scene->components()) {
        if (show_cameras) {
            Camera* camera = component->as<Camera>();
            if (camera && camera->is_valid() && camera->entity() && camera->entity()->is_valid()) {
                Entity* entity = camera->entity();
                const float3 position = entity->world_from_object_matrix().get_col(3).xyz();
                std::optional<ProjectedGizmo> projected
                    = project_gizmo(entity, position, view_projection, viewport_position, viewport_size);
                if (projected) {
                    const bool selected = selected_object == entity;
                    const float4 color = selected ? float4(1.f, 0.9f, 0.25f, 1.f) : float4(0.65f, 0.8f, 1.f, 0.95f);
                    append_visible_gizmo(
                        *projected,
                        m_icons->icon(Icon::camera),
                        color,
                        selected,
                        scene->active_camera() == camera
                    );
                    if (selected) {
                        const float display_length
                            = std::max(0.25f, 0.15f * sgl::math::length(position - viewport_camera_position));
                        draw_camera_frustum(context, make_camera_gizmo_frustum(camera, display_length));
                    }
                }
            }
        }

        if (show_lights) {
            Light* light = component->as<Light>();
            if (light && light->is_valid() && light->entity() && light->entity()->is_valid()) {
                Entity* entity = light->entity();
                const float4x4 world_from_object = entity->world_from_object_matrix();
                const float3 center = world_from_object.get_col(3).xyz();
                std::optional<ProjectedGizmo> projected
                    = project_gizmo(entity, center, view_projection, viewport_position, viewport_size);
                if (projected) {
                    const bool selected = selected_object == entity;
                    const float4 color = selected
                        ? float4(1.f, 0.9f, 0.25f, 1.f)
                        : (light->active() ? float4(1.f, 0.65f, 0.12f, 0.95f) : float4(0.55f, 0.55f, 0.55f, 0.8f));
                    append_visible_gizmo(*projected, m_icons->icon(light_icon(light)), color, selected, false);
                    if (selected)
                        draw_selected_light_gizmo(context, light, world_from_object, center, viewport_camera_position);
                }
            }
        }
    }

    // Draw farther icons first so nearer icons appear on top, regardless of gizmo type.
    std::stable_sort(
        m_visible_gizmos.begin(),
        m_visible_gizmos.end(),
        [](const VisibleGizmo& lhs, const VisibleGizmo& rhs)
        {
            return lhs.projected.depth > rhs.projected.depth;
        }
    );

    // Keep all image submissions adjacent so ImGui can batch them with one texture binding.
    for (const VisibleGizmo& gizmo : m_visible_gizmos) {
        if (!gizmo.selected)
            continue;
        const float halo_half_size = 0.5f * gizmo.image.preferred_size + HALO_PADDING;
        const ImVec2 center(gizmo.projected.screen_position.x, gizmo.projected.screen_position.y);
        draw_list->AddImage(
            gizmo.image.texture,
            ImVec2(center.x - halo_half_size, center.y - halo_half_size),
            ImVec2(center.x + halo_half_size, center.y + halo_half_size),
            ImVec2(gizmo.image.uv_min.x, gizmo.image.uv_min.y),
            ImVec2(gizmo.image.uv_max.x, gizmo.image.uv_max.y),
            ImGui::ColorConvertFloat4ToU32(ImVec4(0.95f, 0.85f, 0.2f, 0.5f))
        );
    }

    for (const VisibleGizmo& gizmo : m_visible_gizmos) {
        const float icon_half_size = 0.5f * gizmo.image.preferred_size;
        const ImVec2 center(gizmo.projected.screen_position.x, gizmo.projected.screen_position.y);
        draw_list->AddImage(
            gizmo.image.texture,
            ImVec2(center.x - icon_half_size, center.y - icon_half_size),
            ImVec2(center.x + icon_half_size, center.y + icon_half_size),
            ImVec2(gizmo.image.uv_min.x, gizmo.image.uv_min.y),
            ImVec2(gizmo.image.uv_max.x, gizmo.image.uv_max.y),
            ImGui::ColorConvertFloat4ToU32(ImVec4(gizmo.color.x, gizmo.color.y, gizmo.color.z, gizmo.color.w))
        );
    }

    for (const VisibleGizmo& gizmo : m_visible_gizmos) {
        if (!gizmo.active_accent)
            continue;
        const float radius = 0.5f * gizmo.image.preferred_size + 3.f;
        draw_list->AddCircle(
            ImVec2(gizmo.projected.screen_position.x, gizmo.projected.screen_position.y),
            radius,
            ImGui::ColorConvertFloat4ToU32(ImVec4(0.2f, 1.f, 0.65f, 1.f)),
            24,
            2.f
        );
    }

    draw_list->PopClipRect();
}

} // namespace falcor::ui::detail
