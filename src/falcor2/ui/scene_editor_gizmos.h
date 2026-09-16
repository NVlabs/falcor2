// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/core/macros.h"
#include "falcor2/core/object.h"
#include "falcor2/core/types.h"
#include "falcor2/render/fwd.h"
#include "falcor2/ui/icon_library.h"

#include <array>
#include <optional>
#include <span>
#include <vector>

namespace falcor::ui::detail {

/// Screen-space representation of one selectable editor gizmo.
struct ProjectedGizmo {
    Entity* entity{nullptr};
    float2 screen_position{0.f};
    float depth{0.f};
    float hit_radius{0.f};
};

/// Editor-scaled world-space camera frustum used for selected-camera guides.
struct CameraGizmoFrustum {
    float3 origin{0.f};
    std::array<float3, 4> corners{};
};

/// Axial and radial coordinates of a cone boundary at a fixed distance from its apex.
struct LightGizmoConeProfile {
    float axial_offset{0.f};
    float radius{0.f};
};

/// Draws and picks all editor-only scene gizmos.
class SceneGizmoRenderer : public Object {
    FALCOR_OBJECT(SceneGizmoRenderer)
public:
    explicit SceneGizmoRenderer(ref<IconLibrary> icons);

    void draw(
        Scene* scene,
        SceneObject* selected_object,
        const float4x4& view,
        const float4x4& projection,
        float2 viewport_position,
        float2 viewport_size,
        bool show_cameras,
        bool show_lights
    );

    Entity* pick(
        Scene* scene,
        const float4x4& view,
        const float4x4& projection,
        float2 viewport_position,
        float2 viewport_size,
        float2 screen_position,
        bool show_cameras,
        bool show_lights
    ) const;

private:
    struct VisibleGizmo {
        ProjectedGizmo projected;
        IconImage image;
        float4 color{1.f};
        bool selected{false};
        bool active_accent{false};
    };

    ref<IconLibrary> m_icons;
    std::vector<VisibleGizmo> m_visible_gizmos;
};

/// Project a world-space point into viewport screen space.
/// The returned position may be outside the viewport rectangle so partially visible
/// icons can still be drawn and picked. Returns no value when the point is outside
/// the depth range, behind the camera, or cannot be projected to finite coordinates.
FALCOR_API std::optional<ProjectedGizmo> project_gizmo(
    Entity* entity,
    float3 world_position,
    const float4x4& view_projection,
    float2 viewport_position,
    float2 viewport_size
);

/// Project and clip a world-space line segment to a viewport rectangle.
/// Returns the two screen-space endpoints of the visible segment, or no value when
/// the segment does not intersect the view frustum.
FALCOR_API std::optional<std::array<float2, 2>> project_gizmo_segment(
    float3 start,
    float3 end,
    const float4x4& view_projection,
    float2 viewport_position,
    float2 viewport_size
);

/// Pick the closest projected gizmo inside its screen-space hit radius.
/// Screen-space distance is the primary ordering key. Depth resolves exact overlaps.
FALCOR_API Entity* pick_gizmo(std::span<const ProjectedGizmo> gizmos, float2 screen_position);

/// Construct a camera frustum whose image plane lies @p display_length units from the camera.
FALCOR_API CameraGizmoFrustum make_camera_gizmo_frustum(const Camera* camera, float display_length);

/// Construct a cone boundary profile that supports half-angles over the full [0, 180] degree range.
FALCOR_API LightGizmoConeProfile make_light_gizmo_cone_profile(float guide_length, float half_angle);

} // namespace falcor::ui::detail
