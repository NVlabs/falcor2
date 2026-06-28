// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/core/object.h"
#include "falcor2/core/types.h"
#include "falcor2/render/transform.h"

#include <sgl/core/input.h>

#include <array>

namespace falcor::ui {

/// Controls camera navigation in a 3D viewport.
///
/// Provides Unreal Editor-style camera controls with multiple interaction modes:
///
/// ## Navigation Modes
///
/// - **First-person (RMB held):** WASD/QE movement with mouse-look rotation.
///   Shift for fast movement, Ctrl for slow. Scroll wheel adjusts move speed.
/// - **Pan (MMB held):** Pans the camera in the view plane.
/// - **Orbit (Alt+LMB held):** Tumbles the camera around the pivot point.
///   Vertical rotation is clamped to prevent gimbal lock.
/// - **Dolly (Alt+RMB held):** Zooms toward or away from the pivot point.
/// - **Track (Alt+MMB held):** Pans the camera and pivot together in the view plane.
///   Pan speed scales with orbit distance for a consistent feel at any zoom level.
/// - **Scroll (idle):** Dollies toward or away from the pivot point.
///
/// ## Pivot Point
///
/// The pivot is the point around which orbit and dolly operations revolve.
/// It is automatically kept at `orbit_distance` ahead of the camera along its forward
/// vector after first-person movement, and can be set explicitly via `set_pivot()` or
/// `focus()`. Drag and track operations translate the pivot along with the camera.
///
/// ## Usage
///
/// Feed keyboard and mouse events via `handle_keyboard_event()` / `handle_mouse_event()`,
/// then call `update(dt)` once per frame. The controller updates its internal transform;
/// the caller is responsible for applying it back to the camera entity.
///
class FALCOR_API CameraController : public Object {
public:
    /// Minimum movement speed.
    static constexpr float MIN_MOVE_SPEED = 0.01f;
    /// Maximum movement speed.
    static constexpr float MAX_MOVE_SPEED = 1000.f;

    /// Interaction state of the controller.
    enum class State {
        /// No active interaction.
        idle,
        /// RMB held: fly-mode with WASD/QE.
        first_person,
        /// MMB held: pan in view plane.
        pan,
        /// Alt+LMB held: tumble around pivot.
        orbit,
        /// Alt+RMB held: zoom toward/away from pivot.
        dolly,
        /// Alt+MMB held: pan camera and pivot together.
        track,
    };
    SGL_ENUM_INFO(
        State,
        {
            {State::idle, "idle"},
            {State::first_person, "first_person"},
            {State::pan, "pan"},
            {State::orbit, "orbit"},
            {State::dolly, "dolly"},
            {State::track, "track"},
        }
    );

    CameraController();

    /// Camera transform.
    const Transform& transform() const { return m_transform; }

    /// Set the camera transform. Also recomputes the pivot from the current orbit distance.
    void set_transform(const Transform& transform);

    /// Pivot point used as the center of orbit/dolly operations.
    float3 pivot() const { return m_pivot; }

    /// Set the pivot point. Updates orbit distance to match the camera-to-pivot distance.
    void set_pivot(const float3& pivot);

    /// Distance from the camera to the pivot point.
    float orbit_distance() const { return m_orbit_distance; }

    /// Set the orbit distance. Recomputes the pivot along the camera's forward vector.
    void set_orbit_distance(float distance);

    /// Movement speed in world units per second (first-person mode).
    float move_speed() const { return m_move_speed; }

    /// Set the movement speed, clamped to [MIN_MOVE_SPEED, MAX_MOVE_SPEED].
    void set_move_speed(float speed);

    /// Whether smoothing/inertia is enabled for orbit and scroll.
    bool smoothing() const { return m_smoothing_enabled; }

    /// Enable or disable smoothing/inertia for orbit and scroll.
    void set_smoothing(bool enabled);

    /// Whether the controller is currently in an active interaction mode.
    bool is_interacting() const { return m_state != State::idle; }

    /// Current interaction mode state.
    State state() const { return m_state; }

    /// Focus the camera on a target point.
    /// Sets the pivot to `target` and repositions the camera at `distance` along its
    /// current backward direction so that it looks at the target.
    /// @param target World-space point to focus on.
    /// @param distance Distance from target to place the camera.
    void focus(const float3& target, float distance);

    /// Process a keyboard event. Updates modifier and movement key state.
    /// @return True if the event is consumed by camera movement.
    bool handle_keyboard_event(const sgl::KeyboardEvent& event);

    /// Process a mouse event. Manages mode transitions and accumulates mouse delta/scroll.
    /// @return True if the event affected or was owned by camera interaction.
    bool handle_mouse_event(const sgl::MouseEvent& event);

    /// Cancel any active interaction and clear transient input state.
    void cancel_interaction();

    /// Advance the controller by `dt` seconds.
    /// Applies movement, rotation, orbit, etc. based on the current mode and accumulated input.
    /// @return True if the transform changed this frame.
    bool update(float dt);

private:
    /// Recompute the pivot point from the camera position and forward direction.
    void update_pivot_from_camera();

    /// Per-mode update functions. Each returns true if the transform was modified.
    bool update_scroll(float dt, bool apply_smoothing);
    bool update_first_person(float dt, bool apply_smoothing);
    bool update_pan();
    bool update_orbit(float dt, bool apply_smoothing);
    bool update_dolly();
    bool update_track();

    /// Apply a scroll dolly by the given amount. Positive scrolls toward the pivot.
    /// Uses multiplicative zoom so the camera can never reach the pivot.
    bool apply_scroll_dolly(float log_amount);

    /// Apply an orbit by the given yaw/pitch angular deltas (radians).
    /// Core implementation shared by all orbit paths.
    bool apply_orbit_delta(float yaw_delta, float pitch_delta);

    /// Apply orbit at the given yaw/pitch angular velocities (radians/second).
    /// Convenience wrapper over apply_orbit_delta for inertia coast.
    bool apply_orbit(float yaw_speed, float pitch_speed, float dt);

    /// Number of movement keys (WASD + QE).
    static constexpr size_t MOVE_KEY_COUNT = 6;
    /// Speed multiplier when Shift is held.
    static constexpr float MOVE_FAST_FACTOR = 10.f;
    /// Speed multiplier when Ctrl is held.
    static constexpr float MOVE_SLOW_FACTOR = 0.1f;
    /// Radians per pixel of mouse movement for orbit.
    static constexpr float ORBIT_SPEED = 0.005f;
    /// Dolly factor per pixel of mouse movement.
    static constexpr float DOLLY_SPEED = 0.005f;
    /// Base track/pan speed per pixel of mouse movement.
    static constexpr float TRACK_SPEED = 0.001f;
    /// Minimum camera-to-pivot distance.
    static constexpr float MIN_ORBIT_DISTANCE = 0.01f;
    /// Multiplicative speed change per scroll tick.
    static constexpr float SCROLL_SPEED_FACTOR = 1.1f;
    /// Time constant for orbit velocity tracking during active drag (seconds). Lower = more responsive.
    static constexpr float ORBIT_SMOOTH_TIME = 0.05f;
    /// Time constant for orbit inertia decay after button release (seconds).
    static constexpr float ORBIT_INERTIA_TIME = 0.15f;
    /// Time constant for first-person look smoothing (seconds).
    static constexpr float LOOK_SMOOTH_TIME = 0.03f;
    /// Time constant for scroll velocity decay (seconds).
    static constexpr float SCROLL_SMOOTH_TIME = 0.1f;
    /// Orbit angular velocity below this is zeroed (radians/second).
    static constexpr float ORBIT_STOP_THRESHOLD = 0.05f;
    /// Scroll velocity below this is zeroed (units/second).
    static constexpr float SCROLL_STOP_THRESHOLD = 0.5f;
    /// Look angular velocity below this is zeroed (radians/second).
    static constexpr float LOOK_STOP_THRESHOLD = 0.05f;

    /// Camera transform (position, rotation, scale).
    Transform m_transform;
    /// Current interaction state.
    State m_state{State::idle};
    /// First-person movement speed in world units per second.
    float m_move_speed{5.f};
    /// Mouse rotation sensitivity in radians per pixel.
    float m_rotate_speed{0.002f};
    /// World-space pivot point for orbit/dolly operations.
    float3 m_pivot{0.f, 0.f, -5.f};
    /// Distance from camera to pivot.
    float m_orbit_distance{5.f};

    // Accumulated mouse input state.
    float2 m_mouse_pos{0.f};
    float2 m_mouse_delta{0.f};
    bool m_first_mouse_delta{true};
    float2 m_mouse_scroll_delta{0.f};

    // Keyboard state.
    std::array<bool, MOVE_KEY_COUNT> m_move_key_down{};
    bool m_shift_down{false};
    bool m_ctrl_down{false};
    bool m_alt_down{false};

    /// Whether smoothing/inertia is enabled.
    bool m_smoothing_enabled{true};
    /// Orbit angular velocity in radians/second (yaw, pitch) for inertia.
    float2 m_orbit_velocity{0.f};
    /// Smoothed first-person look velocity (yaw, pitch) in radians/second.
    float2 m_look_velocity{0.f};
    /// Scroll dolly velocity in units/second for smooth scrolling.
    float m_scroll_velocity{0.f};
};

SGL_ENUM_REGISTER(CameraController::State);

} // namespace falcor::ui
