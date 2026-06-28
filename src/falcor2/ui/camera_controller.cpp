// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "camera_controller.h"

#include <algorithm>
#include <cmath>

namespace falcor::ui {

namespace {

struct MoveKeyBinding {
    sgl::KeyCode key;
    float3 direction;
};

static const MoveKeyBinding MOVE_KEYS[] = {
    {sgl::KeyCode::a, float3(-1, 0, 0)},
    {sgl::KeyCode::d, float3(1, 0, 0)},
    {sgl::KeyCode::e, float3(0, 1, 0)},
    {sgl::KeyCode::q, float3(0, -1, 0)},
    {sgl::KeyCode::w, float3(0, 0, 1)},
    {sgl::KeyCode::s, float3(0, 0, -1)},
};

bool is_move_key(sgl::KeyCode key)
{
    for (const MoveKeyBinding& binding : MOVE_KEYS) {
        if (binding.key == key)
            return true;
    }
    return false;
}

bool is_shift_key(sgl::KeyCode key)
{
    return key == sgl::KeyCode::left_shift || key == sgl::KeyCode::right_shift;
}

bool is_ctrl_key(sgl::KeyCode key)
{
    return key == sgl::KeyCode::left_control || key == sgl::KeyCode::right_control;
}

bool is_alt_key(sgl::KeyCode key)
{
    return key == sgl::KeyCode::left_alt || key == sgl::KeyCode::right_alt;
}

} // anonymous namespace

CameraController::CameraController()
{
    static_assert(std::size(MOVE_KEYS) == MOVE_KEY_COUNT);
    m_move_key_down.fill(false);
}

void CameraController::set_transform(const Transform& transform)
{
    m_transform = transform;
    update_pivot_from_camera();
}

void CameraController::set_pivot(const float3& pivot)
{
    m_pivot = pivot;
    m_orbit_distance = length(m_transform.translation() - m_pivot);
    m_orbit_distance = std::max(m_orbit_distance, MIN_ORBIT_DISTANCE);
}

void CameraController::set_orbit_distance(float distance)
{
    m_orbit_distance = std::max(distance, MIN_ORBIT_DISTANCE);
    update_pivot_from_camera();
}

void CameraController::set_move_speed(float speed)
{
    m_move_speed = std::clamp(speed, MIN_MOVE_SPEED, MAX_MOVE_SPEED);
}

void CameraController::set_smoothing(bool enabled)
{
    m_smoothing_enabled = enabled;
    if (!enabled) {
        m_orbit_velocity = float2(0.f);
        m_look_velocity = float2(0.f);
        m_scroll_velocity = 0.f;
    }
}

void CameraController::focus(const float3& target, float distance)
{
    distance = std::max(distance, MIN_ORBIT_DISTANCE);
    m_pivot = target;
    m_orbit_distance = distance;

    // Reposition camera to look at target from current direction.
    float3x3 rot = sgl::math::matrix_from_quat(m_transform.rotation());
    float3 fwd = -rot.get_col(2);
    float3 new_pos = target - fwd * distance;
    m_transform.set_translation(new_pos);
}

bool CameraController::handle_keyboard_event(const sgl::KeyboardEvent& event)
{
    bool camera_key = is_move_key(event.key) || is_shift_key(event.key) || is_ctrl_key(event.key);

    if (event.is_key_press() || event.is_key_release()) {
        bool down = event.is_key_press();
        bool move_key = false;
        for (size_t i = 0; i < MOVE_KEY_COUNT; ++i) {
            if (MOVE_KEYS[i].key == event.key) {
                if (m_state == State::first_person || !down)
                    m_move_key_down[i] = down;
                move_key = true;
                break;
            }
        }
        if (!move_key) {
            if (is_shift_key(event.key))
                m_shift_down = down;
            else if (is_ctrl_key(event.key))
                m_ctrl_down = down;
            else if (is_alt_key(event.key))
                m_alt_down = down;
        }
    }

    return m_state == State::first_person && camera_key
        && (event.is_key_press() || event.is_key_release() || event.is_key_repeat());
}

bool CameraController::handle_mouse_event(const sgl::MouseEvent& event)
{
    m_shift_down = event.has_modifier(sgl::KeyModifier::shift);
    m_ctrl_down = event.has_modifier(sgl::KeyModifier::ctrl);
    m_alt_down = event.has_modifier(sgl::KeyModifier::alt);

    bool handled = m_state != State::idle;

    switch (m_state) {
    case State::idle: {
        m_first_mouse_delta = true;
        if (event.is_button_down()) {
            if (m_alt_down) {
                // Alt + button: orbit/dolly/track modes.
                if (event.button == sgl::MouseButton::left) {
                    m_state = State::orbit;
                    handled = true;
                } else if (event.button == sgl::MouseButton::right) {
                    m_state = State::dolly;
                    handled = true;
                } else if (event.button == sgl::MouseButton::middle) {
                    m_state = State::track;
                    handled = true;
                }
            } else {
                // No Alt: original behavior.
                if (event.button == sgl::MouseButton::right) {
                    m_state = State::first_person;
                    handled = true;
                } else if (event.button == sgl::MouseButton::middle) {
                    m_state = State::pan;
                    handled = true;
                }
            }
        }
        break;
    }
    case State::first_person:
        if (event.is_button_up() && event.button == sgl::MouseButton::right) {
            m_state = State::idle;
            m_look_velocity = float2(0.f);
        }
        break;
    case State::pan:
        if (event.is_button_up() && event.button == sgl::MouseButton::middle) {
            m_state = State::idle;
        }
        break;
    case State::orbit:
        if (event.is_button_up() && event.button == sgl::MouseButton::left) {
            m_state = State::idle;
        }
        break;
    case State::dolly:
        if (event.is_button_up() && event.button == sgl::MouseButton::right) {
            m_state = State::idle;
        }
        break;
    case State::track:
        if (event.is_button_up() && event.button == sgl::MouseButton::middle) {
            m_state = State::idle;
        }
        break;
    }

    if (event.is_move()) {
        m_mouse_delta = event.pos - m_mouse_pos;
        if (m_first_mouse_delta) {
            m_mouse_delta = float2(0.f);
            m_first_mouse_delta = false;
        }
        m_mouse_pos = event.pos;
    }

    if (event.is_scroll()) {
        m_mouse_scroll_delta = event.scroll;
        handled = true;
    }

    return handled;
}

bool CameraController::update(float dt)
{
    // When dt is non-positive, time-dependent smoothing/inertia cannot be applied.
    bool apply_smoothing = m_smoothing_enabled && dt > 0.f;

    bool changed = false;
    switch (m_state) {
    case State::idle:
        changed = update_scroll(dt, apply_smoothing);
        break;
    case State::first_person:
        changed = update_first_person(dt, apply_smoothing);
        break;
    case State::pan:
        changed = update_pan();
        break;
    case State::orbit:
        changed = update_orbit(dt, apply_smoothing);
        break;
    case State::dolly:
        changed = update_dolly();
        break;
    case State::track:
        changed = update_track();
        break;
    }

    // Apply orbit inertia when idle and smoothing is enabled.
    if (m_state == State::idle && apply_smoothing && length(m_orbit_velocity) > ORBIT_STOP_THRESHOLD) {
        changed |= apply_orbit(m_orbit_velocity.x, m_orbit_velocity.y, dt);
        m_orbit_velocity *= std::exp(-dt / ORBIT_INERTIA_TIME);
        if (length(m_orbit_velocity) <= ORBIT_STOP_THRESHOLD)
            m_orbit_velocity = float2(0.f);
    }

    m_mouse_delta = float2(0.f);
    m_mouse_scroll_delta = float2(0.f);
    return changed;
}

void CameraController::update_pivot_from_camera()
{
    float3x3 rot = sgl::math::matrix_from_quat(m_transform.rotation());
    float3 fwd = -rot.get_col(2);
    m_pivot = m_transform.translation() + fwd * m_orbit_distance;
}

void CameraController::cancel_interaction()
{
    m_state = State::idle;
    m_mouse_delta = float2(0.f);
    m_mouse_scroll_delta = float2(0.f);
    m_first_mouse_delta = true;
    m_move_key_down.fill(false);
    m_shift_down = false;
    m_ctrl_down = false;
    m_alt_down = false;
    m_orbit_velocity = float2(0.f);
    m_look_velocity = float2(0.f);
    m_scroll_velocity = 0.f;
}

bool CameraController::update_scroll(float dt, bool apply_smoothing)
{
    bool changed = false;

    if (m_mouse_scroll_delta.y != 0.f) {
        if (apply_smoothing) {
            // Accumulate scroll ticks into log-space velocity.
            m_scroll_velocity += m_mouse_scroll_delta.y * 3.f;
        } else {
            // No smoothing: apply directly.
            changed |= apply_scroll_dolly(m_mouse_scroll_delta.y * 0.1f);
        }
    }

    // Apply and decay smooth scroll velocity.
    if (apply_smoothing && std::abs(m_scroll_velocity) > SCROLL_STOP_THRESHOLD) {
        changed |= apply_scroll_dolly(m_scroll_velocity * dt);
        m_scroll_velocity *= std::exp(-dt / SCROLL_SMOOTH_TIME);
        if (std::abs(m_scroll_velocity) <= SCROLL_STOP_THRESHOLD)
            m_scroll_velocity = 0.f;
    }

    return changed;
}

bool CameraController::update_first_person(float dt, bool apply_smoothing)
{
    bool changed = false;

    float3 pos = m_transform.translation();
    float3x3 rot = sgl::math::matrix_from_quat(m_transform.rotation());
    float3 fwd = -rot.get_col(2);
    float3 right = rot.get_col(0);
    float3 up = rot.get_col(1);

    // Scroll wheel adjusts movement speed in first-person mode.
    if (m_mouse_scroll_delta.y != 0.f) {
        float factor = (m_mouse_scroll_delta.y > 0.f) ? SCROLL_SPEED_FACTOR : (1.f / SCROLL_SPEED_FACTOR);
        int ticks = (int)std::abs(m_mouse_scroll_delta.y);
        for (int i = 0; i < ticks; ++i)
            m_move_speed *= factor;
        m_move_speed = std::clamp(m_move_speed, MIN_MOVE_SPEED, MAX_MOVE_SPEED);
    }

    // Move
    float3 move_delta{0.f};
    for (size_t i = 0; i < MOVE_KEY_COUNT; ++i) {
        if (m_move_key_down[i])
            move_delta += MOVE_KEYS[i].direction;
    }
    float speed = m_move_speed;
    if (m_shift_down)
        speed = speed * MOVE_FAST_FACTOR;
    if (m_ctrl_down)
        speed = speed * MOVE_SLOW_FACTOR;
    move_delta *= speed;
    if (length(move_delta) > 0.f) {
        float3 offset = right * move_delta.x;
        offset += up * move_delta.y;
        offset += fwd * move_delta.z;
        pos += offset * dt;
        m_transform.set_translation(pos);
        changed = true;
    }

    // Rotate
    if (apply_smoothing) {
        // Smooth mouse look: blend look velocity toward instantaneous input.
        float2 target_look{0.f};
        if (length(m_mouse_delta) > 0.f)
            target_look = float2(-m_mouse_delta.x * m_rotate_speed / dt, -m_mouse_delta.y * m_rotate_speed / dt);
        float blend = 1.f - std::exp(-dt / LOOK_SMOOTH_TIME);
        m_look_velocity += (target_look - m_look_velocity) * blend;
        if (length(m_look_velocity) > LOOK_STOP_THRESHOLD) {
            quatf horiz = sgl::math::quat_from_angle_axis(m_look_velocity.x * dt, float3(0, 1, 0));
            quatf vert = sgl::math::quat_from_angle_axis(m_look_velocity.y * dt, float3(1, 0, 0));
            m_transform.set_rotation(mul(mul(horiz, m_transform.rotation()), vert));
            changed = true;
        }
    } else {
        if (length(m_mouse_delta) > 0.f) {
            quatf horiz = sgl::math::quat_from_angle_axis(-m_mouse_delta.x * m_rotate_speed, float3(0, 1, 0));
            quatf vert = sgl::math::quat_from_angle_axis(-m_mouse_delta.y * m_rotate_speed, float3(1, 0, 0));
            m_transform.set_rotation(mul(mul(horiz, m_transform.rotation()), vert));
            changed = true;
        }
    }

    // Keep pivot ahead of camera after movement/rotation.
    if (changed)
        update_pivot_from_camera();

    return changed;
}

bool CameraController::update_pan()
{
    bool changed = false;

    float3 pos = m_transform.translation();
    float3x3 rot = sgl::math::matrix_from_quat(m_transform.rotation());
    float3 right = rot.get_col(0);
    float3 up = rot.get_col(1);

    // Pan (scale by orbit distance for consistent feel at different zoom levels).
    if (length(m_mouse_delta) > 0.f) {
        float scale = TRACK_SPEED * m_orbit_distance;
        float3 offset = -right * m_mouse_delta.x * scale;
        offset += up * m_mouse_delta.y * scale;
        pos += offset;
        m_transform.set_translation(pos);
        // Also move pivot so orbit stays consistent.
        m_pivot += offset;
        changed = true;
    }

    return changed;
}

bool CameraController::apply_scroll_dolly(float log_amount)
{
    float3 pos = m_transform.translation();
    float3 to_pivot = m_pivot - pos;
    float dist = length(to_pivot);
    if (dist <= 0.f)
        return false;

    // Multiplicative zoom: new_dist = dist * exp(-log_amount).
    // Can never reach zero, and backing out is always proportionally fast.
    float3 dir = to_pivot / dist;
    float new_dist = dist * std::exp(-log_amount);
    new_dist = std::max(new_dist, MIN_ORBIT_DISTANCE);

    m_transform.set_translation(m_pivot - dir * new_dist);
    m_orbit_distance = new_dist;
    return true;
}

bool CameraController::apply_orbit_delta(float yaw_delta, float pitch_delta)
{
    float3 pos = m_transform.translation();
    float3 offset = pos - m_pivot;
    float dist = length(offset);
    if (dist < 1e-6f)
        return false;

    float3 dir = offset / dist;

    // Extract spherical angles from offset direction.
    // Using explicit spherical coordinates avoids the degenerate cross product
    // that occurs when the look-at forward vector approaches world up near the poles.
    float current_yaw = std::atan2(dir.x, dir.z);
    float current_pitch = std::asin(std::clamp(dir.y, -1.f, 1.f));

    // Apply angular delta and clamp pitch to avoid gimbal lock.
    float max_pitch = sgl::math::radians(89.f);
    float new_yaw = current_yaw + yaw_delta;
    float new_pitch = std::clamp(current_pitch + pitch_delta, -max_pitch, max_pitch);

    // Convert back to Cartesian offset.
    float cos_pitch = std::cos(new_pitch);
    float3 new_dir = float3(cos_pitch * std::sin(new_yaw), std::sin(new_pitch), cos_pitch * std::cos(new_yaw));

    float3 new_pos = m_pivot + new_dir * dist;
    m_transform.set_translation(new_pos);

    // Build orientation from spherical angles.
    // The right vector is always horizontal, derived directly from yaw.
    float3 fwd = normalize(m_pivot - new_pos);
    float3 new_right = float3(std::cos(new_yaw), 0.f, -std::sin(new_yaw));
    float3 new_up = cross(new_right, fwd);
    float3x3 rot_mat;
    rot_mat.set_col(0, new_right);
    rot_mat.set_col(1, new_up);
    rot_mat.set_col(2, -fwd);
    m_transform.set_rotation(sgl::math::quat_from_matrix(rot_mat));

    m_orbit_distance = dist;
    return true;
}

bool CameraController::apply_orbit(float yaw_speed, float pitch_speed, float dt)
{
    return apply_orbit_delta(yaw_speed * dt, pitch_speed * dt);
}

bool CameraController::update_orbit(float dt, bool apply_smoothing)
{
    if (apply_smoothing) {
        // Compute instantaneous angular velocity from mouse delta.
        float2 target_velocity{0.f};
        if (length(m_mouse_delta) > 0.f) {
            target_velocity.x = -m_mouse_delta.x * ORBIT_SPEED / dt;
            target_velocity.y = -m_mouse_delta.y * ORBIT_SPEED / dt;
        }
        // EMA blend velocity toward target for smooth tracking during drag.
        // When mouse stops, velocity smoothly decays toward zero so release feels natural.
        float blend = 1.f - std::exp(-dt / ORBIT_SMOOTH_TIME);
        m_orbit_velocity += (target_velocity - m_orbit_velocity) * blend;

        if (length(m_orbit_velocity) > ORBIT_STOP_THRESHOLD)
            return apply_orbit(m_orbit_velocity.x, m_orbit_velocity.y, dt);
        return false;
    } else {
        // No smoothing: apply raw mouse delta directly as angular displacement, no inertia.
        m_orbit_velocity = float2(0.f);
        if (length(m_mouse_delta) > 0.f) {
            float yaw_delta = -m_mouse_delta.x * ORBIT_SPEED;
            float pitch_delta = -m_mouse_delta.y * ORBIT_SPEED;
            return apply_orbit_delta(yaw_delta, pitch_delta);
        }
        return false;
    }
}

bool CameraController::update_dolly()
{
    bool changed = false;

    if (length(m_mouse_delta) > 0.f) {
        // Use horizontal + vertical mouse delta to dolly.
        float delta = (m_mouse_delta.x + m_mouse_delta.y) * DOLLY_SPEED;

        float3 pos = m_transform.translation();
        float3 to_pivot = m_pivot - pos;
        float dist = length(to_pivot);
        if (dist > 0.f) {
            float3 dir = to_pivot / dist;
            float new_dist = dist * (1.f - delta);
            new_dist = std::max(new_dist, MIN_ORBIT_DISTANCE);
            pos = m_pivot - dir * new_dist;
            m_transform.set_translation(pos);
            m_orbit_distance = new_dist;
            changed = true;
        }
    }

    return changed;
}

bool CameraController::update_track()
{
    bool changed = false;

    if (length(m_mouse_delta) > 0.f) {
        float3x3 rot = sgl::math::matrix_from_quat(m_transform.rotation());
        float3 right = rot.get_col(0);
        float3 up = rot.get_col(1);

        // Scale track speed by orbit distance for consistent feel at different zoom levels.
        float scale = TRACK_SPEED * m_orbit_distance;
        float3 offset = -right * m_mouse_delta.x * scale;
        offset += up * m_mouse_delta.y * scale;

        m_transform.set_translation(m_transform.translation() + offset);
        m_pivot += offset;
        changed = true;
    }

    return changed;
}

} // namespace falcor::ui
