// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "widgets.h"

#include "falcor2/render/transform.h"

#include <sgl/math/vector_math.h>

#include <imgui.h>
#include <imgui_internal.h>

namespace falcor::ui {

void draw_viewport_overlay(const ViewportOverlayDesc& desc)
{
    if (desc.screen_size.x <= 0.f || desc.screen_size.y <= 0.f)
        return;

    ImGui::SetNextWindowPos(desc.screen_pos + float2(4.f));
    ImGui::SetNextWindowBgAlpha(0.35f);

    ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoMove
        | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav
        | ImGuiWindowFlags_NoInputs | ImGuiWindowFlags_AlwaysAutoResize;

    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    if (ImGui::Begin("ViewportOverlay", nullptr, flags)) {
        ImGui::Text("FPS: %.1f", desc.fps);
    }
    ImGui::End();
    ImGui::PopStyleVar();
}

bool toggle_button(const char* label, bool active)
{
    if (active) {
        ImGui::PushStyleColor(ImGuiCol_Button, ImGui::GetStyleColorVec4(ImGuiCol_ButtonActive));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImGui::GetStyleColorVec4(ImGuiCol_ButtonActive));
    }
    bool clicked = ImGui::Button(label);
    if (active) {
        ImGui::PopStyleColor(2);
    }
    return clicked;
}

bool transform_editor(const char* label, Transform& transform)
{
    if (!ImGui::CollapsingHeader(label ? label : "Transform", ImGuiTreeNodeFlags_DefaultOpen))
        return false;

    ImGui::Indent();

    float3 translation = transform.translation();
    float3 scale = transform.scale();

    // Retrieve per-instance rotation-editing state from ImGui's internal storage.
    // IDs are scoped under a sub-level to avoid collisions with the DragFloat3 widgets.
    ImGuiStorage* storage = ImGui::GetStateStorage();
    ImGui::PushID("##transform_state");
    ImGuiID id_editing = ImGui::GetID("ed");
    ImGuiID id_rx = ImGui::GetID("rx");
    ImGuiID id_ry = ImGui::GetID("ry");
    ImGuiID id_rz = ImGui::GetID("rz");
    ImGui::PopID();

    bool editing_rotation = storage->GetBool(id_editing, false);
    float3 cached_rotation{
        storage->GetFloat(id_rx, 0.f),
        storage->GetFloat(id_ry, 0.f),
        storage->GetFloat(id_rz, 0.f),
    };

    // Use cached rotation while actively dragging to prevent wrapping.
    float3 rotation = editing_rotation ? cached_rotation : sgl::math::degrees(transform.rotation_xyz());

    bool changed = false;
    changed |= ImGui::DragFloat3("Translation", &translation.x, 0.1f);
    bool rotation_changed = ImGui::DragFloat3("Rotation", &rotation.x, 0.1f);
    changed |= rotation_changed;
    changed |= ImGui::DragFloat3("Scale", &scale.x, 0.1f);

    if (rotation_changed && !editing_rotation)
        editing_rotation = true;
    if (editing_rotation) {
        cached_rotation = rotation;
        if (!ImGui::GetIO().MouseDown[0])
            editing_rotation = false;
    }

    // Write state back.
    storage->SetBool(id_editing, editing_rotation);
    storage->SetFloat(id_rx, cached_rotation.x);
    storage->SetFloat(id_ry, cached_rotation.y);
    storage->SetFloat(id_rz, cached_rotation.z);

    if (changed) {
        transform.set_translation(translation);
        transform.set_rotation_xyz(sgl::math::radians(rotation));
        transform.set_scale(scale);
    }

    ImGui::Unindent();

    return changed;
}

} // namespace falcor::ui
