// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "falcor2/ui/scene_editor.h"
#include "falcor2/ui/icon_library.h"
#include "falcor2/ui/scene_editor_gizmos.h"
#include "falcor2/ui/scene_editor_outliner.h"
#include "falcor2/ui/scene_picker.h"
#include "falcor2/ui/camera_controller.h"

#include "falcor2/render/scene.h"
#include "falcor2/render/component/geometry_instance.h"
#include "falcor2/render/component/camera.h"
#include "falcor2/render/component/light.h"
#include "falcor2/ui/widgets.h"
#include "falcor2/core/reflection/type_registry.h"

#include <imgui.h>
#include <imgui_internal.h>
#include <imguizmo/ImGuizmo.h>

#include <sgl/device/resource.h>

#include <cmath>
#include <cstdio>
#include <vector>

namespace falcor::ui {
namespace {

namespace command_ids {
const CommandId TOOL_SELECT{"scene_editor.tool.select"};
const CommandId TOOL_MOVE{"scene_editor.tool.move"};
const CommandId TOOL_ROTATE{"scene_editor.tool.rotate"};
const CommandId TOOL_SCALE{"scene_editor.tool.scale"};
const CommandId TOGGLE_TRANSFORM_SPACE{"scene_editor.transform.toggle_space"};
const CommandId DELETE_SELECTED{"scene_editor.edit.delete_selected"};
const CommandId CREATE_ENTITY{"scene_editor.create.entity"};
const CommandId CREATE_GEOMETRY{"scene_editor.create.geometry"};
const CommandId CREATE_MATERIAL{"scene_editor.create.material"};
const CommandId FRAME_SELECTED{"scene_editor.view.frame_selected"};
const CommandId TOGGLE_PLAYBACK{"scene_editor.animation.toggle_playback"};
const CommandId RESET_PLAYBACK{"scene_editor.animation.reset_playback"};
const CommandId TOGGLE_HELP{"scene_editor.help.toggle"};
const CommandId TOGGLE_CAMERA_GIZMOS{"scene_editor.viewport.toggle_camera_gizmos"};
const CommandId TOGGLE_LIGHT_GIZMOS{"scene_editor.viewport.toggle_light_gizmos"};
const CommandId TOGGLE_OUTLINER{"scene_editor.window.toggle_outliner"};
const CommandId TOGGLE_GRAPH{"scene_editor.window.toggle_graph"};
const CommandId TOGGLE_INSPECTOR{"scene_editor.window.toggle_inspector"};
const CommandId TOGGLE_TRANSPORT{"scene_editor.window.toggle_transport"};
const CommandId RESET_LAYOUT{"scene_editor.window.reset_layout"};
} // namespace command_ids

CommandShortcut key_shortcut(sgl::KeyCode key, const char* display_name)
{
    return {key, sgl::KeyModifierFlags::none, display_name};
}

ImRect main_viewport_work_rect()
{
    return static_cast<ImGuiViewportP*>(ImGui::GetMainViewport())->GetBuildWorkRect();
}

bool tool_icon_button(const IconLibrary* icons, const EditorCommand& command, Icon icon, bool active)
{
    constexpr ImVec2 BUTTON_SIZE{32.f, 32.f};
    if (active) {
        ImGui::PushStyleColor(ImGuiCol_Button, ImGui::GetStyleColorVec4(ImGuiCol_ButtonActive));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImGui::GetStyleColorVec4(ImGuiCol_ButtonActive));
    }
    ImGui::PushID(command.id.c_str());
    const bool clicked = ImGui::Button("##Command", BUTTON_SIZE);
    ImGui::PopID();
    if (active)
        ImGui::PopStyleColor(2);

    const ImVec2 item_min = ImGui::GetItemRectMin();
    const ImVec2 item_max = ImGui::GetItemRectMax();
    const ImVec2 center(0.5f * (item_min.x + item_max.x), 0.5f * (item_min.y + item_max.y));
    if (icons) {
        const IconImage image = icons->icon(icon);
        const float half_size = 0.5f * image.preferred_size;
        ImGui::GetWindowDrawList()->AddImage(
            image.texture,
            ImVec2(center.x - half_size, center.y - half_size),
            ImVec2(center.x + half_size, center.y + half_size),
            ImVec2(image.uv_min.x, image.uv_min.y),
            ImVec2(image.uv_max.x, image.uv_max.y),
            ImGui::GetColorU32(ImGuiCol_Text)
        );
    }

    if (ImGui::IsItemHovered()) {
        ImGui::BeginTooltip();
        ImGui::TextUnformatted(command.label.c_str());
        if (command.shortcut) {
            ImGui::SameLine(0.f, 0.f);
            ImGui::Text(" (%s)", command.shortcut->display_name.c_str());
        }
        ImGui::EndTooltip();
    }
    return clicked;
}

template<typename DrawNode>
bool draw_forced_open_tree_node(ImGuiID id, bool force_open, bool persist_open, DrawNode draw_node)
{
    if (!force_open && !persist_open)
        return draw_node();

    ImGuiStorage* storage = ImGui::GetStateStorage();
    detail::OutlinerTreeOpenState open_state
        = detail::outliner_tree_open_state(storage->GetBool(id, false), force_open, persist_open);
    ImGui::SetNextItemOpen(open_state.draw_open, ImGuiCond_Always);
    bool is_open = draw_node();
    storage->SetBool(id, open_state.stored_open);
    return is_open;
}

bool draw_forced_open_tree_node(const char* label, bool force_open, bool persist_open = false)
{
    return draw_forced_open_tree_node(
        ImGui::GetID(label),
        force_open,
        persist_open,
        [label]()
        {
            return ImGui::TreeNode(label);
        }
    );
}

bool draw_forced_open_entity_node(Entity* entity, int flags, bool force_open, bool persist_open)
{
    return draw_forced_open_tree_node(
        ImGui::GetID(entity),
        force_open,
        persist_open,
        [entity, flags]()
        {
            return ImGui::TreeNodeEx(entity, flags, "%s", entity->name().c_str());
        }
    );
}

} // namespace

SceneEditor::SceneEditor()
{
    register_commands();
}

void SceneEditor::set_scene(ref<Scene> scene)
{
    if (scene != m_scene) {
        m_scene = std::move(scene);
        if (!m_scene) {
            m_scene_gizmo_renderer = nullptr;
            m_icon_library = nullptr;
        } else if (!m_icon_library || m_icon_library->device() != m_scene->device()) {
            m_icon_library = make_ref<IconLibrary>(ref<sgl::Device>(m_scene->device()));
            m_scene_gizmo_renderer = make_ref<detail::SceneGizmoRenderer>(m_icon_library);
        }
        set_selected_object(nullptr);
    }
}

void SceneEditor::set_camera_controller(ref<CameraController> controller)
{
    m_camera_controller = std::move(controller);
}

void SceneEditor::set_selected_object(SceneObject* object)
{
    FALCOR_CHECK(
        object == nullptr || object->scene() == m_scene.get(),
        "Selected object must be part of the current scene."
    );
    m_inspector_return_entity = {};
    m_reveal_selection_in_outliner = false;
    if (object != m_selected_object) {
        m_selected_object = object;
        ++m_selection_version;
    }
}

void SceneEditor::remove_selected_object()
{
    if (m_selected_object) {
        m_selected_object->remove();
        set_selected_object(nullptr);
    }
}

bool SceneEditor::frame_selected(Camera* camera)
{
    if (!can_frame_selected())
        return false;

    Entity* entity = m_selected_object->as<Entity>();
    AABB aabb = entity->world_aabb();
    float distance = sgl::math::length(aabb.size()) * 1.5f;
    m_camera_controller->focus(aabb.center(), distance);
    if (camera && camera->entity())
        camera->entity()->set_world_transform(m_camera_controller->transform());
    return true;
}

bool SceneEditor::can_frame_selected() const
{
    if (!m_camera_controller || !m_selected_object)
        return false;

    Entity* entity = m_selected_object->as<Entity>();
    return entity && entity->world_aabb().is_valid();
}

void SceneEditor::set_tool_mode(ToolMode mode)
{
    m_tool_mode = mode;
}

void SceneEditor::set_transform_space(TransformSpace space)
{
    m_transform_space = space;
}

void SceneEditor::set_active_camera(const Camera* camera)
{
    if (!camera) {
        set_camera_matrices(float4x4::identity(), float4x4::identity());
        return;
    }

    set_camera_matrices(camera->calc_view_from_world(), camera->calc_clip_from_view());
}

void SceneEditor::set_camera_matrices(const float4x4& view, const float4x4& proj)
{
    m_view_matrix = view;
    m_proj_matrix = proj;

    // Build a scale-free view matrix for ImGuizmo.
    // If the camera sits under a scaled parent, the view matrix
    // contains inverse scale that would distort the gizmo.
    float4x4 camera_world = inverse(view);
    {
        float3 scale, translation, skew;
        quatf rotation;
        float4 perspective;
        if (sgl::math::decompose(camera_world, scale, rotation, translation, skew, perspective)) {
            camera_world
                = mul(sgl::math::matrix_from_translation(translation), float4x4(sgl::math::matrix_from_quat(rotation)));
        }
    }
    m_view_matrix_no_scale = inverse(camera_world);
}

void SceneEditor::update_playback(double dt)
{
    if (!m_playing || !m_scene || !m_scene->has_animation())
        return;

    double duration = m_scene->animation_duration();
    if (duration <= 0.0)
        return;

    double time = m_scene->time() + dt;
    if (time > duration) {
        if (m_looping)
            time = std::fmod(time, duration);
        else {
            time = duration;
            m_playing = false;
        }
    }
    m_scene->set_time(time);
}

bool SceneEditor::handle_keyboard_shortcut(const sgl::KeyboardEvent& event)
{
    return m_commands.dispatch_shortcut(event);
}

void SceneEditor::begin_frame()
{
    m_gizmo_frame_initialized = ImGui::GetFrameCount();
    ImGuizmo::SetImGuiContext(ImGui::GetCurrentContext());
    ImGuizmo::BeginFrame();
    ImGuizmo::SetOrthographic(false);
}

void SceneEditor::editor_ui()
{
    main_menu_ui();
    setup_dockspace();

    help_ui();

    tool_rail_ui();
    graph_ui();
    outliner_ui();
    inspector_ui();
    transport_ui();
}

void SceneEditor::viewport_ui(sgl::Texture* output_texture, float fps)
{
    if (m_gizmo_frame_initialized != ImGui::GetFrameCount())
        begin_frame();

    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
    if (ImGui::Begin("Viewport", nullptr, ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoScrollbar)) {
        viewport_header_ui();

        ImVec2 pos = ImGui::GetCursorScreenPos();
        ImVec2 size = ImGui::GetContentRegionAvail();
        m_viewport_state.pos = float2(pos.x, pos.y);
        m_viewport_state.size = float2(size.x, size.y);
        m_viewport_state.hovered
            = ImGui::IsWindowHovered() && ImGui::IsMouseHoveringRect(pos, ImVec2(pos.x + size.x, pos.y + size.y));
        m_viewport_state.focused = ImGui::IsWindowFocused();
        m_viewport_state.valid = size.x > 0.f && size.y > 0.f;
        if (output_texture && size.x > 0 && size.y > 0) {
            ImGui::Image(output_texture, size);
        }

        // ImGuizmo: render gizmo inside the viewport window.
        ImGuizmo::SetDrawlist();
        ImGuizmo::SetRect(pos.x, pos.y, size.x, size.y);

        // Render editor-only scene gizmos.
        if (m_scene_gizmo_renderer && (m_show_camera_gizmos || m_show_light_gizmos)) {
            m_scene_gizmo_renderer->draw(
                m_scene.get(),
                m_selected_object,
                m_view_matrix,
                m_proj_matrix,
                m_viewport_state.pos,
                m_viewport_state.size,
                m_show_camera_gizmos,
                m_show_light_gizmos
            );
        }
        viewport_gizmo();
    } else {
        m_viewport_state = {};
    }
    ImGui::End();
    ImGui::PopStyleVar();

    draw_viewport_overlay({
        .screen_pos = m_viewport_state.pos,
        .screen_size = m_viewport_state.size,
        .fps = fps,
    });
}

bool SceneEditor::is_gizmo_active() const
{
    if (m_tool_mode == ToolMode::select)
        return false;

    Entity* entity = m_selected_object ? m_selected_object->as<Entity>() : nullptr;
    if (!entity)
        return false;

    return ImGuizmo::IsOver() || ImGuizmo::IsUsing();
}

bool SceneEditor::is_viewport_interactive() const
{
    return m_viewport_state.valid && m_viewport_state.hovered;
}

bool SceneEditor::can_pick_at(float2 screen_pos, uint2& local_pos) const
{
    if (!m_viewport_state.valid)
        return false;
    if (is_gizmo_active())
        return false;
    float local_x = screen_pos.x - m_viewport_state.pos.x;
    float local_y = screen_pos.y - m_viewport_state.pos.y;
    if (local_x < 0 || local_y < 0 || local_x >= m_viewport_state.size.x || local_y >= m_viewport_state.size.y)
        return false;
    local_pos = uint2(uint32_t(local_x), uint32_t(local_y));
    return true;
}

Entity* SceneEditor::pick_gizmo_at(float2 screen_pos) const
{
    if (!m_scene || !m_viewport_state.valid)
        return nullptr;

    FALCOR_ASSERT(m_scene_gizmo_renderer);

    return m_scene_gizmo_renderer->pick(
        m_scene.get(),
        m_view_matrix,
        m_proj_matrix,
        m_viewport_state.pos,
        m_viewport_state.size,
        screen_pos,
        m_show_camera_gizmos,
        m_show_light_gizmos
    );
}

void SceneEditor::setup_dockspace()
{
    ImGuiViewport* vp = ImGui::GetMainViewport();
    const ImRect work_rect = main_viewport_work_rect();
    ImGui::SetNextWindowPos(work_rect.Min);
    ImGui::SetNextWindowSize(work_rect.GetSize());
    ImGui::SetNextWindowViewport(vp->ID);

    ImGuiWindowFlags flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize
        | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus;

    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
    ImGui::Begin("##DockSpaceWindow", nullptr, flags);
    ImGui::PopStyleVar(3);

    ImGuiID dockspace_id = ImGui::GetID("MainDockSpace");

    // Build default layout before the first DockSpace() call.
    if (!m_layout_initialized) {
        m_layout_initialized = true;
        setup_default_layout(dockspace_id);
    }

    ImGui::DockSpace(dockspace_id, ImVec2(0.0f, 0.0f), ImGuiDockNodeFlags_NoDockingOverCentralNode);

    ImGui::End();
}

void SceneEditor::setup_default_layout(uint32_t dockspace_id)
{
    ImGui::DockBuilderRemoveNode(dockspace_id);
    ImGui::DockBuilderAddNode(dockspace_id, ImGuiDockNodeFlags_DockSpace);
    const ImRect work_rect = main_viewport_work_rect();
    ImGui::DockBuilderSetNodeSize(dockspace_id, work_rect.GetSize());

    ImGuiID rest = dockspace_id;

    // Split off the right panel first so it spans the full dockspace height.
    ImGuiID right;
    ImGui::DockBuilderSplitNode(rest, ImGuiDir_Right, RIGHT_PANEL_SIZE, &right, &rest);

    // Split the transport from the viewport area only.
    ImGuiID bottom;
    ImGui::DockBuilderSplitNode(rest, ImGuiDir_Down, 0.15f, &bottom, &rest);

    // Split the viewport tool rail from the left of the center area.
    ImGuiID tool_rail;
    const float center_width = work_rect.GetWidth() * (1.f - RIGHT_PANEL_SIZE);
    ImGui::DockBuilderSplitNode(rest, ImGuiDir_Left, TOOL_RAIL_WIDTH / center_width, &tool_rail, &rest);
    ImGuiID viewport = rest;

    // Split right: right_top (outliner) | right_bottom (inspector)
    ImGuiID right_top, right_bottom;
    ImGui::DockBuilderSplitNode(right, ImGuiDir_Up, 0.50f, &right_top, &right_bottom);

    // Mark the viewport as central node with no tab bar.
    ImGuiDockNode* viewport_node = ImGui::DockBuilderGetNode(viewport);
    if (viewport_node)
        viewport_node->LocalFlags |= ImGuiDockNodeFlags_CentralNode | ImGuiDockNodeFlags_NoTabBar;
    ImGuiDockNode* tool_rail_node = ImGui::DockBuilderGetNode(tool_rail);
    if (tool_rail_node) {
        tool_rail_node->LocalFlags |= ImGuiDockNodeFlags_NoResize;
        tool_rail_node->LocalFlags |= ImGuiDockNodeFlags_NoTabBar;
    }
    ImGuiDockNode* right_top_node = ImGui::DockBuilderGetNode(right_top);
    if (right_top_node)
        right_top_node->SelectedTabId = ImHashStr("Outliner");

    // Dock windows into slots.
    ImGui::DockBuilderDockWindow("##ToolRail", tool_rail);
    ImGui::DockBuilderDockWindow("Viewport", viewport);
    ImGui::DockBuilderDockWindow("Outliner", right_top);
    ImGui::DockBuilderDockWindow("Graph", right_top);
    ImGui::DockBuilderDockWindow("Inspector", right_bottom);
    ImGui::DockBuilderDockWindow("Transport", bottom);

    ImGui::DockBuilderFinish(dockspace_id);
}

void SceneEditor::register_commands()
{
    const auto scene_loaded = [this]
    {
        return m_scene != nullptr;
    };

    const auto add_tool_command
        = [this, scene_loaded](const CommandId& id, std::string label, CommandShortcut shortcut, ToolMode mode)
    {
        m_commands.register_command({
            .id = id,
            .label = std::move(label),
            .shortcut = std::move(shortcut),
            .execute =
                [this, mode]
            {
                set_tool_mode(mode);
            },
            .can_execute = scene_loaded,
            .is_checked =
                [this, mode]
            {
                return m_tool_mode == mode;
            },
        });
    };

    const auto add_panel_command = [this](const CommandId& id, std::string label, bool SceneEditor::* visibility_member)
    {
        m_commands.register_command({
            .id = id,
            .label = std::move(label),
            .execute =
                [this, visibility_member]
            {
                this->*visibility_member = !(this->*visibility_member);
            },
            .is_checked =
                [this, visibility_member]
            {
                return this->*visibility_member;
            },
        });
    };

    const auto add_create_popup_command
        = [this, scene_loaded](const CommandId& id, std::string label, CreatePopup popup)
    {
        m_commands.register_command({
            .id = id,
            .label = std::move(label),
            .execute =
                [this, popup]
            {
                m_show_outliner = true;
                m_pending_create_popup = popup;
            },
            .can_execute = scene_loaded,
        });
    };

    add_tool_command(command_ids::TOOL_SELECT, "Select mode", key_shortcut(sgl::KeyCode::q, "Q"), ToolMode::select);
    add_tool_command(command_ids::TOOL_MOVE, "Move mode", key_shortcut(sgl::KeyCode::w, "W"), ToolMode::move);
    add_tool_command(command_ids::TOOL_ROTATE, "Rotate mode", key_shortcut(sgl::KeyCode::e, "E"), ToolMode::rotate);
    add_tool_command(command_ids::TOOL_SCALE, "Scale mode", key_shortcut(sgl::KeyCode::r, "R"), ToolMode::scale);

    m_commands.register_command({
        .id = command_ids::TOGGLE_TRANSFORM_SPACE,
        .label = "Toggle world/local transform space",
        .shortcut = key_shortcut(sgl::KeyCode::x, "X"),
        .execute =
            [this]
        {
            set_transform_space(
                transform_space() == TransformSpace::local ? TransformSpace::world : TransformSpace::local
            );
        },
        .can_execute = scene_loaded,
        .is_checked =
            [this]
        {
            return m_transform_space == TransformSpace::world;
        },
    });
    m_commands.register_command({
        .id = command_ids::DELETE_SELECTED,
        .label = "Remove selected object",
        .shortcut = key_shortcut(sgl::KeyCode::delete_, "Delete"),
        .execute =
            [this]
        {
            remove_selected_object();
        },
        .can_execute =
            [this]
        {
            return m_scene && m_selected_object;
        },
    });
    m_commands.register_command({
        .id = command_ids::CREATE_ENTITY,
        .label = "Entity",
        .execute =
            [this]
        {
            m_scene->create_entity();
        },
        .can_execute = scene_loaded,
    });
    add_create_popup_command(command_ids::CREATE_GEOMETRY, "Geometry...", CreatePopup::geometry);
    add_create_popup_command(command_ids::CREATE_MATERIAL, "Material...", CreatePopup::material);
    m_commands.register_command({
        .id = command_ids::FRAME_SELECTED,
        .label = "Frame Selected",
        .shortcut = key_shortcut(sgl::KeyCode::f, "F"),
        .execute =
            [this]
        {
            FALCOR_ASSERT(m_scene && m_scene->active_camera());
            frame_selected(m_scene->active_camera());
        },
        .can_execute =
            [this]
        {
            return m_scene && m_scene->active_camera() && can_frame_selected();
        },
    });
    m_commands.register_command({
        .id = command_ids::TOGGLE_PLAYBACK,
        .label = "Toggle playback",
        .shortcut = key_shortcut(sgl::KeyCode::space, "Space"),
        .execute =
            [this]
        {
            set_playing(!playing());
        },
        .can_execute =
            [this]
        {
            return m_scene && m_scene->has_animation();
        },
        .is_checked =
            [this]
        {
            return m_playing;
        },
    });
    m_commands.register_command({
        .id = command_ids::RESET_PLAYBACK,
        .label = "Reset playback",
        .execute =
            [this]
        {
            FALCOR_ASSERT(m_scene && m_scene->has_animation());
            set_playing(false);
            m_scene->set_time(0.0);
        },
        .can_execute =
            [this]
        {
            return m_scene && m_scene->has_animation();
        },
    });
    m_commands.register_command({
        .id = command_ids::TOGGLE_HELP,
        .label = "Toggle controls and shortcuts",
        .shortcut = key_shortcut(sgl::KeyCode::f1, "F1"),
        .execute =
            [this]
        {
            m_show_help = !m_show_help;
        },
        .is_checked =
            [this]
        {
            return m_show_help;
        },
    });
    m_commands.register_command({
        .id = command_ids::TOGGLE_CAMERA_GIZMOS,
        .label = "Camera gizmos",
        .execute =
            [this]
        {
            m_show_camera_gizmos = !m_show_camera_gizmos;
        },
        .can_execute = scene_loaded,
        .is_checked =
            [this]
        {
            return m_show_camera_gizmos;
        },
    });
    m_commands.register_command({
        .id = command_ids::TOGGLE_LIGHT_GIZMOS,
        .label = "Light gizmos",
        .execute =
            [this]
        {
            m_show_light_gizmos = !m_show_light_gizmos;
        },
        .can_execute = scene_loaded,
        .is_checked =
            [this]
        {
            return m_show_light_gizmos;
        },
    });

    add_panel_command(command_ids::TOGGLE_OUTLINER, "Outliner", &SceneEditor::m_show_outliner);
    add_panel_command(command_ids::TOGGLE_GRAPH, "Graph", &SceneEditor::m_show_graph);
    add_panel_command(command_ids::TOGGLE_INSPECTOR, "Inspector", &SceneEditor::m_show_inspector);
    add_panel_command(command_ids::TOGGLE_TRANSPORT, "Transport", &SceneEditor::m_show_transport);
    m_commands.register_command({
        .id = command_ids::RESET_LAYOUT,
        .label = "Reset Layout",
        .execute = [this]
        {
            m_show_outliner = true;
            m_show_graph = true;
            m_show_inspector = true;
            m_show_transport = true;
            m_layout_initialized = false;
        },
    });
}

void SceneEditor::main_menu_ui()
{
    if (ImGui::BeginMainMenuBar()) {
        constexpr const char* PLACEHOLDER_MENUS[]{"File"};
        for (const char* label : PLACEHOLDER_MENUS) {
            if (ImGui::BeginMenu(label)) {
                ImGui::MenuItem("Coming soon", nullptr, false, false);
                ImGui::EndMenu();
            }
        }
        if (ImGui::BeginMenu("Edit")) {
            command_menu_item(command_ids::DELETE_SELECTED);
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Create")) {
            command_menu_item(command_ids::CREATE_ENTITY);
            command_menu_item(command_ids::CREATE_GEOMETRY);
            command_menu_item(command_ids::CREATE_MATERIAL);
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("View")) {
            command_menu_item(command_ids::FRAME_SELECTED);
            ImGui::Separator();
            command_menu_item(command_ids::TOGGLE_CAMERA_GIZMOS);
            command_menu_item(command_ids::TOGGLE_LIGHT_GIZMOS);
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Window")) {
            command_menu_item(command_ids::TOGGLE_OUTLINER);
            command_menu_item(command_ids::TOGGLE_GRAPH);
            command_menu_item(command_ids::TOGGLE_INSPECTOR);
            command_menu_item(command_ids::TOGGLE_TRANSPORT);
            ImGui::Separator();
            command_menu_item(command_ids::RESET_LAYOUT);
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Help")) {
            command_menu_item(command_ids::TOGGLE_HELP);
            ImGui::EndMenu();
        }
        ImGui::EndMainMenuBar();
    }
}

bool SceneEditor::command_menu_item(const CommandId& id)
{
    const EditorCommand* command = m_commands.find(id);
    FALCOR_ASSERT(command);
    const char* shortcut = command->shortcut ? command->shortcut->display_name.c_str() : nullptr;
    if (ImGui::MenuItem(command->label.c_str(), shortcut, m_commands.is_checked(id), m_commands.can_execute(id))) {
        return m_commands.execute(id);
    }
    return false;
}

void SceneEditor::tool_rail_ui()
{
    ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize
        | ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus;

    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(8.f, 8.f));
    if (ImGui::Begin("##ToolRail", nullptr, flags)) {
        // Leave an empty cap aligned with the viewport header so the tools line up
        // with the viewport image.
        ImGui::SetCursorPosY(VIEWPORT_HEADER_HEIGHT + ImGui::GetStyle().WindowPadding.y);

        auto command_button = [this](const CommandId& id, Icon icon)
        {
            const EditorCommand* command = m_commands.find(id);
            FALCOR_ASSERT(command);
            ImGui::BeginDisabled(!m_commands.can_execute(id));
            const bool clicked = tool_icon_button(m_icon_library.get(), *command, icon, m_commands.is_checked(id));
            ImGui::EndDisabled();
            if (clicked)
                m_commands.execute(id);
        };

        command_button(command_ids::TOOL_SELECT, Icon::select);
        command_button(command_ids::TOOL_MOVE, Icon::move);
        command_button(command_ids::TOOL_ROTATE, Icon::rotate);
        command_button(command_ids::TOOL_SCALE, Icon::scale);
    }
    ImGui::End();
    ImGui::PopStyleVar(2);
}

void SceneEditor::viewport_header_ui()
{
    const ImVec2 header_pos = ImGui::GetCursorScreenPos();
    ImGui::PushStyleColor(ImGuiCol_ChildBg, ImGui::GetStyleColorVec4(ImGuiCol_MenuBarBg));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(8.f, 0.f));
    const ImGuiChildFlags child_flags = ImGuiChildFlags_AlwaysUseWindowPadding;
    const ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse;
    if (ImGui::BeginChild("##ViewportHeader", ImVec2(0.f, VIEWPORT_HEADER_HEIGHT), child_flags, window_flags)) {
        ImGui::SetCursorPosY(0.5f * (VIEWPORT_HEADER_HEIGHT - ImGui::GetFrameHeight()));

        active_camera_ui();

        ImGui::SameLine();
        ImGui::SeparatorEx(ImGuiSeparatorFlags_Vertical);
        ImGui::SameLine();
        const char* space_label = (m_transform_space == TransformSpace::local) ? "Local (X)" : "World (X)";
        ImGui::BeginDisabled(!m_commands.can_execute(command_ids::TOGGLE_TRANSFORM_SPACE));
        if (toggle_button(space_label, false))
            m_commands.execute(command_ids::TOGGLE_TRANSFORM_SPACE);
        ImGui::EndDisabled();

        if (m_camera_controller) {
            ImGui::SameLine();
            ImGui::SeparatorEx(ImGuiSeparatorFlags_Vertical);
            ImGui::SameLine();

            char speed_label[32];
            std::snprintf(speed_label, sizeof(speed_label), "Speed %.2f", m_camera_controller->move_speed());
            if (ImGui::Button(speed_label))
                ImGui::OpenPopup("Camera Speed");
            if (ImGui::BeginPopup("Camera Speed")) {
                ImGui::SetNextItemWidth(180.f);
                float speed = m_camera_controller->move_speed();
                if (ImGui::SliderFloat(
                        "##CameraSpeed",
                        &speed,
                        CameraController::MIN_MOVE_SPEED,
                        CameraController::MAX_MOVE_SPEED,
                        "%.2f",
                        ImGuiSliderFlags_Logarithmic
                    ))
                    m_camera_controller->set_move_speed(speed);
                ImGui::EndPopup();
            }
        }

        ImGui::SameLine();
        ImGui::SeparatorEx(ImGuiSeparatorFlags_Vertical);
        ImGui::SameLine();
        viewport_display_menu_ui();
    }
    ImGui::EndChild();
    ImGui::PopStyleVar();
    ImGui::PopStyleColor();

    // BeginChild() includes item spacing after the child. Position the viewport image
    // immediately below the reserved header instead.
    ImGui::SetCursorScreenPos(ImVec2(header_pos.x, header_pos.y + VIEWPORT_HEADER_HEIGHT));
}

void SceneEditor::active_camera_ui()
{
    Camera* active_camera = m_scene ? m_scene->active_camera() : nullptr;
    const char* active_name
        = active_camera && active_camera->entity() ? active_camera->entity()->name().c_str() : "No camera";

    ImGui::AlignTextToFramePadding();
    ImGui::TextUnformatted("Camera");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(160.f);
    if (ImGui::BeginCombo("##ActiveCamera", active_name)) {
        if (m_scene) {
            for (Component* component : m_scene->components()) {
                Camera* camera = component->as<Camera>();
                if (!camera || !camera->is_valid() || !camera->entity() || !camera->entity()->is_valid())
                    continue;

                const bool selected = camera == active_camera;
                ImGui::PushID(camera);
                if (ImGui::Selectable(camera->entity()->name().c_str(), selected) && !selected)
                    m_scene->set_active_camera(camera);
                if (selected)
                    ImGui::SetItemDefaultFocus();
                ImGui::PopID();
            }
        }
        ImGui::EndCombo();
    }

    ImGui::SameLine();
    ImGui::BeginDisabled(!active_camera || !active_camera->entity());
    if (ImGui::Button("Select##ActiveCamera"))
        select_and_reveal(active_camera->entity());
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
        ImGui::SetTooltip("Select and reveal the active camera entity");
    ImGui::EndDisabled();
}

void SceneEditor::viewport_display_menu_ui()
{
    constexpr const char* POPUP_LABEL = "Viewport Display";
    if (ImGui::Button("Display"))
        ImGui::OpenPopup(POPUP_LABEL);

    if (ImGui::BeginPopup(POPUP_LABEL)) {
        ImGui::SeparatorText("Gizmos");
        command_menu_item(command_ids::TOGGLE_CAMERA_GIZMOS);
        command_menu_item(command_ids::TOGGLE_LIGHT_GIZMOS);
        ImGui::EndPopup();
    }
}

void SceneEditor::graph_ui()
{
    if (!m_show_graph)
        return;

    if (ImGui::Begin("Graph", &m_show_graph) && m_graph_ui_callback)
        m_graph_ui_callback();
    ImGui::End();
}

void SceneEditor::outliner_ui()
{
    if (!m_show_outliner)
        return;

    if (m_pending_create_popup != CreatePopup::none)
        ImGui::SetNextWindowFocus();

    SceneObject* clicked_object = nullptr;
    if (ImGui::Begin("Outliner", &m_show_outliner)) {
        if (!m_scene) {
            ImGui::TextDisabled("No scene loaded");
            ImGui::End();
            return;
        }

        if (ImGui::Button("Add Geometry"))
            m_commands.execute(command_ids::CREATE_GEOMETRY);
        if (m_pending_create_popup == CreatePopup::geometry) {
            ImGui::OpenPopup("Add Geometry");
            m_pending_create_popup = CreatePopup::none;
        }
        add_scene_object_popup<Geometry>("Add Geometry");

        ImGui::SameLine();
        if (ImGui::Button("Add Material"))
            m_commands.execute(command_ids::CREATE_MATERIAL);
        if (m_pending_create_popup == CreatePopup::material) {
            ImGui::OpenPopup("Add Material");
            m_pending_create_popup = CreatePopup::none;
        }
        add_scene_object_popup<Material>("Add Material");

        ImGui::SameLine();
        if (ImGui::Button("Add Entity"))
            m_commands.execute(command_ids::CREATE_ENTITY);

        ImGui::Checkbox("Sort A-Z", &m_sort_outliner);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(-1.f);
        ImGui::InputTextWithHint(
            "##outliner_search",
            "Search names or t:component",
            m_outliner_search.data(),
            m_outliner_search.size()
        );

        detail::OutlinerQuery query(m_outliner_search.data());
        Geometry* reveal_geometry
            = m_reveal_selection_in_outliner && m_selected_object ? m_selected_object->as<Geometry>() : nullptr;
        Material* reveal_material
            = m_reveal_selection_in_outliner && m_selected_object ? m_selected_object->as<Material>() : nullptr;
        Entity* reveal_entity
            = m_reveal_selection_in_outliner && m_selected_object ? m_selected_object->as<Entity>() : nullptr;

        ImGui::BeginChild("##scrollregion");
        if (draw_forced_open_tree_node("Geometries", !query.empty(), reveal_geometry != nullptr)) {
            auto& geometry_collection = m_scene->geometries();
            std::vector<Geometry*> geometries;
            geometries.reserve(geometry_collection.size());
            for (Geometry* geometry : geometry_collection)
                if (query.matches_object(geometry))
                    geometries.push_back(geometry);
            if (m_sort_outliner)
                detail::sort_outliner_objects(geometries);
            for (Geometry* geometry : geometries)
                scene_object_node(geometry, m_selected_object, clicked_object);
            ImGui::TreePop();
        }
        if (draw_forced_open_tree_node("Materials", !query.empty(), reveal_material != nullptr)) {
            auto& material_collection = m_scene->materials();
            std::vector<Material*> materials;
            materials.reserve(material_collection.size());
            for (Material* material : material_collection)
                if (query.matches_object(material))
                    materials.push_back(material);
            if (m_sort_outliner)
                detail::sort_outliner_objects(materials);
            for (Material* material : materials)
                scene_object_node(material, m_selected_object, clicked_object);
            ImGui::TreePop();
        }
        if (draw_forced_open_tree_node("Entities", !query.empty(), reveal_entity != nullptr)) {
            auto& entities = m_scene->entities();
            std::vector<Entity*> root_entities;
            root_entities.reserve(entities.size());
            for (size_t i = 0; i < entities.size(); ++i) {
                Entity* entity = entities[i];
                if (entity->parent() == nullptr)
                    root_entities.push_back(entity);
            }

            std::unordered_set<const Entity*> visible_entities;
            const std::unordered_set<const Entity*>* visible_entities_ptr = nullptr;
            if (!query.empty()) {
                visible_entities.reserve(entities.size());
                for (const Entity* entity : root_entities)
                    query.collect_visible_entities(entity, visible_entities);
                visible_entities_ptr = &visible_entities;
            }

            if (m_sort_outliner)
                detail::sort_outliner_objects(root_entities);
            for (Entity* entity : root_entities)
                entity_node(entity, m_selected_object, clicked_object, visible_entities_ptr, reveal_entity);
            ImGui::TreePop();
        }

        ImGui::EndChild();
    }
    ImGui::End();
    if (clicked_object) {
        set_selected_object(clicked_object);
    }
}

void SceneEditor::inspector_ui()
{
    if (!m_show_inspector)
        return;

    if (ImGui::Begin("Inspector", &m_show_inspector)) {
        if (!m_scene) {
            ImGui::TextDisabled("No scene loaded");
        } else if (m_selected_object) {
            if (!inspector_navigation_ui()) {
                SceneObject* selected_object = m_selected_object;
                scene_object_ui<Geometry>(selected_object, &SceneEditor::geometry_ui);
                scene_object_ui<Material>(selected_object, &SceneEditor::material_ui);
                scene_object_ui<Entity>(selected_object, &SceneEditor::entity_ui);
            }
        } else {
            ImGui::TextDisabled("No object selected");
        }
    }
    ImGui::End();
}

void SceneEditor::transport_ui()
{
    if (!m_show_transport)
        return;

    if (ImGui::Begin("Transport", &m_show_transport)) {
        if (!m_scene) {
            ImGui::TextDisabled("No scene loaded");
        } else if (m_scene->has_animation()) {
            // Play/pause toggle.
            if (ImGui::Button(playing() ? "Pause" : "Play"))
                m_commands.execute(command_ids::TOGGLE_PLAYBACK);
            ImGui::SameLine();
            if (ImGui::Button("Reset"))
                m_commands.execute(command_ids::RESET_PLAYBACK);
            ImGui::SameLine();
            ImGui::Checkbox("Loop", &m_looping);

            // Time slider.
            ImGui::SetNextItemWidth(-1.f);
            double time = m_scene->time();
            double min_time = 0.0;
            double max_time = m_scene->animation_duration();
            if (ImGui::SliderScalar("##time", ImGuiDataType_Double, &time, &min_time, &max_time, "%.3f s"))
                m_scene->set_time(time);
        } else {
            ImGui::TextDisabled("No animation data");
        }
    }
    ImGui::End();
}

void SceneEditor::geometry_ui(Geometry* geometry)
{
    FALCOR_UNUSED(geometry);
}

void SceneEditor::material_ui(Material* material)
{
    properties_editor(*material, m_property_editor_ctx);
}

bool SceneEditor::inspector_navigation_ui()
{
    if (!m_inspector_return_entity)
        return false;

    Entity* return_entity = m_inspector_return_entity.get();
    if (!return_entity->is_valid() || return_entity->scene() != m_scene.get()) {
        m_inspector_return_entity = {};
        return false;
    }

    std::string label = "< Back to " + return_entity->name();
    if (ImGui::Button(label.c_str())) {
        select_and_reveal(return_entity);
        return true;
    }
    ImGui::Separator();
    return false;
}

void SceneEditor::select_and_reveal(SceneObject* object)
{
    set_selected_object(object);
    m_outliner_search.fill('\0');
    m_reveal_selection_in_outliner = object != nullptr;
}

void SceneEditor::inspect_referenced_object(SceneObject* object, Entity* return_entity)
{
    if (!object)
        return;
    select_and_reveal(object);
    m_inspector_return_entity = ref<Entity>(return_entity);
}

void SceneEditor::geometry_instance_ui(GeometryInstance* instance)
{
    Geometry* geometry = instance->geometry();
    const ImGuiTableFlags reference_table_flags = ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_PadOuterX;
    if (ImGui::BeginTable("##geometry_reference", 3, reference_table_flags)) {
        ImGui::TableSetupColumn("Reference", ImGuiTableColumnFlags_WidthFixed);
        ImGui::TableSetupColumn("Geometry", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("Actions", ImGuiTableColumnFlags_WidthFixed);
        ImGui::TableNextRow();

        ImGui::TableNextColumn();
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted("Geometry");

        ImGui::TableNextColumn();
        ImGui::SetNextItemWidth(-1.f);
        if (scene_collection_combo<Geometry>("##geometry", &geometry, m_scene->geometries()))
            instance->set_geometry(geometry);

        ImGui::TableNextColumn();
        ImGui::BeginDisabled(geometry == nullptr);
        if (ImGui::Button("->"))
            inspect_referenced_object(geometry, instance->entity());
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Open geometry in Inspector");
        ImGui::EndDisabled();

        ImGui::EndTable();
    }

    ImGui::SeparatorText("Materials");
    size_t slot_count = instance->material_slot_count();
    if (slot_count == 0) {
        ImGui::TextDisabled("Assign geometry to edit materials.");
        return;
    }

    if (ImGui::BeginTable("##material_slots", 3, reference_table_flags)) {
        ImGui::TableSetupColumn("Slot", ImGuiTableColumnFlags_WidthFixed);
        ImGui::TableSetupColumn("Material", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("Actions", ImGuiTableColumnFlags_WidthFixed);

        for (size_t slot_index = 0; slot_index < slot_count; ++slot_index) {
            Material* material
                = slot_index < instance->materials().size() ? instance->materials()[slot_index] : nullptr;

            ImGui::PushID(static_cast<int>(slot_index));
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::AlignTextToFramePadding();
            ImGui::Text("%zu", slot_index);

            ImGui::TableNextColumn();
            ImGui::SetNextItemWidth(-1.f);
            if (scene_collection_combo<Material>("##material", &material, m_scene->materials(), true))
                instance->set_material(slot_index, material);

            ImGui::TableNextColumn();
            ImGui::BeginDisabled(material == nullptr);
            if (ImGui::Button("->"))
                inspect_referenced_object(material, instance->entity());
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Open material in Inspector");
            ImGui::EndDisabled();
            ImGui::PopID();
        }
        ImGui::EndTable();
    }
}

void SceneEditor::component_ui(Component* component)
{
    ImGui::PushID(component);
    if (ImGui::CollapsingHeader(component->class_name(), ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Indent();
        if (auto* gi = component->as<GeometryInstance>()) {
            geometry_instance_ui(gi);
        }
        if (auto* camera = component->as<Camera>()) {
            const bool active = m_scene->active_camera() == camera;
            ImGui::BeginDisabled(active);
            if (ImGui::Button("Make Active"))
                m_scene->set_active_camera(camera);
            if (active && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
                ImGui::SetTooltip("This camera is already active");
            ImGui::EndDisabled();
        }
        properties_editor(*component, m_property_editor_ctx);
        if (ImGui::Button("Remove Component"))
            component->remove();
        ImGui::Unindent();
    }
    ImGui::PopID();
}

void SceneEditor::entity_ui(Entity* entity)
{
    if (ImGui::Button("Add Component"))
        ImGui::OpenPopup("Add Component");
    add_component_popup("Add Component");

    Transform transform = entity->transform();
    if (transform_editor("Transform", transform))
        entity->set_transform(transform);

    for (Component* component : entity->components())
        component_ui(component);
}

void SceneEditor::scene_object_node(SceneObject* object, SceneObject* selected, SceneObject*& clicked)
{
    if (!object->is_valid())
        return;
    int node_flags = ImGuiTreeNodeFlags_Leaf;
    if (object == selected)
        node_flags |= ImGuiTreeNodeFlags_Selected;
    bool node_open = ImGui::TreeNodeEx(object, node_flags, "%s", object->name().c_str());
    bool item_clicked = ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen();
    reveal_outliner_item(object);
    if (item_clicked)
        clicked = object;
    if (node_open) {
        ImGui::TreePop();
    }
}

void SceneEditor::entity_node(
    Entity* entity,
    SceneObject* selected,
    SceneObject*& clicked,
    const std::unordered_set<const Entity*>* visible_entities,
    const Entity* reveal_entity
)
{
    if (!entity->is_valid() || (visible_entities && !visible_entities->contains(entity)))
        return;
    int node_flags = 0;
    bool has_visible_children = false;
    for (const Entity* child : entity->children()) {
        if (child->is_valid() && (!visible_entities || visible_entities->contains(child))) {
            has_visible_children = true;
            break;
        }
    }
    if (!has_visible_children)
        node_flags |= ImGuiTreeNodeFlags_Leaf;
    if (entity == selected)
        node_flags |= ImGuiTreeNodeFlags_Selected;
    bool persist_open = reveal_entity && detail::is_entity_ancestor(entity, reveal_entity);
    bool node_open
        = draw_forced_open_entity_node(entity, node_flags, visible_entities && has_visible_children, persist_open);
    bool item_clicked = ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen();
    reveal_outliner_item(entity);
    if (item_clicked)
        clicked = entity;
    if (node_open) {
        const std::vector<Entity*>& children = entity->children();
        if (m_sort_outliner) {
            std::vector<Entity*> sorted_children;
            sorted_children.reserve(children.size());
            sorted_children.insert(sorted_children.end(), children.begin(), children.end());
            detail::sort_outliner_objects(sorted_children);
            for (Entity* child : sorted_children)
                entity_node(child, selected, clicked, visible_entities, reveal_entity);
        } else {
            for (Entity* child : children)
                entity_node(child, selected, clicked, visible_entities, reveal_entity);
        }
        ImGui::TreePop();
    }
}

void SceneEditor::reveal_outliner_item(SceneObject* object)
{
    if (m_reveal_selection_in_outliner && object == m_selected_object) {
        ImGui::SetScrollHereY(0.5f);
        m_reveal_selection_in_outliner = false;
    }
}

template<typename T>
void SceneEditor::add_scene_object_popup(const char* label)
{
    if (ImGui::BeginPopup(label)) {
        for (const auto& class_info : SceneObjectFactory<T>::get().class_infos()) {
            if (ImGui::Selectable(class_info.name.c_str())) {
                m_scene->_create_object<T>(class_info.name);
                ImGui::CloseCurrentPopup();
            }
        }
        ImGui::EndPopup();
    }
}

void SceneEditor::add_component_popup(const char* label)
{
    Entity* entity = m_selected_object ? m_selected_object->as<Entity>() : nullptr;
    if (!entity)
        return;
    if (ImGui::BeginPopup(label)) {
        for (const auto& class_info : SceneObjectFactory<Component>::get().class_infos()) {
            if (ImGui::Selectable(class_info.name.c_str())) {
                entity->create_component(class_info.name);
                ImGui::CloseCurrentPopup();
            }
        }
        ImGui::EndPopup();
    }
}

template<typename T>
bool SceneEditor::scene_collection_combo(
    const char* label,
    T** selected,
    SceneObjectCollectionView<T>& collection,
    bool allow_none
)
{
    bool changed = false;
    if (ImGui::BeginCombo(label, (*selected) ? (*selected)->name().c_str() : "None")) {
        if (allow_none) {
            bool is_selected = *selected == nullptr;
            if (ImGui::Selectable("None", is_selected) && !is_selected) {
                *selected = nullptr;
                changed = true;
            }
            if (is_selected)
                ImGui::SetItemDefaultFocus();
            ImGui::Separator();
        }
        for (size_t i = 0; i < collection.size(); ++i) {
            T* item = collection[i];
            bool is_selected = (item == *selected);
            ImGui::PushID(item);
            if (ImGui::Selectable(item->name().c_str(), is_selected)) {
                *selected = item;
                changed = true;
            }
            if (is_selected)
                ImGui::SetItemDefaultFocus();
            ImGui::PopID();
        }
        ImGui::EndCombo();
    }
    return changed;
}

template<typename T, typename Func>
void SceneEditor::scene_object_ui(SceneObject* object, Func func)
{
    if (T* typed_object = object->as<T>()) {
        ImGui::Text("%s: %s", object->class_name(), object->name().c_str());
        ImGui::Separator();
        (this->*func)(typed_object);
    }
}

void SceneEditor::viewport_gizmo()
{
    if (m_tool_mode == ToolMode::select)
        return;

    Entity* entity = m_selected_object ? m_selected_object->as<Entity>() : nullptr;
    if (!entity)
        return;

    float4x4 parent_world_from_object
        = entity->parent() ? entity->parent()->world_from_object_matrix() : float4x4::identity();
    float4x4 world_from_object = mul(parent_world_from_object, entity->transform().matrix());
    float4x4 edit_matrix = transpose(world_from_object);

    ImGuizmo::OPERATION operation{};
    switch (m_tool_mode) {
    case ToolMode::move:
        operation = ImGuizmo::TRANSLATE;
        break;
    case ToolMode::rotate:
        operation = ImGuizmo::ROTATE;
        break;
    case ToolMode::scale:
        operation = ImGuizmo::SCALE;
        break;
    default:
        return;
    }
    ImGuizmo::MODE mode = (m_transform_space == TransformSpace::local) ? ImGuizmo::LOCAL : ImGuizmo::WORLD;

    bool camera_interacting = m_camera_controller && m_camera_controller->is_interacting();
    ImGuizmo::Enable(!camera_interacting);
    if (ImGuizmo::Manipulate(
            transpose(m_view_matrix_no_scale).data(),
            transpose(m_proj_matrix).data(),
            operation,
            mode,
            edit_matrix.data()
        )) {
        world_from_object = transpose(edit_matrix);
        float4x4 transform_matrix = mul(sgl::math::inverse(parent_world_from_object), world_from_object);
        entity->set_transform(Transform(transform_matrix));
    }
    ImGuizmo::Enable(true);
}

void SceneEditor::help_ui()
{
    if (!m_show_help)
        return;

    static const SceneEditor::HelpSection CAMERA_HELP_SECTION = {
        "Camera Controls",
        {
            {"RMB + Mouse", "Look around (first person)"},
            {"RMB + W/A/S/D", "Move forward/left/back/right"},
            {"RMB + Q/E", "Move down/up"},
            {"RMB + Shift", "10x speed multiplier"},
            {"RMB + Ctrl", "0.1x speed multiplier"},
            {"RMB + Scroll", "Adjust movement speed"},
            {"MMB + Mouse", "Pan camera"},
            {"Alt + LMB + Mouse", "Orbit around pivot"},
            {"Alt + RMB + Mouse", "Dolly (zoom to pivot)"},
            {"Alt + MMB + Mouse", "Track (pan + move pivot)"},
            {"Scroll Wheel", "Zoom in/out"},
        },
    };

    HelpSection editor_help_section{.title = "Editor Tools"};
    const std::array editor_help_commands{
        &command_ids::TOOL_SELECT,
        &command_ids::TOOL_MOVE,
        &command_ids::TOOL_ROTATE,
        &command_ids::TOOL_SCALE,
        &command_ids::TOGGLE_TRANSFORM_SPACE,
        &command_ids::FRAME_SELECTED,
        &command_ids::DELETE_SELECTED,
        &command_ids::TOGGLE_HELP,
    };
    for (const CommandId* id : editor_help_commands) {
        const EditorCommand* command = m_commands.find(*id);
        FALCOR_ASSERT(command && command->shortcut);
        editor_help_section.entries.push_back({command->shortcut->display_name, command->label});
    }

    auto help_entry = [](const char* key, const char* description)
    {
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TextUnformatted(key);
        ImGui::TableNextColumn();
        ImGui::TextUnformatted(description);
    };

    auto help_section = [&help_entry](int id, const HelpSection& section)
    {
        const ImGuiTableFlags table_flags = ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_PadOuterX;
        ImGui::SeparatorText(section.title.c_str());
        ImGui::PushID(id);
        if (ImGui::BeginTable("##section", 2, table_flags)) {
            ImGui::TableSetupColumn("Key", ImGuiTableColumnFlags_WidthFixed, 200.f);
            ImGui::TableSetupColumn("Action", ImGuiTableColumnFlags_WidthStretch);
            for (const auto& entry : section.entries)
                help_entry(entry.key.c_str(), entry.description.c_str());
            ImGui::EndTable();
        }
        ImGui::PopID();
    };

    ImGui::OpenPopup("Help");

    ImVec2 center = ImGui::GetMainViewport()->GetCenter();
    ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));

    ImGuiWindowFlags flags = ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings;
    if (ImGui::BeginPopupModal("Help", &m_show_help, flags)) {
        // Keep F1 in sync with the command shortcut. The modal captures keyboard input, so close it locally while
        // ignoring the F1 press used to open it on its appearing frame.
        if (!ImGui::IsWindowAppearing() && (ImGui::IsKeyPressed(ImGuiKey_F1) || ImGui::IsKeyPressed(ImGuiKey_Escape)))
            m_show_help = false;

        int id = 0;
        help_section(id++, CAMERA_HELP_SECTION);
        help_section(id++, editor_help_section);
        for (const auto& section : m_help_sections)
            help_section(id++, section);
        ImGui::EndPopup();
    }
}

} // namespace falcor::ui
