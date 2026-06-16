// SPDX-License-Identifier: Apache-2.0

#include "falcor2/ui/scene_editor.h"
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

namespace falcor::ui {

void SceneEditor::set_scene(ref<Scene> scene)
{
    if (scene != m_scene) {
        m_scene = std::move(scene);
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

    float aspect_ratio = static_cast<float>(camera->width()) / static_cast<float>(camera->height());
    float4x4 view_matrix = inverse(camera->entity()->world_from_object_matrix());
    float4x4 proj_matrix = sgl::math::perspective(sgl::math::radians(camera->fov_y()), aspect_ratio, 0.1f, 1000.0f);
    set_camera_matrices(view_matrix, proj_matrix);
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

void SceneEditor::begin_frame()
{
    m_gizmo_frame_initialized = ImGui::GetFrameCount();
    ImGuizmo::SetImGuiContext(ImGui::GetCurrentContext());
    ImGuizmo::BeginFrame();
    ImGuizmo::SetOrthographic(false);
}

void SceneEditor::editor_ui()
{
    setup_dockspace();

    // F1 toggles help -- available even without a scene loaded.
    if (ImGui::IsKeyPressed(ImGuiKey_F1))
        m_show_help = !m_show_help;

    help_ui();

    bool disabled = m_scene == nullptr;
    if (disabled)
        ImGui::BeginDisabled();

    toolbar_ui();
    outliner_ui();
    inspector_ui();
    transport_ui();

    if (disabled) {
        ImGui::EndDisabled();
        return;
    }

    if (ImGui::IsKeyPressed(ImGuiKey_Delete))
        remove_selected_object();
    if (ImGui::IsKeyPressed(ImGuiKey_Space))
        set_playing(!playing());
    if (ImGui::IsKeyPressed(ImGuiKey_Q))
        m_tool_mode = ToolMode::select;
    if (ImGui::IsKeyPressed(ImGuiKey_W))
        m_tool_mode = ToolMode::move;
    if (ImGui::IsKeyPressed(ImGuiKey_E))
        m_tool_mode = ToolMode::rotate;
    if (ImGui::IsKeyPressed(ImGuiKey_R))
        m_tool_mode = ToolMode::scale;
    if (ImGui::IsKeyPressed(ImGuiKey_X))
        m_transform_space
            = (m_transform_space == TransformSpace::local) ? TransformSpace::world : TransformSpace::local;
}

void SceneEditor::viewport_ui(sgl::Texture* output_texture, float fps)
{
    if (m_gizmo_frame_initialized != ImGui::GetFrameCount())
        begin_frame();

    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
    if (ImGui::Begin("Viewport", nullptr, ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoScrollbar)) {
        ImVec2 pos = ImGui::GetCursorScreenPos();
        ImVec2 size = ImGui::GetContentRegionAvail();
        m_viewport_state.pos = float2(pos.x, pos.y);
        m_viewport_state.size = float2(size.x, size.y);
        m_viewport_state.hovered = ImGui::IsWindowHovered();
        m_viewport_state.focused = ImGui::IsWindowFocused();
        m_viewport_state.valid = size.x > 0.f && size.y > 0.f;
        if (output_texture && size.x > 0 && size.y > 0) {
            ImGui::Image(output_texture, size);
        }

        // ImGuizmo: render gizmo inside the viewport window.
        ImGuizmo::SetDrawlist();
        ImGuizmo::SetRect(pos.x, pos.y, size.x, size.y);
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

void SceneEditor::setup_dockspace()
{
    ImGuiViewport* vp = ImGui::GetMainViewport();
    ImVec2 dockspace_pos(vp->WorkPos.x, vp->WorkPos.y + TOOLBAR_HEIGHT);
    ImVec2 dockspace_size(vp->WorkSize.x, vp->WorkSize.y - TOOLBAR_HEIGHT);
    ImGui::SetNextWindowPos(dockspace_pos);
    ImGui::SetNextWindowSize(dockspace_size);
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
    ImGuiViewport* vp = ImGui::GetMainViewport();
    ImGui::DockBuilderSetNodeSize(dockspace_id, ImVec2(vp->WorkSize.x, vp->WorkSize.y - TOOLBAR_HEIGHT));

    ImGuiID rest = dockspace_id;

    // Split: bottom (transport) | rest
    ImGuiID bottom;
    ImGui::DockBuilderSplitNode(rest, ImGuiDir_Down, 0.15f, &bottom, &rest);

    // Split rest: left (settings) | rest
    ImGuiID left;
    ImGui::DockBuilderSplitNode(rest, ImGuiDir_Left, LEFT_PANEL_SIZE, &left, &rest);

    // Split rest: right (outliner/inspector) | viewport
    ImGuiID right, viewport;
    ImGui::DockBuilderSplitNode(rest, ImGuiDir_Right, RIGHT_PANEL_SIZE / (1.f - LEFT_PANEL_SIZE), &right, &viewport);

    // Split right: right_top (outliner) | right_bottom (inspector)
    ImGuiID right_top, right_bottom;
    ImGui::DockBuilderSplitNode(right, ImGuiDir_Up, 0.50f, &right_top, &right_bottom);

    // Mark the viewport as central node with no tab bar.
    ImGuiDockNode* viewport_node = ImGui::DockBuilderGetNode(viewport);
    if (viewport_node)
        viewport_node->LocalFlags |= ImGuiDockNodeFlags_CentralNode | ImGuiDockNodeFlags_NoTabBar;

    // Dock windows into slots.
    ImGui::DockBuilderDockWindow("Settings", left);
    ImGui::DockBuilderDockWindow("Viewport", viewport);
    ImGui::DockBuilderDockWindow("Outliner", right_top);
    ImGui::DockBuilderDockWindow("Inspector", right_bottom);
    ImGui::DockBuilderDockWindow("Transport", bottom);

    ImGui::DockBuilderFinish(dockspace_id);
}

void SceneEditor::toolbar_ui()
{
    ImGuiViewport* vp = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(vp->WorkPos);
    ImGui::SetNextWindowSize(ImVec2(vp->WorkSize.x, TOOLBAR_HEIGHT));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);

    ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize
        | ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus;

    if (ImGui::Begin("##Toolbar", nullptr, flags)) {
        if (toggle_button("Select (Q)", m_tool_mode == ToolMode::select))
            m_tool_mode = ToolMode::select;
        ImGui::SameLine();
        if (toggle_button("Move (W)", m_tool_mode == ToolMode::move))
            m_tool_mode = ToolMode::move;
        ImGui::SameLine();
        if (toggle_button("Rotate (E)", m_tool_mode == ToolMode::rotate))
            m_tool_mode = ToolMode::rotate;
        ImGui::SameLine();
        if (toggle_button("Scale (R)", m_tool_mode == ToolMode::scale))
            m_tool_mode = ToolMode::scale;
        ImGui::SameLine();
        ImGui::SeparatorEx(ImGuiSeparatorFlags_Vertical);
        ImGui::SameLine();
        const char* space_label = (m_transform_space == TransformSpace::local) ? "Local (X)" : "World (X)";
        if (toggle_button(space_label, false))
            m_transform_space
                = (m_transform_space == TransformSpace::local) ? TransformSpace::world : TransformSpace::local;

        // Camera move speed slider.
        if (m_camera_controller) {
            ImGui::SameLine();
            ImGui::SeparatorEx(ImGuiSeparatorFlags_Vertical);
            ImGui::SameLine();
            ImGui::SetNextItemWidth(120.f);
            float speed = m_camera_controller->move_speed();
            if (ImGui::SliderFloat(
                    "Speed",
                    &speed,
                    CameraController::MIN_MOVE_SPEED,
                    CameraController::MAX_MOVE_SPEED,
                    "%.2f",
                    ImGuiSliderFlags_Logarithmic
                ))
                m_camera_controller->set_move_speed(speed);
        }

        // Right-aligned help button.
        const char* help_button_label = "Help (F1)";
        float button_width = ImGui::CalcTextSize(help_button_label).x + ImGui::GetStyle().FramePadding.x * 2.f;
        ImGui::SameLine(ImGui::GetWindowWidth() - button_width - ImGui::GetStyle().WindowPadding.x);
        if (ImGui::Button(help_button_label))
            m_show_help = !m_show_help;
    }
    ImGui::End();
    ImGui::PopStyleVar();
}

void SceneEditor::outliner_ui()
{
    SceneObject* clicked_object = nullptr;
    if (ImGui::Begin("Outliner")) {

        if (ImGui::Button("Add Geometry"))
            ImGui::OpenPopup("Add Geometry");
        add_scene_object_popup<Geometry>("Add Geometry");

        ImGui::SameLine();
        if (ImGui::Button("Add Material"))
            ImGui::OpenPopup("Add Material");
        add_scene_object_popup<Material>("Add Material");

        ImGui::SameLine();
        if (ImGui::Button("Add Entity"))
            m_scene->create_entity();

        ImGui::BeginChild("##scrollregion");
        if (ImGui::TreeNode("Geometries")) {
            auto& geometries = m_scene->geometries();
            for (size_t i = 0; i < geometries.size(); ++i)
                scene_object_node(geometries[i], m_selected_object, clicked_object);
            ImGui::TreePop();
        }
        if (ImGui::TreeNode("Materials")) {
            auto& materials = m_scene->materials();
            for (size_t i = 0; i < materials.size(); ++i)
                scene_object_node(materials[i], m_selected_object, clicked_object);
            ImGui::TreePop();
        }
        if (ImGui::TreeNode("Entities")) {
            auto& entities = m_scene->entities();
            for (size_t i = 0; i < entities.size(); ++i) {
                Entity* entity = entities[i];
                if (entity->parent() == nullptr)
                    entity_node(entity, m_selected_object, clicked_object);
            }
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
    if (ImGui::Begin("Inspector")) {
        if (m_selected_object) {
            scene_object_ui<Geometry>(m_selected_object, &SceneEditor::geometry_ui);
            scene_object_ui<Material>(m_selected_object, &SceneEditor::material_ui);
            scene_object_ui<Entity>(m_selected_object, &SceneEditor::entity_ui);
        } else {
            ImGui::Text("No object selected!");
        }
    }
    ImGui::End();
}

void SceneEditor::transport_ui()
{
    if (ImGui::Begin("Transport")) {
        if (m_scene && m_scene->has_animation()) {
            // Play/pause toggle.
            if (ImGui::Button(playing() ? "Pause" : "Play"))
                set_playing(!playing());
            ImGui::SameLine();
            if (ImGui::Button("Reset")) {
                set_playing(false);
                m_scene->set_time(0.0);
            }
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

void SceneEditor::geometry_instance_ui(GeometryInstance* instance)
{
    Geometry* geometry = instance->geometry();
    if (scene_collection_combo<Geometry>("Geometry", &geometry, m_scene->geometries()))
        instance->set_geometry(geometry);
}

void SceneEditor::component_ui(Component* component)
{
    ImGui::PushID(component);
    if (ImGui::CollapsingHeader(component->class_name(), ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Indent();
        if (auto* gi = component->as<GeometryInstance>()) {
            geometry_instance_ui(gi);
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
    if (ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen())
        clicked = object;
    if (node_open) {
        ImGui::TreePop();
    }
}

void SceneEditor::entity_node(Entity* entity, SceneObject* selected, SceneObject*& clicked)
{
    if (!entity->is_valid())
        return;
    int node_flags = 0;
    if (entity->children().empty())
        node_flags |= ImGuiTreeNodeFlags_Leaf;
    if (entity == selected)
        node_flags |= ImGuiTreeNodeFlags_Selected;
    bool node_open = ImGui::TreeNodeEx(entity, node_flags, "%s", entity->name().c_str());
    if (ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen())
        clicked = entity;
    if (node_open) {
        for (Entity* child : entity->children())
            entity_node(child, selected, clicked);
        ImGui::TreePop();
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
bool SceneEditor::scene_collection_combo(const char* label, T** selected, SceneObjectCollectionView<T>& collection)
{
    bool changed = false;
    if (ImGui::BeginCombo(label, (*selected) ? (*selected)->name().c_str() : "None")) {
        for (size_t i = 0; i < collection.size(); ++i) {
            T* item = collection[i];
            bool is_selected = (item == *selected);
            if (ImGui::Selectable(item->name().c_str(), is_selected)) {
                *selected = item;
                changed = true;
            }
            if (is_selected) {
                ImGui::SetItemDefaultFocus();
            }
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
            {"F", "Focus camera on selection"},
        },
    };

    static const SceneEditor::HelpSection EDITOR_HELP_SECTION = {
        "Editor Tools",
        {
            {"Q", "Select mode"},
            {"W", "Move mode"},
            {"E", "Rotate mode"},
            {"R", "Scale mode"},
            {"X", "Toggle world/local transform space"},
            {"Delete", "Remove selected object"},
            {"F1", "Toggle this help screen"},
        },
    };

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
        int id = 0;
        help_section(id++, CAMERA_HELP_SECTION);
        help_section(id++, EDITOR_HELP_SECTION);
        for (const auto& section : m_help_sections)
            help_section(id++, section);
        ImGui::EndPopup();
    }
}

} // namespace falcor::ui
