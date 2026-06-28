// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/core/macros.h"
#include "falcor2/core/object.h"
#include "falcor2/core/types.h"

#include "falcor2/ui/fwd.h"
#include "falcor2/ui/property_editor.h"

#include "falcor2/render/fwd.h"

#include <sgl/core/input.h>
#include <sgl/device/fwd.h>

#include <string>
#include <vector>

namespace falcor::ui {

class FALCOR_API SceneEditor : public Object {
    FALCOR_OBJECT(SceneEditor)
public:
    enum class ToolMode {
        select,
        move,
        rotate,
        scale,
    };
    SGL_ENUM_INFO(
        ToolMode,
        {
            {ToolMode::select, "select"},
            {ToolMode::move, "move"},
            {ToolMode::rotate, "rotate"},
            {ToolMode::scale, "scale"},
        }
    );

    enum class TransformSpace {
        world,
        local,
    };
    SGL_ENUM_INFO(
        TransformSpace,
        {
            {TransformSpace::world, "world"},
            {TransformSpace::local, "local"},
        }
    );

    struct ViewportState {
        /// Viewport position in screen coordinates (top-left corner).
        float2 pos{0.f, 0.f};
        /// Viewport size in pixels.
        float2 size{0.f, 0.f};
        /// Whether the viewport was hovered in the last rendered viewport snapshot.
        bool hovered{false};
        /// Whether the viewport was focused in the last rendered viewport snapshot.
        bool focused{false};
        /// Whether the last rendered viewport snapshot has a non-zero size.
        bool valid{false};
    };

    /// A single key/description pair shown in the help popup.
    struct HelpEntry {
        std::string key;
        std::string description;
    };

    /// A titled group of help entries shown in the help popup.
    struct HelpSection {
        std::string title;
        std::vector<HelpEntry> entries;
    };

    SceneEditor() = default;

    /// The scene being edited, or nullptr if no scene is loaded.
    Scene* scene() const { return m_scene; }
    void set_scene(ref<Scene> scene);

    /// Optional camera controller whose move speed is shown in the toolbar.
    CameraController* camera_controller() const { return m_camera_controller.get(); }
    void set_camera_controller(ref<CameraController> controller);

    /// The currently selected scene object, or nullptr if no object is selected.
    SceneObject* selected_object() const { return m_selected_object; }
    void set_selected_object(SceneObject* object);

    /// Monotonically increasing version counter that changes whenever the selection changes.
    /// Consumers can cache the value and compare to detect changes without consuming state.
    uint64_t selection_version() const { return m_selection_version; }

    /// Remove the currently selected object from the scene and clear the selection.
    void remove_selected_object();

    /// Current tool mode (select, move, rotate, scale).
    ToolMode tool_mode() const { return m_tool_mode; }
    void set_tool_mode(ToolMode mode);

    /// Current transform space for the gizmo (world or local).
    TransformSpace transform_space() const { return m_transform_space; }
    void set_transform_space(TransformSpace space);

    /// Set the active camera used for gizmo rendering.
    void set_active_camera(const Camera* camera);

    /// Set the camera view and projection matrices used for rendering the gizmo.
    void set_camera_matrices(const float4x4& view, const float4x4& proj);

    /// Whether animation playback is active.
    bool playing() { return m_playing; }
    /// Start or stop animation playback.
    void set_playing(bool playing) { m_playing = playing; }

    /// Whether animation looping is enabled.
    bool looping() { return m_looping; }
    /// Enable or disable animation looping.
    void set_looping(bool looping) { m_looping = looping; }

    /// Advance animation playback by @p dt seconds (call once per frame).
    /// When playing, advances the scene time and optionally loops.
    void update_playback(double dt);

    /// Whether the editor UI is visible.
    /// Used by SceneInteractionController to gate viewport interactions.
    bool visible() const { return m_visible; }
    void set_visible(bool visible) { m_visible = visible; }

    /// Last rendered viewport state.
    ViewportState viewport_state() const { return m_viewport_state; }

    /// Handle scene editor keyboard shortcuts.
    /// @return True if the event was consumed by an editor shortcut.
    bool handle_keyboard_shortcut(const sgl::KeyboardEvent& event);

    /// Optional explicit ImGuizmo initialization for the current ImGui frame.
    /// viewport_ui() will initialize ImGuizmo automatically if this was not called.
    void begin_frame();

    /// Draw the scene editor UI.
    void editor_ui();

    /// Draw the viewport with the rendered scene and the gizmo.
    void viewport_ui(sgl::Texture* output_texture, float fps);

    /// Returns true if the gizmo is currently hovered or being manipulated.
    bool is_gizmo_active() const;

    /// Whether the viewport is currently interactive (valid and hovered).
    /// Uses the viewport state from the last rendered frame.
    bool is_viewport_interactive() const;

    /// Test whether a screen-space position can be used for viewport picking.
    /// Returns true if the position is inside the viewport and the gizmo is not active.
    /// On success, writes the viewport-local pixel coordinates to @p local_pos.
    bool can_pick_at(float2 screen_pos, uint2& local_pos) const;

    /// Add an extra section of help entries (e.g. application-level shortcuts).
    /// These are displayed after the built-in camera and editor sections.
    void add_help_section(HelpSection section) { m_help_sections.push_back(std::move(section)); }

private:
    /// Size of the toolbar at the top of the viewport.
    static constexpr float TOOLBAR_HEIGHT = 54.f;
    /// Relative size of the left docked panel.
    static constexpr float LEFT_PANEL_SIZE = 0.25f;
    /// Relative size of the right docked panel.
    static constexpr float RIGHT_PANEL_SIZE = 0.25f;

    ref<Scene> m_scene;
    ref<CameraController> m_camera_controller;
    SceneObject* m_selected_object{nullptr};
    uint64_t m_selection_version{0};

    ToolMode m_tool_mode{ToolMode::select};
    TransformSpace m_transform_space{TransformSpace::local};

    /// Animation transport state.
    bool m_playing{false};
    bool m_looping{true};

    PropertyEditorContext m_property_editor_ctx;

    ViewportState m_viewport_state;

    float4x4 m_view_matrix{float4x4::identity()};
    float4x4 m_view_matrix_no_scale{float4x4::identity()};
    float4x4 m_proj_matrix{float4x4::identity()};

    bool m_visible{true};
    bool m_show_help{false};

    std::vector<HelpSection> m_help_sections;

    bool m_layout_initialized{false};
    int m_gizmo_frame_initialized{-1};

    void setup_dockspace();
    void setup_default_layout(uint32_t dockspace_id);

    void toolbar_ui();
    void outliner_ui();
    void inspector_ui();
    void transport_ui();
    void help_ui();

    void geometry_ui(Geometry* geometry);
    void material_ui(Material* material);
    void geometry_instance_ui(GeometryInstance* instance);
    void component_ui(Component* component);
    void entity_ui(Entity* entity);

    void scene_object_node(SceneObject* object, SceneObject* selected, SceneObject*& clicked);
    void entity_node(Entity* entity, SceneObject* selected, SceneObject*& clicked);

    template<typename T>
    void add_scene_object_popup(const char* label);

    void add_component_popup(const char* label);

    template<typename T>
    bool scene_collection_combo(const char* label, T** selected, SceneObjectCollectionView<T>& collection);

    template<typename T, typename Func>
    void scene_object_ui(SceneObject* object, Func func);

    void viewport_gizmo();
};

SGL_ENUM_REGISTER(SceneEditor::ToolMode);
SGL_ENUM_REGISTER(SceneEditor::TransformSpace);

} // namespace falcor::ui
