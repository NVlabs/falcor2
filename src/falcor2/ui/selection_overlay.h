// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/core/macros.h"
#include "falcor2/core/object.h"

#include "falcor2/render/fwd.h"
#include "falcor2/render/shared_scene_types.h"
#include "falcor2/render/scene_requirements.h"
#include "falcor2/utils/aabb.h"

#include <sgl/device/fwd.h>

#include <span>
#include <optional>
#include <cstdint>

namespace falcor::ui {

/// GPU compute pass that renders selection outlines/highlights onto an output texture.
/// Uses a 5x5 neighborhood sampling approach to detect silhouette edges of selected entities.
/// Supports rendering outlines for both visible and occluded selected objects using
/// a selection probe ray pass that traces through non-selected geometry.
class FALCOR_API SelectionOverlay : public Object {
    FALCOR_OBJECT(SelectionOverlay)
public:
    /// Options for SelectionOverlay.
    struct Options {
        /// Whether occluded selections are shown.
        /// This comes with a performance cost due an additional probe ray pass.
        bool show_occluded{true};
        /// Color used for outlines and fill.
        float3 selection_color{0.7f, 0.7f, 0.2f};
        /// Opacity of the fill overlay (0 = no fill).
        float fill_opacity{0.f};
    };

    explicit SelectionOverlay(ref<sgl::Device> device, std::optional<Options> options = std::nullopt);

    /// Options.
    const Options& options() const { return m_options; }

    /// Set options.
    void set_options(const Options& options);

    /// Set the selection from a list of entities.
    /// Clears the previous selection.
    /// @param entities Entites to select.
    void set_selected_entities(std::span<Entity*> entities);

    /// Set the selection from a single entity.
    /// Clears the previous selection.
    /// @param entity Entity to select, or nullptr to clear selection.
    void set_selected_entity(Entity* entity);

    /// Set the selection from selected geometry instance IDs directly.
    /// Clears the previous selection.
    /// @param ids Geometry instance IDs to select.
    void set_selected_geometry_instance_ids(std::span<const shared::GeometryInstanceID> ids);

    /// Clear the selection.
    void clear_selection();

    /// Execute the selection overlay pass.
    /// Reads selected geometry instances from geometry_instance_id_texture and writes
    /// outlines/highlights onto output_texture. When Options::show_occluded is true, an
    /// additional probe ray tracing pass is dispatched against the scene to detect selected
    /// objects behind occluders; when it is false, probe tracing is skipped and only the
    /// geometry_instance_id_texture is used for outline detection.
    /// @param command_encoder Command encoder to record commands into.
    /// @param output_texture Texture to write the selection overlay output to.
    /// @param geometry_instance_id_texture Texture containing geometry instance IDs.
    /// @param scene The scene to trace probe rays against.
    /// @param camera The camera to generate probe rays from.
    void draw_overlay(
        sgl::CommandEncoder* command_encoder,
        sgl::Texture* output_texture,
        sgl::Texture* geometry_instance_id_texture,
        const Scene* scene,
        const Camera* camera
    );

private:
    void selection_bitmap_clear()
    {
        m_selection_bitmap.clear();
        m_selection_bitmap_dirty = true;
        m_aabb_valid = false;
    }

    void selection_bitmap_set(uint32_t id)
    {
        uint32_t index = id / 32;
        uint32_t bit = 1u << (id % 32);
        if (index >= m_selection_bitmap.size())
            m_selection_bitmap.resize(index + 1, 0);
        m_selection_bitmap[index] |= bit;
        m_selection_bitmap_dirty = true;
    }

    void create_probe_kernel(const Scene* scene);
    void update_aabb();

    /// Maximum number of selected entities for which AABB culling is enabled.
    static constexpr uint32_t SELECTION_AABB_CULLING_MAX_OBJECTS = 16;

    ref<sgl::Device> m_device;
    Options m_options;

    ref<sgl::ComputeKernel> m_kernel;
    ref<sgl::ComputeKernel> m_probe_kernel;
    ref<sgl::Texture> m_selected_hit_texture;
    SceneRequirements m_scene_requirements;
    std::vector<uint32_t> m_selection_bitmap;
    bool m_selection_bitmap_dirty{false};
    ref<sgl::Buffer> m_selection_bitmap_buffer;

    // Selected entities for AABB recomputation.
    std::vector<Entity*> m_selected_entities;

    // AABB culling state.
    AABB m_selection_aabb;
    bool m_aabb_valid{false};

    // Raytracing pipeline state (for CUDA/OptiX).
    bool m_use_raytracing_pipeline{false};
    uint64_t m_requirements_generation{0};
    struct ProbeRT {
        ref<sgl::ShaderProgram> program;
        ref<sgl::RayTracingPipeline> pipeline;
        ref<sgl::ShaderTable> shader_table;
    } m_probe_rt;
};

} // namespace falcor::ui
