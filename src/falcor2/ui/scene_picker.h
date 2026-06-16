// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/core/macros.h"
#include "falcor2/core/object.h"
#include "falcor2/core/types.h"

#include "falcor2/render/fwd.h"
#include "falcor2/render/shared_scene_types.h"

#include <sgl/device/fwd.h>

#include <cstdint>

namespace falcor::ui {

/// GPU-based object picking service.
/// Renders geometry instance IDs to an internal texture using a standalone ray-tracing
/// compute pass, then performs single-pixel readback to determine which entity was clicked.
class FALCOR_API ScenePicker : public Object {
    FALCOR_OBJECT(ScenePicker)
public:
    explicit ScenePicker(ref<sgl::Device> device);

    /// Render geometry instance IDs into the internal texture.
    /// Creates/resizes the texture as needed based on camera dimensions.
    /// @param command_encoder The command encoder to record GPU commands into.
    /// @param scene The scene to render for picking.
    /// @param camera The camera to render from (used for dimensions and picking rays).
    void render(sgl::CommandEncoder* command_encoder, const Scene* scene, const Camera* camera);

    /// The internal geometry instance ID texture.
    /// Returns nullptr if render() has not been called yet.
    sgl::Texture* geometry_instance_id_texture() const { return m_geometry_instance_id_texture; }

    /// Pick a geometry instance ID at the given pixel position using either
    /// the internal texture or a user-provided external texture.
    /// @param position Pixel coordinates in the texture (viewport-local).
    /// @param geometry_instance_id_texture Optional texture containing geometry instance IDs.
    /// @return The geometry instance ID, or GeometryInstanceID::invalid if nothing was hit.
    shared::GeometryInstanceID pick(uint2 position, sgl::Texture* geometry_instance_id_texture = nullptr);

    /// Pick an entity at the given pixel position using either
    /// the internal texture or a user-provided external texture.
    /// @param scene The scene to search for entities.
    /// @param position Pixel coordinates in the texture (viewport-local).
    /// @param geometry_instance_id_texture Optional texture containing geometry instance IDs.
    /// @return The entity at the position, or nullptr if nothing was hit.
    Entity* pick_entity(Scene* scene, uint2 position, sgl::Texture* geometry_instance_id_texture = nullptr);

    /// Find the entity that owns the given geometry instance ID.
    /// Searches through all GeometryInstance components in the scene.
    /// @return The owning entity, or nullptr if the ID is invalid.
    static Entity* find_entity_by_geometry_instance_id(Scene* scene, shared::GeometryInstanceID geometry_instance_id);

private:
    void create_pipelines(const Scene* scene);

    ref<sgl::Device> m_device;
    ref<sgl::Buffer> m_readback_buffer;
    ref<sgl::Texture> m_geometry_instance_id_texture;

    bool m_use_raytracing_pipeline{false};
    uint64_t m_requirements_generation{0};

    // Compute pipeline.
    struct RenderIdsCS {
        ref<sgl::ComputeKernel> kernel;
    } m_render_ids_cs;

    // Ray-tracing pipeline.
    struct RenderIdsRT {
        ref<sgl::ShaderProgram> program;
        ref<sgl::RayTracingPipeline> pipeline;
        ref<sgl::ShaderTable> shader_table;
    } m_render_ids_rt;
};

} // namespace falcor::ui
