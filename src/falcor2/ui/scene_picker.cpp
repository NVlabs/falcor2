// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "falcor2/ui/scene_picker.h"

#include "falcor2/render/scene.h"
#include "falcor2/render/ray_tracing_setup.h"
#include "falcor2/render/component/camera.h"
#include "falcor2/render/component/geometry_instance.h"

#include <sgl/device/device.h>
#include <sgl/device/resource.h>
#include <sgl/device/command.h>
#include <sgl/device/shader.h>
#include <sgl/device/kernel.h>
#include <sgl/device/pipeline.h>
#include <sgl/device/raytracing.h>
#include <sgl/device/shader_cursor.h>

namespace falcor::ui {

ScenePicker::ScenePicker(ref<sgl::Device> device)
    : m_device(std::move(device))
{
    m_readback_buffer = m_device->create_buffer({
        .size = sizeof(uint32_t),
        .usage = sgl::BufferUsage::unordered_access,
    });

    // Use ray tracing pipeline on CUDA, and compute shader pipeline on other APIs.
    m_use_raytracing_pipeline = (m_device->type() == sgl::DeviceType::cuda);
}

void ScenePicker::create_pipelines(const Scene* scene)
{
    // Early out if the pipelines are already created and the scene requirements haven't changed.
    if (m_use_raytracing_pipeline) {
        if (m_render_ids_rt.pipeline && m_requirements_generation == scene->requirements_generation())
            return;
    } else {
        if (m_render_ids_cs.kernel && m_requirements_generation == scene->requirements_generation())
            return;
    }

    m_requirements_generation = scene->requirements_generation();

    const SceneRequirements& requirements = scene->requirements();

    std::vector<ref<sgl::SlangModule>> modules;
    modules.push_back(m_device->load_module("falcor2/ui/kernels/scene_picker.slang"));
    modules.insert(modules.end(), requirements.modules.begin(), requirements.modules.end());
    ref<sgl::SlangModule> module = m_device->compose_modules("scene_picker", modules, requirements.type_conformances);

    if (m_use_raytracing_pipeline) {
        SceneRayTracingSetup::RayDesc intersect_ray_type;
        intersect_ray_type.name = "intersect";
        intersect_ray_type.has_miss = true;
        intersect_ray_type.has_closest_hit = true;
        SceneRayTracingSetup rt_setup = SceneRayTracingSetup::create(scene, {intersect_ray_type});

        m_render_ids_rt.program = rt_setup.link_program(module, {"render_ids_ray_gen"});
        m_render_ids_rt.pipeline = rt_setup.create_pipeline({
            .program = m_render_ids_rt.program,
            .max_recursion = 1,
            .max_ray_payload_size = 128,
        });
        m_render_ids_rt.shader_table = rt_setup.create_shader_table(m_render_ids_rt.program, {"render_ids_ray_gen"});
    } else {
        m_render_ids_cs.kernel = m_device->create_compute_kernel({
            .program = m_device->link_program({module}, {module->entry_point("render_ids_compute")}),
        });
    }
}

void ScenePicker::render(sgl::CommandEncoder* command_encoder, const Scene* scene, const Camera* camera)
{
    uint32_t width = camera->width();
    uint32_t height = camera->height();

    if (!m_geometry_instance_id_texture || m_geometry_instance_id_texture->width() != width
        || m_geometry_instance_id_texture->height() != height) {
        m_geometry_instance_id_texture = m_device->create_texture({
            .format = sgl::Format::r32_uint,
            .width = width,
            .height = height,
            .usage = sgl::TextureUsage::shader_resource | sgl::TextureUsage::unordered_access,
            .label = "geometry_instance_id_texture",
        });
    }

    create_pipelines(scene);

    if (m_use_raytracing_pipeline) {
        ref<sgl::RayTracingPassEncoder> pass_encoder = command_encoder->begin_ray_tracing_pass();
        sgl::ShaderObject* shader_object
            = pass_encoder->bind_pipeline(m_render_ids_rt.pipeline, m_render_ids_rt.shader_table);
        sgl::ShaderCursor cursor = sgl::ShaderCursor(shader_object);
        scene->bind(cursor);
        cursor = cursor["g_render_ids_params"];
        camera->bind(cursor["camera"]);
        cursor["output"] = m_geometry_instance_id_texture;
        pass_encoder->dispatch_rays(0, {width, height, 1});
        pass_encoder->end();
    } else {
        m_render_ids_cs.kernel->dispatch(
            uint3(width, height, 1),
            [&](sgl::ShaderCursor cursor)
            {
                scene->bind(cursor);
                cursor = cursor.find_entry_point(0);
                camera->bind(cursor["camera"]);
                cursor["output"] = m_geometry_instance_id_texture;
            },
            command_encoder
        );
    }
}

shared::GeometryInstanceID ScenePicker::pick(uint2 position, sgl::Texture* geometry_instance_id_texture)
{
    if (!geometry_instance_id_texture)
        geometry_instance_id_texture = m_geometry_instance_id_texture;
    if (!geometry_instance_id_texture)
        return shared::GeometryInstanceID::invalid;

    auto command_encoder = m_device->create_command_encoder();
    command_encoder->copy_texture_to_buffer(
        m_readback_buffer,
        0,
        4,
        256,
        geometry_instance_id_texture,
        0,
        0,
        uint3(position.x, position.y, 0),
        uint3(1, 1, 1)
    );
    m_device->submit_command_buffer(command_encoder->finish());

    shared::GeometryInstanceID id = shared::GeometryInstanceID::invalid;
    m_readback_buffer->get_data(&id, sizeof(id));
    return id;
}

Entity* ScenePicker::pick_entity(Scene* scene, uint2 position, sgl::Texture* geometry_instance_id_texture)
{
    shared::GeometryInstanceID id = pick(position, geometry_instance_id_texture);
    if (id == shared::GeometryInstanceID::invalid)
        return nullptr;
    return find_entity_by_geometry_instance_id(scene, id);
}

Entity* ScenePicker::find_entity_by_geometry_instance_id(Scene* scene, shared::GeometryInstanceID geometry_instance_id)
{
    // Iterate through all GeometryInstance components in the scene to find the one that owns the given geometry
    // instance ID.
    uint32_t geometry_instance_index = static_cast<uint32_t>(geometry_instance_id);
    for (size_t i = 0; i < scene->components().size(); ++i) {
        if (auto geometry_instance = scene->components()[i]->as<GeometryInstance>()) {
            uint32_t first_index = static_cast<uint32_t>(geometry_instance->geometry_instance_id());
            uint32_t count = geometry_instance->geometry_instance_count();
            if (geometry_instance_index >= first_index && geometry_instance_index < first_index + count) {
                return geometry_instance->entity();
            }
        }
    }
    return nullptr;
}

} // namespace falcor::ui
