// SPDX-License-Identifier: Apache-2.0

#include "falcor2/ui/selection_overlay.h"

#include "falcor2/core/error.h"
#include "falcor2/render/scene.h"
#include "falcor2/render/entity.h"
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

SelectionOverlay::SelectionOverlay(ref<sgl::Device> device, std::optional<Options> options)
    : m_device(std::move(device))
    , m_options(options.value_or(Options{}))
{
    ref<sgl::ShaderProgram> program
        = m_device->load_program("falcor2/ui/kernels/selection_overlay.slang", {"selection_overlay"});
    m_kernel = m_device->create_compute_kernel({.program = program});

    m_use_raytracing_pipeline = (m_device->type() == sgl::DeviceType::cuda);
}

void SelectionOverlay::set_options(const Options& options)
{
    m_options = options;
}

void SelectionOverlay::set_selected_entities(std::span<Entity*> entities)
{
    selection_bitmap_clear();

    m_selected_entities.clear();

    for (Entity* entity : entities) {
        if (!entity)
            continue;
        m_selected_entities.push_back(entity);
        for (Component* component : entity->components()) {
            if (GeometryInstance* geometry_instance = component->as<GeometryInstance>()) {
                uint32_t first = static_cast<uint32_t>(geometry_instance->geometry_instance_id());
                for (uint32_t i = 0; i < geometry_instance->geometry_instance_count(); ++i) {
                    selection_bitmap_set(first + i);
                }
            }
        }
    }

    update_aabb();
}

void SelectionOverlay::set_selected_entity(Entity* entity)
{
    set_selected_entities({&entity, 1});
}

void SelectionOverlay::set_selected_geometry_instance_ids(std::span<const shared::GeometryInstanceID> ids)
{
    selection_bitmap_clear();
    m_aabb_valid = false;
    m_selected_entities.clear();

    for (shared::GeometryInstanceID id : ids)
        selection_bitmap_set(static_cast<uint32_t>(id));
}

void SelectionOverlay::clear_selection()
{
    selection_bitmap_clear();
    m_selected_entities.clear();
}

void SelectionOverlay::draw_overlay(
    sgl::CommandEncoder* command_encoder,
    sgl::Texture* output_texture,
    sgl::Texture* geometry_instance_id_texture,
    const Scene* scene,
    const Camera* camera
)
{
    FALCOR_CHECK_NOT_NULL(command_encoder);
    FALCOR_CHECK_NOT_NULL(output_texture);
    FALCOR_CHECK_NOT_NULL(geometry_instance_id_texture);
    FALCOR_CHECK_NOT_NULL(scene);
    FALCOR_CHECK_NOT_NULL(camera);

    if (m_selection_bitmap.empty())
        return;

    // Recompute AABB each frame to track entity movement.
    update_aabb();

    uint32_t width = output_texture->width();
    uint32_t height = output_texture->height();

    if (!m_selection_bitmap_buffer
        || m_selection_bitmap.size() * sizeof(uint32_t) > m_selection_bitmap_buffer->size()) {
        m_selection_bitmap_buffer = m_device->create_buffer({
            .size = m_selection_bitmap.size() * sizeof(uint32_t),
            .usage = sgl::BufferUsage::shader_resource,
        });
    }

    if (m_selection_bitmap_dirty) {
        command_encoder->upload_buffer_data(
            m_selection_bitmap_buffer,
            0,
            m_selection_bitmap.size() * sizeof(uint32_t),
            m_selection_bitmap.data()
        );
        m_selection_bitmap_dirty = false;
    }

    // Create/resize selected hit texture.
    if (!m_selected_hit_texture || m_selected_hit_texture->width() != width
        || m_selected_hit_texture->height() != height) {
        m_selected_hit_texture = m_device->create_texture({
            .format = sgl::Format::r32_uint,
            .width = width,
            .height = height,
            .usage = sgl::TextureUsage::shader_resource | sgl::TextureUsage::unordered_access,
            .label = "selected_hit_texture",
        });
    }

    // Dispatch probe kernel to detect selected objects behind occluders.
    if (m_options.show_occluded) {
        create_probe_kernel(scene);
    }
    uint32_t bitmap_bit_count = static_cast<uint32_t>(m_selection_bitmap.size() * 32);

    // Prepare AABB uniform data.
    float3 aabb_min = m_aabb_valid ? m_selection_aabb.min : float3(0.f);
    float3 aabb_max = m_aabb_valid ? m_selection_aabb.max : float3(0.f);

    if (m_options.show_occluded) {
        if (m_use_raytracing_pipeline) {
            ref<sgl::RayTracingPassEncoder> pass_encoder = command_encoder->begin_ray_tracing_pass();
            sgl::ShaderObject* shader_object
                = pass_encoder->bind_pipeline(m_probe_rt.pipeline, m_probe_rt.shader_table);
            sgl::ShaderCursor cursor = sgl::ShaderCursor(shader_object);
            scene->bind(cursor);
            cursor = cursor["g_selection_probe_params"];
            camera->bind(cursor["camera"]);
            cursor["selection_bitmap"] = m_selection_bitmap_buffer;
            cursor["selection_bitmap_bit_count"] = bitmap_bit_count;
            cursor["selected_hit_texture"] = m_selected_hit_texture;
            cursor["aabb_min"] = aabb_min;
            cursor["aabb_max"] = aabb_max;
            cursor["aabb_enabled"] = m_aabb_valid;
            pass_encoder->dispatch_rays(0, {width, height, 1});
            pass_encoder->end();
        } else {
            m_probe_kernel->dispatch(
                uint3(width, height, 1),
                [&](sgl::ShaderCursor cursor)
                {
                    scene->bind(cursor);
                    cursor = cursor.find_entry_point(0);
                    camera->bind(cursor["camera"]);
                    cursor["selection_bitmap"] = m_selection_bitmap_buffer;
                    cursor["selection_bitmap_bit_count"] = bitmap_bit_count;
                    cursor["selected_hit_texture"] = m_selected_hit_texture;
                    cursor["aabb_min"] = aabb_min;
                    cursor["aabb_max"] = aabb_max;
                    cursor["aabb_enabled"] = m_aabb_valid;
                },
                command_encoder
            );
        }
    } else {
        // Clear selected hit texture when probe is skipped.
        command_encoder->clear_texture_uint(m_selected_hit_texture, {}, uint4(0));
    }

    // Dispatch overlay kernel to draw outlines.
    m_kernel->dispatch(
        uint3(width, height, 1),
        [&](sgl::ShaderCursor cursor)
        {
            cursor = cursor.find_entry_point(0);
            cursor["output_texture"] = ref(output_texture);
            cursor["geometry_instance_id_texture"] = ref(geometry_instance_id_texture);
            cursor["selected_hit_texture"] = m_selected_hit_texture;
            cursor["selection_bitmap"] = m_selection_bitmap_buffer;
            cursor["selection_bitmap_bit_count"] = bitmap_bit_count;
            cursor["selection_color"] = m_options.selection_color;
            cursor["show_occluded"] = m_options.show_occluded;
            cursor["fill_opacity"] = m_options.fill_opacity;
        },
        command_encoder
    );
}

void SelectionOverlay::create_probe_kernel(const Scene* scene)
{
    FALCOR_ASSERT(scene);

    if (m_use_raytracing_pipeline) {
        if (m_probe_rt.pipeline && m_requirements_generation == scene->requirements_generation())
            return;
    } else {
        if (m_probe_kernel && m_scene_requirements == scene->requirements())
            return;
    }

    m_requirements_generation = scene->requirements_generation();
    m_scene_requirements = scene->requirements();

    const SceneRequirements& requirements = scene->requirements();

    std::vector<ref<sgl::SlangModule>> modules;
    modules.push_back(m_device->load_module("falcor2/ui/kernels/selection_probe.slang"));
    modules.insert(modules.end(), requirements.modules.begin(), requirements.modules.end());
    ref<sgl::SlangModule> module
        = m_device->compose_modules("selection_probe", modules, requirements.type_conformances);

    if (m_use_raytracing_pipeline) {
        SceneRayTracingSetup::RayDesc probe_ray_type;
        probe_ray_type.name = "probe";
        probe_ray_type.has_miss = true;
        probe_ray_type.has_any_hit = true;
        SceneRayTracingSetup rt_setup = SceneRayTracingSetup::create(scene, {probe_ray_type});

        m_probe_rt.program = rt_setup.link_program(module, {"selection_probe_ray_gen"});
        m_probe_rt.pipeline = rt_setup.create_pipeline({
            .program = m_probe_rt.program,
            .max_recursion = 1,
            .max_ray_payload_size = 128,
        });
        m_probe_rt.shader_table = rt_setup.create_shader_table(m_probe_rt.program, {"selection_probe_ray_gen"});
    } else {
        m_probe_kernel = m_device->create_compute_kernel({
            .program = m_device->link_program({module}, {module->entry_point("selection_probe")}),
        });
    }
}

void SelectionOverlay::update_aabb()
{
    m_selection_aabb = AABB();
    m_aabb_valid = false;

    if (m_selected_entities.size() > SELECTION_AABB_CULLING_MAX_OBJECTS)
        return;

    bool valid = true;
    for (Entity* entity : m_selected_entities) {
        AABB entity_aabb = entity->world_aabb();
        if (entity_aabb.is_valid()) {
            m_selection_aabb.expand(entity_aabb);
        } else {
            valid = false;
            break;
        }
    }

    m_aabb_valid = valid && m_selection_aabb.is_valid();
}

} // namespace falcor::ui
