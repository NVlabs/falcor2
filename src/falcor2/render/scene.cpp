// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "scene.h"

#include "falcor2/render/texture_manager.h"
#include "falcor2/render/scene_globals.h"
#include "falcor2/render/scene_system.h"
#include "falcor2/render/material_system.h"
#include "falcor2/render/light_system.h"
#include "falcor2/render/emissive_geometry_system.h"
#include "falcor2/render/component/camera.h"
#include "falcor2/render/component/transform_animation.h"

#include <sgl/device/device.h>
#include <sgl/device/command.h>
#include <sgl/device/shader.h>
#include <sgl/device/shader_object.h>
#include <sgl/device/shader_cursor.h>
#include <sgl/core/timer.h>

namespace falcor {

// Definition of get_scene_object_factory() - returns singleton factory instance.
template<typename TObject>
SceneObjectFactory<TObject>& get_scene_object_factory()
{
    static SceneObjectFactory<TObject> instance;
    return instance;
}

// Explicit template instantiations for get_scene_object_factory.
// This ensures a single instance of each factory across shared library boundaries.
template FALCOR_API SceneObjectFactory<Geometry>& get_scene_object_factory<Geometry>();
template FALCOR_API SceneObjectFactory<Material>& get_scene_object_factory<Material>();
template FALCOR_API SceneObjectFactory<Animation>& get_scene_object_factory<Animation>();
template FALCOR_API SceneObjectFactory<Entity>& get_scene_object_factory<Entity>();
template FALCOR_API SceneObjectFactory<Component>& get_scene_object_factory<Component>();

static uint64_t next_generation_id()
{
    static std::atomic<uint64_t> id{0};
    return ++id;
}

static ref<sgl::SlangModule> create_settings_module(sgl::Device* device, UVOrigin uv_origin)
{
    bool is_lower_left = uv_origin == UVOrigin::lower_left;
    const char* module_name
        = is_lower_left ? "falcor2_scene_uv_origin_lower_left" : "falcor2_scene_uv_origin_upper_left";
    const char* source = is_lower_left
        ? "export static const bool FALCOR_TEXTURE_COORDINATE_ORIGIN_LOWER_LEFT = true;\n"
        : "export static const bool FALCOR_TEXTURE_COORDINATE_ORIGIN_LOWER_LEFT = false;\n";
    return device->load_module_from_source(module_name, source);
}

Scene::Scene(ref<sgl::Device> device, const SceneOptions& options)
    : m_device(std::move(device))
    , m_options(options)
    , m_material_collection(this)
    , m_materials(m_material_collection)
    , m_geometry_collection(this)
    , m_geometries(m_geometry_collection)
    , m_animation_collection(this)
    , m_animations(m_animation_collection)
    , m_entity_collection(this)
    , m_entities(m_entity_collection)
    , m_component_collection(this)
    , m_components(m_component_collection)
{
    m_render_module = m_device->load_module("falcor2.render");
    m_settings_module = create_settings_module(m_device.get(), m_options.uv_origin);
    m_requirements.modules.push_back(m_settings_module);
    m_requirements_generation = next_generation_id();

    m_texture_manager.reset(new TextureManager(m_device));

    m_material_system = ref<MaterialSystem>(new MaterialSystem(this));
    m_light_system = ref<LightSystem>(new LightSystem(this));
    m_emissive_geometry_system = ref<EmissiveGeometrySystem>(new EmissiveGeometrySystem(this));

    m_render_scene = ref<RenderScene>(new RenderScene(m_device));
}

Scene::~Scene() { }

ref<Scene> Scene::create(ref<sgl::Device> device, std::optional<UVOrigin> uv_origin)
{
    return ref<Scene>{new Scene(std::move(device), SceneOptions(uv_origin.value_or(UVOrigin::upper_left)))};
}

ref<Scene>
Scene::create(ref<sgl::Device> device, const ImporterScene& importer_scene, std::optional<UVOrigin> uv_origin)
{
    return detail::create_scene(std::move(device), importer_scene, uv_origin);
}

ref<Scene> Scene::create(
    ref<sgl::Device> device,
    const std::filesystem::path& path,
    bool recompute_normals,
    std::optional<UVOrigin> uv_origin
)
{
    return detail::create_scene(std::move(device), path, recompute_normals, uv_origin);
}

Material* Scene::create_material(std::string_view type, std::optional<Properties> props)
{
    return _create_object<Material>(type, props);
}

Material* Scene::create_material(const std::type_info& type, std::optional<Properties> props)
{
    return _create_object<Material>(type, props);
}

Geometry* Scene::create_geometry(std::string_view type, std::optional<Properties> props)
{
    return _create_object<Geometry>(type, props);
}

Geometry* Scene::create_geometry(const std::type_info& type, std::optional<Properties> props)
{
    return _create_object<Geometry>(type, props);
}

Animation* Scene::create_animation()
{
    return _create_object<Animation>();
}

Entity* Scene::create_entity(std::optional<Properties> props)
{
    return _create_object<Entity>(props);
}

void Scene::add_scene_globals(ref<SceneGlobals> globals)
{
    m_scene_globals.push_back(std::move(globals));
}

ref<SceneGlobals> Scene::get_or_create_scene_globals(std::string_view name, std::function<ref<SceneGlobals>()> factory)
{
    FALCOR_CHECK(!name.empty(), "Scene globals name must not be empty.");
    FALCOR_CHECK(factory, "Scene globals factory must be valid.");

    auto refcounted_it = m_refcounted_scene_globals.find(name);
    if (refcounted_it != m_refcounted_scene_globals.end()) {
        FALCOR_ASSERT(refcounted_it->second);
        return ref<SceneGlobals>(refcounted_it->second);
    }

    ref<SceneGlobals> globals = factory();
    FALCOR_CHECK(globals, "Scene globals factory for '{}' returned null.", name);
    m_scene_globals.push_back(globals);
    m_refcounted_scene_globals.emplace(std::string(name), globals.get());
    return globals;
}

void Scene::remove_scene_globals(SceneGlobals* globals)
{
    auto it = std::find_if(
        m_scene_globals.begin(),
        m_scene_globals.end(),
        [globals](const ref<SceneGlobals>& g)
        {
            return g.get() == globals;
        }
    );
    if (it != m_scene_globals.end())
        m_scene_globals.erase(it);
}

void Scene::run_garbage_collect()
{
    auto refcounted_it = m_refcounted_scene_globals.begin();
    while (refcounted_it != m_refcounted_scene_globals.end()) {
        SceneGlobals* globals = refcounted_it->second;
        FALCOR_ASSERT(globals);
        if (globals->ref_count() > 1) {
            ++refcounted_it;
            continue;
        }

        auto scene_globals_it = std::find_if(
            m_scene_globals.begin(),
            m_scene_globals.end(),
            [globals](const ref<SceneGlobals>& scene_globals)
            {
                return scene_globals.get() == globals;
            }
        );

        FALCOR_ASSERT(scene_globals_it != m_scene_globals.end());
        m_scene_globals.erase(scene_globals_it);
        refcounted_it = m_refcounted_scene_globals.erase(refcounted_it);
    }
}

Camera* Scene::active_camera() const
{
    if (m_active_camera && m_active_camera->is_valid())
        return m_active_camera;
    return nullptr;
}

void Scene::set_active_camera(Camera* camera)
{
    if (camera) {
        FALCOR_CHECK(camera->scene() == this, "Camera does not belong to this scene.");
        m_active_camera = ref(camera);
    } else {
        m_active_camera = nullptr;
    }
}

void Scene::set_time(double time)
{
    m_time = time;
}

double Scene::animation_duration() const
{
    double duration = 0.0;
    for (const ref<Animation>& anim : m_animation_collection.objects()) {
        if (anim->is_valid())
            duration = std::max(duration, static_cast<double>(anim->duration()));
    }
    return duration;
}

bool Scene::has_animation() const
{
    for (const ref<Animation>& anim : m_animation_collection.objects()) {
        if (anim->is_valid() && !anim->channels.empty())
            return true;
    }
    return false;
}

SceneUpdateFlags Scene::update()
{
    SceneUpdateContext ctx(device());
    SceneUpdateFlags update_flags = _update(ctx);
    ctx.submit();
    return update_flags;
}

SceneUpdateFlags Scene::_update(SceneUpdateContext& ctx)
{
    SceneUpdateFlags update_flags = SceneUpdateFlags::none;

    // sgl::Timer timer;
    _handle_removed_objects();
    // fmt::println("elasped {}", timer.elapsed_s());

#if 0
    // Allocate shader object.
    // TODO(scene): Fails to find "g_scene" on CUDA.
    if (!m_scene_shader_object) {
        auto g_scene = m_render_module->layout()->globals_type_layout()->find_field_by_name("g_scene");
        m_scene_shader_object = m_device->create_shader_object(g_scene->type_layout());
    }
#endif

    // Auto-set active camera if none is set and a camera component was added.
    if (!m_active_camera && m_component_collection.has_dirty(Component::DirtyFlags::added)) {
        for (Component* component : m_component_collection.objects()) {
            if (component->is_valid()) {
                if (Camera* camera = component->as<Camera>()) {
                    m_active_camera = ref(camera);
                    break;
                }
            }
        }
    }


    // Call SceneObject::on_add_to_scene on all added objects.
    m_material_collection.for_each_dirty(&Material::on_add_to_scene, Material::DirtyFlags::added);
    m_geometry_collection.for_each_dirty(&Geometry::on_add_to_scene, Geometry::DirtyFlags::added);
    m_animation_collection.for_each_dirty(&Animation::on_add_to_scene, Animation::DirtyFlags::added);
    m_entity_collection.for_each_dirty(&Entity::on_add_to_scene, Entity::DirtyFlags::added);
    m_component_collection.for_each_dirty(&Component::on_add_to_scene, Component::DirtyFlags::added);

    // Call SceneObject::on_remove_from_scene on all removed objects.
    m_material_collection.for_each_dirty(&Material::on_remove_from_scene, Material::DirtyFlags::removed);
    m_geometry_collection.for_each_dirty(&Geometry::on_remove_from_scene, Geometry::DirtyFlags::removed);
    m_animation_collection.for_each_dirty(&Animation::on_remove_from_scene, Animation::DirtyFlags::removed);
    m_entity_collection.for_each_dirty(&Entity::on_remove_from_scene, Entity::DirtyFlags::removed);
    m_component_collection.for_each_dirty(&Component::on_remove_from_scene, Component::DirtyFlags::removed);

    // Call SceneObject::on_load_resources on all objects with the resources or added dirty flag.
    m_material_collection.for_each_dirty(
        &Material::on_load_resources,
        Material::DirtyFlags::resources | Material::DirtyFlags::added
    );
    m_geometry_collection.for_each_dirty(
        &Geometry::on_load_resources,
        Geometry::DirtyFlags::resources | Geometry::DirtyFlags::added
    );
    m_entity_collection.for_each_dirty(
        &Entity::on_load_resources,
        Entity::DirtyFlags::resources | Entity::DirtyFlags::added
    );
    m_component_collection.for_each_dirty(
        &Component::on_load_resources,
        Component::DirtyFlags::resources | Component::DirtyFlags::added
    );

    // Update texture manager (load pending textures, garbage collect unused textures, etc).
    m_texture_manager->update();

    // Call SceneObject::update_resources on all dirty objects.

    // TODO(scene): Update texture manager here.

    // Update animations.
    update_flags |= _update_animations(ctx);

    // Update materials.
    update_flags |= m_material_system->update(ctx);

    // Update lights.
    update_flags |= m_light_system->update(ctx);

    // Update geometries.
    m_geometry_collection.for_each(
        [&update_flags](Geometry* geometry, Geometry::DirtyFlags flags)
        {
            if (is_set(flags, Geometry::DirtyFlags::render_state)) {
                geometry->update_render_state();
                update_flags |= SceneUpdateFlags::geometry | SceneUpdateFlags::render_state;
            }
        }
    );

    // Update entities.
    m_entity_collection.for_each(
        [&update_flags](Entity* entity, Entity::DirtyFlags flags)
        {
            if (is_set(flags, Entity::DirtyFlags::transform)) {
                entity->update_transform();
                update_flags |= SceneUpdateFlags::transforms;
            }
        }
    );

    // Update components.
    m_component_collection.for_each(
        [&update_flags](Component* component, Component::DirtyFlags flags)
        {
            if (is_set(flags, Component::DirtyFlags::render_state)) {
                component->update_render_state();
                update_flags |= SceneUpdateFlags::render_state;
            }
        }
    );

    // Cache required modules and type conformances, detect changes.
    // Note: This needs to happen before updating the emissive geometry system, as it relies on the
    // scene requirements to determine if shaders need to be recompiled (for sampling material emission).
    {
        SceneRequirements new_requirements;
        new_requirements.modules.push_back(m_settings_module);
        const auto& mm = m_material_system->required_modules();
        new_requirements.modules.insert(new_requirements.modules.end(), mm.begin(), mm.end());
        const auto& mc = m_material_system->required_type_conformances();
        new_requirements.type_conformances.assign(mc.begin(), mc.end());
        const auto& lc = m_light_system->required_type_conformances();
        new_requirements.type_conformances.insert(new_requirements.type_conformances.end(), lc.begin(), lc.end());

        new_requirements.ray_tracing_pipeline_flags
            = (m_render_scene->has_lss_geometry() && m_render_scene->lss_mode() == RenderLSSMode::hardware)
            ? sgl::RayTracingPipelineFlags::enable_linear_swept_spheres
            : sgl::RayTracingPipelineFlags::none;

        if (new_requirements != m_requirements) {
            m_requirements = std::move(new_requirements);
            m_requirements_generation = next_generation_id();
            update_flags |= SceneUpdateFlags::requirements;
        }
    }

    // Update render scene.
    if (m_render_scene->update(m_hit_group_policy))
        update_flags |= SceneUpdateFlags::render_state;

    // Update emissive geometry system.
    // TODO(scene): Providing info if materials updated would benefit speed of the rebuild checks.
    update_flags |= m_emissive_geometry_system->update(ctx);

#if 0
    // Bind scene systems.
    sgl::ShaderCursor cursor(m_scene_shader_object);
    m_material_system->bind_to_scene(cursor);
    // m_render_scene->bind(cursor);
#endif

    // Remove marked objects and reset dirty flags.
    m_material_collection.remove_marked_and_compact();
    m_material_collection.reset_dirty_flags();
    m_geometry_collection.remove_marked_and_compact();
    m_geometry_collection.reset_dirty_flags();
    m_animation_collection.remove_marked_and_compact();
    m_animation_collection.reset_dirty_flags();
    m_entity_collection.remove_marked_and_compact();
    m_entity_collection.reset_dirty_flags();
    m_component_collection.remove_marked_and_compact();
    m_component_collection.reset_dirty_flags();

    run_garbage_collect();

    m_update_flags = update_flags;
    if (update_flags != SceneUpdateFlags::none)
        m_update_generation = next_generation_id();

    m_updated_signal.emit(update_flags);

    return update_flags;
}

SceneUpdateFlags Scene::_update_animations(SceneUpdateContext& ctx)
{
    FALCOR_UNUSED(ctx);

    bool animations_changed = m_animation_collection.is_dirty();
    bool animation_state_changed = m_component_collection.has_dirty(Component::DirtyFlags::animation_state);
    bool time_changed = m_time != m_update_time;

    if (!animations_changed && !animation_state_changed && !time_changed)
        return SceneUpdateFlags::none;

    float t = static_cast<float>(m_time);

    for (Component* component : m_component_collection.objects()) {
        TransformAnimation* ta = component->as<TransformAnimation>();
        if (!ta || !ta->is_valid() || !ta->has_channels())
            continue;

        Animation* anim = ta->animation();
        if (!anim || !anim->is_valid() || anim->channels.empty())
            continue;

        Entity* entity = ta->entity();
        if (!entity || !entity->is_valid())
            continue;

        Transform transform = entity->transform();
        bool changed = false;

        if (const AnimationChannel* channel = anim->channel(ta->translation_channel())) {
            transform.set_translation(channel->evaluate_float3(t));
            changed = true;
        }

        if (const AnimationChannel* channel = anim->channel(ta->rotation_channel())) {
            transform.set_rotation(channel->evaluate_quatf(t));
            changed = true;
        }

        if (const AnimationChannel* channel = anim->channel(ta->scale_channel())) {
            transform.set_scale(channel->evaluate_float3(t));
            changed = true;
        }

        if (changed)
            entity->set_transform(transform);
    }

    m_update_time = m_time;

    return SceneUpdateFlags::animation;
}

const SceneRequirements& Scene::requirements() const
{
    return m_requirements;
}

void Scene::bind(sgl::ShaderCursor globals, SceneBindFlags flags) const
{
    sgl::ShaderCursor scene = globals["g_scene"];
    if (is_set(flags, SceneBindFlags::materials))
        m_material_system->bind_to_scene(scene);
    if (is_set(flags, SceneBindFlags::lights))
        m_light_system->bind_to_scene(scene);
    if (is_set(flags, SceneBindFlags::emissive_geometry))
        m_emissive_geometry_system->bind_to_scene(scene);
    if (is_set(flags, SceneBindFlags::render_scene))
        m_render_scene->bind_to_scene(scene);

    for (const ref<SceneGlobals>& sg : m_scene_globals)
        sg->bind(globals);
}

bool Scene::has_geometry_type(shared::GeometryType type) const
{
    return m_render_scene->has_geometry_type(type);
}

void Scene::_mark_material_removed(Material* material)
{
    m_material_collection.mark_remove(material);
}

void Scene::_mark_material_dirty(Material* material, Material::DirtyFlags dirty_flags)
{
    m_material_collection.mark_dirty(material, dirty_flags);
}

void Scene::_mark_geometry_removed(Geometry* geometry)
{
    m_geometry_collection.mark_remove(geometry);
}

void Scene::_mark_geometry_dirty(Geometry* geometry, Geometry::DirtyFlags flags)
{
    m_geometry_collection.mark_dirty(geometry, flags);
}

void Scene::_mark_entity_removed(Entity* entity)
{
    m_entity_collection.mark_remove(entity);
}

void Scene::_mark_entity_dirty(Entity* entity, Entity::DirtyFlags flags)
{
    m_entity_collection.mark_dirty(entity, flags);
}

void Scene::_mark_component_removed(Component* component)
{
    m_component_collection.mark_remove(component);
}

void Scene::_mark_component_dirty(Component* component, Component::DirtyFlags flags)
{
    m_component_collection.mark_dirty(component, flags);
}

void Scene::_handle_removed_objects()
{
    bool has_removed = false;
    has_removed |= m_material_collection.has_dirty(Material::DirtyFlags::removed);
    has_removed |= m_geometry_collection.has_dirty(Geometry::DirtyFlags::removed);
    has_removed |= m_animation_collection.has_dirty(Animation::DirtyFlags::removed);
    has_removed |= m_entity_collection.has_dirty(Entity::DirtyFlags::removed);
    has_removed |= m_component_collection.has_dirty(Component::DirtyFlags::removed);

    if (!has_removed)
        return;

    // Auto-replace active camera if it was removed.
    if (m_active_camera && !m_active_camera->is_valid()) {
        m_active_camera = nullptr;
        for (Component* component : m_component_collection.objects()) {
            if (component->is_valid()) {
                if (Camera* camera = component->as<Camera>()) {
                    m_active_camera = ref(camera);
                    break;
                }
            }
        }
    }

    for (Material* material : m_material_collection.objects())
        material->clear_invalid_references();
    for (Geometry* geometry : m_geometry_collection.objects())
        geometry->clear_invalid_references();
    for (Entity* entity : m_entity_collection.objects())
        entity->clear_invalid_references();
    for (Component* component : m_component_collection.objects())
        component->clear_invalid_references();
}

} // namespace falcor
