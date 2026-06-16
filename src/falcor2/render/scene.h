// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "scene_requirements.h"

#include "falcor2/render/fwd.h"
#include "falcor2/render/scene_object.h"
#include "falcor2/render/scene_update.h"
#include "falcor2/render/material.h"
#include "falcor2/render/geometry.h"
#include "falcor2/render/animation.h"
#include "falcor2/render/entity.h"
#include "falcor2/render/component.h"
#include "falcor2/render/scene_import.h"
#include "falcor2/render/hit_group_policy.h"
#include "falcor2/render/render_scene.h"

#include "falcor2/core/fwd.h"
#include "falcor2/core/object.h"
#include "falcor2/core/properties.h"
#include "falcor2/core/signal.h"

#include <sgl/device/fwd.h>
#include <sgl/device/shader.h>

#include <optional>
#include <span>


namespace falcor {

// Prevent implicit instantiation in other translation units.
// Explicit instantiations are provided in scene.cpp.
extern template FALCOR_API SceneObjectFactory<Geometry>& get_scene_object_factory<Geometry>();
extern template FALCOR_API SceneObjectFactory<Material>& get_scene_object_factory<Material>();
extern template FALCOR_API SceneObjectFactory<Animation>& get_scene_object_factory<Animation>();
extern template FALCOR_API SceneObjectFactory<Entity>& get_scene_object_factory<Entity>();
extern template FALCOR_API SceneObjectFactory<Component>& get_scene_object_factory<Component>();

/// Flags to determine which parts of the scene to bind to a shader.
enum class SceneBindFlags {
    none = 0,
    materials = (1 << 0),
    lights = (1 << 1),
    emissive_geometry = (1 << 2),
    render_scene = (1 << 3),
    all = materials | lights | emissive_geometry | render_scene,
};
FALCOR_ENUM_CLASS_OPERATORS(SceneBindFlags);
SGL_ENUM_FLAGS_INFO(
    SceneBindFlags,
    {
        {SceneBindFlags::none, "none"},
        {SceneBindFlags::materials, "materials"},
        {SceneBindFlags::lights, "lights"},
        {SceneBindFlags::emissive_geometry, "emissive_geometry"},
        {SceneBindFlags::render_scene, "render_scene"},
        {SceneBindFlags::all, "all"},
    }
);
SGL_ENUM_REGISTER(SceneBindFlags);

using SceneUpdatedSignal = Signal<void(SceneUpdateFlags)>;

/// Scene container.
///
/// The Scene is the top-level container for all renderable content and serves as the central
/// hub for managing scene objects and their lifecycles. It coordinates between the high-level
/// scene graph representation and the low-level device resources needed for rendering.
///
/// Scene objects are organized into four main categories:
/// - Materials (Material): Define surface appearance and shading properties.
/// - Geometries (Geometry): Define geometry data (meshes, implicits, etc.).
/// - Entities (Entity): Scene graph nodes with transforms that form a hierarchy.
/// - Components (Component): Attached to entities to add functionality (e.g., GeometryInstance, Camera, Light).
///
/// The scene uses a deferred update model. When objects are created, modified, or removed,
/// changes are tracked via dirty flags. The update() method must be called to process these
/// changes, update device resources, and prepare the scene for rendering.
///
/// The scene can be loaded from various file formats (glTF, USD, etc.) via importers,
/// or constructed programmatically using the create_* methods.
class FALCOR_API Scene : public Object {
    FALCOR_OBJECT(Scene)
public:
    /// Constructor.
    /// @param device The device to use for rendering.
    Scene(ref<sgl::Device> device);

    /// Constructor.
    /// @param device The device to use for rendering.
    /// @param importer_scene The importer scene to create the scene from.
    Scene(ref<sgl::Device> device, ref<ImporterScene> importer_scene);

    /// Constructor.
    /// @param device The device to use for rendering.
    /// @param path Path to the scene file to load.
    /// @param recompute_normals If true, recompute normals for all meshes.
    Scene(ref<sgl::Device> device, const std::filesystem::path& path, bool recompute_normals = false);

    /// Destructor.
    virtual ~Scene() override;

    /// Create an empty scene.
    /// @param device The device to use for rendering.
    /// @return The created scene.
    static ref<Scene> create(ref<sgl::Device> device);

    /// Create a scene from an importer scene.
    /// @param device The device to use for rendering.
    /// @param importer_scene The importer scene to create the scene from.
    /// @return The created scene.
    static ref<Scene> create(ref<sgl::Device> device, ref<ImporterScene> importer_scene);

    /// Create a scene by loading from a file.
    /// @param device The device to use for rendering.
    /// @param path Path to the scene file to load.
    /// @param recompute_normals If true, recompute normals for all meshes.
    /// @return The created scene.
    static ref<Scene>
    create(ref<sgl::Device> device, const std::filesystem::path& path, bool recompute_normals = false);

    /// The devce device used for creating resources and executing rendering commands.
    sgl::Device* device() const { return m_device; }

    /// The Slang module containing the scene's rendering shader code (falcor2.render).
    sgl::SlangModule* render_module() const { return m_render_module.get(); }

    /// The texture manager responsible for loading, caching, and managing scene textures.
    TextureManager* texture_manager() const { return m_texture_manager.get(); }

    /// Add scene globals to the scene.
    /// Scene globals are shader globals that are set on the scene and automatically bound when rendering.
    /// @param globals The scene globals to add.
    void add_scene_globals(ref<SceneGlobals> globals);

    /// Remove scene globals from the scene.
    /// @param globals The scene globals to remove.
    void remove_scene_globals(SceneGlobals* globals);

    /// List of all scene globals currently added to the scene.
    const std::vector<ref<SceneGlobals>>& scene_globals() const { return m_scene_globals; }

    /// View providing indexed access to all materials in the scene.
    MaterialCollectionView& materials() { return m_materials; }
    /// View providing indexed access to all geometries in the scene.
    GeometryCollectionView& geometries() { return m_geometries; }
    /// View providing indexed access to all animations in the scene.
    AnimationCollectionView& animations() { return m_animations; }
    /// View providing indexed access to all entities in the scene.
    EntityCollectionView& entities() { return m_entities; }
    /// View providing indexed access to all components in the scene.
    ComponentCollectionView& components() { return m_components; }

    /// The underlying material collection with dirty flag tracking (const).
    const MaterialCollection& material_collection() const { return m_material_collection; }
    /// The underlying material collection with dirty flag tracking (non-const).
    MaterialCollection& material_collection() { return m_material_collection; }
    /// The underlying geometry collection with dirty flag tracking (const).
    const GeometryCollection& geometry_collection() const { return m_geometry_collection; }
    /// The underlying geometry collection with dirty flag tracking (non-const).
    GeometryCollection& geometry_collection() { return m_geometry_collection; }
    /// The underlying animation collection with dirty flag tracking (const).
    const AnimationCollection& animation_collection() const { return m_animation_collection; }
    /// The underlying animation collection with dirty flag tracking (non-const).
    AnimationCollection& animation_collection() { return m_animation_collection; }
    /// The underlying entity collection with dirty flag tracking (const).
    const EntityCollection& entity_collection() const { return m_entity_collection; }
    /// The underlying entity collection with dirty flag tracking (non-const).
    EntityCollection& entity_collection() { return m_entity_collection; }
    /// The underlying component collection with dirty flag tracking (const).
    const ComponentCollection& component_collection() const { return m_component_collection; }
    /// The underlying component collection with dirty flag tracking (non-const).
    ComponentCollection& component_collection() { return m_component_collection; }

    /// The active camera used for rendering.
    /// This is automatically set if a Camera component is added and no active camera is set,
    /// and automatically cleared if the active camera is removed.
    Camera* active_camera() const;
    void set_active_camera(Camera* camera);

    /// Create a material of a specific type and add it to the scene.
    /// The material is owned by the scene and will be marked with DirtyFlags::added.
    /// @tparam T The material type to create (must derive from Material).
    /// @param props Optional properties to initialize the material with.
    /// @return Pointer to the created material (owned by the scene).
    template<typename T>
        requires std::is_base_of_v<Material, T>
    T* create_material(std::optional<Properties> props = {})
    {
        return _create_object<T>(props);
    }

    /// Create a material by registered type name (e.g., "StandardMaterial").
    /// @param type The registered name of the material type to create.
    /// @param props Optional properties to initialize the material with.
    /// @return Pointer to the created material (owned by the scene).
    Material* create_material(std::string_view type, std::optional<Properties> props = {});

    /// Create a material by C++ type info.
    /// @param type The type info of the material type to create.
    /// @param props Optional properties to initialize the material with.
    /// @return Pointer to the created material (owned by the scene).
    Material* create_material(const std::type_info& type, std::optional<Properties> props = {});

    /// Create a geometry of a specific type and add it to the scene.
    /// The geometry is owned by the scene and will be marked with DirtyFlags::added.
    /// @tparam T The geometry type to create (e.g., StaticMeshGeometry, GeometryGroup).
    /// @param props Optional properties to initialize the geometry with.
    /// @return Pointer to the created geometry (owned by the scene).
    template<typename T>
    T* create_geometry(std::optional<Properties> props = {})
    {
        return _create_object<T>(props);
    }

    /// Create a geometry by registered type name (e.g., "StaticMeshGeometry").
    /// @param type The registered name of the geometry type to create.
    /// @param props Optional properties to initialize the geometry with.
    /// @return Pointer to the created geometry (owned by the scene).
    Geometry* create_geometry(std::string_view type, std::optional<Properties> props = {});

    /// Create a geometry by C++ type info.
    /// @param type The type info of the geometry type to create.
    /// @param props Optional properties to initialize the geometry with.
    /// @return Pointer to the created geometry (owned by the scene).
    Geometry* create_geometry(const std::type_info& type, std::optional<Properties> props = {});

    /// Create an animation and add it to the scene.
    /// @return Pointer to the created animation (owned by the scene).
    Animation* create_animation();

    /// Create a new entity and add it to the scene.
    /// Entities are scene graph nodes that hold transforms and components.
    /// Use Entity::set_parent() to build a hierarchy after creation.
    /// @param props Optional properties to initialize the entity with.
    /// @return Pointer to the created entity (owned by the scene).
    Entity* create_entity(std::optional<Properties> props = {});

    /// The current scene animation time.
    double time() const { return m_time; }

    /// Set the current scene animation time.
    /// This will only take effect on the next update() call.
    void set_time(double time);

    /// Get the maximum duration across all animations (0 if no animations).
    double animation_duration() const;

    /// Returns true if this scene has any animation data.
    bool has_animation() const;

    /// Update the scene, processing all pending changes.
    /// This must be called before rendering to ensure device resources are up to date.
    /// Internally creates a command encoder and submits it to the device.
    /// The update processes: added/removed objects, dirty materials, lights, geometries,
    /// entity transforms, and component render states.
    /// @return Flags indicating what changed during the update.
    SceneUpdateFlags update();

    /// TODO(scene) make private?
    /// Called by Scene::update() to do the actual update.
    /// @param ctx The update context (command encoder, etc.).
    /// @return Flags indicating what changed during the update.
    SceneUpdateFlags _update(SceneUpdateContext& ctx);

    /// Update the scene from animations.
    /// @param ctx The update context (command encoder, etc.).
    /// @return Flags indicating what changed during the update.
    SceneUpdateFlags _update_animations(SceneUpdateContext& ctx);

    /// Flags indicating what changed during the last update() call.
    SceneUpdateFlags update_flags() const { return m_update_flags; }

    /// A monotonically increasing generation counter that is incremented on each update with dirty changes.
    /// This can be used to track if the scene has changed since a certain point in time.
    uint64_t update_generation() const { return m_update_generation; }

    /// Signal emitted when the scene is updated. The signal parameter indicates what changed.
    SceneUpdatedSignal& updated() { return m_updated_signal; }

    /// Get the modules and conformances required to make the scene fully work.
    const SceneRequirements& requirements() const;

    /// A monotonically increasing generation counter that is incremented when scene requirements change.
    /// This can be used to track if the scene requirements have changed since a certain point in time.
    uint64_t requirements_generation() const { return m_requirements_generation; }

    /// Hit group policy used when building TLAS instance metadata.
    const HitGroupPolicy& hit_group_policy() const { return m_hit_group_policy; }

    /// Returns true if the scene currently has geometry of the given type.
    bool has_geometry_type(shared::GeometryType type) const;

    /// Bind scene data to shader globals for rendering.
    /// @param globals The shader cursor pointing to the global parameters.
    /// @param flags Flags indicating which parts of the scene to bind (materials, lights, etc.).
    void bind(sgl::ShaderCursor globals, SceneBindFlags flags = SceneBindFlags::all) const;

public:
    /// Get the low-level render scene that manages geometry data and acceleration structures.
    RenderScene* _render_scene() { return m_render_scene; }
    const RenderScene* _render_scene() const { return m_render_scene; }

    /// Add an existing object to the scene and register it with the appropriate collection.
    /// @tparam T The type of object to add.
    /// @param object The object to add.
    template<typename T>
    void _add_object(ref<T> object)
    {
        if constexpr (std::is_base_of_v<Material, T>)
            m_material_collection.add(object);
        if constexpr (std::is_base_of_v<Geometry, T>)
            m_geometry_collection.add(object);
        if constexpr (std::is_base_of_v<Animation, T>)
            m_animation_collection.add(object);
        if constexpr (std::is_base_of_v<Entity, T>)
            m_entity_collection.add(object);
        if constexpr (std::is_base_of_v<Component, T>)
            m_component_collection.add(object);
        object->_set_default_name();
    }

    /// Create an object of a specific type.
    /// @tparam T The type of object to create.
    /// @param props Optional properties to initialize the object with.
    /// @return Pointer to the created object.
    template<typename T>
    T* _create_object(std::optional<Properties> props = {})
    {
        static_assert(std::is_base_of_v<SceneObject, T>, "T must be derived from SceneObject");
        ref<T> object = ref<T>(new T());
        _add_object<T>(object);
        object->_create(props);
        return object;
    }

    /// Create an object by type name.
    /// @tparam T The base type of object to create.
    /// @param type The name of the type to create.
    /// @param props Optional properties to initialize the object with.
    /// @return Pointer to the created object.
    template<typename T>
    T* _create_object(std::string_view type, std::optional<Properties> props = {})
    {
        static_assert(std::is_base_of_v<SceneObject, T>, "T must be derived from SceneObject");
        ref<T> object = SceneObjectFactory<T>::get().create(type);
        _add_object<T>(object);
        object->_create(props);
        return object;
    }

    /// Create an object by type info.
    /// @tparam T The base type of object to create.
    /// @param type The type info of the type to create.
    /// @param props Optional properties to initialize the object with.
    /// @return Pointer to the created object.
    template<typename T>
    T* _create_object(const std::type_info& type, std::optional<Properties> props = {})
    {
        static_assert(std::is_base_of_v<SceneObject, T>, "T must be derived from SceneObject");
        ref<T> object = SceneObjectFactory<T>::get().create(type);
        _add_object<T>(object);
        object->_create(props);
        return object;
    }

    /// Mark a material as removed. Called by Material::remove().
    void _mark_material_removed(Material* material);
    /// Mark a material as dirty. Called by Material::mark_dirty().
    void _mark_material_dirty(Material* material, Material::DirtyFlags flags);
    /// Mark a geometry as removed. Called by Geometry::remove().
    void _mark_geometry_removed(Geometry* geometry);
    /// Mark a geometry as dirty. Called by Geometry::mark_dirty().
    void _mark_geometry_dirty(Geometry* geometry, Geometry::DirtyFlags flags);
    /// Mark an entity as removed. Called by Entity::remove().
    void _mark_entity_removed(Entity* entity);
    /// Mark an entity as dirty. Called by Entity::mark_dirty().
    void _mark_entity_dirty(Entity* entity, Entity::DirtyFlags flags);
    /// Mark a component as removed. Called by Component::remove().
    void _mark_component_removed(Component* component);
    /// Mark a component as dirty. Called by Component::mark_dirty().
    void _mark_component_dirty(Component* component, Component::DirtyFlags flags);

    /// Called by Scene::update() to allow scene objects to clear invalid references.
    /// Calls SceneObject::clear_invalid_references() on each all scene objects to
    /// allow clearing references to now removed scene objects.
    /// This is a no-op if no objects have been removed since last call to Scene::update().
    void _handle_removed_objects();

private:
    ref<sgl::Device> m_device;

    /// Collection of all materials.
    MaterialCollection m_material_collection;
    /// View of materials collection.
    MaterialCollectionView m_materials;
    /// Collection of all geometries.
    GeometryCollection m_geometry_collection;
    /// View of geometries collection.
    GeometryCollectionView m_geometries;
    /// Collection of all animations.
    AnimationCollection m_animation_collection;
    /// View of animations collection.
    AnimationCollectionView m_animations;
    /// Collection of all entities.
    EntityCollection m_entity_collection;
    /// View of entities collection.
    EntityCollectionView m_entities;
    /// Collection of all components.
    ComponentCollection m_component_collection;
    /// View of components collection.
    ComponentCollectionView m_components;

    /// The active camera for rendering. Held as ref to prevent premature deletion.
    ref<Camera> m_active_camera;

    /// Texture manager for loading and caching textures.
    ref<TextureManager> m_texture_manager;

    /// List of scene globals to bind to the shader when rendering the scene.
    std::vector<ref<SceneGlobals>> m_scene_globals;

    /// Material system for managing materials.
    ref<MaterialSystem> m_material_system;
    /// Light system for managing lights.
    ref<LightSystem> m_light_system;
    /// Emissive geometry system for managing emissive geometry.
    ref<EmissiveGeometrySystem> m_emissive_geometry_system;

    /// Slang module containing rendering shader code.
    ref<sgl::SlangModule> m_render_module;
    /// Shader object for binding scene data.
    ref<sgl::ShaderObject> m_scene_shader_object;

    /// Internal render scene representation.
    ref<RenderScene> m_render_scene;

    /// Hit group policy used for scene ray tracing setup.
    /// This is currently hard-coded but will eventually be configurable.
    HitGroupPolicy m_hit_group_policy{
        .mode = HitGroupPolicy::Mode::per_geometry_type,
        .ray_type_count = 2,
        .geometry_type_count = 2,
    };

    /// Current animation time.
    double m_time{0.0};

    /// The animation time for which the scene was last updated.
    /// Used to track if animation time has changed since last update.
    double m_update_time{0.0};

    /// Flags indicating what changed in the last update.
    SceneUpdateFlags m_update_flags{SceneUpdateFlags::none};

    /// A monotonically increasing generation counter that is incremented on each update with dirty changes.
    uint64_t m_update_generation{0};

    /// Signal emitted when the scene is updated. The signal parameter indicates what changed.
    SceneUpdatedSignal m_updated_signal;

    /// Cached requirements for scene to work.
    SceneRequirements m_requirements;

    /// A monotonically increasing generation counter for requirements changes.
    uint64_t m_requirements_generation{0};

    FALCOR_NON_COPYABLE_AND_MOVABLE(Scene);
};

/// Create a component of a specific type, attach it to this entity, and add it to the scene.
/// Components add functionality to entities (e.g., GeometryInstance for rendering geometry,
/// Camera for camera behavior). The component is owned by the scene.
/// @tparam T The component type to create (must derive from Component).
/// @param props Optional properties to initialize the component with.
/// @return Pointer to the created component (owned by the scene, attached to this entity).
template<typename T>
    requires std::is_base_of_v<Component, T>
T* Entity::create_component(std::optional<Properties> props)
{
    T* component = m_scene->_create_object<T>(props);
    component->m_entity = this;
    m_components.push_back(component);
    return component;
}

} // namespace falcor
