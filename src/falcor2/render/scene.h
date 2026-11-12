// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/render/fwd.h"
#include "falcor2/render/scene_requirements.h"
#include "falcor2/render/scene_capabilities.h"
#include "falcor2/render/scene_render_request.h"
#include "falcor2/render/scene_object.h"
#include "falcor2/render/scene_update.h"
#include "falcor2/render/material.h"
#include "falcor2/render/material_aov_registry.h"
#include "falcor2/render/scene_globals.h"
#include "falcor2/render/geometry.h"
#include "falcor2/render/animation.h"
#include "falcor2/render/entity.h"
#include "falcor2/render/component.h"
#include "falcor2/render/scene_config.h"
#include "falcor2/render/ray_tracing_policy.h"
#include "falcor2/render/render_scene.h"

#include "falcor2/core/fwd.h"
#include "falcor2/core/object.h"
#include "falcor2/core/properties.h"
#include "falcor2/core/signal.h"
#include "falcor2/importers/fwd.h"
#include "falcor2/importers/importer.h"
#include "falcor2/importers/importer_types.h"

#include <sgl/device/fwd.h>
#include <sgl/device/shader.h>

#include <pxr/pxr.h>

#include <algorithm>
#include <functional>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <typeinfo>
#include <utility>

PXR_NAMESPACE_OPEN_SCOPE
class UsdStage;
PXR_NAMESPACE_CLOSE_SCOPE

namespace falcor {

namespace mtlx {
class CodeGenCache;
}

enum class MeshAttributeSemantic;

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
    static constexpr uint32_t NAMED_ATTRIBUTE_ID_BEGIN = 16;
    static constexpr uint32_t NAMED_ATTRIBUTE_ID_END = 32;

    /// Destructor.
    virtual ~Scene() override;

    /// Create an empty scene.
    /// @param device The device to use for rendering.
    /// @param config Immutable live-scene configuration.
    /// @return The created scene.
    static ref<Scene> create(ref<sgl::Device> device, const SceneConfig& config = {});

    /// Load a scene from a file.
    /// @param device The device to use for rendering.
    /// @param path Path to the scene file to load.
    /// @param import_options Optional import behavior. If unset, uses default import options.
    /// @param config Optional live-scene configuration. If unset, uses the imported scene's preferred UV origin.
    /// @return The loaded scene.
    static ref<Scene> load(
        ref<sgl::Device> device,
        const std::filesystem::path& path,
        std::optional<ImportOptions> import_options = {},
        std::optional<SceneConfig> config = {}
    );

    /// Create a scene from a caller-owned USD stage without modifying it.
    /// @param device The device to use for rendering.
    /// @param stage USD stage to import.
    /// @param import_options Optional import behavior. If unset, uses default import options.
    /// @param config Optional live-scene configuration. If unset, uses the imported scene's preferred UV origin.
    /// @return The created scene.
    static ref<Scene> from_stage(
        ref<sgl::Device> device,
        const pxr::UsdStage& stage,
        std::optional<ImportOptions> import_options = {},
        std::optional<SceneConfig> config = {}
    );

    /// Create a scene from imported CPU-side scene data.
    /// @param device The device to use for rendering.
    /// @param importer_scene Imported scene data to load.
    /// @param config Optional live-scene configuration. If unset, uses importer_scene.uv_origin.
    /// @return The created scene.
    static ref<Scene> from_importer_scene(
        ref<sgl::Device> device,
        const ImporterScene& importer_scene,
        std::optional<SceneConfig> config = {}
    );

    /// Create a scene from the edits recorded by an importer.
    /// @param device The device to use for rendering.
    /// @param importer The importer whose recorded edits and loaded callbacks to apply.
    /// @param config Optional live-scene configuration. If unset, uses the importer scene's preferred UV origin.
    /// @return The created scene.
    static ref<Scene>
    from_importer(ref<sgl::Device> device, const Importer& importer, std::optional<SceneConfig> config = {});

    /// Append a scene file to this live scene.
    /// @param path Path to the scene file to append.
    /// @param import_options Optional import behavior. If unset, uses default import options.
    void append(const std::filesystem::path& path, std::optional<ImportOptions> import_options = {});

    /// Append imported CPU-side scene data to this live scene.
    void append(const ImporterScene& importer_scene);

    /// Build and append importer edits, then run loaded callbacks.
    void append(const Importer& importer);

    /// The devce device used for creating resources and executing rendering commands.
    sgl::Device* device() const { return m_device; }

    /// The Slang module containing the scene's rendering shader code (falcor2.render).
    sgl::SlangModule* render_module() const { return m_render_module.get(); }

    /// The texture manager responsible for loading, caching, and managing scene textures.
    TextureManager* texture_manager() const { return m_texture_manager.get(); }

    /// Immutable live-scene configuration.
    const SceneConfig& config() const { return m_config; }

    /// Scene-local catalog of permanent typed material AOV declarations.
    MaterialAOVRegistry* material_aov_registry() const { return m_material_aov_registry.get(); }
    mtlx::CodeGenCache* materialx_codegen_cache() const { return m_materialx_codegen_cache.get(); }

    /// Add scene globals to the scene.
    /// Scene globals are shader globals that are set on the scene and automatically bound when rendering.
    /// @param globals The scene globals to add.
    void add_scene_globals(ref<SceneGlobals> globals);

    /// Get named scene globals, or create and add them to the scene.
    /// Expected usage is for scene-owned shared globals used by materials: the material keeps the returned ref
    /// while it needs the shared globals, and Scene garbage collection reclaims the object after that ref is released.
    /// Manual globals added with add_scene_globals() are not reused or removed by this helper. The caller is
    /// responsible for using a unique name for each shared globals object.
    /// The factory must create a new SceneGlobals object, so each managed SceneGlobals* is registered under exactly
    /// one name. Managed globals must not be passed to remove_scene_globals(); a SceneGlobals object is either managed
    /// by this helper and garbage collected by Scene, or manually added/removed by the caller.
    /// @param name Unique scene globals name.
    /// @param factory Factory called only when no scene globals with \p name exist yet.
    /// @return The existing or newly created scene globals.
    ref<SceneGlobals> get_or_create_scene_globals(std::string_view name, std::function<ref<SceneGlobals>()> factory);

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
    /// The first authored camera is selected on update() or import completion unless a selection already exists.
    Camera* active_camera() const;
    /// Select a camera without changing its view. Later camera creation or imports preserve the selection.
    /// @param camera Live camera in this scene, or nullptr to select the free camera.
    void set_active_camera(Camera* camera);

    /// Built-in permanent navigation camera with editable pose and camera settings.
    Camera* free_camera() const { return m_free_camera; }

    /// Calculate exact mesh bounds by combining the bounds of all live geometry instances.
    /// Call update() first to evaluate the desired animation pose. This method does not update the scene.
    /// Empty scenes return an invalid AABB and zero radius. Unsupported geometry throws.
    /// @param transform Finite affine transform applied after each instance's world transform.
    /// @return Vertex bounds and maximum distance from the transformed coordinate system's origin.
    MeshBounds calculate_bounds(const float4x4& transform = float4x4::identity()) const;

    /// Apply a transform to each authored root entity, leaving hierarchy and descendant local transforms unchanged.
    /// Composes transform * root.transform using Transform's TRS decomposition, which removes shear.
    /// Excludes built-in entities. Later animation evaluation can replace keyed root transform components.
    /// Call update() afterwards to publish the transforms; this method does not update the scene.
    /// @param transform Finite transform to apply to the authored scene.
    void apply_transform(const Transform& transform);

    /// Fit the free camera to valid, finite geometry-instance bounds using its current lens.
    /// Updates pose, focus distance, and clipping planes; preserves camera dimensions and active-camera selection.
    /// Call after update() for current bounds; this method does not update the scene.
    /// Without usable bounds, looks at the origin from (0, 0, 5) and uses the baseline clipping range.
    /// @param aspect Width/height ratio used for fitting. Must be positive and finite; defaults to a square view.
    /// @param default_depth_range Finite near/far baseline in scene units, with 0 < near < far.
    void frame_free_camera(float aspect = 1.f, float2 default_depth_range = float2(0.01f, 1000.f));

    /// Copy the source's evaluated world pose and camera settings to the free camera, then activate it.
    /// Call after update() to capture the current animation pose.
    /// @param source Non-null, live camera in this scene.
    /// @return The free camera.
    Camera* activate_free_camera_from(const Camera* source);

    /// Find valid objects by scanning all scene collections.
    /// @param source_asset_ref Source asset reference to match exactly.
    /// @param include_builtins Include built-in objects in the results.
    std::vector<SceneObject*>
    find_objects_by_source_asset_ref(const SourceAssetRef& source_asset_ref, bool include_builtins = false) const;

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

    /// Get the imported playback interval in seconds, falling back to the union of animation key/sample intervals.
    /// New animated scenes start at this interval's beginning. Appending assets preserves the current time.
    AnimationTimeRange animation_range() const;

    /// True if a live entity has a bound animation channel with keyframes.
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

    /// Returns the capabilities required by the scene after its last completed update.
    /// Combines requirements from all valid materials. Attribute availability alone does not add requirements.
    /// Requirements are conservative. Returns an empty set before the first update.
    SceneCapabilities required_capabilities() const { return m_required_capabilities; }

    /// Shared immutable snapshot with scene-required capabilities enabled.
    /// Repeated calls reuse the snapshot cached by effective capability mask; no materials are rescanned here.
    /// Retained snapshots survive scene updates until shader or ray-tracing inputs change.
    ref<const SceneRequirements> requirements() const { return resolve_requirements(); }

    /// Resolve shader and ray-tracing requirements after the last completed update, without modifying the scene.
    /// Requests add consumer shader operations and material AOV dependencies. Call after update().
    /// SceneCapabilityPolicy::specified requires an explicit specified_capabilities value.
    /// Throws for unknown AOVs, unsupported requirements, or specified selections omitting required capabilities.
    /// Equivalent effective selections share a snapshot, independently of requested output bindings.
    ref<const SceneRequirements> resolve_requirements(
        const SceneRenderRequest& request = {},
        SceneCapabilityPolicy policy = SceneCapabilityPolicy::required,
        std::optional<SceneCapabilities> specified_capabilities = std::nullopt
    ) const;

    /// Advances when synchronized scene changes may affect requirements.
    /// Compare resolved shader inputs or full requirements before rebuilding shaders or pipelines.
    /// Consumer or selection changes require resolution independently of this counter.
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

    /// Get the shader ID for a geometry attribute without modifying the scene registry.
    std::optional<uint32_t>
    _get_geometry_attribute_id(MeshAttributeSemantic semantic, uint32_t index, std::string_view name) const;

    /// Reserve a shader ID for a named geometry attribute before its source stream is available.
    uint32_t _reserve_geometry_attribute_id(std::string_view name);

    /// Get or create the shader ID for a geometry attribute and validate its stored source signature.
    std::optional<uint32_t> _get_or_create_geometry_attribute_id(
        MeshAttributeSemantic semantic,
        uint32_t index,
        std::string_view name,
        shared::AdditionalAttributeType type,
        uint32_t component_count
    );

    // Get shared storage for material parameter data.
    MaterialSystem* _material_system() { return m_material_system; }

    // Get internal light data for renderer-owned sampling strategies and tests.
    LightSystem* _light_system() { return m_light_system; }
    const LightSystem* _light_system() const { return m_light_system; }

    // Get internal emissive geometry data so it can be tested
    EmissiveGeometrySystem* _emissive_geometry_system() { return m_emissive_geometry_system; }
    const EmissiveGeometrySystem* _emissive_geometry_system() const { return m_emissive_geometry_system; }

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
        if constexpr (std::is_base_of_v<Geometry, T> || std::is_base_of_v<Component, T>)
            m_geometry_animation_cache_dirty = true;
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
    /// Mark animation data as dirty. Called by Animation setters.
    void _mark_animation_dirty(Animation* animation, Animation::DirtyFlags flags);
    /// Mark a geometry as removed. Called by Geometry::remove().
    void _mark_geometry_removed(Geometry* geometry);
    /// Mark a geometry as dirty. Called by Geometry::mark_dirty().
    void _mark_geometry_dirty(Geometry* geometry, Geometry::DirtyFlags flags);
    /// Mark an animation as removed. Called by Animation::remove().
    void _mark_animation_removed(Animation* animation);
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
    /// The source type/component count checked for all streams sharing a named ID, independent of material reads.
    struct NamedAttributeSourceSignature {
        shared::AdditionalAttributeType type;
        uint32_t component_count;
    };

    struct NamedAttributeRegistryEntry {
        uint32_t id;
        /// Empty after name-only reservation; populated by the first geometry source.
        std::optional<NamedAttributeSourceSignature> source_signature;
    };

    /// Construct a scene with a fully resolved configuration.
    Scene(ref<sgl::Device> device, const SceneConfig& config);

    /// Select the first authored camera if initial selection is pending.
    void select_initial_camera();

    /// Garbage collect refcounted scene globals with no remaining external owners.
    void run_garbage_collect();

    /// Refresh the geometry contribution to animation queries after data or instance edits.
    void update_geometry_animation_cache() const;

    ref<sgl::Device> m_device;

    /// Resolved immutable live-scene configuration.
    const SceneConfig m_config;

    /// Collection of all materials.
    MaterialCollection m_material_collection;
    /// View of materials collection.
    MaterialCollectionView m_materials;
    /// Scene-local catalog of permanent typed material AOV declarations.
    ref<MaterialAOVRegistry> m_material_aov_registry{make_ref<MaterialAOVRegistry>()};
    std::unique_ptr<mtlx::CodeGenCache> m_materialx_codegen_cache;
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

    // Shared by animation_range() and has_animation(); playback and transforms do not invalidate it.
    AnimationTimeRange m_imported_playback_range;
    mutable AnimationTimeRange m_geometry_animation_range;
    mutable bool m_has_geometry_animation{false};
    mutable bool m_geometry_animation_cache_dirty{true};

    /// Active rendering camera. Initialized to the free camera.
    ref<Camera> m_active_camera;
    /// Built-in free camera entity and component.
    ref<Camera> m_free_camera;
    /// Indicates whether the initial camera selection is pending.
    bool m_initial_camera_selection_pending{true};

    /// Texture manager for loading and caching textures.
    ref<TextureManager> m_texture_manager;

    /// List of scene globals to bind to the shader when rendering the scene.
    std::vector<ref<SceneGlobals>> m_scene_globals;
    /// Non-owning globals managed by get_or_create_scene_globals().
    std::map<std::string, SceneGlobals*, std::less<>> m_refcounted_scene_globals;

    /// Material system for managing materials.
    ref<MaterialSystem> m_material_system;
    /// Light system for managing lights.
    ref<LightSystem> m_light_system;
    /// Emissive geometry system for managing emissive geometry.
    ref<EmissiveGeometrySystem> m_emissive_geometry_system;

    /// Slang module containing rendering shader code.
    ref<sgl::SlangModule> m_render_module;
    /// Slang settings module that specializes scene conventions.
    ref<sgl::SlangModule> m_settings_module;
    /// Caches settings modules by capability selection for reuse across consumers.
    /// Mutable so resolve_requirements() can populate it lazily without changing the all-capabilities snapshot.
    mutable std::map<SceneCapabilityFlags, ref<sgl::SlangModule>> m_specialized_settings_modules;
    /// Shader object for binding scene data.
    ref<sgl::ShaderObject> m_scene_shader_object;

    /// Internal render scene representation.
    ref<RenderScene> m_render_scene;

    /// Stable scene-wide mappings for named geometry attributes.
    std::map<std::string, NamedAttributeRegistryEntry, std::less<>> m_named_attribute_registry;

    /// Hit group policy used for scene ray tracing setup.
    /// This is currently hard-coded but will eventually be configurable.
    HitGroupPolicy m_hit_group_policy{
        .mode = HitGroupPolicy::Mode::per_geometry_type,
        .ray_type_count = 3,
        .geometry_type_count = 3,
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

    /// Internal immutable snapshot with all capabilities enabled.
    ref<const SceneRequirements> m_requirements;

    /// Snapshots cached by effective capability selection, populated lazily by resolve_requirements().
    /// Cleared when the all-capabilities snapshot changes.
    mutable std::map<SceneCapabilityFlags, ref<const SceneRequirements>> m_selected_requirements;

    /// Capabilities required by synchronized scene materials, excluding consumer demand.
    SceneCapabilities m_required_capabilities;

    /// AOV catalog revision observed by the last update(); bounds the output IDs accepted in render requests.
    uint64_t m_material_aov_output_generation{0};

    /// Globally unique revision assigned at construction and advanced when synchronized requirements inputs change.
    /// Covers snapshot, material capability, and AOV catalog changes so consumers know when to resolve again.
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
    FALCOR_CHECK(!is_builtin(), "Cannot add components to a built-in object.");
    T* component = m_scene->_create_object<T>(props);
    component->m_entity = this;
    m_components.push_back(component);
    return component;
}

} // namespace falcor
