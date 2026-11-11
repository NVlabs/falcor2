// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/render/fwd.h"
#include "falcor2/render/source_asset_ref.h"

#include "falcor2/core/fwd.h"
#include "falcor2/core/object.h"
#include "falcor2/core/error.h"
#include "falcor2/core/enum.h"
#include "falcor2/core/reflected_object.h"
#include "falcor2/core/properties.h"
#include "falcor2/core/reflection.h"

#include <atomic>
#include <vector>
#include <span>
#include <initializer_list>
#include <functional>
#include <memory>
#include <type_traits>

namespace falcor {

/// Type used for scene object collection indices.
using SceneObjectCollectionIndex = uint32_t;
/// Invalid scene object collection index.
static constexpr SceneObjectCollectionIndex INVALID_SCENE_OBJECT_COLLECTION_INDEX = 0xffffffff;

/// Kind of scene object.
enum class SceneObjectKind {
    material,
    geometry,
    animation,
    entity,
    component,
};
SGL_ENUM_INFO(
    SceneObjectKind,
    {
        {SceneObjectKind::material, "material"},
        {SceneObjectKind::geometry, "geometry"},
        {SceneObjectKind::animation, "animation"},
        {SceneObjectKind::entity, "entity"},
        {SceneObjectKind::component, "component"},
    }
);
SGL_ENUM_REGISTER(SceneObjectKind);

/// Base class for all scene objects.
class FALCOR_API SceneObject : public ReflectedObject {
    FALCOR_OBJECT(SceneObject)
public:
    /// Dirty flags shared by all scene objects.
    enum class DirtyFlags : uint32_t {
        none = 0,
        resources = (1u << 29),
        added = (1u << 30),
        removed = (1u << 31),
    };

    /// Constructor.
    SceneObject();

    /// Destructor.
    virtual ~SceneObject();

    /// The kind of this scene object.
    virtual SceneObjectKind kind() const = 0;

    /// Check if this object is of type T.
    template<typename T>
    bool is() const
    {
        return dynamic_cast<const T*>(this) != nullptr;
    }

    /// Cast this object to type T.
    template<typename T>
    T* as()
    {
        return dynamic_cast<T*>(this);
    }

    /// Cast this object to type T.
    template<typename T>
    T* as() const
    {
        return dynamic_cast<T*>(this);
    }

    /// The scene this scene object belongs to.
    Scene* scene() const { return m_scene; }

    /// The index in the collection this object belongs to.
    SceneObjectCollectionIndex collection_index() const { return m_collection_index; }

    /// The name of the scene object.
    const std::string& name() const { return m_name; }
    void set_name(std::string_view name);

    /// Whether this is a built-in object managed by the scene.
    /// Built-in objects cannot be removed or renamed.
    bool is_builtin() const;

    /// Reference to the object in the source asset from which this object was imported.
    /// Returns nullptr for objects without provenance.
    const SourceAssetRef* source_asset_ref() const { return m_source_asset_ref.get(); }

    /// Destroy this object.
    /// This does not immediately delete the object, but has the following effects:
    /// - Marks the object as removed
    /// - Object will be deleted when the scene is next updated
    virtual void remove();

    /// Check if the object is valid (not removed).
    bool is_valid() const { return !m_removed; }

    /// Called during Scene::update() when other objects have been removed.
    /// This allows to clear invalid references in this object.
    virtual void clear_invalid_references() { }

    /// Called when the object is created.
    /// This is called in the context of Scene::create_object().
    /// Both the scene and collection index of the object are valid at this point.
    /// The default implementation calls ReflectedObject::set_properties() to set properties on the object.
    virtual void on_create(std::optional<const Properties> props);

    /// Called when the object is destroyed.
    /// This is called during Scene::update() after the object has been marked as removed.
    /// After this call, the scene and collection index of the object are invalid and should not be accessed.
    virtual void on_destroy();

    /// Called during Scene::update() when object is added to the scene.
    virtual void on_add_to_scene() { }
    /// Called during Scene::update() when object is removed from the scene.
    virtual void on_remove_from_scene() { }

    /// Called during Scene::update() to let the object load resources.
    /// Note: This is called for all objects with the resources dirty flag set, as well as all newly added objects.
    virtual void on_load_resources() { }
    /// Called during Scene::update() to let the object update resources.
    // virtual void update_resources(SceneUpdateContext& ctx) override;

public:
    virtual void _create(std::optional<const Properties> props);
    virtual void _destroy();

    /// Called when an object is added to the scene to set a default name.
    void _set_default_name();

    /// Set importer provenance. Passing nullopt clears the reference.
    void _set_source_asset_ref(std::optional<SourceAssetRef> source_asset_ref);

protected:
    /// Scene this object belongs to.
    Scene* m_scene{nullptr};
    /// Index in the scene object collection.
    SceneObjectCollectionIndex m_collection_index{INVALID_SCENE_OBJECT_COLLECTION_INDEX};
    /// Name of the object.
    std::string m_name;
    /// Allocate provenance only for objects that have a source asset reference.
    std::unique_ptr<SourceAssetRef> m_source_asset_ref;

    bool m_removed{false};

private:
    friend class Scene;
    template<typename>
    friend class SceneObjectCollection;

    FALCOR_NON_COPYABLE_AND_MOVABLE(SceneObject);
};

FALCOR_ENUM_CLASS_OPERATORS(SceneObject::DirtyFlags);

/// Forward declaration of factory class.
template<typename TObject>
class SceneObjectFactory;

/// Forward declaration of get function - defined in scene.cpp.
/// This is used to ensure a single instance of each factory across shared library boundaries.
template<typename TObject>
FALCOR_API SceneObjectFactory<TObject>& get_scene_object_factory();

/// Factory class for creating all scene objects of a specific base type.
template<typename TObject>
class SceneObjectFactory {
public:
    static_assert(std::is_base_of_v<SceneObject, TObject>, "TObject must be derived from SceneObject");

    using CreateFn = std::function<ref<TObject>()>;

    struct ClassInfo {
        std::string name;
        const std::type_info& type;
        CreateFn create_fn{nullptr};
    };

    /// Constructor.
    SceneObjectFactory() = default;

    /// Get the singleton instance of the factory.
    /// Delegates to get_scene_object_factory() which is explicitly instantiated in scene.cpp
    /// to ensure a single instance across shared library boundaries.
    static SceneObjectFactory& get() { return get_scene_object_factory<TObject>(); }

    /// List of registered class infos.
    const std::vector<ClassInfo>& class_infos() const { return m_class_infos; }

    /// Register a class.
    /// @tparam T The class type to register.
    /// @param name The name of the class.
    /// @param create_fn Function to create an instance of the class.
    template<typename T>
    void register_class(std::string_view name, CreateFn create_fn)
    {
        static_assert(std::is_base_of_v<TObject, T>, "T must be derived from TObject");
        m_class_infos.push_back({
            .name = std::string(name),
            .type = typeid(T),
            .create_fn = std::move(create_fn),
        });
    }

    /// Find class info by name.
    /// @param name The name of the class.
    /// @return The class info or nullptr if not found.
    const ClassInfo* find_class_info(std::string_view name) const
    {
        for (const auto& info : m_class_infos)
            if (name == info.name)
                return &info;
        return nullptr;
    }

    /// Find class info by type.
    /// @param type The type info of the class.
    /// @return The class info or nullptr if not found.
    const ClassInfo* find_class_info(const std::type_info& type) const
    {
        for (const auto& info : m_class_infos)
            if (type == info.type)
                return &info;
        return nullptr;
    }

    /// Get class info by name or throw if not found.
    /// @param name The name of the class.
    /// @return The class info.
    const ClassInfo& class_info(std::string_view name) const
    {
        const ClassInfo* info = find_class_info(name);
        if (info)
            return *info;
        FALCOR_THROW("Unknown class \"{}\"", name);
    }

    /// Get class info by type or throw if not found.
    /// @param type The type info of the class.
    /// @return The class info.
    const ClassInfo& class_info(const std::type_info& type) const
    {
        const ClassInfo* info = find_class_info(type);
        if (info)
            return *info;
        FALCOR_THROW("Unknown class \"{}\"", type.name());
    }

    /// Create an instance of a class.
    /// @param info The class info.
    /// @return The created instance.
    ref<TObject> create(const ClassInfo& info) const { return info.create_fn(); }

    /// Create an instance of a class by name.
    /// @param name The name of the class.
    /// @return The created instance.
    ref<TObject> create(std::string_view name) const { return create(class_info(name)); }

    /// Create an instance of a class by type.
    /// @param type The type info of the class.
    /// @return The created instance.
    ref<TObject> create(const std::type_info& type) const { return create(class_info(type)); }

private:
    std::vector<ClassInfo> m_class_infos;

    FALCOR_NON_COPYABLE_AND_MOVABLE(SceneObjectFactory);
};

/// Collection of scene objects of a specific type.
/// Objects are owned by the collection.
/// Keeps a list of dirty flags as well as combined dirty flags (or'd) for the entire collection.
/// This allows to quickly check for changes.
template<typename TObject>
class SceneObjectCollection {
public:
    static_assert(std::is_base_of_v<SceneObject, TObject>, "TObject must be derived from SceneObject");

    using ObjectType = TObject;
    using DirtyFlags = TObject::DirtyFlags;

    static_assert(sizeof(DirtyFlags) == sizeof(uint32_t));

    // Make sure that object class reuses the same dirty flags for none/added/removed flags.
    // clang-format off
    static_assert(static_cast<uint32_t>(DirtyFlags::none) == static_cast<uint32_t>(SceneObject::DirtyFlags::none));
    static_assert(static_cast<uint32_t>(DirtyFlags::added) == static_cast<uint32_t>(SceneObject::DirtyFlags::added));
    static_assert(static_cast<uint32_t>(DirtyFlags::removed) == static_cast<uint32_t>(SceneObject::DirtyFlags::removed));
    // clang-format on

    /// Constructor.
    /// @param scene Scene the collection belongs to.
    SceneObjectCollection(Scene* scene)
        : m_scene(scene)
    {
    }

    /// Destructor.
    ~SceneObjectCollection() = default;

    /// The size of the collection.
    size_t size() const { return m_objects.size(); }

    /// Check if the collection is empty.
    bool empty() const { return m_objects.empty(); }

    /// List of objects in the collection.
    const std::vector<ref<TObject>>& objects() const { return m_objects; }
    /// List of dirty flags for each object in the collection.
    const std::vector<DirtyFlags>& dirty_flags() const { return m_dirty_flags; }

    /// Access object by index (const).
    const TObject* operator[](size_t index) const { return m_objects[index]; }
    /// Access object by index (non-const).
    TObject* operator[](size_t index) { return m_objects[index]; }

    /// The combined dirty flags (or'd flags of all objects since last call to reset_dirty_flags()).
    DirtyFlags combined_dirty_flags() const { return static_cast<DirtyFlags>(m_combined_dirty_flags.load()); }

    /// Return true if any object is marked dirty.
    bool is_dirty() const { return combined_dirty_flags() != DirtyFlags::none; }

    /// Return true if any object in the collection is marked with any of the supplied flags.
    bool has_dirty(DirtyFlags flags) const { return is_set(combined_dirty_flags(), flags); }

    /// Clear the collection.
    void clear()
    {
        m_objects.clear();
        m_dirty_flags.clear();
        m_combined_dirty_flags.store(static_cast<uint32_t>(DirtyFlags::none));
    }

    /// Reset all dirty flags.
    void reset_dirty_flags()
    {
        if (is_dirty()) {
            std::fill(m_dirty_flags.begin(), m_dirty_flags.end(), DirtyFlags::none);
            m_combined_dirty_flags.store(static_cast<uint32_t>(DirtyFlags::none));
        }
    }

    /// Add an object to the collection.
    /// @param object The object to add.
    void add(ref<TObject> object)
    {
        FALCOR_ASSERT(object);
        FALCOR_ASSERT(object->m_scene == nullptr);
        FALCOR_ASSERT(object->m_collection_index == INVALID_SCENE_OBJECT_COLLECTION_INDEX);

        size_t index = m_objects.size();
        resize(index + 1);
        m_objects[index] = object;
        object->m_scene = m_scene;
        object->m_collection_index = SceneObjectCollectionIndex(index);
        mark_dirty(object, DirtyFlags::added);
    }

    /// Mark an object for removal.
    /// @param object The object to mark for removal.
    void mark_remove(TObject* object) { mark_dirty(object, DirtyFlags::removed); }

    /// Mark an object as dirty with the specified flags.
    /// @param object The object to mark as dirty.
    /// @param flags The dirty flags to set.
    void mark_dirty(TObject* object, DirtyFlags flags)
    {
        FALCOR_ASSERT(object);
        FALCOR_ASSERT(object->m_scene == m_scene);
        FALCOR_ASSERT(object->m_collection_index != INVALID_SCENE_OBJECT_COLLECTION_INDEX);

        m_dirty_flags[object->m_collection_index] |= flags;
        m_combined_dirty_flags.fetch_or(static_cast<uint32_t>(flags));
    }

    template<typename Func>
    void for_each(Func func, DirtyFlags flags = ~DirtyFlags::none) const
    {
        if (!has_dirty(flags))
            return;
        for (size_t i = 0; i < m_objects.size(); ++i) {
            DirtyFlags object_flags = m_dirty_flags[i];
            if (is_set(object_flags, flags)) {
                func(m_objects[i].get(), object_flags);
            }
        }
    }

    template<typename MemberFunc>
    void for_each_dirty(MemberFunc func, DirtyFlags flags)
    {
        for_each(
            [func](TObject* object, DirtyFlags /*flags*/)
            {
                (object->*func)();
            },
            flags
        );
    }

    /// Remove objects marked for removal and compact the collection.
    /// This retains the order of the objects in the collection.
    void remove_marked_and_compact()
    {
        // Early out if empty or no objects are marked for removal.
        if (empty() || !has_dirty(DirtyFlags::removed))
            return;
        size_t target_index = 0;
        for (size_t source_index = 0; source_index < m_objects.size(); ++source_index) {
            // Teardown must also run for consecutive and trailing removals when
            // callers retain references after this collection releases ownership.
            if (is_set(m_dirty_flags[source_index], DirtyFlags::removed)) {
                m_objects[source_index]->_destroy();
                continue;
            }
            if (source_index != target_index) {
                m_objects[target_index] = m_objects[source_index];
                m_objects[target_index]->m_collection_index = SceneObjectCollectionIndex(target_index);
                m_dirty_flags[target_index] = m_dirty_flags[source_index];
            }
            ++target_index;
        }
        resize(target_index);
    }

private:
    void resize(size_t new_size)
    {
        m_objects.resize(new_size);
        m_dirty_flags.resize(new_size);
    }

    Scene* m_scene{nullptr};
    std::vector<ref<TObject>> m_objects;
    std::vector<DirtyFlags> m_dirty_flags;
    std::atomic<uint32_t> m_combined_dirty_flags{static_cast<uint32_t>(DirtyFlags::none)};

    FALCOR_NON_COPYABLE_AND_MOVABLE(SceneObjectCollection);
};

/// View of a SceneObjectCollection.
/// Raw iteration, size, and indexed access include built-in objects and preserve collection indices.
/// find(), find_all(), and iter() exclude built-in objects unless include_builtins is true.
/// Do not change collection membership while iterating. Ranges and iterators borrow this view.
template<typename TObject>
class SceneObjectCollectionView {
public:
    using Collection = SceneObjectCollection<TObject>;

    SceneObjectCollectionView(Collection& collection)
        : m_collection{collection}
    {
    }

    /// The size of the collection.
    size_t size() const { return m_collection.size(); }

    /// Access object by index (const).
    const TObject* operator[](size_t index) const { return m_collection[index]; }
    /// Access object by index (non-const).
    TObject* operator[](size_t index) { return m_collection[index]; }

    /// Iterator for iterating over objects in the collection view.
    template<bool IsConst>
    class IteratorImpl {
    public:
        using ViewType = std::conditional_t<IsConst, const SceneObjectCollectionView, SceneObjectCollectionView>;
        using ValueType = std::conditional_t<IsConst, const TObject*, TObject*>;

        IteratorImpl(ViewType& view, size_t index, bool include_builtins = true)
            : m_view(view)
            , m_index(index)
            , m_include_builtins(include_builtins)
        {
            skip_builtins();
        }

        ValueType operator*() const { return m_view[m_index]; }

        IteratorImpl& operator++()
        {
            ++m_index;
            skip_builtins();
            return *this;
        }

        bool operator==(const IteratorImpl& other) const { return m_index == other.m_index; }
        bool operator!=(const IteratorImpl& other) const { return m_index != other.m_index; }

    private:
        void skip_builtins()
        {
            if (!m_include_builtins)
                while (m_index < m_view.size() && m_view[m_index]->is_builtin())
                    ++m_index;
        }

        ViewType& m_view;
        size_t m_index;
        bool m_include_builtins;
    };

    /// Iterator type for iterating over objects in the collection view.
    using Iterator = IteratorImpl<false>;
    /// Const iterator type for iterating over objects in the collection view.
    using ConstIterator = IteratorImpl<true>;

    /// Get an iterator to the beginning of the collection view.
    Iterator begin() { return Iterator(*this, 0); }
    /// Get an iterator to the end of the collection view.
    Iterator end() { return Iterator(*this, size()); }

    /// Get a const iterator to the beginning of the collection view.
    ConstIterator begin() const { return ConstIterator(*this, 0); }
    /// Get a const iterator to the end of the collection view.
    ConstIterator end() const { return ConstIterator(*this, size()); }

    template<bool IsConst>
    class RangeImpl {
    public:
        using ViewType = typename IteratorImpl<IsConst>::ViewType;

        RangeImpl(ViewType& view, bool include_builtins)
            : m_view(view)
            , m_include_builtins(include_builtins)
        {
        }

        IteratorImpl<IsConst> begin() const { return {m_view, 0, m_include_builtins}; }
        IteratorImpl<IsConst> end() const { return {m_view, m_view.size(), m_include_builtins}; }

    private:
        ViewType& m_view;
        bool m_include_builtins;
    };

    /// Iterate without allocating a result list, excluding built-in objects by default.
    /// Set include_builtins to true to include them. The view must outlive the range and its iterators.
    RangeImpl<false> iter(bool include_builtins = false) { return {*this, include_builtins}; }
    RangeImpl<true> iter(bool include_builtins = false) const { return {*this, include_builtins}; }

    /// Find the first object with the given name.
    /// @param name The name to search for.
    /// @param include_builtins Include built-in objects; excluded by default.
    /// @return Pointer to the first matching object, or nullptr if not found.
    TObject* find(std::string_view name, bool include_builtins = false) const
    {
        for (size_t i = 0; i < size(); ++i) {
            if (!include_builtins && m_collection[i]->is_builtin())
                continue;
            if (m_collection[i]->name() == name)
                return m_collection[i];
        }
        return nullptr;
    }

    /// Find the first object of the given type.
    /// @tparam T The type to search for (must be derived from TObject).
    /// @param include_builtins Include built-in objects; excluded by default.
    /// @return Pointer to the first matching object, or nullptr if not found.
    template<typename T>
    T* find(bool include_builtins = false) const
    {
        static_assert(std::is_base_of_v<TObject, T>, "T must be derived from TObject");
        for (size_t i = 0; i < size(); ++i) {
            if (!include_builtins && m_collection[i]->is_builtin())
                continue;
            if (T* obj = dynamic_cast<T*>(m_collection[i]))
                return obj;
        }
        return nullptr;
    }

    /// Find the first object of the given type with the given name.
    /// @tparam T The type to search for (must be derived from TObject).
    /// @param name The name to search for.
    /// @param include_builtins Include built-in objects; excluded by default.
    /// @return Pointer to the first matching object, or nullptr if not found.
    template<typename T>
    T* find(std::string_view name, bool include_builtins = false) const
    {
        static_assert(std::is_base_of_v<TObject, T>, "T must be derived from TObject");
        for (size_t i = 0; i < size(); ++i) {
            if (!include_builtins && m_collection[i]->is_builtin())
                continue;
            if (m_collection[i]->name() == name)
                if (T* obj = dynamic_cast<T*>(m_collection[i]))
                    return obj;
        }
        return nullptr;
    }

    // Route C strings to name lookup instead of pointer-to-bool conversion.
    template<typename T>
    T* find(const char* name, bool include_builtins = false) const
    {
        return find<T>(std::string_view(name), include_builtins);
    }

    /// Find all objects with the given name.
    /// @param name The name to search for.
    /// @param include_builtins Include built-in objects; excluded by default.
    /// @return Vector of pointers to all matching objects.
    std::vector<TObject*> find_all(std::string_view name, bool include_builtins = false) const
    {
        std::vector<TObject*> result;
        for (size_t i = 0; i < size(); ++i) {
            if (!include_builtins && m_collection[i]->is_builtin())
                continue;
            if (m_collection[i]->name() == name)
                result.push_back(m_collection[i]);
        }
        return result;
    }

    /// Find all objects of the given type.
    /// @tparam T The type to search for (must be derived from TObject).
    /// @param include_builtins Include built-in objects; excluded by default.
    /// @return Vector of pointers to all matching objects.
    template<typename T>
    std::vector<T*> find_all(bool include_builtins = false) const
    {
        static_assert(std::is_base_of_v<TObject, T>, "T must be derived from TObject");
        std::vector<T*> result;
        for (size_t i = 0; i < size(); ++i) {
            if (!include_builtins && m_collection[i]->is_builtin())
                continue;
            if (T* obj = dynamic_cast<T*>(m_collection[i]))
                result.push_back(obj);
        }
        return result;
    }

    /// Find all objects of the given type with the given name.
    /// @tparam T The type to search for (must be derived from TObject).
    /// @param name The name to search for.
    /// @param include_builtins Include built-in objects; excluded by default.
    /// @return Vector of pointers to all matching objects.
    template<typename T>
    std::vector<T*> find_all(std::string_view name, bool include_builtins = false) const
    {
        static_assert(std::is_base_of_v<TObject, T>, "T must be derived from TObject");
        std::vector<T*> result;
        for (size_t i = 0; i < size(); ++i) {
            if (!include_builtins && m_collection[i]->is_builtin())
                continue;
            if (m_collection[i]->name() == name)
                if (T* obj = dynamic_cast<T*>(m_collection[i]))
                    result.push_back(obj);
        }
        return result;
    }

    // Route C strings to name lookup instead of pointer-to-bool conversion.
    template<typename T>
    std::vector<T*> find_all(const char* name, bool include_builtins = false) const
    {
        return find_all<T>(std::string_view(name), include_builtins);
    }

private:
    SceneObjectCollection<TObject>& m_collection;
};

/// Macro to declare a scene object class.
/// This macro should be placed at the top of the class declaration.
/// @param name The class name.
/// @param base The reflected base class.
#define FALCOR_SCENE_OBJECT(name, base) FALCOR_REFLECTED_OBJECT(name, base)

/// Macro to register a scene object class.
/// @param name The class name.
/// @param base The base class name.
#define FALCOR_SCENE_REGISTER_CLASS(name, base)                                                                        \
    FALCOR_STATIC_ONCE(reflection::register_type<name>());                                                             \
    FALCOR_STATIC_ONCE({                                                                                               \
        SceneObjectFactory<base>::get().register_class<name>(                                                          \
            #name,                                                                                                     \
            []() -> ref<base>                                                                                          \
            {                                                                                                          \
                return ref<base>(new name());                                                                          \
            }                                                                                                          \
        );                                                                                                             \
    })

/// Helper function to erase invalid scene objects objects from a vector.
/// @tparam TObject Type of scene object.
/// @param objects Vector of pointers to scene objects.
template<typename TObject>
    requires std::is_base_of_v<SceneObject, TObject>
void erase_invalid_objects(std::vector<TObject*>& objects)
{
    objects.erase(
        std::remove_if(
            objects.begin(),
            objects.end(),
            [&](TObject* object)
            {
                return !object->is_valid();
            }
        ),
        objects.end()
    );
}

} // namespace falcor
