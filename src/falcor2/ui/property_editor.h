// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/core/fwd.h"

#include "falcor2/core/macros.h"
#include "falcor2/core/reflection/class_descriptor.h"
#include "falcor2/core/reflection/dynamic_properties.h"

#include <cstdint>
#include <list>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace falcor::ui {

// ----------------------------------------------------------------------------
// PropertyLayoutNode / PropertyLayout
// ----------------------------------------------------------------------------

/// A node in the grouped property layout tree.
struct PropertyLayoutNode {
    /// Group name (empty for the root node).
    std::string name;
    /// Properties in this group, in registration order.
    std::vector<const reflection::PropertyDescriptor*> properties;
    /// Nested subgroups, ordered by first appearance.
    std::vector<PropertyLayoutNode> children;
};

/// Cached tree structure representing the grouped, ordered property list
/// for a given class + dynamic property set combination.
struct PropertyLayout {
    /// Top-level node; root.properties = ungrouped properties.
    PropertyLayoutNode root;
    /// ClassDescriptor used to build this layout.
    const reflection::ClassDescriptor* class_desc{nullptr};
    /// Pointer to the DynamicPropertySet used to build this layout (nullptr if none).
    const reflection::DynamicPropertySet* dynamic_set{nullptr};
    /// Generation of DynamicPropertySet when layout was built (0 if no dynamic props).
    uint64_t dynamic_generation{0};
};

// ----------------------------------------------------------------------------
// PropertyLayoutCache
// ----------------------------------------------------------------------------

/// Bounded LRU cache of PropertyLayout objects, keyed by instance pointer.
class FALCOR_API PropertyLayoutCache {
public:
    static constexpr size_t DEFAULT_CAPACITY = 32;

    /// Constructor.
    explicit PropertyLayoutCache(size_t capacity = DEFAULT_CAPACITY);

    /// Look up a layout for the given instance. Returns nullptr if not cached.
    PropertyLayout* find(const void* instance);

    /// Insert or replace a layout for the given instance, evicting LRU if at capacity.
    PropertyLayout& insert(const void* instance, PropertyLayout layout);

    /// Remove all cached entries.
    void clear();

private:
    struct Entry {
        const void* key;
        PropertyLayout layout;
    };

    size_t m_capacity;
    /// LRU list: most recently used at front.
    std::list<Entry> m_entries;
    /// Map from instance pointer to list iterator for O(1) lookup.
    std::unordered_map<const void*, std::list<Entry>::iterator> m_index;
};

// ----------------------------------------------------------------------------
// PropertyEditorContext
// ----------------------------------------------------------------------------

/// Context passed through to property editor functions.
/// Provides options and caches layout information across frames.
struct PropertyEditorContext {
    PropertyEditorContext() = default;

    /// Whether to show Advanced-tagged properties.
    bool show_advanced{false};
    /// Whether to show read-only properties.
    bool show_read_only{true};
    /// Cached layout data, persists across frames.
    PropertyLayoutCache layout_cache;
};

// ----------------------------------------------------------------------------
// Layout building
// ----------------------------------------------------------------------------

/// Build a PropertyLayout from a class descriptor and optional dynamic property set.
FALCOR_API PropertyLayout build_property_layout(
    const reflection::ClassDescriptor& class_desc,
    const reflection::DynamicPropertySet* dynamic_set = nullptr
);

/// Build a PropertyLayout from a flat span of property descriptors.
/// This overload is source-agnostic: it works for Python reflected properties,
/// dynamic property sets, or any other collection of descriptors.
FALCOR_API PropertyLayout build_property_layout(std::span<const reflection::PropertyDescriptor* const> descriptors);

// ----------------------------------------------------------------------------
// Property editor functions
// ----------------------------------------------------------------------------

/// Editor for a single property. Returns true if value changed.
FALCOR_API bool property_editor(const reflection::PropertyDescriptor& desc, void* instance, PropertyEditorContext& ctx);

/// Property editor for all properties of a class (including base classes and dynamic properties).
/// Returns true if any value changed.
FALCOR_API bool properties_editor(ReflectedObject& object, PropertyEditorContext& ctx);

/// Property editor for a flat span of property descriptors.
/// The cache_key is used to cache (and invalidate) the layout.
/// The generation counter detects when the layout needs rebuilding.
/// Returns true if any value changed.
FALCOR_API bool properties_editor(
    std::span<const reflection::PropertyDescriptor* const> descriptors,
    void* instance,
    const void* cache_key,
    uint64_t generation,
    PropertyEditorContext& ctx
);

/// Property editor for a Properties objects.
/// Returns true if any value changed.
FALCOR_API bool properties_editor(Properties& properties);

} // namespace falcor::ui
