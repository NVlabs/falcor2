// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "nanobind.h"
#include "core/reflection/python_property_descriptor.h"

#include "falcor2/core/macros.h"
#include "falcor2/core/reflection/property_descriptor.h"

#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace falcor::reflection {

/// Adapter that reads a Python class's ``_reflected_properties`` list and
/// presents the entries as a span of C++ ``PropertyDescriptor`` pointers,
/// compatible with the native property-editor pipeline.
///
/// Descriptors are cached per Python *type* identity so that multiple
/// ``PythonClassReflection`` instances for the same class share the same
/// underlying descriptor objects.  The cache is invalidated when the identity
/// of any element in ``_reflected_properties`` changes (e.g. dynamic schema).
///
/// A monotonically-increasing *generation* counter is bumped on every rebuild
/// so that consumers (UI layout caches, etc.) can detect structural changes.
///
/// NOTE: This class assumes single-threaded access from the Python context
/// (GIL held).  No internal locking is performed.
class PythonClassReflection {
public:
    /// Construct from a Python class (type object) or an instance.
    /// If an instance is passed, its ``type()`` is used.
    explicit PythonClassReflection(nb::object py_class);

    /// Ensure the descriptors are up-to-date with ``_reflected_properties``.
    /// Checks if the property list has changed (via fingerprinting) and rebuilds
    /// only if necessary.  Safe to call even if nothing changed.
    void ensure_up_to_date();

    /// Force a full rebuild of descriptors from ``_reflected_properties``.
    /// WARNING: Invalidates all previously returned descriptor pointers from
    /// this instance (callers that hold raw pointers should re-query).
    void rebuild();

    /// The descriptors as a span of base-class pointers.
    std::span<const PropertyDescriptor* const> descriptors() const;

    /// Find a descriptor by property name.  Returns nullptr if not found.
    const PropertyDescriptor* find_property(std::string_view name) const;

    /// The Python class name (``__name__``).
    std::string class_name() const;

    /// Number of descriptors.
    size_t size() const;

    /// Access a descriptor by index.
    const PropertyDescriptor* operator[](size_t i) const;

    /// Generation counter, incremented on every rebuild.
    /// Can be used by callers to invalidate layout caches.
    uint64_t generation() const;

    /// Clear the global per-class cache.
    /// Primarily useful for testing and interpreter shutdown cleanup.
    static void clear_cache();

    /// Find or create a cached PythonClassReflection for the given instance's type.
    /// Used by the convenience ``python_properties_editor`` overload.
    static PythonClassReflection& find_or_create(nb::handle instance);

private:
    /// Shared descriptor data cached per Python type.
    struct CachedDescriptors {
        std::vector<std::unique_ptr<PythonPropertyDescriptor>> descriptors;
        std::vector<const PropertyDescriptor*> descriptor_ptrs;
        std::unordered_map<std::string_view, const PropertyDescriptor*> name_map;
        nb::object py_class_ref;        ///< Prevent GC of the Python type while cached.
        uint64_t source_fingerprint{0}; ///< Identity fingerprint of _reflected_properties when built.
        uint64_t generation{0};
    };

    /// Compute an identity-based fingerprint of a ``_reflected_properties`` list.
    /// Combines the list length with the ``PyObject*`` identity of each element.
    static uint64_t compute_fingerprint(nb::handle props_list);

    /// Build (or rebuild) the CachedDescriptors from the Python class.
    static std::shared_ptr<CachedDescriptors>
    build_descriptors(nb::handle py_class, std::shared_ptr<CachedDescriptors> existing);

    /// The Python class (type object).  Kept alive to prevent GC.
    nb::object m_py_class;

    /// Shared, cached descriptor data for this class.
    std::shared_ptr<CachedDescriptors> m_cached;

    /// Global cache: Python type object pointer -> cached descriptors.
    /// Keys are raw PyObject* but the corresponding CachedDescriptors always
    /// holds a strong reference via py_class_ref, preventing GC.
    static std::unordered_map<PyObject*, std::shared_ptr<CachedDescriptors>> s_cache;
    /// Global cache: Python type object pointer -> PythonClassReflection (convenience overload).
    static std::unordered_map<PyObject*, std::unique_ptr<PythonClassReflection>> s_reflections;

    FALCOR_NON_COPYABLE_AND_MOVABLE(PythonClassReflection);
};

} // namespace falcor::reflection
