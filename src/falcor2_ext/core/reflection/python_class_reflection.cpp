// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "python_class_reflection.h"

#include "falcor2/core/error.h"

#include <sgl/core/hash.h>

namespace falcor::reflection {

// Static members.
std::unordered_map<PyObject*, std::shared_ptr<PythonClassReflection::CachedDescriptors>> PythonClassReflection::s_cache;
std::unordered_map<PyObject*, std::unique_ptr<PythonClassReflection>> PythonClassReflection::s_reflections;

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

PythonClassReflection::PythonClassReflection(nb::object py_class)
{
    // If the user passed an instance, resolve to its type.
    if (!nb::isinstance<nb::type_object>(py_class))
        py_class = nb::borrow<nb::object>(nb::handle(reinterpret_cast<PyObject*>(Py_TYPE(py_class.ptr()))));

    m_py_class = std::move(py_class);

    auto it = s_cache.find(m_py_class.ptr());
    if (it != s_cache.end()) {
        m_cached = it->second;
        return;
    }

    m_cached = build_descriptors(m_py_class, nullptr);
    s_cache[m_py_class.ptr()] = m_cached;
}

// ---------------------------------------------------------------------------
// Cache management
// ---------------------------------------------------------------------------

void PythonClassReflection::ensure_up_to_date()
{
    nb::object props_list = m_py_class.attr("_reflected_properties");
    uint64_t current_fp = compute_fingerprint(props_list);

    if (m_cached && m_cached->source_fingerprint == current_fp)
        return; // Nothing changed.

    rebuild();
}

void PythonClassReflection::rebuild()
{
    m_cached = build_descriptors(m_py_class, m_cached);
    s_cache[m_py_class.ptr()] = m_cached;
}

void PythonClassReflection::clear_cache()
{
    s_cache.clear();
    s_reflections.clear();
}

PythonClassReflection& PythonClassReflection::find_or_create(nb::handle instance)
{
    PyObject* type_ptr = reinterpret_cast<PyObject*>(Py_TYPE(instance.ptr()));

    auto it = s_reflections.find(type_ptr);
    if (it != s_reflections.end())
        return *it->second;

    auto refl = std::make_unique<PythonClassReflection>(nb::borrow<nb::object>(nb::handle(type_ptr)));
    it = s_reflections.emplace(type_ptr, std::move(refl)).first;
    return *it->second;
}

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------

std::span<const PropertyDescriptor* const> PythonClassReflection::descriptors() const
{
    return {m_cached->descriptor_ptrs.data(), m_cached->descriptor_ptrs.size()};
}

const PropertyDescriptor* PythonClassReflection::find_property(std::string_view name) const
{
    auto it = m_cached->name_map.find(name);
    if (it != m_cached->name_map.end())
        return it->second;
    return nullptr;
}

std::string PythonClassReflection::class_name() const
{
    return nb::cast<std::string>(m_py_class.attr("__name__"));
}

size_t PythonClassReflection::size() const
{
    return m_cached->descriptor_ptrs.size();
}

const PropertyDescriptor* PythonClassReflection::operator[](size_t i) const
{
    return m_cached->descriptor_ptrs[i];
}

uint64_t PythonClassReflection::generation() const
{
    return m_cached->generation;
}

// ---------------------------------------------------------------------------
// Internal
// ---------------------------------------------------------------------------

std::shared_ptr<PythonClassReflection::CachedDescriptors>
PythonClassReflection::build_descriptors(nb::handle py_class, std::shared_ptr<CachedDescriptors> existing)
{
    auto data = std::make_shared<CachedDescriptors>();

    nb::object props_list = py_class.attr("_reflected_properties");
    data->py_class_ref = nb::borrow(py_class);
    data->source_fingerprint = compute_fingerprint(props_list);

    for (nb::handle info : props_list)
        data->descriptors.push_back(std::make_unique<PythonPropertyDescriptor>(nb::borrow(info)));

    data->descriptor_ptrs.reserve(data->descriptors.size());
    for (auto& desc : data->descriptors) {
        data->descriptor_ptrs.push_back(desc.get());
        // name() returns a string_view into the descriptor's owned string,
        // which is stable for the lifetime of the CachedDescriptors.
        data->name_map[desc->name()] = desc.get();
    }

    // Bump generation from the previous version (if any).
    data->generation = existing ? existing->generation + 1 : 0;

    return data;
}

uint64_t PythonClassReflection::compute_fingerprint(nb::handle props_list)
{
    Py_ssize_t len = nb::len(props_list);
    size_t hash = sgl::hash(len);
    for (nb::handle item : props_list) {
        // Use PyObject* identity (stable for the object's lifetime).
        uint64_t ptr_val = reinterpret_cast<uintptr_t>(item.ptr());
        hash = sgl::hash_combine(hash, ptr_val);
    }
    return hash;
}

} // namespace falcor::reflection
