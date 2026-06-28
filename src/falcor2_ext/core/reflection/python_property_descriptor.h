// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "nanobind.h"
#include "core/property_type_map.h"

#include "falcor2/core/reflection/property_descriptor.h"
#include "falcor2/core/reflection/metadata.h"

namespace falcor::reflection {

/// PropertyDescriptor subclass whose values live on the Python side.
///
/// Created from a Python ``PythonPropertyInfo`` object.  All get/set
/// operations acquire the GIL before calling into Python.
///
/// INSTANCE PARAMETER CONVENTION:
/// The ``void* instance`` parameter in the PropertyDescriptor interface is
/// reinterpreted as a ``nb::handle*`` pointing to the Python object that owns
/// the property.  This allows the type-erased C++ base-class interface to
/// work with Python objects while preserving GIL semantics.
///
/// Usage pattern (internal)::
///
///     nb::handle py_obj = ...;
///     descriptor->get_any(static_cast<const void*>(&py_obj));
///
class PythonPropertyDescriptor : public PropertyDescriptor {
public:
    /// Construct from a Python PythonPropertyInfo object.
    /// The info object must have: name, getter, setter (or None), value_type, enum_type,
    /// doc, default_value, value_range, ui_flags, ui_label, ui_group, ui_drag_speed, ui_enable_if.
    explicit PythonPropertyDescriptor(nb::object info);

    // PropertyDescriptor interface

    const std::type_info& type() const override;

    bool is_default(const void* instance) const override;
    bool has_default_value() const override;

    Any get_any(const void* instance) const override;
    void set_any(void* instance, const Any& value) const override;

    int64_t get_enum_as_int64(const void* instance) const override;
    void set_enum_from_int64(void* instance, int64_t value) const override;

    bool is_serializable_to_properties() const override;
    void write_to_properties(const void* instance, Properties& props) const override;
    bool read_from_properties(void* instance, const Properties& props) const override;

    void reset(void* instance) const override;

protected:
    void get_value(const void* instance, void* out) const override;
    void set_value(void* instance, const void* value) const override;

private:
    /// Call the Python getter on the instance, returning a Python object.
    nb::object py_get(const void* instance) const;

    /// Call the Python setter on the instance with a Python value.
    void py_set(void* instance, nb::handle value) const;

    /// The Python PythonPropertyInfo (kept alive to prevent GC of getter/setter).
    nb::object m_info;

    /// Cached Python callables.
    nb::object m_getter;
    nb::object m_setter; // may be nb::none()

    /// Default value as Python object (may be nb::none()).
    nb::object m_default_value;

    /// Type map entry for the property's C++ type, or nullptr for unsupported types.
    const PropertyTypeMapEntry* m_type_entry{nullptr};

    /// Whether this is an enum property.
    bool m_is_enum{false};

    /// Python enum type (only set if m_is_enum).
    nb::object m_enum_type;

    FALCOR_NON_COPYABLE_AND_MOVABLE(PythonPropertyDescriptor);
};

} // namespace falcor::reflection
