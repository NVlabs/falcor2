// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "nanobind.h"

#include "falcor2/core/types.h"
#include "falcor2/core/properties.h"

#include <string>
#include <typeinfo>
#include <vector>

namespace falcor {

/// Describes how to convert between a Python property value and its C++ counterpart.
///
/// Each entry maps a C++ type to its PropertyType, Python type object, and
/// a set of function-pointer callbacks for conversion.
/// This table is the single source of truth for Python<->C++ property type dispatch.
struct PropertyTypeMapEntry {
    /// C++ type_info for the mapped type.
    const std::type_info* type_info;

    /// The PropertyType enum value for this type.
    PropertyType property_type;

    /// The corresponding Python type object (e.g. PyBool_Type for bool, nb::type<float3>() for float3).
    nb::object python_type;

    /// Convert a Python object to Any containing the C++ type.
    Any (*to_any)(nb::handle);

    /// Convert an Any to a Python object.
    nb::object (*from_any)(const Any&);

    /// Write a value (from Python object) to Properties under the given key.
    void (*write_to_props)(Properties&, std::string_view, nb::handle);

    /// Read a value from Properties and return as Python object. Returns None if key absent.
    nb::object (*read_from_props)(const Properties&, std::string_view);

    /// Returns true if a Python value is convertible to this C++ type.
    /// Scalars use try_cast semantics; vectors/matrices use isinstance semantics.
    bool (*matches_python)(nb::handle);

    /// Check if first_elem matches this type; if so, cast the full sequence and set_list on props.
    /// Returns false if first_elem does not match.
    bool (*try_set_list)(nb::handle first_elem, nb::handle seq, Properties&, std::string_view);
};

/// Build the global type map table. Entries are returned in a static vector.
/// Thread-safe (initialized on first call).
/// Table order is load-bearing: bool before int64_t, int64_t before double,
/// vectors via isinstance, string last.
const std::vector<PropertyTypeMapEntry>& property_type_map();

/// Find a type map entry by C++ type_info. Returns nullptr if not found.
const PropertyTypeMapEntry* property_type_map_find(const std::type_info& ti);

/// Find a type map entry by PropertyType. O(1) lookup. Returns nullptr for special types.
const PropertyTypeMapEntry* property_type_map_find(PropertyType pt);

/// Find a type map entry by Python type object (identity comparison). Returns nullptr if not found.
const PropertyTypeMapEntry* property_type_map_find(nb::handle py_type);

} // namespace falcor
