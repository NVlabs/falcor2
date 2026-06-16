// SPDX-License-Identifier: Apache-2.0

#pragma once

/// @file
/// The reflection system allows C++ classes to declare their properties
/// via a static reflect() method, which can then be consumed by multiple
/// backends:
///
///   - register_type<T>(): populates a global TypeRegistry with ClassDescriptor
///     and PropertyDescriptor instances for runtime introspection.
///   - serialize_properties / deserialize_properties: directly serialize or
///     deserialize an object's properties to/from a Properties dictionary.
///
/// A reflected class implements the ReflectedClass concept by providing:
/// @code
///   static void reflect(auto& r) {
///       r.def_prop_rw("size", &MyClass::size, &MyClass::set_size, "Doc string.")
///        .def_prop_ro("name", &MyClass::name);
///   }
/// @endcode
///
/// In addition to static (class-level) properties, DynamicPropertySet provides
/// per-instance properties that share the same PropertyDescriptor interface,
/// allowing UI and serialization code to treat both uniformly.

#include "falcor2/core/reflection/reflector.h"
#include "falcor2/core/reflection/metadata.h"
#include "falcor2/core/reflection/properties_reflector.h"
#include "falcor2/core/reflection/property_descriptor.h"
#include "falcor2/core/reflection/property_range.h"
#include "falcor2/core/reflection/class_descriptor.h"
#include "falcor2/core/reflection/type_registry.h"
#include "falcor2/core/reflection/registry_reflector.h"
#include "falcor2/core/reflection/dynamic_properties.h"
