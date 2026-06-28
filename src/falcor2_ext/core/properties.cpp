// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "nanobind.h"
#include "any.h"
#include "property_type_map.h"

#include "falcor2/core/properties.h"
#include "falcor2/core/error.h"

#include <filesystem>

namespace falcor {

nb::object properties_getitem(const Properties& props, std::string_view key)
{
    // We need to ask for type information to return the right cast
    PropertyType type = props.type(key);

    if (type == PropertyType::invalid)
        FALCOR_THROW("Invalid property value");

    // Handle simple scalar/vector types via the property type map.
    const auto* entry = property_type_map_find(type);
    if (entry)
        return entry->read_from_props(props, key);

    // Special types with non-trivial conversion logic.
    switch (type) {
    case PropertyType::enum_: {
        const detail::PropertyEnumValue& ev = props.get<detail::PropertyEnumValue>(key);
        if (ev.type) {
            // Try to reconstruct the Python enum if the C++ type is bound
            nb::handle py_obj = nanobind::detail::enum_from_cpp(ev.type, ev.value);
            if (py_obj.is_valid())
                return nb::steal(py_obj);
        }
        // Fallback: return just the integer value
        return nb::cast(ev.value);
    }
    case PropertyType::object:
        return nb::cast(props.get<ref<Object>>(key));
    case PropertyType::any: {
        const Any& any_val = props.get<Any>(key);
        auto py_handle = any_try_unwrap(any_val);
        if (py_handle)
            return nb::borrow(py_handle);
        nb::handle py_obj = nb_type_to_python(any_val.type(), const_cast<void*>(any_val.data()));
        if (!py_obj.is_valid())
            FALCOR_THROW("Property \"{}\" is not a known Python object", key);
        return nb::steal(py_obj);
    }
    case PropertyType::list: {
        const detail::PropertyList& pl = props.get<detail::PropertyList>(key);
        return pl.dispatch(
            [](const auto& vec) -> nb::object
            {
                return nb::cast(vec);
            }
        );
    }
    default:
        break;
    }
    FALCOR_THROW("Unsupported property type");
}

bool properties_set_enum(Properties& props, std::string_view key, const nb::handle& value)
{
    // For bound enums, preserve type identity
    // Check for __nb_enum__ attribute which marks nanobind enum types
    nb::handle type = value.type();
    if (nb::hasattr(type, "__nb_enum__")) {
        nb::capsule enum_capsule = nb::borrow<nb::capsule>(type.attr("__nb_enum__"));
        auto* type_data = static_cast<nanobind::detail::type_data*>(enum_capsule.data());
        const std::type_info* ti = type_data->type;

        int64_t int_val;
        if (nanobind::detail::enum_from_python(ti, value.ptr(), &int_val, 0)) {
            // Store as PropertyEnumValue with type info preserved
            props.set(key, detail::PropertyEnumValue{ti, int_val});
            return true;
        }
    }
    return false;
}

void properties_set_any(Properties& props, std::string_view key, const nb::handle& value)
{
    try {
        props.set(key, any_wrap(value));
    } catch (...) {
        FALCOR_THROW(
            "Could not assign an object of type '{}' to property '{}'. "
            "The value is not convertible to any of the supported property types "
            "(bool, int, float, vector, string, path, Object). "
            "It also could not be stored as a property of type 'any', "
            "as the underlying type is not exposed via nanobind.",
            nb::inst_name(value).c_str(),
            key
        );
    }
}

bool properties_setitem(Properties& props, std::string_view key, const nb::handle& value)
{
    // Handle simple scalar/vector types via the property type map.
    for (const auto& entry : property_type_map()) {
        if (entry.matches_python(value)) {
            entry.write_to_props(props, key, value);
            return true;
        }
    }

    // Handle special types with non-trivial conversion logic.
    {
        std::filesystem::path typed_value;
        if (nb::try_cast<std::filesystem::path>(value, typed_value)) {
            props.set(key, typed_value);
            return true;
        }
    }
    {
        ref<Object> typed_value;
        if (nb::try_cast<ref<Object>>(value, typed_value)) {
            props.set(key, typed_value);
            return true;
        }
    }

    // Try to handle bound enums with type preservation
    if (properties_set_enum(props, key, value))
        return true;

    // Try to handle Python list/tuple as PropertyList
    if (nb::isinstance<nb::list>(value) || nb::isinstance<nb::tuple>(value)) {
        nb::sequence seq = nb::borrow<nb::sequence>(value);
        size_t len = nb::len(seq);
        if (len == 0)
            FALCOR_THROW("Cannot assign an empty list to property \"{}\": element type cannot be inferred", key);

        nb::handle first = seq[0];

        // Detect element type from first element via the type map.
        for (const auto& entry : property_type_map()) {
            if (entry.try_set_list(first, value, props, key))
                return true;
        }

        // ref<Object> lists: not in the table.
        {
            ref<Object> tmp;
            if (nb::try_cast<ref<Object>>(first, tmp)) {
                props.set_list(key, nb::cast<std::vector<ref<Object>>>(value));
                return true;
            }
        }
    }

    properties_set_any(props, key, value);
    return false;
}

} // namespace falcor

#define DEF_SET_ITEM(T, ...)                                                                                           \
    def(                                                                                                               \
        "__setitem__",                                                                                                 \
        [](Properties& self, std::string_view key, const T& value)                                                     \
        {                                                                                                              \
            self.set(key, value, false);                                                                               \
        },                                                                                                             \
        ##__VA_ARGS__                                                                                                  \
    )

// Typed getters: return properly typed values, with optional default.
// Throws on type mismatch if the property exists but has an incompatible type.
#define DEF_TYPED_GETTER(name, T)                                                                                      \
    def(                                                                                                               \
        name,                                                                                                          \
        [](const Properties& self, std::string_view key, std::optional<T> default_value) -> T                          \
        {                                                                                                              \
            return default_value.has_value() ? self.get<T>(key, default_value.value()) : self.get<T>(key);             \
        },                                                                                                             \
        "key"_a,                                                                                                       \
        "default_value"_a.none() = nb::none()                                                                          \
    )

// Typed list getter: returns a Python list of the specified element type.
#define DEF_TYPED_LIST_GETTER(name, T)                                                                                 \
    def(                                                                                                               \
        name,                                                                                                          \
        [](const Properties& self, std::string_view key) -> std::vector<T>                                             \
        {                                                                                                              \
            return self.get_list<T>(key);                                                                              \
        },                                                                                                             \
        "key"_a                                                                                                        \
    )


FALCOR_PY_EXPORT(core_properties)
{
    using namespace falcor;

    nb::sgl_enum<PropertyType>(m, "PropertyType", D(PropertyType));

    m.attr("EnumT") = nb::type_var("EnumT");

    nb::class_<Properties>(m, "Properties", D(Properties))
        .def(nb::init<>(), D(Properties, Properties))
        .def(nb::init<const Properties&>(), D(Properties, Properties, 2))
        .def(
            "__init__",
            [](Properties* self, nb::dict dict)
            {
                new (self) Properties();
                for (const auto& [k, v] : dict)
                    properties_setitem(*self, nb::cast<std::string_view>(k), v);
            }
        )
        .def("clear", &Properties::clear, D(Properties, clear))
        .def("empty", &Properties::empty, D(Properties, empty))
        .def("size", &Properties::size, D_NA(Properties, size))
        .def("merge", &Properties::merge, D(Properties, merge))
        .def("has_property", &Properties::has_property, D(Properties, has_property))
        .def("remove_property", &Properties::remove_property, D(Properties, remove_property))
        .def(
            "find",
            [](const Properties& self, std::string_view name) -> nb::object
            {
                if (auto entry = self.find(name); entry != self.end()) {
                    return nb::make_tuple(entry->name(), properties_getitem(self, entry->name()));
                }
                return nb::none();
            },
            "name"_a,
            "Return a (key, value) tuple if found, or None if not found."
        )
        .def(
            "rename_property",
            &Properties::rename_property,
            "old_name"_a,
            "new_name"_a,
            D(Properties, rename_property)
        )
        .def("keys", &Properties::keys, D(Properties, keys))
        .def(
            "items",
            [](const Properties& self)
            {
                nb::list result;
                for (const auto& entry : self) {
                    result.append(nb::make_tuple(entry->name(), properties_getitem(self, entry->name())));
                }
                return result;
            },
            "Return a list of (key, value) tuples"
        )
        .def("type", &Properties::type, D(Properties, type))
        .def("as_string", &Properties::as_string, D(Properties, as_string))
        .def("hash", &Properties::hash, D(Properties, hash))
        .DEF_SET_ITEM(bool)
        .DEF_SET_ITEM(int64_t)
        .DEF_SET_ITEM(double)
        .DEF_SET_ITEM(int2)
        .DEF_SET_ITEM(int3)
        .DEF_SET_ITEM(int4)
        .DEF_SET_ITEM(uint2)
        .DEF_SET_ITEM(uint3)
        .DEF_SET_ITEM(uint4)
        .DEF_SET_ITEM(float2)
        .DEF_SET_ITEM(float3)
        .DEF_SET_ITEM(float4)
        .DEF_SET_ITEM(float3x3)
        .DEF_SET_ITEM(float4x4)
        .DEF_SET_ITEM(std::string)
        .DEF_SET_ITEM(std::filesystem::path)
        .def(
            "__setitem__",
            [](Properties& self, std::string_view key, Object* value)
            {
                self.set(key, ref<Object>(value));
            }
        )
        .def(
            "__setitem__",
            [](Properties& self, std::string_view key, nb::list value)
            {
                properties_setitem(self, key, value);
            }
        )
        .def(
            "__setitem__",
            [](Properties& self, std::string_view key, nb::tuple value)
            {
                properties_setitem(self, key, value);
            }
        )
        .def(
            "__setitem__",
            [](Properties& self, std::string_view key, nb::fallback value)
            {
                if (properties_set_enum(self, key, value))
                    return;
                properties_set_any(self, key, value);
            }
        )
        .def(
            "get",
            [](const Properties& self, std::string_view key, nb::object default_value)
            {
                return self.has_property(key) ? properties_getitem(self, key) : default_value;
            },
            "key"_a,
            "default_value"_a.none() = nb::none(),
            D(Properties, get)
        )
        .DEF_TYPED_GETTER("get_bool", bool)
        .DEF_TYPED_GETTER("get_int", int64_t)
        .DEF_TYPED_GETTER("get_float", double)
        .DEF_TYPED_GETTER("get_int2", int2)
        .DEF_TYPED_GETTER("get_int3", int3)
        .DEF_TYPED_GETTER("get_int4", int4)
        .DEF_TYPED_GETTER("get_uint2", uint2)
        .DEF_TYPED_GETTER("get_uint3", uint3)
        .DEF_TYPED_GETTER("get_uint4", uint4)
        .DEF_TYPED_GETTER("get_float2", float2)
        .DEF_TYPED_GETTER("get_float3", float3)
        .DEF_TYPED_GETTER("get_float4", float4)
        .DEF_TYPED_GETTER("get_float3x3", float3x3)
        .DEF_TYPED_GETTER("get_float4x4", float4x4)
        .DEF_TYPED_GETTER("get_string", std::string)
        .DEF_TYPED_GETTER("get_path", std::filesystem::path)
        .DEF_TYPED_GETTER("get_object", ref<Object>)
#undef DEF_TYPED_GETTER
        .def(
            "get_enum",
            [](const Properties& self, std::string_view key, nb::object enum_type, nb::object default_value)
                -> nb::object
            {
                if (!self.has_property(key))
                    return default_value;
                PropertyType pt = self.type(key);
                int64_t int_val;
                if (pt == PropertyType::enum_) {
                    int_val = self.get<detail::PropertyEnumValue>(key).value;
                } else if (pt == PropertyType::int_) {
                    int_val = self.get<int64_t>(key);
                } else {
                    FALCOR_THROW("Property \"{}\" is not an enum or integer type", key);
                }
                return enum_type(nb::cast(int_val));
            },
            "key"_a,
            "enum_type"_a,
            "default_value"_a.none() = nb::none(),
            nb::sig("def get_enum(self, key: str, enum_type: type[EnumT], default_value: EnumT | None = None) -> EnumT")
        )
        .DEF_TYPED_LIST_GETTER("get_bool_list", bool)
        .DEF_TYPED_LIST_GETTER("get_int_list", int64_t)
        .DEF_TYPED_LIST_GETTER("get_float_list", double)
        .DEF_TYPED_LIST_GETTER("get_int2_list", int2)
        .DEF_TYPED_LIST_GETTER("get_int3_list", int3)
        .DEF_TYPED_LIST_GETTER("get_int4_list", int4)
        .DEF_TYPED_LIST_GETTER("get_uint2_list", uint2)
        .DEF_TYPED_LIST_GETTER("get_uint3_list", uint3)
        .DEF_TYPED_LIST_GETTER("get_uint4_list", uint4)
        .DEF_TYPED_LIST_GETTER("get_float2_list", float2)
        .DEF_TYPED_LIST_GETTER("get_float3_list", float3)
        .DEF_TYPED_LIST_GETTER("get_float4_list", float4)
        .DEF_TYPED_LIST_GETTER("get_float3x3_list", float3x3)
        .DEF_TYPED_LIST_GETTER("get_float4x4_list", float4x4)
        .DEF_TYPED_LIST_GETTER("get_string_list", std::string)
        .DEF_TYPED_LIST_GETTER("get_path_list", std::filesystem::path)
        .DEF_TYPED_LIST_GETTER("get_object_list", ref<Object>)
#undef DEF_TYPED_LIST_GETTER
        .def(
            "__getitem__",
            [](const Properties& self, std::string_view key)
            {
                return properties_getitem(self, key);
            },
            D(Properties, get)
        )
        .def(
            "__getattr__",
            [](const Properties& self, std::string_view key)
            {
                return properties_getitem(self, key);
            },
            D(Properties, get)
        )
        .def(
            "__contains__",
            [](const Properties& self, std::string_view key)
            {
                return self.has_property(key);
            }
        )
        .def(
            "__delitem__",
            [](Properties& self, std::string_view key)
            {
                if (!self.remove_property(key))
                    throw nb::key_error(std::string(key).c_str());
            }
        )
        .def(
            "__iter__",
            [](const Properties& self)
            {
                nb::list keys;
                for (const auto& entry : self)
                    keys.append(entry->name());
                return nb::iter(keys);
            }
        )
        .def(
            "__dir__",
            [](const Properties& self) -> nb::object
            {
                nb::list result;
                for (const auto& entry : self) {
                    result.append(entry->name());
                }
                return result;
            }
        )
        .def(
            "__len__",
            [](const Properties& self)
            {
                return self.size();
            }
        )
        .def(
            "__hash__",
            [](const Properties& self)
            {
                return self.hash();
            }
        )
        .def(nb::self == nb::self, D(Properties, operator_eq))
        .def(nb::self != nb::self, D(Properties, operator_ne))
        .def("__repr__", &Properties::to_string);

    nb::implicitly_convertible<nb::dict, Properties>();
}
