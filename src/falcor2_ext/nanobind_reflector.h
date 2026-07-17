// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "nanobind.h"
#include "falcor2/core/reflection/metadata.h"
#include "falcor2/core/reflection/property_descriptor.h"

#include <tuple>

namespace falcor::reflection {

/// Check if a type should be filtered out when forwarding to nanobind.
/// Reflection metadata types are filtered; everything else (including const char* doc strings) is forwarded.
template<typename T>
inline constexpr bool is_nanobind_extra_v = !is_reflection_metadata_v<T>;

namespace detail {

/// Helper to filter extras into a tuple of only nanobind-compatible ones.
inline auto filter_nanobind_extras()
{
    return std::tuple<>();
}

template<typename First, typename... Rest>
auto filter_nanobind_extras(First&& first, Rest&&... rest)
{
    if constexpr (is_nanobind_extra_v<First>) {
        return std::tuple_cat(
            std::make_tuple(std::forward<First>(first)),
            filter_nanobind_extras(std::forward<Rest>(rest)...)
        );
    } else {
        return filter_nanobind_extras(std::forward<Rest>(rest)...);
    }
}

/// Class reflector that creates nanobind Python bindings from reflect() calls.
template<typename T, typename... Extra>
class NanobindClassReflector {
public:
    using target_type = T;

    NanobindClassReflector(nb::class_<T, Extra...> nb_class)
        : m_nb_class(nb_class)
    {
    }

    /// Define a read-write property.
    template<typename Getter, typename Setter, typename... Extras>
    NanobindClassReflector& def_prop_rw(const char* name, Getter getter, Setter setter, Extras&&... extras)
    {
        auto nb_extras = detail::filter_nanobind_extras(std::forward<Extras>(extras)...);
        if constexpr (detail::has_on_change_v<Extras...>) {
            using ReturnType = std::invoke_result_t<Getter, const T&>;
            using ValueType = std::remove_cv_t<std::remove_reference_t<ReturnType>>;
            auto extracted = detail::extract_all<ValueType>(std::forward<Extras>(extras)...);
            auto on_change_callback = std::move(extracted.on_change->callback);
            auto wrapped_setter = [setter = std::move(setter),
                                   on_change_callback = std::move(on_change_callback)](T& obj, const ValueType& value)
            {
                std::invoke(setter, obj, value);
                on_change_callback(&obj);
            };
            std::apply(
                [&](auto&&... args)
                {
                    m_nb_class
                        .def_prop_rw(name, getter, std::move(wrapped_setter), std::forward<decltype(args)>(args)...);
                },
                nb_extras
            );
        } else {
            std::apply(
                [&](auto&&... args)
                {
                    m_nb_class.def_prop_rw(name, getter, setter, std::forward<decltype(args)>(args)...);
                },
                nb_extras
            );
        }
        return *this;
    }

    /// Define a read-only property.
    template<typename Getter, typename... Extras>
    NanobindClassReflector& def_prop_ro(const char* name, Getter getter, Extras&&... extras)
    {
        auto nb_extras = detail::filter_nanobind_extras(std::forward<Extras>(extras)...);
        std::apply(
            [&](auto&&... args)
            {
                m_nb_class.def_prop_ro(name, getter, std::forward<decltype(args)>(args)...);
            },
            nb_extras
        );
        return *this;
    }

    /// Define a read-write property from a member pointer.
    /// If OnChange is present, synthesizes getter/setter lambdas and uses def_prop_rw
    /// to ensure Python attribute assignment goes through the wrapped setter.
    template<typename V, typename... Extras>
    NanobindClassReflector& def_rw(const char* name, V T::* member, Extras&&... extras)
    {
        if constexpr (detail::has_on_change_v<Extras...>) {
            // OnChange present: must use def_prop_rw with lambdas to ensure
            // Python obj.field = x goes through the same wrapped setter.
            auto getter = [member](const T& obj) -> V
            {
                return obj.*member;
            };
            auto setter = [member](T& obj, const V& val)
            {
                obj.*member = val;
            };
            // Forward to def_prop_rw which will filter extras for nanobind.
            return def_prop_rw(name, getter, setter, std::forward<Extras>(extras)...);
        } else {
            auto nb_extras = detail::filter_nanobind_extras(std::forward<Extras>(extras)...);
            std::apply(
                [&](auto&&... args)
                {
                    m_nb_class.def_rw(name, member, std::forward<decltype(args)>(args)...);
                },
                nb_extras
            );
            return *this;
        }
    }

    /// Define a read-only property from a member pointer.
    template<typename V, typename... Extras>
    NanobindClassReflector& def_ro(const char* name, V T::* member, Extras&&... extras)
    {
        auto nb_extras = detail::filter_nanobind_extras(std::forward<Extras>(extras)...);
        std::apply(
            [&](auto&&... args)
            {
                m_nb_class.def_ro(name, member, std::forward<decltype(args)>(args)...);
            },
            nb_extras
        );
        return *this;
    }

    /// Get the underlying nanobind class for additional nanobind-specific configuration.
    nb::class_<T, Extra...>& nb_class() { return m_nb_class; }

private:
    nb::class_<T, Extra...> m_nb_class;
};

} // namespace detail

/// Bind a ReflectedClass type by creating an nb::class_ from its metadata and calling reflect().
template<typename T>
void bind(nb::module_& m)
{
    if constexpr (std::is_same_v<typename T::ReflectedBase, void>) {
        auto cls = nb::class_<T>(m, T::reflected_class_name());
        detail::NanobindClassReflector<T> reflector(cls);
        T::reflect(reflector);
    } else {
        auto cls = nb::class_<T, typename T::ReflectedBase>(m, T::reflected_class_name());
        detail::NanobindClassReflector<T, typename T::ReflectedBase> reflector(cls);
        T::reflect(reflector);
    }
}

/// Bind a ReflectedClass type into an existing nb::class_ binding by calling reflect().
/// Use this when you need custom nanobind configuration before or after binding properties.
template<typename T, typename... Extra>
void bind(nb::class_<T, Extra...>& cls)
{
    detail::NanobindClassReflector<T, Extra...> reflector(cls);
    T::reflect(reflector);
}

} // namespace falcor::reflection
