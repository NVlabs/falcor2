// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <concepts>
#include <type_traits>

namespace falcor::reflection {

namespace detail {

/// Sentinel type used in concept checks to verify that reflector methods
/// accept an arbitrary parameter pack of extras (metadata).
/// If a method accepts this unknown type it must be using a template parameter pack.
struct ExtraProbe { };

/// Opaque value type used as the property type in concept checks.
struct ValueProbe { };

/// Implementation concept that checks all four reflector methods against a known target type T.
/// T is obtained from R::target_type so this is not exposed directly to users.
template<typename R, typename T>
concept ClassReflectorImpl = requires(
    R r,
    const char* name,
    ValueProbe T::* member,
    ValueProbe (T::*getter)() const,
    void (T::*setter)(ValueProbe)
) {
    // def_prop_ro: read-only property via getter.
    { r.def_prop_ro(name, getter) } -> std::same_as<R&>;
    // def_prop_ro with two extras (proves true variadic pack).
    { r.def_prop_ro(name, getter, ExtraProbe{}, ExtraProbe{}) } -> std::same_as<R&>;

    // def_prop_rw: read-write property via getter/setter pair.
    { r.def_prop_rw(name, getter, setter) } -> std::same_as<R&>;
    // def_prop_rw with two extras (proves true variadic pack).
    { r.def_prop_rw(name, getter, setter, ExtraProbe{}, ExtraProbe{}) } -> std::same_as<R&>;

    // def_rw: read-write property via member pointer.
    { r.def_rw(name, member) } -> std::same_as<R&>;
    // def_rw with two extras (proves true variadic pack).
    { r.def_rw(name, member, ExtraProbe{}, ExtraProbe{}) } -> std::same_as<R&>;

    // def_ro: read-only property via member pointer.
    { r.def_ro(name, member) } -> std::same_as<R&>;
    // def_ro with two extras (proves true variadic pack).
    { r.def_ro(name, member, ExtraProbe{}, ExtraProbe{}) } -> std::same_as<R&>;
};

} // namespace detail

/// Concept for a class reflector.
///
/// A ClassReflector must expose a `target_type` alias and support def_prop_rw,
/// def_prop_ro, def_rw, and def_ro for chaining property declarations.
/// Each method accepts a property name, a getter/setter or member pointer,
/// and an arbitrary number of additional metadata arguments (doc strings,
/// DefaultValue, ValueRange, etc.).
///
/// The variadic extras requirement is verified by passing an unknown sentinel
/// type (ExtraProbe) as an extra argument; a method that accepts it must be
/// using a template parameter pack.
template<typename R>
concept ClassReflector
    = requires { typename R::target_type; } && detail::ClassReflectorImpl<R, typename R::target_type>;

namespace detail {

/// Helper target type providing canonical getter, setter, and member pointer
/// for concept checking without depending on any real reflected class.
/// Uses ValueProbe as the property type to ensure reflector methods are generic.
struct ReflectorProbeTarget {
    ValueProbe value{};
    ValueProbe get_value() const { return value; }
    void set_value(ValueProbe v) { value = v; }
};

/// Minimal reflector satisfying the ClassReflector concept, used only for ReflectedClass trait checking.
struct ClassReflectorProbe {
    using target_type = ReflectorProbeTarget;

    template<typename... Args>
    ClassReflectorProbe& def_prop_ro(const char*, Args&&...)
    {
        return *this;
    }
    template<typename... Args>
    ClassReflectorProbe& def_prop_rw(const char*, Args&&...)
    {
        return *this;
    }
    template<typename... Args>
    ClassReflectorProbe& def_rw(const char*, Args&&...)
    {
        return *this;
    }
    template<typename... Args>
    ClassReflectorProbe& def_ro(const char*, Args&&...)
    {
        return *this;
    }
};

static_assert(ClassReflector<ClassReflectorProbe>);

} // namespace detail

/// Concept to check if a class is a fully reflected type.
/// Requires:
///   - A `ReflectedBase` type alias (the reflected base class, or void for roots).
///   - A static `reflected_class_name()` returning the class name.
///   - A static `reflect(r)` method accepting a ClassReflector.
template<typename T>
concept ReflectedClass = requires(detail::ClassReflectorProbe r) {
    typename T::ReflectedBase;
    { T::reflected_class_name() } -> std::convertible_to<const char*>;
    { T::reflect(r) } -> std::same_as<void>;
};

} // namespace falcor::reflection
