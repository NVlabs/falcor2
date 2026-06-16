// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/core/enum.h"

#include <functional>
#include <string>
#include <type_traits>
#include <vector>

namespace falcor::ui {
struct PropertyEditorContext;
} // namespace falcor::ui

namespace falcor::reflection {

/// Explicit default value for a property.
/// Used to reset a property back to its default value.
template<typename T>
struct DefaultValue {
    T value;

    constexpr DefaultValue() = default;
    constexpr explicit DefaultValue(T val)
        : value(std::move(val))
    {
    }
};

/// Convenience factory for DefaultValue.
template<typename T>
DefaultValue<T> default_value(T val)
{
    return DefaultValue<T>(std::move(val));
}

/// Numeric range constraint for property values.
/// Used as a hint for UI sliders. Not enforced by property setters.
/// Stores min/max as double so a single non-templated struct works for all numeric types.
struct ValueRange {
    double min{0};
    double max{0};

    constexpr ValueRange() = default;
    constexpr ValueRange(double min, double max)
        : min(min)
        , max(max)
    {
    }
};

/// Convenience factory for ValueRange.
inline ValueRange value_range(double min, double max)
{
    return ValueRange{min, max};
}

/// A single enum value and its name.
struct EnumItem {
    int64_t value;
    std::string name;
};

/// Descriptor for an enum type.
struct EnumDescriptor {
    /// All registered values.
    std::vector<EnumItem> items;
    /// True for flag/bitmask enums.
    bool is_flags{false};
};

/// Convenience factory for EnumDescriptor.
inline EnumDescriptor enum_descriptor(std::vector<EnumItem> items, bool is_flags = false)
{
    EnumDescriptor desc;
    desc.items = std::move(items);
    desc.is_flags = is_flags;
    return desc;
}

namespace detail {
/// Create an EnumDescriptor from SGL's EnumInfo<T>.
template<sgl::has_enum_info T>
EnumDescriptor enum_descriptor_from_sgl_enum()
{
    EnumDescriptor desc;
    desc.is_flags = sgl::EnumInfo<T>::is_flags;
    for (const auto& [value, name] : sgl::EnumInfo<T>::items)
        desc.items.push_back(EnumItem{static_cast<int64_t>(value), std::string(name)});
    return desc;
}
} // namespace detail


/// Change notification callback for a property.
/// Extracted during property construction (like DefaultValue) and used to wrap the setter.
/// NOT stored in the metadata bag. Constructed via on_change<T>() which hides the void* cast.
struct OnChange {
    std::function<void(void* instance)> callback;
};

/// Convenience factory for OnChange. Hides the void* cast from the caller.
template<typename T, typename F>
    requires std::is_invocable_v<F, T&>
OnChange on_change(F&& func)
{
    return OnChange{[func = std::forward<F>(func)](void* instance)
                    {
                        func(*static_cast<T*>(instance));
                    }};
}

/// Convenience factory for OnChange from a member function pointer.
/// Usage: on_change<MyClass>(&MyClass::mark_dirty)
template<typename T>
OnChange on_change(void (T::*member_func)())
{
    return OnChange{[member_func](void* instance)
                    {
                        (static_cast<T*>(instance)->*member_func)();
                    }};
}

/// Override display name for a property (default: property name).
struct UILabel {
    std::string text;
};

/// Convenience factory for UILabel.
inline UILabel ui_label(std::string_view text)
{
    return UILabel{std::string{text}};
}

/// Hierarchical grouping using '/' delimiter (e.g., "Appearance/Color").
/// Rendered as collapsing headers in the UI.
struct UIGroup {
    std::string path;
};

/// Convenience factory for UIGroup.
inline UIGroup ui_group(std::string_view path)
{
    return UIGroup{std::string{path}};
}

/// Override default drag speed for numeric properties.
struct UIDragSpeed {
    float speed;
};

/// Convenience factory for UIDragSpeed.
inline UIDragSpeed ui_drag_speed(float speed)
{
    return UIDragSpeed{speed};
}

/// Conditional enable/disable for a property in the UI.
/// Constructed via ui_enable_if<T>() which hides the void* cast.
struct UIEnableIf {
    std::function<bool(const void* instance)> predicate;
};

/// Convenience factory for UIEnableIf. Hides the void* cast from the caller.
template<typename T, typename F>
    requires std::is_invocable_r_v<bool, F, const T&>
UIEnableIf ui_enable_if(F&& pred)
{
    return UIEnableIf{
        [pred = std::forward<F>(pred)](const void* instance) -> bool
        {
            return pred(*static_cast<const T*>(instance));
        }
    };
}

/// Fully custom callback for a property editor.
/// Uses a plain function pointer to avoid heap allocation inside Any storage.
/// Returns true if property value changed.
struct UIEditor {
    bool (*func)(void* instance, const class PropertyDescriptor& desc, falcor::ui::PropertyEditorContext& ctx);
};

/// Convenience factory for UIEditor.
inline UIEditor
ui_editor(bool (*func)(void* instance, const PropertyDescriptor& desc, falcor::ui::PropertyEditorContext& ctx))
{
    return UIEditor{func};
}

/// UI hint flags for property rendering.
/// Multiple flags can be combined and are OR-accumulated during property registration.
enum class UIFlags : uint32_t {
    none = 0,
    /// Display float3/float4 as a color picker instead of three/four separate drags.
    display_as_color = 1 << 0,
    /// Hide unless "show advanced" is enabled.
    advanced = 1 << 1,
    /// Never show the property in the UI.
    hidden = 1 << 2,
};
SGL_ENUM_FLAGS_INFO(
    UIFlags,
    {
        {UIFlags::none, "none"},
        {UIFlags::display_as_color, "display_as_color"},
        {UIFlags::advanced, "advanced"},
        {UIFlags::hidden, "hidden"},
    }
);
SGL_ENUM_REGISTER(UIFlags);
FALCOR_ENUM_CLASS_OPERATORS(UIFlags)

// ----------------------------------------------------------------------------
// Metadata traits
// ----------------------------------------------------------------------------

/// Trait to check if a type is a falcor reflection metadata type.
template<typename T>
struct is_reflection_metadata : std::false_type { };

template<typename T>
struct is_reflection_metadata<DefaultValue<T>> : std::true_type { };
template<>
struct is_reflection_metadata<ValueRange> : std::true_type { };
template<>
struct is_reflection_metadata<EnumDescriptor> : std::true_type { };
template<>
struct is_reflection_metadata<OnChange> : std::true_type { };
template<>
struct is_reflection_metadata<UILabel> : std::true_type { };
template<>
struct is_reflection_metadata<UIGroup> : std::true_type { };
template<>
struct is_reflection_metadata<UIDragSpeed> : std::true_type { };
template<>
struct is_reflection_metadata<UIEnableIf> : std::true_type { };
template<>
struct is_reflection_metadata<UIEditor> : std::true_type { };
template<>
struct is_reflection_metadata<UIFlags> : std::true_type { };

template<typename T>
inline constexpr bool is_reflection_metadata_v = is_reflection_metadata<std::decay_t<T>>::value;

/// Trait to check if a type is a reflection flags enum (OR-accumulated).
/// Flags types are also valid reflection metadata. During extract_metadata,
/// multiple values of the same flags type are OR'd together into a single entry.
template<typename T>
struct is_reflection_flags : std::false_type { };

template<>
struct is_reflection_flags<UIFlags> : std::true_type { };

template<typename T>
inline constexpr bool is_reflection_flags_v = is_reflection_flags<std::decay_t<T>>::value;

namespace detail {
/// Compile-time check: true if OnChange appears anywhere in the parameter pack.
template<typename... Ts>
inline constexpr bool has_on_change_v = (std::is_same_v<std::decay_t<Ts>, OnChange> || ...);
} // namespace detail

} // namespace falcor::reflection
