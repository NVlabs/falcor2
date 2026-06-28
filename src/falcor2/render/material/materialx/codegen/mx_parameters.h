// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/render/texture_manager.h"
#include "falcor2/core/types.h"
#include "falcor2/core/object.h"
#include "falcor2/core/error.h"
#include "falcor2/core/reflection.h"

#include "falcor2/utils/idictionary.h"

#include <sgl/device/resource.h>
#include <sgl/device/buffer_cursor.h>

#include <string>
#include <optional>
#include <vector>
#include <any>
#include <variant>

namespace falcor {

class Properties;

namespace materialx {
namespace params {
/// Contains all the settings MaterialX uses to describe desired texture loading.
struct TextureInfo {
    std::string mx_input_name;  ///< Name of the materialx input
    std::string file_name;      ///< Filename of the texture to load
    std::string layer;          ///< Layered within the texture. IGNORED
    std::string uaddressmode;   ///< U-direction wrap mode used for texture sampler creation.
    std::string vaddressmode;   ///< V-direction wrap mode used for texture sampler creation.
    std::string filtertype;     ///< Filter mode used for texture sampler creation.
    std::string framerange;     ///< Used for per-frame (animated) textures. IGNORED.
    std::string frameoffset;    ///< Used for per-frame (animated) textures. IGNORED.
    std::string frameendaction; ///< Used for per-frame (animated) textures. IGNORED.
    /// MaterialX image default value, used as sampler border color for constant address mode.
    float4 border_color = float4(0.f, 0.f, 0.f, 1.f);
    /// ColorX outputs should read as srgb for LDR images, while the rest read as is in file.
    bool is_color_output_type = false;

    // Updated after we load the texture
    TextureHandle texture_handle; ///< Value of the handle
};

/// @Describe user interface information for a parameter
struct UserInterfaceInfo {
    std::string doc;                  ///< Doc string
    std::string name;                 ///< UI name of the parameter (prefer to display this)
    std::vector<std::string> folders; ///< Folder nesting of the parameter in the UI, always starts with Params

    bool operator==(const UserInterfaceInfo& rhs) const = default;
    // Sort best UI representation, for each folder, have the direct content first,
    // sorted by name, and then sort by the name of the subfolder.
    bool operator<(const UserInterfaceInfo& rhs) const;
};


/// The type of the parameter data
enum class Type {
    int_,
    bool_,
    float_,
    float2_,
    float3_,
    float4_,
    float4x4_,
    texture,
    unsupported, ///< Type that is not supported by the UI (strings, arrays)
};

using Value = std::variant<int, bool, float, float2, float3, float4, float4x4, std::unique_ptr<TextureInfo>>;

template<typename T>
struct TypeTraits {
    static constexpr Type type = Type::unsupported;
};

#define PARAM_TRAIT(T, type_enum)                                                                                      \
    template<>                                                                                                         \
    struct TypeTraits<T> {                                                                                             \
        static constexpr Type type = type_enum;                                                                        \
        using ReturnType = T;                                                                                          \
        using StoredType = T;                                                                                          \
        static const ReturnType& extract_value(const StoredType& v)                                                    \
        {                                                                                                              \
            return v;                                                                                                  \
        }                                                                                                              \
        static ReturnType& extract_value(StoredType& v)                                                                \
        {                                                                                                              \
            return v;                                                                                                  \
        }                                                                                                              \
    }

PARAM_TRAIT(int, Type::int_);
PARAM_TRAIT(bool, Type::bool_);
PARAM_TRAIT(float, Type::float_);
PARAM_TRAIT(float2, Type::float2_);
PARAM_TRAIT(float3, Type::float3_);
PARAM_TRAIT(float4, Type::float4_);
PARAM_TRAIT(float4x4, Type::float4x4_);

#undef PARAM_TRAIT

template<>
struct TypeTraits<std::unique_ptr<TextureInfo>> {
    static constexpr Type type = Type::texture;
    using ReturnType = TextureHandle;
    using StoredType = std::unique_ptr<TextureInfo>;
    static const ReturnType& extract_value(const StoredType& v) { return v->texture_handle; }
    static ReturnType& extract_value(StoredType& v) { return v->texture_handle; }
};
template<>
struct TypeTraits<TextureInfo> {
    static constexpr Type type = Type::texture;
    using ReturnType = TextureInfo;
    using StoredType = std::unique_ptr<TextureInfo>;
    static const ReturnType& extract_value(const StoredType& v) { return *v; }
    static ReturnType& extract_value(StoredType& v) { return *v; }
};

template<typename T>
concept HasExtractValue = requires(typename TypeTraits<T>::StoredType& stored) {
    { TypeTraits<T>::extract_value(stored) } -> std::convertible_to<typename TypeTraits<T>::ReturnType&>;
};

} // namespace params

/// Reprensents the MaterialX public parameter.
struct FALCOR_API MxParamInfo {
    struct MxMaterialInformation {
        bool hasEntryPointVolumeProperties;
    };

    /// Name of the parameter in the generated Slang code.
    std::string param_name;
    /// Type of the parameter, if support.
    params::Type param_type;
    /// Current value of the parameter, only valid for editable parameters with supported param_type;
    std::optional<params::Value> param_value;
    // Describes interface
    params::UserInterfaceInfo interface;
    /// Parameter's type in MaterialX system.
    std::string mx_type_string;
    /// True when the parameter can be changed from outside. False if it is a compile-time constant.
    bool is_editable = false;

    // Sets value from Properties
    void set_value(const Properties& props);
    void set_value(const reflection::DynamicPropertySet& props);

    template<typename TDst, typename TSrc>
    void set_value(const TSrc& value)
    {
        static_assert(sizeof(TDst) == sizeof(TSrc));
        FALCOR_ASSERT(params::TypeTraits<TDst>::type == param_type);
        param_value = reinterpret_cast<const TDst&>(value);
    }

    void set_value(const params::TextureInfo&& value)
    {
        FALCOR_ASSERT(param_type == params::Type::texture);
        param_value = std::make_unique<params::TextureInfo>(std::move(value));
    }

    void to_dictionary(IDictionary& dict) const;
    template<typename CursorT>
    void set_cursor(CursorT& cursor) const;

    // True if `param_value` holds an actual value, mandatory for all is_editable
    bool holds_value() const { return param_value.has_value(); }

    template<typename T>
        requires params::HasExtractValue<T>
    T* extract_value()
    {
        if (param_value.has_value()) {
            using Trait = typename params::TypeTraits<T>;
            if (auto stored = std::get_if<typename Trait::StoredType>(&param_value.value()))
                return &Trait::extract_value(*stored);
        }
        return nullptr;
    }

    template<typename T>
        requires params::HasExtractValue<T>
    const T* extract_value() const
    {
        if (param_value.has_value()) {
            using Trait = typename params::TypeTraits<T>;
            if (auto stored = std::get_if<typename Trait::StoredType>(&param_value.value()))
                return &Trait::extract_value(*stored);
        }
        return nullptr;
    }

private:
};

class MxParamList {
public:
    void add_param_info(MxParamInfo param);

public:
    uint32_t m_editable_count = 0;
    std::vector<MxParamInfo> m_params;
};

} // namespace materialx
} // namespace falcor
