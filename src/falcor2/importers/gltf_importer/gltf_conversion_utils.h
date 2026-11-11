// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "gltf_conversions.h"
#include "falcor2/core/error.h"

namespace falcor::gltf_conversion {

inline void warn(Properties& result, const ImporterMaterial& source, std::string_view target, std::string message)
{
    auto warnings
        = result.get_list_optional<std::string>("_gltf_conversion_warnings").value_or(std::vector<std::string>{});
    warnings.push_back(message);
    result.set_list("_gltf_conversion_warnings", warnings);
    sgl::log_warn("glTF material '{}', {}: {}", source.name, target, message);
}

template<typename T>
void copy(const Properties& params, Properties& result, std::string_view input, std::string_view output)
{
    if (auto value = params.get_optional<T>(input))
        result.set(output, *value);
}

inline void texture(
    const ImporterMaterial& source,
    Properties& result,
    std::span<const ImporterTexture> textures,
    const GLTFTextureSetter& setter,
    std::string_view input,
    std::string_view output,
    TextureLoadFlags flags,
    std::string_view target
)
{
    const auto index = source.params.get_optional<int>(input);
    if (!index)
        return;
    FALCOR_CHECK(*index >= 0 && size_t(*index) < textures.size(), "Invalid glTF texture index for '{}'", input);
    const int uv = source.params.get(fmt::format("{}:texcoord", input), 0);
    FALCOR_CHECK(uv >= 0 && uv < 8, "Invalid glTF UV set {} for '{}'", uv, input);
    setter(result, output, textures[*index], flags);
    result.set(fmt::format("{}_texcoord", output), uint32_t(uv));
    if (source.params.get(fmt::format("{}:has_transform", input), false))
        warn(
            result,
            source,
            target,
            fmt::format("{} texture transform is unsupported; sampling untransformed UVs", input)
        );
}

inline void unsupported(
    const ImporterMaterial& source,
    Properties& result,
    std::string_view target,
    std::initializer_list<std::string_view> features
)
{
    for (auto feature : features) {
        if (source.params.get(feature, false))
            warn(result, source, target, fmt::format("{} is not mapped", feature));
    }
}

} // namespace falcor::gltf_conversion
