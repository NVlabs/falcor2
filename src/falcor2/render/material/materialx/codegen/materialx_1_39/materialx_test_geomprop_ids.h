// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <string_view>
#include <unordered_map>

namespace falcor::materialx::mx139 {

inline std::optional<uint32_t> materialx_test_geomprop_id(std::string_view geomprop)
{
    static const std::unordered_map<std::string_view, uint32_t> k_ids = {
        {"geompropvalue_integer", 1},
        {"geompropvalue_bool", 2},
        {"geompropvalue_boolean", 2},
        {"geompropvalue_float", 3},
        {"geompropvalue_vector2", 4},
        {"geompropvalue_vector3", 5},
        {"geompropvalue_vector4", 6},
        {"geompropvalue_color2", 7},
        {"geompropvalue_color3", 8},
        {"geompropvalue_color4", 9},
        {"myvalue", 10},
        {"texcoord_1", 11},
        {"color_0", 12},
        {"color_1", 13},
    };

    const auto it = k_ids.find(geomprop);
    if (it == k_ids.end())
        return {};
    return it->second;
}

} // namespace falcor::materialx::mx139
