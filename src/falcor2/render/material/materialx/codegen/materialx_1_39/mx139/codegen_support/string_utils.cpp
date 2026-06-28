// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "string_utils.h"

#include <algorithm>
#include <cctype>
#include <sstream>

namespace falcor {
namespace materialx {
namespace mx139 {
namespace codegen_support {

std::string trim(std::string value)
{
    auto is_space = [](unsigned char c)
    {
        return std::isspace(c) != 0;
    };
    value.erase(
        value.begin(),
        std::find_if(
            value.begin(),
            value.end(),
            [&](char c)
            {
                return !is_space(c);
            }
        )
    );
    value.erase(
        std::find_if(
            value.rbegin(),
            value.rend(),
            [&](char c)
            {
                return !is_space(c);
            }
        ).base(),
        value.end()
    );
    return value;
}

std::string trim_trailing_whitespace_lines(const std::string& value)
{
    std::string result;
    result.reserve(value.size());

    size_t line_start = 0;
    while (line_start < value.size()) {
        const size_t line_end = value.find('\n', line_start);
        const size_t content_end = line_end == std::string::npos ? value.size() : line_end;
        size_t trim_end = content_end;
        while (trim_end > line_start
               && (value[trim_end - 1] == ' ' || value[trim_end - 1] == '\t' || value[trim_end - 1] == '\r'))
            --trim_end;

        result.append(value, line_start, trim_end - line_start);
        if (line_end == std::string::npos)
            break;

        result.push_back('\n');
        line_start = line_end + 1;
    }

    while (!result.empty() && result.back() == '\n')
        result.pop_back();
    result.push_back('\n');
    return result;
}

std::string sanitize_identifier(std::string value)
{
    if (value.empty())
        value = "material";

    for (char& c : value) {
        if (!std::isalnum(static_cast<unsigned char>(c)))
            c = '_';
    }

    if (!std::isalpha(static_cast<unsigned char>(value.front())) && value.front() != '_')
        value = "_" + value;

    return value;
}

std::string snake_case_identifier(std::string value)
{
    value = sanitize_identifier(std::move(value));

    std::string result;
    result.reserve(value.size());
    for (size_t i = 0; i < value.size(); ++i) {
        const unsigned char c = static_cast<unsigned char>(value[i]);
        if (c == '_') {
            if (!result.empty() && result.back() != '_')
                result.push_back('_');
            continue;
        }

        const bool is_upper = std::isupper(c) != 0;
        if (is_upper && !result.empty() && result.back() != '_') {
            const unsigned char previous = static_cast<unsigned char>(value[i - 1]);
            const unsigned char next = i + 1 < value.size() ? static_cast<unsigned char>(value[i + 1]) : 0;
            const bool previous_wants_separator = std::islower(previous) != 0 || std::isdigit(previous) != 0;
            const bool next_wants_separator = next != 0 && std::islower(next) != 0;
            if (previous_wants_separator || next_wants_separator)
                result.push_back('_');
        }

        result.push_back(static_cast<char>(std::tolower(c)));
    }

    while (result.size() > 1 && result.back() == '_')
        result.pop_back();
    return result.empty() ? "_" : result;
}

void replace_all(std::string& text, const std::string& needle, const std::string& replacement)
{
    if (needle.empty())
        return;
    size_t pos = 0;
    while ((pos = text.find(needle, pos)) != std::string::npos) {
        text.replace(pos, needle.size(), replacement);
        pos += replacement.size();
    }
}

std::vector<float> parse_float_list(std::string value)
{
    std::replace(value.begin(), value.end(), ',', ' ');

    std::stringstream stream(value);
    std::vector<float> result;
    float item = 0.0f;
    while (stream >> item)
        result.push_back(item);
    return result;
}

} // namespace codegen_support
} // namespace mx139
} // namespace materialx
} // namespace falcor
