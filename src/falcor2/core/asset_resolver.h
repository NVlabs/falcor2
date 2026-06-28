// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/core/error.h"
#include "falcor2/core/macros.h"
#include "falcor2/core/object.h"

#include <sgl/core/string.h>

#include <filesystem>
#include <regex>
#include <span>
#include <string>
#include <string_view>

namespace falcor {

class FALCOR_API AssetResolver : public Object {
    FALCOR_OBJECT(AssetResolver)
public:
    AssetResolver()
        : m_search_paths_stack(1)
    {
    }

    /// Clones the whole AssetResolver, including the stack.
    ref<AssetResolver> clone() const { return make_ref<AssetResolver>(*this); }

    /// Clones only the current level of AssetResolver, it can no longer be popped.
    ref<AssetResolver> shallow_clone() const
    {
        auto result = make_ref<AssetResolver>();
        result->m_search_paths_stack.back() = m_search_paths_stack.back();
        return result;
    }

    void add_search_path(const std::filesystem::path& path) { m_search_paths_stack.back().push_back(path); }
    void add_search_paths(std::string_view paths)
    {
        for (std::string path : sgl::string::split(paths, ";")) {
            path = sgl::string::remove_leading_trailing_whitespace(path);
            if (!path.empty())
                add_search_path(path);
        }
    }
    void push_scope() { m_search_paths_stack.push_back(m_search_paths_stack.back()); }
    void pop_scope()
    {
        m_search_paths_stack.pop_back();
        FALCOR_CHECK(!m_search_paths_stack.empty(), "AssetResolver scope empty.");
    }

    std::span<const std::filesystem::path> get_search_paths() const { return m_search_paths_stack.back(); }

    std::filesystem::path resolve_path(const std::filesystem::path& asset_path) const
    {
        {
            std::filesystem::path absolute = std::filesystem::absolute(asset_path);
            if (std::filesystem::exists(absolute))
                return std::filesystem::canonical(absolute);
        }

        for (auto& search_path : m_search_paths_stack.back()) {
            auto absolute = search_path / asset_path;
            if (std::filesystem::exists(absolute))
                return std::filesystem::canonical(absolute);
        }

        return {};
    }

    std::vector<std::filesystem::path> resolve_path_pattern(
        const std::filesystem::path& directory_path,
        const std::string& filename_pattern,
        bool first_match_only = true
    ) const
    {
        std::regex filename_regex(filename_pattern);

        // If this is an existing absolute path, or a relative path to the working directory, search it.
        {
            std::filesystem::path absolute = std::filesystem::absolute(directory_path);
            std::vector<std::filesystem::path> resolved = glob_files(absolute, filename_regex, first_match_only);
            if (!resolved.empty())
                return resolved;
        }

        for (auto& search_path : m_search_paths_stack.back()) {
            auto absolute = search_path / directory_path;
            auto resolved = glob_files(absolute, filename_regex, first_match_only);
            if (!resolved.empty())
                return resolved;
        }

        return {};
    }

    /// TODO: This probably should be a more global utility
    std::vector<std::filesystem::path> static glob_files(
        const std::filesystem::path& path,
        const std::regex& filename_regex,
        bool first_match_only
    )
    {
        std::vector<std::filesystem::path> result;
        if (!std::filesystem::exists(path))
            return {};

        for (const auto& entry : std::filesystem::directory_iterator(path)) {
            if (!entry.is_regular_file())
                continue;
            std::string filename = entry.path().filename().string();
            if (std::regex_match(filename, filename_regex)) {
                result.push_back(entry.path());
                if (first_match_only)
                    return result;
            }
        }

        return result;
    }

private:
    std::vector<std::vector<std::filesystem::path>> m_search_paths_stack;
};

} // namespace falcor
