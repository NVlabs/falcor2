// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "falcor2/core/asset_resolver.h"
#include "falcor2/core/error.h"

#include <sgl/core/string.h>

#include <algorithm>
#include <regex>

namespace falcor {

namespace {

void append_unique(std::vector<std::filesystem::path>& paths, const std::filesystem::path& path)
{
    if (path.empty())
        return;

    const auto normalized = std::filesystem::absolute(path).lexically_normal();
    if (std::find(paths.begin(), paths.end(), normalized) == paths.end())
        paths.push_back(normalized);
}

std::vector<std::filesystem::path> split_paths(const std::string& paths)
{
    std::vector<std::filesystem::path> result;
    for (std::string item : sgl::string::split(paths, ";")) {
        item = sgl::string::remove_leading_trailing_whitespace(item);
        if (!item.empty())
            result.emplace_back(std::move(item));
    }
    return result;
}

std::vector<std::filesystem::path> normalize_paths(std::span<const std::filesystem::path> paths)
{
    std::vector<std::filesystem::path> result;
    result.reserve(paths.size());
    for (const auto& path : paths)
        append_unique(result, path);
    return result;
}

std::vector<std::filesystem::path> normalize_split_paths(const std::string& paths)
{
    const auto split = split_paths(paths);
    return normalize_paths(split);
}

std::filesystem::path try_canonical(const std::filesystem::path& path)
{
    std::error_code error;
    auto canonical = std::filesystem::canonical(path, error);
    return error ? std::filesystem::path{} : canonical;
}

std::vector<std::filesystem::path>
glob_files(const std::filesystem::path& path, const std::regex& filename_regex, bool first_match_only)
{
    std::vector<std::filesystem::path> result;
    std::error_code error;
    std::filesystem::directory_iterator iterator(
        path,
        std::filesystem::directory_options::skip_permission_denied,
        error
    );
    const std::filesystem::directory_iterator end;

    while (!error && iterator != end) {
        std::error_code entry_error;
        if (iterator->is_regular_file(entry_error) && !entry_error) {
            const std::string filename = iterator->path().filename().string();
            if (std::regex_match(filename, filename_regex))
                result.push_back(iterator->path().lexically_normal());
        }
        iterator.increment(error);
    }

    if (result.empty())
        return {};

    std::sort(
        result.begin(),
        result.end(),
        [](const auto& lhs, const auto& rhs)
        {
            return lhs.generic_string() < rhs.generic_string();
        }
    );
    if (first_match_only)
        result.resize(1);
    return result;
}

} // namespace

AssetResolver::AssetResolver() = default;

AssetResolver::AssetResolver(const std::filesystem::path& path)
{
    append_unique(m_search_paths, path);
}

AssetResolver::AssetResolver(const std::string& paths)
    : m_search_paths(normalize_split_paths(paths))
{
}

AssetResolver::AssetResolver(const char* paths)
    : AssetResolver(std::string(paths))
{
}

AssetResolver::AssetResolver(std::span<const std::filesystem::path> paths)
    : m_search_paths(normalize_paths(paths))
{
}

ref<AssetResolver> AssetResolver::push(const std::filesystem::path& path) const
{
    return push(std::span<const std::filesystem::path>(&path, 1));
}

ref<AssetResolver> AssetResolver::push(const std::string& paths) const
{
    return push(split_paths(paths));
}

ref<AssetResolver> AssetResolver::push(const char* paths) const
{
    return push(std::string(paths));
}

ref<AssetResolver> AssetResolver::push(std::span<const std::filesystem::path> paths) const
{
    std::vector<std::filesystem::path> result;
    result.reserve(paths.size() + m_search_paths.size());
    for (const auto& path : paths)
        append_unique(result, path);
    for (const auto& path : m_search_paths)
        append_unique(result, path);

    auto resolver = ref<AssetResolver>(new AssetResolver());
    resolver->m_search_paths = std::move(result);
    return resolver;
}

std::string AssetResolver::to_string() const
{
    std::string result = "AssetResolver(search_paths=[";
    for (size_t i = 0; i < m_search_paths.size(); ++i) {
        if (i > 0)
            result += ", ";
        result += fmt::format("\"{}\"", m_search_paths[i]);
    }

    std::error_code error;
    const auto working_directory = std::filesystem::current_path(error);
    result += "], working_directory_fallback=";
    result += error ? "<unavailable>" : fmt::format("\"{}\"", working_directory);
    result += ")";
    return result;
}

std::filesystem::path AssetResolver::resolve_path(const std::filesystem::path& asset_path, bool required) const
{
    std::filesystem::path resolved;
    if (asset_path.is_absolute()) {
        resolved = try_canonical(asset_path);
    } else {
        for (const auto& search_path : m_search_paths) {
            resolved = try_canonical(search_path / asset_path);
            if (!resolved.empty())
                return resolved;
        }

        std::error_code error;
        const auto absolute = std::filesystem::absolute(asset_path, error);
        if (!error)
            resolved = try_canonical(absolute);
    }

    if (resolved.empty() && required)
        FALCOR_THROW("Failed to resolve required asset path \"{}\". {}.", asset_path, to_string());
    return resolved;
}

std::vector<std::filesystem::path> AssetResolver::resolve_path_pattern(
    const std::filesystem::path& directory_path,
    const std::string& filename_pattern,
    bool first_match_only
) const
{
    const std::regex filename_regex(filename_pattern);

    if (directory_path.is_absolute())
        return glob_files(directory_path.lexically_normal(), filename_regex, first_match_only);

    for (const auto& search_path : m_search_paths) {
        auto resolved = glob_files((search_path / directory_path).lexically_normal(), filename_regex, first_match_only);
        if (!resolved.empty())
            return resolved;
    }

    std::error_code error;
    const auto absolute = std::filesystem::absolute(directory_path, error);
    return error ? std::vector<std::filesystem::path>{}
                 : glob_files(absolute.lexically_normal(), filename_regex, first_match_only);
}

} // namespace falcor
