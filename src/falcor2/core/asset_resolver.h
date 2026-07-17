// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/core/macros.h"
#include "falcor2/core/object.h"

#include <filesystem>
#include <span>
#include <string>
#include <vector>

namespace falcor {

/// Immutable ordered search roots used to resolve asset paths.
class FALCOR_API AssetResolver : public Object {
    FALCOR_OBJECT(AssetResolver)
    FALCOR_NON_COPYABLE_AND_MOVABLE(AssetResolver);

public:
    /// Create a resolver without explicit search roots.
    AssetResolver();

    /// Create a resolver from one literal path.
    explicit AssetResolver(const std::filesystem::path& path);

    /// Create a resolver from a semicolon-separated path list.
    explicit AssetResolver(const std::string& paths);

    /// Create a resolver from a semicolon-separated path list.
    explicit AssetResolver(const char* paths);

    /// Create a resolver from literal paths in the supplied order.
    explicit AssetResolver(std::span<const std::filesystem::path> paths);

    /// Return a resolver with one literal path at highest priority.
    [[nodiscard]] ref<AssetResolver> push(const std::filesystem::path& path) const;

    /// Return a resolver with a semicolon-separated path list at highest priority.
    [[nodiscard]] ref<AssetResolver> push(const std::string& paths) const;

    /// Return a resolver with a semicolon-separated path list at highest priority.
    [[nodiscard]] ref<AssetResolver> push(const char* paths) const;

    /// Return a resolver with the literal paths at highest priority in the supplied order.
    [[nodiscard]] ref<AssetResolver> push(std::span<const std::filesystem::path> paths) const;

    /// Return explicit search roots in effective priority order.
    std::span<const std::filesystem::path> get_search_paths() const { return m_search_paths; }

    /// Return a human-readable description of explicit roots and the working-directory fallback.
    std::string to_string() const override;

    /// Resolve one asset path, returning an empty path when it cannot be found.
    /// If required is true, throw a runtime error when the asset cannot be resolved.
    std::filesystem::path resolve_path(const std::filesystem::path& asset_path, bool required = false) const;

    /// Resolve a filename regular expression in the first candidate directory containing matches.
    std::vector<std::filesystem::path> resolve_path_pattern(
        const std::filesystem::path& directory_path,
        const std::string& filename_pattern,
        bool first_match_only = true
    ) const;

private:
    std::vector<std::filesystem::path> m_search_paths;
};

} // namespace falcor
