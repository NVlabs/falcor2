// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "platform.h"

#include <algorithm>
#include <string_view>

#if defined(__GNUG__)
#include <cxxabi.h>
#endif

namespace falcor::platform {

// -------------------------------------------------------------------------------------------------
// System paths
// -------------------------------------------------------------------------------------------------

const std::filesystem::path& project_directory()
{
    static std::filesystem::path path = []()
    {
        std::filesystem::path path_ = std::filesystem::path{FALCOR_PROJECT_DIR}.lexically_normal();
        if (path_.empty())
            return runtime_directory();
        return path_;
    }();
    return path;
}

// -------------------------------------------------------------------------------------------------
// Utilities
// -------------------------------------------------------------------------------------------------

namespace {

/// Remove all occurrences of a substring from a string.
void remove_all(std::string& str, std::string_view substr)
{
    size_t pos;
    while ((pos = str.find(substr)) != std::string::npos)
        str.erase(pos, substr.size());
}

/// Normalize a type name to be consistent across compilers.
/// This strips MSVC-specific prefixes and decorations.
std::string normalize_type_name(std::string name)
{
    // Strip leading type keywords (MSVC includes these, GCC/Clang don't)
    for (std::string_view prefix : {"enum ", "class ", "struct "}) {
        if (name.starts_with(prefix)) {
            name.erase(0, prefix.size());
            break;
        }
    }

    // Remove MSVC pointer decorations
    remove_all(name, " __ptr64");

    // Remove MSVC calling conventions
    for (std::string_view cc : {"__cdecl", "__stdcall", "__fastcall", "__thiscall", "__vectorcall"})
        remove_all(name, cc);

    // Clean up any resulting double spaces
    size_t pos;
    while ((pos = name.find("  ")) != std::string::npos)
        name.erase(pos, 1);

    // Trim leading/trailing whitespace
    auto not_space = [](unsigned char c)
    {
        return !std::isspace(c);
    };
    name.erase(name.begin(), std::find_if(name.begin(), name.end(), not_space));
    name.erase(std::find_if(name.rbegin(), name.rend(), not_space).base(), name.end());

    return name;
}

} // namespace

std::string demangle_type_name(const std::type_info& t)
{
    const char* name_in = t.name();

#if defined(__GNUG__)
    int status = 0;
    char* name = abi::__cxa_demangle(name_in, nullptr, nullptr, &status);
    if (!name)
        return normalize_type_name(std::string(name_in));
    std::string result(name);
    free(name);
    return normalize_type_name(std::move(result));
#else
    return normalize_type_name(std::string(name_in));
#endif
}

} // namespace falcor::platform
