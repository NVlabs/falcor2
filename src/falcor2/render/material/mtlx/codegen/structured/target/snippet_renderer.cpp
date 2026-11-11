// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "snippet_renderer.h"

#include "falcor2/core/error.h"
#include "falcor2/core/platform.h"

#include <MaterialXCore/Util.h>
#include <MaterialXGenShader/Util.h>

#include <filesystem>
#include <fstream>
#include <mutex>
#include <regex>
#include <sstream>
#include <unordered_map>

namespace falcor::mtlx::structured {

namespace {

void replace_all(std::string& text, std::string_view from, std::string_view to)
{
    std::size_t position = 0;
    while ((position = text.find(from, position)) != std::string::npos) {
        text.replace(position, from.size(), to);
        position += to.size();
    }
}

void strip_file_header(std::string& text)
{
    while (text.starts_with("// SPDX-")) {
        const std::size_t newline = text.find('\n');
        if (newline == std::string::npos) {
            text.clear();
            return;
        }
        text.erase(0, newline + 1);
    }
    while (!text.empty() && text.front() == '\n')
        text.erase(text.begin());
    constexpr std::string_view MODULE_MARKER = "implementing mtlx_snippets;";
    if (text.starts_with(MODULE_MARKER)) {
        const std::size_t newline = text.find('\n');
        text.erase(0, newline == std::string::npos ? text.size() : newline + 1);
        while (!text.empty() && text.front() == '\n')
            text.erase(text.begin());
    }
}

} // namespace

std::string load_snippet(std::string_view name)
{
    static std::mutex mutex;
    static std::unordered_map<std::string, std::string> cache;
    const std::string key(name);
    {
        std::scoped_lock lock(mutex);
        if (auto found = cache.find(key); found != cache.end())
            return found->second;
    }

    const std::filesystem::path path = platform::runtime_directory() / "mtlx/snippets" / key;
    std::ifstream stream(path, std::ios::binary);
    FALCOR_CHECK(stream.good(), "Unable to read structured MaterialX snippet '{}'.", path.string());
    std::stringstream buffer;
    buffer << stream.rdbuf();
    FALCOR_CHECK(!stream.bad(), "Failed while reading structured MaterialX snippet '{}'.", path.string());
    std::string text = buffer.str();
    replace_all(text, "\r\n", "\n");
    replace_all(text, "\r", "\n");
    strip_file_header(text);
    while (!text.empty() && text.back() == '\n')
        text.pop_back();
    text.push_back('\n');

    std::scoped_lock lock(mutex);
    return cache.emplace(key, std::move(text)).first->second;
}

std::string render_snippet(std::string_view name, const SnippetTokens& tokens)
{
    std::string source = load_snippet(name);
    MaterialX::StringMap substitutions(tokens.begin(), tokens.end());
    MaterialX::tokenSubstitution(substitutions, source);
    static const std::regex LEFTOVER_TOKEN(R"(\$Mx[A-Za-z0-9]+)");
    std::smatch match;
    FALCOR_CHECK(
        !std::regex_search(source, match, LEFTOVER_TOKEN),
        "Structured MaterialX snippet '{}' retains token '{}'.",
        name,
        match.empty() ? std::string{} : match.str()
    );
    return source;
}

} // namespace falcor::mtlx::structured
