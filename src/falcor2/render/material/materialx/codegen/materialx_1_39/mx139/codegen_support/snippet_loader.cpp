// SPDX-License-Identifier: Apache-2.0

#include "snippet_loader.h"

#include "slang_source_postprocess.h"
#include "string_utils.h"

#include "falcor2/core/error.h"

#include <MaterialXCore/Util.h>
#include <MaterialXGenShader/Util.h>

#include <filesystem>
#include <fstream>
#include <mutex>
#include <regex>
#include <sstream>
#include <unordered_map>

namespace mx = MaterialX;

namespace falcor {
namespace materialx {
namespace mx139 {
namespace codegen_support {

namespace {

void scan_for_leftover_tokens(const std::string& source, std::string_view snippet_name)
{
    static const std::regex leftover_pattern("\\$Mx[A-Za-z0-9]+");
    std::smatch match;
    if (std::regex_search(source, match, leftover_pattern)) {
        FALCOR_THROW(
            "MX139 render_snippet('{}'): leftover token '{}' in rendered source (no value supplied).",
            snippet_name,
            match.str()
        );
    }
}

} // namespace

std::string load_snippet_source(std::string_view snippet_name)
{
#if !defined(MX139_SNIPPET_PATH)
    FALCOR_THROW("MX139 rewrite snippet path is not configured.");
    return {};
#else
    static std::unordered_map<std::string, std::string> cache;
    static std::mutex cache_mutex;

    std::string key(snippet_name);
    {
        std::scoped_lock lock(cache_mutex);
        auto it = cache.find(key);
        if (it != cache.end())
            return it->second;
    }

    const std::filesystem::path path = std::filesystem::path(MX139_SNIPPET_PATH) / key;
    std::ifstream stream(path, std::ios::binary);
    FALCOR_CHECK(stream.good(), "Unable to read MX139 rewrite Slang snippet '{}'.", path.string());

    std::stringstream buffer;
    buffer << stream.rdbuf();
    FALCOR_CHECK(!stream.bad(), "Failed while reading MX139 rewrite Slang snippet '{}'.", path.string());

    std::string text = buffer.str();
    replace_all(text, "\r\n", "\n");
    replace_all(text, "\r", "\n");
    strip_slang_snippet_file_header(text);
    strip_slang_snippet_module_marker(text);
    while (!text.empty() && text.back() == '\n')
        text.pop_back();
    text.push_back('\n');

    std::scoped_lock lock(cache_mutex);
    auto [inserted_it, inserted] = cache.emplace(std::move(key), std::move(text));
    FALCOR_UNUSED(inserted);
    return inserted_it->second;
#endif
}

std::string render_snippet(std::string_view snippet_name, const TokenMap& tokens)
{
    std::string source = load_snippet_source(snippet_name);
    if (tokens.empty()) {
        scan_for_leftover_tokens(source, snippet_name);
        return source;
    }

    static const std::regex key_shape("^\\$Mx[A-Za-z0-9]+$");
    mx::StringMap mx_tokens;
    for (const auto& [key, value] : tokens) {
        FALCOR_CHECK(
            std::regex_match(key, key_shape),
            "MX139 render_snippet('{}'): invalid token key '{}'. Keys must match ^\\$Mx[A-Za-z0-9]+$.",
            snippet_name,
            key
        );
        mx_tokens.emplace(key, value);
    }
    mx::tokenSubstitution(mx_tokens, source);
    scan_for_leftover_tokens(source, snippet_name);
    return source;
}

void ensure_snippet_emitted(mx::ShaderStage& stage, SnippetEmitState& emit_state, std::string_view snippet_name)
{
    if (!emit_state.emitted_snippets.insert(std::string(snippet_name)).second)
        return;
    std::string source = stage.getSourceCode();
    source += load_snippet_source(snippet_name);
    stage.setSourceCode(source);
}

} // namespace codegen_support
} // namespace mx139
} // namespace materialx
} // namespace falcor
