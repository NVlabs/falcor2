// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "source_code_node.h"
#include <sgl/core/hash.h>

#include <MaterialXCore/Definition.h>
#include <MaterialXFormat/Util.h>
#include <MaterialXGenShader/Exception.h>
#include <MaterialXGenShader/GenContext.h>
#include <MaterialXGenShader/Nodes/SourceCodeNode.h>
#include <MaterialXGenShader/ShaderNode.h>

#include <array>
#include <mutex>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <utility>

namespace falcor::mtlx::structured {

namespace mx = MaterialX;

namespace {

class SourceCodeCache {
    using Key = std::tuple<std::string, std::string>;
    using KeyView = std::tuple<std::string_view, std::string_view>;

    struct KeyHash {
        using is_transparent = void;
        size_t operator()(const KeyView& key) const { return sgl::hash(std::get<0>(key), std::get<1>(key)); }
    };

public:
    struct Source {
        mx::FilePath filename;
        std::string text;
    };

    std::shared_ptr<const Source> read(const std::string& file, const std::string& uri, mx::GenContext& context)
    {
        // All requests use the standard fixed search paths. Keep local source
        // locations distinct even when implementations request the same filename.
        const KeyView key{uri, file};
        {
            std::lock_guard lock(m_mutex);
            if (auto it = m_sources.find(key); it != m_sources.end())
                return it->second;
        }
        const mx::FilePath filename = context.resolveSourceFile(file, mx::FilePath(uri).getParentPath());
        auto loaded = std::make_shared<Source>(Source{filename, mx::readFile(filename)});
        // Library shader files are fixed for the process lifetime, including
        // source-tree fallbacks. Restart after editing them.
        if (loaded->text.empty())
            return loaded;
        // Duplicate first reads are harmless. No I/O or source copy under the lock.
        std::lock_guard lock(m_mutex);
        return m_sources.try_emplace(Key{uri, file}, std::move(loaded)).first->second;
    }

private:
    std::mutex m_mutex;
    std::unordered_map<Key, std::shared_ptr<const Source>, KeyHash, std::equal_to<>> m_sources;
};

SourceCodeCache& source_code_cache()
{
    static SourceCodeCache cache;
    return cache;
}

FormalNodeOperation operation_from_category(std::string_view category)
{
    static constexpr std::array<std::pair<std::string_view, FormalNodeOperation>, 20> OPERATIONS = {{
        {"constant", FormalNodeOperation::constant},
        {"convert", FormalNodeOperation::convert},
        {"multiply", FormalNodeOperation::multiply},
        {"add", FormalNodeOperation::add},
        {"subtract", FormalNodeOperation::subtract},
        {"divide", FormalNodeOperation::divide},
        {"mix", FormalNodeOperation::mix},
        {"layer", FormalNodeOperation::layer},
        {"weighted_layer", FormalNodeOperation::weighted_layer},
        {"clamp", FormalNodeOperation::clamp},
        {"min", FormalNodeOperation::minimum},
        {"max", FormalNodeOperation::maximum},
        {"absval", FormalNodeOperation::absolute_value},
        {"invert", FormalNodeOperation::invert},
        {"dot", FormalNodeOperation::dot},
        {"luminance", FormalNodeOperation::luminance},
        {"extract", FormalNodeOperation::extract},
        {"uniform_edf", FormalNodeOperation::uniform_edf},
        {"generalized_schlick_edf", FormalNodeOperation::generalized_schlick_edf},
        {"surface", FormalNodeOperation::surface},
    }};
    for (const auto& [name, operation] : OPERATIONS) {
        if (category == name)
            return operation;
    }
    return FormalNodeOperation::unknown;
}

class GenslangPtSourceCodeNode final : public mx::SourceCodeNode {
public:
    void initialize(const mx::InterfaceElement& element, mx::GenContext& context) override
    {
        mx::SourceCodeNode::initialize(element, context);
        // MaterialX's SourceCodeNode retains the implementation but not its
        // NodeDef. Capture the formal category here while the relationship is
        // explicit, so later analysis never reconstructs semantics from the
        // implementation filename.
        const auto& implementation = static_cast<const mx::Implementation&>(element);
        const mx::ConstNodeDefPtr node_def = implementation.getNodeDef();
        m_operation = node_def ? operation_from_category(node_def->getNodeString()) : FormalNodeOperation::unknown;
        if (node_def) {
            const std::string& category = node_def->getNodeString();
            m_has_geometry_selectors = category == "position" || category == "normal" || category == "tangent"
                || category == "bitangent" || category == "transformpoint" || category == "transformvector"
                || category == "transformnormal" || category == "geompropvalueuniform";
        }
    }

    bool isEditable(const mx::ShaderInput& input) const override
    {
        if (m_has_geometry_selectors) {
            const std::string& name = input.getName();
            if (name == "index" || name == "space" || name == "fromspace" || name == "tospace" || name == "geomprop")
                return false;
        }
        return mx::SourceCodeNode::isEditable(input);
    }

    FormalNodeOperation operation() const { return m_operation; }

private:
    void resolveSourceCode(const mx::InterfaceElement& element, mx::GenContext& context) override
    {
        const auto source
            = source_code_cache().read(element.getAttribute("file"), element.getActiveSourceUri(), context);
        _sourceFilename = source->filename;
        _functionSource = source->text;
        if (_functionSource.empty())
            throw mx::ExceptionShaderGenError(
                "Failed to get source code from file '" + _sourceFilename.asString() + "' used by implementation '"
                + element.getName() + "'"
            );
    }

    FormalNodeOperation m_operation = FormalNodeOperation::unknown;
    bool m_has_geometry_selectors = false;
};

} // namespace

mx::ShaderNodeImplPtr create_source_code_node()
{
    return std::make_shared<GenslangPtSourceCodeNode>();
}

FormalNodeOperation formal_node_operation(const mx::ShaderNode& node)
{
    const auto* implementation = dynamic_cast<const GenslangPtSourceCodeNode*>(&node.getImplementation());
    return implementation ? implementation->operation() : FormalNodeOperation::unknown;
}

} // namespace falcor::mtlx::structured
