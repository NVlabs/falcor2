// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "material_network.h"

#include "falcor2/core/error.h"

namespace falcor {
namespace importer_material {

NodeNetwork::NodeNetwork(const std::vector<Properties>* nodes)
    : m_properties(nodes)
{
    FALCOR_CHECK(m_properties && !m_properties->empty(), "Cannot build NodeNetwork on empty Properties.");
}

Node NodeNetwork::get_node(std::string_view path) const
{
    FALCOR_CHECK(!path.empty(), "Path cannot be empty");

    auto it = m_path_cache.find(path);
    if (it != m_path_cache.end())
        return Node(this, it->second);

    for (const auto& props : *m_properties) {
        std::string_view candidate_path = props.get<std::string_view>("_path", "");
        if (candidate_path == path) {
            auto [inserted_it, _] = m_path_cache.emplace(std::string(path), &props);
            return Node(this, inserted_it->second);
        }
    }

    return {};
}

Node NodeNetwork::terminal() const
{
    return Node(this, &m_properties->back());
}

NodeConnection::NodeConnection(const NodeNetwork* network, std::string_view connection)
{
    const size_t dot_pos = connection.find('.');
    FALCOR_CHECK(
        dot_pos != std::string::npos,
        "Value expected to be to contain `.`, it is a connection. Actual value: {}",
        connection
    );

    std::string_view path = connection.substr(0, dot_pos);
    m_source = network->get_node(path);
    m_output = connection.substr(dot_pos + 1);
    FALCOR_CHECK(m_source.is_valid(), "Path {} does not lead to any known node.", path);
}

NodeInput::NodeInput(Properties::iterator it, const Node& node)
{
    if (it.name().starts_with("info:"))
        return;
    if (it.name().ends_with(":colorspace"))
        return;
    if (it.name().ends_with(":typename"))
        return;
    if (it.name() == "_path" || it.name() == "_type")
        return;

    m_property = it;
    m_node = node;
    if (auto pos = it.name().find(":connect"); pos != std::string_view::npos) {
        FALCOR_CHECK(
            it.type() == PropertyType::string,
            "{} value type expected to be a string, it is a connection.",
            it.name()
        );
        m_connection = NodeConnection(node.m_network, it.get<std::string_view>());
        m_name = it.name().substr(0, pos);
    } else {
        m_name = it.name();
    }

    std::string type_name(std::string(m_name) + ":typename");
    auto type_it = node.m_props->find(type_name);
    FALCOR_CHECK(
        type_it != node.m_props->end(),
        "Missing required '{}' property on node input '{}'.",
        type_name,
        m_name
    );
    m_type = type_it.get<std::string_view>();

    std::string colorspace_name(std::string(m_name) + ":colorspace");
    auto colorspace_it = node.m_props->find(colorspace_name);
    if (colorspace_it != node.m_props->end())
        m_colorspace = colorspace_it.get<std::string_view>();
}

NodeInput Node::get_input(std::string_view name) const
{
    auto it = m_props->find(name);
    if (it != m_props->end()) {
        return NodeInput(it, *this);
    }

    std::string connection_name(name);
    connection_name += ":connect";
    it = m_props->find(connection_name);
    if (it != m_props->end()) {
        return NodeInput(it, *this);
    }

    return {};
}

Node::iterator::iterator(const Node& node, Properties::iterator it, Properties::iterator end)
    : m_node(node)
    , m_it(it)
    , m_end(end)
{
    advance_to_valid();
}

const NodeInput& Node::iterator::operator*() const
{
    return m_current;
}

const NodeInput* Node::iterator::operator->() const
{
    return &m_current;
}

Node::iterator& Node::iterator::operator++()
{
    if (m_is_end)
        return *this;

    ++m_it;
    advance_to_valid();
    return *this;
}

bool Node::iterator::operator==(const iterator& other) const
{
    if (m_is_end && other.m_is_end)
        return true;

    return m_is_end == other.m_is_end && m_node.m_props == other.m_node.m_props && m_it == other.m_it;
}

void Node::iterator::advance_to_valid()
{
    if (!m_node || m_it == m_end) {
        m_is_end = true;
        m_current = {};
        return;
    }

    while (m_it != m_end) {
        m_current = NodeInput(m_it, m_node);
        if (m_current) {
            m_is_end = false;
            return;
        }
        ++m_it;
    }

    m_is_end = true;
    m_current = {};
}

Node::iterator Node::begin() const
{
    if (!is_valid())
        return end();
    return iterator(*this, m_props->begin(), m_props->end());
}

Node::iterator Node::end() const
{
    if (!is_valid())
        return iterator();
    return iterator(*this, m_props->end(), m_props->end());
}

} // namespace importer_material
} // namespace falcor
