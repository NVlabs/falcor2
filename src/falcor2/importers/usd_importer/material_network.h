// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/core/properties.h"

#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace falcor {
namespace importer_material {

class NodeNetwork;
class NodeInput;

class Node {
    friend class NodeNetwork;
    friend class NodeInput;
    friend class NodeConnection;

public:
    class iterator;

    Node() = default;

    bool is_valid() const { return m_network && m_props; }
    explicit operator bool() const { return is_valid(); }

    NodeInput get_input(std::string_view name) const;
    iterator begin() const;
    iterator end() const;

    const Properties& props() const { return *m_props; }

private:
    Node(const NodeNetwork* network, const Properties* props)
        : m_network(network)
        , m_props(props)
    {
    }

    const NodeNetwork* m_network = nullptr;
    const Properties* m_props = nullptr;
};

class NodeNetwork {
public:
    explicit NodeNetwork(const std::vector<Properties>* nodes);

    Node get_node(std::string_view path) const;
    Node terminal() const;

private:
    const std::vector<Properties>* m_properties;
    mutable std::map<std::string, const Properties*, std::less<>> m_path_cache;
};

class NodeConnection {
public:
    NodeConnection() = default;

    // Node the NodeInput is connected to.
    const Node& source() const { return m_source; }
    // Name of the output NodeInput is connected to (e.g., `r` on a texture).
    std::string_view output() const { return m_output; }

    NodeConnection(const NodeNetwork* network, std::string_view connection);

private:
    Node m_source;
    std::string m_output;
};

class NodeInput {
    friend class Node;

public:
    NodeInput() = default;

    // Property can have a mangled name, use name instead.
    const Properties::iterator& property() const { return m_property; }
    std::string_view name() const { return m_name; }
    std::string_view type() const { return m_type; }
    const std::optional<std::string_view>& colorspace() const { return m_colorspace; }
    const std::optional<NodeConnection>& connection() const { return m_connection; }
    const Node& node() const { return m_node; }

    explicit operator bool() const { return is_valid(); }
    bool is_valid() const { return m_node.is_valid(); }

private:
    NodeInput(Properties::iterator it, const Node& node);

private:
    Properties::iterator m_property;
    std::string_view m_name;
    std::string_view m_type;
    std::optional<std::string_view> m_colorspace;
    std::optional<NodeConnection> m_connection;
    Node m_node;
};

class Node::iterator {
public:
    iterator() = default;

    const NodeInput& operator*() const;
    const NodeInput* operator->() const;

    iterator& operator++();
    bool operator==(const iterator& other) const;
    bool operator!=(const iterator& other) const { return !(*this == other); }

private:
    friend class Node;
    iterator(const Node& node, Properties::iterator it, Properties::iterator end);

    void advance_to_valid();

    Node m_node;
    Properties::iterator m_it;
    Properties::iterator m_end;
    NodeInput m_current;
    bool m_is_end = true;
};

} // namespace importer_material
} // namespace falcor
