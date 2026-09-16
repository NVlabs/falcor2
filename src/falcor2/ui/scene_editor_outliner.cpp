// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "falcor2/ui/scene_editor_outliner.h"

#include "falcor2/render/component.h"
#include "falcor2/render/entity.h"
#include "falcor2/render/scene_object.h"

#include <cctype>

namespace falcor::ui::detail {
namespace {

char ascii_lower(char c)
{
    if (c >= 'A' && c <= 'Z')
        return c - 'A' + 'a';
    return c;
}

bool contains_case_insensitive(std::string_view text, std::string_view search)
{
    return std::search(
               text.begin(),
               text.end(),
               search.begin(),
               search.end(),
               [](char lhs, char rhs)
               {
                   return ascii_lower(lhs) == ascii_lower(rhs);
               }
           )
        != text.end();
}

} // namespace

bool is_entity_ancestor(const Entity* ancestor, const Entity* entity)
{
    for (const Entity* parent = entity ? entity->parent() : nullptr; parent; parent = parent->parent()) {
        if (parent == ancestor)
            return true;
    }
    return false;
}

OutlinerQuery::OutlinerQuery(std::string_view query)
{
    size_t token_begin = 0;
    while (token_begin < query.size()) {
        while (token_begin < query.size() && std::isspace(static_cast<unsigned char>(query[token_begin])))
            ++token_begin;
        if (token_begin == query.size())
            break;

        size_t token_end = token_begin;
        while (token_end < query.size() && !std::isspace(static_cast<unsigned char>(query[token_end])))
            ++token_end;

        std::string_view token = query.substr(token_begin, token_end - token_begin);
        if (token.size() > 2 && ascii_lower(token[0]) == 't' && token[1] == ':')
            m_component_terms.emplace_back(token.substr(2));
        else if (token != "t:" && token != "T:")
            m_name_terms.emplace_back(token);

        token_begin = token_end;
    }
}

bool OutlinerQuery::matches_name(std::string_view name) const
{
    return std::ranges::all_of(
        m_name_terms,
        [&](const std::string& term)
        {
            return contains_case_insensitive(name, term);
        }
    );
}

bool OutlinerQuery::matches_object(const SceneObject* object) const
{
    return object && object->is_valid() && m_component_terms.empty() && matches_name(object->name());
}

bool OutlinerQuery::matches_entity(const Entity* entity) const
{
    if (!entity || !entity->is_valid() || !matches_name(entity->name()))
        return false;

    return std::ranges::all_of(
        m_component_terms,
        [&](const std::string& term)
        {
            return std::ranges::any_of(
                entity->components(),
                [&](const Component* component)
                {
                    return component->is_valid() && contains_case_insensitive(component->class_name(), term);
                }
            );
        }
    );
}

bool OutlinerQuery::collect_visible_entities(
    const Entity* entity,
    std::unordered_set<const Entity*>& visible_entities
) const
{
    if (!entity || !entity->is_valid())
        return false;

    bool visible = matches_entity(entity);
    for (const Entity* child : entity->children())
        visible |= collect_visible_entities(child, visible_entities);

    if (visible)
        visible_entities.insert(entity);
    return visible;
}

} // namespace falcor::ui::detail
