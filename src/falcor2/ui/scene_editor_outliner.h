// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/core/macros.h"
#include "falcor2/render/fwd.h"

#include <algorithm>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace falcor::ui::detail {

/// Open state used while drawing a tree node under an active filter.
struct OutlinerTreeOpenState {
    bool draw_open;
    bool stored_open;
};

/// Resolve transient and persistent requests to open an outliner tree node.
inline OutlinerTreeOpenState outliner_tree_open_state(bool stored_open, bool force_open, bool persist_open = false)
{
    bool updated_stored_open = stored_open || persist_open;
    return {updated_stored_open || force_open, updated_stored_open};
}

/// Return true if @p ancestor is a strict ancestor of @p entity.
FALCOR_API bool is_entity_ancestor(const Entity* ancestor, const Entity* entity);

/// Parsed scene outliner search query.
/// Plain tokens match object names and "t:" tokens match entity component types.
class FALCOR_API OutlinerQuery {
public:
    explicit OutlinerQuery(std::string_view query);

    bool empty() const { return m_name_terms.empty() && m_component_terms.empty(); }

    /// Return true if a non-entity scene object matches the query.
    bool matches_object(const SceneObject* object) const;

    /// Return true if an entity matches the query directly.
    bool matches_entity(const Entity* entity) const;

    /// Add matching entities and their ancestors to @p visible_entities.
    /// Returns true if @p entity or any of its descendants matches the query.
    bool collect_visible_entities(const Entity* entity, std::unordered_set<const Entity*>& visible_entities) const;

private:
    bool matches_name(std::string_view name) const;

    std::vector<std::string> m_name_terms;
    std::vector<std::string> m_component_terms;
};

/// Stable sort scene objects by name.
template<typename TObject>
void sort_outliner_objects(std::vector<TObject*>& objects)
{
    std::stable_sort(
        objects.begin(),
        objects.end(),
        [](const TObject* lhs, const TObject* rhs)
        {
            return lhs->name() < rhs->name();
        }
    );
}

} // namespace falcor::ui::detail
