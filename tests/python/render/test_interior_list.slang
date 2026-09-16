// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "falcor2/testing/shader_test.slangh"

import falcor2.render;

BasicInteriorListEntry make_entry(uint material_id, uint nested_priority, float interior_ior)
{
    return BasicInteriorListEntry(MaterialID(material_id), nested_priority, interior_ior, {});
}

HomogeneousMediumProperties make_medium(float3 sigma_a, float3 sigma_s, float anisotropy)
{
    HomogeneousMediumProperties result = {};
    result.sigma_a = sigma_a;
    result.sigma_s = sigma_s;
    result.anisotropy = anisotropy;
    return result;
}

HomogeneousInteriorListEntry
make_medium_entry(uint material_id, uint nested_priority, float interior_ior, HomogeneousMediumProperties medium)
{
    return HomogeneousInteriorListEntry(MaterialID(material_id), nested_priority, interior_ior, medium);
}

struct ExplicitEmptyInteriorListEntry : IInteriorListEntry
{
    MaterialID stored_material_id = MaterialID(0);
    uint stored_nested_priority = 0;
    float stored_interior_ior = 1.f;
    HomogeneousMediumProperties stored_interior_medium = {};
    bool occupied = false;

    property MaterialID material_id
    {
        get {
            return stored_material_id;
        }
    }
    property uint nested_priority
    {
        get {
            return stored_nested_priority;
        }
    }
    property float interior_ior
    {
        get {
            return stored_interior_ior;
        }
    }
    property HomogeneousMediumProperties interior_medium
    {
        get {
            return stored_interior_medium;
        }
    }

    __init(
        MaterialID material_id,
        uint nested_priority,
        float interior_ior,
        HomogeneousMediumProperties interior_medium
    )
    {
        stored_material_id = material_id;
        stored_nested_priority = nested_priority;
        stored_interior_ior = interior_ior;
        stored_interior_medium = interior_medium;
        occupied = true;
    }

    bool is_empty() { return !occupied; }
    bool matches(MaterialID candidate_material_id) { return occupied && material_id == candidate_material_id; }

    [mutating]
    void clear() { occupied = false; }
}

TEST_CASE(interior_list_default_state)
{
    InteriorListImpl<BasicInteriorListEntry, 4> list = {};
    CHECK(list.is_empty());
    CHECK_FALSE(list.contains_only(MaterialID(1)));
    CHECK_EQ(uint(list.top_material_id()), uint(MaterialID::invalid));
    CHECK(list.entries[0].is_empty());
    CHECK(list.is_true_interface(1));
    CHECK_EQ(list.active_medium().sigma_a, float3(0.f));
    CHECK_EQ(list.active_medium().sigma_s, float3(0.f));
}

TEST_CASE(interior_list_reports_sole_material)
{
    InteriorListImpl<BasicInteriorListEntry, 4> list = {};

    CHECK_EQ(list.enter(make_entry(1, 1, 1.1)), InteriorListUpdateResult::success);
    CHECK(list.contains_only(MaterialID(1)));
    CHECK_FALSE(list.contains_only(MaterialID(2)));

    CHECK_EQ(list.enter(make_entry(2, 2, 1.2)), InteriorListUpdateResult::success);
    CHECK_FALSE(list.contains_only(MaterialID(1)));
    CHECK_FALSE(list.contains_only(MaterialID(2)));

    CHECK_EQ(list.exit(MaterialID(2)), InteriorListUpdateResult::success);
    CHECK(list.contains_only(MaterialID(1)));
}

TEST_CASE(interior_list_uses_entry_empty_state)
{
    InteriorListImpl<ExplicitEmptyInteriorListEntry, 2> list = {};
    CHECK_EQ(uint(list.top_material_id()), uint(MaterialID::invalid));

    CHECK_EQ(list.enter(ExplicitEmptyInteriorListEntry(MaterialID(1), 2, 1.5, {})), InteriorListUpdateResult::success);
    CHECK_FALSE(list.is_empty());
    CHECK_FALSE(list.entries[0].is_empty());
    CHECK(list.contains(MaterialID(1)));

    CHECK_EQ(list.exit(MaterialID(1)), InteriorListUpdateResult::success);
    CHECK(list.is_empty());
    CHECK(list.entries[0].is_empty());
    CHECK_FALSE(list.contains(MaterialID(1)));
}

TEST_CASE(interior_list_entry_packs_valid_ior_range)
{
    CHECK_EQ(float(make_entry(1, 1, 1.).interior_ior), 1.);
    CHECK_EQ(float(make_entry(1, 1, 1.5).interior_ior), 1.5);
    CHECK_EQ(float(make_entry(1, 1, 65504.).interior_ior), 65504.);
}

TEST_CASE(homogeneous_interior_entry_packs_medium)
{
    let entry = make_medium_entry(1, 2, 1.5, make_medium(float3(0.125f, 1.f, 12.f), float3(0.25f, 2.f, 24.f), 0.75f));
    let medium = entry.interior_medium;

    CHECK_APPROX(medium.sigma_a.x, 0.125f, 1e-3f);
    CHECK_APPROX(medium.sigma_a.y, 1.f, 1e-3f);
    CHECK_APPROX(medium.sigma_a.z, 12.f, 1e-3f);
    CHECK_APPROX(medium.sigma_s.x, 0.25f, 1e-3f);
    CHECK_APPROX(medium.sigma_s.y, 2.f, 1e-3f);
    CHECK_APPROX(medium.sigma_s.z, 24.f, 1e-3f);
    CHECK_APPROX(medium.anisotropy, 0.75f, 1e-3f);
}

TEST_CASE(homogeneous_interior_entry_packs_valid_medium_range)
{
    let entry
        = make_medium_entry(1, 2, 1.5, make_medium(float3(0.f, 0.125f, 65504.f), float3(65504.f, 0.25f, 0.f), -0.999f));
    let medium = entry.interior_medium;

    CHECK_EQ(medium.sigma_a.x, 0.f);
    CHECK_APPROX(medium.sigma_a.y, 0.125f, 1e-3f);
    CHECK_EQ(medium.sigma_a.z, 65504.f);
    CHECK_EQ(medium.sigma_s.x, 65504.f);
    CHECK_APPROX(medium.sigma_s.y, 0.25f, 1e-3f);
    CHECK_EQ(medium.sigma_s.z, 0.f);
    CHECK_APPROX(medium.anisotropy, -0.999f, 1e-3f);
}

TEST_CASE(interior_list_selects_active_medium)
{
    InteriorListImpl<HomogeneousInteriorListEntry, 2> list = {};
    let low_medium = make_medium(float3(0.1f), float3(0.2f), -0.25f);
    let high_medium = make_medium(float3(1.f), float3(2.f), 0.75f);

    CHECK_EQ(list.enter(make_medium_entry(1, 1, 1.1, low_medium)), InteriorListUpdateResult::success);
    CHECK_APPROX(list.active_medium().sigma_a.x, 0.1f, 1e-3f);

    CHECK_EQ(list.enter(make_medium_entry(2, 3, 1.2, high_medium)), InteriorListUpdateResult::success);
    CHECK_APPROX(list.active_medium().sigma_a.x, 1.f, 1e-3f);
    CHECK_APPROX(list.active_medium().sigma_s.x, 2.f, 1e-3f);
    CHECK_APPROX(list.active_medium().anisotropy, 0.75f, 1e-3f);

    CHECK_EQ(list.exit(MaterialID(2)), InteriorListUpdateResult::success);
    CHECK_APPROX(list.active_medium().sigma_a.x, 0.1f, 1e-3f);
    CHECK_APPROX(list.active_medium().anisotropy, -0.25f, 1e-3f);

    CHECK_EQ(list.exit(MaterialID(1)), InteriorListUpdateResult::success);
    CHECK_EQ(list.active_medium().sigma_a, float3(0.f));
    CHECK_EQ(list.active_medium().sigma_s, float3(0.f));
}

TEST_CASE(interior_list_orders_priorities_and_ties)
{
    InteriorListImpl<BasicInteriorListEntry, 4> list = {};

    CHECK_EQ(list.enter(make_entry(1, 1, 1.1)), InteriorListUpdateResult::success);
    CHECK_EQ(list.enter(make_entry(2, 3, 1.2)), InteriorListUpdateResult::success);
    CHECK_EQ(list.enter(make_entry(3, 0, 1.3)), InteriorListUpdateResult::success);
    CHECK_EQ(list.enter(make_entry(4, 3, 1.4)), InteriorListUpdateResult::success);

    CHECK_EQ(uint(list.entries[0].material_id), 3);
    CHECK_EQ(uint(list.entries[1].material_id), 4);
    CHECK_EQ(uint(list.entries[2].material_id), 2);
    CHECK_EQ(uint(list.entries[3].material_id), 1);
    CHECK_EQ(uint(list.entries[0].nested_priority), 0);
    CHECK_APPROX(float(list.entries[1].interior_ior), 1.4f, 1e-3);
}

TEST_CASE(interior_list_supports_configured_sorting_networks)
{
    InteriorListImpl<BasicInteriorListEntry, 1> list1 = {};
    CHECK_EQ(list1.enter(make_entry(1, 1, 1.1)), InteriorListUpdateResult::success);
    CHECK_EQ(list1.enter(make_entry(2, 2, 1.2)), InteriorListUpdateResult::capacity_exceeded);

    InteriorListImpl<BasicInteriorListEntry, 2> list2 = {};
    CHECK_EQ(list2.enter(make_entry(1, 1, 1.1)), InteriorListUpdateResult::success);
    CHECK_EQ(list2.enter(make_entry(2, 2, 1.2)), InteriorListUpdateResult::success);
    CHECK_EQ(uint(list2.entries[0].material_id), 2);
    CHECK_EQ(uint(list2.entries[1].material_id), 1);

    InteriorListImpl<BasicInteriorListEntry, 3> list3 = {};
    CHECK_EQ(list3.enter(make_entry(1, 1, 1.1)), InteriorListUpdateResult::success);
    CHECK_EQ(list3.enter(make_entry(2, 3, 1.2)), InteriorListUpdateResult::success);
    CHECK_EQ(list3.enter(make_entry(3, 2, 1.3)), InteriorListUpdateResult::success);
    CHECK_EQ(uint(list3.entries[0].material_id), 2);
    CHECK_EQ(uint(list3.entries[1].material_id), 3);
    CHECK_EQ(uint(list3.entries[2].material_id), 1);
}

TEST_CASE(interior_list_reports_structural_failures_without_mutation)
{
    InteriorListImpl<BasicInteriorListEntry, 4> list = {};
    CHECK_EQ(list.enter(make_entry(1, 1, 1.1)), InteriorListUpdateResult::success);
    CHECK_EQ(list.exit(MaterialID(2)), InteriorListUpdateResult::entry_not_found);
    CHECK_EQ(uint(list.top_material_id()), 1);

    CHECK_EQ(list.enter(make_entry(2, 3, 1.2)), InteriorListUpdateResult::success);
    CHECK_EQ(list.enter(make_entry(3, 0, 1.3)), InteriorListUpdateResult::success);
    CHECK_EQ(list.enter(make_entry(4, 3, 1.4)), InteriorListUpdateResult::success);
    CHECK_EQ(list.enter(make_entry(5, 2, 1.5)), InteriorListUpdateResult::capacity_exceeded);
    CHECK_FALSE(list.contains(MaterialID(5)));
}

TEST_CASE(interior_list_tracks_duplicate_material_entries)
{
    InteriorListImpl<BasicInteriorListEntry, 4> list = {};

    CHECK_EQ(list.enter(make_entry(1, 2, 1.5)), InteriorListUpdateResult::success);
    CHECK_APPROX(list.exterior_ior(MaterialID(1), true), 1.5f, 1e-3);

    CHECK_EQ(list.enter(make_entry(1, 2, 1.5)), InteriorListUpdateResult::success);
    CHECK_EQ(uint(list.entries[0].material_id), 1);
    CHECK_EQ(uint(list.entries[1].material_id), 1);
    CHECK_APPROX(list.exterior_ior(MaterialID(1), false), 1.5f, 1e-3);

    CHECK_EQ(list.exit(MaterialID(1)), InteriorListUpdateResult::success);
    CHECK(list.contains(MaterialID(1)));
    CHECK_APPROX(list.exterior_ior(MaterialID(1), false), 1.f, 1e-3);

    CHECK_EQ(list.exit(MaterialID(1)), InteriorListUpdateResult::success);
    CHECK_FALSE(list.contains(MaterialID(1)));
    CHECK_EQ(list.exit(MaterialID(1)), InteriorListUpdateResult::entry_not_found);
}

TEST_CASE(interior_list_removes_resorts_and_computes_exterior_ior)
{
    InteriorListImpl<BasicInteriorListEntry, 4> list = {};
    CHECK_EQ(list.exterior_ior(MaterialID(1), true), 1.);

    CHECK_EQ(list.enter(make_entry(1, 1, 1.1)), InteriorListUpdateResult::success);
    CHECK_EQ(list.enter(make_entry(2, 3, 1.2)), InteriorListUpdateResult::success);
    CHECK_APPROX(list.exterior_ior(MaterialID(4), true), 1.2f, 1e-3);
    CHECK_APPROX(list.exterior_ior(MaterialID(2), false), 1.1f, 1e-3);
    CHECK_APPROX(list.exterior_ior(MaterialID(1), false), 1.2f, 1e-3);

    CHECK_EQ(list.exit(MaterialID(2)), InteriorListUpdateResult::success);
    CHECK_EQ(uint(list.top_material_id()), 1);
    CHECK(list.entries[1].is_empty());
    CHECK_EQ(list.exterior_ior(MaterialID(1), false), 1.);
}

TEST_CASE(interior_list_classifies_true_and_false_interfaces)
{
    InteriorListImpl<BasicInteriorListEntry, 4> list = {};
    CHECK(list.is_true_interface(1));

    CHECK_EQ(list.enter(make_entry(2, 3, 1.2)), InteriorListUpdateResult::success);
    CHECK_FALSE(list.is_true_interface(1));
    CHECK(list.is_true_interface(3));
    CHECK(list.is_true_interface(0));
}
