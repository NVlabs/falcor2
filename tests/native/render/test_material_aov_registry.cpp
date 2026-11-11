// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "testing.h"
#include "falcor2/render/material_aov_registry.h"

using namespace falcor;

TEST_CASE("material AOV registry retains permanent typed IDs")
{
    auto registry = make_ref<MaterialAOVRegistry>();
    CHECK_FALSE(registry->find_output("missing"));
    CHECK_FALSE(registry->has_output("missing"));
    CHECK(registry->outputs().empty());
    CHECK_EQ(registry->register_output("mask", sgl::DataType::float32, 1), 0u);
    CHECK_EQ(registry->register_output("mask", sgl::DataType::float32, 1), 0u);
    auto snapshot = registry->outputs();
    REQUIRE_EQ(snapshot.size(), 1u);
    CHECK_EQ(snapshot[0].id, 0u);
    CHECK_EQ(snapshot[0].type, sgl::DataType::float32);
    CHECK(registry->find_output("mask") == snapshot[0]);
    CHECK_THROWS_WITH(
        registry->register_output("mask", sgl::DataType::float32, 2),
        doctest::Contains("incompatible value type")
    );
    CHECK_THROWS(registry->register_output("", sgl::DataType::float32, 1));
    CHECK_THROWS(registry->register_output("bad", sgl::DataType::uint32, 1));
    CHECK_THROWS(registry->register_output("bad", sgl::DataType::float32, 0));
    CHECK_THROWS(registry->register_output("bad", sgl::DataType::float32, 5));
    CHECK(registry->outputs() == snapshot);
    CHECK_EQ(registry->register_output("Mask", sgl::DataType::float32, 2), 1u);
    CHECK_EQ(registry->output_generation(), 2u);
    CHECK_EQ(registry->find_output("mask")->id, 0u);
    CHECK_EQ(snapshot[0].component_count, 1u);
    registry = nullptr;
    CHECK_EQ(snapshot[0].name, "mask");
}
