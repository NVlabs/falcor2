// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "testing.h"

#include "falcor2/core/reflection.h"

using namespace falcor;
using namespace falcor::reflection;

namespace {
enum class TestEnum : int {
    off = 0,
    on = 1,
};
} // namespace

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST_SUITE_BEGIN("reflection");

// --- DynamicPropertySet ---

TEST_CASE("DynamicPropertySet add/get/set/remove")
{
    DynamicPropertySet dps;

    dps.add_property<int>("dyn_int", 10, false, "A dynamic int.", default_value(10));
    dps.add_property<float>("dyn_float", 3.14f, false, "A dynamic float.", default_value(3.14f));
    dps.add_property<std::string>("dyn_str", std::string("hello"), false, default_value(std::string("hello")));
    dps.add_property<TestEnum>("dyn_enum", TestEnum::off, false, "A dynamic enum.", default_value(TestEnum::off));

    CHECK(dps.size() == 4);

    SUBCASE("get returns correct values")
    {
        Any v = dps.get_any("dyn_int");
        CHECK(*any_cast<int>(&v) == 10);

        Any vf = dps.get_any("dyn_float");
        CHECK(*any_cast<float>(&vf) == doctest::Approx(3.14f));

        Any vs = dps.get_any("dyn_str");
        CHECK(*any_cast<std::string>(&vs) == "hello");

        Any ve = dps.get_any("dyn_enum");
        CHECK(*any_cast<TestEnum>(&ve) == TestEnum::off);
    }

    SUBCASE("set updates value and fires callback")
    {
        int callback_count = 0;
        DynamicPropertySet dps2;
        dps2.add_property<int>(
            "val",
            10,
            false,
            on_change<int>(
                [&callback_count](int&)
                {
                    ++callback_count;
                }
            )
        );

        dps2.set_any("val", Any(42));
        Any v = dps2.get_any("val");
        CHECK(*any_cast<int>(&v) == 42);
        CHECK(callback_count == 1);
    }

    SUBCASE("remove_property")
    {
        dps.remove_property("dyn_float");
        CHECK(dps.size() == 3);
        CHECK(dps.find_property("dyn_float") == nullptr);
        CHECK(dps.find_property("dyn_int") != nullptr);
        CHECK(dps.find_property("dyn_str") != nullptr);
        CHECK(dps.find_property("dyn_enum") != nullptr);
    }

    SUBCASE("add_property throws on duplicate name")
    {
        CHECK_THROWS(dps.add_property<int>("dyn_int", 0));
    }

    SUBCASE("get throws on missing name")
    {
        CHECK_THROWS(dps.get_any("nonexistent"));
    }

    SUBCASE("set throws on missing name")
    {
        CHECK_THROWS(dps.set_any("nonexistent", Any(0)));
    }
}

TEST_CASE("DynamicPropertySet descriptors returns PropertyDescriptor pointers")
{
    DynamicPropertySet dps;
    dps.add_property<int>("a", 1);
    dps.add_property<float>("b", 2.0f);

    auto descs = dps.descriptors();
    CHECK(descs.size() == 2);
    CHECK(descs[0]->name() == "a");
    CHECK(descs[1]->name() == "b");

    // Both are PropertyDescriptor* -- same interface as TypedPropertyDescriptor.
    CHECK(descs[0]->type() == typeid(int));
    CHECK(descs[1]->type() == typeid(float));
}

TEST_CASE("DynamicPropertySet write_to_properties and read_from_properties")
{
    int callback_count = 0;
    auto count_cb = on_change<int>(
        [&callback_count](int&)
        {
            ++callback_count;
        }
    );
    // Use a separate counter for string properties since on_change<T> is typed.
    int str_callback_count = 0;
    auto str_count_cb = on_change<std::string>(
        [&str_callback_count](std::string&)
        {
            ++str_callback_count;
        }
    );
    int enum_callback_count = 0;
    auto enum_count_cb = on_change<TestEnum>(
        [&enum_callback_count](TestEnum&)
        {
            ++enum_callback_count;
        }
    );

    DynamicPropertySet dps;
    dps.add_property<int>("x", 42, false, default_value(0), count_cb);
    dps.add_property<std::string>("y", std::string("hello"), false, default_value(std::string("")), str_count_cb);
    dps.add_property<TestEnum>("mode", TestEnum::off, false, default_value(TestEnum::off), enum_count_cb);

    Properties props;
    dps.write_to_properties(props);

    CHECK(props.get<int>("x") == 42);
    CHECK(props.get<std::string>("y") == "hello");
    CHECK(props.get<TestEnum>("mode") == TestEnum::off);
    // Modify via read_from_properties
    Properties props2;
    props2.set<int>("x", 99);
    props2.set<std::string>("y", "world");
    props2.set<TestEnum>("mode", TestEnum::on);

    // Reset counters before read_from_properties.
    callback_count = 0;
    str_callback_count = 0;
    enum_callback_count = 0;

    dps.read_from_properties(props2);

    Any vx = dps.get_any("x");
    CHECK(*any_cast<int>(&vx) == 99);
    Any vy = dps.get_any("y");
    CHECK(*any_cast<std::string>(&vy) == "world");
    Any vm = dps.get_any("mode");
    CHECK(*any_cast<TestEnum>(&vm) == TestEnum::on);

    CHECK(callback_count == 1);
    CHECK(str_callback_count == 1);
    CHECK(enum_callback_count == 1);
}

TEST_CASE("DynamicPropertySet enum read_from_properties accepts integer values")
{
    DynamicPropertySet dps;
    dps.add_property<TestEnum>("mode", TestEnum::off, false, default_value(TestEnum::off));

    Properties props;
    props.set<int64_t>("mode", static_cast<int64_t>(TestEnum::on));

    dps.read_from_properties(props);

    CHECK(dps.get<TestEnum>("mode") == TestEnum::on);
}

TEST_CASE("DynamicPropertySet reset")
{
    DynamicPropertySet dps;
    dps.add_property<int>("x", 42, false, default_value(0));

    dps.reset();

    Any v = dps.get_any("x");
    CHECK(*any_cast<int>(&v) == 0);
}

TEST_CASE("DynamicPropertySet reset fires on-change callback")
{
    int callback_count = 0;
    DynamicPropertySet dps;
    dps.add_property<int>(
        "x",
        42,
        false,
        default_value(0),
        on_change<int>(
            [&callback_count](int&)
            {
                ++callback_count;
            }
        )
    );

    // reset() should fire the callback.
    dps.reset();
    CHECK(dps.get<int>("x") == 0);
    CHECK(callback_count == 1);

    // reset(name) should also fire the callback.
    dps.set<int>("x", 99);
    callback_count = 0;
    dps.reset("x");
    CHECK(dps.get<int>("x") == 0);
    CHECK(callback_count == 1);
}

TEST_CASE("DynamicPropertySet has_property")
{
    DynamicPropertySet dps;
    dps.add_property<int>("x", 10);

    CHECK(dps.has_property("x"));
    CHECK_FALSE(dps.has_property("nonexistent"));
}

TEST_CASE("DynamicPropertySet typed get/set")
{
    DynamicPropertySet dps;
    dps.add_property<int>("i", 10, false, default_value(10));
    dps.add_property<float>("f", 3.14f, false, default_value(3.14f));
    dps.add_property<std::string>("s", std::string("hello"), false, default_value(std::string("hello")));
    dps.add_property<TestEnum>("e", TestEnum::off, false, default_value(TestEnum::off));

    SUBCASE("typed get returns correct values")
    {
        CHECK(dps.get<int>("i") == 10);
        CHECK(dps.get<float>("f") == doctest::Approx(3.14f));
        CHECK(dps.get<std::string>("s") == "hello");
        CHECK(dps.get<TestEnum>("e") == TestEnum::off);
    }

    SUBCASE("typed set updates value")
    {
        dps.set<int>("i", 42);
        CHECK(dps.get<int>("i") == 42);

        dps.set<float>("f", 2.71f);
        CHECK(dps.get<float>("f") == doctest::Approx(2.71f));

        dps.set<std::string>("s", std::string("world"));
        CHECK(dps.get<std::string>("s") == "world");

        dps.set<TestEnum>("e", TestEnum::on);
        CHECK(dps.get<TestEnum>("e") == TestEnum::on);
    }

    SUBCASE("typed set fires on-change callback")
    {
        int callback_count = 0;
        DynamicPropertySet dps2;
        dps2.add_property<int>(
            "val",
            10,
            false,
            on_change<int>(
                [&callback_count](int&)
                {
                    ++callback_count;
                }
            )
        );

        dps2.set<int>("val", 99);
        CHECK(dps2.get<int>("val") == 99);
        CHECK(callback_count == 1);
    }

    SUBCASE("typed get throws on wrong type")
    {
        CHECK_THROWS(dps.get<float>("i"));
        CHECK_THROWS(dps.get<int>("f"));
        CHECK_THROWS(dps.get<TestEnum>("s"));
        CHECK_THROWS(dps.get<std::string>("e"));
    }

    SUBCASE("typed get throws on missing name")
    {
        CHECK_THROWS(dps.get<int>("nonexistent"));
    }

    SUBCASE("typed set throws on missing name")
    {
        CHECK_THROWS(dps.set<int>("nonexistent", 0));
    }

    SUBCASE("get_ref returns const reference to stored value")
    {
        const int& ref_i = dps.get_ref<int>("i");
        CHECK(ref_i == 10);

        dps.set<int>("i", 42);
        CHECK(ref_i == 42); // Reference tracks the stored value.
    }

    SUBCASE("get_ref throws on wrong type")
    {
        CHECK_THROWS(dps.get_ref<float>("i"));
    }

    SUBCASE("get_ref throws on missing name")
    {
        CHECK_THROWS(dps.get_ref<int>("nonexistent"));
    }
}

TEST_CASE("DynamicPropertySet reset by name")
{
    DynamicPropertySet dps;
    dps.add_property<int>("a", 10, false, default_value(0));
    dps.add_property<float>("b", 3.14f, false, default_value(1.0f));

    dps.set<int>("a", 42);
    dps.set<float>("b", 2.71f);

    // Reset only "a".
    dps.reset("a");
    CHECK(dps.get<int>("a") == 0);
    CHECK(dps.get<float>("b") == doctest::Approx(2.71f));

    // Throws on missing name.
    CHECK_THROWS(dps.reset("nonexistent"));
}

TEST_CASE("DynamicPropertySet clear")
{
    DynamicPropertySet dps;
    dps.add_property<int>("x", 1);
    dps.add_property<float>("y", 2.0f);
    CHECK(dps.size() == 2);

    dps.clear();

    CHECK(dps.size() == 0);
    CHECK(dps.empty());
    CHECK(dps.find_property("x") == nullptr);
    CHECK(dps.find_property("y") == nullptr);

    // Can add properties again after clear.
    dps.add_property<int>("z", 3);
    CHECK(dps.size() == 1);
    CHECK(dps.get<int>("z") == 3);
}

TEST_CASE("DynamicPropertySet is_equal_property")
{
    DynamicPropertySet dps;
    dps.add_property<int>("x", 42);
    dps.add_property<std::string>("s", std::string("hello"));

    SUBCASE("equal values")
    {
        Properties props;
        props.set<int>("x", 42);
        props.set<std::string>("s", "hello");
        CHECK(dps.is_equal_property("x", props));
        CHECK(dps.is_equal_property("s", props));
    }

    SUBCASE("different values")
    {
        Properties props;
        props.set<int>("x", 99);
        props.set<std::string>("s", "world");
        CHECK_FALSE(dps.is_equal_property("x", props));
        CHECK_FALSE(dps.is_equal_property("s", props));
    }

    SUBCASE("property missing in other")
    {
        Properties props;
        CHECK_FALSE(dps.is_equal_property("x", props));
    }

    SUBCASE("property missing in this set")
    {
        Properties props;
        props.set<int>("missing", 0);
        CHECK_FALSE(dps.is_equal_property("missing", props));
    }
}

TEST_CASE("DynamicPropertySet is_default")
{
    DynamicPropertySet dps;
    dps.add_property<int>("x", 42, false, default_value(42));

    CHECK(dps.is_default("x"));

    dps.set<int>("x", 99);
    CHECK_FALSE(dps.is_default("x"));

    dps.reset("x");
    CHECK(dps.is_default("x"));

    // Returns false for nonexistent property.
    CHECK_FALSE(dps.is_default("nonexistent"));
}

TEST_CASE("DynamicPropertySet generation counter")
{
    DynamicPropertySet dps;
    CHECK(dps.generation() == 0);

    // add_property increments generation.
    dps.add_property<int>("x", 1);
    CHECK(dps.generation() == 1);

    dps.add_property<float>("y", 2.0f);
    CHECK(dps.generation() == 2);

    // set does NOT increment generation (not a structural change).
    dps.set<int>("x", 42);
    CHECK(dps.generation() == 2);

    // remove_property increments generation.
    dps.remove_property("x");
    CHECK(dps.generation() == 3);

    // remove_property of nonexistent name does NOT increment.
    dps.remove_property("nonexistent");
    CHECK(dps.generation() == 3);

    // clear increments generation.
    dps.clear();
    CHECK(dps.generation() == 4);

    // Adding after clear continues incrementing.
    dps.add_property<int>("z", 10);
    CHECK(dps.generation() == 5);
}

TEST_CASE("DynamicPropertySet with new metadata types")
{
    DynamicPropertySet dyn;
    dyn.add_property<float>(
        "intensity",
        1.0f,
        false,
        ui_label("Light Intensity"),
        ui_drag_speed(0.1f),
        UIFlags::advanced
    );

    const PropertyDescriptor* desc = dyn.find_property("intensity");
    REQUIRE(desc != nullptr);

    REQUIRE(desc->ui_label() != nullptr);
    CHECK(desc->ui_label()->text == "Light Intensity");

    REQUIRE(desc->ui_drag_speed() != nullptr);
    CHECK(desc->ui_drag_speed()->speed == doctest::Approx(0.1f));

    CHECK(desc->ui_flags() == UIFlags::advanced);
}

TEST_CASE("DynamicPropertySet integral enum access requires enum metadata")
{
    int callback_count = 0;
    DynamicPropertySet dyn;
    dyn.add_property<int>(
        "mode",
        20,
        false,
        enum_descriptor({{10, "Ten"}, {20, "Twenty"}}),
        on_change<int>(
            [](int& count)
            {
                ++count;
            }
        )
    );
    dyn.add_property<int>("count", 20);

    const PropertyDescriptor* enum_prop = dyn.get_property("mode");
    CHECK(enum_prop->get_enum_as_int64(&callback_count) == 20);

    enum_prop->set_enum_from_int64(&callback_count, 10);
    CHECK(dyn.get<int>("mode") == 10);
    CHECK(callback_count == 1);

    const PropertyDescriptor* plain_prop = dyn.get_property("count");
    CHECK_THROWS_WITH(
        plain_prop->get_enum_as_int64(&callback_count),
        doctest::Contains("integral type has no enum metadata")
    );
    CHECK_THROWS_WITH(
        plain_prop->set_enum_from_int64(&callback_count, 10),
        doctest::Contains("integral type has no enum metadata")
    );
    CHECK(dyn.get<int>("count") == 20);
}

TEST_CASE("DynamicPropertySet read_from_properties skips read-only properties")
{
    DynamicPropertySet dps;
    dps.add_property<int>("ro", 42, true);
    dps.add_property<int>("rw", 10, false);

    Properties props;
    props.set<int>("ro", 99);
    props.set<int>("rw", 77);

    dps.read_from_properties(props);

    CHECK(dps.get<int>("ro") == 42); // unchanged, read-only
    CHECK(dps.get<int>("rw") == 77); // updated
}

TEST_SUITE_END();
