// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "testing.h"
#include "falcor2/core/properties.h"
#include "falcor2/core/object.h"

using namespace falcor;

namespace doctest {
template<>
struct StringMaker<Properties> {
    static String convert(const Properties& value) { return value.to_string().c_str(); }
};
} // namespace doctest

TEST_SUITE_BEGIN("properties");

namespace {

class TestObjectA : public Object {
    FALCOR_OBJECT(TestObjectA)
};
class TestObjectB : public Object {
    FALCOR_OBJECT(TestObjectB)
};

enum class TestEnum : int {
    foo = 0,
    bar = 1,
    baz = 42,
};

// ---------------------------------------------------------------------------
// Traits for type-parameterized tests
// ---------------------------------------------------------------------------

/// Maps each storable type to its PropertyType tag, a test value, and a default value.
template<typename T>
struct TestTypeTraits;

// Use brace-init so commas in constructors don't confuse the preprocessor.
#define DEFINE_TYPE_TRAITS(T, prop_type, test_val, default_val)                                                        \
    template<>                                                                                                         \
    struct TestTypeTraits<T> {                                                                                         \
        static constexpr PropertyType type = prop_type;                                                                \
        static T test_value()                                                                                          \
        {                                                                                                              \
            return test_val;                                                                                           \
        }                                                                                                              \
        static T default_value()                                                                                       \
        {                                                                                                              \
            return default_val;                                                                                        \
        }                                                                                                              \
    }

DEFINE_TYPE_TRAITS(bool, PropertyType::bool_, true, false);
DEFINE_TYPE_TRAITS(int, PropertyType::int_, 42, 123);
DEFINE_TYPE_TRAITS(int64_t, PropertyType::int_, 123456789012345LL, int64_t{0});
DEFINE_TYPE_TRAITS(float, PropertyType::float_, 42.0f, 123.0f);
DEFINE_TYPE_TRAITS(double, PropertyType::float_, 42.0, 123.0);
DEFINE_TYPE_TRAITS(int2, PropertyType::int2, (int2{1, 2}), (int2{3, 4}));
DEFINE_TYPE_TRAITS(int3, PropertyType::int3, (int3{1, 2, 3}), (int3{4, 5, 6}));
DEFINE_TYPE_TRAITS(int4, PropertyType::int4, (int4{1, 2, 3, 4}), (int4{5, 6, 7, 8}));
DEFINE_TYPE_TRAITS(uint2, PropertyType::uint2, (uint2{1, 2}), (uint2{3, 4}));
DEFINE_TYPE_TRAITS(uint3, PropertyType::uint3, (uint3{1, 2, 3}), (uint3{4, 5, 6}));
DEFINE_TYPE_TRAITS(uint4, PropertyType::uint4, (uint4{1, 2, 3, 4}), (uint4{5, 6, 7, 8}));
DEFINE_TYPE_TRAITS(float2, PropertyType::float2, (float2{1, 2}), (float2{3, 4}));
DEFINE_TYPE_TRAITS(float3, PropertyType::float3, (float3{1, 2, 3}), (float3{4, 5, 6}));
DEFINE_TYPE_TRAITS(float4, PropertyType::float4, (float4{1, 2, 3, 4}), (float4{5, 6, 7, 8}));
DEFINE_TYPE_TRAITS(
    float3x3,
    PropertyType::float3x3,
    (float3x3{{1, 2, 3, 4, 5, 6, 7, 8, 9}}),
    (float3x3{{9, 8, 7, 6, 5, 4, 3, 2, 1}})
);
DEFINE_TYPE_TRAITS(
    float4x4,
    PropertyType::float4x4,
    (float4x4{{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}}),
    (float4x4{{16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1}})
);
DEFINE_TYPE_TRAITS(std::string, PropertyType::string, "foo", "bar");
DEFINE_TYPE_TRAITS(TestEnum, PropertyType::enum_, TestEnum::bar, TestEnum::baz);

#undef DEFINE_TYPE_TRAITS

/// Provides sample vector data for PropertyList round-trip tests.
template<typename T>
struct TestListTraits;

#define DEFINE_LIST_TRAITS(T, ...)                                                                                     \
    template<>                                                                                                         \
    struct TestListTraits<T> {                                                                                         \
        static std::vector<T> data()                                                                                   \
        {                                                                                                              \
            return {__VA_ARGS__};                                                                                      \
        }                                                                                                              \
    }

DEFINE_LIST_TRAITS(bool, true, false, true);
DEFINE_LIST_TRAITS(int64_t, 10, 20, 30);
DEFINE_LIST_TRAITS(double, 1.5, 2.5, 3.5);
DEFINE_LIST_TRAITS(int2, (int2{1, 2}), (int2{3, 4}));
DEFINE_LIST_TRAITS(int3, (int3{1, 2, 3}), (int3{4, 5, 6}));
DEFINE_LIST_TRAITS(int4, (int4{1, 2, 3, 4}), (int4{5, 6, 7, 8}));
DEFINE_LIST_TRAITS(uint2, (uint2{1, 2}), (uint2{3, 4}));
DEFINE_LIST_TRAITS(uint3, (uint3{1, 2, 3}), (uint3{4, 5, 6}));
DEFINE_LIST_TRAITS(uint4, (uint4{1, 2, 3, 4}), (uint4{5, 6, 7, 8}));
DEFINE_LIST_TRAITS(float2, (float2{1, 2}), (float2{3, 4}));
DEFINE_LIST_TRAITS(float3, (float3{1, 0, 0}), (float3{0, 1, 0}), (float3{0, 0, 1}));
DEFINE_LIST_TRAITS(float4, (float4{1, 2, 3, 4}), (float4{5, 6, 7, 8}));
DEFINE_LIST_TRAITS(float3x3, (float3x3{{1, 0, 0, 0, 1, 0, 0, 0, 1}}), (float3x3{{2, 0, 0, 0, 2, 0, 0, 0, 2}}));
DEFINE_LIST_TRAITS(
    float4x4,
    (float4x4{{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}}),
    (float4x4{{2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2}})
);
DEFINE_LIST_TRAITS(std::string, "alpha", "beta", "gamma");

#undef DEFINE_LIST_TRAITS

/// All types that can be stored as scalar properties.
using AllPropertyTypes = std::tuple<
    bool,
    int,
    int64_t,
    float,
    double,
    int2,
    int3,
    int4,
    uint2,
    uint3,
    uint4,
    float2,
    float3,
    float4,
    float3x3,
    float4x4,
    std::string,
    TestEnum>;

/// All types that can be stored as PropertyList elements.
using AllListElementTypes = std::tuple<
    bool,
    int64_t,
    double,
    int2,
    int3,
    int4,
    uint2,
    uint3,
    uint4,
    float2,
    float3,
    float4,
    float3x3,
    float4x4,
    std::string>;

} // namespace

TYPE_TO_STRING(bool);
TYPE_TO_STRING(int);
TYPE_TO_STRING(int64_t);
TYPE_TO_STRING(float);
TYPE_TO_STRING(double);
TYPE_TO_STRING(int2);
TYPE_TO_STRING(int3);
TYPE_TO_STRING(int4);
TYPE_TO_STRING(uint2);
TYPE_TO_STRING(uint3);
TYPE_TO_STRING(uint4);
TYPE_TO_STRING(float2);
TYPE_TO_STRING(float3);
TYPE_TO_STRING(float4);
TYPE_TO_STRING(float3x3);
TYPE_TO_STRING(float4x4);
TYPE_TO_STRING(std::string);
TYPE_TO_STRING(TestEnum);

// =============================================================================
// Basic type storage (parameterized)
// =============================================================================

TEST_CASE_TEMPLATE_DEFINE("type_roundtrip", T, type_roundtrip_id)
{
    using Traits = TestTypeTraits<T>;
    Properties props;

    const T test_val = Traits::test_value();
    const T default_val = Traits::default_value();

    CHECK(props.has_property("v") == false);
    CHECK(props.get<T>("v", default_val) == default_val);

    props.set("v", test_val);

    CHECK(props.type("v") == Traits::type);
    CHECK(props.has_property("v") == true);
    CHECK(props.get<T>("v") == test_val);
    CHECK(props.get<T>("v", default_val) == test_val);

    CHECK(props.remove_property("v") == true);
    CHECK(props.has_property("v") == false);
    CHECK(props.get<T>("v", default_val) == default_val);
}
TEST_CASE_TEMPLATE_APPLY(type_roundtrip_id, AllPropertyTypes);

// =============================================================================
// Numeric edge cases
// =============================================================================

TEST_CASE("integer_range_checking")
{
    Properties props;

    SUBCASE("int stored, retrieved as various integer types")
    {
        props.set("val", 42);
        CHECK(props.get<int>("val") == 42);
        CHECK(props.get<int64_t>("val") == 42);
        CHECK(props.get<uint32_t>("val") == 42u);
    }

    SUBCASE("negative int cannot be retrieved as unsigned")
    {
        props.set("val", -1);
        CHECK_THROWS(props.get<uint32_t>("val"));
    }

    SUBCASE("large int64 cannot be retrieved as int")
    {
        props.set("val", int64_t(3000000000LL));
        CHECK_THROWS(props.get<int>("val"));
        CHECK(props.get<int64_t>("val") == 3000000000LL);
    }
}

TEST_CASE("float_double_unification")
{
    Properties props;

    SUBCASE("float stored as double")
    {
        props.set("val", 3.14f);
        CHECK(props.type("val") == PropertyType::float_);
        CHECK(props.get<float>("val") == doctest::Approx(3.14f));
        CHECK(props.get<double>("val") == doctest::Approx(3.14));
    }

    SUBCASE("double stored and retrieved")
    {
        props.set("val", 3.14);
        CHECK(props.type("val") == PropertyType::float_);
        CHECK(props.get<double>("val") == doctest::Approx(3.14));
        CHECK(props.get<float>("val") == doctest::Approx(3.14f));
    }
}

TEST_CASE("float16_maps_to_float_storage")
{
    Properties props;

    SUBCASE("scalar")
    {
        props.set("val", float16_t(0.5f));
        CHECK(props.type("val") == PropertyType::float_);
        CHECK(static_cast<float>(props.get<float16_t>("val")) == doctest::Approx(0.5f));
        CHECK(props.get<float>("val") == doctest::Approx(0.5f));
        CHECK(Properties::check_type<float16_t>() == PropertyType::float_);
    }

    SUBCASE("vector")
    {
        props.set("val", float16_t3{0.25f, 0.5f, 0.75f});
        CHECK(props.type("val") == PropertyType::float3);
        CHECK(float3(props.get<float16_t3>("val")) == float3{0.25f, 0.5f, 0.75f});
        CHECK(props.get<float3>("val") == float3{0.25f, 0.5f, 0.75f});
        CHECK(Properties::check_type<float16_t3>() == PropertyType::float3);
    }

    SUBCASE("scalar list")
    {
        props.set_list("val", std::vector<float16_t>{float16_t(0.25f), float16_t(0.5f)});
        CHECK(props.type("val") == PropertyType::list);
        CHECK(props.get<detail::PropertyList>("val").element_type() == PropertyType::float_);

        const auto& double_values = props.get_list<double>("val");
        REQUIRE(double_values.size() == 2);
        CHECK(double_values[0] == doctest::Approx(0.25));
        CHECK(double_values[1] == doctest::Approx(0.5));

        std::vector<float16_t> half_values = props.get_list<float16_t>("val");
        REQUIRE(half_values.size() == 2);
        CHECK(static_cast<float>(half_values[0]) == doctest::Approx(0.25f));
        CHECK(static_cast<float>(half_values[1]) == doctest::Approx(0.5f));
    }

    SUBCASE("vector list")
    {
        props.set_list("val", std::vector<float16_t3>{float16_t3{0.25f, 0.5f, 0.75f}});
        CHECK(props.type("val") == PropertyType::list);
        CHECK(props.get<detail::PropertyList>("val").element_type() == PropertyType::float3);

        const auto& float_values = props.get_list<float3>("val");
        REQUIRE(float_values.size() == 1);
        CHECK(float_values[0] == float3{0.25f, 0.5f, 0.75f});

        std::vector<float16_t3> half_values = props.get_list<float16_t3>("val");
        REQUIRE(half_values.size() == 1);
        CHECK(float3(half_values[0]) == float3{0.25f, 0.5f, 0.75f});
    }
}

TEST_CASE("filesystem_path")
{
    Properties props;

    SUBCASE("set and get path")
    {
        std::filesystem::path p("/some/test/path");
        props.set("path", p);
        CHECK(props.type("path") == PropertyType::string);
        CHECK(props.get<std::filesystem::path>("path") == p);
        CHECK(props.get<std::string>("path") == p.string());
    }

    SUBCASE("default value")
    {
        std::filesystem::path default_path("/default");
        CHECK(props.get<std::filesystem::path>("missing", default_path) == default_path);
    }
}

// =============================================================================
// Container operations
// =============================================================================

TEST_CASE("empty_clear")
{
    Properties props;
    CHECK(props.empty());
    CHECK(props.size() == 0);

    props.set("a", 1);
    props.set("b", 2);
    CHECK(!props.empty());
    CHECK(props.size() == 2);

    props.clear();
    CHECK(props.empty());
    CHECK(props.size() == 0);
}

TEST_CASE("insertion_order")
{
    Properties props;
    props.set("c", 3);
    props.set("a", 1);
    props.set("b", 2);

    auto k = props.keys();
    CHECK(k.size() == 3);
    CHECK(k[0] == "c");
    CHECK(k[1] == "a");
    CHECK(k[2] == "b");
}

TEST_CASE("copy_move")
{
    Properties props1;
    props1.set("int", 42);
    props1.set("str", std::string("hello"));
    props1.set("vec", float3(1, 2, 3));

    SUBCASE("copy constructor")
    {
        Properties props2(props1);
        CHECK(props1 == props2);
    }

    SUBCASE("move constructor")
    {
        Properties copy(props1);
        Properties props2(std::move(props1));
        CHECK(props2 == copy);
    }

    SUBCASE("copy assignment")
    {
        Properties props2;
        props2 = props1;
        CHECK(props1 == props2);
    }

    SUBCASE("move assignment")
    {
        Properties copy(props1);
        Properties props2;
        props2 = std::move(props1);
        CHECK(props2 == copy);
    }
}

TEST_CASE("equality")
{
    Properties props1;
    Properties props2;

    CHECK(props1 == props1);
    CHECK(props1 == props2);

    props1.set("a", 1);
    props1.set("b", std::string("hello"));
    CHECK(props1 != props2);

    props2.set("a", 1);
    props2.set("b", std::string("hello"));
    CHECK(props1 == props2);
}

TEST_CASE("merge")
{
    Properties props1;
    props1.set("foo", 42);
    props1.set("bar", 42);
    props1.set("baz", 42);

    Properties props2;
    props2.set("foo", 1);
    props2.set("bar", 1);
    props2.set("new", 1);

    Properties merged(props1);
    CHECK(merged == props1);
    merged.merge(props1);
    CHECK(merged == props1);
    merged.merge(props2);
    CHECK(merged.get<int>("foo") == 1);
    CHECK(merged.get<int>("bar") == 1);
    CHECK(merged.get<int>("baz") == 42);
    CHECK(merged.get<int>("new") == 1);
}

TEST_CASE("hash")
{
    Properties props1;
    props1.set("a", 1);
    props1.set("b", 2.0);
    props1.set("c", std::string("hello"));

    Properties props2;
    props2.set("a", 1);
    props2.set("b", 2.0);
    props2.set("c", std::string("hello"));

    SUBCASE("equal properties have equal hash")
    {
        CHECK(props1.hash() == props2.hash());
    }

    SUBCASE("insertion order does not affect hash")
    {
        Properties props3;
        props3.set("c", std::string("hello"));
        props3.set("a", 1);
        props3.set("b", 2.0);
        CHECK(props1.hash() == props3.hash());
    }

    SUBCASE("different values produce different hash")
    {
        Properties props3;
        props3.set("a", 999);
        props3.set("b", 2.0);
        props3.set("c", std::string("hello"));
        CHECK(props1.hash() != props3.hash());
    }

    SUBCASE("empty properties")
    {
        Properties empty1;
        Properties empty2;
        CHECK(empty1.hash() == empty2.hash());
    }
}

TEST_CASE("remove_property")
{
    Properties props;
    props.set("foo", 42);

    CHECK(props.remove_property("foo") == true);
    CHECK(props.remove_property("foo") == false);
    CHECK(props.remove_property("nonexistent") == false);
}

TEST_CASE("rename_property")
{
    Properties props;
    props.set("old_name", 42);
    props.set("other", 99);

    SUBCASE("basic rename")
    {
        CHECK(props.rename_property("old_name", "new_name") == true);
        CHECK(props.has_property("new_name") == true);
        CHECK(props.has_property("old_name") == false);
        CHECK(props.get<int>("new_name") == 42);
    }

    SUBCASE("rename nonexistent")
    {
        CHECK(props.rename_property("nonexistent", "new_name") == false);
    }

    SUBCASE("rename to existing name")
    {
        CHECK(props.rename_property("old_name", "other") == false);
        CHECK(props.has_property("old_name") == true);
        CHECK(props.get<int>("old_name") == 42);
    }
}

TEST_CASE("set_throw_if_exists")
{
    Properties props;

    SUBCASE("throws when property exists")
    {
        props.set("val", 42);
        CHECK_THROWS(props.set("val", 99, true));
        CHECK(props.get<int>("val") == 42);
    }

    SUBCASE("succeeds when property does not exist")
    {
        CHECK_NOTHROW(props.set("val", 42, true));
        CHECK(props.get<int>("val") == 42);
    }
}

// =============================================================================
// Accessors
// =============================================================================

TEST_CASE("get_optional")
{
    Properties props;
    props.set("int_val", 42);
    props.set("str_val", std::string("hello"));

    SUBCASE("returns value when present")
    {
        auto result = props.get_optional<int>("int_val");
        REQUIRE(result.has_value());
        CHECK(*result == 42);
    }

    SUBCASE("returns nullopt when missing")
    {
        auto result = props.get_optional<int>("nonexistent");
        CHECK(!result.has_value());
    }

    SUBCASE("returns nullopt on type mismatch by default")
    {
        auto result = props.get_optional<double>("int_val");
        CHECK(!result.has_value());
    }

    SUBCASE("throws on type mismatch when requested")
    {
        CHECK_THROWS(props.get_optional<double>("int_val", true));
    }

    SUBCASE("string type")
    {
        auto result = props.get_optional<std::string>("str_val");
        REQUIRE(result.has_value());
        CHECK(*result == "hello");
    }

    SUBCASE("enum")
    {
        props.set("enum_val", TestEnum::bar);
        auto result = props.get_optional<TestEnum>("enum_val");
        REQUIRE(result.has_value());
        CHECK(*result == TestEnum::bar);

        auto missing = props.get_optional<TestEnum>("nonexistent");
        CHECK(!missing.has_value());
    }
}

TEST_CASE("try_get")
{
    Properties props;
    props.set("int_val", 42);
    props.set("str_val", std::string("hello"));
    props.set("float_val", 3.14);

    SUBCASE("returns pointer for matching type")
    {
        const int64_t* p = props.try_get<int64_t>("int_val");
        REQUIRE(p != nullptr);
        CHECK(*p == 42);
    }

    SUBCASE("returns nullptr for wrong type")
    {
        const double* p = props.try_get<double>("int_val");
        CHECK(p == nullptr);
    }

    SUBCASE("returns nullptr for missing key")
    {
        const int64_t* p = props.try_get<int64_t>("nonexistent");
        CHECK(p == nullptr);
    }

    SUBCASE("string type")
    {
        const std::string* p = props.try_get<std::string>("str_val");
        REQUIRE(p != nullptr);
        CHECK(*p == "hello");
    }
}

TEST_CASE("any_storage")
{
    Properties props;

    struct MyData {
        int x;
        float y;
    };

    props.set_any("data", MyData{42, 3.14f});
    CHECK(props.type("data") == PropertyType::any);

    const auto& data = props.get_any<MyData>("data");
    CHECK(data.x == 42);
    CHECK(data.y == doctest::Approx(3.14f));

    CHECK_THROWS(props.get_any<int>("data"));
}

TEST_CASE("object_type")
{
    Properties props;

    SUBCASE("correct object type")
    {
        ref<TestObjectA> obj = make_ref<TestObjectA>();
        props.set("obj", obj);
        CHECK(props.get<ref<TestObjectA>>("obj") == obj);
    }

    SUBCASE("wrong object type throws")
    {
        ref<TestObjectA> obj = make_ref<TestObjectA>();
        props.set("obj", obj);
        CHECK_THROWS(props.get<ref<TestObjectB>>("obj"));
    }
}

TEST_CASE("as_string")
{
    Properties props;
    props.set("int_val", 42);
    props.set("str_val", std::string("hello"));
    props.set("float_val", 3.14);
    props.set("bool_val", true);
    props.set("vec_val", float3(1, 2, 3));

    SUBCASE("returns non-empty string for each type")
    {
        CHECK(!props.as_string("int_val").empty());
        CHECK(!props.as_string("str_val").empty());
        CHECK(!props.as_string("float_val").empty());
        CHECK(!props.as_string("bool_val").empty());
        CHECK(!props.as_string("vec_val").empty());
    }

    SUBCASE("throws for missing key")
    {
        CHECK_THROWS(props.as_string("nonexistent"));
    }
}

TEST_CASE("to_string")
{
    SUBCASE("empty properties")
    {
        Properties props;
        std::string str = props.to_string();
        CHECK(str.find("Properties") != std::string::npos);
    }

    SUBCASE("non-empty properties")
    {
        Properties props;
        props.set("a", 1);
        props.set("b", std::string("hello"));
        std::string str = props.to_string();
        CHECK(str.find("Properties") != std::string::npos);
        CHECK(str.find("a") != std::string::npos);
        CHECK(str.find("b") != std::string::npos);
    }
}

TEST_CASE("check_type")
{
    CHECK(Properties::check_type<int>() == PropertyType::int_);
    CHECK(Properties::check_type<double>() == PropertyType::float_);
    CHECK(Properties::check_type<std::string>() == PropertyType::string);
    CHECK(Properties::check_type<float3>() == PropertyType::float3);
    CHECK(Properties::check_type<ref<Object>>() == PropertyType::object);
    CHECK(Properties::check_type<bool>() == PropertyType::bool_);
    CHECK(Properties::check_type<TestEnum>() == PropertyType::enum_);
    CHECK(Properties::check_type<Any>() == PropertyType::any);
}

TEST_CASE("is_equal_property")
{
    Properties props1;
    Properties props2;

    SUBCASE("both present and equal")
    {
        props1.set("val", 42);
        props2.set("val", 42);
        CHECK(props1.is_equal_property("val", props2));
    }

    SUBCASE("both present but different values")
    {
        props1.set("val", 42);
        props2.set("val", 99);
        CHECK(!props1.is_equal_property("val", props2));
    }

    SUBCASE("both present but different types")
    {
        props1.set("val", 42);
        props2.set("val", std::string("hello"));
        CHECK(!props1.is_equal_property("val", props2));
    }

    SUBCASE("both missing")
    {
        CHECK(props1.is_equal_property("val", props2));
    }

    SUBCASE("one missing")
    {
        props1.set("val", 42);
        CHECK(!props1.is_equal_property("val", props2));
        CHECK(!props2.is_equal_property("val", props1));
    }
}

TEST_CASE("copy_property")
{
    Properties src;
    src.set("int_val", 42);
    src.set("str_val", std::string("hello"));

    SUBCASE("copy to empty")
    {
        Properties dst;
        auto it = src.find("int_val");
        REQUIRE(it != src.end());
        dst.copy_property(it);
        CHECK(dst.has_property("int_val"));
        CHECK(dst.get<int>("int_val") == 42);
    }

    SUBCASE("copy overwrites existing")
    {
        Properties dst;
        dst.set("int_val", 99);
        auto it = src.find("int_val");
        dst.copy_property(it);
        CHECK(dst.get<int>("int_val") == 42);
    }

    SUBCASE("copy string property")
    {
        Properties dst;
        auto it = src.find("str_val");
        dst.copy_property(it);
        CHECK(dst.get<std::string>("str_val") == "hello");
    }
}

// =============================================================================
// Iterator
// =============================================================================

TEST_CASE("iterator")
{
    Properties props;
    props.set("a", 1);
    props.set("b", std::string("hello"));
    props.set("c", float3(1, 2, 3));

    SUBCASE("iterate all entries")
    {
        size_t count = 0;
        for (const auto& entry : props) {
            CHECK(!entry->name().empty());
            count++;
        }
        CHECK(count == props.size());
    }

    SUBCASE("skips tombstones")
    {
        size_t original_size = props.size();
        props.remove_property("b");
        size_t count = 0;
        for (const auto& entry : props) {
            CHECK(entry->name() != "b");
            count++;
        }
        CHECK(count == original_size - 1);
    }

    SUBCASE("get<T> retrieves value")
    {
        auto it = props.begin();
        CHECK(it->name() == "a");
        CHECK(it->get<int>() == 1);
    }

    SUBCASE("type and index")
    {
        auto it = props.begin();
        CHECK(it->type() == PropertyType::int_);
        // index() should be a valid non-sentinel value
        CHECK(it->index() != size_t(-1));
    }
}

TEST_CASE("find")
{
    Properties props;
    props.set("alpha", 1);
    props.set("beta", std::string("hello"));

    SUBCASE("finds existing key")
    {
        auto it = props.find("alpha");
        REQUIRE(it != props.end());
        CHECK(it->name() == "alpha");
        CHECK(it->type() == PropertyType::int_);
        CHECK(it->get<int>() == 1);
    }

    SUBCASE("returns end for missing key")
    {
        auto it = props.find("nonexistent");
        CHECK(it == props.end());
    }

    SUBCASE("finds string property")
    {
        auto it = props.find("beta");
        REQUIRE(it != props.end());
        CHECK(it->name() == "beta");
        CHECK(it->get<std::string>() == "hello");
    }
}

// =============================================================================
// PropertyList (parameterized)
// =============================================================================

TEST_CASE_TEMPLATE_DEFINE("property_list_roundtrip", T, list_roundtrip_id)
{
    Properties props;
    std::vector<T> data = TestListTraits<T>::data();

    props.set_list("v", data);
    CHECK(props.type("v") == PropertyType::list);

    const auto& result = props.get_list<T>("v");
    REQUIRE(result.size() == data.size());
    for (size_t i = 0; i < data.size(); ++i)
        CHECK(result[i] == data[i]);
}
TEST_CASE_TEMPLATE_APPLY(list_roundtrip_id, AllListElementTypes);

// =============================================================================
// PropertyList (non-parameterized)
// =============================================================================

TEST_CASE("property_list_object")
{
    Properties props;

    SUBCASE("roundtrip")
    {
        ref<TestObjectA> a = make_ref<TestObjectA>();
        ref<TestObjectA> b = make_ref<TestObjectA>();
        std::vector<ref<Object>> data = {a, b};
        props.set_list("objs", data);
        CHECK(props.type("objs") == PropertyType::list);
        CHECK(props.get<detail::PropertyList>("objs").element_type() == PropertyType::object);
        auto result = props.get_list<ref<Object>>("objs");
        REQUIRE(result.size() == 2);
        CHECK(result[0] == a);
        CHECK(result[1] == b);
    }

    SUBCASE("try_get_list")
    {
        ref<TestObjectA> a = make_ref<TestObjectA>();
        props.set_list("objs", std::vector<ref<Object>>{a});
        const auto* p = props.try_get_list<ref<Object>>("objs");
        REQUIRE(p != nullptr);
        REQUIRE(p->size() == 1);
        CHECK((*p)[0] == a);
    }

    SUBCASE("equality")
    {
        ref<TestObjectA> a = make_ref<TestObjectA>();
        Properties p1, p2;
        p1.set_list("objs", std::vector<ref<Object>>{a});
        p2.set_list("objs", std::vector<ref<Object>>{a});
        CHECK(p1 == p2);
    }

    SUBCASE("to_string")
    {
        ref<TestObjectA> a = make_ref<TestObjectA>();
        detail::PropertyList list = detail::PropertyList::create(std::vector<ref<Object>>{a, nullptr});
        std::string str = list.to_string();
        CHECK(str.front() == '[');
        CHECK(str.back() == ']');
    }
}

TEST_CASE("set_list_throw_if_exists")
{
    Properties props;

    SUBCASE("throws when property exists")
    {
        props.set_list("v", std::vector<float3>{float3(1, 0, 0)});
        CHECK_THROWS(props.set_list("v", std::vector<float3>{float3(0, 1, 0)}, true));
        auto result = props.get_list<float3>("v");
        REQUIRE(result.size() == 1);
        CHECK(result[0] == float3(1, 0, 0));
    }

    SUBCASE("succeeds when property does not exist")
    {
        CHECK_NOTHROW(props.set_list("v", std::vector<float3>{float3(1, 0, 0)}, true));
        auto result = props.get_list<float3>("v");
        REQUIRE(result.size() == 1);
        CHECK(result[0] == float3(1, 0, 0));
    }
}

TEST_CASE("property_list_type_mismatch")
{
    Properties props;
    props.set_list("v", std::vector<float3>{float3(1, 0, 0)});

    CHECK_THROWS(props.get_list<int3>("v"));
    CHECK_THROWS(props.get_list<double>("v"));
}

TEST_CASE("property_list_type_conversion")
{
    Properties props;

    SUBCASE("int widens to int64_t")
    {
        props.set_list("v", std::vector<int>{1, 2, 3});
        CHECK(props.get<detail::PropertyList>("v").element_type() == PropertyType::int_);
        const auto& result = props.get_list<int64_t>("v");
        REQUIRE(result.size() == 3);
        CHECK(result[0] == 1);
        CHECK(result[1] == 2);
        CHECK(result[2] == 3);
    }

    SUBCASE("float widens to double")
    {
        props.set_list("v", std::vector<float>{1.5f, 2.5f});
        CHECK(props.get<detail::PropertyList>("v").element_type() == PropertyType::float_);
        const auto& result = props.get_list<double>("v");
        REQUIRE(result.size() == 2);
        CHECK(result[0] == doctest::Approx(1.5));
        CHECK(result[1] == doctest::Approx(2.5));
    }

    SUBCASE("uint16_t widens to int64_t")
    {
        props.set_list("v", std::vector<uint16_t>{100, 200});
        CHECK(props.get<detail::PropertyList>("v").element_type() == PropertyType::int_);
        const auto& result = props.get_list<int64_t>("v");
        REQUIRE(result.size() == 2);
        CHECK(result[0] == 100);
        CHECK(result[1] == 200);
    }

    SUBCASE("filesystem::path converts to string")
    {
        props.set_list("v", std::vector<std::filesystem::path>{"/foo/bar", "/baz/qux"});
        CHECK(props.get<detail::PropertyList>("v").element_type() == PropertyType::string);
        const auto& result = props.get_list<std::string>("v");
        REQUIRE(result.size() == 2);
        CHECK(result[0] == "/foo/bar");
        CHECK(result[1] == "/baz/qux");
    }

    SUBCASE("get_list narrows int64_t to int")
    {
        props.set_list("v", std::vector<int64_t>{10, 20, 30});
        auto result = props.get_list<int>("v");
        REQUIRE(result.size() == 3);
        CHECK(result[0] == 10);
        CHECK(result[1] == 20);
        CHECK(result[2] == 30);
    }

    SUBCASE("get_list narrows double to float")
    {
        props.set_list("v", std::vector<double>{1.5, 2.5});
        auto result = props.get_list<float>("v");
        REQUIRE(result.size() == 2);
        CHECK(result[0] == doctest::Approx(1.5f));
        CHECK(result[1] == doctest::Approx(2.5f));
    }

    SUBCASE("get_list retrieves strings as paths")
    {
        props.set_list("v", std::vector<std::string>{"/foo", "/bar"});
        auto result = props.get_list<std::filesystem::path>("v");
        REQUIRE(result.size() == 2);
        CHECK(result[0] == std::filesystem::path("/foo"));
        CHECK(result[1] == std::filesystem::path("/bar"));
    }

    SUBCASE("get_list range check throws for out-of-bounds")
    {
        props.set_list("v", std::vector<int64_t>{int64_t(3000000000LL)});
        CHECK_THROWS(props.get_list<int>("v"));
    }

    SUBCASE("get_list narrows int64_t to uint32_t")
    {
        props.set_list("v", std::vector<int64_t>{100, 200});
        auto result = props.get_list<uint32_t>("v");
        REQUIRE(result.size() == 2);
        CHECK(result[0] == 100u);
        CHECK(result[1] == 200u);
    }

    SUBCASE("get_list negative to unsigned throws")
    {
        props.set_list("v", std::vector<int64_t>{-1});
        CHECK_THROWS(props.get_list<uint32_t>("v"));
    }
}

TEST_CASE("property_list_empty_vector")
{
    Properties props;
    props.set_list<float3>("v", {});
    CHECK(props.type("v") == PropertyType::list);
    const auto& result = props.get_list<float3>("v");
    CHECK(result.empty());
}

TEST_CASE("property_list_get_optional")
{
    Properties props;
    props.set_list("v", std::vector<float3>{float3(1, 2, 3)});

    SUBCASE("returns value when present")
    {
        auto result = props.get_list_optional<float3>("v");
        REQUIRE(result.has_value());
        REQUIRE(result->size() == 1);
        CHECK((*result)[0] == float3(1, 2, 3));
    }

    SUBCASE("returns nullopt when missing")
    {
        auto result = props.get_list_optional<float3>("nonexistent");
        CHECK(!result.has_value());
    }

    SUBCASE("returns nullopt on element type mismatch by default")
    {
        auto result = props.get_list_optional<int3>("v");
        CHECK(!result.has_value());
    }

    SUBCASE("throws on element type mismatch when requested")
    {
        CHECK_THROWS(props.get_list_optional<int3>("v", true));
    }
}

TEST_CASE("property_list_try_get_list")
{
    Properties props;
    props.set_list("v", std::vector<int64_t>{10, 20, 30});
    props.set("scalar", 42);

    SUBCASE("returns pointer for matching type")
    {
        const std::vector<int64_t>* p = props.try_get_list<int64_t>("v");
        REQUIRE(p != nullptr);
        REQUIRE(p->size() == 3);
        CHECK((*p)[0] == 10);
    }

    SUBCASE("returns nullptr for wrong element type")
    {
        const std::vector<double>* p = props.try_get_list<double>("v");
        CHECK(p == nullptr);
    }

    SUBCASE("returns nullptr for missing key")
    {
        const std::vector<int64_t>* p = props.try_get_list<int64_t>("nonexistent");
        CHECK(p == nullptr);
    }

    SUBCASE("returns nullptr for non-list property")
    {
        const std::vector<int64_t>* p = props.try_get_list<int64_t>("scalar");
        CHECK(p == nullptr);
    }
}

TEST_CASE("property_list_hash")
{
    Properties props1;
    props1.set_list("v", std::vector<float3>{float3(1, 0, 0), float3(0, 1, 0)});

    Properties props2;
    props2.set_list("v", std::vector<float3>{float3(1, 0, 0), float3(0, 1, 0)});

    CHECK(props1.hash() == props2.hash());

    Properties props3;
    props3.set_list("v", std::vector<float3>{float3(0, 0, 1), float3(0, 1, 0)});

    CHECK(props1.hash() != props3.hash());
}

TEST_CASE("property_list_equality")
{
    Properties props1;
    props1.set_list("v", std::vector<float3>{float3(1, 0, 0)});

    Properties props2;
    props2.set_list("v", std::vector<float3>{float3(1, 0, 0)});

    CHECK(props1 == props2);

    Properties props3;
    props3.set_list("v", std::vector<float3>{float3(0, 1, 0)});

    CHECK(props1 != props3);
}

TEST_CASE("property_list_iteration")
{
    Properties props;
    props.set("scalar", 42);
    props.set_list("vec", std::vector<float3>{float3(1, 0, 0)});

    size_t count = 0;
    bool found_list = false;
    for (const auto& entry : props) {
        count++;
        if (entry->name() == "vec") {
            CHECK(entry->type() == PropertyType::list);
            found_list = true;
        }
    }
    CHECK(count == 2);
    CHECK(found_list);
}

TEST_CASE("property_list_copy_move")
{
    Properties props1;
    props1.set_list("v", std::vector<float3>{float3(1, 0, 0), float3(0, 1, 0)});
    props1.set("scalar", 42);

    SUBCASE("copy")
    {
        Properties props2(props1);
        CHECK(props2 == props1);
        const auto& result = props2.get_list<float3>("v");
        REQUIRE(result.size() == 2);
        CHECK(result[0] == float3(1, 0, 0));
    }

    SUBCASE("move")
    {
        Properties copy(props1);
        Properties props2(std::move(props1));
        CHECK(props2 == copy);
    }
}

TEST_CASE("property_list_to_string")
{
    detail::PropertyList list = detail::PropertyList::create(std::vector<double>{1.0, 2.0, 3.0});
    std::string str = list.to_string();
    CHECK(str.front() == '[');
    CHECK(str.back() == ']');

    detail::PropertyList empty_list;
    CHECK(empty_list.to_string() == "[]");
}

TEST_CASE("property_list_merge")
{
    Properties props1;
    props1.set_list("v", std::vector<float3>{float3(1, 0, 0)});
    props1.set("scalar", 42);

    Properties props2;
    props2.set_list("v", std::vector<float3>{float3(0, 1, 0), float3(0, 0, 1)});

    props1.merge(props2);
    const auto& result = props1.get_list<float3>("v");
    REQUIRE(result.size() == 2);
    CHECK(result[0] == float3(0, 1, 0));
    CHECK(result[1] == float3(0, 0, 1));
    CHECK(props1.get<int>("scalar") == 42);
}

TEST_SUITE_END();
