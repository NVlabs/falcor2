// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "testing.h"

#include "falcor2/core/reflection.h"
#include "falcor2/core/enum.h"

using namespace falcor;
using namespace falcor::reflection;

// ---------------------------------------------------------------------------
// Test helper class
// ---------------------------------------------------------------------------

class MetadataTestClass {
public:
    float albedo() const { return m_albedo; }
    void set_albedo(float v) { m_albedo = v; }

    int count() const { return m_count; }
    void set_count(int v) { m_count = v; }

    bool use_feature() const { return m_use_feature; }
    void set_use_feature(bool v) { m_use_feature = v; }

    float speed() const { return m_speed; }
    void set_speed(float v) { m_speed = v; }

private:
    float m_albedo{0.5f};
    int m_count{0};
    bool m_use_feature{false};
    float m_speed{1.0f};
};

// Build a TypedPropertyDescriptor directly for testing metadata propagation.
template<typename T, typename... Extras>
std::unique_ptr<TypedPropertyDescriptor<T, MetadataTestClass>> make_prop(
    const char* name,
    std::function<T(const MetadataTestClass&)> getter,
    std::function<void(MetadataTestClass&, const T&)> setter,
    Extras&&... extras
)
{
    return std::make_unique<TypedPropertyDescriptor<T, MetadataTestClass>>(
        name,
        std::move(getter),
        std::move(setter),
        std::forward<Extras>(extras)...
    );
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST_SUITE_BEGIN("reflection");

TEST_CASE("is_reflection_metadata trait for new types")
{
    // Metadata types satisfy is_reflection_metadata_v.
    CHECK(is_reflection_metadata_v<DefaultValue<int>>);
    CHECK(is_reflection_metadata_v<ValueRange>);
    CHECK(is_reflection_metadata_v<UILabel>);
    CHECK(is_reflection_metadata_v<UIGroup>);
    CHECK(is_reflection_metadata_v<UIDragSpeed>);
    CHECK(is_reflection_metadata_v<UIEnableIf>);
    CHECK(is_reflection_metadata_v<UIFlags>);

    // Non-metadata types do not.
    CHECK_FALSE(is_reflection_metadata_v<int>);
    CHECK_FALSE(is_reflection_metadata_v<float>);
    CHECK_FALSE(is_reflection_metadata_v<std::string>);
}

TEST_CASE("is_reflection_flags trait")
{
    // Flags metadata types satisfy is_reflection_flags_v.
    CHECK(is_reflection_flags_v<UIFlags>);

    // Non-flags metadata types do not.
    CHECK_FALSE(is_reflection_flags_v<UILabel>);
    CHECK_FALSE(is_reflection_flags_v<int>);
}

TEST_CASE("UILabel metadata storage and retrieval")
{
    auto prop = make_prop<float>(
        "albedo",
        &MetadataTestClass::albedo,
        [](MetadataTestClass& obj, const float& v)
        {
            obj.set_albedo(v);
        },
        ui_label("Albedo Color")
    );

    REQUIRE(prop->ui_label() != nullptr);
    CHECK(prop->ui_label()->text == "Albedo Color");
}

TEST_CASE("UIGroup metadata storage and retrieval")
{
    auto prop = make_prop<float>(
        "albedo",
        &MetadataTestClass::albedo,
        [](MetadataTestClass& obj, const float& v)
        {
            obj.set_albedo(v);
        },
        ui_group("Appearance/Color")
    );

    REQUIRE(prop->ui_group() != nullptr);
    CHECK(prop->ui_group()->path == "Appearance/Color");
}

TEST_CASE("UIDragSpeed metadata storage and retrieval")
{
    auto prop = make_prop<float>(
        "albedo",
        &MetadataTestClass::albedo,
        [](MetadataTestClass& obj, const float& v)
        {
            obj.set_albedo(v);
        },
        ui_drag_speed(0.01f)
    );

    REQUIRE(prop->ui_drag_speed() != nullptr);
    CHECK(prop->ui_drag_speed()->speed == doctest::Approx(0.01f));
}

TEST_CASE("UIEnableIf metadata storage and retrieval")
{
    auto prop = make_prop<float>(
        "speed",
        &MetadataTestClass::speed,
        [](MetadataTestClass& obj, const float& v)
        {
            obj.set_speed(v);
        },
        ui_enable_if<MetadataTestClass>(
            [](const MetadataTestClass& obj)
            {
                return obj.use_feature();
            }
        )
    );

    const UIEnableIf* ei = prop->ui_enable_if();
    REQUIRE(ei != nullptr);

    MetadataTestClass obj;
    obj.set_use_feature(false);
    CHECK_FALSE(ei->predicate(&obj));

    obj.set_use_feature(true);
    CHECK(ei->predicate(&obj));
}

TEST_CASE("UIFlags OR-accumulation")
{
    SUBCASE("single flag")
    {
        auto prop = make_prop<float>(
            "albedo",
            &MetadataTestClass::albedo,
            [](MetadataTestClass& obj, const float& v)
            {
                obj.set_albedo(v);
            },
            UIFlags::advanced
        );

        CHECK(prop->ui_flags() == UIFlags::advanced);
    }

    SUBCASE("multiple flags are OR'd together")
    {
        auto prop = make_prop<float>(
            "albedo",
            &MetadataTestClass::albedo,
            [](MetadataTestClass& obj, const float& v)
            {
                obj.set_albedo(v);
            },
            UIFlags::display_as_color,
            UIFlags::advanced,
            UIFlags::hidden
        );

        CHECK(prop->ui_flags() == (UIFlags::display_as_color | UIFlags::advanced | UIFlags::hidden));
    }

    SUBCASE("no flags returns none")
    {
        auto prop = make_prop<float>(
            "albedo",
            &MetadataTestClass::albedo,
            [](MetadataTestClass& obj, const float& v)
            {
                obj.set_albedo(v);
            }
        );

        CHECK(prop->ui_flags() == UIFlags::none);
    }
}

TEST_CASE("Multiple metadata types on one property")
{
    auto prop = make_prop<float>(
        "albedo",
        &MetadataTestClass::albedo,
        [](MetadataTestClass& obj, const float& v)
        {
            obj.set_albedo(v);
        },
        default_value(0.5f),
        value_range(0.0, 1.0),
        ui_label("Albedo"),
        ui_group("Appearance"),
        ui_drag_speed(0.01f),
        UIFlags::advanced,
        "An albedo property."
    );

    REQUIRE(prop->ui_label() != nullptr);
    CHECK(prop->ui_label()->text == "Albedo");

    REQUIRE(prop->ui_group() != nullptr);
    CHECK(prop->ui_group()->path == "Appearance");

    REQUIRE(prop->ui_drag_speed() != nullptr);
    CHECK(prop->ui_drag_speed()->speed == doctest::Approx(0.01f));

    CHECK(prop->ui_flags() == UIFlags::advanced);

    const ValueRange* vr = prop->value_range();
    REQUIRE(vr != nullptr);
    CHECK(vr->min == doctest::Approx(0.0));
    CHECK(vr->max == doctest::Approx(1.0));

    CHECK(prop->has_default_value());
    CHECK(prop->doc() == "An albedo property.");
}

TEST_CASE("Convenience accessors return defaults when metadata absent")
{
    auto prop = make_prop<float>(
        "plain",
        &MetadataTestClass::albedo,
        [](MetadataTestClass& obj, const float& v)
        {
            obj.set_albedo(v);
        }
    );

    CHECK(prop->doc().empty());
    CHECK(prop->value_range() == nullptr);
    CHECK(prop->ui_flags() == UIFlags::none);
    CHECK(prop->ui_label() == nullptr);
    CHECK(prop->ui_group() == nullptr);
    CHECK(prop->ui_drag_speed() == nullptr);
    CHECK(prop->ui_enable_if() == nullptr);
    CHECK(prop->enum_descriptor() == nullptr);
}

// ---------------------------------------------------------------------------
// EnumDescriptor tests
// ---------------------------------------------------------------------------

namespace {

enum class TestColor : int {
    red = 0,
    green = 1,
    blue = 2,
};

SGL_ENUM_INFO(
    TestColor,
    {
        {TestColor::red, "red"},
        {TestColor::green, "green"},
        {TestColor::blue, "blue"},
    }
)
SGL_ENUM_REGISTER(TestColor)

enum class TestFlags : uint32_t {
    none = 0,
    readable = 1 << 0,
    writable = 1 << 1,
    executable = 1 << 2,
};
FALCOR_ENUM_CLASS_OPERATORS(TestFlags)

SGL_ENUM_FLAGS_INFO(
    TestFlags,
    {
        {TestFlags::readable, "readable"},
        {TestFlags::writable, "writable"},
        {TestFlags::executable, "executable"},
    }
)
SGL_ENUM_REGISTER(TestFlags)

} // namespace

TEST_CASE("is_reflection_metadata trait for EnumDescriptor")
{
    CHECK(is_reflection_metadata_v<EnumDescriptor>);
}

TEST_CASE("enum_descriptor_from_sgl_enum for non-flags enum")
{
    EnumDescriptor desc = reflection::detail::enum_descriptor_from_sgl_enum<TestColor>();
    CHECK_FALSE(desc.is_flags);
    REQUIRE(desc.items.size() == 3);
    CHECK(desc.items[0].value == 0);
    CHECK(desc.items[0].name == "red");
    CHECK(desc.items[1].value == 1);
    CHECK(desc.items[1].name == "green");
    CHECK(desc.items[2].value == 2);
    CHECK(desc.items[2].name == "blue");
}

TEST_CASE("enum_descriptor_from_sgl_enum for flags enum")
{
    EnumDescriptor desc = reflection::detail::enum_descriptor_from_sgl_enum<TestFlags>();
    CHECK(desc.is_flags);
    REQUIRE(desc.items.size() == 3);
    CHECK(desc.items[0].value == 1);
    CHECK(desc.items[0].name == "readable");
    CHECK(desc.items[1].value == 2);
    CHECK(desc.items[1].name == "writable");
    CHECK(desc.items[2].value == 4);
    CHECK(desc.items[2].name == "executable");
}

TEST_CASE("EnumDescriptor auto-injected by extract_all")
{
    // Create a property descriptor for an enum type -- EnumDescriptor should be auto-injected.
    auto prop = std::make_unique<TypedPropertyDescriptor<TestColor, MetadataTestClass>>(
        "color",
        [](const MetadataTestClass&)
        {
            return TestColor::red;
        },
        [](MetadataTestClass&, const TestColor&)
        {
        },
        "A color property."
    );

    const EnumDescriptor* ed = prop->enum_descriptor();
    REQUIRE(ed != nullptr);
    CHECK_FALSE(ed->is_flags);
    REQUIRE(ed->items.size() == 3);
    CHECK(ed->items[0].name == "red");
}

TEST_CASE("EnumDescriptor not injected for non-enum types")
{
    auto prop = make_prop<float>(
        "albedo",
        &MetadataTestClass::albedo,
        [](MetadataTestClass& obj, const float& v)
        {
            obj.set_albedo(v);
        }
    );

    CHECK(prop->enum_descriptor() == nullptr);
}

TEST_CASE("get_enum_as_int64 and set_enum_from_int64 for non-flags enum")
{
    MetadataTestClass obj;
    TestColor color = TestColor::red;

    auto prop = std::make_unique<TypedPropertyDescriptor<TestColor, MetadataTestClass>>(
        "color",
        [&](const MetadataTestClass&)
        {
            return color;
        },
        [&](MetadataTestClass&, const TestColor& v)
        {
            color = v;
        },
        "A color property."
    );

    // Read enum as int64.
    int64_t val = prop->get_enum_as_int64(&obj);
    CHECK(val == 0); // TestColor::red == 0

    // Set enum from int64.
    prop->set_enum_from_int64(&obj, 2);
    CHECK(color == TestColor::blue);

    // Verify round-trip.
    val = prop->get_enum_as_int64(&obj);
    CHECK(val == 2);
}

TEST_CASE("get_enum_as_int64 and set_enum_from_int64 for flags enum")
{
    MetadataTestClass obj;
    TestFlags flags = TestFlags::none;

    auto prop = std::make_unique<TypedPropertyDescriptor<TestFlags, MetadataTestClass>>(
        "flags",
        [&](const MetadataTestClass&)
        {
            return flags;
        },
        [&](MetadataTestClass&, const TestFlags& v)
        {
            flags = v;
        },
        "A flags property."
    );

    int64_t combined = static_cast<int64_t>(TestFlags::readable | TestFlags::executable);
    prop->set_enum_from_int64(&obj, combined);
    CHECK(flags == (TestFlags::readable | TestFlags::executable));

    int64_t val = prop->get_enum_as_int64(&obj);
    CHECK(val == combined);
}

TEST_CASE("Integral typed property enum access requires enum metadata")
{
    MetadataTestClass obj;
    auto setter = [](MetadataTestClass& instance, const int& value)
    {
        instance.set_count(value);
    };
    auto enum_prop
        = make_prop<int>("mode", &MetadataTestClass::count, setter, enum_descriptor({{10, "Ten"}, {20, "Twenty"}}));
    auto plain_prop = make_prop<int>("count", &MetadataTestClass::count, setter);

    obj.set_count(20);
    CHECK(enum_prop->get_enum_as_int64(&obj) == 20);

    enum_prop->set_enum_from_int64(&obj, 10);
    CHECK(obj.count() == 10);

    CHECK_THROWS_WITH(plain_prop->get_enum_as_int64(&obj), doctest::Contains("integral type has no enum metadata"));
    CHECK_THROWS_WITH(
        plain_prop->set_enum_from_int64(&obj, 20),
        doctest::Contains("integral type has no enum metadata")
    );
    CHECK(obj.count() == 10);
}

TEST_CASE("enum property read_from_properties accepts integer values")
{
    MetadataTestClass obj;
    TestFlags flags = TestFlags::none;

    auto prop = std::make_unique<TypedPropertyDescriptor<TestFlags, MetadataTestClass>>(
        "flags",
        [&](const MetadataTestClass&)
        {
            return flags;
        },
        [&](MetadataTestClass&, const TestFlags& v)
        {
            flags = v;
        },
        "A flags property."
    );

    const int64_t combined = static_cast<int64_t>(TestFlags::readable | TestFlags::executable);
    Properties props;
    props.set<int64_t>("flags", combined);

    CHECK(prop->read_from_properties(&obj, props));
    CHECK(flags == (TestFlags::readable | TestFlags::executable));
}

TEST_CASE("get_enum_as_int64 throws for non-enum property")
{
    auto prop = make_prop<float>(
        "albedo",
        &MetadataTestClass::albedo,
        [](MetadataTestClass& obj, const float& v)
        {
            obj.set_albedo(v);
        }
    );

    MetadataTestClass obj;
    CHECK_THROWS(prop->get_enum_as_int64(&obj));
    CHECK_THROWS(prop->set_enum_from_int64(&obj, 42));
}

TEST_SUITE_END();
