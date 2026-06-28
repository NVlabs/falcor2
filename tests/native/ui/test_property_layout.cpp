// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "testing.h"

#include "falcor2/core/reflection.h"
#include "falcor2/core/reflection/dynamic_properties.h"
#include "falcor2/ui/property_editor.h"

using namespace falcor;
using namespace falcor::reflection;
using namespace falcor::ui;

// ---------------------------------------------------------------------------
// Test helper classes
// ---------------------------------------------------------------------------

/// A simple class with ungrouped and grouped properties for layout tests.
class LayoutTestClass {
public:
    using ReflectedBase = void;
    static constexpr const char* reflected_class_name() { return "LayoutTestClass"; }

    float albedo() const { return m_albedo; }
    void set_albedo(float v) { m_albedo = v; }

    float roughness() const { return m_roughness; }
    void set_roughness(float v) { m_roughness = v; }

    float3 color() const { return m_color; }
    void set_color(float3 v) { m_color = v; }

    float metallic() const { return m_metallic; }
    void set_metallic(float v) { m_metallic = v; }

    float ior() const { return m_ior; }
    void set_ior(float v) { m_ior = v; }

    bool enabled() const { return m_enabled; }
    void set_enabled(bool v) { m_enabled = v; }

    /// Reflect this class.
    template<reflection::ClassReflector R>
    static void reflect(R& r)
    {
        r
            // Ungrouped property
            .def_prop_rw("enabled", &LayoutTestClass::enabled, &LayoutTestClass::set_enabled)
            // Group "Appearance"
            .def_prop_rw("albedo", &LayoutTestClass::albedo, &LayoutTestClass::set_albedo, ui_group("Appearance"))
            // Nested group "Appearance/Color"
            .def_prop_rw(
                "color",
                &LayoutTestClass::color,
                &LayoutTestClass::set_color,
                ui_group("Appearance/Color"),
                UIFlags::display_as_color
            )
            // Group "Appearance" again (same group, second property)
            .def_prop_rw(
                "roughness",
                &LayoutTestClass::roughness,
                &LayoutTestClass::set_roughness,
                ui_group("Appearance")
            )
            // Group "Advanced"
            .def_prop_rw("metallic", &LayoutTestClass::metallic, &LayoutTestClass::set_metallic, ui_group("Advanced"))
            // Group "Advanced" with nested path
            .def_prop_rw("ior", &LayoutTestClass::ior, &LayoutTestClass::set_ior, ui_group("Advanced/Optics"));
    }

private:
    float m_albedo{0.5f};
    float m_roughness{0.5f};
    float3 m_color{1.f, 1.f, 1.f};
    float m_metallic{0.0f};
    float m_ior{1.5f};
    bool m_enabled{true};
};

FALCOR_STATIC_ONCE(register_type<LayoutTestClass>());

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST_SUITE_BEGIN("ui");

TEST_CASE("PropertyLayout: ungrouped properties at root")
{
    const auto& cd = get_class_descriptor(typeid(LayoutTestClass));
    PropertyLayout layout = build_property_layout(cd);

    // "enabled" is not in any group, should be at the root level.
    REQUIRE(layout.root.properties.size() == 1);
    CHECK(layout.root.properties[0]->name() == "enabled");
}

TEST_CASE("PropertyLayout: group hierarchy")
{
    const auto& cd = get_class_descriptor(typeid(LayoutTestClass));
    PropertyLayout layout = build_property_layout(cd);

    // Root should have two top-level groups: "Appearance" and "Advanced"
    // (in first-mention order).
    REQUIRE(layout.root.children.size() == 2);
    CHECK(layout.root.children[0].name == "Appearance");
    CHECK(layout.root.children[1].name == "Advanced");
}

TEST_CASE("PropertyLayout: properties within groups")
{
    const auto& cd = get_class_descriptor(typeid(LayoutTestClass));
    PropertyLayout layout = build_property_layout(cd);

    // "Appearance" group should have "albedo" and "roughness" in registration order.
    const auto& appearance = layout.root.children[0];
    REQUIRE(appearance.name == "Appearance");
    REQUIRE(appearance.properties.size() == 2);
    CHECK(appearance.properties[0]->name() == "albedo");
    CHECK(appearance.properties[1]->name() == "roughness");
}

TEST_CASE("PropertyLayout: nested group hierarchy")
{
    const auto& cd = get_class_descriptor(typeid(LayoutTestClass));
    PropertyLayout layout = build_property_layout(cd);

    // "Appearance" should have a child group "Color" containing "color".
    const auto& appearance = layout.root.children[0];
    REQUIRE(appearance.children.size() == 1);
    CHECK(appearance.children[0].name == "Color");
    REQUIRE(appearance.children[0].properties.size() == 1);
    CHECK(appearance.children[0].properties[0]->name() == "color");

    // "Advanced" should have a child group "Optics" containing "ior".
    const auto& advanced = layout.root.children[1];
    REQUIRE(advanced.name == "Advanced");
    REQUIRE(advanced.properties.size() == 1);
    CHECK(advanced.properties[0]->name() == "metallic");
    REQUIRE(advanced.children.size() == 1);
    CHECK(advanced.children[0].name == "Optics");
    REQUIRE(advanced.children[0].properties.size() == 1);
    CHECK(advanced.children[0].properties[0]->name() == "ior");
}

TEST_CASE("PropertyLayout: groups ordered by first mention")
{
    const auto& cd = get_class_descriptor(typeid(LayoutTestClass));
    PropertyLayout layout = build_property_layout(cd);

    // "Appearance" appears first (via "albedo"), "Advanced" second (via "metallic").
    CHECK(layout.root.children[0].name == "Appearance");
    CHECK(layout.root.children[1].name == "Advanced");
}

TEST_CASE("PropertyLayout: dynamic properties included")
{
    const auto& cd = get_class_descriptor(typeid(LayoutTestClass));

    DynamicPropertySet dyn;
    dyn.add_property<float>("dyn_float", 0.0f);
    dyn.add_property<int>("dyn_grouped", 0, false, ui_group("DynGroup"));

    PropertyLayout layout = build_property_layout(cd, &dyn);

    // "dyn_float" should be ungrouped at root alongside "enabled".
    bool found_dyn_float = false;
    for (const auto* prop : layout.root.properties) {
        if (prop->name() == "dyn_float")
            found_dyn_float = true;
    }
    CHECK(found_dyn_float);

    // "dyn_grouped" should be in a "DynGroup" child.
    bool found_dyn_group = false;
    for (const auto& child : layout.root.children) {
        if (child.name == "DynGroup") {
            found_dyn_group = true;
            REQUIRE(child.properties.size() == 1);
            CHECK(child.properties[0]->name() == "dyn_grouped");
        }
    }
    CHECK(found_dyn_group);
}

TEST_CASE("PropertyLayout: stores class descriptor and dynamic set info")
{
    const auto& cd = get_class_descriptor(typeid(LayoutTestClass));

    SUBCASE("without dynamic properties")
    {
        PropertyLayout layout = build_property_layout(cd);
        CHECK(layout.class_desc == &cd);
        CHECK(layout.dynamic_set == nullptr);
        CHECK(layout.dynamic_generation == 0);
    }

    SUBCASE("with dynamic properties")
    {
        DynamicPropertySet dyn;
        dyn.add_property<float>("dyn_val", 0.0f);
        uint64_t gen = dyn.generation();

        PropertyLayout layout = build_property_layout(cd, &dyn);
        CHECK(layout.class_desc == &cd);
        CHECK(layout.dynamic_set == &dyn);
        CHECK(layout.dynamic_generation == gen);
    }
}

TEST_CASE("PropertyLayout: generation-based invalidation")
{
    const auto& cd = get_class_descriptor(typeid(LayoutTestClass));

    DynamicPropertySet dyn;
    dyn.add_property<float>("val1", 0.0f);
    uint64_t gen1 = dyn.generation();

    PropertyLayout layout1 = build_property_layout(cd, &dyn);
    CHECK(layout1.dynamic_generation == gen1);

    // Structural change increments generation.
    dyn.add_property<int>("val2", 0);
    uint64_t gen2 = dyn.generation();
    CHECK(gen2 > gen1);

    PropertyLayout layout2 = build_property_layout(cd, &dyn);
    CHECK(layout2.dynamic_generation == gen2);

    // New layout includes the new property.
    bool found_val2 = false;
    for (const auto* prop : layout2.root.properties) {
        if (prop->name() == "val2")
            found_val2 = true;
    }
    CHECK(found_val2);
}

TEST_CASE("PropertyLayoutCache: basic insert and find")
{
    PropertyLayoutCache cache;
    const auto& cd = get_class_descriptor(typeid(LayoutTestClass));
    PropertyLayout layout = build_property_layout(cd);

    int dummy_instance = 42;
    const void* key = &dummy_instance;

    CHECK(cache.find(key) == nullptr);

    cache.insert(key, std::move(layout));

    PropertyLayout* found = cache.find(key);
    REQUIRE(found != nullptr);
    CHECK(found->class_desc == &cd);
}

TEST_CASE("PropertyLayoutCache: evicts LRU entry at capacity")
{
    PropertyLayoutCache cache(2); // Capacity of 2.
    const auto& cd = get_class_descriptor(typeid(LayoutTestClass));

    int a = 0, b = 0, c = 0;
    cache.insert(&a, build_property_layout(cd));
    cache.insert(&b, build_property_layout(cd));

    // Both should be present.
    CHECK(cache.find(&a) != nullptr);
    CHECK(cache.find(&b) != nullptr);

    // Inserting a third should evict the LRU.
    // After finding &a and &b above, &b was found last, then &a was found,
    // so &a is MRU, &b is second. Actually let's be precise:
    // After insert(&a), insert(&b): order is [b, a] (b is MRU).
    // find(&a) makes a MRU: [a, b].
    // find(&b) makes b MRU: [b, a].
    // Now insert(&c) should evict &a (LRU).
    cache.insert(&c, build_property_layout(cd));

    CHECK(cache.find(&a) == nullptr);
    CHECK(cache.find(&b) != nullptr);
    CHECK(cache.find(&c) != nullptr);
}

TEST_CASE("PropertyLayoutCache: clear removes all entries")
{
    PropertyLayoutCache cache;
    const auto& cd = get_class_descriptor(typeid(LayoutTestClass));

    int a = 0, b = 0;
    cache.insert(&a, build_property_layout(cd));
    cache.insert(&b, build_property_layout(cd));

    cache.clear();

    CHECK(cache.find(&a) == nullptr);
    CHECK(cache.find(&b) == nullptr);
}

TEST_CASE("PropertyLayoutCache: update replaces existing entry")
{
    PropertyLayoutCache cache;

    int instance = 0;
    const void* key = &instance;

    // Insert with LayoutTestClass.
    const auto& cd1 = get_class_descriptor(typeid(LayoutTestClass));
    cache.insert(key, build_property_layout(cd1));

    PropertyLayout* found = cache.find(key);
    REQUIRE(found != nullptr);
    CHECK(found->class_desc == &cd1);
    CHECK(found->dynamic_set == nullptr);

    // Replace with another LayoutTestClass + dynamic properties.
    const auto& cd2 = get_class_descriptor(typeid(LayoutTestClass));
    DynamicPropertySet dyn;
    cache.insert(key, build_property_layout(cd2, &dyn));

    found = cache.find(key);
    REQUIRE(found != nullptr);
    CHECK(found->class_desc == &cd2);
    CHECK(found->dynamic_set == &dyn);
}

TEST_CASE("PropertyLayout: span-based overload groups properties correctly")
{
    // Build descriptors manually using StoredPropertyDescriptor.
    StoredPropertyDescriptor<float> ungrouped("ungrouped", 0.0f, false);
    StoredPropertyDescriptor<float> grouped("grouped", 0.5f, false, ui_group("MyGroup"));
    StoredPropertyDescriptor<int> nested("nested", 42, false, ui_group("MyGroup/Sub"));

    const PropertyDescriptor* descs[] = {&ungrouped, &grouped, &nested};
    std::span<const PropertyDescriptor* const> span(descs);

    PropertyLayout layout = build_property_layout(span);

    // Ungrouped property at root.
    REQUIRE(layout.root.properties.size() == 1);
    CHECK(layout.root.properties[0]->name() == "ungrouped");

    // class_desc and dynamic_set should be null.
    CHECK(layout.class_desc == nullptr);
    CHECK(layout.dynamic_set == nullptr);

    // "MyGroup" group with "grouped" property and "Sub" child.
    REQUIRE(layout.root.children.size() == 1);
    CHECK(layout.root.children[0].name == "MyGroup");
    REQUIRE(layout.root.children[0].properties.size() == 1);
    CHECK(layout.root.children[0].properties[0]->name() == "grouped");

    REQUIRE(layout.root.children[0].children.size() == 1);
    CHECK(layout.root.children[0].children[0].name == "Sub");
    REQUIRE(layout.root.children[0].children[0].properties.size() == 1);
    CHECK(layout.root.children[0].children[0].properties[0]->name() == "nested");
}

TEST_CASE("PropertyLayout: span-based overload with empty span")
{
    std::span<const PropertyDescriptor* const> span;
    PropertyLayout layout = build_property_layout(span);

    CHECK(layout.root.properties.empty());
    CHECK(layout.root.children.empty());
    CHECK(layout.class_desc == nullptr);
    CHECK(layout.dynamic_set == nullptr);
}

TEST_SUITE_END();
