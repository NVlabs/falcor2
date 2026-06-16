// SPDX-License-Identifier: Apache-2.0

#include "testing.h"

#include "falcor2/core/reflection.h"
#include "falcor2/core/properties.h"
#include "falcor2/core/reflected_object.h"
#include "falcor2/render/scene_object.h"

using namespace falcor;
using namespace falcor::reflection;

// ---------------------------------------------------------------------------
// Test classes
// ---------------------------------------------------------------------------

namespace {
enum class TestMode : int {
    off = 0,
    fast = 1,
    quality = 2,
};
} // namespace

/// A simple ReflectedObject with reflected properties for direct serializer/deserializer tests.
class SimpleReflected : public ReflectedObject {
    FALCOR_REFLECTED_OBJECT(SimpleReflected, ReflectedObject)
public:
    int foo() const { return m_foo; }
    void set_foo(int value) { m_foo = value; }

    float bar() const { return m_bar; }
    void set_bar(float value) { m_bar = value; }

    const std::string& label() const { return m_label; }
    void set_label(const std::string& value) { m_label = value; }

    int read_only_prop() const { return 42; }

    TestMode mode() const { return m_mode; }
    void set_mode(TestMode value) { m_mode = value; }

    /// Reflect this class.
    template<reflection::ClassReflector R>
    static void reflect(R& r)
    {
        r //
            .def_prop_rw(
                "foo",
                &SimpleReflected::foo,
                &SimpleReflected::set_foo,
                "An integer property.",
                default_value(0),
                value_range(-10, 10)
            )
            .def_prop_rw(
                "bar",
                &SimpleReflected::bar,
                &SimpleReflected::set_bar,
                "A float property.",
                default_value(1.f),
                value_range(0.0, 100.0)
            )
            .def_prop_rw(
                "label",
                &SimpleReflected::label,
                &SimpleReflected::set_label,
                "A string property.",
                default_value(std::string("default"))
            )
            .def_prop_ro("read_only_prop", &SimpleReflected::read_only_prop, "A read-only property.", default_value(42))
            .def_prop_rw("mode", &SimpleReflected::mode, &SimpleReflected::set_mode, "Rendering mode.");
    }

private:
    int m_foo{0};
    float m_bar{1.f};
    std::string m_label{"default"};
    TestMode m_mode{TestMode::off};
};

FALCOR_STATIC_ONCE(register_type<SimpleReflected>());

/// A non-reflected class for concept testing.
class NotReflected : public Object {
    FALCOR_OBJECT(NotReflected)
};

/// A test SceneObject with reflected properties.
class TestSceneObj : public SceneObject {
    FALCOR_SCENE_OBJECT(TestSceneObj, SceneObject)
public:
    SceneObjectKind kind() const override { return SceneObjectKind::component; }

    int value() const { return m_value; }
    void set_value(int v) { m_value = v; }

    const std::string& label() const { return m_label; }
    void set_label(const std::string& v) { m_label = v; }

    /// Reflect this class.
    template<reflection::ClassReflector R>
    static void reflect(R& r)
    {
        r //
            .def_prop_rw(
                "value",
                &TestSceneObj::value,
                &TestSceneObj::set_value,
                "Integer value.",
                default_value(0),
                value_range(0, 100)
            )
            .def_prop_rw(
                "label",
                &TestSceneObj::label,
                &TestSceneObj::set_label,
                "String label.",
                default_value(std::string("default"))
            );
    }

private:
    int m_value{0};
    std::string m_label{"default"};
};

FALCOR_STATIC_ONCE(register_type<TestSceneObj>());

/// A derived SceneObject for testing inheritance chaining via FALCOR_SCENE_OBJECT.
class DerivedSceneObj : public TestSceneObj {
    FALCOR_SCENE_OBJECT(DerivedSceneObj, TestSceneObj)
public:
    double extra() const { return m_extra; }
    void set_extra(double v) { m_extra = v; }

    /// Reflect this class.
    template<reflection::ClassReflector R>
    static void reflect(R& r)
    {
        r //
            .def_prop_rw(
                "extra",
                &DerivedSceneObj::extra,
                &DerivedSceneObj::set_extra,
                "Extra value.",
                default_value(3.14),
                value_range(0.0, 1000.0)
            );
    }

private:
    double m_extra{3.14};
};

FALCOR_STATIC_ONCE(register_type<DerivedSceneObj>());

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST_SUITE_BEGIN("reflection");

TEST_CASE("ReflectedClass concept")
{
    // Types with a static reflect() template accepting a ClassReflector satisfy ReflectedClass.
    CHECK(ReflectedClass<SimpleReflected>);
    CHECK(ReflectedClass<TestSceneObj>);
    CHECK(ReflectedClass<DerivedSceneObj>);
    // Object itself and non-reflected types do not.
    CHECK_FALSE(ReflectedClass<Object>);
    CHECK_FALSE(ReflectedClass<ReflectedObject>);
    CHECK_FALSE(ReflectedClass<NotReflected>);
}

TEST_CASE("is_reflection_metadata trait")
{
    CHECK(is_reflection_metadata_v<DefaultValue<int>>);
    CHECK(is_reflection_metadata_v<DefaultValue<float>>);
    CHECK(is_reflection_metadata_v<DefaultValue<std::string>>);
    CHECK(is_reflection_metadata_v<ValueRange>);
    CHECK_FALSE(is_reflection_metadata_v<int>);
    CHECK_FALSE(is_reflection_metadata_v<float>);
    CHECK_FALSE(is_reflection_metadata_v<std::string>);
}

TEST_CASE("PropertiesSerializer writes properties from object")
{
    SimpleReflected obj;
    obj.set_foo(42);
    obj.set_bar(3.14f);
    obj.set_label("hello");
    obj.set_mode(TestMode::quality);

    Properties props;
    serialize_properties(obj, props);

    CHECK(props.get<int>("foo") == 42);
    CHECK(props.get<float>("bar") == doctest::Approx(3.14f));
    CHECK(props.get<std::string>("label") == "hello");
    // Read-only properties are serialized too.
    CHECK(props.get<int>("read_only_prop") == 42);
    // Enum property.
    CHECK(props.type("mode") == PropertyType::enum_);
    CHECK(props.get<TestMode>("mode") == TestMode::quality);
}

TEST_CASE("PropertiesDeserializer reads properties into object")
{
    SimpleReflected obj;

    Properties props;
    props.set<int>("foo", 99);
    props.set<float>("bar", 7.5f);
    props.set<std::string>("label", "world");
    props.set<TestMode>("mode", TestMode::fast);

    deserialize_properties(obj, props);

    CHECK(obj.foo() == 99);
    CHECK(obj.bar() == doctest::Approx(7.5f));
    CHECK(obj.label() == "world");
    // Read-only property is unaffected.
    CHECK(obj.read_only_prop() == 42);
    CHECK(obj.mode() == TestMode::fast);
}

TEST_CASE("PropertiesDeserializer skips missing properties")
{
    SimpleReflected obj;
    obj.set_foo(5);
    obj.set_bar(2.f);
    obj.set_label("original");

    Properties props;
    props.set<int>("foo", 99); // Only set foo.

    deserialize_properties(obj, props);

    CHECK(obj.foo() == 99);
    CHECK(obj.bar() == doctest::Approx(2.f));
    CHECK(obj.label() == "original");
}

TEST_CASE("Round-trip serialization through Properties")
{
    SimpleReflected obj1;
    obj1.set_foo(123);
    obj1.set_bar(45.6f);
    obj1.set_label("test");
    obj1.set_mode(TestMode::quality);

    // Serialize.
    Properties props;
    serialize_properties(obj1, props);

    // Deserialize into a new object.
    SimpleReflected obj2;
    deserialize_properties(obj2, props);

    CHECK(obj2.foo() == 123);
    CHECK(obj2.bar() == doctest::Approx(45.6f));
    CHECK(obj2.label() == "test");
    CHECK(obj2.mode() == TestMode::quality);
}

TEST_CASE("ReflectedObject::properties serialization")
{
    auto obj = ref<SimpleReflected>(new SimpleReflected());
    obj->set_foo(42);
    obj->set_bar(3.14f);
    obj->set_label("hello");
    obj->set_mode(TestMode::fast);

    Properties props = obj->properties();
    CHECK(props.get<int>("foo") == 42);
    CHECK(props.get<float>("bar") == doctest::Approx(3.14f));
    CHECK(props.get<std::string>("label") == "hello");
    CHECK(props.get<int>("read_only_prop") == 42);
    CHECK(props.get<TestMode>("mode") == TestMode::fast);
}

TEST_CASE("ReflectedObject::set_properties deserialization")
{
    auto obj = ref<SimpleReflected>(new SimpleReflected());

    Properties props;
    props.set<int>("foo", 99);
    props.set<float>("bar", 7.5f);
    props.set<std::string>("label", "world");
    obj->set_properties(props);

    CHECK(obj->foo() == 99);
    CHECK(obj->bar() == doctest::Approx(7.5f));
    CHECK(obj->label() == "world");
    CHECK(obj->read_only_prop() == 42);
}

TEST_CASE("ReflectedObject::properties round-trip")
{
    auto obj1 = ref<SimpleReflected>(new SimpleReflected());
    obj1->set_foo(123);
    obj1->set_bar(45.6f);
    obj1->set_label("test");
    obj1->set_mode(TestMode::quality);

    Properties props = obj1->properties();

    auto obj2 = ref<SimpleReflected>(new SimpleReflected());
    obj2->set_properties(props);

    CHECK(obj2->foo() == 123);
    CHECK(obj2->bar() == doctest::Approx(45.6f));
    CHECK(obj2->label() == "test");
    CHECK(obj2->mode() == TestMode::quality);
}

TEST_CASE("SceneObject::properties serialization")
{
    auto obj = ref<TestSceneObj>(new TestSceneObj());
    obj->set_value(42);
    obj->set_label("hello");

    Properties props = obj->properties();
    CHECK(props.get<int>("value") == 42);
    CHECK(props.get<std::string>("label") == "hello");
}

TEST_CASE("SceneObject::set_properties deserialization")
{
    auto obj = ref<TestSceneObj>(new TestSceneObj());

    Properties props;
    props.set<int>("value", 99);
    props.set<std::string>("label", "world");
    obj->set_properties(props);

    CHECK(obj->value() == 99);
    CHECK(obj->label() == "world");
}

TEST_CASE("SceneObject::set_properties round-trip")
{
    auto obj1 = ref<TestSceneObj>(new TestSceneObj());
    obj1->set_value(77);
    obj1->set_label("round-trip");

    Properties props = obj1->properties();

    auto obj2 = ref<TestSceneObj>(new TestSceneObj());
    obj2->set_properties(props);

    CHECK(obj2->value() == 77);
    CHECK(obj2->label() == "round-trip");
}

TEST_CASE("Inheritance chaining via TypeRegistry")
{
    auto obj = ref<DerivedSceneObj>(new DerivedSceneObj());
    obj->set_value(7);
    obj->set_label("base");
    obj->set_extra(2.71);

    SUBCASE("properties includes both base and derived")
    {
        Properties props = obj->properties();
        CHECK(props.get<int>("value") == 7);
        CHECK(props.get<std::string>("label") == "base");
        CHECK(props.get<double>("extra") == doctest::Approx(2.71));
    }

    SUBCASE("set_properties applies to both base and derived")
    {
        Properties props;
        props.set<int>("value", 100);
        props.set<std::string>("label", "changed");
        props.set<double>("extra", 1.23);
        obj->set_properties(props);

        CHECK(obj->value() == 100);
        CHECK(obj->label() == "changed");
        CHECK(obj->extra() == doctest::Approx(1.23));
    }

    SUBCASE("round-trip through Properties preserves all values")
    {
        Properties props = obj->properties();

        auto obj2 = ref<DerivedSceneObj>(new DerivedSceneObj());
        obj2->set_properties(props);

        CHECK(obj2->value() == 7);
        CHECK(obj2->label() == "base");
        CHECK(obj2->extra() == doctest::Approx(2.71));
    }

    SUBCASE("partial set_properties only updates present keys")
    {
        Properties props;
        props.set<double>("extra", 9.99);
        obj->set_properties(props);

        // Only extra is updated; base properties keep their values.
        CHECK(obj->value() == 7);
        CHECK(obj->label() == "base");
        CHECK(obj->extra() == doctest::Approx(9.99));
    }
}

TEST_CASE("ClassDescriptor::base() returns parent descriptor")
{
    auto& reg = TypeRegistry::get();
    const ClassDescriptor* derived_desc = reg.find(typeid(DerivedSceneObj));
    REQUIRE(derived_desc != nullptr);

    const ClassDescriptor* base_desc = derived_desc->base();
    REQUIRE(base_desc != nullptr);
    CHECK(base_desc->name() == "TestSceneObj");

    // TestSceneObj's base should be SceneObject (or nullptr if SceneObject is not registered).
    const ClassDescriptor* base_base = base_desc->base();
    // Either way, it's not DerivedSceneObj.
    if (base_base)
        CHECK(base_base->name() != "DerivedSceneObj");
}

TEST_CASE("ClassDescriptor::all_property_range includes base chain")
{
    auto& reg = TypeRegistry::get();
    const ClassDescriptor* desc = reg.find(typeid(DerivedSceneObj));
    REQUIRE(desc != nullptr);

    auto range = desc->all_property_range();
    // TestSceneObj(2: value, label) + DerivedSceneObj(1: extra) = 3
    CHECK(range.size() == 3);

    std::vector<const PropertyDescriptor*> all(range.begin(), range.end());
    // Base properties come first.
    CHECK(all[0]->name() == "value");
    CHECK(all[1]->name() == "label");
    CHECK(all[2]->name() == "extra");
}

// ---------------------------------------------------------------------------
// def_rw / def_ro test class
// ---------------------------------------------------------------------------

/// A ReflectedObject using def_rw/def_ro with member pointers instead of getters/setters.
class MemberReflected : public ReflectedObject {
    FALCOR_REFLECTED_OBJECT(MemberReflected, ReflectedObject)
public:
    int m_value{0};
    float m_factor{1.f};
    std::string m_name{"default"};
    int m_read_only{42};

    /// Reflect this class.
    template<reflection::ClassReflector R>
    static void reflect(R& r)
    {
        r //
            .def_rw("value", &MemberReflected::m_value, "An integer.", default_value(0), value_range(-10, 10))
            .def_rw("factor", &MemberReflected::m_factor, "A float.", default_value(1.f))
            .def_rw("name", &MemberReflected::m_name, default_value(std::string("default")))
            .def_ro("read_only", &MemberReflected::m_read_only, "Read-only member.");
    }
};

FALCOR_STATIC_ONCE(register_type<MemberReflected>());

TEST_CASE("def_rw/def_ro member pointer properties")
{
    auto obj = ref<MemberReflected>(new MemberReflected());
    const auto* desc = TypeRegistry::get().find(typeid(MemberReflected));
    REQUIRE(desc != nullptr);

    SUBCASE("read and write through ClassDescriptor")
    {
        obj->m_value = 7;
        obj->m_factor = 2.5f;
        obj->m_name = "hello";

        CHECK(desc->find_property("value")->get<int>(obj.get()) == 7);
        CHECK(desc->find_property("factor")->get<float>(obj.get()) == doctest::Approx(2.5f));
        CHECK(desc->find_property("name")->get<std::string>(obj.get()) == "hello");

        desc->set_any(obj.get(), "value", Any(99));
        desc->set_any(obj.get(), "factor", Any(0.5f));
        desc->set_any(obj.get(), "name", Any(std::string("world")));

        CHECK(obj->m_value == 99);
        CHECK(obj->m_factor == doctest::Approx(0.5f));
        CHECK(obj->m_name == "world");
    }

    SUBCASE("read-only property rejects writes")
    {
        CHECK(desc->find_property("read_only")->get<int>(obj.get()) == 42);
        CHECK_THROWS(desc->set_any(obj.get(), "read_only", Any(123)));
    }

    SUBCASE("default values and is_default")
    {
        const auto* value_prop = desc->find_property("value");
        REQUIRE(value_prop != nullptr);
        CHECK(value_prop->has_default_value());
        CHECK(value_prop->is_default(obj.get())); // m_value == 0 == default
        obj->m_value = 5;
        CHECK_FALSE(value_prop->is_default(obj.get()));
    }

    SUBCASE("metadata preserved")
    {
        const auto* value_prop = desc->find_property("value");
        REQUIRE(value_prop != nullptr);
        CHECK(value_prop->doc() == "An integer.");
        CHECK(value_prop->value_range() != nullptr);
        CHECK(value_prop->value_range()->min == -10);
        CHECK(value_prop->value_range()->max == 10);
    }

    SUBCASE("properties round-trip")
    {
        obj->m_value = 123;
        obj->m_factor = 45.6f;
        obj->m_name = "test";

        Properties props = obj->properties();

        auto obj2 = ref<MemberReflected>(new MemberReflected());
        obj2->set_properties(props);

        CHECK(obj2->m_value == 123);
        CHECK(obj2->m_factor == doctest::Approx(45.6f));
        CHECK(obj2->m_name == "test");
    }

    SUBCASE("reset restores default values")
    {
        obj->m_value = 99;
        obj->m_factor = 99.f;
        obj->m_name = "changed";

        desc->reset(obj.get());

        CHECK(obj->m_value == 0);
        CHECK(obj->m_factor == doctest::Approx(1.f));
        CHECK(obj->m_name == "default");
    }
}

// ---------------------------------------------------------------------------
// OnChange test class
// ---------------------------------------------------------------------------

/// A ReflectedObject that uses OnChange with both def_rw and def_prop_rw.
class OnChangeReflected : public ReflectedObject {
    FALCOR_REFLECTED_OBJECT(OnChangeReflected, ReflectedObject)
public:
    int m_field{0};
    int m_on_change_count{0};

    float getter_value() const { return m_getter_value; }
    void set_getter_value(float v) { m_getter_value = v; }

    int on_change_count() const { return m_on_change_count; }

    /// Reflect this class.
    template<reflection::ClassReflector R>
    static void reflect(R& r)
    {
        r //
            .def_rw(
                "field",
                &OnChangeReflected::m_field,
                default_value(0),
                on_change<OnChangeReflected>(
                    [](OnChangeReflected& self)
                    {
                        self.m_on_change_count++;
                    }
                )
            )
            .def_prop_rw(
                "getter_value",
                &OnChangeReflected::getter_value,
                &OnChangeReflected::set_getter_value,
                default_value(0.f),
                on_change<OnChangeReflected>(
                    [](OnChangeReflected& self)
                    {
                        self.m_on_change_count++;
                    }
                )
            );
    }

private:
    float m_getter_value{0.f};
};

FALCOR_STATIC_ONCE(register_type<OnChangeReflected>());

TEST_CASE("OnChange callback fires on property set")
{
    auto obj = ref<OnChangeReflected>(new OnChangeReflected());
    const auto* desc = TypeRegistry::get().find(typeid(OnChangeReflected));
    REQUIRE(desc != nullptr);

    SUBCASE("OnChange fires on def_rw set_any")
    {
        CHECK(obj->on_change_count() == 0);
        desc->set_any(obj.get(), "field", Any(42));
        CHECK(obj->m_field == 42);
        CHECK(obj->on_change_count() == 1);
        desc->set_any(obj.get(), "field", Any(99));
        CHECK(obj->on_change_count() == 2);
    }

    SUBCASE("OnChange fires on def_prop_rw set_any")
    {
        CHECK(obj->on_change_count() == 0);
        desc->set_any(obj.get(), "getter_value", Any(3.14f));
        CHECK(obj->getter_value() == doctest::Approx(3.14f));
        CHECK(obj->on_change_count() == 1);
    }

    SUBCASE("OnChange fires on read_from_properties")
    {
        Properties props;
        props.set<int>("field", 10);
        props.set<float>("getter_value", 2.5f);

        obj->set_properties(props);

        CHECK(obj->m_field == 10);
        CHECK(obj->getter_value() == doctest::Approx(2.5f));
        CHECK(obj->on_change_count() == 2); // One for each property.
    }

    SUBCASE("OnChange fires on reset")
    {
        obj->m_field = 99;
        desc->reset(obj.get());
        // reset calls the setter which fires OnChange.
        CHECK(obj->m_field == 0);
        CHECK(obj->on_change_count() >= 1);
    }

    SUBCASE("OnChange fires via PropertiesDeserializer")
    {
        Properties props;
        props.set<int>("field", 55);

        deserialize_properties(*obj, props);

        CHECK(obj->m_field == 55);
        CHECK(obj->on_change_count() == 1);
    }
}

// ---------------------------------------------------------------------------
// OnChange with member function pointer test class
// ---------------------------------------------------------------------------

/// A ReflectedObject that uses on_change with a member function pointer.
class MemberFuncOnChange : public ReflectedObject {
    FALCOR_REFLECTED_OBJECT(MemberFuncOnChange, ReflectedObject)
public:
    int m_value{0};
    int m_dirty_count{0};

    void mark_dirty() { m_dirty_count++; }

    /// Reflect this class.
    template<reflection::ClassReflector R>
    static void reflect(R& r)
    {
        r //
            .def_rw(
                "value",
                &MemberFuncOnChange::m_value,
                default_value(0),
                on_change(&MemberFuncOnChange::mark_dirty)
            );
    }
};

FALCOR_STATIC_ONCE(register_type<MemberFuncOnChange>());

TEST_CASE("OnChange with member function pointer")
{
    auto obj = ref<MemberFuncOnChange>(new MemberFuncOnChange());
    const auto* desc = TypeRegistry::get().find(typeid(MemberFuncOnChange));
    REQUIRE(desc != nullptr);

    CHECK(obj->m_dirty_count == 0);
    desc->set_any(obj.get(), "value", Any(42));
    CHECK(obj->m_value == 42);
    CHECK(obj->m_dirty_count == 1);
    desc->set_any(obj.get(), "value", Any(99));
    CHECK(obj->m_dirty_count == 2);
}

// ---------------------------------------------------------------------------
// ReflectedObject per-property access
// ---------------------------------------------------------------------------

TEST_CASE("ReflectedObject::find_property")
{
    auto obj = ref<SimpleReflected>(new SimpleReflected());

    SUBCASE("finds static properties")
    {
        const auto* prop = obj->find_property("foo");
        REQUIRE(prop != nullptr);
        CHECK(prop->name() == "foo");
    }

    SUBCASE("finds inherited properties")
    {
        auto derived = ref<DerivedSceneObj>(new DerivedSceneObj());
        CHECK(derived->find_property("extra") != nullptr);
        CHECK(derived->find_property("value") != nullptr);
        CHECK(derived->find_property("label") != nullptr);
    }

    SUBCASE("returns nullptr for nonexistent property")
    {
        CHECK(obj->find_property("nonexistent") == nullptr);
    }
}

TEST_CASE("ReflectedObject::has_property")
{
    auto obj = ref<SimpleReflected>(new SimpleReflected());
    CHECK(obj->has_property("foo"));
    CHECK(obj->has_property("bar"));
    CHECK(obj->has_property("label"));
    CHECK(obj->has_property("read_only_prop"));
    CHECK_FALSE(obj->has_property("nonexistent"));
}

TEST_CASE("ReflectedObject::get_property typed")
{
    auto obj = ref<SimpleReflected>(new SimpleReflected());
    obj->set_foo(42);
    obj->set_bar(3.14f);
    obj->set_label("hello");

    CHECK(obj->get_property<int>("foo") == 42);
    CHECK(obj->get_property<float>("bar") == doctest::Approx(3.14f));
    CHECK(obj->get_property<std::string>("label") == "hello");
    CHECK(obj->get_property<int>("read_only_prop") == 42);
}

TEST_CASE("ReflectedObject::set_property typed")
{
    auto obj = ref<SimpleReflected>(new SimpleReflected());

    obj->set_property<int>("foo", 99);
    obj->set_property<float>("bar", 7.5f);
    obj->set_property<std::string>("label", "world");

    CHECK(obj->foo() == 99);
    CHECK(obj->bar() == doctest::Approx(7.5f));
    CHECK(obj->label() == "world");
}

TEST_CASE("ReflectedObject::get_property type-erased (Any)")
{
    auto obj = ref<SimpleReflected>(new SimpleReflected());
    obj->set_foo(42);

    Any value = obj->get_property("foo");
    const int* p = any_cast<int>(&value);
    REQUIRE(p != nullptr);
    CHECK(*p == 42);
}

TEST_CASE("ReflectedObject::set_property type-erased (Any)")
{
    auto obj = ref<SimpleReflected>(new SimpleReflected());

    obj->set_property("foo", Any(77));
    CHECK(obj->foo() == 77);
}

TEST_CASE("ReflectedObject per-property access on derived objects")
{
    auto obj = ref<DerivedSceneObj>(new DerivedSceneObj());
    obj->set_value(10);
    obj->set_extra(2.5);

    CHECK(obj->get_property<int>("value") == 10);
    CHECK(obj->get_property<double>("extra") == doctest::Approx(2.5));

    obj->set_property<int>("value", 20);
    obj->set_property<double>("extra", 9.9);

    CHECK(obj->value() == 20);
    CHECK(obj->extra() == doctest::Approx(9.9));
}

TEST_SUITE_END();
