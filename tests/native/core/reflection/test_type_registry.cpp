// SPDX-License-Identifier: Apache-2.0

#include "testing.h"

#include "falcor2/core/reflection.h"
#include "falcor2/core/properties.h"
#include "falcor2/core/reflected_object.h"

using namespace falcor;
using namespace falcor::reflection;

// ---------------------------------------------------------------------------
// Test helper class (registered locally for isolated tests)
// ---------------------------------------------------------------------------

/// A simple class for testing the registry reflector directly.
class RegistryTestObj : public ReflectedObject {
    FALCOR_REFLECTED_OBJECT(RegistryTestObj, ReflectedObject)
public:
    int value() const { return m_value; }
    void set_value(int v) { m_value = v; }

    float ratio() const { return m_ratio; }
    void set_ratio(float v) { m_ratio = v; }

    const std::string& name() const { return m_name; }
    void set_name(const std::string& v) { m_name = v; }

    int constant() const { return 99; }

    /// Reflect this class.
    template<reflection::ClassReflector R>
    static void reflect(R& r)
    {
        r //
            .def_prop_rw(
                "value",
                &RegistryTestObj::value,
                &RegistryTestObj::set_value,
                "An integer value.",
                default_value(0),
                value_range(-100, 100)
            )
            .def_prop_rw(
                "ratio",
                &RegistryTestObj::ratio,
                &RegistryTestObj::set_ratio,
                "A float ratio.",
                default_value(0.5f),
                value_range(0.0, 1.0)
            )
            .def_prop_rw(
                "name",
                &RegistryTestObj::name,
                &RegistryTestObj::set_name,
                "A string name.",
                default_value(std::string("default"))
            )
            .def_prop_ro("constant", &RegistryTestObj::constant, "A read-only constant.", default_value(99));
    }

private:
    int m_value{0};
    float m_ratio{0.5f};
    std::string m_name{"default"};
};

// Register RegistryTestObj in the type registry via static init.
FALCOR_STATIC_ONCE(register_type<RegistryTestObj>());

/// A custom struct that does NOT satisfy PropertyValueType.
struct CustomData {
    int x{0};
    int y{0};

    bool operator==(const CustomData&) const = default;
};

/// A class with a non-serializable property (CustomData).
class NonSerializableObj : public ReflectedObject {
    FALCOR_REFLECTED_OBJECT(NonSerializableObj, ReflectedObject)
public:
    CustomData data() const { return m_data; }
    void set_data(const CustomData& v) { m_data = v; }

    int tag() const { return m_tag; }
    void set_tag(int v) { m_tag = v; }

    /// Reflect this class.
    template<reflection::ClassReflector R>
    static void reflect(R& r)
    {
        r //
            .def_prop_rw("data", &NonSerializableObj::data, &NonSerializableObj::set_data, default_value(CustomData{}))
            .def_prop_rw("tag", &NonSerializableObj::tag, &NonSerializableObj::set_tag, default_value(0));
    }

private:
    CustomData m_data{};
    int m_tag{0};
};

FALCOR_STATIC_ONCE(register_type<NonSerializableObj>());

/// A dummy type used to test duplicate name registration.
struct DuplicateNameDummy { };

/// A class with both static reflected properties and dynamic properties.
class DynPropsObj : public ReflectedObject {
    FALCOR_REFLECTED_OBJECT(DynPropsObj, ReflectedObject)
public:
    int static_val() const { return m_static_val; }
    void set_static_val(int v) { m_static_val = v; }

    /// Reflect this class.
    template<reflection::ClassReflector R>
    static void reflect(R& r)
    {
        r //
            .def_prop_rw("static_val", &DynPropsObj::static_val, &DynPropsObj::set_static_val, default_value(0));
    }

    DynamicPropertySet* dynamic_properties() override { return &m_dyn; }
    const DynamicPropertySet* dynamic_properties() const override { return &m_dyn; }

private:
    int m_static_val{0};
    DynamicPropertySet m_dyn;
};

FALCOR_STATIC_ONCE(register_type<DynPropsObj>());

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST_SUITE_BEGIN("reflection");

// --- Concept checks ---

TEST_CASE("ClassReflector concept")
{
    CHECK(ClassReflector<reflection::detail::RegistryClassReflector<RegistryTestObj>>);
}

// --- TypeRegistry basics ---

TEST_CASE("TypeRegistry find by type_info")
{
    auto& reg = TypeRegistry::get();
    const ClassDescriptor* desc = reg.find(typeid(RegistryTestObj));
    REQUIRE(desc != nullptr);
    CHECK(desc->name() == "RegistryTestObj");
}

TEST_CASE("TypeRegistry find by name")
{
    auto& reg = TypeRegistry::get();
    const ClassDescriptor* desc = reg.find("RegistryTestObj");
    REQUIRE(desc != nullptr);
    CHECK(desc->type() == typeid(RegistryTestObj));
}

TEST_CASE("TypeRegistry find returns null for unregistered type")
{
    auto& reg = TypeRegistry::get();
    CHECK(reg.find(typeid(int)) == nullptr);
    CHECK(reg.find("NonExistentClass") == nullptr);
}

// --- ClassDescriptor for RegistryTestObj ---

TEST_CASE("ClassDescriptor has correct properties")
{
    auto& reg = TypeRegistry::get();
    const ClassDescriptor* desc = reg.find(typeid(RegistryTestObj));
    REQUIRE(desc != nullptr);

    auto props = desc->properties();
    CHECK(props.size() == 4);

    SUBCASE("property names")
    {
        CHECK(props[0]->name() == "value");
        CHECK(props[1]->name() == "ratio");
        CHECK(props[2]->name() == "name");
        CHECK(props[3]->name() == "constant");
    }

    SUBCASE("property types")
    {
        CHECK(props[0]->type() == typeid(int));
        CHECK(props[1]->type() == typeid(float));
        CHECK(props[2]->type() == typeid(std::string));
        CHECK(props[3]->type() == typeid(int));
    }

    SUBCASE("read-only flags")
    {
        CHECK_FALSE(props[0]->is_read_only());
        CHECK_FALSE(props[1]->is_read_only());
        CHECK_FALSE(props[2]->is_read_only());
        CHECK(props[3]->is_read_only());
    }

    SUBCASE("serializable to properties flags")
    {
        CHECK(props[0]->is_serializable_to_properties());
        CHECK(props[1]->is_serializable_to_properties());
        CHECK(props[2]->is_serializable_to_properties());
        CHECK(props[3]->is_serializable_to_properties());
    }
}

// --- Metadata ---

TEST_CASE("PropertyDescriptor metadata - doc string")
{
    auto& reg = TypeRegistry::get();
    const ClassDescriptor* desc = reg.find(typeid(RegistryTestObj));
    REQUIRE(desc != nullptr);

    CHECK(desc->properties()[0]->doc() == "An integer value.");
    CHECK(desc->properties()[1]->doc() == "A float ratio.");
    CHECK(desc->properties()[2]->doc() == "A string name.");
    CHECK(desc->properties()[3]->doc() == "A read-only constant.");
}

TEST_CASE("PropertyDescriptor metadata - ValueRange")
{
    auto& reg = TypeRegistry::get();
    const ClassDescriptor* desc = reg.find(typeid(RegistryTestObj));
    REQUIRE(desc != nullptr);

    const ValueRange* vr0 = desc->properties()[0]->value_range();
    REQUIRE(vr0 != nullptr);
    CHECK(vr0->min == doctest::Approx(-100.0));
    CHECK(vr0->max == doctest::Approx(100.0));

    const ValueRange* vr1 = desc->properties()[1]->value_range();
    REQUIRE(vr1 != nullptr);
    CHECK(vr1->min == doctest::Approx(0.0));
    CHECK(vr1->max == doctest::Approx(1.0));

    // "name" has no ValueRange.
    CHECK(desc->properties()[2]->value_range() == nullptr);
}

TEST_CASE("PropertyDescriptor metadata - DefaultValue")
{
    auto& reg = TypeRegistry::get();
    const ClassDescriptor* desc = reg.find(typeid(RegistryTestObj));
    REQUIRE(desc != nullptr);

    // All properties with DefaultValue(...) report has_default_value() == true.
    CHECK(desc->properties()[0]->has_default_value());
    CHECK(desc->properties()[1]->has_default_value());
    CHECK(desc->properties()[2]->has_default_value());
    CHECK(desc->properties()[3]->has_default_value());

    // Verify defaults via is_default() on a default-constructed object.
    RegistryTestObj obj;
    CHECK(desc->properties()[0]->is_default(&obj)); // value=0
    CHECK(desc->properties()[1]->is_default(&obj)); // ratio=0.5f
    CHECK(desc->properties()[2]->is_default(&obj)); // name="default"

    // Changing a value makes is_default() return false.
    obj.set_value(42);
    CHECK_FALSE(desc->properties()[0]->is_default(&obj));
}

// --- Type-erased get/set ---

TEST_CASE("Type-erased get/set via PropertyDescriptor")
{
    auto& reg = TypeRegistry::get();
    const ClassDescriptor* desc = reg.find(typeid(RegistryTestObj));
    REQUIRE(desc != nullptr);

    RegistryTestObj obj;
    obj.set_value(42);
    obj.set_ratio(0.75f);
    obj.set_name("hello");

    SUBCASE("get returns correct values")
    {
        Any v0 = desc->properties()[0]->get_any(&obj);
        CHECK(*any_cast<int>(&v0) == 42);

        Any v1 = desc->properties()[1]->get_any(&obj);
        CHECK(*any_cast<float>(&v1) == doctest::Approx(0.75f));

        Any v2 = desc->properties()[2]->get_any(&obj);
        CHECK(*any_cast<std::string>(&v2) == "hello");
    }

    SUBCASE("set applies values")
    {
        desc->properties()[0]->set_any(&obj, Any(99));
        CHECK(obj.value() == 99);

        desc->properties()[1]->set_any(&obj, Any(0.1f));
        CHECK(obj.ratio() == doctest::Approx(0.1f));

        desc->properties()[2]->set_any(&obj, Any(std::string("world")));
        CHECK(obj.name() == "world");
    }

    SUBCASE("set on read-only property throws")
    {
        CHECK_THROWS(desc->properties()[3]->set_any(&obj, Any(0)));
    }

    SUBCASE("set with wrong type throws")
    {
        CHECK_THROWS(desc->properties()[0]->set_any(&obj, Any(std::string("wrong"))));
    }
}

// --- ClassDescriptor get/set by name ---

TEST_CASE("ClassDescriptor get/set by name")
{
    auto& reg = TypeRegistry::get();
    const ClassDescriptor* desc = reg.find(typeid(RegistryTestObj));
    REQUIRE(desc != nullptr);

    RegistryTestObj obj;
    obj.set_value(10);

    Any val = desc->get_any(&obj, "value");
    CHECK(*any_cast<int>(&val) == 10);

    desc->set_any(&obj, "value", Any(77));
    CHECK(obj.value() == 77);
}

// --- is_default ---

TEST_CASE("PropertyDescriptor is_default")
{
    auto& reg = TypeRegistry::get();
    const ClassDescriptor* desc = reg.find(typeid(RegistryTestObj));
    REQUIRE(desc != nullptr);

    RegistryTestObj obj;
    // obj has default values from member initializers.
    CHECK(desc->properties()[0]->is_default(&obj)); // value=0
    CHECK(desc->properties()[1]->is_default(&obj)); // ratio=0.5f
    CHECK(desc->properties()[2]->is_default(&obj)); // name="default"

    obj.set_value(42);
    CHECK_FALSE(desc->properties()[0]->is_default(&obj));
}

// --- reset ---

TEST_CASE("PropertyDescriptor reset restores default")
{
    auto& reg = TypeRegistry::get();
    const ClassDescriptor* desc = reg.find(typeid(RegistryTestObj));
    REQUIRE(desc != nullptr);

    RegistryTestObj obj;
    obj.set_value(42);
    obj.set_ratio(0.9f);
    obj.set_name("changed");

    desc->properties()[0]->reset(&obj);
    desc->properties()[1]->reset(&obj);
    desc->properties()[2]->reset(&obj);

    CHECK(obj.value() == 0);
    CHECK(obj.ratio() == doctest::Approx(0.5f));
    CHECK(obj.name() == "default");
}

TEST_CASE("ClassDescriptor reset restores all defaults")
{
    auto& reg = TypeRegistry::get();
    const ClassDescriptor* desc = reg.find(typeid(RegistryTestObj));
    REQUIRE(desc != nullptr);

    RegistryTestObj obj;
    obj.set_value(42);
    obj.set_ratio(0.9f);
    obj.set_name("changed");

    desc->reset(&obj);

    CHECK(obj.value() == 0);
    CHECK(obj.ratio() == doctest::Approx(0.5f));
    CHECK(obj.name() == "default");
}

// --- write_to_properties / read_from_properties ---

TEST_CASE("ClassDescriptor write_to_properties and read_from_properties")
{
    auto& reg = TypeRegistry::get();
    const ClassDescriptor* desc = reg.find(typeid(RegistryTestObj));
    REQUIRE(desc != nullptr);

    RegistryTestObj obj;
    obj.set_value(42);
    obj.set_ratio(0.75f);
    obj.set_name("test");

    Properties props;
    desc->write_to_properties(&obj, props);

    CHECK(props.get<int>("value") == 42);
    CHECK(props.get<float>("ratio") == doctest::Approx(0.75f));
    CHECK(props.get<std::string>("name") == "test");
    // read_only property is also serialized
    CHECK(props.get<int>("constant") == 99);

    // read_from_properties into a fresh object
    RegistryTestObj obj2;
    desc->read_from_properties(&obj2, props);

    CHECK(obj2.value() == 42);
    CHECK(obj2.ratio() == doctest::Approx(0.75f));
    CHECK(obj2.name() == "test");
    // constant is read-only, so it stays at 99 (its only value).
    CHECK(obj2.constant() == 99);
}

// --- StoredPropertyDescriptor ---

TEST_CASE("StoredPropertyDescriptor stores values directly")
{
    StoredPropertyDescriptor<int> desc("test_int", 42, false, "A test int.", default_value(42), value_range(0, 100));

    CHECK(desc.name() == "test_int");
    CHECK(desc.type() == typeid(int));
    CHECK_FALSE(desc.is_read_only());
    CHECK(desc.is_serializable_to_properties());

    Any val = desc.get_any(nullptr);
    CHECK(*any_cast<int>(&val) == 42);

    desc.set_any(nullptr, Any(99));
    Any val2 = desc.get_any(nullptr);
    CHECK(*any_cast<int>(&val2) == 99);

    CHECK_FALSE(desc.is_default(nullptr));
    desc.reset(nullptr);
    Any val3 = desc.get_any(nullptr);
    CHECK(*any_cast<int>(&val3) == 42);
    CHECK(desc.is_default(nullptr));

    CHECK(desc.doc() == "A test int.");
    const ValueRange* vr = desc.value_range();
    REQUIRE(vr != nullptr);
    CHECK(vr->min == doctest::Approx(0.0));
    CHECK(vr->max == doctest::Approx(100.0));
}

TEST_CASE("StoredPropertyDescriptor read-only prevents set")
{
    StoredPropertyDescriptor<int> desc("ro_int", 42, true);
    CHECK(desc.is_read_only());
    CHECK_THROWS(desc.set_any(nullptr, Any(99)));
}

TEST_CASE("DynamicPropertySet add_property_checked rejects static collisions")
{
    auto& reg = TypeRegistry::get();
    const ClassDescriptor* desc = reg.find(typeid(RegistryTestObj));
    REQUIRE(desc != nullptr);

    DynamicPropertySet dps;
    // "value" is a static property on RegistryTestObj.
    CHECK_THROWS(dps.add_property_checked<int>(*desc, "value", 0));

    // A non-colliding name should work fine.
    CHECK_NOTHROW(dps.add_property_checked<int>(*desc, "dynamic_only", 0));
}

// --- Typed get<T>/set<T> on PropertyDescriptor ---

TEST_CASE("PropertyDescriptor typed get/set on TypedPropertyDescriptor")
{
    auto& reg = TypeRegistry::get();
    const ClassDescriptor* cls = reg.find(typeid(RegistryTestObj));
    REQUIRE(cls != nullptr);

    RegistryTestObj obj;
    obj.set_value(42);
    obj.set_ratio(0.75f);

    const PropertyDescriptor* prop_value = cls->find_property("value");
    const PropertyDescriptor* prop_ratio = cls->find_property("ratio");
    REQUIRE(prop_value != nullptr);
    REQUIRE(prop_ratio != nullptr);

    // Typed get
    CHECK(prop_value->get<int>(&obj) == 42);
    CHECK(prop_ratio->get<float>(&obj) == doctest::Approx(0.75f));

    // Typed set
    prop_value->set<int>(&obj, 99);
    CHECK(obj.value() == 99);

    prop_ratio->set<float>(&obj, 0.1f);
    CHECK(obj.ratio() == doctest::Approx(0.1f));

    // Type mismatch throws
    CHECK_THROWS(prop_value->get<float>(&obj));
    CHECK_THROWS(prop_value->set<std::string>(&obj, std::string("wrong")));
}

TEST_CASE("PropertyDescriptor typed get/set on StoredPropertyDescriptor")
{
    StoredPropertyDescriptor<float> desc("test_float", 3.14f, false, default_value(3.14f));

    // Typed get
    CHECK(desc.get<float>(nullptr) == doctest::Approx(3.14f));

    // Typed set
    desc.set<float>(nullptr, 2.71f);
    CHECK(desc.get<float>(nullptr) == doctest::Approx(2.71f));

    // Type mismatch throws
    CHECK_THROWS(desc.get<int>(nullptr));
    CHECK_THROWS(desc.set<int>(nullptr, 42));
}

// --- Uniform interface: TypedPropertyDescriptor and StoredPropertyDescriptor ---

TEST_CASE("Uniform PropertyDescriptor interface for static and dynamic properties")
{
    // Static property from registry.
    auto& reg = TypeRegistry::get();
    const ClassDescriptor* desc = reg.find(typeid(RegistryTestObj));
    REQUIRE(desc != nullptr);
    const PropertyDescriptor* static_prop = desc->find_property("value");
    REQUIRE(static_prop != nullptr);

    // Dynamic property.
    DynamicPropertySet dps;
    dps.add_property<int>("dyn_value", 0, false, default_value(0));
    const PropertyDescriptor* dyn_prop = dps.find_property("dyn_value");
    REQUIRE(dyn_prop != nullptr);

    // Both have the same interface.
    CHECK(static_prop->type() == typeid(int));
    CHECK(dyn_prop->type() == typeid(int));
    CHECK_FALSE(static_prop->is_read_only());
    CHECK_FALSE(dyn_prop->is_read_only());
    CHECK(static_prop->is_serializable_to_properties());
    CHECK(dyn_prop->is_serializable_to_properties());
}

// --- Duplicate registration ---

TEST_CASE("Duplicate register_class throws")
{
    // Call register_class directly (not via register_type) to avoid throwing from
    // RegistryClassReflector's destructor, which would call std::terminate.
    auto* desc = new ClassDescriptor("RegistryTestObj", typeid(RegistryTestObj), typeid(void));
    CHECK_THROWS(TypeRegistry::get().register_class(desc));
    // desc ownership was not taken by the registry since it threw, so clean up.
    delete desc;
}

// --- ReflectedObject::dynamic_properties default ---

TEST_CASE("ReflectedObject::dynamic_properties returns nullptr by default")
{
    RegistryTestObj obj;
    CHECK(obj.dynamic_properties() == nullptr);
    CHECK(static_cast<const RegistryTestObj&>(obj).dynamic_properties() == nullptr);
}

// --- Non-serializable property type ---

TEST_CASE("Non-serializable property: is_serializable_to_properties returns false")
{
    auto& reg = TypeRegistry::get();
    const ClassDescriptor* desc = reg.find(typeid(NonSerializableObj));
    REQUIRE(desc != nullptr);

    const PropertyDescriptor* data_prop = desc->find_property("data");
    const PropertyDescriptor* tag_prop = desc->find_property("tag");
    REQUIRE(data_prop != nullptr);
    REQUIRE(tag_prop != nullptr);

    CHECK_FALSE(data_prop->is_serializable_to_properties());
    CHECK(tag_prop->is_serializable_to_properties());
}

TEST_CASE("Non-serializable property: write_to_properties throws")
{
    auto& reg = TypeRegistry::get();
    const ClassDescriptor* desc = reg.find(typeid(NonSerializableObj));
    REQUIRE(desc != nullptr);

    NonSerializableObj obj;
    Properties props;

    const PropertyDescriptor* data_prop = desc->find_property("data");
    REQUIRE(data_prop != nullptr);

    CHECK_THROWS(data_prop->write_to_properties(&obj, props));
}

TEST_CASE("Non-serializable property: read_from_properties returns false")
{
    auto& reg = TypeRegistry::get();
    const ClassDescriptor* desc = reg.find(typeid(NonSerializableObj));
    REQUIRE(desc != nullptr);

    NonSerializableObj obj;
    Properties props;
    // Even if there were a key named "data", read_from_properties should return false
    // because the type is not serializable.
    CHECK_FALSE(desc->find_property("data")->read_from_properties(&obj, props));
}

TEST_CASE("Non-serializable property: get_any and set_any still work")
{
    auto& reg = TypeRegistry::get();
    const ClassDescriptor* desc = reg.find(typeid(NonSerializableObj));
    REQUIRE(desc != nullptr);

    NonSerializableObj obj;
    obj.set_data(CustomData{1, 2});

    const PropertyDescriptor* data_prop = desc->find_property("data");
    REQUIRE(data_prop != nullptr);

    Any val = data_prop->get_any(&obj);
    const CustomData* cd = any_cast<CustomData>(&val);
    REQUIRE(cd != nullptr);
    CHECK(cd->x == 1);
    CHECK(cd->y == 2);

    data_prop->set_any(&obj, Any(CustomData{3, 4}));
    CHECK(obj.data().x == 3);
    CHECK(obj.data().y == 4);
}

TEST_CASE("Non-serializable property: is_default works")
{
    auto& reg = TypeRegistry::get();
    const ClassDescriptor* desc = reg.find(typeid(NonSerializableObj));
    REQUIRE(desc != nullptr);

    NonSerializableObj obj;
    const PropertyDescriptor* data_prop = desc->find_property("data");
    REQUIRE(data_prop != nullptr);

    CHECK(data_prop->is_default(&obj));

    obj.set_data(CustomData{1, 2});
    CHECK_FALSE(data_prop->is_default(&obj));
}

TEST_CASE("Non-serializable property: ClassDescriptor write_to_properties skips non-serializable")
{
    auto& reg = TypeRegistry::get();
    const ClassDescriptor* desc = reg.find(typeid(NonSerializableObj));
    REQUIRE(desc != nullptr);

    NonSerializableObj obj;
    obj.set_tag(42);
    obj.set_data(CustomData{1, 2});

    // ClassDescriptor::write_to_properties only writes serializable properties.
    Properties props;
    desc->write_to_properties(&obj, props);

    CHECK(props.has_property("tag"));
    CHECK(props.get<int>("tag") == 42);
    CHECK_FALSE(props.has_property("data"));
}

TEST_CASE("Non-serializable property: ClassDescriptor read_from_properties skips non-serializable")
{
    auto& reg = TypeRegistry::get();
    const ClassDescriptor* desc = reg.find(typeid(NonSerializableObj));
    REQUIRE(desc != nullptr);

    NonSerializableObj obj;
    obj.set_data(CustomData{1, 2});

    Properties props;
    props.set<int>("tag", 99);
    desc->read_from_properties(&obj, props);

    CHECK(obj.tag() == 99);
    // data is untouched because it is non-serializable.
    CHECK(obj.data().x == 1);
    CHECK(obj.data().y == 2);
}

TEST_CASE(
    "Non-serializable StoredPropertyDescriptor: is_serializable_to_properties false and write_to_properties throws"
)
{
    StoredPropertyDescriptor<CustomData> desc("stored_data", CustomData{5, 6}, false, default_value(CustomData{}));

    CHECK_FALSE(desc.is_serializable_to_properties());

    Properties props;
    CHECK_THROWS(desc.write_to_properties(nullptr, props));
    CHECK_FALSE(desc.read_from_properties(nullptr, props));
}

// --- Duplicate name registration ---

TEST_CASE("Duplicate register_class by name throws")
{
    // Use a novel type but an already-registered name to exercise the name-collision check.
    auto* desc = new ClassDescriptor("RegistryTestObj", typeid(DuplicateNameDummy), typeid(void));
    CHECK_THROWS(TypeRegistry::get().register_class(desc));
    // desc ownership was not taken by the registry since it threw, so clean up.
    delete desc;
}

// --- ReflectedObject with dynamic_properties ---

TEST_CASE("ReflectedObject with dynamic_properties merges static and dynamic")
{
    auto obj = ref<DynPropsObj>(new DynPropsObj());
    obj->set_static_val(42);
    obj->dynamic_properties()->add_property<float>("dyn_float", 3.14f);

    SUBCASE("properties() includes both static and dynamic")
    {
        Properties props = obj->properties();
        CHECK(props.get<int>("static_val") == 42);
        CHECK(props.get<float>("dyn_float") == doctest::Approx(3.14f));
    }

    SUBCASE("set_properties() applies to both static and dynamic")
    {
        Properties props;
        props.set<int>("static_val", 99);
        props.set<float>("dyn_float", 2.71f);
        obj->set_properties(props);

        CHECK(obj->static_val() == 99);
        CHECK(obj->dynamic_properties()->get<float>("dyn_float") == doctest::Approx(2.71f));
    }

    SUBCASE("round-trip preserves both static and dynamic")
    {
        Properties props = obj->properties();

        auto obj2 = ref<DynPropsObj>(new DynPropsObj());
        obj2->dynamic_properties()->add_property<float>("dyn_float", 0.0f);
        obj2->set_properties(props);

        CHECK(obj2->static_val() == 42);
        CHECK(obj2->dynamic_properties()->get<float>("dyn_float") == doctest::Approx(3.14f));
    }
}

TEST_SUITE_END();
