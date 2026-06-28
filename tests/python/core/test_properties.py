# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import falcor2 as f2
from falcor2 import Properties, PropertyType
import slangpy as spy
from pathlib import Path
from typing import Any


# =============================================================================
# Test data
# =============================================================================

TEST_VALUES = [
    ("bool", PropertyType.bool, True, False, "true"),
    ("int", PropertyType.int, 10, 20, "10"),
    ("float", PropertyType.float, 10.0, 20.0, "10"),
    ("int2", PropertyType.int2, spy.int2(10, 20), spy.int2(20, 30), "{10, 20}"),
    ("int3", PropertyType.int3, spy.int3(10, 20, 30), spy.int3(20, 30, 40), "{10, 20, 30}"),
    (
        "int4",
        PropertyType.int4,
        spy.int4(10, 20, 30, 40),
        spy.int4(20, 30, 40, 50),
        "{10, 20, 30, 40}",
    ),
    ("uint2", PropertyType.uint2, spy.uint2(10, 20), spy.uint2(20, 30), "{10, 20}"),
    (
        "uint3",
        PropertyType.uint3,
        spy.uint3(10, 20, 30),
        spy.uint3(20, 30, 40),
        "{10, 20, 30}",
    ),
    (
        "uint4",
        PropertyType.uint4,
        spy.uint4(10, 20, 30, 40),
        spy.uint4(20, 30, 40, 50),
        "{10, 20, 30, 40}",
    ),
    (
        "float2",
        PropertyType.float2,
        spy.float2(10.5, 20.5),
        spy.float2(20.5, 30.5),
        "{10.5, 20.5}",
    ),
    (
        "float3",
        PropertyType.float3,
        spy.float3(10.5, 20.5, 30.5),
        spy.float3(20.5, 30.5, 40.5),
        "{10.5, 20.5, 30.5}",
    ),
    (
        "float4",
        PropertyType.float4,
        spy.float4(10.5, 20.5, 30.5, 40.5),
        spy.float4(20.5, 30.5, 40.5, 50.5),
        "{10.5, 20.5, 30.5, 40.5}",
    ),
    (
        "float3x3",
        PropertyType.float3x3,
        spy.float3x3([1, 2, 3, 4, 5, 6, 7, 8, 9]),
        spy.float3x3([9, 8, 7, 6, 5, 4, 3, 2, 1]),
        "{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}",
    ),
    (
        "float4x4",
        PropertyType.float4x4,
        spy.float4x4([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]),
        spy.float4x4([16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]),
        "{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}}",
    ),
    ("string", PropertyType.string, "foo", "bar", '"foo"'),
    ("enum", PropertyType.enum, PropertyType.int, PropertyType.float, "falcor::PropertyType(2)"),
]


def fill_properties(props: Properties):
    props["bool"] = True
    props["int"] = 42
    props["float"] = 42.0
    props["int2"] = spy.int2(1, 2)
    props["int3"] = spy.int3(1, 2, 3)
    props["int4"] = spy.int4(1, 2, 3, 4)
    props["uint2"] = spy.uint2(1, 2)
    props["uint3"] = spy.uint3(1, 2, 3)
    props["uint4"] = spy.uint4(1, 2, 3, 4)
    props["float2"] = spy.float2(1.0, 2.0)
    props["float3"] = spy.float3(1.0, 2.0, 3.0)
    props["float4"] = spy.float4(1.0, 2.0, 3.0, 4.0)
    props["float3x3"] = spy.float3x3([1, 0, 0, 0, 1, 0, 0, 0, 1])
    props["float4x4"] = spy.float4x4([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])
    props["string"] = "foo"


# =============================================================================
# Basic type storage (parametrized)
# =============================================================================


@pytest.mark.parametrize("name, type, test_value, default_value, string_value", TEST_VALUES)
def test_type_roundtrip(
    name: str, type: PropertyType, test_value: Any, default_value: Any, string_value: str
):
    p = Properties()

    # Checks while the property is not set
    assert name not in p
    assert not p.has_property(name)
    assert p.get(name) is None
    assert p.get(name, default_value) == default_value

    # Set the property
    p[name] = test_value

    # Checks after the property is set
    assert name in p
    assert p.has_property(name)
    assert p.get(name) == test_value
    assert p.get(name, default_value) == test_value
    assert p.type(name) == type
    assert p.as_string(name) == string_value
    assert p.find(name) == (name, test_value)

    # Remove the property
    del p[name]

    # Checks after the property is removed
    assert name not in p
    assert not p.has_property(name)
    assert p.get(name) is None
    assert p.get(name, default_value) == default_value
    assert p.find(name) is None


def test_type_update():
    """Test that updating a property with a different type works."""
    p = Properties()
    p["val"] = "hello"
    assert p.type("val") == PropertyType.string
    assert p["val"] == "hello"

    p["val"] = 42
    assert p.type("val") == PropertyType.int
    assert p["val"] == 42

    p["val"] = 3.14
    assert p.type("val") == PropertyType.float
    assert p["val"] == pytest.approx(3.14)


def test_pathlib_path():
    """Test that pathlib.Path values are stored as strings."""
    p = Properties()
    p["path"] = Path("/some/test/path")
    assert p.type("path") == PropertyType.string
    assert p["path"] == str(Path("/some/test/path"))

    p2 = Properties({"output": Path("results/output.exr")})
    assert p2.type("output") == PropertyType.string
    assert p2["output"] == str(Path("results/output.exr"))


def test_setitem_error_message():
    """Test that __setitem__ with unsupported types gives a descriptive error."""
    p = Properties()
    with pytest.raises(Exception, match="Could not assign"):
        p["bad"] = object()


# =============================================================================
# Container operations
# =============================================================================


def test_empty_clear():
    """Test empty() and clear() container operations."""
    props = Properties()
    assert props.empty()
    assert len(props) == 0

    fill_properties(props)
    assert not props.empty()
    assert len(props) > 0

    props.clear()
    assert props.empty()
    assert len(props) == 0


def test_size():
    """Test size() and __len__."""
    p = Properties()
    assert p.size() == 0
    assert len(p) == 0

    p["a"] = 1
    assert p.size() == 1
    assert len(p) == 1

    p["b"] = 2.0
    assert p.size() == 2
    assert len(p) == 2

    del p["a"]
    assert p.size() == 1
    assert len(p) == 1


def test_insertion_order():
    """Test that Properties maintains insertion order."""
    props = Properties()
    props["c"] = 3
    props["a"] = 1
    props["b"] = 2

    keys = props.keys()
    assert keys == ["c", "a", "b"]


def test_insertion_order_with_deletion():
    """Test that Properties maintains insertion order even after deletion."""
    props = Properties()
    props["first"] = 1.0
    props["second"] = "hello"
    props["third"] = spy.float3(1, 2, 3)
    props["fourth"] = True
    props["fifth"] = 42

    assert props.keys() == ["first", "second", "third", "fourth", "fifth"]

    del props["third"]
    assert props.keys() == ["first", "second", "fourth", "fifth"]

    assert props["first"] == 1.0
    assert props["second"] == "hello"
    assert props["fourth"] == True
    assert props["fifth"] == 42

    props["sixth"] = spy.float3(0.5, 0.5, 0.5)
    assert props.keys() == ["first", "second", "fourth", "fifth", "sixth"]


# =============================================================================
# Copy & construction
# =============================================================================


def test_create():
    p = Properties()
    assert p is not None
    assert p.empty()


def test_copy_constructor():
    """Test copy constructor produces an independent copy."""
    p1 = Properties()
    fill_properties(p1)

    p2 = Properties(p1)
    assert p1 == p2

    # Modify the copy and verify independence
    p2["extra"] = 999
    assert "extra" not in p1
    assert p1 != p2


def test_dict_construction():
    """Test constructing Properties from a Python dict."""
    p = Properties({"a": 1, "b": 2.0, "c": "hello", "d": True})
    assert len(p) == 4
    assert p["a"] == 1
    assert p["b"] == 2.0
    assert p["c"] == "hello"
    assert p["d"] == True


def test_dict_construction_with_object_and_any():
    """Test constructing Properties from a dict with Object, Any, and scalar values."""
    from falcor2 import AssetResolver, AABB

    resolver = AssetResolver()
    box = AABB(spy.float3(-1, -1, -1), spy.float3(1, 1, 1))
    p = Properties(
        {
            "resolver": resolver,
            "bounds": box,
            "scale": 2.5,
            "name": "scene",
            "enabled": True,
        }
    )
    assert len(p) == 5
    assert p.type("resolver") == PropertyType.object
    assert isinstance(p["resolver"], AssetResolver)
    assert p.type("bounds") == PropertyType.any
    retrieved_box = p["bounds"]
    assert isinstance(retrieved_box, AABB)
    assert retrieved_box.min == spy.float3(-1, -1, -1)
    assert retrieved_box.max == spy.float3(1, 1, 1)
    assert p["scale"] == 2.5
    assert p["name"] == "scene"
    assert p["enabled"] == True


# =============================================================================
# Equality & hashing
# =============================================================================


def test_equality():
    props1 = Properties()
    props2 = Properties()

    assert props1 == props1
    assert props2 == props2
    assert props1 == props2

    fill_properties(props1)
    assert props1 != props2

    fill_properties(props2)
    assert props1 == props2


def test_hash():
    props1 = Properties()
    props1["a"] = 1
    props1["b"] = 2.0
    props1["c"] = "hello"

    props2 = Properties()
    props2["a"] = 1
    props2["b"] = 2.0
    props2["c"] = "hello"

    # Equal properties have equal hash
    assert props1.hash() == props2.hash()

    # Insertion order does not affect hash
    props3 = Properties()
    props3["c"] = "hello"
    props3["a"] = 1
    props3["b"] = 2.0
    assert props1.hash() == props3.hash()

    # Different values produce different hash
    props4 = Properties()
    props4["a"] = 999
    props4["b"] = 2.0
    props4["c"] = "hello"
    assert props1.hash() != props4.hash()


def test_hash_empty():
    """Test that empty Properties have consistent hash."""
    assert Properties().hash() == Properties().hash()


# =============================================================================
# Merge
# =============================================================================


def test_merge():
    props1 = Properties()
    props1["foo"] = 42
    props1["bar"] = 42
    props1["baz"] = 42

    props2 = Properties()
    props2["foo"] = 1
    props2["bar"] = 1
    props2["new"] = 1

    merged = Properties(props1)
    assert merged == props1
    merged.merge(props1)
    assert merged == props1
    merged.merge(props2)
    assert merged["foo"] == 1
    assert merged["bar"] == 1
    assert merged["baz"] == 42
    assert merged["new"] == 1


def test_merge_empty():
    """Test merge edge cases with empty Properties."""
    populated = Properties()
    populated["a"] = 1
    populated["b"] = "hello"

    # Merge empty into populated (no change)
    merged = Properties(populated)
    merged.merge(Properties())
    assert merged == populated

    # Merge populated into empty
    empty = Properties()
    empty.merge(populated)
    assert empty == populated


# =============================================================================
# Mutation
# =============================================================================


def test_remove_property():
    props = Properties()
    props["foo"] = 42
    assert props.remove_property("foo") == True
    assert props.remove_property("foo") == False
    assert props.remove_property("nonexistent") == False


def test_delitem_missing_key():
    """Test that __delitem__ raises KeyError for missing keys."""
    props = Properties()
    with pytest.raises(KeyError):
        del props["nonexistent"]


def test_rename_property():
    props = Properties()
    props["old"] = 42
    props["other"] = 99

    # Basic rename
    assert props.rename_property("old", "new") == True
    assert "new" in props
    assert "old" not in props
    assert props["new"] == 42

    # Rename nonexistent
    assert props.rename_property("nonexistent", "abc") == False

    # Rename to existing name
    assert props.rename_property("new", "other") == False
    assert "new" in props


# =============================================================================
# Accessors
# =============================================================================


def test_getitem_missing_key():
    """Test that __getitem__ raises on missing key."""
    p = Properties()
    with pytest.raises(Exception):
        _ = p["nonexistent"]


def test_get_default_none():
    """Test that get() returns None by default when property is missing."""
    p = Properties()
    assert p.get("missing") is None
    assert p.get("missing", 4.0) == 4.0
    assert p.get("missing", 4) == 4
    assert type(p.get("missing", 4)) == int
    assert p.get("missing", "foo") == "foo"

    p["foo"] = "test"
    assert p.get("foo", 4.0) == "test"


def test_type_missing_key():
    """Test that type() raises on a missing key."""
    p = Properties()
    with pytest.raises(Exception):
        p.type("nonexistent")


def test_as_string_missing_key():
    """Test that as_string() raises on a missing key."""
    p = Properties()
    with pytest.raises(Exception):
        p.as_string("nonexistent")


def test_find_returns_tuple():
    """Test that find returns a (name, value) tuple when property exists."""
    p = Properties()
    p["count"] = 42
    p["label"] = "hello"
    p["vec"] = spy.float3(1, 2, 3)

    result = p.find("count")
    assert result is not None
    assert isinstance(result, tuple)
    assert result == ("count", 42)

    result = p.find("label")
    assert result == ("label", "hello")

    result = p.find("vec")
    assert result == ("vec", spy.float3(1, 2, 3))


def test_find_returns_none():
    """Test that find returns None when property does not exist."""
    p = Properties()
    assert p.find("missing") is None

    p["exists"] = 1
    assert p.find("missing") is None

    del p["exists"]
    assert p.find("exists") is None


# =============================================================================
# Typed getters (scalar)
# =============================================================================


def test_typed_getters():
    """Test typed getter methods return correctly typed values."""
    p = Properties()
    p["b"] = True
    p["i"] = 42
    p["f"] = 3.14
    p["s"] = "hello"
    p["i2"] = spy.int2(1, 2)
    p["i3"] = spy.int3(1, 2, 3)
    p["i4"] = spy.int4(1, 2, 3, 4)
    p["u2"] = spy.uint2(1, 2)
    p["u3"] = spy.uint3(1, 2, 3)
    p["u4"] = spy.uint4(1, 2, 3, 4)
    p["f2"] = spy.float2(1.0, 2.0)
    p["f3"] = spy.float3(1.0, 2.0, 3.0)
    p["f4"] = spy.float4(1.0, 2.0, 3.0, 4.0)
    p["m3"] = spy.float3x3([1, 0, 0, 0, 1, 0, 0, 0, 1])
    p["m4"] = spy.float4x4([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])
    p["path"] = Path("results/output.exr")

    assert p.get_bool("b") == True
    assert p.get_int("i") == 42
    assert p.get_float("f") == pytest.approx(3.14)
    assert p.get_string("s") == "hello"
    assert p.get_int2("i2") == spy.int2(1, 2)
    assert p.get_int3("i3") == spy.int3(1, 2, 3)
    assert p.get_int4("i4") == spy.int4(1, 2, 3, 4)
    assert p.get_uint2("u2") == spy.uint2(1, 2)
    assert p.get_uint3("u3") == spy.uint3(1, 2, 3)
    assert p.get_uint4("u4") == spy.uint4(1, 2, 3, 4)
    assert p.get_float2("f2") == spy.float2(1.0, 2.0)
    assert p.get_float3("f3") == spy.float3(1.0, 2.0, 3.0)
    assert p.get_float4("f4") == spy.float4(1.0, 2.0, 3.0, 4.0)
    assert p.get_float3x3("m3") == spy.float3x3([1, 0, 0, 0, 1, 0, 0, 0, 1])
    assert p.get_float4x4("m4") == spy.float4x4([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])
    assert p.get_path("path") == Path("results/output.exr")
    assert isinstance(p.get_path("path"), Path)

    # get_path also works when stored as a plain string
    p["dir"] = "/some/directory"
    assert p.get_path("dir") == Path("/some/directory")


def test_typed_getters_default():
    """Test typed getters return defaults when property is missing."""
    p = Properties()

    assert p.get_bool("missing", False) == False
    assert p.get_int("missing", 99) == 99
    assert p.get_float("missing", 1.5) == pytest.approx(1.5)
    assert p.get_string("missing", "default") == "default"
    assert p.get_float3("missing", spy.float3(0, 0, 0)) == spy.float3(0, 0, 0)
    assert p.get_path("missing", Path("fallback.exr")) == Path("fallback.exr")


def test_typed_getters_missing_no_default():
    """Test typed getters throw when property is missing and no default is given."""
    p = Properties()

    with pytest.raises(Exception, match="not found"):
        p.get_int("missing")

    with pytest.raises(Exception, match="not found"):
        p.get_string("missing")

    with pytest.raises(Exception, match="not found"):
        p.get_path("missing")


def test_typed_getters_type_mismatch():
    """Test typed getters throw on type mismatch."""
    p = Properties()
    p["value"] = "hello"

    with pytest.raises(Exception):
        p.get_int("value")

    with pytest.raises(Exception):
        p.get_bool("value")


# =============================================================================
# Object typed getter
# =============================================================================


def test_get_object():
    """Test the get_object() typed getter."""
    from falcor2 import AssetResolver

    p = Properties()
    resolver = AssetResolver()
    p["resolver"] = resolver

    # Successful retrieval
    result = p.get_object("resolver")
    assert isinstance(result, AssetResolver)

    # Default value when missing
    default_resolver = AssetResolver()
    result = p.get_object("missing", default_resolver)
    assert result is default_resolver

    # Throws when missing with no default
    with pytest.raises(Exception, match="not found"):
        p.get_object("missing")

    # Type mismatch (string is not an Object)
    p["str_val"] = "hello"
    with pytest.raises(Exception):
        p.get_object("str_val")


# =============================================================================
# Enum typed getter
# =============================================================================


def test_get_enum():
    """Test the get_enum() typed getter for bound enums."""
    p = Properties()

    # Store a nanobind-bound enum value
    p["pt"] = PropertyType.float3

    # Retrieve via get_enum
    result = p.get_enum("pt", PropertyType)
    assert result == PropertyType.float3
    assert isinstance(result, PropertyType)

    # Default when missing
    result = p.get_enum("missing", PropertyType, PropertyType.int)
    assert result == PropertyType.int

    # Missing without default returns None
    result = p.get_enum("missing", PropertyType)
    assert result is None

    # Integer stored, retrieved as enum
    p["int_as_enum"] = 2  # PropertyType.int has value 2
    result = p.get_enum("int_as_enum", PropertyType)
    assert result == PropertyType.int

    # Type mismatch (string is not an enum or int)
    p["str_val"] = "hello"
    with pytest.raises(Exception):
        p.get_enum("str_val", PropertyType)


def test_python_enum():
    """Test that Python stdlib IntEnum is stored as int (implicit int conversion)."""
    import enum

    class Color(enum.IntEnum):
        red = 0
        green = 1
        blue = 2

    p = Properties()
    p["color"] = Color.green
    # IntEnum supports implicit int conversion, so nanobind should cast to int64_t.
    assert p.type("color") == PropertyType.int
    assert p["color"] == 1


# =============================================================================
# Any storage
# =============================================================================


def test_any_roundtrip():
    """Test storing and retrieving a nanobind-bound C++ object via Any."""
    from falcor2 import AABB

    p = Properties()
    box = AABB(spy.float3(0, 0, 0), spy.float3(1, 1, 1))
    p["box"] = box
    assert p.type("box") == PropertyType.any

    retrieved = p["box"]
    assert isinstance(retrieved, AABB)
    assert retrieved.min == spy.float3(0, 0, 0)
    assert retrieved.max == spy.float3(1, 1, 1)


# =============================================================================
# Iteration & repr
# =============================================================================


def test_iter_yields_keys():
    """Test that iterating over Properties yields key strings (dict-like semantics)."""
    p = Properties()
    p["alpha"] = 1
    p["beta"] = 2.0
    p["gamma"] = "hello"

    keys = list(p)
    assert keys == ["alpha", "beta", "gamma"]
    assert all(isinstance(k, str) for k in keys)

    collected = {k: p[k] for k in p}
    assert collected == {"alpha": 1, "beta": 2.0, "gamma": "hello"}


def test_iter_empty():
    """Test that iterating over empty Properties yields nothing."""
    p = Properties()
    assert list(p) == []


def test_items():
    props = Properties()
    props["a"] = 1
    props["b"] = 2.0
    props["c"] = "hello"

    items = props.items()
    assert len(items) == 3
    item_dict = {k: v for k, v in items}
    assert item_dict["a"] == 1
    assert item_dict["b"] == 2.0
    assert item_dict["c"] == "hello"


def test_dir_yields_keys():
    p = Properties()
    p["alpha"] = 1
    p["beta"] = 2.0
    p["gamma"] = "hello"

    assert dir(p) == ["alpha", "beta", "gamma"]


def test_dir_empty():
    assert dir(Properties()) == []


def test_repr():
    """Test string representation."""
    p = Properties()
    assert repr(p) == "Properties({})"

    p["prop_1"] = 1
    p["prop_2"] = "hello"
    r = repr(p)
    assert "Properties" in r
    assert "prop_1" in r
    assert "prop_2" in r


# =============================================================================
# PropertyList roundtrips (parametrized)
# =============================================================================

LIST_ROUNDTRIP_CASES = [
    ("bool", [True, False, True]),
    ("int", [10, 20, 30]),
    ("float", [1.5, 2.5, 3.5]),
    ("string", ["alpha", "beta", "gamma"]),
    ("int2", [spy.int2(1, 2), spy.int2(3, 4)]),
    ("int3", [spy.int3(1, 2, 3), spy.int3(4, 5, 6)]),
    ("int4", [spy.int4(1, 2, 3, 4), spy.int4(5, 6, 7, 8)]),
    ("uint2", [spy.uint2(1, 2), spy.uint2(3, 4)]),
    ("uint3", [spy.uint3(1, 2, 3), spy.uint3(4, 5, 6)]),
    ("uint4", [spy.uint4(1, 2, 3, 4), spy.uint4(5, 6, 7, 8)]),
    ("float2", [spy.float2(1.0, 2.0), spy.float2(3.0, 4.0)]),
    ("float3", [spy.float3(1, 0, 0), spy.float3(0, 1, 0), spy.float3(0, 0, 1)]),
    ("float4", [spy.float4(1, 2, 3, 4), spy.float4(5, 6, 7, 8)]),
    (
        "float3x3",
        [
            spy.float3x3([1, 0, 0, 0, 1, 0, 0, 0, 1]),
            spy.float3x3([2, 0, 0, 0, 2, 0, 0, 0, 2]),
        ],
    ),
    (
        "float4x4",
        [
            spy.float4x4([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]),
            spy.float4x4([2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2]),
        ],
    ),
]


@pytest.mark.parametrize("type_name, data", LIST_ROUNDTRIP_CASES)
def test_list_roundtrip(type_name: str, data: list):
    """Test setting and getting a homogeneous list via __setitem__ / __getitem__."""
    p = Properties()
    p["v"] = data
    assert p.type("v") == PropertyType.list
    result = p["v"]
    assert isinstance(result, list)
    assert len(result) == len(data)
    for i in range(len(data)):
        if isinstance(data[i], float):
            assert result[i] == pytest.approx(data[i])
        else:
            assert result[i] == data[i]


# =============================================================================
# PropertyList typed getters
# =============================================================================


def test_list_typed_getters():
    """Test typed list getter methods."""
    p = Properties()

    p["bools"] = [True, False]
    assert p.get_bool_list("bools") == [True, False]

    p["ints"] = [1, 2, 3]
    assert p.get_int_list("ints") == [1, 2, 3]

    p["floats"] = [1.0, 2.0]
    result = p.get_float_list("floats")
    assert len(result) == 2
    assert result[0] == pytest.approx(1.0)

    p["strs"] = ["a", "b", "c"]
    assert p.get_string_list("strs") == ["a", "b", "c"]


def test_list_typed_getters_vector():
    """Test typed list getters for all vector/matrix types."""
    p = Properties()

    p["i2"] = [spy.int2(1, 2), spy.int2(3, 4)]
    assert p.get_int2_list("i2") == [spy.int2(1, 2), spy.int2(3, 4)]

    p["i3"] = [spy.int3(1, 2, 3)]
    assert p.get_int3_list("i3") == [spy.int3(1, 2, 3)]

    p["i4"] = [spy.int4(1, 2, 3, 4)]
    assert p.get_int4_list("i4") == [spy.int4(1, 2, 3, 4)]

    p["u2"] = [spy.uint2(1, 2)]
    assert p.get_uint2_list("u2") == [spy.uint2(1, 2)]

    p["u3"] = [spy.uint3(1, 2, 3)]
    assert p.get_uint3_list("u3") == [spy.uint3(1, 2, 3)]

    p["u4"] = [spy.uint4(1, 2, 3, 4)]
    assert p.get_uint4_list("u4") == [spy.uint4(1, 2, 3, 4)]

    p["f2"] = [spy.float2(1, 2)]
    assert p.get_float2_list("f2") == [spy.float2(1, 2)]

    p["f3"] = [spy.float3(1, 2, 3), spy.float3(4, 5, 6)]
    assert p.get_float3_list("f3") == [spy.float3(1, 2, 3), spy.float3(4, 5, 6)]

    p["f4"] = [spy.float4(1, 2, 3, 4)]
    assert p.get_float4_list("f4") == [spy.float4(1, 2, 3, 4)]

    p["m3"] = [spy.float3x3([1, 0, 0, 0, 1, 0, 0, 0, 1])]
    assert p.get_float3x3_list("m3") == [spy.float3x3([1, 0, 0, 0, 1, 0, 0, 0, 1])]

    p["m4"] = [spy.float4x4([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])]
    assert p.get_float4x4_list("m4") == [
        spy.float4x4([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])
    ]


def test_list_path_roundtrip():
    """Test get_path_list() retrieves string lists as Path objects."""
    p = Properties()
    # Paths are stored as strings; store as strings and retrieve via get_path_list
    p["paths"] = ["/foo/bar", "/baz/qux"]
    assert p.type("paths") == PropertyType.list
    result = p.get_path_list("paths")
    assert len(result) == 2
    assert result[0] == Path("/foo/bar")
    assert result[1] == Path("/baz/qux")
    assert all(isinstance(r, Path) for r in result)


def test_list_type_mismatch():
    """Test that typed list getters throw on element type mismatch."""
    p = Properties()
    p["f3s"] = [spy.float3(1, 0, 0)]

    with pytest.raises(Exception):
        p.get_int3_list("f3s")

    with pytest.raises(Exception):
        p.get_float_list("f3s")


def test_list_missing_key():
    """Test that typed list getters throw on missing key."""
    p = Properties()

    with pytest.raises(Exception):
        p.get_float3_list("nonexistent")

    with pytest.raises(Exception):
        p.get_string_list("nonexistent")


# =============================================================================
# PropertyList misc
# =============================================================================


def test_list_empty_raises():
    """Test that assigning an empty list raises an error."""
    p = Properties()
    with pytest.raises(Exception, match="empty list"):
        p["v"] = []


def test_list_tuple_input():
    """Test that tuples are also accepted as list input."""
    p = Properties()
    p["v"] = (spy.float3(1, 0, 0), spy.float3(0, 1, 0))
    assert p.type("v") == PropertyType.list
    result = p["v"]
    assert len(result) == 2  # type: ignore
    assert result[0] == spy.float3(1, 0, 0)  # type: ignore


def test_list_dict_construction():
    """Test constructing Properties from a dict with lists."""
    p = Properties(
        {
            "pts": [spy.float3(1, 0, 0), spy.float3(0, 1, 0)],
            "names": ["a", "b"],
            "scale": 2.5,
        }
    )
    assert p.type("pts") == PropertyType.list
    assert p.type("names") == PropertyType.list
    assert p.type("scale") == PropertyType.float
    assert len(p["pts"]) == 2  # type: ignore
    assert p["names"] == ["a", "b"]


def test_list_equality():
    """Test Properties equality with lists."""
    p1 = Properties()
    p1["v"] = [spy.float3(1, 0, 0)]

    p2 = Properties()
    p2["v"] = [spy.float3(1, 0, 0)]

    assert p1 == p2

    p3 = Properties()
    p3["v"] = [spy.float3(0, 1, 0)]

    assert p1 != p3


def test_list_hash():
    """Test Properties hash with lists."""
    p1 = Properties()
    p1["v"] = [spy.float3(1, 0, 0), spy.float3(0, 1, 0)]

    p2 = Properties()
    p2["v"] = [spy.float3(1, 0, 0), spy.float3(0, 1, 0)]

    assert p1.hash() == p2.hash()

    p3 = Properties()
    p3["v"] = [spy.float3(0, 0, 1)]

    assert p1.hash() != p3.hash()


# =============================================================================
# PropertyList object
# =============================================================================


def test_list_object_roundtrip():
    """Test setting and getting a list of Objects."""
    from falcor2 import AssetResolver

    r1 = AssetResolver()
    r2 = AssetResolver()
    p = Properties()
    p["resolvers"] = [r1, r2]
    assert p.type("resolvers") == PropertyType.list
    result = p["resolvers"]
    assert isinstance(result, list)
    assert len(result) == 2
    assert isinstance(result[0], AssetResolver)
    assert isinstance(result[1], AssetResolver)


def test_list_object_typed_getter():
    """Test the typed get_object_list getter."""
    from falcor2 import AssetResolver

    r1 = AssetResolver()
    p = Properties()
    p["v"] = [r1]
    result = p.get_object_list("v")
    assert len(result) == 1
    assert isinstance(result[0], AssetResolver)


def test_list_object_equality():
    """Test Properties equality with object lists."""
    from falcor2 import AssetResolver

    r = AssetResolver()
    p1 = Properties()
    p1["v"] = [r]

    p2 = Properties()
    p2["v"] = [r]

    assert p1 == p2

    p3 = Properties()
    p3["v"] = [AssetResolver()]

    assert p1 != p3


if __name__ == "__main__":
    pytest.main([__file__, "-vvv"])
