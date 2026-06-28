# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Property descriptor and @reflected class decorator."""

import enum

import pytest

from falcor2.reflection import PythonPropertyInfo, reflected, Property, UIFlags


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class Color(enum.IntEnum):
    RED = 0
    GREEN = 1
    BLUE = 2


# ---------------------------------------------------------------------------
# Simple (stored-value) form
# ---------------------------------------------------------------------------


class TestStoredValueForm:
    def test_basic_stored_property(self):
        @reflected
        class Obj:
            roughness = Property(0.5, doc="Roughness", value_range=(0.0, 1.0))

        o = Obj()
        assert o.roughness == 0.5
        o.roughness = 0.8
        assert o.roughness == 0.8

    def test_default_value_preserved(self):
        @reflected
        class Obj:
            x = Property(42)

        a = Obj()
        b = Obj()
        assert a.x == 42
        assert b.x == 42
        a.x = 100
        assert a.x == 100
        assert b.x == 42  # instances are independent

    def test_bool_property(self):
        @reflected
        class Obj:
            flag = Property(True, doc="A flag")

        o = Obj()
        assert o.flag is True
        o.flag = False
        assert o.flag is False

    def test_string_property(self):
        @reflected
        class Obj:
            name = Property("default", doc="Name")

        o = Obj()
        assert o.name == "default"
        o.name = "changed"
        assert o.name == "changed"

    def test_int_property(self):
        @reflected
        class Obj:
            count = Property(10)

        o = Obj()
        assert o.count == 10

    def test_class_access_returns_descriptor(self):
        @reflected
        class Obj:
            val = Property(1.0)

        assert isinstance(Obj.val, Property)


# ---------------------------------------------------------------------------
# Getter/setter (computed) form
# ---------------------------------------------------------------------------


class TestGetterSetterForm:
    def test_getter_only(self):
        @reflected
        class Obj:
            @Property
            def derived(self) -> float:
                return 3.14

        o = Obj()
        assert o.derived == pytest.approx(3.14)

    def test_read_only_rejects_write(self):
        @reflected
        class Obj:
            @Property
            def derived(self) -> float:
                return 3.14

        o = Obj()
        with pytest.raises(AttributeError, match="read-only"):
            o.derived = 1.0

    def test_getter_and_setter(self):
        @reflected
        class Obj:
            def __init__(self):
                super().__init__()
                self._val = 0.0

            @Property(doc="A computed value")
            def val(self) -> float:
                return self._val

            @val.setter
            def val(self, value: float) -> None:
                self._val = value

        o = Obj()
        assert o.val == 0.0
        o.val = 5.5
        assert o.val == 5.5

    def test_keyword_decorator_form(self):
        @reflected
        class Obj:
            def __init__(self):
                super().__init__()
                self._x = 0.0

            @Property(doc="X value", value_range=(0.0, 10.0))
            def x(self) -> float:
                return self._x

            @x.setter
            def x(self, value: float) -> None:
                self._x = value

        o = Obj()
        o.x = 7.0
        assert o.x == 7.0

    def test_cannot_add_setter_to_stored(self):
        with pytest.raises(TypeError, match="Cannot add setter"):
            prop = Property(1.0)

            @prop.setter
            def _set(self, value):
                pass


# ---------------------------------------------------------------------------
# @reflected decorator and _reflected_properties
# ---------------------------------------------------------------------------


class TestReflectedDecorator:
    def test_creates_reflected_properties(self):
        @reflected
        class Obj:
            a = Property(1)
            b = Property(2.0)

        assert hasattr(Obj, "_reflected_properties")
        assert isinstance(Obj._reflected_properties, list)
        assert len(Obj._reflected_properties) == 2

    def test_property_names(self):
        @reflected
        class Obj:
            alpha = Property(0.0)
            beta = Property(0.0)

        names = [p.name for p in Obj._reflected_properties]
        assert "alpha" in names
        assert "beta" in names

    def test_metadata_captured(self):
        @reflected
        class Obj:
            roughness = Property(
                0.5,
                doc="Surface roughness",
                value_range=(0.0, 1.0),
                ui_flags=UIFlags.advanced,
                ui_label="Roughness",
                ui_group="Appearance",
                ui_drag_speed=0.01,
            )

        info = Obj._reflected_properties[0]
        assert isinstance(info, PythonPropertyInfo)
        assert info.doc == "Surface roughness"
        assert info.value_range == (0.0, 1.0)
        assert info.ui_flags == UIFlags.advanced
        assert info.ui_label == "Roughness"
        assert info.ui_group == "Appearance"
        assert info.ui_drag_speed == 0.01

    def test_enable_if_captured(self):
        predicate = lambda self: self.enabled

        @reflected
        class Obj:
            enabled = Property(True)
            value = Property(1.0, ui_enable_if=predicate)

        info = [p for p in Obj._reflected_properties if p.name == "value"][0]
        assert isinstance(info, PythonPropertyInfo)
        assert info.ui_enable_if is predicate

    def test_enum_type_captured(self):
        @reflected
        class Obj:
            color = Property(Color.RED, enum_type=Color)

        info = Obj._reflected_properties[0]
        assert isinstance(info, PythonPropertyInfo)
        assert info.enum_type is Color

    def test_declaration_order_preserved(self):
        @reflected
        class Obj:
            a = Property(1)
            b = Property(2)
            c = Property(3)

        names = [p.name for p in Obj._reflected_properties]
        assert names == ["a", "b", "c"]

    def test_getter_setter_info(self):
        @reflected
        class Obj:
            def __init__(self):
                super().__init__()
                self._x = 0.0

            @Property(doc="X")
            def x(self) -> float:
                return self._x

            @x.setter
            def x(self, value: float) -> None:
                self._x = value

        info = Obj._reflected_properties[0]
        assert isinstance(info, PythonPropertyInfo)
        assert info.name == "x"
        assert info.getter is not None
        assert info.setter is not None
        assert info.doc == "X"

    def test_read_only_info(self):
        @reflected
        class Obj:
            @Property
            def derived(self) -> float:
                return 1.0

        info = Obj._reflected_properties[0]
        assert isinstance(info, PythonPropertyInfo)
        assert info.getter is not None
        assert info.setter is None


# ---------------------------------------------------------------------------
# Inheritance
# ---------------------------------------------------------------------------


class TestInheritance:
    def test_subclass_inherits_base_properties(self):
        @reflected
        class Base:
            a = Property(1)

        @reflected
        class Child(Base):
            b = Property(2)

        names = [p.name for p in Child._reflected_properties]
        assert "a" in names
        assert "b" in names

    def test_base_properties_come_first(self):
        @reflected
        class Base:
            a = Property(1)

        @reflected
        class Child(Base):
            b = Property(2)

        names = [p.name for p in Child._reflected_properties]
        assert names.index("a") < names.index("b")

    def test_subclass_can_override_property(self):
        @reflected
        class Base:
            x = Property(1, doc="base")

        @reflected
        class Child(Base):
            x = Property(2, doc="child")

        info = [p for p in Child._reflected_properties if p.name == "x"][0]
        assert info.doc == "child"
        # Only one entry for 'x'
        assert sum(1 for p in Child._reflected_properties if p.name == "x") == 1

    def test_deep_inheritance(self):
        @reflected
        class A:
            x = Property(1)

        @reflected
        class B(A):
            y = Property(2)

        @reflected
        class C(B):
            z = Property(3)

        names = [p.name for p in C._reflected_properties]
        assert names == ["x", "y", "z"]

    def test_instances_isolated_across_inheritance(self):
        @reflected
        class Base:
            val = Property(10)

        @reflected
        class Child(Base):
            pass

        b = Base()
        c = Child()
        b.val = 100
        assert c.val == 10


# ---------------------------------------------------------------------------
# Type inference
# ---------------------------------------------------------------------------


class TestTypeInference:
    def test_bool_type(self):
        @reflected
        class Obj:
            flag = Property(True)

        info = Obj._reflected_properties[0]
        assert isinstance(info, PythonPropertyInfo)
        assert info.value_type is bool

    def test_int_type(self):
        @reflected
        class Obj:
            count = Property(42)

        info = Obj._reflected_properties[0]
        assert isinstance(info, PythonPropertyInfo)
        assert info.value_type is int

    def test_float_type(self):
        @reflected
        class Obj:
            roughness = Property(0.5)

        info = Obj._reflected_properties[0]
        assert isinstance(info, PythonPropertyInfo)
        assert info.value_type is float

    def test_string_type(self):
        @reflected
        class Obj:
            name = Property("hello")

        info = Obj._reflected_properties[0]
        assert isinstance(info, PythonPropertyInfo)
        assert info.value_type is str

    def test_getter_return_annotation_used(self):
        @reflected
        class Obj:
            @Property
            def val(self) -> float:
                return 0.0

        info = Obj._reflected_properties[0]
        assert isinstance(info, PythonPropertyInfo)
        assert info.value_type is float

    def test_explicit_value_type_overrides(self):
        @reflected
        class Obj:
            val = Property(0, value_type=float)

        info = Obj._reflected_properties[0]
        assert isinstance(info, PythonPropertyInfo)
        assert info.value_type is float

    def test_enum_inferred(self):
        @reflected
        class Obj:
            color = Property(Color.RED, enum_type=Color)

        info = Obj._reflected_properties[0]
        assert isinstance(info, PythonPropertyInfo)
        assert info.enum_type is Color


# ---------------------------------------------------------------------------
# PythonPropertyInfo getter/setter callables
# ---------------------------------------------------------------------------


class TestInfoCallables:
    def test_stored_getter_setter(self):
        @reflected
        class Obj:
            val = Property(10)

        o = Obj()
        info = Obj._reflected_properties[0]
        assert isinstance(info, PythonPropertyInfo)
        assert info.getter is not None
        assert info.getter(o) == 10
        assert info.setter is not None
        info.setter(o, 20)
        assert info.getter(o) == 20
        assert o.val == 20

    def test_computed_getter_setter(self):
        @reflected
        class Obj:
            def __init__(self):
                super().__init__()
                self._x = 0.0

            @Property
            def x(self) -> float:
                return self._x

            @x.setter
            def x(self, value: float) -> None:
                self._x = value

        o = Obj()
        info = Obj._reflected_properties[0]
        assert isinstance(info, PythonPropertyInfo)
        assert info.getter is not None
        assert info.getter(o) == 0.0
        assert info.setter is not None
        info.setter(o, 3.14)
        assert info.getter(o) == pytest.approx(3.14)

    def test_default_value_stored(self):
        @reflected
        class Obj:
            val = Property(42)

        info = Obj._reflected_properties[0]
        assert isinstance(info, PythonPropertyInfo)
        assert info.default_value == 42

    def test_computed_has_no_default(self):
        @reflected
        class Obj:
            @Property
            def val(self) -> float:
                return 1.0

        info = Obj._reflected_properties[0]
        assert isinstance(info, PythonPropertyInfo)
        assert info.default_value is None


# ---------------------------------------------------------------------------
# OnChange callback
# ---------------------------------------------------------------------------


class TestOnChange:
    def test_on_change_captured(self):
        callback = lambda self: None

        @reflected
        class Obj:
            val = Property(1.0, on_change=callback)

        info = Obj._reflected_properties[0]
        assert isinstance(info, PythonPropertyInfo)
        assert info.on_change is callback

    def test_on_change_default_none(self):
        @reflected
        class Obj:
            val = Property(1.0)

        info = Obj._reflected_properties[0]
        assert isinstance(info, PythonPropertyInfo)
        assert info.on_change is None

    def test_on_change_called_on_stored_set(self):
        calls = []

        @reflected
        class Obj:
            val = Property(1.0, on_change=lambda self: calls.append(self))

        obj = Obj()
        obj.val = 2.0
        assert len(calls) == 1
        assert calls[0] is obj

    def test_on_change_called_on_getter_setter_set(self):
        calls = []

        @reflected
        class Obj:
            @Property(on_change=lambda self: calls.append(self))
            def val(self) -> float:
                return self._val if hasattr(self, "_val") else 0.0

            @val.setter
            def val(self, value: float) -> None:
                self._val = value

        obj = Obj()
        obj.val = 5.0
        assert len(calls) == 1
        assert calls[0] is obj

    def test_on_change_not_called_on_get(self):
        calls = []

        @reflected
        class Obj:
            val = Property(1.0, on_change=lambda self: calls.append(self))

        obj = Obj()
        _ = obj.val
        assert len(calls) == 0

    def test_on_change_called_even_if_value_unchanged(self):
        calls = []

        @reflected
        class Obj:
            val = Property(1.0, on_change=lambda self: calls.append(self))

        obj = Obj()
        obj.val = 1.0  # Same as default.
        obj.val = 1.0
        assert len(calls) == 2

    def test_on_change_receives_instance(self):
        received = []

        @reflected
        class Obj:
            val = Property(0, on_change=lambda self: received.append(self))

        a = Obj()
        b = Obj()
        a.val = 1
        b.val = 2
        assert received[0] is a
        assert received[1] is b


if __name__ == "__main__":
    pytest.main([__file__, "-vvv"])
