# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the reflected_property descriptor and @reflected class decorator."""

from __future__ import annotations

import enum

import pytest

from falcor2.reflection import (
    PythonPropertyInfo,
    UIFlags,
    object_factory,
    reflected,
    reflected_property,
)


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
            roughness = reflected_property(0.5, doc="Roughness", value_range=(0.0, 1.0))

        o = Obj()
        assert o.roughness == 0.5
        o.roughness = 0.8
        assert o.roughness == 0.8

    def test_default_value_preserved(self):
        @reflected
        class Obj:
            x = reflected_property(42)

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
            flag = reflected_property(True, doc="A flag")

        o = Obj()
        assert o.flag is True
        o.flag = False
        assert o.flag is False

    def test_string_property(self):
        @reflected
        class Obj:
            name = reflected_property("default", doc="Name")

        o = Obj()
        assert o.name == "default"
        o.name = "changed"
        assert o.name == "changed"

    def test_int_property(self):
        @reflected
        class Obj:
            count = reflected_property(10)

        o = Obj()
        assert o.count == 10

    def test_class_access_returns_descriptor(self):
        @reflected
        class Obj:
            val = reflected_property(1.0)

        assert isinstance(Obj.val, reflected_property)

    def test_property_is_builtin_property_subclass(self):
        class Obj:
            val = reflected_property(1.0)

        assert isinstance(Obj.val, property)

    def test_delete_is_rejected(self):
        class Obj:
            val = reflected_property(1.0)

        with pytest.raises(AttributeError, match="cannot be deleted"):
            del Obj().val


# ---------------------------------------------------------------------------
# Getter/setter (computed) form
# ---------------------------------------------------------------------------


class TestGetterSetterForm:
    def test_getter_replacement_is_rejected(self):
        prop = reflected_property(lambda _: 1.0)

        with pytest.raises(TypeError, match="does not support getter replacement"):
            prop.getter(lambda _: 2.0)

    def test_deleter_is_rejected(self):
        prop = reflected_property(lambda _: 1.0)

        with pytest.raises(TypeError, match="does not support deletion"):
            prop.deleter(lambda _: None)

    def test_getter_only(self):
        @reflected
        class Obj:
            @reflected_property
            def derived(self) -> float:
                return 3.14

        o = Obj()
        assert o.derived == pytest.approx(3.14)

    def test_read_only_rejects_write(self):
        @reflected
        class Obj:
            @reflected_property
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

            @reflected_property(doc="A computed value")
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

            @reflected_property(doc="X value", value_range=(0.0, 10.0))
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
            prop = reflected_property(1.0)

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
            a = reflected_property(1)
            b = reflected_property(2.0)

        assert hasattr(Obj, "_reflected_properties")
        assert isinstance(Obj._reflected_properties, list)
        assert len(Obj._reflected_properties) == 2

    def test_property_names(self):
        @reflected
        class Obj:
            alpha = reflected_property(0.0)
            beta = reflected_property(0.0)

        names = [p.name for p in Obj._reflected_properties]
        assert "alpha" in names
        assert "beta" in names

    def test_metadata_captured(self):
        @reflected
        class Obj:
            roughness = reflected_property(
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
            enabled = reflected_property(True)
            value = reflected_property(1.0, ui_enable_if=predicate)

        info = [p for p in Obj._reflected_properties if p.name == "value"][0]
        assert isinstance(info, PythonPropertyInfo)
        assert info.ui_enable_if is predicate

    def test_declaration_order_preserved(self):
        @reflected
        class Obj:
            a = reflected_property(1)
            b = reflected_property(2)
            c = reflected_property(3)

        names = [p.name for p in Obj._reflected_properties]
        assert names == ["a", "b", "c"]

    def test_getter_setter_info(self):
        @reflected
        class Obj:
            def __init__(self):
                super().__init__()
                self._x = 0.0

            @reflected_property(doc="X")
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
            @reflected_property
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
            a = reflected_property(1)

        @reflected
        class Child(Base):
            b = reflected_property(2)

        names = [p.name for p in Child._reflected_properties]
        assert "a" in names
        assert "b" in names

    def test_base_properties_come_first(self):
        @reflected
        class Base:
            a = reflected_property(1)

        @reflected
        class Child(Base):
            b = reflected_property(2)

        names = [p.name for p in Child._reflected_properties]
        assert names.index("a") < names.index("b")

    def test_subclass_can_override_property(self):
        @reflected
        class Base:
            x = reflected_property(1, doc="base")

        @reflected
        class Child(Base):
            x = reflected_property(2, doc="child")

        info = [p for p in Child._reflected_properties if p.name == "x"][0]
        assert info.doc == "child"
        # Only one entry for 'x'
        assert sum(1 for p in Child._reflected_properties if p.name == "x") == 1

    def test_deep_inheritance(self):
        @reflected
        class A:
            x = reflected_property(1)

        @reflected
        class B(A):
            y = reflected_property(2)

        @reflected
        class C(B):
            z = reflected_property(3)

        names = [p.name for p in C._reflected_properties]
        assert names == ["x", "y", "z"]

    def test_instances_isolated_across_inheritance(self):
        @reflected
        class Base:
            val = reflected_property(10)

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
            flag = reflected_property(True)

        info = Obj._reflected_properties[0]
        assert isinstance(info, PythonPropertyInfo)
        assert info.value_type is bool

    def test_int_type(self):
        @reflected
        class Obj:
            count = reflected_property(42)

        info = Obj._reflected_properties[0]
        assert isinstance(info, PythonPropertyInfo)
        assert info.value_type is int

    def test_float_type(self):
        @reflected
        class Obj:
            roughness = reflected_property(0.5)

        info = Obj._reflected_properties[0]
        assert isinstance(info, PythonPropertyInfo)
        assert info.value_type is float

    def test_string_type(self):
        @reflected
        class Obj:
            name = reflected_property("hello")

        info = Obj._reflected_properties[0]
        assert isinstance(info, PythonPropertyInfo)
        assert info.value_type is str

    def test_computed_property_infers_return_type(self):
        @reflected
        class Obj:
            @reflected_property
            def val(self) -> float:
                return 0.0

        info = Obj._reflected_properties[0]
        assert isinstance(info, PythonPropertyInfo)
        assert info.value_type is float

    def test_string_return_annotation_is_resolved(self):
        @reflected
        class Obj:
            @reflected_property
            def val(self) -> "float":
                return 0.0

        info = Obj._reflected_properties[0]
        assert isinstance(info, PythonPropertyInfo)
        assert info.value_type is float

    def test_self_return_annotation_is_resolved(self):
        @reflected
        class Obj:
            @reflected_property
            def val(self) -> "Obj":
                return self

        info = Obj._reflected_properties[0]
        assert isinstance(info, PythonPropertyInfo)
        assert info.value_type is Obj

    def test_unresolved_return_annotation_requires_value_type(self):
        with pytest.raises(TypeError, match="requires value_type") as exc_info:

            @reflected
            class Obj:
                @reflected_property
                def val(self) -> "MissingType":
                    return None

        assert isinstance(exc_info.value.__cause__, NameError)

    def test_computed_property_uses_explicit_value_type(self):
        @reflected
        class Obj:
            @reflected_property(value_type=float)
            def val(self) -> "float":
                return 0.0

        info = Obj._reflected_properties[0]
        assert isinstance(info, PythonPropertyInfo)
        assert info.value_type is float

    def test_explicit_value_type_overrides(self):
        @reflected
        class Obj:
            val = reflected_property(0, value_type=float)

        info = Obj._reflected_properties[0]
        assert isinstance(info, PythonPropertyInfo)
        assert info.value_type is float

    def test_enum_inferred(self):
        @reflected
        class Obj:
            color = reflected_property(Color.RED)

        info = Obj._reflected_properties[0]
        assert isinstance(info, PythonPropertyInfo)
        assert info.value_type is Color

    def test_optional_return_annotation_is_unwrapped(self):
        @reflected
        class Obj:
            @reflected_property
            def val(self) -> dict | None:
                return None

        info = Obj._reflected_properties[0]
        assert isinstance(info, PythonPropertyInfo)
        assert info.value_type is dict

    def test_object_factories_are_normalized(self):
        class Base:
            pass

        class DefaultConstructed(Base):
            pass

        class OwnerConstructed(Base):
            def __init__(self, owner):
                self.owner = owner

        custom = object_factory(
            OwnerConstructed,
            factory=lambda owner: OwnerConstructed(owner),
            label="Owner constructed",
        )

        @reflected
        class Obj:
            @reflected_property(
                value_type=Base,
                object_factories=(DefaultConstructed, custom),
            )
            def value(self) -> Base:
                return None

        factories = Obj._reflected_properties[0].object_factories
        assert [entry.object_type for entry in factories] == [
            DefaultConstructed,
            OwnerConstructed,
        ]
        assert [entry.label for entry in factories] == [
            "DefaultConstructed",
            "Owner constructed",
        ]

        owner = object()
        assert isinstance(factories[0].factory(owner), DefaultConstructed)
        assert factories[1].factory(owner).owner is owner

    @pytest.mark.parametrize("factories", [(dict,), (None,)])
    def test_object_factories_require_value_type(self, factories: tuple[type | None, ...]) -> None:
        with pytest.raises(TypeError, match="requires value_type"):

            @reflected
            class Obj:
                value = reflected_property(None, object_factories=factories)

    def test_none_object_factory_is_preserved(self):
        @reflected
        class Obj:
            @reflected_property(object_factories=(dict, None))
            def value(self) -> dict | None:
                return None

        factories = Obj._reflected_properties[0].object_factories
        assert factories[0] is not None
        assert factories[0].object_type is dict
        assert factories[1] is None

    def test_none_object_factory_must_be_unique(self):
        with pytest.raises(ValueError, match="duplicate None"):

            @reflected
            class Obj:
                @reflected_property(object_factories=(None, None))
                def value(self) -> dict | None:
                    return None

    def test_object_factory_type_must_derive_from_value_type(self):
        with pytest.raises(TypeError, match="must derive"):

            @reflected
            class Obj:
                @reflected_property(object_factories=(dict,))
                def value(self) -> list:
                    return []

    def test_object_factory_types_must_be_unique(self):
        with pytest.raises(ValueError, match="duplicate"):

            @reflected
            class Obj:
                @reflected_property(object_factories=(dict, dict))
                def value(self) -> dict:
                    return {}

    def test_object_factory_label_must_be_a_string(self):
        with pytest.raises(TypeError, match="label must be a string"):
            object_factory(dict, label=1)


# ---------------------------------------------------------------------------
# PythonPropertyInfo getter/setter callables
# ---------------------------------------------------------------------------


class TestInfoCallables:
    def test_stored_getter_setter(self):
        @reflected
        class Obj:
            val = reflected_property(10)

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

            @reflected_property
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
            val = reflected_property(42)

        info = Obj._reflected_properties[0]
        assert isinstance(info, PythonPropertyInfo)
        assert info.has_default_value is True
        assert info.default_value == 42

    def test_none_default_value_stored(self):
        @reflected
        class Obj:
            val = reflected_property(None, value_type=str)

        obj = Obj()
        info = Obj._reflected_properties[0]
        assert obj.val is None
        assert info.has_default_value is True
        assert info.default_value is None

        obj.val = "value"
        assert obj.val == "value"

    def test_computed_has_no_default(self):
        @reflected
        class Obj:
            @reflected_property
            def val(self) -> float:
                return 1.0

        info = Obj._reflected_properties[0]
        assert isinstance(info, PythonPropertyInfo)
        assert info.has_default_value is False
        assert info.default_value is None


# ---------------------------------------------------------------------------
# OnChange callback
# ---------------------------------------------------------------------------


class TestOnChange:
    def test_on_change_captured(self):
        callback = lambda self: None

        @reflected
        class Obj:
            val = reflected_property(1.0, on_change=callback)

        info = Obj._reflected_properties[0]
        assert isinstance(info, PythonPropertyInfo)
        assert info.on_change is callback

    def test_on_change_default_none(self):
        @reflected
        class Obj:
            val = reflected_property(1.0)

        info = Obj._reflected_properties[0]
        assert isinstance(info, PythonPropertyInfo)
        assert info.on_change is None

    def test_on_change_called_on_stored_set(self):
        calls = []

        @reflected
        class Obj:
            val = reflected_property(1.0, on_change=lambda self: calls.append(self))

        obj = Obj()
        obj.val = 2.0
        assert len(calls) == 1
        assert calls[0] is obj

    def test_on_change_called_on_getter_setter_set(self):
        calls = []

        @reflected
        class Obj:
            @reflected_property(on_change=lambda self: calls.append(self))
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
            val = reflected_property(1.0, on_change=lambda self: calls.append(self))

        obj = Obj()
        _ = obj.val
        assert len(calls) == 0

    def test_on_change_called_even_if_value_unchanged(self):
        calls = []

        @reflected
        class Obj:
            val = reflected_property(1.0, on_change=lambda self: calls.append(self))

        obj = Obj()
        obj.val = 1.0  # Same as default.
        obj.val = 1.0
        assert len(calls) == 2

    def test_on_change_receives_instance(self):
        received = []

        @reflected
        class Obj:
            val = reflected_property(0, on_change=lambda self: received.append(self))

        a = Obj()
        b = Obj()
        a.val = 1
        b.val = 2
        assert received[0] is a
        assert received[1] is b


if __name__ == "__main__":
    pytest.main([__file__, "-vvv"])
