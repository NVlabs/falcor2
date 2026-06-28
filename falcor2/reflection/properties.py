# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Decorator-based Python property reflection system.

Provides ``Property`` descriptors and the ``@reflected`` class decorator
to expose Python object properties to the C++ property system.

Architecture
------------
1. Users declare ``Property`` descriptors on their classes.
2. The ``@reflected`` decorator collects them into a ``_reflected_properties``
   list of frozen ``PythonPropertyInfo`` dataclass instances.
3. On the C++ side, ``PythonPropertyDescriptor`` reads each info object and
   presents it as a ``PropertyDescriptor`` compatible with the native
   property-editor pipeline.

Storage Convention
------------------
Stored-value properties keep their per-instance data in the instance's
``__dict__`` under a prefixed key (``_rp_<name>``) to avoid collisions with
user-defined attributes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Optional, overload

if TYPE_CHECKING:
    from falcor2.reflection import UIFlags

# Storage key prefix for reflected properties ("rp" = reflected property).
# Prevents collision with user-defined attributes on instances.
_REFLECTED_PROPERTY_PREFIX = "_rp_"

# ---------------------------------------------------------------------------
# PythonPropertyInfo  (consumed from C++ via PythonPropertyDescriptor)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PythonPropertyInfo:
    """Frozen description of a single property on a class."""

    name: str
    value_type: Optional[type] = None
    enum_type: Optional[type] = None
    getter: Optional[Callable[[Any], Any]] = None
    setter: Optional[Callable[[Any, Any], None]] = None
    default_value: Any = None
    doc: Optional[str] = None
    value_range: Optional[tuple[float, float]] = None
    ui_flags: Optional["UIFlags"] = None
    ui_label: Optional[str] = None
    ui_group: Optional[str] = None
    ui_drag_speed: Optional[float] = None
    ui_enable_if: Optional[Callable[[Any], bool]] = None
    on_change: Optional[Callable[[Any], None]] = None


# ---------------------------------------------------------------------------
# Property descriptor
# ---------------------------------------------------------------------------


class Property:
    """Descriptor that declares a property on a Python class.

    Simple (stored-value) form::

        class Foo:
            roughness = Property(0.5, doc="Roughness", value_range=(0.0, 1.0))

    Getter/setter (computed) form::

        class Foo:
            @Property(doc="Computed value")
            def my_prop(self) -> float:
                return self._my_prop

            @my_prop.setter
            def my_prop(self, value: float) -> None:
                self._my_prop = value
    """

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @overload
    def __init__(self, initial_value: Any, /, **kwargs: Any) -> None: ...

    @overload
    def __init__(self, fget: Callable[..., Any], /, **kwargs: Any) -> None: ...

    @overload
    def __init__(self, /, *, doc: Optional[str] = ..., **kwargs: Any) -> None: ...

    def __init__(self, first_arg: Any = None, /, **kwargs: Any) -> None:
        super().__init__()

        from falcor2.reflection import UIFlags

        # Metadata common to both forms.
        self._enum_type: Optional[type] = kwargs.get("enum_type", None)
        self._doc: Optional[str] = kwargs.get("doc", None)
        self._value_range: Optional[tuple[float, float]] = kwargs.get("value_range", None)
        self._ui_flags: UIFlags = kwargs.get("ui_flags", UIFlags.none)
        self._ui_label: Optional[str] = kwargs.get("ui_label", None)
        self._ui_group: Optional[str] = kwargs.get("ui_group", None)
        self._ui_drag_speed: Optional[float] = kwargs.get("ui_drag_speed", None)
        self._ui_enable_if: Optional[Callable[..., bool]] = kwargs.get("ui_enable_if", None)
        self._on_change: Optional[Callable[..., None]] = kwargs.get("on_change", None)

        # Property name (filled by __set_name__).
        self._name: Optional[str] = None
        # Storage key in instance __dict__ for simple form.
        self._storage_key: Optional[str] = None
        # Getter / setter for computed form.
        self._fget: Optional[Callable[..., Any]] = None
        self._fset: Optional[Callable[..., None]] = None
        # Whether this was created via the simple (stored-value) form.
        self._is_stored: bool = False
        # Default value (simple form only).
        self._default_value: Any = None
        # Explicit value type annotation (may be None).
        self._value_type: Optional[type] = kwargs.get("value_type", None)

        # State machine for the three construction forms:
        #   first_arg is None  -> pending decorator (keyword-only form)
        #   first_arg callable -> getter form
        #   otherwise          -> stored-value form
        if first_arg is None:
            self._pending_decorator = True
        elif callable(first_arg) and not isinstance(first_arg, type):
            self._pending_decorator = False
            self._fget = first_arg
            self._doc = self._doc or first_arg.__doc__
            # Try to infer value type from return annotation.
            annotations = getattr(first_arg, "__annotations__", {})
            if "return" in annotations and self._value_type is None:
                self._value_type = annotations["return"]
        else:
            self._pending_decorator = False
            self._is_stored = True
            self._default_value = first_arg

    # Allow ``@Property(doc=...)`` to be used as a decorator.
    def __call__(self, fget: Callable[..., Any]) -> Property:
        if not self._pending_decorator:
            raise TypeError(
                "Cannot use Property as a decorator in this form. "
                "Use @Property or @Property(doc=...) for decorator form, "
                "or Property(value) for stored-value form."
            )
        self._pending_decorator = False
        self._fget = fget
        self._doc = self._doc or fget.__doc__
        annotations = getattr(fget, "__annotations__", {})
        if "return" in annotations and self._value_type is None:
            self._value_type = annotations["return"]
        return self

    # ------------------------------------------------------------------
    # Setter decorator (mirrors @property)
    # ------------------------------------------------------------------

    def setter(self, fset: Callable[..., None]) -> Property:
        """Register a setter function. Usage: ``@my_prop.setter``."""
        if self._is_stored:
            raise TypeError("Cannot add setter to a stored-value Property")
        self._fset = fset
        return self

    # ------------------------------------------------------------------
    # Descriptor protocol
    # ------------------------------------------------------------------

    def __set_name__(self, owner: type, name: str) -> None:
        self._name = name
        self._storage_key = f"{_REFLECTED_PROPERTY_PREFIX}{name}"

    def __get__(self, instance: Any, owner: type) -> Any:
        if instance is None:
            return self
        if self._is_stored:
            try:
                return instance.__dict__[self._storage_key]
            except KeyError:
                return self._default_value
        if self._fget is not None:
            return self._fget(instance)
        raise AttributeError(f"Property '{self._name}' has no getter")

    def __set__(self, instance: Any, value: Any) -> None:
        if self._is_stored:
            instance.__dict__[self._storage_key] = value
            if self._on_change is not None:
                self._on_change(instance)
            return
        if self._fset is not None:
            self._fset(instance, value)
            if self._on_change is not None:
                self._on_change(instance)
            return
        raise AttributeError(f"Property '{self._name}' is read-only")

    # ------------------------------------------------------------------
    # Info builder  (used by @reflected)
    # ------------------------------------------------------------------

    def _make_info(self, name: str, owner: type) -> PythonPropertyInfo:
        """Build a :class:`PythonPropertyInfo` for this descriptor."""
        # Resolve value type.
        value_type = self._value_type
        if value_type is None and self._enum_type is not None:
            value_type = self._enum_type
        if value_type is None and self._is_stored and self._default_value is not None:
            value_type = type(self._default_value)

        # Build getter/setter callables that work on instances.
        if self._is_stored:
            storage_key = self._storage_key
            default = self._default_value

            def getter(inst: Any) -> Any:
                try:
                    return inst.__dict__[storage_key]
                except KeyError:
                    return default

            def _setter(inst: Any, val: Any) -> None:
                inst.__dict__[storage_key] = val

            setter = _setter

        else:
            fget = self._fget
            fset = self._fset

            def getter(inst: Any) -> Any:
                assert fget is not None
                return fget(inst)

            setter: Optional[Callable[..., None]] = None
            if fset is not None:

                def _setter(inst: Any, val: Any) -> None:
                    fset(inst, val)

                setter = _setter

        return PythonPropertyInfo(
            name=name,
            value_type=value_type,
            enum_type=self._enum_type,
            getter=getter,
            setter=setter,
            default_value=self._default_value if self._is_stored else None,
            doc=self._doc,
            value_range=self._value_range,
            ui_flags=self._ui_flags,
            ui_label=self._ui_label,
            ui_group=self._ui_group,
            ui_drag_speed=self._ui_drag_speed,
            ui_enable_if=self._ui_enable_if,
            on_change=self._on_change,
        )


# ---------------------------------------------------------------------------
# @reflected class decorator
# ---------------------------------------------------------------------------


def reflected(cls: type) -> type:
    """Class decorator that collects ``Property`` descriptors from a class
    and its bases, and stores the result as ``_reflected_properties`` on the class.

    Properties are collected by walking the MRO in reverse (base classes first)
    so that subclass overrides replace base-class properties.  The result is a
    flat list with base-class properties appearing before subclass properties,
    and each property name appearing at most once.

    Usage::

        @reflected
        class MyObject:
            roughness = Property(0.5, doc="Surface roughness", value_range=(0.0, 1.0))
            metallic = Property(0.0, doc="Metallic factor", value_range=(0.0, 1.0))
    """
    props: list[PythonPropertyInfo] = []
    seen: set[str] = set()

    # Walk MRO in reverse so that base-class properties come first,
    # and subclass overrides replace them (same name = remove old + append new).
    for base in reversed(cls.__mro__):
        for attr_name, attr_value in base.__dict__.items():
            if isinstance(attr_value, Property):
                name = attr_value._name or attr_name
                if name in seen:
                    props = [p for p in props if p.name != name]
                seen.add(name)
                props.append(attr_value._make_info(name, cls))

    cls._reflected_properties = props
    return cls
