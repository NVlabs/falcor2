# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Decorator-based Python property reflection system.

Provides ``reflected_property`` descriptors and the ``@reflected`` class decorator
to expose Python object properties to the C++ property system.

Architecture
------------
1. Users declare ``reflected_property`` descriptors on their classes.
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

import types
from collections.abc import Iterable
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Optional,
    Union,
    get_args,
    get_origin,
    get_type_hints,
    overload,
)

if TYPE_CHECKING:
    from falcor2.reflection import UIFlags

# Storage key prefix for reflected properties ("rp" = reflected property).
# Prevents collision with user-defined attributes on instances.
_REFLECTED_PROPERTY_PREFIX = "_rp_"

# Distinguishes the configured-decorator form from a stored property whose
# explicit initial value is None.
_MISSING = object()

# ---------------------------------------------------------------------------
# PythonPropertyInfo  (consumed from C++ via PythonPropertyDescriptor)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PythonPropertyInfo:
    """Frozen description of a single property on a class."""

    name: str
    value_type: Optional[type] = None
    object_factories: tuple[PythonObjectFactoryInfo | None, ...] = ()
    getter: Optional[Callable[[Any], Any]] = None
    setter: Optional[Callable[[Any, Any], None]] = None
    has_default_value: bool = False
    default_value: Any = None
    doc: Optional[str] = None
    value_range: Optional[tuple[float, float]] = None
    ui_flags: Optional["UIFlags"] = None
    ui_label: Optional[str] = None
    ui_group: Optional[str] = None
    ui_drag_speed: Optional[float] = None
    ui_enable_if: Optional[Callable[[Any], bool]] = None
    on_change: Optional[Callable[[Any], None]] = None


@dataclass(frozen=True)
class PythonObjectFactoryInfo:
    """Frozen description of one concrete reflected-object factory type."""

    object_type: type
    factory: Callable[[Any], Any]
    label: str


def object_factory(
    object_type: type,
    factory: Optional[Callable[[Any], Any]] = None,
    label: Optional[str] = None,
) -> PythonObjectFactoryInfo:
    """Describe a concrete type for an object-valued reflected property.

    If ``factory`` is omitted, ``object_type`` is default-constructed. Custom
    factories receive the Python object that owns the reflected property.
    """
    if not isinstance(object_type, type):
        raise TypeError("object factory type must be a type")

    if factory is None:

        def default_factory(_owner: Any) -> Any:
            return object_type()

        factory = default_factory
    elif not callable(factory):
        raise TypeError("object factory must be callable")

    if label is not None and not isinstance(label, str):
        raise TypeError("object factory label must be a string")
    resolved_label = label if label is not None else object_type.__name__
    if not resolved_label:
        raise ValueError("object factory label must not be empty")
    return PythonObjectFactoryInfo(object_type, factory, resolved_label)


def _normalize_object_factories(
    value_type: type,
    entries: Optional[Iterable[type | PythonObjectFactoryInfo | None]],
) -> tuple[PythonObjectFactoryInfo | None, ...]:
    """Validate and normalize the factory types for an object-valued property."""
    if entries is None:
        return ()

    entries = tuple(entries)
    if not entries:
        return ()
    factories: list[PythonObjectFactoryInfo | None] = []
    seen_object_types: set[type] = set()
    seen_none = False
    for entry in entries:
        if entry is None:
            if seen_none:
                raise ValueError("duplicate None object factory")
            seen_none = True
            factories.append(None)
            continue

        info = object_factory(entry) if isinstance(entry, type) else entry
        if not isinstance(info, PythonObjectFactoryInfo):
            raise TypeError(
                "object_factories entries must be None, types, or values returned by object_factory()"
            )
        if not issubclass(info.object_type, value_type):
            raise TypeError(
                f"object factory type {info.object_type.__name__} must derive from "
                f"{value_type.__name__}"
            )
        if info.object_type in seen_object_types:
            raise ValueError(f"duplicate object factory type {info.object_type.__name__}")
        seen_object_types.add(info.object_type)
        factories.append(info)

    return tuple(factories)


def _annotation_value_type(annotation: Any) -> Optional[type]:
    """Return the concrete property type represented by an annotation."""
    if isinstance(annotation, type):
        return annotation

    if get_origin(annotation) in (Union, types.UnionType):
        members = tuple(member for member in get_args(annotation) if member is not type(None))
        if len(members) == 1 and isinstance(members[0], type):
            return members[0]

    return None


# ---------------------------------------------------------------------------
# Reflected property descriptor
# ---------------------------------------------------------------------------


class reflected_property(property):
    """Descriptor that declares a property on a Python class.

    Simple (stored-value) form::

        class Foo:
            roughness = reflected_property(0.5, doc="Roughness", value_range=(0.0, 1.0))

    Getter/setter (computed) form::

        class Foo:
            @reflected_property(doc="Computed value")
            def my_prop(self) -> float:
                return self._my_prop

            @my_prop.setter
            def my_prop(self, value: float) -> None:
                self._my_prop = value

    Object-valued form::

        @reflected_property(
            object_factories=(None, ConcreteA, ConcreteB),
        )
        def child(self) -> Base | None:
            return self._child

    Object-valued properties are non-null by default. Include ``None`` in
    ``object_factories`` to expose and accept an empty value.
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

    def __init__(self, first_arg: Any = _MISSING, /, **kwargs: Any) -> None:
        super().__init__()

        from falcor2.reflection import UIFlags

        if "enum_type" in kwargs:
            raise TypeError(
                "enum_type is no longer supported; use value_type or a return annotation"
            )
        if "object_type" in kwargs:
            raise TypeError(
                "object_type is no longer supported; use value_type or a return annotation"
            )

        # Metadata common to both forms.
        object_factories = kwargs.get("object_factories")
        self._object_factory_entries = None if object_factories is None else tuple(object_factories)
        self._doc: Optional[str] = kwargs.get("doc", None)
        self._value_range: Optional[tuple[float, float]] = kwargs.get("value_range", None)
        self._ui_flags: UIFlags = kwargs.get("ui_flags", UIFlags.none)
        self._ui_label: Optional[str] = kwargs.get("ui_label", None)
        self._ui_group: Optional[str] = kwargs.get("ui_group", None)
        self._ui_drag_speed: Optional[float] = kwargs.get("ui_drag_speed", None)
        self._ui_enable_if: Optional[Callable[..., bool]] = kwargs.get("ui_enable_if", None)
        self._on_change: Optional[Callable[..., None]] = kwargs.get("on_change", None)

        # Reflected property name (filled by __set_name__).
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
        #   first_arg omitted  -> pending decorator (keyword-only form)
        #   first_arg callable -> getter form
        #   otherwise          -> stored-value form
        if first_arg is _MISSING:
            self._pending_decorator = True
        elif callable(first_arg) and not isinstance(first_arg, type):
            self._pending_decorator = False
            self._fget = first_arg
            self._doc = self._doc or first_arg.__doc__
        else:
            self._pending_decorator = False
            self._is_stored = True
            self._default_value = first_arg

    # Allow ``@reflected_property(doc=...)`` to be used as a decorator.
    def __call__(self, fget: Callable[..., Any]) -> reflected_property:
        if not self._pending_decorator:
            raise TypeError(
                "Cannot use reflected_property as a decorator in this form. "
                "Use @reflected_property or @reflected_property(doc=...) for decorator form, "
                "or reflected_property(value) for stored-value form."
            )
        self._pending_decorator = False
        self._fget = fget
        self._doc = self._doc or fget.__doc__
        return self

    # ------------------------------------------------------------------
    # Setter decorator (mirrors @property)
    # ------------------------------------------------------------------

    def setter(self, fset: Callable[..., None]) -> reflected_property:
        """Register a setter function. Usage: ``@my_prop.setter``."""
        if self._is_stored:
            raise TypeError("Cannot add setter to a stored-value reflected_property")
        self._fset = fset
        return self

    def getter(self, fget: Callable[..., Any]) -> reflected_property:
        """Reject built-in-style getter replacement, which would lose reflection metadata."""
        raise TypeError(
            "reflected_property does not support getter replacement. "
            "Declare a new @reflected_property instead."
        )

    def deleter(self, fdel: Callable[..., None]) -> reflected_property:
        """Reject deletion, which has no reflected-property semantics."""
        raise TypeError("reflected_property does not support deletion.")

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
        raise AttributeError(f"Reflected property '{self._name}' has no getter")

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
        raise AttributeError(f"Reflected property '{self._name}' is read-only")

    def __delete__(self, instance: Any) -> None:
        """Reject deleting a reflected property value."""
        raise AttributeError(f"Reflected property '{self._name}' cannot be deleted")

    # ------------------------------------------------------------------
    # Info builder  (used by @reflected)
    # ------------------------------------------------------------------

    def _make_info(self, name: str, owner: type) -> PythonPropertyInfo:
        """Build a :class:`PythonPropertyInfo` for this descriptor."""
        # Resolve value type.
        type_inference_error: Exception | None = None
        value_type = self._value_type
        if value_type is not None and not isinstance(value_type, type):
            raise TypeError(
                f"reflected_property '{owner.__name__}.{name}' value_type must be a type"
            )
        if value_type is None and self._is_stored and self._default_value is not None:
            value_type = type(self._default_value)
        if value_type is None and self._fget is not None:
            annotations = getattr(self._fget, "__annotations__", {})
            if "return" in annotations:
                localns = dict(vars(owner))
                localns[owner.__name__] = owner
                try:
                    return_type = get_type_hints(
                        self._fget,
                        globalns=self._fget.__globals__,
                        localns=localns,
                    )["return"]
                except (NameError, TypeError) as exc:
                    type_inference_error = exc
                else:
                    value_type = _annotation_value_type(return_type)
        if value_type is None:
            raise TypeError(
                f"reflected_property '{owner.__name__}.{name}' requires value_type= "
                "when its type cannot be inferred from an initial value or return annotation"
            ) from type_inference_error

        object_factories = _normalize_object_factories(value_type, self._object_factory_entries)

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
            object_factories=object_factories,
            getter=getter,
            setter=setter,
            has_default_value=self._is_stored,
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
    """Class decorator that collects ``reflected_property`` descriptors from a class
    and its bases, and stores the result as ``_reflected_properties`` on the class.

    Properties are collected by walking the MRO in reverse (base classes first)
    so that subclass overrides replace base-class properties.  The result is a
    flat list with base-class properties appearing before subclass properties,
    and each property name appearing at most once.

    Usage::

        @reflected
        class MyObject:
            roughness = reflected_property(0.5, doc="Surface roughness", value_range=(0.0, 1.0))
            metallic = reflected_property(0.0, doc="Metallic factor", value_range=(0.0, 1.0))
    """
    props: list[PythonPropertyInfo] = []
    seen: set[str] = set()

    # Walk MRO in reverse so that base-class properties come first,
    # and subclass overrides replace them (same name = remove old + append new).
    for base in reversed(cls.__mro__):
        for attr_name, attr_value in base.__dict__.items():
            if isinstance(attr_value, reflected_property):
                name = attr_value._name or attr_name
                if name in seen:
                    props = [p for p in props if p.name != name]
                seen.add(name)
                props.append(attr_value._make_info(name, cls))

    cls._reflected_properties = props
    return cls
