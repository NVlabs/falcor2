# SPDX-License-Identifier: Apache-2.0

from typing import Any, Callable, Generic, Mapping, Optional, TypeVar

import slangpy as spy

ValueT = TypeVar("ValueT")


class Property(Generic[ValueT]):
    def __init__(
        self,
        value: Optional[ValueT] = None,
        fget: Optional[Callable[[Any], ValueT]] = None,
        fset: Optional[Callable[[Any, ValueT], None]] = None,
        fdel: Optional[Callable[[Any], None]] = None,
        notify: Optional[Callable[[Any], None]] = None,
        label: Optional[str] = None,
        doc: Optional[str] = None,
        min: Optional[ValueT] = None,
        max: Optional[ValueT] = None,
        enable_if: Optional[str] = None,
    ):
        super().__init__()
        if value and (fget or fset or fdel):
            raise ValueError("Cannot specify both value and getter/setter/deleter")
        self.value = value
        self.fget = fget
        self.fset = fset
        self.fdel = fdel
        self.notify = notify
        self.label = label
        self.__doc__ = doc
        self.min = min
        self.max = max
        self.enable_if = enable_if

    def __get__(self, instance: Any, owner: Any) -> ValueT:
        if instance is None:
            return None  # type: ignore
        if self.fget is None:
            raise AttributeError("Property is not readable")
        return self.fget(instance)

    def __set__(self, instance: Any, value: ValueT):
        if self.fset is None:
            raise AttributeError("Property is not writable")
        if self.min is not None and value < self.min:  # type: ignore
            raise ValueError(f"Value must be >= {self.min}")
        if self.max is not None and value > self.max:  # type: ignore
            raise ValueError(f"Value must be <= {self.max}")
        self.fset(instance, value)
        if self.notify:
            self.notify(instance)
        if instance.__listeners__:
            for listener in instance.__listeners__:
                listener(self, value)

    def __delete__(self, instance: Any):
        if self.fdel is None:
            raise AttributeError("Property is not deletable")
        self.fdel(instance)


class PropertyObject:
    def __init__(self):
        super().__init__()
        self.__properties__: Mapping[str, Any] = {}
        self.__property_values__: Mapping[Property[Any], Any] = {}
        for k, v in self.__class__.__dict__.items():
            if isinstance(v, Property):
                prop = v
                self.__properties__[k] = prop
                # Setup values for properties without getter/setters
                if prop.value is not None:
                    prop.fget = lambda self, prop=prop: self.__property_values__[prop]
                    prop.fset = lambda self, value, prop=prop: self.__property_values__.__setitem__(
                        prop, value
                    )
                    prop.fdel = lambda self, prop=prop: self.__property_values__.__delitem__(prop)
                    self.__property_values__[prop] = prop.value

        self.__listeners__ = []

    def add_properties_listener(self, listener: Callable[[Any, Any], None]):
        self.__listeners__.append(listener)

    def remove_properties_listener(self, listener: Callable[[Any, Any], None]):
        self.__listeners__.remove(listener)

    def dump_properties(self):
        d = dict()
        for k in self.__properties__.keys():
            d[k] = getattr(self, k)
        return d

    def load_properties(self, d: Mapping[str, Any]):
        for k, v in d.items():
            setattr(self, k, v)


class PropertiesGroup:
    def __init__(self, parent: spy.ui.Window, object: PropertyObject, label: str):
        super().__init__()
        self.object = object
        self.group = spy.ui.Group(parent, label)
        self.widgets: dict[Property, spy.ui.Widget] = {}  # type: ignore

        for name, property in object.__properties__.items():
            value = getattr(object, name)
            if isinstance(value, bool):
                self.widgets[property] = spy.ui.CheckBox(
                    parent=self.group,
                    label=property.label,
                    value=value,
                    callback=lambda value, name=name: setattr(object, name, value),
                )
            elif isinstance(value, int):
                self.widgets[property] = spy.ui.DragInt(
                    parent=self.group,
                    label=property.label,
                    value=value,
                    min=property.min or 0,
                    max=property.max or 0,
                    callback=lambda value, name=name: setattr(object, name, value),
                )

        self.update_active()
        self.object.add_properties_listener(self.on_property_change)

    def on_property_change(self, property: Property[Any], value: Any):
        if property in self.widgets:
            widget = self.widgets[property]
            if isinstance(widget, spy.ui.CheckBox):
                widget.value = value
            if isinstance(widget, spy.ui.DragInt):
                widget.value = value
        self.update_active()

    def update_active(self):
        for property, widget in self.widgets.items():
            if property.enable_if and property.enable_if[0] == "@":
                enabled = eval(property.enable_if[1:], {"self": self.object})
                widget.enabled = enabled


class PropertiesWindow:
    def __init__(self, screen: spy.ui.Screen, title: str = "Properties"):
        super().__init__()
        self.screen = screen
        self.window = spy.ui.Window(parent=self.screen, title=title, size=spy.float2(500, 300))
        self.groups: dict[PropertyObject, PropertiesGroup] = {}

    def add_properties(self, object: PropertyObject, label: Optional[str] = None):
        if label is None:
            label = object.__class__.__name__
        self.groups[object] = PropertiesGroup(self.window, object, label)

    def remove_properties(self, object: PropertyObject):
        self.groups.pop(object)
