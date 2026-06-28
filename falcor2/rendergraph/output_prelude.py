# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate and bind output prelude implementations for named containers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import slangpy as spy
from slangpy.core.native import get_value_signature
from slangpy.reflection import SlangFunction, SlangType

from . import container_torch
from .container import Container
from .image_format import SINT_FORMATS, UINT_FORMATS

BLOCK_TYPE_PREFIX = "__falcor_OutputBlock"
BLOCK_VAR_PREFIX = "__g_falcor_OutputBlock"
IMPL_TYPE_PREFIX = "__falcor_OutputImpl"
OUTPUT_SPEC_ATTRIBUTE = "OutputSpec"
_FORMATS_BY_NAME = {output_format.name: output_format for output_format in spy.Format}


@dataclass(frozen=True)
class OutputSpec:
    """Description of one reflected output target."""

    name: str
    format: spy.Format
    clear_value: spy.float4 | spy.int4 | spy.uint4


@dataclass(frozen=True)
class _ReflectedOutput:
    name: str
    coord_name: str
    coord_type: str
    value_name: str
    value_type: str
    enabled: bool
    storage_type: str | None = None
    write_expr: str | None = None


class OutputPrelude:
    """Generate prelude code that implements a static output interface."""

    @classmethod
    def create(
        cls,
        module: spy.Module,
        interface_name: str,
        concrete_type_name: str | None = None,
    ) -> "OutputPrelude":
        """Create an output prelude generator for one static writer interface."""
        return cls(module, interface_name, concrete_type_name)

    @staticmethod
    def read_specs(
        module: spy.Module,
        interface_name: str,
        attribute_name: str = OUTPUT_SPEC_ATTRIBUTE,
    ) -> tuple[OutputSpec, ...]:
        """Reflect output specs from attributes on an output interface's functions."""
        interface_type = module.layout.find_type_by_name(interface_name)
        if interface_type is None:
            raise RuntimeError(f"Unable to find {interface_name} in module.")
        if interface_type.type_reflection.kind != spy.TypeReflection.Kind.interface:
            raise RuntimeError(f"{interface_name} must be a Slang interface type.")

        specs: list[OutputSpec] = []
        for function_name in _void_function_names(module, interface_name):
            function: SlangFunction = module.layout.require_function_by_name_in_type(
                interface_type, function_name
            )
            attribute = function.reflection.find_user_attribute_by_name(attribute_name)
            if attribute is None:
                continue
            if attribute.argument_count < 2:
                raise RuntimeError(
                    f"{interface_name}::{function.name} {attribute_name} attribute "
                    "must provide format and clear value arguments."
                )
            output_format = _format_from_attribute(attribute.argument_value_string(0))
            specs.append(
                OutputSpec(
                    name=function.name,
                    format=output_format,
                    clear_value=_clear_value_from_attribute(
                        output_format,
                        attribute.argument_value_string(1),
                    ),
                )
            )
        return tuple(specs)

    def __init__(
        self,
        module: spy.Module,
        interface_name: str,
        concrete_type_name: str | None = None,
    ) -> None:
        """
        Create an output prelude generator for one static writer interface.

        ``interface_name`` names the static output interface to implement.
        ``concrete_type_name`` names the exported concrete type. When omitted,
        the interface must use the ``IBla`` convention and the concrete type is
        inferred as ``Bla``.
        """
        super().__init__()

        interface_type = module.layout.find_type_by_name(interface_name)
        if interface_type is None:
            raise RuntimeError(f"Unable to find {interface_name} in module.")
        if interface_type.type_reflection.kind != spy.TypeReflection.Kind.interface:
            raise RuntimeError(f"{interface_name} must be a Slang interface type.")

        self._module = module
        self._interface_name = interface_name
        self._interface_type = interface_type
        if concrete_type_name is None:
            self._concrete_type_name = _concrete_type_name(interface_name)
        elif concrete_type_name:
            self._concrete_type_name = concrete_type_name
        else:
            raise ValueError("Output prelude concrete type name must not be empty.")
        self._block_type_name = BLOCK_TYPE_PREFIX + interface_name
        self._block_var_name = BLOCK_VAR_PREFIX + interface_name
        self._impl_type_name = IMPL_TYPE_PREFIX + interface_name
        self._signature_prefix = self._interface_name + "\n" + self._concrete_type_name + "\n"
        self._specs = {
            spec.name: spec
            for spec in OutputPrelude.read_specs(
                module,
                interface_name,
                OUTPUT_SPEC_ATTRIBUTE,
            )
        }

    @property
    def specs(self) -> Mapping[str, OutputSpec]:
        """Reflected output specs by output name."""
        return self._specs

    def signature(self, targets: Mapping[str, object | None]) -> str:
        """Return SlangPy's value signature for the output target dictionary."""
        return self._signature_prefix + get_value_signature(targets)

    def generate(self, targets: Mapping[str, object | None]) -> str:
        """
        Generate a concrete implementation of a static writer interface.

        ``targets`` contains named containers; each container name is matched to a
        static interface function with the same name. The first function parameter
        is treated as the write coordinate and the second as the value to store.
        """
        output_names = _void_function_names(self._module, self._interface_name)
        output_name_set = set(output_names)
        unknown_names = [
            name
            for name, container in targets.items()
            if container is not None and name not in output_name_set
        ]
        if unknown_names:
            raise RuntimeError(
                f"{self._interface_type.full_name} does not define expected output function "
                f"'{unknown_names[0]}'."
            )

        # Reflect every output function in the interface. Outputs without supplied
        # targets become no-op implementations that report disabled.
        outputs = [
            _reflect_output(
                self._module,
                self._interface_type,
                self._block_var_name,
                name,
                targets.get(name),
            )
            for name in output_names
        ]
        enabled_outputs = [output for output in outputs if output.enabled]

        lines = []

        # Generate the storage block containing one writable resource per enabled output.
        lines.extend(
            [
                f"public struct {self._block_type_name}",
                "{",
            ]
        )
        for output in enabled_outputs:
            assert output.storage_type is not None
            lines.append(f"    public {output.storage_type} {output.name};")
        lines.extend(["}", ""])

        if enabled_outputs:
            lines.extend(
                [
                    f"public ParameterBlock<{self._block_type_name}> {self._block_var_name};",
                    "",
                ]
            )

        # Generate the concrete interface implementation. Disabled outputs compile
        # to no-op functions so the interface does not need default implementations.
        lines.extend(
            [
                f"public struct {self._impl_type_name} : {self._interface_name}",
                "{",
            ]
        )
        for output in outputs:
            lines.append(
                "    public static override void "
                f"{output.name}({output.coord_type} {output.coord_name}, "
                f"{output.value_type} {output.value_name})"
            )
            lines.append("    {")
            if output.enabled:
                assert output.write_expr is not None
                lines.append(f"        {output.write_expr}")
            lines.append("    }")
            lines.append("")

        for output in outputs:
            enabled_literal = "true" if output.enabled else "false"
            lines.append(f"    public static bool {output.name}_enabled()")
            lines.append("    {")
            lines.append(f"        return {enabled_literal};")
            lines.append("    }")
            lines.append("")

        # Export the implementation for the extern interface type.
        lines.extend(
            [
                "}",
                "",
                f"export struct {self._concrete_type_name} : {self._interface_name} = {self._impl_type_name};",
            ]
        )

        # Export link-time flags consumed by shader code to specialize away disabled outputs.
        for output in outputs:
            enabled_literal = "true" if output.enabled else "false"
            lines.append(
                f"export static const bool {self._concrete_type_name}_{output.name} = "
                f"{enabled_literal};"
            )
        return "\n".join(lines) + "\n"

    def bind(self, cursor: Any, targets: Mapping[str, object | None]) -> None:
        """Bind the named containers in ``targets`` to the generated parameter block."""
        if not any(container is not None for container in targets.values()):
            return

        block = cursor[self._block_var_name]
        for name, container in targets.items():
            if container is None:
                continue
            block[name] = Container.to_render_layout(container)


def _concrete_type_name(interface_name: str) -> str:
    if len(interface_name) <= 1 or interface_name[0] != "I" or not interface_name[1].isupper():
        raise ValueError(
            f"Output prelude interface name '{interface_name}' must be in the form 'IBla'."
        )
    return interface_name[1:]


def _function_returns_void(function: spy.FunctionReflection) -> bool:
    return_type = function.return_type
    if return_type is None:
        return True
    return return_type.full_name == "void" or (
        return_type.kind == spy.TypeReflection.Kind.scalar
        and return_type.scalar_type == spy.TypeReflection.ScalarType.void
    )


def _void_function_names(module: spy.Module, type_name: str) -> list[str]:
    type_decl = _find_type_decl(module, type_name)
    return [
        function.name
        for function in (
            function_decl.as_function()
            for function_decl in type_decl.children_of_kind(spy.DeclReflection.Kind.func)
        )
        if _function_returns_void(function)
    ]


def _reflect_output(
    module: spy.Module,
    interface_type: SlangType,
    block_var_name: str,
    name: str,
    container: object | None,
) -> _ReflectedOutput:
    function: SlangFunction = module.layout.require_function_by_name_in_type(interface_type, name)
    if not function.static or function.have_return_value or len(function.parameters) != 2:
        raise RuntimeError(
            f"{interface_type.full_name}::{name} must be a static void function with two parameters."
        )

    coord_parameter = function.parameters[0]
    value_parameter = function.parameters[1]
    if container is None:
        return _ReflectedOutput(
            name=function.name,
            coord_name=coord_parameter.name,
            coord_type=coord_parameter.type.full_name,
            value_name=value_parameter.name,
            value_type=value_parameter.type.full_name,
            enabled=False,
        )

    storage_type, write_expr = _storage_type_and_write_expr(
        container=container,
        block_var_name=block_var_name,
        name=function.name,
        coord_name=coord_parameter.name,
        coord_type=coord_parameter.type.full_name,
        value_name=value_parameter.name,
        value_type=value_parameter.type.full_name,
    )
    return _ReflectedOutput(
        name=function.name,
        coord_name=coord_parameter.name,
        coord_type=coord_parameter.type.full_name,
        value_name=value_parameter.name,
        value_type=value_parameter.type.full_name,
        enabled=True,
        storage_type=storage_type,
        write_expr=write_expr,
    )


def _storage_type_and_write_expr(
    *,
    container: object,
    block_var_name: str,
    name: str,
    coord_name: str,
    coord_type: str,
    value_name: str,
    value_type: str,
) -> tuple[str, str]:
    block_field = f"{block_var_name}.{name}"
    if isinstance(container, spy.Texture):
        if container.type != spy.TextureType.texture_2d:
            raise TypeError(f"Unsupported texture type for output prelude: {container.type!r}")
        return (
            f"RWTexture2D<{value_type}>",
            f"{block_field}[{coord_name}] = {value_name};",
        )

    if isinstance(container, spy.Tensor) or container_torch.is_torch_tensor(container):
        _, dims = Container.format_and_dims(container)
        dimension_count = len(dims)
        return (
            f"WTensor<{value_type}, {dimension_count}>",
            f"{block_field}.store({_tensor_index_expr(coord_name, coord_type, dimension_count)}, "
            f"{value_name});",
        )

    raise TypeError(f"Unsupported container type for output prelude: {type(container)!r}")


def _find_type_decl(module: spy.Module, type_name: str) -> spy.DeclReflection:
    for source_module in _source_modules(module):
        try:
            module_decl = source_module.module_decl
        except RuntimeError:
            continue

        for child_decl in module_decl.children:
            try:
                type_reflection = child_decl.as_type()
            except RuntimeError:
                continue
            if type_reflection is not None and type_reflection.name == type_name:
                return child_decl

    raise RuntimeError(f"Unable to find {type_name} declaration in module sources.")


def _source_modules(module: spy.Module) -> tuple[spy.SlangModule, ...]:
    source_modules: list[spy.SlangModule] = []

    def visit(source_module: spy.SlangModule) -> None:
        if source_module.is_composed:
            for child_module in source_module.source_modules:
                visit(child_module)
            return
        source_modules.append(source_module)

    visit(module.device_module)
    return tuple(source_modules)


def _format_from_attribute(value: str) -> spy.Format:
    try:
        return _FORMATS_BY_NAME[value]
    except KeyError as exc:
        raise ValueError(f"Unknown output format '{value}'.") from exc


def _clear_value_from_attribute(
    format: spy.Format, value: str
) -> spy.float4 | spy.int4 | spy.uint4:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if len(parts) > 4:
        raise ValueError(f"Output clear value '{value}' must have at most four components.")

    if _is_uint_format(format):
        values = [int(part, 0) for part in parts]
        values.extend([0] * (4 - len(values)))
        return spy.uint4(*values[:4])
    if _is_sint_format(format):
        values = [int(part, 0) for part in parts]
        values.extend([0] * (4 - len(values)))
        return spy.int4(*values[:4])

    values = [float(part) for part in parts]
    values.extend([0.0] * (4 - len(values)))
    return spy.float4(*values[:4])


def _is_uint_format(format: spy.Format) -> bool:
    return format in UINT_FORMATS


def _is_sint_format(format: spy.Format) -> bool:
    return format in SINT_FORMATS


def _tensor_index_expr(coord_name: str, coord_type: str, dimension_count: int) -> str:
    if dimension_count == 1:
        if coord_type.startswith("vector<"):
            return f"int({coord_name}.x)"
        return f"int({coord_name})"
    if dimension_count == 2 and coord_type in ("vector<uint,2>", "vector<int,2>"):
        return f"int({coord_name}.y), int({coord_name}.x)"
    raise ValueError(
        f"Unsupported tensor coordinate type {coord_type!r} for {dimension_count}D tensor."
    )
