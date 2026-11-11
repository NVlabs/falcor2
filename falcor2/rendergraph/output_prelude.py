# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate and bind output prelude implementations for named containers."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Mapping, Sequence

import slangpy as spy
from slangpy.core.native import get_value_signature
from slangpy.reflection import SlangFunction, SlangType

from . import container_torch
from .container import Container
from .image_format import SINT_FORMATS, UINT_FORMATS, channel_count, format_to_typename

OUTPUT_VAR_PREFIX = "__g_falcor_OutputBlock"
IMPL_TYPE_PREFIX = "__falcor_OutputImpl"
OUTPUT_SPEC_ATTRIBUTE = "OutputSpec"
_INDEXED_OUTPUT_FUNCTION = "generic_data"
_FORMATS_BY_NAME = {output_format.name: output_format for output_format in spy.Format}
_INDEXED_FLOAT_FORMATS = {
    1: spy.Format.r32_float,
    2: spy.Format.rg32_float,
    3: spy.Format.rgba32_float,
    4: spy.Format.rgba32_float,
}
_WRITABLE_FLOAT_FORMATS = frozenset(
    (
        spy.Format.r16_float,
        spy.Format.rg16_float,
        spy.Format.rgba16_float,
        spy.Format.r32_float,
        spy.Format.rg32_float,
        spy.Format.rgb32_float,
        spy.Format.rgba32_float,
    )
)


class OutputOperation(Enum):
    """Operation used when writing an output value."""

    write = "write"
    increment = "increment"


@dataclass(frozen=True)
class OutputSpec:
    """Description of one reflected output target."""

    name: str
    format: spy.Format
    clear_value: spy.float4 | spy.int4 | spy.uint4
    operation: OutputOperation = OutputOperation.write


@dataclass(frozen=True)
class IndexedOutput:
    """One stable-ID output target handled by an indexed writer function."""

    id: int
    name: str
    component_count: int
    target: object
    operation: OutputOperation = OutputOperation.write


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


@dataclass(frozen=True)
class _IndexedFunction:
    name: str
    coord_name: str
    coord_type: str
    id_name: str
    id_type: str
    value_name: str
    value_type: str
    enabled_name: str
    enabled_id_name: str
    enabled_id_type: str


@dataclass(frozen=True)
class _PreparedIndexedOutput:
    desc: IndexedOutput
    field_name: str
    storage_type: str
    write_expr: str


class OutputPrelude:
    """Generate prelude code that implements a static output interface."""

    @classmethod
    def create(
        cls,
        module: spy.Module,
        interface_name: str,
        concrete_type_name: str | None = None,
        operation_overrides: Mapping[str, OutputOperation] | None = None,
    ) -> "OutputPrelude":
        """Create an output prelude generator for one static writer interface."""
        return cls(module, interface_name, concrete_type_name, operation_overrides)

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
            operation = (
                _operation_from_attribute(attribute.argument_value_string(2))
                if attribute.argument_count >= 3
                else OutputOperation.write
            )
            specs.append(
                OutputSpec(
                    name=function.name,
                    format=output_format,
                    clear_value=_clear_value_from_attribute(
                        output_format,
                        attribute.argument_value_string(1),
                    ),
                    operation=operation,
                )
            )
        return tuple(specs)

    @staticmethod
    def indexed_output_format(component_count: int) -> spy.Format:
        """Return the default float format for an indexed output width."""
        if (
            isinstance(component_count, bool)
            or not isinstance(component_count, int)
            or component_count not in _INDEXED_FLOAT_FORMATS
        ):
            raise ValueError("Indexed outputs must have one through four components.")
        return _INDEXED_FLOAT_FORMATS[component_count]

    def __init__(
        self,
        module: spy.Module,
        interface_name: str,
        concrete_type_name: str | None = None,
        operation_overrides: Mapping[str, OutputOperation] | None = None,
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
        self._output_var_prefix = OUTPUT_VAR_PREFIX + interface_name
        self._impl_type_name = IMPL_TYPE_PREFIX + interface_name
        self._signature_prefix = self._interface_name + "\n" + self._concrete_type_name + "\n"

        self._indexed_input: tuple[IndexedOutput, ...] | None = None
        self._indexed_sorted: tuple[IndexedOutput, ...] = ()
        self._indexed_signature: str = ""
        void_function_names = _void_function_names(module, interface_name)
        self._indexed_function = _reflect_indexed_function(
            module,
            interface_type,
            void_function_names,
        )
        self._output_names = [
            name
            for name in void_function_names
            if self._indexed_function is None or name != self._indexed_function.name
        ]
        self._specs = {
            spec.name: spec
            for spec in OutputPrelude.read_specs(
                module,
                interface_name,
                OUTPUT_SPEC_ATTRIBUTE,
            )
        }

        # Include per-output operations in the signature so different accumulation
        # choices generate distinct shaders.
        if operation_overrides:
            for name, operation in operation_overrides.items():
                if name not in self._specs:
                    raise ValueError(
                        f"Cannot override operation for output {name!r} without an OutputSpec."
                    )
                self._specs[name] = replace(self._specs[name], operation=operation)
            self._signature_prefix += (
                "operations="
                + repr(
                    tuple(
                        sorted((name, spec.operation.value) for name, spec in self._specs.items())
                    )
                )
                + "\n"
            )

    @property
    def specs(self) -> Mapping[str, OutputSpec]:
        """Reflected output specs by output name."""
        return self._specs

    def signature(
        self,
        targets: Mapping[str, object | None],
        indexed_outputs: Sequence[IndexedOutput] = (),
    ) -> str:
        """Return the signature for named and stable-ID output targets."""
        indexed_outputs = self._sorted_indexed_outputs(indexed_outputs)
        signature = self._signature_prefix + get_value_signature(targets)
        if self._indexed_function is None or not indexed_outputs:
            return signature
        if indexed_outputs is self._indexed_sorted:
            return signature + self._indexed_signature
        indexed_signature = tuple(
            (
                output.id,
                output.name,
                output.component_count,
                output.operation.value,
                get_value_signature(output.target),
            )
            for output in indexed_outputs
        )
        return signature + "\nindexed=" + repr(indexed_signature)

    def generate(
        self,
        targets: Mapping[str, object | None],
        indexed_outputs: Sequence[IndexedOutput] = (),
    ) -> str:
        """
        Generate a concrete implementation of a static writer interface.

        ``targets`` contains named containers; each container name is matched to a
        static interface function with the same name. The first function parameter
        is treated as the write coordinate and the second as the value to store.
        """
        output_names = self._output_names
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
                self._output_var_prefix,
                name,
                targets.get(name),
                (self._specs[name].operation if name in self._specs else OutputOperation.write),
            )
            for name in output_names
        ]
        indexed_outputs = self._prepare_indexed_outputs(indexed_outputs)
        lines = []
        # A tensor-only ParameterBlock can silently miss writes on the pinned
        # Vulkan stack. All output resources use the same global binding path.
        for output in outputs:
            if not output.enabled:
                continue
            assert output.storage_type is not None
            lines.extend(
                [
                    f"public {output.storage_type} {self._output_var_prefix}_{output.name};",
                    "",
                ]
            )

        for output in indexed_outputs:
            lines.extend(
                [
                    f"public {output.storage_type} {self._output_var_prefix}_{output.field_name};",
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

        if self._indexed_function is not None:
            function = self._indexed_function
            lines.append(
                "    public static override void "
                f"{function.name}({function.coord_type} {function.coord_name}, "
                f"{function.id_type} {function.id_name}, "
                f"{function.value_type} {function.value_name})"
            )

            lines.append("    {")
            lines.append(f"        switch ({function.id_name})")
            lines.append("        {")
            for output in indexed_outputs:
                lines.append(f"        case {output.desc.id}u:")
                lines.append(f"            {output.write_expr}")
                lines.append("            break;")
            lines.append("        default:")
            lines.append("            break;")
            lines.append("        }")
            lines.append("    }")
            lines.append("")

            lines.append(
                f"    public static bool {function.enabled_name}("
                f"{function.enabled_id_type} {function.enabled_id_name})"
            )
            lines.append("    {")
            lines.append(f"        switch ({function.enabled_id_name})")
            lines.append("        {")
            for output in indexed_outputs:
                lines.append(f"        case {output.desc.id}u:")
                lines.append("            return true;")
            lines.append("        default:")
            lines.append("            return false;")
            lines.append("        }")
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

    def bind(
        self,
        cursor: Any,
        targets: Mapping[str, object | None],
        indexed_outputs: Sequence[IndexedOutput] = (),
    ) -> None:
        """Bind enabled containers to their generated global variables."""
        indexed_outputs = self._sorted_indexed_outputs(indexed_outputs)
        for name, container in targets.items():
            if container is None:
                continue
            cursor[f"{self._output_var_prefix}_{name}"] = _output_binding(container)

        for output in indexed_outputs:
            field_name = _indexed_field_name(output.id)
            cursor[f"{self._output_var_prefix}_{field_name}"] = _output_binding(output.target)

    def clear_indexed(
        self,
        indexed_outputs: Sequence[IndexedOutput],
        *,
        command_encoder: Any | None = None,
        device: spy.Device | None = None,
    ) -> None:
        """Clear indexed output targets to zero."""
        for output in self._sorted_indexed_outputs(indexed_outputs):
            Container.clear(
                output.target,
                clear_value=spy.float4(0.0),
                command_encoder=command_encoder,
                device=device,
            )

    def _sorted_indexed_outputs(
        self,
        indexed_outputs: Sequence[IndexedOutput],
    ) -> tuple[IndexedOutput, ...]:
        if indexed_outputs is self._indexed_input:
            return self._indexed_sorted
        # Release the previous targets even when the replacement is empty or invalid.
        self._indexed_input = None
        self._indexed_sorted = ()
        self._indexed_signature = ""
        if not indexed_outputs:
            return ()
        outputs = tuple(sorted(indexed_outputs, key=lambda output: output.id))
        if outputs and self._indexed_function is None:
            raise RuntimeError(
                f"{self._interface_type.full_name} does not define an indexed output function."
            )

        ids: set[int] = set()
        names: set[str] = set()
        for output in outputs:
            if isinstance(output.id, bool) or not isinstance(output.id, int):
                raise TypeError("Indexed output IDs must be integers.")
            if output.id < 0 or output.id > 0xFFFFFFFF:
                raise ValueError("Indexed output IDs must fit in uint32.")
            field_name = _indexed_field_name(output.id)
            if field_name in self._output_names:
                raise ValueError(
                    f"Indexed output ID {output.id} conflicts with fixed output {field_name!r}."
                )
            if not isinstance(output.name, str) or not output.name:
                raise ValueError("Indexed output names must not be empty.")
            if (
                isinstance(output.component_count, bool)
                or not isinstance(output.component_count, int)
                or output.component_count not in _INDEXED_FLOAT_FORMATS
            ):
                raise ValueError("Indexed outputs must have one through four components.")
            if output.id in ids:
                raise ValueError(f"Duplicate indexed output ID {output.id}.")
            if output.name in names:
                raise ValueError(f"Duplicate indexed output name {output.name!r}.")
            ids.add(output.id)
            names.add(output.name)

            format_value = Container.format(output.target)
            if (
                format_value not in _WRITABLE_FLOAT_FORMATS
                or channel_count(format_value) < output.component_count
            ):
                raise TypeError(
                    f"Indexed output {output.name!r} target format {format_value.name} cannot preserve "
                    f"{output.component_count} float components."
                )
        if isinstance(indexed_outputs, tuple) and all(
            isinstance(output.target, spy.Texture) for output in outputs
        ):
            indexed_signature = tuple(
                (
                    output.id,
                    output.name,
                    output.component_count,
                    output.operation.value,
                    get_value_signature(output.target),
                )
                for output in outputs
            )
            self._indexed_input = indexed_outputs
            self._indexed_sorted = outputs
            self._indexed_signature = "\nindexed=" + repr(indexed_signature)
        return outputs

    def _prepare_indexed_outputs(
        self,
        indexed_outputs: Sequence[IndexedOutput],
    ) -> tuple[_PreparedIndexedOutput, ...]:
        outputs = self._sorted_indexed_outputs(indexed_outputs)
        if not outputs or self._indexed_function is None:
            return ()

        function = self._indexed_function
        prepared: list[_PreparedIndexedOutput] = []
        for output in outputs:
            target_component_count = channel_count(Container.format(output.target))
            target_value_type = _indexed_target_value_type(
                output.target,
                target_component_count,
            )
            field_name = _indexed_field_name(output.id)
            storage_type, write_expr = _storage_type_and_write_expr(
                container=output.target,
                output_var_prefix=self._output_var_prefix,
                name=field_name,
                coord_name=function.coord_name,
                coord_type=function.coord_type,
                value_name=_indexed_value_expr(
                    function.value_name,
                    output.component_count,
                    target_component_count,
                    target_value_type,
                ),
                value_type=target_value_type,
                operation=output.operation,
            )
            prepared.append(
                _PreparedIndexedOutput(
                    desc=output,
                    field_name=field_name,
                    storage_type=storage_type,
                    write_expr=write_expr,
                )
            )
        return tuple(prepared)


def _concrete_type_name(interface_name: str) -> str:
    if len(interface_name) <= 1 or interface_name[0] != "I" or not interface_name[1].isupper():
        raise ValueError(
            f"Output prelude interface name '{interface_name}' must be in the form 'IBla'."
        )
    return interface_name[1:]


def _output_binding(container: object) -> object:
    if isinstance(container, spy.Texture):
        return container
    return Container.to_tensor_binding(container)


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


def _reflect_indexed_function(
    module: spy.Module,
    interface_type: SlangType,
    void_function_names: Sequence[str],
) -> _IndexedFunction | None:
    if _INDEXED_OUTPUT_FUNCTION not in void_function_names:
        return None

    function: SlangFunction = module.layout.require_function_by_name_in_type(
        interface_type,
        _INDEXED_OUTPUT_FUNCTION,
    )
    parameter_types = tuple(parameter.type.full_name for parameter in function.parameters)
    expected_types = ("vector<uint,2>", "uint", "vector<float,4>")
    if parameter_types != expected_types:
        return None
    if not function.static or function.have_return_value:
        raise RuntimeError(
            f"{interface_type.full_name}::{function.name} must be a static void function with "
            "(uint2, uint, float4) parameters."
        )

    enabled_name = f"{function.name}_enabled"
    try:
        enabled_function: SlangFunction = module.layout.require_function_by_name_in_type(
            interface_type, enabled_name
        )
    except RuntimeError as exc:
        raise RuntimeError(f"{interface_type.full_name} must define {enabled_name}(uint).") from exc
    if (
        not enabled_function.static
        or not enabled_function.have_return_value
        or enabled_function.return_type.full_name != "bool"
        or len(enabled_function.parameters) != 1
        or enabled_function.parameters[0].type.full_name != "uint"
    ):
        raise RuntimeError(
            f"{interface_type.full_name}::{enabled_name} must be a static bool function with one uint parameter."
        )

    return _IndexedFunction(
        name=function.name,
        coord_name=function.parameters[0].name,
        coord_type=function.parameters[0].type.full_name,
        id_name=function.parameters[1].name,
        id_type=function.parameters[1].type.full_name,
        value_name=function.parameters[2].name,
        value_type=function.parameters[2].type.full_name,
        enabled_name=enabled_name,
        enabled_id_name=enabled_function.parameters[0].name,
        enabled_id_type=enabled_function.parameters[0].type.full_name,
    )


def _indexed_field_name(output_id: int) -> str:
    return f"indexed_output_{output_id}"


def _float_value_type(component_count: int) -> str:
    return "float" if component_count == 1 else f"vector<float,{component_count}>"


def _indexed_target_value_type(target: object, target_component_count: int) -> str:
    if isinstance(target, spy.Tensor):
        return target.dtype.full_name
    if container_torch.is_torch_tensor(target):
        return format_to_typename(Container.format(target))
    return _float_value_type(target_component_count)


def _indexed_value_expr(
    value_name: str,
    component_count: int,
    target_component_count: int,
    target_value_type: str,
) -> str:
    swizzle = "xyzw"[:component_count]
    value_expr = value_name if component_count == 4 else f"{value_name}.{swizzle}"
    padding = ["0.0"] * (target_component_count - component_count)
    arguments = ", ".join([value_expr, *padding])
    return f"{target_value_type}({arguments})"


def _reflect_output(
    module: spy.Module,
    interface_type: SlangType,
    output_var_prefix: str,
    name: str,
    container: object | None,
    operation: OutputOperation,
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
        output_var_prefix=output_var_prefix,
        name=function.name,
        coord_name=coord_parameter.name,
        coord_type=coord_parameter.type.full_name,
        value_name=value_parameter.name,
        value_type=value_parameter.type.full_name,
        operation=operation,
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
    output_var_prefix: str,
    name: str,
    coord_name: str,
    coord_type: str,
    value_name: str,
    value_type: str,
    operation: OutputOperation,
) -> tuple[str, str]:
    output_var = f"{output_var_prefix}_{name}"
    if isinstance(container, spy.Texture):
        if container.type != spy.TextureType.texture_2d:
            raise TypeError(f"Unsupported texture type for output prelude: {container.type!r}")
        if operation == OutputOperation.increment:
            if container.format != spy.Format.r32_uint:
                raise TypeError("Incrementing texture outputs require r32_uint format.")
            write_expr = (
                f"{output_var}[{coord_name}] = " f"{output_var}[{coord_name}] + {value_name};"
            )
            storage_type = f'[format("r32ui")] RWTexture2D<{value_type}>'
        else:
            write_expr = f"{output_var}[{coord_name}] = {value_name};"
            storage_type = f"RWTexture2D<{value_type}>"
        return storage_type, write_expr

    if container_torch.is_torch_tensor(container):
        # Torch images are CHW, so each channel is a strided scalar tensor element.
        target_format = Container.format(container)
        channel_total = channel_count(target_format)
        scalar_type = format_to_typename(target_format).removeprefix("vector<").split(",", 1)[0]
        index_prefix = _tensor_index_expr(coord_name, coord_type, 2)
        storage_type = (
            f"RWTensor<{scalar_type}, 3>"
            if operation == OutputOperation.increment
            else f"WTensor<{scalar_type}, 3>"
        )
        stores = []
        for channel in range(channel_total):
            index_expr = f"{index_prefix}, {channel}"
            component = (
                f"({value_name}).{'xyzw'[channel]}"
                if value_type.startswith("vector<")
                else value_name
            )
            stored_value = f"{scalar_type}({component})"
            if operation == OutputOperation.increment:
                stored_value = f"{output_var}.load({index_expr}) + {stored_value}"
            stores.append(f"{output_var}.store({index_expr}, {stored_value});")
        return storage_type, " ".join(stores)

    if isinstance(container, spy.Tensor):
        dims = container.shape
        dimension_count = len(dims)
        index_expr = _tensor_index_expr(coord_name, coord_type, dimension_count)
        # The output parameter may be float while the requested tensor stores half.
        target_value_type = container.dtype.full_name
        stored_value = (
            value_name if target_value_type == value_type else f"{target_value_type}({value_name})"
        )
        if operation == OutputOperation.increment:
            storage_type = f"RWTensor<{target_value_type}, {dimension_count}>"
            write_expr = (
                f"{output_var}.store({index_expr}, "
                f"{output_var}.load({index_expr}) + {stored_value});"
            )
        else:
            storage_type = f"WTensor<{target_value_type}, {dimension_count}>"
            write_expr = f"{output_var}.store({index_expr}, {stored_value});"
        return storage_type, write_expr

    raise TypeError(f"Unsupported container type for output prelude: {type(container)!r}")


def _find_type_decl(module: spy.Module, type_name: str) -> spy.DeclReflection:
    for source_module in _source_modules(module):
        try:
            module_decl = source_module.module_decl
        except RuntimeError:
            continue

        # Slang's __include places declarations under an include scope rather
        # than directly under the module declaration.
        pending = deque(module_decl.children)
        while pending:
            child_decl = pending.popleft()
            try:
                type_reflection = child_decl.as_type()
            except RuntimeError:
                type_reflection = None
            if type_reflection is not None and type_reflection.name == type_name:
                return child_decl
            pending.extend(child_decl.children)

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


def _operation_from_attribute(value: str) -> OutputOperation:
    try:
        return OutputOperation(value)
    except ValueError as exc:
        raise ValueError(f"Unknown output operation '{value}'.") from exc


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
