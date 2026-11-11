# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate inline stubs for OpenUSD's Boost.Python modules.

OpenUSD's generated ``__DOC.py`` files add human-readable signatures to
objects that otherwise do not support ``inspect.signature``. This generator
combines those signatures with runtime reflection and intentionally falls back
to ``Any`` when a C++ type cannot be mapped safely.
"""

from __future__ import annotations

import argparse
import ast
import builtins
from dataclasses import dataclass
import importlib
import inspect
import keyword
import os
from pathlib import Path
import pkgutil
import re
import sys
import types
from typing import Any, Iterable, Sequence


_SIGNATURE_RE = re.compile(
    r"^(?:(\*\*(?P<kind>class|static)method\*\*)\s+)?"
    r"(?P<name>[A-Za-z_]\w*)\((?P<parameters>.*)\)"
    r"(?:\s*->\s*(?P<return>.+?))?\s*$"
)
_PARAMETER_TYPE_RE = re.compile(r"^(?P<name>[A-Za-z_]\w*)\s*:\s*(?P<type>.+?)\s*$")
_IDENTIFIER_RE = re.compile(r"\b[A-Za-z_]\w*\b")
_DOCSTRING_SEPARATOR_RE = re.compile(r"^-{3,}$")
_PUBLIC_DUNDERS = {
    "__bool__",
    "__call__",
    "__contains__",
    "__enter__",
    "__eq__",
    "__exit__",
    "__getitem__",
    "__hash__",
    "__init__",
    "__iter__",
    "__len__",
    "__lt__",
    "__ne__",
    "__next__",
    "__repr__",
    "__setitem__",
    "__str__",
}
_BUILTIN_TYPES = {
    "Any",
    "None",
    "bool",
    "bytes",
    "complex",
    "dict",
    "float",
    "int",
    "list",
    "object",
    "set",
    "str",
    "tuple",
    "type",
}
_TYPING_TYPES = {
    "Callable",
    "ClassVar",
    "Collection",
    "Iterable",
    "Iterator",
    "Literal",
    "Mapping",
    "MutableMapping",
    "Optional",
    "Sequence",
    "Union",
}
_TYPING_NAMES = _TYPING_TYPES | {"Any", "ClassVar", "overload"}
_GENERIC_DEFAULTS = {
    "Callable": "Callable[..., Any]",
    "Collection": "Collection[Any]",
    "Iterable": "Iterable[Any]",
    "Iterator": "Iterator[Any]",
    "Mapping": "Mapping[Any, Any]",
    "MutableMapping": "MutableMapping[Any, Any]",
    "Sequence": "Sequence[Any]",
}
_TYPE_REPLACEMENTS = {
    "TfToken": "str",
    "VtValue": "Any",
    "void": "None",
    "size_t": "int",
    "uint32_t": "int",
    "uint64_t": "int",
    "int32_t": "int",
    "int64_t": "int",
}


@dataclass(frozen=True)
class Parameter:
    name: str
    annotation: str
    has_default: bool


@dataclass(frozen=True)
class Signature:
    kind: str | None
    parameters: tuple[Parameter, ...]
    return_type: str
    documentation: str


@dataclass
class Coverage:
    modules: int = 0
    classes: int = 0
    callables: int = 0
    documented_callables: int = 0
    fallback_callables: int = 0
    unknown_types: int = 0


def _split_parameters(parameters: str) -> list[str]:
    result: list[str] = []
    start = 0
    depth = 0
    for index, char in enumerate(parameters):
        if char in "[({":
            depth += 1
        elif char in "])}":
            depth = max(0, depth - 1)
        elif char == "," and depth == 0:
            result.append(parameters[start:index].strip())
            start = index + 1
    tail = parameters[start:].strip()
    if tail:
        result.append(tail)
    return result


def _safe_parameter_name(name: str, index: int) -> str:
    name = name.strip()
    if not name or not name.isidentifier():
        return f"arg{index}"
    if keyword.iskeyword(name):
        return f"{name}_"
    return name


def parse_docstring_signatures(docstring: str | None, expected_name: str) -> list[Signature]:
    """Parse the signature blocks emitted by OpenUSD's Doxygen converter."""
    if not docstring:
        return []

    lines = inspect.cleandoc(docstring).splitlines()
    signature_lines: list[tuple[int, re.Match[str]]] = []
    for index, line in enumerate(lines):
        match = _SIGNATURE_RE.match(line.strip())
        if match is not None and match.group("name") == expected_name:
            signature_lines.append((index, match))

    result: list[Signature] = []
    for signature_index, (index, match) in enumerate(signature_lines):
        parameter_types: dict[str, str] = {}
        description_index = index + 1
        while description_index < len(lines) and not lines[description_index].strip():
            description_index += 1
        while description_index < len(lines):
            type_match = _PARAMETER_TYPE_RE.match(lines[description_index].strip())
            if type_match is None:
                break
            parameter_types[type_match.group("name")] = type_match.group("type")
            description_index += 1

        next_signature_index = (
            signature_lines[signature_index + 1][0]
            if signature_index + 1 < len(signature_lines)
            else len(lines)
        )
        documentation_lines = [
            line
            for line in lines[description_index:next_signature_index]
            if not _DOCSTRING_SEPARATOR_RE.fullmatch(line.strip())
        ]
        documentation = inspect.cleandoc("\n".join(documentation_lines)).strip()

        parsed_parameters: list[Parameter] = []
        for parameter_index, raw_parameter in enumerate(
            _split_parameters(match.group("parameters")), start=1
        ):
            if raw_parameter in {"/", "*"}:
                continue
            raw_name, separator, _default = raw_parameter.partition("=")
            name = _safe_parameter_name(raw_name, parameter_index)
            parsed_parameters.append(
                Parameter(
                    name=name,
                    annotation=parameter_types.get(raw_name.strip(), "Any"),
                    has_default=bool(separator),
                )
            )

        result.append(
            Signature(
                kind=match.group("kind"),
                parameters=tuple(parsed_parameters),
                return_type=(match.group("return") or "Any").strip(),
                documentation=documentation,
            )
        )

    unique: list[Signature] = []
    for signature in result:
        matching_index = next(
            (
                index
                for index, candidate in enumerate(unique)
                if candidate.kind == signature.kind
                and candidate.parameters == signature.parameters
                and candidate.return_type == signature.return_type
            ),
            None,
        )
        if matching_index is None:
            unique.append(signature)
        elif not unique[matching_index].documentation and signature.documentation:
            unique[matching_index] = signature
    return unique


class TypeResolver:
    def __init__(
        self,
        package_name: str,
        module_name: str,
        symbols: dict[str, list[tuple[str, str]]],
        coverage: Coverage,
    ) -> None:
        self._package_name = package_name
        self._module_name = module_name
        self._symbols = symbols
        self._module_names = {module for candidates in symbols.values() for module, _ in candidates}
        self._symbol_paths = {
            (module, symbol) for candidates in symbols.values() for module, symbol in candidates
        }
        self._coverage = coverage
        self.imports: set[str] = set()

    def annotation_for_value(self, value: object) -> str:
        if value is None:
            return "None"
        value_type = type(value)
        if value_type in {bool, bytes, complex, float, int, str}:
            return value_type.__name__
        return self.annotation_for_class(value_type)

    def annotation_for_class(self, value: type[Any]) -> str:
        module_name = getattr(value, "__module__", "")
        class_name = getattr(value, "__name__", "")
        if module_name == "builtins" and getattr(builtins, class_name, None) is value:
            return "Any" if value is object else class_name
        prefix = f"{self._package_name}."
        if module_name.startswith(prefix) and class_name:
            usd_module = module_name[len(prefix) :].split(".", maxsplit=1)[0]
            candidates = [
                symbol for module, symbol in self._symbols.get(class_name, []) if module == usd_module
            ]
            if not candidates:
                return "Any"
            symbol_name = min(candidates, key=lambda candidate: (candidate.count("."), candidate))
            if usd_module == self._module_name:
                return symbol_name
            self.imports.add(usd_module)
            return f"_{usd_module}.{symbol_name}"
        return "Any"

    def normalize(self, raw_type: str) -> str:
        raw_type = raw_type.strip()
        raw_type = re.sub(r"\bconst\s+", "", raw_type)
        raw_type = raw_type.replace("&", "").replace("*", "")
        raw_type = raw_type.replace("unsigned int", "int")
        raw_type = raw_type.replace("unsigned long", "int")
        raw_type = raw_type.replace("long int", "int")
        raw_type = raw_type.replace("std.", "")
        raw_type = raw_type.replace("std::", "")

        def replace_module_qualifier(match: re.Match[str]) -> str:
            module_name = match.group(1)
            if module_name == self._module_name:
                return ""
            self.imports.add(module_name)
            return f"_{module_name}."

        if self._module_names:
            module_pattern = "|".join(re.escape(name) for name in sorted(self._module_names, key=len, reverse=True))
            raw_type = re.sub(rf"\b({module_pattern})\.", replace_module_qualifier, raw_type)

        unknown = False

        def validate_qualified_name(match: re.Match[str]) -> str:
            nonlocal unknown
            qualified_name = match.group(0)
            first, _, remainder = qualified_name.partition(".")
            if first.startswith("_") and first[1:] in self._module_names:
                module_name = first[1:]
                symbol_name = remainder
            else:
                module_name = self._module_name
                symbol_name = qualified_name
            if (module_name, symbol_name) in self._symbol_paths:
                return qualified_name
            unknown = True
            return "Any"

        raw_type = re.sub(
            r"\b_?[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+\b",
            validate_qualified_name,
            raw_type,
        )

        def replace_identifier(match: re.Match[str]) -> str:
            nonlocal unknown
            name = match.group(0)
            if name.startswith("_") and name[1:] in self._module_names:
                return name
            if match.start() > 0 and raw_type[match.start() - 1] == ".":
                return name
            if name in _TYPE_REPLACEMENTS:
                return _TYPE_REPLACEMENTS[name]
            if name in _BUILTIN_TYPES:
                return name
            if name in _TYPING_TYPES:
                return name
            if name in {"True", "False"}:
                return name

            candidates = self._symbols.get(name, [])
            if len(candidates) == 1:
                module_name, symbol_name = candidates[0]
                if module_name == self._module_name:
                    return symbol_name
                self.imports.add(module_name)
                return f"_{module_name}.{symbol_name}"

            for module_name, symbol_name in candidates:
                if module_name == self._module_name:
                    return symbol_name

            unknown = True
            return "Any"

        annotation = _IDENTIFIER_RE.sub(replace_identifier, raw_type)
        annotation = re.sub(r"\s+", " ", annotation).strip()
        annotation = _GENERIC_DEFAULTS.get(annotation, annotation)
        if re.search(r"\b(?:Any|bool|bytes|complex|float|int|str)\.", annotation):
            unknown = True
        try:
            ast.parse(annotation, mode="eval")
        except SyntaxError:
            unknown = True
        if unknown:
            self._coverage.unknown_types += 1
            annotation = "Any"
        return annotation


class StubRenderer:
    def __init__(
        self,
        package_name: str,
        module_name: str,
        symbols: dict[str, list[tuple[str, str]]],
        coverage: Coverage,
    ) -> None:
        self._package_name = package_name
        self._module_name = module_name
        self._coverage = coverage
        self._resolver = TypeResolver(package_name, module_name, symbols, coverage)

    @staticmethod
    def _is_public(name: str) -> bool:
        if not name.isidentifier() or keyword.iskeyword(name):
            return False
        return not name.startswith("_") or name in _PUBLIC_DUNDERS

    @staticmethod
    def _is_boost_enum(value: type[Any]) -> bool:
        return isinstance(getattr(value, "values", None), dict) and isinstance(
            getattr(value, "names", None), dict
        )

    @staticmethod
    def _render_docstring(docstring: str | None, indent: str) -> list[str]:
        if not docstring:
            return []
        cleaned = inspect.cleandoc(docstring).strip()
        if not cleaned:
            return []

        escaped = cleaned.replace("\\", "\\\\").replace('"""', '\\"\\"\\"')
        lines = escaped.splitlines()
        if len(lines) == 1:
            return [f'{indent}"""{lines[0]}"""']
        return [
            f'{indent}"""{lines[0]}',
            *(f"{indent}{line}" for line in lines[1:]),
            f'{indent}"""',
        ]

    def _render_function(
        self,
        definition: str,
        docstring: str | None,
        indent: str,
    ) -> list[str]:
        documentation = self._render_docstring(docstring, indent + "    ")
        if not documentation:
            return [f"{indent}{definition}: ..."]
        return [
            f"{indent}{definition}:",
            *documentation,
            f"{indent}    ...",
        ]

    def _render_signature(
        self,
        name: str,
        signature: Signature,
        indent: str,
        is_method: bool,
    ) -> list[str]:
        parameters: list[str] = []
        if is_method and signature.kind != "static":
            parameters.append("cls" if signature.kind == "class" else "self")
        for parameter in signature.parameters:
            annotation = self._resolver.normalize(parameter.annotation)
            default = " = ..." if parameter.has_default else ""
            parameters.append(f"{parameter.name}: {annotation}{default}")

        return_type = "None" if name == "__init__" else self._resolver.normalize(signature.return_type)
        definition = f"def {name}({', '.join(parameters)}) -> {return_type}"
        try:
            ast.parse(f"{definition}: ...")
        except SyntaxError:
            receiver = "cls, " if is_method and signature.kind == "class" else "self, " if is_method else ""
            definition = f"def {name}({receiver}*args: Any, **kwargs: Any) -> Any"

        result: list[str] = []
        if signature.kind == "class":
            result.append(f"{indent}@classmethod")
        elif signature.kind == "static":
            result.append(f"{indent}@staticmethod")
        result.extend(self._render_function(definition, signature.documentation, indent))
        return result

    def _render_callable(
        self,
        name: str,
        value: object,
        indent: str,
        is_method: bool,
    ) -> list[str]:
        self._coverage.callables += 1
        signatures = parse_docstring_signatures(getattr(value, "__doc__", None), name)
        if not signatures:
            self._coverage.fallback_callables += 1
            receiver = "self, " if is_method else ""
            return_type = "None" if name == "__init__" else "Any"
            definition = f"def {name}({receiver}*args: Any, **kwargs: Any) -> {return_type}"
            return self._render_function(definition, getattr(value, "__doc__", None), indent)

        self._coverage.documented_callables += 1
        shared_kind = next((signature.kind for signature in signatures if signature.kind), None)
        if shared_kind is not None:
            signatures = [
                Signature(
                    shared_kind,
                    signature.parameters,
                    signature.return_type,
                    signature.documentation,
                )
                for signature in signatures
            ]
        result: list[str] = []
        for signature in signatures:
            if len(signatures) > 1:
                result.append(f"{indent}@overload")
            result.extend(self._render_signature(name, signature, indent, is_method))
        return result

    def _render_enum(self, name: str, value: type[Any], indent: str) -> list[str]:
        self._coverage.classes += 1
        result = [f"{indent}class {name}(int):"]
        result.extend(self._render_docstring(getattr(value, "__doc__", None), indent + "    "))
        names = getattr(value, "names", {})
        if not names:
            if len(result) == 1:
                result.append(f"{indent}    ...")
            return result
        for member_name in sorted(names):
            if member_name.isidentifier() and not keyword.iskeyword(member_name):
                member_type = "int" if indent else f'"{name}"'
                result.append(f"{indent}    {member_name}: ClassVar[{member_type}]")
        return result

    def _render_property(self, name: str, value: property, indent: str) -> list[str]:
        return_type = "Any"
        docstring = getattr(value, "__doc__", None)
        if docstring:
            for line in inspect.cleandoc(docstring).splitlines():
                match = _PARAMETER_TYPE_RE.match(line.strip())
                if match and match.group("name") == "type":
                    return_type = self._resolver.normalize(match.group("type"))
                    break
        result = [f"{indent}@property"]
        result.extend(
            self._render_function(
                f"def {name}(self) -> {return_type}",
                docstring,
                indent,
            )
        )
        if value.fset is not None:
            result.extend(
                [
                    f"{indent}@{name}.setter",
                    f"{indent}def {name}(self, value: {return_type}) -> None: ...",
                ]
            )
        return result

    def _render_class(self, name: str, value: type[Any], indent: str = "") -> list[str]:
        if self._is_boost_enum(value):
            return self._render_enum(name, value, indent)

        self._coverage.classes += 1
        bases: list[str] = []
        for base in getattr(value, "__bases__", ()):
            annotation = self._resolver.annotation_for_class(base)
            if annotation != "Any" and annotation != name:
                bases.append(annotation)
        base_clause = f"({', '.join(bases)})" if bases else ""
        result = [f"{indent}class {name}{base_clause}:"]
        body: list[str] = []
        class_indent = indent + "    "
        body.extend(self._render_docstring(getattr(value, "__doc__", None), class_indent))

        for member_name, member in getattr(value, "__dict__", {}).items():
            if not self._is_public(member_name):
                continue
            if isinstance(member, (classmethod, staticmethod)):
                bound_member = getattr(value, member_name)
                body.extend(self._render_callable(member_name, bound_member, class_indent, True))
            elif inspect.isclass(member):
                body.extend(self._render_class(member_name, member, class_indent))
            elif isinstance(member, property):
                body.extend(self._render_property(member_name, member, class_indent))
            elif callable(member):
                body.extend(self._render_callable(member_name, member, class_indent, True))
            elif not inspect.ismodule(member):
                annotation = self._resolver.annotation_for_value(member)
                body.append(f"{class_indent}{member_name}: ClassVar[{annotation}]")

        result.extend(body or [f"{class_indent}..."])
        return result

    def render(self, module: types.ModuleType) -> str:
        body: list[str] = []
        seen_values: dict[int, str] = {}
        for name, value in module.__dict__.items():
            if not self._is_public(name) or inspect.ismodule(value):
                continue
            if inspect.isclass(value):
                canonical_name = seen_values.get(id(value))
                if canonical_name is not None:
                    body.append(f"{name} = {canonical_name}")
                else:
                    seen_values[id(value)] = name
                    body.extend(self._render_class(name, value))
            elif callable(value):
                body.extend(self._render_callable(name, value, "", False))
            else:
                body.append(f"{name}: {self._resolver.annotation_for_value(value)}")

        body_text = "\n".join(body)
        body_tree = ast.parse(body_text)
        imports = [
            "# pyright: reportIncompatibleMethodOverride=false, "
            "reportIncompatibleVariableOverride=false, reportOverlappingOverload=false",
            "from __future__ import annotations",
            "",
        ]
        used_names = {
            node.id
            for node in ast.walk(body_tree)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        }
        typing_names = sorted(_TYPING_NAMES & used_names)
        if typing_names:
            imports.append(f"from typing import {', '.join(typing_names)}")
        for module_name in sorted(self._resolver.imports):
            imports.append(f"from {self._package_name} import {module_name} as _{module_name}")
        imports.append("")

        output = "\n".join(imports + body + [""])
        ast.parse(output)
        return output


def _build_symbol_index(
    package_name: str, modules: dict[str, types.ModuleType]
) -> dict[str, list[tuple[str, str]]]:
    symbols: dict[str, list[tuple[str, str]]] = {}

    def add_class(module_name: str, symbol_name: str, value: type[Any], seen: set[int]) -> None:
        if id(value) in seen:
            return
        seen.add(id(value))
        short_name = symbol_name.rsplit(".", maxsplit=1)[-1]
        symbols.setdefault(short_name, []).append((module_name, symbol_name))
        cpp_name = getattr(value, "__name__", short_name)
        symbols.setdefault(f"{module_name}{cpp_name}", []).append((module_name, symbol_name))
        for child_name, child in getattr(value, "__dict__", {}).items():
            if child_name.isidentifier() and not keyword.iskeyword(child_name) and inspect.isclass(child):
                add_class(module_name, f"{symbol_name}.{child_name}", child, seen)

    for module_name, module in modules.items():
        seen: set[int] = set()
        for symbol_name, value in module.__dict__.items():
            if symbol_name.startswith("_") or not inspect.isclass(value):
                continue
            add_class(module_name, symbol_name, value, seen)
    return symbols


def generate_stubs(package_name: str, module_names: Sequence[str], output: Path) -> Coverage:
    modules = {
        module_name: importlib.import_module(f"{package_name}.{module_name}")
        for module_name in module_names
    }
    coverage = Coverage(modules=len(modules))
    symbols = _build_symbol_index(package_name, modules)

    for module_name, module in modules.items():
        module_output = output / module_name / "__init__.pyi"
        module_output.parent.mkdir(parents=True, exist_ok=True)
        module_output.write_text(
            StubRenderer(package_name, module_name, symbols, coverage).render(module),
            encoding="utf-8",
            newline="\n",
        )

    root_lines = ["from __future__ import annotations", ""]
    root_lines.extend(f"from . import {name} as {name}" for name in module_names)
    root_lines.extend(["", f"__all__: list[str]", ""])
    (output / "__init__.pyi").write_text("\n".join(root_lines), encoding="utf-8", newline="\n")
    return coverage


def _discover_modules(package_name: str) -> list[str]:
    package = importlib.import_module(package_name)
    module_names = getattr(package, "__all__", None)
    if not isinstance(module_names, Iterable):
        raise RuntimeError(f"{package_name} does not define an iterable __all__")
    discovered = {str(name) for name in module_names}
    discovered.update(module.name for module in pkgutil.iter_modules(package.__path__))
    return sorted(discovered)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", default="falcor2.pxr")
    parser.add_argument("--module", action="append", dest="modules")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python-path", type=Path, action="append", default=[])
    parser.add_argument("--dll-path", type=Path, action="append", default=[])
    args = parser.parse_args(argv)

    for python_path in reversed(args.python_path):
        sys.path.insert(0, str(python_path))
    dll_directories = []
    if os.name == "nt":
        dll_directories = [os.add_dll_directory(path.resolve()) for path in args.dll_path]

    module_names = sorted(args.modules) if args.modules else _discover_modules(args.package)
    coverage = generate_stubs(args.package, module_names, args.output)
    print(
        "Generated USD stubs: "
        f"modules={coverage.modules}, classes={coverage.classes}, "
        f"callables={coverage.callables}, documented={coverage.documented_callables}, "
        f"fallback={coverage.fallback_callables}, unknown_types={coverage.unknown_types}"
    )
    for dll_directory in dll_directories:
        dll_directory.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
