# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Select Falcor2's public API and reuse SlangPy's native reference renderer."""

from __future__ import annotations

import contextlib
import importlib
import importlib.util
import inspect
import io
import json
from pathlib import Path
import re
import tempfile
from types import ModuleType

from sphinx.errors import ExtensionError
from sphinx.ext.napoleon.docstring import GoogleDocstring
from sphinx.util import logging

_DOCS_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _DOCS_DIR.parent
_NATIVE_MODULES = (
    "falcor2",
    "falcor2.ui",
    "falcor2.reflection",
    "falcor2.platform",
    "falcor2.ngx",
)
_PYTHON_MODULES = (
    "falcor2",
    "falcor2.editor",
    "falcor2.rendergraph",
    "falcor2.rendernodes",
    "falcor2.pyscene",
    "falcor2.reflection",
)
_LOGGER = logging.getLogger(__name__)


def _load_shared_generator() -> ModuleType:
    path = _ROOT_DIR / "external/slangpy/docs/generate_api.py"
    spec = importlib.util.spec_from_file_location("slangpy_docs_generate_api", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load the shared SlangPy generator at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    required = {
        "module_name",
        "api_order",
        "output_path",
        "include_module",
        "include_member",
        "format_docstring",
        "skip_empty_sections",
    }
    if not required.issubset(inspect.signature(module.generate_api).parameters):
        raise RuntimeError(
            "The SlangPy checkout lacks the shared documentation generator parameters."
        )
    return module


def _is_native_member(name: str, obj: object) -> bool:
    namespace = name.rpartition(".")[0]
    if namespace not in _NATIVE_MODULES:
        return False
    if isinstance(obj, (str, int, float, bool)):
        return name.rpartition(".")[2].isupper()
    return getattr(obj, "__module__", None) in _NATIVE_MODULES and (
        inspect.isclass(obj) or type(obj).__module__ == "nanobind"
    )


def _python_exports(module: ModuleType) -> list[tuple[str, object]]:
    names = getattr(module, "__all__", vars(module))
    result = []
    for name in sorted(names):
        if name.startswith("_"):
            continue
        obj = getattr(module, name)
        qualified = f"{module.__name__}.{name}"
        if _is_native_member(qualified, obj):
            continue
        owner = getattr(obj, "__module__", type(obj).__module__)
        owned = owner == "falcor2" or owner.startswith("falcor2.")
        if (inspect.isclass(obj) or inspect.isfunction(obj)) and owned:
            result.append((qualified, obj))
        elif name.isupper() and (owned or isinstance(obj, (str, int, float, bool, tuple))):
            result.append((qualified, obj))
    return result


def _render_python_api(modules: list[ModuleType]) -> str:
    sections = []
    documented: dict[int, str] = {}
    for module in modules:
        entries = []
        for name, obj in _python_exports(module):
            kind = (
                "class"
                if inspect.isclass(obj)
                else "function" if inspect.isfunction(obj) else "data"
            )
            canonical = documented.get(id(obj)) if kind != "data" else None
            if canonical is not None:
                role = "func" if kind == "function" else kind
                entries.append(f"``{name}``\n    Alias of :py:{role}:`{canonical}`.\n")
                continue
            entries.append(f".. auto{kind}:: {name}\n")
            if kind == "class":
                entries[-1] += (
                    "    :members:\n"
                    "    :undoc-members:\n"
                    "    :special-members: __init__, __call__\n"
                    "    :show-inheritance:\n"
                )
            if kind != "data":
                documented[id(obj)] = name
        if entries:
            sections.append(
                f"{module.__name__}\n{'-' * len(module.__name__)}\n\n" + "\n".join(entries)
            )
    return "\n".join(sections)


def _write_if_changed(path: Path, text: str) -> None:
    if not path.exists() or path.read_text(encoding="utf-8") != text:
        path.write_text(text, encoding="utf-8")


def _format_native_docstring(text: str) -> str:
    """Normalize the Google sections and Markdown bullets in binding docstrings."""
    text = str(GoogleDocstring(text))
    paragraphs = []
    for paragraph in text.split("\n\n"):
        if paragraph.startswith("## "):
            paragraph = ".. rubric:: " + paragraph[3:]
        elif paragraph.startswith("- "):
            # mkdoc wraps C++ Markdown lists without preserving continuation indentation.
            paragraph = " ".join(line.strip() for line in paragraph.splitlines())
            paragraph = re.sub(r" - (?=\*\*)", "\n\n- ", paragraph)
        paragraphs.append(paragraph)
    return "\n\n".join(paragraphs)


def _normalize_native_signatures(text: str) -> str:
    """Hide process-specific object reprs in defaults without changing documented values."""
    return re.sub(
        r"(?m)^(\s*\.\. py:(?:method|staticmethod|function)::[^\n]*)$",
        lambda match: re.sub(r"<[\w.]+ object at 0x[0-9a-fA-F]+>", "...", match.group(0)),
        text,
    )


def generate_api() -> None:
    """Generate both reference fragments, requiring current repository-local bindings."""
    try:
        package = importlib.import_module("falcor2")
        expected = _ROOT_DIR / "falcor2/__init__.py"
        if Path(package.__file__).resolve() != expected:
            raise RuntimeError(f"Expected {expected}, imported {package.__file__}")
        for name in _NATIVE_MODULES:
            importlib.import_module(name)
        python_modules = [importlib.import_module(name) for name in _PYTHON_MODULES]
        generator = _load_shared_generator()
        ordering = json.loads((_DOCS_DIR / "api_order.json").read_text(encoding="utf-8"))
        _LOGGER.info("Documenting Falcor2 %s from %s", package.__version__, package.__file__)
        python_text = _render_python_api(python_modules)
        with tempfile.TemporaryDirectory(prefix="falcor2-docs-") as temporary:
            target = Path(temporary) / "native.rst"
            diagnostics = io.StringIO()
            with contextlib.redirect_stdout(diagnostics):
                generator.generate_api(
                    "falcor2",
                    api_order=ordering,
                    output_path=target,
                    include_module=lambda name: name in _NATIVE_MODULES,
                    include_member=_is_native_member,
                    format_docstring=_format_native_docstring,
                    skip_empty_sections=True,
                )
            if "Error processing property" in diagnostics.getvalue():
                raise RuntimeError(diagnostics.getvalue())
            native_text = _normalize_native_signatures(target.read_text(encoding="utf-8"))
        generated = _DOCS_DIR / "generated"
        generated.mkdir(exist_ok=True)
        _write_if_changed(generated / "native-api.rst", native_text)
        _write_if_changed(generated / "python-api.rst", python_text)
    except Exception as exc:
        raise ExtensionError(
            "Falcor2 API generation failed. Use this checkout's development interpreter, "
            "build Release bindings, and initialize its pinned SlangPy submodule. "
            f"The documentation build cannot continue with stale API pages: {exc}"
        ) from exc
