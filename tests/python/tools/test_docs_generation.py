# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path
import shutil
from types import ModuleType

import pytest

pytest.importorskip(
    "sphinx", reason="Install docs/requirements.txt to test documentation generation"
)

from sphinx.errors import ExtensionError


@pytest.fixture
def docs_generator() -> ModuleType:
    path = Path(__file__).resolve().parents[3] / "docs/generate_api.py"
    spec = importlib.util.spec_from_file_location("falcor2_docs_generator_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_public_reference_and_repeatability(
    docs_generator: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    shutil.copy2(docs_generator._DOCS_DIR / "api_order.json", tmp_path / "api_order.json")
    monkeypatch.setattr(docs_generator, "_DOCS_DIR", tmp_path)
    docs_generator.generate_api()
    native = tmp_path / "generated/native-api.rst"
    python = tmp_path / "generated/python-api.rst"
    native_text = native.read_text(encoding="utf-8")
    python_text = python.read_text(encoding="utf-8")
    for name in (
        "Scene",
        "AABB",
        "PropertyType",
        "ui.CameraController",
        "reflection.PythonPropertyDescriptor",
    ):
        assert f".. py:class:: falcor2.{name}" in native_text
    for name in (
        "Editor",
        "EditorConfig",
        "ContainerSpec",
        "RenderNode",
        "rendernodes.ReferencePathTracerNode",
    ):
        assert f".. autoclass:: falcor2.{name}" in python_text
    assert ".. autofunction:: falcor2.pyscene.load_asset" in python_text
    assert ".. autofunction:: falcor2.reflection.reflected" in python_text
    assert ".. autodoc" not in native_text
    assert ".. autoclass:: falcor2.PropertyType" not in python_text
    assert "Alias of :py:class:`falcor2.Editor`" in python_text
    assert "Alias of :py:func:`falcor2.create_device`" in python_text
    for excluded in ("falcor2.testing", "falcor2.mcp", "falcor2.minitracer", "falcor2.pxr"):
        for directive in ("py:class", "py:function", "autoclass", "autofunction", "automodule"):
            assert f".. {directive}:: {excluded}." not in native_text + python_text
    assert ".. py:class:: slangpy." not in native_text
    assert "object at 0x" not in native_text
    modified = (native.stat().st_mtime_ns, python.stat().st_mtime_ns)
    docs_generator.generate_api()
    assert modified == (native.stat().st_mtime_ns, python.stat().st_mtime_ns)
    assert native.read_text(encoding="utf-8") == native_text
    assert python.read_text(encoding="utf-8") == python_text


def test_python_selection_aliases_and_removed_exports(docs_generator: ModuleType) -> None:
    root = ModuleType("falcor2")
    child = ModuleType("falcor2.editor")

    class Undocumented:
        pass

    Undocumented.__module__ = "falcor2.editor.implementation"
    root.Undocumented = Undocumented
    child.Alias = Undocumented
    child._private = Undocumented
    child.Foreign = Path
    child.ONE = 1
    child.ALSO_ONE = 1
    text = docs_generator._render_python_api([root, child])
    assert text.count(".. autoclass::") == 1
    assert "Alias of :py:class:`falcor2.Undocumented`" in text
    assert "_private" not in text and "Foreign" not in text
    assert ".. autodata:: falcor2.editor.ONE" in text
    assert ".. autodata:: falcor2.editor.ALSO_ONE" in text
    del root.Undocumented
    text = docs_generator._render_python_api([root, child])
    assert ".. autoclass:: falcor2.editor.Alias" in text
    assert "falcor2.Undocumented" not in text


def test_missing_bindings_fail_even_with_stale_fragments(
    docs_generator: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    generated = tmp_path / "generated"
    generated.mkdir()
    existing = generated / "native-api.rst"
    existing.write_text("stale reference", encoding="utf-8")
    monkeypatch.setattr(docs_generator, "_DOCS_DIR", tmp_path)

    def fail_import(name: str) -> ModuleType:
        raise ImportError("missing test bindings")

    monkeypatch.setattr(docs_generator.importlib, "import_module", fail_import)
    with pytest.raises(ExtensionError, match="build Release bindings"):
        docs_generator.generate_api()
    assert existing.read_text(encoding="utf-8") == "stale reference"


def test_old_shared_generator_has_actionable_failure(
    docs_generator: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "external/slangpy/docs/generate_api.py"
    path.parent.mkdir(parents=True)
    path.write_text("def generate_api(): pass\n", encoding="utf-8")
    monkeypatch.setattr(docs_generator, "_ROOT_DIR", tmp_path)
    with pytest.raises(RuntimeError, match="lacks the shared documentation generator parameters"):
        docs_generator._load_shared_generator()


def test_native_docstring_formats_google_sections_and_wrapped_lists(
    docs_generator: ModuleType,
) -> None:
    text = docs_generator._format_native_docstring(
        "Connect a slot.\n\nArgs:\n    owner: Owner of the connection.\n"
        "        Retained until disconnect.\n\n## Modes\n\n"
        "- **First:** Wrapped\ncontinuation. - **Second:** Another mode."
    )
    assert ":param owner:" in text
    assert ".. rubric:: Modes" in text
    assert "- **First:** Wrapped continuation.\n\n- **Second:** Another mode." in text


def test_native_defaults_omit_addresses_but_keep_real_values(docs_generator: ModuleType) -> None:
    text = (
        ".. py:function:: falcor2.create(options: falcor2.Options = "
        "<falcor2.Options object at 0x12345678>, handle: int = 0x00000000) -> None\n\n"
        "    Example repr: <falcor2.Options object at 0x12345678>\n"
    )
    result = docs_generator._normalize_native_signatures(text)
    assert "options: falcor2.Options = ..." in result
    assert "handle: int = 0x00000000" in result
    assert "Example repr: <falcor2.Options object at 0x12345678>" in result
