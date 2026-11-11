# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build Falcor2's narrative documentation and public Python API reference."""

import importlib.util
import builtins
import inspect
from pathlib import Path
import sys

from sphinx.application import Sphinx
from sphinx import addnodes
from docutils import nodes

_DOCS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_DOCS_DIR.parent))

project = "Falcor2"
author = "NVIDIA"
copyright = "2026, NVIDIA"
extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
]
source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
root_doc = "index"
exclude_patterns = ["_build/**", "generated/**"]
myst_heading_anchors = 6
html_theme = "furo"
html_title = "Falcor2"
autodoc_typehints_format = "short"
# SlangPy defines this descriptor union in its type stubs, not at runtime.
autodoc_type_aliases = {"DeviceDescParam": "slangpy.DeviceDescParam"}


def _generate_api(app: Sphinx) -> None:
    spec = importlib.util.spec_from_file_location(
        "falcor2_docs_generate_api", _DOCS_DIR / "generate_api.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.generate_api()
    app.config.release = app.config.release or sys.modules["falcor2"].__version__
    app.config.html_title = f"Falcor2 {app.config.release}"


def setup(app: Sphinx) -> None:
    app.connect("builder-inited", _generate_api)
    app.connect("doctree-read", _resolve_builtin_types)


def _resolve_builtin_types(app: Sphinx, doctree: nodes.document) -> None:
    """Keep builtin type annotations from linking to same-named enum values."""
    for node in list(doctree.findall(addnodes.pending_xref)):
        if (
            node.get("refdomain") == "py"
            and node.get("reftype") in {"class", "obj"}
            and inspect.isclass(getattr(builtins, node.get("reftarget", ""), None))
        ):
            node.replace_self(node.children)
