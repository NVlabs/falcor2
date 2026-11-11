# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# pyright: reportUnusedImport=false

import os
from importlib import import_module as _import


_import("slangpy")

_package_dir = os.path.normpath(os.path.dirname(__file__))
_runtime_dir = _package_dir
_build_marker = os.path.join(_package_dir, ".build_dir")
_dll_directories = []
if os.name == "nt":
    if os.path.exists(os.path.join(_package_dir, "falcor2.dll")):
        # This is a deployed package containing all the DLLs.
        # No need to setup a dll directory.
        pass
    elif os.path.exists(_build_marker):
        # Loading package from a development build.
        # The DLLs are in the build directory.
        with open(_build_marker, encoding="utf-8") as _file:
            _runtime_dir = _file.readline().strip()
        _dll_directories.append(os.add_dll_directory(_runtime_dir))
    else:
        raise ImportError("Cannot locate falcor2.dll.")
elif os.path.exists(_build_marker):
    with open(_build_marker, encoding="utf-8") as _file:
        _runtime_dir = _file.readline().strip()

# Generated OpenUSD bindings live in the selected build configuration during
# development. Extend this package's search path without writing them into the
# shared source tree.
_development_package_dir = os.path.join(_runtime_dir, "python", "falcor2")
if os.path.isdir(_development_package_dir):
    __path__.insert(0, _development_package_dir)

_plugin_dir = os.path.join(_runtime_dir, "usd")
_plugin_path = None
if os.path.exists(_plugin_dir):
    _plugin_path = os.environ.get("FALCOR2_USD_PLUGIN_PATH")
    os.environ["FALCOR2_USD_PLUGIN_PATH"] = os.pathsep.join(
        path for path in (_plugin_dir, _plugin_path) if path
    )

_import("falcor2.falcor2_ext")
_import("falcor2.extensions")

from falcor2.rendergraph import Container, ContainerSpec, RenderNode  # noqa: E402,F401

from falcor2.editor import (  # noqa: E402,F401
    CameraController,
    Editor,
    EditorConfig,
    FrameOutput,
    RenderMode,
    StreamingConfig,
    SceneShaderHelper,
    create_device,
    get_slang_include_paths,
    load_scene,
    save_image,
)


del (
    os,
    _import,
    _package_dir,
    _runtime_dir,
    _build_marker,
    _development_package_dir,
    _plugin_dir,
    _plugin_path,
)
