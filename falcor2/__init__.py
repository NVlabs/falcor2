# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# pyright: reportUnusedImport=false

import os
import sys
from importlib import import_module as _import

_import("slangpy")

if os.name == "nt":
    package_dir = os.path.normpath(os.path.dirname(__file__))
    if os.path.exists(os.path.join(package_dir, "falcor2.dll")):
        # This is a deployed package containing all the DLLs.
        # No need to setup a dll directory.
        pass
    elif os.path.exists(os.path.join(package_dir, ".build_dir")):
        # Loading package from a development build.
        # The DLLs are in the build directory.
        build_dir = open(os.path.join(package_dir, ".build_dir")).readline().strip()
        os.add_dll_directory(build_dir)
    else:
        print("Cannot locate falcor2.dll.")
        sys.exit(1)

del os, sys

_import("falcor2.falcor2_ext")
_import("falcor2.extensions")
del _import

from falcor2.rendergraph import Container, ContainerSpec, RenderNode  # noqa: E402,F401

from falcor2.editor import (  # noqa: E402,F401
    CameraController,
    Editor,
    EditorConfig,
    FrameOutput,
    RenderMode,
    SceneShaderHelper,
    create_device,
    get_slang_include_paths,
    load_scene,
    save_image,
)
