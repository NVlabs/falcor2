# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared environment setup for local and CI performance commands."""

import os
from pathlib import Path
import shutil

from tools.build import _get_msvc_env


def prepare_environment(python: str | Path, checkout: str | Path, environ=None):
    env = dict(os.environ if environ is None else environ)
    # Keep Linux venv symlinks: resolving one loses the selected environment.
    executable = Path(python).expanduser().absolute()
    if os.name == "nt" and executable.drive.lower() != Path(checkout).absolute().drive.lower():
        raise RuntimeError(
            f"Python {executable} and checkout {checkout} must be on the same drive "
            "for D3D12 Agility SDK initialization. Select a prepared environment "
            "on the checkout drive with --python."
        )
    if not executable.is_file():
        raise RuntimeError(f"Python interpreter does not exist: {executable}")
    # Support both Conda's root interpreter and venv's Scripts/bin interpreter.
    prefix = executable.parent
    if prefix.name.lower() in ("scripts", "bin"):
        prefix = prefix.parent
    if os.name == "nt":
        env = {key.upper(): value for key, value in env.items()}
        paths = [prefix, prefix / "Scripts", prefix / "Library/bin"]
    else:
        paths = [prefix / "bin"]
    env["PATH"] = os.pathsep.join([*(str(path) for path in paths), env.get("PATH", "")])
    if (prefix / "pyvenv.cfg").is_file():
        env["VIRTUAL_ENV"] = str(prefix)
        env["Python3_ROOT_DIR"] = str(prefix)
    else:
        env["CONDA_PREFIX"] = str(prefix)
        if os.name != "nt" and (prefix / "conda-meta").is_dir():
            env["LD_LIBRARY_PATH"] = os.pathsep.join(
                [str(prefix / "lib"), *filter(None, [env.get("LD_LIBRARY_PATH")])]
            )
    if os.name == "nt":
        env = {key.upper(): value for key, value in _get_msvc_env("x64", env).items()}
    missing = [
        name for name in ("git", "cmake", "ninja") if not shutil.which(name, path=env["PATH"])
    ]
    if missing:
        raise RuntimeError("Required tools are missing from PATH: " + ", ".join(missing))
    return executable, env
