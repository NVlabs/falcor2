# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from typing import Any

from tools import build


def test_get_build_env_preserves_lowercase_vcpkg_keep_env_vars(monkeypatch: Any) -> None:
    monkeypatch.setattr(build, "os", SimpleNamespace(name="nt", environ={}))
    monkeypatch.setattr(
        build,
        "_get_msvc_env",
        lambda _platform, _env: {"vcpkg_keep_env_vars": "EXISTING_ONE;EXISTING_TWO"},
    )

    env = build.get_build_env()

    assert set(env["VCPKG_KEEP_ENV_VARS"].split(";")) == {
        "EXISTING_ONE",
        "EXISTING_TWO",
        "FALCOR_PYTHON_ABI",
        "FALCOR_PYTHON_EXECUTABLE",
    }
    assert "vcpkg_keep_env_vars" not in env
