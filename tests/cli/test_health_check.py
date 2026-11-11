# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def check(monkeypatch: pytest.MonkeyPatch) -> object:
    path = (
        Path(__file__).resolve().parents[2] / "skills/user/falcor2-health-check/scripts/verify.py"
    )
    spec = importlib.util.spec_from_file_location("health_check", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "version", lambda name: "1.0")
    device = SimpleNamespace(
        info=SimpleNamespace(adapter_name="GPU", api_name="test"), wait=lambda: None
    )
    monkeypatch.setattr(
        module.importlib,
        "import_module",
        lambda name: SimpleNamespace(__file__=name, create_device=lambda: device),
    )
    monkeypatch.setattr(module.shutil, "which", lambda name: None)
    return module


def test_health_success(check: object, capsys: pytest.CaptureFixture) -> None:
    assert check.main([]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "passed"
    assert result["gpu"]["adapter"] == "GPU"
    assert result["driver"]["status"] == "unavailable"


def test_imports_only(
    check: object, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    def no_gpu() -> None:
        pytest.fail("GPU/driver check must be skipped")

    monkeypatch.setattr(check, "_driver_info", no_gpu)
    monkeypatch.setattr(
        check.importlib, "import_module", lambda name: SimpleNamespace(__file__=name)
    )
    assert check.main(["--imports-only"]) == 0
    assert json.loads(capsys.readouterr().out)["gpu"] == "not_checked"


@pytest.mark.parametrize("stage", ["metadata", "import", "gpu"])
def test_failed_stage(
    check: object, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture, stage: str
) -> None:
    def fail(*args: object) -> None:
        raise RuntimeError("diagnostic detail")

    if stage == "metadata":
        monkeypatch.setattr(check, "version", fail)
    elif stage == "import":
        monkeypatch.setattr(check.importlib, "import_module", fail)
    else:
        monkeypatch.setattr(
            check.importlib,
            "import_module",
            lambda name: SimpleNamespace(__file__=name, create_device=fail),
        )
    assert check.main([]) == 1
    result = json.loads(capsys.readouterr().out)
    assert "diagnostic detail" in result["error"]
    assert (
        result["stage"]
        == {
            "metadata": "slangpy package metadata",
            "import": "slangpy import",
            "gpu": "GPU device creation",
        }[stage]
    )


def test_driver_timeout_is_diagnostic(
    check: object, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setattr(check.shutil, "which", lambda name: name)

    def timeout(*args: object, **kwargs: object) -> None:
        raise subprocess.TimeoutExpired("nvidia-smi", 10)

    monkeypatch.setattr(check.subprocess, "run", timeout)
    assert check.main([]) == 0
    assert json.loads(capsys.readouterr().out)["driver"]["status"] == "query_failed"
