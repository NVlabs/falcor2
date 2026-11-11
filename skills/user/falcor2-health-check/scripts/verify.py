# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import importlib
import json
import os
import shutil
import subprocess
import sys
from importlib.metadata import version


def _driver_info() -> dict:
    command = shutil.which("nvidia-smi")
    if not command:
        return {"status": "unavailable", "detail": "nvidia-smi not on PATH; use the GPU check."}
    try:
        result = subprocess.run(
            [command, "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return {
            "status": "reported" if result.returncode == 0 else "query_failed",
            "detail": (result.stdout + result.stderr).strip(),
        }
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"status": "query_failed", "detail": str(error)}


def main(argv: list[str] | None = None) -> int:
    if os.name == "nt":
        import ctypes

        ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002 | 0x8000)
    parser = argparse.ArgumentParser(description="Check Falcor2 imports and GPU availability.")
    parser.add_argument("--imports-only", action="store_true", help="Skip driver and GPU checks")
    args = parser.parse_args(argv)
    report = {
        "python": sys.executable,
        "python_version": sys.version,
        "packages": {},
        "gpu": "not_checked",
    }
    stage = "installation"
    try:
        for name in ("slangpy", "falcor2"):
            stage = f"{name} package metadata"
            report["packages"][name] = {"version": version(name)}
            stage = f"{name} import"
            module = importlib.import_module(name)
            report["packages"][name]["module"] = module.__file__
        if not args.imports_only:
            report["driver"] = _driver_info()
            stage = "GPU device creation"
            device = module.create_device()
            device.wait()
            report["gpu"] = {"adapter": device.info.adapter_name, "api": device.info.api_name}
        report["status"] = "imports_only" if args.imports_only else "passed"
        code = 0
    except (Exception, SystemExit) as error:
        report.update(status="failed", stage=stage, error=f"{type(error).__name__}: {error}")
        code = 1
    print(json.dumps(report, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
