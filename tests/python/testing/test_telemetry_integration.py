# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid


PROJECT_ROOT = Path(__file__).parents[3]


def test_telemetry_plugin_merges_xdist_workers_after_a_failure() -> None:
    scratch_root = PROJECT_ROOT / ".tmp" / "test-telemetry-integration"
    run_dir = scratch_root / uuid.uuid4().hex
    test_path = run_dir / "test_sample.py"
    output_path = run_dir / "test-telemetry.json"
    run_dir.mkdir(parents=True)
    test_path.write_text(
        "def test_pass():\n" "    assert True\n\n" "def test_fail():\n" "    assert False\n",
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment.pop("PYTEST_ADDOPTS", None)

    try:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-p",
                "tests.python.conftest",
                str(test_path),
                "-q",
                "-n",
                "2",
                "--maxprocesses=2",
                "--telemetry",
                f"--telemetry-output={output_path}",
            ],
            cwd=PROJECT_ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )

        assert completed.returncode == 1, completed.stdout + completed.stderr
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        assert payload["schema_version"] == 1
        assert payload["exitstatus"] == 1
        relative_test_path = test_path.relative_to(PROJECT_ROOT).as_posix()
        assert {record["nodeid"] for record in payload["tests"]} == {
            f"{relative_test_path}::test_pass",
            f"{relative_test_path}::test_fail",
        }
        assert payload["workers"]
        for record in payload["tests"]:
            assert record["worker"] in payload["workers"]
            assert record["durations_seconds"].keys() >= {
                "setup",
                "call",
                "teardown",
                "wall",
            }
    finally:
        shutil.rmtree(run_dir)
        try:
            scratch_root.rmdir()
        except OSError:
            pass
