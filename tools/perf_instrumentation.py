# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare historical instrumentation from main history and local adapters."""

import json
from pathlib import Path
import subprocess
from typing import Any

from tools.ci_submodules import inherit_git_config

PROFILES = Path(__file__).with_name("perf_compat") / "profiles.json"


def load_profiles(profiles_path: Path = PROFILES) -> dict[str, Any]:
    return json.loads(profiles_path.read_text(encoding="utf-8"))


def select_instrumentation(
    source: str, repository: Path, profiles: dict[str, Any]
) -> dict[str, Any]:
    history = subprocess.check_output(
        ["git", "rev-list", "--first-parent", "--reverse", profiles["last"]],
        cwd=repository,
        text=True,
    ).splitlines()
    starts = {entry["first"]: entry for entry in profiles["ranges"]}
    if len(starts) != len(profiles["ranges"]) or not starts:
        raise ValueError("Compatibility ranges must have unique starting revisions")
    positions = {commit: i for i, commit in enumerate(history)}
    boundaries = [positions.get(entry["first"], -1) for entry in profiles["ranges"]]
    if -1 in boundaries or boundaries != sorted(boundaries):
        raise ValueError("Compatibility boundaries require complete first-parent history")
    if source not in positions or positions[source] < boundaries[0]:
        raise ValueError(f"No validated instrumentation range for Falcor2 {source}")
    selected = None
    for commit in history:
        selected = starts.get(commit, selected)
        if commit == source:
            return selected


def apply_instrumentation(
    repository: Path, profiles: dict[str, Any], selected: dict[str, Any]
) -> None:
    """Copy measurement files from main, retaining the historical dependency pins."""

    def read(root: Path, *args: str) -> bytes:
        return subprocess.check_output(["git", *args], cwd=root)

    base = profiles["base"]
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", base, profiles["last"]],
        cwd=repository,
        check=True,
    )
    reporting = repository / "external/slangpy"
    inherit_git_config(repository, reporting)
    reporting_ref = read(repository, "rev-parse", f"{base}:external/slangpy").decode().strip()
    subprocess.run(
        ["git", "fetch", "--depth=1", "origin", reporting_ref], cwd=reporting, check=True
    )
    paths = [path for path in profiles["files"] if path not in selected.get("omit", [])]
    for path in paths:
        if path.startswith("falcor2/testing/benchmark/reporting/"):
            data = read(
                reporting, "show", f"{reporting_ref}:slangpy/testing/benchmark/{Path(path).name}"
            )
        else:
            data = read(repository, "show", f"{base}:{path}")
        target = repository / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    subprocess.run(["git", "add", "--", *paths], cwd=repository, check=True)
    patches = profiles["common_patches"] + selected["patches"]
    if patches:
        subprocess.run(
            [
                "git",
                "apply",
                "--index",
                *[str(PROFILES.parent / "patches" / name) for name in patches],
            ],
            cwd=repository,
            check=True,
        )
