# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare submodules for CI's experimental fetch-based checkout.

GitLab Runner fetches only the superproject. This helper first resets, cleans,
and updates the cached recursive submodule graph. If that fails, it discards
the submodule worktrees and Git metadata and retries once from clean state.
Keeping that recovery policy in one file makes the fetch experiment removable
as a unit if CI needs to return to clone-based checkout.
"""

import configparser
import os
from pathlib import Path
import shlex
import shutil
import stat
import subprocess
import sys
from collections.abc import Callable, Sequence
from types import TracebackType
from typing import Any, Optional


GitConfig = tuple[str, ...]
VCPKG_PATH = Path("external/slangpy/external/vcpkg")


def _run_git(
    arguments: Sequence[str],
    repo_root: Path,
    git_config: GitConfig = (),
    *,
    check: bool = True,
    capture_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    command = ["git"]
    for entry in git_config:
        command.extend(("-c", entry))
    command.extend(arguments)
    if not capture_output:
        print(f"+ {shlex.join(command)}", flush=True)
    return subprocess.run(
        command,
        cwd=repo_root,
        check=check,
        stdout=subprocess.PIPE if capture_output else None,
        stderr=subprocess.PIPE if capture_output else None,
        text=True,
    )


def _external_git_config(repo_root: Path) -> GitConfig:
    result = _run_git(
        ("config", "--get-all", "include.path"),
        repo_root,
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        return ()
    return tuple(f"include.path={path}" for path in result.stdout.splitlines() if path)


def _submodule_paths(repo_root: Path) -> tuple[Path, ...]:
    gitmodules_path = repo_root / ".gitmodules"
    parser = configparser.ConfigParser(interpolation=None)
    with gitmodules_path.open(encoding="utf-8") as stream:
        parser.read_file(stream)

    root = repo_root.resolve()
    paths: list[Path] = []
    for section in parser.sections():
        if not section.startswith("submodule ") or not parser.has_option(section, "path"):
            continue
        relative_path = Path(parser.get(section, "path"))
        resolved_path = (root / relative_path).resolve()
        if (
            relative_path.is_absolute()
            or resolved_path == root
            or root not in resolved_path.parents
        ):
            raise ValueError(f"Submodule path is outside the repository: {relative_path}")
        paths.append(resolved_path)
    return tuple(paths)


def _make_writable_and_retry(
    function: Callable[..., Any],
    path: str,
    _exc_info: tuple[type[BaseException], BaseException, Optional[TracebackType]],
) -> None:
    os.chmod(path, stat.S_IWRITE)
    function(path)


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path, onerror=_make_writable_and_retry)


def _update(repo_root: Path, git_config: GitConfig, depth: int, clean: bool) -> None:
    _run_git(("submodule", "sync", "--recursive"), repo_root, git_config)
    if clean:
        _run_git(
            ("submodule", "foreach", "--recursive", "git reset --hard"),
            repo_root,
            git_config,
        )
        _run_git(
            ("submodule", "foreach", "--recursive", "git clean -ffdx"),
            repo_root,
            git_config,
        )

    arguments = ["submodule", "update", "--init", "--recursive", "--force"]
    if depth > 0:
        arguments.extend(("--depth", str(depth)))
    _run_git(arguments, repo_root, git_config)


def _discard_submodules(repo_root: Path, git_config: GitConfig) -> None:
    submodule_paths = _submodule_paths(repo_root)
    _run_git(
        ("submodule", "deinit", "--all", "--force"),
        repo_root,
        git_config,
        check=False,
    )
    for path in submodule_paths:
        _remove_path(path)

    git_dir = repo_root / ".git"
    if not git_dir.is_dir():
        raise RuntimeError(f"Refusing to remove an unexpected Git directory: {git_dir}")
    _remove_path(git_dir / "modules")


def update_submodules(repo_root: Path, depth: int) -> None:
    if depth < 0:
        raise ValueError("Submodule depth must not be negative")

    repo_root = repo_root.resolve()
    git_config = _external_git_config(repo_root)
    try:
        _update(repo_root, git_config, depth, clean=True)
        return
    except subprocess.CalledProcessError as error:
        print(
            f"Cached submodule update failed with exit code {error.returncode}; "
            "discarding the submodule cache and retrying.",
            file=sys.stderr,
            flush=True,
        )

    _discard_submodules(repo_root, git_config)
    _update(repo_root, git_config, depth, clean=False)


def ensure_full_history(repo_root: Path, relative_path: Path) -> None:
    repository_path = repo_root.resolve() / relative_path
    if not repository_path.is_dir():
        raise RuntimeError(f"Submodule repository does not exist: {repository_path}")

    git_config = _external_git_config(repo_root.resolve())
    result = _run_git(
        ("rev-parse", "--is-shallow-repository"),
        repository_path,
        git_config,
        capture_output=True,
    )
    if result.stdout.strip() == "true":
        _run_git(("fetch", "--unshallow"), repository_path, git_config)


def main() -> int:
    depth_text = os.environ.get("GIT_SUBMODULE_DEPTH", "1")
    try:
        depth = int(depth_text)
    except ValueError as error:
        raise ValueError(f"Invalid GIT_SUBMODULE_DEPTH value: {depth_text}") from error
    repo_root = Path.cwd()
    update_submodules(repo_root, depth)
    # Vcpkg versioning requires commit history; all other submodules stay shallow.
    ensure_full_history(repo_root, VCPKG_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
