# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
import subprocess
import zipfile

import pytest

from tools import docs


def _site(root: Path, text: str) -> Path:
    root.mkdir()
    (root / "index.html").write_text(text, encoding="utf-8")
    (root / "_static").mkdir()
    (root / "_static/style.css").write_text(text, encoding="utf-8")
    return root


def _files(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def _wheel(root: Path, version: str) -> Path:
    path = root / f"falcor2-{version}-cp312-cp312-linux_x86_64.whl"
    with zipfile.ZipFile(path, "w") as wheel:
        wheel.writestr(
            f"falcor2-{version}.dist-info/METADATA", f"Name: falcor2\nVersion: {version}\n"
        )
    return path


def test_releases_retain_history_replace_latest_and_allow_retries(tmp_path: Path) -> None:
    html = _site(tmp_path / "html", "first")
    (html / "removed.html").write_text("old page", encoding="utf-8")
    first = "0.1.0+rtr.100"
    second = "0.1.0+rtr.101"
    docs.stage_site(html, first, 100, tmp_path / "public1", tmp_path / "first.zip")
    original = _files(tmp_path / "public1" / first)
    assert original == _files(tmp_path / "public1/latest")
    (html / "removed.html").unlink()
    (html / "index.html").write_text("second", encoding="utf-8")
    docs.stage_site(
        html, second, 101, tmp_path / "public2", tmp_path / "second.zip", tmp_path / "first.zip"
    )
    assert original == _files(tmp_path / "public2" / first)
    assert _files(html) == _files(tmp_path / "public2/latest")
    assert _files(html) == _files(tmp_path / "public2" / second)
    assert not (tmp_path / "public2/latest/removed.html").exists()
    for version in (first, second):
        assert f'href="{version}/"' in (tmp_path / "public2/index.html").read_text()

    # A retried publication replaces its own complete tree without retaining removed pages.
    (html / "_static/style.css").unlink()
    docs.stage_site(
        html, second, 101, tmp_path / "retry", tmp_path / "retry.zip", tmp_path / "second.zip"
    )
    assert _files(tmp_path / "retry/latest") == _files(html)
    assert _files(tmp_path / "retry" / first) == original
    with zipfile.ZipFile(tmp_path / "retry.zip") as archive:
        assert {name: archive.read(name) for name in archive.namelist()} == _files(
            tmp_path / "retry"
        )


def test_delayed_pipeline_adds_version_without_rolling_latest_back(tmp_path: Path) -> None:
    html = _site(tmp_path / "html", "newer")
    docs.stage_site(html, "0.1.0+rtr.20", 20, tmp_path / "first", tmp_path / "first.zip")
    (html / "index.html").write_text("older", encoding="utf-8")
    docs.stage_site(
        html,
        "0.1.0+rtr.19",
        19,
        tmp_path / "second",
        tmp_path / "second.zip",
        tmp_path / "first.zip",
    )
    assert (tmp_path / "second/latest/index.html").read_text() == "newer"
    assert (tmp_path / "second/0.1.0+rtr.19/index.html").read_text() == "older"
    assert json.loads((tmp_path / "second/versions.json").read_text())["latest"] == "0.1.0+rtr.20"


@pytest.mark.parametrize(
    "version", ["../escape", "/tmp/site", "latest", "0.1/a", "0.1\\a", "0.1:bad", ""]
)
def test_invalid_version_never_creates_site(tmp_path: Path, version: str) -> None:
    html = _site(tmp_path / "html", "site")
    with pytest.raises(ValueError, match="Invalid documentation version"):
        docs.stage_site(html, version, 1, tmp_path / "public", tmp_path / "site.zip")
    assert not (tmp_path / "public").exists()
    assert not (tmp_path / "site.zip").exists()


@pytest.mark.parametrize(
    "entry", ["../escape", "/absolute", "0.1/../../escape", "0.1\\escape", "C:/escape"]
)
def test_archive_path_traversal_fails_before_publication(tmp_path: Path, entry: str) -> None:
    html = _site(tmp_path / "html", "site")
    archive = tmp_path / "previous.zip"
    with zipfile.ZipFile(archive, "w") as saved:
        saved.writestr("versions.json", json.dumps({"latest": "0.1", "versions": {"0.1": 1}}))
        saved.writestr("0.1/index.html", "old")
        info = zipfile.ZipInfo()
        info.filename = entry  # Preserve backslashes that ZipInfo normalizes on Windows.
        saved.writestr(info, "bad")
    with pytest.raises(ValueError, match="Unsafe documentation archive path"):
        docs.stage_site(html, "0.2", 2, tmp_path / "public", tmp_path / "site.zip", archive)
    assert not (tmp_path / "public").exists()
    assert not (tmp_path / "site.zip").exists()
    assert not (tmp_path / "escape").exists()


def test_missing_html_corrupt_history_and_existing_output_fail(tmp_path: Path) -> None:
    html = tmp_path / "html"
    with pytest.raises(FileNotFoundError):
        docs.stage_site(html, "0.1", 1, tmp_path / "public", tmp_path / "site.zip")
    _site(html, "site")
    previous = tmp_path / "previous.zip"
    previous.write_bytes(b"not a zip archive")
    with pytest.raises(zipfile.BadZipFile):
        docs.stage_site(html, "0.1", 1, tmp_path / "public", tmp_path / "site.zip", previous)
    assert not (tmp_path / "public").exists()
    _site(tmp_path / "public", "preserved")
    with pytest.raises(FileExistsError, match="Refusing to reuse"):
        docs.stage_site(html, "0.1", 1, tmp_path / "public", tmp_path / "site.zip")
    assert (tmp_path / "public/index.html").read_text() == "preserved"


def test_version_from_wheel_metadata_includes_release_suffix(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Expected one Falcor2 wheel"):
        docs._wheel_version(tmp_path)
    _wheel(tmp_path, "0.1.0+rtr.321")
    assert docs._wheel_version(tmp_path) == "0.1.0+rtr.321"
    _wheel(tmp_path, "0.1.0+rtr.322")
    with pytest.raises(ValueError, match="Expected one Falcor2 wheel"):
        docs._wheel_version(tmp_path)


def test_failed_build_does_not_leave_publishable_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(docs, "_ROOT", tmp_path)
    _wheel(tmp_path, "0.1.0+rtr.321")
    version_file = tmp_path / "docs/_build/version.txt"
    version_file.parent.mkdir(parents=True)
    version_file.write_text("stale version", encoding="utf-8")

    def fail_build(*args: object, **kwargs: object) -> None:
        raise subprocess.CalledProcessError(1, "sphinx")

    monkeypatch.setattr(docs.subprocess, "run", fail_build)
    with pytest.raises(subprocess.CalledProcessError):
        docs.build_docs(tmp_path)
    assert not version_file.exists()
