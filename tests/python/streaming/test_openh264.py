# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import bz2
import hashlib
import io
from pathlib import Path

import pytest

from falcor2.streaming import openh264


@pytest.fixture
def runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, bytes]:
    library = b"test codec runtime"
    archive = bz2.compress(library)
    monkeypatch.setattr(openh264, "runtime_directory", lambda: tmp_path)
    monkeypatch.setattr(
        openh264,
        "_artifact",
        lambda: (
            "test-codec",
            hashlib.sha256(archive).hexdigest(),
            hashlib.sha256(library).hexdigest(),
        ),
    )
    return tmp_path, archive


def test_install_verified_and_idempotent(
    runtime: tuple[Path, bytes], monkeypatch: pytest.MonkeyPatch
) -> None:
    directory, archive = runtime
    requests = []

    def download(url: str, timeout: int) -> io.BytesIO:
        requests.append(url)
        assert timeout > 0
        return io.BytesIO(archive)

    monkeypatch.setattr(openh264.urllib.request, "urlopen", download)
    installed = openh264.install()
    assert installed.read_bytes() == bz2.decompress(archive)
    assert openh264.install() == installed
    assert requests == ["https://ciscobinary.openh264.org/test-codec.bz2"]
    assert list(directory.iterdir()) == [installed]


def test_corrupt_download_does_not_install(
    runtime: tuple[Path, bytes], monkeypatch: pytest.MonkeyPatch
) -> None:
    directory, _ = runtime
    monkeypatch.setattr(openh264.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(b"corrupt"))
    with pytest.raises(RuntimeError, match="SHA-256"):
        openh264.install()
    assert not list(directory.iterdir())


def test_install_uninstall_and_reinstall(
    runtime: tuple[Path, bytes], monkeypatch: pytest.MonkeyPatch
) -> None:
    directory, archive = runtime
    monkeypatch.setattr(openh264.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(archive))
    assert openh264.main(["install"]) == 0
    unrelated = directory / "another-library"
    unrelated.write_bytes(b"keep")
    assert openh264.main(["uninstall"]) == 0
    assert not (directory / "test-codec").exists()
    assert unrelated.read_bytes() == b"keep"
    assert openh264.main(["uninstall"]) == 0
    assert openh264.main(["install"]) == 0
    assert (directory / "test-codec").read_bytes() == bz2.decompress(archive)


def test_install_rejects_corrupt_existing_library(runtime: tuple[Path, bytes]) -> None:
    directory, _ = runtime
    (directory / "test-codec").write_bytes(b"corrupt")
    assert openh264.main(["install"]) == 1


def test_uninstall_reports_failure(
    runtime: tuple[Path, bytes], monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(path: Path, missing_ok: bool = False) -> None:
        raise PermissionError("Close active encoders first")

    monkeypatch.setattr(Path, "unlink", fail)
    assert openh264.main(["uninstall"]) == 1


def test_package_contains_binary_license() -> None:
    license_text = openh264.LICENSE_PATH.read_text(encoding="utf-8")
    assert openh264.ATTRIBUTION in license_text
    assert "http://www.openh264.org/" in license_text


@pytest.mark.parametrize("available", [False, True])
def test_check_runtime(
    available: bool, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls = []

    def check() -> bool:
        calls.append(True)
        return available

    def unexpected_download(*args: object, **kwargs: object) -> None:
        pytest.fail("Checking availability must not download the runtime")

    monkeypatch.setattr(openh264.f2.streaming.VideoEncoder, "check_available", check)
    monkeypatch.setattr(openh264.urllib.request, "urlopen", unexpected_download)
    assert openh264.main(["check"]) == (0 if available else 1)
    assert calls == [True]
    assert (
        "OpenH264 is available" if available else "OpenH264 is unavailable"
    ) in capsys.readouterr().out
