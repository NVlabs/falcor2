# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import glob
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
from typing import Any, Optional

PROJECT_DIR = Path(__file__).resolve().parent.parent

RUST_MINIDUMP_VERSION = "0.26.1"
RUST_MINIDUMP_BASE_URL = (
    f"https://github.com/rust-minidump/rust-minidump/releases/download/v{RUST_MINIDUMP_VERSION}"
)

RUST_MINIDUMP_ASSETS = {
    ("Windows", "AMD64"): "minidump-stackwalk-x86_64-pc-windows-msvc.zip",
    ("Windows", "x86_64"): "minidump-stackwalk-x86_64-pc-windows-msvc.zip",
    ("Linux", "x86_64"): "minidump-stackwalk-x86_64-unknown-linux-gnu.tar.xz",
    ("Darwin", "x86_64"): "minidump-stackwalk-x86_64-apple-darwin.tar.xz",
    ("Darwin", "arm64"): "minidump-stackwalk-aarch64-apple-darwin.tar.xz",
    ("Darwin", "aarch64"): "minidump-stackwalk-aarch64-apple-darwin.tar.xz",
}

VALID_KINDS = {"native", "python"}


def database_dir(kind: str) -> Path:
    _check_kind(kind)
    return PROJECT_DIR / ".crashpad" / kind


def setup(kind: str) -> Path:
    database = database_dir(kind)
    shutil.rmtree(database, ignore_errors=True)
    database.mkdir(parents=True, exist_ok=True)
    return database


def notify_current_test(kind: str, name: str) -> None:
    database = database_dir(kind)
    database.mkdir(parents=True, exist_ok=True)
    path = database / f"{os.getpid()}.txt"
    path.write_text(f"{name}\n", encoding="utf-8")


def find_test_name(kind: str, pid: int) -> Optional[str]:
    path = database_dir(kind) / f"{pid}.txt"
    if not path.exists():
        return None

    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) == 0:
        return None
    return lines[-1].strip()


def report(kind: str, writer: Any = None) -> bool:
    database = database_dir(kind)
    _copy_posix_pending_reports(database)

    reports = sorted((database / "reports").glob("*.dmp"))
    if len(reports) == 0:
        return False

    _sep(writer, "=", f"CRASHPAD REPORTS ({kind})")
    _postprocess_reports(database)

    for index, report_path in enumerate(reports):
        _write(writer, f"Crash report {index}: {report_path.name}\n")

        report_json_path = report_path.with_suffix(".json")
        if report_json_path.exists():
            try:
                report_json = json.loads(report_json_path.read_text(encoding="utf-8"))
                pid = int(report_json["pid"])
                test_name = find_test_name(kind, pid)
                if test_name:
                    _write(writer, f"Crash most probably originated in {test_name}\n")
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                pass

        report_txt_path = report_path.with_suffix(".txt")
        if report_txt_path.exists():
            _write(writer, report_txt_path.read_text(encoding="utf-8", errors="replace"))
            _sep(writer, "-")

    return True


def _check_kind(kind: str) -> None:
    if kind not in VALID_KINDS:
        raise ValueError(f"Unknown Crashpad report kind: {kind}")


def _copy_posix_pending_reports(database: Path) -> None:
    if os.name != "posix":
        return

    reports_dir = database / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    for file in glob.glob(str(database / "pending" / "*.dmp")):
        shutil.copy2(file, reports_dir / Path(file).name)


def _postprocess_reports(database: Path) -> None:
    minidump_stackwalk = _get_minidump_stackwalk_path()
    if minidump_stackwalk is None:
        system = platform.system()
        machine = platform.machine()
        print(
            f"minidump-stackwalk is not available for {system}/{machine}. "
            "Skipping crash report postprocessing."
        )
        return

    for report_path in sorted((database / "reports").glob("*.dmp")):
        report_txt = report_path.with_suffix(".txt")
        report_json = report_path.with_suffix(".json")
        subprocess.run(
            [
                str(minidump_stackwalk),
                str(report_path),
                "--brief",
                "--use-local-debuginfo",
                "--output-file",
                str(report_txt),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        subprocess.run(
            [
                str(minidump_stackwalk),
                str(report_path),
                "--json",
                "--pretty",
                "--use-local-debuginfo",
                "--output-file",
                str(report_json),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )


def _get_minidump_stackwalk_path() -> Optional[Path]:
    system = platform.system()
    machine = platform.machine()
    asset_name = _get_minidump_asset(system, machine)
    if asset_name is None:
        return None

    exe_name = "minidump-stackwalk.exe" if system == "Windows" else "minidump-stackwalk"
    exe_path = PROJECT_DIR / ".tools" / "rust-minidump" / exe_name
    if exe_path.exists():
        return exe_path

    if not _download_minidump_stackwalk(asset_name, exe_path.parent):
        return None

    return exe_path if exe_path.exists() else None


def _get_minidump_asset(system: str, machine: str) -> Optional[str]:
    return RUST_MINIDUMP_ASSETS.get((system, machine))


def _download_minidump_stackwalk(asset_name: str, extract_dir: Path) -> bool:
    import tarfile
    import urllib.request
    import zipfile

    url = f"{RUST_MINIDUMP_BASE_URL}/{asset_name}"
    print(f"Downloading minidump-stackwalk from {url} ...")

    try:
        extract_dir.mkdir(parents=True, exist_ok=True)
        archive_path = extract_dir / asset_name
        urllib.request.urlretrieve(url, archive_path)

        if asset_name.endswith(".zip"):
            with zipfile.ZipFile(archive_path, "r") as archive:
                archive.extractall(extract_dir)
        elif asset_name.endswith(".tar.xz"):
            with tarfile.open(archive_path, "r:xz") as archive:
                archive.extractall(extract_dir, members=_strip_tar_member_prefix(archive))
        else:
            print(f"Unknown archive format: {asset_name}")
            return False

        archive_path.unlink()

        if platform.system() != "Windows":
            exe_path = extract_dir / "minidump-stackwalk"
            if exe_path.exists():
                exe_path.chmod(exe_path.stat().st_mode | 0o755)

        print(f"Successfully installed minidump-stackwalk to {extract_dir}")
        return True

    except Exception as exc:
        print(f"Failed to download minidump-stackwalk: {exc}")
        return False


def _strip_tar_member_prefix(archive: Any, strip: int = 1) -> list[Any]:
    members = []
    for member in archive.getmembers():
        path = Path(member.path)
        if len(path.parts) > strip:
            member.path = str(Path(*path.parts[strip:]))
            members.append(member)
    return members


def _write(writer: Any, text: str) -> None:
    if writer is None:
        print(text, end="")
    else:
        writer.write(text)


def _sep(writer: Any, separator: str, title: str = "") -> None:
    if writer is not None and hasattr(writer, "sep"):
        writer.sep(separator, title)
        return

    if title:
        print(f"{separator * 10} {title} {separator * 10}")
    else:
        print(separator * 40)
