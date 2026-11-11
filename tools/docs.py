# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build wheel-versioned documentation and assemble a cumulative GitLab Pages site."""

import argparse
from email.parser import BytesParser
from html import escape
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile

_ROOT = Path(__file__).resolve().parents[1]


def _validate_version(version: str) -> str:
    if not re.fullmatch(r"[0-9][A-Za-z0-9.+_-]*", version):
        raise ValueError(f"Invalid documentation version: {version!r}")
    return version


def _wheel_version(wheelhouse: Path) -> str:
    wheels = list(wheelhouse.glob("falcor2-*.whl"))
    if len(wheels) != 1:
        raise ValueError(f"Expected one Falcor2 wheel in {wheelhouse}, found {len(wheels)}")
    with zipfile.ZipFile(wheels[0]) as wheel:
        metadata = [name for name in wheel.namelist() if name.endswith(".dist-info/METADATA")]
        if len(metadata) != 1:
            raise ValueError("Expected one distribution metadata file in Falcor2 wheel")
        headers = BytesParser().parsebytes(wheel.read(metadata[0]))
    if headers["Name"] != "falcor2" or not headers["Version"]:
        raise ValueError("Missing Falcor2 name/version in wheel metadata")
    return _validate_version(headers["Version"])


def build_docs(wheelhouse: Path) -> None:
    """Document current source bindings, labeling the site with the actual wheel version."""
    version = _wheel_version(wheelhouse)
    version_file = _ROOT / "docs/_build/version.txt"
    version_file.unlink(missing_ok=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "sphinx",
            "-M",
            "html",
            "docs",
            "docs/_build",
            "-E",
            "-a",
            "-D",
            f"release={version}",
        ],
        cwd=_ROOT,
        check=True,
    )
    version_file.write_text(version + "\n", encoding="utf-8")


def _restore_history(archive: Path, target: Path, replacing: str) -> dict[str, int]:
    with zipfile.ZipFile(archive) as source:
        manifest = json.loads(source.read("versions.json"))
        versions = manifest["versions"]
        if not isinstance(versions, dict) or not versions or manifest["latest"] not in versions:
            raise ValueError("Invalid documentation history manifest")
        for version, pipeline in versions.items():
            _validate_version(version)
            if type(pipeline) is not int or pipeline < 1:
                raise ValueError("Invalid pipeline number in documentation history")
            if not source.read(f"{version}/index.html").strip():
                raise ValueError(f"Missing index in archived documentation {version}")
        for entry in source.infolist():
            path = PurePosixPath(entry.orig_filename)
            if (
                path.is_absolute()
                or ".." in path.parts
                or "\\" in entry.orig_filename
                or ":" in entry.orig_filename
                or (entry.external_attr >> 16) & 0o170000 == 0o120000
            ):
                raise ValueError(f"Unsafe documentation archive path: {entry.filename!r}")
            # Rebuild latest and the root index; replace this pipeline's own version on retry.
            if path.parts and path.parts[0] in versions and path.parts[0] != replacing:
                source.extract(entry, target)
    return versions


def stage_site(
    html: Path,
    version: str,
    pipeline_iid: int,
    output: Path,
    archive: Path,
    previous: Path | None = None,
) -> None:
    """Preserve historical versions, update latest, and write a complete deployment archive."""
    _validate_version(version)
    if pipeline_iid < 1:
        raise ValueError("Pipeline number must be positive")
    if not (html / "index.html").read_bytes().strip():
        raise ValueError("Documentation index.html is empty")
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to reuse an existing Pages output directory: {output}")
    archive = archive.resolve()
    if archive.is_relative_to(output):
        raise ValueError("The history archive must be outside the Pages output directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="falcor2-pages-", dir=output.parent) as temporary:
        staged = Path(temporary) / "public"
        staged.mkdir()
        versions = _restore_history(previous, staged, version) if previous else {}
        if version in versions and versions[version] != pipeline_iid:
            raise ValueError(f"Documentation version {version} belongs to another pipeline")
        if pipeline_iid in versions.values() and versions.get(version) != pipeline_iid:
            raise ValueError("Pipeline number already belongs to another documentation version")
        shutil.copytree(html, staged / version)
        versions[version] = pipeline_iid
        latest = max(versions, key=versions.__getitem__)
        shutil.copytree(staged / latest, staged / "latest")
        (staged / "versions.json").write_text(
            json.dumps({"latest": latest, "versions": versions}, indent=2) + "\n",
            encoding="utf-8",
        )
        links = "\n".join(
            f'<li><a href="{escape(item)}/">{escape(item)}</a></li>'
            for item in sorted(versions, key=versions.__getitem__, reverse=True)
        )
        (staged / "index.html").write_text(
            '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            "<title>Falcor2 documentation</title></head><body>"
            '<h1>Falcor2 documentation</h1><p><a href="latest/">Latest documentation</a></p>'
            f"<ul>{links}</ul></body></html>\n",
            encoding="utf-8",
        )
        archive.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as saved:
            for path in sorted(staged.rglob("*")):
                if path.is_file():
                    saved.write(path, path.relative_to(staged).as_posix())
        staged.rename(output)
    print(f"Documentation staged at {output}: {version}; latest is {latest}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="Build HTML using a wheel's release version")
    build.add_argument("--wheelhouse", type=Path, required=True)
    stage = commands.add_parser("stage", help="Assemble a versioned Pages tree and history archive")
    stage.add_argument("--html", type=Path, default=_ROOT / "docs/_build/html")
    stage.add_argument("--version-file", type=Path, default=_ROOT / "docs/_build/version.txt")
    stage.add_argument("--pipeline-iid", type=int, required=True)
    stage.add_argument("--output", type=Path, default=_ROOT / "public")
    stage.add_argument("--archive", type=Path, default=_ROOT / ".tmp/docs-pages/site.zip")
    stage.add_argument("--previous", type=Path)
    args = parser.parse_args()
    if args.command == "build":
        build_docs(args.wheelhouse)
    else:
        stage_site(
            args.html,
            args.version_file.read_text(encoding="utf-8").strip(),
            args.pipeline_iid,
            args.output,
            args.archive,
            args.previous,
        )


if __name__ == "__main__":
    main()
