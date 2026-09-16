# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build and publish a static visual report for pytest image comparisons."""

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import html
import json
import mimetypes
import os
from pathlib import Path
import re
import shutil
from typing import Any, Dict, List, Optional

import falcor2.testing.helpers as helpers


BUCKET_NAME = "nvr-ci-imagetests"
PUBLIC_BUCKET_URL = "https://pdx.s8k.io/v1/AUTH_team-nvr-ci/nvr-ci-imagetests"
_SAFE_JOB_ID = re.compile(r"^[A-Za-z0-9._-]+$")


@dataclass(frozen=True)
class ImageResult:
    """One image comparison captured by the pytest plugin."""

    id: str
    test_name: str
    test_file: str
    metadata_file: str
    status: str
    difference: Optional[float]
    tolerance: float
    current_hash: str
    reference_hash: Optional[str]
    message: Optional[str]
    assets: Dict[str, Optional[str]]

    @classmethod
    def from_json(cls, path: Path) -> "ImageResult":
        with open(path, "r") as file:
            data = json.load(file)
        if data.get("schema_version") != 1:
            raise ValueError(f"Unsupported image result schema in {path}")
        return cls(
            id=str(data["id"]),
            test_name=str(data["test_name"]),
            test_file=str(data["test_file"]),
            metadata_file=str(data["metadata_file"]),
            status=str(data["status"]),
            difference=(float(data["difference"]) if data.get("difference") is not None else None),
            tolerance=float(data["tolerance"]),
            current_hash=str(data["current_hash"]),
            reference_hash=(
                str(data["reference_hash"]) if data.get("reference_hash") is not None else None
            ),
            message=str(data["message"]) if data.get("message") is not None else None,
            assets={
                str(name): str(value) if value is not None else None
                for name, value in dict(data["assets"]).items()
            },
        )


@dataclass(frozen=True)
class ReportContext:
    """GitLab job information displayed in the report."""

    job_id: str
    job_name: str
    job_url: Optional[str]
    commit_sha: str
    commit_url: Optional[str]
    test_exit_code: int
    generated_at: str


@dataclass(frozen=True)
class ReportSummary:
    """Counts and output path returned after report generation."""

    total: int
    passed: int
    failed: int
    generated: int
    index_path: Path


def load_records(input_dir: Path) -> List[ImageResult]:
    """Load complete worker records, ignoring interrupted temporary files."""
    record_dir = input_dir / "records"
    if not record_dir.exists():
        return []
    return [ImageResult.from_json(path) for path in sorted(record_dir.glob("*.json"))]


def _has_failures(input_dir: Path) -> bool:
    """Return whether any captured image comparison failed."""
    return any(record.status == "failed" for record in load_records(input_dir))


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _format_difference(value: Optional[float]) -> str:
    return "not available" if value is None else f"{value:.6f}"


def _reference_url(reference_hash: str) -> str:
    return f"{PUBLIC_BUCKET_URL}/references/{reference_hash}.exr"


def _asset_img(source: Optional[str], label: str, detail: bool) -> str:
    if source is None:
        return '<div class="missing">Not available</div>'
    escaped_source = _escape(source)
    image_class = "detail-image" if detail else "thumbnail"
    return (
        f'<a href="{escaped_source}">'
        f'<img class="{image_class}" src="{escaped_source}" alt="{_escape(label)}" loading="lazy">'
        "</a>"
    )


def _status_label(status: str) -> str:
    if status == "passed":
        return "PASS"
    if status == "failed":
        return "FAIL"
    if status == "generated":
        return "GENERATED"
    return status.upper()


def _page_shell(title: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_escape(title)}</title>
<style>
:root {{
  color-scheme: dark;
  --bg: #11151b;
  --panel: #1a2029;
  --panel-2: #222b37;
  --text: #eef3f8;
  --muted: #aab7c4;
  --pass: #48c78e;
  --fail: #ff6685;
  --generated: #e9b949;
  --border: #344154;
  --link: #7dc4ff;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif;
}}
main {{ width: min(1500px, calc(100% - 32px)); margin: 24px auto 64px; }}
a {{ color: var(--link); }}
h1, h2, h3 {{ line-height: 1.2; }}
.meta, .summary {{ color: var(--muted); }}
.summary {{ display: flex; gap: 12px; flex-wrap: wrap; margin: 18px 0 28px; }}
.count {{
  border: 1px solid var(--border);
  border-radius: 999px;
  padding: 5px 12px;
  background: var(--panel);
}}
.cards {{
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(360px, 1fr));
  gap: 16px;
}}
.card, .panel {{
  border: 1px solid var(--border);
  border-radius: 10px;
  background: var(--panel);
  overflow: hidden;
}}
.card.failed {{ border-color: var(--fail); }}
.card-header {{ padding: 12px 14px 8px; }}
.card h3 {{ margin: 0 0 5px; font-size: 15px; overflow-wrap: anywhere; }}
.status {{ font-weight: 700; letter-spacing: 0.04em; }}
.status.passed {{ color: var(--pass); }}
.status.failed {{ color: var(--fail); }}
.status.generated {{ color: var(--generated); }}
.thumb-row {{
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 2px;
  background: var(--border);
}}
.thumbnail {{
  display: block;
  width: 100%;
  height: 180px;
  object-fit: contain;
  background: #080a0d;
}}
.metrics {{ padding: 10px 14px 13px; color: var(--muted); }}
.failure-message {{
  color: #ffd5de;
  background: #391923;
  border-radius: 6px;
  padding: 8px 10px;
  margin-top: 8px;
  overflow-wrap: anywhere;
}}
.section {{ margin-top: 36px; }}
.comparison-grid {{
  display: grid;
  grid-template-columns: repeat(3, minmax(300px, 1fr));
  gap: 16px;
  align-items: start;
}}
.panel h2 {{ margin: 0; padding: 12px 14px; font-size: 17px; }}
.image-scroll {{ overflow: auto; max-height: 80vh; background: #080a0d; }}
.detail-image {{ display: block; max-width: 100%; height: auto; margin: auto; }}
.missing {{ padding: 40px; text-align: center; color: var(--muted); }}
code {{ overflow-wrap: anywhere; }}
@media (max-width: 1000px) {{
  .comparison-grid {{ grid-template-columns: 1fr; }}
}}
</style>
</head>
<body>
<main>
{body}
</main>
</body>
</html>
"""


def _card_html(record: ImageResult) -> str:
    current = record.assets.get("current")
    reference = record.assets.get("reference")
    difference = record.assets.get("diff")
    message = (
        f'<div class="failure-message">{_escape(record.message)}</div>'
        if record.message is not None
        else ""
    )
    return f"""
<article class="card {_escape(record.status)}">
  <div class="card-header">
    <h3><a href="tests/{_escape(record.id)}.html">{_escape(record.test_name)}</a></h3>
    <span class="status {_escape(record.status)}">{_status_label(record.status)}</span>
  </div>
  <div class="thumb-row">
    {_asset_img(current, "Current image", detail=False)}
    {_asset_img(reference, "Reference image", detail=False)}
    {_asset_img(difference, "Difference image", detail=False)}
  </div>
  <div class="metrics">
    Difference: {_format_difference(record.difference)}; tolerance: {record.tolerance:.6f}
    {message}
  </div>
</article>
"""


def _detail_html(
    record: ImageResult,
    context: ReportContext,
) -> str:
    assets = {
        name: f"../{path}" if path is not None else None for name, path in record.assets.items()
    }
    reference_link = (
        f'<a href="{_escape(_reference_url(record.reference_hash))}">Download reference EXR</a>'
        if record.reference_hash is not None
        else "No reference EXR is available."
    )
    message = (
        f'<div class="failure-message">{_escape(record.message)}</div>'
        if record.message is not None
        else ""
    )
    body = f"""
<p><a href="../index.html">Back to all image tests</a></p>
<h1>{_escape(record.test_name)}</h1>
<p>
  <span class="status {_escape(record.status)}">{_status_label(record.status)}</span>
  &middot; Difference {_format_difference(record.difference)}
  &middot; Tolerance {record.tolerance:.6f}
</p>
{message}
<div class="comparison-grid section">
  <section class="panel">
    <h2>Current</h2>
    <div class="image-scroll">{_asset_img(assets.get("current"), "Current image", True)}</div>
  </section>
  <section class="panel">
    <h2>Reference</h2>
    <div class="image-scroll">{_asset_img(assets.get("reference"), "Reference image", True)}</div>
  </section>
  <section class="panel">
    <h2>Difference (10x)</h2>
    <div class="image-scroll">{_asset_img(assets.get("diff"), "Difference image", True)}</div>
  </section>
</div>
<section class="section">
  <h2>Comparison data</h2>
  <p>{reference_link}</p>
  <p class="meta">
    Current hash: <code>{_escape(record.current_hash)}</code><br>
    Reference hash: <code>{_escape(record.reference_hash or "none")}</code><br>
    Metadata: <code>{_escape(record.metadata_file)}</code><br>
    GitLab job: {_escape(context.job_name)} ({_escape(context.job_id)})
  </p>
</section>
"""
    return _page_shell(f"{record.test_name} - image test", body)


def _index_html(
    records: List[ImageResult],
    context: ReportContext,
    summary: ReportSummary,
) -> str:
    failures = [record for record in records if record.status == "failed"]
    all_records = sorted(records, key=lambda record: record.test_name.lower())
    failure_section = (
        '<section class="section"><h2>Failures</h2><div class="cards">'
        + "".join(_card_html(record) for record in failures)
        + "</div></section>"
        if failures
        else '<section class="section"><h2>Failures</h2><p>No image comparisons failed.</p></section>'
    )
    job_link = (
        f'<a href="{_escape(context.job_url)}">{_escape(context.job_name)}</a>'
        if context.job_url
        else _escape(context.job_name)
    )
    commit_link = (
        f'<a href="{_escape(context.commit_url)}"><code>{_escape(context.commit_sha[:12])}</code></a>'
        if context.commit_url
        else f"<code>{_escape(context.commit_sha[:12])}</code>"
    )
    body = f"""
<h1>Falcor2 image-test report</h1>
<p class="meta">
  Job {job_link} ({_escape(context.job_id)}) &middot;
  Commit {commit_link} &middot;
  Pytest exit code {context.test_exit_code} &middot;
  Generated {_escape(context.generated_at)}
</p>
<div class="summary">
  <span class="count">{summary.total} total</span>
  <span class="count">{summary.failed} failed</span>
  <span class="count">{summary.passed} passed</span>
  <span class="count">{summary.generated} generated</span>
</div>
{failure_section}
<section class="section">
  <h2>All image comparisons</h2>
  <div class="cards">{"".join(_card_html(record) for record in all_records)}</div>
</section>
"""
    return _page_shell("Falcor2 image-test report", body)


def _reset_output_directory(output_dir: Path) -> None:
    resolved = output_dir.resolve()
    if resolved == Path(resolved.anchor) or resolved == Path.cwd().resolve():
        raise ValueError(f"Refusing to replace unsafe report directory: {resolved}")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)


def _copy_record_assets(input_dir: Path, output_dir: Path, record: ImageResult) -> ImageResult:
    copied_assets: Dict[str, Optional[str]] = {}
    input_root = input_dir.resolve()
    for name, relative_path in record.assets.items():
        if relative_path is None:
            copied_assets[name] = None
            continue
        source = (input_dir / relative_path).resolve()
        try:
            source.relative_to(input_root)
        except ValueError as exc:
            raise ValueError(f"Image result asset escapes input directory: {source}") from exc
        if not source.is_file():
            copied_assets[name] = None
            continue
        destination = output_dir / "assets" / record.id / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        copied_assets[name] = destination.relative_to(output_dir).as_posix()
    return ImageResult(
        id=record.id,
        test_name=record.test_name,
        test_file=record.test_file,
        metadata_file=record.metadata_file,
        status=record.status,
        difference=record.difference,
        tolerance=record.tolerance,
        current_hash=record.current_hash,
        reference_hash=record.reference_hash,
        message=record.message,
        assets=copied_assets,
    )


def build_report(
    input_dir: Path,
    output_dir: Path,
    context: ReportContext,
) -> ReportSummary:
    """Build a complete static site from captured worker records."""
    raw_records = load_records(input_dir)
    _reset_output_directory(output_dir)
    records = [_copy_record_assets(input_dir, output_dir, record) for record in raw_records]

    tests_dir = output_dir / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    for record in records:
        detail_path = tests_dir / f"{record.id}.html"
        detail_path.write_text(
            _detail_html(record, context),
            encoding="utf-8",
        )

    summary = ReportSummary(
        total=len(records),
        passed=sum(record.status == "passed" for record in records),
        failed=sum(record.status == "failed" for record in records),
        generated=sum(record.status == "generated" for record in records),
        index_path=output_dir / "index.html",
    )
    summary.index_path.write_text(
        _index_html(records, context, summary),
        encoding="utf-8",
    )
    return summary


def upload_report(
    site_dir: Path,
    job_id: str,
    client: Any,
    bucket_name: str = BUCKET_NAME,
    public_bucket_url: str = PUBLIC_BUCKET_URL,
) -> str:
    """Upload a static site below results/<job id>, with index.html last."""
    if not _SAFE_JOB_ID.fullmatch(job_id):
        raise ValueError(f"Unsafe GitLab job id: {job_id!r}")
    if not (site_dir / "index.html").is_file():
        raise FileNotFoundError(f"Report index not found: {site_dir / 'index.html'}")

    files = [path for path in site_dir.rglob("*") if path.is_file()]
    files.sort(key=lambda path: (path.name == "index.html", path.as_posix()))
    prefix = f"results/{job_id}"
    for path in files:
        relative_path = path.relative_to(site_dir).as_posix()
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        cache_control = "no-cache" if path.suffix.lower() == ".html" else "public, max-age=31536000"
        client.upload_file(
            str(path),
            bucket_name,
            f"{prefix}/{relative_path}",
            ExtraArgs={
                "ContentType": content_type,
                "CacheControl": cache_control,
            },
        )
    return f"{public_bucket_url}/{prefix}/index.html"


def _default_commit_url(commit_sha: str) -> Optional[str]:
    project_url = os.environ.get("CI_PROJECT_URL")
    if not project_url or not commit_sha:
        return None
    return f"{project_url}/-/commit/{commit_sha}"


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("reports/image-tests/raw"))
    parser.add_argument("--output", type=Path, default=Path("reports/image-tests/site"))
    parser.add_argument("--job-id", default=os.environ.get("CI_JOB_ID", "local"))
    parser.add_argument("--job-name", default=os.environ.get("CI_JOB_NAME", "local image tests"))
    parser.add_argument("--job-url", default=os.environ.get("CI_JOB_URL"))
    parser.add_argument("--commit-sha", default=os.environ.get("CI_COMMIT_SHA", "unknown"))
    parser.add_argument("--commit-url", default=None)
    parser.add_argument("--test-exit-code", type=int, default=0)
    parser.add_argument("--upload", action="store_true")
    parser.add_argument(
        "--skip-if-no-failures",
        action="store_true",
        help="Skip report generation when no failed image comparisons were captured",
    )
    return parser


def main() -> None:
    args = create_parser().parse_args()
    if args.skip_if_no_failures and not _has_failures(args.input):
        print("Skipping image test report because no failed image comparisons were captured.")
        return

    commit_url = args.commit_url or _default_commit_url(args.commit_sha)
    context = ReportContext(
        job_id=args.job_id,
        job_name=args.job_name,
        job_url=args.job_url,
        commit_sha=args.commit_sha,
        commit_url=commit_url,
        test_exit_code=args.test_exit_code,
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    summary = build_report(
        input_dir=args.input,
        output_dir=args.output,
        context=context,
    )
    print(
        f"Generated image test report with {summary.total} comparisons "
        f"({summary.failed} failed)."
    )

    if not args.upload:
        print(f"Image test report: {summary.index_path.resolve()}")
        return

    client = helpers.create_boto3_client()
    try:
        report_url = upload_report(args.output, args.job_id, client)
    finally:
        close = getattr(client, "close", None)
        if close is not None:
            close()
    print(f"Image test report: {report_url}")


if __name__ == "__main__":
    main()
