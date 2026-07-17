# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Image-set comparison and HTML report generation for material image runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import html
from pathlib import Path
import shutil
from typing import Any

import numpy as np
from slangpy import Bitmap

from examples.render_material.render_material_manifest import (
    RenderResult,
    read_manifest,
    read_results,
)


LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
STRUCTURAL_SEVERITY = {
    "missing_reference_entry": 100,
    "missing_tested_entry": 95,
    "missing_reference_image": 90,
    "missing_tested_image": 85,
    "load_error": 80,
    "shape_mismatch": 70,
    "no_finite_pixels": 60,
}


@dataclass(frozen=True)
class ImageMetrics:
    """Numeric RGB comparison metrics for one image pair."""

    rmse_rgb: float
    p95_abs_rgb: float
    max_abs_rgb: float
    mean_luma_bias: float
    finite_fraction: float


@dataclass(frozen=True)
class ComparisonReport:
    """Report rows and output location produced by `compare_image_sets`."""

    output_dir: Path
    rows: tuple[dict[str, Any], ...]


def compare_image_sets(
    reference_root: str | Path,
    tested_root: str | Path,
    output_dir: str | Path,
) -> ComparisonReport:
    """Compare two render directories and write visual HTML output."""

    reference_dir = Path(reference_root)
    tested_dir = Path(tested_root)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    reference_entries = {
        entry.entry_id: entry for entry in read_manifest(reference_dir / "manifest.json").entries
    }
    tested_entries = {
        entry.entry_id: entry for entry in read_manifest(tested_dir / "manifest.json").entries
    }
    reference_results = _result_map(reference_dir / "manifest.json")
    tested_results = _result_map(tested_dir / "manifest.json")

    entry_ids = sorted(set(reference_entries) | set(tested_entries))
    rows = [
        _build_row(
            entry_id,
            reference_dir=reference_dir,
            tested_dir=tested_dir,
            reference_entry=reference_entries.get(entry_id),
            tested_entry=tested_entries.get(entry_id),
            reference_result=reference_results.get(entry_id),
            tested_result=tested_results.get(entry_id),
        )
        for entry_id in entry_ids
    ]
    rows.sort(key=_sort_key)
    _write_outputs(out_dir, rows, reference_dir, tested_dir)
    return ComparisonReport(output_dir=out_dir, rows=tuple(rows))


def load_rgb(path: Path) -> np.ndarray:
    """Load an image as linear-ish float RGB data for metric computation."""

    bitmap = Bitmap.load_from_file(path)
    image = bitmap.convert(
        pixel_format=Bitmap.PixelFormat.rgba,
        component_type=Bitmap.ComponentType.float32,
    )
    data = np.asarray(image, copy=False)
    return data[:, :, :3].astype(np.float32, copy=False)


def compare_images(reference_rgb: np.ndarray, tested_rgb: np.ndarray) -> ImageMetrics | str:
    """Compare two RGB arrays, returning metrics or a structural error code."""

    if reference_rgb.shape != tested_rgb.shape:
        return "shape_mismatch"
    finite = np.isfinite(reference_rgb) & np.isfinite(tested_rgb)
    finite_count = int(np.count_nonzero(finite))
    if finite_count == 0:
        return "no_finite_pixels"
    delta = tested_rgb - reference_rgb
    finite_delta = delta[finite]
    abs_delta = np.abs(finite_delta)
    luma_delta = delta[:, :, :3] @ LUMA
    finite_luma = np.isfinite(luma_delta)
    return ImageMetrics(
        rmse_rgb=float(np.sqrt(np.mean(finite_delta * finite_delta))),
        p95_abs_rgb=float(np.percentile(abs_delta, 95.0)),
        max_abs_rgb=float(np.max(abs_delta)),
        mean_luma_bias=float(np.mean(luma_delta[finite_luma])) if np.any(finite_luma) else 0.0,
        finite_fraction=float(finite_count) / float(reference_rgb.size),
    )


def _build_row(
    entry_id: str,
    *,
    reference_dir: Path,
    tested_dir: Path,
    reference_entry: Any,
    tested_entry: Any,
    reference_result: RenderResult | None,
    tested_result: RenderResult | None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "entry_id": entry_id,
        "label": (
            (reference_entry or tested_entry).label if (reference_entry or tested_entry) else ""
        ),
        "provider": (
            (reference_entry or tested_entry).provider if (reference_entry or tested_entry) else ""
        ),
        "material_class": (
            (reference_entry or tested_entry).material_class
            if (reference_entry or tested_entry)
            else ""
        ),
        "output": _output_label(reference_entry or tested_entry),
        "status": "compared",
        "message": "",
        "reference_image": reference_result.image if reference_result else "",
        "tested_image": tested_result.image if tested_result else "",
    }
    if reference_entry is None:
        row["status"] = "missing_reference_entry"
        return row
    if tested_entry is None:
        row["status"] = "missing_tested_entry"
        return row
    if reference_result is None or not reference_result.image:
        row["status"] = "missing_reference_image"
        row["message"] = reference_result.message if reference_result else ""
        return row
    if tested_result is None or not tested_result.image:
        row["status"] = "missing_tested_image"
        row["message"] = tested_result.message if tested_result else ""
        return row
    reference_path = _find_source_image(reference_dir, reference_result.image)
    tested_path = _find_source_image(tested_dir, tested_result.image)
    if reference_path is None:
        row["status"] = "missing_reference_image"
        return row
    if tested_path is None:
        row["status"] = "missing_tested_image"
        return row
    try:
        metrics = compare_images(load_rgb(reference_path), load_rgb(tested_path))
    except Exception as exc:  # noqa: BLE001 - report rows should survive reader failures.
        row["status"] = "load_error"
        row["message"] = str(exc)
        return row
    if isinstance(metrics, str):
        row["status"] = metrics
        return row
    row.update(asdict(metrics))
    return row


def _result_map(path: Path) -> dict[str, RenderResult]:
    return {result.entry_id: result for result in read_results(path)}


def _output_label(entry: Any) -> str:
    if entry is None:
        return ""
    if entry.output.kind == "material":
        return "material"
    return f"{entry.output.kind}:{entry.output.name}"


def _sort_key(row: dict[str, Any]) -> tuple[int, float, float, str]:
    status = str(row.get("status", ""))
    if status != "compared":
        return (-STRUCTURAL_SEVERITY.get(status, 1), 0.0, 0.0, str(row.get("entry_id", "")))
    return (
        0,
        -float(row.get("rmse_rgb", 0.0)),
        -float(row.get("max_abs_rgb", 0.0)),
        str(row.get("entry_id", "")),
    )


def _write_outputs(
    output_dir: Path,
    rows: list[dict[str, Any]],
    reference_dir: Path,
    tested_dir: Path,
) -> None:
    assets_dir = output_dir / "html_assets"
    if assets_dir.exists():
        shutil.rmtree(assets_dir)
    assets_dir.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(rows, start=1):
        if row.get("status") == "compared":
            _write_preview_assets(reference_dir, tested_dir, assets_dir, index, row)
    _write_html(output_dir / "comparison.html", rows, reference_dir, tested_dir)


def _write_html(
    path: Path,
    rows: list[dict[str, Any]],
    reference_root: Path,
    tested_root: Path,
) -> None:
    """Write a card-based HTML report with summary metrics and visual comparison cards."""

    summary = _html_summary(rows)
    cards = "\n".join(_row_card(index, row) for index, row in enumerate(rows, start=1))
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Material Image Comparison</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #111318;
      --panel: #1b2028;
      --panel2: #202631;
      --text: #e8edf4;
      --muted: #9aa7b7;
      --line: #364253;
      --accent: #78b7ff;
      --warn: #f0b45d;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      padding: 24px;
      background: var(--bg);
      color: var(--text);
      font-family: Segoe UI, Arial, sans-serif;
      line-height: 1.4;
    }}
    header, .summary, .row-card {{
      max-width: 1480px;
      margin: 0 auto 18px auto;
    }}
    header h1 {{
      margin: 0 0 8px 0;
      font-size: 28px;
      font-weight: 650;
    }}
    header p, .summary p {{
      color: var(--muted);
      margin: 4px 0;
    }}
    code {{
      color: #d8e8ff;
      background: #141922;
      padding: 1px 4px;
      border-radius: 3px;
      overflow-wrap: anywhere;
    }}
    .summary {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
    }}
    .summary-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 10px;
      margin: 14px 0 0 0;
    }}
    .summary-grid div {{
      background: var(--panel2);
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px;
    }}
    .summary-grid span {{
      display: block;
      color: var(--muted);
      font-size: 12px;
    }}
    .summary-grid strong {{
      display: block;
      font-size: 22px;
      margin-top: 4px;
    }}
    .row-card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }}
    .row-card.structural {{
      border-color: var(--warn);
    }}
    .row-head {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 16px;
      padding: 14px 16px;
      border-bottom: 1px solid var(--line);
      background: #171c24;
    }}
    .rank {{
      color: var(--accent);
      font-size: 13px;
      font-weight: 700;
      margin-bottom: 4px;
    }}
    .row-head h2 {{
      margin: 0;
      font-size: 16px;
      font-weight: 620;
      overflow-wrap: anywhere;
    }}
    .row-head p {{
      margin: 4px 0 0 0;
      color: var(--muted);
      font-size: 13px;
    }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(2, minmax(120px, 1fr));
      gap: 6px;
      align-content: start;
      min-width: 330px;
      font-size: 12px;
      color: var(--muted);
    }}
    .metrics span {{
      background: var(--panel2);
      border: 1px solid var(--line);
      border-radius: 5px;
      padding: 5px 7px;
    }}
    .metrics strong {{ color: var(--text); }}
    .images {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 0;
    }}
    figure {{
      margin: 0;
      border-right: 1px solid var(--line);
      background: #0f1218;
    }}
    figure:last-child {{ border-right: 0; }}
    img {{
      display: block;
      width: 100%;
      image-rendering: auto;
    }}
    figcaption {{
      padding: 7px 10px;
      color: var(--muted);
      border-top: 1px solid var(--line);
      font-size: 12px;
    }}
    details {{
      padding: 10px 16px;
      color: var(--muted);
      border-top: 1px solid var(--line);
    }}
    details p {{ margin: 5px 0; }}
    .controls {{
      max-width: 1480px;
      margin: 0 auto 18px auto;
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px 16px;
      color: var(--muted);
    }}
    .controls label {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      font-size: 13px;
    }}
    .controls select {{
      color: var(--text);
      background: var(--panel2);
      border: 1px solid var(--line);
      border-radius: 5px;
      padding: 5px 7px;
    }}
    @media (max-width: 900px) {{
      body {{ padding: 12px; }}
      .row-head {{ grid-template-columns: 1fr; }}
      .metrics {{ min-width: 0; }}
      .images {{ grid-template-columns: 1fr; }}
      figure {{ border-right: 0; border-bottom: 1px solid var(--line); }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Material Image Comparison</h1>
    <p>Rows are sorted worst-first by raw RGB RMSE, with structural issues first.</p>
    <p>Reference root: <code>{html.escape(str(reference_root))}</code></p>
    <p>Tested root: <code>{html.escape(str(tested_root))}</code></p>
  </header>
  <section class="controls" aria-label="Sort controls">
    <label for="sort-key">Sort by
      <select id="sort-key">
        <option value="rmse" selected>RMSE</option>
        <option value="p95">P95 abs</option>
        <option value="max">Max abs</option>
        <option value="luma">Luma error (abs)</option>
        <option value="name">Name</option>
        <option value="id">Entry ID</option>
      </select>
    </label>
    <label for="sort-direction">Order
      <select id="sort-direction">
        <option value="desc" selected>Descending</option>
        <option value="asc">Ascending</option>
      </select>
    </label>
    <label><input id="structural-first" type="checkbox" checked> Structural issues first</label>
  </section>
  <section class="summary">
    <p>Diff thumbnails are abs(tested - reference) x4.</p>
    <div class="summary-grid">
      <div><span>Rows</span><strong>{summary['row_count']}</strong></div>
      <div><span>Compared</span><strong>{summary['compared_count']}</strong></div>
      <div><span>Structural issues</span><strong>{summary['structural_count']}</strong></div>
      <div><span>Max RMSE</span><strong>{summary['max_rmse']}</strong></div>
      <div><span>Max P95 abs</span><strong>{summary['max_p95']}</strong></div>
      <div><span>Max abs</span><strong>{summary['max_abs']}</strong></div>
      <div><span>Max |luma bias|</span><strong>{summary['max_luma_error']}</strong></div>
    </div>
  </section>
  <section id="comparison-rows">
    {cards}
  </section>
  <script>
    (function () {{
      const rowsRoot = document.getElementById("comparison-rows");
      const sortKey = document.getElementById("sort-key");
      const sortDirection = document.getElementById("sort-direction");
      const structuralFirst = document.getElementById("structural-first");
      const numericKeys = new Set(["rmse", "p95", "max", "luma"]);

      function numericValue(row, key) {{
        const value = Number.parseFloat(row.dataset[key] || "0");
        return Number.isFinite(value) ? value : 0;
      }}

      function textValue(row, key) {{
        return (row.dataset[key] || "").toLocaleLowerCase();
      }}

      function compareRows(a, b) {{
        if (structuralFirst.checked) {{
          const structuralDelta = Number(b.dataset.structural) - Number(a.dataset.structural);
          if (structuralDelta !== 0) {{
            return structuralDelta;
          }}
        }}

        const key = sortKey.value;
        const direction = sortDirection.value === "asc" ? 1 : -1;
        let result = 0;
        if (numericKeys.has(key)) {{
          result = numericValue(a, key) - numericValue(b, key);
        }} else {{
          result = textValue(a, key).localeCompare(textValue(b, key));
        }}
        if (result !== 0) {{
          return result * direction;
        }}
        result = textValue(a, "name").localeCompare(textValue(b, "name"));
        if (result !== 0) {{
          return result;
        }}
        return Number(a.dataset.originalIndex) - Number(b.dataset.originalIndex);
      }}

      function applySort() {{
        const rows = Array.from(rowsRoot.querySelectorAll(".row-card"));
        rows.sort(compareRows);
        rows.forEach((row, index) => {{
          rowsRoot.appendChild(row);
          const rank = row.querySelector(".rank");
          if (rank) {{
            rank.textContent = "#" + String(index + 1);
          }}
        }});
      }}

      sortKey.addEventListener("change", applySort);
      sortDirection.addEventListener("change", applySort);
      structuralFirst.addEventListener("change", applySort);
    }})();
  </script>
</body>
</html>
"""
    path.write_text(document, encoding="utf-8")


def _row_card(index: int, row: dict[str, Any]) -> str:
    status = str(row.get("status", ""))
    card_class = "row-card structural" if status != "compared" else "row-card"
    entry = html.escape(str(row.get("entry_id", "")))
    label = html.escape(str(row.get("label", "")))
    provider = html.escape(str(row.get("provider", "")))
    material_class = html.escape(str(row.get("material_class", "")))
    output = html.escape(str(row.get("output", "")))
    metrics = _metric_markup(row)
    visuals = _visual_markup(index, row)
    data_attrs = {
        "original-index": str(index),
        "structural": "1" if status != "compared" else "0",
        "status": status,
        "name": str(row.get("label", "")),
        "id": str(row.get("entry_id", "")),
        "rmse": _data_number(row.get("rmse_rgb")),
        "p95": _data_number(row.get("p95_abs_rgb")),
        "max": _data_number(row.get("max_abs_rgb")),
        "luma": _data_number(abs(float(row.get("mean_luma_bias", 0.0) or 0.0))),
    }
    attrs = " ".join(
        f'data-{name}="{html.escape(value, quote=True)}"' for name, value in data_attrs.items()
    )
    return f"""<article class="{card_class}" {attrs}>
  <div class="row-head">
    <div>
      <div class="rank">#{index}</div>
      <h2>{label}</h2>
      <p><code>{entry}</code></p>
      <p>{provider} / {material_class} / {output}</p>
    </div>
    <div class="metrics">{metrics}</div>
  </div>
  {visuals}
</article>"""


def _metric_markup(row: dict[str, Any]) -> str:
    status = html.escape(str(row.get("status", "")))
    return "\n      ".join(
        [
            f"<span>Status <strong>{status}</strong></span>",
            f"<span>RMSE <strong>{html.escape(_format_number(row.get('rmse_rgb')))}</strong></span>",
            f"<span>P95 abs <strong>{html.escape(_format_number(row.get('p95_abs_rgb')))}</strong></span>",
            f"<span>Max abs <strong>{html.escape(_format_number(row.get('max_abs_rgb')))}</strong></span>",
            f"<span>Luma bias <strong>{html.escape(_format_number(row.get('mean_luma_bias')))}</strong></span>",
            f"<span>Finite <strong>{html.escape(_format_number(row.get('finite_fraction')))}</strong></span>",
        ]
    )


def _visual_markup(index: int, row: dict[str, Any]) -> str:
    if row.get("status") != "compared":
        message = html.escape(str(row.get("message", "")) or "No image pair was compared.")
        return f"<details open><summary>Structural issue</summary><p>{message}</p></details>"
    prefix = f"html_assets/{index:04d}_{_safe_name(str(row['entry_id']))}"
    return f"""<div class="images">
    <figure><img src="{prefix}_reference.png" alt="Reference image"><figcaption>Reference</figcaption></figure>
    <figure><img src="{prefix}_tested.png" alt="Tested image"><figcaption>Tested</figcaption></figure>
    <figure><img src="{prefix}_diff.png" alt="Absolute difference image"><figcaption>Abs diff x4</figcaption></figure>
  </div>"""


def _html_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    compared = [row for row in rows if row.get("status") == "compared"]
    return {
        "row_count": len(rows),
        "compared_count": len(compared),
        "structural_count": len(rows) - len(compared),
        "max_rmse": _format_number(_max_metric(compared, "rmse_rgb")),
        "max_p95": _format_number(_max_metric(compared, "p95_abs_rgb")),
        "max_abs": _format_number(_max_metric(compared, "max_abs_rgb")),
        "max_luma_error": _format_number(
            max((abs(float(row.get("mean_luma_bias", 0.0) or 0.0)) for row in compared), default="")
        ),
    }


def _max_metric(rows: list[dict[str, Any]], key: str) -> float | str:
    values = [float(row[key]) for row in rows if row.get(key) not in (None, "")]
    return max(values, default="")


def _write_preview_assets(
    reference_dir: Path,
    tested_dir: Path,
    assets_dir: Path,
    index: int,
    row: dict[str, Any],
) -> None:
    reference_image = row.get("reference_image")
    tested_image = row.get("tested_image")
    reference_path = _find_source_image(reference_dir, str(reference_image))
    tested_path = _find_source_image(tested_dir, str(tested_image))
    if reference_path is None or tested_path is None:
        return
    reference = load_rgb(reference_path)
    tested = load_rgb(tested_path)
    diff = np.abs(tested - reference) * 4.0
    prefix = assets_dir / f"{index:04d}_{_safe_name(str(row['entry_id']))}"
    _write_png(prefix.with_name(prefix.name + "_reference.png"), reference)
    _write_png(prefix.with_name(prefix.name + "_tested.png"), tested)
    _write_png(prefix.with_name(prefix.name + "_diff.png"), diff)


def _find_source_image(root: Path, relative: str) -> Path | None:
    relative_path = Path(relative)
    if relative_path.is_absolute():
        return None
    root_resolved = root.resolve(strict=False)
    path = (root_resolved / relative_path).resolve(strict=False)
    try:
        path.relative_to(root_resolved)
    except ValueError:
        return None
    return path if path.is_file() else None


def _write_png(path: Path, rgb: np.ndarray) -> None:
    image = np.nan_to_num(rgb[:, :, :3], nan=0.0, posinf=1.0, neginf=0.0)
    image = np.clip(image, 0.0, 1.0)
    bitmap = Bitmap(
        data=image,
        pixel_format=Bitmap.PixelFormat.rgb,
        srgb_gamma=False,
    ).convert(
        component_type=Bitmap.ComponentType.uint8,
        srgb_gamma=True,
    )
    bitmap.write(path, Bitmap.FileFormat.png)


def _format_number(value: Any) -> str:
    if value is None or value == "":
        return ""
    return f"{float(value):.6g}"


def _data_number(value: Any) -> str:
    if value is None or value == "":
        return ""
    return f"{float(value):.17g}"


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)
