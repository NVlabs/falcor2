# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read BenchView metric histories and compare matched Falcor2 revisions."""

from __future__ import annotations

import argparse
import json
import math
import sys
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen


DEFAULT_API_BASE = "http://rtrci.nvidia.com/benchview/api"
STATISTICS = ("median", "mean", "min", "max")


class BenchViewError(Exception):
    """Describe an API or comparison failure without hiding it in output."""


def _get_json(
    api_base: str, path: str, params: list[tuple[str, str]], timeout: float
) -> dict[str, Any]:
    url = f"{api_base.rstrip('/')}{path}"
    if params:
        url += "?" + urlencode(params)
    request = Request(url, headers={"Accept": "application/json"}, method="GET")
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
    except HTTPError as error:
        raise BenchViewError(f"BenchView returned HTTP {error.code} for {path}") from error
    except (URLError, TimeoutError, OSError) as error:
        raise BenchViewError(f"Could not reach BenchView: {error}") from error
    except (json.JSONDecodeError, UnicodeError) as error:
        raise BenchViewError(f"BenchView returned invalid JSON for {path}") from error
    if not isinstance(payload, dict):
        raise BenchViewError(f"BenchView returned an unexpected response for {path}")
    return payload


def _find_test(api_base: str, project: str, name: str, source: str | None, timeout: float) -> str:
    path = f"/v1/projects/{quote(project, safe='')}/tests"
    cursor = None
    seen_cursors: set[str] = set()
    items: list[dict[str, Any]] = []
    while True:
        params = [("test", name), ("limit", "200")]
        if source:
            params.append(("source", source))
        if cursor:
            params.append(("cursor", cursor))
        page = _get_json(api_base, path, params, timeout)
        page_items = page.get("items")
        if not isinstance(page_items, list):
            raise BenchViewError("BenchView test catalogue has no items list")
        items.extend(item for item in page_items if isinstance(item, dict))
        cursor = page.get("nextCursor")
        if cursor is None:
            break
        if not isinstance(cursor, str) or cursor in seen_cursors:
            raise BenchViewError("BenchView test catalogue returned an invalid cursor")
        seen_cursors.add(cursor)

    exact = [
        item
        for item in items
        if str(item.get("name", "")).casefold() == name.casefold() or item.get("testId") == name
    ]
    matches = exact or items
    if len(matches) != 1:
        choices = [f"{item.get('name')}: {item.get('testId')}" for item in matches[:10]]
        detail = "; ".join(choices) if choices else "no matching tests"
        raise BenchViewError(
            f"Test search for {name!r} is not unique ({len(matches)} matches): "
            f"{detail}. Pass --test-id for an exact selection."
        )
    test_id = matches[0].get("testId")
    if not isinstance(test_id, str):
        raise BenchViewError("BenchView test catalogue omitted testId")
    return test_id


def _point_value(point: dict[str, Any], statistic: str) -> float:
    summary = point.get("summary")
    if point.get("status") != "passed" or not isinstance(summary, dict):
        raise BenchViewError("Selected observation has no passed metric summary")
    value = summary.get(statistic)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BenchViewError(f"Selected observation has no numeric {statistic}")
    if not math.isfinite(value):
        raise BenchViewError(f"Selected observation has non-finite {statistic}")
    return float(value)


def _brief_point(point: dict[str, Any], statistic: str) -> dict[str, Any]:
    return {
        "revision": point.get("revision"),
        "commitTime": point.get("x"),
        "observedAt": point.get("observedAt"),
        "observationId": point.get("observationId"),
        "runId": point.get("runId"),
        "value": _point_value(point, statistic),
        "sampleCount": point["summary"].get("count"),
    }


def _matching_point(points: list[dict[str, Any]], revision_prefix: str) -> dict[str, Any] | None:
    matches = [
        point
        for point in points
        if isinstance(point.get("revision"), str) and point["revision"].startswith(revision_prefix)
    ]
    if len(matches) > 1:
        observations = ", ".join(str(item.get("observationId")) for item in matches)
        raise BenchViewError(
            f"Revision {revision_prefix} has multiple matching runs: {observations}"
        )
    return matches[0] if matches else None


def _detail(api_base: str, point: dict[str, Any], timeout: float) -> dict[str, Any]:
    observation_id = point.get("observationId")
    if not isinstance(observation_id, str):
        raise BenchViewError("Series point omitted observationId")
    return _get_json(
        api_base,
        f"/v1/observations/{quote(observation_id, safe='')}",
        [],
        timeout,
    )


def _environment(detail: dict[str, Any]) -> dict[str, Any]:
    environment = detail.get("environment", {})
    identity = environment.get("identity", {})
    gpus = identity.get("gpus", [])
    gpu = gpus[0] if isinstance(gpus, list) and gpus else {}
    os_info = identity.get("os", {})
    return {
        "fingerprint": environment.get("fingerprint"),
        "machine": identity.get("machine"),
        "os": os_info.get("name"),
        "gpu": gpu.get("name"),
        "driver": gpu.get("driverVersion"),
    }


def _change(before: float, after: float) -> dict[str, float | None]:
    delta = after - before
    return {
        "absolute": delta,
        "percent": 100.0 * delta / before if before else None,
        "ratio": after / before if before else None,
    }


def _components(
    before: dict[str, Any], after: dict[str, Any], statistic: str
) -> list[dict[str, Any]]:
    left = {item["id"]: item for item in before.get("breakdown", {}).get("components", [])}
    right = {item["id"]: item for item in after.get("breakdown", {}).get("components", [])}
    result = []
    for component_id in left.keys() & right.keys():
        old = left[component_id]["summary"].get(statistic)
        new = right[component_id]["summary"].get(statistic)
        if not isinstance(old, (int, float)) or not isinstance(new, (int, float)):
            continue
        result.append(
            {
                "id": component_id,
                "name": right[component_id].get("name"),
                "before": old,
                "after": new,
                "change": _change(old, new),
            }
        )
    return sorted(result, key=lambda item: item["id"])


def _compare(
    api_base: str,
    series: dict[str, Any],
    before: dict[str, Any],
    after: dict[str, Any],
    metric_id: str,
    statistic: str,
    timeout: float,
) -> dict[str, Any]:
    old_detail = _detail(api_base, before, timeout)
    new_detail = _detail(api_base, after, timeout)
    old_environment = old_detail.get("environment", {})
    new_environment = new_detail.get("environment", {})
    if (
        old_environment.get("fingerprint") != series.get("environmentFingerprint")
        or new_environment.get("fingerprint") != series.get("environmentFingerprint")
        or old_environment.get("identity") != new_environment.get("identity")
    ):
        raise BenchViewError("Compared observations do not share an environment identity")

    old_metrics = old_detail.get("observation", {}).get("metrics", [])
    new_metrics = new_detail.get("observation", {}).get("metrics", [])
    old_metric = next((item for item in old_metrics if item.get("id") == metric_id), None)
    new_metric = next((item for item in new_metrics if item.get("id") == metric_id), None)
    if old_metric is None or new_metric is None:
        raise BenchViewError(f"Observation detail omitted metric {metric_id}")
    direction = old_metric.get("direction")
    if direction != new_metric.get("direction"):
        raise BenchViewError(f"Metric {metric_id} changed direction between observations")
    old_value = _point_value(before, statistic)
    new_value = _point_value(after, statistic)
    delta = new_value - old_value
    directional_change = "neutral"
    if delta == 0:
        directional_change = "unchanged"
    elif direction == "lower":
        directional_change = "worse" if delta > 0 else "better"
    elif direction == "higher":
        directional_change = "better" if delta > 0 else "worse"

    old_point = _brief_point(before, statistic)
    new_point = _brief_point(after, statistic)
    for point, detail in ((old_point, old_detail), (new_point, new_detail)):
        metadata = detail.get("rawObservation", {}).get("metadata", {})
        point["gpuClocksLocked"] = metadata.get("gpuClocksLocked")
        point["provenance"] = metadata.get("provenance")

    return {
        "caseId": series.get("caseId"),
        "dimensions": series.get("dimensions"),
        "environment": _environment(old_detail),
        "direction": direction,
        "directionalChange": directional_change,
        "before": old_point,
        "after": new_point,
        "change": _change(old_value, new_value),
        "components": _components(before, after, statistic),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--test", help="Search for one BenchView test by name")
    selection.add_argument("--test-id", help="Select an exact BenchView test ID")
    parser.add_argument("--metric", required=True, help="Metric ID, such as gpu_time")
    parser.add_argument("--project", default="falcor2")
    parser.add_argument("--api-base", default=DEFAULT_API_BASE)
    parser.add_argument("--source", help="Exact source-file filter for test discovery")
    parser.add_argument("--dimension", action="append", default=[], metavar="NAME=VALUE")
    parser.add_argument("--branch", help="Optional exact VCS branch filter")
    parser.add_argument("--environment", help="Optional environment fingerprint filter")
    parser.add_argument("--from", dest="from_date", help="Inclusive UTC start date or timestamp")
    parser.add_argument("--to", dest="to_date", help="Inclusive UTC end date or timestamp")
    parser.add_argument(
        "--x-axis", choices=("commitTime", "observedAt", "receivedAt"), default="commitTime"
    )
    parser.add_argument("--stat", choices=STATISTICS, default="median")
    parser.add_argument("--before", help="Earlier revision SHA or unambiguous prefix")
    parser.add_argument("--after", help="Later revision SHA or unambiguous prefix")
    parser.add_argument("--timeout", type=float, default=20.0)
    args = parser.parse_args(argv)
    if bool(args.before) != bool(args.after):
        parser.error("--before and --after must be provided together")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if not args.api_base.startswith(("http://", "https://")):
        parser.error("--api-base must be an HTTP or HTTPS URL")
    for dimension in args.dimension:
        name, separator, value = dimension.partition("=")
        if not separator or not name or not value:
            parser.error("--dimension must use NAME=VALUE")

    try:
        test_id = args.test_id or _find_test(
            args.api_base, args.project, args.test, args.source, args.timeout
        )
        params = [
            ("testId", test_id),
            ("metric", args.metric),
            ("xAxis", args.x_axis),
            ("sampling", "all"),
            ("status", "passed"),
        ]
        for option, value in (
            ("branch", args.branch),
            ("environment", args.environment),
            ("from", args.from_date),
            ("to", args.to_date),
        ):
            if value:
                params.append((option, value))
        params.extend(("dimension", value) for value in args.dimension)
        series = _get_json(
            args.api_base,
            f"/v1/projects/{quote(args.project, safe='')}/series",
            params,
            args.timeout,
        )
        if (
            series.get("testId") != test_id
            or series.get("metricId") != args.metric
            or series.get("appliedSampling") != "all"
            or not isinstance(series.get("series"), list)
            or not isinstance(series.get("totalPoints"), int)
            or not isinstance(series.get("returnedPoints"), int)
        ):
            raise BenchViewError("BenchView returned an unexpected series response")
        if series.get("totalPoints") != series.get("returnedPoints"):
            raise BenchViewError("BenchView did not return all matching points; narrow the filters")

        report: dict[str, Any] = {
            "project": args.project,
            "testId": test_id,
            "testName": series.get("testName"),
            "metric": args.metric,
            "metricName": series.get("metricName"),
            "unit": series.get("unit"),
            "statistic": args.stat,
            "xAxis": series.get("xAxis"),
            "totalPoints": series.get("totalPoints"),
        }
        if args.before:
            comparisons = []
            unmatched = []
            for item in series.get("series", []):
                points = item.get("points", [])
                before = _matching_point(points, args.before)
                after = _matching_point(points, args.after)
                if before is None or after is None:
                    unmatched.append(
                        {
                            "dimensions": item.get("dimensions"),
                            "environmentFingerprint": item.get("environmentFingerprint"),
                        }
                    )
                    continue
                comparisons.append(
                    _compare(
                        args.api_base,
                        item,
                        before,
                        after,
                        args.metric,
                        args.stat,
                        args.timeout,
                    )
                )
            if not comparisons:
                raise BenchViewError("No matched case and environment contains both revisions")
            report["comparisons"] = comparisons
            report["unmatchedSeries"] = unmatched
        else:
            report["series"] = [
                {
                    "caseId": item.get("caseId"),
                    "dimensions": item.get("dimensions"),
                    "environmentFingerprint": item.get("environmentFingerprint"),
                    "points": [_brief_point(point, args.stat) for point in item.get("points", [])],
                }
                for item in series.get("series", [])
            ]
        print(json.dumps(report, indent=2, ensure_ascii=True))
        return 0
    except (BenchViewError, KeyError, TypeError, ValueError) as error:
        print(f"benchview_metric: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
