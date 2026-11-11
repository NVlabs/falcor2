---
name: falcor2-benchview-analysis
description: Query BenchView's read-only API and analyze Falcor2 benchmark metrics across revisions, scenes, and GPU backends. Use for performance-history investigations and backfilled results, not for running or submitting benchmarks.
---

# Analyze Falcor2 metrics in BenchView

BenchView is served at `http://rtrci.nvidia.com/benchview/api`. Its read routes
under `/v1` need no credentials. Do not open BenchView's ignored `env.json` or use
its write key for analysis. If a sandbox blocks sockets, retry an authorized
read-only request outside the sandbox; a socket denial does not establish that
BenchView is down. The API base can be overridden for another deployment.

Use [scripts/benchview_metric.py](scripts/benchview_metric.py) for test discovery,
series queries, and matched-revision comparisons. It uses only Python's standard
library and sends GET requests. For example:

```text
<python> .agents/skills/falcor2-benchview-analysis/scripts/benchview_metric.py \
  --test Sponza --metric gpu_time --dimension scene=sponza \
  --dimension device_type=cuda --from 2026-10-02 --to 2026-10-05 \
  --before 4ab8df48 --after c5793ab2
```

The script emits JSON with the returned unit, statistic, source observations,
and any matched comparisons. Omit `--before` and `--after` to inspect a metric's
history. Use `--test-id` if a test-name search is ambiguous. Use `--branch` only
when that branch is part of the question; backfilled runs may need an unfiltered
branch search. Use repeated `--dimension name=value` filters to select a scene,
backend, or another test parameter. For other endpoints, consult BenchView's
`apps/api/src/routes/queries.ts` and `packages/contracts/src/query-api.ts`.

## Interpret a result

- The default time axis is source **commit time**. This finds backfills even when
  `observedAt` is weeks later. State both dates when provenance matters. Date-only
  `--from` and `--to` bounds cover complete UTC days. The helper requests every
  point (`sampling=all`) and rejects an incomplete response.
- Compare only the same case dimensions and environment fingerprint. The helper
  verifies the environment identity in each pair and reports the GPU, driver,
  OS, and capture dates. If a revision has multiple matching runs, inspect their
  observation IDs and provenance rather than selecting one silently.
- Report the metric ID, unit, statistic, sample count, both values, and change.
  Use the metric's declared `direction` to interpret whether a change is better
  or worse. BenchView Overview uses medians by default. One 32-sample `gpu_time`
  observation is one benchmark run, not 32 independent runs.
- `compile_time` is one compilation total per scene/backend; it is not startup
  wall time. `gpu_time_diagnostic` is a separate instrumented pass. Its phase
  **means** add to its total; independently computed medians need not. Do not
  equate its total with the primary uninstrumented `gpu_time` samples.
- A time-series jump identifies a revision interval, not the exact instruction
  responsible. Inspect adjacent source commits and hold the benchmark workload,
  hardware, driver, compiler/dependency pins, and harness recipe in view. Mark
  a pin comparison unverified when the API omits its provenance. Use a controlled
  A/B run or shader profiling to confirm a code-level mechanism.

For Falcor2's benchmark and backfill semantics, read
`docs/performance-ci.md` and `docs/benchmark-backfill.md` in the Falcor2 source
root. An installed `source-context.md` identifies that root in a consuming
project. When discussing
path-tracer performance, note that today's `scene_capability_policy=required`
selects features needed by the scene, including interior tracking when solid
transmission requires it. It can compile more variants across scenes but smaller
specializations can render up to about 2x faster; `all` is an explicit option.
