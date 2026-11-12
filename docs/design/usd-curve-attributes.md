---
status: draft
audience: Falcor2 contributors and coding agents
last_verified: 2026-11-11
verified_against: "892f7b3bec19aebcb1f645206e7089e81387586a"
---

(falcor2-design-usd-curve-attributes)=
# USD curve attribute counts and interpolation

Falcor2's USD importer separates authored curve attribute counts from tessellated vertex counts. Its shared
adapter flattens primvars into a borrowed `CurveInput`, then the same evaluator supplies round curves and
triangle ribbons. Varying attributes interpolate linearly between span boundaries; vertex attributes follow
control-point basis weights, even when their array lengths coincide. A schema-based count helper preserves
strand boundaries through pinned endpoint evaluation, periodic compatibility behavior and changes in
refinement. Representation-specific validation retains round-width fallbacks without redefining the authored
layout. This contract prevents one strand from consuming another strand's widths, UVs or oriented-curve normals.

## Purpose and scope

This page explains who determines authored cardinality, how flattened arrays are partitioned, and how their
values reach both curve representations. A strand is one curve in a `curveVertexCounts` batch; a span is one
polynomial interval. Normals can select a different surface representation without changing USD's interpolation
rules.

[Curve import and tessellation](curve-import-and-tessellation.md) covers representation selection, refinement,
ribbon frames, generated UVs and output assembly. [Scene loading and authoring](scene-loading-and-authoring.md)
covers the surrounding importer, and [Geometry deformation](geometry-deformation.md) covers runtime sample
storage and playback. This page concerns authored attributes and import-time evaluation.

## Architecture

`compute_curve_varying_counts()` in the USD importer utilities owns the per-strand authored varying count.
It calls `UsdGeomBasisCurves::ComputeSegmentCounts()` at the requested time and adds a final boundary for
nonperiodic and pinned strands. Periodic strands reuse their first boundary and need no extra authored value.
The helper assumes valid USD topology; it is not a general topology validator or repair pass.

`convert_usd_curves()` resolves and reads the attributes, converts interpolation tokens, and supplies those
counts in `curve_tessellator::CurveInput::varying_counts` whenever varying data is present. The adapter owns
the USD arrays and converted vectors during evaluation. `CurveInput` borrows their flattened spans and contains
no USD objects. First-sample effective normals choose the writer; `UsdImporterContext::extract_curve()` routes
the resulting `ImporterCurve` or `ImporterMesh` into the imported scene.

The shared evaluator maintains independent control-point and varying offsets for each strand. Offsets advance
by authored counts even if a writer samples fewer spans or emits more vertices. The round writer converts
evaluated widths from diameters to radii. The ribbon writer uses evaluated widths and normals to place edge
vertices; it copies each authored centerline UV to both edges. Returned importer geometry owns its output
storage. Neither writer infers the input layout from the output vertex count.

## Attribute flow

### Resolve, flatten and partition

A valued, possibly inherited `primvars:widths` or `primvars:normals` takes precedence over its built-in attribute.
The adapter resolves this choice before collecting motion times. Indexed primvars are flattened before strand
partitioning, so authored cardinality describes the flattened array, or index-array length, rather than the
compact value table. An indexed varying width table with two unique values can describe many boundaries if
its indices supply the required flattened entries. Inherited `st` is likewise flattened while preserving its
interpolation mode.

| Interpolation | Authored layout | Evaluation |
| --- | --- | --- |
| Constant | One value for the primitive | Broadcast to every strand |
| Uniform | One value per strand | Select by strand index |
| Vertex | One value per control point | Evaluate with the position basis |
| Varying | One value per span boundary | Interpolate linearly within each span |

For example, pinned spline strands with four and five controls require four and five varying values. The second
strand begins at varying offset four, regardless of refinement or how many tessellated samples the first strand
produces. Compact primvar table entries may be reused, but the flattened strand ranges must remain separate.

### Evaluate control polygons and polynomial spans

At refinement zero, cubic curves use their control polygon and varying data is mapped across vertex ordinals.
Pinned B-spline and Catmull-Rom curves have N varying values for N controls, so each polygon vertex receives its
corresponding varying value. Positive refinement changes the output sample density: vertex attributes use the
cubic basis, while varying widths, UVs and ribbon normals remain linear between span boundaries.

Equal counts do not imply equal interpolation. For a positively refined pinned B-spline with four width values
`[2, 2, 8, 2]`, the start of its second span evaluates vertex width as `(2 + 4 * 2 + 8) / 6 = 3`; varying width
at that same boundary is the second authored value, `2`. These are diameters in both representations, with the
round writer halving the evaluated result afterward.

At positive refinement, pinned B-spline and Catmull-Rom evaluation reflects phantom endpoint controls:
`P[-1] = 2 * P[0] - P[1]` and `P[N] = 2 * P[N-1] - P[N-2]`. Vertex attributes use the same reflection,
including widths, UVs and ribbon normals. All N - 1 spans are evaluated, including endpoint spans on two- and
three-control strands. Varying attributes use the N authored boundary values directly; they receive neither
phantom values nor an interior offset.

Pinned linear and Bezier curves follow nonperiodic behavior. Periodic ribbons close their geometry, while
round curves retain open geometric scheduling, including the nonperiodic fallback for cubic input. Both still
partition varying data using the authored periodic counts. A geometric fallback must not shift the next strand's
attribute offset or reinterpret periodic boundary values as an open layout.

### Validate values and preserve motion layout

The shared interpretation does not imply identical input validation. Nonempty ribbon normals, widths and UVs
must have exactly the count required by their interpolation. Normals are required for ribbon evaluation; empty
widths can use the default diameter. The USD adapter requires exact flattened `st` cardinality for round curves
as well, and rejects an authored empty or unreadable `st` instead of treating it as missing.

Round widths retain compatibility behavior: missing built-in widths can select the default radius, and a
singleton broadcasts regardless of its interpolation token. Larger width arrays must cover all required strand
ranges, but surplus values are permitted. An effective empty width primvar is rejected for round import.
These exceptions do not create an alternate varying-count formula or an N - 2 partition for pinned splines.

With `import_curve_motion`, the adapter checks fixed point counts and raw normal/width table lengths before
flattening each sample. `curveVertexCounts` and normal, width and UV index arrays must remain identical across
their authored times. This also prevents changing representation by switching between empty and nonempty
normals. Type, basis, wrap and interpolation are read once as fixed metadata; the adapter does not claim to
validate every possible animated metadata change.

`st` values are read only at the initial geometry sample and do not contribute motion times. Later UV value-table
changes are not imported, even though index changes are checked. Later deformation samples reuse the reference
indices and UVs and write only round positions/radii or ribbon positions/normals into allocated streams.

## Invariants and rationale

### The schema determines authored cardinality

For valid topology, let N be the number of authored controls in one strand. The
[USD BasisCurves specification](https://openusd.org/release/api/class_usd_geom_basis_curves.html) defines these counts:

| Curve type and basis | Nonperiodic varying count | Periodic varying count | Pinned varying count |
| --- | --- | --- | --- |
| Linear | N | N | N |
| Cubic Bezier | (N - 4) / 3 + 2 | N / 3 | (N - 4) / 3 + 2 |
| Cubic B-spline / Catmull-Rom | N - 2 | N | N |

Pinned affects B-spline and Catmull-Rom endpoints; it does not add an analogous Bezier rule. A pinned spline
has N - 1 spans and therefore N boundary values. Refinement, ribbon edge duplication and runtime storage
cannot change these authored counts. The count helper derives partitions from topology without branching
on an OpenUSD version.

### Why counts come from segment topology

Falcor2 uses segment topology because agreement between upstream varying-size helpers does not establish the
schema's authored layout. [OpenUSD issue #2775](https://github.com/PixarAnimationStudios/OpenUSD/issues/2775)
records Storm rejecting seven varying widths on a seven-control pinned curve because it expected five,
then using a fallback width. That N - 2 count omits the pinned endpoint boundaries.

The [Hdsi pinned-curve correction](https://github.com/PixarAnimationStudios/OpenUSD/commit/7b3db0b60b6d36f4c1378388d5b076912410e469)
uses N input varying values for N pinned controls. Delegate-specific endpoint expansion is a later representation
conversion; any values it adds are not extra authored USD entries. The separate
[pinned endpoint discussion](https://forum.aousd.org/t/evaluating-pinned-curves/2868) distinguishes specification
reflection from legacy endpoint duplication. That geometric conversion must not replace the authored-count rule.

Falcor2 derives counts from segment topology and verifies imported values and strand isolation. Tests assert
schema-derived results rather than requiring an upstream defect to persist. Width compatibility fallbacks
remain explicit exceptions to validation, not evidence for changing the helper's layout.

## Source landmarks

- `src/falcor2/importers/usd_importer/usd_importer_utils.h` and
  `src/falcor2/importers/usd_importer/usd_importer_utils.cpp` - authored counts and fixed-index checks.
- `src/falcor2/importers/usd_importer/usd_importer_curve_adapter.cpp` - effective sources, flattening, cardinality
  validation and motion extraction.
- `src/falcor2/importers/curve_tessellator/curve_tessellator.h` and
  `src/falcor2/importers/curve_tessellator/curve_tessellator_internal.h` - borrowed input, strand ranges and evaluation.
- `src/falcor2/importers/curve_tessellator/curve_tessellator.cpp` and
  `src/falcor2/importers/curve_tessellator/curve_tessellator_ribbons.cpp` - width conversion, output-specific
  validation and ribbon attributes.
- `src/falcor2_ext/importers/importer_types.cpp` - owned-output Python views and sample access.
- `tests/native/usd_importer/test_usd_importer.cpp` - schema-count matrix, mixed strands, short pinned endpoints
  and an in-memory indexed varying-motion example.
- `tests/native/importers/test_ribbon_tessellator.cpp` - ribbon interpolation, malformed attributes and seams.
- `tests/python/importers/test_ribbons.py`, `tests/python/importers/test_ribbon_motion.py` and
  `tests/python/importers/test_curve_motion.py` - shared evaluation, frozen samples and layout rejection.

## Related design documents

- [Curve import and tessellation](curve-import-and-tessellation.md)
- [Scene loading and authoring](scene-loading-and-authoring.md)
- [Geometry deformation](geometry-deformation.md)
- [Geometry and ray tracing](geometry-and-ray-tracing.md)
- [Falcor2 design documentation](index.md)
