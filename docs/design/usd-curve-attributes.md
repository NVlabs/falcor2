---
status: draft
audience: Falcor2 contributors and coding agents
last_verified: 2026-10-19
verified_against: "82b7ae929"
---

(falcor2-design-usd-curve-attributes)=
# USD curve attribute counts and interpolation

Falcor2's USD importer separates the number of authored curve attributes from the number of vertices produced by
tessellation. A varying attribute supplies linearly interpolated values at curve-span boundaries; it is not a
per-control-point attribute merely because the two counts sometimes coincide. The importer owns this distinction
before curves become linear swept spheres. Its shared varying-count helper follows the USD schema's segment rules,
including the end spans of pinned splines, instead of reproducing conflicting behavior in OpenUSD's varying-size
helper and Storm. This contract keeps batched strands from reading one another's widths or texture coordinates.

## Purpose and scope

This page explains how the importer sizes, partitions, and samples authored varying data on USD `BasisCurves`, and
why its count differs from some upstream implementations. The same authored cardinality applies whether a consumer
represents a curve as a round primitive or a ribbon: normals and the eventual surface representation do not change
USD's interpolation rules.

The implemented consumer described here is round-curve import. Ribbon construction, generated texture-coordinate
conventions, and shader shading frames are separate concerns. General stage traversal belongs in
[Scene loading and authoring](scene-loading-and-authoring.md); live geometry and ray traversal belong in
[Geometry and ray tracing](geometry-and-ray-tracing.md).

## Architecture

`compute_curve_varying_counts()` in the USD importer utilities owns the authored varying count for each strand. A
strand is one curve within the batch described by `curveVertexCounts`. The helper takes a `UsdGeomBasisCurves` and a
time code, obtains its segment counts, and adds the final boundary value for every nonperiodic or pinned strand.
Periodic strands share their first and final boundary and need no additional value. The helper expects valid USD
topology; it is not a general topology validator or a repair mechanism for malformed curves.

`UsdImporterContext::extract_curve()` reads widths, texture coordinates, interpolation tokens, and topology. It
records the helper's per-strand counts in `TesselatorInputCurve`. The tessellator maintains separate offsets for
control-point data and varying data as it walks a batch. Those offsets must follow the authored topology even when
the renderer omits some spans or chooses a different sampling rate.

The tessellator produces `ImporterCurve` positions, radii, UVs, and line-segment indices. Widths are diameters and
are converted to radii after interpolation. Motion import reuses the reference layout for subsequent position and
radius samples; changing topology or primvar indices is rejected by the existing fixed-layout motion checks.

## Attribute flow

The importer prefers an authored `primvars:widths` over the schema `widths` attribute. Indexed primvars are flattened
before partitioning them into strands. Therefore the varying count describes the flattened values, or the primvar
index-array length, rather than the size of the compact value table. An authored `st` primvar similarly retains its
interpolation after flattening, and its flattened count is validated before tessellation.

Constant attributes apply to the whole primitive, uniform attributes supply one value per strand, and vertex
attributes use control-point ranges. Varying attributes use the shared count helper. Existing round-width behavior
still allows an absent array to use the default radius and a single value to broadcast. These conventions do not
change the number of varying values specified by USD.

At refinement zero, cubic curves use their control polygon. Varying data is distributed across that polygon's
vertex ordinals. For pinned B-spline and Catmull-Rom curves there are N varying values for N control points, so each
control vertex receives the corresponding varying value. At positive refinement, vertex data is evaluated with the
cubic basis, while varying widths and UVs remain linear between span boundaries.

At positive refinement, pinned B-spline and Catmull-Rom curves reflect one phantom control at each endpoint:
`P[-1] = 2 * P[0] - P[1]` and `P[N] = 2 * P[N-1] - P[N-2]`. The same reflection applies to vertex widths and UVs.
All N - 1 spans are evaluated, including both endpoint spans, so two- and three-point pinned strands are supported.
Varying widths and UVs use the N authored boundary values directly, without phantom values or an interior offset.
Motion samples use the same endpoint evaluation and attribute layout.

For linear and Bezier curves, pinned has the same behavior as nonperiodic. Periodic round cubic geometry retains
its existing nonperiodic fallback; its authored attribute count must nevertheless remain periodic.

## Invariants and rationale

### The schema determines authored cardinality

For valid topology, let N be the number of authored control points in one strand. The
[USD BasisCurves specification](https://openusd.org/release/api/class_usd_geom_basis_curves.html) defines these counts:

| Curve type and basis | Nonperiodic varying count | Periodic varying count | Pinned varying count |
| --- | --- | --- | --- |
| Linear | N | N | N |
| Cubic Bezier | (N - 4) / 3 + 2 | N / 3 | (N - 4) / 3 + 2 |
| Cubic B-spline / Catmull-Rom | N - 2 | N | N |

Pinned affects B-spline and Catmull-Rom endpoint evaluation; it does not add an analogous Bezier rule. A pinned
spline has N - 1 spans, including the two end spans, and therefore N boundary values. Varying interpolation remains
linear even where its count equals the vertex count. Tessellation refinement changes output sampling density,
never the required authored array size.

### OpenUSD has conflicting implementations

`UsdGeomBasisCurves::ComputeVaryingDataSize()` and `HdBasisCurvesTopology` historically treat pinned spline varying
data like ordinary nonperiodic data. The
[UsdGeom implementation rationale](https://github.com/PixarAnimationStudios/OpenUSD/commit/587cbc36d2a74ff5be6d40d482dfb660f192cbe5)
asserts that the extra pinned spans should not require authored values. That assertion conflicts with the schema.
The affected implementation also attempts to clamp short curves with unsigned arithmetic after subtracting four, so its
count can wrap before the clamp takes effect. In OpenUSD 26.08, for N equal to 2, 3, 4, and 7 it returns 0, 1, 2, and 5; the Hydra
topology helper clamps in signed arithmetic and returns 2, 2, 2, and 5. Neither sequence is the specified pinned
count of 2, 3, 4, and 7.

[OpenUSD issue #2775](https://github.com/PixarAnimationStudios/OpenUSD/issues/2775) reports the concrete consequence:
a seven-point pinned curve with seven varying widths is rejected by Storm because it expects five. The discussion
also identifies the UsdGeom calculation as incorrect, records internal issue USD-8899, and includes maintainer
requests for a contribution to fix it. Agreement between those two legacy helpers is therefore insufficient
validation of an importer.

Hydra itself is not uniform. The
[correction to HdsiPinnedCurveExpandingSceneIndex](https://github.com/PixarAnimationStudios/OpenUSD/commit/7b3db0b60b6d36f4c1378388d5b076912410e469),
used by HdPrman, explicitly derives N input varying values for N pinned control points. Its tests use four values
for four-point curves and seven for seven-point curves. Endpoint expansion for that delegate is a subsequent
representation conversion: Catmull-Rom can retain the varying array, whereas its duplicated-endpoint B-spline
conversion adds one varying value at each end. Those expanded counts are not the authored USD counts.

The separate [pinned endpoint discussion](https://forum.aousd.org/t/evaluating-pinned-curves/2868) explains why Hydra's
endpoint duplication differs from the specification's reflected phantom points. Maintainers advise following the
specification and describe the duplication as legacy pipeline behavior. That discussion concerns geometry; it
does not substitute for the varying-count evidence above.

Falcor2 consequently uses `ComputeSegmentCounts()`, whose pinned treatment agrees with the schema, and derives its
own varying counts from those spans. It neither branches on an OpenUSD version nor accepts an N - 2 pinned array
as an alternate valid layout. The small helper can remain correct after the upstream varying-size routines are
fixed. Tests assert schema-derived counts and imported values, rather than requiring an upstream bug to persist.

## Source landmarks

- `src/falcor2/importers/usd_importer/usd_importer_utils.h` and
  `src/falcor2/importers/usd_importer/usd_importer_utils.cpp` - shared varying-count contract and rationale.
- `src/falcor2/importers/usd_importer/usd_importer_curve_adapter.cpp` - attribute selection, flattening, per-strand counts,
  UV validation, and motion sample extraction.
- `src/falcor2/importers/curve_tessellator/curve_tessellator.h` - shared Falcor input types and round/ribbon entry points.
- `src/falcor2/importers/curve_tessellator/curve_tessellator_internal.h` - shared strand offsets, basis evaluation,
  varying interpolation, and reflected pinned endpoints.
- `src/falcor2/importers/curve_tessellator/curve_tessellator.cpp` and `curve_tessellator_ribbons.cpp` in the same
  directory - round output and ribbon frame/joint/triangle assembly.
- `src/falcor2/importers/importer_types.h` - `ImporterCurve` storage and sample streams.
- `tests/native/usd_importer/test_usd_importer.cpp` - schema-count matrix, mixed strand lengths, short pinned curves,
  nonconstant widths and UVs, indexed attributes, and sampled radii.
- `tests/python/importers/test_curve_motion.py` - motion sample correspondence and layout rejection.

## Related design documents

- [Scene loading and authoring](scene-loading-and-authoring.md)
- [Geometry and ray tracing](geometry-and-ray-tracing.md)
- [Falcor2 design documentation](index.md)
