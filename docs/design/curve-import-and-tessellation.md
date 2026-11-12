---
status: draft
audience: Falcor2 contributors and coding agents
last_verified: 2026-11-11
verified_against: "142c593fb4c498971cbdb09a5dd308d95c794262"
---

(falcor2-design-curve-import-and-tessellation)=
# Curve import and tessellation

Falcor2 imports USD BasisCurves through one attribute adapter and a shared curve evaluator. Effective normals
at the first geometry sample select the representation: curves without normals become round swept-sphere
segments, while oriented curves become triangle ribbons. Both outputs share basis evaluation, attribute
interpolation and motion sampling, but retain distinct refinement, closure and UV policies. The adapter owns
decoded values; tessellators borrow them and write owned importer geometry. Motion preserves the first sample's
topology and static attributes, then fills deformation streams for ordinary scene realization. Ribbon frame
repair handles local singularities without borrowing orientation from another strand.

## Purpose and scope

This page explains how authored curves become renderable geometry, which responsibilities are shared, and
which output differences are intentional. It covers import-time evaluation and fixed-topology motion extraction.
[USD curve attribute counts and interpolation](usd-curve-attributes.md) explains authored cardinality and
strand offsets. [Scene loading and authoring](scene-loading-and-authoring.md) covers the surrounding importer;
[Geometry deformation](geometry-deformation.md) covers sample storage, timing and GPU playback.

## Architecture

### Adapter, evaluator and output writers

The USD adapter resolves effective attributes, flattens indexed primvars, selects sample times and constructs
the USD-independent `curve_tessellator::CurveInput`. This input borrows point, normal, width, UV and strand-count
arrays. The adapter keeps their owners alive during each evaluation; no tessellator retains USD objects or
borrowed spans in its result. Widths are diameters at this boundary.

`CurveInput` also carries type, basis, wrap, interpolation modes, refinement and import properties. The shared
internal evaluator computes strand offsets, polynomial spans, sample locations and interpolated attributes.
The round writer produces `ImporterCurve` position/radius streams and segment indices. The ribbon writer
produces `ImporterMesh` positions, normals, UVs and triangles. Sample-zero overloads allocate all required
deformation streams and create static topology; later-sample overloads write only positions/radii or
positions/normals into caller-owned streams of exactly the reference size.

Ribbon material assignment, deduplication, tangent generation and optional normal recomputation use the mesh
pipeline. Scene realization chooses static or dynamic geometry from the resulting motion keys. A ribbon with
multiple keys becomes `DynamicMeshGeometry`; a round curve becomes `DynamicCurveGeometry`. Both use the
ordinary vertex-cache deformation machinery rather than a runtime USD evaluator.

### Shared evaluation with explicit output policies

The input supports linear curves and cubic Bezier, B-spline and Catmull-Rom bases. Refinement accepts levels
0 through 256; the default curve level is zero. An authored `refinementLevel` overrides the import option for
both outputs. Refinement controls the interval count, not an exponential subdivision depth: positive level N gives
N+1 intervals per evaluated span where refinement applies.

| Input or policy | Round curves | Ribbons |
| --- | --- | --- |
| Refinement zero | Authored control polygon | Authored control polygon |
| Positive cubic refinement | N+1 intervals per polynomial span | N+1 intervals per polynomial span |
| Positive linear refinement | Keeps authored control points | N+1 intervals per control edge |
| Periodic wrap | Geometry remains open | Closes the strand with a duplicated UV seam |
| Missing UVs | Longitudinal U, V = 0.5 | Transverse U, longitudinal V |

The round writer deliberately preserves its linear and periodic compatibility behavior. Periodic cubic input
warns and uses open geometric scheduling, while attribute counts still come from the authored periodic
topology. Sharing the evaluator does not make these output policies interchangeable.

At positive refinement, pinned B-spline and Catmull-Rom curves reflect endpoint controls and vertex attributes
to evaluate the end spans. They have one span per adjacent authored control pair, including two-control curves.
Pinned linear and Bezier input follows the corresponding nonperiodic behavior. Level zero never introduces
phantom controls into the control-polygon output.

## Runtime flow

### Resolve attributes and choose representation

The adapter first resolves normals and widths, including inherited primvars. A valued `primvars:normals` or
`primvars:widths` replaces its built-in attribute before sample-time collection. Indexed data is flattened before
evaluation. An effective empty normal array selects round output, even when the prim has built-in normals
overridden by an empty primvar. A nonempty array selects ribbons. Later samples cannot change that choice.

Interpolation remains part of each attribute's meaning: constant values apply to the whole input, uniform
values to one strand, vertex values follow control-point basis weights, and varying values interpolate linearly
between span boundaries. Normals and widths need not use the same mode. The adapter supplies authored varying
counts whenever needed; the evaluator keeps those counts separate from geometric sample counts. Round
tessellation accepts empty widths as the default diameter and broadcasts singleton values. The USD adapter
rejects an effective empty width primvar; missing built-in widths retain the default-radius fallback.
[Attribute validation](usd-curve-attributes.md#validate-values-and-preserve-motion-layout) explains these
exceptions. Ribbon validation enforces the declared layout and valid oriented input.

The adapter reads `st` at the initial geometry sample and preserves its interpolation. Authored coordinates
take precedence over generated coordinates and stay static through motion. Material binding and orientation
are also resolved before tessellation.

### Schedule samples and preserve topology

`import_curve_motion` controls motion extraction for both outputs, including ribbons that become meshes.
`import_mesh_motion` alone does not enable ribbon animation. The common geometry-time helper collects times
from points and the effective normal/width sources. Ignored built-in attributes and animated `st` do not add
deformation keys. The helper produces the native time, interpolation and stored-sample mapping used at realization;
with curve motion disabled, the adapter uses the earliest geometry sample.

The first sample establishes the output layout. The adapter checks fixed strand counts, point/attribute
cardinalities and primvar index layouts; type, basis, wrap and interpolation are read once as fixed metadata.
Animated indices are checked as authored data, rather than accepted merely because flattening would leave the
same array length. Each later sample uses the same evaluator and writes directly into the allocated streams;
indices and UVs are not regenerated.

Multi-sample round output retains coincident control edges that may separate later. Multi-sample ribbons retain
collapsed widths and degenerate triangles so vertex correspondence survives. Static ribbon output may discard
degenerate triangles and compact unused vertices. Subsequent mesh deduplication considers deformation streams,
so vertices that coincide initially but separate later are kept distinct. Normal recomputation and tangent
transfer follow the shared mesh processing rules.

### Assemble round segments and ribbon frames

The round writer joins successive samples within each strand and converts interpolated diameter to radius.
Missing widths use twice the input's default radius before that conversion; the default radius is 0.01.
For static cubic control polygons at level zero, exactly coincident adjacent centers can omit their segment
indices. This cleanup does not compact the allocated position array and does not apply to linear curves.

Ribbons emit two edge vertices at each centerline sample. A usable tangent and the authored normal determine
the transverse direction through `cross(normal, tangent)`; half the width offsets the two edges. Adjacent
intervals share one cross-section and use a fixed two-triangle quad split. Left-handed orientation reverses
triangle winding. Usable authored shading normals are normalized and retain any tangent component, rather
than being recomputed from the flat triangles unless explicitly requested. Repaired samples use the transported
perpendicular orientation as their shading normal.

The ribbon writer repairs singular tangent/normal frames using usable neighbors within the same strand.
It transports neighboring frames through tangent changes and blends compatible anchors, including across a
periodic seam. Repair never borrows another strand's orientation or chooses an arbitrary axis when transport
is ambiguous. Missing orientation anchors and ambiguous opposing-frame repairs are diagnosed. Static strands
with no usable tangent can be omitted; motion rejects them because their scheduled layout must survive.
Motion repairs each sample spatially; it does not fill missing orientation from a previous time sample.

Generated ribbon U is zero at one edge and one at the edge toward `cross(normal, tangent)`. V advances from
zero to one along the strand with equal ranges per basis span, or per control-polygon edge at level zero;
it is not normalized arc length. Periodic output duplicates the seam so coincident endpoints carry different
longitudinal coordinates. Authored `st` instead gives both edges the same interpolated centerline coordinate,
without synthesizing a transverse gradient. The adapter therefore derives tangents from normals for authored
`st`, and from UVs for generated coordinates, subject to the common normal-recomputation pass.

## Invariants and rationale

- **One attribute interpretation feeds both writers.** Basis weights, strand offsets and varying interpolation
  must remain consistent across positions, widths, normals and UVs. Equal array lengths do not make vertex and
  varying semantics equivalent. The separate attribute-count page owns the detailed cardinality rules.
- **Representation and topology are fixed for motion.** First-sample normals select the geometry family, and
  later data must fit its streams. Preserving initially degenerate geometry is necessary because it can become
  visible later; static cleanup cannot be reused indiscriminately for animated output.
- **Orientation and shading have different jobs.** The frame determines edge placement, while authored normals
  determine default shading. Projecting every shading normal perpendicular to the tangent would change the
  authored field. Frame repair is local and deterministic, not a promise that all singular inputs are valid.
- **UV conventions are part of the output contract.** Generated ribbon coordinates describe width and spine;
  authored centerline coordinates remain authored. Shared joints, fixed diagonals and finite sampling define
  the rendered triangle surface. Shared cross-sections avoid cracks, but sharp corners can overlap and do not
  preserve constant perpendicular width; tessellation performs no miter or self-intersection repair. The
  implementation does not establish exact parity with another renderer for twisted ribbons.

## Source landmarks

- `src/falcor2/importers/usd_importer/usd_importer_curve_adapter.cpp` - effective attributes, representation
  selection, motion loop and mesh postprocessing.
- `src/falcor2/importers/usd_importer/usd_importer_utils.cpp` - authored counts and shared time/index checks.
- `src/falcor2/importers/curve_tessellator/curve_tessellator.h` and
  `src/falcor2/importers/curve_tessellator/curve_tessellator_internal.h` - borrowed input, scheduling and evaluation.
- `src/falcor2/importers/curve_tessellator/curve_tessellator.cpp` and
  `src/falcor2/importers/curve_tessellator/curve_tessellator_ribbons.cpp` - round and ribbon output policies.
- `src/falcor2/importers/importer_types.cpp`, `src/falcor2/render/scene_import.cpp` and
  `src/falcor2_ext/importers/importer_types.cpp` - owned streams, shared mesh processing, realization and Python views.
- `tests/native/importers/test_ribbon_tessellator.cpp` and `tests/native/usd_importer/test_usd_importer.cpp` -
  basis/interpolation contracts, frame repair, joints, generated coordinates and malformed-input diagnostics.
- `tests/python/importers/test_ribbons.py`, `tests/python/importers/test_ribbon_motion.py` and
  `tests/python/importers/test_curve_motion.py` - shared evaluation, fixed layouts and frozen-sample comparisons.
- `tests/python/render/test_usd_ribbons.py` and `data/assets/test_usd_ribbons/textured.usda` - material/UV/transform
  probes, textured fixtures and dynamic ribbon submission.

## Related design documents

- [Scene loading and authoring](scene-loading-and-authoring.md)
- [USD curve attribute counts and interpolation](usd-curve-attributes.md)
- [Geometry and ray tracing](geometry-and-ray-tracing.md)
- [Geometry deformation](geometry-deformation.md)
- [Falcor2 design documentation](index.md)
