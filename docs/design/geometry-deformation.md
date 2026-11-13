---
status: draft
audience: Falcor2 contributors and coding agents
last_verified: 2026-11-11
verified_against: "579ee8fb3fcd71df5935f2c5b787783b088775e8"
---

(falcor2-design-geometry-deformation)=
# Geometry deformation

Falcor2 evaluates fixed-topology geometry motion into pooled current and previous GPU vertices. Each dynamic
geometry owns its topology and one typed deformer; render collections own output storage and shared updaters
that batch deformation work. The implemented vertex-cache deformer selects and interpolates fully resident
samples, tracks evaluated-output history and supplies conservative bounds. Importers translate source animation
into native sample references before runtime evaluation. This separation keeps sample ownership, output
allocation and dispatch scheduling explicit while letting ray tracing, motion vectors and emissive-light updates
consume the same final geometry.

## Purpose and scope

This page explains who owns deformation inputs and outputs, how time selects stored samples, and how an update
produces coherent current/previous vertices. It covers triangle meshes and linear swept spheres (LSS), Falcor2's
round-curve representation.

[Geometry and ray tracing](geometry-and-ray-tracing.md) covers pool layout, additional attributes, acceleration
structures and traversal. [Scene loading and authoring](scene-loading-and-authoring.md) covers decoding and
realization; source-specific tessellation and USD composition are importer responsibilities.

## Architecture

### Geometry, deformer and updater ownership

`DynamicMeshGeometry` and `DynamicCurveGeometry` derive directly from `Geometry`. Each owns prepared topology,
initial packed vertices, static attributes and one `unique_ptr<GeometryDeformer<Vertex>>`. Meshes provide one
output range per submesh; curves provide one. Installing new data replaces topology and deformer together and
requires the geometry to belong to a Scene.

`GeometryDeformer<Vertex>` names the final packed output format. Its input representation and count need not
match that output. Configuration, output layout and animation range stay fixed until reconfiguration. The
deformer owns or references its inputs and decides when current and previous outputs need work. RenderScene's
mesh and LSS collections own the actual pooled output buffers.

Geometry attaches its deformer only after render outputs exist, supplying a retained callback that does not
own the geometry. The callback resolves each range's current/previous pointers and vertex count, publishes
bounds, and marks current-output changes. These pointers address the range starts and are resolved for each
update that needs work. The deformer consumes these targets during preparation and does not retain them.

Each collection has a lazy updater registry. `get_or_create_geometry_updater<Updater>()` selects the collection
using the updater's vertex type and returns a stable, heap-owned instance for that concrete updater type.
Registry entries are ordered by execution phase. Production currently uses the vertex-cache phase; this is
batched execution, with no implemented deformer chain or dependency graph.

`VertexCacheUpdater<Vertex>` holds borrowed pointers to attached deformers, reusable work records and a lazy
compute kernel. It prepares participants and dispatches their combined work. It owns neither source samples nor
final output buffers. Replacement and removal detach the deformer before destroying outputs. Updater destruction
clears surviving participants' borrowed links and callbacks, so externally retained geometry can later be
destroyed safely after its Scene.

### Vertex-cache inputs

`create_vertex_cache()` takes a Device and sample descriptions, returning an owned, unattached deformer together
with matching topology outputs. The caller explicitly installs both. Mesh setup computes submesh remaps from
the first sample and applies them to every sample; curve setup similarly fixes its topology and remap. Static
attributes come from the first sample. Bounds include all packed samples, including curve radii.

`VertexCacheDeformer<Vertex>` owns packed sample stores, copied timestamps, an optional timeline-to-sample map,
interpolation mode, the selected source pair, history state and conservative bounds. Each `GeometrySampleStorage`
retains its CPU samples and fully resident immutable GPU copies. Buffers contain whole samples in sample-major
order, with a maximum size capped at 2 GiB; one sample must fit within that limit. Large stores can span several
buffers. Preparing a sample resolves its buffer and element offset and declares the source resource state. It
does not stream samples into a bounded GPU cache.

Timeline keys, stored sample IDs and GPU locations are separate identities. A timeline may reference the same
stored pose repeatedly without duplicating its GPU vertices. The optional sample-index map expresses that reuse;
an empty map gives one stored sample per key.

### Importer boundary

Importer motion records carry times in seconds, interpolation and optional sample indices. Scene realization
selects dynamic geometry for records with multiple motion keys and installs the native cache through its
factory. Python exposes these importer records and the dynamic geometry classes; native deformer installation
and `create_vertex_cache()` are not Python authoring APIs.

For USD clips, USD resolves composition and source time. The importer builds an event schedule, interns complete
tuples of the selected deformation attributes before tessellation, and emits native times and sample references.
Mesh tuples include points and applicable authored normals; curve tuples include points and any effective normal
and width values present. Hash matches require exact value equality, so matching positions alone do not establish
sample identity. Oriented curves tessellate into triangle ribbons and follow the mesh deformation path.

Native timestamps use float seconds. Clip import rejects distinct ordinary events that collapse during
conversion; synthetic jump events become explicit left/right pairs. Runtime evaluation consumes this native
representation without consulting USD. Repeated timeline events can still require repeated USD reads during
import.

## Runtime flow

### Select inputs and decide history

Each geometry collection updates its pools before preparing its updaters. Preparation runs even at unchanged
Scene time, allowing a deformer to inspect other inputs. Vertex-cache preparation selects a source pair and
interpolation weight, then compares that tuple with the previous selection. This compares evaluation inputs,
not timeline-key identity or numerical equality of entire resulting vertex arrays.

Times must be finite and nondecreasing, with at most two keys at one time. Step and linear interpolation clamp
finite queries outside the range to endpoints; a nonfinite query selects the first sample. At a paired-time jump,
an exact boundary query chooses the right value and the preceding linear interval approaches the left value.
Equal resolved source IDs collapse to a single source with zero interpolation weight.

The deformer owns the history state machine:

| Condition | Output work |
| --- | --- |
| First evaluation or forced initialization | Evaluate current and write the same result to previous. |
| Changed selection | Preserve old current in previous, then evaluate current. |
| First unchanged update after a change | Copy current to previous to clear motion. |
| Further unchanged updates | No output work. |

Previous therefore means the preceding evaluated final result, including a seek. It does not mean the adjacent
source sample or a fixed time offset. The first unchanged update is called catch-up. Catch-up changes only
previous vertices and does not mark current geometry deformed or invalidate acceleration structures and
emissive geometry. The no-work path returns before resolving output ranges or source buffers.

### Dispatch and publish results

When work exists, the output callback resolves current ranges after pool allocation, and source preparation
declares shader-resource states that retain inputs through execution. The deformer appends one direct-pointer
work record per output range. The updater assembles records across participants; dispatch spans vertices and
records and tiles work to device dispatch limits.

The collection transitions both members of its dynamic output tuple to unordered access, dispatches the batch,
then restores shader-resource access. Mesh kernels interpolate positions and normalize interpolated normals and
tangents. LSS kernels interpolate positions and radii. Other packed fields and additional attributes remain
static. Initialization writes both outputs; changed evaluation captures current before replacing it; catch-up
only copies current to previous.

Procedural LSS bounds are generated after deformation. RenderScene refreshes the common buffer-pointer table
after geometry collection updates and then updates acceleration structures. Scene updates emissive geometry
after RenderScene, so light sampling sees the same deformed surfaces as traversal.

## Invariants and rationale

All samples must describe the same topology and vertex correspondence. Cache factories check matching vertex
counts, but do not comprehensively prove later-sample topology equivalence. Producers must preserve that
contract through decoding and tessellation. Topology and additional attributes stay static because each cache
evaluation writes into a fixed final layout.

Conservative bounds cover the cache's entire motion range, avoiding a CPU readback of each evaluated pose.
History uses final outputs so arbitrary seeks and repeated timestamps have a consistent motion-vector meaning.
Separating immutable source storage from writable output pools lets many geometries share a dispatch without
sharing sample ownership.

Creating, destroying or updating dynamic geometry currently forces initialization of every attached deformer
in that output collection. Dynamic pool reconfiguration therefore resets both outputs and their history;
clients cannot assume history survives pool growth. Source residency and output-history preservation are
separate concerns.

Deformer attachment and borrowed updater references are valid only while their owning render collection exists.
Safe destruction of retained geometry after Scene teardown does not imply that it can still be evaluated.

## Source landmarks

- `src/falcor2/render/geometry/geometry_deformer.h` and `src/falcor2/render/geometry/geometry_updater.h` -
  ownership, output callbacks and updater registry.
- `src/falcor2/render/geometry/dynamic_mesh_geometry.cpp` and
  `src/falcor2/render/geometry/dynamic_curve_geometry.cpp` - topology installation, attachment and output callbacks.
- `src/falcor2/render/geometry/vertex_cache_deformer.h`,
  `src/falcor2/render/geometry/vertex_cache_deformer.cpp`,
  `src/falcor2/render/geometry/vertex_cache_updater.cpp`,
  `src/falcor2/render/geometry/geometry_animation.h`,
  `src/falcor2/render/geometry/geometry_animation.cpp` and
  `src/falcor2/render/geometry/geometry_sample_storage.h` - factories, selection, history and storage.
- `src/falcor2/render/render_scene.cpp`, `src/falcor2/render/scene.cpp` and
  `slang/falcor2/render/kernels/geometry_animation.slang` - scheduling, downstream updates and GPU evaluation.
- `src/falcor2/importers/usd_importer/usd_importer_utils.cpp`,
  `src/falcor2/importers/usd_importer/usd_importer_mesh_adapter.cpp`,
  `src/falcor2/importers/usd_importer/usd_importer_curve_adapter.cpp` and `src/falcor2/render/scene_import.cpp` -
  clip lowering and native realization.
- `src/falcor2_ext/importers/importer_types.cpp` and `src/falcor2_ext/render/mesh.cpp` - Python exposure boundaries.
- `tests/native/render/test_dynamic_geometry.cpp` and `tests/native/render/test_geometry_sample_storage.cpp` -
  timing/history, GPU outputs, lifecycle and storage boundaries.
- `tests/python/render/test_mesh_motion.py` and `tests/python/importers/test_geometry_clips.py` - executable
  examples and integration coverage for motion, seeking and clip reuse.

## Related design documents

- [Geometry and ray tracing](geometry-and-ray-tracing.md)
- [Scene loading and authoring](scene-loading-and-authoring.md)
- [Scene model](scene-model.md)
- [Buffer pointers and indirect GPU access](buffer-pointers.md)
