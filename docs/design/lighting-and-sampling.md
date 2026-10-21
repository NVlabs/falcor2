---
status: current
audience: Falcor2 contributors and coding agents
last_verified: 2026-09-15
verified_against: "d4d9693f4cfcc31a65fc6504183f2dcb6630670b"
---

(falcor2-design-lighting-and-sampling)=
# Lighting and sampling

Falcor2 represents authored analytic and environment lights as scene components and derives emissive triangle
emitters from geometry and materials. `LightSystem` and `EmissiveGeometrySystem` turn those mutable scene objects
into synchronized device data, power estimates, and generation counters. Renderers then choose their own
`LightSampler`: uniform selection, radiometric power selection, or power selection augmented by a spatial
`EmissiveTriangleTree`. This separation keeps scene ownership independent of an integrator's sampling strategy.
Light samples carry the complete scene-selection probability needed by transport, while environment emitters and
finite emitters remain separate probability domains because their power estimates have different measures.

## Purpose and scope

This page explains light authoring and radiometry, analytic and environment implementations, emissive geometry,
scene synchronization, and renderer-owned sampling. It also describes the current emissive-triangle hierarchy.
Material emission and device records belong in [Material system](material-system.md); importer conversion of area
lights belongs in [Scene loading and authoring](scene-loading-and-authoring.md); path-tracing consumption belongs in
[Render pipeline](render-pipeline.md) and [BSDF contract](bsdf-contract.md).

A light sampler is a proposal distribution used for next-event estimation. Its probability density function (PDF)
includes both selecting an emitter and selecting a direction or point on that emitter.

## Architecture

### Authoring and radiometry

`Light` is a `Component` attached to an entity, so its world transform and lifecycle follow the scene model. Shared
authoring controls are active state, RGB color in render space, nonnegative intensity, exposure in stops, optional
blackbody color temperature, and normalization. Effective emission is color times intensity and `2^exposure`, with
the normalized temperature tint when enabled. A normalized finite or angular light divides by its final world-space
size factor, keeping the intended emission measure stable when size changes.

Constant and distant lights are environment emitters. A constant light is uniform over the sphere; a distant light
samples a world-oriented angular cone. `EnvMapLight` owns a texture through `TextureManager` and builds a fixed-size
importance map and mip hierarchy used for directional sampling. Point, sphere, disk, and rectangle lights are
finite emitters. Sphere transforms use a volume-equivalent scale under nonuniform transforms; disk and rectangle
area comes from transformed basis vectors. Surface samplers convert their area density to solid-angle density at
the receiver.

Point and analytic area lights can compose `LightShaping`. Shaping defines a cone, soft transition, and focus term
around the light's local negative Z axis without changing the emitter geometry. Its sampling power estimate uses a
compact angular approximation, while the evaluated falloff follows the full shaping expression.

### Environment-map orientation

`EnvMapLight.pole_axis` selects whether the texture's top pole aligns with positive Y or positive Z in the owning
entity's local space; Y is the default. When writing GPU light data, the component composes the entity's world
transform with a positive 90-degree rotation about local X for Z alignment. Evaluation, sampling, and PDF evaluation
use the resulting light orientation. Editing the axis marks render state dirty so the next scene update uploads the
change.

The USD importer resolves `DomeLight_1`'s `poleAxis` value while the stage is available: explicit Y and Z are retained,
and `scene` becomes the stage's up axis. Legacy `DomeLight` uses Y. The resolved value travels through `ImporterLight`
to the runtime environment component because the runtime has no USD stage-axis setting. A dome without a texture
becomes a direction-independent `ConstantLight`.

Pole alignment belongs to the environment light, so it leaves the owning entity and its descendants' transforms
unchanged and requires no synthetic child node. Other light types continue to use their own orientation conventions.

### Synchronized scene representations

`LightSystem` collects active `Light` components during `Scene.update()`. It places analytic lights first and
environment lights in contiguous ranges, assigns snapshot `LightID` values, serializes each concrete reflected
Slang type into fixed records, and publishes type conformances. A GPU kernel evaluates scalar selection weights.
The system exposes topology, geometry, and selection-weight generations so samplers can rebuild only when their
inputs change. The current implementation advances all three generations for any synchronized light change, which
is conservative.

`EmissiveGeometrySystem` scans potentially emissive triangle geometry independently of authored light components.
It combines geometry and material requirements, constructs triangle and instance mappings, evaluates material
emission, and computes one-sided luminance flux. Only active triangles with positive flux enter the sampling set.
Buffers map global triangle IDs to dense active IDs and back, while topology, geometry, and flux generations let a
sampler distinguish a changed set from changed values.

### Renderer-owned sampling strategies

The scene owns emitter data but not a global sampling policy. Each renderer owns a `LightSampler` implementing the
Slang `ILightSampler` interface and updates it after the scene. The reference path tracer defaults to
`PowerLightSampler`; replacing it changes shader specialization and therefore the renderer's linked module.

`UniformLightSampler` gives equal selection probability to active component lights and active emissive triangles.
`PowerLightSampler` builds alias tables for analytic, environment, and emissive sets. Finite analytic lights and
emissive triangles are mixed by estimated flux. Environment and finite domains use unlike power measures, so when
both domains are nonempty the sampler chooses each domain with probability one half, then uses normalized weights
inside that domain. Multiple environment lights form a mixture, and PDF evaluation includes every environment
component that can produce the queried direction.

`HierarchicalLightSampler` preserves power sampling for analytic and environment lights but replaces emissive
triangle alias selection with `EmissiveTriangleTree`. The tree is a flat, binary, preorder hierarchy built over
active triangle IDs. Nodes store a spatial bound, total flux, and an emitter-normal cone; the left child is implicit
and the right child is indexed. Traversal weights children using receiver position and normal, flux, distance,
receiver cosine bounds, and emitter orientation. The default leaf policy is uniform, while a selectable contextual
leaf policy applies the same local reasoning within leaves.

Tree topology is currently built on the CPU using equal partitioning, binned surface-area heuristic (the default),
or binned solid-angle heuristic. If the active set is unchanged, GPU refit kernels can update leaf and internal
bounds, flux, and cones without replacing topology. Sampling and reverse PDF evaluation use the same traversal; an
active-triangle-to-leaf map supports exact evaluation of the distribution that sampling used.

## Runtime flow

```text
authored Light components     emissive materials + triangles
            |                              |
            v                              v
      LightSystem                 EmissiveGeometrySystem
       records + weights          active triangles + flux
            |                              |
            +---------------+--------------+
                            |
                            v
                 renderer-owned LightSampler
                 /           |             \
             uniform       power       hierarchical
                                           |
                                  EmissiveTriangleTree
                            |
                            v
                 LightSample: Li, PDF, emitter ID
                            |
                            v
                    path-tracer NEE and MIS
```

1. Authoring or transform edits mark component and scene state dirty. Material, geometry, instance, or transform
   edits similarly invalidate the emissive-geometry inputs.
2. `Scene.update()` synchronizes component-light records, texture-backed environment maps, emissive triangles,
   selection weights, and generation counters.
3. The renderer updates its selected sampler. Alias tables rebuild from relevant weights; the hierarchical sampler
   rebuilds when active topology changes and may refit when only geometry or flux changes.
4. Shader specialization binds the concrete sampler. A sample first chooses the appropriate emitter domain and
   emitter, then samples its direction or surface and multiplies those conditional probabilities into one PDF.
5. The path tracer evaluates visibility and scattering. It uses the complete light PDF for multiple importance
   sampling when the emitter supports the corresponding competing path.

## Invariants and rationale

Light IDs, active triangle IDs, and sampler tables belong to one synchronized scene snapshot. They are dense lookup
keys, not durable object identities. A renderer must update its sampler after the scene and bind records,
generations, and linked shader requirements from the same state.

Finite and infinite emitters are separate selection domains. Treating unlike power estimates as one normalized
quantity would imply a unit conversion that the system does not define. The fixed half split when both domains are
present is therefore intentional; power weighting still applies within each compatible domain.

The returned PDF is complete. It includes hidden branch choices, emitter selection, and conditional direction or
surface sampling. The environment PDF is a mixture over all relevant environment lights, and hierarchical
`sample()` and `eval_pdf()` must traverse equivalent probabilities. These properties let generic transport perform
MIS without knowing the sampler's internal branch.

Current point and analytic area samples are marked next-event-estimation-only. The reference path tracer therefore
does not compare a later path hit against their light-sampling density. This flag is part of the sampling contract,
not a statement that the light has no surface query methods.

The emissive hierarchy separates stable topology from changing values so refit can preserve buffers and traversal
indices. Its spatial and angular bounds are conservative; they influence efficiency but must not remove a
positive-probability emitter. CPU topology construction and the selectable experimental policies are current
implementation choices, not promises that every scene uses a hierarchy.

## Source landmarks

- `src/falcor2/render/component/light.h` and `src/falcor2/render/component/light.cpp` - shared authoring controls,
  transforms, normalization, shaping, environment pole alignment, concrete light data, and power estimates.
- `src/falcor2/importers/usd_importer/usd_importer_context.cpp` and `src/falcor2/render/scene_import.cpp` - USD dome
  pole-axis resolution and transfer into runtime environment lights.
- `src/falcor2/render/light_system.h` and `src/falcor2/render/light_system.cpp` - active-light ordering, IDs, records,
  type conformances, selection weights, generations, and scene binding.
- `src/falcor2/render/emissive_geometry_system.h` and `src/falcor2/render/emissive_geometry_system.cpp` - emissive
  triangle discovery, emission evaluation, flux, mappings, generations, and device buffers.
- `src/falcor2/render/light_sampler.h` and `src/falcor2/render/light_sampler.cpp` - uniform, power, and hierarchical
  sampler ownership, invalidation, alias tables, specialization, and binding.
- `src/falcor2/render/emissive_triangle_tree.h` and `src/falcor2/render/emissive_triangle_tree.cpp` - hierarchy
  construction, packed nodes, options, mappings, rebuild, and refit control.
- `slang/falcor2/render/light.slang`, `slang/falcor2/render/light_system.slang`, and
  `slang/falcor2/render/lights/` - device light interfaces, records, radiometry, shaping, and concrete sampling.
- `slang/falcor2/render/light_sampler.slang`, `slang/falcor2/render/emissive_geometry.slang`, and
  `slang/falcor2/render/emissive_triangle_tree.slang` - complete-scene sampling, domain mixtures, emissive-triangle
  queries, tree traversal, and PDF evaluation.
- `falcor2/rendernodes/reference_pathtracer_node.py` and
  `slang/falcor2/rendernodes/reference_pathtracer.slang` - renderer ownership, specialization, binding, next-event
  estimation, and multiple importance sampling.
- `tests/python/render/test_light_sampler.py`, `tests/python/render/test_light_surface_sampling.py`,
  `tests/python/render/test_hierarchical_light_sampler.py`, and
  `tests/native/render/test_emissive_triangle_tree.cpp` - radiometry, distributions, sample/PDF agreement,
  invalidation, rebuild/refit, packing, and conservative bounds.
- `tests/native/usd_importer/test_usd_importer.cpp` and `tests/native/render/test_scene.cpp` - dome pole-axis import,
  preservation of child transforms, runtime property transfer, and textureless fallback.

## Related design documents

- [Falcor2 system overview](overview.md)
- [Scene model](scene-model.md)
- [Render pipeline](render-pipeline.md)
- [Scene loading and authoring](scene-loading-and-authoring.md)
- [Geometry and ray tracing](geometry-and-ray-tracing.md)
- [Material system](material-system.md)
- [BSDF contract](bsdf-contract.md)
- [Falcor2 design documentation](index.md)
