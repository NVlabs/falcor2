---
status: current
audience: Falcor2 contributors and coding agents
last_verified: 2026-11-11
verified_against: "937fa44437ce197d952f959179a6ffddac523487"
---

(falcor2-design-lighting-and-sampling)=
# Lighting and sampling

Falcor2 separates authored emitters, synchronized scene data and renderer-owned proposal distributions.
`LightSystem` manages analytic and environment components, including procedural surfaces for finite area lights;
`EmissiveGeometrySystem` derives triangle emitters from materials and geometry. Renderers choose uniform,
power-weighted or hierarchical light sampling. Samples expose unweighted radiance and the complete proposal
probability, while retained references preserve an endpoint for evaluation at another receiver. The reference
path tracer combines next-event estimation with emitter-hit transport through multiple importance sampling.
Analytic visibility and hidden-surface continuation determine which hits terminate a path without changing the
meaning of the sampling distribution.

## Purpose and scope

This page explains emitter ownership, radiometry, sampling distributions, retained endpoints and their transport
contract. Next-event estimation (NEE) explicitly samples an emitter; multiple importance sampling (MIS) combines
that technique with paths that reach an emitter through scattering. A probability density function (PDF) measures
the proposal used, including emitter selection at scene level.

[Material system](material-system.md) covers material emission, [Scene loading and authoring](scene-loading-and-authoring.md)
covers asset translation, and [Geometry and ray tracing](geometry-and-ray-tracing.md) covers shared acceleration
structures. [Render pipeline](render-pipeline.md) and [BSDF contract](bsdf-contract.md) cover renderer composition
and scattering.

## Architecture

### Authored lights and radiometry

`Light` is an entity-owned component. Effective emission is render-space RGB color times nonnegative intensity,
`2^exposure` and an optional luminance-normalized blackbody tint. Normalization divides by the concrete light's
world-space size factor where applicable; point intensity and surface radiance remain different quantities.
Point, sphere, cylinder, disk and rectangle lights compose `LightShaping`, which evaluates cone softness and focus
around local negative Z without changing the emitting surface. A shaped point is still a `PointLight`.
Selection power uses an angular-midpoint approximation for soft shaping; emitted radiance uses the full profile.

Constant, distant and texture-backed environment lights emit at infinity. `EnvMapLight` owns its texture through
`TextureManager` and builds an importance-map mip hierarchy. Its local pole defaults to Y; Z alignment composes
a positive 90-degree X rotation into the light record, leaving entity and child transforms unchanged. USD
`DomeLight_1` resolves `scene` pole alignment against the stage up axis during import; legacy domes use Y.
A textureless imported dome becomes a constant light. An unassigned runtime environment map contributes no light;
a black importance map has no sampling distribution and evaluates to zero PDF.

Finite sphere, cylinder, disk and rectangle lights have analytic surfaces. Sphere radius uses volume-equivalent
transform scale; its sampler chooses directions uniformly over the visible solid-angle cone, then intersects
the sphere. Disk, rectangle and cylinder samplers use area densities converted to solid angle. Planar surfaces
use transformed basis vectors. Cylinders emit only from the open lateral surface along local X; axial and uniform
radial scaling are exact, while radial nonuniform scale or shear uses an area-equivalent circular approximation,
with a warning during USD import.

Opt-in virtual sphere shrinking makes the sampled sphere receiver-dependent and compensates radiance for its
smaller area. USD import enables this compatibility option for recognized Unreal provenance. Samples that use
the proxy remain NEE-only and cannot retain a reusable endpoint; authored-surface hits and reevaluation are
unavailable in those receiver contexts. Ordinary sphere samples support complementary hit transport.

### Synchronized emitter data and geometry

`LightSystem` collects active components, places finite analytic records before environment records, assigns
snapshot `LightID` values and publishes reflected device records, conformances and selection weights. Topology,
geometry and selection-weight generations currently advance together on a synchronized light change.

For analytic area lights, the system owns one canonical procedural bounding box and shared bottom-level
acceleration structure (BLAS), with persistent per-light instances in the top-level structure (TLAS). Instance
data carries the current light ID and visibility flag; shader intersection resolves the actual surface.
Radiometric changes retain instances, while transforms, masks and identity are updated as needed. Adding the
first or removing the last area-light geometry changes the geometry-type requirements and ray-tracing layout;
it need not recompile an otherwise unchanged linked shader module.

`EmissiveGeometrySystem` discovers potentially emissive triangles and estimates their material emission.
Positive estimated emission defines the dense active mapping; active membership does not guarantee positive
flux after geometry degenerates. World-space area and emission determine one-sided luminance flux. Independent
topology, geometry and flux generations track changes. Rigid movement of contributing entities can preserve
cached flux; deformation and area-changing transforms invalidate it. Unrelated nonemissive transforms do not
require rebuilding light distributions.

### Renderer-owned sampling

The scene owns emitter data; each renderer owns its `LightSampler` and updates it after the scene. The reference
path tracer defaults to `PowerLightSampler`. Replacing the concrete sampler changes shader specialization.
`UniformLightSampler` selects equally among active component lights and active emissive triangles, independently
of intensity.

`PowerLightSampler` builds alias tables for finite analytic, environment and emissive-triangle sets. Compatible
finite flux estimates determine the analytic/triangle mixture. Environment estimates measure directional
radiance integrated over solid angle, with different units from finite flux. When both domains have positive
estimates, selection gives each domain probability one half and uses relative weights within it. Alias tables
track relevant topology and weight/flux changes; pure emissive geometry motion need not rebuild them.

`HierarchicalLightSampler` keeps analytic/environment power selection and replaces triangle selection with
`EmissiveTriangleTree`. CPU construction produces a flat binary tree with spatial bounds, flux and normal cones;
binned surface-area splitting is the default, with equal and solid-angle alternatives. Traversal considers the
receiver position and normal. Leaves default to uniform selection, with an optional contextual policy.
Unchanged active triangle IDs and order allow GPU refitting of bounds, cones and flux; changed membership or
build options require rebuilding. Sampling and reverse PDF evaluation use matching branch and leaf probabilities.

## Runtime flow

1. Authoring, material or geometry edits mark scene data dirty. `Scene.update()` synchronizes emitter records,
   procedural light instances, triangle emission, weights and generations.
2. The renderer updates and binds its sampler from that scene state. `ILight.sample()` consumes explicit `float2`
   coordinates; `ILightSampler.sample()` consumes `LightSampleInput`, separating emitter selection from light sampling.
3. The sampler chooses an emitter and evaluates a direction or endpoint. `LightSample.radiance` is unweighted,
   unoccluded incident radiance; `weight` is radiance divided by PDF, or zero when PDF is zero. Individual-light
   PDFs are conditional; scene-sampler PDFs include selection. Point-light probabilities are discrete; other
   sample PDFs use solid angle.
4. The path tracer evaluates scattering and visibility and combines competing techniques. Environment sampling
   and hits use summed environment radiance and the marginal PDF over all relevant components.

### Analytic visibility and emitter-hit MIS

`geometry_visible` defaults to false and controls the area-light surface's transport behavior:

| Surface state | Primary rays | Shadow rays | Secondary radiance rays |
| --- | --- | --- | --- |
| Hidden | Ignore surface | Ignore surface | Accumulate emission and continue |
| Visible | Terminal hit | Block | Terminal hit |

Hidden non-emitting back faces are skipped. Visible planar surfaces remain blockers from either side.
Point lights have no corresponding surface. For supported emitter hits, the path tracer uses the previous
scatter's non-delta state and actual NEE hemisphere coverage to decide whether a competing technique exists.
With NEE enabled and MIS disabled, covered sampleable hits are suppressed to avoid double counting. With MIS,
analytic, triangle and environment hits use their complete sampler PDF; delta and uncovered paths retain their
hit contribution.

Only triangles currently participate in material-surface light sampling. Native curves and other unsupported
emissive surfaces retain hit emission even when triangle NEE is enabled without MIS. Curve emission does not
imply curve-light sampling support.

A hidden analytic hit advances the prepared segment's lower distance bound without consuming path depth.
It preserves the original terminal distance, sampled homogeneous-medium event, stochastic-opacity evaluator,
scatter PDF and coverage. Intermediate emission evaluates attenuation from the original segment origin without
committing throughput repeatedly. Resampling at each hidden surface would change the proposal and medium weight.

### Retained endpoints

`LightSampleRef` stores emitter type, emitter ID and two intrinsic parameters in 16 bytes. Parameters describe
triangle barycentrics or an analytic surface location; sphere normals and infinite world directions use
octahedral encoding. The reference contains no PDF, visibility or reservoir weight. Ordinary sampling overloads
can discard it.

`eval_sample()` reconstructs that endpoint for a destination receiver without random inputs and computes the
current proposal density. Reconstruction may succeed with zero radiance or zero PDF, including a radiating
triangle outside the hierarchy's receiver-normal support. Coincident connections fail. Scene-level environment
reevaluation uses the complete environment mixture, independently of which component originally supplied the
direction. Virtual sphere proxies cannot supply or consume retained references.

## Invariants and rationale

Scene IDs, active triangle mappings, sampler distributions and linked requirements must describe the same
synchronized state. IDs are lookup keys, not permanent object identities. Retained references require unchanged
emitter identities and parameterizations; their `is_valid()` check only says an endpoint type was retained.
Callers remain responsible for reference contents, and the interface supplies no temporal identity remapping.

A complete PDF includes every hidden domain, emitter and conditional-sampling choice. Direction evaluation,
surface-hit evaluation and retained reevaluation must use the proposal appropriate to the current receiver.
Keeping unweighted radiance separate from proposal weight preserves meaningful endpoint evaluation when the
current proposal has zero support. Reference reconstruction does not establish visibility.

NEE-only is a property of the available competing strategy, not of the emitter being analytically represented.
Point and virtual-proxy samples use it; authored analytic surfaces can participate in hit MIS. Visibility controls
path termination and occlusion separately. Conservative tree bounds and matching sample/PDF traversal preserve
the intended distribution; receiver-context restrictions can still intentionally assign zero proposal support.

## Source landmarks

- `src/falcor2/render/component/light.h` and `src/falcor2/render/component/light.cpp` - authoring, radiometry,
  transforms, shaping and environment orientation/resources.
- `src/falcor2/importers/usd_importer/usd_importer_context.cpp` and `src/falcor2/importers/light_import.cpp` -
  USD classification, pole-axis resolution and shared runtime transfer.
- `src/falcor2/render/light_system.cpp` and `src/falcor2/render/emissive_geometry_system.cpp` - records, procedural
  instances, generations, triangle emission and flux updates.
- `src/falcor2/render/light_sampler.cpp` and `src/falcor2/render/emissive_triangle_tree.cpp` - distribution ownership,
  invalidation, hierarchy construction and refit.
- `slang/falcor2/render/light.slang` and `slang/falcor2/render/light_sampler.slang` - sample/reference contracts,
  complete PDFs, mixtures and endpoint evaluation.
- `slang/falcor2/render/lights/` and `slang/falcor2/render/emissive_triangle_tree.slang` - concrete emitters,
  procedural surface queries and matching tree traversal.
- `slang/falcor2/render/kernels/emissive_geometry_kernels.slang` - active mapping and area/flux computation.
- `falcor2/rendernodes/reference_pathtracer_node.py` and `slang/falcor2/rendernodes/reference_pathtracer.slang` -
  renderer sampler ownership, NEE, hit MIS and segment continuation.
- `src/falcor2_ext/render/scene.cpp` and `src/falcor2_ext/render/light_sampler.cpp` - reflected light/sampler exposure.
- `tests/python/render/test_light_sampler.py`, `tests/python/render/test_light_surface_sampling.py` and
  `tests/python/render/test_light_sample_ref.py` - distributions, visibility, surface PDFs and endpoint reconstruction.
- `tests/python/render/test_hierarchical_light_sampler.py` and `tests/native/render/test_emissive_triangle_tree.cpp` -
  rebuild/refit, contextual support, traversal agreement and conservative packing.
- `tests/python/pathtracer/test_pathtracer.py` and `tests/python/pathtracer/test_curve_emission.py` - hit energy,
  hidden continuation, opacity/media, shadow implementations and unsampled curve emission.
- `tests/native/usd_importer/test_usd_importer.cpp` and `data/scenes/emissive-strength.py` - in-memory USD light
  examples and material/environment authoring.

## Related design documents

- [Sampling inputs](sampling-inputs.md) - caller budgets and coordinate reuse across sampling hierarchies.
- [Falcor2 system overview](overview.md)
- [Scene model](scene-model.md)
- [Render pipeline](render-pipeline.md)
- [Scene loading and authoring](scene-loading-and-authoring.md)
- [Geometry and ray tracing](geometry-and-ray-tracing.md)
- [Material system](material-system.md)
- [BSDF contract](bsdf-contract.md)
- [Falcor2 design documentation](index.md)
