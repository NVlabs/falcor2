---
status: draft
audience: Falcor2 contributors and coding agents
last_verified: 2026-11-11
verified_against: "de2eda3d86ff9104cd529812696806a62989fe44"
---

(falcor2-design-render-pipeline)=
# Render pipeline

Falcor2's render pipeline composes Python render nodes around a scene-specialized Slang path tracer.
`PathTracerPipeline` synchronizes the scene, chooses a camera, renders linear radiance, accumulates or reconstructs
samples, optionally denoises them, and tone maps the result. Scene requirements determine shader and ray-tracing
setup, while renderers own output storage, dispatch specialization, pixel filters, and temporal history. These
resources have different reuse rules: a scene edit can preserve a linked module while changing its dispatch or
invalidating accumulated color. Containers carry the resulting images to the editor, headless callers, or another
Python processing stage.

## Purpose and scope

For contributors familiar with Python and path tracing, this page explains how a synchronized scene becomes a
frame and which changes invalidate modules, dispatches, or image history. It covers ownership, sample delivery,
pixel filters, denoiser integration, and presentation.

[Scene model](scene-model.md) explains requirement derivation and scene updates; [Geometry and ray tracing](geometry-and-ray-tracing.md)
explains traversal; [Lighting and sampling](lighting-and-sampling.md) explains light selection. Transport and
denoiser algorithms are outside this page.

## Architecture

### Nodes, containers, and outputs

`RenderNode.forward()` tracks nested calls and invokes the subclass's `_exec()`; the outermost node supplies
the active editor's settings tree. `PathTracerPipeline` owns and explicitly orders its tracer, accumulators,
denoisers, and tone mapper.

`ContainerSpec` describes container kind, format, and logical `(height, width)`, resolving automatic fields against
a fallback. `Container` supports SGL textures, SlangPy tensors, and supported CUDA Torch tensors. Torch exposes
channel-first images adapted at shader-call boundaries. Nodes can reuse non-Torch output allocations; callers
must copy results that need to survive later calls overwriting them. Temporary Torch outputs are newly allocated.

`ReferencePathTracerNode` owns color and guide targets. `OutputPrelude` generates writers for enabled fixed guides
and indexed material AOVs (auxiliary outputs). Formats and target types affect generated writers; actual resources
bind on each call. Requested material names must already exist in the synchronized scene catalog. Storage remains
renderer-owned, as described in [Named material outputs](material-aovs.md).

### Requirements, modules, and dispatch reuse

After `Scene.update()`, `SceneShaderHelper.prepare()` returns a linked module and the exact immutable
`SceneRequirements` snapshot used to prepare it. Helper and scene must share a device. The reference tracer
defaults to `scene_capability_policy=required`, selecting scene and consumer demand, including interior tracking
when `solid_transmission` is selected. `all` explicitly enables the broader feature set; `specified` validates
the supplied mask against supported and required capabilities. Required specialization can reduce shader work
while producing more variants across scenes.

The preparation and dispatch layers compare different inputs:

| Layer | Reuse condition |
| --- | --- |
| Requirement resolution | Same scene requirement generation, policy, specified mask, requested AOV names, and explicit consumer capabilities. |
| Linked module | Same source module and matching snapshot shader inputs: capabilities, conformances, and underlying Slang component identities. |
| Tracer dispatch | Same constants, guide-writer signature, light-sampler type and shader generation, sample-generator choice, and full requirements snapshot. |

The helper copies AOV sequences into its resolution key, so editing the same list is observed. A new requirement
generation triggers resolution, not necessarily compilation. Full snapshot comparison additionally includes
geometry types, hit-group policy, LSS intersection mode, and pipeline flags. Adding the first geometry of a type
can therefore preserve shader inputs while changing ray-tracing setup. Equivalent scenes can reuse a module while
binding different scene data.

The tracer passes the returned snapshot to `SceneRayTracingSetup.create()` for hit groups, shader-table ordering,
and flags. Its generated prelude also includes ray-tracing metadata so downstream SlangPy code-based caching
distinguishes pipeline layouts. Runtime settings such as maximum depth and background color bind per call.
Changing these values or replacing a guide target with the same signature need not change specialization.

The Python helper retains its current module and the tracer its current dispatch. Failed helper preparation
preserves the previous successful module, snapshot, keys, and bound scene; invalid inputs retry on subsequent calls.


### Pixel filters and camera jitter

`ReferencePathTracerNode.pixel_filter` is a renderer-owned native `ReconstructionFilter`, assigned through a plain
Python field. The default is `GaussianReconstructionFilter(sigma=0.5, radius=1.5)`.
Filter distances are in input-render pixels, independent of final display resolution.


Slang's `IReconstructionFilter.sample()` takes two caller-supplied uniform values and returns an offset and
importance weight: normalized kernel value divided by sampling density. `eval()` returns the normalized kernel
per square input pixel, zero outside support. Neither method exposes a probability density function (PDF).

- Box samples a square of full `width`, with unit weights. The default width is one pixel. Width zero samples
  the center with weight one and evaluates to zero as a delta distribution.
- Gaussian is the normalized product of two boundary-subtracted kernels,
  `max(exp(-x*x/(2*sigma*sigma)) - exp(-radius*radius/(2*sigma*sigma)), 0)`. It uses square support and truncated
  Gaussian proposals with compensating weights.

Before the camera UV Y flip, film position is `pixel + 0.5 + pixel_offset + filter_offset`.
`pixel_offset` defaults to zero and is supplied per call. Samples contribute only to the owning output pixel,
without splatting or clamping to pixel/image boundaries. Two random values are consumed even for width-zero Box,
preserving subsequent sampling dimensions. Radiance receives the filter weight; guides remain unweighted.

`ReconstructionFilterMarshall` resolves the native object's concrete Slang type and delegates parameter upload
to its cursor writer. Signatures contain the type, excluding parameters, generation, and object identity.
Same-type replacements and parameter edits reuse specialization while uploading current values. Binding validates
finite, representable parameters and prepares Gaussian normalization/sampling constants; construction alone does
not validate them.


### Filter edits and temporal ownership

Filter construction and persistent reflected writes receive globally unique generations, including same-value
writes. Direct C++ field edits require `notify_changed()`. The pipeline polls its effective filter: replacement
or edits request a history reset independently for each consumer. Reassigning the same unchanged object does not.
Generation controls history invalidation; parameters upload on every dispatch regardless.

On entering DLSS Ray Reconstruction, the pipeline saves the ordinary filter and installs a persistent zero-width
Box; on exit it restores the saved filter. The swap occurs only on transitions, so callers select their ordinary
filter outside DLSS mode. Assignments during DLSS are not intercepted. Each frame passes shared Halton jitter as
`pixel_offset` and reports its negative to DLSS. Width-zero Box adds no per-ray film jitter. The separate DLSS
Super Resolution example uses the same shared-offset policy for all color and guide samples.

Changing `pixel_offset` does not advance filter generation or request reset; callers changing a fixed target
position must invalidate history themselves. Standalone tracer users own accumulation and its invalidation.
Ordinary filtering keeps a fixed kernel center; averaging independently shifted centers changes the effective
kernel. DLSS's temporal reconstruction does not establish equivalence to ordinary filtered accumulation.

## Runtime flow

The pipeline first updates the scene, then resolves the explicit or active camera. Every sample in that call sees
the same synchronized scene. A standalone tracer expects callers to perform this update themselves. The pipeline
selects the effective filter and checks history before rendering.

```text
Scene.update -> camera -> tracer: linear color + guides
                            |                  |
                  accumulation -> OptiX     DLSS-RR
                            |                  |
                            +---- tone map ----+
                                      |
                           returned container -> consumer
```

OptiX and tone mapping are optional. DLSS bypasses both ordinary accumulators and OptiX. The normal branch uses:

| Accumulation mode | Samples and delivery |
| --- | --- |
| `off` | One raw sample per call, ignoring `spp`. |
| `normal` (default) | `spp` samples, delivered as equal-size batch means using `samples_per_dispatch` (default one). |
| `weighted` | Each of `spp` raw samples enters the luminance-binned estimator separately, followed by one resolve. |

Batching above one requires normal accumulation without DLSS, and `spp` must be divisible by batch size.
Each shader batch uses consecutive sample indices and averages already filter-weighted radiance. The pipeline
advances its iteration by sample count; the conventional accumulator counts batch means. Equal-size batches
preserve relative sample weighting, but sanitization and luminance clamping act on each batch mean, so batching
need not reproduce individually clamped samples. Guide writes retain their overwrite/increment semantics rather
than becoming averaged guides.

Before dispatch the tracer resolves outputs, updates its light sampler, prepares requirements and dispatch,
and binds scene, camera, filter, settings, and guide resources. Previous-camera state supplies motion-vector guides.
OptiX receives linear color after the selected non-DLSS accumulation policy and automatically requested
albedo/normal guides. DLSS instead renders one sample at its chosen internal resolution, requires its complete
guide set, and reconstructs the target size.

`TonemapperNode` follows either branch, preserving container kind. ACES film is the default curve; auto exposure
is disabled by default. When enabled in its default `previous_frame` mode, it applies retained exposure, meters
the current image, and adapts the next exposure using `delta_time`; initialization applies current exposure
immediately. The pipeline returns an image or `(image, guides)`. DLSS guides also include its low-resolution input
color. The editor presents supported containers; headless consumers can process or save them.

## Invariants and rationale

Shader reuse and valid image history are independent. Normal accumulation resets its iteration and both color
histories for pending resets, changed scene update generation, or changed camera uniforms. Changing batch size,
sample generator, or effective filter requests reset. Full `reset()` additionally clears previous-camera state,
NaN tracking, DLSS history, and exposure history. Programmatic tracer-setting changes may require an explicit
history reset even when the next dispatch correctly receives the new setting.

DLSS consumes pending resets and separately keys history on input size, output size, and quality. It does not run
the normal scene/camera reset check; callers remain responsible for other invalidation. Allocation compatibility
likewise differs from history validity: the conventional accumulator restarts when its complete input specification
changes, and pipeline output-spec edits request reset.

Color stays linear through tracing, accumulation, and denoising. Ordinary averaging divides by sample/batch count,
not by a random sum of filter weights. Weighted firefly suppression is a distinct estimator; filter importance
sampling does not make it unbiased. Only normalized nonnegative reconstruction filters are supported.

Stages accept a caller-owned command encoder where supported, but `PathTracerPipeline` calls `Scene.update()`
separately without forwarding that encoder. OptiX can append to a supplied encoder only on CUDA; a graphics-backend
encoder cannot supply its required CUDA stream. Guide names, formats, dimensions, and backend support remain
consumer contracts even when storage uses a common container abstraction.

## Source landmarks

- `falcor2/rendergraph/render_node.py`, `falcor2/rendergraph/container_spec.py`,
  `falcor2/rendergraph/container.py`, and `falcor2/rendergraph/output_prelude.py` - node execution, storage, and writers.
- `falcor2/editor/scene_shader.py`, `src/falcor2/render/scene_requirements.h`,
  `src/falcor2/render/ray_tracing_setup.cpp`, and `src/falcor2_ext/render/scene.cpp` - preparation and snapshot contracts.
- `falcor2/rendernodes/pathtracer_pipeline_node.py`, `falcor2/rendernodes/reference_pathtracer_node.py`, and
  `slang/falcor2/rendernodes/reference_pathtracer.slang` - orchestration, dispatch reuse, and sample batching.
- `src/falcor2/render/reconstruction_filter.cpp`, `src/falcor2_ext/render/reconstruction_filter.cpp`,
  `slang/falcor2/render/reconstruction_filter.slang`, and `falcor2/extensions/reconstructionfiltermarshall.py` - filters and binding.

- `falcor2/rendernodes/accumulator_node.py`, `falcor2/rendernodes/reweighting_accumulator_node.py`,
  `falcor2/rendernodes/tonemapper_node.py`, `falcor2/rendernodes/optix_denoiser_node.py`, and
  `falcor2/rendernodes/dlss_ray_recon_node.py` - image histories and downstream consumers.
- `falcor2/utils/jitter.py`, `falcor2/editor/editor.py`, `examples/pathtracer/headless.py`,
  `examples/pathtracer/dlss.py`, and `examples/pathtracer/dlss_rr.py` - jitter and presentation compositions.
- `tests/python/render/test_scene.py`, `tests/python/pathtracer/test_pathtracer.py`, and
  `tests/python/pathtracer/test_pipeline.py` - requirement/dispatch reuse, batching, and resets.
- `tests/python/render/test_pixel_filter.py`, `tests/python/pathtracer/test_accumulator_node.py`,
  `tests/python/pathtracer/test_tone_mapper.py`, and `tests/python/rendergraph/test_container_spec.py` - numerical and storage contracts.

## Related design documents

- [Scene model](scene-model.md) - synchronized scene state and capability selection.
- [Geometry and ray tracing](geometry-and-ray-tracing.md) - traversal and acceleration structures.
- [Lighting and sampling](lighting-and-sampling.md) - sampler ownership and transport integration.
- [Named material outputs](material-aovs.md) - declarations and indexed guide delivery.
- [Python and native integration](python-native-integration.md) - ownership across language boundaries.
