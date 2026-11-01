---
status: current
audience: Falcor2 contributors and coding agents
last_verified: 2026-10-09
verified_against: "9c2e77d60f9a07fd62394c5cf7b22dc0623d0174"
---

(falcor2-design-render-pipeline)=
# Render pipeline

Falcor2's render pipeline is a Python composition of callable render nodes around a scene-specialized Slang path
tracer. `PathTracerPipeline` updates the scene, chooses a camera, renders one or more raw linear-radiance samples,
selects an accumulation policy, optionally denoises the result, and tone maps it for display. Renderer-owned pixel
filters choose primary-ray positions independently of temporal reconstruction. `ContainerSpec` and `Container`
carry images across SGL textures, SlangPy tensors, and supported CUDA Torch tensors. The returned container can serve
an interactive editor, a headless image writer, or another Python processing pipeline.

## Purpose and scope

This page explains how Falcor2's Python render-node layer, reference path tracer, accumulation modes, denoising
boundaries, tone mapper, and presentation paths compose into a frame. It covers pixel-filter ownership and sampling,
data contracts, ordering, and temporal reset rules.

The scene's update internals belong in [Scene model](scene-model.md). Ray traversal and acceleration structures
belong in [Geometry and ray tracing](geometry-and-ray-tracing.md); material evaluation belongs in
[Material system](material-system.md); and light selection belongs in
[Lighting and sampling](lighting-and-sampling.md). The transport estimator and denoiser algorithms are also outside
this page; only their integration contracts are described here.

## Architecture

### Node and container layer

`RenderNode` is a small callable protocol, not a static directed graph. A subclass implements `_exec()`, while
`forward()` records nested node execution and calls that implementation. When an editor is active, the outermost
node becomes the root shown by the graph-settings panel. A composite node can expose named child settings and react
to their edits; `PathTracerPipeline` uses this mechanism to expose its path tracer, accumulators, denoisers, and tone
mapper.

`ContainerSpec` describes container type, pixel format, and logical dimensions, resolving automatic fields against
a fallback such as the camera's `(height, width)`. `Container` allocates and reuses storage. Textures and SlangPy
tensors use render layout; Torch images have a public channel-first shape adapted at shader-call boundaries.
Nodes own their outputs and history while preserving the requested container kind where their contracts allow it.

`OutputPrelude` reflects fixed shader guides from `IWriteGuide`, including formats, clear values, and write or
increment behavior. It generates code specialized to concrete targets; disabled outputs become no-ops.
Material-defined outputs use the indexed delivery described in [Named material outputs](material-aovs.md).

### Pipeline stages

`ReferencePathTracerNode` owns the renderer module, scene-aware shader linker, light sampler, output specifications,
and previous-camera state used by motion-vector guides. Before dispatch it updates the light sampler and obtains a
cached module linked against the scene's current modules and type conformances. When the scene requirements or
specialization inputs change, native `SceneRayTracingSetup` supplies the required miss shaders, hit groups,
shader-table ordering, and pipeline flags for a rebuilt dispatch. One invocation then emits one stochastic,
scene-linear RGBA sample per pixel plus requested guides.

`PathTracerPipeline` explicitly orders its owned nodes. The normal branch chooses no accumulation, a conventional
running mean (the default), or a luminance-binned reweighting estimator. OptiX can denoise the accumulated linear
HDR result using diffuse-albedo and normal guides. The alternative DLSS Ray Reconstruction branch chooses its own
internal render resolution, requires a larger guide set, applies subpixel jitter, and produces the final-size image
without using either normal accumulator or OptiX. `TonemapperNode` follows either branch and preserves the selected
container while applying exposure and a display curve; auto exposure and the ACES film curve are the defaults.

### Pixel filters and camera jitter

`ReferencePathTracerNode.pixel_filter` is a plain Python field of type `ReconstructionFilter`, independent of
Scene ownership. Python selects an implementation by assigning a native Box or Gaussian object; the field is not
reflected as a settings-editor selector.
The reference path tracer and native `scene_test` path-traced mode default to `GaussianReconstructionFilter(sigma=0.5,
radius=1.5)`, matching PBRT's filter shape and parameters.

The Slang module `falcor2.render` owns `IReconstructionFilter`, `ReconstructionFilterSample`,
`BoxReconstructionFilter`, and `GaussianReconstructionFilter`. The interface exposes instance methods;
each concrete Slang struct holds its own parameters.

Python passes the native object directly to the shader's `IReconstructionFilter pixel_filter` argument alongside
the camera. `ReconstructionFilterMarshall` follows the Material marshalling pattern: it resolves `slang_type_name`
to the concrete shader type and uses SlangPy's `WriteToCursorMarshall`. A registered native cursor writer delegates
the upload to the filter's `write_to_cursor()`. The filter is one uniform value shared across the dispatch.

The native and Python marshalling signatures identify the concrete shader type, excluding parameter values,
generation, and object identity. SlangPy specializes calls for Box or Gaussian; parameter edits and replacement
with another object of the same type reuse that specialization while uploading current values.

Only native `scene_test` uses a filter link-time type. Its shader declares
`extern struct ReconstructionFilter : IReconstructionFilter = BoxReconstructionFilter`. Its existing generated
constants module exports the selected concrete type using the native object's `slang_type_name()`. The filter is
an ordinary field of `RenderParams`, uploaded through `g_params.pixel_filter`, with Box as the shader fallback and
Gaussian selected at initialization. Native pipeline caching does not track filter-type changes; switching the
type after compilation requires invalidating that cache.

Gaussian's native implementation derives normalization and sampling constants from `sigma` and `radius`
at binding time and uploads them once per dispatch as fields of the Gaussian instance. Box uploads only its width
and calculates its inexpensive reciprocal-area kernel value when evaluated.

All filter distances are in input-render pixels, independent of output resolution. Samples return an offset
and importance weight: the normalized filter value divided by the internal sampling PDF. The renderer weights radiance
and contributes only to the owning pixel, even if the ray position lies outside that pixel or image. There is no
splatting or pixel-boundary clamping.

`eval(offset)` returns the normalized kernel value as a `float` at a center-relative offset,
excluding shared jitter. The value is zero outside support, in units of inverse square input pixels. Evaluation
permits future splatting approaches such as Mitsuba's; selecting neighboring pixels and accumulating their
contributions are separate renderer work. Neither sampling nor evaluation exposes a PDF.

- `BoxReconstructionFilter(width=1)` samples a full-pixel square, with offsets from -0.5 to +0.5 on each axis and unit weights.
  `width=0` samples the center exactly: offset zero and weight one. Its continuous evaluation is zero,
  matching the BSDF delta convention.
- `GaussianReconstructionFilter(sigma=0.5, radius=1.5)` uses square support and the normalized product of
  `max(exp(-x*x/(2*sigma*sigma)) - exp(-radius*radius/(2*sigma*sigma)), 0)` on each axis. This is PBRT/Mitsuba's
  boundary-subtracted shape. Its defaults match PBRT's equal-axis Gaussian; `radius=4*sigma` matches Mitsuba's
  cutoff. Sampling uses two independent truncated, unshifted Gaussian proposals through inverse erf, with a
  `value/pdf` weight. PBRT instead uses a tabulated approximation; Mitsuba evaluates its reconstruction filter
  for splatting. Equivalent kernel shape does not mean identical samples or reconstruction behavior.

The per-call renderer argument `pixel_offset`, default `(0, 0)`, shifts the pixel's sampling footprint in input pixels.
Before the camera UV Y flip, film position is pixel index plus `(0.5, 0.5)`, plus `pixel_offset`, plus the sampled
filter offset. The output pixel address is unchanged. Filters contain no jitter state.

Ordinary reconstruction keeps the filter centered on the output pixel. Importance sampling that fixed kernel and
splatting uniformly distributed film samples target the same normalized convolution when they use the same film
domain and boundary handling, with different sample sharing and noise. Independently jittering the center and
averaging into the same pixel would convolve the filter with the center-jitter distribution, changing the intended
kernel. DLSS instead uses shared jitter to displace its temporal
sampling grid and receives that displacement explicitly.

The renderer consumes two values from its random generator and passes them to `sample()`, even at width zero,
preserving subsequent RNG dimensions. There is no filter-local stratification. Constructors and edits store values;
binding validates finite nonnegative Box width, positive Gaussian parameters, and representable support and density. Stable Gaussian
formulas handle tight cutoffs; guards reject parameters unsuitable for the shader's finite precision.

### Filter edits and temporal ownership

Each filter receives a globally unique generation at construction and after persistent reflected writes, including
same-value assignments. Each pipeline polls its effective filter after DLSS selection and before rendering.
A changed generation requests a history reset; independent consumers therefore detect shared edits separately.
Replacing a filter resets history even with equal parameters; reassigning the same object does not.
Direct C++ field edits require `notify_changed()`; reflected Python/editor writes notify automatically.
The counter invalidates accumulated images; parameters are uploaded on every dispatch regardless of generation.
Polling detects both replacement and in-place edits without a property setter or change subscriptions on the tracer.

The caller passes `pixel_offset` on each renderer invocation. It does not change filter generation or request
a history reset. A caller changing a fixed target position must explicitly reset history. Standalone tracer
users own accumulation and its invalidation.

On entry to DLSS rendering, `PathTracerPipeline` saves the ordinary filter and selects a persistent zero-width Box;
on exit it restores the saved filter. Each DLSS frame passes shared Halton jitter to the renderer and reports it
to the DLSS node as `(-jitter_x, -jitter_y)`. Callers select their ordinary filter outside DLSS mode: the swap occurs
only on mode transitions, so assignments made during DLSS are not intercepted. History resets leave pending
filter transitions intact.
The standalone `examples/pathtracer/dlss.py` demo uses a zero-width Box for guides and all color samples.
Every sample uses the same frame jitter, which is also reported to DLSS, without additional per-ray
pixel-filter jitter. The demo retains cross-frame accumulation until camera
uniforms change. These input-sampling policies do not establish equivalence between DLSS reconstruction and the
ordinary accumulated pixel filter.

## Runtime flow

```text
                         Scene.update()
                               |
                     camera and reset decision
                               |
               ReferencePathTracerNode iteration(s)
                    |                 |
               linear color     optional guides
                    \                 /
                     +-------+-------+
                             |
                 +-----------+-----------+
                 |                       |
       off / running mean /        DLSS-RR temporal path
       weighted resolve                   |
                 |                       |
          optional OptiX                  |
                 +-----------+-----------+
                             |
                   optional tone mapping
                             |
                      returned container
                       /             \
                Editor.present()    headless consumer
```

1. Call `Scene.update()`, then resolve the explicit or active camera; return `None` if none exists. Every sample
   in the call sees the same synchronized scene state. Select the effective filter and poll its generation.
2. In the normal branch, pending configuration changes, a changed scene update generation, or changed camera
   uniforms restart the iteration counter and both accumulation histories. Accumulation `off` renders exactly one
   raw iteration. `normal` renders `spp` iterations and updates the conventional mean after each. `weighted` adds all
   `spp` samples to reweighting history and resolves once after the batch.
3. The tracer resolves and prepares color/guide containers, binds scene and filter data, and dispatches ray tracing.
   The iteration number seeds sampling. OptiX automatically requests missing diffuse-albedo and normal guides.
4. OptiX, when supported by a CUDA or CUDA-interoperable device, runs after normal accumulation and before tone
   mapping. DLSS-RR instead requests its optimal low-resolution color and complete guide set, resets history when its
   render size, target size, or quality key changes, and reconstructs the target-size result.
5. Tone mapping is optional and always follows the selected temporal branch. Its default auto-exposure mode uses one
   stored exposure value from the previous frame, meters the current image, and rate-limits the value prepared for
   the next call using `delta_time`. The first call after a full reset applies current-frame exposure immediately.
6. Return the image, or `(image, guides)` if requested. The editor presents supported containers; headless callers
   can process or save them without an editor.

## Invariants and rationale

Radiance remains linear through tracing, accumulation, and denoising; display mapping follows. Each raw ray sample
enters the chosen accumulator separately. In particular, Zirr's luminance-binned reweighting receives individual
samples rather than a pre-averaged batch. The conventional accumulator averages already importance-weighted radiance
by sample count; it does not divide by a random sum of filter weights. Guides describe the sampled ray and remain
unweighted. Filter importance sampling preserves the per-sample input contract, not a guarantee that firefly
reweighting is unbiased. Negative-lobed filters are outside the supported nonnegative-filter contract.

Container dimensions use `(height, width)` even though camera and presentation APIs commonly speak in width and
height. A concrete container's type, format, dimensions, and device are part of stage compatibility. The standard
accumulator starts new history when its complete input specification changes, and pipeline output-spec changes
request a normal-history reset. Nodes may cache and reuse allocations only while those contracts continue to match.

Normal accumulation reset and full pipeline reset have different scopes in the current code.
Scene or camera changes restart normal color histories and iteration count. `reset()` additionally clears
previous-camera and NaN state, DLSS-RR history, and exposure history. DLSS consumes pending configuration/filter
resets and separately keys history on render dimensions, target dimensions, and quality. It does not use the normal
scene/camera `_needs_reset()` branch; callers retain explicit reset responsibility for other invalidation.

Guide availability is a compile-time and storage contract, not merely a returned dictionary entry. Enabling a guide
changes generated shader code and dispatch specialization, and each consumer requires matching names, formats, and
dimensions. OptiX accepts optional albedo and normal guides but the pipeline supplies both; DLSS-RR requires every
guide in its declared set.

Most stages accept an optional caller-owned command encoder so their GPU work can be recorded in pipeline order;
without one, each stage uses its standalone execution path. Backend boundaries still apply: OptiX can record into a
caller-owned encoder only for a CUDA device, because graphics-backend encoders do not expose the CUDA stream it
needs.

## Source landmarks

- `falcor2/rendergraph/render_node.py`, `falcor2/rendergraph/container_spec.py`, and
  `falcor2/rendergraph/container.py` - callable-node tracking and concrete container contracts.
- `falcor2/rendergraph/output_prelude.py` - reflection, specialization, and binding of optional shader outputs.
- `falcor2/rendernodes/pathtracer_pipeline_node.py` - frame orchestration, accumulation selection, reset decisions,
  denoiser branches, tone mapping, and returned guides.
- `falcor2/rendernodes/reference_pathtracer_node.py` and
  `slang/falcor2/rendernodes/reference_pathtracer.slang` - scene-specialized dispatch, stochastic iterations, and
  guide production.
- `src/falcor2/render/reconstruction_filter.h`, `src/falcor2/render/reconstruction_filter.cpp`, and
  `src/falcor2_ext/render/reconstruction_filter.cpp` - native reflection, validation, generations, and Python bindings.
- `slang/falcor2/render/reconstruction_filter.slang` - shared interface, sample result, concrete filter structs, and sampling maps.
- `falcor2/extensions/reconstructionfiltermarshall.py` - SlangPy interface-argument resolution and native cursor marshalling.
- `src/internal/scene_test/scene_test.cpp` and `src/internal/scene_test/scene_test.slang` - native filter specialization
  through the generated constants module and binding inside `g_params`.
- `falcor2/utils/jitter.py` and `examples/pathtracer/dlss.py` - shared frame jitter and the SR demo's sampling policy.
- `falcor2/rendernodes/accumulator_node.py`, `falcor2/rendernodes/reweighting_accumulator_node.py`, and
  `falcor2/rendernodes/tonemapper_node.py` - temporal color histories, weighted resolve, exposure, and display
  mapping.
- `falcor2/rendernodes/optix_denoiser_node.py`, `falcor2/rendernodes/dlss_ray_recon_node.py`, and
  `src/falcor2/denoisers/optix_denoiser.h` - denoiser integration and native OptiX boundary.
- `src/falcor2/render/ray_tracing_setup.h` and `src/falcor2_ext/render/ray_tracing_setup.cpp` - native ray-type,
  hit-group, and pipeline metadata exposed to Python.
- `falcor2/editor/editor.py` - frame acquisition, root-node settings discovery, container presentation, UI drawing,
  submission, and swapchain presentation.
- `examples/pathtracer/simple.py`, `examples/pathtracer/headless.py`, and `examples/pathtracer/dlss_rr.py` -
  interactive, storage-selectable headless, and DLSS-RR compositions.
- `tests/python/pathtracer/test_pipeline.py`, `tests/python/pathtracer/test_pathtracer.py`,
  `tests/python/pathtracer/test_accumulator_node.py`, `tests/python/pathtracer/test_tone_mapper.py`, and
  `tests/python/rendergraph/test_container_spec.py` - stage ordering, resets, guides, containers, and output behavior.
- `tests/python/render/test_pixel_filter.py` - deterministic Box/Gaussian sampling and evaluation, parameter edits,
  and validation through a shared interface-argument callable.

## Related design documents

- [Falcor2 system overview](overview.md)
- [Scene model](scene-model.md)
- [Scene loading and authoring](scene-loading-and-authoring.md)
- [Falcor2 design documentation](index.md)
- [Geometry and ray tracing](geometry-and-ray-tracing.md)
- [Material system](material-system.md)
- [BSDF contract](bsdf-contract.md)
- [Lighting and sampling](lighting-and-sampling.md)
