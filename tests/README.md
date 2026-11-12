# Running the regular and nightly test tiers

From an activated development environment, run both phases with the CI driver:

```powershell
python tools/ci.py --config Release unit-test
python tools/ci.py --config Release --nightly unit-test
```

The regular tier is the default. Nightly includes regular tests and every deferred
case. Python CI runs four workers, enables slow tests and uses the persistent
shader cache. Release also enables image comparisons. Individual phases remain
available as `unit-test-cpp` and `unit-test-python`.

Debug CI selects only D3D12 on Windows and Vulkan on Linux, including nightly
runs. Release retains the complete backend matrix. This policy is applied by
`tools/ci.py`; direct native/pytest runs keep their usual backend defaults.
For an explicit local selection, use native `-device-type=d3d12` or pytest
`--device-type=d3d12` (substitute `vulkan` as needed). The pytest option also
filters explicit backend parameters and skips hardcoded requests for another
backend before creating a device. `SLANGPY_DEVICE` continues to set the default
Python parameter matrix; an explicit `--device-type` takes precedence.
Existing Vulkan-only regular cases therefore run in Windows Debug nightlies;
their regular coverage remains in Linux Debug and Release.

Regular CI overlaps the native process and the four Python workers. Nightly
runs always execute the phases sequentially. To enable overlap in a direct
regular run of the CI driver:

```powershell
python tools/ci.py --config Release --parallel-test-phases unit-test
```

`CI_NIGHTLY` and `CI_PARALLEL_TEST_PHASES` provide the equivalent CI settings.
CI defaults `CI_PARALLEL_TEST_PHASES` to `true`; use `false` or
`--no-parallel-test-phases` to disable overlap. Nightly takes precedence over an
explicit overlap request. Direct CI-driver runs without that environment setting
default to sequential phases. The combined
command saves phase timings to `reports/test-phases.json`, separate native and
Python logs, both JUnit reports, and `reports/system-telemetry-combined.json`.
The combined sampler includes both process trees. The existing
`reports/system-telemetry.json` process metrics cover pytest; its host and GPU
metrics are system-wide in either mode.

For direct runs, native accepts `-nightly` (also `--nightly`), while pytest accepts
`--nightly` or `--all`. Pytest `--nightly` enables slow tests; `--all` additionally
enables image comparisons. Existing capability skips still apply. Deferred tests
are deselected before scheduling and do not count as successful coverage.

## Assigning a tier

New tests are regular unless explicitly tagged. Keep representative assertions in
regular tests and defer repeated or expensive matrices using source annotations.

Python whole modules use `pytestmark = pytest.mark.nightly`; whole functions use
`@pytest.mark.nightly`. Parameter lists can attach
`marks=pytest.mark.nightly` to individual `pytest.param` values. For intersecting
parameter dimensions, use explicit excluded values:

```python
@pytest.mark.nightly_if(variant=["large"], format=["float16"])
```

All named conditions must match; separate `nightly_if` marks are alternatives.
Keys must name collected parameters, including indirectly parametrized fixtures.
Values must be lists or tuples. Unknown parameters are collection errors.
Newly added values do not match an explicit exclusion unless the annotation is
updated.

For tests that use Vulkan as their regular backend, use the simpler marker:

```python
@pytest.mark.nightly_if_not_vk
```

It reads the collected `device_type` parameter and defers every non-Vulkan
backend until nightly, including D3D12, CUDA, Metal and any future backend.
This is independent of OS or API ordering. `SLANGPY_DEVICE` still controls which
backends are collected; selecting only a non-Vulkan backend requires `--nightly`
to run these marked cases. The marker never adds Vulkan to the collected matrix.

Additional parameter conditions remain conjunctive, for example
`@pytest.mark.nightly_if_not_vk(variant=["large"])`. An indirectly parametrized
device fixture can name its backend parameter with
`@pytest.mark.nightly_if_not_vk(device_param="_bsdf_resources")`. The selected
parameter must contain `DeviceType` values. Deferred cases are removed before
fixtures create devices or other resources. Separate nightly marks still combine
as alternatives, so a whole-case nightly mark also defers Vulkan.

Native uses `TEST_CASE_NIGHTLY`, `TEST_CASE_GPU_NIGHTLY`, and `SUBCASE_NIGHTLY`.
For cases that are expensive specifically in Debug, use `TEST_CASE_DEBUG_NIGHTLY`
or `TEST_CASE_GPU_DEBUG_NIGHTLY`. These require `-nightly` in a compiled Debug
binary and remain regular in Release. Native GPU cases run every supported
graphics API unless `-device-type` selects one. Whole-case tier metadata reserves
the doctest descriptions `falcor2 nightly` and `falcor2 debug nightly`; case and
suite names stay unchanged. Device-specific host-style cases use
`TEST_CASE_DEVICE(name, cuda)` so graphics-only runs exclude them before the body.
Keep at least one regular branch in a mixed-subcase case. If every branch is
deferred, tag the whole case instead.

The initial Debug-only native set contains the five families taking more than
10 seconds each in the Windows Debug serial profile: MaterialX emitted-operation
geometry capabilities, MaterialX uniform emission eligibility, emissive transform
stretch classification, emissive transform cache contributing entities, and the
NGX native smoke test. The Python MDL emission, MaterialX emission, glTF backface
culling, and LSS closest-hit modules are nightly in all configurations.

## Test assets and regression fixtures

Keep reusable asset bundles in the `data` submodule, including scenes with companion layers or textures and
external examples with their licenses and provenance. Small, self-contained regression fixtures may live beside
their tests. Compiler import fixtures and adjacent Python helper tests may also span multiple files when that
structure is part of the behavior under test. Keep executable test shaders and image-reference manifests with
the tests. Generate disposable fixtures in each test's temporary directory.
