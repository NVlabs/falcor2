# Optional USD FBX plugin

This overlay builds selected Adobe file-format plugins and shared utilities
against Falcor2's existing private OpenUSD 26.08 monolith. It leaves the `usd` overlay and triplets
unchanged and does not enable USD imaging or Hio.

Source is pinned to `4997b5f3b6f0e4797a91c9a971becbae9704c54f` of the
[integration fork](https://github.com/skallweitNV/USD-Fileformat-plugins/tree/dev/falcor2-integration),
based on Adobe 2026.07. Importer fixes, fixed-rate resampling, bounded adaptive animation refinement,
independent error-based channel reduction, and the OpenImageIO backend live in
that fork. This overlay applies no source patches. Its minimal
feature-selected CMake wrapper supplies private library names and SDK linkage.

## Port features

The port has no default features. Its core package contains shared utilities and
requires no FBX SDK. The `fbx` feature adds the FBX plugin and LibXml2; SDK
acquisition, linkage and notices apply only when that feature is selected.
Core-only packages contain no FBX plugin library, registry or FBX test fixture.

Falcor2's `usd-fbx` feature explicitly depends on
`usd-fileformat-plugins[fbx]` with `default-features: false`. The selection is:

```text
FALCOR_ENABLE_USD_FBX -> usd-fbx -> usd-fileformat-plugins[fbx]
```

Additional formats can get their own port features and Falcor2 options later.
Only `fbx` is currently implemented. SDK packages are selected for Windows x64
(VS2019 binaries, linked with VS2022), Linux x64 (GCC), and macOS x64/arm64 (Clang universal libraries).
Linux ARM64 and Windows ARM64 are not enabled by this overlay.

## Build

The plugin is **disabled by default**. Normal Falcor2 builds do not need the FBX
SDK or build this plugin's additional dependencies. When enabled,
vcpkg downloads the platform's Autodesk FBX SDK **2020.3.11**, verifies its SHA512, and
extracts it under the port's build directory. The installer is never executed;
no system-wide SDK installation or administrator privileges are needed. The
download is cached by vcpkg for subsequent builds.

Windows uses Autodesk's VS2019 binary package of the same SDK version, so it also
links with VS2022 17.9. The VS2022 SDK package requires newer STL helper symbols.
Falcor2 and the plugin are still compiled with the configured VS2022 toolchain.

From an initialized x64 MSVC environment with the development Python active:

```powershell
cmake --preset windows-msvc -DFALCOR_ENABLE_USD_FBX=ON
cmake --build --preset windows-msvc-release
```

For Linux, use `linux-gcc` / `linux-gcc-release`. For macOS, use
`macos-arm64-clang` / `macos-arm64-clang-release`, or the corresponding
`macos-x64-clang` presets. Pass the same `-DFALCOR_ENABLE_USD_FBX=ON` option.
Linux extraction uses the active development Python; macOS uses the system BSD
tar to unpack the package and its payload. Neither runs the vendor installer.

`FALCOR_ENABLE_USD_FBX` selects the optional `usd-fbx` vcpkg manifest feature
before dependency installation. It preserves other selected features. Set the
option to `OFF` to disable it on a subsequent configure. If you separately set
`VCPKG_MANIFEST_FEATURES=usd-fbx`, remove that explicit selection as well.
Unsupported platforms fail during configuration.

To use an existing unmodified SDK of the same version and platform variant instead,
set `FBXSDK_ROOT` in the environment and preserve it through vcpkg's environment
isolation before configuring:

```powershell
$env:FBXSDK_ROOT = "C:/path/to/FBX SDK/2020.3.11"
$env:VCPKG_KEEP_ENV_VARS = (@($env:VCPKG_KEEP_ENV_VARS, "FBXSDK_ROOT") | Where-Object { $_ }) -join ";"
```

On Linux/macOS:

```sh
export FBXSDK_ROOT="/path/to/fbxsdk/2020.3.11"
export VCPKG_KEEP_ENV_VARS="${VCPKG_KEEP_ENV_VARS:+${VCPKG_KEEP_ENV_VARS};}FBXSDK_ROOT"
```

Both acquisition paths validate the SDK's Debug and Release library hashes.
An invalid override fails rather than falling back to a download. Unset
`FBXSDK_ROOT` to restore automatic acquisition. The override selects the same
pinned SDK and does not force a rebuild when a binary package is already cached.

The SDK is statically linked, so it is required to build the plugin from source,
not to use an already built package. Autodesk's SDK license applies to its use;
SDK and Adobe license notices are included in the plugin package.

Run CMake and tests outside the sandbox as required by the repository instructions.
OpenImageIO and LibXml2 come from the existing vcpkg baseline. Plugin libraries
have private `falcor2_adobe_` names; Debug and Release have separate registries.
Automatic deployment and wheel inclusion are outside this integration.

## Load an FBX

Register the installed plugin explicitly before opening an FBX. Set `prefix` to
your build's installed vcpkg triplet directory (the example uses Windows):

```python
import ctypes
import os
from pathlib import Path
import falcor2 as f2
from falcor2.pxr import Plug

prefix = Path("build/windows-msvc/vcpkg_installed/x64-windows-static-md-fix").resolve()
if os.name == "nt":
    dll_directory = os.add_dll_directory(str(prefix / "bin"))  # Keep this handle alive.
    plugin_dll = ctypes.WinDLL(str(prefix / "bin/falcor2_adobe_usdFbx.dll"))
registry = prefix / "share/usd-fileformat-plugins/plugin/usdFbx/resources/plugInfo.json"
Plug.Registry().RegisterPlugins(str(registry))
scene = f2.UsdImporter().load_scene(Path("scene.fbx").resolve())
```

On Windows, preloading supplies Python's DLL dependency search flags before USD
loads the plugin; keep both handles alive. On Linux/macOS, the registry points to
the library under `lib`, with relative runtime paths to its dependencies.
Use `falcor2.pxr`, not an unrelated `pxr` runtime. For Debug builds, append `debug`
to `prefix` to select the matching library and registry.
Generic scene-file dispatch is separate work. Exported ordinary USD assets do not
require this plugin or the SDK to load.

## Animation and image behavior

The fork accepts `animationMode=adaptive` (default), `animationMode=resample`, or
`animationMode=keyframes`:

```python
from falcor2.pxr import Sdf, Usd

layer = Sdf.Layer.FindOrOpen(str(Path("scene.fbx").resolve()), {"animationMode": "keyframes"})
stage = Usd.Stage.Open(layer)
stage.Export("scene.usdc")
```

Adaptive mode evaluates node transforms at source frames and refines between
frames to approximate cubic curves, pivot motion and Euler revolutions with USD
samples. Keyframe mode evaluates original key times only; it is compact but can
lose curved motion, pivot paths, held transitions and full revolutions. All modes
retain source-frame time codes and playback duration. These modes affect node
transforms, including camera transforms; skeletal curve sampling is unchanged.
Unknown values, including `splines`, fail. Arguments apply when FBX is imported,
not to already exported USD. Dynamic payloads also support `animationMode` metadata.

Resample mode bakes transforms at a fixed rate, then reduces translation, rotation
and scale independently against those baked samples. `animationSampleRate=0`
(default) uses the source scene's FPS, including custom rates; a positive finite
value selects an explicit rate in Hz without changing playback FPS or duration.
Track endpoints are preserved even when off-grid. Resampling uses the full
reduction tolerance budget and performs no adaptive refinement. Held transitions
are represented at frame resolution, so subframe motion can differ. This mode
also accepts the tolerance arguments below and `animationSampleRate` as dynamic
payload metadata. The scene-conversion scripts explicitly select resample mode.

Adaptive mode accepts these finite file-format arguments (also available
as string metadata on dynamic payloads):

| Argument | Default | Meaning |
| --- | --- | --- |
| `animationPositionTolerance` | `0.0001` | Translation error in meters, accounting for parent scale and float precision |
| `animationRotationTolerance` | `0.01` | Local rotation error in degrees; must be less than 180 |
| `animationScaleTolerance` | `0.00001` | Relative local scale error, with a unit floor |
| `animationMaxSampleRate` | `0` | Optional maximum frequency of additional adaptive subdivisions, in Hz; zero disables the rate cap |

Tolerances must be positive; the rate must be non-negative. Refinement and channel
reduction each use half the error budget. Translation,
rotation and scale are reduced independently; nearly constant channels collapse
to their evaluated value and retain clip endpoints. Source keys and one-sided
samples around held transitions are not discarded by the frequency limit; the
subsequent reduction only removes samples within its error budget. The importer
warns when a frequency or recursion limit prevents meeting the refinement tolerance. These
are sampled error controls, not a bound on accumulated world-space error through
a hierarchy. Translation uses a conservative bound on parent transform stretch
over evaluated times; this avoids oversampling rigs scaled down relative to the
scene's units. Rotation and scale tolerances are local. Keyframe mode retains
its existing behavior.

Rotation checks cover both USD's orientation-matrix evaluation and unit-quaternion
slerp used by direct-channel consumers. USD's close-angle interpolation can yield
non-unit quaternions; these paths otherwise differ near zero rotation.

Rotations are normalized double quaternions, and scales retain float precision. Dense
animated joint hierarchies can be expensive to convert; arbitrary affine/sheared
FBX inheritance has not been established by the focused checks.

Image helpers use OpenImageIO and `ArResolver` for package-relative assets,
preserve stored pixel values and unassociated alpha, and support DDS. They do not
implicitly convert sRGB values to linear. Empty texture slots are skipped and
same-named images retain distinct data. SBSAR/Hio image providers are unsupported.

## Validation

After building Release, run the plugin integration tests:

```powershell
<python> -m pytest tests/python/importers/test_usd_fbx_plugin.py -v
```

Tests automatically use the active development build's CMake configuration and
vcpkg installation. They skip when the FBX feature is disabled or when running
outside a supported development build. With the feature enabled, missing files
and load failures are errors. Tests import Adobe's cube fixture and round-trip an
embedded texture through the plugin and Falcor2 importer, without converter dependencies.

The fork contains independent USD-only importer regression tests in
`fbx/tests/test_regressions.py` and an `imageIoTests` CTest target in `utils/tests`.
Image-helper coverage lives there: package-relative PNG, DDS, orientation, alpha,
HDR EXR, memory encoding, and unsupported, missing and corrupt inputs. Full
converter and scene-catalog validation belongs to the separate conversion integration.
