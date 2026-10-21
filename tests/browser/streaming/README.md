# Streaming tests

These are standalone tests, run from the repository root. See
[Streaming](../../../docs/streaming.md) for server and codec setup. Live playback
and soak tests need a streaming-enabled Release build, OpenH264, and a GPU.

Use the development Python environment, Node.js, and an installed Edge or Chrome:

```powershell
npm.cmd install --prefix build/streaming-deps/browser-tools playwright-core@1.58.2
$env:NODE_PATH = Join-Path (Get-Location) 'build/streaming-deps/browser-tools/node_modules'
$env:STREAMING_PYTHON = (Get-Command python).Source
node --test tests/browser/streaming/test_client.cjs
node --test tests/browser/streaming/test_server.cjs
node tests/browser/streaming/test_input.cjs msedge
node tests/browser/streaming/test_streaming.cjs msedge d3d12
node tests/browser/streaming/test_streaming.cjs msedge d3d12 camera
node tests/browser/streaming/test_editor.cjs msedge d3d12
node tests/browser/streaming/test_editor.cjs msedge d3d12 window
node tests/browser/streaming/test_editor.cjs msedge d3d12 headless usd_stage
```

Use `chrome` instead of `msedge`, and `vulkan` or `cuda` instead of `d3d12`.
Client and server-harness unit tests require only Node.js. Input tests use a real browser data
channel without a GPU; composition events are simulated, so native IME behavior
still needs platform validation. Live tests check decoded color/motion, input,
auto-connect, optional statistics, viewer exclusivity, reconnect, and shutdown.
Set `STREAMING_TEST_TOKEN` to also check rejection of missing/wrong tokens and
successful authenticated playback in the standalone server tests, which stop their
server after 40 seconds. The camera demo checks motion, focus-loss release, and
fresh state on reconnect.
The editor variant runs the normal path-tracer application's render loop at 960x540,
checks complete UI playback, F5/F12 shortcuts, camera control, preserved application
state on reconnect and continued playback after Escape. Python editor tests cover path-tracing
accumulation and reset after remote camera input. The optional `window` argument also
opens a native editor window. A fourth argument of `usd_stage` exercises the same
streaming CLI with an application that creates its scene in memory. The harness
passes `--stream` explicitly; `--headless` alone does not start a server. Editor
readiness is detected through `/health` on the selected port. The harness owns
the timeout and verifies Escape leaves the editor running; editor examples need no
duration or JSON-readiness arguments.
Screenshots are written under `build/streaming-deps/`.

The Python and native regression suites remain in their usual locations:

```powershell
python -m pytest tests/python/streaming -n 4 --shader-cache
.\build\windows-msvc\Release\falcor2_tests.exe -ts=webrtc
```

Unavailable backends and codecs may skip tests; check the skip summary.

## Soak testing

The optional soak harness checks sustained 720p30 pattern playback and resource
growth across reconnects. Install its extra dependency and run:

```powershell
python -m pip install psutil==7.2.2
node tests/browser/streaming/soak.cjs msedge output/streaming-soak d3d12 600 20
```

The last two arguments are playback seconds and reconnect count (defaults: 600
and 20). Reports and samples go into the supplied output directory. The sampler
in `tests/python/streaming/process_sample.py` records CPU/process resources and
GPU memory where driver accounting is available. Set `STREAMING_SOAK_URL` to
target an existing pattern server; remote process resources are not sampled.
The harness stops its local server on completion and leaves remote servers running.
