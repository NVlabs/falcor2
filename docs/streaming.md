(falcor2-streaming)=
# Streaming

Stream the Falcor2 editor to a browser with keyboard and mouse control over WebRTC.
CUDA, D3D12, and Vulkan are supported. The editor supports multiple viewers of
the same scene and camera, with keyboard and mouse input enabled for everyone.

## Setup

Install Falcor2 with the `streaming` extra, or install `aiohttp>=3.11,<4` in a
[development environment](../README.md). Native builds need
`FALCOR_ENABLE_STREAMING=ON` (the default).

Install OpenH264 separately on the rendering machine:

```sh
python -m falcor2.streaming.openh264 install
python -m falcor2.streaming.openh264 check
```

## Editor applications

```sh
python examples/pathtracer/simple.py --headless --stream --width 1280 --height 720
```

Open the URL printed by the editor. Omit `--headless` to also show the native
window. The `dlss.py`, `dlss_rr.py`, and `usd_stage.py` examples support the same
streaming options; run an example with `--help` for the full list.

- `--stream-port`: HTTP port (default: 8080).
- `--stream-max-viewers`: optional positive simultaneous viewer limit (default: unlimited).
- `--stream-fps`: capture rate (default: 30); does not limit rendering.
- `--stream-bitrate`: encoder target in bits/s (default: 8000000).
- `--stream-token [VALUE]`: require a token; omit the value to generate a private URL.
- `--device-type`: choose `automatic`, `d3d12`, `vulkan`, or `cuda`.

Frame dimensions must be even, between 16x16 and 3840x2160. The browser displays
the stream fitted to the available main view by default, preserving its aspect ratio.
Use **Fit to view** to switch to actual-size display with scrolling. The sidebar
collapses to an icon rail; expand it for control labels, statistics, and codec attribution.
**Full screen** shows only the stream, fitted to the display, where the browser supports
it. Press Esc to exit and restore the previous layout and sizing mode.
These display options do not change the encoded resolution. If the browser rejects
the video configuration, try 1280x720 at 30 FPS and 8 Mbit/s.

All connected browsers control the same editor; local input resumes after the
last viewer disconnects. Inputs are processed on the rendering thread, but
overlapping clicks, drags, or held keys from different viewers can interfere.
There is no exclusive controller or input ownership arbitration.

Rendering is shared, with one encoder and WebRTC connection per viewer. A slow
viewer skips its own frames and adjusts its own bitrate. Encoding cost and
outgoing bandwidth grow with the viewer count; the bitrate setting is per viewer.
Rendering continues without a viewer, and reconnecting preserves the scene and
settings. Click **Connect** to reconnect, or expand **Statistics** for stream metrics.

To enable streaming in your own application:

```python
from falcor2.editor import Editor, EditorConfig, StreamingConfig

editor = Editor.create(device, scene=scene, config=EditorConfig(
    width=1280, height=720,
    window=False,
    streaming=StreamingConfig(port=8080, fps=30),
))
editor.run(lambda scene: pipeline(scene, delta_time=editor.dt))
```

`Editor.run()` closes the editor when it finishes, leaving your device open.
Use `EditorArgumentParser` to add the shared CLI options to an application and
pass the resulting `args.editor_config` to `Editor.create()`.

## Standalone demos

```sh
python -m falcor2.streaming.server --backend d3d12
python -m falcor2.streaming.server --backend d3d12 --demo camera
```

The default demo is an animated pattern. The camera demo supports right-drag to
look, right mouse + WASD/QE to move, middle-drag to pan, and scrolling to dolly.
Standalone options use `--fps`, `--port`, `--token`, etc. without the `--stream-`
prefix; see `--help`.
Standalone demos still accept one viewer at a time; multi-viewer sharing applies
to editor applications.

## Remote access

The server listens on all IPv4 interfaces (`0.0.0.0:8080`) by default. HTTP works
without a certificate, tunnel, or origin list:

```sh
python examples/pathtracer/simple.py --headless --stream \
  --width 1280 --height 720
```

Open `http://<workstation-ip-or-hostname>:8080` from the other machine. Allow
inbound TCP 8080 for the page and signaling, plus the media UDP ports below.
Use `--stream-bind 127.0.0.1` for local-only access or `--stream-bind ::` for an
IPv6 wildcard listener. On machines with multiple network interfaces, use
`--stream-ice-bind <workstation-ip>` to select the reachable media interface.
The standalone demos use `--bind` and `--ice-bind` instead of the `--stream-`
options.

Authentication is off unless a token is configured. HTTP leaves the page, token,
and signaling unencrypted. HTTPS remains optional using `--stream-cert` and
`--stream-key`, or an HTTPS reverse proxy with explicit allowed origins. Proxies
must forward WebSocket upgrades on `/ws`. By default, browser signaling must
come from the same origin as the requested server address. Use `--stream-origin`
to restrict access to explicit origins; HTTPS reverse proxies require this list.

WebRTC media uses UDP ports 50000..50100 by default. Configure STUN/TURN through
the ICE options when needed; no public servers are configured automatically.
TURN supports UDP only. An SSH tunnel for HTTP does not carry the media traffic.

## Codec notices

OpenH264 Video Codec provided by Cisco Systems, Inc.

When distributing applications, preserve this notice, the
[Cisco binary license](../falcor2/streaming/OPENH264_BINARY_LICENSE.txt), and codec
installation/removal controls. Close active encoders before running
`python -m falcor2.streaming.openh264 uninstall`.

See [streaming tests](../tests/browser/streaming/README.md) for playback and input checks.
