# Falcor2 MCP

Falcor2 exposes development and debugging hooks to AI agents through a small MCP stack. The MCP server lives outside this repository and is intended to stay mostly stable: it resolves the relevant workspace, reads endpoint declarations, forwards online calls to a running Falcor2 bridge, and runs offline calls in the target Falcor2 Python environment.

Falcor2-owned endpoint declarations live in `falcor2/mcp/config.json`. Projects that use Falcor2 can add their own project config with extra endpoints and launch profiles.

## Current Endpoints

| Endpoint | Type | Purpose |
| --- | --- | --- |
| `falcor2.bridge.ping` | Online | Checks that the selected running Falcor2 bridge is alive. |
| `falcor2.app.status` | Online | Returns lightweight status for the selected running Falcor2 app or Editor. |
| `falcor2.app.screenshot` | Online | Saves the current Editor output texture to an image file and returns the path and image metadata. |
| `falcor2.offline.python_info` | Offline | Runs in the resolved target Python and reports interpreter/workspace context. |
| `falcor2.offline.render_scene` | Offline | Loads a scene without a running app, renders every camera, and returns PNG paths and image metadata. |
| `falcor2.echo` | Online | Echoes a JSON payload; useful as a simple bridge test. |

The MCP server exposes stable tools such as target resolution, endpoint listing, calling an endpoint, refresh, launch, and stop. The endpoints above are the Falcor2 capabilities those tools discover and invoke.

## Adding Online Endpoints

Use online endpoints for capabilities that require a running Falcor2 instance: scene state, Editor state, current output textures, GPU resources, or app-specific runtime data.

Add an endpoint entry to `falcor2/mcp/config.json` or to a project config:

```json
{
  "name": "falcor2.app.example",
  "title": "Example",
  "description": "Return example live app data.",
  "input_schema": {
    "type": "object",
    "additionalProperties": false
  },
  "output_schema": {
    "type": "object",
    "additionalProperties": true
  }
}
```

Then register a matching handler on the running bridge and pump the bridge from the application update loop:

```python
from pathlib import Path

from falcor2.mcp import Bridge


def example(params: dict[str, object]) -> dict[str, object]:
    return {"ok": True}


bridge = Bridge(workspace_root=Path.cwd())
bridge.register_handler("falcor2.app.example", example)
bridge.start()

try:
    while running:
        update_app()
        bridge.update()
finally:
    bridge.stop()
```

Handlers accept and return JSON objects. The bridge socket thread only reads incoming requests and queues them; handlers run when the application calls `bridge.update()`, on the same thread that called `bridge.update()`. Responses are sent from that update call. The running bridge advertises registered handlers through its heartbeat record, and the MCP server forwards calls to the selected instance.

Editor-specific endpoints should live behind `falcor2/mcp/editor_bridge.py`, which registers Editor status and screenshot handlers onto a normal `Bridge`. The Editor itself still only starts, updates, and stops that bridge.

## Adding Offline Endpoints

Use offline endpoints for work that can run without a live app: asset inspection, static analysis, small render tools, generated previews, or scripts that launch their own Falcor2 work.

Add an endpoint entry with `module` and `function`:

```json
{
  "name": "falcor2.offline.example",
  "title": "Example Offline Call",
  "description": "Run a Python function in the target Falcor2 environment.",
  "input_schema": {
    "type": "object",
    "additionalProperties": true
  },
  "output_schema": {
    "type": "object",
    "additionalProperties": true
  },
  "module": "falcor2.mcp.offline",
  "function": "example"
}
```

Implement the function with this shape:

```python
def example(params: dict[str, object], context: dict[str, object]) -> dict[str, object]:
    return {
        "ok": True,
        "params": dict(params),
        "workspace": context.get("workspace_root"),
    }
```

Offline endpoints run in a subprocess, not inside the MCP server process. The server resolves the target Python in this order: `FALCOR2_MCP_PYTHON`, project config `falcor2.python`, active `VIRTUAL_ENV`, active `CONDA_PREFIX`, workspace `.conda`, workspace `.venv`, workspace `venv`, then the MCP server Python only when a project config was explicitly loaded.

`falcor2.offline.render_scene` requires `scene_path` and accepts `out`, `width`, `height`, `spp`, and `device_type`. Relative scene and output paths resolve against the selected workspace. If `out` is omitted, rendering uses a unique directory below the system temporary directory. Calls that may compile shaders should set the MCP tool's top-level `timeout_s` to 300 seconds; `timeout_s` is not an endpoint parameter. Image-producing endpoint descriptions require the calling agent to include both a clickable link and an inline image in its final response instead of relying on collapsible tool output.
