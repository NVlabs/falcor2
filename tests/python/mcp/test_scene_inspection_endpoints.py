# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from falcor2.mcp import offline
from falcor2.mcp.editor_bridge import _inspect_editor_scene


def test_offline_inspect_scene_loads_filters_and_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scene_path = tmp_path / "scene.py"
    scene_path.write_text("", encoding="utf-8")
    scene = object()
    monkeypatch.setattr(offline, "create_device", lambda device_type: object())
    monkeypatch.setattr(
        offline,
        "f2",
        SimpleNamespace(Scene=SimpleNamespace(load=lambda device, path: scene)),
    )
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        offline,
        "scene_to_dict",
        lambda actual_scene, **kwargs: captured.update(scene=actual_scene, kwargs=kwargs)
        or {"nodes": []},
    )
    monkeypatch.setattr(
        offline,
        "write_scene_json",
        lambda actual_scene, path, **kwargs: captured.update(path=path, write_kwargs=kwargs)
        or path,
    )

    result = offline.inspect_scene(
        {"scene_path": "scene.py", "node_name_pattern": "*cam*", "out": "scene.json"},
        {"workspace_root": str(tmp_path)},
    )

    assert result["scene"] == {"nodes": []}
    assert result["output_path"] == str((tmp_path / "scene.json").resolve())
    assert captured["scene"] is scene
    assert captured["kwargs"] == {"node_name_pattern": "*cam*", "case_sensitive": False}


def test_editor_inspect_scene_uses_current_scene(monkeypatch: pytest.MonkeyPatch) -> None:
    scene = object()
    monkeypatch.setattr(
        "falcor2.mcp.editor_bridge.scene_to_dict",
        lambda actual_scene, **kwargs: {"same_scene": actual_scene is scene, **kwargs},
    )

    result = _inspect_editor_scene(
        SimpleNamespace(scene=scene), {"node_name_pattern": "Camera", "case_sensitive": True}
    )

    assert result == {
        "same_scene": True,
        "node_name_pattern": "Camera",
        "case_sensitive": True,
    }


def test_editor_inspect_scene_requires_loaded_scene() -> None:
    with pytest.raises(ValueError, match="does not have a loaded scene"):
        _inspect_editor_scene(SimpleNamespace(scene=None), {})
