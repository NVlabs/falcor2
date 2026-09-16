# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from enum import IntEnum
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from weakref import ref

import pytest
import falcor2 as f2
import falcor2.testing.helpers as helpers
import slangpy as spy

from falcor2.editor.graph_settings import GraphSettingsPanel
from falcor2.editor.editor import Editor, _get_active_editor, _set_active_editor
from falcor2.reflection import reflected, reflected_property


class _Mode(IntEnum):
    off = 0
    enabled = 1


@reflected
class _SampleSettings:
    enabled = reflected_property(True)
    mode = reflected_property(_Mode.off, ui_group="Rendering")
    samples = reflected_property(2, value_range=(1, 8), ui_group="Rendering")
    scale = reflected_property(0.5, value_range=(0.0, 1.0))
    name = reflected_property("sample")


@reflected
class _ObjectSettings:
    def __init__(self, value: f2.LightSampler | None) -> None:
        self._value = value

    @reflected_property(
        object_factories=(None, f2.UniformLightSampler, f2.PowerLightSampler),
    )
    def value(self) -> f2.LightSampler | None:
        return self._value

    @value.setter
    def value(self, value: f2.LightSampler | None) -> None:
        self._value = value


@reflected
class _TestRenderNode(f2.RenderNode):
    enabled = reflected_property(True)

    def __init__(self, child: _TestRenderNode | None = None) -> None:
        super().__init__()
        self.child = child

    def _exec(self, value: Any) -> Any:
        if self.child is not None:
            self.child(value)
        return value


@reflected
class _RootSettings:
    enabled = reflected_property(True)

    def __init__(self, children: dict[str, Any]) -> None:
        self._children = children
        self.changed_sources: list[Any] = []

    def graph_settings_children(self) -> dict[str, Any]:
        return self._children

    def on_graph_settings_changed(self, source: Any) -> None:
        self.changed_sources.append(source)


class _FakePropertyEditor:
    def __init__(self, changed: Any | None = None) -> None:
        self.calls: list[tuple[str, Any]] = []
        self.changed = changed

    def render(self, name: str, settings: Any) -> bool:
        self.calls.append((name, settings))
        return settings is self.changed


def test_graph_panel_renders_current_node() -> None:
    scene_editor = SimpleNamespace(graph_ui_callback=None)
    property_editor = _FakePropertyEditor()
    first = _SampleSettings()
    second = _SampleSettings()
    current: list[Any | None] = [first]
    GraphSettingsPanel(scene_editor, lambda: current[0], property_editor)

    scene_editor.graph_ui_callback()
    current[0] = second
    scene_editor.graph_ui_callback()
    current[0] = None
    scene_editor.graph_ui_callback()

    assert property_editor.calls == [("_SampleSettings", first), ("_SampleSettings", second)]


def test_graph_panel_ignores_non_reflected_nodes() -> None:
    scene_editor = SimpleNamespace(graph_ui_callback=None)
    property_editor = _FakePropertyEditor()
    GraphSettingsPanel(scene_editor, object, property_editor)

    scene_editor.graph_ui_callback()
    assert property_editor.calls == []


def test_graph_panel_renders_explicit_children_and_notifies_root() -> None:
    scene_editor = SimpleNamespace(graph_ui_callback=None)
    child = _SampleSettings()
    second_child = _SampleSettings()
    ignored_child = object()
    root = _RootSettings(
        {
            "primary": child,
            "secondary": second_child,
            "ignored": ignored_child,
        }
    )
    property_editor = _FakePropertyEditor(changed=child)
    GraphSettingsPanel(scene_editor, lambda: root, property_editor)

    scene_editor.graph_ui_callback()

    assert property_editor.calls == [
        ("_RootSettings", root),
        ("_SampleSettings (primary)", child),
        ("_SampleSettings (secondary)", second_child),
    ]
    assert root.changed_sources == [child]


def test_graph_panel_identifies_changed_root() -> None:
    scene_editor = SimpleNamespace(graph_ui_callback=None)
    root = _RootSettings({})
    property_editor = _FakePropertyEditor(changed=root)
    GraphSettingsPanel(scene_editor, lambda: root, property_editor)

    scene_editor.graph_ui_callback()

    assert root.changed_sources == [root]


def test_editor_render_node_reference_is_weak() -> None:
    editor = Editor.__new__(Editor)
    editor._render_node = None
    node = _TestRenderNode()
    node_ref = ref(node)

    editor._set_render_node(node)
    assert editor._get_render_node() is node

    del node
    assert node_ref() is None
    assert editor._get_render_node() is None


def test_active_editor_tracks_only_the_root_render_node() -> None:
    scene_editor = SimpleNamespace(graph_ui_callback=None)
    property_editor = _FakePropertyEditor()
    editor = Editor.__new__(Editor)
    editor._render_node = None
    editor._graph_settings_panel = GraphSettingsPanel(
        scene_editor,
        editor._get_render_node,
        property_editor,
    )
    editor._closed = False
    editor._stop_mcp_bridge = lambda: None
    editor._end_profile_frame = lambda: None
    editor._wait_for_device = lambda: None
    editor.window = SimpleNamespace(close=lambda: None)
    child = _TestRenderNode()
    root = _TestRenderNode(child)
    payload = object()

    _set_active_editor(editor)
    try:
        assert _get_active_editor() is editor
        assert root(payload) is payload
    finally:
        editor.close()
    scene_editor.graph_ui_callback()

    assert editor._closed is True
    assert _get_active_editor() is None
    assert property_editor.calls == [("_TestRenderNode", root)]

    property_editor.calls.clear()
    later_root = _TestRenderNode()
    later_root(payload)
    scene_editor.graph_ui_callback()
    assert property_editor.calls == [("_TestRenderNode", root)]


def test_native_property_editor_and_graph_callback_are_bound() -> None:
    property_editor = f2.ui.PropertyEditor()
    context = f2.ui.PropertyEditorContext()
    scene_editor = f2.ui.SceneEditor()
    callback_calls: list[bool] = []

    scene_editor.graph_ui_callback = lambda: callback_calls.append(True)

    assert property_editor is not None
    assert property_editor.context.show_advanced is False
    assert property_editor.context.show_read_only is True
    property_editor.context.show_advanced = True
    property_editor.context.show_read_only = False
    assert property_editor.context.show_advanced is True
    assert property_editor.context.show_read_only is False
    assert context.show_advanced is False
    assert context.show_read_only is True
    assert callable(f2.ui.properties_editor)
    assert scene_editor.graph_ui_callback is not None


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_native_property_editor_renders_reflected_settings(
    device_type: spy.DeviceType,
    device: spy.Device,
    empty_scene: f2.Scene,
    workspace_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(workspace_tmp_path)
    ui_context = spy.ui.Context(device)
    scene_editor = f2.ui.SceneEditor()
    scene_editor.scene = empty_scene
    settings = _SampleSettings()
    panel = GraphSettingsPanel(scene_editor, lambda: settings)
    target = device.create_texture(
        width=320,
        height=200,
        format=spy.Format.rgba8_unorm,
        usage=spy.TextureUsage.render_target | spy.TextureUsage.unordered_access,
    )
    command_encoder = device.create_command_encoder()

    ui_context.begin_frame(target.width, target.height)
    scene_editor.editor_ui()
    ui_context.end_frame(target, command_encoder)
    device.submit_command_buffer(command_encoder.finish())
    device.wait()


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_direct_properties_editor_renders_multiple_objects(
    device_type: spy.DeviceType,
    device: spy.Device,
    empty_scene: f2.Scene,
    workspace_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(workspace_tmp_path)
    ui_context = spy.ui.Context(device)
    scene_editor = f2.ui.SceneEditor()
    scene_editor.scene = empty_scene
    context = f2.ui.PropertyEditorContext()
    first = _SampleSettings()
    second = _SampleSettings()
    populated_object = _ObjectSettings(f2.UniformLightSampler())
    null_object = _ObjectSettings(None)
    changes: list[bool] = []

    def render_graph() -> None:
        changes.append(f2.ui.properties_editor(first, context))
        changes.append(f2.ui.properties_editor(second, context))
        changes.append(f2.ui.properties_editor(populated_object, context))
        changes.append(f2.ui.properties_editor(null_object, context))

    scene_editor.graph_ui_callback = render_graph
    target = device.create_texture(
        width=320,
        height=200,
        format=spy.Format.rgba8_unorm,
        usage=spy.TextureUsage.render_target | spy.TextureUsage.unordered_access,
    )
    command_encoder = device.create_command_encoder()

    ui_context.begin_frame(target.width, target.height)
    scene_editor.editor_ui()
    ui_context.end_frame(target, command_encoder)
    device.submit_command_buffer(command_encoder.finish())
    device.wait()

    assert changes == [False, False, False, False]
