# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import pytest

import falcor2 as f2
from falcor2.rendergraph.render_node import _current_render_node_stack


class RecordingNode(f2.RenderNode):
    def __init__(self):
        super().__init__()
        self.calls = []

    def _exec(self, *args: Any, **kwargs: Any) -> tuple[tuple[Any, ...], dict[str, Any]]:
        self.calls.append((args, kwargs))
        return args, kwargs


class StackRecordingNode(f2.RenderNode):
    def __init__(self, child: StackRecordingNode | None = None):
        super().__init__()
        self.child = child
        self.stacks: list[tuple[f2.RenderNode, ...]] = []

    def _exec(self) -> None:
        self.stacks.append(_current_render_node_stack())
        if self.child is not None:
            self.child()


def test_render_node_is_exported():
    """RenderNode is available on the public falcor2 module."""
    node = f2.RenderNode()
    assert isinstance(node, f2.RenderNode)


def test_render_node_base_exec_requires_override():
    """The base RenderNode raises until a subclass implements _exec()."""
    node = f2.RenderNode()

    with pytest.raises(NotImplementedError, match="_exec"):
        node(1, test=True)


def test_render_node_call_forwards_verbatim_arguments():
    """__call__ passes arbitrary positional and keyword arguments through to _exec()."""
    node = RecordingNode()
    payload = object()

    result_args, result_kwargs = node("sample", payload, samples=4, enabled=True)

    assert result_args == ("sample", payload)
    assert result_kwargs == {"samples": 4, "enabled": True}
    assert node.calls == [(("sample", payload), {"samples": 4, "enabled": True})]


def test_render_node_tracks_nested_stack() -> None:
    child = StackRecordingNode()
    root = StackRecordingNode(child)

    root()

    assert root.stacks == [(root,)]
    assert child.stacks == [(root, child)]
    assert _current_render_node_stack() == ()


def test_render_node_restores_stack_after_failure() -> None:
    class FailingNode(f2.RenderNode):
        def _exec(self) -> None:
            raise RuntimeError("expected failure")

    with pytest.raises(RuntimeError, match="expected failure"):
        FailingNode()()

    assert _current_render_node_stack() == ()
