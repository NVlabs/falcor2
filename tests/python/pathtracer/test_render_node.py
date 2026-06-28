# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import pytest

import falcor2 as f2


class RecordingNode(f2.RenderNode):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, *args: Any, **kwargs: Any):
        self.calls.append((args, kwargs))
        return args, kwargs


def test_render_node_is_exported():
    """RenderNode is available on the public falcor2 module."""
    node = f2.RenderNode()
    assert isinstance(node, f2.RenderNode)


def test_render_node_base_forward_requires_override():
    """The base RenderNode raises until a subclass implements forward()."""
    node = f2.RenderNode()

    with pytest.raises(NotImplementedError, match="forward"):
        node(1, test=True)


def test_render_node_call_forwards_verbatim_arguments():
    """__call__ passes arbitrary positional and keyword arguments through to forward()."""
    node = RecordingNode()
    payload = object()

    result_args, result_kwargs = node("sample", payload, samples=4, enabled=True)

    assert result_args == ("sample", payload)
    assert result_kwargs == {"samples": 4, "enabled": True}
    assert node.calls == [(("sample", payload), {"samples": 4, "enabled": True})]
