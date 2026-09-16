# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Execution tracking for Python-side render graph nodes."""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any


_render_node_stack: ContextVar[tuple["RenderNode", ...]] = ContextVar(
    "falcor2_render_node_stack",
    default=(),
)


def _current_render_node_stack() -> tuple["RenderNode", ...]:
    """Return the node stack for the current execution context."""
    return _render_node_stack.get()


class RenderNode:
    """
    Base class for Python render nodes.

    Nodes follow a simple callable protocol: subclasses implement ``_exec()`` and instances
    are invoked through ``__call__``. ``forward()`` tracks nested execution before delegating
    to the implementation.
    """

    def _exec(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the node. Subclasses must override this."""
        raise NotImplementedError("Subclasses of RenderNode must implement the _exec method")

    def graph_settings_children(self) -> dict[str, Any]:
        """Return role-labelled child objects to include in the graph settings UI."""
        return {}

    def on_graph_settings_changed(self, source: Any) -> None:
        """Handle a graph settings edit made on ``source``."""

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Track this node's execution and dispatch to ``_exec()``."""
        stack = _render_node_stack.get()
        is_root = not stack
        stack_token = _render_node_stack.set((*stack, self))
        try:
            if is_root:
                # Import lazily to avoid a module cycle while falcor2 initializes.
                from falcor2.editor.editor import _get_active_editor

                editor = _get_active_editor()
                if editor is not None:
                    editor._set_render_node(self)
            return self._exec(*args, **kwargs)
        finally:
            _render_node_stack.reset(stack_token)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Dispatch to the tracked ``forward()`` entry point."""
        return self.forward(*args, **kwargs)
