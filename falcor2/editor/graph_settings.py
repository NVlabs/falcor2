# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Property UI for the currently executing render graph."""

from __future__ import annotations

from typing import Any, Callable

import falcor2 as f2


class GraphSettingsPanel:
    """Render the current reflected root node and its explicit settings children."""

    def __init__(
        self,
        scene_editor: Any,
        get_render_node: Callable[[], Any | None],
        property_editor: Any = None,
    ) -> None:
        self._get_render_node = get_render_node
        self._property_editor = (
            property_editor if property_editor is not None else f2.ui.PropertyEditor()
        )
        scene_editor.graph_ui_callback = self.render

    def render(self) -> None:
        """Render reflected settings exposed by the active root node."""
        node = self._get_render_node()
        if node is None:
            return

        get_children = getattr(node, "graph_settings_children", None)
        children = get_children() if get_children is not None else {}
        settings_entries = (
            (type(node).__name__, node),
            *(
                (f"{type(settings).__name__} ({role})", settings)
                for role, settings in children.items()
            ),
        )
        on_changed = getattr(node, "on_graph_settings_changed", None)
        for label, settings_object in settings_entries:
            if getattr(settings_object, "_reflected_properties", None) is None:
                continue
            if self._property_editor.render(label, settings_object) and on_changed is not None:
                on_changed(settings_object)
