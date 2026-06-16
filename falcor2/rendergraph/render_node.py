# SPDX-License-Identifier: Apache-2.0
"""Minimal base class for Python-side render graph nodes."""


from typing import Any


class RenderNode:
    """
    Base class for Python render nodes.

    Nodes follow a simple callable protocol: subclasses implement ``forward()`` and instances
    are invoked through ``__call__``.
    """

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the node. Subclasses must override this."""
        raise NotImplementedError("Subclasses of RenderNode must implement the forward method")

    def __call__(self, *args: Any, **kwargs: Any):
        """Dispatch directly to ``forward()``."""
        return self.forward(*args, **kwargs)
