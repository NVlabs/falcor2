# SPDX-License-Identifier: Apache-2.0
"""
Declarative render-container descriptions for the Python graph layer.

``ContainerSpec`` captures the logical contract of a render container rather than its exact
storage layout. The current convention is to represent render targets with ``dims=(H, W)``
and use ``format`` to describe channel packing. Concrete backends may store that contract
differently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final, TypeAlias

import slangpy as spy

from . import container_torch


class _AutoValue:
    """Sentinel used for spec fields that should be resolved from a fallback contract."""

    def __repr__(self) -> str:
        return "auto"


AUTO: Final = _AutoValue()

ContainerType: TypeAlias = type[Any] | _AutoValue
ContainerFormat: TypeAlias = spy.Format
ContainerDims: TypeAlias = tuple[int, ...] | _AutoValue


@dataclass(frozen=True)
class ContainerSpec:
    """
    Logical contract for a texture-like render container.

    Current conventions:
    - Texture2D: dims=(H, W), format defines channels
    - render-like spy.Tensor: dims=(H, W), format defines dtype
    - render-like torch.Tensor: dims=(H, W), tensor shape is (C, H, W), though memory layout is still (H, W, C)
    """

    container_type: ContainerType = AUTO
    format: ContainerFormat = spy.Format.undefined
    dims: ContainerDims = AUTO

    @classmethod
    def auto(cls) -> "ContainerSpec":
        """Return a fully automatic spec that resolves all fields from a fallback."""
        return cls()

    @staticmethod
    def texture2d(
        format: ContainerFormat = spy.Format.undefined, dims: ContainerDims = AUTO
    ) -> "ContainerSpec":
        """Construct a spec for a 2D texture container."""
        return ContainerSpec(container_type=spy.Texture, format=format, dims=dims)

    @staticmethod
    def tensor(
        format: ContainerFormat = spy.Format.undefined, dims: ContainerDims = AUTO
    ) -> "ContainerSpec":
        """Construct a spec for a render-like ``spy.Tensor`` container."""
        return ContainerSpec(container_type=spy.Tensor, format=format, dims=dims)

    @staticmethod
    def torch(
        format: ContainerFormat = spy.Format.undefined, dims: ContainerDims = AUTO
    ) -> "ContainerSpec":
        """Construct a spec for a torch-backed render container."""
        return ContainerSpec(
            container_type=container_torch.tensor_type(),
            format=format,
            dims=dims,
        )

    def resolved(self, fallback: "ContainerSpec | None" = None) -> "ContainerSpec":
        """Fill any automatic fields in this spec from ``fallback``."""
        fallback = fallback or ContainerSpec.auto()
        container_type = (
            fallback.container_type if self.container_type is AUTO else self.container_type
        )
        format_value = fallback.format if self.format is spy.Format.undefined else self.format
        dims_value = fallback.dims if self.dims is AUTO else self.dims
        return ContainerSpec(container_type=container_type, format=format_value, dims=dims_value)

    @classmethod
    def from_container(cls, container: Any) -> "ContainerSpec":
        """Build a spec from a concrete container by delegating to ``Container.format_and_dims()``."""
        from .container import Container

        format_value, dims = Container.format_and_dims(container)

        if isinstance(container, spy.Tensor):
            return cls.tensor(format=format_value, dims=dims)

        if isinstance(container, spy.Texture):
            if container.type == spy.TextureType.texture_2d:
                return cls.texture2d(format=format_value, dims=dims)
            raise ValueError(f"Unsupported texture type: {container.type!r}")

        if container_torch.is_torch_tensor(container):
            return cls.torch(format=format_value, dims=dims)

        raise TypeError(f"Unsupported container type: {type(container)!r}")

    def with_format(self, format: ContainerFormat) -> "ContainerSpec":
        """Return a copy of this spec with the given format."""
        return ContainerSpec(container_type=self.container_type, format=format, dims=self.dims)

    def with_dims(self, dims: ContainerDims) -> "ContainerSpec":
        """Return a copy of this spec with the given dims."""
        return ContainerSpec(container_type=self.container_type, format=self.format, dims=dims)
