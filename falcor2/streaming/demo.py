# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Diagnostic pattern and interactive camera demos for browser streaming."""

from typing import Any

import slangpy as spy
import falcor2 as f2
from falcor2.ui import CameraController


class PatternRenderer:
    _entry_point = "make_animated_pattern"

    def __init__(self, device: spy.Device, width: int, height: int) -> None:
        super().__init__()
        self._image = device.create_texture(
            type=spy.TextureType.texture_2d,
            format=spy.Format.rgba32_float,
            width=width,
            height=height,
            usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
        )
        self._kernel = device.create_compute_kernel(
            device.load_program("falcor2/streaming/pattern.slang", [self._entry_point])
        )

    def render(self, frame_id: int, dt: float) -> spy.Texture:
        image = self._image
        if image is None or self._kernel is None:
            raise RuntimeError("Renderer is closed")
        self._kernel.dispatch(
            thread_count=[image.width, image.height, 1],
            vars={
                "pattern": image,
                "width": image.width,
                "height": image.height,
                "frame_index": frame_id,
            },
        )
        return image

    def stats(self) -> dict[str, Any]:
        return {"mode": "pattern"}

    def handle_keyboard_event(self, event: spy.KeyboardEvent) -> None:
        """The diagnostic pattern has no interactive behavior."""

    def handle_mouse_event(self, event: spy.MouseEvent) -> None:
        pass

    def update(self, dt: float) -> None:
        pass

    def close(self) -> None:
        self._kernel = None
        self._image = None


class CameraRenderer(PatternRenderer):
    """A shaded sphere and checker floor using Falcor's existing camera controller.

    Hold right mouse for WASD/QE navigation and looking; middle mouse pans.
    Mouse motion uses ordinary viewport drag input.
    """

    _entry_point = "render_camera"

    def __init__(self, device: spy.Device, width: int, height: int) -> None:
        super().__init__(device, width, height)
        self.controller = CameraController()
        self.controller.smoothing = False
        transform = f2.Transform()
        transform.translation = spy.float3(0, 1, 5)
        self.controller.transform = transform

    def handle_keyboard_event(self, event: spy.KeyboardEvent) -> None:
        self.controller.handle_keyboard_event(event)

    def handle_mouse_event(self, event: spy.MouseEvent) -> None:
        self.controller.handle_mouse_event(event)
        # Consume each delta before the next event replaces it. Zero elapsed
        # time applies unsmoothed mouse input without advancing held-key motion.
        self.controller.update(0.0)

    def update(self, dt: float) -> None:
        self.controller.update(dt)

    def render(self, frame_id: int, dt: float) -> spy.Texture:
        image = self._image
        if image is None or self._kernel is None:
            raise RuntimeError("Renderer is closed")
        transform = self.controller.transform
        rotation = spy.math.matrix_from_quat(transform.rotation)
        self._kernel.dispatch(
            thread_count=[image.width, image.height, 1],
            vars={
                "pattern": image,
                "width": image.width,
                "height": image.height,
                "camera_position": transform.translation,
                "camera_right": rotation.get_col(0),
                "camera_up": rotation.get_col(1),
                "camera_forward": -rotation.get_col(2),
            },
        )
        return image

    def stats(self) -> dict[str, Any]:
        pos = self.controller.transform.translation
        return {"mode": "camera", "camera_position": [pos.x, pos.y, pos.z]}
