# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, Any
from slangpy import float4, int2
from slangpy.math import tan, radians, matrix_from_quat
import slangpy as spy
from falcor2.minitracer.posrotscale import PosRotScale

if TYPE_CHECKING:
    from falcor2.minitracer.scene import Scene


class Camera:
    def __init__(self, scene: "Scene", index: int):
        super().__init__()
        self.scene = scene
        self.index = index
        self.transform = PosRotScale()
        self.width = 100
        self.height = 100
        self.aspect_ratio = 1.0
        self.fov_degrees = 70.0
        self.recompute()

    def recompute(self):
        self.aspect_ratio = float(self.width) / float(self.height)

        pr = self.transform

        rot = matrix_from_quat(pr.rot)

        self.pos = pr.pos
        self.fwd = -rot.get_col(2)
        self.right = rot.get_col(0)
        self.up = rot.get_col(1)

        fov = radians(self.fov_degrees)

        self.image_u = self.right * tan(fov * 0.5) * self.aspect_ratio
        self.image_v = self.up * tan(fov * 0.5)
        self.image_w = self.fwd

    def get_uniforms(self):
        self.recompute()
        return {
            "position": self.pos,
            "image_u": self.image_u,
            "image_v": self.image_v,
            "image_w": self.image_w,
            "dims": int2(self.width, self.height),
        }

    def create_output_tensor(self, dtype: Any = float4) -> spy.Tensor:
        return spy.Tensor.empty(
            self.scene.device,
            shape=(self.height, self.width),
            dtype=dtype,
        )
