# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from slangpy import float3, float3x4, float4, float4x4, quatf
from slangpy.math import matrix_from_quat


class PosRotScale:
    def __init__(
        self,
        pos: float3 = float3(0),
        rot: quatf = quatf.identity(),
        scale: float3 = float3(1),
    ):
        super().__init__()
        self.pos = pos
        self.rot = rot
        self.scale = scale

    def to_float3x4(self):
        rot = matrix_from_quat(self.rot)
        mat = float3x4()
        mat.set_row(0, float4(rot.get_row(0) * self.scale.x, self.pos.x))
        mat.set_row(1, float4(rot.get_row(1) * self.scale.y, self.pos.y))
        mat.set_row(2, float4(rot.get_row(2) * self.scale.z, self.pos.z))
        return mat

    def to_float4x4(self):
        rot = matrix_from_quat(self.rot)
        mat = float4x4()
        mat.set_row(0, float4(rot.get_row(0) * self.scale.x, self.pos.x))
        mat.set_row(1, float4(rot.get_row(1) * self.scale.y, self.pos.y))
        mat.set_row(2, float4(rot.get_row(2) * self.scale.z, self.pos.z))
        mat.set_row(3, float4(0, 0, 0, 1))
        return mat

    def __str__(self):
        return f"PosRotScale({self.pos}, {self.rot}, {self.scale})"

    def __repr__(self):
        return str(self)
