# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import slangpy.bindings as spybind
import slangpy.reflection as spyref

import falcor2 as f2


class CameraMarshall(spybind.Marshall):
    def __init__(self, layout: spyref.SlangProgramLayout):
        super().__init__(layout)
        ct = layout.find_type_by_name("Camera")
        assert ct is not None
        self.slang_type = ct

    def get_shape(self, value: Any):
        return spybind.Shape(value.height, value.width)

    def resolve_type(self, context: spybind.BindContext, bound_type: spyref.SlangType):
        if bound_type.full_name in ["Camera", "Ray", "RaySampler"]:
            return bound_type
        return None

    def resolve_dimensionality(
        self,
        context: spybind.BindContext,
        binding: spybind.BoundVariable,
        vector_target_type: spyref.SlangType,
    ):
        if vector_target_type.full_name in ["Camera"]:
            return 0
        elif vector_target_type.full_name in ["Ray", "RaySampler"]:
            return 2
        return None

    def create_calldata(
        self, context: spybind.BindContext, binding: spybind.BoundVariable, data: Any
    ):
        return data.get_uniforms()

    def gen_calldata(
        self,
        cgb: spybind.CodeGenBlock,
        context: spybind.BindContext,
        binding: spybind.BoundVariable,
    ):
        cgb.add_import("falcor2/render.slang")
        binding.gen_calldata_type_name(cgb, "Camera")


def _create_camera_marshall(layout: spyref.SlangProgramLayout, _value: f2.Camera) -> CameraMarshall:
    return CameraMarshall(layout)


spybind.PYTHON_TYPES[f2.Camera] = _create_camera_marshall
spybind.PYTHON_TYPES[f2.CameraUniforms] = _create_camera_marshall
