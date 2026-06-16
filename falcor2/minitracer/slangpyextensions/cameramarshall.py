# SPDX-License-Identifier: Apache-2.0

import slangpy.bindings as spybind
import slangpy.reflection as spyref

from falcor2.minitracer.camera import Camera


class CameraMarshall(spybind.Marshall):
    def __init__(self, layout: spyref.SlangProgramLayout):
        super().__init__(layout)
        ct = layout.find_type_by_name("Camera")
        assert ct is not None
        self.slang_type = ct

    def get_shape(self, value: Camera):
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
        self, context: spybind.BindContext, binding: spybind.BoundVariable, data: "Camera"
    ):
        return data.get_uniforms()

    def gen_calldata(
        self,
        cgb: spybind.CodeGenBlock,
        context: spybind.BindContext,
        binding: spybind.BoundVariable,
    ):
        cgb.add_import("falcor2/minitracer/scene.slang")
        binding.gen_calldata_type_name(cgb, "Camera")


spybind.PYTHON_TYPES[Camera] = lambda layout, value: CameraMarshall(layout)
