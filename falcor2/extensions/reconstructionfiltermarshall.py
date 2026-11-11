# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re

import slangpy.bindings as spybind
import slangpy.reflection as spyref

import falcor2 as f2


class ReconstructionFilterMarshall(spybind.WriteToCursorMarshall):
    def __init__(self, layout: spyref.SlangProgramLayout, value: f2.ReconstructionFilter):
        type_name = value.slang_type_name
        super().__init__(
            layout,
            spybind.WriteToCursorMarshallInfo(
                slang_type_name=type_name,
                signature=f"falcor2.ReconstructionFilter:{type_name}",
                imports=("falcor2/render.slang",),
                accepted_type_regex=re.compile(re.escape(type_name)),
            ),
        )

    def resolve_types(
        self, context: spybind.BindContext, bound_type: spyref.SlangType
    ) -> list[spyref.SlangType]:
        if bound_type.full_name == "IReconstructionFilter":
            return [self.slang_type]
        return super().resolve_types(context, bound_type)


spybind.PYTHON_TYPES[f2.ReconstructionFilter] = ReconstructionFilterMarshall
