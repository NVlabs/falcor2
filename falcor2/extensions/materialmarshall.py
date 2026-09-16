# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re

import slangpy.bindings as spybind
import slangpy.reflection as spyref

import falcor2 as f2


class MaterialMarshall(spybind.WriteToCursorMarshall):
    """Marshall a Material through its value-dependent concrete Slang type."""

    def __init__(self, layout: spyref.SlangProgramLayout, value: f2.Material):
        type_name = value.slang_type_name
        super().__init__(
            layout,
            spybind.WriteToCursorMarshallInfo(
                slang_type_name=type_name,
                signature=f"falcor2.Material:{type_name}",
                imports=("falcor2/render.slang",),
                accepted_type_regex=re.compile(re.escape(type_name.replace("::", "."))),
            ),
        )

    def resolve_types(
        self, context: spybind.BindContext, bound_type: spyref.SlangType
    ) -> list[spyref.SlangType]:
        if bound_type.full_name == "IMaterial":
            return [self.slang_type]
        return super().resolve_types(context, bound_type)


def _create_material_marshall(
    layout: spyref.SlangProgramLayout, value: f2.Material
) -> MaterialMarshall:
    return MaterialMarshall(layout, value)


# Type lookup follows the Python MRO, so this covers native derived materials and
# Python subclasses of Material as well.
spybind.PYTHON_TYPES[f2.Material] = _create_material_marshall
