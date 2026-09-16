# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re

import slangpy.bindings as spybind

import falcor2 as f2


# TextureHandle is generic in Slang. The native fallback metadata names its base type,
# while this registration also accepts specializations used by function parameters.
spybind.register_write_to_cursor_type(
    f2.TextureHandle,
    slang_type_name="TextureHandle",
    imports=("falcor2/render.slang",),
    accepted_type_regex=re.compile(r"TextureHandle(?:<.*>)?"),
)
