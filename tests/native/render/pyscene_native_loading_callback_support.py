# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


def callback_material_name(*, has_existing_material: bool) -> str:
    return (
        "PyScene Append Callback Observed Existing Material"
        if has_existing_material
        else "PyScene Callback Material"
    )
