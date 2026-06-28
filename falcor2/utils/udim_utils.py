# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


def decode_udim_index(udim_index: int) -> tuple[int, int]:
    """Decode a UDIM index into its U and V tile coordinates.
    The tile runs from U to U+1 in the U direction and V to V+1 in the V direction.
    """

    u = (udim_index - 1001) % 10
    v = (udim_index - 1001) // 10
    return (u, v)
