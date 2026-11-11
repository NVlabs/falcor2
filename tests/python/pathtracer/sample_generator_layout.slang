// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import falcor2.rendernodes.reference_pathtracer;
__exported import falcor2.render;
__exported import falcor2.utils;

public uint payload_size()
{
    return sizeof(PathPayload);
}
