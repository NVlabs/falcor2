// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <sgl/core/macros.h>

#ifdef FALCOR_DLL
#define FALCOR_API SGL_API_EXPORT
#else // FALCOR_DLL
#define FALCOR_API SGL_API_IMPORT
#endif // FALCOR_DLL

#define FALCOR_CONCAT(x, y) _FALCOR_CONCAT_IMPL(x, y)
#define _FALCOR_CONCAT_IMPL(x, y) x##y

#define FALCOR_NON_COPYABLE(cls) SGL_NON_COPYABLE(cls)
#define FALCOR_NON_MOVABLE(cls) SGL_NON_MOVABLE(cls)
#define FALCOR_NON_COPYABLE_AND_MOVABLE(cls) SGL_NON_COPYABLE_AND_MOVABLE(cls)

#define FALCOR_UNUSED(...) SGL_UNUSED(__VA_ARGS__)

#define FALCOR_UNIQUE_NAME(base) FALCOR_CONCAT(base, __COUNTER__)

/// Uses static initialization to run the `block` once.
#define FALCOR_STATIC_ONCE(block) _FALCOR_STATIC_ONCE_IMPL(FALCOR_UNIQUE_NAME(do_once), block)
#define _FALCOR_STATIC_ONCE_IMPL(name, block)                                                                          \
    namespace {                                                                                                        \
    static struct name {                                                                                               \
        name()                                                                                                         \
        {                                                                                                              \
            block;                                                                                                     \
        }                                                                                                              \
    } FALCOR_CONCAT(name, _instance);                                                                                  \
    }
