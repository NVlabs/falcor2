// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <sgl/core/macros.h>

#define BEGIN_DISABLE_USD_WARNINGS                                                                                     \
    SGL_DIAGNOSTIC_PUSH                                                                                                \
    SGL_DISABLE_MSVC_WARNING(4003) /* Not enough macro arguments */                                                    \
    SGL_DISABLE_MSVC_WARNING(4127) /* Conditional expression is constant */                                            \
    SGL_DISABLE_MSVC_WARNING(4244) /* Conversion possible loss of data */                                              \
    SGL_DISABLE_MSVC_WARNING(4267) /* Conversion possible loss of data */                                              \
    SGL_DISABLE_MSVC_WARNING(4305) /* Truncation double to float */                                                    \
    SGL_DISABLE_MSVC_WARNING(4324) /* Structure was padded for alignment */                                            \
    SGL_DISABLE_MSVC_WARNING(5033) /* 'register' storage class specifier deprecated */                                 \
    SGL_DISABLE_MSVC_WARNING(4100) /* Unreferenced format parameter */                                                 \
    SGL_DISABLE_CLANG_WARNING("-Wignored-attributes")                                                                  \
    SGL_DISABLE_GCC_WARNING("-Wignored-attributes")                                                                    \
    SGL_DISABLE_GCC_WARNING("-Wparentheses")

#define END_DISABLE_USD_WARNINGS SGL_DIAGNOSTIC_POP
