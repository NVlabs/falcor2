// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <sgl/core/error.h>

/// Helper for throwing exceptions.
/// Logs the exception and a stack trace before throwing.
#define FALCOR_THROW(...) SGL_THROW(__VA_ARGS__)

/// Helper for checking conditions and throwing exceptions.
/// Logs the exception and a stack trace before throwing.
#define FALCOR_CHECK(cond, ...) SGL_CHECK(cond, __VA_ARGS__)

/// Helper for throwing an exception if a pointer is null.
#define FALCOR_CHECK_NOT_NULL(arg) SGL_CHECK_NOT_NULL(arg)

#define FALCOR_CHECK_EQ(arg, value) SGL_CHECK_EQ(arg, value)
#define FALCOR_CHECK_NE(arg, value) SGL_CHECK_NE(arg, value)
#define FALCOR_CHECK_LT(arg, value) SGL_CHECK_LT(arg, value)
#define FALCOR_CHECK_LE(arg, value) SGL_CHECK_LE(arg, value)
#define FALCOR_CHECK_GT(arg, value) SGL_CHECK_GT(arg, value)
#define FALCOR_CHECK_GE(arg, value) SGL_CHECK_GE(arg, value)
#define FALCOR_CHECK_BOUNDS(arg, min, max) SGL_CHECK_BOUNDS(arg, min, max)

/// Helper for marking unimplemented functions.
#define FALCOR_UNIMPLEMENTED() SGL_UNIMPLEMENTED()

/// Helper for marking unreachable code.
#define FALCOR_UNREACHABLE() SGL_UNREACHABLE()

#define FALCOR_ASSERT(cond) SGL_ASSERT(cond)
#define FALCOR_ASSERT_EQ(a, b) SGL_ASSERT_EQ(a, b)
#define FALCOR_ASSERT_NE(a, b) SGL_ASSERT_NE(a, b)
#define FALCOR_ASSERT_GE(a, b) SGL_ASSERT_GE(a, b)
#define FALCOR_ASSERT_GT(a, b) SGL_ASSERT_GT(a, b)
#define FALCOR_ASSERT_LE(a, b) SGL_ASSERT_LE(a, b)
#define FALCOR_ASSERT_LT(a, b) SGL_ASSERT_LT(a, b)
