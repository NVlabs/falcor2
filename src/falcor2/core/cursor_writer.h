// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/core/error.h"

#include <sgl/device/buffer_cursor.h>
#include <sgl/device/shader_cursor.h>

/// Generate a static ShaderCursor writer that delegates to an instance method.
#define FALCOR_STATIC_WRITE_TO_SHADER_CURSOR_METHOD(cls, method)                                                       \
    static void write_to_cursor(sgl::ShaderCursor cursor, const cls* value)                                            \
    {                                                                                                                  \
        FALCOR_CHECK_NOT_NULL(value);                                                                                  \
        value->method(cursor);                                                                                         \
    }

/// Generate a static BufferElementCursor writer that delegates to an instance method.
#define FALCOR_STATIC_WRITE_TO_BUFFER_CURSOR_METHOD(cls, method)                                                       \
    static void write_to_cursor(sgl::BufferElementCursor cursor, const cls* value)                                     \
    {                                                                                                                  \
        FALCOR_CHECK_NOT_NULL(value);                                                                                  \
        value->method(cursor);                                                                                         \
    }

/// Generate static ShaderCursor and BufferElementCursor writers that delegate to an instance method.
#define FALCOR_STATIC_WRITE_TO_CURSOR_METHOD(cls, method)                                                              \
    FALCOR_STATIC_WRITE_TO_SHADER_CURSOR_METHOD(cls, method)                                                           \
    FALCOR_STATIC_WRITE_TO_BUFFER_CURSOR_METHOD(cls, method)

/// Generate a static ShaderCursor writer that delegates to instance write_to_cursor().
#define FALCOR_STATIC_WRITE_TO_SHADER_CURSOR(cls) FALCOR_STATIC_WRITE_TO_SHADER_CURSOR_METHOD(cls, write_to_cursor)

/// Generate a static BufferElementCursor writer that delegates to instance write_to_cursor().
#define FALCOR_STATIC_WRITE_TO_BUFFER_CURSOR(cls) FALCOR_STATIC_WRITE_TO_BUFFER_CURSOR_METHOD(cls, write_to_cursor)

/// Generate static ShaderCursor and BufferElementCursor writers that delegate to instance write_to_cursor().
#define FALCOR_STATIC_WRITE_TO_CURSOR(cls) FALCOR_STATIC_WRITE_TO_CURSOR_METHOD(cls, write_to_cursor)

/// Generate virtual ShaderCursor and BufferElementCursor overrides that delegate to write_to_cursor_impl().
#define FALCOR_WRITE_TO_CURSOR_OVERRIDES()                                                                             \
    virtual void write_to_cursor(sgl::BufferElementCursor cursor) const override                                       \
    {                                                                                                                  \
        write_to_cursor_impl(cursor);                                                                                  \
    }                                                                                                                  \
    virtual void write_to_cursor(sgl::ShaderCursor cursor) const override                                              \
    {                                                                                                                  \
        write_to_cursor_impl(cursor);                                                                                  \
    }
