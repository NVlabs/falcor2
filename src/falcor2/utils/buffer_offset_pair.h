// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/utils/buffer_handle.h"

namespace falcor {

/// Non-owning shader reference to a byte offset within a buffer.
struct FALCOR_API BufferOffsetPair {
    FALCOR_STATIC_WRITE_TO_CURSOR(BufferOffsetPair);

    static constexpr std::string_view slang_type_name = "BufferOffsetPair";
    static constexpr std::array<std::string_view, 1> slangpy_imports() { return {"falcor2/utils.slang"}; }

    BufferHandle buffer = to_handle(ref<sgl::Buffer>{});
    uint32_t offset{0};

    template<typename TCursor>
    void write_to_cursor(TCursor& cursor) const
    {
        cursor["buffer"] = buffer;
        cursor["offset"] = offset;
        cursor["padding"] = uint32_t(0);
    }
};

} // namespace falcor
