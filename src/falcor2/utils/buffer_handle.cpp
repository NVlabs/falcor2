// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "buffer_handle.h"

#include <sgl/device/cursor_utils.h>

namespace falcor {

FALCOR_STATIC_ONCE(sgl::cursor_utils::register_cursor_writer<BufferHandle>());

} // namespace falcor
