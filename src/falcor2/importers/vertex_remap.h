// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/core/macros.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace falcor {

/// One vertex attribute used to determine exact vertex identity.
///
/// All fields are measured in bytes. The data is borrowed for the duration of generate_vertex_remap().
struct VertexRemapStream {
    /// Address of attribute element zero.
    const std::byte* data = nullptr;

    /// Number of meaningful bytes in each element, excluding padding.
    size_t element_size = 0;

    /// Byte distance between consecutive vertex elements.
    size_t stride = 0;
};

/// Maps source vertices to compact vertices and compact vertices back to their first source occurrence.
struct VertexRemap {
    /// Compact vertex index for each source vertex.
    std::vector<uint32_t> old_to_new;

    /// First source occurrence for each compact vertex. Empty for an identity remap.
    std::vector<uint32_t> new_to_old;

    /// Number of compact vertices.
    size_t unique_vertex_count() const { return is_identity() ? old_to_new.size() : new_to_old.size(); }

    /// True when every source vertex already has a unique compact index.
    bool is_identity() const { return new_to_old.empty(); }
};

/// Generate a deterministic remap for vertices that are byte-identical in every supplied stream.
///
/// Every stream describes the same vertex_count elements. For a non-empty input, its data must remain valid throughout
/// the call and provide at least (vertex_count - 1) * stride + element_size readable bytes. Stream padding is ignored,
/// no input storage is retained, and output indices follow the first occurrence of each unique vertex. With no streams,
/// all source vertices are equivalent.
FALCOR_API VertexRemap generate_vertex_remap(std::span<const VertexRemapStream> streams, size_t vertex_count);

} // namespace falcor
