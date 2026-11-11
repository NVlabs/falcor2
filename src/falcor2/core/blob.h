// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/core/object.h"

#include <cstdint>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace falcor {

/// Named immutable binary data.
/// The constructor copies the input, so the caller may release or modify its source afterward.
class FALCOR_API Blob : public Object {
    FALCOR_OBJECT(Blob)
public:
    Blob(std::span<const uint8_t> data, std::string name = {})
        : m_data(data.begin(), data.end())
        , m_name(std::move(name))
    {
    }

    /// Owned bytes. The returned span remains valid while this object lives.
    std::span<const uint8_t> data() const { return m_data; }

    /// Optional descriptive name.
    const std::string& name() const { return m_name; }

private:
    const std::vector<uint8_t> m_data;
    const std::string m_name;
};

} // namespace falcor
