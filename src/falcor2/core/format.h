// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <sgl/core/format.h>

#include "falcor2/core/ustring.h"

template<>
struct fmt::formatter<falcor::ustring> : formatter<std::string_view> {
    template<typename FormatContext>
    auto format(falcor::ustring ustring, FormatContext& ctx) const
    {
        return formatter<std::string_view>::format(std::string_view(ustring), ctx);
    }
};
