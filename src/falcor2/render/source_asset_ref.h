// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/core/ustring.h"

#include <string_view>

namespace falcor {

/// Identifies an object in an external source asset.
///
/// The strings are intentionally format-neutral. For example, object_path is a
/// USD prim path for USD and a JSON pointer for glTF.
/// Identical strings share process-lifetime storage through ustring.
struct FALCOR_API SourceAssetRef {
    SourceAssetRef() = default;
    /// Construct from interned strings without consulting the string registry.
    SourceAssetRef(ustring format, ustring asset_path, ustring object_path)
        : format(format)
        , asset_path(asset_path)
        , object_path(object_path)
    {
    }
    SourceAssetRef(std::string_view format, std::string_view asset_path, std::string_view object_path)
        : SourceAssetRef(ustring(format), ustring(asset_path), ustring(object_path))
    {
    }

    ustring format;
    ustring asset_path;
    ustring object_path;

    bool operator==(const SourceAssetRef&) const = default;
};

} // namespace falcor
