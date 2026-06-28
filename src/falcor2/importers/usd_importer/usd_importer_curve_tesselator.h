// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/importers/importer_types.h"

#include "usd_importer_macros.h"

BEGIN_DISABLE_USD_WARNINGS
#include <pxr/base/gf/vec3f.h>
#include <pxr/base/vt/types.h>
#include <pxr/base/vt/array.h>
END_DISABLE_USD_WARNINGS


namespace falcor {
namespace usd_importer {

struct TesselatorInputCurve {
    enum class Type {
        linear,
        cubic,
    };

    enum class Basis {
        bezier,
        bspline,
        catmull_rom,
    };

    std::string name;
    Type type{Type::linear};
    Basis basis{Basis::bezier};

    pxr::VtArray<pxr::GfVec3f> points;
    pxr::VtArray<float> widths;
    pxr::VtArray<int> vertex_counts;

    /// Number of linear segments to generate per cubic span.
    int segments_per_span{8};

    /// Default radius when widths are not provided.
    float default_radius{0.01f};
};

ImporterCurve tessellate_curves(const TesselatorInputCurve& input);

} // namespace usd_importer
} // namespace falcor
