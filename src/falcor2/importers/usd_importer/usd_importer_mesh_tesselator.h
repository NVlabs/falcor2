// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/importers/importer_types.h"

#include "usd_importer_macros.h"

BEGIN_DISABLE_USD_WARNINGS
#include <pxr/base/gf/vec3f.h>
#include <pxr/base/gf/vec2f.h>
#include <pxr/base/vt/types.h>
#include <pxr/base/vt/array.h>
END_DISABLE_USD_WARNINGS


namespace falcor {
namespace usd_importer {

struct TesselatorInputMesh {
    std::string name;
    pxr::TfToken subdiv_scheme;
    pxr::TfToken orientation;
    pxr::TfToken facevarying_interpolation;
    pxr::TfToken vertex_boundary_interpolation;

    pxr::VtArray<pxr::GfVec3f> positions;
    pxr::VtArray<int> face_vertex_counts;
    pxr::VtArray<int> face_indices;

    pxr::VtArray<int> hole_indices;
    int refinement_level = 0;

    pxr::TfToken uv_interpolation;
    pxr::VtArray<pxr::GfVec2f> uvs;

    pxr::TfToken normals_interpolation;
    pxr::VtArray<pxr::GfVec3f> normals;

    struct Subgeometry {
        std::string name;
        std::string material_name;
        pxr::VtArray<int> face_indices;
    };

    std::vector<Subgeometry> subgeometries;
};

ImporterMesh tessellate(const TesselatorInputMesh& input);

} // namespace usd_importer
} // namespace falcor
