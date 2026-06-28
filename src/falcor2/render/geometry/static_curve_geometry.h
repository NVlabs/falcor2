// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/render/geometry.h"

#include "falcor2/core/types.h"
#include "falcor2/core/properties.h"
#include "falcor2/importers/importer_types.h"

namespace falcor {

struct StaticCurveGeometryDataDesc {
    template<typename T>
    struct VertexAttributeStream {
        const T* data;
        size_t stride;
    };

    template<typename T>
    struct IndexStream {
        const T* data;
        size_t stride;
    };

    std::string_view name;
    uint32_t vertex_count;
    VertexAttributeStream<float3> position_stream;
    VertexAttributeStream<float> radius_stream;
    uint32_t index_count;
    IndexStream<uint32_t> index_stream;
    CurveIndexingMode indexing_mode{CurveIndexingMode::list};
};

/// Static curve geometry.
class FALCOR_API StaticCurveGeometry : public Geometry {
    FALCOR_SCENE_OBJECT(StaticCurveGeometry, Geometry)
public:
    /// Destructor.
    ~StaticCurveGeometry() override;

    /// Set the curve data.
    /// @param desc Description of the curve data.
    void set_curve_data(const StaticCurveGeometryDataDesc& desc);

    // SceneObject interface

    virtual void clear_invalid_references() override;

    // Geometry interface

    virtual void update_render_state() override;

    /// Reflect this class.
    template<reflection::ClassReflector R>
    static void reflect(R& r)
    {
        FALCOR_UNUSED(r);
    }

    /// Convert list-mode indices (pairs per segment) to successive-mode indices
    /// (one index per segment, where positions[index] and positions[index+1] form the segment).
    /// If any pair is non-consecutive, vertices are rearranged so each segment's
    /// endpoints are adjacent.
    static void
    convert_list_to_successive_indices(std::vector<uint32_t>& indices, std::vector<shared::PackedLSSVertex>& vertices);

private:
    std::vector<shared::PackedLSSVertex> m_vertices;
    std::vector<uint32_t> m_indices;

    RenderLSSGeometryHandle m_render_geometry;
};

} // namespace falcor
