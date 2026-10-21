// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/render/geometry/geometry_attribute_classification.h"
#include "falcor2/render/render_scene.h"
#include "falcor2/render/shared_geometry_types.h"

#include <cstdint>
#include <optional>
#include <span>
#include <string_view>

namespace falcor {

class Scene;

/// Map a supported source data type to its packed additional-attribute representation.
std::optional<shared::AdditionalAttributeType> get_additional_attribute_type(sgl::DataType type);

/// Register an authored geometry attribute with the scene if its source type is supported.
void register_geometry_attribute(Scene& scene, const GeometryVertexAttributeDesc& attribute);

/// Register an owned geometry attribute with the scene if its source type is supported.
void register_geometry_attribute(Scene& scene, const GeometryVertexAttribute& attribute);

/// Warn when an authored geometry attribute cannot be made available to shaders.
void warn_if_unsupported_geometry_attribute(
    const GeometryVertexAttributeDesc& attribute,
    std::string_view geometry_type
);

/// Additional geometry attributes prepared for upload to the render scene.
class PreparedGeometryAttributes {
public:
    explicit PreparedGeometryAttributes(size_t capacity = 0);

    PreparedGeometryAttributes(const PreparedGeometryAttributes&) = delete;
    PreparedGeometryAttributes& operator=(const PreparedGeometryAttributes&) = delete;
    PreparedGeometryAttributes(PreparedGeometryAttributes&&) = default;
    PreparedGeometryAttributes& operator=(PreparedGeometryAttributes&&) = default;

    void append(const Scene& scene, const GeometryVertexAttribute& attribute, size_t vertex_count);

    std::span<const RenderGeometryAdditionalAttributeData> attributes() const { return m_attributes; }

private:
    std::vector<RenderGeometryAdditionalAttributeData> m_attributes;
    std::vector<std::vector<uint8_t>> m_converted_data;
};

/// Validate that no two attributes in a geometry resolve to the same attribute identity.
void validate_unique_geometry_attributes(
    std::span<const GeometryVertexAttributeDesc> attributes,
    std::string_view geometry_type
);

} // namespace falcor
