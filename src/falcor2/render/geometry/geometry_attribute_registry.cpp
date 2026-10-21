// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "geometry_attribute_registry.h"

#include "falcor2/render/scene.h"

#include <sgl/core/error.h>
#include <sgl/core/logger.h>

#include <cstring>
#include <string_view>
#include <unordered_set>

namespace falcor {

namespace {

template<typename T>
void register_geometry_attribute_impl(Scene& scene, const T& attribute)
{
    if (const auto type = get_additional_attribute_type(attribute.type))
        scene._get_or_create_geometry_attribute_id(
            attribute.semantic,
            attribute.index,
            attribute.name,
            *type,
            attribute.component_count
        );
}

} // namespace

GeometryAttributeClassification classify_geometry_attribute(MeshAttributeSemantic semantic, uint32_t index)
{
    if (semantic == MeshAttributeSemantic::tex_coord && index < 8)
        return {GeometryAttributeClassificationKind::fixed, index};
    if (semantic == MeshAttributeSemantic::color && index == 0)
        return {GeometryAttributeClassificationKind::fixed, 8};

    switch (semantic) {
    case MeshAttributeSemantic::position:
    case MeshAttributeSemantic::normal:
    case MeshAttributeSemantic::tangent:
    case MeshAttributeSemantic::handedness:
        return {GeometryAttributeClassificationKind::core};
    default:
        return {GeometryAttributeClassificationKind::named};
    }
}

std::optional<shared::AdditionalAttributeType> get_additional_attribute_type(sgl::DataType type)
{
    switch (type) {
    case sgl::DataType::bool_:
        return shared::AdditionalAttributeType::bool_;
    case sgl::DataType::int32:
        return shared::AdditionalAttributeType::int32;
    case sgl::DataType::float16:
        return shared::AdditionalAttributeType::float16;
    case sgl::DataType::float32:
        return shared::AdditionalAttributeType::float32;
    default:
        return std::nullopt;
    }
}

void register_geometry_attribute(Scene& scene, const GeometryVertexAttributeDesc& attribute)
{
    register_geometry_attribute_impl(scene, attribute);
}

void register_geometry_attribute(Scene& scene, const GeometryVertexAttribute& attribute)
{
    register_geometry_attribute_impl(scene, attribute);
}

void warn_if_unsupported_geometry_attribute(
    const GeometryVertexAttributeDesc& attribute,
    std::string_view geometry_type
)
{
    if (get_additional_attribute_type(attribute.type))
        return;

    sgl::log_warn(
        "{} geometry attribute '{}' (semantic '{}', index {}) uses unsupported data type '{}' and will not be "
        "available to shaders.",
        geometry_type,
        attribute.name,
        sgl::enum_to_string(attribute.semantic),
        attribute.index,
        sgl::enum_to_string(attribute.type)
    );
}

PreparedGeometryAttributes::PreparedGeometryAttributes(size_t capacity)
{
    m_attributes.reserve(capacity);
    m_converted_data.reserve(capacity);
}

void PreparedGeometryAttributes::append(
    const Scene& scene,
    const GeometryVertexAttribute& attribute,
    size_t vertex_count
)
{
    const auto type = get_additional_attribute_type(attribute.type);
    if (!type)
        return;

    const auto attribute_id = scene._get_geometry_attribute_id(attribute.semantic, attribute.index, attribute.name);
    if (!attribute_id)
        return;

    std::span<const uint8_t> attribute_data = attribute.data;
    if (*type == shared::AdditionalAttributeType::bool_) {
        const size_t element_count = vertex_count * attribute.component_count;
        std::vector<uint8_t>& normalized = m_converted_data.emplace_back(element_count);
        for (size_t index = 0; index < element_count; ++index) {
            bool value;
            std::memcpy(&value, attribute.data.data() + index * sizeof(bool), sizeof(bool));
            normalized[index] = value ? uint8_t(1) : uint8_t(0);
        }
        attribute_data = normalized;
    }

    m_attributes.push_back({
        .mask_bit = *attribute_id,
        .type = *type,
        .component_count = attribute.component_count,
        .data = attribute_data,
    });
}

void validate_unique_geometry_attributes(
    std::span<const GeometryVertexAttributeDesc> attributes,
    std::string_view geometry_type
)
{
    std::unordered_set<uint32_t> fixed_attribute_ids;
    std::unordered_set<uint64_t> core_attribute_keys;
    std::unordered_set<std::string_view> named_attribute_names;

    for (const GeometryVertexAttributeDesc& attribute : attributes) {
        const GeometryAttributeClassification classification
            = classify_geometry_attribute(attribute.semantic, attribute.index);
        switch (classification.kind) {
        case GeometryAttributeClassificationKind::fixed:
            SGL_CHECK(
                fixed_attribute_ids.insert(classification.fixed_id).second,
                "Duplicate {} vertex attribute with fixed ID {}",
                geometry_type,
                classification.fixed_id
            );
            break;
        case GeometryAttributeClassificationKind::core: {
            const uint64_t key = (uint64_t(attribute.semantic) << 32) | attribute.index;
            SGL_CHECK(
                core_attribute_keys.insert(key).second,
                "Duplicate {} core vertex attribute semantic and index",
                geometry_type
            );
            break;
        }
        case GeometryAttributeClassificationKind::named:
            SGL_CHECK(
                named_attribute_names.insert(attribute.name).second,
                "Duplicate {} named vertex attribute '{}'",
                geometry_type,
                attribute.name
            );
            break;
        }
    }
}

} // namespace falcor
