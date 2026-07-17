// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "falcor2/importers/importer_collections.h"
#include "falcor2/importers/importer.h"
#include "falcor2/importers/importer_edits.h"
#include "falcor2/core/error.h"

#include <utility>

namespace falcor {

ImporterMaterialSelector::ImporterMaterialSelector(Importer& importer, std::string name)
    : m_importer(&importer)
    , m_name(std::move(name))
{
}

void ImporterMaterialSelector::replace(std::string material_type, Properties props) const
{
    if (material_type.empty()) {
        FALCOR_THROW("Importer material replacement type must not be empty");
    }

    m_importer->add_edit(std::make_unique<ReplaceMaterialEdit>(m_name, std::move(material_type), std::move(props)));
}

void ImporterMaterialSelector::replace(ImporterMaterial::Constructor constructor, Properties props) const
{
    if (!constructor) {
        FALCOR_THROW("Importer material replacement constructor must not be empty");
    }

    m_importer->add_edit(std::make_unique<ReplaceMaterialEdit>(m_name, std::move(constructor), std::move(props)));
}

ImporterMaterialCollection::ImporterMaterialCollection(Importer& importer)
    : m_importer(importer)
{
}

ImporterMaterialSelector ImporterMaterialCollection::find(std::string name)
{
    if (name.empty()) {
        FALCOR_THROW("Importer material selector name must not be empty");
    }
    return ImporterMaterialSelector(m_importer, std::move(name));
}

ImporterMaterialSelector ImporterMaterialCollection::operator[](std::string name)
{
    return find(std::move(name));
}

ImporterEnvCollection::ImporterEnvCollection(Importer& importer)
    : m_importer(importer)
{
}

void ImporterEnvCollection::set(std::filesystem::path path, float exposure, std::string name)
{
    m_importer.add_edit(std::make_unique<SetEnvironmentEdit>(std::move(path), exposure, std::move(name)));
}

ImporterCameraCollection::ImporterCameraCollection(Importer& importer)
    : m_importer(importer)
{
}

ImporterCameraSelector ImporterCameraCollection::create(
    std::string name,
    float focus_distance,
    float focal_length,
    float fstop,
    float2 depth_range,
    ImporterCamera::Projection projection,
    ImporterCamera::FOVDirection fov_direction,
    float sensor_size_mm
)
{
    ImporterCamera camera;
    camera.name = std::move(name);
    camera.focus_distance = focus_distance;
    camera.focal_length = focal_length;
    camera.fstop = fstop;
    camera.depth_range = depth_range;
    camera.projection = projection;
    camera.fov_direction = fov_direction;
    camera.sensor_size_mm = sensor_size_mm;

    ImporterCameraSelector selector;
    selector.id = m_importer.next_selector_id();
    m_importer.add_edit(std::make_unique<CreateCameraEdit>(selector.id, std::move(camera)));
    return selector;
}

ImporterCameraSelector ImporterCameraCollection::create_fov(
    std::string name,
    float fov_degrees,
    float focus_distance,
    float fstop,
    float2 depth_range,
    ImporterCamera::Projection projection,
    ImporterCamera::FOVDirection fov_direction,
    float sensor_size_mm
)
{
    const float focal_length = ImporterCamera::focal_length_from_fov_degrees(fov_degrees, sensor_size_mm);
    return create(
        std::move(name),
        focus_distance,
        focal_length,
        fstop,
        depth_range,
        projection,
        fov_direction,
        sensor_size_mm
    );
}

ImporterNodeCollection::ImporterNodeCollection(Importer& importer)
    : m_importer(importer)
{
}

ImporterNodeSelector ImporterNodeCollection::create(
    std::string name,
    float4x4 transform,
    std::optional<ImporterCameraSelector> camera,
    std::optional<ImporterNodeSelector> parent
)
{
    ImporterNodeSelector selector;
    selector.id = m_importer.next_selector_id();
    m_importer.add_edit(
        std::make_unique<CreateNodeEdit>(selector.id, std::move(name), transform, std::move(camera), std::move(parent))
    );
    return selector;
}

ImporterNodeSelector ImporterNodeCollection::create_camera(
    std::string name,
    float4x4 transform,
    float focus_distance,
    float focal_length,
    float fstop,
    float2 depth_range,
    ImporterCamera::Projection projection,
    ImporterCamera::FOVDirection fov_direction,
    float sensor_size_mm
)
{
    ImporterCameraSelector camera = m_importer.cameras().create(
        name,
        focus_distance,
        focal_length,
        fstop,
        depth_range,
        projection,
        fov_direction,
        sensor_size_mm
    );
    return create(std::move(name), transform, camera);
}

ImporterNodeSelector ImporterNodeCollection::create_camera_fov(
    std::string name,
    float4x4 transform,
    float fov_degrees,
    float focus_distance,
    float fstop,
    float2 depth_range,
    ImporterCamera::Projection projection,
    ImporterCamera::FOVDirection fov_direction,
    float sensor_size_mm
)
{
    ImporterCameraSelector camera = m_importer.cameras().create_fov(
        name,
        fov_degrees,
        focus_distance,
        fstop,
        depth_range,
        projection,
        fov_direction,
        sensor_size_mm
    );
    return create(std::move(name), transform, camera);
}

} // namespace falcor
