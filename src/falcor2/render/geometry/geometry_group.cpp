// SPDX-License-Identifier: Apache-2.0

#include "geometry_group.h"

namespace falcor {

GeometryGroup::~GeometryGroup() { }

void GeometryGroup::set_geometries(const std::vector<Geometry*>& geometries)
{
    m_geometries = geometries;
}

void GeometryGroup::clear_invalid_references()
{
    erase_invalid_objects(m_geometries);
}

FALCOR_SCENE_REGISTER_GEOMETRY(GeometryGroup);

} // namespace falcor
