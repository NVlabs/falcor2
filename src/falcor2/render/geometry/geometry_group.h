// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/render/geometry.h"

namespace falcor {

/// TODO(scene): This is not implemented yet.

/// Geometry group.
/// A geometry group is a collection of geometries treated as a single geometry.
class FALCOR_API GeometryGroup : public Geometry {
    FALCOR_SCENE_OBJECT(GeometryGroup, Geometry)
public:
    /// Destructor.
    ~GeometryGroup() override;

    /// List of geometries.
    const std::vector<Geometry*>& geometries() const { return m_geometries; }
    void set_geometries(const std::vector<Geometry*>& geometries);
    void add_geometry(Geometry* geometry);
    void remove_geometry(Geometry* geometry);

    // SceneObject interface

    virtual void clear_invalid_references() override;

    /// Reflect this class.
    template<reflection::ClassReflector R>
    static void reflect(R& r)
    {
        FALCOR_UNUSED(r);
    }

private:
    std::vector<Geometry*> m_geometries;
};

} // namespace falcor
