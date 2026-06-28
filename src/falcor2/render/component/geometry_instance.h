// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/render/component.h"
#include "falcor2/render/render_scene.h"

namespace falcor {

/// Geometry instance component.
/// A geometry instance references instances the reference geometry on the entity.
/// It allows to assign per-instance materials.
class FALCOR_API GeometryInstance : public Component {
    FALCOR_SCENE_OBJECT(GeometryInstance, Component)
public:
    /// Destructor.
    ~GeometryInstance() override;

    /// Geometry of this instance.
    Geometry* geometry() const { return m_geometry; }
    void set_geometry(Geometry* geometry);

    /// Materials assigned to this geometry instance.
    const std::vector<Material*>& materials() const { return m_materials; }
    void set_materials(const std::vector<Material*>& materials);

    /// GeometryInstanceID assigned to this geometry instance.
    shared::GeometryInstanceID geometry_instance_id() const;
    /// Number of geometries belonging to this geometry instance.
    /// All subsequent geometries use successive GeometryInstanceIDs.
    uint32_t geometry_instance_count() const;

    // SceneObject interface

    virtual void clear_invalid_references() override;

    virtual void on_add_to_scene() override;
    virtual void on_remove_from_scene() override;

    // Component interface

    virtual void update_render_state() override;

    /// Reflect this class.
    template<reflection::ClassReflector R>
    static void reflect(R& r)
    {
        FALCOR_UNUSED(r);
    }

private:
    Geometry* m_geometry;
    std::vector<Material*> m_materials;

    RenderGeometryInstanceHandle m_render_geometry_instance;
};

} // namespace falcor
