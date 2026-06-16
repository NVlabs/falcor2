// SPDX-License-Identifier: Apache-2.0

#include "nanobind.h"

#include "falcor2/render/ray_tracing_setup.h"
#include "falcor2/render/scene.h"

#include <sgl/device/device.h>
#include <sgl/device/pipeline.h>
#include <sgl/device/raytracing.h>
#include <sgl/device/shader.h>

namespace nb = nanobind;
using namespace nb::literals;

FALCOR_PY_EXPORT(render_ray_tracing_setup)
{
    using namespace falcor;

    nb::class_<SceneRayTracingSetup> setup(m, "SceneRayTracingSetup", D(SceneRayTracingSetup));

    nb::class_<SceneRayTracingSetup::Options>(setup, "Options", D(SceneRayTracingSetup, Options))
        .def(nb::init<>())
        .def_rw(
            "skip_unused_geometry_types",
            &SceneRayTracingSetup::Options::skip_unused_geometry_types,
            D(SceneRayTracingSetup, Options, skip_unused_geometry_types)
        );

    nb::class_<SceneRayTracingSetup::RayDesc>(setup, "RayDesc", D(SceneRayTracingSetup, RayDesc))
        .def(nb::init<>())
        .def_rw("name", &SceneRayTracingSetup::RayDesc::name, D(SceneRayTracingSetup, RayDesc, name))
        .def_rw("has_miss", &SceneRayTracingSetup::RayDesc::has_miss, D(SceneRayTracingSetup, RayDesc, has_miss))
        .def_rw(
            "has_closest_hit",
            &SceneRayTracingSetup::RayDesc::has_closest_hit,
            D(SceneRayTracingSetup, RayDesc, has_closest_hit)
        )
        .def_rw(
            "has_any_hit",
            &SceneRayTracingSetup::RayDesc::has_any_hit,
            D(SceneRayTracingSetup, RayDesc, has_any_hit)
        );

    nb::class_<SceneRayTracingSetup::PerGeometryTypeRayDesc>(
        setup,
        "PerGeometryTypeRayDesc",
        D(SceneRayTracingSetup, PerGeometryTypeRayDesc)
    )
        .def(nb::init<>())
        .def_rw(
            "miss_entry_point",
            &SceneRayTracingSetup::PerGeometryTypeRayDesc::miss_entry_point,
            D(SceneRayTracingSetup, PerGeometryTypeRayDesc, miss_entry_point)
        )
        .def_rw(
            "hit_groups",
            &SceneRayTracingSetup::PerGeometryTypeRayDesc::hit_groups,
            D(SceneRayTracingSetup, PerGeometryTypeRayDesc, hit_groups)
        );

    setup
        .def_static(
            "create",
            nb::overload_cast<
                const Scene*,
                std::span<const SceneRayTracingSetup::RayDesc>,
                std::optional<SceneRayTracingSetup::Options>>(&SceneRayTracingSetup::create),
            "scene"_a,
            "ray_descs"_a,
            "options"_a.none() = nb::none(),
            D(SceneRayTracingSetup, create)
        )
        .def_static(
            "create",
            nb::overload_cast<
                const Scene*,
                std::span<const SceneRayTracingSetup::PerGeometryTypeRayDesc>,
                std::optional<SceneRayTracingSetup::Options>>(&SceneRayTracingSetup::create),
            "scene"_a,
            "ray_descs"_a,
            "options"_a.none() = nb::none(),
            D(SceneRayTracingSetup, create)
        )
        .def_ro("entry_points", &SceneRayTracingSetup::entry_points, D(SceneRayTracingSetup, entry_points))
        .def_ro("hit_groups", &SceneRayTracingSetup::hit_groups, D(SceneRayTracingSetup, hit_groups))
        .def_ro(
            "sbt_hit_group_names",
            &SceneRayTracingSetup::sbt_hit_group_names,
            D(SceneRayTracingSetup, sbt_hit_group_names)
        )
        .def_ro(
            "sbt_miss_entry_points",
            &SceneRayTracingSetup::sbt_miss_entry_points,
            D(SceneRayTracingSetup, sbt_miss_entry_points)
        )
        .def_ro("pipeline_flags", &SceneRayTracingSetup::pipeline_flags, D(SceneRayTracingSetup, pipeline_flags))
        .def(
            "link_program",
            &SceneRayTracingSetup::link_program,
            "module"_a,
            "additional_entry_points"_a = std::vector<std::string>{},
            D(SceneRayTracingSetup, link_program)
        )
        .def(
            "create_pipeline",
            &SceneRayTracingSetup::create_pipeline,
            "desc"_a,
            D(SceneRayTracingSetup, create_pipeline)
        )
        .def(
            "create_shader_table",
            &SceneRayTracingSetup::create_shader_table,
            "program"_a,
            "ray_gen_entry_points"_a,
            D(SceneRayTracingSetup, create_shader_table)
        );
}
