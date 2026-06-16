// SPDX-License-Identifier: Apache-2.0

#include "nanobind.h"

#include "falcor2/render/transform.h"

namespace nb = nanobind;
using namespace nb::literals;

FALCOR_PY_EXPORT(render_transform)
{
    using namespace falcor;

    nb::class_<Transform> transform(m, "Transform", D(Transform));
    transform //
        .def(nb::init<>(), D(Transform, Transform))
        .def(nb::init<const Transform&>(), "other"_a, D(Transform, Transform, 2))
        .def(
            nb::init<const float3&, const quatf&, const float3&>(),
            "translation"_a,
            "rotation"_a,
            "scale"_a,
            D(Transform, Transform, 3)
        )
        .def(nb::init<const float4x4&>(), "matrix"_a, D(Transform, Transform, 4))
        .def("reset", &Transform::reset, D(Transform, reset))
        .DEF_PROP_RW(Transform, composition_order)
        .DEF_PROP_RW(Transform, translation)
        .DEF_PROP_RW(Transform, rotation)
        .DEF_PROP_RW(Transform, rotation_xyz)
        .DEF_PROP_RW(Transform, scale)
        .DEF_PROP_RO(Transform, matrix, nb::rv_policy::copy)
        .def("is_identity", &Transform::is_identity, D(Transform, is_identity))
        .def("translate", &Transform::translate, "delta"_a, nb::rv_policy::reference_internal, D(Transform, translate))
        .def("rotate", &Transform::rotate, "delta"_a, nb::rv_policy::reference_internal, D(Transform, rotate))
        .def("scale_by", &Transform::scale_by, "factor"_a, nb::rv_policy::reference_internal, D(Transform, scale_by))
        .def("__repr__", &Transform::to_string, D(Transform, to_string));

    nb::sgl_enum<Transform::CompositionOrder>(transform, "CompositionOrder");
}
