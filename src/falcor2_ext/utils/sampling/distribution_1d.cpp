// SPDX-License-Identifier: Apache-2.0

#include "nanobind.h"

#include "falcor2/utils/sampling/distribution_1d.h"

#include <sgl/device/device.h>
#include <sgl/device/resource.h>

namespace nb = nanobind;
using namespace nb::literals;

FALCOR_PY_EXPORT(utils_sampling_distribution_1d)
{
    using namespace falcor;

    nb::class_<DiscreteDistribution1D>(m, "DiscreteDistribution1D", D(DiscreteDistribution1D))
        .def(
            "__init__",
            [](DiscreteDistribution1D* self,
               sgl::Device* device,
               nb::ndarray<float, nb::shape<-1>, nb::device::cpu> func,
               std::string_view label)
            {
                new (self) DiscreteDistribution1D(device, std::span<float>(func.data(), func.shape(0)), label);
            },
            "device"_a,
            "func"_a,
            "label"_a = "",
            D(DiscreteDistribution1D, DiscreteDistribution1D)
        )
        .DEF_PROP_RO(DiscreteDistribution1D, size)
        .DEF_PROP_RO(DiscreteDistribution1D, pdf)
        .DEF_PROP_RO(DiscreteDistribution1D, cdf)
        .DEF_PROP_RO(DiscreteDistribution1D, pdf_buffer)
        .DEF_PROP_RO(DiscreteDistribution1D, cdf_buffer)
        .def(
            "get_this",
            [](DiscreteDistribution1D* self)
            {
                nb::dict result = nb::dict();
                result["_type"] = "DiscreteDistribution1D";
                result["size"] = self->size();
                result["pdf_buffer"] = self->pdf_buffer();
                result["cdf_buffer"] = self->cdf_buffer();
                return result;
            }
        );

    nb::class_<AliasTable1D>(m, "AliasTable1D", D(AliasTable1D))
        .def(
            "__init__",
            [](AliasTable1D* self,
               sgl::Device* device,
               nb::ndarray<float, nb::shape<-1>, nb::device::cpu> func,
               std::string_view label)
            {
                new (self) AliasTable1D(device, std::span<float>(func.data(), func.shape(0)), label);
            },
            "device"_a,
            "func"_a,
            "label"_a = "",
            D(AliasTable1D, AliasTable1D)
        )
        .DEF_PROP_RO(AliasTable1D, size)
        .DEF_PROP_RO(AliasTable1D, pdf)
        .DEF_PROP_RO(AliasTable1D, alias_table)
        .DEF_PROP_RO(AliasTable1D, pdf_buffer)
        .DEF_PROP_RO(AliasTable1D, alias_table_buffer)
        .def(
            "get_this",
            [](AliasTable1D* self)
            {
                nb::dict result = nb::dict();
                result["_type"] = "AliasTable1D";
                result["size"] = self->size();
                result["pdf_buffer"] = self->pdf_buffer();
                result["alias_table_buffer"] = self->alias_table_buffer();
                return result;
            }
        );
}
