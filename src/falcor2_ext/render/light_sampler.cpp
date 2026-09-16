// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "nanobind.h"
#include "nanobind_reflector.h"

#include "falcor2/render/light_sampler.h"
#include "falcor2/render/scene.h"

FALCOR_PY_EXPORT(render_light_sampler)
{
    using namespace falcor;

    nb::sgl_enum<EmissiveTriangleTreeSplitHeuristic>(m, "EmissiveTriangleTreeSplitHeuristic");
    nb::sgl_enum<shared::EmissiveTriangleTreeLeafSamplingMode>(m, "EmissiveTriangleTreeLeafSamplingMode");

    nb::class_<LightSampler, ReflectedObject> sampler(m, "LightSampler", D(LightSampler));
    reflection::bind<LightSampler>(sampler);
    sampler //
        .def_prop_ro("slang_type_name", &LightSampler::slang_type_name, D(LightSampler, slang_type_name))
        .def_prop_ro("shader_generation", &LightSampler::shader_generation, D(LightSampler, shader_generation))
        .def_prop_ro(
            "shader_specialization_source",
            &LightSampler::shader_specialization_source,
            "Slang source exporting sampler-specific link-time constants."
        )
        .def(
            "update",
            &LightSampler::update,
            "scene"_a,
            "command_encoder"_a.none() = nb::none(),
            D(LightSampler, update)
        );

    nb::class_<UniformLightSampler, LightSampler> uniform_sampler(m, "UniformLightSampler", D(UniformLightSampler));
    reflection::bind<UniformLightSampler>(uniform_sampler);
    uniform_sampler.def(nb::init<>());

    nb::class_<PowerLightSampler, LightSampler> power_sampler(m, "PowerLightSampler", D(PowerLightSampler));
    reflection::bind<PowerLightSampler>(power_sampler);
    power_sampler.def(nb::init<>());

    nb::class_<HierarchicalLightSampler, LightSampler> hierarchical_sampler(m, "HierarchicalLightSampler");
    reflection::bind<HierarchicalLightSampler>(hierarchical_sampler);
    hierarchical_sampler.def(nb::init<>());
}
