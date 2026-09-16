// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "light_sampler.h"

#include "falcor2/core/error.h"
#include "falcor2/render/scene.h"

#include <sgl/device/cursor_utils.h>
#include <sgl/device/shader_cursor.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>
#include <span>

namespace falcor {
namespace {

uint64_t next_light_sampler_shader_generation()
{
    static std::atomic<uint64_t> id{0};
    return ++id;
}

float sum_weights(std::span<const float> weights)
{
    double total = 0.0;
    for (float weight : weights) {
        if (!std::isfinite(weight) || weight < 0.f)
            FALCOR_THROW("selection weights must be finite and non-negative");
        total += weight;
    }
    if (total > std::numeric_limits<float>::max())
        FALCOR_THROW("selection weight total exceeds the finite float range");
    return static_cast<float>(total);
}

ref<AliasTable1D> create_distribution(sgl::Device* device, std::span<const float> weights, std::string_view label)
{
    if (weights.empty())
        return nullptr;
    return ref<AliasTable1D>(new AliasTable1D(device, weights, label));
}

} // namespace

LightSampler::LightSampler()
    : m_shader_generation(next_light_sampler_shader_generation())
{
}

void LightSampler::mark_shader_dirty()
{
    m_shader_generation = next_light_sampler_shader_generation();
}

bool UniformLightSampler::update(const Scene* scene, sgl::CommandEncoder* command_encoder)
{
    FALCOR_UNUSED(command_encoder);
    FALCOR_CHECK_NOT_NULL(scene);
    return false;
}

void UniformLightSampler::write_to_cursor(sgl::ShaderCursor cursor) const
{
    FALCOR_UNUSED(cursor);
}

bool PowerLightSampler::update(const Scene* scene, sgl::CommandEncoder* command_encoder)
{
    FALCOR_UNUSED(command_encoder);
    FALCOR_CHECK_NOT_NULL(scene);

    const LightSystem& light_system = *scene->_light_system();
    const EmissiveGeometrySystem& emissive_geometry = *scene->_emissive_geometry_system();
    const bool scene_changed = m_scene.get() != scene;
    const LightGenerations& light_generations = light_system.generations();
    const EmissiveGeometryGenerations& emissive_geometry_generations = emissive_geometry.generations();
    // Power distributions depend on source indexing and scalar selection weights, not spatial data.
    const bool lights_changed = light_generations.topology != m_light_generation_snapshot.topology
        || light_generations.selection_weights != m_light_generation_snapshot.selection_weights;
    const bool emissive_geometry_changed
        = emissive_geometry_generations.topology != m_emissive_geometry_generation_snapshot.topology
        || emissive_geometry_generations.flux != m_emissive_geometry_generation_snapshot.flux;
    if (!scene_changed && !lights_changed && !emissive_geometry_changed)
        return false;

    m_scene = ref<const Scene>(scene);
    m_light_generation_snapshot = light_generations;
    m_emissive_geometry_generation_snapshot = emissive_geometry_generations;

    const std::span<const float> light_powers = light_system.light_powers();
    FALCOR_CHECK_EQ(light_powers.size(), light_system.light_count());
    const size_t analytic_light_count = light_system.analytic_light_count();
    const size_t environment_light_count = light_system.environment_light_count();
    const std::span<const float> analytic_light_powers = light_powers.first(analytic_light_count);
    const std::span<const float> environment_light_powers
        = light_powers.subspan(analytic_light_count, environment_light_count);
    const std::span<const float> emissive_triangle_powers = emissive_geometry.active_triangle_flux();

    // Analytic lights and emissive triangles are finite emitters with compatible flux weights.
    // Environment lights are infinite emitters whose directional-radiance integrals form a separate domain.
    m_analytic_light_power = sum_weights(analytic_light_powers);
    m_environment_light_power = sum_weights(environment_light_powers);
    m_emissive_triangle_power = sum_weights(emissive_triangle_powers);

    m_analytic_light_distribution
        = create_distribution(scene->device(), analytic_light_powers, "PowerLightSampler::analytic_light_distribution");
    m_environment_light_distribution = create_distribution(
        scene->device(),
        environment_light_powers,
        "PowerLightSampler::environment_light_distribution"
    );
    m_emissive_triangle_distribution = create_distribution(
        scene->device(),
        emissive_triangle_powers,
        "PowerLightSampler::emissive_triangle_distribution"
    );

    ++m_rebuild_count;
    return true;
}

void PowerLightSampler::write_to_cursor(sgl::ShaderCursor cursor) const
{
    FALCOR_CHECK(m_scene, "PowerLightSampler::update() must be called before binding.");
    if (m_analytic_light_distribution)
        cursor["analytic_light_distribution"] = *m_analytic_light_distribution;
    if (m_environment_light_distribution)
        cursor["environment_light_distribution"] = *m_environment_light_distribution;
    if (m_emissive_triangle_distribution)
        cursor["emissive_triangle_distribution"] = *m_emissive_triangle_distribution;
    cursor["analytic_light_power"] = m_analytic_light_power;
    cursor["environment_light_power"] = m_environment_light_power;
    cursor["emissive_triangle_power"] = m_emissive_triangle_power;
}

void HierarchicalLightSampler::set_split_heuristic(EmissiveTriangleTreeSplitHeuristic value)
{
    if (m_build_options.split_heuristic != value) {
        m_build_options.split_heuristic = value;
        m_needs_rebuild = true;
    }
}

void HierarchicalLightSampler::set_max_triangle_count_per_leaf(uint32_t value)
{
    value = std::clamp(value, 1u, shared::PackedEmissiveTriangleTreeNode::MAX_TRIANGLE_COUNT);
    if (m_build_options.max_triangle_count_per_leaf != value) {
        m_build_options.max_triangle_count_per_leaf = value;
        m_needs_rebuild = true;
    }
}

void HierarchicalLightSampler::set_bin_count(uint32_t value)
{
    value = std::clamp(value, 2u, 128u);
    if (m_build_options.bin_count != value) {
        m_build_options.bin_count = value;
        m_needs_rebuild = true;
    }
}

void HierarchicalLightSampler::set_leaf_sampling_mode(shared::EmissiveTriangleTreeLeafSamplingMode value)
{
    if (m_leaf_sampling_mode != value) {
        m_leaf_sampling_mode = value;
        mark_shader_dirty();
    }
}

std::string HierarchicalLightSampler::shader_specialization_source() const
{
    return "export static const uint EMISSIVE_TRIANGLE_TREE_LEAF_SAMPLING_MODE = "
        + std::to_string(uint32_t(m_leaf_sampling_mode)) + ";\n";
}

bool HierarchicalLightSampler::update(const Scene* scene, sgl::CommandEncoder* command_encoder)
{
    FALCOR_CHECK_NOT_NULL(scene);

    const LightSystem& light_system = *scene->_light_system();
    const EmissiveGeometrySystem& emissive_geometry = *scene->_emissive_geometry_system();
    const LightGenerations& light_generations = light_system.generations();
    const EmissiveGeometryGenerations& emissive_geometry_generations = emissive_geometry.generations();
    const bool scene_changed = m_scene.get() != scene;
    const bool lights_changed = scene_changed || light_generations.topology != m_light_generation_snapshot.topology
        || light_generations.selection_weights != m_light_generation_snapshot.selection_weights;
    const bool emissive_topology_generation_changed
        = scene_changed || emissive_geometry_generations.topology != m_emissive_geometry_generation_snapshot.topology;
    bool emissive_topology_changed = scene_changed;
    if (emissive_topology_generation_changed) {
        const std::span<const shared::EmissiveTriangleID> active_triangle_ids = emissive_geometry.active_triangle_ids();
        emissive_topology_changed = scene_changed
            || !std::equal(active_triangle_ids.begin(),
                           active_triangle_ids.end(),
                           m_active_triangle_ids_snapshot.begin(),
                           m_active_triangle_ids_snapshot.end());
        m_active_triangle_ids_snapshot.assign(active_triangle_ids.begin(), active_triangle_ids.end());
    }
    const bool emissive_attributes_changed = scene_changed
        || emissive_geometry_generations.geometry != m_emissive_geometry_generation_snapshot.geometry
        || emissive_geometry_generations.flux != m_emissive_geometry_generation_snapshot.flux;
    const bool emissive_power_changed = scene_changed || emissive_topology_generation_changed
        || emissive_geometry_generations.flux != m_emissive_geometry_generation_snapshot.flux;

    bool changed = false;
    if (scene_changed) {
        m_scene = ref<const Scene>(scene);
        m_emissive_triangle_tree = std::make_unique<EmissiveTriangleTree>(scene->device());
        m_needs_rebuild = true;
    }

    if (lights_changed) {
        const std::span<const float> light_powers = light_system.light_powers();
        FALCOR_CHECK_EQ(light_powers.size(), light_system.light_count());
        const size_t analytic_light_count = light_system.analytic_light_count();
        const size_t environment_light_count = light_system.environment_light_count();
        const std::span<const float> analytic_light_powers = light_powers.first(analytic_light_count);
        const std::span<const float> environment_light_powers
            = light_powers.subspan(analytic_light_count, environment_light_count);

        m_analytic_light_power = sum_weights(analytic_light_powers);
        m_environment_light_power = sum_weights(environment_light_powers);
        m_analytic_light_distribution = create_distribution(
            scene->device(),
            analytic_light_powers,
            "HierarchicalLightSampler::analytic_light_distribution"
        );
        m_environment_light_distribution = create_distribution(
            scene->device(),
            environment_light_powers,
            "HierarchicalLightSampler::environment_light_distribution"
        );
        changed = true;
    }

    if (emissive_power_changed) {
        m_emissive_triangle_power = sum_weights(emissive_geometry.active_triangle_flux());
        changed = true;
    }

    FALCOR_CHECK(
        m_emissive_triangle_tree,
        "HierarchicalLightSampler failed to create its EmissiveTriangleTree component."
    );
    if (m_needs_rebuild || emissive_topology_changed || (emissive_attributes_changed && !m_allow_refitting)) {
        m_emissive_triangle_tree->build(emissive_geometry, m_build_options);
        m_needs_rebuild = false;
        ++m_rebuild_count;
        changed = true;
    } else if (emissive_attributes_changed) {
        m_emissive_triangle_tree->refit(emissive_geometry, command_encoder);
        ++m_refit_count;
        changed = true;
    }

    m_light_generation_snapshot = light_generations;
    m_emissive_geometry_generation_snapshot = emissive_geometry_generations;
    return changed;
}

void HierarchicalLightSampler::write_to_cursor(sgl::ShaderCursor cursor) const
{
    FALCOR_CHECK(
        m_scene && m_emissive_triangle_tree && m_rebuild_count > 0,
        "HierarchicalLightSampler::update() must be called before binding."
    );
    if (m_analytic_light_distribution)
        cursor["analytic_light_distribution"] = *m_analytic_light_distribution;
    if (m_environment_light_distribution)
        cursor["environment_light_distribution"] = *m_environment_light_distribution;
    cursor["analytic_light_power"] = m_analytic_light_power;
    cursor["environment_light_power"] = m_environment_light_power;
    cursor["emissive_triangle_power"] = m_emissive_triangle_power;

    sgl::ShaderCursor emissive_triangle_tree_cursor = cursor["emissive_triangle_tree"];
    m_emissive_triangle_tree->write_to_cursor(emissive_triangle_tree_cursor);
}

FALCOR_STATIC_ONCE(reflection::register_type<LightSampler>());
FALCOR_STATIC_ONCE(reflection::register_type<UniformLightSampler>());
FALCOR_STATIC_ONCE(reflection::register_type<PowerLightSampler>());
FALCOR_STATIC_ONCE(reflection::register_type<HierarchicalLightSampler>());
FALCOR_STATIC_ONCE(sgl::cursor_utils::register_cursor_writer<LightSampler>());

} // namespace falcor
