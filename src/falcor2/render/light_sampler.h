// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/core/cursor_writer.h"
#include "falcor2/core/reflected_object.h"
#include "falcor2/core/reflection.h"
#include "falcor2/render/emissive_geometry_system.h"
#include "falcor2/render/emissive_triangle_tree.h"
#include "falcor2/render/light_system.h"
#include "falcor2/utils/sampling/distribution_1d.h"

#include <sgl/device/fwd.h>

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace falcor {

/// Base class for renderer-owned scene light sampling strategies.
class FALCOR_API LightSampler : public ReflectedObject {
    FALCOR_REFLECTED_OBJECT(LightSampler, ReflectedObject)
public:
    FALCOR_STATIC_WRITE_TO_SHADER_CURSOR(LightSampler);

    LightSampler();
    virtual ~LightSampler() = default;

    /// Slang struct implementing ILightSampler.
    virtual std::string_view slang_type_name() const = 0;

    /// Globally unique generation identifying the current shader specialization state.
    uint64_t shader_generation() const { return m_shader_generation; }

    /// Slang source exporting sampler-specific link-time constants.
    virtual std::string shader_specialization_source() const { return {}; }

    /// Update strategy resources from an already updated scene.
    /// @return True if bound resources changed.
    virtual bool update(const Scene* scene, sgl::CommandEncoder* command_encoder = nullptr) = 0;

    /// Bind strategy data to its Slang struct.
    virtual void write_to_cursor(sgl::ShaderCursor cursor) const = 0;

    template<reflection::ClassReflector R>
    static void reflect(R&)
    {
    }

protected:
    /// Advance the shader generation after changing shader specialization state.
    void mark_shader_dirty();

private:
    uint64_t m_shader_generation;
};

/// Samples every active scene light source with uniform probability.
class FALCOR_API UniformLightSampler final : public LightSampler {
    FALCOR_REFLECTED_OBJECT(UniformLightSampler, LightSampler)
public:
    UniformLightSampler() = default;

    std::string_view slang_type_name() const override { return "UniformLightSampler"; }
    bool update(const Scene* scene, sgl::CommandEncoder* command_encoder = nullptr) override;
    void write_to_cursor(sgl::ShaderCursor cursor) const override;

    template<reflection::ClassReflector R>
    static void reflect(R&)
    {
    }
};

/// Samples finite and infinite emitters with equal probability, then power-weights within each domain.
class FALCOR_API PowerLightSampler final : public LightSampler {
    FALCOR_REFLECTED_OBJECT(PowerLightSampler, LightSampler)
public:
    PowerLightSampler() = default;

    std::string_view slang_type_name() const override { return "PowerLightSampler"; }
    bool update(const Scene* scene, sgl::CommandEncoder* command_encoder = nullptr) override;
    void write_to_cursor(sgl::ShaderCursor cursor) const override;

    /// Number of times the sampling distributions have been rebuilt.
    uint64_t rebuild_count() const { return m_rebuild_count; }

    template<reflection::ClassReflector R>
    static void reflect(R& r)
    {
        r.def_prop_ro("rebuild_count", &PowerLightSampler::rebuild_count, "Number of distribution rebuilds.");
    }

private:
    ref<const Scene> m_scene;
    LightGenerations m_light_generation_snapshot;
    EmissiveGeometryGenerations m_emissive_geometry_generation_snapshot;
    ref<AliasTable1D> m_analytic_light_distribution;
    ref<AliasTable1D> m_environment_light_distribution;
    ref<AliasTable1D> m_emissive_triangle_distribution;
    /// Finite analytic-light flux total.
    float m_analytic_light_power{0.f};
    /// Infinite-light directional radiance integral total.
    float m_environment_light_power{0.f};
    /// Finite emissive-triangle flux total.
    float m_emissive_triangle_power{0.f};
    uint64_t m_rebuild_count{0};
};

/// Samples component lights by power and emissive triangles through a spatial light hierarchy.
class FALCOR_API HierarchicalLightSampler final : public LightSampler {
    FALCOR_REFLECTED_OBJECT(HierarchicalLightSampler, LightSampler)
public:
    HierarchicalLightSampler() = default;

    std::string_view slang_type_name() const override { return "HierarchicalLightSampler"; }
    std::string shader_specialization_source() const override;
    bool update(const Scene* scene, sgl::CommandEncoder* command_encoder = nullptr) override;
    void write_to_cursor(sgl::ShaderCursor cursor) const override;

    EmissiveTriangleTreeSplitHeuristic split_heuristic() const { return m_build_options.split_heuristic; }
    void set_split_heuristic(EmissiveTriangleTreeSplitHeuristic value);

    uint32_t max_triangle_count_per_leaf() const { return m_build_options.max_triangle_count_per_leaf; }
    void set_max_triangle_count_per_leaf(uint32_t value);

    uint32_t bin_count() const { return m_build_options.bin_count; }
    void set_bin_count(uint32_t value);

    bool allow_refitting() const { return m_allow_refitting; }
    void set_allow_refitting(bool value) { m_allow_refitting = value; }

    shared::EmissiveTriangleTreeLeafSamplingMode leaf_sampling_mode() const { return m_leaf_sampling_mode; }
    void set_leaf_sampling_mode(shared::EmissiveTriangleTreeLeafSamplingMode value);

    uint64_t rebuild_count() const { return m_rebuild_count; }
    uint64_t refit_count() const { return m_refit_count; }
    uint32_t node_count() const { return m_emissive_triangle_tree ? m_emissive_triangle_tree->stats().node_count : 0; }
    uint32_t leaf_count() const { return m_emissive_triangle_tree ? m_emissive_triangle_tree->stats().leaf_count : 0; }
    uint32_t max_depth() const { return m_emissive_triangle_tree ? m_emissive_triangle_tree->stats().max_depth : 0; }
    uint32_t refit_dispatch_count() const
    {
        return m_emissive_triangle_tree ? m_emissive_triangle_tree->refit_dispatch_count() : 0;
    }
    uint64_t node_buffer_recreation_count() const
    {
        return m_emissive_triangle_tree ? m_emissive_triangle_tree->node_buffer_recreation_count() : 0;
    }

    template<reflection::ClassReflector R>
    static void reflect(R& r)
    {
        r //
            .def_prop_rw(
                "split_heuristic",
                &HierarchicalLightSampler::split_heuristic,
                &HierarchicalLightSampler::set_split_heuristic,
                "Hierarchy split heuristic."
            )
            .def_prop_rw(
                "max_triangle_count_per_leaf",
                &HierarchicalLightSampler::max_triangle_count_per_leaf,
                &HierarchicalLightSampler::set_max_triangle_count_per_leaf,
                "Maximum number of triangles stored in a leaf.",
                reflection::value_range(1u, 64u)
            )
            .def_prop_rw(
                "bin_count",
                &HierarchicalLightSampler::bin_count,
                &HierarchicalLightSampler::set_bin_count,
                "Number of bins used by the binned SAH builder.",
                reflection::value_range(2u, 128u)
            )
            .def_prop_rw(
                "allow_refitting",
                &HierarchicalLightSampler::allow_refitting,
                &HierarchicalLightSampler::set_allow_refitting,
                "Refit unchanged hierarchy layouts after geometry or flux edits."
            )
            .def_prop_rw(
                "leaf_sampling_mode",
                &HierarchicalLightSampler::leaf_sampling_mode,
                &HierarchicalLightSampler::set_leaf_sampling_mode,
                "Strategy used to select emissive triangles within a tree leaf."
            )
            .def_prop_ro(
                "rebuild_count",
                &HierarchicalLightSampler::rebuild_count,
                "Number of full hierarchy rebuilds."
            )
            .def_prop_ro("refit_count", &HierarchicalLightSampler::refit_count, "Number of hierarchy refits.")
            .def_prop_ro("node_count", &HierarchicalLightSampler::node_count, "Current tree node count.")
            .def_prop_ro("leaf_count", &HierarchicalLightSampler::leaf_count, "Current tree leaf count.")
            .def_prop_ro("max_depth", &HierarchicalLightSampler::max_depth, "Current tree maximum depth.")
            .def_prop_ro(
                "refit_dispatch_count",
                &HierarchicalLightSampler::refit_dispatch_count,
                "Number of GPU dispatches required by one refit of the current tree."
            )
            .def_prop_ro(
                "node_buffer_recreation_count",
                &HierarchicalLightSampler::node_buffer_recreation_count,
                "Number of node-buffer allocations performed by the current tree."
            );
    }

private:
    ref<const Scene> m_scene;
    LightGenerations m_light_generation_snapshot;
    EmissiveGeometryGenerations m_emissive_geometry_generation_snapshot;
    std::vector<shared::EmissiveTriangleID> m_active_triangle_ids_snapshot;
    ref<AliasTable1D> m_analytic_light_distribution;
    ref<AliasTable1D> m_environment_light_distribution;
    std::unique_ptr<EmissiveTriangleTree> m_emissive_triangle_tree;
    EmissiveTriangleTree::BuildOptions m_build_options;
    float m_analytic_light_power{0.f};
    float m_environment_light_power{0.f};
    float m_emissive_triangle_power{0.f};
    bool m_needs_rebuild{true};
    bool m_allow_refitting{true};
    shared::EmissiveTriangleTreeLeafSamplingMode m_leaf_sampling_mode{
        shared::EmissiveTriangleTreeLeafSamplingMode::uniform
    };
    uint64_t m_rebuild_count{0};
    uint64_t m_refit_count{0};
};

} // namespace falcor
