// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <sgl/device/command.h>

#include <algorithm>
#include <memory>
#include <span>
#include <typeinfo>
#include <vector>

namespace falcor {

enum class GeometryUpdatePhase {
    vertex_cache,
};

/// Shared algorithm execution for one output format. Preparation collects work before the
/// render collection transitions its output buffers; dispatch consumes that work.
class GeometryUpdater {
public:
    explicit GeometryUpdater(GeometryUpdatePhase phase)
        : m_phase(phase)
    {
    }
    virtual ~GeometryUpdater() = default;

    GeometryUpdatePhase phase() const { return m_phase; }
    virtual void prepare(sgl::CommandEncoder* encoder, double time, bool initialize) = 0;
    virtual bool has_work() const = 0;
    virtual void dispatch(sgl::CommandEncoder* encoder) = 0;

private:
    GeometryUpdatePhase m_phase;
};

/// Owns one instance per concrete updater type. Lookups occur during attachment;
/// heap ownership keeps borrowed updater addresses stable as the registry grows.
class GeometryUpdaterRegistry {
public:
    template<typename Updater>
    Updater& get_or_create()
    {
        for (const auto& updater : m_updaters)
            if (typeid(*updater) == typeid(Updater))
                return static_cast<Updater&>(*updater);

        auto updater = std::make_unique<Updater>();
        auto& result = *updater;
        // Execution order is independent of the order in which deformers attach.
        const auto position = std::upper_bound(
            m_updaters.begin(),
            m_updaters.end(),
            updater->phase(),
            [](GeometryUpdatePhase phase, const auto& existing)
            {
                return phase < existing->phase();
            }
        );
        m_updaters.insert(position, std::move(updater));
        return result;
    }

    std::span<const std::unique_ptr<GeometryUpdater>> updaters() const { return m_updaters; }

private:
    std::vector<std::unique_ptr<GeometryUpdater>> m_updaters;
};

} // namespace falcor
