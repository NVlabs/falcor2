// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/core/cursor_writer.h"
#include "falcor2/core/reflected_object.h"
#include "falcor2/core/reflection.h"

#include <sgl/device/shader_cursor.h>

#include <cstdint>
#include <string_view>

namespace falcor {

/// Renderer-owned normalized, nonnegative reconstruction filter, sampled by its GPU implementation.
/// Describes a center-relative distribution independent of renderer-controlled camera jitter.
class FALCOR_API ReconstructionFilter : public ReflectedObject {
    FALCOR_REFLECTED_OBJECT(ReconstructionFilter, ReflectedObject)
public:
    FALCOR_STATIC_WRITE_TO_SHADER_CURSOR(ReconstructionFilter);

    ReconstructionFilter();

    /// Concrete Slang struct implementing IReconstructionFilter, used for argument marshalling.
    virtual std::string_view slang_type_name() const = 0;
    /// Globally unique revision for accumulation invalidation, independent of parameter uploads.
    /// Advances after reflected configuration writes or notify_changed().
    uint64_t generation() const { return m_generation; }
    /// Advance generation after direct native configuration writes.
    /// Reflected configuration writes advance it automatically.
    void notify_changed();
    /// Validate and upload parameters; rejects invalid or unrepresentable values.
    virtual void write_to_cursor(sgl::ShaderCursor cursor) const = 0;
    /// Identify the shader type for SlangPy caching, independently of parameter edits.
    static void write_slangpy_signature(sgl::SignatureBuffer& signature, const ReconstructionFilter* value);

    template<reflection::ClassReflector R>
    static void reflect(R&)
    {
    }

private:
    uint64_t m_generation;
};

/// Uniform sampling of a square centered at the origin.
/// Each camera sample consumes two random values, including when width is zero.
class FALCOR_API BoxReconstructionFilter : public ReconstructionFilter {
    FALCOR_REFLECTED_OBJECT(BoxReconstructionFilter, ReconstructionFilter)
public:
    explicit BoxReconstructionFilter(float width = 1.f);

    float width;

    std::string_view slang_type_name() const override { return "BoxReconstructionFilter"; }
    void write_to_cursor(sgl::ShaderCursor cursor) const override;

    template<reflection::ClassReflector R>
    static void reflect(R& r)
    {
        r.def_rw(
            "width",
            &BoxReconstructionFilter::width,
            "Full square width in input pixels; zero samples the center.",
            reflection::on_change(&BoxReconstructionFilter::notify_changed)
        );
    }
};

/// Product of two Gaussian kernels shifted to reach zero at their square support boundary.
/// Uses the normalized PBRT/Mitsuba shape. Sampling uses a truncated Gaussian proposal
/// and returns a varying importance weight, rather than PBRT's tabulated approximation.
class FALCOR_API GaussianReconstructionFilter : public ReconstructionFilter {
    FALCOR_REFLECTED_OBJECT(GaussianReconstructionFilter, ReconstructionFilter)
public:
    GaussianReconstructionFilter(float sigma = 0.5f, float radius = 1.5f);

    float sigma;
    float radius;

    std::string_view slang_type_name() const override { return "GaussianReconstructionFilter"; }
    void write_to_cursor(sgl::ShaderCursor cursor) const override;

    template<reflection::ClassReflector R>
    static void reflect(R& r)
    {
        r.def_rw(
             "sigma",
             &GaussianReconstructionFilter::sigma,
             "Gaussian standard deviation in input pixels.",
             reflection::on_change(&GaussianReconstructionFilter::notify_changed)
        )
            .def_rw(
                "radius",
                &GaussianReconstructionFilter::radius,
                "Per-axis cutoff radius in input pixels.",
                reflection::on_change(&GaussianReconstructionFilter::notify_changed)
            );
    }
};

} // namespace falcor
