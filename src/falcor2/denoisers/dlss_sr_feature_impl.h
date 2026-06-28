// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/core/config.h"

#if FALCOR_ENABLE_NGX

#include "falcor2/denoisers/dlss_sr_feature.h"
#include "falcor2/denoisers/ngx.h"

#include <sgl/device/fwd.h>

#include <atomic>
#include <memory>

#include <nvsdk_ngx.h>
#include <nvsdk_ngx_params.h>

namespace falcor::ngx {

struct DLSSSREvaluateResources {
    ref<sgl::Texture> color;
    ref<sgl::Texture> depth;
    ref<sgl::Texture> motion_vectors;
    ref<sgl::Texture> output;
    ref<sgl::Texture> exposure;

    uint32_t render_subrect_width{0};
    uint32_t render_subrect_height{0};

    float jitter_offset_x{0.f};
    float jitter_offset_y{0.f};
    bool reset{false};
    float motion_vector_scale_x{1.f};
    float motion_vector_scale_y{1.f};
    float pre_exposure{1.f};
    float exposure_scale{1.f};
};

class DLSSSRFeature::Impl {
public:
    /// Store the parent NGX context and immutable creation descriptor.
    Impl(ref<NGX> ngx, DLSSSRFeatureDesc desc);

    /// Release backend-specific feature state.
    virtual ~Impl();

    /// Record backend-specific NGX evaluation work into an existing command encoder.
    virtual void
    record_evaluate(ref<DLSSSRFeature> owner, sgl::CommandEncoder* command_encoder, const DLSSSREvaluateDesc& desc)
        = 0;

    /// Return the result from the most recently executed evaluation callback.
    NVSDK_NGX_Result last_evaluate_result() const;

    /// Return and clear the result from the most recently executed evaluation callback.
    NVSDK_NGX_Result get_last_evaluate_result();

protected:
    /// Validate feature dimensions and capability state before creating an NGX feature.
    void validate_feature_desc() const;

    /// Build the common DLSS Super Resolution feature creation parameters.
    NVSDK_NGX_DLSS_Create_Params make_create_params() const;

    /// Retain all resources referenced by an evaluation descriptor until the callback retires.
    static DLSSSREvaluateResources retain_evaluate_resources(const DLSSSREvaluateDesc& desc);

    /// Validate that an evaluation descriptor is complete and matches the feature/device.
    void validate_evaluate_desc(const DLSSSREvaluateDesc& desc) const;

    /// Transition resources into the states expected by NGX evaluation.
    static void
    prepare_evaluate_resources(sgl::CommandEncoder* command_encoder, const DLSSSREvaluateResources& resources);

    void reset_last_evaluate_result();
    void set_last_evaluate_result(NVSDK_NGX_Result result);

    uint32_t evaluate_render_subrect_width(const DLSSSREvaluateResources& resources) const;
    uint32_t evaluate_render_subrect_height(const DLSSSREvaluateResources& resources) const;

    ref<NGX> m_ngx;
    DLSSSRFeatureDesc m_desc;
    std::atomic<NVSDK_NGX_Result> m_last_evaluate_result{static_cast<NVSDK_NGX_Result>(0)};
};

std::unique_ptr<DLSSSRFeature::Impl> make_dlss_sr_feature_impl(ref<NGX> ngx, const DLSSSRFeatureDesc& desc);

} // namespace falcor::ngx

#endif // FALCOR_ENABLE_NGX
