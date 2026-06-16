// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/core/config.h"

#if FALCOR_ENABLE_NGX

#include "falcor2/denoisers/dlss_g_feature.h"
#include "falcor2/denoisers/ngx.h"

#include <sgl/device/fwd.h>

#include <atomic>
#include <memory>

#include <nvsdk_ngx.h>
#include <nvsdk_ngx_params_dlssg.h>

namespace falcor::ngx {

struct DLSSGEvaluateResources {
    ref<sgl::Texture> backbuffer;
    ref<sgl::Texture> depth;
    ref<sgl::Texture> motion_vectors;
    ref<sgl::Texture> output_interpolated_frame;
    DLSSGCameraDesc camera;
};

class DLSSGFeature::Impl {
public:
    /// Store the parent NGX context and immutable creation descriptor.
    Impl(ref<NGX> ngx, DLSSGFeatureDesc desc);

    /// Release backend-specific feature state.
    virtual ~Impl();

    /// Record backend-specific NGX evaluation work into an existing command encoder.
    virtual void
    record_evaluate(ref<DLSSGFeature> owner, sgl::CommandEncoder* command_encoder, const DLSSGEvaluateDesc& desc)
        = 0;

    /// Return the result from the most recently executed evaluation callback.
    NVSDK_NGX_Result last_evaluate_result() const;

    /// Return and clear the result from the most recently executed evaluation callback.
    NVSDK_NGX_Result get_last_evaluate_result();

protected:
    /// Validate feature dimensions and capability state before creating an NGX feature.
    void validate_feature_desc() const;

    /// Build the common DLSS Frame Generation feature creation parameters.
    NVSDK_NGX_DLSSG_Create_Params make_create_params(uint32_t native_backbuffer_format) const;

    /// Build the optional DLSS Frame Generation camera/evaluation parameters.
    NVSDK_NGX_DLSSG_Opt_Eval_Params make_opt_eval_params(const DLSSGEvaluateResources& resources) const;

    /// Retain all resources referenced by an evaluation descriptor until the callback retires.
    static DLSSGEvaluateResources retain_evaluate_resources(const DLSSGEvaluateDesc& desc);

    /// Validate that an evaluation descriptor is complete and matches the feature/device.
    void validate_evaluate_desc(const DLSSGEvaluateDesc& desc) const;

    /// Transition resources into the states expected by NGX evaluation.
    static void
    prepare_evaluate_resources(sgl::CommandEncoder* command_encoder, const DLSSGEvaluateResources& resources);

    uint32_t effective_render_width() const;
    uint32_t effective_render_height() const;

    void reset_last_evaluate_result();
    void set_last_evaluate_result(NVSDK_NGX_Result result);

    ref<NGX> m_ngx;
    DLSSGFeatureDesc m_desc;
    std::atomic<NVSDK_NGX_Result> m_last_evaluate_result{static_cast<NVSDK_NGX_Result>(0)};
};

std::unique_ptr<DLSSGFeature::Impl> make_dlss_g_feature_impl(ref<NGX> ngx, const DLSSGFeatureDesc& desc);

} // namespace falcor::ngx

#endif // FALCOR_ENABLE_NGX
