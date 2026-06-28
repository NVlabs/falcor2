// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/core/object.h"
#include "falcor2/denoisers/ngx_types.h"

#include <sgl/device/fwd.h>

namespace falcor::ngx {

class NGX;

class FALCOR_API DLSSRRFeature : public Object {
    FALCOR_OBJECT(DLSSRRFeature)

public:
    class Impl;

    /// Create a DLSS Ray Reconstruction feature from an initialized NGX context.
    DLSSRRFeature(ref<NGX> ngx, const DLSSRRFeatureDesc& desc);

    /// Release the backend-specific feature implementation.
    ~DLSSRRFeature();

    /// Return the parent NGX context kept alive by this feature.
    NGX* ngx() const { return m_ngx; }

    /// Return the SGL device owned by the parent NGX context.
    sgl::Device* device() const;

    /// Return the feature creation descriptor.
    const DLSSRRFeatureDesc& desc() const { return m_desc; }

    /// Submit DLSS Ray Reconstruction evaluation on a short-lived command buffer.
    void evaluate(const DLSSRREvaluateDesc& desc);

    /// Record a DLSS Ray Reconstruction evaluation into an existing command encoder.
    void evaluate(sgl::CommandEncoder* command_encoder, const DLSSRREvaluateDesc& desc);

    /// Return and clear the NGX result from the most recently executed evaluation callback, or Result::none if none has
    /// executed.
    Result last_evaluate_result();

private:
    void throw_if_last_evaluate_failed() const;

    ref<NGX> m_ngx;
    DLSSRRFeatureDesc m_desc;
    Impl* m_impl = nullptr;
};

} // namespace falcor::ngx
