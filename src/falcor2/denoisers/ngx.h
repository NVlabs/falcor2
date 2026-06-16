// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/core/object.h"
#include "falcor2/denoisers/ngx_types.h"

#include <sgl/device/fwd.h>

#include <cstdint>

namespace falcor::ngx {

class DLSSRRFeature;
class DLSSSRFeature;
class DLSSGFeature;

class FALCOR_API NGX : public Object {
    FALCOR_OBJECT(NGX)

public:
    class Impl;

    /// Return the cached NGX context for the supplied SGL device, creating it if needed.
    static ref<NGX> get(sgl::Device* device);

    /// Shut down the NGX context owned by this object.
    ~NGX();

    /// Return the SGL device used to initialize NGX.
    sgl::Device* device() const { return m_device; }

    /// Return NGX capability information that can be queried without creating a feature.
    const NGXInfo& info() const;

    /// Query NGX for the recommended DLSS-D render size for an output size and quality mode.
    DLSSDOptimalSettings get_dlssd_optimal_settings(
        uint32_t target_width,
        uint32_t target_height,
        QualityMode quality = QualityMode::quality
    ) const;

    /// Query NGX for the recommended DLSS Super Resolution render size for an output size and quality mode.
    DLSSOptimalSettings get_dlss_optimal_settings(
        uint32_t target_width,
        uint32_t target_height,
        QualityMode quality = QualityMode::quality
    ) const;

    /// Create a DLSS Ray Reconstruction feature owned by this NGX context.
    ref<DLSSRRFeature> create_dlss_rr_feature(const DLSSRRFeatureDesc& desc);

    /// Create a DLSS Super Resolution feature owned by this NGX context.
    ref<DLSSSRFeature> create_dlss_sr_feature(const DLSSSRFeatureDesc& desc);

    /// Create a DLSS Frame Generation feature owned by this NGX context.
    ref<DLSSGFeature> create_dlss_g_feature(const DLSSGFeatureDesc& desc);

private:
    friend Impl* get_ngx_impl(NGX* ngx);

    /// Create and initialize an NGX context for the supplied SGL device.
    explicit NGX(ref<sgl::Device> device);

    ref<sgl::Device> m_device;
    Impl* m_impl = nullptr;
};

/// Query Vulkan extensions that NGX requires before an SGL Vulkan device is created.
FALCOR_API NGXInfo get_vulkan_pre_device_info();

} // namespace falcor::ngx
