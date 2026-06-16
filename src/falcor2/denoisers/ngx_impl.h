// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/core/config.h"

#if FALCOR_ENABLE_NGX

#include "falcor2/denoisers/ngx.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

#include <nvsdk_ngx.h>
#include <nvsdk_ngx_defs_dlssd.h>

namespace falcor::ngx {

/// Convert common NGX result codes to readable diagnostic strings.
std::string ngx_result_to_string(int32_t raw_result);

/// Convert the public Falcor2 quality mode to the NGX quality enum.
NVSDK_NGX_PerfQuality_Value to_ngx_perf_quality(QualityMode quality);

/// Convert an NGX result code to the public Falcor2 result enum.
Result to_result(NVSDK_NGX_Result result);

/// Query Vulkan instance/device extensions required before creating a Vulkan NGX context.
void query_vulkan_required_extensions(NGXInfo& info);

class NGX::Impl {
public:
    /// Create the backend-specific implementation for the device type.
    static std::unique_ptr<Impl> create(ref<sgl::Device> device);

    /// Store the parent device for use by backend-specific implementations.
    explicit Impl(ref<sgl::Device> device);

    /// Release backend-neutral implementation state.
    virtual ~Impl();

    /// Return the SGL device that owns the native graphics/CUDA context.
    sgl::Device* device() const { return m_device; }

    /// Return capability information populated during backend initialization.
    const NGXInfo& info() const { return m_info; }

    /// Return the backend-native NGX device handle.
    void* ngx_device() const { return m_ngx_device; }

    /// Validate support and query backend-specific DLSS-D optimal settings.
    DLSSDOptimalSettings
    get_dlssd_optimal_settings(uint32_t target_width, uint32_t target_height, QualityMode quality) const;

    /// Validate support and query backend-specific DLSS Super Resolution optimal settings.
    DLSSOptimalSettings
    get_dlss_optimal_settings(uint32_t target_width, uint32_t target_height, QualityMode quality) const;

protected:
    /// NVIDIA-assigned application ID used by all NGX backend initializers.
    static constexpr uint64_t k_application_id = 231313132ull;

    /// Convert a filesystem path to the wide string form expected by NGX.
    static std::wstring to_wstring_path(const std::filesystem::path& path);

    /// Return the writable app-data directory used by NGX.
    static std::filesystem::path app_data_path();

    /// Build shared NGX feature metadata, including the runtime search path list.
    static NVSDK_NGX_FeatureCommonInfo make_feature_common_info(const wchar_t** path_list);

    /// Query DLSS-D optimal settings from a valid NGX capability parameter block.
    static DLSSDOptimalSettings query_dlssd_optimal_settings_from_params(
        NVSDK_NGX_Parameter* capability_params,
        uint32_t target_width,
        uint32_t target_height,
        QualityMode quality
    );

    /// Query DLSS Super Resolution optimal settings from a valid NGX capability parameter block.
    static DLSSOptimalSettings query_dlss_optimal_settings_from_params(
        NVSDK_NGX_Parameter* capability_params,
        uint32_t target_width,
        uint32_t target_height,
        QualityMode quality
    );

    /// Read DLSSD capability values from an NGX parameter block.
    void read_dlssd_capability_params(NVSDK_NGX_Parameter* capability_params);

    /// Read DLSS Super Resolution capability values from an NGX parameter block.
    void read_dlss_capability_params(NVSDK_NGX_Parameter* capability_params);

    /// Read DLSS Frame Generation capability values from an NGX parameter block.
    void read_frame_generation_capability_params(NVSDK_NGX_Parameter* capability_params);

    /// Backend hook for querying DLSS-D optimal settings.
    virtual DLSSDOptimalSettings
    query_dlssd_optimal_settings(uint32_t target_width, uint32_t target_height, QualityMode quality) const
        = 0;

    /// Backend hook for querying DLSS Super Resolution optimal settings.
    virtual DLSSOptimalSettings
    query_dlss_optimal_settings(uint32_t target_width, uint32_t target_height, QualityMode quality) const
        = 0;

    ref<sgl::Device> m_device;
    NGXInfo m_info;
    void* m_ngx_device{nullptr};
};

/// Return the backend implementation owned by an NGX context.
NGX::Impl* get_ngx_impl(NGX* ngx);

} // namespace falcor::ngx

#endif // FALCOR_ENABLE_NGX
