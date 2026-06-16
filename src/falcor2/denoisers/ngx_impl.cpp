// SPDX-License-Identifier: Apache-2.0

#include "falcor2/core/config.h"

#if FALCOR_ENABLE_NGX

#include "falcor2/denoisers/ngx_impl.h"

#include "falcor2/core/platform.h"
#include "falcor2/falcor2.h"

#include <sgl/device/device.h>

#include <nvsdk_ngx_helpers.h>
#include <nvsdk_ngx_helpers_dlssd_cuda.h>
#include <nvsdk_ngx_defs_dlssg.h>

#include <algorithm>
#include <utility>

namespace falcor::ngx {
std::unique_ptr<NGX::Impl> make_d3d12_ngx_impl(ref<sgl::Device> device);
std::unique_ptr<NGX::Impl> make_vulkan_ngx_impl(ref<sgl::Device> device);
std::unique_ptr<NGX::Impl> make_cuda_ngx_impl(ref<sgl::Device> device);

// clang-format off
static_assert(static_cast<uint32_t>(Result::none) == 0u);
static_assert(static_cast<uint32_t>(Result::success) == static_cast<uint32_t>(NVSDK_NGX_Result_Success));
static_assert(static_cast<uint32_t>(Result::fail) == static_cast<uint32_t>(NVSDK_NGX_Result_Fail));
static_assert(static_cast<uint32_t>(Result::fail_feature_not_supported) == static_cast<uint32_t>(NVSDK_NGX_Result_FAIL_FeatureNotSupported));
static_assert(static_cast<uint32_t>(Result::fail_platform_error) == static_cast<uint32_t>(NVSDK_NGX_Result_FAIL_PlatformError));
static_assert(static_cast<uint32_t>(Result::fail_feature_already_exists) == static_cast<uint32_t>(NVSDK_NGX_Result_FAIL_FeatureAlreadyExists));
static_assert(static_cast<uint32_t>(Result::fail_feature_not_found) == static_cast<uint32_t>(NVSDK_NGX_Result_FAIL_FeatureNotFound));
static_assert(static_cast<uint32_t>(Result::fail_invalid_parameter) == static_cast<uint32_t>(NVSDK_NGX_Result_FAIL_InvalidParameter));
static_assert(static_cast<uint32_t>(Result::fail_scratch_buffer_too_small) == static_cast<uint32_t>(NVSDK_NGX_Result_FAIL_ScratchBufferTooSmall));
static_assert(static_cast<uint32_t>(Result::fail_not_initialized) == static_cast<uint32_t>(NVSDK_NGX_Result_FAIL_NotInitialized));
static_assert(static_cast<uint32_t>(Result::fail_unsupported_input_format) == static_cast<uint32_t>(NVSDK_NGX_Result_FAIL_UnsupportedInputFormat));
static_assert(static_cast<uint32_t>(Result::fail_rw_flag_missing) == static_cast<uint32_t>(NVSDK_NGX_Result_FAIL_RWFlagMissing));
static_assert(static_cast<uint32_t>(Result::fail_missing_input) == static_cast<uint32_t>(NVSDK_NGX_Result_FAIL_MissingInput));
static_assert(static_cast<uint32_t>(Result::fail_unable_to_initialize_feature) == static_cast<uint32_t>(NVSDK_NGX_Result_FAIL_UnableToInitializeFeature));
static_assert(static_cast<uint32_t>(Result::fail_out_of_date) == static_cast<uint32_t>(NVSDK_NGX_Result_FAIL_OutOfDate));
static_assert(static_cast<uint32_t>(Result::fail_out_of_gpu_memory) == static_cast<uint32_t>(NVSDK_NGX_Result_FAIL_OutOfGPUMemory));
static_assert(static_cast<uint32_t>(Result::fail_unsupported_format) == static_cast<uint32_t>(NVSDK_NGX_Result_FAIL_UnsupportedFormat));
static_assert(static_cast<uint32_t>(Result::fail_unable_to_write_to_app_data_path) == static_cast<uint32_t>(NVSDK_NGX_Result_FAIL_UnableToWriteToAppDataPath));
static_assert(static_cast<uint32_t>(Result::fail_unsupported_parameter) == static_cast<uint32_t>(NVSDK_NGX_Result_FAIL_UnsupportedParameter));
static_assert(static_cast<uint32_t>(Result::fail_denied) == static_cast<uint32_t>(NVSDK_NGX_Result_FAIL_Denied));
static_assert(static_cast<uint32_t>(Result::fail_not_implemented) == static_cast<uint32_t>(NVSDK_NGX_Result_FAIL_NotImplemented));
// clang-format on

static std::filesystem::path default_app_data_path()
{
    return falcor::platform::app_data_directory() / "ngx";
}

static bool read_param_i(NVSDK_NGX_Parameter* params, const char* name, int32_t& value)
{
    int int_value = 0;
    NVSDK_NGX_Result result = NVSDK_NGX_Parameter_GetI(params, name, &int_value);
    if (result == NVSDK_NGX_Result_Success) {
        value = int_value;
        return true;
    }

    unsigned int uint_value = 0;
    result = NVSDK_NGX_Parameter_GetUI(params, name, &uint_value);
    if (result == NVSDK_NGX_Result_Success) {
        value = static_cast<int32_t>(uint_value);
        return true;
    }

    return false;
}

NVSDK_NGX_PerfQuality_Value to_ngx_perf_quality(QualityMode quality)
{
    switch (quality) {
    case QualityMode::performance:
        return NVSDK_NGX_PerfQuality_Value_MaxPerf;
    case QualityMode::balanced:
        return NVSDK_NGX_PerfQuality_Value_Balanced;
    case QualityMode::quality:
        return NVSDK_NGX_PerfQuality_Value_MaxQuality;
    case QualityMode::ultra_performance:
        return NVSDK_NGX_PerfQuality_Value_UltraPerformance;
    case QualityMode::dlaa:
        return NVSDK_NGX_PerfQuality_Value_DLAA;
    default:
        return NVSDK_NGX_PerfQuality_Value_MaxQuality;
    }
}

Result to_result(NVSDK_NGX_Result result)
{
    return static_cast<Result>(static_cast<uint32_t>(result));
}

NGX::Impl::Impl(ref<sgl::Device> device)
    : m_device(std::move(device))
{
}

NGX::Impl::~Impl() = default;

DLSSDOptimalSettings
NGX::Impl::get_dlssd_optimal_settings(uint32_t target_width, uint32_t target_height, QualityMode quality) const
{
    FALCOR_CHECK(
        target_width != 0 && target_height != 0,
        "DLSSD optimal-settings query requires non-zero target dimensions."
    );
    if (!m_info.dlssd_supported) {
        const int32_t ngx_success = static_cast<int32_t>(1);

        // Prefer the most actionable failure reason exposed by the capability query.
        if (m_info.dlssd_needs_updated_driver) {
            FALCOR_THROW(
                "DLSSD optimal-settings query requires an updated NVIDIA driver. "
                "SuperSamplingDenoising minimum driver version is {}.{}.",
                m_info.dlssd_minimum_driver_version_major,
                m_info.dlssd_minimum_driver_version_minor
            );
        }
        if (m_info.dlssd_feature_init_result != 0 && m_info.dlssd_feature_init_result != ngx_success) {
            FALCOR_THROW(
                "DLSSD optimal-settings query requires supported NGX SuperSamplingDenoising capability. "
                "FeatureInitResult={}.",
                ngx_result_to_string(m_info.dlssd_feature_init_result)
            );
        }
        FALCOR_THROW(
            "DLSSD optimal-settings query requires supported NGX SuperSamplingDenoising capability. "
            "dlssd_available={}.",
            m_info.dlssd_available ? "true" : "false"
        );
    }
    return query_dlssd_optimal_settings(target_width, target_height, quality);
}

DLSSOptimalSettings
NGX::Impl::get_dlss_optimal_settings(uint32_t target_width, uint32_t target_height, QualityMode quality) const
{
    FALCOR_CHECK(
        target_width != 0 && target_height != 0,
        "DLSS optimal-settings query requires non-zero target dimensions."
    );
    if (!m_info.dlss_supported) {
        const int32_t ngx_success = static_cast<int32_t>(NVSDK_NGX_Result_Success);

        if (m_info.dlss_needs_updated_driver) {
            FALCOR_THROW(
                "DLSS optimal-settings query requires an updated NVIDIA driver. "
                "SuperSampling minimum driver version is {}.{}.",
                m_info.dlss_minimum_driver_version_major,
                m_info.dlss_minimum_driver_version_minor
            );
        }
        if (m_info.dlss_feature_init_result != 0 && m_info.dlss_feature_init_result != ngx_success) {
            FALCOR_THROW(
                "DLSS optimal-settings query requires supported NGX SuperSampling capability. "
                "FeatureInitResult={}.",
                ngx_result_to_string(m_info.dlss_feature_init_result)
            );
        }
        FALCOR_THROW(
            "DLSS optimal-settings query requires supported NGX SuperSampling capability. dlss_available={}.",
            m_info.dlss_available ? "true" : "false"
        );
    }
    return query_dlss_optimal_settings(target_width, target_height, quality);
}

std::unique_ptr<NGX::Impl> NGX::Impl::create(ref<sgl::Device> device)
{
    FALCOR_CHECK(device, "NGX requires a device.");

    // Keep backend selection tied to the SGL device so NGX cannot drift from the
    // native context that will later be used for command submission.
    switch (device->type()) {
    case sgl::DeviceType::d3d12:
#if SGL_WINDOWS
        return make_d3d12_ngx_impl(std::move(device));
#else
        FALCOR_THROW("NGX D3D12 backend is only available on Windows.");
#endif
    case sgl::DeviceType::vulkan:
        return make_vulkan_ngx_impl(std::move(device));
    case sgl::DeviceType::cuda:
        return make_cuda_ngx_impl(std::move(device));
    default:
        FALCOR_THROW("NGX only supports D3D12, Vulkan, and CUDA devices.");
    }
}

std::string ngx_result_to_string(int32_t raw_result)
{
    if (raw_result == 0) {
        return "not-run";
    }

    switch (static_cast<NVSDK_NGX_Result>(static_cast<uint32_t>(raw_result))) {
    case NVSDK_NGX_Result_Success:
        return "NVSDK_NGX_Result_Success";
    case NVSDK_NGX_Result_Fail:
        return "NVSDK_NGX_Result_Fail";
    case NVSDK_NGX_Result_FAIL_FeatureNotSupported:
        return "NVSDK_NGX_Result_FAIL_FeatureNotSupported";
    case NVSDK_NGX_Result_FAIL_PlatformError:
        return "NVSDK_NGX_Result_FAIL_PlatformError";
    case NVSDK_NGX_Result_FAIL_FeatureAlreadyExists:
        return "NVSDK_NGX_Result_FAIL_FeatureAlreadyExists";
    case NVSDK_NGX_Result_FAIL_FeatureNotFound:
        return "NVSDK_NGX_Result_FAIL_FeatureNotFound";
    case NVSDK_NGX_Result_FAIL_InvalidParameter:
        return "NVSDK_NGX_Result_FAIL_InvalidParameter";
    case NVSDK_NGX_Result_FAIL_ScratchBufferTooSmall:
        return "NVSDK_NGX_Result_FAIL_ScratchBufferTooSmall";
    case NVSDK_NGX_Result_FAIL_NotInitialized:
        return "NVSDK_NGX_Result_FAIL_NotInitialized";
    case NVSDK_NGX_Result_FAIL_UnsupportedInputFormat:
        return "NVSDK_NGX_Result_FAIL_UnsupportedInputFormat";
    case NVSDK_NGX_Result_FAIL_RWFlagMissing:
        return "NVSDK_NGX_Result_FAIL_RWFlagMissing";
    case NVSDK_NGX_Result_FAIL_MissingInput:
        return "NVSDK_NGX_Result_FAIL_MissingInput";
    case NVSDK_NGX_Result_FAIL_UnableToInitializeFeature:
        return "NVSDK_NGX_Result_FAIL_UnableToInitializeFeature";
    case NVSDK_NGX_Result_FAIL_OutOfDate:
        return "NVSDK_NGX_Result_FAIL_OutOfDate";
    case NVSDK_NGX_Result_FAIL_OutOfGPUMemory:
        return "NVSDK_NGX_Result_FAIL_OutOfGPUMemory";
    case NVSDK_NGX_Result_FAIL_UnsupportedFormat:
        return "NVSDK_NGX_Result_FAIL_UnsupportedFormat";
    case NVSDK_NGX_Result_FAIL_UnableToWriteToAppDataPath:
        return "NVSDK_NGX_Result_FAIL_UnableToWriteToAppDataPath";
    case NVSDK_NGX_Result_FAIL_UnsupportedParameter:
        return "NVSDK_NGX_Result_FAIL_UnsupportedParameter";
    case NVSDK_NGX_Result_FAIL_Denied:
        return "NVSDK_NGX_Result_FAIL_Denied";
    case NVSDK_NGX_Result_FAIL_NotImplemented:
        return "NVSDK_NGX_Result_FAIL_NotImplemented";
    default:
        return "NVSDK_NGX_Result_" + std::to_string(raw_result);
    }
}

std::wstring NGX::Impl::to_wstring_path(const std::filesystem::path& path)
{
#if SGL_WINDOWS
    return path.wstring();
#else
    const std::string path_string = path.string();
    return std::wstring(path_string.begin(), path_string.end());
#endif
}

std::filesystem::path NGX::Impl::app_data_path()
{
    std::filesystem::path path = default_app_data_path().lexically_normal();
    std::filesystem::create_directories(path);
    return path;
}

NVSDK_NGX_FeatureCommonInfo NGX::Impl::make_feature_common_info(const wchar_t** path_list)
{
    NVSDK_NGX_FeatureCommonInfo info{};
    const std::filesystem::path runtime_dir = falcor::platform::runtime_directory();
    if (!runtime_dir.empty()) {
        static thread_local std::wstring runtime_dir_wide;
        runtime_dir_wide = to_wstring_path(runtime_dir);
        path_list[0] = runtime_dir_wide.c_str();
        info.PathListInfo.Path = path_list;
        info.PathListInfo.Length = 1;
    }
    return info;
}

void NGX::Impl::read_dlssd_capability_params(NVSDK_NGX_Parameter* capability_params)
{
    int32_t available = 0;
    int32_t needs_updated_driver = 0;
    int32_t minimum_driver_major = 0;
    int32_t minimum_driver_minor = 0;
    int32_t feature_init_result = 0;

    read_param_i(capability_params, NVSDK_NGX_Parameter_SuperSamplingDenoising_Available, available);
    read_param_i(
        capability_params,
        NVSDK_NGX_Parameter_SuperSamplingDenoising_NeedsUpdatedDriver,
        needs_updated_driver
    );
    read_param_i(
        capability_params,
        NVSDK_NGX_Parameter_SuperSamplingDenoising_MinDriverVersionMajor,
        minimum_driver_major
    );
    read_param_i(
        capability_params,
        NVSDK_NGX_Parameter_SuperSamplingDenoising_MinDriverVersionMinor,
        minimum_driver_minor
    );
    read_param_i(capability_params, NVSDK_NGX_Parameter_SuperSamplingDenoising_FeatureInitResult, feature_init_result);

    // NGX exposes availability, driver state, and feature init state as separate
    // parameters. Combine them into the single support predicate the public API
    // needs, while preserving the raw fields for diagnostics.
    const int32_t ngx_success = static_cast<int32_t>(NVSDK_NGX_Result_Success);
    m_info.dlssd_available = available != 0;
    m_info.dlssd_supported = available != 0 && needs_updated_driver == 0
        && (feature_init_result == 0 || feature_init_result == ngx_success);
    m_info.dlssd_needs_updated_driver = needs_updated_driver != 0;
    m_info.dlssd_minimum_driver_version_major = static_cast<uint32_t>(std::max<int32_t>(minimum_driver_major, 0));
    m_info.dlssd_minimum_driver_version_minor = static_cast<uint32_t>(std::max<int32_t>(minimum_driver_minor, 0));
    m_info.dlssd_feature_init_result = feature_init_result;
}

void NGX::Impl::read_dlss_capability_params(NVSDK_NGX_Parameter* capability_params)
{
    int32_t available = 0;
    int32_t needs_updated_driver = 0;
    int32_t minimum_driver_major = 0;
    int32_t minimum_driver_minor = 0;
    int32_t feature_init_result = 0;

    read_param_i(capability_params, NVSDK_NGX_Parameter_SuperSampling_Available, available);
    read_param_i(capability_params, NVSDK_NGX_Parameter_SuperSampling_NeedsUpdatedDriver, needs_updated_driver);
    read_param_i(capability_params, NVSDK_NGX_Parameter_SuperSampling_MinDriverVersionMajor, minimum_driver_major);
    read_param_i(capability_params, NVSDK_NGX_Parameter_SuperSampling_MinDriverVersionMinor, minimum_driver_minor);
    read_param_i(capability_params, NVSDK_NGX_Parameter_SuperSampling_FeatureInitResult, feature_init_result);

    const int32_t ngx_success = static_cast<int32_t>(NVSDK_NGX_Result_Success);
    m_info.dlss_available = available != 0;
    m_info.dlss_supported = available != 0 && needs_updated_driver == 0
        && (feature_init_result == 0 || feature_init_result == ngx_success);
    m_info.dlss_needs_updated_driver = needs_updated_driver != 0;
    m_info.dlss_minimum_driver_version_major = static_cast<uint32_t>(std::max<int32_t>(minimum_driver_major, 0));
    m_info.dlss_minimum_driver_version_minor = static_cast<uint32_t>(std::max<int32_t>(minimum_driver_minor, 0));
    m_info.dlss_feature_init_result = feature_init_result;
}

void NGX::Impl::read_frame_generation_capability_params(NVSDK_NGX_Parameter* capability_params)
{
    int32_t available = 0;
    int32_t needs_updated_driver = 0;
    int32_t minimum_driver_major = 0;
    int32_t minimum_driver_minor = 0;
    int32_t feature_init_result = 0;

    read_param_i(capability_params, NVSDK_NGX_Parameter_FrameGeneration_Available, available);
    read_param_i(capability_params, NVSDK_NGX_Parameter_FrameGeneration_NeedsUpdatedDriver, needs_updated_driver);
    read_param_i(capability_params, NVSDK_NGX_Parameter_FrameGeneration_MinDriverVersionMajor, minimum_driver_major);
    read_param_i(capability_params, NVSDK_NGX_Parameter_FrameGeneration_MinDriverVersionMinor, minimum_driver_minor);
    read_param_i(capability_params, NVSDK_NGX_Parameter_FrameGeneration_FeatureInitResult, feature_init_result);

    unsigned int multi_frame_count_max = 1;
    NVSDK_NGX_Parameter_GetUI(capability_params, NVSDK_NGX_DLSSG_Parameter_MultiFrameCountMax, &multi_frame_count_max);

    const int32_t ngx_success = static_cast<int32_t>(NVSDK_NGX_Result_Success);
    m_info.frame_generation_available = available != 0;
    m_info.frame_generation_supported = available != 0 && needs_updated_driver == 0
        && (feature_init_result == 0 || feature_init_result == ngx_success);
    m_info.frame_generation_needs_updated_driver = needs_updated_driver != 0;
    m_info.frame_generation_minimum_driver_version_major
        = static_cast<uint32_t>(std::max<int32_t>(minimum_driver_major, 0));
    m_info.frame_generation_minimum_driver_version_minor
        = static_cast<uint32_t>(std::max<int32_t>(minimum_driver_minor, 0));
    m_info.frame_generation_feature_init_result = feature_init_result;
    m_info.frame_generation_multi_frame_count_max = std::max<uint32_t>(multi_frame_count_max, 1);
}

DLSSDOptimalSettings NGX::Impl::query_dlssd_optimal_settings_from_params(
    NVSDK_NGX_Parameter* capability_params,
    uint32_t target_width,
    uint32_t target_height,
    QualityMode quality
)
{
    DLSSDOptimalSettings settings;

    unsigned int render_width = 0;
    unsigned int render_height = 0;
    unsigned int max_render_width = 0;
    unsigned int max_render_height = 0;
    unsigned int min_render_width = 0;
    unsigned int min_render_height = 0;
    float sharpness = 0.f;

    // NGX writes the recommended render-size range into out-parameters. Keep the
    // wrapper narrow and copy only the values that are part of Falcor2's API.
    NVSDK_NGX_Result result = NGX_DLSSD_GET_OPTIMAL_SETTINGS(
        capability_params,
        target_width,
        target_height,
        to_ngx_perf_quality(quality),
        &render_width,
        &render_height,
        &max_render_width,
        &max_render_height,
        &min_render_width,
        &min_render_height,
        &sharpness
    );

    FALCOR_CHECK(
        result == NVSDK_NGX_Result_Success,
        "DLSSD optimal-settings query failed: {}",
        ngx_result_to_string(static_cast<int32_t>(result))
    );
    FALCOR_CHECK(
        render_width != 0 && render_height != 0,
        "DLSSD optimal-settings query returned zero render dimensions."
    );

    settings.render_width = render_width;
    settings.render_height = render_height;
    settings.max_render_width = max_render_width;
    settings.max_render_height = max_render_height;
    settings.min_render_width = min_render_width;
    settings.min_render_height = min_render_height;
    settings.sharpness = sharpness;
    return settings;
}

DLSSOptimalSettings NGX::Impl::query_dlss_optimal_settings_from_params(
    NVSDK_NGX_Parameter* capability_params,
    uint32_t target_width,
    uint32_t target_height,
    QualityMode quality
)
{
    DLSSOptimalSettings settings;

    unsigned int render_width = 0;
    unsigned int render_height = 0;
    unsigned int max_render_width = 0;
    unsigned int max_render_height = 0;
    unsigned int min_render_width = 0;
    unsigned int min_render_height = 0;
    float sharpness = 0.f;

    NVSDK_NGX_Result result = NGX_DLSS_GET_OPTIMAL_SETTINGS(
        capability_params,
        target_width,
        target_height,
        to_ngx_perf_quality(quality),
        &render_width,
        &render_height,
        &max_render_width,
        &max_render_height,
        &min_render_width,
        &min_render_height,
        &sharpness
    );

    FALCOR_CHECK(
        result == NVSDK_NGX_Result_Success,
        "DLSS optimal-settings query failed: {}",
        ngx_result_to_string(static_cast<int32_t>(result))
    );
    FALCOR_CHECK(
        render_width != 0 && render_height != 0,
        "DLSS optimal-settings query returned zero render dimensions."
    );

    settings.render_width = render_width;
    settings.render_height = render_height;
    settings.max_render_width = max_render_width;
    settings.max_render_height = max_render_height;
    settings.min_render_width = min_render_width;
    settings.min_render_height = min_render_height;
    settings.sharpness = sharpness;
    return settings;
}

} // namespace falcor::ngx

#endif // FALCOR_ENABLE_NGX
