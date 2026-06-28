// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "falcor2/core/config.h"

#if FALCOR_ENABLE_NGX

#include "falcor2/denoisers/ngx_impl.h"

#include "falcor2/falcor2.h"

#include <sgl/device/device.h>

#include <nvsdk_ngx_helpers_dlssd_cuda.h>

#include <memory>
#include <utility>

namespace falcor::ngx {

namespace {

struct CudaHandles {
    NVSDK_NGX_CUDADevice ngx_device{};
};

CudaHandles get_cuda_handles(const sgl::Device* device)
{
    CudaHandles handles;
    if (!device)
        return handles;

    for (const sgl::NativeHandle& handle : device->native_handles()) {
        switch (handle.type()) {
        case sgl::NativeHandleType::CUcontext:
            handles.ngx_device.cudaContext = reinterpret_cast<void*>(handle.value());
            break;
        default:
            break;
        }
    }

    const sgl::NativeHandle stream_handle = device->get_native_command_queue_handle(sgl::CommandQueueType::graphics);
    if (stream_handle.type() == sgl::NativeHandleType::CUstream)
        handles.ngx_device.cudaStream = reinterpret_cast<void*>(stream_handle.value());

    return handles;
}

bool shutdown_cuda_runtime(NVSDK_NGX_CUDADevice* device) noexcept
{
    return NVSDK_NGX_CUDA_Shutdown1(device) == NVSDK_NGX_Result_Success;
}

class CudaNGXImpl final : public NGX::Impl {
public:
    /// Initialize NGX against the current CUDA context owned by the SGL device.
    explicit CudaNGXImpl(ref<sgl::Device> device)
        : NGX::Impl(std::move(device))
    {
        initialize();
    }

    /// Shut down the CUDA NGX runtime for the CUDA context owned by this device.
    ~CudaNGXImpl() { FALCOR_UNUSED(shutdown_cuda_runtime(&m_cuda_device)); }

private:
    /// Initialize the CUDA NGX SDK entry point and cache DLSS-D capability data.
    void initialize()
    {
        // SGL keeps the device context current by design. Passing the explicit
        // CUDA context to NGX still lets the SDK maintain a per-device instance.
        const CudaHandles handles = get_cuda_handles(m_device.get());
        FALCOR_CHECK(
            handles.ngx_device.cudaContext,
            "NGX CUDA initialization could not get a CUcontext native handle."
        );
        m_cuda_device = handles.ngx_device;
        m_ngx_device = &m_cuda_device;

        // NGX requires a writable app-data path and optionally accepts an extra
        // search path where it can find the runtime binary beside the executable.
        const std::filesystem::path app_data_path = NGX::Impl::app_data_path();
        const std::wstring app_data_path_wide = NGX::Impl::to_wstring_path(app_data_path);
        const wchar_t* path_list[1] = {};
        NVSDK_NGX_FeatureCommonInfo feature_common_info = NGX::Impl::make_feature_common_info(path_list);

        // This call creates the NGX CUDA context for the native CUDA context
        // owned by the SGL device.
        NVSDK_NGX_Result init_result = NVSDK_NGX_CUDA_Init1(
            k_application_id,
            app_data_path_wide.c_str(),
            &m_cuda_device,
            &feature_common_info,
            NVSDK_NGX_Version_API
        );

        FALCOR_CHECK(
            init_result == NVSDK_NGX_Result_Success,
            "NGX CUDA initialization failed: {}",
            ngx_result_to_string(static_cast<int32_t>(init_result))
        );

        // Capability parameters are SDK-owned and must be destroyed using the
        // matching CUDA destroy function on every exit path.
        NVSDK_NGX_Parameter* capability_params = nullptr;
        NVSDK_NGX_Result capability_result = NVSDK_NGX_CUDA_GetCapabilityParameters(&capability_params);
        auto capability_params_guard
            = std::unique_ptr<NVSDK_NGX_Parameter, decltype(&NVSDK_NGX_CUDA_DestroyParameters)>(
                capability_params,
                NVSDK_NGX_CUDA_DestroyParameters
            );

        if (capability_result != NVSDK_NGX_Result_Success || !capability_params) {
            // Initialization succeeded but capability query failed, so unwind the
            // SDK context before surfacing the error.
            FALCOR_UNUSED(shutdown_cuda_runtime(&m_cuda_device));
            FALCOR_THROW(
                "NGX CUDA capability query failed: {}",
                ngx_result_to_string(static_cast<int32_t>(capability_result))
            );
        }

        // Cache the DLSS-D support fields needed by public NGX queries.
        read_dlssd_capability_params(capability_params);
        read_dlss_capability_params(capability_params);
        read_frame_generation_capability_params(capability_params);
    }

    /// Query CUDA capability parameters and derive DLSS-D optimal settings.
    DLSSDOptimalSettings
    query_dlssd_optimal_settings(uint32_t target_width, uint32_t target_height, QualityMode quality) const override
    {
        // Request a fresh parameter block because NGX computes optimal settings
        // through the same capability parameter interface.
        NVSDK_NGX_Parameter* capability_params = nullptr;
        NVSDK_NGX_Result capability_result = NVSDK_NGX_CUDA_GetCapabilityParameters(&capability_params);
        auto capability_params_guard
            = std::unique_ptr<NVSDK_NGX_Parameter, decltype(&NVSDK_NGX_CUDA_DestroyParameters)>(
                capability_params,
                NVSDK_NGX_CUDA_DestroyParameters
            );

        FALCOR_CHECK(
            capability_result == NVSDK_NGX_Result_Success && capability_params,
            "DLSSD NGX CUDA capability query for optimal settings failed: {}",
            ngx_result_to_string(static_cast<int32_t>(capability_result))
        );

        return query_dlssd_optimal_settings_from_params(capability_params, target_width, target_height, quality);
    }

    /// Query CUDA capability parameters and derive DLSS Super Resolution optimal settings.
    DLSSOptimalSettings
    query_dlss_optimal_settings(uint32_t target_width, uint32_t target_height, QualityMode quality) const override
    {
        NVSDK_NGX_Parameter* capability_params = nullptr;
        NVSDK_NGX_Result capability_result = NVSDK_NGX_CUDA_GetCapabilityParameters(&capability_params);
        auto capability_params_guard
            = std::unique_ptr<NVSDK_NGX_Parameter, decltype(&NVSDK_NGX_CUDA_DestroyParameters)>(
                capability_params,
                NVSDK_NGX_CUDA_DestroyParameters
            );

        FALCOR_CHECK(
            capability_result == NVSDK_NGX_Result_Success && capability_params,
            "DLSS NGX CUDA capability query for optimal settings failed: {}",
            ngx_result_to_string(static_cast<int32_t>(capability_result))
        );

        return query_dlss_optimal_settings_from_params(capability_params, target_width, target_height, quality);
    }

    NVSDK_NGX_CUDADevice m_cuda_device{};
};

} // namespace

std::unique_ptr<NGX::Impl> make_cuda_ngx_impl(ref<sgl::Device> device)
{
    return std::make_unique<CudaNGXImpl>(std::move(device));
}

} // namespace falcor::ngx

#endif // FALCOR_ENABLE_NGX
