// SPDX-License-Identifier: Apache-2.0

#include "falcor2/core/config.h"

#if FALCOR_ENABLE_NGX

#include "falcor2/denoisers/ngx_impl.h"

#include "falcor2/falcor2.h"

#include <sgl/device/device.h>

#include <vulkan/vulkan.h>
#include <nvsdk_ngx_vk.h>

#include <memory>
#include <utility>

namespace falcor::ngx {

namespace {

struct VulkanHandles {
    VkInstance instance{VK_NULL_HANDLE};
    VkPhysicalDevice physical_device{VK_NULL_HANDLE};
    VkDevice device{VK_NULL_HANDLE};
};

VulkanHandles get_vulkan_handles(const sgl::Device* device)
{
    VulkanHandles handles;
    if (!device)
        return handles;

    for (const sgl::NativeHandle& handle : device->native_handles()) {
        switch (handle.type()) {
        case sgl::NativeHandleType::VkInstance:
            handles.instance = reinterpret_cast<VkInstance>(handle.value());
            break;
        case sgl::NativeHandleType::VkPhysicalDevice:
            handles.physical_device = reinterpret_cast<VkPhysicalDevice>(handle.value());
            break;
        case sgl::NativeHandleType::VkDevice:
            handles.device = reinterpret_cast<VkDevice>(handle.value());
            break;
        default:
            break;
        }
    }

    return handles;
}

class VulkanNGXImpl final : public NGX::Impl {
public:
    /// Initialize NGX against the native Vulkan instance, physical device, and device.
    explicit VulkanNGXImpl(ref<sgl::Device> device)
        : NGX::Impl(std::move(device))
    {
        initialize();
    }

    /// Shut down the Vulkan NGX context if initialization reached the SDK.
    ~VulkanNGXImpl()
    {
        if (!m_vulkan_device)
            return;

        NVSDK_NGX_VULKAN_Shutdown1(m_vulkan_device);
    }

private:
    /// Initialize the Vulkan NGX SDK entry point and cache DLSS-D capability data.
    void initialize()
    {
        // Preserve the required extension list in NGXInfo for diagnostics and
        // pre-device setup, even after a full Vulkan device exists.
        query_vulkan_required_extensions(m_info);

        // NGX Vulkan initialization needs all three native handles from the same
        // SGL device: instance, physical device, and logical device.
        VulkanHandles handles = get_vulkan_handles(m_device.get());
        FALCOR_CHECK(
            handles.instance && handles.physical_device && handles.device,
            "NGX Vulkan initialization could not get complete Vulkan native handles."
        );

        m_vulkan_device = handles.device;
        m_ngx_device = reinterpret_cast<void*>(m_vulkan_device);

        // NGX requires a writable app-data path and optionally accepts an extra
        // search path where it can find the runtime binary beside the executable.
        const std::filesystem::path app_data_path = NGX::Impl::app_data_path();
        const std::wstring app_data_path_wide = NGX::Impl::to_wstring_path(app_data_path);
        const wchar_t* path_list[1] = {};
        NVSDK_NGX_FeatureCommonInfo feature_common_info = NGX::Impl::make_feature_common_info(path_list);

        // This call creates the NGX Vulkan context for the native SGL device.
        NVSDK_NGX_Result init_result = NVSDK_NGX_VULKAN_Init(
            k_application_id,
            app_data_path_wide.c_str(),
            handles.instance,
            handles.physical_device,
            handles.device,
            nullptr,
            nullptr,
            &feature_common_info,
            NVSDK_NGX_Version_API
        );

        FALCOR_CHECK(
            init_result == NVSDK_NGX_Result_Success,
            "NGX Vulkan initialization failed: {}",
            ngx_result_to_string(static_cast<int32_t>(init_result))
        );

        // Capability parameters are SDK-owned and must be destroyed using the
        // matching Vulkan destroy function on every exit path.
        NVSDK_NGX_Parameter* capability_params = nullptr;
        NVSDK_NGX_Result capability_result = NVSDK_NGX_VULKAN_GetCapabilityParameters(&capability_params);
        auto capability_params_guard
            = std::unique_ptr<NVSDK_NGX_Parameter, decltype(&NVSDK_NGX_VULKAN_DestroyParameters)>(
                capability_params,
                NVSDK_NGX_VULKAN_DestroyParameters
            );

        if (capability_result != NVSDK_NGX_Result_Success || !capability_params) {
            // Initialization succeeded but capability query failed, so unwind the
            // SDK context before surfacing the error.
            NVSDK_NGX_VULKAN_Shutdown1(m_vulkan_device);
            FALCOR_THROW(
                "NGX Vulkan capability query failed: {}",
                ngx_result_to_string(static_cast<int32_t>(capability_result))
            );
        }

        // Cache the DLSS-D support fields needed by public NGX queries.
        read_dlssd_capability_params(capability_params);
        read_dlss_capability_params(capability_params);
        read_frame_generation_capability_params(capability_params);
    }

    /// Query Vulkan capability parameters and derive DLSS-D optimal settings.
    DLSSDOptimalSettings
    query_dlssd_optimal_settings(uint32_t target_width, uint32_t target_height, QualityMode quality) const override
    {
        // Request a fresh parameter block because NGX computes optimal settings
        // through the same capability parameter interface.
        NVSDK_NGX_Parameter* capability_params = nullptr;
        NVSDK_NGX_Result capability_result = NVSDK_NGX_VULKAN_GetCapabilityParameters(&capability_params);
        auto capability_params_guard
            = std::unique_ptr<NVSDK_NGX_Parameter, decltype(&NVSDK_NGX_VULKAN_DestroyParameters)>(
                capability_params,
                NVSDK_NGX_VULKAN_DestroyParameters
            );

        FALCOR_CHECK(
            capability_result == NVSDK_NGX_Result_Success && capability_params,
            "DLSSD NGX Vulkan capability query for optimal settings failed: {}",
            ngx_result_to_string(static_cast<int32_t>(capability_result))
        );

        return query_dlssd_optimal_settings_from_params(capability_params, target_width, target_height, quality);
    }

    /// Query Vulkan capability parameters and derive DLSS Super Resolution optimal settings.
    DLSSOptimalSettings
    query_dlss_optimal_settings(uint32_t target_width, uint32_t target_height, QualityMode quality) const override
    {
        NVSDK_NGX_Parameter* capability_params = nullptr;
        NVSDK_NGX_Result capability_result = NVSDK_NGX_VULKAN_GetCapabilityParameters(&capability_params);
        auto capability_params_guard
            = std::unique_ptr<NVSDK_NGX_Parameter, decltype(&NVSDK_NGX_VULKAN_DestroyParameters)>(
                capability_params,
                NVSDK_NGX_VULKAN_DestroyParameters
            );

        FALCOR_CHECK(
            capability_result == NVSDK_NGX_Result_Success && capability_params,
            "DLSS NGX Vulkan capability query for optimal settings failed: {}",
            ngx_result_to_string(static_cast<int32_t>(capability_result))
        );

        return query_dlss_optimal_settings_from_params(capability_params, target_width, target_height, quality);
    }

    VkDevice m_vulkan_device{VK_NULL_HANDLE};
};

} // namespace

std::unique_ptr<NGX::Impl> make_vulkan_ngx_impl(ref<sgl::Device> device)
{
    return std::make_unique<VulkanNGXImpl>(std::move(device));
}

void query_vulkan_required_extensions(NGXInfo& info)
{
    unsigned int instance_extension_count = 0;
    const char** instance_extensions = nullptr;
    unsigned int device_extension_count = 0;
    const char** device_extensions = nullptr;

    info.required_vulkan_instance_extensions.clear();
    info.required_vulkan_device_extensions.clear();

    // This NGX query is available before SDK initialization and is the one piece
    // of information Vulkan needs before creating the device.
    NVSDK_NGX_Result result = NVSDK_NGX_VULKAN_RequiredExtensions(
        &instance_extension_count,
        &instance_extensions,
        &device_extension_count,
        &device_extensions
    );

    FALCOR_CHECK(
        result == NVSDK_NGX_Result_Success,
        "NGX Vulkan extension query failed: {}",
        ngx_result_to_string(static_cast<int32_t>(result))
    );

    for (unsigned int i = 0; i < instance_extension_count; ++i) {
        if (instance_extensions && instance_extensions[i])
            info.required_vulkan_instance_extensions.emplace_back(instance_extensions[i]);
    }

    for (unsigned int i = 0; i < device_extension_count; ++i) {
        if (device_extensions && device_extensions[i])
            info.required_vulkan_device_extensions.emplace_back(device_extensions[i]);
    }
}

} // namespace falcor::ngx

#endif // FALCOR_ENABLE_NGX
