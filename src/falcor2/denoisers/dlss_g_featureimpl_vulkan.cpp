// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "falcor2/core/config.h"

#if FALCOR_ENABLE_NGX

#include "falcor2/denoisers/dlss_g_feature_impl.h"
#include "falcor2/denoisers/ngx_impl.h"
#include "falcor2/falcor2.h"

#include <sgl/device/command.h>
#include <sgl/device/device.h>
#include <sgl/device/formats.h>
#include <sgl/device/resource.h>

#include <vulkan/vulkan.h>
#include <nvsdk_ngx_helpers_dlssg_vk.h>
#include <nvsdk_ngx_helpers_vk.h>

#include <string>
#include <utility>

namespace falcor::ngx {
namespace {

struct VulkanTextureResource {
    ref<sgl::Texture> texture;
    ref<sgl::TextureView> view;
    NVSDK_NGX_Resource_VK resource{};
};

VulkanTextureResource make_vulkan_resource(ref<sgl::Texture> texture, bool read_write, const char* name)
{
    sgl::NativeHandle image_handle = texture->native_handle();
    FALCOR_CHECK(
        image_handle.type() == sgl::NativeHandleType::VkImage && image_handle.value(),
        "Vulkan DLSS-G evaluation {} must expose a VkImage.",
        name
    );

    VulkanTextureResource resource;
    resource.texture = std::move(texture);
    resource.view = resource.texture->create_view({});

    sgl::NativeHandle view_handle = resource.view->native_handle();
    FALCOR_CHECK(
        view_handle.type() == sgl::NativeHandleType::VkImageView && view_handle.value(),
        "Vulkan DLSS-G evaluation {} must expose a VkImageView.",
        name
    );

    VkImageSubresourceRange subresource_range{};
    subresource_range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    subresource_range.baseMipLevel = 0;
    subresource_range.levelCount = 1;
    subresource_range.baseArrayLayer = 0;
    subresource_range.layerCount = 1;

    resource.resource = NVSDK_NGX_Create_ImageView_Resource_VK(
        reinterpret_cast<VkImageView>(view_handle.value()),
        reinterpret_cast<VkImage>(image_handle.value()),
        subresource_range,
        static_cast<VkFormat>(sgl::get_format_info(resource.texture->format()).vk_format),
        resource.texture->width(),
        resource.texture->height(),
        read_write
    );
    return resource;
}

class VulkanDLSSGFeatureImpl final : public DLSSGFeature::Impl {
public:
    /// Create the Vulkan NGX DLSS-G feature on an active Vulkan command buffer.
    VulkanDLSSGFeatureImpl(ref<NGX> ngx, DLSSGFeatureDesc desc)
        : DLSSGFeature::Impl(std::move(ngx), desc)
    {
        try {
            initialize();
        } catch (...) {
            release_resources();
            throw;
        }
    }

    /// Release the Vulkan NGX feature and parameter map.
    ~VulkanDLSSGFeatureImpl() override { release_resources(); }

    /// Record a Vulkan NGX DLSS-G evaluation callback.
    void record_evaluate(
        ref<DLSSGFeature> owner,
        sgl::CommandEncoder* command_encoder,
        const DLSSGEvaluateDesc& desc
    ) override
    {
        validate_evaluate_desc(desc);
        DLSSGEvaluateResources resources = retain_evaluate_resources(desc);
        prepare_evaluate_resources(command_encoder, resources);

        reset_last_evaluate_result();
        command_encoder->execute_callback(
            [this, resources = std::move(resources), owner = std::move(owner)](sgl::NativeHandle native_handle)
            {
                FALCOR_UNUSED(owner);
                if (native_handle.type() != sgl::NativeHandleType::VkCommandBuffer || !native_handle.value()) {
                    set_last_evaluate_result(NVSDK_NGX_Result_FAIL_InvalidParameter);
                    return;
                }

                VulkanTextureResource backbuffer = make_vulkan_resource(resources.backbuffer, false, "backbuffer");
                VulkanTextureResource depth = make_vulkan_resource(resources.depth, false, "depth");
                VulkanTextureResource motion_vectors
                    = make_vulkan_resource(resources.motion_vectors, false, "motion_vectors");
                VulkanTextureResource output_interpolated_frame
                    = make_vulkan_resource(resources.output_interpolated_frame, true, "output_interpolated_frame");

                NVSDK_NGX_VK_DLSSG_Eval_Params eval_params{};
                eval_params.pBackbuffer = &backbuffer.resource;
                eval_params.pDepth = &depth.resource;
                eval_params.pMVecs = &motion_vectors.resource;
                eval_params.pOutputInterpFrame = &output_interpolated_frame.resource;

                NVSDK_NGX_DLSSG_Opt_Eval_Params opt_eval_params = make_opt_eval_params(resources);
                set_last_evaluate_result(NGX_VK_EVALUATE_DLSSG(
                    reinterpret_cast<VkCommandBuffer>(native_handle.value()),
                    m_feature_handle,
                    m_params,
                    &eval_params,
                    &opt_eval_params
                ));
            }
        );
    }

private:
    /// Allocate the NGX parameter map and create the feature through a native command callback.
    void initialize()
    {
        validate_feature_desc();

        NVSDK_NGX_Result allocate_result = NVSDK_NGX_VULKAN_AllocateParameters(&m_params);
        FALCOR_CHECK(
            allocate_result == NVSDK_NGX_Result_Success && m_params,
            "Vulkan DLSS-G parameter allocation failed: {}",
            ngx_result_to_string(static_cast<int32_t>(allocate_result))
        );

        sgl::Device* device = m_ngx->device();
        ref<sgl::CommandEncoder> command_encoder = device->create_command_encoder();

        bool callback_called = false;
        std::string callback_error;
        NVSDK_NGX_Result create_result = NVSDK_NGX_Result_Fail;

        command_encoder->execute_callback(
            [&](sgl::NativeHandle native_handle)
            {
                callback_called = true;
                if (native_handle.type() != sgl::NativeHandleType::VkCommandBuffer || !native_handle.value()) {
                    callback_error = "Vulkan DLSS-G feature creation requires an active VkCommandBuffer.";
                    return;
                }

                const uint32_t vk_format = sgl::get_format_info(m_desc.backbuffer_format).vk_format;
                NVSDK_NGX_DLSSG_Create_Params create_params = make_create_params(vk_format);
                NVSDK_NGX_Parameter_SetUI(m_params, NVSDK_NGX_DLSSG_Parameter_InternalWidth, create_params.RenderWidth);
                NVSDK_NGX_Parameter_SetUI(
                    m_params,
                    NVSDK_NGX_DLSSG_Parameter_InternalHeight,
                    create_params.RenderHeight
                );
                create_result = NGX_VK_CREATE_DLSSG(
                    reinterpret_cast<VkCommandBuffer>(native_handle.value()),
                    1,
                    1,
                    &m_feature_handle,
                    m_params,
                    &create_params
                );
            }
        );

        ref<sgl::CommandBuffer> command_buffer = command_encoder->finish();
        device->submit_command_buffer(command_buffer);
        device->wait();

        FALCOR_CHECK(callback_called, "Vulkan DLSS-G feature creation callback was not executed.");
        FALCOR_CHECK(callback_error.empty(), "{}", callback_error);
        FALCOR_CHECK(
            create_result == NVSDK_NGX_Result_Success && m_feature_handle,
            "Vulkan DLSS-G feature creation failed: {}",
            ngx_result_to_string(static_cast<int32_t>(create_result))
        );
    }

    /// Release backend resources in the order expected by NGX.
    void release_resources() noexcept
    {
        if (m_feature_handle) {
            FALCOR_UNUSED(NVSDK_NGX_VULKAN_ReleaseFeature(m_feature_handle));
            m_feature_handle = nullptr;
        }
        if (m_params) {
            FALCOR_UNUSED(NVSDK_NGX_VULKAN_DestroyParameters(m_params));
            m_params = nullptr;
        }
    }

    NVSDK_NGX_Parameter* m_params{nullptr};
    NVSDK_NGX_Handle* m_feature_handle{nullptr};
};

} // namespace

std::unique_ptr<DLSSGFeature::Impl> make_vulkan_dlss_g_feature_impl(ref<NGX> ngx, const DLSSGFeatureDesc& desc)
{
    return std::make_unique<VulkanDLSSGFeatureImpl>(std::move(ngx), desc);
}

} // namespace falcor::ngx

#endif // FALCOR_ENABLE_NGX
