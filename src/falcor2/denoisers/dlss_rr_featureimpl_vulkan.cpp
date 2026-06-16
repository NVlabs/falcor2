// SPDX-License-Identifier: Apache-2.0

#include "falcor2/core/config.h"

#if FALCOR_ENABLE_NGX

#include "falcor2/denoisers/dlss_rr_feature_impl.h"
#include "falcor2/denoisers/ngx_impl.h"
#include "falcor2/falcor2.h"

#include <sgl/device/command.h>
#include <sgl/device/device.h>
#include <sgl/device/resource.h>

#include <vulkan/vulkan.h>
#include <nvsdk_ngx_helpers_vk.h>
#include <nvsdk_ngx_helpers_dlssd_vk.h>

#include <string>
#include <utility>

namespace falcor::ngx {
namespace {

VkDevice get_vulkan_device(const sgl::Device* device)
{
    if (!device)
        return VK_NULL_HANDLE;

    for (const sgl::NativeHandle& handle : device->native_handles()) {
        if (handle.type() == sgl::NativeHandleType::VkDevice)
            return reinterpret_cast<VkDevice>(handle.value());
    }

    return VK_NULL_HANDLE;
}

struct VulkanTextureResource {
    ref<sgl::Texture> texture;
    ref<sgl::TextureView> view;
    NVSDK_NGX_Resource_VK resource{};
};

VkFormat to_vulkan_format(sgl::Format format);

VulkanTextureResource make_vulkan_resource(ref<sgl::Texture> texture, bool read_write, const char* name)
{
    sgl::NativeHandle image_handle = texture->native_handle();
    FALCOR_CHECK(
        image_handle.type() == sgl::NativeHandleType::VkImage && image_handle.value(),
        "Vulkan DLSS-RR evaluation {} must expose a VkImage.",
        name
    );

    VulkanTextureResource resource;
    resource.texture = std::move(texture);
    resource.view = resource.texture->create_view({});

    sgl::NativeHandle view_handle = resource.view->native_handle();
    FALCOR_CHECK(
        view_handle.type() == sgl::NativeHandleType::VkImageView && view_handle.value(),
        "Vulkan DLSS-RR evaluation {} must expose a VkImageView.",
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
        to_vulkan_format(resource.texture->format()),
        resource.texture->width(),
        resource.texture->height(),
        read_write
    );
    return resource;
}

VkFormat to_vulkan_format(sgl::Format format)
{
    switch (format) {
    case sgl::Format::r16_float:
        return VK_FORMAT_R16_SFLOAT;
    case sgl::Format::rg16_float:
        return VK_FORMAT_R16G16_SFLOAT;
    case sgl::Format::rgba16_float:
        return VK_FORMAT_R16G16B16A16_SFLOAT;
    case sgl::Format::r32_float:
        return VK_FORMAT_R32_SFLOAT;
    case sgl::Format::rg32_float:
        return VK_FORMAT_R32G32_SFLOAT;
    case sgl::Format::rgba32_float:
        return VK_FORMAT_R32G32B32A32_SFLOAT;
    default:
        FALCOR_THROW("Unsupported Vulkan DLSS-RR texture format: {}", format);
    }
}

class VulkanDLSSRRFeatureImpl final : public DLSSRRFeature::Impl {
public:
    /// Create the Vulkan NGX DLSS-RR feature on an active Vulkan command buffer.
    VulkanDLSSRRFeatureImpl(ref<NGX> ngx, DLSSRRFeatureDesc desc)
        : DLSSRRFeature::Impl(std::move(ngx), desc)
    {
        try {
            initialize();
        } catch (...) {
            release_resources();
            throw;
        }
    }

    /// Release the Vulkan NGX feature and parameter map.
    ~VulkanDLSSRRFeatureImpl() override { release_resources(); }

    /// Record a Vulkan NGX DLSS-RR evaluation callback.
    void record_evaluate(
        ref<DLSSRRFeature> owner,
        sgl::CommandEncoder* command_encoder,
        const DLSSRREvaluateDesc& desc
    ) override
    {
        validate_evaluate_desc(desc);
        DLSSRREvaluateResources resources = retain_evaluate_resources(desc);
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

                VulkanTextureResource diffuse_albedo
                    = make_vulkan_resource(resources.diffuse_albedo, false, "diffuse_albedo");
                VulkanTextureResource specular_albedo
                    = make_vulkan_resource(resources.specular_albedo, false, "specular_albedo");
                VulkanTextureResource normals = make_vulkan_resource(resources.normals, false, "normals");
                VulkanTextureResource roughness = make_vulkan_resource(resources.roughness, false, "roughness");
                VulkanTextureResource color = make_vulkan_resource(resources.color, false, "color");
                VulkanTextureResource depth = make_vulkan_resource(resources.depth, false, "depth");
                VulkanTextureResource motion_vectors
                    = make_vulkan_resource(resources.motion_vectors, false, "motion_vectors");
                VulkanTextureResource output = make_vulkan_resource(resources.output, true, "output");
                VulkanTextureResource motion_vectors_reflections;
                if (resources.motion_vectors_reflections) {
                    motion_vectors_reflections = make_vulkan_resource(
                        resources.motion_vectors_reflections,
                        false,
                        "motion_vectors_reflections"
                    );
                }
                VulkanTextureResource specular_hit_distance;
                if (resources.specular_hit_distance) {
                    specular_hit_distance
                        = make_vulkan_resource(resources.specular_hit_distance, false, "specular_hit_distance");
                }

                NVSDK_NGX_VK_DLSSD_Eval_Params eval_params{};
                eval_params.pInDiffuseAlbedo = &diffuse_albedo.resource;
                eval_params.pInSpecularAlbedo = &specular_albedo.resource;
                eval_params.pInNormals = &normals.resource;
                eval_params.pInRoughness = &roughness.resource;
                eval_params.pInColor = &color.resource;
                eval_params.pInOutput = &output.resource;
                eval_params.pInDepth = &depth.resource;
                eval_params.pInMotionVectors = &motion_vectors.resource;
                eval_params.InJitterOffsetX = resources.jitter_offset_x;
                eval_params.InJitterOffsetY = resources.jitter_offset_y;
                eval_params.InRenderSubrectDimensions.Width = m_desc.render_width;
                eval_params.InRenderSubrectDimensions.Height = m_desc.render_height;
                eval_params.InReset = resources.reset ? 1 : 0;
                eval_params.InMVScaleX = resources.motion_vector_scale_x;
                eval_params.InMVScaleY = resources.motion_vector_scale_y;
                eval_params.InPreExposure = resources.pre_exposure;
                eval_params.InExposureScale = resources.exposure_scale;
                eval_params.InFrameTimeDeltaInMsec = resources.frame_time_delta_ms;
                if (resources.motion_vectors_reflections) {
                    eval_params.pInMotionVectorsReflections = &motion_vectors_reflections.resource;
                }
                if (resources.specular_hit_distance) {
                    eval_params.pInSpecularHitDistance = &specular_hit_distance.resource;
                    eval_params.pInWorldToViewMatrix = const_cast<float*>(resources.view_from_world->data());
                    eval_params.pInViewToClipMatrix = const_cast<float*>(resources.clip_from_view->data());
                }

                set_last_evaluate_result(NGX_VULKAN_EVALUATE_DLSSD_EXT(
                    reinterpret_cast<VkCommandBuffer>(native_handle.value()),
                    m_feature_handle,
                    m_params,
                    &eval_params
                ));
            }
        );
    }

private:
    /// Allocate the NGX parameter map and create the feature through a native command callback.
    void initialize()
    {
        validate_feature_desc();

        sgl::Device* device = m_ngx->device();
        m_vulkan_device = get_vulkan_device(device);
        FALCOR_CHECK(m_vulkan_device, "Vulkan DLSS-RR feature creation requires a VkDevice.");

        NVSDK_NGX_Result allocate_result = NVSDK_NGX_VULKAN_AllocateParameters(&m_params);
        FALCOR_CHECK(
            allocate_result == NVSDK_NGX_Result_Success && m_params,
            "Vulkan DLSS-RR parameter allocation failed: {}",
            ngx_result_to_string(static_cast<int32_t>(allocate_result))
        );

        ref<sgl::CommandEncoder> command_encoder = device->create_command_encoder();

        bool callback_called = false;
        std::string callback_error;
        NVSDK_NGX_Result create_result = NVSDK_NGX_Result_Fail;

        command_encoder->execute_callback(
            [&](sgl::NativeHandle native_handle)
            {
                callback_called = true;
                if (native_handle.type() != sgl::NativeHandleType::VkCommandBuffer || !native_handle.value()) {
                    callback_error = "Vulkan DLSS-RR feature creation requires an active VkCommandBuffer.";
                    return;
                }

                NVSDK_NGX_DLSSD_Create_Params create_params = make_create_params();
                create_result = NGX_VULKAN_CREATE_DLSSD_EXT1(
                    m_vulkan_device,
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

        FALCOR_CHECK(callback_called, "Vulkan DLSS-RR feature creation callback was not executed.");
        FALCOR_CHECK(callback_error.empty(), "{}", callback_error);
        FALCOR_CHECK(
            create_result == NVSDK_NGX_Result_Success && m_feature_handle,
            "Vulkan DLSS-RR feature creation failed: {}",
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

    VkDevice m_vulkan_device{VK_NULL_HANDLE};
    NVSDK_NGX_Parameter* m_params{nullptr};
    NVSDK_NGX_Handle* m_feature_handle{nullptr};
};

} // namespace

std::unique_ptr<DLSSRRFeature::Impl> make_vulkan_dlss_rr_feature_impl(ref<NGX> ngx, const DLSSRRFeatureDesc& desc)
{
    return std::make_unique<VulkanDLSSRRFeatureImpl>(std::move(ngx), desc);
}

} // namespace falcor::ngx

#endif // FALCOR_ENABLE_NGX
