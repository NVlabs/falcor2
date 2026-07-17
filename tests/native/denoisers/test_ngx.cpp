// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "testing.h"

#include "falcor2/core/config.h"

#if FALCOR_ENABLE_NGX

#include "falcor2/denoisers/dlss_rr_feature.h"
#include "falcor2/denoisers/ngx.h"

#include <sgl/device/command.h>
#include <sgl/device/device.h>
#include <sgl/device/resource.h>

#include <exception>
#include <string>
#include <utility>
#include <vector>

#endif

using namespace falcor;

TEST_SUITE_BEGIN("denoisers");

#if FALCOR_ENABLE_NGX
namespace {

std::vector<sgl::DeviceType> ngx_device_types()
{
#if SGL_WINDOWS
    return {sgl::DeviceType::d3d12, sgl::DeviceType::vulkan, sgl::DeviceType::cuda};
#elif SGL_LINUX
    return {sgl::DeviceType::vulkan, sgl::DeviceType::cuda};
#else
    return {};
#endif
}

sgl::DeviceDesc make_ngx_device_desc(sgl::DeviceType device_type)
{
    sgl::DeviceDesc desc = falcor::testing::make_test_device_desc(device_type);

    if (device_type == sgl::DeviceType::vulkan) {
        auto ngx_info = ngx::get_vulkan_pre_device_info();
        desc.additional_vulkan_instance_extensions = ngx_info.required_vulkan_instance_extensions;
        desc.additional_vulkan_device_extensions = ngx_info.required_vulkan_device_extensions;
    }

    return desc;
}

void check_dlssd_optimal_settings(
    const ngx::DLSSDOptimalSettings& settings,
    uint32_t target_width,
    uint32_t target_height
)
{
    CHECK(settings.render_width > 0);
    CHECK(settings.render_height > 0);
    CHECK(settings.render_width <= target_width);
    CHECK(settings.render_height <= target_height);
    CHECK(settings.min_render_width <= settings.render_width);
    CHECK(settings.min_render_height <= settings.render_height);
    CHECK(settings.max_render_width >= settings.render_width);
    CHECK(settings.max_render_height >= settings.render_height);
}

ref<sgl::Texture>
create_dlss_rr_texture(sgl::Device* device, uint32_t width, uint32_t height, sgl::Format format, std::string label)
{
    return device->create_texture({
        .format = format,
        .width = width,
        .height = height,
        .usage = sgl::TextureUsage::shader_resource | sgl::TextureUsage::unordered_access,
        .label = std::move(label),
    });
}

struct DLSSRRSmokeResources {
    ref<sgl::Texture> diffuse_albedo;
    ref<sgl::Texture> specular_albedo;
    ref<sgl::Texture> normals;
    ref<sgl::Texture> roughness;
    ref<sgl::Texture> color;
    ref<sgl::Texture> depth;
    ref<sgl::Texture> motion_vectors;
    ref<sgl::Texture> output;
    ref<sgl::Texture> motion_vectors_reflections;

    ngx::DLSSRREvaluateDesc desc() const
    {
        return {
            .diffuse_albedo = diffuse_albedo,
            .specular_albedo = specular_albedo,
            .normals = normals,
            .roughness = roughness,
            .color = color,
            .depth = depth,
            .motion_vectors = motion_vectors,
            .output = output,
            .motion_vectors_reflections = motion_vectors_reflections,
            .reset = true,
        };
    }
};

DLSSRRSmokeResources create_dlss_rr_smoke_resources(
    sgl::Device* device,
    uint32_t render_width,
    uint32_t render_height,
    uint32_t target_width,
    uint32_t target_height
)
{
    DLSSRRSmokeResources resources{
        .diffuse_albedo = create_dlss_rr_texture(
            device,
            render_width,
            render_height,
            sgl::Format::rgba16_float,
            "dlss_rr_diffuse_albedo"
        ),
        .specular_albedo = create_dlss_rr_texture(
            device,
            render_width,
            render_height,
            sgl::Format::rgba16_float,
            "dlss_rr_specular_albedo"
        ),
        .normals
        = create_dlss_rr_texture(device, render_width, render_height, sgl::Format::rgba16_float, "dlss_rr_normals"),
        .roughness
        = create_dlss_rr_texture(device, render_width, render_height, sgl::Format::r16_float, "dlss_rr_roughness"),
        .color
        = create_dlss_rr_texture(device, render_width, render_height, sgl::Format::rgba16_float, "dlss_rr_color"),
        .depth = create_dlss_rr_texture(device, render_width, render_height, sgl::Format::r32_float, "dlss_rr_depth"),
        .motion_vectors = create_dlss_rr_texture(
            device,
            render_width,
            render_height,
            sgl::Format::rg16_float,
            "dlss_rr_motion_vectors"
        ),
        .output
        = create_dlss_rr_texture(device, target_width, target_height, sgl::Format::rgba16_float, "dlss_rr_output"),
        .motion_vectors_reflections = create_dlss_rr_texture(
            device,
            render_width,
            render_height,
            sgl::Format::rg16_float,
            "dlss_rr_motion_vectors_reflections"
        ),
    };

    ref<sgl::CommandEncoder> command_encoder = device->create_command_encoder();
    command_encoder->clear_texture_float(resources.diffuse_albedo, {}, float4(0.5f, 0.5f, 0.5f, 1.f));
    command_encoder->clear_texture_float(resources.specular_albedo, {}, float4(0.04f, 0.04f, 0.04f, 1.f));
    command_encoder->clear_texture_float(resources.normals, {}, float4(0.f, 0.f, 1.f, 0.f));
    command_encoder->clear_texture_float(resources.roughness, {}, float4(0.5f));
    command_encoder->clear_texture_float(resources.color, {}, float4(0.1f, 0.1f, 0.1f, 1.f));
    command_encoder->clear_texture_float(resources.depth, {}, float4(1.f));
    command_encoder->clear_texture_float(resources.motion_vectors, {}, float4(0.f));
    command_encoder->clear_texture_float(resources.motion_vectors_reflections, {}, float4(0.f));
    command_encoder->clear_texture_float(resources.output, {}, float4(0.f));
    ref<sgl::CommandBuffer> command_buffer = command_encoder->finish();
    device->submit_command_buffer(command_buffer);
    device->wait();

    return resources;
}

void run_ngx_smoke_test(sgl::DeviceType device_type)
{
    ref<sgl::Device> device = sgl::Device::create(make_ngx_device_desc(device_type));
    REQUIRE(device);

    ref<ngx::NGX> ngx_context;
    try {
        ngx_context = ngx::NGX::get(device.get());
    } catch (const std::exception& e) {
        INFO("NGX initialization is not available on this test device: " << e.what());
        device->close();
        return;
    }

    {
        ref<ngx::NGX> same_ngx_context = ngx::NGX::get(device.get());
        CHECK(ngx_context);
        CHECK(ngx_context.get() == same_ngx_context.get());
        CHECK(ngx_context->device() == device.get());

        const ngx::NGXInfo& info = ngx_context->info();
        if (!info.dlssd_supported) {
            INFO("NGX SuperSamplingDenoising is not supported on this test device.");
        } else {
            const uint32_t target_width = 1280;
            const uint32_t target_height = 720;
            ngx::DLSSDOptimalSettings settings
                = ngx_context->get_dlssd_optimal_settings(target_width, target_height, ngx::QualityMode::quality);
            check_dlssd_optimal_settings(settings, target_width, target_height);

            ngx::DLSSRRFeatureDesc feature_desc{
                .quality = ngx::QualityMode::quality,
                .render_width = settings.render_width,
                .render_height = settings.render_height,
                .target_width = target_width,
                .target_height = target_height,
            };
            ref<ngx::DLSSRRFeature> feature = ngx_context->create_dlss_rr_feature(feature_desc);
            CHECK(feature);
            CHECK(feature->ngx() == ngx_context.get());
            CHECK(feature->device() == device.get());
            CHECK(feature->desc().render_width == settings.render_width);
            CHECK(feature->desc().target_width == target_width);

            DLSSRRSmokeResources resources = create_dlss_rr_smoke_resources(
                device.get(),
                feature_desc.render_width,
                feature_desc.render_height,
                feature_desc.target_width,
                feature_desc.target_height
            );

            CHECK_NOTHROW(feature->evaluate(resources.desc()));
            device->wait();
            CHECK(feature->last_evaluate_result() == ngx::Result::success);

            ref<sgl::CommandEncoder> command_encoder = device->create_command_encoder();
            CHECK_NOTHROW(feature->evaluate(command_encoder.get(), resources.desc()));
            ref<sgl::CommandBuffer> command_buffer = command_encoder->finish();
            feature = nullptr;
            device->submit_command_buffer(command_buffer);
            device->wait();
        }
    }

    device->close();
}

} // namespace

// Note: Not using 'TEST_CASE_GPU' as we want to create new devices that have NGX support.
TEST_CASE("ngx_dlss_rr_native_smoke")
{
    for (sgl::DeviceType device_type : ngx_device_types()) {
        SUBCASE(enum_to_string(device_type).c_str())
        {
            run_ngx_smoke_test(device_type);
        }
    }
}
#endif

TEST_SUITE_END();
