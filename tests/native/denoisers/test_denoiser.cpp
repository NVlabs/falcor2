// SPDX-License-Identifier: Apache-2.0

#include "testing.h"

#include "falcor2/denoisers/optix_denoiser.h"

#include <sgl/device/device.h>
#include <sgl/device/resource.h>

#include <random>

using namespace falcor;

TEST_SUITE_BEGIN("denoisers");

// Helper function to create a CUDA device for denoiser tests
ref<sgl::Device> create_cuda_device()
{
    sgl::DeviceDesc desc{
        .type = sgl::DeviceType::cuda,
        .enable_debug_layers = true,
        .compiler_options = {
            .include_paths = {
                falcor::testing::project_directory() / "slang",
                falcor::testing::project_directory() / "tests" / "native",
                falcor::testing::project_directory() / "external" / "slangpy" / "slangpy" / "slang",
            },
        }
    };
    return sgl::Device::create(desc);
}

// Helper function to create a checkerboard pattern in a buffer
void create_checkerboard_pattern(
    float* data,
    uint32_t width,
    uint32_t height,
    uint32_t channels,
    uint32_t checker_size = 8
)
{
    for (uint32_t y = 0; y < height; ++y) {
        for (uint32_t x = 0; x < width; ++x) {
            bool checker = ((x / checker_size) + (y / checker_size)) % 2 == 0;
            float value = checker ? 1.0f : 0.0f;

            for (uint32_t c = 0; c < channels; ++c) {
                data[(y * width + x) * channels + c] = value;
            }
        }
    }
}

// Helper function to add noise to an image
void add_noise(float* data, uint32_t width, uint32_t height, uint32_t channels, float noise_strength = 0.9f)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-noise_strength, noise_strength);

    for (uint32_t y = 0; y < height; y += 2) {
        for (uint32_t x = 0; x < width; x += 2) {
            for (uint32_t c = 0; c < channels; ++c) {
                uint32_t idx = (y * width + x) * channels + c;
                data[idx] = std::clamp(data[idx] + dis(gen), 0.0f, 1.0f);
            }
        }
    }
}

// Helper function to calculate mean squared error between two images
float calculate_mse(const float* img1, const float* img2, uint32_t width, uint32_t height, uint32_t channels)
{
    float mse = 0.0f;
    uint32_t num_pixels = width * height * channels;

    for (uint32_t i = 0; i < num_pixels; ++i) {
        float diff = img1[i] - img2[i];
        mse += diff * diff;
    }

    return mse / num_pixels;
}

TEST_CASE("optix_denoiser_creation_destruction")
{
    sgl::ref<sgl::Device> device = create_cuda_device();
    CHECK(device);

    {
        // Test denoiser creation and destruction
        OptixDenoiserDesc desc{
            .model_kind = optix::ModelKind::aov,
            .alpha_mode = optix::AlphaMode::copy,
            .albedo_guide_layer = false,
            .normal_guide_layer = false,
            .max_width = 512,
            .max_height = 512
        };

        auto denoiser = make_ref<OptixDenoiser>(device, desc);
        CHECK(denoiser);
    }

    device->close();
}


TEST_CASE("optix_denoiser_denoise_checkerboard")
{
    sgl::ref<sgl::Device> device = create_cuda_device();
    CHECK(device);

    // Test parameters
    const uint32_t width = 256;
    const uint32_t height = 256;
    const uint32_t channels = 3; // RGB
    const uint32_t pixel_size = channels * sizeof(float);
    const uint32_t row_stride = width * pixel_size;
    const uint32_t buffer_size = height * row_stride;

    // Create denoiser
    OptixDenoiserDesc desc{
        .model_kind = optix::ModelKind::aov,
        .alpha_mode = optix::AlphaMode::copy,
        .albedo_guide_layer = false,
        .normal_guide_layer = false,
        .max_width = width,
        .max_height = height
    };

    auto denoiser = make_ref<OptixDenoiser>(device, desc);
    CHECK(denoiser);

    // Create checkerboard pattern on CPU
    std::vector<float> original_data(width * height * channels);
    create_checkerboard_pattern(original_data.data(), width, height, channels);

    // Create noisy version
    std::vector<float> noisy_data = original_data;
    add_noise(noisy_data.data(), width, height, channels);

    // Create buffers for original, noisy, and denoised images
    auto original_buffer = device->create_buffer(
        {.element_count = width * height * channels,
         .struct_size = sizeof(float),
         .usage = sgl::BufferUsage::shader_resource | sgl::BufferUsage::unordered_access,
         .data = original_data.data(),
         .data_size = original_data.size() * sizeof(float)}
    );

    auto noisy_buffer = device->create_buffer(
        {.element_count = width * height * channels,
         .struct_size = sizeof(float),
         .usage = sgl::BufferUsage::shader_resource | sgl::BufferUsage::unordered_access,
         .data = noisy_data.data(),
         .data_size = noisy_data.size() * sizeof(float)}
    );

    auto denoised_buffer = device->create_buffer(
        {.element_count = width * height * channels,
         .struct_size = sizeof(float),
         .usage = sgl::BufferUsage::shader_resource | sgl::BufferUsage::unordered_access}
    );

    // Set up denoiser parameters
    optix::DenoiserParams params{
        .hdr_intensity = std::nullopt,
        .blend_factor = 0.0f,
        .hdr_average_color = std::nullopt,
        .temporal_mode_use_previous_layers = 0
    };

    // Set up empty guide layer (no auxiliary data)
    optix::DenoiserGuideLayer guide_layer{};

    // Set up denoising layer
    std::vector<optix::DenoiserLayer> layers(1);
    layers[0]
        = {.input
           = {.buffer = noisy_buffer,
              .width = width,
              .height = height,
              .row_stride_in_bytes = row_stride,
              .pixel_stride_in_bytes = static_cast<uint16_t>(pixel_size),
              .format = optix::PixelFormat::float3},
           .previous_output = {}, // Empty for non-temporal mode
           .output
           = {.buffer = denoised_buffer,
              .width = width,
              .height = height,
              .row_stride_in_bytes = row_stride,
              .pixel_stride_in_bytes = static_cast<uint16_t>(pixel_size),
              .format = optix::PixelFormat::float3},
           .type = optix::DenoiserAOVType::beauty};

    // Perform denoising
    CHECK_NOTHROW({ denoiser->denoise(params, guide_layer, layers); });

    // Download denoised result
    std::vector<float> denoised_data(original_data.size());
    denoised_buffer->get_data(denoised_data.data(), buffer_size);

    // Verify that denoising improved the image quality
    // Calculate MSE between original and noisy images
    float mse_noisy = calculate_mse(original_data.data(), noisy_data.data(), width, height, channels);

    // Calculate MSE between original and denoised images
    float mse_denoised = calculate_mse(original_data.data(), denoised_data.data(), width, height, channels);

    // Denoising should reduce the MSE (improve quality)
    // MESSAGE("MSE noisy: ", mse_noisy, ", MSE denoised: ", mse_denoised);
    CHECK(mse_denoised < mse_noisy);

    // Also check that the improvement is significant (at least 20% reduction)
    CHECK(mse_denoised < 0.8f * mse_noisy);

    device->close();
}

TEST_SUITE_END();
