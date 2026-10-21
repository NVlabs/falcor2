// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "nanobind.h"
#include "falcor2/streaming/video_encoder.h"
#include <sgl/device/device.h>
#include <sgl/device/resource.h>

FALCOR_PY_EXPORT(streaming_video_encoder)
{
    namespace streaming = falcor::streaming;
    using namespace falcor::streaming;
    using namespace nb::literals;

    nb::module_ module = nb::module_::import_("falcor2.streaming");

    nb::sgl_enum<InputTransfer>(module, "InputTransfer", D(streaming, InputTransfer));
    nb::sgl_enum<VideoEncoderComplexity>(module, "VideoEncoderComplexity", D(streaming, VideoEncoderComplexity));

    nb::class_<VideoEncoderConfig>(module, "VideoEncoderConfig", D(streaming, VideoEncoderConfig))
        .def(nb::init<>())
        .DEF_RW_2(streaming, VideoEncoderConfig, width)
        .DEF_RW_2(streaming, VideoEncoderConfig, height)
        .DEF_RW_2(streaming, VideoEncoderConfig, fps)
        .DEF_RW_2(streaming, VideoEncoderConfig, target_bitrate)
        .DEF_RW_2(streaming, VideoEncoderConfig, max_bitrate)
        .DEF_RW_2(streaming, VideoEncoderConfig, max_frame_slots)
        .DEF_RW_2(streaming, VideoEncoderConfig, complexity)
        .DEF_RW_2(streaming, VideoEncoderConfig, thread_count)
        .DEF_RW_2(streaming, VideoEncoderConfig, library_path);

    nb::class_<EncodedFrame>(module, "EncodedFrame", D(streaming, EncodedFrame))
        .DEF_RO_2(streaming, EncodedFrame, frame_id)
        .DEF_RO_2(streaming, EncodedFrame, pts_us)
        .DEF_RO_2(streaming, EncodedFrame, duration_us)
        .DEF_RO_2(streaming, EncodedFrame, keyframe)
        .DEF_RO_2(streaming, EncodedFrame, gpu_prepare_ms)
        .DEF_RO_2(streaming, EncodedFrame, encode_wall_ms)
        .DEF_RO_2(streaming, EncodedFrame, submit_to_output_ms)
        .DEF_PROP_RO_2(streaming, EncodedFrame, byte_size)
        .def_prop_ro(
            "data",
            [](const EncodedFrame& frame)
            {
                return nb::bytes(frame.data.data(), frame.data.size());
            },
            D(streaming, EncodedFrame, data)
        );

    nb::class_<VideoEncoder>(module, "VideoEncoder", D(streaming, VideoEncoder))
        .def_static(
            "check_available",
            &VideoEncoder::check_available,
            "library_path"_a = "",
            D(streaming, VideoEncoder, check_available)
        )
        .def(
            nb::init<falcor::ref<sgl::Device>, VideoEncoderConfig>(),
            "device"_a,
            "config"_a,
            D(streaming, VideoEncoder, VideoEncoder)
        )
        .def(
            "try_submit",
            &VideoEncoder::try_submit,
            "texture"_a,
            "frame_id"_a,
            "pts_us"_a,
            "transfer"_a,
            D(streaming, VideoEncoder, try_submit)
        )
        .def("poll_output", &VideoEncoder::poll_output, D(streaming, VideoEncoder, poll_output))
        .def("request_keyframe", &VideoEncoder::request_keyframe, D(streaming, VideoEncoder, request_keyframe))
        .def(
            "set_target_bitrate",
            &VideoEncoder::set_target_bitrate,
            "bits_per_second"_a,
            D(streaming, VideoEncoder, set_target_bitrate)
        )
        .def("close", &VideoEncoder::close, D(streaming, VideoEncoder, close));
}
