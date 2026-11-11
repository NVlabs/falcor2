# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Work around OptiX behavior on Linux NVIDIA driver 535."""

import ctypes
import sys
from functools import cache

import slangpy as spy


_CU_LIMIT_STACK_SIZE = 0
_MIN_CUDA_STACK_SIZE_FOR_OPTIX = 16384
_NVML_DRIVER_VERSION_BUFFER_SIZE = 80


@cache
def _is_linux_driver_535() -> bool:
    if sys.platform != "linux":
        return False

    try:
        nvml = ctypes.CDLL("libnvidia-ml.so.1")
        nvml.nvmlInit_v2.argtypes = []
        nvml.nvmlInit_v2.restype = ctypes.c_int
        nvml.nvmlSystemGetDriverVersion.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.c_uint]
        nvml.nvmlSystemGetDriverVersion.restype = ctypes.c_int
        nvml.nvmlShutdown.argtypes = []
        nvml.nvmlShutdown.restype = ctypes.c_int
        result = nvml.nvmlInit_v2()
        if result != 0:
            spy.log_warn(
                f"NVML initialization failed with error {result}; Linux driver 535 workarounds skipped."
            )
            return False
        try:
            version = ctypes.create_string_buffer(_NVML_DRIVER_VERSION_BUFFER_SIZE)
            result = nvml.nvmlSystemGetDriverVersion(version, len(version))
            if result != 0:
                spy.log_warn(
                    f"NVML driver query failed with error {result}; Linux driver 535 workarounds skipped."
                )
                return False
            return version.value.partition(b".")[0] == b"535"
        finally:
            nvml.nvmlShutdown()
    except (OSError, AttributeError) as exc:
        spy.log_warn(
            f"Cannot query NVIDIA driver version through NVML: {exc}; Linux driver 535 workarounds skipped."
        )
        return False


def ensure_optix_cuda_stack(device: spy.Device) -> None:
    """Grow the CUDA context stack before OptiX rendering on affected drivers.

    CUDA ordinarily grows its per-thread stack for a kernel launch. OptiX path
    tracing on Linux driver 535 can retain the 1 KiB context default instead.
    Deep MaterialX path tracing scenes compiled with fast floating point mode
    need at least 16 KiB on driver 535.
    """
    if device.info.type != spy.DeviceType.cuda or not _is_linux_driver_535():
        return

    try:
        cuda = ctypes.CDLL("libcuda.so.1")
    except OSError as exc:
        raise RuntimeError("Cannot load CUDA driver library for OptiX stack workaround.") from exc
    cuda.cuCtxGetLimit.argtypes = [ctypes.POINTER(ctypes.c_size_t), ctypes.c_int]
    cuda.cuCtxGetLimit.restype = ctypes.c_int
    cuda.cuCtxSetLimit.argtypes = [ctypes.c_int, ctypes.c_size_t]
    cuda.cuCtxSetLimit.restype = ctypes.c_int

    with device.cuda_context_scope():
        current_size = ctypes.c_size_t()
        result = cuda.cuCtxGetLimit(ctypes.byref(current_size), _CU_LIMIT_STACK_SIZE)
        if result != 0:
            raise RuntimeError(
                f"cuCtxGetLimit(CU_LIMIT_STACK_SIZE) failed with CUDA error {result}."
            )
        if current_size.value < _MIN_CUDA_STACK_SIZE_FOR_OPTIX:
            result = cuda.cuCtxSetLimit(_CU_LIMIT_STACK_SIZE, _MIN_CUDA_STACK_SIZE_FOR_OPTIX)
            if result != 0:
                raise RuntimeError(
                    f"cuCtxSetLimit(CU_LIMIT_STACK_SIZE) failed with CUDA error {result}."
                )
