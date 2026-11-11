# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import numpy as np
import slangpy as spy
import falcor2 as f2
import falcor2.testing.helpers as helpers


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_buffer_ptr_functional_access(device_type: spy.DeviceType) -> None:
    device = helpers.get_device(device_type)

    test_data = np.array([0xDEADBEEF, 0xCAFEBABE, 0x12345678, 0x87654321], dtype=np.uint32)
    src_buffer = device.create_buffer(usage=spy.BufferUsage.shader_resource, data=test_data)
    dst_buffer = device.create_buffer(
        size=test_data.nbytes,
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
    )
    module = spy.Module(device.load_module("utils/test_buffer_ptr_access.slang"))
    module["run_test"](
        src=f2.to_buffer_ptr(src_buffer),
        dst=dst_buffer,
        byte_size=test_data.nbytes,
    )
    np.testing.assert_array_equal(np.frombuffer(dst_buffer.to_numpy(), dtype=np.uint32), test_data)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_divergent_buffer_ptrs(device_type: spy.DeviceType) -> None:
    device = helpers.get_device(device_type)
    data = np.arange(64, dtype=np.uint32).reshape(8, 8)
    buffers = [
        device.create_buffer(usage=spy.BufferUsage.shader_resource, data=row) for row in data
    ]
    pointers = [f2.to_buffer_ptr(buffer).data for buffer in buffers]
    indices = np.random.default_rng(42).integers(0, len(buffers), size=1024)
    offsets = np.arange(len(indices), dtype=np.uint32) % data.shape[1]
    packed = np.array([pointers[index] for index in indices], dtype=np.uint64)
    module = spy.Module.load_from_file(device, "utils/test_buffer_ptr_access.slang")
    result = module.load_divergent(
        pointer_data=spy.Tensor.from_numpy(device, packed),
        byte_offset=spy.Tensor.from_numpy(device, offsets * 4),
        _result="numpy",
    )
    expected = data[indices, offsets]
    np.testing.assert_array_equal(result, np.repeat(expected[:, None], 6, axis=1).reshape(-1, 2, 3))


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_rw_buffer_ptr_gpu_access(device_type: spy.DeviceType) -> None:
    device = helpers.get_device(device_type)
    data = np.arange(129 * 3, dtype=np.float32).reshape(-1, 3)
    expected = np.full((len(data) + 4, 3), -1, dtype=np.float32)
    src = device.create_buffer(usage=spy.BufferUsage.shader_resource, data=data)
    dst = device.create_buffer(usage=spy.BufferUsage.unordered_access, data=expected)
    src_ptr, dst_ptr = f2.to_buffer_ptr(src), f2.to_rw_buffer_ptr(dst)
    assert isinstance(dst_ptr, f2.RWBufferPtr)
    copy = device.create_compute_kernel(
        program=device.load_program("utils/test_buffer_ptr_access.slang", ["copy_rw_vectors"])
    )
    update = device.create_compute_kernel(
        program=device.load_program("utils/test_buffer_ptr_access.slang", ["update_rw_vectors"])
    )
    encoder = device.create_command_encoder()
    encoder.set_buffer_state(src, spy.ResourceState.shader_resource)
    encoder.set_buffer_state(dst, spy.ResourceState.unordered_access)
    copy.dispatch(
        (len(data), 1, 1), command_encoder=encoder, src=src_ptr, dst=dst_ptr, count=len(data)
    )
    encoder.set_buffer_state(dst, spy.ResourceState.unordered_access)
    update.dispatch((len(data), 1, 1), command_encoder=encoder, buffer=dst_ptr, count=len(data))
    device.submit_command_buffer(encoder.finish())
    device.wait()
    expected[2:-2] = data + np.array([1, 2, 3], dtype=np.float32)
    np.testing.assert_array_equal(dst.to_numpy().view(np.float32).reshape(-1, 3), expected)


if __name__ == "__main__":
    pytest.main([__file__])
