# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import numpy as np
import slangpy as spy
import falcor2 as f2
import falcor2.testing.helpers as helpers


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_buffer_handle_gpu_access(device_type: spy.DeviceType) -> None:
    device = helpers.get_device(device_type)

    test_data = np.array([0xDEADBEEF, 0xCAFEBABE, 0x12345678, 0x87654321], dtype=np.uint32)
    src_buffer = device.create_buffer(usage=spy.BufferUsage.shader_resource, data=test_data)
    dst_buffer = device.create_buffer(
        size=test_data.nbytes,
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
    )
    module = spy.Module(device.load_module("utils/test_buffer_handle.slang"))
    module["run_test"](
        src=f2.to_handle(src_buffer),
        dst=dst_buffer,
        byte_size=test_data.nbytes,
    )
    np.testing.assert_array_equal(np.frombuffer(dst_buffer.to_numpy(), dtype=np.uint32), test_data)


if __name__ == "__main__":
    pytest.main([__file__])
