# SPDX-License-Identifier: Apache-2.0

import pytest
import numpy as np
import slangpy as spy
import falcor2 as f2
import falcor2.testing.helpers as helpers


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_managed_buffer_upload_and_resize(device_type: spy.DeviceType):
    """Test ManagedBuffer upload, resize, and data preservation."""
    device = helpers.get_device(device_type)

    # Create a ManagedBuffer with initial size of 16 bytes
    initial_size = 16
    desc = spy.BufferDesc({"size": initial_size, "usage": spy.BufferUsage.shader_resource})
    managed_buffer = f2.ManagedBuffer(desc)

    assert managed_buffer.size() == initial_size

    # Set initial data (4 float32 values)
    initial_data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    managed_buffer.set_data(initial_data.tobytes(), 0)

    # Upload to GPU
    managed_buffer.update_buffer(device)

    # Read back and verify the data was uploaded correctly
    read_back_array = np.frombuffer(managed_buffer.buffer.to_numpy(), dtype=np.float32)
    np.testing.assert_array_equal(read_back_array, initial_data)

    # Resize the buffer to hold 8 float32 values (32 bytes)
    new_size = 32
    managed_buffer.resize(new_size)
    assert managed_buffer.size() == new_size

    # Set new data into the old part (first 4 floats should be preserved after we set them again)
    managed_buffer.set_data(initial_data.tobytes(), 0)

    # Set new data into the new part (next 4 floats)
    new_data = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32)
    managed_buffer.set_data(new_data.tobytes(), initial_data.nbytes)

    # Upload to GPU
    managed_buffer.update_buffer(device)

    # Read back and verify all data
    full_array = np.frombuffer(managed_buffer.buffer.to_numpy(), dtype=np.float32)

    expected_full = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=np.float32)
    np.testing.assert_array_equal(full_array, expected_full)

    # Verify old part is preserved
    np.testing.assert_array_equal(full_array[:4], initial_data)

    # Verify new part is correct
    np.testing.assert_array_equal(full_array[4:], new_data)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_managed_buffer_gpu_access(device_type: spy.DeviceType):
    """Test that ManagedBuffer data is accessible on the GPU via a shader.

    This test uses a shader to copy data from a ManagedBuffer to a separate
    RWByteAddressBuffer, demonstrating that the data is on the GPU and can be
    accessed by shaders.
    """
    device = helpers.get_device(device_type)

    # Create a ManagedBuffer and set some data
    buffer_size = 32  # 8 uint32 values
    desc = spy.BufferDesc({"size": buffer_size, "usage": spy.BufferUsage.shader_resource})
    managed_buffer = f2.ManagedBuffer(desc)

    # Set test data
    test_data = np.array(
        [
            0xDEADBEEF,
            0xCAFEBABE,
            0x12345678,
            0x87654321,
            0xABCDEF00,
            0x11223344,
            0x55667788,
            0x99AABBCC,
        ],
        dtype=np.uint32,
    )
    managed_buffer.set_data(test_data.tobytes(), 0)
    managed_buffer.update_buffer(device)

    # Create a destination buffer (RWByteAddressBuffer)
    dst_buffer = device.create_buffer(
        spy.BufferDesc(
            {
                "size": buffer_size,
                "usage": spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
            }
        )
    )

    # Load the shader module
    module = spy.Module(device.load_module("utils/test_managed_buffer.slang"))

    # Run the shader to copy data from ManagedBuffer to dst_buffer
    module["run_test"](
        src=f2.to_handle(managed_buffer),
        dst=dst_buffer,
        byte_size=buffer_size,
    )

    # Read back the destination buffer and verify the data was copied correctly
    read_back_array = np.frombuffer(dst_buffer.to_numpy(), dtype=np.uint32)
    np.testing.assert_array_equal(read_back_array, test_data)


if __name__ == "__main__":
    pytest.main([__file__])
