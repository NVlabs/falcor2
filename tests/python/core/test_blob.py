# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import falcor2 as f2


def test_blob_binary_data_survives_properties_storage() -> None:
    data = b"\x00arbitrary binary data\xff\x00"
    properties = f2.Properties({"payload": f2.Blob(data, name="payload.bin")})

    blob = properties["payload"]
    assert isinstance(blob, f2.Blob)
    assert blob.name == "payload.bin"
    assert blob.data == data
