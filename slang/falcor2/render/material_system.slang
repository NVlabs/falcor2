// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing render;

public struct MaterialSystem
{
    public StructuredBuffer<MaterialData> material_data;

    public MaterialData get_material_data(MaterialID id) { return material_data[uint(id)]; }

    public IMaterial get_material(MaterialID id)
    {
        let data = get_material_data(id);
        return createDynamicObject<IMaterial, MaterialData>(data.header.type_id, data);
    }
};
