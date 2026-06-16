// SPDX-License-Identifier: Apache-2.0

struct $MxStackDataType
{
    float3 bsdf_weights[$MxBsdfCount] = {};
    float3 bsdf_transmission_scales[$MxBsdfCount] = {};
    float3 layer_weights[$MxLayerCount] = {};
    float3 layer_transmission_scales[$MxLayerCount] = {};
    float3 mix_values[$MxLayerCount] = {};
    float3 bsdf_n[$MxBsdfCount] = {};
    float3 bsdf_t[$MxBsdfCount] = {};
    $MxBsdfFramesLine$MxBsdfMemberLines void set_bsdf<TBSDF : ILayerableBSDF>(int bsdf_index, TBSDF bsdf)
    {
        switch (bsdf_index) {
            $MxSetBsdfCases
        }
    }
    void init_vdf_transmission_scales() { $MxInitVdfBsdfLines$MxInitVdfLayerLines }
}
