// SPDX-License-Identifier: Apache-2.0

#include "falcor2/denoisers/ngx.h"

#include "falcor2/denoisers/dlss_g_feature.h"
#include "falcor2/denoisers/dlss_rr_feature.h"
#include "falcor2/denoisers/dlss_sr_feature.h"

#if FALCOR_ENABLE_NGX
#include "falcor2/denoisers/ngx_impl.h"
#endif

#include "falcor2/falcor2.h"

#include <sgl/device/device.h>

#include <map>
#include <utility>

namespace falcor::ngx {

static std::map<sgl::Device*, NGX*>& ngx_contexts_by_device()
{
    // This registry is non-owning. It prevents duplicate NGX initialization for
    // a live device, but NGX lifetime is still controlled entirely by ref<NGX>.
    static std::map<sgl::Device*, NGX*> contexts;
    return contexts;
}

ref<NGX> NGX::get(sgl::Device* device)
{
    FALCOR_CHECK(device, "NGX requires a device.");

    auto& contexts = ngx_contexts_by_device();
    auto it = contexts.find(device);
    if (it != contexts.end())
        return ref<NGX>(it->second);

    ref<NGX> context(new NGX(ref<sgl::Device>(device)));
    contexts.emplace(context->device(), context.get());
    return context;
}

NGX::NGX(ref<sgl::Device> device)
    : m_device(std::move(device))
{
    FALCOR_CHECK(m_device, "NGX requires a device.");
#if FALCOR_ENABLE_NGX
    m_impl = Impl::create(m_device).release();
#else
    FALCOR_THROW("Falcor2 was built without NGX. Reconfigure with FALCOR_ENABLE_NGX=ON.");
#endif
}

NGX::~NGX()
{
    auto& contexts = ngx_contexts_by_device();
    auto it = contexts.find(m_device.get());
    if (it != contexts.end() && it->second == this)
        contexts.erase(it);

#if FALCOR_ENABLE_NGX
    if (m_device)
        m_device->wait();
    delete m_impl;
#endif
}

#if FALCOR_ENABLE_NGX
NGX::Impl* get_ngx_impl(NGX* ngx)
{
    FALCOR_CHECK(ngx, "NGX implementation access requires an NGX object.");
    return ngx->m_impl;
}
#endif

const NGXInfo& NGX::info() const
{
#if FALCOR_ENABLE_NGX
    FALCOR_ASSERT(m_impl);
    return m_impl->info();
#else
    FALCOR_THROW("Falcor2 was built without NGX. Reconfigure with FALCOR_ENABLE_NGX=ON.");
#endif
}

DLSSDOptimalSettings
NGX::get_dlssd_optimal_settings(uint32_t target_width, uint32_t target_height, QualityMode quality) const
{
#if FALCOR_ENABLE_NGX
    FALCOR_ASSERT(m_impl);
    return m_impl->get_dlssd_optimal_settings(target_width, target_height, quality);
#else
    FALCOR_UNUSED(target_width);
    FALCOR_UNUSED(target_height);
    FALCOR_UNUSED(quality);
    FALCOR_THROW("Falcor2 was built without NGX. Reconfigure with FALCOR_ENABLE_NGX=ON.");
#endif
}

DLSSOptimalSettings
NGX::get_dlss_optimal_settings(uint32_t target_width, uint32_t target_height, QualityMode quality) const
{
#if FALCOR_ENABLE_NGX
    FALCOR_ASSERT(m_impl);
    return m_impl->get_dlss_optimal_settings(target_width, target_height, quality);
#else
    FALCOR_UNUSED(target_width);
    FALCOR_UNUSED(target_height);
    FALCOR_UNUSED(quality);
    FALCOR_THROW("Falcor2 was built without NGX. Reconfigure with FALCOR_ENABLE_NGX=ON.");
#endif
}

ref<DLSSRRFeature> NGX::create_dlss_rr_feature(const DLSSRRFeatureDesc& desc)
{
#if FALCOR_ENABLE_NGX
    return make_ref<DLSSRRFeature>(ref<NGX>(this), desc);
#else
    FALCOR_UNUSED(desc);
    FALCOR_THROW("Falcor2 was built without NGX. Reconfigure with FALCOR_ENABLE_NGX=ON.");
#endif
}

ref<DLSSSRFeature> NGX::create_dlss_sr_feature(const DLSSSRFeatureDesc& desc)
{
#if FALCOR_ENABLE_NGX
    return make_ref<DLSSSRFeature>(ref<NGX>(this), desc);
#else
    FALCOR_UNUSED(desc);
    FALCOR_THROW("Falcor2 was built without NGX. Reconfigure with FALCOR_ENABLE_NGX=ON.");
#endif
}

ref<DLSSGFeature> NGX::create_dlss_g_feature(const DLSSGFeatureDesc& desc)
{
#if FALCOR_ENABLE_NGX
    return make_ref<DLSSGFeature>(ref<NGX>(this), desc);
#else
    FALCOR_UNUSED(desc);
    FALCOR_THROW("Falcor2 was built without NGX. Reconfigure with FALCOR_ENABLE_NGX=ON.");
#endif
}

NGXInfo get_vulkan_pre_device_info()
{
    NGXInfo info;
#if FALCOR_ENABLE_NGX
    query_vulkan_required_extensions(info);
#else
    FALCOR_THROW("Falcor2 was built without NGX. Reconfigure with FALCOR_ENABLE_NGX=ON.");
#endif
    return info;
}

} // namespace falcor::ngx
