// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "usd_importer.h"
#include "usd_importer_context.h"
#include "usd_importer_materialx.h"
#include "usd_importer_utils.h"
#include "usd_importer_macros.h"

#include "falcor2/core/types.h"

BEGIN_DISABLE_USD_WARNINGS
#include <pxr/usd/ar/resolver.h>
#include <pxr/usd/ar/resolverContextBinder.h>

#include <pxr/base/tf/diagnosticMgr.h>
#include <pxr/base/tf/errorMark.h>
#include <pxr/base/work/withScopedParallelism.h>

#include <pxr/usd/sdf/layer.h>

#include <pxr/usd/usd/common.h>
#include <pxr/usd/usd/editTarget.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/sdf/layer.h>

#include <pxr/usd/usdGeom/metrics.h>
#include <pxr/usd/usdGeom/tokens.h>
#include <pxr/usd/usdGeom/xform.h>
#include <pxr/usd/usdGeom/xformable.h>
#include <pxr/usd/usdGeom/camera.h>

#include <pxr/usd/usdLux/boundableLightBase.h>
#include <pxr/usd/usdLux/nonboundableLightBase.h>
END_DISABLE_USD_WARNINGS

#include <stdexcept>
#include <string>
#include <memory>
#include <mutex>
#include <vector>
#include <algorithm>
#include "fmt/format.h"

namespace falcor {
namespace usd_importer {

using namespace pxr;

class DiagDelegate : public TfDiagnosticMgr::Delegate {
public:
    void IssueFatalError(const TfCallContext& context, const std::string& msg) override
    {
        // Ensure proper string construction to avoid dangling pointers
        std::string error_message = msg;
        error_message += " ";
        error_message += std::string(context.GetPrettyFunction());
        throw std::runtime_error(error_message);
    }

    void IssueError(const TfError& err) override
    {
        std::string message = formatMessage(&err);
        sgl::log_error("USD Error: {}", message);
    }

    void IssueWarning(const TfWarning& warning) override
    {
        std::string message = formatMessage(&warning);
        if (should_suppress_warning(message))
            return;
        sgl::log_warn("USD Warning: {}", message);
    }

    void IssueStatus(const TfStatus& status) override { sgl::log_info("USD Info: {}", formatMessage(&status)); }

private:
    static bool should_suppress_warning(const std::string& message)
    {
        // Compatibility with legacy assets:
        // - Many of our USD assets were authored/validated with older USD versions where material bindings were
        //   commonly authored without explicitly applying UsdShadeMaterialBindingAPI on every bound prim.
        // - In newer USD releases (roughly from the 21.x era, around 21.08/21.11 and later), validation became
        //   stricter and emits the warning below when MaterialBindingAPI is not applied.
        // - Those assets are intentionally kept as-is, so this warning is expected noise for us.
        if (message.find("but MaterialBindingAPI is not applied on the prim") != std::string::npos)
            return true;

        // This warning is expected for .mtlx references because they are rewritten by process_mtlx_references().
        const bool is_mtlx_open_warning = message.find("Could not open asset @") != std::string::npos
            && message.find(".mtlx") != std::string::npos
            && message.find("Cannot determine file format for @") != std::string::npos;
        return is_mtlx_open_warning;
    }

    std::string formatMessage(const TfDiagnosticBase* elt) { return elt->GetCommentary(); }
};

// USD delegates are process-wide: overlapping imports must share one logging delegate.
class ScopedDiagnosticDelegate {
public:
    ScopedDiagnosticDelegate()
    {
        auto& state = shared_state();
        std::lock_guard lock(state.mutex);
        if (state.users == 0)
            TfDiagnosticMgr::GetInstance().AddDelegate(&state.delegate);
        ++state.users;
    }

    ~ScopedDiagnosticDelegate()
    {
        auto& state = shared_state();
        std::lock_guard lock(state.mutex);
        if (--state.users == 0)
            TfDiagnosticMgr::GetInstance().RemoveDelegate(&state.delegate);
    }

    ScopedDiagnosticDelegate(const ScopedDiagnosticDelegate&) = delete;
    ScopedDiagnosticDelegate& operator=(const ScopedDiagnosticDelegate&) = delete;

private:
    struct SharedState {
        std::mutex mutex;
        size_t users = 0;
        DiagDelegate delegate;
    };

    static SharedState& shared_state()
    {
        static SharedState state;
        return state;
    }
};

UVOrigin get_uv_origin(const UsdStageRefPtr& stage)
{
    static const std::string key = "falcor:uvOriginOverride";
    const VtDictionary& custom_layer_data = stage->GetRootLayer()->GetCustomLayerData();
    auto it = custom_layer_data.find(key);
    if (it == custom_layer_data.end())
        return UVOrigin::lower_left;

    if (!it->second.IsHolding<std::string>()) {
        sgl::log_warn("Ignoring USD customLayerData '{}': expected a string.", key);
        return UVOrigin::lower_left;
    }

    const std::string& value = it->second.UncheckedGet<std::string>();
    if (value == "upperLeft")
        return UVOrigin::upper_left;
    if (value == "lowerLeft")
        return UVOrigin::lower_left;
    sgl::log_warn("Ignoring USD customLayerData '{}' value '{}': expected 'upperLeft' or 'lowerLeft'.", key, value);
    return UVOrigin::lower_left;
}

uint32_t get_max_mip_count(const UsdStageRefPtr& stage)
{
    const auto data = stage->GetRootLayer()->GetCustomLayerData();
    const std::string key = "falcor:maxMipCount";
    const auto it = data.find(key);
    if (it == data.end())
        return 0;
    const auto value = VtValue::Cast<uint32_t>(it->second);
    if (value.IsHolding<uint32_t>())
        return value.UncheckedGet<uint32_t>();
    sgl::log_warn("Ignoring USD customLayerData '{}': expected an unsigned 32-bit integer.", key);
    return 0;
}

UsdStageRefPtr create_import_stage(const UsdStage& source_stage)
{
    SdfLayerRefPtr import_session_layer = SdfLayer::CreateAnonymous("falcor2-stage-import.usda");
    if (const SdfLayerHandle& source_session_layer = source_stage.GetSessionLayer())
        import_session_layer->SetSubLayerPaths({source_session_layer->GetIdentifier()});

    UsdStageRefPtr import_stage = UsdStage::OpenMasked(
        source_stage.GetRootLayer(),
        import_session_layer,
        source_stage.GetPathResolverContext(),
        source_stage.GetPopulationMask(),
        UsdStage::LoadNone
    );
    if (!import_stage)
        throw std::runtime_error("Failed to create a private USD stage for import.");

    import_stage->MuteAndUnmuteLayers(source_stage.GetMutedLayers(), {});
    import_stage->SetLoadRules(source_stage.GetLoadRules());
    import_stage->SetInterpolationType(source_stage.GetInterpolationType());
    import_stage->SetEditTarget(UsdEditTarget(import_session_layer));
    return import_stage;
}

ref<ImporterScene> load_scene_from_stage_impl(
    const UsdStageRefPtr& usd_stage,
    const ImportOptions& import_options,
    const std::string& source_asset_path
)
{
    if (!usd_stage)
        throw std::runtime_error("USD stage must not be null.");

    ArResolverContextBinder binder(usd_stage->GetPathResolverContext());

    // Expands references to .mtlx into simple UsdMaterial/UsdShader that just points to the original mtlx file. This is
    // used instead of usdMtlx, which expands all the Mtlx nodes into UsdShade to allow more complex overrides. We
    // cannot use that since it complicates the compilation (both USD and Falcor need MaterialX, each of a different
    // version) and we do not need nor use it.
    process_mtlx_references(usd_stage);

    auto importer_context = std::make_shared<UsdImporterContext>(usd_stage, import_options, ustring(source_asset_path));

    float4x4 root_transform = sgl::math::matrix_from_scaling(float3(importer_context->get_meters_per_unit()));
    if (UsdGeomGetStageUpAxis(usd_stage) == UsdGeomTokens->z)
        root_transform = mul(sgl::math::matrix_from_rotation_x(sgl::math::radians(-90.f)), root_transform);
    importer_context->push_transform(root_transform, "Root");

    importer_context->run(usd_stage->GetPseudoRoot());

    auto scene = importer_context->get_scene();
    scene->max_mip_count = get_max_mip_count(usd_stage);
    scene->uv_origin = usd_importer::get_uv_origin(usd_stage);
    for (ImporterMesh& mesh : scene->meshes)
        mesh.uv_origin = scene->uv_origin;
    for (ImporterCurve& curve : scene->curves)
        curve.uv_origin = scene->uv_origin;

    return scene;
}

ref<ImporterScene> load_scene_impl(const std::filesystem::path& path, const ImportOptions& import_options)
{
    if (!path.is_absolute())
        throw std::runtime_error("USD path must be resolved to an absolute path.");

    ScopedDiagnosticDelegate diagnostics;

    auto resolver_context = ArGetResolver().CreateDefaultContextForAsset(path.string());
    ArResolverContextBinder binder(resolver_context);

    // OpenMasked creates a fresh stage instead of reusing a cached one, but can still populate caches:
    // https://github.com/PixarAnimationStudios/OpenUSD/issues/4228
    UsdStageRefPtr usd_stage;
    {
        TfErrorMark open_errors;
        usd_stage = UsdStage::OpenMasked(path.string(), UsdStagePopulationMask::All());
        if (!usd_stage) {
            std::string message = "Failed to open USD path: " + path.string();
            for (const auto& error : open_errors)
                message += "\n - " + error.GetCommentary();
            open_errors.Clear();
            throw std::runtime_error(message);
        }
    }

    // New stages target their potentially shared root layer by default. Keep import edits in the session layer.
    usd_stage->SetEditTarget(UsdEditTarget(usd_stage->GetSessionLayer()));

    const std::string source_asset_path = std::filesystem::absolute(path).lexically_normal().string();
    ref<ImporterScene> scene = usd_importer::load_scene_from_stage_impl(usd_stage, import_options, source_asset_path);
    detail::apply_import_options(*scene, import_options, 0.0);
    return scene;
}

ref<ImporterScene> import_stage_impl(const UsdStage& stage, const ImportOptions& import_options)
{
    ScopedDiagnosticDelegate diagnostics;

    pxr::UsdStageRefPtr import_stage = usd_importer::create_import_stage(stage);
    const std::string source_asset_path = stage.GetRootLayer()->GetRealPath();
    ref<ImporterScene> scene
        = usd_importer::load_scene_from_stage_impl(import_stage, import_options, source_asset_path);
    detail::apply_import_options(*scene, import_options, 0.0);
    return scene;
}

} // namespace usd_importer

ref<ImporterScene> UsdImporter::load_scene(const std::filesystem::path& path)
{
    return load_scene(path, ImportOptions{});
}

ref<ImporterScene> UsdImporter::load_scene(const std::filesystem::path& path, const ImportOptions& import_options)
{
    // USD waits must not run another import's tasks under this caller's thread-local error marks.
    return pxr::WorkWithScopedParallelism(
        [&]
        {
            return usd_importer::load_scene_impl(path, import_options);
        },
        /* dropPythonGIL */ false
    );
}

ref<ImporterScene> UsdImporter::import_stage(const pxr::UsdStage& stage)
{
    return import_stage(stage, ImportOptions{});
}

ref<ImporterScene> UsdImporter::import_stage(const pxr::UsdStage& stage, const ImportOptions& import_options)
{
    return pxr::WorkWithScopedParallelism(
        [&]
        {
            return usd_importer::import_stage_impl(stage, import_options);
        },
        /* dropPythonGIL */ false
    );
}

} // namespace falcor
