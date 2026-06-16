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

#include <pxr/usd/usd/common.h>
#include <pxr/usd/usd/stage.h>

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
        m_errors.push_back(message);
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

    std::string format_errors() const
    {
        if (m_errors.empty())
            return {};

        std::string joined = "USD reported errors:";
        for (const auto& error : m_errors)
            joined += "\n - " + error;
        return joined;
    }

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
    std::vector<std::string> m_errors;
};

class ScopeGuard {
public:
    ScopeGuard(const std::function<void(void)>& func)
        : m_func(func)
    {
    }

    ~ScopeGuard() { m_func(); }

private:
    std::function<void(void)> m_func;
};

} // namespace usd_importer

ref<ImporterScene> UsdImporter::load_scene(const std::filesystem::path& path)
{
    using namespace pxr;

    if (!path.is_absolute())
        throw std::runtime_error("USD path must be resolved to an absolute path.");

    usd_importer::DiagDelegate diagnostic_delegate;
    TfDiagnosticMgr::GetInstance().AddDelegate(&diagnostic_delegate);
    usd_importer::ScopeGuard remove_diagnostic(
        [&]()
        {
            TfDiagnosticMgr::GetInstance().RemoveDelegate(&diagnostic_delegate);
        }
    );

    auto resolver_context = ArGetResolver().CreateDefaultContextForAsset(path.string());
    ArResolverContextBinder binder(resolver_context);

    UsdStageRefPtr usd_stage = UsdStage::Open(path.string());
    if (!usd_stage) {
        std::string message = "Failed to open USD path: " + path.string();
        std::string diagnostics = diagnostic_delegate.format_errors();
        if (!diagnostics.empty())
            message += "\n" + diagnostics;
        throw std::runtime_error(message);
    }

    // Expands references to .mtlx into simple UsdMaterial/UsdShader that just points to the original mtlx file. This is
    // used instead of usdMtlx, which expands all the Mtlx nodes into UsdShade to allow more complex overrides. We
    // cannot use that since it complicates the compilation (both USD and Falcor need MaterialX, each of a different
    // version) and we do not need nor use it.
    usd_importer::process_mtlx_references(usd_stage);

    auto importer_context = std::make_shared<usd_importer::UsdImporterContext>(usd_stage);

    float4x4 root_transform = sgl::math::matrix_from_scaling(float3(importer_context->get_meters_per_unit()));
    if (UsdGeomGetStageUpAxis(usd_stage) == UsdGeomTokens->z)
        root_transform = mul(sgl::math::matrix_from_rotation_x(sgl::math::radians(-90.f)), root_transform);
    importer_context->push_transform(root_transform, "Root");

    importer_context->traverse_scene(usd_stage->GetPseudoRoot());
    importer_context->finalize();

    auto scene = importer_context->get_scene();
    return scene;
}

} // namespace falcor
