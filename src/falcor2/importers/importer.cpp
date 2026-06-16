// SPDX-License-Identifier: Apache-2.0

#include "falcor2/importers/importer.h"
#include "falcor2/importers/importer_types.h"
#include "falcor2/importers/usd_importer/usd_importer.h"
#include "falcor2/importers/gltf_importer/gltf_importer.h"
#include "falcor2/core/blob_cache.h"

#include <sgl/core/logger.h>
#include <sgl/core/string.h>
#include <sgl/core/thread.h>

#include <exception>
#include <mutex>

namespace falcor {

namespace {

std::mutex s_importer_cache_mutex;
ref<BlobCache> s_importer_cache;

ref<BlobCache> ensure_importer_cache()
{
    std::lock_guard lock(s_importer_cache_mutex);
    if (s_importer_cache)
        return s_importer_cache;

    try {
        s_importer_cache = make_ref<BlobCache>();
    } catch (const std::exception& e) {
        sgl::log_warn("Failed to create importer blob cache: {}", e.what());
    }

    return s_importer_cache;
}

} // namespace

void set_importer_cache(ref<BlobCache> cache)
{
    std::lock_guard lock(s_importer_cache_mutex);
    s_importer_cache = std::move(cache);
}

ref<BlobCache> importer_cache()
{
    std::lock_guard lock(s_importer_cache_mutex);
    return s_importer_cache;
}

ref<ImporterScene> import_scene(const std::filesystem::path& path, const ImportOptions& import_options)
{
    // Ensure the global importer cache is set up.
    ensure_importer_cache();

    ref<ImporterScene> importer_scene;
    std::string extension = sgl::string::to_lower(path.extension().string());
    if (extension == ".usd" || extension == ".usda" || extension == ".usdc") {
        importer_scene = UsdImporter().load_scene(path);
    } else if (extension == ".gltf" || extension == ".glb") {
        importer_scene = GltfImporter().load_scene(path);
    } else {
        FALCOR_THROW("Unknown scene file extension \"{}\"", extension);
    }

    if (import_options.recompute_normals) {
        sgl::thread::parallel_for(
            sgl::thread::blocked_range<size_t>(0, importer_scene->meshes.size()),
            [&](auto range)
            {
                for (auto i : range) {
                    auto& mesh = importer_scene->meshes[i];
                    mesh.add_normals_from_faces();
                    mesh.add_tangents_from_normals();
                }
            }
        );
    }

    return importer_scene;
}

} // namespace falcor
