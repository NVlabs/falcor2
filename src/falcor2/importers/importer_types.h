// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/core/types.h"
#include "falcor2/core/object.h"
#include "falcor2/core/macros.h"
#include "falcor2/core/properties.h"
#include "falcor2/utils/aabb.h"
#include "falcor2/importers/fwd.h"
#include "falcor2/render/fwd.h"
#include "falcor2/render/scene_types.h"
#include "falcor2/render/animation_timing.h"
#include "falcor2/render/source_asset_ref.h"

#include "sgl/core/data_type.h"
#include "sgl/core/short_vector.h"

#include <cstddef>
#include <iterator>
#include <type_traits>
#include <functional>
#include <optional>
#include <vector>
#include <string>
#include <string_view>
#include <map>
#include <variant>
#include <span>

namespace falcor {

namespace detail {
/// Validate the signed import refinement level before calculating a tessellation rate.
FALCOR_API void validate_refinement_level(int level);
} // namespace detail

/// Semantic meaning that importer assigns to a vertex attribute. For some
/// formats this may be explicitly specified, for others it may be inferred
/// from attribute name (e.g., "POSITION", "pos", "uv1"). If semantic is
/// 'unknown', the importer was unable to assign it any specific meaning.
enum class ImporterSemantic {
    unknown,
    position,
    normal,
    tangent,
    bitangent,
    handedness,
    tex_coord,
    color,
    joints,
    weights
};
SGL_ENUM_INFO(
    ImporterSemantic,
    {
        {ImporterSemantic::unknown, "unknown"},
        {ImporterSemantic::position, "position"},
        {ImporterSemantic::normal, "normal"},
        {ImporterSemantic::tangent, "tangent"},
        {ImporterSemantic::bitangent, "bitangent"},
        {ImporterSemantic::handedness, "handedness"},
        {ImporterSemantic::tex_coord, "tex_coord"},
        {ImporterSemantic::color, "color"},
        {ImporterSemantic::joints, "joints"},
        {ImporterSemantic::weights, "weights"},
    }
);
SGL_ENUM_REGISTER(ImporterSemantic);

/// Describes an attribute of a mesh vertex
struct FALCOR_API ImporterMeshAttribute {

    /// Name of the attribute as specified in the file (e.g., "POSITION", "TEXCOORD_0", "myCustomAttrib")
    std::string name;

    /// Semantic meaning of the attribute assigned by the importer
    ImporterSemantic semantic{ImporterSemantic::unknown};

    /// Index for attributes that can have multiple sets, e.g., TEXCOORD_0, TEXCOORD_1.
    /// Mesh position, normal and tangent indices identify matching motion samples; zero is the reference sample.
    uint32_t index{0};

    /// Type of the attribute data (e.g. float for a float3)
    sgl::DataType type{sgl::DataType::void_};

    /// Number of components in the attribute (e.g. 3 for a float3)
    uint32_t num_components{0};

    /// Buffer index for the attribute
    uint32_t buffer{0};

    /// Offset in bytes from start of vertex
    size_t offset{0};

    bool valid() const { return type != sgl::DataType::void_; }
};

/// Minimal information to describe an attribute to be created using ensure_attributes.
struct FALCOR_API ImporterMeshDefaultAttribute {
    ImporterSemantic semantic{ImporterSemantic::unknown};
    uint32_t index{0};
    uint32_t components{0}; // Leave as 0 to use default for semantic
    /// Exact authored identity for an unknown-semantic attribute.
    std::string name;
    /// Scalar type override; void_ selects the semantic default.
    sgl::DataType type{sgl::DataType::void_};
};

/// Returns true when two attribute descriptions identify the same stream.
/// Known semantics use semantic and index; unknown semantics use an exact nonempty name.
inline bool importer_attribute_identity_equal(
    ImporterSemantic lhs_semantic,
    uint32_t lhs_index,
    std::string_view lhs_name,
    ImporterSemantic rhs_semantic,
    uint32_t rhs_index,
    std::string_view rhs_name
)
{
    if (lhs_semantic == ImporterSemantic::unknown || rhs_semantic == ImporterSemantic::unknown) {
        return lhs_semantic == ImporterSemantic::unknown && rhs_semantic == ImporterSemantic::unknown
            && !lhs_name.empty() && lhs_name == rhs_name;
    }
    return lhs_semantic == rhs_semantic && lhs_index == rhs_index;
}

inline bool
importer_attribute_identity_equal(const ImporterMeshDefaultAttribute& lhs, const ImporterMeshDefaultAttribute& rhs)
{
    return importer_attribute_identity_equal(lhs.semantic, lhs.index, lhs.name, rhs.semantic, rhs.index, rhs.name);
}

inline bool importer_attribute_identity_equal(const ImporterMeshAttribute& lhs, const ImporterMeshDefaultAttribute& rhs)
{
    return importer_attribute_identity_equal(lhs.semantic, lhs.index, lhs.name, rhs.semantic, rhs.index, rhs.name);
}

/// Helper class to make it simple to read/write a stream of a given
/// attribute type for the vertices of a mesh.
/// Like std::span, constness of the wrapper does not affect element mutability.
/// Use a const element type for read-only access.
/// Iterators borrow the underlying storage, not this wrapper, and are invalidated when that storage moves or dies.
template<typename T>
struct ImporterVertexStream {
    std::conditional_t<std::is_const_v<T>, const uint8_t, uint8_t>* data;
    size_t stride;
    size_t size;

    constexpr operator ImporterVertexStream<const T>() const noexcept { return {data, stride, size}; }

    bool valid() const { return data != nullptr; }

    T* first() const;

    T& operator[](size_t index) const;

    class Iterator {
    public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = std::remove_cv_t<T>;
        using difference_type = std::ptrdiff_t;
        using pointer = T*;
        using reference = T&;

        Iterator() = default;
        reference operator*() const { return *operator->(); }
        pointer operator->() const { return reinterpret_cast<T*>(m_data + m_index * m_stride); }

        Iterator& operator++()
        {
            ++m_index;
            return *this;
        }

        Iterator operator++(int)
        {
            Iterator previous = *this;
            ++(*this);
            return previous;
        }

        bool operator==(const Iterator&) const = default;

    private:
        friend struct ImporterVertexStream;
        Iterator(decltype(data) data, size_t stride, size_t index)
            : m_data(data)
            , m_stride(stride)
            , m_index(index)
        {
        }

        decltype(data) m_data = nullptr;
        size_t m_stride = 0;
        // A strided end pointer could exceed the allocation when the attribute has a nonzero offset.
        size_t m_index = 0;
    };

    Iterator begin() const { return Iterator(data, stride, 0); }
    Iterator end() const { return Iterator(data, stride, size); }
};


/// A buffer containing vertex data with a fixed byte stride.
struct FALCOR_API ImporterVertexBuffer {
    /// Buffer data.
    std::vector<uint8_t> data;
    /// Stride in bytes between vertices.
    size_t stride = 0;

    /// Resize storage, preserving existing vertices. Invalidates previously acquired views on reallocation.
    void resize_vertices(size_t count) { data.resize(count * stride); }

    /// View an attribute at a byte offset within each vertex record.
    template<typename T>
    ImporterVertexStream<T> get_stream(size_t offset, size_t count)
    {
        return {data.empty() ? nullptr : data.data() + offset, stride, count};
    }

    template<typename T>
    ImporterVertexStream<const T> get_stream(size_t offset, size_t count) const
    {
        return {data.empty() ? nullptr : data.data() + offset, stride, count};
    }
};


/// Mesh data imported from a file. Represents a set of buffers containing
/// vertex data, and a set of subgeometries with indices into the vertex data.
/// Operations that change the vertex layout or topology invalidate previously acquired streams, pointers, NumPy
/// views, and references to subgeometry indices. Reacquire them after the operation completes.
class FALCOR_API ImporterMesh {
public:
    /// Create import-processing defaults from options. Overlay authored properties afterward.
    static Properties make_properties(const ImportOptions& options);
    /// Import-processing properties. Initialize with make_properties() before processing.
    Properties properties;
    /// Mesh name.
    std::string name;
    /// Reference to the object in the source asset from which this mesh was imported.
    std::optional<SourceAssetRef> source_asset_ref;

    /// Timeline key times in seconds; empty for static geometry. Paired equal times encode a jump (left, right).
    std::vector<float> motion_times;
    /// Stored sample index for each timeline key. Empty means one stored sample per key.
    std::vector<uint32_t> motion_sample_indices;
    /// Interpolation between motion samples.
    AnimationInterpolationMode motion_interpolation_mode{AnimationInterpolationMode::linear};

    /// Authored image origin convention for this mesh's texture coordinates.
    ///
    /// This describes the texcoord data stored on this mesh. It may differ
    /// from the ImporterScene or Scene target UV origin; scene loading converts
    /// mesh UVs when the mesh convention differs from the target convention.
    /// UV-origin conversion does not update tangent handedness; tangent-frame
    /// parity and normal-map channel conventions are separate concerns.
    UVOrigin uv_origin = UVOrigin::upper_left;

    struct Subgeometry {
        /// Subgeometry name.
        std::string name;
        /// Triangle indices.
        std::vector<uint3> indices;
        /// Index into ImporterScene::materials, or -1 if no material is assigned.
        int material_index = -1;
    };

    /// List of subgeometries.
    std::vector<Subgeometry> subgeometries;

    /// Local bounding box of the mesh (calculated from vertices).
    AABB local_aabb;

    /// Calculate the local AABB from the mesh vertices.
    void calculate_local_aabb()
    {
        local_aabb = AABB(); // Reset to invalid AABB
        auto positions = position_stream();
        for (uint32_t i = 0; i < m_vertex_count; i++) {
            auto pos = positions[i];
            local_aabb.expand(pos);
        }
    }

    /// Generate normals for each position sample based on its face geometry.
    void add_normals_from_faces();

    /// Generate tangents using UV coordinates and MikkTSpace, splitting vertices as required by its per-corner output.
    /// Existing vertices are never merged; exact compaction remains a separate operation. This may rewrite indices
    /// and invalidates previously acquired mesh views as described by ImporterMesh.
    void add_tangents_from_uvs();

    /// Generate tangents for mesh vertices using normals (orthonormal basis).
    void add_tangents_from_normals();

    /// Transfer reference tangents to additional position/normal samples without further vertex splits.
    void update_motion_tangents();

    /// Collapse exact local-space, same-material triangles with opposite winding within this mesh.
    /// Corresponding vertices must match in every position sample.
    /// Retains the first authored winding and all same-winding duplicates. Skips meshes with joint or weight
    /// attributes. Removes empty subgeometries, preserves vertex data, and returns the number of removed triangles.
    size_t collapse_exact_opposite_triangles();

    /// Remove duplicate vertices from mesh and update indices accordingly.
    void deduplicate_vertices();

    /// Low level function to get a pointer to vertex attribute data in a stream.
    void get_stream_info(const ImporterMeshAttribute& attrib, void*& out_base_address, size_t& out_stride)
    {
        if (attrib.valid()) {
            auto& stream = m_buffers[attrib.buffer];
            out_base_address = stream.data.data() + attrib.offset;
            out_stride = stream.stride;
        } else {
            out_base_address = nullptr;
            out_stride = 0;
        }
    }

    /// Low level function to get a pointer to vertex attribute data in a stream.
    void get_stream_info(const ImporterMeshAttribute& attrib, const void*& out_base_address, size_t& out_stride) const
    {
        if (attrib.valid()) {
            auto& stream = m_buffers[attrib.buffer];
            out_base_address = stream.data.data() + attrib.offset;
            out_stride = stream.stride;
        } else {
            out_base_address = nullptr;
            out_stride = 0;
        }
    }


    /// Get a typed 'VertexStream' for a given attribute to give an easy way
    /// to read/write the vertex data.
    template<typename T>
    ImporterVertexStream<T> get_stream(const ImporterMeshAttribute& attrib)
    {
        if (attrib.valid()) {
            return m_buffers[attrib.buffer].get_stream<T>(attrib.offset, m_vertex_count);
        } else {
            return {};
        }
    }

    /// Get a typed 'VertexStream' for a given attribute to give an easy way
    /// to read/write the vertex data.
    template<typename T>
    ImporterVertexStream<const T> get_stream(const ImporterMeshAttribute& attrib) const
    {
        if (attrib.valid()) {
            return m_buffers[attrib.buffer].get_stream<T>(attrib.offset, m_vertex_count);
        } else {
            return {};
        }
    }

    /// Find an attribute by semantic and index (e.g., texcoord 0, texcoord 1).
    ImporterMeshAttribute* find_attribute(ImporterSemantic semantic, uint32_t index = 0)
    {
        for (auto& attrib : m_attributes) {
            if (attrib.semantic == semantic && attrib.index == index)
                return &attrib;
        }
        return nullptr;
    }

    /// Find an attribute by name (e.g., "POSITION", "TEXCOORD_0", "myCustomAttrib").
    ImporterMeshAttribute* find_attribute(std::string_view _name)
    {
        for (auto& attrib : m_attributes) {
            if (attrib.name == _name)
                return &attrib;
        }
        return nullptr;
    }

    /// Find an attribute by its semantic/index or exact authored-name identity.
    ImporterMeshAttribute* find_attribute(const ImporterMeshDefaultAttribute& descriptor)
    {
        for (auto& attrib : m_attributes) {
            if (importer_attribute_identity_equal(attrib, descriptor))
                return &attrib;
        }
        return nullptr;
    }

    /// Find a typed 'VertexStream' for a given attribute to give an easy way
    /// to read/write the vertex data.
    template<typename T>
    ImporterVertexStream<T> find_stream(ImporterSemantic semantic, uint32_t index = 0)
    {
        const auto* attrib = find_attribute(semantic, index);
        if (!attrib)
            return ImporterVertexStream<T>();
        if (attrib->buffer >= m_buffers.size())
            return ImporterVertexStream<T>();
        return get_stream<T>(*attrib);
    }

    /// Add a new vertex attribute to the mesh. If the attribute
    /// refers to an existing buffer, the buffer must be empty. If it
    /// refers to a new buffer, it must be created by calling create_buffer()
    /// or create_buffers_from_attributes() before use.
    void add_attribute(const ImporterMeshAttribute& attrib);

    /// Create new buffers for all attributes that refer to a buffer that doesn't exist yet.
    void create_buffers_from_attributes();

    /// Ensures a set of known attriute types exist, creating them with basic
    /// floating point config if needed.
    void ensure_attributes(std::span<ImporterMeshDefaultAttribute> attribs);

    /// Ensures a set of known attriute types exist, creating them with basic
    /// floating point config if needed.
    void ensure_attributes(std::initializer_list<ImporterMeshDefaultAttribute> attribs)
    {
        auto first = const_cast<ImporterMeshDefaultAttribute*>(attribs.begin());
        ensure_attributes(std::span<ImporterMeshDefaultAttribute>(first, attribs.size()));
    }

    /// Allocates space in vertex buffers for given number of vertices and
    /// returns the starting vertex index.
    size_t allocate_vertices(size_t count, bool init_to_zero = false);

    /// Current vertex count.
    size_t vertex_count() const { return m_vertex_count; }

    /// Bake a skeleton pose using one mesh-local transform per joint.
    /// Normalizes finite weights across all sets; zero-sum vertices remain unchanged. Signed weights are allowed.
    /// Consumes joint and weight attributes after deformation.
    /// Fails if the mesh has additional position samples.
    void bake_pose(std::span<const float4x4> mesh_from_bind_mesh_by_joint);

    /// Zero out-of-range joint influences and nonfinite weights; set unused indices to zero.
    /// Requires a nonempty joint palette.
    /// Preserves finite signed weights and surviving sums. Copy shared meshes before binding-specific preparation.
    void sanitize_skinning(size_t joint_count);

    /// Normalize finite weights across all influence sets. Leaves zero sums unchanged, including signed cancellation.
    /// Does not sanitize or limit influences. bake_pose() calls this before deformation.
    void normalize_skinning_weights();

    /// Vertex attributes.
    const std::vector<ImporterMeshAttribute>& attributes() const { return m_attributes; }

    /// Vertex buffers.
    const std::vector<ImporterVertexBuffer>& buffers() const { return m_buffers; }
    std::vector<ImporterVertexBuffer>& buffers() { return m_buffers; }

    size_t sample_count() const { return m_positions.size(); }

    /// Vertex positions.
    const ImporterMeshAttribute& position_attribute(uint32_t sample_index = 0) const
    {
        FALCOR_CHECK(sample_index < m_positions.size(), "Position sample index out of bounds");
        return m_positions[sample_index];
    }
    ImporterVertexStream<float3> position_stream(uint32_t sample_index = 0)
    {
        return get_stream<float3>(position_attribute(sample_index));
    }
    ImporterVertexStream<const float3> position_stream(uint32_t sample_index = 0) const
    {
        return get_stream<float3>(position_attribute(sample_index));
    }

    /// Vertex normals.
    const ImporterMeshAttribute& normal_attribute(uint32_t sample_index = 0) const
    {
        FALCOR_CHECK(sample_index < m_normals.size(), "Normal sample index out of bounds");
        return m_normals[sample_index];
    }
    ImporterVertexStream<float3> normal_stream(uint32_t sample_index = 0)
    {
        return get_stream<float3>(normal_attribute(sample_index));
    }
    ImporterVertexStream<const float3> normal_stream(uint32_t sample_index = 0) const
    {
        return get_stream<float3>(normal_attribute(sample_index));
    }

    /// Vertex tangents.
    const ImporterMeshAttribute& tangent_attribute(uint32_t sample_index = 0) const
    {
        FALCOR_CHECK(sample_index < m_tangents.size(), "Tangent sample index out of bounds");
        return m_tangents[sample_index];
    }
    ImporterVertexStream<float3> tangent_stream(uint32_t sample_index = 0)
    {
        return get_stream<float3>(tangent_attribute(sample_index));
    }
    ImporterVertexStream<const float3> tangent_stream(uint32_t sample_index = 0) const
    {
        return get_stream<float3>(tangent_attribute(sample_index));
    }

    /// Vertex handedness.
    const ImporterMeshAttribute& handedness_attribute() const { return m_handedness; }
    ImporterVertexStream<float> handedness_stream() { return get_stream<float>(m_handedness); }
    ImporterVertexStream<const float> handedness_stream() const { return get_stream<float>(m_handedness); }

    /// Vertex texcoords (0) + helpers to access stream as float2.
    const ImporterMeshAttribute& texcoord_attribute() const { return m_texcoords; }
    ImporterVertexStream<float2> texcoord_stream() { return get_stream<float2>(m_texcoords); }
    ImporterVertexStream<const float2> texcoord_stream() const { return get_stream<float2>(m_texcoords); }

    /// Vertex colors (0) + helpers to access stream as rgb float3.
    const ImporterMeshAttribute& color_attribute() const { return m_colors; }
    ImporterVertexStream<float3> color_stream() { return get_stream<float3>(m_colors); }
    ImporterVertexStream<const float3> color_stream() const { return get_stream<float3>(m_colors); }

private:
    void apply_vertex_remap(const VertexRemap& remap);

    // Current vertex count
    size_t m_vertex_count = 0;

    // Vertex attributes and buffers
    std::vector<ImporterMeshAttribute> m_attributes;
    std::vector<ImporterVertexBuffer> m_buffers;

    // Cached commonly used attributes for easy access
    sgl::short_vector<ImporterMeshAttribute, 1> m_positions{ImporterMeshAttribute{}};
    sgl::short_vector<ImporterMeshAttribute, 1> m_normals{ImporterMeshAttribute{}};
    sgl::short_vector<ImporterMeshAttribute, 1> m_tangents{ImporterMeshAttribute{}};
    ImporterMeshAttribute m_handedness;
    ImporterMeshAttribute m_texcoords;
    ImporterMeshAttribute m_colors;

    /// Create a new empty vertex buffer. The attribute(s) that refer to this buffer
    /// must be created before use. Pass stride==0 to have the stride automatically
    /// calculated from the attributes that refer to this buffer.
    void create_buffer(size_t buffer_index, size_t stride = 0);
};


template<typename T>
T* ImporterVertexStream<T>::first() const
{
    SGL_CHECK(data != nullptr, "Invalid vertex stream");
    return reinterpret_cast<T*>(data);
}

template<typename T>
inline T& ImporterVertexStream<T>::operator[](size_t index) const
{
    SGL_CHECK(index < size, "Vertex index out of bounds");
    return *reinterpret_cast<T*>(data + index * stride);
}


/// Imported curve data.
class FALCOR_API ImporterCurve {
public:
    /// Create import-processing defaults from options. Overlay authored properties afterward.
    static Properties make_properties(const ImportOptions& options);
    /// Import-processing properties. Initialize with make_properties() before processing.
    Properties properties;
    /// Curve name.
    std::string name;
    /// Reference to the object in the source asset from which this curve was imported.
    std::optional<SourceAssetRef> source_asset_ref;
    /// Index into ImporterScene::materials, or -1 if no material is assigned.
    int material_index = -1;
    /// Timeline key times in seconds; empty for static geometry. Paired equal times encode a jump (left, right).
    std::vector<float> motion_times;
    /// Stored sample index for each timeline key. Empty means one stored sample per key.
    std::vector<uint32_t> motion_sample_indices;
    /// Interpolation between motion samples.
    AnimationInterpolationMode motion_interpolation_mode{AnimationInterpolationMode::linear};

    /// Resize all position/radius samples to the same vertex count. Invalidates previously acquired streams/views.
    void resize_vertices(size_t vertex_count, size_t sample_count = 1);
    size_t vertex_count() const { return m_vertex_count; }
    size_t sample_count() const { return m_positions.size(); }

    /// Curve vertex positions for a sample; zero is the reference sample.
    ImporterVertexStream<float3> position_stream(uint32_t sample_index = 0)
    {
        FALCOR_CHECK(sample_index < m_positions.size(), "Position sample index out of bounds");
        return m_positions[sample_index].get_stream<float3>(0, m_vertex_count);
    }
    ImporterVertexStream<const float3> position_stream(uint32_t sample_index = 0) const
    {
        FALCOR_CHECK(sample_index < m_positions.size(), "Position sample index out of bounds");
        return m_positions[sample_index].get_stream<float3>(0, m_vertex_count);
    }
    /// Radius at each vertex for a sample; zero is the reference sample.
    ImporterVertexStream<float> radius_stream(uint32_t sample_index = 0)
    {
        FALCOR_CHECK(sample_index < m_radii.size(), "Radius sample index out of bounds");
        return m_radii[sample_index].get_stream<float>(0, m_vertex_count);
    }
    ImporterVertexStream<const float> radius_stream(uint32_t sample_index = 0) const
    {
        FALCOR_CHECK(sample_index < m_radii.size(), "Radius sample index out of bounds");
        return m_radii[sample_index].get_stream<float>(0, m_vertex_count);
    }
    /// Primary texture coordinates at each curve vertex. Empty leaves render geometry UVs at zero.
    std::vector<float2> texcoords;
    /// Origin convention of texcoords; converted once when loaded into a render scene.
    UVOrigin uv_origin = UVOrigin::upper_left;
    /// Pairs of indices defining line segments.
    std::vector<uint32_t> indices;
    /// Indexing mode for the index buffer.
    CurveIndexingMode indexing_mode{CurveIndexingMode::list};
    /// Local bounding box.
    AABB local_aabb;

    /// Calculate local AABB from positions and radii.
    void calculate_local_aabb();

private:
    size_t m_vertex_count = 0;
    sgl::short_vector<ImporterVertexBuffer, 1> m_positions{ImporterVertexBuffer{.stride = sizeof(float3)}};
    sgl::short_vector<ImporterVertexBuffer, 1> m_radii{ImporterVertexBuffer{.stride = sizeof(float)}};
};

/// Filter modes for texture sampling.
enum class TextureFilterMode {
    nearest = 0,
    linear = 1,
};
SGL_ENUM_INFO(
    TextureFilterMode,
    {
        {TextureFilterMode::nearest, "nearest"},
        {TextureFilterMode::linear, "linear"},
    }
);
SGL_ENUM_REGISTER(TextureFilterMode);

/// Wrap modes for texture sampling
enum class TextureWrapMode {
    repeat = 0,
    mirror_repeat = 1,
    clamp_to_edge = 2,
};
SGL_ENUM_INFO(
    TextureWrapMode,
    {
        {TextureWrapMode::repeat, "repeat"},
        {TextureWrapMode::mirror_repeat, "mirror_repeat"},
        {TextureWrapMode::clamp_to_edge, "clamp_to_edge"},
    }
);
SGL_ENUM_REGISTER(TextureWrapMode);

struct FALCOR_API ImporterTexture {
    // Source channels (rgb, r, g, b etc) read from the texture.
    std::string source_name;
    std::filesystem::path texture_path;
    /// True when USD explicitly specifies that the texture is sRGB.
    bool is_srgb = false;
    float4x4 texture_transform = float4x4::identity();
    /// Optional texture data as bytes (alternative to texture_path)
    std::vector<uint8_t> texture_data;
    /// MIME type for embedded texture data (e.g., "image/png", "image/jpeg")
    std::string mime_type;

    /// Magnification filter mode
    TextureFilterMode mag_filter = TextureFilterMode::linear;
    /// Minification filter within a mip level.
    TextureFilterMode min_filter = TextureFilterMode::linear;
    /// Filter between mip levels.
    TextureFilterMode mip_filter = TextureFilterMode::linear;
    /// False for non-mipmapped minification; keep image mip data for other bindings.
    bool use_mipmaps = true;
    /// S (U) wrapping mode
    TextureWrapMode wrap_s = TextureWrapMode::repeat;
    /// T (V) wrapping mode
    TextureWrapMode wrap_t = TextureWrapMode::repeat;

    auto operator<=>(const ImporterTexture&) const = default;
};

/// Asset data embedded in an imported container.
struct FALCOR_API ImporterAsset {
    /// Resolved asset path.
    std::filesystem::path path;
    /// Encoded asset bytes.
    std::vector<uint8_t> data;
};

/// Prototype for instancing more complex objects than simple meshes.
/// Allows nested instancing and instancing of larger groups of meshes.
struct FALCOR_API ImporterPrototype {
    /// Prototype name.
    std::string name;
    /// Index into ImporterScene::nodes, root node of the hierarchy.
    int root_node = -1;
};

/// A single animation channel storing typed keyframe data.
struct FALCOR_API ImporterAnimationChannel {
    /// Value type of keyframe data.
    AnimationValueType value_type = AnimationValueType::float_;
    /// Interpolation mode.
    AnimationInterpolationMode interpolation_mode = AnimationInterpolationMode::linear;
    /// Keyframe timestamps in seconds.
    std::vector<float> times;
    /// Keyframe values (flat array: 1/3/4 floats per key depending on value_type).
    std::vector<float> values;
    /// In-tangent per keyframe (same layout as values), only populated for cubic_spline.
    std::vector<float> in_tangents;
    /// Out-tangent per keyframe (same layout as values), only populated for cubic_spline.
    std::vector<float> out_tangents;

    /// The number of components per keyframe value.
    int components_per_value() const;
    /// The number of keyframes.
    size_t keyframe_count() const;
};

/// A collection of animation channels.
struct FALCOR_API ImporterAnimation {
    /// Animation name.
    std::string name;
    /// Reference to the object in the source asset from which this animation was imported.
    std::optional<SourceAssetRef> source_asset_ref;
    /// Flat storage of all animation channels.
    std::vector<ImporterAnimationChannel> channels;

    /// The keyframe interval across all channels in seconds; invalid when there are no keys.
    AnimationTimeRange key_range() const;
};

/// Skeleton binding data used to deform importer meshes into an authored pose.
/// Matrices use column vectors: a_from_b maps b into a, with the rightmost factor acting first.
/// For normalized weights, each joint contributes:
///
///     world_from_joint(time) * joint_from_bind[j] * bind_from_mesh * point(mesh_position).
///
/// USD supplies geomBindTransform and inverse(bindTransforms[j]); glTF supplies identity and inverseBindMatrices[j].
/// Keep the mesh binding separate so different mesh attachments can use the same joint inverse binds.
/// Joint nodes hold the rest/default pose and ordinary animation. Their world transforms already include
/// USD's Skeleton placement and the import root's unit/up-axis conversion. Bind and rest poses may differ.
///
/// Example: mesh point x=1, bind_from_mesh=T(10), joint_from_bind=T(-10), and world_from_joint=T(12)
/// give 1 -> 11 -> 1 -> 13. Moving the joint's parent by +5 gives 18. A mesh-node placement of +100
/// must not add another 100: baking stores 18-100=-82 locally so rendering gives world x=18.
struct FALCOR_API ImporterSkin {
    /// Skin name.
    std::string name;
    /// Optional skeleton root node, -1 if unspecified.
    int skeleton_root = -1;
    /// Joint node indices. Their order defines the indices used by mesh joint attributes.
    std::vector<int> joints;
    /// Reference mesh to common bind space: identity for glTF, geomBindTransform for USD.
    float4x4 bind_from_mesh = float4x4::identity();
    /// Common bind space to each joint: inverseBindMatrices for glTF, inverse(bindTransforms) for USD.
    std::vector<float4x4> joint_from_bind;
};

struct FALCOR_API ImporterNode {
    /// Node name.
    std::string name;
    /// Reference to the object in the source asset from which this node was imported.
    std::optional<SourceAssetRef> source_asset_ref;
    /// Node transformation matrix (i.e. transform from object space to parent or world space).
    float4x4 transform = float4x4::identity();

    // TODO(chrisc): Replace use of 'int' indices with mesh/node id types

    /// Index into ImporterScene::meshes, -1 if no mesh.
    int mesh_index = -1;
    /// Index into ImporterScene::skins, -1 if the node's mesh is not skinned.
    int skin_index = -1;
    /// Index into ImporterScene::curves, -1 if no curve.
    int curve_index = -1;
    int light_index = -1;
    int camera_index = -1;
    int prototype_index = -1;
    /// Indices into ImporterScene::nodes (child nodes).
    std::vector<int> children;
    /// Index into ImporterScene::nodes, -1 if root node.
    int parent = -1;

    /// Index into ImporterScene::animation.channels for translation (expects float3), -1 if not animated.
    int animation_translation = -1;
    /// Index into ImporterScene::animation.channels for rotation (expects quatf), -1 if not animated.
    int animation_rotation = -1;
    /// Index into ImporterScene::animation.channels for scale (expects float3), -1 if not animated.
    int animation_scale = -1;

    /// World-space bounding box of this node and all its children.
    AABB world_aabb;
};

struct FALCOR_API ImporterMaterial {
    /// Callback that constructs a live material during importer scene loading.
    ///
    /// The returned material must already be owned by the supplied scene.
    using Constructor = std::function<Material*(Scene&, const ImporterMaterial&)>;

    /// Material name.
    std::string name;
    /// Reference to the object in the source asset from which this material was imported.
    std::optional<SourceAssetRef> source_asset_ref;

    // TODO(chrisc): Replace use of 'int' as texture index with a proper texture id type.

    /// Material parameters.
    /// For Usd, if UsdPreviewSurface is available, it is flattened here. For MDL, nodes must be used.
    Properties params;
    /// Optional live material constructor. When set, this takes precedence over parameter-driven conversion.
    Constructor constructor;
    /// Maps a named material output (e.g. "surface", "displacement") to the material network
    /// that drives it. Each network is represented as an ordered list of nodes, where each node's
    /// properties (type, parameter values, input connections) are captured in a Properties object.
    /// This mirrors the structure of a MaterialNetwork, but is stored here without that dependency.
    std::map<std::string, std::vector<Properties>, std::less<>> output_to_material_network;
};

struct FALCOR_API ImporterCamera {
    enum class Projection { perspective, orthographic };
    SGL_ENUM_INFO(
        Projection,
        {
            {Projection::perspective, "perspective"},
            {Projection::orthographic, "orthographic"},
        }
    );

    enum class FOVDirection { horizontal, vertical };
    SGL_ENUM_INFO(
        FOVDirection,
        {
            {FOVDirection::horizontal, "horizontal"},
            {FOVDirection::vertical, "vertical"},
        }
    );

    /// Camera name.
    std::string name;
    /// Reference to the object in the source asset from which this camera was imported.
    std::optional<SourceAssetRef> source_asset_ref;
    /// Focal length (millimeters).
    float focal_length = 50.f;
    /// F-stop value.
    float fstop = 8.f;
    /// Exposure compensation in stops: +1 doubles rendered RGB; -1 halves it.
    float exposure = 0.f;
    /// Sensor size (millimeters) in fov_direction.
    float sensor_size_mm = 24.f;
    /// Whether thin-lens depth of field is enabled.
    bool enable_depth_of_field = false;
    /// Focus distance.
    float focus_distance = 1.f;
    /// Depth range (near, far).
    float2 depth_range = float2(0.01f, 10000.f);
    /// Projection type.
    Projection projection = Projection::perspective;
    /// Axis to which sensor_size_mm and fov_degrees() apply. Focal length is axis-independent.
    FOVDirection fov_direction = FOVDirection::vertical;

    /// Calculate focal length from a field of view and sensor size.
    static float focal_length_from_fov_degrees(float fov_degrees, float sensor_size_mm);

    /// Calculate the field of view represented by a sensor size and focal length.
    static float fov_degrees_from_focal_length(float sensor_size_mm, float focal_length_mm);

    /// Calculate this camera's field of view along fov_direction.
    float fov_degrees() const;

    /// Normalize the directional sensor size to the vertical film-fit model used by Camera.
    ///
    /// If this camera stores a horizontal sensor size, the vertical sensor size
    /// is derived from the supplied horizontal-to-vertical sensor ratio. The
    /// default ratio is the 4:3 sensor assumption used by pyscene scene conversion.
    float vertical_sensor_size_mm(float horizontal_to_vertical_sensor_ratio = 4.f / 3.f) const;

    /// Calculate this camera's vertical field of view.
    ///
    /// If this camera stores a horizontal FOV, the vertical sensor size is derived
    /// from the supplied horizontal-to-vertical sensor ratio. The default ratio
    /// is the 4:3 sensor assumption used by pyscene scene conversion.
    float vertical_fov_degrees(float horizontal_to_vertical_sensor_ratio = 4.f / 3.f) const;
};
SGL_ENUM_REGISTER(ImporterCamera::Projection);
SGL_ENUM_REGISTER(ImporterCamera::FOVDirection);

struct FALCOR_API ImporterLight {
    enum class Type {
        /// Light emitted from a distance source along the -Z axis.
        distant,
        /// Light emitted from a rectangle of size width x height along the -Z axis.
        /// Rectangle at origin in the XY plane.
        rectangular,
        /// Light emitted out from a sphere at the origin.
        sphere,
        /// Sphere with zero radius.
        point,
        /// Open cylinder centered at the origin, with its axis along X and emission from the lateral surface.
        cylinder,
        /// Disk at the origin, in XY plane, emitting along the -Z axis, optionally with a spot profile.
        disk,
        /// Environment map.
        /// Top pole is aligned with +Y axis, matching OpenEXR latlong specification.
        /// LatLong 0,0 -> points in positive Z direction.
        /// LatLong 0,Pi/2 -> points in positive X direction.
        dome,
        /// Constant environment light.
        constant,
    };
    SGL_ENUM_INFO(
        Type,
        {
            {Type::distant, "distant"},
            {Type::rectangular, "rectangular"},
            {Type::sphere, "sphere"},
            {Type::point, "point"},
            {Type::cylinder, "cylinder"},
            {Type::disk, "disk"},
            {Type::dome, "dome"},
            {Type::constant, "constant"},
        }
    );

    /// Create import-processing defaults for the specified type. Overlay authored properties afterward.
    static Properties make_properties(const ImportOptions& options, Type type);
    /// Import-processing properties. Initialize with make_properties() after the light type is known.
    Properties properties;
    /// Light name.
    std::string name;
    /// Reference to the object in the source asset from which this light was imported.
    std::optional<SourceAssetRef> source_asset_ref;
    /// Light type.
    Type type;
    /// Light color in the rendering color space.
    float3 color = float3(1.f);
    /// Scalar light intensity.
    float intensity = 1.f;
    /// Light exposure in stops.
    float exposure = 0.f;
    /// Whether emission is divided by the light's world-space size factor.
    bool normalize = false;
    /// Whether the color temperature multiplier is enabled.
    bool enable_color_temperature = false;
    /// Color temperature in Kelvin.
    float color_temperature = 6500.f;

    /// Whether the source scene requests render-visible geometry for this light.
    bool geometry_visible = false;

    /// Angular diameter in degrees (for distant lights).
    float degree_angular_diameter = 0.53f;

    /// Width (for rectangular lights).
    float width = 1.f;
    /// Height (for rectangular lights).
    float height = 1.f;

    /// Radius (for sphere/cylinder/disk lights).
    float radius = 0.5f;
    /// Length along local X (for cylinder lights).
    float length = 1.f;
    /// Whether a sphere light should be rendered as a zero-area point light.
    bool treat_as_point = false;
    /// Whether a cylinder light may be approximated as a zero-radius line light.
    bool treat_as_line = false;
    /// Whether a sphere light keeps its sampled radius smaller than the receiver distance.
    bool enable_virtual_sphere_shrinking = false;
    /// Whether the light uses an angular emission profile.
    bool enable_shaping = false;
    /// Outer half-angle in degrees for directional emission shaping.
    float shaping_cone_angle = 90.f;
    /// Fraction of the cone occupied by the smooth falloff.
    float shaping_cone_softness = 0.f;
    /// Off-axis cosine power exponent used to focus emission.
    float shaping_focus = 0.f;

    /// Environment map path (for dome lights).
    std::string env_map_path;
    /// Environment map pole axis in the node's local space (for dome lights).
    EnvMapPoleAxis pole_axis = EnvMapPoleAxis::y;
};
SGL_ENUM_REGISTER(ImporterLight::Type);

class FALCOR_API ImporterScene : public Object {
public:
    /// Authored texture cap for scene creation. Merging takes the smaller nonzero limit; zero means unrestricted.
    uint32_t max_mip_count{0};

    /// Target image origin convention for scenes created from this importer scene.
    ///
    /// This is the preferred scene-wide destination convention. Individual
    /// ImporterMesh instances still describe their own authored texcoord
    /// convention via ImporterMesh::uv_origin and are converted on load if it
    /// differs from this scene convention.
    UVOrigin uv_origin = UVOrigin::upper_left;

    /// Material array.
    std::vector<ImporterMaterial> materials;
    /// Embedded asset array.
    std::vector<ImporterAsset> assets;
    /// Texture array.
    std::vector<ImporterTexture> textures;
    /// Mesh array.
    std::vector<ImporterMesh> meshes;
    /// Curve array.
    std::vector<ImporterCurve> curves;
    /// Node array.
    std::vector<ImporterNode> nodes;
    /// Skin array.
    std::vector<ImporterSkin> skins;
    /// Camera array.
    std::vector<ImporterCamera> cameras;
    /// Light array.
    std::vector<ImporterLight> lights;
    /// Prototype array.
    std::vector<ImporterPrototype> prototypes;
    /// Root node indices.
    std::vector<int> root_nodes;

    /// Animation data (single animation per scene, empty channels if no animation).
    ImporterAnimation animation;

    /// Authored playback interval in seconds, independent of key times. Invalid when unspecified.
    /// Composing imports unions their effective animation ranges when either has an authored interval.
    AnimationTimeRange playback_range;

    /// Authored playback interval, or the union of transform and geometry sample intervals when unspecified.
    /// Invalid when neither an authored interval nor animation samples exist. Does not shift key timestamps.
    AnimationTimeRange animation_range() const;

    /// Calculate AABBs for all meshes and nodes in the scene.
    void calculate_aabbs();

    /// Bake a skeleton pose into static mesh vertices. Omitted time uses the stored node transforms;
    /// time_seconds evaluates imported animation at that time in seconds, including animated joint ancestors.
    /// Output node transforms and ordinary animation remain unchanged. The sampled world pose is converted into
    /// each output mesh's stored local frame; subsequent scene animation can still move the baked geometry.
    /// Consumes all skin associations and binding data after deformation.
    void bake_pose(std::optional<double> time_seconds = std::nullopt);

    /// Collapse exact opposite-winding triangles independently within each mesh, using local positions.
    /// Never compares triangles across meshes or instances. Shared meshes are processed once without changing
    /// mesh indices or node references. Skips meshes referenced by skinned nodes or carrying skinning attributes.
    /// Returns the number of triangles removed from mesh definitions, independent of their instance counts.
    size_t collapse_exact_opposite_triangles();

    /// Rescale the scene to fit within a uniform box of the specified size
    /// This applies a uniform scale and translation to center the scene
    /// @param box_size The size of the target box (default: 1.0)
    void rescale_to_fit(float box_size = 1.0f);

    /// Get the overall AABB of the entire scene (union of all root node AABBs).
    AABB get_scene_aabb() const;

    /// Whether a camera is attached to a node reachable from a root, including prototype instances.
    /// Unused camera definitions and disconnected nodes do not count.
    bool has_camera() const;

    /// Append a perspective camera and root node fitted using the free-camera best-view algorithm.
    /// Recomputes mesh and curve bounds and includes transformed prototype instances. Uses stored node
    /// transforms and geometry sample zero; animation and live-scene callbacks are not evaluated.
    /// Always adds a camera, even when others exist. Without finite geometry bounds, looks at the origin
    /// from (0, 0, 5). Appending may invalidate references to camera and node vector elements.
    /// @param fov_degrees Vertical field of view in degrees, strictly between 0 and 180.
    /// @param aspect Width/height ratio used for fitting. Must be positive and finite.
    /// @param name Name assigned to both the camera and its root node.
    /// @return Index of the added camera in cameras.
    int add_framed_camera(float fov_degrees = 50.f, float aspect = 1.f, std::string_view name = "FramedCamera");

    /// Flatten the node hierarchy by removing parent-child relationships
    /// and accumulating transforms into world space. Only nodes with meshes are preserved.
    void flatten_node_hierarchy();

    /// Replace all materials to be gray diffuse only.
    /// Remove all lights from the scene and add a single constant light.
    void make_clay_scene();
};

} // namespace falcor
