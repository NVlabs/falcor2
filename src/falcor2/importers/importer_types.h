// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/core/types.h"
#include "falcor2/core/object.h"
#include "falcor2/core/macros.h"
#include "falcor2/core/properties.h"
#include "falcor2/utils/aabb.h"
#include "falcor2/importers/fwd.h"

#include "sgl/core/data_type.h"

#include <cstddef>
#include <functional>
#include <vector>
#include <string>
#include <map>
#include <variant>
#include <span>

namespace falcor {

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

    /// Index for attributes that can have multiple sets, e.g., TEXCOORD_0, TEXCOORD_1
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

/// Minimal information to describe an attribute to be created using ensure_attributes,
/// given known semantic and index.
struct FALCOR_API ImporterMeshDefaultAttribute {
    ImporterSemantic semantic{ImporterSemantic::unknown};
    uint32_t index{0};
    uint32_t components{0}; // Leave as 0 to use default for semantic
};

/// A buffer containing vertex data. Layout is described by ImporterMeshAttribute.
struct FALCOR_API ImporterMeshBuffer {

    /// Buffer data
    std::vector<uint8_t> data;

    // Stride in bytes between vertices
    size_t stride;
};

/// Helper class to make it simple to read/write a stream of a given
/// attribute type for the vertices of a mesh.
template<typename T>
struct ImporterVertexStream {
    uint8_t* data;
    size_t stride;
    size_t size;

    bool valid() const { return data != nullptr; }

    T* first();
    const T* first() const;

    const T& operator[](size_t index) const;
    T& operator[](size_t index);
};

/// Helper class to make it simple to read/write a stream of a given
/// attribute type for the vertices of a mesh.
template<typename T>
struct ImporterVertexStreamRO {
    const uint8_t* data;
    size_t stride;
    size_t size;

    bool valid() const { return data != nullptr; }

    const T* first() const;

    const T& operator[](size_t index) const;
};


/// Mesh data imported from a file. Represents a set of buffers containing
/// vertex data, and a set of subgeometries with indices into the vertex data.
class FALCOR_API ImporterMesh {
public:
    /// Mesh name.
    std::string name;

    struct Subgeometry {
        /// Subgeometry name.
        std::string name;
        /// Triangle indices.
        std::vector<uint3> indices;
        /// Material name.
        std::string material_name;
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

    /// Generate normals for mesh vertices based on face geometry.
    void add_normals_from_faces();

    /// Generate tangents for mesh vertices using UV coordinates and MikkTSpace algorithm.
    void add_tangents_from_uvs();

    /// Generate tangents for mesh vertices using normals (orthonormal basis).
    void add_tangents_from_normals();

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
            void* data = nullptr;
            size_t stride = 0;
            get_stream_info(attrib, data, stride);
            return ImporterVertexStream<T>{reinterpret_cast<uint8_t*>(data), stride, m_vertex_count};
        } else {
            return {};
        }
    }

    /// Get a typed 'VertexStream' for a given attribute to give an easy way
    /// to read/write the vertex data.
    template<typename T>
    ImporterVertexStreamRO<T> get_stream(const ImporterMeshAttribute& attrib) const
    {
        if (attrib.valid()) {
            const void* data = nullptr;
            size_t stride = 0;
            get_stream_info(attrib, data, stride);
            return ImporterVertexStreamRO<T>{reinterpret_cast<const uint8_t*>(data), stride, m_vertex_count};
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

    /// Get a hash value for a vertex at given index based on its attribute values, correctly
    /// skipping over any padding between elements.
    size_t hash_vertex(size_t vertex_index);

    /// Current vertex count.
    size_t vertex_count() const { return m_vertex_count; }

    /// Vertex attributes.
    const std::vector<ImporterMeshAttribute>& attributes() const { return m_attributes; }

    /// Vertex buffers.
    const std::vector<ImporterMeshBuffer>& buffers() const { return m_buffers; }
    std::vector<ImporterMeshBuffer>& buffers() { return m_buffers; }

    /// Vertex positions.
    const ImporterMeshAttribute& position_attribute() const { return m_positions; }
    ImporterVertexStream<float3> position_stream() { return get_stream<float3>(m_positions); }
    ImporterVertexStreamRO<float3> position_stream() const { return get_stream<float3>(m_positions); }

    /// Vertex normals.
    const ImporterMeshAttribute& normal_attribute() const { return m_normals; }
    ImporterVertexStream<float3> normal_stream() { return get_stream<float3>(m_normals); }
    ImporterVertexStreamRO<float3> normal_stream() const { return get_stream<float3>(m_normals); }

    /// Vertex tangents.
    const ImporterMeshAttribute& tangent_attribute() const { return m_tangents; }
    ImporterVertexStream<float3> tangent_stream() { return get_stream<float3>(m_tangents); }
    ImporterVertexStreamRO<float3> tangent_stream() const { return get_stream<float3>(m_tangents); }

    /// Vertex handedness.
    const ImporterMeshAttribute& handedness_attribute() const { return m_handedness; }
    ImporterVertexStream<float> handedness_stream() { return get_stream<float>(m_handedness); }
    ImporterVertexStreamRO<float> handedness_stream() const { return get_stream<float>(m_handedness); }

    /// Vertex texcoords (0) + helpers to access stream as float2.
    const ImporterMeshAttribute& texcoord_attribute() const { return m_texcoords; }
    ImporterVertexStream<float2> texcoord_stream() { return get_stream<float2>(m_texcoords); }
    ImporterVertexStreamRO<float2> texcoord_stream() const { return get_stream<float2>(m_texcoords); }

    /// Vertex colors (0) + helpers to access stream as rgb float3.
    const ImporterMeshAttribute& color_attribute() const { return m_colors; }
    ImporterVertexStream<float3> color_stream() { return get_stream<float3>(m_colors); }
    ImporterVertexStreamRO<float3> color_stream() const { return get_stream<float3>(m_colors); }

private:
    // Current vertex count
    size_t m_vertex_count = 0;

    // Vertex attributes and buffers
    std::vector<ImporterMeshAttribute> m_attributes;
    std::vector<ImporterMeshBuffer> m_buffers;

    // Cached commonly used attributes for easy access
    ImporterMeshAttribute m_positions;
    ImporterMeshAttribute m_normals;
    ImporterMeshAttribute m_tangents;
    ImporterMeshAttribute m_handedness;
    ImporterMeshAttribute m_texcoords;
    ImporterMeshAttribute m_colors;

    /// Create a new empty vertex buffer. The attribute(s) that refer to this buffer
    /// must be created before use. Pass stride==0 to have the stride automatically
    /// calculated from the attributes that refer to this buffer.
    void create_buffer(size_t buffer_index, size_t stride = 0);
};


template<typename T>
T* ImporterVertexStream<T>::first()
{
    SGL_CHECK(data != nullptr, "Invalid vertex stream");
    return reinterpret_cast<T*>(data);
}

template<typename T>
const T* ImporterVertexStream<T>::first() const
{
    SGL_CHECK(data != nullptr, "Invalid vertex stream");
    return reinterpret_cast<const T*>(data);
}

template<typename T>
inline const T& ImporterVertexStream<T>::operator[](size_t index) const
{
    SGL_CHECK(index < size, "Vertex index out of bounds");
    return *reinterpret_cast<const T*>(data + index * stride);
}

template<typename T>
inline T& ImporterVertexStream<T>::operator[](size_t index)
{
    SGL_CHECK(index < size, "Vertex index out of bounds");
    return *reinterpret_cast<T*>(data + index * stride);
}


template<typename T>
const T* ImporterVertexStreamRO<T>::first() const
{
    SGL_CHECK(data != nullptr, "Invalid vertex stream");
    return reinterpret_cast<const T*>(data);
}

template<typename T>
inline const T& ImporterVertexStreamRO<T>::operator[](size_t index) const
{
    SGL_CHECK(index < size, "Vertex index out of bounds");
    return *reinterpret_cast<const T*>(data + index * stride);
}

/// Indexing mode for curve line segments.
enum class CurveIndexingMode {
    /// Each segment is defined by a pair of indices: indices[2k], indices[2k+1].
    list,
    /// Each segment is defined by a single index: positions[indices[k]] and positions[indices[k]+1].
    successive,
};
SGL_ENUM_INFO(
    CurveIndexingMode,
    {
        {CurveIndexingMode::list, "list"},
        {CurveIndexingMode::successive, "successive"},
    }
);
SGL_ENUM_REGISTER(CurveIndexingMode);

/// Imported curve data.
struct FALCOR_API ImporterCurve {
    /// Curve name.
    std::string name;
    /// Material name.
    std::string material_name;
    /// Curve vertex positions.
    std::vector<float3> positions;
    /// Radius at each vertex.
    std::vector<float> radii;
    /// Pairs of indices defining line segments.
    std::vector<uint32_t> indices;
    /// Indexing mode for the index buffer.
    CurveIndexingMode indexing_mode{CurveIndexingMode::list};
    /// Local bounding box.
    AABB local_aabb;

    /// Calculate local AABB from positions and radii.
    void calculate_local_aabb();
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

/// Alpha modes for material transparency (from glTF spec)
enum class AlphaMode {
    /// The alpha value is ignored and the rendered output is fully opaque.
    opaque = 0,
    /// The rendered output is either fully opaque or fully transparent depending on the alpha value and the
    /// specified alpha cutoff value.
    mask = 1,
    // The alpha value is used to composite the source and destination areas.
    blend = 2,
};
SGL_ENUM_INFO(
    AlphaMode,
    {
        {AlphaMode::opaque, "opaque"},
        {AlphaMode::mask, "mask"},
        {AlphaMode::blend, "blend"},
    }
);
SGL_ENUM_REGISTER(AlphaMode);

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
    /// Minification filter mode (simplified from glTF mipmap modes)
    TextureFilterMode min_filter = TextureFilterMode::linear;
    /// S (U) wrapping mode
    TextureWrapMode wrap_s = TextureWrapMode::repeat;
    /// T (V) wrapping mode
    TextureWrapMode wrap_t = TextureWrapMode::repeat;

    auto operator<=>(const ImporterTexture&) const = default;
};

/// Prototype for instancing more complex objects than simple meshes.
/// Allows nested instancing and instancing of larger groups of meshes.
struct FALCOR_API ImporterPrototype {
    /// Prototype name.
    std::string name;
    /// Index into ImporterScene::nodes, root node of the hierarchy.
    int root_node = -1;
};

/// Determines component count and interpolation method for animation keyframes.
enum class AnimationValueType {
    /// Scalar float (lerp)
    float_,
    /// 3-component float vector(lerp)
    float3,
    /// 4-component float quaternion (slerp)
    quatf,
};
SGL_ENUM_INFO(
    AnimationValueType,
    {
        {AnimationValueType::float_, "float"},
        {AnimationValueType::float3, "float3"},
        {AnimationValueType::quatf, "quatf"},
    }
);
SGL_ENUM_REGISTER(AnimationValueType);

/// Interpolation mode for animation keyframes.
enum class AnimationInterpolationMode {
    /// No interpolation, use left keyframe value
    step,
    /// Linear interpolation (lerp for float/float3, slerp for quatf)
    linear,
    /// Catmull-Rom auto-tangents
    cubic_hermite,
    /// Explicit in/out tangents per keyframe
    cubic_spline,
};
SGL_ENUM_INFO(
    AnimationInterpolationMode,
    {
        {AnimationInterpolationMode::step, "step"},
        {AnimationInterpolationMode::linear, "linear"},
        {AnimationInterpolationMode::cubic_hermite, "cubic_hermite"},
        {AnimationInterpolationMode::cubic_spline, "cubic_spline"},
    }
);
SGL_ENUM_REGISTER(AnimationInterpolationMode);

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
    /// Flat storage of all animation channels.
    std::vector<ImporterAnimationChannel> channels;

    /// The duration (max keyframe time across all channels).
    float duration() const;
};

struct FALCOR_API ImporterNode {
    /// Node name.
    std::string name;
    /// Node transformation matrix (i.e. transform from object space to parent or world space).
    float4x4 transform = float4x4::identity();

    // TODO(chrisc): Replace use of 'int' indices with mesh/node id types

    /// Index into ImporterScene::meshes, -1 if no mesh.
    int mesh_index = -1;
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
    /// Material name.
    std::string name;

    // TODO(chrisc): Replace use of 'int' as texture index with a proper texture id type.

    /// Material parameters.
    /// For Usd, if UsdPreviewSurface is available, it is flattened here. For MDL, nodes must be used.
    Properties params;
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
    /// Focus distance.
    float focus_distance;
    /// Focal length (millimeters).
    float focal_length;
    /// F-stop value.
    float fstop;
    /// Depth range (near, far).
    float2 depth_range;
    /// Projection type.
    Projection projection;
    /// Field of view direction.
    FOVDirection fov_direction;
    /// Aperture size.
    float aperture;
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
        /// Disk at the origin, in XY plane, emitting along the -Z axis.
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
            {Type::disk, "disk"},
            {Type::dome, "dome"},
            {Type::constant, "constant"},
        }
    );

    /// Light name.
    std::string name;
    /// Light type.
    Type type;
    /// Light intensity.
    float3 intensity;

    /// Angular diameter in degrees (for distant lights).
    float degree_angular_diameter = 0.53f;

    /// Width (for rectangular lights).
    float width = 1.f;
    /// Height (for rectangular lights).
    float height = 1.f;

    /// Radius (for sphere/disk lights).
    float radius = 1.f;

    /// Environment map path (for dome lights).
    std::string env_map_path;
};
SGL_ENUM_REGISTER(ImporterLight::Type);

class FALCOR_API ImporterScene : public Object {
public:
    /// Material array.
    std::vector<ImporterMaterial> materials;
    /// Texture array.
    std::vector<ImporterTexture> textures;
    /// Mesh array.
    std::vector<ImporterMesh> meshes;
    /// Curve array.
    std::vector<ImporterCurve> curves;
    /// Node array.
    std::vector<ImporterNode> nodes;
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

    /// Calculate AABBs for all meshes and nodes in the scene.
    void calculate_aabbs();

    /// Rescale the scene to fit within a uniform box of the specified size
    /// This applies a uniform scale and translation to center the scene
    /// @param box_size The size of the target box (default: 1.0)
    void rescale_to_fit(float box_size = 1.0f);

    /// Get the overall AABB of the entire scene (union of all root node AABBs).
    AABB get_scene_aabb() const;

    /// Adds a new camera and returns its index.
    /// Uses outlier-resistant bounds/center from mesh AABBs, appends a camera node,
    /// and returns the newly added camera index.
    int add_default_camera_robust_fit(
        float fov_degrees = 50.f,
        float aspect = 16.f / 9.f,
        float coverage = 0.9f,
        float outlier_percentile = 0.05f
    );

    /// Adds a new camera and returns its index.
    /// Evaluates multiple view directions over per-mesh AABBs, appends a camera node,
    /// and returns the newly added camera index.
    int add_default_camera_best_view(
        float fov_degrees = 50.f,
        float aspect = 16.f / 9.f,
        float coverage = 0.9f,
        float outlier_percentile = 0.05f
    );

    /// Flatten the node hierarchy by removing parent-child relationships
    /// and accumulating transforms into world space. Only nodes with meshes are preserved.
    void flatten_node_hierarchy();

    /// Replace all materials to be gray diffuse only.
    /// Remove all lights from the scene and add a single constant light.
    void make_clay_scene();
};

} // namespace falcor
