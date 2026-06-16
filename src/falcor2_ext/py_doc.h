/*
  This file contains docstrings for use in the Python bindings.
  Do not edit! They were automatically extracted by pybind11_mkdoc.
 */

#define __EXPAND(x)                                      x
#define __COUNT(_1, _2, _3, _4, _5, _6, _7, COUNT, ...)  COUNT
#define __VA_SIZE(...)                                   __EXPAND(__COUNT(__VA_ARGS__, 7, 6, 5, 4, 3, 2, 1))
#define __CAT1(a, b)                                     a ## b
#define __CAT2(a, b)                                     __CAT1(a, b)
#define __DOC1(n1)                                       __doc_##n1
#define __DOC2(n1, n2)                                   __doc_##n1##_##n2
#define __DOC3(n1, n2, n3)                               __doc_##n1##_##n2##_##n3
#define __DOC4(n1, n2, n3, n4)                           __doc_##n1##_##n2##_##n3##_##n4
#define __DOC5(n1, n2, n3, n4, n5)                       __doc_##n1##_##n2##_##n3##_##n4##_##n5
#define __DOC6(n1, n2, n3, n4, n5, n6)                   __doc_##n1##_##n2##_##n3##_##n4##_##n5##_##n6
#define __DOC7(n1, n2, n3, n4, n5, n6, n7)               __doc_##n1##_##n2##_##n3##_##n4##_##n5##_##n6##_##n7
#define DOC(...)                                         __EXPAND(__EXPAND(__CAT2(__DOC, __VA_SIZE(__VA_ARGS__)))(__VA_ARGS__))

#if defined(__GNUG__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#endif


static const char *__doc_MaterialX_v1_39_5_GenContext = R"doc()doc";

static const char *__doc_MaterialX_v1_39_5_ShaderGraph = R"doc()doc";

static const char *__doc_falcor_AABB = R"doc(Axis-Aligned Bounding Box.)doc";

static const char *__doc_falcor_AABB_AABB = R"doc(Default constructor. Creates an invalid AABB.)doc";

static const char *__doc_falcor_AABB_AABB_2 = R"doc(Constructor with explicit min/max points.)doc";

static const char *__doc_falcor_AABB_center = R"doc(Get the center of the AABB.)doc";

static const char *__doc_falcor_AABB_expand = R"doc(Expand the AABB to include a point.)doc";

static const char *__doc_falcor_AABB_expand_2 = R"doc(Expand the AABB to include a sphere (point + radius).)doc";

static const char *__doc_falcor_AABB_expand_3 = R"doc(Expand the AABB to include another AABB.)doc";

static const char *__doc_falcor_AABB_is_valid = R"doc(Check if the AABB is valid (min <= max for all components).)doc";

static const char *__doc_falcor_AABB_max = R"doc(Maximum point of the AABB.)doc";

static const char *__doc_falcor_AABB_min = R"doc(Minimum point of the AABB.)doc";

static const char *__doc_falcor_AABB_size = R"doc(Get the size (extent) of the AABB.)doc";

static const char *__doc_falcor_AABB_transform =
R"doc(Transform the AABB by a matrix (creates new AABB containing all 8
corners).)doc";

static const char *__doc_falcor_AliasTable1D =
R"doc(Discrete 1D probability distribution for importance sampling.

Given a set of non-negative weights, this class builds a discrete
probability distribution that can be sampled on the device. It
computes a normalized PDF and creates an alias table for efficient
sampling using Vose's alias method (Vose, 1991), which allows O(n)
construction and O(1) sampling.

The distribution stores both PDF and alias table in device buffers for
shader access. Sampling returns an index in [0, size-1] with
probability proportional to the original weight at that index.

Definitions: sum = sum_{i=0}^{size-1} weight[i] pdf[i] = weight[i] /
sum alias_table[i] = probability of the main index i, with the alias
index for the remaining probability.)doc";

static const char *__doc_falcor_AliasTable1D_AliasTable1D =
R"doc(Construct a discrete distribution from a set of weights.

Parameter ``device``:
    Device for buffer allocation.

Parameter ``func``:
    Non-negative weights defining the distribution.

Parameter ``label``:
    Debug label device resources.)doc";

static const char *__doc_falcor_AliasTable1D_AliasTable1D_2 = R"doc()doc";

static const char *__doc_falcor_AliasTable1D_AliasTable1D_3 = R"doc()doc";

static const char *__doc_falcor_AliasTable1D_Entry = R"doc(Alias table entry.)doc";

static const char *__doc_falcor_AliasTable1D_Entry_alias = R"doc(Alias index for the remaining probability.)doc";

static const char *__doc_falcor_AliasTable1D_Entry_prob = R"doc(Probability of the original index.)doc";

static const char *__doc_falcor_AliasTable1D_alias_table = R"doc(Alias table entries.)doc";

static const char *__doc_falcor_AliasTable1D_alias_table_buffer = R"doc(Device buffer containing the alias table.)doc";

static const char *__doc_falcor_AliasTable1D_m_alias_table = R"doc()doc";

static const char *__doc_falcor_AliasTable1D_m_alias_table_buffer = R"doc()doc";

static const char *__doc_falcor_AliasTable1D_m_pdf = R"doc()doc";

static const char *__doc_falcor_AliasTable1D_m_pdf_buffer = R"doc()doc";

static const char *__doc_falcor_AliasTable1D_m_size = R"doc()doc";

static const char *__doc_falcor_AliasTable1D_operator_assign = R"doc()doc";

static const char *__doc_falcor_AliasTable1D_operator_assign_2 = R"doc()doc";

static const char *__doc_falcor_AliasTable1D_pdf = R"doc(Normalize probability density function.)doc";

static const char *__doc_falcor_AliasTable1D_pdf_buffer = R"doc(Device buffer containing the probability density function.)doc";

static const char *__doc_falcor_AliasTable1D_size = R"doc(Number of elements in the distribution.)doc";

static const char *__doc_falcor_AliasTable1D_write_to_cursor = R"doc()doc";

static const char *__doc_falcor_AliasTable1D_write_to_cursor_2 =
R"doc(Write the handle data to a shader cursor for GPU binding.

Template parameter ``TCursor``:
    Shader cursor type.

Parameter ``cursor``:
    Shader cursor to write to.)doc";

static const char *__doc_falcor_AlphaMode = R"doc(Alpha modes for material transparency (from glTF spec))doc";

static const char *__doc_falcor_AlphaMode_blend =
R"doc(The rendered output is either fully opaque or fully transparent
depending on the alpha value and the specified alpha cutoff value.)doc";

static const char *__doc_falcor_AlphaMode_info = R"doc()doc";

static const char *__doc_falcor_AlphaMode_mask =
R"doc(The rendered output is either fully opaque or fully transparent
depending on the alpha value and the specified alpha cutoff value.)doc";

static const char *__doc_falcor_AlphaMode_opaque = R"doc(The alpha value is ignored and the rendered output is fully opaque.)doc";

static const char *__doc_falcor_Animation = R"doc(A collection of animation channels.)doc";

static const char *__doc_falcor_AnimationChannel = R"doc(A single animation channel storing typed keyframe data.)doc";

static const char *__doc_falcor_AnimationChannel_Accessor =
R"doc(Accessor caches the last queried keyframe index for sequential
playback optimization. Use the default (nullptr) for single-threaded
evaluation, or provide an external Accessor for thread-safe concurrent
evaluation of the same channel.)doc";

static const char *__doc_falcor_AnimationChannel_Accessor_last_index = R"doc()doc";

static const char *__doc_falcor_AnimationChannel_components_per_value = R"doc(Returns the number of components per keyframe value.)doc";

static const char *__doc_falcor_AnimationChannel_evaluate_float = R"doc()doc";

static const char *__doc_falcor_AnimationChannel_evaluate_float3 = R"doc()doc";

static const char *__doc_falcor_AnimationChannel_evaluate_impl =
R"doc(Evaluate N-component channel at time t. IsQuat enables quaternion-
specific behavior (slerp, hemisphere alignment, normalization).)doc";

static const char *__doc_falcor_AnimationChannel_evaluate_quatf = R"doc()doc";

static const char *__doc_falcor_AnimationChannel_find_keyframe_index =
R"doc(Find the keyframe segment index for time t, using the accessor as a
starting hint.)doc";

static const char *__doc_falcor_AnimationChannel_in_tangents =
R"doc(In-tangent per keyframe (same layout as values), only populated for
cubic_spline.)doc";

static const char *__doc_falcor_AnimationChannel_interpolation_mode = R"doc(Interpolation mode.)doc";

static const char *__doc_falcor_AnimationChannel_keyframe_count = R"doc(Returns the number of keyframes.)doc";

static const char *__doc_falcor_AnimationChannel_m_accessor = R"doc(Default accessor used when no external one is provided.)doc";

static const char *__doc_falcor_AnimationChannel_out_tangents =
R"doc(Out-tangent per keyframe (same layout as values), only populated for
cubic_spline.)doc";

static const char *__doc_falcor_AnimationChannel_times = R"doc(Keyframe timestamps in seconds.)doc";

static const char *__doc_falcor_AnimationChannel_value_type = R"doc(Value type of keyframe data.)doc";

static const char *__doc_falcor_AnimationChannel_values =
R"doc(Keyframe values (flat array: 1/3/4 floats per key depending on
value_type).)doc";

static const char *__doc_falcor_AnimationInterpolationMode = R"doc(Interpolation mode for animation keyframes.)doc";

static const char *__doc_falcor_AnimationInterpolationMode_cubic_hermite = R"doc(Catmull-Rom auto-tangents)doc";

static const char *__doc_falcor_AnimationInterpolationMode_cubic_spline = R"doc(Explicit in/out tangents per keyframe)doc";

static const char *__doc_falcor_AnimationInterpolationMode_info = R"doc()doc";

static const char *__doc_falcor_AnimationInterpolationMode_linear = R"doc(Linear interpolation (lerp for float/float3, slerp for quatf))doc";

static const char *__doc_falcor_AnimationInterpolationMode_step = R"doc(No interpolation, use left keyframe value)doc";

static const char *__doc_falcor_AnimationValueType =
R"doc(Determines component count and interpolation method for animation
keyframes.)doc";

static const char *__doc_falcor_AnimationValueType_float = R"doc(Scalar float (lerp))doc";

static const char *__doc_falcor_AnimationValueType_float3 = R"doc(3-component float vector(lerp))doc";

static const char *__doc_falcor_AnimationValueType_info = R"doc()doc";

static const char *__doc_falcor_AnimationValueType_quatf = R"doc(4-component float quaternion (slerp))doc";

static const char *__doc_falcor_Animation_DirtyFlags = R"doc(Dirty flags for Animation.)doc";

static const char *__doc_falcor_Animation_DirtyFlags_added = R"doc()doc";

static const char *__doc_falcor_Animation_DirtyFlags_none = R"doc()doc";

static const char *__doc_falcor_Animation_DirtyFlags_removed = R"doc()doc";

static const char *__doc_falcor_Animation_channel = R"doc(Get channel by index. Returns nullptr if index is out of range.)doc";

static const char *__doc_falcor_Animation_channels = R"doc(Flat storage of all animation channels.)doc";

static const char *__doc_falcor_Animation_class_descriptor = R"doc()doc";

static const char *__doc_falcor_Animation_class_name = R"doc()doc";

static const char *__doc_falcor_Animation_duration = R"doc(Returns the duration (max keyframe time across all channels).)doc";

static const char *__doc_falcor_Animation_kind = R"doc()doc";

static const char *__doc_falcor_Animation_reflect = R"doc(Reflect this class.)doc";

static const char *__doc_falcor_Animation_reflected_class_name = R"doc()doc";

static const char *__doc_falcor_Animation_static_class_descriptor = R"doc()doc";

static const char *__doc_falcor_Any =
R"doc(Type-erased storage for arbitrary objects.

This class resembles `std::any` but supports advanced customization by
exposing the underlying type-erased storage implementation Any::Base.
The Properties class uses Any when it needs to store things that
aren't part of the supported set of property types. The main reason
for using Any (as opposed to the builtin `std::any`) is to enable
seamless use in Python bindings, which requires access to Any::Base.

Instances of this type are copyable with reference semantics. The
class is not thread-safe(i.e., it may not be copied concurrently).)doc";

static const char *__doc_falcor_Any_Any = R"doc()doc";

static const char *__doc_falcor_Any_Any_2 = R"doc()doc";

static const char *__doc_falcor_Any_Any_3 = R"doc()doc";

static const char *__doc_falcor_Any_Any_4 = R"doc()doc";

static const char *__doc_falcor_Any_Any_5 = R"doc()doc";

static const char *__doc_falcor_Any_Base = R"doc()doc";

static const char *__doc_falcor_Any_Base_dec_ref = R"doc()doc";

static const char *__doc_falcor_Any_Base_inc_ref = R"doc()doc";

static const char *__doc_falcor_Any_Base_ptr = R"doc()doc";

static const char *__doc_falcor_Any_Base_ref_count = R"doc()doc";

static const char *__doc_falcor_Any_Base_type = R"doc()doc";

static const char *__doc_falcor_Any_Storage = R"doc()doc";

static const char *__doc_falcor_Any_Storage_Storage = R"doc()doc";

static const char *__doc_falcor_Any_Storage_ptr = R"doc()doc";

static const char *__doc_falcor_Any_Storage_type = R"doc()doc";

static const char *__doc_falcor_Any_Storage_value = R"doc()doc";

static const char *__doc_falcor_Any_base_ptr = R"doc(Access the underlying type-erased storage.)doc";

static const char *__doc_falcor_Any_base_ptr_2 = R"doc()doc";

static const char *__doc_falcor_Any_data = R"doc()doc";

static const char *__doc_falcor_Any_data_2 = R"doc()doc";

static const char *__doc_falcor_Any_has_value = R"doc()doc";

static const char *__doc_falcor_Any_m_storage = R"doc()doc";

static const char *__doc_falcor_Any_operator_assign = R"doc()doc";

static const char *__doc_falcor_Any_operator_assign_2 = R"doc()doc";

static const char *__doc_falcor_Any_operator_eq = R"doc()doc";

static const char *__doc_falcor_Any_operator_ne = R"doc()doc";

static const char *__doc_falcor_Any_release = R"doc()doc";

static const char *__doc_falcor_Any_type = R"doc()doc";

static const char *__doc_falcor_ArchiveBlobView = R"doc(Zero-copy view of a sized archive payload.)doc";

static const char *__doc_falcor_ArchiveBlobView_bytes = R"doc(Borrowed bytes. The backing archive bytes must outlive this view.)doc";

static const char *__doc_falcor_ArchiveCodec =
R"doc(Type-specific archive payload codec. Types may specialize this
template explicitly or define ADL archive hooks.)doc";

static const char *__doc_falcor_ArchiveCodec_2 = R"doc()doc";

static const char *__doc_falcor_ArchiveCodec_3 = R"doc()doc";

static const char *__doc_falcor_ArchiveCodec_4 = R"doc()doc";

static const char *__doc_falcor_ArchiveCodec_5 = R"doc()doc";

static const char *__doc_falcor_ArchiveCodec_6 = R"doc()doc";

static const char *__doc_falcor_ArchiveCodec_7 = R"doc()doc";

static const char *__doc_falcor_ArchiveCodec_8 = R"doc()doc";

static const char *__doc_falcor_ArchiveCodec_9 = R"doc()doc";

static const char *__doc_falcor_ArchiveCodec_10 = R"doc()doc";

static const char *__doc_falcor_ArchiveCodec_11 = R"doc()doc";

static const char *__doc_falcor_ArchiveCodec_12 = R"doc()doc";

static const char *__doc_falcor_ArchiveCodec_13 = R"doc()doc";

static const char *__doc_falcor_ArchiveCodec_14 = R"doc()doc";

static const char *__doc_falcor_ArchiveCodec_15 = R"doc()doc";

static const char *__doc_falcor_ArchiveCodec_16 = R"doc()doc";

static const char *__doc_falcor_ArchiveCodec_17 = R"doc()doc";

static const char *__doc_falcor_ArchiveCodec_18 = R"doc()doc";

static const char *__doc_falcor_ArchiveCodec_19 = R"doc()doc";

static const char *__doc_falcor_ArchiveCodec_20 = R"doc()doc";

static const char *__doc_falcor_ArchiveCodec_21 = R"doc()doc";

static const char *__doc_falcor_ArchiveCodec_22 = R"doc()doc";

static const char *__doc_falcor_ArchiveCodec_23 = R"doc()doc";

static const char *__doc_falcor_ArchiveCodec_24 = R"doc()doc";

static const char *__doc_falcor_ArchiveCodec_25 = R"doc()doc";

static const char *__doc_falcor_ArchiveCodec_26 = R"doc()doc";

static const char *__doc_falcor_ArchiveCodec_27 = R"doc()doc";

static const char *__doc_falcor_ArchiveDecoder = R"doc(Reads primitive archive wire encodings from an ArchiveSource.)doc";

static const char *__doc_falcor_ArchiveDecoder_ArchiveDecoder = R"doc(Create a decoder that reads from the given source.)doc";

static const char *__doc_falcor_ArchiveDecoder_empty = R"doc(Return true if the source has no unread bytes.)doc";

static const char *__doc_falcor_ArchiveDecoder_finish = R"doc(Verify that the source has been fully consumed.)doc";

static const char *__doc_falcor_ArchiveDecoder_m_source = R"doc()doc";

static const char *__doc_falcor_ArchiveDecoder_offset = R"doc(Return the current read offset in the source.)doc";

static const char *__doc_falcor_ArchiveDecoder_read_byte = R"doc(Read one raw byte.)doc";

static const char *__doc_falcor_ArchiveDecoder_read_bytes = R"doc(Read raw bytes without a length prefix.)doc";

static const char *__doc_falcor_ArchiveDecoder_read_sized = R"doc(Read a sized payload and return a borrowed span into the source.)doc";

static const char *__doc_falcor_ArchiveDecoder_read_sized_2 =
R"doc(Read a sized payload with an exact byte count and return a borrowed
pointer into the source.)doc";

static const char *__doc_falcor_ArchiveDecoder_read_varsint = R"doc(Read a zig-zag signed integer.)doc";

static const char *__doc_falcor_ArchiveDecoder_read_varsint_2 = R"doc(Read a varsint and reject values outside the inclusive range.)doc";

static const char *__doc_falcor_ArchiveDecoder_read_varuint = R"doc(Read a canonical unsigned LEB128 integer.)doc";

static const char *__doc_falcor_ArchiveDecoder_read_varuint_2 = R"doc(Read a varuint and reject values above max_value.)doc";

static const char *__doc_falcor_ArchiveDecoder_remaining = R"doc(Return the number of unread bytes.)doc";

static const char *__doc_falcor_ArchiveDecoder_skip = R"doc(Skip raw bytes.)doc";

static const char *__doc_falcor_ArchiveDeserializer =
R"doc(Reads ordered archive fields and nested records from an
ArchiveDecoder.)doc";

static const char *__doc_falcor_ArchiveDeserializer_ArchiveDeserializer = R"doc(Create a root deserializer over a caller-provided decoder.)doc";

static const char *__doc_falcor_ArchiveDeserializer_ArchiveDeserializer_2 = R"doc()doc";

static const char *__doc_falcor_ArchiveDeserializer_ArchiveDeserializer_3 = R"doc(Move construct a deserializer, transferring any owned nested source.)doc";

static const char *__doc_falcor_ArchiveDeserializer_ArchiveDeserializer_4 = R"doc()doc";

static const char *__doc_falcor_ArchiveDeserializer_FieldHeader = R"doc()doc";

static const char *__doc_falcor_ArchiveDeserializer_FieldHeader_encoding = R"doc()doc";

static const char *__doc_falcor_ArchiveDeserializer_FieldHeader_id = R"doc()doc";

static const char *__doc_falcor_ArchiveDeserializer_commit_requested_field_id = R"doc()doc";

static const char *__doc_falcor_ArchiveDeserializer_decoder = R"doc()doc";

static const char *__doc_falcor_ArchiveDeserializer_field =
R"doc(Read a field by ID. Returns false when missing and leaves value
unchanged.)doc";

static const char *__doc_falcor_ArchiveDeserializer_finish =
R"doc(Skip and validate all remaining fields, then verify that the record is
fully consumed.)doc";

static const char *__doc_falcor_ArchiveDeserializer_list = R"doc(Read a counted list of codec-backed element payloads.)doc";

static const char *__doc_falcor_ArchiveDeserializer_m_decoder = R"doc()doc";

static const char *__doc_falcor_ArchiveDeserializer_m_finished = R"doc()doc";

static const char *__doc_falcor_ArchiveDeserializer_m_owned_decoder = R"doc()doc";

static const char *__doc_falcor_ArchiveDeserializer_m_owned_source = R"doc()doc";

static const char *__doc_falcor_ArchiveDeserializer_m_pending_field = R"doc()doc";

static const char *__doc_falcor_ArchiveDeserializer_m_previous_encoded_field_id = R"doc()doc";

static const char *__doc_falcor_ArchiveDeserializer_m_previous_requested_field_id = R"doc()doc";

static const char *__doc_falcor_ArchiveDeserializer_operator_assign = R"doc()doc";

static const char *__doc_falcor_ArchiveDeserializer_operator_assign_2 = R"doc()doc";

static const char *__doc_falcor_ArchiveDeserializer_read_field_header = R"doc()doc";

static const char *__doc_falcor_ArchiveDeserializer_read_next_field_if_needed = R"doc()doc";

static const char *__doc_falcor_ArchiveDeserializer_record_list = R"doc(Read a counted list of nested archive records.)doc";

static const char *__doc_falcor_ArchiveDeserializer_section = R"doc(Read a nested record field. Returns std::nullopt when missing.)doc";

static const char *__doc_falcor_ArchiveDeserializer_skip_pending_field = R"doc()doc";

static const char *__doc_falcor_ArchiveDeserializer_validate_next_requested_field_id = R"doc()doc";

static const char *__doc_falcor_ArchiveEncoder = R"doc(Writes primitive archive wire encodings to an ArchiveSink.)doc";

static const char *__doc_falcor_ArchiveEncoder_ArchiveEncoder = R"doc(Create an encoder that writes to the given sink.)doc";

static const char *__doc_falcor_ArchiveEncoder_m_sink = R"doc()doc";

static const char *__doc_falcor_ArchiveEncoder_write_byte = R"doc(Write one raw byte.)doc";

static const char *__doc_falcor_ArchiveEncoder_write_bytes = R"doc(Write raw bytes without a length prefix.)doc";

static const char *__doc_falcor_ArchiveEncoder_write_sized = R"doc(Write a sized payload: varuint byte count followed by raw bytes.)doc";

static const char *__doc_falcor_ArchiveEncoder_write_sized_2 = R"doc(Write a sized payload: varuint byte count followed by raw bytes.)doc";

static const char *__doc_falcor_ArchiveEncoder_write_varsint = R"doc(Write a zig-zag signed integer.)doc";

static const char *__doc_falcor_ArchiveEncoder_write_varuint = R"doc(Write a canonical unsigned LEB128 integer.)doc";

static const char *__doc_falcor_ArchiveEncoding =
R"doc(Payload encoding stored in the low three bits of each archive field
header.)doc";

static const char *__doc_falcor_ArchiveEncoding_byte = R"doc(A single byte payload.)doc";

static const char *__doc_falcor_ArchiveEncoding_fixed12 = R"doc(Twelve raw bytes.)doc";

static const char *__doc_falcor_ArchiveEncoding_fixed16 = R"doc(Sixteen raw bytes.)doc";

static const char *__doc_falcor_ArchiveEncoding_fixed4 = R"doc(Four raw bytes.)doc";

static const char *__doc_falcor_ArchiveEncoding_fixed8 = R"doc(Eight raw bytes.)doc";

static const char *__doc_falcor_ArchiveEncoding_sized = R"doc(Varuint byte count followed by that many raw bytes.)doc";

static const char *__doc_falcor_ArchiveEncoding_varsint = R"doc(Zig-zag signed integer payload encoded as varuint.)doc";

static const char *__doc_falcor_ArchiveEncoding_varuint = R"doc(Canonical unsigned LEB128 payload.)doc";

static const char *__doc_falcor_ArchiveSerializer = R"doc(Writes ordered archive fields and nested records to an ArchiveEncoder.)doc";

static const char *__doc_falcor_ArchiveSerializer_ArchiveSerializer = R"doc(Create a root serializer over a caller-provided encoder.)doc";

static const char *__doc_falcor_ArchiveSerializer_ArchiveSerializer_2 = R"doc()doc";

static const char *__doc_falcor_ArchiveSerializer_ArchiveSerializer_3 =
R"doc(Move construct a serializer. Moving a serializer with open child
sections is rejected.)doc";

static const char *__doc_falcor_ArchiveSerializer_ArchiveSerializer_4 = R"doc()doc";

static const char *__doc_falcor_ArchiveSerializer_commit_field_id = R"doc()doc";

static const char *__doc_falcor_ArchiveSerializer_encoder = R"doc()doc";

static const char *__doc_falcor_ArchiveSerializer_field =
R"doc(Write a field. Field IDs must be nonzero and strictly increasing
within this record.)doc";

static const char *__doc_falcor_ArchiveSerializer_finish =
R"doc(Finish this serializer. Section serializers flush their sized payload
to the parent here.)doc";

static const char *__doc_falcor_ArchiveSerializer_list =
R"doc(Write a counted list of codec-backed element payloads as a sized
field.)doc";

static const char *__doc_falcor_ArchiveSerializer_list_2 =
R"doc(Write a counted list of codec-backed element payloads as a sized
field.)doc";

static const char *__doc_falcor_ArchiveSerializer_m_encoder = R"doc()doc";

static const char *__doc_falcor_ArchiveSerializer_m_finished = R"doc()doc";

static const char *__doc_falcor_ArchiveSerializer_m_open_child_sections = R"doc()doc";

static const char *__doc_falcor_ArchiveSerializer_m_owned_encoder = R"doc()doc";

static const char *__doc_falcor_ArchiveSerializer_m_owned_sink = R"doc()doc";

static const char *__doc_falcor_ArchiveSerializer_m_parent = R"doc()doc";

static const char *__doc_falcor_ArchiveSerializer_m_previous_field_id = R"doc()doc";

static const char *__doc_falcor_ArchiveSerializer_m_section_id = R"doc()doc";

static const char *__doc_falcor_ArchiveSerializer_operator_assign = R"doc()doc";

static const char *__doc_falcor_ArchiveSerializer_operator_assign_2 = R"doc()doc";

static const char *__doc_falcor_ArchiveSerializer_record_list = R"doc(Write a counted list of nested archive records as a sized field.)doc";

static const char *__doc_falcor_ArchiveSerializer_record_list_2 = R"doc(Write a counted list of nested archive records as a sized field.)doc";

static const char *__doc_falcor_ArchiveSerializer_section =
R"doc(Create a nested record field. The returned section must be finished
before the parent can continue.)doc";

static const char *__doc_falcor_ArchiveSerializer_validate_next_field_id = R"doc()doc";

static const char *__doc_falcor_ArchiveSink = R"doc(Raw byte destination used by ArchiveEncoder.)doc";

static const char *__doc_falcor_ArchiveSink_write_bytes = R"doc(Append raw bytes to the sink.)doc";

static const char *__doc_falcor_ArchiveSource = R"doc(Raw byte source used by ArchiveDecoder.)doc";

static const char *__doc_falcor_ArchiveSource_offset = R"doc(Return the current read offset.)doc";

static const char *__doc_falcor_ArchiveSource_read_bytes =
R"doc(Read raw bytes and advance the cursor. The returned pointer is
borrowed from the source.)doc";

static const char *__doc_falcor_ArchiveSource_remaining = R"doc(Return the number of unread bytes.)doc";

static const char *__doc_falcor_ArchiveSource_skip = R"doc(Advance the cursor without returning bytes.)doc";

static const char *__doc_falcor_ArrayView = R"doc()doc";

static const char *__doc_falcor_ArrayView_ArrayView = R"doc()doc";

static const char *__doc_falcor_ArrayView_ArrayView_2 = R"doc()doc";

static const char *__doc_falcor_ArrayView_get_ptr = R"doc()doc";

static const char *__doc_falcor_ArrayView_get_ptr_2 = R"doc()doc";

static const char *__doc_falcor_ArrayView_m_byte_stride = R"doc()doc";

static const char *__doc_falcor_ArrayView_m_data = R"doc()doc";

static const char *__doc_falcor_ArrayView_m_size = R"doc()doc";

static const char *__doc_falcor_ArrayView_operator_array = R"doc()doc";

static const char *__doc_falcor_ArrayView_operator_array_2 = R"doc()doc";

static const char *__doc_falcor_ArrayView_size = R"doc()doc";

static const char *__doc_falcor_AssetResolver = R"doc()doc";

static const char *__doc_falcor_AssetResolver_2 = R"doc()doc";

static const char *__doc_falcor_AssetResolver_AssetResolver = R"doc()doc";

static const char *__doc_falcor_AssetResolver_add_search_path = R"doc()doc";

static const char *__doc_falcor_AssetResolver_add_search_paths = R"doc()doc";

static const char *__doc_falcor_AssetResolver_class_name = R"doc()doc";

static const char *__doc_falcor_AssetResolver_clone = R"doc(Clones the whole AssetResolver, including the stack.)doc";

static const char *__doc_falcor_AssetResolver_get_search_paths = R"doc()doc";

static const char *__doc_falcor_AssetResolver_glob_files = R"doc(TODO: This probably should be a more global utility)doc";

static const char *__doc_falcor_AssetResolver_m_search_paths_stack = R"doc()doc";

static const char *__doc_falcor_AssetResolver_pop_scope = R"doc()doc";

static const char *__doc_falcor_AssetResolver_push_scope = R"doc()doc";

static const char *__doc_falcor_AssetResolver_resolve_path = R"doc()doc";

static const char *__doc_falcor_AssetResolver_resolve_path_pattern = R"doc()doc";

static const char *__doc_falcor_AssetResolver_shallow_clone =
R"doc(Clones only the current level of AssetResolver, it can no longer be
popped.)doc";

static const char *__doc_falcor_BitField =
R"doc(Utility for packing and unpacking bitfields within an unsigned integer
type.

Template parameter ``T``:
    Unsigned integer type used as the storage.

Template parameter ``BitOffset``:
    Bit position of the field's least significant bit.

Template parameter ``BitSize``:
    Number of bits in the field.)doc";

static const char *__doc_falcor_BitField_extract =
R"doc(Extract the field value from an integer.

Parameter ``base``:
    The integer to extract from.

Returns:
    The extracted field value.)doc";

static const char *__doc_falcor_BitField_insert =
R"doc(Insert a value into the field, preserving other bits.

Parameter ``base``:
    The integer to modify.

Parameter ``field``:
    The value to insert (must not exceed MAX_VALUE).)doc";

static const char *__doc_falcor_BlobCache =
R"doc(File-based binary blob cache indexed by SHA1 digests. Each cache key
maps to a directory containing one or more named payloads. The key
directory timestamp is used for LRU eviction.)doc";

static const char *__doc_falcor_BlobCache_BlobCache =
R"doc(Create a cache using the supplied options, or defaults when no options
are provided.

Parameter ``options``:
    Optional cache configuration.)doc";

static const char *__doc_falcor_BlobCache_BlobCache_2 = R"doc()doc";

static const char *__doc_falcor_BlobCache_BlobCache_3 = R"doc()doc";

static const char *__doc_falcor_BlobCache_CacheEntry = R"doc()doc";

static const char *__doc_falcor_BlobCache_CacheEntry_key_dir = R"doc()doc";

static const char *__doc_falcor_BlobCache_CacheEntry_last_access = R"doc()doc";

static const char *__doc_falcor_BlobCache_CacheEntry_total_size = R"doc()doc";

static const char *__doc_falcor_BlobCache_Error = R"doc(Blob cache error codes returned through std::error_code.)doc";

static const char *__doc_falcor_BlobCache_Error_invalid_sub_data_name = R"doc()doc";

static const char *__doc_falcor_BlobCache_Error_lock_timeout = R"doc()doc";

static const char *__doc_falcor_BlobCache_Error_ok = R"doc()doc";

static const char *__doc_falcor_BlobCache_Error_read_failed = R"doc()doc";

static const char *__doc_falcor_BlobCache_Error_write_failed = R"doc()doc";

static const char *__doc_falcor_BlobCache_Options = R"doc()doc";

static const char *__doc_falcor_BlobCache_Options_eviction_target_pct = R"doc(Evict down to this percentage of max_size.)doc";

static const char *__doc_falcor_BlobCache_Options_eviction_threshold_pct = R"doc(Trigger eviction when total size exceeds this percentage of max_size.)doc";

static const char *__doc_falcor_BlobCache_Options_flush_after_write =
R"doc(Whether to flush file streams after writes (default false for
performance).)doc";

static const char *__doc_falcor_BlobCache_Options_max_size = R"doc(Maximum total cache size in bytes (default 128 GB).)doc";

static const char *__doc_falcor_BlobCache_Options_root_dir =
R"doc(Root directory for the cache. Defaults to
platform::app_data_directory() / "blob_cache".)doc";

static const char *__doc_falcor_BlobCache_ScanResult = R"doc()doc";

static const char *__doc_falcor_BlobCache_ScanResult_entries = R"doc()doc";

static const char *__doc_falcor_BlobCache_ScanResult_total_size = R"doc()doc";

static const char *__doc_falcor_BlobCache_Stats = R"doc()doc";

static const char *__doc_falcor_BlobCache_Stats_entry_count = R"doc(Number of cache keys that contain at least one live payload.)doc";

static const char *__doc_falcor_BlobCache_Stats_max_size = R"doc(Configured cache size limit in bytes.)doc";

static const char *__doc_falcor_BlobCache_Stats_total_size = R"doc(Total size in bytes across all live payload files.)doc";

static const char *__doc_falcor_BlobCache_WriteStream = R"doc()doc";

static const char *__doc_falcor_BlobCache_WriteStream_WriteStream = R"doc()doc";

static const char *__doc_falcor_BlobCache_WriteStream_WriteStream_2 = R"doc()doc";

static const char *__doc_falcor_BlobCache_WriteStream_WriteStream_3 = R"doc()doc";

static const char *__doc_falcor_BlobCache_WriteStream_WriteStream_4 = R"doc()doc";

static const char *__doc_falcor_BlobCache_WriteStream_abort =
R"doc(Cancel the pending write and remove the temporary file if it still
exists.)doc";

static const char *__doc_falcor_BlobCache_WriteStream_commit =
R"doc(Publish the temporary file as the cache payload or throw
std::system_error on failure.)doc";

static const char *__doc_falcor_BlobCache_WriteStream_m_cache = R"doc()doc";

static const char *__doc_falcor_BlobCache_WriteStream_m_key = R"doc()doc";

static const char *__doc_falcor_BlobCache_WriteStream_m_stream = R"doc()doc";

static const char *__doc_falcor_BlobCache_WriteStream_m_sub_data_name = R"doc()doc";

static const char *__doc_falcor_BlobCache_WriteStream_m_tmp_path = R"doc()doc";

static const char *__doc_falcor_BlobCache_WriteStream_operator_assign = R"doc()doc";

static const char *__doc_falcor_BlobCache_WriteStream_operator_assign_2 = R"doc()doc";

static const char *__doc_falcor_BlobCache_WriteStream_stream = R"doc(The writable temporary file stream owned by this pending write.)doc";

static const char *__doc_falcor_BlobCache_WriteStream_stream_2 = R"doc()doc";

static const char *__doc_falcor_BlobCache_WriteStream_try_commit =
R"doc(Try to publish the temporary file as the cache payload.

Parameter ``ec``:
    Receives the failure reason when the commit fails.

Returns:
    True when the payload was published successfully.)doc";

static const char *__doc_falcor_BlobCache_class_name = R"doc()doc";

static const char *__doc_falcor_BlobCache_cleanup_temp_files = R"doc()doc";

static const char *__doc_falcor_BlobCache_clear =
R"doc(Remove all cache entries while preserving cache metadata such as the
lock file.)doc";

static const char *__doc_falcor_BlobCache_contains =
R"doc(Return true when a payload exists for key and optional sub-data name.

Parameter ``key``:
    Key identifying the cache entry.

Parameter ``sub_data_name``:
    Optional payload name within the cache entry. Uses
    DEFAULT_SUB_DATA_NAME when omitted.

Returns:
    True when the payload exists.)doc";

static const char *__doc_falcor_BlobCache_digest_to_hex = R"doc()doc";

static const char *__doc_falcor_BlobCache_entry_has_payload = R"doc()doc";

static const char *__doc_falcor_BlobCache_evict =
R"doc(Evict least-recently-used cache entries until the configured target
size is reached.)doc";

static const char *__doc_falcor_BlobCache_get_blob_path = R"doc()doc";

static const char *__doc_falcor_BlobCache_get_entry_size =
R"doc(Return the payload size in bytes, or std::nullopt when the payload is
missing.

Parameter ``key``:
    Key identifying the cache entry.

Parameter ``sub_data_name``:
    Optional payload name within the cache entry. Uses
    DEFAULT_SUB_DATA_NAME when omitted.

Returns:
    The payload size in bytes, or std::nullopt when the payload is
    missing.)doc";

static const char *__doc_falcor_BlobCache_get_key_dir = R"doc()doc";

static const char *__doc_falcor_BlobCache_get_path =
R"doc(Return the filesystem path that would contain the payload. The payload
may not exist.

Parameter ``key``:
    Key identifying the cache entry.

Parameter ``sub_data_name``:
    Optional payload name within the cache entry. Uses
    DEFAULT_SUB_DATA_NAME when omitted.

Returns:
    Filesystem path for the requested payload.)doc";

static const char *__doc_falcor_BlobCache_get_temp_path = R"doc()doc";

static const char *__doc_falcor_BlobCache_is_valid_sub_data_name = R"doc()doc";

static const char *__doc_falcor_BlobCache_m_lock_path = R"doc()doc";

static const char *__doc_falcor_BlobCache_m_mutex = R"doc()doc";

static const char *__doc_falcor_BlobCache_m_options = R"doc()doc";

static const char *__doc_falcor_BlobCache_m_root_dir = R"doc()doc";

static const char *__doc_falcor_BlobCache_maybe_evict = R"doc()doc";

static const char *__doc_falcor_BlobCache_operator_assign = R"doc()doc";

static const char *__doc_falcor_BlobCache_operator_assign_2 = R"doc()doc";

static const char *__doc_falcor_BlobCache_prepare_key_dir = R"doc()doc";

static const char *__doc_falcor_BlobCache_publish_temp_file = R"doc()doc";

static const char *__doc_falcor_BlobCache_read =
R"doc(Read an existing payload into out_data. The buffer size must exactly
match the payload size.

Parameter ``key``:
    Key identifying the cache entry.

Parameter ``out_data``:
    Destination buffer. Its size must exactly match the payload size.

Parameter ``sub_data_name``:
    Optional payload name within the cache entry. Uses
    DEFAULT_SUB_DATA_NAME when omitted.

Returns:
    True when the payload exists and was read into out_data.)doc";

static const char *__doc_falcor_BlobCache_read_2 =
R"doc(Read a payload into a new byte vector, or return std::nullopt when the
payload is missing.

Parameter ``key``:
    Key identifying the cache entry.

Parameter ``sub_data_name``:
    Optional payload name within the cache entry. Uses
    DEFAULT_SUB_DATA_NAME when omitted.

Returns:
    The payload bytes, or std::nullopt when the payload is missing.)doc";

static const char *__doc_falcor_BlobCache_read_as_file_stream =
R"doc(Open an existing payload as a read-only file stream, or return nullptr
when missing.

Parameter ``key``:
    Key identifying the cache entry.

Parameter ``sub_data_name``:
    Optional payload name within the cache entry. Uses
    DEFAULT_SUB_DATA_NAME when omitted.

Returns:
    A readable file stream for the payload, or nullptr when the
    payload is missing.)doc";

static const char *__doc_falcor_BlobCache_read_as_memory_mapped_file_stream =
R"doc(Open a non-empty payload as a read-only memory-mapped stream, or
return nullptr when missing or empty.

Parameter ``key``:
    Key identifying the cache entry.

Parameter ``sub_data_name``:
    Optional payload name within the cache entry. Uses
    DEFAULT_SUB_DATA_NAME when omitted.

Returns:
    A readable memory-mapped stream for the payload, or nullptr when
    the payload is missing or empty.)doc";

static const char *__doc_falcor_BlobCache_remove_payload = R"doc()doc";

static const char *__doc_falcor_BlobCache_scan_entries = R"doc()doc";

static const char *__doc_falcor_BlobCache_stats =
R"doc(Scan the cache directory and return current cache statistics.

Returns:
    Current cache statistics.)doc";

static const char *__doc_falcor_BlobCache_touch_key_dir = R"doc()doc";

static const char *__doc_falcor_BlobCache_try_read =
R"doc(Non-throwing fixed-buffer read variant.

Parameter ``key``:
    Key identifying the cache entry.

Parameter ``out_data``:
    Destination buffer. Its size must exactly match the payload size.

Parameter ``sub_data_name``:
    Optional payload name within the cache entry. Uses
    DEFAULT_SUB_DATA_NAME when omitted.

Parameter ``ec``:
    Receives the failure reason on validation or I/O failure.

Returns:
    True when the payload exists and was read into out_data. Missing
    payloads or size mismatches return false without setting ec.)doc";

static const char *__doc_falcor_BlobCache_try_write =
R"doc(Non-throwing write variant.

Parameter ``key``:
    Key identifying the cache entry.

Parameter ``data``:
    Bytes to store in the cache.

Parameter ``sub_data_name``:
    Optional payload name within the cache entry. Uses
    DEFAULT_SUB_DATA_NAME when omitted.

Parameter ``ec``:
    Receives the failure reason on validation, locking, or I/O
    failure.

Returns:
    True when the payload was written successfully.)doc";

static const char *__doc_falcor_BlobCache_validate_sub_data_name = R"doc()doc";

static const char *__doc_falcor_BlobCache_write =
R"doc(Store data under key and optional sub-data name, replacing any
existing payload.

Parameter ``key``:
    Key identifying the cache entry.

Parameter ``data``:
    Bytes to store in the cache.

Parameter ``sub_data_name``:
    Optional payload name within the cache entry. Uses
    DEFAULT_SUB_DATA_NAME when omitted.)doc";

static const char *__doc_falcor_BlobCache_write_stream =
R"doc(Start a streaming write to a temporary file. The payload is visible
only after WriteStream::commit().

Parameter ``key``:
    Key identifying the cache entry.

Parameter ``sub_data_name``:
    Optional payload name within the cache entry. Uses
    DEFAULT_SUB_DATA_NAME when omitted.

Returns:
    A pending write stream that publishes the payload on commit.)doc";

static const char *__doc_falcor_BufferArchiveSink = R"doc(ArchiveSink implementation backed by a growing byte buffer.)doc";

static const char *__doc_falcor_BufferArchiveSink_buffer = R"doc(Return the backing byte buffer.)doc";

static const char *__doc_falcor_BufferArchiveSink_bytes = R"doc(Return the written bytes.)doc";

static const char *__doc_falcor_BufferArchiveSink_empty = R"doc(Return true if no bytes have been written.)doc";

static const char *__doc_falcor_BufferArchiveSink_m_buffer = R"doc()doc";

static const char *__doc_falcor_BufferArchiveSink_reserve = R"doc(Reserve storage in the backing buffer.)doc";

static const char *__doc_falcor_BufferArchiveSink_size = R"doc(Return the number of bytes written.)doc";

static const char *__doc_falcor_BufferArchiveSink_write_bytes = R"doc(Append raw bytes to the backing buffer.)doc";

static const char *__doc_falcor_BufferHandle =
R"doc(An object used to convert CPU Buffer to the GPU BufferHandle, by
converting to either pointer or descriptor handle, based on the
backend.)doc";

static const char *__doc_falcor_BufferHandle_BufferHandle = R"doc()doc";

static const char *__doc_falcor_BufferHandle_BufferHandle_2 = R"doc()doc";

static const char *__doc_falcor_BufferHandle_data = R"doc(Data for direct upload into memory location with GPU BufferHandle.)doc";

static const char *__doc_falcor_BufferHandle_m_handle_data = R"doc()doc";

static const char *__doc_falcor_BufferHandle_to_dictionary =
R"doc(Used in get_uniforms() and get_this() calls to produce Python
dictionary.)doc";

static const char *__doc_falcor_BufferHandle_use_gpu_pointer = R"doc()doc";

static const char *__doc_falcor_BufferHandle_write_to_cursor = R"doc()doc";

static const char *__doc_falcor_BufferHandle_write_to_cursor_2 = R"doc()doc";

static const char *__doc_falcor_BufferHandle_write_to_cursor_3 = R"doc(Used when writing using Buffer or Shader Cursors.)doc";

static const char *__doc_falcor_Camera = R"doc()doc";

static const char *__doc_falcor_Camera_2 = R"doc()doc";

static const char *__doc_falcor_CameraUniforms = R"doc()doc";

static const char *__doc_falcor_CameraUniforms_bind = R"doc(Bind camera uniforms to a shader cursor.)doc";

static const char *__doc_falcor_CameraUniforms_dims = R"doc(Viewport dimensions in pixels.)doc";

static const char *__doc_falcor_CameraUniforms_height = R"doc(Viewport height in pixels.)doc";

static const char *__doc_falcor_CameraUniforms_image_u =
R"doc(Computed image_u vector (right direction scaled by FOV and aspect
ratio).)doc";

static const char *__doc_falcor_CameraUniforms_image_v = R"doc(Computed image_v vector (up direction scaled by FOV).)doc";

static const char *__doc_falcor_CameraUniforms_image_w = R"doc(Computed image_w vector (forward direction).)doc";

static const char *__doc_falcor_CameraUniforms_position = R"doc(Computed camera position (world space).)doc";

static const char *__doc_falcor_CameraUniforms_to_dictionary = R"doc(Write camera uniforms to a dictionary.)doc";

static const char *__doc_falcor_CameraUniforms_width = R"doc(Viewport width in pixels.)doc";

static const char *__doc_falcor_Camera_bind = R"doc(Bind camera uniforms to a shader cursor.)doc";

static const char *__doc_falcor_Camera_calc_clip_from_view = R"doc(Calculate the clip-from-view projection matrix.)doc";

static const char *__doc_falcor_Camera_calc_clip_from_view_2 =
R"doc(Calculate the clip-from-view projection matrix for the given viewport
dimensions.)doc";

static const char *__doc_falcor_Camera_calc_uniforms = R"doc(Calculate camera uniforms for the camera's viewport dimensions.)doc";

static const char *__doc_falcor_Camera_calc_uniforms_2 = R"doc(Calculate camera uniforms for the given viewport dimensions.)doc";

static const char *__doc_falcor_Camera_calc_view_from_world = R"doc(Calculate the view-from-world matrix.)doc";

static const char *__doc_falcor_Camera_class_descriptor = R"doc()doc";

static const char *__doc_falcor_Camera_class_name = R"doc()doc";

static const char *__doc_falcor_Camera_focal_length = R"doc(Focal length in mm.)doc";

static const char *__doc_falcor_Camera_focus_distance = R"doc(Focus distance in world units.)doc";

static const char *__doc_falcor_Camera_fov_y = R"doc(Vertical field of view in degrees.)doc";

static const char *__doc_falcor_Camera_fstop = R"doc(f-stop value.)doc";

static const char *__doc_falcor_Camera_height = R"doc(Viewport height in pixels.)doc";

static const char *__doc_falcor_Camera_m_focal_length = R"doc(Focal length in mm.)doc";

static const char *__doc_falcor_Camera_m_focus_distance = R"doc(Focus distance in world units.)doc";

static const char *__doc_falcor_Camera_m_fov_y = R"doc(Vertical field of view in degrees.)doc";

static const char *__doc_falcor_Camera_m_fstop = R"doc(f-stop value.)doc";

static const char *__doc_falcor_Camera_m_height = R"doc(Viewport height in pixels.)doc";

static const char *__doc_falcor_Camera_m_needs_recompute = R"doc(Flag indicating if the computed values need to be recomputed.)doc";

static const char *__doc_falcor_Camera_m_uniforms = R"doc(Computed camera uniforms for the camera's viewport dimensions.)doc";

static const char *__doc_falcor_Camera_m_width = R"doc(Viewport width in pixels.)doc";

static const char *__doc_falcor_Camera_needs_recompute = R"doc()doc";

static const char *__doc_falcor_Camera_on_entity_transform_changed = R"doc()doc";

static const char *__doc_falcor_Camera_recompute = R"doc(Recompute camera basis vectors from the entity's world transform.)doc";

static const char *__doc_falcor_Camera_reflect = R"doc(Reflect this class.)doc";

static const char *__doc_falcor_Camera_reflected_class_name = R"doc()doc";

static const char *__doc_falcor_Camera_set_focal_length = R"doc()doc";

static const char *__doc_falcor_Camera_set_focus_distance = R"doc()doc";

static const char *__doc_falcor_Camera_set_fov_y = R"doc()doc";

static const char *__doc_falcor_Camera_set_fstop = R"doc()doc";

static const char *__doc_falcor_Camera_set_height = R"doc()doc";

static const char *__doc_falcor_Camera_set_width = R"doc()doc";

static const char *__doc_falcor_Camera_static_class_descriptor = R"doc()doc";

static const char *__doc_falcor_Camera_to_dictionary = R"doc(Write camera uniforms to a dictionary.)doc";

static const char *__doc_falcor_Camera_width = R"doc(Viewport width in pixels.)doc";

static const char *__doc_falcor_Component = R"doc(Base class for scene components.)doc";

static const char *__doc_falcor_Component_Component = R"doc(Constructor.)doc";

static const char *__doc_falcor_Component_DirtyFlags = R"doc(Dirty flags for Component.)doc";

static const char *__doc_falcor_Component_DirtyFlags_added = R"doc()doc";

static const char *__doc_falcor_Component_DirtyFlags_animation_state = R"doc()doc";

static const char *__doc_falcor_Component_DirtyFlags_entity_transform = R"doc()doc";

static const char *__doc_falcor_Component_DirtyFlags_none = R"doc()doc";

static const char *__doc_falcor_Component_DirtyFlags_removed = R"doc()doc";

static const char *__doc_falcor_Component_DirtyFlags_render_state = R"doc()doc";

static const char *__doc_falcor_Component_DirtyFlags_resources = R"doc()doc";

static const char *__doc_falcor_Component_class_descriptor = R"doc()doc";

static const char *__doc_falcor_Component_class_name = R"doc()doc";

static const char *__doc_falcor_Component_entity = R"doc(The entity this component belongs to.)doc";

static const char *__doc_falcor_Component_kind = R"doc()doc";

static const char *__doc_falcor_Component_m_entity = R"doc(The entity this component belongs to.)doc";

static const char *__doc_falcor_Component_mark_dirty = R"doc()doc";

static const char *__doc_falcor_Component_on_entity_transform_changed = R"doc(Called immediately when entity transform changed.)doc";

static const char *__doc_falcor_Component_reflected_class_name = R"doc()doc";

static const char *__doc_falcor_Component_remove = R"doc()doc";

static const char *__doc_falcor_Component_static_class_descriptor = R"doc()doc";

static const char *__doc_falcor_Component_update_render_state =
R"doc(Called during Scene::update() to update the render state of the
component.)doc";

static const char *__doc_falcor_ConstantLight = R"doc()doc";

static const char *__doc_falcor_ConstantLight_ConstantLight = R"doc()doc";

static const char *__doc_falcor_ConstantLight_class_descriptor = R"doc()doc";

static const char *__doc_falcor_ConstantLight_class_name = R"doc()doc";

static const char *__doc_falcor_ConstantLight_m_radiance = R"doc()doc";

static const char *__doc_falcor_ConstantLight_radiance = R"doc()doc";

static const char *__doc_falcor_ConstantLight_reflect = R"doc(Reflect this class.)doc";

static const char *__doc_falcor_ConstantLight_reflected_class_name = R"doc()doc";

static const char *__doc_falcor_ConstantLight_set_radiance = R"doc()doc";

static const char *__doc_falcor_ConstantLight_static_class_descriptor = R"doc()doc";

static const char *__doc_falcor_ConstantLight_write_to_cursor = R"doc()doc";

static const char *__doc_falcor_ConstantLight_write_to_cursor_2 = R"doc()doc";

static const char *__doc_falcor_ConstantLight_write_to_cursor_impl = R"doc()doc";

static const char *__doc_falcor_CurveIndexingMode = R"doc(Indexing mode for curve line segments.)doc";

static const char *__doc_falcor_CurveIndexingMode_info = R"doc()doc";

static const char *__doc_falcor_CurveIndexingMode_list =
R"doc(Each segment is defined by a pair of indices: indices[2k],
indices[2k+1].)doc";

static const char *__doc_falcor_CurveIndexingMode_successive =
R"doc(Each segment is defined by a single index: positions[indices[k]] and
positions[indices[k]+1].)doc";

static const char *__doc_falcor_DiscreteDistribution1D =
R"doc(Discrete 1D probability distribution for importance sampling.

Given a set of non-negative weights, this class builds a discrete
probability distribution that can be sampled on the device. It
computes a normalized PDF and a CDF for efficient inverse transform
sampling using binary search.

The distribution stores both PDF and CDF in device buffers for shader
access. Sampling returns an index in [0, size-1] with probability
proportional to the original weight at that index.

Definitions: sum = sum_{i=0}^{size-1} weight[i] pdf[i] = weight[i] /
sum cdf[i] = sum(pdf[0..i]), so cdf[size-1] == 1.0 (if sum > 0).)doc";

static const char *__doc_falcor_DiscreteDistribution1D_DiscreteDistribution1D =
R"doc(Construct a discrete distribution from a set of weights.

Parameter ``device``:
    Device for buffer allocation.

Parameter ``func``:
    Non-negative weights defining the distribution.

Parameter ``label``:
    Debug label device resources.)doc";

static const char *__doc_falcor_DiscreteDistribution1D_DiscreteDistribution1D_2 = R"doc()doc";

static const char *__doc_falcor_DiscreteDistribution1D_DiscreteDistribution1D_3 = R"doc()doc";

static const char *__doc_falcor_DiscreteDistribution1D_cdf = R"doc(Cumulative distribution function.)doc";

static const char *__doc_falcor_DiscreteDistribution1D_cdf_buffer = R"doc(Device buffer containing the cumulative distribution function.)doc";

static const char *__doc_falcor_DiscreteDistribution1D_m_cdf = R"doc()doc";

static const char *__doc_falcor_DiscreteDistribution1D_m_cdf_buffer = R"doc()doc";

static const char *__doc_falcor_DiscreteDistribution1D_m_pdf = R"doc()doc";

static const char *__doc_falcor_DiscreteDistribution1D_m_pdf_buffer = R"doc()doc";

static const char *__doc_falcor_DiscreteDistribution1D_m_size = R"doc()doc";

static const char *__doc_falcor_DiscreteDistribution1D_operator_assign = R"doc()doc";

static const char *__doc_falcor_DiscreteDistribution1D_operator_assign_2 = R"doc()doc";

static const char *__doc_falcor_DiscreteDistribution1D_pdf = R"doc(Normalize probability density function.)doc";

static const char *__doc_falcor_DiscreteDistribution1D_pdf_buffer = R"doc(Device buffer containing the probability density function.)doc";

static const char *__doc_falcor_DiscreteDistribution1D_size = R"doc(Number of elements in the distribution.)doc";

static const char *__doc_falcor_DiscreteDistribution1D_write_to_cursor = R"doc()doc";

static const char *__doc_falcor_DiscreteDistribution1D_write_to_cursor_2 =
R"doc(Write the handle data to a shader cursor for GPU binding.

Template parameter ``TCursor``:
    Shader cursor type.

Parameter ``cursor``:
    Shader cursor to write to.)doc";

static const char *__doc_falcor_DistantLight = R"doc()doc";

static const char *__doc_falcor_DistantLight_DistantLight = R"doc()doc";

static const char *__doc_falcor_DistantLight_class_descriptor = R"doc()doc";

static const char *__doc_falcor_DistantLight_class_name = R"doc()doc";

static const char *__doc_falcor_DistantLight_cutoff_angle = R"doc()doc";

static const char *__doc_falcor_DistantLight_m_cutoff_angle = R"doc()doc";

static const char *__doc_falcor_DistantLight_m_radiance = R"doc()doc";

static const char *__doc_falcor_DistantLight_radiance = R"doc()doc";

static const char *__doc_falcor_DistantLight_reflect = R"doc(Reflect this class.)doc";

static const char *__doc_falcor_DistantLight_reflected_class_name = R"doc()doc";

static const char *__doc_falcor_DistantLight_set_cutoff_angle = R"doc()doc";

static const char *__doc_falcor_DistantLight_set_radiance = R"doc()doc";

static const char *__doc_falcor_DistantLight_static_class_descriptor = R"doc()doc";

static const char *__doc_falcor_DistantLight_write_to_cursor = R"doc()doc";

static const char *__doc_falcor_DistantLight_write_to_cursor_2 = R"doc()doc";

static const char *__doc_falcor_DistantLight_write_to_cursor_impl = R"doc()doc";

static const char *__doc_falcor_EmissiveGeometrySystem = R"doc(Scene system responsible for managing emissive geometry.)doc";

static const char *__doc_falcor_EmissiveGeometrySystem_EmissiveGeometrySystem =
R"doc(Constructor.

Parameter ``scene``:
    The scene this system belongs to.)doc";

static const char *__doc_falcor_EmissiveGeometrySystem_bind_to_scene = R"doc()doc";

static const char *__doc_falcor_EmissiveGeometrySystem_class_name = R"doc()doc";

static const char *__doc_falcor_EmissiveGeometrySystem_create_kernels = R"doc()doc";

static const char *__doc_falcor_EmissiveGeometrySystem_m_accumulate_triangle_emission_kernel = R"doc()doc";

static const char *__doc_falcor_EmissiveGeometrySystem_m_active_triangle_alias_table =
R"doc(TODO(scene): We might wanna move this to a light sampler object
instead. Alias table over all active emissive triangles (for
sampling).)doc";

static const char *__doc_falcor_EmissiveGeometrySystem_m_active_triangle_count = R"doc()doc";

static const char *__doc_falcor_EmissiveGeometrySystem_m_active_triangle_flux = R"doc(List of active emissive triangle fluxes.)doc";

static const char *__doc_falcor_EmissiveGeometrySystem_m_active_triangle_flux_buffer = R"doc()doc";

static const char *__doc_falcor_EmissiveGeometrySystem_m_active_triangle_id_to_triangle_id_buffer = R"doc()doc";

static const char *__doc_falcor_EmissiveGeometrySystem_m_active_triangle_ids = R"doc(List of active emissive triangles that have non-zero emission.)doc";

static const char *__doc_falcor_EmissiveGeometrySystem_m_components = R"doc()doc";

static const char *__doc_falcor_EmissiveGeometrySystem_m_compute_triangle_max_emission_factor_kernel = R"doc()doc";

static const char *__doc_falcor_EmissiveGeometrySystem_m_compute_triangle_max_emission_kernel = R"doc()doc";

static const char *__doc_falcor_EmissiveGeometrySystem_m_device = R"doc()doc";

static const char *__doc_falcor_EmissiveGeometrySystem_m_entities = R"doc()doc";

static const char *__doc_falcor_EmissiveGeometrySystem_m_finalize_triangle_emission_kernel = R"doc()doc";

static const char *__doc_falcor_EmissiveGeometrySystem_m_gather_active_triangles_kernel = R"doc()doc";

static const char *__doc_falcor_EmissiveGeometrySystem_m_gather_triangle_count_kernel = R"doc()doc";

static const char *__doc_falcor_EmissiveGeometrySystem_m_geometries = R"doc()doc";

static const char *__doc_falcor_EmissiveGeometrySystem_m_geometry_instance_to_triangle_id_buffer = R"doc()doc";

static const char *__doc_falcor_EmissiveGeometrySystem_m_materials = R"doc()doc";

static const char *__doc_falcor_EmissiveGeometrySystem_m_requirements_generation = R"doc()doc";

static const char *__doc_falcor_EmissiveGeometrySystem_m_setup_active_triangle_mapping_kernel = R"doc()doc";

static const char *__doc_falcor_EmissiveGeometrySystem_m_setup_geometry_instance_to_triangle_id_kernel = R"doc()doc";

static const char *__doc_falcor_EmissiveGeometrySystem_m_setup_triangle_kernel = R"doc()doc";

static const char *__doc_falcor_EmissiveGeometrySystem_m_setup_triangle_sampling_kernel = R"doc()doc";

static const char *__doc_falcor_EmissiveGeometrySystem_m_triangle_count = R"doc()doc";

static const char *__doc_falcor_EmissiveGeometrySystem_m_triangle_emission_buffer = R"doc()doc";

static const char *__doc_falcor_EmissiveGeometrySystem_m_triangle_flux = R"doc(List of (potentially) emissive triangle fluxes.)doc";

static const char *__doc_falcor_EmissiveGeometrySystem_m_triangle_flux_buffer = R"doc()doc";

static const char *__doc_falcor_EmissiveGeometrySystem_m_triangle_id_to_active_triangle_id_buffer = R"doc()doc";

static const char *__doc_falcor_EmissiveGeometrySystem_m_triangles = R"doc(List of (potentially) emissive triangles.)doc";

static const char *__doc_falcor_EmissiveGeometrySystem_m_triangles_buffer = R"doc()doc";

static const char *__doc_falcor_EmissiveGeometrySystem_m_update_triangle_kernel = R"doc()doc";

static const char *__doc_falcor_EmissiveGeometrySystem_update = R"doc()doc";

static const char *__doc_falcor_Entity =
R"doc(Scene entity. An entity represents an object in the scene graph with a
transform and a set of components.)doc";

static const char *__doc_falcor_Entity_DirtyFlags = R"doc(Dirty flags for Entity.)doc";

static const char *__doc_falcor_Entity_DirtyFlags_added = R"doc()doc";

static const char *__doc_falcor_Entity_DirtyFlags_none = R"doc()doc";

static const char *__doc_falcor_Entity_DirtyFlags_removed = R"doc()doc";

static const char *__doc_falcor_Entity_DirtyFlags_resources = R"doc()doc";

static const char *__doc_falcor_Entity_DirtyFlags_transform = R"doc()doc";

static const char *__doc_falcor_Entity_children = R"doc(List of child entities.)doc";

static const char *__doc_falcor_Entity_class_descriptor = R"doc()doc";

static const char *__doc_falcor_Entity_class_name = R"doc()doc";

static const char *__doc_falcor_Entity_clear_invalid_references = R"doc()doc";

static const char *__doc_falcor_Entity_components = R"doc(List of components belonging to this entity.)doc";

static const char *__doc_falcor_Entity_create_component =
R"doc(Create a component instance.

Template parameter ``T``:
    Type of component to create.

Parameter ``props``:
    Properties to initialize the component (optional).

Returns:
    The created component.)doc";

static const char *__doc_falcor_Entity_create_component_2 =
R"doc(Create a component instance.

Parameter ``type``:
    Type name of the component to create.

Parameter ``props``:
    Properties to initialize the component (optional).

Returns:
    The created component.)doc";

static const char *__doc_falcor_Entity_create_component_3 =
R"doc(Create a component instance.

Parameter ``type``:
    Type info of the component to create.

Parameter ``props``:
    Properties to initialize the component (optional).

Returns:
    The created component.)doc";

static const char *__doc_falcor_Entity_kind = R"doc()doc";

static const char *__doc_falcor_Entity_m_children = R"doc()doc";

static const char *__doc_falcor_Entity_m_components = R"doc()doc";

static const char *__doc_falcor_Entity_m_parent = R"doc()doc";

static const char *__doc_falcor_Entity_m_render_transform = R"doc()doc";

static const char *__doc_falcor_Entity_m_transform = R"doc()doc";

static const char *__doc_falcor_Entity_m_world_from_object_matrix = R"doc()doc";

static const char *__doc_falcor_Entity_m_world_from_object_matrix_dirty = R"doc()doc";

static const char *__doc_falcor_Entity_mark_dirty = R"doc()doc";

static const char *__doc_falcor_Entity_on_add_to_scene = R"doc()doc";

static const char *__doc_falcor_Entity_on_remove_from_scene = R"doc()doc";

static const char *__doc_falcor_Entity_parent = R"doc(Parent entity or nullptr if this is a root entity.)doc";

static const char *__doc_falcor_Entity_reflect = R"doc(Reflect this class.)doc";

static const char *__doc_falcor_Entity_reflected_class_name = R"doc()doc";

static const char *__doc_falcor_Entity_remove = R"doc()doc";

static const char *__doc_falcor_Entity_render_transform = R"doc()doc";

static const char *__doc_falcor_Entity_set_parent = R"doc()doc";

static const char *__doc_falcor_Entity_set_transform = R"doc()doc";

static const char *__doc_falcor_Entity_set_world_transform =
R"doc(Set the world-space transform of this entity. If the entity has a
parent, the local transform is computed by multiplying with the
inverse of the parent's world matrix.)doc";

static const char *__doc_falcor_Entity_static_class_descriptor = R"doc()doc";

static const char *__doc_falcor_Entity_transform = R"doc(Transform of the entity (local space, relative to parent).)doc";

static const char *__doc_falcor_Entity_transform_changed =
R"doc(Called immediately when the transform of this entity changed, or the
entity was reparented. This marks the transform as dirty, and notifies
components and child entities about the change.)doc";

static const char *__doc_falcor_Entity_update_transform = R"doc(Called during Scene::update() to update the transform.)doc";

static const char *__doc_falcor_Entity_world_aabb =
R"doc(Compute the world-space axis-aligned bounding box of this entity.
Returns an invalid AABB if the entity has no geometry.)doc";

static const char *__doc_falcor_Entity_world_from_object_matrix = R"doc(World-from-object transformation matrix.)doc";

static const char *__doc_falcor_EnvMapLight = R"doc()doc";

static const char *__doc_falcor_EnvMapLight_EnvMapLight = R"doc()doc";

static const char *__doc_falcor_EnvMapLight_class_descriptor = R"doc()doc";

static const char *__doc_falcor_EnvMapLight_class_name = R"doc()doc";

static const char *__doc_falcor_EnvMapLight_env_map_path = R"doc()doc";

static const char *__doc_falcor_EnvMapLight_m_build_importance_map_kernel = R"doc()doc";

static const char *__doc_falcor_EnvMapLight_m_env_map_loaded = R"doc()doc";

static const char *__doc_falcor_EnvMapLight_m_env_map_path = R"doc()doc";

static const char *__doc_falcor_EnvMapLight_m_env_map_texture_handle = R"doc()doc";

static const char *__doc_falcor_EnvMapLight_m_importance_map_dirty = R"doc()doc";

static const char *__doc_falcor_EnvMapLight_m_importance_map_texture = R"doc()doc";

static const char *__doc_falcor_EnvMapLight_on_load_resources = R"doc()doc";

static const char *__doc_falcor_EnvMapLight_reflect = R"doc(Reflect this class.)doc";

static const char *__doc_falcor_EnvMapLight_reflected_class_name = R"doc()doc";

static const char *__doc_falcor_EnvMapLight_set_env_map_path = R"doc()doc";

static const char *__doc_falcor_EnvMapLight_static_class_descriptor = R"doc()doc";

static const char *__doc_falcor_EnvMapLight_update = R"doc()doc";

static const char *__doc_falcor_EnvMapLight_write_to_cursor = R"doc()doc";

static const char *__doc_falcor_EnvMapLight_write_to_cursor_2 = R"doc()doc";

static const char *__doc_falcor_EnvMapLight_write_to_cursor_impl = R"doc()doc";

static const char *__doc_falcor_FNVHash =
R"doc(Accumulates Fowler-Noll-Vo hash for inserted data. To hash multiple
items, create one Hash and insert all the items into it if at all
possible. This is superior to hashing the items individually and
combining the hashes.

Template parameter ``TUInt``:
    Type of the storage for the hash, either 32 or 64 unsigned
    integer.)doc";

static const char *__doc_falcor_FNVHash_get = R"doc()doc";

static const char *__doc_falcor_FNVHash_insert =
R"doc(Inserts all data between [begin,end) into the hash.

Parameter ``begin``:
    $Parameter ``end``:)doc";

static const char *__doc_falcor_FNVHash_insert_2 =
R"doc(Inserts all data starting at data and going for size bytes into the
hash

Parameter ``data``:
    $Parameter ``size``:)doc";

static const char *__doc_falcor_FNVHash_insert_3 = R"doc()doc";

static const char *__doc_falcor_FNVHash_m_state = R"doc()doc";

static const char *__doc_falcor_FNVHash_operator_le = R"doc()doc";

static const char *__doc_falcor_Geometry = R"doc(Base class for scene geometry.)doc";

static const char *__doc_falcor_Geometry_2 = R"doc(Base class for scene geometry.)doc";

static const char *__doc_falcor_GeometryGroup =
R"doc(Geometry group. A geometry group is a collection of geometries treated
as a single geometry.)doc";

static const char *__doc_falcor_GeometryGroup_add_geometry = R"doc()doc";

static const char *__doc_falcor_GeometryGroup_class_descriptor = R"doc()doc";

static const char *__doc_falcor_GeometryGroup_class_name = R"doc()doc";

static const char *__doc_falcor_GeometryGroup_clear_invalid_references = R"doc()doc";

static const char *__doc_falcor_GeometryGroup_geometries = R"doc(List of geometries.)doc";

static const char *__doc_falcor_GeometryGroup_m_geometries = R"doc()doc";

static const char *__doc_falcor_GeometryGroup_reflect = R"doc(Reflect this class.)doc";

static const char *__doc_falcor_GeometryGroup_reflected_class_name = R"doc()doc";

static const char *__doc_falcor_GeometryGroup_remove_geometry = R"doc()doc";

static const char *__doc_falcor_GeometryGroup_set_geometries = R"doc()doc";

static const char *__doc_falcor_GeometryGroup_static_class_descriptor = R"doc()doc";

static const char *__doc_falcor_GeometryInstance =
R"doc(Geometry instance component. A geometry instance references instances
the reference geometry on the entity. It allows to assign per-instance
materials.)doc";

static const char *__doc_falcor_GeometryInstance_class_descriptor = R"doc()doc";

static const char *__doc_falcor_GeometryInstance_class_name = R"doc()doc";

static const char *__doc_falcor_GeometryInstance_clear_invalid_references = R"doc()doc";

static const char *__doc_falcor_GeometryInstance_geometry = R"doc(Geometry of this instance.)doc";

static const char *__doc_falcor_GeometryInstance_geometry_instance_count =
R"doc(Number of geometries belonging to this geometry instance. All
subsequent geometries use successive GeometryInstanceIDs.)doc";

static const char *__doc_falcor_GeometryInstance_geometry_instance_id = R"doc(GeometryInstanceID assigned to this geometry instance.)doc";

static const char *__doc_falcor_GeometryInstance_m_geometry = R"doc()doc";

static const char *__doc_falcor_GeometryInstance_m_materials = R"doc()doc";

static const char *__doc_falcor_GeometryInstance_m_render_geometry_instance = R"doc()doc";

static const char *__doc_falcor_GeometryInstance_materials = R"doc(Materials assigned to this geometry instance.)doc";

static const char *__doc_falcor_GeometryInstance_on_add_to_scene = R"doc()doc";

static const char *__doc_falcor_GeometryInstance_on_remove_from_scene = R"doc()doc";

static const char *__doc_falcor_GeometryInstance_reflect = R"doc(Reflect this class.)doc";

static const char *__doc_falcor_GeometryInstance_reflected_class_name = R"doc()doc";

static const char *__doc_falcor_GeometryInstance_set_geometry = R"doc()doc";

static const char *__doc_falcor_GeometryInstance_set_materials = R"doc()doc";

static const char *__doc_falcor_GeometryInstance_static_class_descriptor = R"doc()doc";

static const char *__doc_falcor_GeometryInstance_update_render_state = R"doc()doc";

static const char *__doc_falcor_Geometry_DirtyFlags = R"doc(Dirty flags for Geometry.)doc";

static const char *__doc_falcor_Geometry_DirtyFlags_added = R"doc()doc";

static const char *__doc_falcor_Geometry_DirtyFlags_none = R"doc()doc";

static const char *__doc_falcor_Geometry_DirtyFlags_removed = R"doc()doc";

static const char *__doc_falcor_Geometry_DirtyFlags_render_state = R"doc()doc";

static const char *__doc_falcor_Geometry_DirtyFlags_resources = R"doc()doc";

static const char *__doc_falcor_Geometry_Geometry = R"doc(Constructor.)doc";

static const char *__doc_falcor_Geometry_class_descriptor = R"doc()doc";

static const char *__doc_falcor_Geometry_class_name = R"doc()doc";

static const char *__doc_falcor_Geometry_kind = R"doc()doc";

static const char *__doc_falcor_Geometry_local_aabb = R"doc(Local-space axis-aligned bounding box of the geometry.)doc";

static const char *__doc_falcor_Geometry_m_geometry_instances = R"doc(List of geometry instances referencing this geometry.)doc";

static const char *__doc_falcor_Geometry_m_local_aabb = R"doc(Local-space axis-aligned bounding box.)doc";

static const char *__doc_falcor_Geometry_m_render_geometry_group = R"doc(Render geometry group representing this geometry.)doc";

static const char *__doc_falcor_Geometry_mark_dirty = R"doc()doc";

static const char *__doc_falcor_Geometry_reflected_class_name = R"doc()doc";

static const char *__doc_falcor_Geometry_remove = R"doc()doc";

static const char *__doc_falcor_Geometry_render_geometry_group = R"doc()doc";

static const char *__doc_falcor_Geometry_static_class_descriptor = R"doc()doc";

static const char *__doc_falcor_Geometry_update_render_state =
R"doc(Called during Scene::update() to update the render state of the
component.)doc";

static const char *__doc_falcor_GltfImporter = R"doc(GLTF importer.)doc";

static const char *__doc_falcor_GltfImporter_GltfImporter = R"doc()doc";

static const char *__doc_falcor_GltfImporter_build_node_hierarchy = R"doc()doc";

static const char *__doc_falcor_GltfImporter_class_name = R"doc()doc";

static const char *__doc_falcor_GltfImporter_extract_animations = R"doc()doc";

static const char *__doc_falcor_GltfImporter_extract_cameras = R"doc()doc";

static const char *__doc_falcor_GltfImporter_extract_materials = R"doc()doc";

static const char *__doc_falcor_GltfImporter_extract_meshes = R"doc()doc";

static const char *__doc_falcor_GltfImporter_extract_nodes = R"doc()doc";

static const char *__doc_falcor_GltfImporter_extract_textures = R"doc()doc";

static const char *__doc_falcor_GltfImporter_load_gltf_file = R"doc()doc";

static const char *__doc_falcor_GltfImporter_load_scene =
R"doc(Load a GLTF scene from the specified file path.

Parameter ``path``:
    Path to the GLTF file.

Returns:
    The imported scene.)doc";

static const char *__doc_falcor_GltfImporter_m_model = R"doc()doc";

static const char *__doc_falcor_GltfImporter_m_path = R"doc()doc";

static const char *__doc_falcor_GltfImporter_m_scene = R"doc()doc";

static const char *__doc_falcor_HitGroupPolicy = R"doc(Policy to determine how hit groups are generated for the scene.)doc";

static const char *__doc_falcor_HitGroupPolicy_Mode = R"doc()doc";

static const char *__doc_falcor_HitGroupPolicy_Mode_info = R"doc()doc";

static const char *__doc_falcor_HitGroupPolicy_Mode_per_geometry_instance_id = R"doc()doc";

static const char *__doc_falcor_HitGroupPolicy_Mode_per_geometry_type = R"doc()doc";

static const char *__doc_falcor_HitGroupPolicy_geometry_type_count = R"doc(Number of geometry types (used for per_geometry_type mode).)doc";

static const char *__doc_falcor_HitGroupPolicy_hit_group_index = R"doc()doc";

static const char *__doc_falcor_HitGroupPolicy_instance_contribution_to_hit_group_index = R"doc()doc";

static const char *__doc_falcor_HitGroupPolicy_mode = R"doc(Policy mode.)doc";

static const char *__doc_falcor_HitGroupPolicy_multiplier_for_geometry_contribution_to_hit_group_index = R"doc()doc";

static const char *__doc_falcor_HitGroupPolicy_ray_type_count = R"doc(Number of ray types.)doc";

static const char *__doc_falcor_IDictionary = R"doc()doc";

static const char *__doc_falcor_IDictionary_Accessor = R"doc()doc";

static const char *__doc_falcor_IDictionary_Accessor_Accessor = R"doc()doc";

static const char *__doc_falcor_IDictionary_Accessor_m_name = R"doc()doc";

static const char *__doc_falcor_IDictionary_Accessor_m_wrapper = R"doc()doc";

static const char *__doc_falcor_IDictionary_Accessor_operator_assign = R"doc()doc";

static const char *__doc_falcor_IDictionary_operator_array = R"doc()doc";

static const char *__doc_falcor_IDictionary_pop = R"doc()doc";

static const char *__doc_falcor_IDictionary_push = R"doc()doc";

static const char *__doc_falcor_IDictionary_set = R"doc()doc";

static const char *__doc_falcor_IDictionary_set_2 = R"doc()doc";

static const char *__doc_falcor_IDictionary_set_3 = R"doc()doc";

static const char *__doc_falcor_IDictionary_set_4 = R"doc()doc";

static const char *__doc_falcor_IDictionary_set_5 = R"doc()doc";

static const char *__doc_falcor_IDictionary_set_6 = R"doc()doc";

static const char *__doc_falcor_IDictionary_set_7 = R"doc()doc";

static const char *__doc_falcor_IDictionary_set_8 = R"doc()doc";

static const char *__doc_falcor_IDictionary_set_9 = R"doc()doc";

static const char *__doc_falcor_IDictionary_set_10 = R"doc()doc";

static const char *__doc_falcor_IDictionary_set_11 = R"doc()doc";

static const char *__doc_falcor_IDictionary_set_12 = R"doc()doc";

static const char *__doc_falcor_IDictionary_set_13 = R"doc()doc";

static const char *__doc_falcor_IDictionary_set_14 = R"doc()doc";

static const char *__doc_falcor_IDictionary_set_15 = R"doc()doc";

static const char *__doc_falcor_IDictionary_set_16 = R"doc()doc";

static const char *__doc_falcor_IDictionary_set_17 = R"doc()doc";

static const char *__doc_falcor_IDictionary_set_18 = R"doc()doc";

static const char *__doc_falcor_IDictionary_set_19 = R"doc()doc";

static const char *__doc_falcor_IDictionary_set_20 = R"doc()doc";

static const char *__doc_falcor_IDictionary_set_21 = R"doc()doc";

static const char *__doc_falcor_IDictionary_set_22 = R"doc()doc";

static const char *__doc_falcor_IDictionary_set_23 = R"doc()doc";

static const char *__doc_falcor_IDictionary_set_24 = R"doc()doc";

static const char *__doc_falcor_IDictionary_set_25 = R"doc()doc";

static const char *__doc_falcor_IDictionary_set_26 = R"doc()doc";

static const char *__doc_falcor_IDictionary_set_27 = R"doc()doc";

static const char *__doc_falcor_IDictionary_set_28 = R"doc()doc";

static const char *__doc_falcor_IDictionary_set_29 = R"doc()doc";

static const char *__doc_falcor_IDictionary_set_30 = R"doc()doc";

static const char *__doc_falcor_IDictionary_set_31 = R"doc()doc";

static const char *__doc_falcor_IDictionary_set_32 = R"doc()doc";

static const char *__doc_falcor_IDictionary_set_33 = R"doc()doc";

static const char *__doc_falcor_IDictionary_set_34 = R"doc()doc";

static const char *__doc_falcor_IDictionary_set_35 = R"doc()doc";

static const char *__doc_falcor_IDictionary_set_36 = R"doc()doc";

static const char *__doc_falcor_IDictionary_set_37 = R"doc()doc";

static const char *__doc_falcor_IDictionary_set_38 = R"doc()doc";

static const char *__doc_falcor_IDictionary_set_39 = R"doc()doc";

static const char *__doc_falcor_IDictionary_set_40 = R"doc()doc";

static const char *__doc_falcor_IDictionary_set_41 = R"doc()doc";

static const char *__doc_falcor_IDictionary_set_42 = R"doc()doc";

static const char *__doc_falcor_IDictionary_set_43 = R"doc()doc";

static const char *__doc_falcor_IDictionary_set_44 = R"doc()doc";

static const char *__doc_falcor_IDictionary_set_45 = R"doc()doc";

static const char *__doc_falcor_IDictionary_set_46 = R"doc()doc";

static const char *__doc_falcor_IDictionary_set_47 = R"doc()doc";

static const char *__doc_falcor_IDictionary_set_48 = R"doc()doc";

static const char *__doc_falcor_IDictionary_set_49 = R"doc()doc";

static const char *__doc_falcor_IDictionary_set_50 = R"doc()doc";

static const char *__doc_falcor_IDictionary_set_51 = R"doc()doc";

static const char *__doc_falcor_IDictionary_set_52 = R"doc()doc";

static const char *__doc_falcor_IDictionary_set_53 = R"doc()doc";

static const char *__doc_falcor_IDictionary_set_54 = R"doc()doc";

static const char *__doc_falcor_IDictionary_set_55 = R"doc()doc";

static const char *__doc_falcor_IDictionary_set_56 = R"doc()doc";

static const char *__doc_falcor_IDictionary_set_57 = R"doc()doc";

static const char *__doc_falcor_IDictionary_set_58 = R"doc()doc";

static const char *__doc_falcor_IDictionary_set_59 = R"doc()doc";

static const char *__doc_falcor_IDictionary_set_60 = R"doc()doc";

static const char *__doc_falcor_ImportOptions = R"doc()doc";

static const char *__doc_falcor_ImportOptions_recompute_normals = R"doc()doc";

static const char *__doc_falcor_ImporterAnimation = R"doc(A collection of animation channels.)doc";

static const char *__doc_falcor_ImporterAnimationChannel = R"doc(A single animation channel storing typed keyframe data.)doc";

static const char *__doc_falcor_ImporterAnimationChannel_components_per_value = R"doc(The number of components per keyframe value.)doc";

static const char *__doc_falcor_ImporterAnimationChannel_in_tangents =
R"doc(In-tangent per keyframe (same layout as values), only populated for
cubic_spline.)doc";

static const char *__doc_falcor_ImporterAnimationChannel_interpolation_mode = R"doc(Interpolation mode.)doc";

static const char *__doc_falcor_ImporterAnimationChannel_keyframe_count = R"doc(The number of keyframes.)doc";

static const char *__doc_falcor_ImporterAnimationChannel_out_tangents =
R"doc(Out-tangent per keyframe (same layout as values), only populated for
cubic_spline.)doc";

static const char *__doc_falcor_ImporterAnimationChannel_times = R"doc(Keyframe timestamps in seconds.)doc";

static const char *__doc_falcor_ImporterAnimationChannel_value_type = R"doc(Value type of keyframe data.)doc";

static const char *__doc_falcor_ImporterAnimationChannel_values =
R"doc(Keyframe values (flat array: 1/3/4 floats per key depending on
value_type).)doc";

static const char *__doc_falcor_ImporterAnimation_channels = R"doc(Flat storage of all animation channels.)doc";

static const char *__doc_falcor_ImporterAnimation_duration = R"doc(The duration (max keyframe time across all channels).)doc";

static const char *__doc_falcor_ImporterAnimation_name = R"doc(Animation name.)doc";

static const char *__doc_falcor_ImporterCamera = R"doc()doc";

static const char *__doc_falcor_ImporterCamera_FOVDirection = R"doc()doc";

static const char *__doc_falcor_ImporterCamera_FOVDirection_horizontal = R"doc()doc";

static const char *__doc_falcor_ImporterCamera_FOVDirection_info = R"doc()doc";

static const char *__doc_falcor_ImporterCamera_FOVDirection_vertical = R"doc()doc";

static const char *__doc_falcor_ImporterCamera_Projection = R"doc()doc";

static const char *__doc_falcor_ImporterCamera_Projection_info = R"doc()doc";

static const char *__doc_falcor_ImporterCamera_Projection_orthographic = R"doc()doc";

static const char *__doc_falcor_ImporterCamera_Projection_perspective = R"doc()doc";

static const char *__doc_falcor_ImporterCamera_aperture = R"doc(Aperture size.)doc";

static const char *__doc_falcor_ImporterCamera_depth_range = R"doc(Depth range (near, far).)doc";

static const char *__doc_falcor_ImporterCamera_focal_length = R"doc(Focal length (millimeters).)doc";

static const char *__doc_falcor_ImporterCamera_focus_distance = R"doc(Focus distance.)doc";

static const char *__doc_falcor_ImporterCamera_fov_direction = R"doc(Field of view direction.)doc";

static const char *__doc_falcor_ImporterCamera_fstop = R"doc(F-stop value.)doc";

static const char *__doc_falcor_ImporterCamera_name = R"doc(Camera name.)doc";

static const char *__doc_falcor_ImporterCamera_projection = R"doc(Projection type.)doc";

static const char *__doc_falcor_ImporterCurve = R"doc(Imported curve data.)doc";

static const char *__doc_falcor_ImporterCurve_calculate_local_aabb = R"doc(Calculate local AABB from positions and radii.)doc";

static const char *__doc_falcor_ImporterCurve_indexing_mode = R"doc(Indexing mode for the index buffer.)doc";

static const char *__doc_falcor_ImporterCurve_indices = R"doc(Pairs of indices defining line segments.)doc";

static const char *__doc_falcor_ImporterCurve_local_aabb = R"doc(Local bounding box.)doc";

static const char *__doc_falcor_ImporterCurve_material_name = R"doc(Material name.)doc";

static const char *__doc_falcor_ImporterCurve_name = R"doc(Curve name.)doc";

static const char *__doc_falcor_ImporterCurve_positions = R"doc(Curve vertex positions.)doc";

static const char *__doc_falcor_ImporterCurve_radii = R"doc(Radius at each vertex.)doc";

static const char *__doc_falcor_ImporterLight = R"doc()doc";

static const char *__doc_falcor_ImporterLight_Type = R"doc()doc";

static const char *__doc_falcor_ImporterLight_Type_constant = R"doc(Constant environment light.)doc";

static const char *__doc_falcor_ImporterLight_Type_disk = R"doc(Disk at the origin, in XY plane, emitting along the -Z axis.)doc";

static const char *__doc_falcor_ImporterLight_Type_distant = R"doc(Light emitted from a distance source along the -Z axis.)doc";

static const char *__doc_falcor_ImporterLight_Type_dome =
R"doc(Environment map. Top pole is aligned with +Y axis, matching OpenEXR
latlong specification. LatLong 0,0 -> points in positive Z direction.
LatLong 0,Pi/2 -> points in positive X direction.)doc";

static const char *__doc_falcor_ImporterLight_Type_info = R"doc()doc";

static const char *__doc_falcor_ImporterLight_Type_point = R"doc(Sphere with zero radius.)doc";

static const char *__doc_falcor_ImporterLight_Type_rectangular =
R"doc(Light emitted from a rectangle of size width x height along the -Z
axis. Rectangle at origin in the XY plane.)doc";

static const char *__doc_falcor_ImporterLight_Type_sphere = R"doc(Light emitted out from a sphere at the origin.)doc";

static const char *__doc_falcor_ImporterLight_degree_angular_diameter = R"doc(Angular diameter in degrees (for distant lights).)doc";

static const char *__doc_falcor_ImporterLight_env_map_path = R"doc(Environment map path (for dome lights).)doc";

static const char *__doc_falcor_ImporterLight_height = R"doc(Height (for rectangular lights).)doc";

static const char *__doc_falcor_ImporterLight_intensity = R"doc(Light intensity.)doc";

static const char *__doc_falcor_ImporterLight_name = R"doc(Light name.)doc";

static const char *__doc_falcor_ImporterLight_radius = R"doc(Radius (for sphere/disk lights).)doc";

static const char *__doc_falcor_ImporterLight_type = R"doc(Light type.)doc";

static const char *__doc_falcor_ImporterLight_width = R"doc(Width (for rectangular lights).)doc";

static const char *__doc_falcor_ImporterMaterial = R"doc()doc";

static const char *__doc_falcor_ImporterMaterial_2 = R"doc()doc";

static const char *__doc_falcor_ImporterMaterial_3 = R"doc()doc";

static const char *__doc_falcor_ImporterMaterial_4 = R"doc()doc";

static const char *__doc_falcor_ImporterMaterial_5 = R"doc()doc";

static const char *__doc_falcor_ImporterMaterial_name = R"doc(Material name.)doc";

static const char *__doc_falcor_ImporterMaterial_output_to_material_network =
R"doc(Maps a named material output (e.g. "surface", "displacement") to the
material network that drives it. Each network is represented as an
ordered list of nodes, where each node's properties (type, parameter
values, input connections) are captured in a Properties object. This
mirrors the structure of a MaterialNetwork, but is stored here without
that dependency.)doc";

static const char *__doc_falcor_ImporterMaterial_params =
R"doc(Material parameters. For Usd, if UsdPreviewSurface is available, it is
flattened here. For MDL, nodes must be used.)doc";

static const char *__doc_falcor_ImporterMesh =
R"doc(Mesh data imported from a file. Represents a set of buffers containing
vertex data, and a set of subgeometries with indices into the vertex
data.)doc";

static const char *__doc_falcor_ImporterMeshAttribute = R"doc(Describes an attribute of a mesh vertex)doc";

static const char *__doc_falcor_ImporterMeshAttribute_buffer = R"doc(Buffer index for the attribute)doc";

static const char *__doc_falcor_ImporterMeshAttribute_index =
R"doc(Index for attributes that can have multiple sets, e.g., TEXCOORD_0,
TEXCOORD_1)doc";

static const char *__doc_falcor_ImporterMeshAttribute_name =
R"doc(Name of the attribute as specified in the file (e.g., "POSITION",
"TEXCOORD_0", "myCustomAttrib"))doc";

static const char *__doc_falcor_ImporterMeshAttribute_num_components = R"doc(Number of components in the attribute (e.g. 3 for a float3))doc";

static const char *__doc_falcor_ImporterMeshAttribute_offset = R"doc(Offset in bytes from start of vertex)doc";

static const char *__doc_falcor_ImporterMeshAttribute_semantic = R"doc(Semantic meaning of the attribute assigned by the importer)doc";

static const char *__doc_falcor_ImporterMeshAttribute_type = R"doc(Type of the attribute data (e.g. float for a float3))doc";

static const char *__doc_falcor_ImporterMeshAttribute_valid = R"doc()doc";

static const char *__doc_falcor_ImporterMeshBuffer =
R"doc(A buffer containing vertex data. Layout is described by
ImporterMeshAttribute.)doc";

static const char *__doc_falcor_ImporterMeshBuffer_data = R"doc(Buffer data)doc";

static const char *__doc_falcor_ImporterMeshBuffer_stride = R"doc()doc";

static const char *__doc_falcor_ImporterMeshDefaultAttribute =
R"doc(Minimal information to describe an attribute to be created using
ensure_attributes, given known semantic and index.)doc";

static const char *__doc_falcor_ImporterMeshDefaultAttribute_components = R"doc()doc";

static const char *__doc_falcor_ImporterMeshDefaultAttribute_index = R"doc()doc";

static const char *__doc_falcor_ImporterMeshDefaultAttribute_semantic = R"doc()doc";

static const char *__doc_falcor_ImporterMesh_Subgeometry = R"doc()doc";

static const char *__doc_falcor_ImporterMesh_Subgeometry_indices = R"doc(Triangle indices.)doc";

static const char *__doc_falcor_ImporterMesh_Subgeometry_material_name = R"doc(Material name.)doc";

static const char *__doc_falcor_ImporterMesh_Subgeometry_name = R"doc(Subgeometry name.)doc";

static const char *__doc_falcor_ImporterMesh_add_attribute =
R"doc(Add a new vertex attribute to the mesh. If the attribute refers to an
existing buffer, the buffer must be empty. If it refers to a new
buffer, it must be created by calling create_buffer() or
create_buffers_from_attributes() before use.)doc";

static const char *__doc_falcor_ImporterMesh_add_normals_from_faces = R"doc(Generate normals for mesh vertices based on face geometry.)doc";

static const char *__doc_falcor_ImporterMesh_add_tangents_from_normals = R"doc(Generate tangents for mesh vertices using normals (orthonormal basis).)doc";

static const char *__doc_falcor_ImporterMesh_add_tangents_from_uvs =
R"doc(Generate tangents for mesh vertices using UV coordinates and
MikkTSpace algorithm.)doc";

static const char *__doc_falcor_ImporterMesh_allocate_vertices =
R"doc(Allocates space in vertex buffers for given number of vertices and
returns the starting vertex index.)doc";

static const char *__doc_falcor_ImporterMesh_attributes = R"doc(Vertex attributes.)doc";

static const char *__doc_falcor_ImporterMesh_buffers = R"doc(Vertex buffers.)doc";

static const char *__doc_falcor_ImporterMesh_buffers_2 = R"doc()doc";

static const char *__doc_falcor_ImporterMesh_calculate_local_aabb = R"doc(Calculate the local AABB from the mesh vertices.)doc";

static const char *__doc_falcor_ImporterMesh_color_attribute = R"doc(Vertex colors (0) + helpers to access stream as rgb float3.)doc";

static const char *__doc_falcor_ImporterMesh_color_stream = R"doc()doc";

static const char *__doc_falcor_ImporterMesh_color_stream_2 = R"doc()doc";

static const char *__doc_falcor_ImporterMesh_create_buffer =
R"doc(Create a new empty vertex buffer. The attribute(s) that refer to this
buffer must be created before use. Pass stride==0 to have the stride
automatically calculated from the attributes that refer to this
buffer.)doc";

static const char *__doc_falcor_ImporterMesh_create_buffers_from_attributes =
R"doc(Create new buffers for all attributes that refer to a buffer that
doesn't exist yet.)doc";

static const char *__doc_falcor_ImporterMesh_deduplicate_vertices = R"doc(Remove duplicate vertices from mesh and update indices accordingly.)doc";

static const char *__doc_falcor_ImporterMesh_ensure_attributes =
R"doc(Ensures a set of known attriute types exist, creating them with basic
floating point config if needed.)doc";

static const char *__doc_falcor_ImporterMesh_ensure_attributes_2 =
R"doc(Ensures a set of known attriute types exist, creating them with basic
floating point config if needed.)doc";

static const char *__doc_falcor_ImporterMesh_find_attribute =
R"doc(Find an attribute by semantic and index (e.g., texcoord 0, texcoord
1).)doc";

static const char *__doc_falcor_ImporterMesh_find_attribute_2 =
R"doc(Find an attribute by name (e.g., "POSITION", "TEXCOORD_0",
"myCustomAttrib").)doc";

static const char *__doc_falcor_ImporterMesh_find_stream =
R"doc(Find a typed 'VertexStream' for a given attribute to give an easy way
to read/write the vertex data.)doc";

static const char *__doc_falcor_ImporterMesh_get_stream =
R"doc(Get a typed 'VertexStream' for a given attribute to give an easy way
to read/write the vertex data.)doc";

static const char *__doc_falcor_ImporterMesh_get_stream_2 =
R"doc(Get a typed 'VertexStream' for a given attribute to give an easy way
to read/write the vertex data.)doc";

static const char *__doc_falcor_ImporterMesh_get_stream_info =
R"doc(Low level function to get a pointer to vertex attribute data in a
stream.)doc";

static const char *__doc_falcor_ImporterMesh_get_stream_info_2 =
R"doc(Low level function to get a pointer to vertex attribute data in a
stream.)doc";

static const char *__doc_falcor_ImporterMesh_handedness_attribute = R"doc(Vertex handedness.)doc";

static const char *__doc_falcor_ImporterMesh_handedness_stream = R"doc()doc";

static const char *__doc_falcor_ImporterMesh_handedness_stream_2 = R"doc()doc";

static const char *__doc_falcor_ImporterMesh_hash_vertex =
R"doc(Get a hash value for a vertex at given index based on its attribute
values, correctly skipping over any padding between elements.)doc";

static const char *__doc_falcor_ImporterMesh_local_aabb = R"doc(Local bounding box of the mesh (calculated from vertices).)doc";

static const char *__doc_falcor_ImporterMesh_m_attributes = R"doc()doc";

static const char *__doc_falcor_ImporterMesh_m_buffers = R"doc()doc";

static const char *__doc_falcor_ImporterMesh_m_colors = R"doc()doc";

static const char *__doc_falcor_ImporterMesh_m_handedness = R"doc()doc";

static const char *__doc_falcor_ImporterMesh_m_normals = R"doc()doc";

static const char *__doc_falcor_ImporterMesh_m_positions = R"doc()doc";

static const char *__doc_falcor_ImporterMesh_m_tangents = R"doc()doc";

static const char *__doc_falcor_ImporterMesh_m_texcoords = R"doc()doc";

static const char *__doc_falcor_ImporterMesh_m_vertex_count = R"doc()doc";

static const char *__doc_falcor_ImporterMesh_name = R"doc(Mesh name.)doc";

static const char *__doc_falcor_ImporterMesh_normal_attribute = R"doc(Vertex normals.)doc";

static const char *__doc_falcor_ImporterMesh_normal_stream = R"doc()doc";

static const char *__doc_falcor_ImporterMesh_normal_stream_2 = R"doc()doc";

static const char *__doc_falcor_ImporterMesh_position_attribute = R"doc(Vertex positions.)doc";

static const char *__doc_falcor_ImporterMesh_position_stream = R"doc()doc";

static const char *__doc_falcor_ImporterMesh_position_stream_2 = R"doc()doc";

static const char *__doc_falcor_ImporterMesh_subgeometries = R"doc(List of subgeometries.)doc";

static const char *__doc_falcor_ImporterMesh_tangent_attribute = R"doc(Vertex tangents.)doc";

static const char *__doc_falcor_ImporterMesh_tangent_stream = R"doc()doc";

static const char *__doc_falcor_ImporterMesh_tangent_stream_2 = R"doc()doc";

static const char *__doc_falcor_ImporterMesh_texcoord_attribute = R"doc(Vertex texcoords (0) + helpers to access stream as float2.)doc";

static const char *__doc_falcor_ImporterMesh_texcoord_stream = R"doc()doc";

static const char *__doc_falcor_ImporterMesh_texcoord_stream_2 = R"doc()doc";

static const char *__doc_falcor_ImporterMesh_vertex_count = R"doc(Current vertex count.)doc";

static const char *__doc_falcor_ImporterNode = R"doc()doc";

static const char *__doc_falcor_ImporterNode_animation_rotation =
R"doc(Index into ImporterScene::animation.channels for rotation (expects
quatf), -1 if not animated.)doc";

static const char *__doc_falcor_ImporterNode_animation_scale =
R"doc(Index into ImporterScene::animation.channels for scale (expects
float3), -1 if not animated.)doc";

static const char *__doc_falcor_ImporterNode_animation_translation =
R"doc(Index into ImporterScene::animation.channels for translation (expects
float3), -1 if not animated.)doc";

static const char *__doc_falcor_ImporterNode_camera_index = R"doc()doc";

static const char *__doc_falcor_ImporterNode_children = R"doc(Indices into ImporterScene::nodes (child nodes).)doc";

static const char *__doc_falcor_ImporterNode_curve_index = R"doc(Index into ImporterScene::curves, -1 if no curve.)doc";

static const char *__doc_falcor_ImporterNode_light_index = R"doc()doc";

static const char *__doc_falcor_ImporterNode_mesh_index = R"doc(Index into ImporterScene::meshes, -1 if no mesh.)doc";

static const char *__doc_falcor_ImporterNode_name = R"doc(Node name.)doc";

static const char *__doc_falcor_ImporterNode_parent = R"doc(Index into ImporterScene::nodes, -1 if root node.)doc";

static const char *__doc_falcor_ImporterNode_prototype_index = R"doc()doc";

static const char *__doc_falcor_ImporterNode_transform =
R"doc(Node transformation matrix (i.e. transform from object space to parent
or world space).)doc";

static const char *__doc_falcor_ImporterNode_world_aabb = R"doc(World-space bounding box of this node and all its children.)doc";

static const char *__doc_falcor_ImporterPrototype =
R"doc(Prototype for instancing more complex objects than simple meshes.
Allows nested instancing and instancing of larger groups of meshes.)doc";

static const char *__doc_falcor_ImporterPrototype_name = R"doc(Prototype name.)doc";

static const char *__doc_falcor_ImporterPrototype_root_node = R"doc(Index into ImporterScene::nodes, root node of the hierarchy.)doc";

static const char *__doc_falcor_ImporterScene = R"doc()doc";

static const char *__doc_falcor_ImporterScene_add_default_camera_best_view =
R"doc(Adds a new camera and returns its index. Evaluates multiple view
directions over per-mesh AABBs, appends a camera node, and returns the
newly added camera index.)doc";

static const char *__doc_falcor_ImporterScene_add_default_camera_robust_fit =
R"doc(Adds a new camera and returns its index. Uses outlier-resistant
bounds/center from mesh AABBs, appends a camera node, and returns the
newly added camera index.)doc";

static const char *__doc_falcor_ImporterScene_animation =
R"doc(Animation data (single animation per scene, empty channels if no
animation).)doc";

static const char *__doc_falcor_ImporterScene_calculate_aabbs = R"doc(Calculate AABBs for all meshes and nodes in the scene.)doc";

static const char *__doc_falcor_ImporterScene_cameras = R"doc(Camera array.)doc";

static const char *__doc_falcor_ImporterScene_curves = R"doc(Curve array.)doc";

static const char *__doc_falcor_ImporterScene_flatten_node_hierarchy =
R"doc(Flatten the node hierarchy by removing parent-child relationships and
accumulating transforms into world space. Only nodes with meshes are
preserved.)doc";

static const char *__doc_falcor_ImporterScene_get_scene_aabb =
R"doc(Get the overall AABB of the entire scene (union of all root node
AABBs).)doc";

static const char *__doc_falcor_ImporterScene_lights = R"doc(Light array.)doc";

static const char *__doc_falcor_ImporterScene_make_clay_scene =
R"doc(Replace all materials to be gray diffuse only. Remove all lights from
the scene and add a single constant light.)doc";

static const char *__doc_falcor_ImporterScene_materials = R"doc(Material array.)doc";

static const char *__doc_falcor_ImporterScene_meshes = R"doc(Mesh array.)doc";

static const char *__doc_falcor_ImporterScene_nodes = R"doc(Node array.)doc";

static const char *__doc_falcor_ImporterScene_prototypes = R"doc(Prototype array.)doc";

static const char *__doc_falcor_ImporterScene_rescale_to_fit =
R"doc(Rescale the scene to fit within a uniform box of the specified size
This applies a uniform scale and translation to center the scene

Parameter ``box_size``:
    The size of the target box (default: 1.0))doc";

static const char *__doc_falcor_ImporterScene_root_nodes = R"doc(Root node indices.)doc";

static const char *__doc_falcor_ImporterScene_textures = R"doc(Texture array.)doc";

static const char *__doc_falcor_ImporterSemantic =
R"doc(Semantic meaning that importer assigns to a vertex attribute. For some
formats this may be explicitly specified, for others it may be
inferred from attribute name (e.g., "POSITION", "pos", "uv1"). If
semantic is 'unknown', the importer was unable to assign it any
specific meaning.)doc";

static const char *__doc_falcor_ImporterSemantic_bitangent = R"doc()doc";

static const char *__doc_falcor_ImporterSemantic_color = R"doc()doc";

static const char *__doc_falcor_ImporterSemantic_handedness = R"doc()doc";

static const char *__doc_falcor_ImporterSemantic_info = R"doc()doc";

static const char *__doc_falcor_ImporterSemantic_joints = R"doc()doc";

static const char *__doc_falcor_ImporterSemantic_normal = R"doc()doc";

static const char *__doc_falcor_ImporterSemantic_position = R"doc()doc";

static const char *__doc_falcor_ImporterSemantic_tangent = R"doc()doc";

static const char *__doc_falcor_ImporterSemantic_tex_coord = R"doc()doc";

static const char *__doc_falcor_ImporterSemantic_unknown = R"doc()doc";

static const char *__doc_falcor_ImporterSemantic_weights = R"doc()doc";

static const char *__doc_falcor_ImporterTexture = R"doc()doc";

static const char *__doc_falcor_ImporterTexture_2 = R"doc()doc";

static const char *__doc_falcor_ImporterTexture_is_srgb = R"doc(True when USD explicitly specifies that the texture is sRGB.)doc";

static const char *__doc_falcor_ImporterTexture_mag_filter = R"doc(Magnification filter mode)doc";

static const char *__doc_falcor_ImporterTexture_mime_type = R"doc(MIME type for embedded texture data (e.g., "image/png", "image/jpeg"))doc";

static const char *__doc_falcor_ImporterTexture_min_filter = R"doc(Minification filter mode (simplified from glTF mipmap modes))doc";

static const char *__doc_falcor_ImporterTexture_operator_le = R"doc()doc";

static const char *__doc_falcor_ImporterTexture_source_name = R"doc()doc";

static const char *__doc_falcor_ImporterTexture_texture_data = R"doc(Optional texture data as bytes (alternative to texture_path))doc";

static const char *__doc_falcor_ImporterTexture_texture_path = R"doc()doc";

static const char *__doc_falcor_ImporterTexture_texture_transform = R"doc()doc";

static const char *__doc_falcor_ImporterTexture_wrap_s = R"doc(S (U) wrapping mode)doc";

static const char *__doc_falcor_ImporterTexture_wrap_t = R"doc(T (V) wrapping mode)doc";

static const char *__doc_falcor_ImporterVertexStream =
R"doc(Helper class to make it simple to read/write a stream of a given
attribute type for the vertices of a mesh.)doc";

static const char *__doc_falcor_ImporterVertexStreamRO =
R"doc(Helper class to make it simple to read/write a stream of a given
attribute type for the vertices of a mesh.)doc";

static const char *__doc_falcor_ImporterVertexStreamRO_data = R"doc()doc";

static const char *__doc_falcor_ImporterVertexStreamRO_first = R"doc()doc";

static const char *__doc_falcor_ImporterVertexStreamRO_operator_array = R"doc()doc";

static const char *__doc_falcor_ImporterVertexStreamRO_size = R"doc()doc";

static const char *__doc_falcor_ImporterVertexStreamRO_stride = R"doc()doc";

static const char *__doc_falcor_ImporterVertexStreamRO_valid = R"doc()doc";

static const char *__doc_falcor_ImporterVertexStream_data = R"doc()doc";

static const char *__doc_falcor_ImporterVertexStream_first = R"doc()doc";

static const char *__doc_falcor_ImporterVertexStream_first_2 = R"doc()doc";

static const char *__doc_falcor_ImporterVertexStream_operator_array = R"doc()doc";

static const char *__doc_falcor_ImporterVertexStream_operator_array_2 = R"doc()doc";

static const char *__doc_falcor_ImporterVertexStream_size = R"doc()doc";

static const char *__doc_falcor_ImporterVertexStream_stride = R"doc()doc";

static const char *__doc_falcor_ImporterVertexStream_valid = R"doc()doc";

static const char *__doc_falcor_IndexedVector =
R"doc(Turns array of values into a set of values, and an array of indices to
them. Effectively deduplicating the values.)doc";

static const char *__doc_falcor_IndexedVector_append =
R"doc(Appends value to the vector, returns the index at which it is
positioned, and returns true if the value already existed.)doc";

static const char *__doc_falcor_IndexedVector_append_2 = R"doc()doc";

static const char *__doc_falcor_IndexedVector_get_indices = R"doc()doc";

static const char *__doc_falcor_IndexedVector_get_values = R"doc()doc";

static const char *__doc_falcor_IndexedVector_m_index_map = R"doc()doc";

static const char *__doc_falcor_IndexedVector_m_indices = R"doc()doc";

static const char *__doc_falcor_IndexedVector_m_values = R"doc()doc";

static const char *__doc_falcor_Light =
R"doc(Base class for lights. Each sub-class represents a different light
type on the shader side.)doc";

static const char *__doc_falcor_LightSystem = R"doc(Scene system responsible for managing lights.)doc";

static const char *__doc_falcor_LightSystem_LightSystem =
R"doc(Constructor.

Parameter ``scene``:
    The scene this system belongs to.)doc";

static const char *__doc_falcor_LightSystem_bind_to_scene = R"doc()doc";

static const char *__doc_falcor_LightSystem_class_name = R"doc()doc";

static const char *__doc_falcor_LightSystem_m_analytic_light_count = R"doc()doc";

static const char *__doc_falcor_LightSystem_m_analytic_light_selection_distribution = R"doc()doc";

static const char *__doc_falcor_LightSystem_m_components = R"doc()doc";

static const char *__doc_falcor_LightSystem_m_environment_light_id = R"doc()doc";

static const char *__doc_falcor_LightSystem_m_light_count = R"doc()doc";

static const char *__doc_falcor_LightSystem_m_light_data = R"doc()doc";

static const char *__doc_falcor_LightSystem_m_type_conformances = R"doc()doc";

static const char *__doc_falcor_LightSystem_required_type_conformances = R"doc(Get the type conformances for all light types.)doc";

static const char *__doc_falcor_LightSystem_update = R"doc()doc";

static const char *__doc_falcor_Light_Light = R"doc(Constructor.)doc";

static const char *__doc_falcor_Light_active =
R"doc(Active flag. Light does not contribute to the scene lighting if not
active.)doc";

static const char *__doc_falcor_Light_class_descriptor = R"doc()doc";

static const char *__doc_falcor_Light_class_name = R"doc()doc";

static const char *__doc_falcor_Light_exposure = R"doc(Light exposure value in exposure stops.)doc";

static const char *__doc_falcor_Light_light_flags = R"doc(Light flags.)doc";

static const char *__doc_falcor_Light_light_id = R"doc(Light ID.)doc";

static const char *__doc_falcor_Light_light_type = R"doc(Light type.)doc";

static const char *__doc_falcor_Light_m_active = R"doc()doc";

static const char *__doc_falcor_Light_m_exposure = R"doc()doc";

static const char *__doc_falcor_Light_m_light_flags = R"doc()doc";

static const char *__doc_falcor_Light_m_light_id = R"doc()doc";

static const char *__doc_falcor_Light_m_light_type = R"doc()doc";

static const char *__doc_falcor_Light_m_slang_type_name = R"doc()doc";

static const char *__doc_falcor_Light_on_entity_transform_changed = R"doc()doc";

static const char *__doc_falcor_Light_reflect = R"doc(Reflect this class.)doc";

static const char *__doc_falcor_Light_reflected_class_name = R"doc()doc";

static const char *__doc_falcor_Light_set_active = R"doc()doc";

static const char *__doc_falcor_Light_set_exposure = R"doc()doc";

static const char *__doc_falcor_Light_set_light_flags = R"doc()doc";

static const char *__doc_falcor_Light_set_light_id = R"doc()doc";

static const char *__doc_falcor_Light_set_light_type = R"doc()doc";

static const char *__doc_falcor_Light_set_slang_type_name = R"doc()doc";

static const char *__doc_falcor_Light_slang_type_name =
R"doc(Slang type name of the struct representing this light type. Specifies
the type used for write_to_cursor().)doc";

static const char *__doc_falcor_Light_static_class_descriptor = R"doc()doc";

static const char *__doc_falcor_Light_update =
R"doc(Called during Scene::update() to update this light.

Parameter ``ctx``:
    Update context.)doc";

static const char *__doc_falcor_Light_write_slangpy_signature = R"doc(Write the SlangPy cache signature for cursor-writer binding.)doc";

static const char *__doc_falcor_Light_write_to_cursor = R"doc()doc";

static const char *__doc_falcor_Light_write_to_cursor_2 = R"doc()doc";

static const char *__doc_falcor_Light_write_to_cursor_3 =
R"doc(Called during Scene::update() to write this light to the corresponding
slang struct.)doc";

static const char *__doc_falcor_Light_write_to_cursor_4 = R"doc()doc";

static const char *__doc_falcor_ManagedBuffer =
R"doc(A simple Wrapper over a Buffer, allowing setting CPU data and syncing
only in update. GPU address/handle can be obtained at any time, even
before calling `update`.)doc";

static const char *__doc_falcor_ManagedBuffer_ManagedBuffer =
R"doc(Constructor.

Parameter ``desc``:
    Buffer descriptor.)doc";

static const char *__doc_falcor_ManagedBuffer_buffer =
R"doc(Device buffer. This is only available if not empty and after calling
`update_buffer`.)doc";

static const char *__doc_falcor_ManagedBuffer_cdata =
R"doc(Always returns constant data (without setting dirty flags), even on
non-const ManagedBuffer.)doc";

static const char *__doc_falcor_ManagedBuffer_class_name = R"doc()doc";

static const char *__doc_falcor_ManagedBuffer_create =
R"doc(Create a new ManagedBuffer.

Parameter ``desc``:
    Buffer descriptor.)doc";

static const char *__doc_falcor_ManagedBuffer_data =
R"doc(Returns data and marks entire buffer dirty, as it can be arbitrarily
modified by the caller.)doc";

static const char *__doc_falcor_ManagedBuffer_data_2 = R"doc(Host data (const access).)doc";

static const char *__doc_falcor_ManagedBuffer_data_unsafe =
R"doc(Returns a mutable pointer to the host data without setting dirty
flags. The caller is responsible for manually marking dirty regions if
data is modified.)doc";

static const char *__doc_falcor_ManagedBuffer_is_dirty = R"doc(True if some host data has not been uploaded to the device buffer.)doc";

static const char *__doc_falcor_ManagedBuffer_m_buffer = R"doc(Device buffer.)doc";

static const char *__doc_falcor_ManagedBuffer_m_buffer_desc = R"doc()doc";

static const char *__doc_falcor_ManagedBuffer_m_data = R"doc(Host data.)doc";

static const char *__doc_falcor_ManagedBuffer_m_dirty_range = R"doc(Current dirty range.)doc";

static const char *__doc_falcor_ManagedBuffer_m_keep_alive =
R"doc(Keeps m_buffer alive until next `update` is called, so we don't kill a
used GPU resource with resize.)doc";

static const char *__doc_falcor_ManagedBuffer_mark_all_dirty = R"doc(Mark the entire buffer as dirty.)doc";

static const char *__doc_falcor_ManagedBuffer_mark_dirty =
R"doc(Explicitly mark a byte range as dirty.

Parameter ``begin``:
    Start of the dirty range (byte offset).

Parameter ``end``:
    End of the dirty range (byte offset, exclusive).)doc";

static const char *__doc_falcor_ManagedBuffer_reset_dirty_range = R"doc(Explicitly reset the dirty range.)doc";

static const char *__doc_falcor_ManagedBuffer_resize =
R"doc(Resize buffer to a new size.

Parameter ``size``:
    New size in bytes.)doc";

static const char *__doc_falcor_ManagedBuffer_set_data =
R"doc(Set data from a source buffer.

Parameter ``src``:
    Source data pointer.

Parameter ``src_size``:
    Size in bytes to copy.

Parameter ``offset``:
    Byte offset into the buffer.)doc";

static const char *__doc_falcor_ManagedBuffer_set_data_2 = R"doc()doc";

static const char *__doc_falcor_ManagedBuffer_size = R"doc(Current size in bytes.)doc";

static const char *__doc_falcor_ManagedBuffer_update_buffer =
R"doc(Update the device buffer to reflect the host data. Creates a temporary
command encoder, uploads dirty data, and submits.

Parameter ``device``:
    Device to use.)doc";

static const char *__doc_falcor_ManagedBuffer_update_buffer_2 =
R"doc(Update the device buffer to reflect the host data. If the host buffer
is not empty or different size than the device buffer, a new device
buffer is allocated. The device buffer is updated with the data that
changed on the host buffer since the last call to `update_buffer`.

Parameter ``command_encoder``:
    Command encoder to use for upload.)doc";

static const char *__doc_falcor_ManagedVector =
R"doc(A vector that maintains synchronized host and device copies with dirty
tracking. ManagedVector provides a convenient way to manage device
buffers that are frequently updated from the CPU. It maintains a host-
side vector and a device-side buffer, tracking which elements have
been modified to minimize data transfer.)doc";

static const char *__doc_falcor_ManagedVectorDesc = R"doc(ManagedVector descriptor.)doc";

static const char *__doc_falcor_ManagedVectorDesc_label = R"doc(Device buffer label.)doc";

static const char *__doc_falcor_ManagedVectorDesc_max_buffer_size =
R"doc(Maximum size of device buffer (defaults to 2GB which is a D3D12 limit
mostly).)doc";

static const char *__doc_falcor_ManagedVectorDesc_memory_type = R"doc(Device buffer memory type.)doc";

static const char *__doc_falcor_ManagedVectorDesc_usage = R"doc(Device buffer usage.)doc";

static const char *__doc_falcor_ManagedVector_ManagedVector = R"doc(Constructor.)doc";

static const char *__doc_falcor_ManagedVector_ManagedVector_2 = R"doc()doc";

static const char *__doc_falcor_ManagedVector_ManagedVector_3 = R"doc()doc";

static const char *__doc_falcor_ManagedVector_buffer =
R"doc(Device buffer. This is only available if not empty and after calling
`update_buffer`.)doc";

static const char *__doc_falcor_ManagedVector_data = R"doc(Host data.)doc";

static const char *__doc_falcor_ManagedVector_data_2 = R"doc()doc";

static const char *__doc_falcor_ManagedVector_ensure_size =
R"doc(Ensure the buffer has a minimum size. Size must not exceed capacity.

Parameter ``size``:
    New minimum size.)doc";

static const char *__doc_falcor_ManagedVector_is_dirty = R"doc(True if some host data has not been uploaded to the device buffer.)doc";

static const char *__doc_falcor_ManagedVector_m_buffer = R"doc(Device buffer.)doc";

static const char *__doc_falcor_ManagedVector_m_data = R"doc(Host data.)doc";

static const char *__doc_falcor_ManagedVector_m_desc = R"doc()doc";

static const char *__doc_falcor_ManagedVector_m_dirty_range = R"doc(Current dirty range.)doc";

static const char *__doc_falcor_ManagedVector_mark_all_dirty = R"doc(Mark the entire buffer as dirty.)doc";

static const char *__doc_falcor_ManagedVector_mark_dirty = R"doc(Explicitly mark a range as dirty.)doc";

static const char *__doc_falcor_ManagedVector_operator_array = R"doc(Element accessor (const access).)doc";

static const char *__doc_falcor_ManagedVector_operator_array_2 = R"doc(Element accessor (non-const access). Marks the element as dirty.)doc";

static const char *__doc_falcor_ManagedVector_operator_assign = R"doc()doc";

static const char *__doc_falcor_ManagedVector_operator_assign_2 = R"doc()doc";

static const char *__doc_falcor_ManagedVector_reset_dirty_range = R"doc(Explicitly reset the dirty range.)doc";

static const char *__doc_falcor_ManagedVector_resize =
R"doc(Resize buffer to a new size. Size must not exceed capacity.

Parameter ``size``:
    New size.)doc";

static const char *__doc_falcor_ManagedVector_size = R"doc(Current number of elements in the buffer.)doc";

static const char *__doc_falcor_ManagedVector_update_buffer =
R"doc(Update the device buffer to reflect the host data. If the host buffer
is not empty or bigger than the device buffer, a new device buffer is
allocated. The device buffer is updated with the data that changed on
the host buffer since the last call to `update_buffer`.

Parameter ``command_encoder``:
    Command encoder to use for upload.

Parameter ``upload_dirty``:
    Upload dirty data (enabled by default).)doc";

static const char *__doc_falcor_Material = R"doc(Base class for scene materials.)doc";

static const char *__doc_falcor_Material_2 = R"doc(Base class for scene materials.)doc";

static const char *__doc_falcor_MaterialSystem = R"doc(Scene system responsible for managing materials.)doc";

static const char *__doc_falcor_MaterialSystem_MaterialSystem =
R"doc(Constructor.

Parameter ``scene``:
    The scene this system belongs to.)doc";

static const char *__doc_falcor_MaterialSystem_allocate_material_id = R"doc(Allocate a new material ID.)doc";

static const char *__doc_falcor_MaterialSystem_bind_to_scene = R"doc()doc";

static const char *__doc_falcor_MaterialSystem_class_name = R"doc()doc";

static const char *__doc_falcor_MaterialSystem_free_material_id = R"doc(Free a material ID.)doc";

static const char *__doc_falcor_MaterialSystem_m_free_material_ids = R"doc()doc";

static const char *__doc_falcor_MaterialSystem_m_material_data = R"doc()doc";

static const char *__doc_falcor_MaterialSystem_m_materials = R"doc()doc";

static const char *__doc_falcor_MaterialSystem_m_next_material_id = R"doc()doc";

static const char *__doc_falcor_MaterialSystem_m_required_modules = R"doc()doc";

static const char *__doc_falcor_MaterialSystem_m_type_conformances = R"doc()doc";

static const char *__doc_falcor_MaterialSystem_required_modules = R"doc(Get the additional modules required by all active materials.)doc";

static const char *__doc_falcor_MaterialSystem_required_type_conformances = R"doc(Get the type conformances for all active materials.)doc";

static const char *__doc_falcor_MaterialSystem_update = R"doc()doc";

static const char *__doc_falcor_MaterialXMaterial = R"doc(MaterialX material.)doc";

static const char *__doc_falcor_MaterialXMaterial_MaterialXMaterial = R"doc(Constructor.)doc";

static const char *__doc_falcor_MaterialXMaterial_all_material_params = R"doc()doc";

static const char *__doc_falcor_MaterialXMaterial_build_texture_list =
R"doc(List the texture handles used by this material's bound parameters.
Extracts TextureInfo values from the codegen param list. Empty before
on_load_resources() has run.)doc";

static const char *__doc_falcor_MaterialXMaterial_class_descriptor = R"doc()doc";

static const char *__doc_falcor_MaterialXMaterial_class_name = R"doc()doc";

static const char *__doc_falcor_MaterialXMaterial_data_buffer = R"doc()doc";

static const char *__doc_falcor_MaterialXMaterial_dynamic_properties = R"doc()doc";

static const char *__doc_falcor_MaterialXMaterial_dynamic_properties_2 = R"doc()doc";

static const char *__doc_falcor_MaterialXMaterial_m_codegen_desc = R"doc()doc";

static const char *__doc_falcor_MaterialXMaterial_m_codegen_properties_hash = R"doc()doc";

static const char *__doc_falcor_MaterialXMaterial_m_codegen_result = R"doc()doc";

static const char *__doc_falcor_MaterialXMaterial_m_data_buffer = R"doc()doc";

static const char *__doc_falcor_MaterialXMaterial_m_debug_load_shader_path = R"doc()doc";

static const char *__doc_falcor_MaterialXMaterial_m_debug_write_shader_path = R"doc()doc";

static const char *__doc_falcor_MaterialXMaterial_m_has_entry_point_volume_properties = R"doc()doc";

static const char *__doc_falcor_MaterialXMaterial_m_lut_globals = R"doc()doc";

static const char *__doc_falcor_MaterialXMaterial_m_material_properties = R"doc(Dynamic properties (generated from MaterialX definition).)doc";

static const char *__doc_falcor_MaterialXMaterial_m_mtlx_auto_transmission = R"doc()doc";

static const char *__doc_falcor_MaterialXMaterial_m_mtlx_autogamma = R"doc()doc";

static const char *__doc_falcor_MaterialXMaterial_m_mtlx_basepath = R"doc()doc";

static const char *__doc_falcor_MaterialXMaterial_m_mtlx_buffer = R"doc()doc";

static const char *__doc_falcor_MaterialXMaterial_m_mtlx_compensation = R"doc()doc";

static const char *__doc_falcor_MaterialXMaterial_m_mtlx_editable_params = R"doc()doc";

static const char *__doc_falcor_MaterialXMaterial_m_mtlx_flip_v_texcoord = R"doc()doc";

static const char *__doc_falcor_MaterialXMaterial_m_mtlx_geomprop_ids = R"doc()doc";

static const char *__doc_falcor_MaterialXMaterial_m_mtlx_geomprop_names = R"doc()doc";

static const char *__doc_falcor_MaterialXMaterial_m_mtlx_layering_mode = R"doc()doc";

static const char *__doc_falcor_MaterialXMaterial_m_mtlx_layeringmethod = R"doc()doc";

static const char *__doc_falcor_MaterialXMaterial_m_mtlx_node_name = R"doc()doc";

static const char *__doc_falcor_MaterialXMaterial_m_mtlx_optimize_graph = R"doc()doc";

static const char *__doc_falcor_MaterialXMaterial_m_mtlx_path = R"doc()doc";

static const char *__doc_falcor_MaterialXMaterial_m_mtlx_path_resolver = R"doc(Attempts to resolve paths based on the library path)doc";

static const char *__doc_falcor_MaterialXMaterial_m_mtlx_search_paths = R"doc()doc";

static const char *__doc_falcor_MaterialXMaterial_m_mtlx_target_color_space_override = R"doc()doc";

static const char *__doc_falcor_MaterialXMaterial_m_mtlx_transmissive_bsdfs = R"doc()doc";

static const char *__doc_falcor_MaterialXMaterial_m_require_codegen = R"doc()doc";

static const char *__doc_falcor_MaterialXMaterial_m_slang_module = R"doc()doc";

static const char *__doc_falcor_MaterialXMaterial_mtlx_geomprop_ids_property = R"doc()doc";

static const char *__doc_falcor_MaterialXMaterial_mtlx_geomprop_names_property = R"doc()doc";

static const char *__doc_falcor_MaterialXMaterial_on_load_resources = R"doc()doc";

static const char *__doc_falcor_MaterialXMaterial_on_remove_from_scene = R"doc()doc";

static const char *__doc_falcor_MaterialXMaterial_reflect = R"doc(Reflect this class.)doc";

static const char *__doc_falcor_MaterialXMaterial_reflected_class_name = R"doc()doc";

static const char *__doc_falcor_MaterialXMaterial_remove_lut_scene_globals = R"doc()doc";

static const char *__doc_falcor_MaterialXMaterial_require_codegen = R"doc()doc";

static const char *__doc_falcor_MaterialXMaterial_required_data_buffer_size = R"doc()doc";

static const char *__doc_falcor_MaterialXMaterial_required_module = R"doc()doc";

static const char *__doc_falcor_MaterialXMaterial_requires_data_buffer = R"doc()doc";

static const char *__doc_falcor_MaterialXMaterial_requires_data_buffer_2 = R"doc()doc";

static const char *__doc_falcor_MaterialXMaterial_run_codegen = R"doc()doc";

static const char *__doc_falcor_MaterialXMaterial_set_mtlx_geomprop_ids_property = R"doc()doc";

static const char *__doc_falcor_MaterialXMaterial_set_mtlx_geomprop_names_property = R"doc()doc";

static const char *__doc_falcor_MaterialXMaterial_set_properties = R"doc()doc";

static const char *__doc_falcor_MaterialXMaterial_static_class_descriptor = R"doc()doc";

static const char *__doc_falcor_MaterialXMaterial_update = R"doc()doc";

static const char *__doc_falcor_MaterialXMaterial_update_lut_scene_globals = R"doc()doc";

static const char *__doc_falcor_MaterialXMaterial_validate_device_support = R"doc()doc";

static const char *__doc_falcor_MaterialXMaterial_write_to_cursor = R"doc()doc";

static const char *__doc_falcor_MaterialXMaterial_write_to_cursor_2 = R"doc()doc";

static const char *__doc_falcor_MaterialXMaterial_write_to_cursor_impl = R"doc()doc";

static const char *__doc_falcor_Material_DirtyFlags = R"doc(Dirty flags for Material.)doc";

static const char *__doc_falcor_Material_DirtyFlags_added = R"doc()doc";

static const char *__doc_falcor_Material_DirtyFlags_info = R"doc()doc";

static const char *__doc_falcor_Material_DirtyFlags_none = R"doc()doc";

static const char *__doc_falcor_Material_DirtyFlags_properties = R"doc()doc";

static const char *__doc_falcor_Material_DirtyFlags_removed = R"doc()doc";

static const char *__doc_falcor_Material_DirtyFlags_resources = R"doc()doc";

static const char *__doc_falcor_Material_Material = R"doc(Constructor.)doc";

static const char *__doc_falcor_Material_class_descriptor = R"doc()doc";

static const char *__doc_falcor_Material_class_name = R"doc()doc";

static const char *__doc_falcor_Material_flags = R"doc(Returns material flags used by the scene material header.)doc";

static const char *__doc_falcor_Material_kind = R"doc()doc";

static const char *__doc_falcor_Material_m_material_id = R"doc()doc";

static const char *__doc_falcor_Material_m_slang_type_name = R"doc()doc";

static const char *__doc_falcor_Material_mark_dirty = R"doc()doc";

static const char *__doc_falcor_Material_material_id = R"doc(Material ID.)doc";

static const char *__doc_falcor_Material_reflect = R"doc(Reflect this class.)doc";

static const char *__doc_falcor_Material_reflected_class_name = R"doc()doc";

static const char *__doc_falcor_Material_remove = R"doc()doc";

static const char *__doc_falcor_Material_required_module = R"doc(Returns an additional Slang module required by this material.)doc";

static const char *__doc_falcor_Material_set_material_id = R"doc()doc";

static const char *__doc_falcor_Material_set_slang_type_name = R"doc()doc";

static const char *__doc_falcor_Material_slang_type_name =
R"doc(Slang type name of the struct representing this material type.
Specifies the type used for write_to_cursor().)doc";

static const char *__doc_falcor_Material_static_class_descriptor = R"doc()doc";

static const char *__doc_falcor_Material_update =
R"doc(Called during Scene::update() to update this material.

Parameter ``ctx``:
    Update context.)doc";

static const char *__doc_falcor_Material_write_slangpy_signature = R"doc(Write the SlangPy cache signature for cursor-writer binding.)doc";

static const char *__doc_falcor_Material_write_to_cursor = R"doc()doc";

static const char *__doc_falcor_Material_write_to_cursor_2 = R"doc()doc";

static const char *__doc_falcor_Material_write_to_cursor_3 =
R"doc(Called during Scene::update() to write this material to the
corresponding slang struct (BufferElementCursor).)doc";

static const char *__doc_falcor_Material_write_to_cursor_4 =
R"doc(Called during Scene::update() to write this material to the
corresponding slang struct (ShaderCursor).)doc";

static const char *__doc_falcor_MeshWriter =
R"doc(MeshWriter provides a simple interface for constructing and exporting
triangle meshes to glTF format.

Usage: 1. Call begin_submesh() to start a new submesh. 2. Add
triangles using triangle(). 3. Call end_submesh() to finalize the
submesh. 4. Repeat steps 1-3 for additional submeshes. 5. Call write()
to export all submeshes to a .gltf or .glb file.

Each submesh becomes a separate mesh/node in the resulting glTF scene.)doc";

static const char *__doc_falcor_MeshWriter_Submesh = R"doc()doc";

static const char *__doc_falcor_MeshWriter_Submesh_indices = R"doc()doc";

static const char *__doc_falcor_MeshWriter_Submesh_name = R"doc()doc";

static const char *__doc_falcor_MeshWriter_Submesh_positions = R"doc()doc";

static const char *__doc_falcor_MeshWriter_begin_submesh =
R"doc(Begin defining a new submesh.

Parameter ``name``:
    Name for the submesh (used as mesh and node name in glTF).)doc";

static const char *__doc_falcor_MeshWriter_class_name = R"doc()doc";

static const char *__doc_falcor_MeshWriter_end_submesh = R"doc(End the current submesh definition.)doc";

static const char *__doc_falcor_MeshWriter_m_current_submesh = R"doc(Currently active submesh, or nullptr if none.)doc";

static const char *__doc_falcor_MeshWriter_m_submeshes = R"doc()doc";

static const char *__doc_falcor_MeshWriter_triangle =
R"doc(Add a triangle to the current submesh.

Parameter ``v0``:
    First vertex position.

Parameter ``v1``:
    Second vertex position.

Parameter ``v2``:
    Third vertex position.)doc";

static const char *__doc_falcor_MeshWriter_write =
R"doc(Write all submeshes to a glTF file.

Parameter ``path``:
    Output file path. Use .gltf for JSON format or .glb for binary
    format.)doc";

static const char *__doc_falcor_OffsetAllocator = R"doc()doc";

static const char *__doc_falcor_OffsetAllocator_Allocation = R"doc()doc";

static const char *__doc_falcor_OffsetAllocator_Allocation_is_valid = R"doc()doc";

static const char *__doc_falcor_OffsetAllocator_Allocation_metadata = R"doc()doc";

static const char *__doc_falcor_OffsetAllocator_Allocation_offset = R"doc()doc";

static const char *__doc_falcor_OffsetAllocator_Allocation_operator_bool = R"doc()doc";

static const char *__doc_falcor_OffsetAllocator_Node = R"doc()doc";

static const char *__doc_falcor_OffsetAllocator_Node_bin_list_next = R"doc()doc";

static const char *__doc_falcor_OffsetAllocator_Node_bin_list_prev = R"doc()doc";

static const char *__doc_falcor_OffsetAllocator_Node_data_offset = R"doc()doc";

static const char *__doc_falcor_OffsetAllocator_Node_data_size = R"doc()doc";

static const char *__doc_falcor_OffsetAllocator_Node_neighbor_next = R"doc()doc";

static const char *__doc_falcor_OffsetAllocator_Node_neighbor_prev = R"doc()doc";

static const char *__doc_falcor_OffsetAllocator_Node_used = R"doc()doc";

static const char *__doc_falcor_OffsetAllocator_OffsetAllocator = R"doc()doc";

static const char *__doc_falcor_OffsetAllocator_OffsetAllocator_2 = R"doc()doc";

static const char *__doc_falcor_OffsetAllocator_StorageReport = R"doc()doc";

static const char *__doc_falcor_OffsetAllocator_StorageReportFull = R"doc()doc";

static const char *__doc_falcor_OffsetAllocator_StorageReportFull_Region = R"doc()doc";

static const char *__doc_falcor_OffsetAllocator_StorageReportFull_Region_count = R"doc()doc";

static const char *__doc_falcor_OffsetAllocator_StorageReportFull_Region_size = R"doc()doc";

static const char *__doc_falcor_OffsetAllocator_StorageReportFull_free_regions = R"doc()doc";

static const char *__doc_falcor_OffsetAllocator_StorageReport_largest_free_region = R"doc()doc";

static const char *__doc_falcor_OffsetAllocator_StorageReport_total_free_space = R"doc()doc";

static const char *__doc_falcor_OffsetAllocator_allocate = R"doc()doc";

static const char *__doc_falcor_OffsetAllocator_allocation_size = R"doc()doc";

static const char *__doc_falcor_OffsetAllocator_free = R"doc()doc";

static const char *__doc_falcor_OffsetAllocator_get_current_allocs = R"doc()doc";

static const char *__doc_falcor_OffsetAllocator_get_free_storage = R"doc()doc";

static const char *__doc_falcor_OffsetAllocator_get_max_allocs = R"doc()doc";

static const char *__doc_falcor_OffsetAllocator_get_size = R"doc()doc";

static const char *__doc_falcor_OffsetAllocator_insert_node_into_bin = R"doc()doc";

static const char *__doc_falcor_OffsetAllocator_m_bin_indices = R"doc()doc";

static const char *__doc_falcor_OffsetAllocator_m_current_allocs = R"doc()doc";

static const char *__doc_falcor_OffsetAllocator_m_free_nodes = R"doc()doc";

static const char *__doc_falcor_OffsetAllocator_m_free_offset = R"doc()doc";

static const char *__doc_falcor_OffsetAllocator_m_free_storage = R"doc()doc";

static const char *__doc_falcor_OffsetAllocator_m_max_allocs = R"doc()doc";

static const char *__doc_falcor_OffsetAllocator_m_nodes = R"doc()doc";

static const char *__doc_falcor_OffsetAllocator_m_size = R"doc()doc";

static const char *__doc_falcor_OffsetAllocator_m_used_bins = R"doc()doc";

static const char *__doc_falcor_OffsetAllocator_m_used_bins_top = R"doc()doc";

static const char *__doc_falcor_OffsetAllocator_remove_node_from_bin = R"doc()doc";

static const char *__doc_falcor_OffsetAllocator_reset = R"doc()doc";

static const char *__doc_falcor_OffsetAllocator_storage_report = R"doc()doc";

static const char *__doc_falcor_OffsetAllocator_storage_report_full = R"doc()doc";

static const char *__doc_falcor_OpenPBRMaterial = R"doc(OpenPBR material.)doc";

static const char *__doc_falcor_OpenPBRMaterial_Attribute = R"doc()doc";

static const char *__doc_falcor_OpenPBRMaterial_AttributeFlags = R"doc()doc";

static const char *__doc_falcor_OpenPBRMaterial_AttributeFlags_none = R"doc()doc";

static const char *__doc_falcor_OpenPBRMaterial_AttributeInfo = R"doc()doc";

static const char *__doc_falcor_OpenPBRMaterial_AttributeInfo_default_value = R"doc()doc";

static const char *__doc_falcor_OpenPBRMaterial_AttributeInfo_flags = R"doc()doc";

static const char *__doc_falcor_OpenPBRMaterial_AttributeInfo_group = R"doc()doc";

static const char *__doc_falcor_OpenPBRMaterial_AttributeInfo_kind = R"doc()doc";

static const char *__doc_falcor_OpenPBRMaterial_AttributeInfo_label = R"doc()doc";

static const char *__doc_falcor_OpenPBRMaterial_AttributeInfo_name = R"doc()doc";

static const char *__doc_falcor_OpenPBRMaterial_AttributeInfo_slang_name = R"doc()doc";

static const char *__doc_falcor_OpenPBRMaterial_AttributeInfo_type = R"doc()doc";

static const char *__doc_falcor_OpenPBRMaterial_AttributeKind = R"doc()doc";

static const char *__doc_falcor_OpenPBRMaterial_AttributeKind_color = R"doc(Color attribute in [0, 1] range.)doc";

static const char *__doc_falcor_OpenPBRMaterial_AttributeKind_ior = R"doc(Index of refraction in [1, +inf) range.)doc";

static const char *__doc_falcor_OpenPBRMaterial_AttributeKind_positive = R"doc(Generic attribute in [0, +inf) range.)doc";

static const char *__doc_falcor_OpenPBRMaterial_AttributeKind_weight = R"doc(Weight attribute in [0, 1] range.)doc";

static const char *__doc_falcor_OpenPBRMaterial_AttributeType = R"doc()doc";

static const char *__doc_falcor_OpenPBRMaterial_AttributeType_float = R"doc()doc";

static const char *__doc_falcor_OpenPBRMaterial_AttributeType_float3 = R"doc()doc";

static const char *__doc_falcor_OpenPBRMaterial_Attribute_get_float = R"doc()doc";

static const char *__doc_falcor_OpenPBRMaterial_Attribute_get_float3 = R"doc()doc";

static const char *__doc_falcor_OpenPBRMaterial_Attribute_info = R"doc()doc";

static const char *__doc_falcor_OpenPBRMaterial_Attribute_set_float = R"doc()doc";

static const char *__doc_falcor_OpenPBRMaterial_Attribute_set_float3 = R"doc()doc";

static const char *__doc_falcor_OpenPBRMaterial_Attribute_texture = R"doc()doc";

static const char *__doc_falcor_OpenPBRMaterial_Attribute_texture_channel = R"doc()doc";

static const char *__doc_falcor_OpenPBRMaterial_Attribute_texture_handle = R"doc()doc";

static const char *__doc_falcor_OpenPBRMaterial_Attribute_texture_path = R"doc()doc";

static const char *__doc_falcor_OpenPBRMaterial_Attribute_value = R"doc()doc";

static const char *__doc_falcor_OpenPBRMaterial_OpenPBRMaterial = R"doc(Constructor.)doc";

static const char *__doc_falcor_OpenPBRMaterial_attribute_by_name = R"doc()doc";

static const char *__doc_falcor_OpenPBRMaterial_class_descriptor = R"doc()doc";

static const char *__doc_falcor_OpenPBRMaterial_class_name = R"doc()doc";

static const char *__doc_falcor_OpenPBRMaterial_dynamic_properties = R"doc()doc";

static const char *__doc_falcor_OpenPBRMaterial_dynamic_properties_2 = R"doc()doc";

static const char *__doc_falcor_OpenPBRMaterial_flags = R"doc()doc";

static const char *__doc_falcor_OpenPBRMaterial_initialize_attributes = R"doc()doc";

static const char *__doc_falcor_OpenPBRMaterial_initialize_properties = R"doc()doc";

static const char *__doc_falcor_OpenPBRMaterial_m_attributes = R"doc()doc";

static const char *__doc_falcor_OpenPBRMaterial_m_attributes_buffer = R"doc()doc";

static const char *__doc_falcor_OpenPBRMaterial_m_geometry_thin_walled = R"doc()doc";

static const char *__doc_falcor_OpenPBRMaterial_m_globals = R"doc()doc";

static const char *__doc_falcor_OpenPBRMaterial_m_material_properties = R"doc(Dynamic properties (generated from attributes).)doc";

static const char *__doc_falcor_OpenPBRMaterial_m_normal_texture = R"doc()doc";

static const char *__doc_falcor_OpenPBRMaterial_m_normal_texture_handle = R"doc()doc";

static const char *__doc_falcor_OpenPBRMaterial_m_normal_texture_path = R"doc()doc";

static const char *__doc_falcor_OpenPBRMaterial_m_normal_texture_scale = R"doc()doc";

static const char *__doc_falcor_OpenPBRMaterial_on_load_resources = R"doc()doc";

static const char *__doc_falcor_OpenPBRMaterial_reflected_class_name = R"doc()doc";

static const char *__doc_falcor_OpenPBRMaterial_require_load_resources = R"doc()doc";

static const char *__doc_falcor_OpenPBRMaterial_require_update = R"doc()doc";

static const char *__doc_falcor_OpenPBRMaterial_set_properties = R"doc()doc";

static const char *__doc_falcor_OpenPBRMaterial_static_class_descriptor = R"doc()doc";

static const char *__doc_falcor_OpenPBRMaterial_sync_attributes_from_properties = R"doc()doc";

static const char *__doc_falcor_OpenPBRMaterial_update = R"doc()doc";

static const char *__doc_falcor_OpenPBRMaterial_write_to_cursor = R"doc()doc";

static const char *__doc_falcor_OpenPBRMaterial_write_to_cursor_2 = R"doc()doc";

static const char *__doc_falcor_OpenPBRMaterial_write_to_cursor_impl = R"doc()doc";

static const char *__doc_falcor_OpenPBRSceneGlobals = R"doc()doc";

static const char *__doc_falcor_OptixDenoiser = R"doc(OptiX denoiser wrapper.)doc";

static const char *__doc_falcor_OptixDenoiserDesc = R"doc()doc";

static const char *__doc_falcor_OptixDenoiserDesc_albedo_guide_layer = R"doc(Whether to use albedo guide layer.)doc";

static const char *__doc_falcor_OptixDenoiserDesc_alpha_mode = R"doc(Alpha processing mode.)doc";

static const char *__doc_falcor_OptixDenoiserDesc_max_height = R"doc(Maximum image height for denoising.)doc";

static const char *__doc_falcor_OptixDenoiserDesc_max_width = R"doc(Maximum image width for denoising.)doc";

static const char *__doc_falcor_OptixDenoiserDesc_model_kind = R"doc(OptiX denoiser model kind.)doc";

static const char *__doc_falcor_OptixDenoiserDesc_normal_guide_layer = R"doc(Whether to use normal guide layer.)doc";

static const char *__doc_falcor_OptixDenoiser_Impl = R"doc()doc";

static const char *__doc_falcor_OptixDenoiser_OptixDenoiser =
R"doc(Create an OptiX denoiser.

Parameter ``device``:
    The device to use for denoising operations.

Parameter ``desc``:
    Descriptor for denoiser initialization.)doc";

static const char *__doc_falcor_OptixDenoiser_class_name = R"doc()doc";

static const char *__doc_falcor_OptixDenoiser_denoise =
R"doc(Denoise an image.

Parameter ``params``:
    Denoiser parameters.

Parameter ``guide_layer``:
    Guide layer containing auxiliary data (leave empty if not used).

Parameter ``layers``:
    List of layers to denoise.

Parameter ``cuda_stream``:
    Optional CUDA stream to use for denoising (default: null stream).)doc";

static const char *__doc_falcor_OptixDenoiser_device = R"doc(Device used by this denoiser.)doc";

static const char *__doc_falcor_OptixDenoiser_m_device = R"doc()doc";

static const char *__doc_falcor_OptixDenoiser_m_impl = R"doc()doc";

static const char *__doc_falcor_PackChunkFlags = R"doc(Flags stored in each pack chunk header.)doc";

static const char *__doc_falcor_PackChunkFlags_compressed = R"doc(Reserved for future use; currently rejected by PackReader.)doc";

static const char *__doc_falcor_PackChunkFlags_container = R"doc(Chunk payload contains an internal PTOC followed by child chunks.)doc";

static const char *__doc_falcor_PackChunkFlags_none = R"doc(Plain data chunk.)doc";

static const char *__doc_falcor_PackOptions = R"doc()doc";

static const char *__doc_falcor_PackOptions_clamp = R"doc(Clamp out-of-range values to valid range.)doc";

static const char *__doc_falcor_PackOptions_nan_to_zero = R"doc(Convert NaNs to zero.)doc";

static const char *__doc_falcor_PackOptions_safe =
R"doc(Safe conversion that clamps out-of-range values and converts NaNs to
zero.)doc";

static const char *__doc_falcor_PackOptions_unsafe = R"doc(No special handling of out-of-range values or NaNs.)doc";

static const char *__doc_falcor_PackReader = R"doc(Reads and validates a pack file from disk or a borrowed memory span.)doc";

static const char *__doc_falcor_PackReader_ChunkIterator = R"doc(Input iterator over chunk views.)doc";

static const char *__doc_falcor_PackReader_ChunkIterator_2 = R"doc(Input iterator over chunk views.)doc";

static const char *__doc_falcor_PackReader_ChunkIterator_ChunkIterator = R"doc()doc";

static const char *__doc_falcor_PackReader_ChunkIterator_m_data = R"doc()doc";

static const char *__doc_falcor_PackReader_ChunkIterator_m_entries = R"doc()doc";

static const char *__doc_falcor_PackReader_ChunkIterator_m_index = R"doc()doc";

static const char *__doc_falcor_PackReader_ChunkIterator_operator_eq = R"doc(Compare iterator positions.)doc";

static const char *__doc_falcor_PackReader_ChunkIterator_operator_inc = R"doc(Advance to the next chunk.)doc";

static const char *__doc_falcor_PackReader_ChunkIterator_operator_inc_2 = R"doc(Advance to the next chunk and return the previous iterator.)doc";

static const char *__doc_falcor_PackReader_ChunkIterator_operator_mul = R"doc(Return the current chunk view.)doc";

static const char *__doc_falcor_PackReader_ChunkIterator_operator_ne = R"doc(Compare iterator positions.)doc";

static const char *__doc_falcor_PackReader_ChunkRange = R"doc(Iterable range of chunk views.)doc";

static const char *__doc_falcor_PackReader_ChunkRange_2 = R"doc(Iterable range of chunk views.)doc";

static const char *__doc_falcor_PackReader_ChunkRange_ChunkRange = R"doc()doc";

static const char *__doc_falcor_PackReader_ChunkRange_begin = R"doc(Return an iterator to the first chunk.)doc";

static const char *__doc_falcor_PackReader_ChunkRange_end = R"doc(Return the end iterator.)doc";

static const char *__doc_falcor_PackReader_ChunkRange_m_count = R"doc()doc";

static const char *__doc_falcor_PackReader_ChunkRange_m_data = R"doc()doc";

static const char *__doc_falcor_PackReader_ChunkRange_m_entries = R"doc()doc";

static const char *__doc_falcor_PackReader_ChunkView = R"doc(Lightweight view of a single chunk inside a PackReader.)doc";

static const char *__doc_falcor_PackReader_ChunkView_2 = R"doc(Lightweight view of a single chunk inside a PackReader.)doc";

static const char *__doc_falcor_PackReader_ChunkView_ChunkView = R"doc()doc";

static const char *__doc_falcor_PackReader_ChunkView_child_count =
R"doc(Return the number of direct child chunks. Throws if this is a data
chunk.)doc";

static const char *__doc_falcor_PackReader_ChunkView_children = R"doc(Iterate over direct child chunks. Throws if this is a data chunk.)doc";

static const char *__doc_falcor_PackReader_ChunkView_data = R"doc(Return the data payload bytes. Throws if this is a container chunk.)doc";

static const char *__doc_falcor_PackReader_ChunkView_find_child =
R"doc(Find the first direct child with the given tag. Throws if this is a
data chunk.)doc";

static const char *__doc_falcor_PackReader_ChunkView_flags = R"doc(Return the chunk flags.)doc";

static const char *__doc_falcor_PackReader_ChunkView_is_container = R"doc(Return true if this chunk contains child chunks.)doc";

static const char *__doc_falcor_PackReader_ChunkView_m_data = R"doc()doc";

static const char *__doc_falcor_PackReader_ChunkView_m_header = R"doc()doc";

static const char *__doc_falcor_PackReader_ChunkView_size = R"doc(Return the data payload size. Container chunks report zero.)doc";

static const char *__doc_falcor_PackReader_ChunkView_tag = R"doc(Return the chunk FourCC tag.)doc";

static const char *__doc_falcor_PackReader_ChunkView_version = R"doc(Return the chunk version.)doc";

static const char *__doc_falcor_PackReader_PackReader = R"doc(Open, memory-map, and validate a pack file from disk.)doc";

static const char *__doc_falcor_PackReader_PackReader_2 =
R"doc(Validate and read a pack file from a borrowed memory span. The span
must outlive the reader, and its data pointer must be aligned to
detail::PackChunkHeader.)doc";

static const char *__doc_falcor_PackReader_PackReader_3 = R"doc(Move construct a reader, transferring ownership of any memory mapping.)doc";

static const char *__doc_falcor_PackReader_PackReader_4 = R"doc()doc";

static const char *__doc_falcor_PackReader_chunk_count = R"doc(Return the number of top-level chunks.)doc";

static const char *__doc_falcor_PackReader_chunks = R"doc(Iterate over top-level chunks.)doc";

static const char *__doc_falcor_PackReader_find_chunk = R"doc(Find the first top-level chunk with the given tag.)doc";

static const char *__doc_falcor_PackReader_m_data = R"doc()doc";

static const char *__doc_falcor_PackReader_m_mmap = R"doc()doc";

static const char *__doc_falcor_PackReader_operator_assign = R"doc(Move assign a reader, transferring ownership of any memory mapping.)doc";

static const char *__doc_falcor_PackReader_operator_assign_2 = R"doc()doc";

static const char *__doc_falcor_PackReader_tag = R"doc(Return the root chunk FourCC tag.)doc";

static const char *__doc_falcor_PackReader_validate = R"doc()doc";

static const char *__doc_falcor_PackReader_version = R"doc(Return the root chunk version.)doc";

static const char *__doc_falcor_PackWriter =
R"doc(Incrementally builds a pack file chunk tree and serializes it to a
file or stream.)doc";

static const char *__doc_falcor_PackWriter_ChunkNode = R"doc()doc";

static const char *__doc_falcor_PackWriter_ChunkNode_absolute_offset = R"doc()doc";

static const char *__doc_falcor_PackWriter_ChunkNode_children = R"doc()doc";

static const char *__doc_falcor_PackWriter_ChunkNode_data_offset = R"doc()doc";

static const char *__doc_falcor_PackWriter_ChunkNode_data_size = R"doc()doc";

static const char *__doc_falcor_PackWriter_ChunkNode_flags = R"doc()doc";

static const char *__doc_falcor_PackWriter_ChunkNode_payload_size = R"doc(< Cached serialized payload size (set by compute_layout).)doc";

static const char *__doc_falcor_PackWriter_ChunkNode_tag = R"doc()doc";

static const char *__doc_falcor_PackWriter_ChunkNode_version = R"doc()doc";

static const char *__doc_falcor_PackWriter_PackWriter = R"doc(Create a writer with the given root chunk tag and version.)doc";

static const char *__doc_falcor_PackWriter_PackWriter_2 = R"doc()doc";

static const char *__doc_falcor_PackWriter_PackWriter_3 = R"doc()doc";

static const char *__doc_falcor_PackWriter_begin_chunk_internal = R"doc()doc";

static const char *__doc_falcor_PackWriter_begin_container_chunk = R"doc(Begin a container child chunk under the current open container chunk.)doc";

static const char *__doc_falcor_PackWriter_begin_data_chunk = R"doc(Begin a data child chunk under the current open container chunk.)doc";

static const char *__doc_falcor_PackWriter_compute_layout = R"doc()doc";

static const char *__doc_falcor_PackWriter_end_chunk = R"doc(End the current chunk and return to its parent.)doc";

static const char *__doc_falcor_PackWriter_finish =
R"doc(Finalize the chunk tree. All chunks must be closed before
finalization.)doc";

static const char *__doc_falcor_PackWriter_m_buffer = R"doc()doc";

static const char *__doc_falcor_PackWriter_m_finalized = R"doc()doc";

static const char *__doc_falcor_PackWriter_m_root = R"doc()doc";

static const char *__doc_falcor_PackWriter_m_stack = R"doc()doc";

static const char *__doc_falcor_PackWriter_operator_assign = R"doc()doc";

static const char *__doc_falcor_PackWriter_operator_assign_2 = R"doc()doc";

static const char *__doc_falcor_PackWriter_ptoc_payload_size = R"doc()doc";

static const char *__doc_falcor_PackWriter_write = R"doc(Append raw payload bytes to the current chunk.)doc";

static const char *__doc_falcor_PackWriter_write_2 = R"doc(Append raw payload bytes to the current chunk.)doc";

static const char *__doc_falcor_PackWriter_write_node = R"doc()doc";

static const char *__doc_falcor_PackWriter_write_to_file = R"doc(Write the finalized pack to a file.)doc";

static const char *__doc_falcor_PackWriter_write_to_stream =
R"doc(Write the pack to an arbitrary stream. The stream must be writable,
empty, and positioned at the beginning (offset 0), because PTOC
entries store absolute byte offsets from the stream start.)doc";

static const char *__doc_falcor_ParallelReduction =
R"doc(Utility for computing parallel reduction (sum, min, max) on buffers
and textures. Results are written to user-specified BufferOffsetPair
destinations (GPU-side, not host readback). At least one of
sum/min/max must be provided.)doc";

static const char *__doc_falcor_ParallelReduction_ParallelReduction = R"doc(Constructor.)doc";

static const char *__doc_falcor_ParallelReduction_PipelineKey = R"doc(Key for pipeline cache: (data type or format config, op mask).)doc";

static const char *__doc_falcor_ParallelReduction_PipelineKey_is_texture = R"doc()doc";

static const char *__doc_falcor_ParallelReduction_PipelineKey_op_mask = R"doc()doc";

static const char *__doc_falcor_ParallelReduction_PipelineKey_operator_le = R"doc()doc";

static const char *__doc_falcor_ParallelReduction_PipelineKey_type_key =
R"doc(DataType enum value for buffers, or encoded format config for
textures.)doc";

static const char *__doc_falcor_ParallelReduction_Pipelines = R"doc()doc";

static const char *__doc_falcor_ParallelReduction_Pipelines_final_reduce_pipeline = R"doc()doc";

static const char *__doc_falcor_ParallelReduction_Pipelines_reduce_pipeline = R"doc()doc";

static const char *__doc_falcor_ParallelReduction_class_name = R"doc()doc";

static const char *__doc_falcor_ParallelReduction_execute =
R"doc(Computes parallel reduction over a buffer of scalar elements. Supports
int32, uint32, int64, uint64, float32 types. element_count must be
greater than zero. Each provided output writes one scalar of the
requested type at the given offset. int32, uint32, and float32 outputs
reserve 4 bytes and require 4-byte aligned offsets. int64 and uint64
outputs reserve 8 bytes and require 8-byte aligned offsets. At least
one of sum/min/max must be provided.

Parameter ``command_encoder``:
    Command encoder.

Parameter ``type``:
    Data type of the elements.

Parameter ``data``:
    Data buffer and byte offset to reduce. The offset must be aligned
    to the element size.

Parameter ``element_count``:
    Number of elements to reduce.

Parameter ``sum``:
    (Optional) Buffer and offset to which the sum is written.

Parameter ``min``:
    (Optional) Buffer and offset to which the min is written.

Parameter ``max``:
    (Optional) Buffer and offset to which the max is written.)doc";

static const char *__doc_falcor_ParallelReduction_execute_2 =
R"doc(Computes parallel reduction over a 2D texture. Results are vec4
(float4 for float/unorm/snorm, int4 for sint, uint4 for uint). Each
provided output writes one 16-byte vec4 and requires a 4-byte aligned
offset. At least one of sum/min/max must be provided.

Parameter ``command_encoder``:
    Command encoder.

Parameter ``input``:
    Input non-empty 2D texture (non-owning).

Parameter ``sum``:
    (Optional) Buffer and offset to which the sum is written (16
    bytes).

Parameter ``min``:
    (Optional) Buffer and offset to which the min is written (16
    bytes).

Parameter ``max``:
    (Optional) Buffer and offset to which the max is written (16
    bytes).)doc";

static const char *__doc_falcor_ParallelReduction_get_pipelines = R"doc()doc";

static const char *__doc_falcor_ParallelReduction_m_device = R"doc()doc";

static const char *__doc_falcor_ParallelReduction_m_intermediate =
R"doc(Intermediate ping-pong buffers (stores ReduceState/TexReduceState
elements).)doc";

static const char *__doc_falcor_ParallelReduction_m_pipelines = R"doc()doc";

static const char *__doc_falcor_PointLight = R"doc()doc";

static const char *__doc_falcor_PointLight_PointLight = R"doc()doc";

static const char *__doc_falcor_PointLight_class_descriptor = R"doc()doc";

static const char *__doc_falcor_PointLight_class_name = R"doc()doc";

static const char *__doc_falcor_PointLight_intensity = R"doc()doc";

static const char *__doc_falcor_PointLight_m_intensity = R"doc()doc";

static const char *__doc_falcor_PointLight_reflect = R"doc(Reflect this class.)doc";

static const char *__doc_falcor_PointLight_reflected_class_name = R"doc()doc";

static const char *__doc_falcor_PointLight_set_intensity = R"doc()doc";

static const char *__doc_falcor_PointLight_static_class_descriptor = R"doc()doc";

static const char *__doc_falcor_PointLight_write_to_cursor = R"doc()doc";

static const char *__doc_falcor_PointLight_write_to_cursor_2 = R"doc()doc";

static const char *__doc_falcor_PointLight_write_to_cursor_impl = R"doc()doc";

static const char *__doc_falcor_PrefixSum =
R"doc(Utility for computing prefix sum in parallel. The prefix sum is
computed in place using exclusive scan. Each new element is y[i] =
x[0] + ... + x[i-1], for i=1..N and y[0] = 0.)doc";

static const char *__doc_falcor_PrefixSum_Pipelines = R"doc()doc";

static const char *__doc_falcor_PrefixSum_Pipelines_finalize_pipeline = R"doc()doc";

static const char *__doc_falcor_PrefixSum_Pipelines_group_scan_pipeline = R"doc()doc";

static const char *__doc_falcor_PrefixSum_PrefixSum = R"doc(Constructor.)doc";

static const char *__doc_falcor_PrefixSum_class_name = R"doc()doc";

static const char *__doc_falcor_PrefixSum_execute =
R"doc(Computes the parallel prefix sum over an array of elements. This
supports int32, uint32, int64, uint64, float32 types.

Parameter ``command_encoder``:
    Command encoder.

Parameter ``type``:
    Data type of the elements.

Parameter ``data``:
    Data buffer and offset to compute prefix sum over.

Parameter ``element_count``:
    Number of elements to compute prefix sum over.

Parameter ``total_sum``:
    (Optional) Buffer and offset to which the total sum is copied.)doc";

static const char *__doc_falcor_PrefixSum_get_pipelines = R"doc()doc";

static const char *__doc_falcor_PrefixSum_m_device = R"doc()doc";

static const char *__doc_falcor_PrefixSum_m_pipelines = R"doc()doc";

static const char *__doc_falcor_PrefixSum_m_prefix_group_sums = R"doc(Temporary buffer for prefix sum computation.)doc";

static const char *__doc_falcor_PrefixSum_m_prev_total_sum = R"doc(Temporary buffer for prev total sum of an iteration.)doc";

static const char *__doc_falcor_PrefixSum_m_total_sum = R"doc(Temporary buffer for total sum of an iteration.)doc";

static const char *__doc_falcor_Properties =
R"doc(Associative container for passing typed configuration parameters.

Properties maps string keys to values of various types: booleans,
integers, floats, vectors, strings, Object references, and type-erased
Any values.

Key features: - O(1) insertion and lookup by property name. -
Traversal of properties in the original insertion order. - Type-safe
storage with automatic widening (e.g. int -> int64_t, float ->
double). - Range-checked retrieval for integer types. - Type-safe
dynamic_cast for Object-derived property values.

Deleting properties leaves tombstone entries that are skipped during
iteration but occupy memory until the container is cleared.)doc";

static const char *__doc_falcor_Properties_2 =
R"doc(Associative container for passing typed configuration parameters.

Properties maps string keys to values of various types: booleans,
integers, floats, vectors, strings, Object references, and type-erased
Any values.

Key features: - O(1) insertion and lookup by property name. -
Traversal of properties in the original insertion order. - Type-safe
storage with automatic widening (e.g. int -> int64_t, float ->
double). - Range-checked retrieval for integer types. - Type-safe
dynamic_cast for Object-derived property values.

Deleting properties leaves tombstone entries that are skipped during
iteration but occupy memory until the container is cleared.)doc";

static const char *__doc_falcor_Properties_3 =
R"doc(Associative container for passing typed configuration parameters.

Properties maps string keys to values of various types: booleans,
integers, floats, vectors, strings, Object references, and type-erased
Any values.

Key features: - O(1) insertion and lookup by property name. -
Traversal of properties in the original insertion order. - Type-safe
storage with automatic widening (e.g. int -> int64_t, float ->
double). - Range-checked retrieval for integer types. - Type-safe
dynamic_cast for Object-derived property values.

Deleting properties leaves tombstone entries that are skipped during
iteration but occupy memory until the container is cleared.)doc";

static const char *__doc_falcor_Properties_Impl = R"doc()doc";

static const char *__doc_falcor_Properties_Properties = R"doc(Default constructor.)doc";

static const char *__doc_falcor_Properties_Properties_2 = R"doc(Copy constructor.)doc";

static const char *__doc_falcor_Properties_Properties_3 = R"doc(Move constructor.)doc";

static const char *__doc_falcor_Properties_append =
R"doc(Append a new property.

Parameter ``name``:
    Property name.

Parameter ``throw_if_exists``:
    Throw if a property with the same name exists.

Returns:
    The index of the property.)doc";

static const char *__doc_falcor_Properties_as_string = R"doc(Return a human-readable string representation of a property value.)doc";

static const char *__doc_falcor_Properties_begin = R"doc(Iterator to the first property.)doc";

static const char *__doc_falcor_Properties_check_type = R"doc(Returns PropertyType for the corresponding type T.)doc";

static const char *__doc_falcor_Properties_check_type_impl = R"doc(Convert a storage type to a PropertyType enum value.)doc";

static const char *__doc_falcor_Properties_clear = R"doc(Clear all properties.)doc";

static const char *__doc_falcor_Properties_copy_property =
R"doc(Copy a property pointed to by an iterator of another Properties
object.)doc";

static const char *__doc_falcor_Properties_empty = R"doc(True if the container is empty.)doc";

static const char *__doc_falcor_Properties_end = R"doc(Past-the-end iterator.)doc";

static const char *__doc_falcor_Properties_entry_name =
R"doc(Get the name of a property.

Parameter ``index``:
    Index of the property.

Returns:
    The name of the property.)doc";

static const char *__doc_falcor_Properties_find = R"doc(Returns iterator to a property of the given name, or end().)doc";

static const char *__doc_falcor_Properties_get =
R"doc(Get a property value. Throws if the property does not exist or has an
incompatible type.

Parameter ``name``:
    Property name.

Returns:
    The value.)doc";

static const char *__doc_falcor_Properties_get_2 =
R"doc(Get a property value, returning a default value if the property does
not exist. Throws on type mismatch if the property exists but has an
incompatible type.

Parameter ``name``:
    Property name.

Returns:
    The value or the default value if the property does not exist.)doc";

static const char *__doc_falcor_Properties_get_3 =
R"doc(Retrieve a property value by index.

Parameter ``index``:
    Index of the property.

Returns:
    The value of the property.)doc";

static const char *__doc_falcor_Properties_get_any =
R"doc(Retrieve a value previously stored using set_any(). Throws if the
property does not exist or cannot be cast to the requested type.

Parameter ``name``:
    Property name.

Returns:
    The value.)doc";

static const char *__doc_falcor_Properties_get_impl =
R"doc(Low-level access to a property's stored value by index.

Parameter ``index``:
    Index of the property.

Parameter ``throw_if_incompatible``:
    If true, throw on type mismatch, otherwise return nullptr.

Returns:
    A pointer to the stored value, or nullptr on type mismatch when
    not throwing.)doc";

static const char *__doc_falcor_Properties_get_list =
R"doc(Get a list property as a typed vector. Throws if the property does not
exist or has an incompatible element type.

Parameter ``name``:
    Property name.

Returns:
    The vector.)doc";

static const char *__doc_falcor_Properties_get_list_optional =
R"doc(Get a list property as a typed vector, returning std::nullopt if the
property does not exist or has an incompatible type.

Parameter ``name``:
    Property name.

Parameter ``throw_if_incompatible``:
    If true, throw on type/element mismatch.

Returns:
    The vector or std::nullopt.)doc";

static const char *__doc_falcor_Properties_get_optional =
R"doc(Get a property value, returning std::nullopt if the property does not
exist or has an incompatible type.

Parameter ``name``:
    Property name.

Parameter ``throw_if_incompatible``:
    If true, throw on type mismatch.

Returns:
    The value or std::nullopt if the property does not exist or has an
    incompatible type.)doc";

static const char *__doc_falcor_Properties_has_property = R"doc(Check if a property exists.)doc";

static const char *__doc_falcor_Properties_hash =
R"doc(Compute a hash of the object. The hash is order-independent (sorted by
key).)doc";

static const char *__doc_falcor_Properties_is_equal_property =
R"doc(Returns true if either both this and other Properties miss the
property of a given name, or if the property is present and equal in
both.)doc";

static const char *__doc_falcor_Properties_iterator =
R"doc(Forward iterator over the entries of a Properties container.

The iterator dereferences to itself, exposing name(), type(), and
get<T>() accessors on the current property. Tombstone entries (deleted
properties) are automatically skipped.

It is legal to mutate the container while iterating (e.g. removing
entries), but newly added entries may or may not be visited.)doc";

static const char *__doc_falcor_Properties_key_index =
R"doc(Get the index of a property by name.

Parameter ``name``:
    Property name.

Returns:
    The index of the property, or `npos` if not found.)doc";

static const char *__doc_falcor_Properties_keys = R"doc(Get all property keys (in insertion order).)doc";

static const char *__doc_falcor_Properties_m_impl = R"doc()doc";

static const char *__doc_falcor_Properties_merge =
R"doc(Merge properties from another Properties object. Existing properties
will be overridden.)doc";

static const char *__doc_falcor_Properties_operator_assign = R"doc(Assignment operator.)doc";

static const char *__doc_falcor_Properties_operator_assign_2 = R"doc(Move assignment operator.)doc";

static const char *__doc_falcor_Properties_operator_eq = R"doc(Comparison operator.)doc";

static const char *__doc_falcor_Properties_operator_ne = R"doc()doc";

static const char *__doc_falcor_Properties_remove_property =
R"doc(Remove a property. Returns true if the property was found and removed,
false otherwise.)doc";

static const char *__doc_falcor_Properties_rename_property =
R"doc(Rename a property. Returns true if the property was renamed, false if
old_name doesn't exist or new_name already exists.)doc";

static const char *__doc_falcor_Properties_require_key_index =
R"doc(Get the index of a property by name or throw an exception if the
property was not found.

Parameter ``name``:
    Property name.

Returns:
    The index of the property.)doc";

static const char *__doc_falcor_Properties_set =
R"doc(Set a property value.

Parameter ``name``:
    Property name.

Parameter ``value``:
    Value to set.

Parameter ``throw_if_exist``:
    Throw an exception if a property with the same name already
    exists.)doc";

static const char *__doc_falcor_Properties_set_2 = R"doc(@overload)doc";

static const char *__doc_falcor_Properties_set_any =
R"doc(Store an arbitrary value wrapped in a type-erased Any container.

This enables storing types that are not natively supported by the
Properties system. The value can be retrieved later using
get_any<T>().

See also:
    Any

Parameter ``name``:
    Property name.

Parameter ``value``:
    Value to set.)doc";

static const char *__doc_falcor_Properties_set_any_2 = R"doc(@overload)doc";

static const char *__doc_falcor_Properties_set_impl =
R"doc(Set a property value.

Parameter ``index``:
    Index of the property.

Parameter ``value``:
    Property value.)doc";

static const char *__doc_falcor_Properties_set_impl_2 =
R"doc(Set a property value.

Parameter ``index``:
    Index of the property.

Parameter ``value``:
    Property value.)doc";

static const char *__doc_falcor_Properties_set_list =
R"doc(Set a list property from a typed vector.

Parameter ``name``:
    Property name.

Parameter ``data``:
    The vector to store.

Parameter ``throw_if_exists``:
    Throw if a property with the same name already exists.)doc";

static const char *__doc_falcor_Properties_size = R"doc(Number of properties in the container.)doc";

static const char *__doc_falcor_Properties_throw_any_type_error = R"doc(Throw an exception when get_any() encounters an incompatible type.)doc";

static const char *__doc_falcor_Properties_throw_object_type_error = R"doc(Throw an exception when an object has incompatible type.)doc";

static const char *__doc_falcor_Properties_to_string = R"doc(String representation of all properties.)doc";

static const char *__doc_falcor_Properties_try_get =
R"doc(Try to retrieve a property value without implicit conversions. Returns
a pointer to the stored value, or nullptr if the property doesn't
exist or has a different type.

Parameter ``name``:
    Property name.

Returns:
    The value or nullptr if the property does not exist.)doc";

static const char *__doc_falcor_Properties_try_get_2 =
R"doc(Try to get a property value by index without implicit conversions.

Parameter ``index``:
    Index of the property.

Returns:
    A pointer to the stored value, or nullptr on type mismatch.)doc";

static const char *__doc_falcor_Properties_try_get_list =
R"doc(Try to retrieve a list property without implicit conversions. Returns
a pointer to the stored vector, or nullptr if the property doesn't
exist or has a different element type.

Parameter ``name``:
    Property name.

Returns:
    Pointer to the stored vector, or nullptr.)doc";

static const char *__doc_falcor_Properties_type = R"doc(Get the type of a stored property.)doc";

static const char *__doc_falcor_PropertyType =
R"doc(Enumeration of property value types. The order of entries must exactly
match the order of types in the PropertyValue variant defined in
properties.cpp.)doc";

static const char *__doc_falcor_PropertyType_any = R"doc(< Type-erased storage for arbitrary data (see Any).)doc";

static const char *__doc_falcor_PropertyType_bool = R"doc(< Boolean value.)doc";

static const char *__doc_falcor_PropertyType_count = R"doc(< Number of property types (not a valid type).)doc";

static const char *__doc_falcor_PropertyType_enum =
R"doc(< Enum value with preserved type identity (see
detail::PropertyEnumValue).)doc";

static const char *__doc_falcor_PropertyType_float = R"doc(< Double-precision float (float and double are both stored as double).)doc";

static const char *__doc_falcor_PropertyType_float2 = R"doc(< 2-component float vector.)doc";

static const char *__doc_falcor_PropertyType_float3 = R"doc(< 3-component float vector.)doc";

static const char *__doc_falcor_PropertyType_float3x3 = R"doc(< 3x3 float matrix.)doc";

static const char *__doc_falcor_PropertyType_float4 = R"doc(< 4-component float vector.)doc";

static const char *__doc_falcor_PropertyType_float4x4 = R"doc(< 4x4 float matrix.)doc";

static const char *__doc_falcor_PropertyType_info = R"doc()doc";

static const char *__doc_falcor_PropertyType_int = R"doc(< 64-bit signed integer (all integer types are widened to this).)doc";

static const char *__doc_falcor_PropertyType_int2 = R"doc(< 2-component signed integer vector.)doc";

static const char *__doc_falcor_PropertyType_int3 = R"doc(< 3-component signed integer vector.)doc";

static const char *__doc_falcor_PropertyType_int4 = R"doc(< 4-component signed integer vector.)doc";

static const char *__doc_falcor_PropertyType_invalid = R"doc(< Unknown/deleted property (used for tombstones).)doc";

static const char *__doc_falcor_PropertyType_list = R"doc(< Homogeneous typed list (see detail::PropertyList).)doc";

static const char *__doc_falcor_PropertyType_object = R"doc(< Reference-counted Object (ref<Object>).)doc";

static const char *__doc_falcor_PropertyType_string = R"doc(< String value (std::string). Also used for filesystem paths.)doc";

static const char *__doc_falcor_PropertyType_uint2 = R"doc(< 2-component unsigned integer vector.)doc";

static const char *__doc_falcor_PropertyType_uint3 = R"doc(< 3-component unsigned integer vector.)doc";

static const char *__doc_falcor_PropertyType_uint4 = R"doc(< 4-component unsigned integer vector.)doc";

static const char *__doc_falcor_PythonContext =
R"doc(An isolated Python execution context with its own globals dict.

Created via PythonInterpreter::create_context(). Each context has a
fresh globals dict seeded only with __builtins__, providing full
isolation between contexts. State accumulates within a context across
calls (e.g. imports and variables persist).

@note All public methods automatically acquire/release the GIL, making
them safe to call from any thread, including from Python (where the
GIL is already held).

@note Each PythonContext instance should only be used from a single
thread at a time. While individual operations are GIL-safe, concurrent
calls on the same context may interleave and cause logical races on
shared state in the globals dict.

.. warning::
    PythonContext objects must not outlive program main(). Storing
    them in static/global variables risks destruction after the Python
    interpreter state becomes invalid.

Usage:

```
auto ctx = PythonInterpreter::get().create_context();
ctx.execute_string("x = 42");
ctx.execute_string("print(x)");  // works, same context
```)doc";

static const char *__doc_falcor_PythonContext_PythonContext = R"doc()doc";

static const char *__doc_falcor_PythonContext_PythonContext_2 = R"doc()doc";

static const char *__doc_falcor_PythonContext_PythonContext_3 = R"doc()doc";

static const char *__doc_falcor_PythonContext_execute_file =
R"doc(Execute a Python file in this context.

Parameter ``path``:
    Path to the .py file to execute.

Throws:
    PythonException on failure with the full Python traceback.)doc";

static const char *__doc_falcor_PythonContext_execute_string =
R"doc(Execute a string of Python code in this context.

Parameter ``code``:
    Python source code to execute.

Throws:
    PythonException on failure with the full Python traceback.)doc";

static const char *__doc_falcor_PythonContext_m_globals = R"doc(Opaque pointer to the PyObject* globals dict.)doc";

static const char *__doc_falcor_PythonContext_operator_assign = R"doc()doc";

static const char *__doc_falcor_PythonContext_operator_assign_2 = R"doc()doc";

static const char *__doc_falcor_PythonException =
R"doc(Exception thrown when Python code execution fails. Contains the full
Python stack trace in the message.)doc";

static const char *__doc_falcor_PythonException_PythonException = R"doc()doc";

static const char *__doc_falcor_PythonInterpreter =
R"doc(Singleton interface for the process-global Python interpreter.

CPython is inherently process-global, so this class exposes it as a
singleton. The interpreter is lazily initialized on the first call to
get().

Usage:

```
auto& py = PythonInterpreter::get();

auto ctx = py.create_context();
try {
    ctx.execute_string("print('hello')");
} catch (const PythonException& e) {
    log_error("{}", e.what());
}
```)doc";

static const char *__doc_falcor_PythonInterpreter_2 =
R"doc(Singleton interface for the process-global Python interpreter.

CPython is inherently process-global, so this class exposes it as a
singleton. The interpreter is lazily initialized on the first call to
get().

Usage:

```
auto& py = PythonInterpreter::get();

auto ctx = py.create_context();
try {
    ctx.execute_string("print('hello')");
} catch (const PythonException& e) {
    log_error("{}", e.what());
}
```)doc";

static const char *__doc_falcor_PythonInterpreter_PythonInterpreter = R"doc()doc";

static const char *__doc_falcor_PythonInterpreter_PythonInterpreter_2 = R"doc()doc";

static const char *__doc_falcor_PythonInterpreter_PythonInterpreter_3 = R"doc()doc";

static const char *__doc_falcor_PythonInterpreter_capture_python_error =
R"doc(Capture the current Python exception as a string and clear it. Returns
an empty string if no exception is pending. @pre Caller must hold the
GIL.)doc";

static const char *__doc_falcor_PythonInterpreter_create_context =
R"doc(Create a new isolated execution context. The context has a fresh
globals dict seeded only with __builtins__.)doc";

static const char *__doc_falcor_PythonInterpreter_get =
R"doc(Get the singleton instance, lazily initializing the interpreter.

Throws:
    PythonException if initialization fails.)doc";

static const char *__doc_falcor_PythonInterpreter_initialize =
R"doc(Initialize the Python interpreter.

Throws:
    PythonException if initialization fails.)doc";

static const char *__doc_falcor_PythonInterpreter_m_owns_python = R"doc(Whether this instance initialized the interpreter.)doc";

static const char *__doc_falcor_PythonInterpreter_operator_assign = R"doc()doc";

static const char *__doc_falcor_PythonInterpreter_operator_assign_2 = R"doc()doc";

static const char *__doc_falcor_ReflectedObject =
R"doc(Base class for objects with reflection-based property serialization.

Serialization is driven by the TypeRegistry: properties() and
set_properties() look up the ClassDescriptor for the concrete type and
delegate to ClassDescriptor::write_to_properties /
read_from_properties, which walk the inheritance chain automatically.
Dynamic per-instance properties are handled separately via the
dynamic_properties() accessor.)doc";

static const char *__doc_falcor_ReflectedObject_class_descriptor = R"doc()doc";

static const char *__doc_falcor_ReflectedObject_class_name = R"doc()doc";

static const char *__doc_falcor_ReflectedObject_dynamic_properties =
R"doc(Access per-instance dynamic properties, or nullptr if this object has
none. Objects that own dynamic properties (e.g. MDL materials)
override this.)doc";

static const char *__doc_falcor_ReflectedObject_dynamic_properties_2 = R"doc()doc";

static const char *__doc_falcor_ReflectedObject_find_property =
R"doc(Find a property descriptor by name. Searches static properties (via
ClassDescriptor) first, then dynamic properties. Returns nullptr if no
property with the given name exists.)doc";

static const char *__doc_falcor_ReflectedObject_get_property =
R"doc(Type-erased get: returns the current value of the named property
wrapped in Any. Throws if the property does not exist.)doc";

static const char *__doc_falcor_ReflectedObject_get_property_2 =
R"doc(Typed get: returns the current value of the named property. Throws if
the property does not exist or holds a different type.)doc";

static const char *__doc_falcor_ReflectedObject_has_property = R"doc(Check if a property with the given name exists.)doc";

static const char *__doc_falcor_ReflectedObject_properties = R"doc()doc";

static const char *__doc_falcor_ReflectedObject_set_properties = R"doc()doc";

static const char *__doc_falcor_ReflectedObject_set_property =
R"doc(Type-erased set: sets the named property from an Any value. Throws if
the property does not exist, is read-only, or if the Any holds the
wrong type.)doc";

static const char *__doc_falcor_ReflectedObject_set_property_2 =
R"doc(Typed set: sets the named property value directly. Throws if the
property does not exist, is read-only, or holds a different type.)doc";

static const char *__doc_falcor_RenderGeometryGroup = R"doc()doc";

static const char *__doc_falcor_RenderGeometryGroup_2 = R"doc()doc";

static const char *__doc_falcor_RenderGeometryGroupDesc = R"doc()doc";

static const char *__doc_falcor_RenderGeometryGroupDesc_geometries = R"doc()doc";

static const char *__doc_falcor_RenderGeometryGroup_blas = R"doc()doc";

static const char *__doc_falcor_RenderGeometryGroup_geometries = R"doc()doc";

static const char *__doc_falcor_RenderGeometryGroup_geometry_type = R"doc()doc";

static const char *__doc_falcor_RenderGeometryInstance = R"doc()doc";

static const char *__doc_falcor_RenderGeometryInstance_2 = R"doc()doc";

static const char *__doc_falcor_RenderGeometryInstanceDesc = R"doc()doc";

static const char *__doc_falcor_RenderGeometryInstanceDesc_geometry_group = R"doc()doc";

static const char *__doc_falcor_RenderGeometryInstanceDesc_material_ids = R"doc()doc";

static const char *__doc_falcor_RenderGeometryInstanceDesc_transform = R"doc()doc";

static const char *__doc_falcor_RenderGeometryInstanceDirtyFlags = R"doc()doc";

static const char *__doc_falcor_RenderGeometryInstanceDirtyFlags_created = R"doc()doc";

static const char *__doc_falcor_RenderGeometryInstanceDirtyFlags_destroyed = R"doc()doc";

static const char *__doc_falcor_RenderGeometryInstanceDirtyFlags_geometry_group = R"doc()doc";

static const char *__doc_falcor_RenderGeometryInstanceDirtyFlags_none = R"doc()doc";

static const char *__doc_falcor_RenderGeometryInstanceDirtyFlags_transform = R"doc()doc";

static const char *__doc_falcor_RenderGeometryInstance_geometry_group = R"doc()doc";

static const char *__doc_falcor_RenderGeometryInstance_geometry_instance_count = R"doc()doc";

static const char *__doc_falcor_RenderGeometryInstance_geometry_instance_id = R"doc()doc";

static const char *__doc_falcor_RenderGeometryInstance_material_ids = R"doc()doc";

static const char *__doc_falcor_RenderGeometryInstance_transform = R"doc()doc";

static const char *__doc_falcor_RenderHandle = R"doc()doc";

static const char *__doc_falcor_RenderHandle_2 = R"doc()doc";

static const char *__doc_falcor_RenderHandle_RenderHandle = R"doc()doc";

static const char *__doc_falcor_RenderHandle_RenderHandle_2 = R"doc()doc";

static const char *__doc_falcor_RenderHandle_RenderHandle_3 = R"doc()doc";

static const char *__doc_falcor_RenderHandle_index = R"doc()doc";

static const char *__doc_falcor_RenderHandle_is_valid = R"doc()doc";

static const char *__doc_falcor_RenderHandle_m_index = R"doc()doc";

static const char *__doc_falcor_RenderHandle_m_type = R"doc()doc";

static const char *__doc_falcor_RenderHandle_operator_assign = R"doc()doc";

static const char *__doc_falcor_RenderHandle_operator_assign_2 = R"doc()doc";

static const char *__doc_falcor_RenderHandle_operator_bool = R"doc()doc";

static const char *__doc_falcor_RenderHandle_operator_eq = R"doc()doc";

static const char *__doc_falcor_RenderHandle_operator_ne = R"doc()doc";

static const char *__doc_falcor_RenderHandle_type = R"doc()doc";

static const char *__doc_falcor_RenderLSSGeometries = R"doc()doc";

static const char *__doc_falcor_RenderLSSGeometries_RenderLSSGeometries = R"doc()doc";

static const char *__doc_falcor_RenderLSSGeometries_bind = R"doc()doc";

static const char *__doc_falcor_RenderLSSGeometries_create = R"doc()doc";

static const char *__doc_falcor_RenderLSSGeometries_destroy = R"doc()doc";

static const char *__doc_falcor_RenderLSSGeometries_lss_mode = R"doc()doc";

static const char *__doc_falcor_RenderLSSGeometries_m_aabb_buffer = R"doc()doc";

static const char *__doc_falcor_RenderLSSGeometries_m_compute_aabbs_kernel = R"doc()doc";

static const char *__doc_falcor_RenderLSSGeometries_m_descs = R"doc()doc";

static const char *__doc_falcor_RenderLSSGeometries_m_index_allocator = R"doc()doc";

static const char *__doc_falcor_RenderLSSGeometries_m_indices = R"doc()doc";

static const char *__doc_falcor_RenderLSSGeometries_m_lss_mode = R"doc()doc";

static const char *__doc_falcor_RenderLSSGeometries_m_pool = R"doc()doc";

static const char *__doc_falcor_RenderLSSGeometries_m_vertex_allocator = R"doc()doc";

static const char *__doc_falcor_RenderLSSGeometries_m_vertices = R"doc()doc";

static const char *__doc_falcor_RenderLSSGeometries_update = R"doc()doc";

static const char *__doc_falcor_RenderLSSGeometries_update_buffers = R"doc()doc";

static const char *__doc_falcor_RenderLSSGeometries_write_acceleration_structure_build_input = R"doc()doc";

static const char *__doc_falcor_RenderLSSGeometries_write_geometry_instance_desc = R"doc()doc";

static const char *__doc_falcor_RenderLSSGeometry = R"doc()doc";

static const char *__doc_falcor_RenderLSSGeometry_2 = R"doc()doc";

static const char *__doc_falcor_RenderLSSGeometryData = R"doc()doc";

static const char *__doc_falcor_RenderLSSGeometryData_indices = R"doc()doc";

static const char *__doc_falcor_RenderLSSGeometryData_vertices = R"doc()doc";

static const char *__doc_falcor_RenderLSSGeometryDesc = R"doc()doc";

static const char *__doc_falcor_RenderLSSGeometryDesc_index_count = R"doc()doc";

static const char *__doc_falcor_RenderLSSGeometryDesc_vertex_count = R"doc()doc";

static const char *__doc_falcor_RenderLSSGeometry_geometry_groups = R"doc(List of geometry groups referencing this geometry.)doc";

static const char *__doc_falcor_RenderLSSGeometry_index_allocation = R"doc()doc";

static const char *__doc_falcor_RenderLSSGeometry_index_count = R"doc()doc";

static const char *__doc_falcor_RenderLSSGeometry_index_offset = R"doc()doc";

static const char *__doc_falcor_RenderLSSGeometry_vertex_allocation = R"doc()doc";

static const char *__doc_falcor_RenderLSSGeometry_vertex_count = R"doc()doc";

static const char *__doc_falcor_RenderLSSGeometry_vertex_offset = R"doc()doc";

static const char *__doc_falcor_RenderLSSMode = R"doc(Mode used for LSS (Linear Swept Sphere) intersection.)doc";

static const char *__doc_falcor_RenderLSSMode_hardware = R"doc(Use native hardware LSS acceleration structure primitives.)doc";

static const char *__doc_falcor_RenderLSSMode_info = R"doc()doc";

static const char *__doc_falcor_RenderLSSMode_procedural =
R"doc(Use procedural AABB acceleration structure primitives with custom
intersection shader.)doc";

static const char *__doc_falcor_RenderObjectHandle = R"doc()doc";

static const char *__doc_falcor_RenderObjectHandle_2 = R"doc()doc";

static const char *__doc_falcor_RenderObjectHandle_DebugInfo = R"doc()doc";

static const char *__doc_falcor_RenderObjectHandle_DebugInfo_self = R"doc()doc";

static const char *__doc_falcor_RenderObjectHandle_RenderObjectHandle = R"doc()doc";

static const char *__doc_falcor_RenderObjectHandle_RenderObjectHandle_2 = R"doc()doc";

static const char *__doc_falcor_RenderObjectHandle_RenderObjectHandle_3 = R"doc()doc";

static const char *__doc_falcor_RenderObjectHandle_RenderObjectHandle_4 = R"doc()doc";

static const char *__doc_falcor_RenderObjectHandle_RenderObjectHandle_5 = R"doc()doc";

static const char *__doc_falcor_RenderObjectHandle_RenderObjectHandle_6 = R"doc()doc";

static const char *__doc_falcor_RenderObjectHandle_index = R"doc()doc";

static const char *__doc_falcor_RenderObjectHandle_is_valid = R"doc()doc";

static const char *__doc_falcor_RenderObjectHandle_m_debug_info = R"doc()doc";

static const char *__doc_falcor_RenderObjectHandle_m_index = R"doc()doc";

static const char *__doc_falcor_RenderObjectHandle_operator_RenderHandle = R"doc()doc";

static const char *__doc_falcor_RenderObjectHandle_operator_assign = R"doc()doc";

static const char *__doc_falcor_RenderObjectHandle_operator_assign_2 = R"doc()doc";

static const char *__doc_falcor_RenderObjectHandle_operator_bool = R"doc()doc";

static const char *__doc_falcor_RenderObjectHandle_operator_eq = R"doc()doc";

static const char *__doc_falcor_RenderObjectHandle_operator_ne = R"doc()doc";

static const char *__doc_falcor_RenderObjectHandle_type = R"doc()doc";

static const char *__doc_falcor_RenderObjectPool = R"doc()doc";

static const char *__doc_falcor_RenderObjectPool_Iterator = R"doc()doc";

static const char *__doc_falcor_RenderObjectPool_Iterator_Iterator = R"doc()doc";

static const char *__doc_falcor_RenderObjectPool_Iterator_m_allocated = R"doc()doc";

static const char *__doc_falcor_RenderObjectPool_Iterator_m_index = R"doc()doc";

static const char *__doc_falcor_RenderObjectPool_Iterator_m_pool = R"doc()doc";

static const char *__doc_falcor_RenderObjectPool_Iterator_operator_inc = R"doc()doc";

static const char *__doc_falcor_RenderObjectPool_Iterator_operator_mul = R"doc()doc";

static const char *__doc_falcor_RenderObjectPool_Iterator_operator_mul_2 = R"doc()doc";

static const char *__doc_falcor_RenderObjectPool_Iterator_operator_ne = R"doc()doc";

static const char *__doc_falcor_RenderObjectPool_Iterator_skip_free_indices = R"doc()doc";

static const char *__doc_falcor_RenderObjectPool_RenderObjectPool = R"doc()doc";

static const char *__doc_falcor_RenderObjectPool_allocate = R"doc()doc";

static const char *__doc_falcor_RenderObjectPool_begin = R"doc()doc";

static const char *__doc_falcor_RenderObjectPool_capacity = R"doc()doc";

static const char *__doc_falcor_RenderObjectPool_combined_dirty_flags = R"doc()doc";

static const char *__doc_falcor_RenderObjectPool_count = R"doc()doc";

static const char *__doc_falcor_RenderObjectPool_dirty_flags = R"doc()doc";

static const char *__doc_falcor_RenderObjectPool_end = R"doc()doc";

static const char *__doc_falcor_RenderObjectPool_free = R"doc()doc";

static const char *__doc_falcor_RenderObjectPool_is_dirty = R"doc()doc";

static const char *__doc_falcor_RenderObjectPool_is_dirty_2 = R"doc()doc";

static const char *__doc_falcor_RenderObjectPool_m_allocated = R"doc()doc";

static const char *__doc_falcor_RenderObjectPool_m_combined_dirty_flags = R"doc()doc";

static const char *__doc_falcor_RenderObjectPool_m_debug_infos = R"doc()doc";

static const char *__doc_falcor_RenderObjectPool_m_dirty_flags = R"doc()doc";

static const char *__doc_falcor_RenderObjectPool_m_free_indices = R"doc()doc";

static const char *__doc_falcor_RenderObjectPool_m_next_index = R"doc()doc";

static const char *__doc_falcor_RenderObjectPool_m_objects = R"doc()doc";

static const char *__doc_falcor_RenderObjectPool_mark_dirty = R"doc()doc";

static const char *__doc_falcor_RenderObjectPool_operator_array = R"doc()doc";

static const char *__doc_falcor_RenderObjectPool_operator_array_2 = R"doc()doc";

static const char *__doc_falcor_RenderObjectPool_reset_dirty_flags = R"doc()doc";

static const char *__doc_falcor_RenderObjectType = R"doc()doc";

static const char *__doc_falcor_RenderObjectType_geometry_group = R"doc()doc";

static const char *__doc_falcor_RenderObjectType_geometry_instance = R"doc()doc";

static const char *__doc_falcor_RenderObjectType_lss_geometry = R"doc()doc";

static const char *__doc_falcor_RenderObjectType_transform = R"doc()doc";

static const char *__doc_falcor_RenderObjectType_triangle_geometry = R"doc()doc";

static const char *__doc_falcor_RenderScene = R"doc()doc";

static const char *__doc_falcor_RenderScene_DirtyFlags = R"doc()doc";

static const char *__doc_falcor_RenderScene_DirtyFlags_geometry_group = R"doc()doc";

static const char *__doc_falcor_RenderScene_DirtyFlags_geometry_instance = R"doc()doc";

static const char *__doc_falcor_RenderScene_DirtyFlags_lss_geometry = R"doc()doc";

static const char *__doc_falcor_RenderScene_DirtyFlags_none = R"doc()doc";

static const char *__doc_falcor_RenderScene_DirtyFlags_transform = R"doc()doc";

static const char *__doc_falcor_RenderScene_DirtyFlags_triangle_geometry = R"doc()doc";

static const char *__doc_falcor_RenderScene_RenderScene = R"doc()doc";

static const char *__doc_falcor_RenderScene_UpdateContext = R"doc()doc";

static const char *__doc_falcor_RenderScene_bind_to_scene = R"doc()doc";

static const char *__doc_falcor_RenderScene_build_tlas = R"doc()doc";

static const char *__doc_falcor_RenderScene_class_name = R"doc()doc";

static const char *__doc_falcor_RenderScene_create = R"doc()doc";

static const char *__doc_falcor_RenderScene_create_geometry_group = R"doc()doc";

static const char *__doc_falcor_RenderScene_create_geometry_instance = R"doc()doc";

static const char *__doc_falcor_RenderScene_create_lss_geometry = R"doc()doc";

static const char *__doc_falcor_RenderScene_create_transform = R"doc()doc";

static const char *__doc_falcor_RenderScene_create_triangle_geometry = R"doc()doc";

static const char *__doc_falcor_RenderScene_destroy_geometry_group = R"doc()doc";

static const char *__doc_falcor_RenderScene_destroy_geometry_instance = R"doc()doc";

static const char *__doc_falcor_RenderScene_destroy_lss_geometry = R"doc()doc";

static const char *__doc_falcor_RenderScene_destroy_transform = R"doc()doc";

static const char *__doc_falcor_RenderScene_destroy_triangle_geometry = R"doc()doc";

static const char *__doc_falcor_RenderScene_geometry_instance_desc_count = R"doc()doc";

static const char *__doc_falcor_RenderScene_get_geometry_group = R"doc()doc";

static const char *__doc_falcor_RenderScene_get_geometry_instance = R"doc()doc";

static const char *__doc_falcor_RenderScene_get_lss_geometry = R"doc()doc";

static const char *__doc_falcor_RenderScene_get_lss_geometry_2 = R"doc()doc";

static const char *__doc_falcor_RenderScene_get_scratch_buffer = R"doc()doc";

static const char *__doc_falcor_RenderScene_get_transform = R"doc()doc";

static const char *__doc_falcor_RenderScene_get_transform_2 = R"doc()doc";

static const char *__doc_falcor_RenderScene_get_triangle_geometry = R"doc()doc";

static const char *__doc_falcor_RenderScene_get_triangle_geometry_2 = R"doc()doc";

static const char *__doc_falcor_RenderScene_has_geometry_type = R"doc()doc";

static const char *__doc_falcor_RenderScene_has_lss_geometry = R"doc()doc";

static const char *__doc_falcor_RenderScene_has_triangle_geometry = R"doc()doc";

static const char *__doc_falcor_RenderScene_lss_mode = R"doc()doc";

static const char *__doc_falcor_RenderScene_m_compute_inverse_kernel = R"doc()doc";

static const char *__doc_falcor_RenderScene_m_device = R"doc()doc";

static const char *__doc_falcor_RenderScene_m_geometry_group_pool = R"doc()doc";

static const char *__doc_falcor_RenderScene_m_geometry_instance_descs = R"doc()doc";

static const char *__doc_falcor_RenderScene_m_geometry_instance_pool = R"doc()doc";

static const char *__doc_falcor_RenderScene_m_instance_data = R"doc()doc";

static const char *__doc_falcor_RenderScene_m_lss_geometries = R"doc()doc";

static const char *__doc_falcor_RenderScene_m_object_from_world = R"doc()doc";

static const char *__doc_falcor_RenderScene_m_prev_transforms_need_update = R"doc()doc";

static const char *__doc_falcor_RenderScene_m_prev_world_from_object = R"doc()doc";

static const char *__doc_falcor_RenderScene_m_render_module = R"doc()doc";

static const char *__doc_falcor_RenderScene_m_scratch_buffer = R"doc()doc";

static const char *__doc_falcor_RenderScene_m_tlas = R"doc()doc";

static const char *__doc_falcor_RenderScene_m_tlas_instance_descs = R"doc()doc";

static const char *__doc_falcor_RenderScene_m_tlas_native_instance_descs = R"doc()doc";

static const char *__doc_falcor_RenderScene_m_tlas_native_instance_descs_buffer = R"doc()doc";

static const char *__doc_falcor_RenderScene_m_tlas_prefer_refit = R"doc(If true, TLAS updates use refit when only transforms have changed.)doc";

static const char *__doc_falcor_RenderScene_m_transform_pool = R"doc()doc";

static const char *__doc_falcor_RenderScene_m_triangle_geometries = R"doc()doc";

static const char *__doc_falcor_RenderScene_m_update_tlas_instance_desc_transforms_kernel = R"doc()doc";

static const char *__doc_falcor_RenderScene_m_world_from_object = R"doc()doc";

static const char *__doc_falcor_RenderScene_set_tlas_prefer_refit = R"doc()doc";

static const char *__doc_falcor_RenderScene_tlas_prefer_refit =
R"doc(If true, TLAS updates use refit when only transforms have changed
(default: true).)doc";

static const char *__doc_falcor_RenderScene_update = R"doc()doc";

static const char *__doc_falcor_RenderScene_update_2 = R"doc()doc";

static const char *__doc_falcor_RenderScene_update_blases = R"doc()doc";

static const char *__doc_falcor_RenderScene_update_lss_geometry = R"doc()doc";

static const char *__doc_falcor_RenderScene_update_transform = R"doc()doc";

static const char *__doc_falcor_RenderScene_update_triangle_geometry = R"doc()doc";

static const char *__doc_falcor_RenderTransform = R"doc()doc";

static const char *__doc_falcor_RenderTransform_2 = R"doc()doc";

static const char *__doc_falcor_RenderTransformDesc = R"doc()doc";

static const char *__doc_falcor_RenderTransformDesc_world_from_object = R"doc()doc";

static const char *__doc_falcor_RenderTransformDirtyFlags = R"doc()doc";

static const char *__doc_falcor_RenderTransformDirtyFlags_created = R"doc()doc";

static const char *__doc_falcor_RenderTransformDirtyFlags_none = R"doc()doc";

static const char *__doc_falcor_RenderTransformDirtyFlags_updated = R"doc()doc";

static const char *__doc_falcor_RenderTransform_geometry_instances = R"doc(List of geometry instances referencing this transform.)doc";

static const char *__doc_falcor_RenderTransform_world_from_object = R"doc()doc";

static const char *__doc_falcor_RenderTriangleGeometries = R"doc()doc";

static const char *__doc_falcor_RenderTriangleGeometries_RenderTriangleGeometries = R"doc()doc";

static const char *__doc_falcor_RenderTriangleGeometries_bind = R"doc()doc";

static const char *__doc_falcor_RenderTriangleGeometries_create = R"doc()doc";

static const char *__doc_falcor_RenderTriangleGeometries_destroy = R"doc()doc";

static const char *__doc_falcor_RenderTriangleGeometries_m_descs = R"doc()doc";

static const char *__doc_falcor_RenderTriangleGeometries_m_index_allocator = R"doc()doc";

static const char *__doc_falcor_RenderTriangleGeometries_m_indices = R"doc()doc";

static const char *__doc_falcor_RenderTriangleGeometries_m_pool = R"doc()doc";

static const char *__doc_falcor_RenderTriangleGeometries_m_vertex_allocator = R"doc()doc";

static const char *__doc_falcor_RenderTriangleGeometries_m_vertices = R"doc()doc";

static const char *__doc_falcor_RenderTriangleGeometries_update = R"doc()doc";

static const char *__doc_falcor_RenderTriangleGeometries_update_buffers = R"doc()doc";

static const char *__doc_falcor_RenderTriangleGeometries_write_acceleration_structure_build_input = R"doc()doc";

static const char *__doc_falcor_RenderTriangleGeometries_write_geometry_instance_desc = R"doc()doc";

static const char *__doc_falcor_RenderTriangleGeometry = R"doc()doc";

static const char *__doc_falcor_RenderTriangleGeometry_2 = R"doc()doc";

static const char *__doc_falcor_RenderTriangleGeometryData = R"doc()doc";

static const char *__doc_falcor_RenderTriangleGeometryData_indices = R"doc()doc";

static const char *__doc_falcor_RenderTriangleGeometryData_vertices = R"doc()doc";

static const char *__doc_falcor_RenderTriangleGeometryDesc = R"doc()doc";

static const char *__doc_falcor_RenderTriangleGeometryDesc_index_count = R"doc()doc";

static const char *__doc_falcor_RenderTriangleGeometryDesc_vertex_count = R"doc()doc";

static const char *__doc_falcor_RenderTriangleGeometry_geometry_groups = R"doc(List of geometry groups referencing this geometry.)doc";

static const char *__doc_falcor_RenderTriangleGeometry_index_allocation = R"doc()doc";

static const char *__doc_falcor_RenderTriangleGeometry_index_count = R"doc()doc";

static const char *__doc_falcor_RenderTriangleGeometry_index_offset = R"doc()doc";

static const char *__doc_falcor_RenderTriangleGeometry_vertex_allocation = R"doc()doc";

static const char *__doc_falcor_RenderTriangleGeometry_vertex_count = R"doc()doc";

static const char *__doc_falcor_RenderTriangleGeometry_vertex_offset = R"doc()doc";

static const char *__doc_falcor_Scene =
R"doc(Scene container.

The Scene is the top-level container for all renderable content and
serves as the central hub for managing scene objects and their
lifecycles. It coordinates between the high-level scene graph
representation and the low-level device resources needed for
rendering.

Scene objects are organized into four main categories: - Materials
(Material): Define surface appearance and shading properties. -
Geometries (Geometry): Define geometry data (meshes, implicits, etc.).
- Entities (Entity): Scene graph nodes with transforms that form a
hierarchy. - Components (Component): Attached to entities to add
functionality (e.g., GeometryInstance, Camera, Light).

The scene uses a deferred update model. When objects are created,
modified, or removed, changes are tracked via dirty flags. The
update() method must be called to process these changes, update device
resources, and prepare the scene for rendering.

The scene can be loaded from various file formats (glTF, USD, etc.)
via importers, or constructed programmatically using the create_*
methods.)doc";

static const char *__doc_falcor_Scene_2 = R"doc()doc";

static const char *__doc_falcor_SceneBindFlags = R"doc(Flags to determine which parts of the scene to bind to a shader.)doc";

static const char *__doc_falcor_SceneBindFlags_all = R"doc()doc";

static const char *__doc_falcor_SceneBindFlags_emissive_geometry = R"doc()doc";

static const char *__doc_falcor_SceneBindFlags_info = R"doc()doc";

static const char *__doc_falcor_SceneBindFlags_lights = R"doc()doc";

static const char *__doc_falcor_SceneBindFlags_materials = R"doc()doc";

static const char *__doc_falcor_SceneBindFlags_none = R"doc()doc";

static const char *__doc_falcor_SceneBindFlags_render_scene = R"doc()doc";

static const char *__doc_falcor_SceneGlobals = R"doc()doc";

static const char *__doc_falcor_SceneGlobals_2 = R"doc()doc";

static const char *__doc_falcor_SceneGlobals_bind =
R"doc(Bind this scene globals object to the shader.

Parameter ``globals``:
    The shader cursor pointing to the global parameters.)doc";

static const char *__doc_falcor_SceneGlobals_class_name = R"doc()doc";

static const char *__doc_falcor_SceneObject = R"doc(Base class for all scene objects.)doc";

static const char *__doc_falcor_SceneObjectCollection =
R"doc(Collection of scene objects of a specific type. Objects are owned by
the collection. Keeps a list of dirty flags as well as combined dirty
flags (or'd) for the entire collection. This allows to quickly check
for changes.)doc";

static const char *__doc_falcor_SceneObjectCollectionView = R"doc(View of a SceneObjectCollection.)doc";

static const char *__doc_falcor_SceneObjectCollectionView_IteratorImpl = R"doc(Iterator for iterating over objects in the collection view.)doc";

static const char *__doc_falcor_SceneObjectCollectionView_IteratorImpl_IteratorImpl = R"doc()doc";

static const char *__doc_falcor_SceneObjectCollectionView_IteratorImpl_m_index = R"doc()doc";

static const char *__doc_falcor_SceneObjectCollectionView_IteratorImpl_m_view = R"doc()doc";

static const char *__doc_falcor_SceneObjectCollectionView_IteratorImpl_operator_eq = R"doc()doc";

static const char *__doc_falcor_SceneObjectCollectionView_IteratorImpl_operator_inc = R"doc()doc";

static const char *__doc_falcor_SceneObjectCollectionView_IteratorImpl_operator_mul = R"doc()doc";

static const char *__doc_falcor_SceneObjectCollectionView_IteratorImpl_operator_ne = R"doc()doc";

static const char *__doc_falcor_SceneObjectCollectionView_SceneObjectCollectionView = R"doc()doc";

static const char *__doc_falcor_SceneObjectCollectionView_begin = R"doc(Get an iterator to the beginning of the collection view.)doc";

static const char *__doc_falcor_SceneObjectCollectionView_begin_2 = R"doc(Get a const iterator to the beginning of the collection view.)doc";

static const char *__doc_falcor_SceneObjectCollectionView_end = R"doc(Get an iterator to the end of the collection view.)doc";

static const char *__doc_falcor_SceneObjectCollectionView_end_2 = R"doc(Get a const iterator to the end of the collection view.)doc";

static const char *__doc_falcor_SceneObjectCollectionView_find =
R"doc(Find the first object with the given name.

Parameter ``name``:
    The name to search for.

Returns:
    Pointer to the first matching object, or nullptr if not found.)doc";

static const char *__doc_falcor_SceneObjectCollectionView_find_2 =
R"doc(Find the first object of the given type.

Template parameter ``T``:
    The type to search for (must be derived from TObject).

Returns:
    Pointer to the first matching object, or nullptr if not found.)doc";

static const char *__doc_falcor_SceneObjectCollectionView_find_3 =
R"doc(Find the first object of the given type with the given name.

Template parameter ``T``:
    The type to search for (must be derived from TObject).

Parameter ``name``:
    The name to search for.

Returns:
    Pointer to the first matching object, or nullptr if not found.)doc";

static const char *__doc_falcor_SceneObjectCollectionView_find_all =
R"doc(Find all objects with the given name.

Parameter ``name``:
    The name to search for.

Returns:
    Vector of pointers to all matching objects.)doc";

static const char *__doc_falcor_SceneObjectCollectionView_find_all_2 =
R"doc(Find all objects of the given type.

Template parameter ``T``:
    The type to search for (must be derived from TObject).

Returns:
    Vector of pointers to all matching objects.)doc";

static const char *__doc_falcor_SceneObjectCollectionView_find_all_3 =
R"doc(Find all objects of the given type with the given name.

Template parameter ``T``:
    The type to search for (must be derived from TObject).

Parameter ``name``:
    The name to search for.

Returns:
    Vector of pointers to all matching objects.)doc";

static const char *__doc_falcor_SceneObjectCollectionView_m_collection = R"doc()doc";

static const char *__doc_falcor_SceneObjectCollectionView_operator_array = R"doc(Access object by index (const).)doc";

static const char *__doc_falcor_SceneObjectCollectionView_operator_array_2 = R"doc(Access object by index (non-const).)doc";

static const char *__doc_falcor_SceneObjectCollectionView_size = R"doc(The size of the collection.)doc";

static const char *__doc_falcor_SceneObjectCollection_SceneObjectCollection =
R"doc(Constructor.

Parameter ``scene``:
    Scene the collection belongs to.)doc";

static const char *__doc_falcor_SceneObjectCollection_SceneObjectCollection_2 = R"doc()doc";

static const char *__doc_falcor_SceneObjectCollection_SceneObjectCollection_3 = R"doc()doc";

static const char *__doc_falcor_SceneObjectCollection_add =
R"doc(Add an object to the collection.

Parameter ``object``:
    The object to add.)doc";

static const char *__doc_falcor_SceneObjectCollection_clear = R"doc(Clear the collection.)doc";

static const char *__doc_falcor_SceneObjectCollection_combined_dirty_flags =
R"doc(The combined dirty flags (or'd flags of all objects since last call to
reset_dirty_flags()).)doc";

static const char *__doc_falcor_SceneObjectCollection_dirty_flags = R"doc(List of dirty flags for each object in the collection.)doc";

static const char *__doc_falcor_SceneObjectCollection_empty = R"doc(Check if the collection is empty.)doc";

static const char *__doc_falcor_SceneObjectCollection_for_each = R"doc()doc";

static const char *__doc_falcor_SceneObjectCollection_for_each_dirty = R"doc()doc";

static const char *__doc_falcor_SceneObjectCollection_has_dirty =
R"doc(Return true if any object in the collection is marked with any of the
supplied flags.)doc";

static const char *__doc_falcor_SceneObjectCollection_is_dirty = R"doc(Return true if any object is marked dirty.)doc";

static const char *__doc_falcor_SceneObjectCollection_m_combined_dirty_flags = R"doc()doc";

static const char *__doc_falcor_SceneObjectCollection_m_dirty_flags = R"doc()doc";

static const char *__doc_falcor_SceneObjectCollection_m_objects = R"doc()doc";

static const char *__doc_falcor_SceneObjectCollection_m_scene = R"doc()doc";

static const char *__doc_falcor_SceneObjectCollection_mark_dirty =
R"doc(Mark an object as dirty with the specified flags.

Parameter ``object``:
    The object to mark as dirty.

Parameter ``flags``:
    The dirty flags to set.)doc";

static const char *__doc_falcor_SceneObjectCollection_mark_remove =
R"doc(Mark an object for removal.

Parameter ``object``:
    The object to mark for removal.)doc";

static const char *__doc_falcor_SceneObjectCollection_objects = R"doc(List of objects in the collection.)doc";

static const char *__doc_falcor_SceneObjectCollection_operator_array = R"doc(Access object by index (const).)doc";

static const char *__doc_falcor_SceneObjectCollection_operator_array_2 = R"doc(Access object by index (non-const).)doc";

static const char *__doc_falcor_SceneObjectCollection_operator_assign = R"doc()doc";

static const char *__doc_falcor_SceneObjectCollection_operator_assign_2 = R"doc()doc";

static const char *__doc_falcor_SceneObjectCollection_remove_marked_and_compact =
R"doc(Remove objects marked for removal and compact the collection. This
retains the order of the objects in the collection.)doc";

static const char *__doc_falcor_SceneObjectCollection_reset_dirty_flags = R"doc(Reset all dirty flags.)doc";

static const char *__doc_falcor_SceneObjectCollection_resize = R"doc()doc";

static const char *__doc_falcor_SceneObjectCollection_size = R"doc(The size of the collection.)doc";

static const char *__doc_falcor_SceneObjectFactory = R"doc(Forward declaration of factory class.)doc";

static const char *__doc_falcor_SceneObjectFactory_2 = R"doc(Forward declaration of factory class.)doc";

static const char *__doc_falcor_SceneObjectFactory_ClassInfo = R"doc()doc";

static const char *__doc_falcor_SceneObjectFactory_ClassInfo_create_fn = R"doc()doc";

static const char *__doc_falcor_SceneObjectFactory_ClassInfo_name = R"doc()doc";

static const char *__doc_falcor_SceneObjectFactory_ClassInfo_type = R"doc()doc";

static const char *__doc_falcor_SceneObjectFactory_SceneObjectFactory = R"doc(Constructor.)doc";

static const char *__doc_falcor_SceneObjectFactory_SceneObjectFactory_2 = R"doc()doc";

static const char *__doc_falcor_SceneObjectFactory_SceneObjectFactory_3 = R"doc()doc";

static const char *__doc_falcor_SceneObjectFactory_class_info =
R"doc(Get class info by name or throw if not found.

Parameter ``name``:
    The name of the class.

Returns:
    The class info.)doc";

static const char *__doc_falcor_SceneObjectFactory_class_info_2 =
R"doc(Get class info by type or throw if not found.

Parameter ``type``:
    The type info of the class.

Returns:
    The class info.)doc";

static const char *__doc_falcor_SceneObjectFactory_class_infos = R"doc(List of registered class infos.)doc";

static const char *__doc_falcor_SceneObjectFactory_create =
R"doc(Create an instance of a class.

Parameter ``info``:
    The class info.

Returns:
    The created instance.)doc";

static const char *__doc_falcor_SceneObjectFactory_create_2 =
R"doc(Create an instance of a class by name.

Parameter ``name``:
    The name of the class.

Returns:
    The created instance.)doc";

static const char *__doc_falcor_SceneObjectFactory_create_3 =
R"doc(Create an instance of a class by type.

Parameter ``type``:
    The type info of the class.

Returns:
    The created instance.)doc";

static const char *__doc_falcor_SceneObjectFactory_find_class_info =
R"doc(Find class info by name.

Parameter ``name``:
    The name of the class.

Returns:
    The class info or nullptr if not found.)doc";

static const char *__doc_falcor_SceneObjectFactory_find_class_info_2 =
R"doc(Find class info by type.

Parameter ``type``:
    The type info of the class.

Returns:
    The class info or nullptr if not found.)doc";

static const char *__doc_falcor_SceneObjectFactory_get =
R"doc(Get the singleton instance of the factory. Delegates to
get_scene_object_factory() which is explicitly instantiated in
scene.cpp to ensure a single instance across shared library
boundaries.)doc";

static const char *__doc_falcor_SceneObjectFactory_m_class_infos = R"doc()doc";

static const char *__doc_falcor_SceneObjectFactory_operator_assign = R"doc()doc";

static const char *__doc_falcor_SceneObjectFactory_operator_assign_2 = R"doc()doc";

static const char *__doc_falcor_SceneObjectFactory_register_class =
R"doc(Register a class.

Template parameter ``T``:
    The class type to register.

Parameter ``name``:
    The name of the class.

Parameter ``create_fn``:
    Function to create an instance of the class.)doc";

static const char *__doc_falcor_SceneObjectKind = R"doc(Kind of scene object.)doc";

static const char *__doc_falcor_SceneObjectKind_animation = R"doc()doc";

static const char *__doc_falcor_SceneObjectKind_component = R"doc()doc";

static const char *__doc_falcor_SceneObjectKind_entity = R"doc()doc";

static const char *__doc_falcor_SceneObjectKind_geometry = R"doc()doc";

static const char *__doc_falcor_SceneObjectKind_info = R"doc()doc";

static const char *__doc_falcor_SceneObjectKind_material = R"doc()doc";

static const char *__doc_falcor_SceneObject_DirtyFlags = R"doc(Dirty flags shared by all scene objects.)doc";

static const char *__doc_falcor_SceneObject_DirtyFlags_added = R"doc()doc";

static const char *__doc_falcor_SceneObject_DirtyFlags_none = R"doc()doc";

static const char *__doc_falcor_SceneObject_DirtyFlags_removed = R"doc()doc";

static const char *__doc_falcor_SceneObject_DirtyFlags_resources = R"doc()doc";

static const char *__doc_falcor_SceneObject_SceneObject = R"doc(Constructor.)doc";

static const char *__doc_falcor_SceneObject_SceneObject_2 = R"doc()doc";

static const char *__doc_falcor_SceneObject_SceneObject_3 = R"doc()doc";

static const char *__doc_falcor_SceneObject_as = R"doc(Cast this object to type T.)doc";

static const char *__doc_falcor_SceneObject_as_2 = R"doc(Cast this object to type T.)doc";

static const char *__doc_falcor_SceneObject_class_name = R"doc()doc";

static const char *__doc_falcor_SceneObject_clear_invalid_references =
R"doc(Called during Scene::update() when other objects have been removed.
This allows to clear invalid references in this object.)doc";

static const char *__doc_falcor_SceneObject_collection_index = R"doc(The index in the collection this object belongs to.)doc";

static const char *__doc_falcor_SceneObject_create = R"doc()doc";

static const char *__doc_falcor_SceneObject_destroy = R"doc()doc";

static const char *__doc_falcor_SceneObject_is = R"doc(Check if this object is of type T.)doc";

static const char *__doc_falcor_SceneObject_is_valid = R"doc(Check if the object is valid (not removed).)doc";

static const char *__doc_falcor_SceneObject_kind = R"doc(The kind of this scene object.)doc";

static const char *__doc_falcor_SceneObject_m_collection_index = R"doc(Index in the scene object collection.)doc";

static const char *__doc_falcor_SceneObject_m_name = R"doc(Name of the object.)doc";

static const char *__doc_falcor_SceneObject_m_removed = R"doc()doc";

static const char *__doc_falcor_SceneObject_m_scene = R"doc(Scene this object belongs to.)doc";

static const char *__doc_falcor_SceneObject_name = R"doc(The name of the scene object.)doc";

static const char *__doc_falcor_SceneObject_on_add_to_scene = R"doc(Called during Scene::update() when object is added to the scene.)doc";

static const char *__doc_falcor_SceneObject_on_create =
R"doc(Called when the object is created. This is called in the context of
Scene::create_object(). Both the scene and collection index of the
object are valid at this point. The default implementation calls
ReflectedObject::set_properties() to set properties on the object.)doc";

static const char *__doc_falcor_SceneObject_on_destroy =
R"doc(Called when the object is destroyed. This is called during
Scene::update() after the object has been marked as removed. After
this call, the scene and collection index of the object are invalid
and should not be accessed.)doc";

static const char *__doc_falcor_SceneObject_on_load_resources =
R"doc(Called during Scene::update() to let the object load resources. Note:
This is called for all objects with the resources dirty flag set, as
well as all newly added objects.)doc";

static const char *__doc_falcor_SceneObject_on_remove_from_scene = R"doc(Called during Scene::update() when object is removed from the scene.)doc";

static const char *__doc_falcor_SceneObject_operator_assign = R"doc()doc";

static const char *__doc_falcor_SceneObject_operator_assign_2 = R"doc()doc";

static const char *__doc_falcor_SceneObject_remove =
R"doc(Destroy this object. This does not immediately delete the object, but
has the following effects: - Marks the object as removed - Object will
be deleted when the scene is next updated)doc";

static const char *__doc_falcor_SceneObject_scene = R"doc(The scene this scene object belongs to.)doc";

static const char *__doc_falcor_SceneObject_set_default_name = R"doc(Called when an object is added to the scene to set a default name.)doc";

static const char *__doc_falcor_SceneObject_set_name = R"doc()doc";

static const char *__doc_falcor_SceneRayTracingSetup =
R"doc(Helper to create a ray tracing pipeline and shader table for a scene.

Based on the scene's hit group policy the following is provided: - a
list of entry point names to link into the ray tracing shader program.
- a list of hit group descriptors for pipeline creation. - a list of
hit group names for ShaderTableDesc (ordered per scene's hit group
policy). - a list of miss entry point names for ShaderTableDesc (one
per ray type). - ray tracing pipeline flags.

In addition, this helper also provides functions to further simplify
pipeline and shader table creation from the collected data: -
link_program() to link a shader program with the collected entry
points plus additional ones (e.g., ray gen). - create_pipeline() to
create a ray tracing pipeline from the linked shader program. -
create_shader_table() to create a shader table from the linked shader
program and ray gen entry points.)doc";

static const char *__doc_falcor_SceneRayTracingSetup_Options = R"doc(Options for creating the ray tracing setup.)doc";

static const char *__doc_falcor_SceneRayTracingSetup_Options_skip_unused_geometry_types =
R"doc(Skip hit groups for geometry types that are not present in the scene
(replace with dummy entries).)doc";

static const char *__doc_falcor_SceneRayTracingSetup_PerGeometryTypeRayDesc =
R"doc(Description of a single ray type with its miss shader and hit groups
per geometry type.)doc";

static const char *__doc_falcor_SceneRayTracingSetup_PerGeometryTypeRayDesc_hit_groups = R"doc(Hit groups keyed by geometry type.)doc";

static const char *__doc_falcor_SceneRayTracingSetup_PerGeometryTypeRayDesc_miss_entry_point = R"doc(Miss shader entry point name.)doc";

static const char *__doc_falcor_SceneRayTracingSetup_RayDesc =
R"doc(Description of a single ray type. This describes ray types defined
with the helper macros found in scene_ray_tracing.slangh.)doc";

static const char *__doc_falcor_SceneRayTracingSetup_RayDesc_has_any_hit =
R"doc(Whether this ray type has an associated any hit shader. Needs
SCENE_RAY_TYPE_ANY_HIT to be defined for this ray type.)doc";

static const char *__doc_falcor_SceneRayTracingSetup_RayDesc_has_closest_hit =
R"doc(Whether this ray type has an associated closest hit shader. Needs
SCENE_RAY_TYPE_CLOSEST_HIT to be defined for this ray type.)doc";

static const char *__doc_falcor_SceneRayTracingSetup_RayDesc_has_miss =
R"doc(Whether this ray type has an associated miss shader. Needs
SCENE_RAY_TYPE_MISS to be defined for this ray type.)doc";

static const char *__doc_falcor_SceneRayTracingSetup_RayDesc_name =
R"doc(Ray type name (for example "intersect", "visibility", etc.). Needs to
match the ray type name used in the SCENE_RAY_TYPE_XXX macros.)doc";

static const char *__doc_falcor_SceneRayTracingSetup_create =
R"doc(Create a ray tracing setup for a scene given ray type descriptions.
Missing ray types, geometry types, and unused scene geometry types are
filled with dummy entries.

Parameter ``scene``:
    The scene to create the setup for.

Parameter ``ray_descs``:
    Per-ray-type descriptions (miss shader + hit groups keyed by
    geometry type).

Parameter ``options``:
    Optional configuration options.

Returns:
    The populated ray tracing setup.)doc";

static const char *__doc_falcor_SceneRayTracingSetup_create_2 = R"doc()doc";

static const char *__doc_falcor_SceneRayTracingSetup_create_3 =
R"doc(Create a ray tracing setup for a scene given per-ray-type
descriptions. Missing ray types, geometry types, and unused scene
geometry types are filled with dummy entries.

Parameter ``scene``:
    The scene to create the setup for.

Parameter ``ray_descs``:
    Per-ray-type descriptions (miss shader + hit groups keyed by
    geometry type).

Parameter ``options``:
    Optional configuration options.

Returns:
    The populated ray tracing setup.)doc";

static const char *__doc_falcor_SceneRayTracingSetup_create_4 = R"doc()doc";

static const char *__doc_falcor_SceneRayTracingSetup_create_pipeline =
R"doc(Create a ray tracing pipeline from this setup. The hit_groups field is
overwritten and flags are OR'ed with the setup's pipeline_flags.

Parameter ``desc``:
    Pipeline descriptor (hit_groups is overwritten, flags are
    combined).

Returns:
    The created ray tracing pipeline.)doc";

static const char *__doc_falcor_SceneRayTracingSetup_create_shader_table =
R"doc(Create a shader table from this setup.

Parameter ``program``:
    The linked shader program.

Parameter ``ray_gen_entry_points``:
    Ray generation entry point names.

Returns:
    The created shader table.)doc";

static const char *__doc_falcor_SceneRayTracingSetup_entry_points = R"doc(Entry point names needed for the ray tracing pipeline.)doc";

static const char *__doc_falcor_SceneRayTracingSetup_hit_groups = R"doc(Hit group descriptors for RayTracingPipelineDesc.)doc";

static const char *__doc_falcor_SceneRayTracingSetup_link_program =
R"doc(Link a shader program with the setup's entry points plus additional
ones (e.g., ray gen).

Parameter ``module``:
    The composed Slang module containing all entry points.

Parameter ``additional_entry_points``:
    Additional entry point names to include (e.g., ray gen shaders).

Returns:
    The linked shader program.)doc";

static const char *__doc_falcor_SceneRayTracingSetup_pipeline_flags = R"doc(Ray tracing pipeline flags (e.g., enable_linear_swept_spheres).)doc";

static const char *__doc_falcor_SceneRayTracingSetup_sbt_hit_group_names =
R"doc(Hit group names for ShaderTableDesc (ordered per scene's hit group
policy).)doc";

static const char *__doc_falcor_SceneRayTracingSetup_sbt_miss_entry_points = R"doc(Miss entry point names for ShaderTableDesc (one per ray type).)doc";

static const char *__doc_falcor_SceneRequirements = R"doc(Scene requirements for the shader compilation.)doc";

static const char *__doc_falcor_SceneRequirements_modules = R"doc()doc";

static const char *__doc_falcor_SceneRequirements_operator_le = R"doc()doc";

static const char *__doc_falcor_SceneRequirements_ray_tracing_pipeline_flags = R"doc(Required ray-tracing pipeline flags.)doc";

static const char *__doc_falcor_SceneRequirements_type_conformances = R"doc()doc";

static const char *__doc_falcor_SceneSystem = R"doc(Base class for scene systems.)doc";

static const char *__doc_falcor_SceneSystem_SceneSystem =
R"doc(Constructor.

Parameter ``scene``:
    The scene this system belongs to.)doc";

static const char *__doc_falcor_SceneSystem_SceneSystem_2 = R"doc()doc";

static const char *__doc_falcor_SceneSystem_SceneSystem_3 = R"doc()doc";

static const char *__doc_falcor_SceneSystem_bind_to_scene =
R"doc(Bind the system to the given shader cursor.

Parameter ``cursor``:
    Shader cursor to bind to.)doc";

static const char *__doc_falcor_SceneSystem_class_name = R"doc()doc";

static const char *__doc_falcor_SceneSystem_m_scene = R"doc(Scene this system belongs to.)doc";

static const char *__doc_falcor_SceneSystem_operator_assign = R"doc()doc";

static const char *__doc_falcor_SceneSystem_operator_assign_2 = R"doc()doc";

static const char *__doc_falcor_SceneSystem_scene = R"doc(The scene this system belongs to.)doc";

static const char *__doc_falcor_SceneSystem_update =
R"doc(Called during Scene::update() to update the system.

Parameter ``ctx``:
    Update context.

Returns:
    Flags indicating what was updated.)doc";

static const char *__doc_falcor_SceneUpdateContext = R"doc(Context passed to scene update functions.)doc";

static const char *__doc_falcor_SceneUpdateContext_SceneUpdateContext = R"doc()doc";

static const char *__doc_falcor_SceneUpdateContext_SceneUpdateContext_2 = R"doc()doc";

static const char *__doc_falcor_SceneUpdateContext_SceneUpdateContext_3 = R"doc()doc";

static const char *__doc_falcor_SceneUpdateContext_command_encoder = R"doc(Command encoder to record commands into.)doc";

static const char *__doc_falcor_SceneUpdateContext_device = R"doc(The device used for this update.)doc";

static const char *__doc_falcor_SceneUpdateContext_m_command_encoder = R"doc()doc";

static const char *__doc_falcor_SceneUpdateContext_m_device = R"doc()doc";

static const char *__doc_falcor_SceneUpdateContext_operator_assign = R"doc()doc";

static const char *__doc_falcor_SceneUpdateContext_operator_assign_2 = R"doc()doc";

static const char *__doc_falcor_SceneUpdateContext_submit = R"doc(Submit the recorded commands to the device.)doc";

static const char *__doc_falcor_SceneUpdateFlags = R"doc(Flags indicating what was updated during a scene update.)doc";

static const char *__doc_falcor_SceneUpdateFlags_animation = R"doc()doc";

static const char *__doc_falcor_SceneUpdateFlags_geometry = R"doc()doc";

static const char *__doc_falcor_SceneUpdateFlags_info = R"doc()doc";

static const char *__doc_falcor_SceneUpdateFlags_lights = R"doc()doc";

static const char *__doc_falcor_SceneUpdateFlags_materials = R"doc()doc";

static const char *__doc_falcor_SceneUpdateFlags_none = R"doc()doc";

static const char *__doc_falcor_SceneUpdateFlags_render_state = R"doc()doc";

static const char *__doc_falcor_SceneUpdateFlags_requirements = R"doc()doc";

static const char *__doc_falcor_SceneUpdateFlags_transforms = R"doc()doc";

static const char *__doc_falcor_Scene_Scene =
R"doc(Constructor.

Parameter ``device``:
    The device to use for rendering.)doc";

static const char *__doc_falcor_Scene_Scene_2 =
R"doc(Constructor.

Parameter ``device``:
    The device to use for rendering.

Parameter ``importer_scene``:
    The importer scene to create the scene from.)doc";

static const char *__doc_falcor_Scene_Scene_3 =
R"doc(Constructor.

Parameter ``device``:
    The device to use for rendering.

Parameter ``path``:
    Path to the scene file to load.

Parameter ``recompute_normals``:
    If true, recompute normals for all meshes.)doc";

static const char *__doc_falcor_Scene_Scene_4 = R"doc()doc";

static const char *__doc_falcor_Scene_Scene_5 = R"doc()doc";

static const char *__doc_falcor_Scene_active_camera =
R"doc(The active camera used for rendering. This is automatically set if a
Camera component is added and no active camera is set, and
automatically cleared if the active camera is removed.)doc";

static const char *__doc_falcor_Scene_add_object =
R"doc(Add an existing object to the scene and register it with the
appropriate collection.

Template parameter ``T``:
    The type of object to add.

Parameter ``object``:
    The object to add.)doc";

static const char *__doc_falcor_Scene_add_scene_globals =
R"doc(Add scene globals to the scene. Scene globals are shader globals that
are set on the scene and automatically bound when rendering.

Parameter ``globals``:
    The scene globals to add.)doc";

static const char *__doc_falcor_Scene_animation_collection = R"doc(The underlying animation collection with dirty flag tracking (const).)doc";

static const char *__doc_falcor_Scene_animation_collection_2 =
R"doc(The underlying animation collection with dirty flag tracking (non-
const).)doc";

static const char *__doc_falcor_Scene_animation_duration = R"doc(Get the maximum duration across all animations (0 if no animations).)doc";

static const char *__doc_falcor_Scene_animations = R"doc(View providing indexed access to all animations in the scene.)doc";

static const char *__doc_falcor_Scene_bind =
R"doc(Bind scene data to shader globals for rendering.

Parameter ``globals``:
    The shader cursor pointing to the global parameters.

Parameter ``flags``:
    Flags indicating which parts of the scene to bind (materials,
    lights, etc.).)doc";

static const char *__doc_falcor_Scene_class_name = R"doc()doc";

static const char *__doc_falcor_Scene_component_collection = R"doc(The underlying component collection with dirty flag tracking (const).)doc";

static const char *__doc_falcor_Scene_component_collection_2 =
R"doc(The underlying component collection with dirty flag tracking (non-
const).)doc";

static const char *__doc_falcor_Scene_components = R"doc(View providing indexed access to all components in the scene.)doc";

static const char *__doc_falcor_Scene_create =
R"doc(Create an empty scene.

Parameter ``device``:
    The device to use for rendering.

Returns:
    The created scene.)doc";

static const char *__doc_falcor_Scene_create_2 =
R"doc(Create a scene from an importer scene.

Parameter ``device``:
    The device to use for rendering.

Parameter ``importer_scene``:
    The importer scene to create the scene from.

Returns:
    The created scene.)doc";

static const char *__doc_falcor_Scene_create_3 =
R"doc(Create a scene by loading from a file.

Parameter ``device``:
    The device to use for rendering.

Parameter ``path``:
    Path to the scene file to load.

Parameter ``recompute_normals``:
    If true, recompute normals for all meshes.

Returns:
    The created scene.)doc";

static const char *__doc_falcor_Scene_create_animation =
R"doc(Create an animation and add it to the scene.

Returns:
    Pointer to the created animation (owned by the scene).)doc";

static const char *__doc_falcor_Scene_create_entity =
R"doc(Create a new entity and add it to the scene. Entities are scene graph
nodes that hold transforms and components. Use Entity::set_parent() to
build a hierarchy after creation.

Parameter ``props``:
    Optional properties to initialize the entity with.

Returns:
    Pointer to the created entity (owned by the scene).)doc";

static const char *__doc_falcor_Scene_create_geometry =
R"doc(Create a geometry of a specific type and add it to the scene. The
geometry is owned by the scene and will be marked with
DirtyFlags::added.

Template parameter ``T``:
    The geometry type to create (e.g., StaticMeshGeometry,
    GeometryGroup).

Parameter ``props``:
    Optional properties to initialize the geometry with.

Returns:
    Pointer to the created geometry (owned by the scene).)doc";

static const char *__doc_falcor_Scene_create_geometry_2 =
R"doc(Create a geometry by registered type name (e.g.,
"StaticMeshGeometry").

Parameter ``type``:
    The registered name of the geometry type to create.

Parameter ``props``:
    Optional properties to initialize the geometry with.

Returns:
    Pointer to the created geometry (owned by the scene).)doc";

static const char *__doc_falcor_Scene_create_geometry_3 =
R"doc(Create a geometry by C++ type info.

Parameter ``type``:
    The type info of the geometry type to create.

Parameter ``props``:
    Optional properties to initialize the geometry with.

Returns:
    Pointer to the created geometry (owned by the scene).)doc";

static const char *__doc_falcor_Scene_create_material =
R"doc(Create a material of a specific type and add it to the scene. The
material is owned by the scene and will be marked with
DirtyFlags::added.

Template parameter ``T``:
    The material type to create (must derive from Material).

Parameter ``props``:
    Optional properties to initialize the material with.

Returns:
    Pointer to the created material (owned by the scene).)doc";

static const char *__doc_falcor_Scene_create_material_2 =
R"doc(Create a material by registered type name (e.g., "StandardMaterial").

Parameter ``type``:
    The registered name of the material type to create.

Parameter ``props``:
    Optional properties to initialize the material with.

Returns:
    Pointer to the created material (owned by the scene).)doc";

static const char *__doc_falcor_Scene_create_material_3 =
R"doc(Create a material by C++ type info.

Parameter ``type``:
    The type info of the material type to create.

Parameter ``props``:
    Optional properties to initialize the material with.

Returns:
    Pointer to the created material (owned by the scene).)doc";

static const char *__doc_falcor_Scene_create_object =
R"doc(Create an object of a specific type.

Template parameter ``T``:
    The type of object to create.

Parameter ``props``:
    Optional properties to initialize the object with.

Returns:
    Pointer to the created object.)doc";

static const char *__doc_falcor_Scene_create_object_2 =
R"doc(Create an object by type name.

Template parameter ``T``:
    The base type of object to create.

Parameter ``type``:
    The name of the type to create.

Parameter ``props``:
    Optional properties to initialize the object with.

Returns:
    Pointer to the created object.)doc";

static const char *__doc_falcor_Scene_create_object_3 =
R"doc(Create an object by type info.

Template parameter ``T``:
    The base type of object to create.

Parameter ``type``:
    The type info of the type to create.

Parameter ``props``:
    Optional properties to initialize the object with.

Returns:
    Pointer to the created object.)doc";

static const char *__doc_falcor_Scene_device =
R"doc(The devce device used for creating resources and executing rendering
commands.)doc";

static const char *__doc_falcor_Scene_entities = R"doc(View providing indexed access to all entities in the scene.)doc";

static const char *__doc_falcor_Scene_entity_collection = R"doc(The underlying entity collection with dirty flag tracking (const).)doc";

static const char *__doc_falcor_Scene_entity_collection_2 = R"doc(The underlying entity collection with dirty flag tracking (non-const).)doc";

static const char *__doc_falcor_Scene_geometries = R"doc(View providing indexed access to all geometries in the scene.)doc";

static const char *__doc_falcor_Scene_geometry_collection = R"doc(The underlying geometry collection with dirty flag tracking (const).)doc";

static const char *__doc_falcor_Scene_geometry_collection_2 =
R"doc(The underlying geometry collection with dirty flag tracking (non-
const).)doc";

static const char *__doc_falcor_Scene_handle_removed_objects =
R"doc(Called by Scene::update() to allow scene objects to clear invalid
references. Calls SceneObject::clear_invalid_references() on each all
scene objects to allow clearing references to now removed scene
objects. This is a no-op if no objects have been removed since last
call to Scene::update().)doc";

static const char *__doc_falcor_Scene_has_animation = R"doc(Returns true if this scene has any animation data.)doc";

static const char *__doc_falcor_Scene_has_geometry_type = R"doc(Returns true if the scene currently has geometry of the given type.)doc";

static const char *__doc_falcor_Scene_hit_group_policy = R"doc(Hit group policy used when building TLAS instance metadata.)doc";

static const char *__doc_falcor_Scene_m_active_camera =
R"doc(The active camera for rendering. Held as ref to prevent premature
deletion.)doc";

static const char *__doc_falcor_Scene_m_animation_collection = R"doc(Collection of all animations.)doc";

static const char *__doc_falcor_Scene_m_animations = R"doc(View of animations collection.)doc";

static const char *__doc_falcor_Scene_m_component_collection = R"doc(Collection of all components.)doc";

static const char *__doc_falcor_Scene_m_components = R"doc(View of components collection.)doc";

static const char *__doc_falcor_Scene_m_device = R"doc()doc";

static const char *__doc_falcor_Scene_m_emissive_geometry_system = R"doc(Emissive geometry system for managing emissive geometry.)doc";

static const char *__doc_falcor_Scene_m_entities = R"doc(View of entities collection.)doc";

static const char *__doc_falcor_Scene_m_entity_collection = R"doc(Collection of all entities.)doc";

static const char *__doc_falcor_Scene_m_geometries = R"doc(View of geometries collection.)doc";

static const char *__doc_falcor_Scene_m_geometry_collection = R"doc(Collection of all geometries.)doc";

static const char *__doc_falcor_Scene_m_hit_group_policy =
R"doc(Hit group policy used for scene ray tracing setup. This is currently
hard-coded but will eventually be configurable.)doc";

static const char *__doc_falcor_Scene_m_light_system = R"doc(Light system for managing lights.)doc";

static const char *__doc_falcor_Scene_m_material_collection = R"doc(Collection of all materials.)doc";

static const char *__doc_falcor_Scene_m_material_system = R"doc(Material system for managing materials.)doc";

static const char *__doc_falcor_Scene_m_materials = R"doc(View of materials collection.)doc";

static const char *__doc_falcor_Scene_m_render_module = R"doc(Slang module containing rendering shader code.)doc";

static const char *__doc_falcor_Scene_m_render_scene = R"doc(Internal render scene representation.)doc";

static const char *__doc_falcor_Scene_m_requirements = R"doc(Cached requirements for scene to work.)doc";

static const char *__doc_falcor_Scene_m_requirements_generation =
R"doc(A monotonically increasing generation counter for requirements
changes.)doc";

static const char *__doc_falcor_Scene_m_scene_globals = R"doc(List of scene globals to bind to the shader when rendering the scene.)doc";

static const char *__doc_falcor_Scene_m_scene_shader_object = R"doc(Shader object for binding scene data.)doc";

static const char *__doc_falcor_Scene_m_texture_manager = R"doc(Texture manager for loading and caching textures.)doc";

static const char *__doc_falcor_Scene_m_time = R"doc(Current animation time.)doc";

static const char *__doc_falcor_Scene_m_update_flags = R"doc(Flags indicating what changed in the last update.)doc";

static const char *__doc_falcor_Scene_m_update_generation =
R"doc(A monotonically increasing generation counter that is incremented on
each update with dirty changes.)doc";

static const char *__doc_falcor_Scene_m_update_time =
R"doc(The animation time for which the scene was last updated. Used to track
if animation time has changed since last update.)doc";

static const char *__doc_falcor_Scene_m_updated_signal =
R"doc(Signal emitted when the scene is updated. The signal parameter
indicates what changed.)doc";

static const char *__doc_falcor_Scene_mark_component_dirty = R"doc(Mark a component as dirty. Called by Component::mark_dirty().)doc";

static const char *__doc_falcor_Scene_mark_component_removed = R"doc(Mark a component as removed. Called by Component::remove().)doc";

static const char *__doc_falcor_Scene_mark_entity_dirty = R"doc(Mark an entity as dirty. Called by Entity::mark_dirty().)doc";

static const char *__doc_falcor_Scene_mark_entity_removed = R"doc(Mark an entity as removed. Called by Entity::remove().)doc";

static const char *__doc_falcor_Scene_mark_geometry_dirty = R"doc(Mark a geometry as dirty. Called by Geometry::mark_dirty().)doc";

static const char *__doc_falcor_Scene_mark_geometry_removed = R"doc(Mark a geometry as removed. Called by Geometry::remove().)doc";

static const char *__doc_falcor_Scene_mark_material_dirty = R"doc(Mark a material as dirty. Called by Material::mark_dirty().)doc";

static const char *__doc_falcor_Scene_mark_material_removed = R"doc(Mark a material as removed. Called by Material::remove().)doc";

static const char *__doc_falcor_Scene_material_collection = R"doc(The underlying material collection with dirty flag tracking (const).)doc";

static const char *__doc_falcor_Scene_material_collection_2 =
R"doc(The underlying material collection with dirty flag tracking (non-
const).)doc";

static const char *__doc_falcor_Scene_materials = R"doc(View providing indexed access to all materials in the scene.)doc";

static const char *__doc_falcor_Scene_operator_assign = R"doc()doc";

static const char *__doc_falcor_Scene_operator_assign_2 = R"doc()doc";

static const char *__doc_falcor_Scene_remove_scene_globals =
R"doc(Remove scene globals from the scene.

Parameter ``globals``:
    The scene globals to remove.)doc";

static const char *__doc_falcor_Scene_render_module =
R"doc(The Slang module containing the scene's rendering shader code
(falcor2.render).)doc";

static const char *__doc_falcor_Scene_render_scene =
R"doc(Get the low-level render scene that manages geometry data and
acceleration structures.)doc";

static const char *__doc_falcor_Scene_render_scene_2 = R"doc()doc";

static const char *__doc_falcor_Scene_requirements =
R"doc(Get the modules and conformances required to make the scene fully
work.)doc";

static const char *__doc_falcor_Scene_requirements_generation =
R"doc(A monotonically increasing generation counter that is incremented when
scene requirements change. This can be used to track if the scene
requirements have changed since a certain point in time.)doc";

static const char *__doc_falcor_Scene_scene_globals = R"doc(List of all scene globals currently added to the scene.)doc";

static const char *__doc_falcor_Scene_set_active_camera = R"doc()doc";

static const char *__doc_falcor_Scene_set_time =
R"doc(Set the current scene animation time. This will only take effect on
the next update() call.)doc";

static const char *__doc_falcor_Scene_texture_manager =
R"doc(The texture manager responsible for loading, caching, and managing
scene textures.)doc";

static const char *__doc_falcor_Scene_time = R"doc(The current scene animation time.)doc";

static const char *__doc_falcor_Scene_update =
R"doc(Update the scene, processing all pending changes. This must be called
before rendering to ensure device resources are up to date. Internally
creates a command encoder and submits it to the device. The update
processes: added/removed objects, dirty materials, lights, geometries,
entity transforms, and component render states.

Returns:
    Flags indicating what changed during the update.)doc";

static const char *__doc_falcor_Scene_update_2 =
R"doc(TODO(scene) make private? Called by Scene::update() to do the actual
update.

Parameter ``ctx``:
    The update context (command encoder, etc.).

Returns:
    Flags indicating what changed during the update.)doc";

static const char *__doc_falcor_Scene_update_animations =
R"doc(Update the scene from animations.

Parameter ``ctx``:
    The update context (command encoder, etc.).

Returns:
    Flags indicating what changed during the update.)doc";

static const char *__doc_falcor_Scene_update_flags = R"doc(Flags indicating what changed during the last update() call.)doc";

static const char *__doc_falcor_Scene_update_generation =
R"doc(A monotonically increasing generation counter that is incremented on
each update with dirty changes. This can be used to track if the scene
has changed since a certain point in time.)doc";

static const char *__doc_falcor_Scene_updated =
R"doc(Signal emitted when the scene is updated. The signal parameter
indicates what changed.)doc";

static const char *__doc_falcor_Signal = R"doc(Primary template (undefined). Use Signal<void(Args...)>.)doc";

static const char *__doc_falcor_SignalConnection =
R"doc(RAII handle representing a signal-slot connection. Auto-disconnects on
destruction. Move-only.)doc";

static const char *__doc_falcor_SignalConnectionGroup =
R"doc(Manages a group of signal connections with bulk disconnect. RAII:
auto-disconnects all on destruction and on move-assignment.)doc";

static const char *__doc_falcor_SignalConnectionGroup_2 =
R"doc(Manages a group of signal connections with bulk disconnect. RAII:
auto-disconnects all on destruction and on move-assignment.)doc";

static const char *__doc_falcor_SignalConnectionGroup_SignalConnectionGroup = R"doc()doc";

static const char *__doc_falcor_SignalConnectionGroup_SignalConnectionGroup_2 = R"doc()doc";

static const char *__doc_falcor_SignalConnectionGroup_SignalConnectionGroup_3 = R"doc()doc";

static const char *__doc_falcor_SignalConnectionGroup_add = R"doc(Add a connection to the group.)doc";

static const char *__doc_falcor_SignalConnectionGroup_disconnect_all = R"doc(Disconnect all tracked connections.)doc";

static const char *__doc_falcor_SignalConnectionGroup_empty = R"doc(Returns true if no connections are tracked.)doc";

static const char *__doc_falcor_SignalConnectionGroup_m_connections = R"doc()doc";

static const char *__doc_falcor_SignalConnectionGroup_operator_assign = R"doc()doc";

static const char *__doc_falcor_SignalConnectionGroup_operator_assign_2 = R"doc()doc";

static const char *__doc_falcor_SignalConnectionGroup_prune = R"doc(Remove dead connections to keep the vector from growing unboundedly.)doc";

static const char *__doc_falcor_SignalConnectionGroup_size = R"doc(Returns the number of live (connected) connections.)doc";

static const char *__doc_falcor_SignalConnection_SignalConnection = R"doc()doc";

static const char *__doc_falcor_SignalConnection_SignalConnection_2 = R"doc()doc";

static const char *__doc_falcor_SignalConnection_SignalConnection_3 = R"doc()doc";

static const char *__doc_falcor_SignalConnection_SignalConnection_4 = R"doc()doc";

static const char *__doc_falcor_SignalConnection_connected = R"doc(Check if this connection is still active.)doc";

static const char *__doc_falcor_SignalConnection_disconnect = R"doc(Disconnect the slot from the signal.)doc";

static const char *__doc_falcor_SignalConnection_m_control = R"doc()doc";

static const char *__doc_falcor_SignalConnection_operator_assign = R"doc()doc";

static const char *__doc_falcor_SignalConnection_operator_assign_2 = R"doc()doc";

static const char *__doc_falcor_SignalConnection_share =
R"doc(Create a shared duplicate that shares the same control block. Both
SignalConnection objects can independently call disconnect(). First
destructor disconnects; second is a no-op.)doc";

static const char *__doc_falcor_SpanArchiveSink =
R"doc(ArchiveSink implementation that writes into caller-provided fixed
storage.)doc";

static const char *__doc_falcor_SpanArchiveSink_SpanArchiveSink = R"doc(Create a sink over a mutable byte span.)doc";

static const char *__doc_falcor_SpanArchiveSink_bytes = R"doc(Return the written prefix of the provided span.)doc";

static const char *__doc_falcor_SpanArchiveSink_capacity = R"doc(Return the total writable capacity.)doc";

static const char *__doc_falcor_SpanArchiveSink_empty = R"doc(Return true if no bytes have been written.)doc";

static const char *__doc_falcor_SpanArchiveSink_m_bytes = R"doc()doc";

static const char *__doc_falcor_SpanArchiveSink_m_offset = R"doc()doc";

static const char *__doc_falcor_SpanArchiveSink_offset = R"doc(Return the current write offset.)doc";

static const char *__doc_falcor_SpanArchiveSink_remaining = R"doc(Return the remaining writable capacity.)doc";

static const char *__doc_falcor_SpanArchiveSink_size = R"doc(Return the number of bytes written.)doc";

static const char *__doc_falcor_SpanArchiveSink_write_bytes = R"doc(Append raw bytes. Throws if the span capacity would be exceeded.)doc";

static const char *__doc_falcor_SpanArchiveSource = R"doc(ArchiveSource implementation over a borrowed bounded byte span.)doc";

static const char *__doc_falcor_SpanArchiveSource_SpanArchiveSource = R"doc(Create a source over borrowed archive bytes.)doc";

static const char *__doc_falcor_SpanArchiveSource_m_bytes = R"doc()doc";

static const char *__doc_falcor_SpanArchiveSource_m_offset = R"doc()doc";

static const char *__doc_falcor_SpanArchiveSource_offset = R"doc(Return the current read offset.)doc";

static const char *__doc_falcor_SpanArchiveSource_read_bytes =
R"doc(Read raw bytes and advance the cursor. Throws if the read extends past
the span.)doc";

static const char *__doc_falcor_SpanArchiveSource_remaining = R"doc(Return the number of unread bytes.)doc";

static const char *__doc_falcor_SpanArchiveSource_skip = R"doc(Advance the cursor. Throws if the skip extends past the span.)doc";

static const char *__doc_falcor_SpotLight = R"doc()doc";

static const char *__doc_falcor_SpotLight_SpotLight = R"doc()doc";

static const char *__doc_falcor_SpotLight_class_descriptor = R"doc()doc";

static const char *__doc_falcor_SpotLight_class_name = R"doc()doc";

static const char *__doc_falcor_SpotLight_cutoff_angle = R"doc()doc";

static const char *__doc_falcor_SpotLight_falloff_angle = R"doc()doc";

static const char *__doc_falcor_SpotLight_intensity = R"doc()doc";

static const char *__doc_falcor_SpotLight_m_cutoff_angle = R"doc()doc";

static const char *__doc_falcor_SpotLight_m_falloff_angle = R"doc()doc";

static const char *__doc_falcor_SpotLight_m_intensity = R"doc()doc";

static const char *__doc_falcor_SpotLight_reflect = R"doc(Reflect this class.)doc";

static const char *__doc_falcor_SpotLight_reflected_class_name = R"doc()doc";

static const char *__doc_falcor_SpotLight_set_cutoff_angle = R"doc()doc";

static const char *__doc_falcor_SpotLight_set_falloff_angle = R"doc()doc";

static const char *__doc_falcor_SpotLight_set_intensity = R"doc()doc";

static const char *__doc_falcor_SpotLight_static_class_descriptor = R"doc()doc";

static const char *__doc_falcor_SpotLight_write_to_cursor = R"doc()doc";

static const char *__doc_falcor_SpotLight_write_to_cursor_2 = R"doc()doc";

static const char *__doc_falcor_SpotLight_write_to_cursor_impl = R"doc()doc";

static const char *__doc_falcor_StandardMaterial = R"doc(Standard material.)doc";

static const char *__doc_falcor_StandardMaterial_StandardMaterial = R"doc(Constructor.)doc";

static const char *__doc_falcor_StandardMaterial_base_color_factor = R"doc()doc";

static const char *__doc_falcor_StandardMaterial_base_color_texture_handle = R"doc()doc";

static const char *__doc_falcor_StandardMaterial_class_descriptor = R"doc()doc";

static const char *__doc_falcor_StandardMaterial_class_name = R"doc()doc";

static const char *__doc_falcor_StandardMaterial_diffuse_transmission_factor = R"doc()doc";

static const char *__doc_falcor_StandardMaterial_double_sided = R"doc()doc";

static const char *__doc_falcor_StandardMaterial_emissive_factor = R"doc()doc";

static const char *__doc_falcor_StandardMaterial_emissive_texture_handle = R"doc()doc";

static const char *__doc_falcor_StandardMaterial_ior = R"doc()doc";

static const char *__doc_falcor_StandardMaterial_m_base_color_factor = R"doc()doc";

static const char *__doc_falcor_StandardMaterial_m_base_color_texture = R"doc()doc";

static const char *__doc_falcor_StandardMaterial_m_base_color_texture_handle = R"doc()doc";

static const char *__doc_falcor_StandardMaterial_m_base_color_texture_path = R"doc()doc";

static const char *__doc_falcor_StandardMaterial_m_diffuse_transmission_factor = R"doc()doc";

static const char *__doc_falcor_StandardMaterial_m_double_sided = R"doc()doc";

static const char *__doc_falcor_StandardMaterial_m_emissive_factor = R"doc()doc";

static const char *__doc_falcor_StandardMaterial_m_emissive_texture = R"doc()doc";

static const char *__doc_falcor_StandardMaterial_m_emissive_texture_handle = R"doc()doc";

static const char *__doc_falcor_StandardMaterial_m_emissive_texture_path = R"doc()doc";

static const char *__doc_falcor_StandardMaterial_m_ior = R"doc()doc";

static const char *__doc_falcor_StandardMaterial_m_metallic_factor = R"doc()doc";

static const char *__doc_falcor_StandardMaterial_m_metallic_roughness_texture = R"doc()doc";

static const char *__doc_falcor_StandardMaterial_m_metallic_roughness_texture_handle = R"doc()doc";

static const char *__doc_falcor_StandardMaterial_m_metallic_roughness_texture_path = R"doc()doc";

static const char *__doc_falcor_StandardMaterial_m_metallic_texture_channel = R"doc()doc";

static const char *__doc_falcor_StandardMaterial_m_normal_texture = R"doc()doc";

static const char *__doc_falcor_StandardMaterial_m_normal_texture_handle = R"doc()doc";

static const char *__doc_falcor_StandardMaterial_m_normal_texture_path = R"doc()doc";

static const char *__doc_falcor_StandardMaterial_m_normal_texture_scale = R"doc()doc";

static const char *__doc_falcor_StandardMaterial_m_roughness_factor = R"doc()doc";

static const char *__doc_falcor_StandardMaterial_m_roughness_texture_channel = R"doc()doc";

static const char *__doc_falcor_StandardMaterial_m_specular_transmission_factor = R"doc()doc";

static const char *__doc_falcor_StandardMaterial_m_thin_surface = R"doc()doc";

static const char *__doc_falcor_StandardMaterial_m_transmission_factor = R"doc()doc";

static const char *__doc_falcor_StandardMaterial_m_transmission_texture = R"doc()doc";

static const char *__doc_falcor_StandardMaterial_m_transmission_texture_handle = R"doc()doc";

static const char *__doc_falcor_StandardMaterial_m_transmission_texture_path = R"doc()doc";

static const char *__doc_falcor_StandardMaterial_mark_dirty_properties = R"doc()doc";

static const char *__doc_falcor_StandardMaterial_mark_dirty_resources = R"doc()doc";

static const char *__doc_falcor_StandardMaterial_metallic_factor = R"doc()doc";

static const char *__doc_falcor_StandardMaterial_metallic_roughness_texture_handle = R"doc()doc";

static const char *__doc_falcor_StandardMaterial_metallic_texture_channel = R"doc()doc";

static const char *__doc_falcor_StandardMaterial_normal_texture_handle = R"doc()doc";

static const char *__doc_falcor_StandardMaterial_normal_texture_scale = R"doc()doc";

static const char *__doc_falcor_StandardMaterial_on_load_resources = R"doc()doc";

static const char *__doc_falcor_StandardMaterial_reflect = R"doc(Reflect this class.)doc";

static const char *__doc_falcor_StandardMaterial_reflected_class_name = R"doc()doc";

static const char *__doc_falcor_StandardMaterial_roughness_factor = R"doc()doc";

static const char *__doc_falcor_StandardMaterial_roughness_texture_channel = R"doc()doc";

static const char *__doc_falcor_StandardMaterial_specular_transmission_factor = R"doc()doc";

static const char *__doc_falcor_StandardMaterial_static_class_descriptor = R"doc()doc";

static const char *__doc_falcor_StandardMaterial_thin_surface = R"doc()doc";

static const char *__doc_falcor_StandardMaterial_transmission_factor = R"doc()doc";

static const char *__doc_falcor_StandardMaterial_transmission_texture_handle = R"doc()doc";

static const char *__doc_falcor_StandardMaterial_update = R"doc()doc";

static const char *__doc_falcor_StandardMaterial_write_to_cursor = R"doc()doc";

static const char *__doc_falcor_StandardMaterial_write_to_cursor_2 = R"doc()doc";

static const char *__doc_falcor_StandardMaterial_write_to_cursor_impl = R"doc()doc";

static const char *__doc_falcor_StandardSpecGlossMaterial = R"doc(Standard specular-glossiness material.)doc";

static const char *__doc_falcor_StandardSpecGlossMaterial_StandardSpecGlossMaterial = R"doc(Constructor.)doc";

static const char *__doc_falcor_StandardSpecGlossMaterial_class_descriptor = R"doc()doc";

static const char *__doc_falcor_StandardSpecGlossMaterial_class_name = R"doc()doc";

static const char *__doc_falcor_StandardSpecGlossMaterial_diffuse_factor = R"doc()doc";

static const char *__doc_falcor_StandardSpecGlossMaterial_diffuse_texture_handle = R"doc()doc";

static const char *__doc_falcor_StandardSpecGlossMaterial_diffuse_transmission_factor = R"doc()doc";

static const char *__doc_falcor_StandardSpecGlossMaterial_double_sided = R"doc()doc";

static const char *__doc_falcor_StandardSpecGlossMaterial_emissive_factor = R"doc()doc";

static const char *__doc_falcor_StandardSpecGlossMaterial_emissive_texture_handle = R"doc()doc";

static const char *__doc_falcor_StandardSpecGlossMaterial_glossiness_factor = R"doc()doc";

static const char *__doc_falcor_StandardSpecGlossMaterial_ior = R"doc()doc";

static const char *__doc_falcor_StandardSpecGlossMaterial_m_diffuse_factor = R"doc()doc";

static const char *__doc_falcor_StandardSpecGlossMaterial_m_diffuse_texture = R"doc()doc";

static const char *__doc_falcor_StandardSpecGlossMaterial_m_diffuse_texture_handle = R"doc()doc";

static const char *__doc_falcor_StandardSpecGlossMaterial_m_diffuse_texture_path = R"doc()doc";

static const char *__doc_falcor_StandardSpecGlossMaterial_m_diffuse_transmission_factor = R"doc()doc";

static const char *__doc_falcor_StandardSpecGlossMaterial_m_double_sided = R"doc()doc";

static const char *__doc_falcor_StandardSpecGlossMaterial_m_emissive_factor = R"doc()doc";

static const char *__doc_falcor_StandardSpecGlossMaterial_m_emissive_texture = R"doc()doc";

static const char *__doc_falcor_StandardSpecGlossMaterial_m_emissive_texture_handle = R"doc()doc";

static const char *__doc_falcor_StandardSpecGlossMaterial_m_emissive_texture_path = R"doc()doc";

static const char *__doc_falcor_StandardSpecGlossMaterial_m_glossiness_factor = R"doc()doc";

static const char *__doc_falcor_StandardSpecGlossMaterial_m_ior = R"doc()doc";

static const char *__doc_falcor_StandardSpecGlossMaterial_m_normal_texture = R"doc()doc";

static const char *__doc_falcor_StandardSpecGlossMaterial_m_normal_texture_handle = R"doc()doc";

static const char *__doc_falcor_StandardSpecGlossMaterial_m_normal_texture_path = R"doc()doc";

static const char *__doc_falcor_StandardSpecGlossMaterial_m_normal_texture_scale = R"doc()doc";

static const char *__doc_falcor_StandardSpecGlossMaterial_m_specular_factor = R"doc()doc";

static const char *__doc_falcor_StandardSpecGlossMaterial_m_specular_glossiness_texture = R"doc()doc";

static const char *__doc_falcor_StandardSpecGlossMaterial_m_specular_glossiness_texture_handle = R"doc()doc";

static const char *__doc_falcor_StandardSpecGlossMaterial_m_specular_glossiness_texture_path = R"doc()doc";

static const char *__doc_falcor_StandardSpecGlossMaterial_m_specular_transmission_factor = R"doc()doc";

static const char *__doc_falcor_StandardSpecGlossMaterial_m_thin_surface = R"doc()doc";

static const char *__doc_falcor_StandardSpecGlossMaterial_m_transmission_factor = R"doc()doc";

static const char *__doc_falcor_StandardSpecGlossMaterial_m_transmission_texture = R"doc()doc";

static const char *__doc_falcor_StandardSpecGlossMaterial_m_transmission_texture_handle = R"doc()doc";

static const char *__doc_falcor_StandardSpecGlossMaterial_m_transmission_texture_path = R"doc()doc";

static const char *__doc_falcor_StandardSpecGlossMaterial_mark_dirty_properties = R"doc()doc";

static const char *__doc_falcor_StandardSpecGlossMaterial_mark_dirty_resources = R"doc()doc";

static const char *__doc_falcor_StandardSpecGlossMaterial_normal_texture_handle = R"doc()doc";

static const char *__doc_falcor_StandardSpecGlossMaterial_normal_texture_scale = R"doc()doc";

static const char *__doc_falcor_StandardSpecGlossMaterial_on_load_resources = R"doc()doc";

static const char *__doc_falcor_StandardSpecGlossMaterial_reflect = R"doc(Reflect this class.)doc";

static const char *__doc_falcor_StandardSpecGlossMaterial_reflected_class_name = R"doc()doc";

static const char *__doc_falcor_StandardSpecGlossMaterial_specular_factor = R"doc()doc";

static const char *__doc_falcor_StandardSpecGlossMaterial_specular_glossiness_texture_handle = R"doc()doc";

static const char *__doc_falcor_StandardSpecGlossMaterial_specular_transmission_factor = R"doc()doc";

static const char *__doc_falcor_StandardSpecGlossMaterial_static_class_descriptor = R"doc()doc";

static const char *__doc_falcor_StandardSpecGlossMaterial_thin_surface = R"doc()doc";

static const char *__doc_falcor_StandardSpecGlossMaterial_transmission_factor = R"doc()doc";

static const char *__doc_falcor_StandardSpecGlossMaterial_transmission_texture_handle = R"doc()doc";

static const char *__doc_falcor_StandardSpecGlossMaterial_update = R"doc()doc";

static const char *__doc_falcor_StandardSpecGlossMaterial_write_to_cursor = R"doc()doc";

static const char *__doc_falcor_StandardSpecGlossMaterial_write_to_cursor_2 = R"doc()doc";

static const char *__doc_falcor_StandardSpecGlossMaterial_write_to_cursor_impl = R"doc()doc";

static const char *__doc_falcor_StaticCurveGeometry = R"doc(Static curve geometry.)doc";

static const char *__doc_falcor_StaticCurveGeometryDataDesc = R"doc()doc";

static const char *__doc_falcor_StaticCurveGeometryDataDesc_IndexStream = R"doc()doc";

static const char *__doc_falcor_StaticCurveGeometryDataDesc_IndexStream_data = R"doc()doc";

static const char *__doc_falcor_StaticCurveGeometryDataDesc_IndexStream_stride = R"doc()doc";

static const char *__doc_falcor_StaticCurveGeometryDataDesc_VertexAttributeStream = R"doc()doc";

static const char *__doc_falcor_StaticCurveGeometryDataDesc_VertexAttributeStream_data = R"doc()doc";

static const char *__doc_falcor_StaticCurveGeometryDataDesc_VertexAttributeStream_stride = R"doc()doc";

static const char *__doc_falcor_StaticCurveGeometryDataDesc_index_count = R"doc()doc";

static const char *__doc_falcor_StaticCurveGeometryDataDesc_index_stream = R"doc()doc";

static const char *__doc_falcor_StaticCurveGeometryDataDesc_indexing_mode = R"doc()doc";

static const char *__doc_falcor_StaticCurveGeometryDataDesc_name = R"doc()doc";

static const char *__doc_falcor_StaticCurveGeometryDataDesc_position_stream = R"doc()doc";

static const char *__doc_falcor_StaticCurveGeometryDataDesc_radius_stream = R"doc()doc";

static const char *__doc_falcor_StaticCurveGeometryDataDesc_vertex_count = R"doc()doc";

static const char *__doc_falcor_StaticCurveGeometry_class_descriptor = R"doc()doc";

static const char *__doc_falcor_StaticCurveGeometry_class_name = R"doc()doc";

static const char *__doc_falcor_StaticCurveGeometry_clear_invalid_references = R"doc()doc";

static const char *__doc_falcor_StaticCurveGeometry_convert_list_to_successive_indices =
R"doc(Convert list-mode indices (pairs per segment) to successive-mode
indices (one index per segment, where positions[index] and
positions[index+1] form the segment). If any pair is non-consecutive,
vertices are rearranged so each segment's endpoints are adjacent.)doc";

static const char *__doc_falcor_StaticCurveGeometry_m_indices = R"doc()doc";

static const char *__doc_falcor_StaticCurveGeometry_m_render_geometry = R"doc()doc";

static const char *__doc_falcor_StaticCurveGeometry_m_vertices = R"doc()doc";

static const char *__doc_falcor_StaticCurveGeometry_reflect = R"doc(Reflect this class.)doc";

static const char *__doc_falcor_StaticCurveGeometry_reflected_class_name = R"doc()doc";

static const char *__doc_falcor_StaticCurveGeometry_set_curve_data =
R"doc(Set the curve data.

Parameter ``desc``:
    Description of the curve data.)doc";

static const char *__doc_falcor_StaticCurveGeometry_static_class_descriptor = R"doc()doc";

static const char *__doc_falcor_StaticCurveGeometry_update_render_state = R"doc()doc";

static const char *__doc_falcor_StaticMeshGeometry =
R"doc(Static mesh geometry. A static mesh geometry contains one or more sub-
meshes with triangle data. Each sub-mesh has its own material (which
can be overriden per geometry instance).)doc";

static const char *__doc_falcor_StaticMeshGeometryDataDesc = R"doc()doc";

static const char *__doc_falcor_StaticMeshGeometryDataDesc_IndexStream = R"doc()doc";

static const char *__doc_falcor_StaticMeshGeometryDataDesc_IndexStream_data = R"doc()doc";

static const char *__doc_falcor_StaticMeshGeometryDataDesc_IndexStream_stride = R"doc()doc";

static const char *__doc_falcor_StaticMeshGeometryDataDesc_SubMesh = R"doc()doc";

static const char *__doc_falcor_StaticMeshGeometryDataDesc_SubMesh_index_count = R"doc()doc";

static const char *__doc_falcor_StaticMeshGeometryDataDesc_SubMesh_index_stream = R"doc()doc";

static const char *__doc_falcor_StaticMeshGeometryDataDesc_SubMesh_name = R"doc()doc";

static const char *__doc_falcor_StaticMeshGeometryDataDesc_VertexAttributeStream = R"doc()doc";

static const char *__doc_falcor_StaticMeshGeometryDataDesc_VertexAttributeStream_data = R"doc()doc";

static const char *__doc_falcor_StaticMeshGeometryDataDesc_VertexAttributeStream_stride = R"doc()doc";

static const char *__doc_falcor_StaticMeshGeometryDataDesc_handedness_stream = R"doc()doc";

static const char *__doc_falcor_StaticMeshGeometryDataDesc_name = R"doc()doc";

static const char *__doc_falcor_StaticMeshGeometryDataDesc_normal_stream = R"doc()doc";

static const char *__doc_falcor_StaticMeshGeometryDataDesc_position_stream = R"doc()doc";

static const char *__doc_falcor_StaticMeshGeometryDataDesc_sub_meshes = R"doc()doc";

static const char *__doc_falcor_StaticMeshGeometryDataDesc_tangent_stream = R"doc()doc";

static const char *__doc_falcor_StaticMeshGeometryDataDesc_texcoord_stream = R"doc()doc";

static const char *__doc_falcor_StaticMeshGeometryDataDesc_vertex_count = R"doc()doc";

static const char *__doc_falcor_StaticMeshGeometry_SubMesh = R"doc()doc";

static const char *__doc_falcor_StaticMeshGeometry_SubMesh_indices = R"doc()doc";

static const char *__doc_falcor_StaticMeshGeometry_SubMesh_material = R"doc()doc";

static const char *__doc_falcor_StaticMeshGeometry_SubMesh_name = R"doc()doc";

static const char *__doc_falcor_StaticMeshGeometry_SubMesh_vertices = R"doc()doc";

static const char *__doc_falcor_StaticMeshGeometry_class_descriptor = R"doc()doc";

static const char *__doc_falcor_StaticMeshGeometry_class_name = R"doc()doc";

static const char *__doc_falcor_StaticMeshGeometry_clear_invalid_references = R"doc()doc";

static const char *__doc_falcor_StaticMeshGeometry_index_count = R"doc(Number of indices in the given sub-mesh.)doc";

static const char *__doc_falcor_StaticMeshGeometry_indices = R"doc(Triangle indices of the given sub-mesh (CPU-side).)doc";

static const char *__doc_falcor_StaticMeshGeometry_m_dirty_data = R"doc()doc";

static const char *__doc_falcor_StaticMeshGeometry_m_render_geometries = R"doc()doc";

static const char *__doc_falcor_StaticMeshGeometry_m_sub_meshes = R"doc()doc";

static const char *__doc_falcor_StaticMeshGeometry_mark_vertices_dirty =
R"doc(Flag vertex data dirty so update_render_state() re-uploads it on the
next scene update.)doc";

static const char *__doc_falcor_StaticMeshGeometry_mutable_indices = R"doc(Triangle indices of the given sub-mesh (CPU-side, mutable).)doc";

static const char *__doc_falcor_StaticMeshGeometry_mutable_vertices = R"doc(Packed vertices of the given sub-mesh (CPU-side, mutable).)doc";

static const char *__doc_falcor_StaticMeshGeometry_recompute_local_aabb = R"doc(Recompute local bounds from the CPU-side packed vertex positions.)doc";

static const char *__doc_falcor_StaticMeshGeometry_reflect = R"doc(Reflect this class.)doc";

static const char *__doc_falcor_StaticMeshGeometry_reflected_class_name = R"doc()doc";

static const char *__doc_falcor_StaticMeshGeometry_set_mesh_data =
R"doc(Set the mesh data.

Parameter ``desc``:
    Description of the mesh data.)doc";

static const char *__doc_falcor_StaticMeshGeometry_static_class_descriptor = R"doc()doc";

static const char *__doc_falcor_StaticMeshGeometry_sub_mesh_count = R"doc(Number of sub-meshes in this geometry.)doc";

static const char *__doc_falcor_StaticMeshGeometry_update_render_state = R"doc()doc";

static const char *__doc_falcor_StaticMeshGeometry_vertex_count = R"doc(Number of vertices in the given sub-mesh.)doc";

static const char *__doc_falcor_StaticMeshGeometry_vertices = R"doc(Packed vertices of the given sub-mesh (CPU-side).)doc";

static const char *__doc_falcor_TextureFilterMode = R"doc(Filter modes for texture sampling.)doc";

static const char *__doc_falcor_TextureFilterMode_info = R"doc()doc";

static const char *__doc_falcor_TextureFilterMode_linear = R"doc()doc";

static const char *__doc_falcor_TextureFilterMode_nearest = R"doc()doc";

static const char *__doc_falcor_TextureHandle =
R"doc(Handle to a texture managed by TextureManager. This is a reference-
counted handle that automatically manages the lifetime of the
underlying texture/sampler resources. When all handles to a texture
are destroyed, the texture becomes eligible for garbage collection
during the next TextureManager::update() call. Handles can be copied
and moved, and support both regular textures and UDIM texture sets.)doc";

static const char *__doc_falcor_TextureHandle_2 =
R"doc(Handle to a texture managed by TextureManager. This is a reference-
counted handle that automatically manages the lifetime of the
underlying texture/sampler resources. When all handles to a texture
are destroyed, the texture becomes eligible for garbage collection
during the next TextureManager::update() call. Handles can be copied
and moved, and support both regular textures and UDIM texture sets.)doc";

static const char *__doc_falcor_TextureHandle_TextureHandle = R"doc(Default constructor. Creates an invalid handle.)doc";

static const char *__doc_falcor_TextureHandle_TextureHandle_2 = R"doc(Copy constructor.)doc";

static const char *__doc_falcor_TextureHandle_TextureHandle_3 = R"doc(Move constructor.)doc";

static const char *__doc_falcor_TextureHandle_TextureHandle_4 = R"doc()doc";

static const char *__doc_falcor_TextureHandle_UDIMTile = R"doc(Forward declaration of UDIM tile information.)doc";

static const char *__doc_falcor_TextureHandle_data = R"doc(Packed handle data for use on the device.)doc";

static const char *__doc_falcor_TextureHandle_is_finalized =
R"doc(Returns true if the handle is finalized and ready for use. This can be
false if the handle was created with deferred loading.)doc";

static const char *__doc_falcor_TextureHandle_is_udim = R"doc(Returns true if this handle represents a UDIM texture.)doc";

static const char *__doc_falcor_TextureHandle_is_valid = R"doc(Returns true if handle is valid.)doc";

static const char *__doc_falcor_TextureHandle_m_handle_id = R"doc()doc";

static const char *__doc_falcor_TextureHandle_m_texture_manager = R"doc()doc";

static const char *__doc_falcor_TextureHandle_operator_assign = R"doc(Copy assignment.)doc";

static const char *__doc_falcor_TextureHandle_operator_assign_2 = R"doc(Move assignment.)doc";

static const char *__doc_falcor_TextureHandle_operator_bool = R"doc()doc";

static const char *__doc_falcor_TextureHandle_operator_eq = R"doc(Equality operator.)doc";

static const char *__doc_falcor_TextureHandle_operator_ne = R"doc(Inequality operator.)doc";

static const char *__doc_falcor_TextureHandle_path =
R"doc(Absolute path of the original texture file. Only available if the
texture was loaded from a file.)doc";

static const char *__doc_falcor_TextureHandle_sampler = R"doc(Sampler resource.)doc";

static const char *__doc_falcor_TextureHandle_texture =
R"doc(Texture resource. May not be available if the texture is not yet
loaded.)doc";

static const char *__doc_falcor_TextureHandle_to_dictionary =
R"doc(Serialize the handle data to a dictionary.

Parameter ``dict``:
    Dictionary to write to.)doc";

static const char *__doc_falcor_TextureHandle_udim_tiles =
R"doc(List of UDIM tiles referenced by this texture. Each tile contains the
tile index (e.g., 1001), UV coordinates, and a texture handle for
accessing the individual tile texture. Empty if this is not a UDIM
texture.)doc";

static const char *__doc_falcor_TextureHandle_write_to_cursor = R"doc()doc";

static const char *__doc_falcor_TextureHandle_write_to_cursor_2 = R"doc()doc";

static const char *__doc_falcor_TextureHandle_write_to_cursor_3 =
R"doc(Write the handle data to a shader cursor for GPU binding.

Template parameter ``TCursor``:
    Shader cursor type.

Parameter ``cursor``:
    Shader cursor to write to.)doc";

static const char *__doc_falcor_TextureManager =
R"doc(Manages textures and samplers for rendering.

The texture manager handles loading textures from files, registering
existing textures, and creating handles that can be used on the
device. It supports deferred loading, UDIM textures, and automatic
garbage collection of unused resources.

## Bindless Design

The texture manager uses a fully bindless design where textures and
samplers are accessed via descriptor indices rather than bound
resource slots. This allows shaders to access any texture without
rebinding, enabling efficient rendering of scenes with many
materials/textures.

Each TextureHandle is represented on the device as a 32-bit packed
value. The bit layout differs between CUDA and non-CUDA backends:

D3D12/Vulkan: - Bits 0-19: Texture descriptor index (supports up to
~1M textures) - Bits 20-29: Sampler descriptor index (supports up to
~1K samplers) - Bit 31: UDIM flag (indicates the texture is a UDIM
indirection texture)

CUDA: - Bits 0-29: Combined texture/sampler descriptor index - Bit 31:
UDIM flag

CUDA does not support independent sampler objects - texture objects
always have an associated sampler. To handle this, the texture manager
creates a texture view with the required sampler settings, yielding a
combined texture/sampler descriptor. This allows more bits for the
texture index since no separate sampler index is needed.

Shaders use this packed handle to look up the texture (and sampler on
non-CUDA) from global descriptor arrays, allowing texture sampling
without any binding changes.

## UDIM Support

UDIM textures are represented using a two-level indirection scheme: 1.
The main handle points to a small indirection texture (r32_uint
format) 2. Each texel in the indirection texture contains another
packed handle for the actual tile 3. Shaders use the UV coordinates to
index into the indirection texture, then use the resulting handle to
sample the actual tile texture

## Resource Lifetime

Resources are reference-counted through TextureHandle instances. When
all handles to a texture/sampler pair are released, the resources
become eligible for garbage collection during the next update() call.
This allows efficient resource reuse while ensuring resources stay
alive as long as they're needed.

## Thread Safety

TextureManager is currently not thread-safe.)doc";

static const char *__doc_falcor_TextureManager_HandleEntry = R"doc()doc";

static const char *__doc_falcor_TextureManager_HandleEntry_allocated = R"doc()doc";

static const char *__doc_falcor_TextureManager_HandleEntry_atomic_ref_count = R"doc(Returns an atomic reference to ref_count for thread-safe operations.)doc";

static const char *__doc_falcor_TextureManager_HandleEntry_data = R"doc()doc";

static const char *__doc_falcor_TextureManager_HandleEntry_info = R"doc()doc";

static const char *__doc_falcor_TextureManager_HandleEntry_is_finalized = R"doc()doc";

static const char *__doc_falcor_TextureManager_HandleEntry_ref_count =
R"doc(Reference count. Should not be used directly but through
`atomic_ref_count` accessor.)doc";

static const char *__doc_falcor_TextureManager_HandleEntry_texture_view = R"doc(Texture view used on CUDA to get combined texture/sampler descriptor.)doc";

static const char *__doc_falcor_TextureManager_HandleID = R"doc()doc";

static const char *__doc_falcor_TextureManager_HandleID_invalid = R"doc()doc";

static const char *__doc_falcor_TextureManager_HandleInfo = R"doc()doc";

static const char *__doc_falcor_TextureManager_HandleInfo_operator_le = R"doc()doc";

static const char *__doc_falcor_TextureManager_HandleInfo_sampler_id = R"doc()doc";

static const char *__doc_falcor_TextureManager_HandleInfo_texture_id = R"doc()doc";

static const char *__doc_falcor_TextureManager_LoadTextureDesc = R"doc(Parameters for loading a texture from file.)doc";

static const char *__doc_falcor_TextureManager_LoadTextureDesc_generate_mips = R"doc(Generate mipmaps for the texture.)doc";

static const char *__doc_falcor_TextureManager_LoadTextureDesc_load_deferred = R"doc(Defer texture loading until `update()` is called.)doc";

static const char *__doc_falcor_TextureManager_LoadTextureDesc_path = R"doc(Path to the texture file.)doc";

static const char *__doc_falcor_TextureManager_LoadTextureDesc_resolver = R"doc(Asset resolver to use for resolving relative paths (optional).)doc";

static const char *__doc_falcor_TextureManager_LoadTextureDesc_sampler = R"doc(Sampler or sampler configuration (optional).)doc";

static const char *__doc_falcor_TextureManager_LoadTextureDesc_srgb = R"doc(Treat the texture as sRGB.)doc";

static const char *__doc_falcor_TextureManager_LoadTextureDesc_usage = R"doc(Texture usage flags.)doc";

static const char *__doc_falcor_TextureManager_RegisterTextureDesc = R"doc(Parameters for registering a texture.)doc";

static const char *__doc_falcor_TextureManager_RegisterTextureDesc_sampler = R"doc(Sampler or sampler configuration (optional).)doc";

static const char *__doc_falcor_TextureManager_RegisterTextureDesc_texture = R"doc(Texture to register.)doc";

static const char *__doc_falcor_TextureManager_SamplerConfig = R"doc(Configuration for a texture sampler.)doc";

static const char *__doc_falcor_TextureManager_SamplerConfig_address_u = R"doc(Addressing mode for U coordinate.)doc";

static const char *__doc_falcor_TextureManager_SamplerConfig_address_v = R"doc(Addressing mode for V coordinate.)doc";

static const char *__doc_falcor_TextureManager_SamplerConfig_address_w = R"doc(Addressing mode for W coordinate.)doc";

static const char *__doc_falcor_TextureManager_SamplerConfig_border_color = R"doc(Border color used when any address mode is clamp_to_border.)doc";

static const char *__doc_falcor_TextureManager_SamplerConfig_mag_filter = R"doc(Magnification filter.)doc";

static const char *__doc_falcor_TextureManager_SamplerConfig_min_filter = R"doc(Minification filter.)doc";

static const char *__doc_falcor_TextureManager_SamplerConfig_mip_filter = R"doc(Mipmapping filter.)doc";

static const char *__doc_falcor_TextureManager_SamplerConfig_operator_le = R"doc()doc";

static const char *__doc_falcor_TextureManager_SamplerEntry = R"doc()doc";

static const char *__doc_falcor_TextureManager_SamplerEntry_allocated = R"doc()doc";

static const char *__doc_falcor_TextureManager_SamplerEntry_sampler = R"doc()doc";

static const char *__doc_falcor_TextureManager_SamplerEntry_sampler_config = R"doc()doc";

static const char *__doc_falcor_TextureManager_SamplerID = R"doc()doc";

static const char *__doc_falcor_TextureManager_SamplerID_invalid = R"doc()doc";

static const char *__doc_falcor_TextureManager_Stats = R"doc(Statistics.)doc";

static const char *__doc_falcor_TextureManager_Stats_regular_handle_count = R"doc(Number of texture handles representing regular (non-UDIM) textures.)doc";

static const char *__doc_falcor_TextureManager_Stats_regular_texture_count = R"doc(Number of regular (non-UDIM) textures.)doc";

static const char *__doc_falcor_TextureManager_Stats_total_handle_count = R"doc(Total number of texture handles.)doc";

static const char *__doc_falcor_TextureManager_Stats_total_sampler_count = R"doc(Total number of samplers.)doc";

static const char *__doc_falcor_TextureManager_Stats_total_texture_count = R"doc(Total number textures.)doc";

static const char *__doc_falcor_TextureManager_Stats_udim_handle_count = R"doc(Number of texture handles representing UDIM textures.)doc";

static const char *__doc_falcor_TextureManager_Stats_udim_texture_count = R"doc(Number of UDIM indirection textures.)doc";

static const char *__doc_falcor_TextureManager_TextureEntry = R"doc()doc";

static const char *__doc_falcor_TextureManager_TextureEntry_allocated = R"doc()doc";

static const char *__doc_falcor_TextureManager_TextureEntry_info = R"doc()doc";

static const char *__doc_falcor_TextureManager_TextureEntry_is_loaded = R"doc()doc";

static const char *__doc_falcor_TextureManager_TextureEntry_texture = R"doc()doc";

static const char *__doc_falcor_TextureManager_TextureEntry_udim_tiles = R"doc()doc";

static const char *__doc_falcor_TextureManager_TextureID = R"doc()doc";

static const char *__doc_falcor_TextureManager_TextureID_invalid = R"doc()doc";

static const char *__doc_falcor_TextureManager_TextureInfo = R"doc()doc";

static const char *__doc_falcor_TextureManager_TextureInfo_operator_le = R"doc()doc";

static const char *__doc_falcor_TextureManager_TextureInfo_options = R"doc()doc";

static const char *__doc_falcor_TextureManager_TextureInfo_path = R"doc()doc";

static const char *__doc_falcor_TextureManager_TextureInfo_udim_paths = R"doc()doc";

static const char *__doc_falcor_TextureManager_TextureManager = R"doc()doc";

static const char *__doc_falcor_TextureManager_TextureManager_2 = R"doc()doc";

static const char *__doc_falcor_TextureManager_TextureManager_3 = R"doc()doc";

static const char *__doc_falcor_TextureManager_TextureOptions = R"doc()doc";

static const char *__doc_falcor_TextureManager_TextureOptions_generate_mips = R"doc()doc";

static const char *__doc_falcor_TextureManager_TextureOptions_operator_le = R"doc()doc";

static const char *__doc_falcor_TextureManager_TextureOptions_srgb = R"doc()doc";

static const char *__doc_falcor_TextureManager_TextureOptions_usage = R"doc()doc";

static const char *__doc_falcor_TextureManager_UDIMPath = R"doc()doc";

static const char *__doc_falcor_TextureManager_UDIMPath_operator_le = R"doc()doc";

static const char *__doc_falcor_TextureManager_UDIMPath_path = R"doc()doc";

static const char *__doc_falcor_TextureManager_UDIMPath_tile_index = R"doc()doc";

static const char *__doc_falcor_TextureManager_UDIMTile = R"doc()doc";

static const char *__doc_falcor_TextureManager_UDIMTile_handle_id = R"doc()doc";

static const char *__doc_falcor_TextureManager_UDIMTile_tile_index = R"doc()doc";

static const char *__doc_falcor_TextureManager_UDIMTile_tile_u = R"doc()doc";

static const char *__doc_falcor_TextureManager_UDIMTile_tile_v = R"doc()doc";

static const char *__doc_falcor_TextureManager_allocate_handle = R"doc()doc";

static const char *__doc_falcor_TextureManager_allocate_sampler = R"doc()doc";

static const char *__doc_falcor_TextureManager_allocate_texture = R"doc()doc";

static const char *__doc_falcor_TextureManager_class_name = R"doc()doc";

static const char *__doc_falcor_TextureManager_collect_stats = R"doc()doc";

static const char *__doc_falcor_TextureManager_create_udim_indirection_texture = R"doc()doc";

static const char *__doc_falcor_TextureManager_dec_handle_ref =
R"doc(Decrement handle reference count. Handle is garbage collected in next
call to update() if reference count reaches zero.)doc";

static const char *__doc_falcor_TextureManager_default_sampler = R"doc(Default texture sampler.)doc";

static const char *__doc_falcor_TextureManager_device = R"doc(Device associated with the texture manager.)doc";

static const char *__doc_falcor_TextureManager_finalize_handle = R"doc()doc";

static const char *__doc_falcor_TextureManager_finalize_handles = R"doc()doc";

static const char *__doc_falcor_TextureManager_finalize_udim_handle = R"doc()doc";

static const char *__doc_falcor_TextureManager_finalize_udim_handles = R"doc()doc";

static const char *__doc_falcor_TextureManager_free_handle = R"doc()doc";

static const char *__doc_falcor_TextureManager_free_sampler = R"doc()doc";

static const char *__doc_falcor_TextureManager_free_texture = R"doc()doc";

static const char *__doc_falcor_TextureManager_get_handle_data = R"doc()doc";

static const char *__doc_falcor_TextureManager_get_handle_entry = R"doc()doc";

static const char *__doc_falcor_TextureManager_get_handle_entry_2 = R"doc()doc";

static const char *__doc_falcor_TextureManager_get_handle_path = R"doc()doc";

static const char *__doc_falcor_TextureManager_get_handle_sampler = R"doc()doc";

static const char *__doc_falcor_TextureManager_get_handle_texture = R"doc()doc";

static const char *__doc_falcor_TextureManager_get_or_create_handle = R"doc()doc";

static const char *__doc_falcor_TextureManager_get_or_create_sampler = R"doc()doc";

static const char *__doc_falcor_TextureManager_get_or_create_sampler_2 = R"doc()doc";

static const char *__doc_falcor_TextureManager_get_or_create_sampler_3 = R"doc()doc";

static const char *__doc_falcor_TextureManager_get_or_create_texture = R"doc()doc";

static const char *__doc_falcor_TextureManager_get_or_create_texture_2 = R"doc()doc";

static const char *__doc_falcor_TextureManager_get_or_create_udim_texture = R"doc()doc";

static const char *__doc_falcor_TextureManager_get_sampler_entry = R"doc()doc";

static const char *__doc_falcor_TextureManager_get_sampler_entry_2 = R"doc()doc";

static const char *__doc_falcor_TextureManager_get_texture_entry = R"doc()doc";

static const char *__doc_falcor_TextureManager_get_texture_entry_2 = R"doc()doc";

static const char *__doc_falcor_TextureManager_inc_handle_ref = R"doc(Increment handle reference count.)doc";

static const char *__doc_falcor_TextureManager_is_handle_finalized = R"doc()doc";

static const char *__doc_falcor_TextureManager_is_handle_udim = R"doc()doc";

static const char *__doc_falcor_TextureManager_is_texture_loaded = R"doc()doc";

static const char *__doc_falcor_TextureManager_load_texture =
R"doc(Load a texture from file and create a texture handle for it. Uses the
default sampler if no sampler is specified. If the texture/sampler is
already registered, returns the existing handle.

Parameter ``path``:
    Path to the texture file.

Parameter ``resolver``:
    Asset resolver to use for resolving relative paths (optional).

Parameter ``sampler``:
    Sampler or sampler configuration (optional).

Parameter ``generate_mips``:
    Generate mipmaps for the texture.

Parameter ``srgb``:
    Treat the texture as sRGB.

Parameter ``usage``:
    Texture usage flags.

Parameter ``load_deferred``:
    Defer texture loading until `update()` is called.

Returns:
    Texture handle.)doc";

static const char *__doc_falcor_TextureManager_load_texture_2 = R"doc()doc";

static const char *__doc_falcor_TextureManager_load_textures = R"doc()doc";

static const char *__doc_falcor_TextureManager_m_default_asset_resolver =
R"doc(Default asset resolver. Used when no resolver is specified to resolve
relative texture paths.)doc";

static const char *__doc_falcor_TextureManager_m_default_sampler = R"doc()doc";

static const char *__doc_falcor_TextureManager_m_default_sampler_id = R"doc()doc";

static const char *__doc_falcor_TextureManager_m_device = R"doc()doc";

static const char *__doc_falcor_TextureManager_m_free_handle_indices = R"doc()doc";

static const char *__doc_falcor_TextureManager_m_free_sampler_indices = R"doc()doc";

static const char *__doc_falcor_TextureManager_m_free_texture_indices = R"doc()doc";

static const char *__doc_falcor_TextureManager_m_handle_entries = R"doc()doc";

static const char *__doc_falcor_TextureManager_m_handle_info_to_handle_id = R"doc()doc";

static const char *__doc_falcor_TextureManager_m_handles_marked = R"doc()doc";

static const char *__doc_falcor_TextureManager_m_pending_garbage_collect = R"doc()doc";

static const char *__doc_falcor_TextureManager_m_pending_handles_to_finalize = R"doc()doc";

static const char *__doc_falcor_TextureManager_m_pending_textures_to_load = R"doc()doc";

static const char *__doc_falcor_TextureManager_m_pending_udim_handles_to_finalize = R"doc()doc";

static const char *__doc_falcor_TextureManager_m_sampler_config_to_sampler_id = R"doc()doc";

static const char *__doc_falcor_TextureManager_m_sampler_entries = R"doc()doc";

static const char *__doc_falcor_TextureManager_m_sampler_marked = R"doc()doc";

static const char *__doc_falcor_TextureManager_m_sampler_to_sampler_id = R"doc()doc";

static const char *__doc_falcor_TextureManager_m_stats = R"doc()doc";

static const char *__doc_falcor_TextureManager_m_stats_dirty = R"doc()doc";

static const char *__doc_falcor_TextureManager_m_texture_entries = R"doc()doc";

static const char *__doc_falcor_TextureManager_m_texture_info_to_texture_id = R"doc()doc";

static const char *__doc_falcor_TextureManager_m_texture_loader = R"doc()doc";

static const char *__doc_falcor_TextureManager_m_texture_marked = R"doc()doc";

static const char *__doc_falcor_TextureManager_m_texture_to_texture_id = R"doc()doc";

static const char *__doc_falcor_TextureManager_normalize_udim_paths = R"doc()doc";

static const char *__doc_falcor_TextureManager_notify_textures_loaded = R"doc()doc";

static const char *__doc_falcor_TextureManager_operator_assign = R"doc()doc";

static const char *__doc_falcor_TextureManager_operator_assign_2 = R"doc()doc";

static const char *__doc_falcor_TextureManager_register_texture =
R"doc(Register a texture and create a texture handle from it. Uses the
default sampler if no sampler is specified. If the texture/sampler is
already registered, returns the existing handle.

Parameter ``texture``:
    Texture to register.

Parameter ``sampler``:
    Sampler or sampler configuration (optional).

Returns:
    Texture handle.)doc";

static const char *__doc_falcor_TextureManager_run_garbage_collect = R"doc()doc";

static const char *__doc_falcor_TextureManager_stats = R"doc(Statistics. These are only updated during `update()` calls.)doc";

static const char *__doc_falcor_TextureManager_update =
R"doc(Process pending work. This loads any deferred textures, finalizes
pending handles, and runs garbage collection for resources that are no
longer referenced. Should be called once per frame or after a batch of
texture loading operations.)doc";

static const char *__doc_falcor_TextureWrapMode = R"doc(Wrap modes for texture sampling)doc";

static const char *__doc_falcor_TextureWrapMode_clamp_to_edge = R"doc()doc";

static const char *__doc_falcor_TextureWrapMode_info = R"doc()doc";

static const char *__doc_falcor_TextureWrapMode_mirror_repeat = R"doc()doc";

static const char *__doc_falcor_TextureWrapMode_repeat = R"doc()doc";

static const char *__doc_falcor_Transform =
R"doc(Represents a 3D transformation with separate translation, rotation,
and scale components. The transformation matrix is computed on-demand
and cached until any component changes. By default, the composition
order is Scale-Rotate-Translate (SRT).)doc";

static const char *__doc_falcor_TransformAnimation =
R"doc(Transform animation component. Binds animation channels to an entity's
transform (translation, rotation, scale).)doc";

static const char *__doc_falcor_TransformAnimation_animation = R"doc(The animation object containing the channels.)doc";

static const char *__doc_falcor_TransformAnimation_class_descriptor = R"doc()doc";

static const char *__doc_falcor_TransformAnimation_class_name = R"doc()doc";

static const char *__doc_falcor_TransformAnimation_clear_invalid_references = R"doc()doc";

static const char *__doc_falcor_TransformAnimation_has_channels = R"doc(Returns true if any channel is bound.)doc";

static const char *__doc_falcor_TransformAnimation_m_animation = R"doc()doc";

static const char *__doc_falcor_TransformAnimation_m_rotation = R"doc()doc";

static const char *__doc_falcor_TransformAnimation_m_scale = R"doc()doc";

static const char *__doc_falcor_TransformAnimation_m_translation = R"doc()doc";

static const char *__doc_falcor_TransformAnimation_reflect = R"doc(Reflect this class.)doc";

static const char *__doc_falcor_TransformAnimation_reflected_class_name = R"doc()doc";

static const char *__doc_falcor_TransformAnimation_rotation_channel =
R"doc(Index into Animation::channels for rotation (expects quatf), -1 if
none.)doc";

static const char *__doc_falcor_TransformAnimation_scale_channel = R"doc(Index into Animation::channels for scale (expects float3), -1 if none.)doc";

static const char *__doc_falcor_TransformAnimation_set_animation = R"doc()doc";

static const char *__doc_falcor_TransformAnimation_set_rotation_channel = R"doc()doc";

static const char *__doc_falcor_TransformAnimation_set_scale_channel = R"doc()doc";

static const char *__doc_falcor_TransformAnimation_set_translation_channel = R"doc()doc";

static const char *__doc_falcor_TransformAnimation_static_class_descriptor = R"doc()doc";

static const char *__doc_falcor_TransformAnimation_translation_channel =
R"doc(Index into Animation::channels for translation (expects float3), -1 if
none.)doc";

static const char *__doc_falcor_Transform_CompositionOrder =
R"doc(Order in which transformation components are composed into the final
matrix. For example, 'srt' means: first scale, then rotate, then
translate.)doc";

static const char *__doc_falcor_Transform_CompositionOrder_info = R"doc()doc";

static const char *__doc_falcor_Transform_CompositionOrder_rst = R"doc(< rotate -> scale -> translate)doc";

static const char *__doc_falcor_Transform_CompositionOrder_rts = R"doc(< rotate -> translate -> scale)doc";

static const char *__doc_falcor_Transform_CompositionOrder_srt = R"doc(< scale -> rotate -> translate)doc";

static const char *__doc_falcor_Transform_CompositionOrder_str = R"doc(< scale -> translate -> rotate)doc";

static const char *__doc_falcor_Transform_CompositionOrder_trs = R"doc(< translate -> rotate -> scale)doc";

static const char *__doc_falcor_Transform_CompositionOrder_tsr = R"doc(< translate -> scale -> rotate)doc";

static const char *__doc_falcor_Transform_IdentityFlags =
R"doc(Flags indicating which transformation components are identity (no
effect).)doc";

static const char *__doc_falcor_Transform_IdentityFlags_all = R"doc()doc";

static const char *__doc_falcor_Transform_IdentityFlags_none = R"doc()doc";

static const char *__doc_falcor_Transform_IdentityFlags_rotation = R"doc()doc";

static const char *__doc_falcor_Transform_IdentityFlags_scale = R"doc()doc";

static const char *__doc_falcor_Transform_IdentityFlags_translation = R"doc()doc";

static const char *__doc_falcor_Transform_Transform = R"doc(Default constructor. Creates an identity transform.)doc";

static const char *__doc_falcor_Transform_Transform_2 = R"doc(Copy constructor.)doc";

static const char *__doc_falcor_Transform_Transform_3 = R"doc(Constructor with explicit translation, rotation, and scale components.)doc";

static const char *__doc_falcor_Transform_Transform_4 =
R"doc(Constructor from a 4x4 transformation matrix (decomposes into TRS
components). Note: Skew and perspective components are not supported,
the result will be undefined.)doc";

static const char *__doc_falcor_Transform_composition_order = R"doc(Composition order for building the matrix.)doc";

static const char *__doc_falcor_Transform_identity_flags = R"doc(Flags indicating which components are identity.)doc";

static const char *__doc_falcor_Transform_is_identity = R"doc(Check if this is an identity transform.)doc";

static const char *__doc_falcor_Transform_m_composition_order = R"doc()doc";

static const char *__doc_falcor_Transform_m_dirty = R"doc()doc";

static const char *__doc_falcor_Transform_m_identity_flags = R"doc()doc";

static const char *__doc_falcor_Transform_m_matrix = R"doc()doc";

static const char *__doc_falcor_Transform_m_rotation = R"doc()doc";

static const char *__doc_falcor_Transform_m_scale = R"doc()doc";

static const char *__doc_falcor_Transform_m_translation = R"doc()doc";

static const char *__doc_falcor_Transform_mark_dirty = R"doc()doc";

static const char *__doc_falcor_Transform_matrix = R"doc(Transformation matrix (cached and recomputed when dirty).)doc";

static const char *__doc_falcor_Transform_operator_assign = R"doc(Copy assignment operator.)doc";

static const char *__doc_falcor_Transform_reset = R"doc(Reset the transform to identity.)doc";

static const char *__doc_falcor_Transform_rotate = R"doc(Apply a rotation delta (accumulates with existing rotation).)doc";

static const char *__doc_falcor_Transform_rotation = R"doc(Rotation component (quaternion).)doc";

static const char *__doc_falcor_Transform_rotation_xyz = R"doc(Rotation as euler angles in radians, XYZ order.)doc";

static const char *__doc_falcor_Transform_scale = R"doc(Scale component.)doc";

static const char *__doc_falcor_Transform_scale_by = R"doc(Apply a scale factor (multiplies with existing scale).)doc";

static const char *__doc_falcor_Transform_set_composition_order = R"doc(Set the composition order for building the matrix.)doc";

static const char *__doc_falcor_Transform_set_rotation = R"doc(Set the rotation component (quaternion).)doc";

static const char *__doc_falcor_Transform_set_rotation_xyz = R"doc(Set the rotation from euler angles in radians, XYZ order.)doc";

static const char *__doc_falcor_Transform_set_scale = R"doc(Set the scale component.)doc";

static const char *__doc_falcor_Transform_set_translation = R"doc(Set the translation component.)doc";

static const char *__doc_falcor_Transform_to_string = R"doc(Get a string representation of the transform.)doc";

static const char *__doc_falcor_Transform_translate = R"doc(Apply a translation delta (accumulates with existing translation).)doc";

static const char *__doc_falcor_Transform_translation = R"doc(Translation component.)doc";

static const char *__doc_falcor_Transform_update_matrix = R"doc()doc";

static const char *__doc_falcor_TypedID = R"doc(Strongly typed ID.)doc";

static const char *__doc_falcor_TypedID_TypedID = R"doc()doc";

static const char *__doc_falcor_TypedID_TypedID_2 = R"doc()doc";

static const char *__doc_falcor_TypedID_TypedID_3 = R"doc()doc";

static const char *__doc_falcor_TypedID_TypedID_4 = R"doc()doc";

static const char *__doc_falcor_TypedID_index = R"doc()doc";

static const char *__doc_falcor_TypedID_m_index = R"doc()doc";

static const char *__doc_falcor_TypedID_operator_assign = R"doc()doc";

static const char *__doc_falcor_TypedID_operator_assign_2 = R"doc()doc";

static const char *__doc_falcor_TypedID_operator_eq = R"doc()doc";

static const char *__doc_falcor_TypedID_operator_ne = R"doc()doc";

static const char *__doc_falcor_UDIMTile = R"doc(Forward declaration of UDIM tile information.)doc";

static const char *__doc_falcor_UDIMTile_texture_handle = R"doc(Tile texture handle.)doc";

static const char *__doc_falcor_UDIMTile_tile_index = R"doc(UDIM tile index (e.g. 1001, 1002, etc.).)doc";

static const char *__doc_falcor_UDIMTile_tile_u = R"doc(Tile U coordinate (zero-based).)doc";

static const char *__doc_falcor_UDIMTile_tile_v = R"doc(Tile V coordinate (zero-based).)doc";

static const char *__doc_falcor_UsdImporter = R"doc(USD importer.)doc";

static const char *__doc_falcor_UsdImporter_class_name = R"doc()doc";

static const char *__doc_falcor_UsdImporter_load_scene =
R"doc(Load a USD scene from the specified file path.

Parameter ``path``:
    Path to the USD file.

Returns:
    The imported scene.)doc";

static const char *__doc_falcor_any_cast = R"doc()doc";

static const char *__doc_falcor_any_cast_2 = R"doc()doc";

static const char *__doc_falcor_blob_cache_category =
R"doc(Error category for blob cache specific errors.

Returns:
    The process-wide blob cache error category.)doc";

static const char *__doc_falcor_check_type = R"doc(Returns PropertyType for the corresponding type T.)doc";

static const char *__doc_falcor_checked_cast = R"doc()doc";

static const char *__doc_falcor_convert_material = R"doc()doc";

static const char *__doc_falcor_create_component =
R"doc(Create a component of a specific type, attach it to this entity, and
add it to the scene. Components add functionality to entities (e.g.,
GeometryInstance for rendering geometry, Camera for camera behavior).
The component is owned by the scene.

Template parameter ``T``:
    The component type to create (must derive from Component).

Parameter ``props``:
    Optional properties to initialize the component with.

Returns:
    Pointer to the created component (owned by the scene, attached to
    this entity).)doc";

static const char *__doc_falcor_detail_ArchiveBlobViewCodec = R"doc()doc";

static const char *__doc_falcor_detail_ArchiveBlobViewCodec_read_payload = R"doc()doc";

static const char *__doc_falcor_detail_ArchiveBlobViewCodec_write_payload = R"doc()doc";

static const char *__doc_falcor_detail_ArchiveBoolCodec = R"doc()doc";

static const char *__doc_falcor_detail_ArchiveBoolCodec_read_payload = R"doc()doc";

static const char *__doc_falcor_detail_ArchiveBoolCodec_write_payload = R"doc()doc";

static const char *__doc_falcor_detail_ArchiveByteCodec = R"doc()doc";

static const char *__doc_falcor_detail_ArchiveByteCodec_read_payload = R"doc()doc";

static const char *__doc_falcor_detail_ArchiveByteCodec_write_payload = R"doc()doc";

static const char *__doc_falcor_detail_ArchivePathCodec = R"doc()doc";

static const char *__doc_falcor_detail_ArchivePathCodec_read_payload = R"doc()doc";

static const char *__doc_falcor_detail_ArchivePathCodec_write_payload = R"doc()doc";

static const char *__doc_falcor_detail_ArchiveRawFixedCodec = R"doc()doc";

static const char *__doc_falcor_detail_ArchiveRawFixedCodec_read_payload = R"doc()doc";

static const char *__doc_falcor_detail_ArchiveRawFixedCodec_write_payload = R"doc()doc";

static const char *__doc_falcor_detail_ArchiveRawSizedCodec = R"doc()doc";

static const char *__doc_falcor_detail_ArchiveRawSizedCodec_read_payload = R"doc()doc";

static const char *__doc_falcor_detail_ArchiveRawSizedCodec_write_payload = R"doc()doc";

static const char *__doc_falcor_detail_ArchiveStringCodec = R"doc()doc";

static const char *__doc_falcor_detail_ArchiveStringCodec_read_payload = R"doc()doc";

static const char *__doc_falcor_detail_ArchiveStringCodec_write_payload = R"doc()doc";

static const char *__doc_falcor_detail_ArchiveVarsintCodec = R"doc()doc";

static const char *__doc_falcor_detail_ArchiveVarsintCodec_read_payload = R"doc()doc";

static const char *__doc_falcor_detail_ArchiveVarsintCodec_write_payload = R"doc()doc";

static const char *__doc_falcor_detail_ArchiveVaruintCodec = R"doc()doc";

static const char *__doc_falcor_detail_ArchiveVaruintCodec_read_payload = R"doc()doc";

static const char *__doc_falcor_detail_ArchiveVaruintCodec_write_payload = R"doc()doc";

static const char *__doc_falcor_detail_ConnectionControl =
R"doc(Shared control block for a connection. Allows SignalConnection objects
to outlive the Signal they were connected to.)doc";

static const char *__doc_falcor_detail_ConnectionControl_active = R"doc(Flag indicating this slot is active for emission purposes.)doc";

static const char *__doc_falcor_detail_ConnectionControl_connected = R"doc()doc";

static const char *__doc_falcor_detail_ConnectionControl_disconnect_fn = R"doc(Pointer to the disconnect function. Null if signal was destroyed.)doc";

static const char *__doc_falcor_detail_ConnectionControl_slot_id = R"doc()doc";

static const char *__doc_falcor_detail_FNVHashConstants = R"doc()doc";

static const char *__doc_falcor_detail_FNVHashConstants_2 = R"doc()doc";

static const char *__doc_falcor_detail_FNVHashConstants_3 = R"doc()doc";

static const char *__doc_falcor_detail_PackChunkHeader = R"doc()doc";

static const char *__doc_falcor_detail_PackChunkHeader_flags = R"doc()doc";

static const char *__doc_falcor_detail_PackChunkHeader_reserved = R"doc()doc";

static const char *__doc_falcor_detail_PackChunkHeader_size = R"doc()doc";

static const char *__doc_falcor_detail_PackChunkHeader_tag = R"doc()doc";

static const char *__doc_falcor_detail_PackChunkHeader_uncompressed_size = R"doc()doc";

static const char *__doc_falcor_detail_PackChunkHeader_version = R"doc()doc";

static const char *__doc_falcor_detail_PackTocEntry = R"doc()doc";

static const char *__doc_falcor_detail_PackTocEntry_offset = R"doc()doc";

static const char *__doc_falcor_detail_PackTocEntry_tag = R"doc()doc";

static const char *__doc_falcor_detail_PackTocEntry_version = R"doc()doc";

static const char *__doc_falcor_detail_PackTocHeader = R"doc()doc";

static const char *__doc_falcor_detail_PackTocHeader_child_count = R"doc()doc";

static const char *__doc_falcor_detail_PackTocHeader_reserved = R"doc()doc";

static const char *__doc_falcor_detail_PropertyEnumValue =
R"doc(Type-erased enum value that preserves the original enum's type
identity. Stored as part of the PropertyValue variant for enum types.)doc";

static const char *__doc_falcor_detail_PropertyEnumValue_operator_eq = R"doc()doc";

static const char *__doc_falcor_detail_PropertyEnumValue_type = R"doc()doc";

static const char *__doc_falcor_detail_PropertyEnumValue_value = R"doc()doc";

static const char *__doc_falcor_detail_PropertyList =
R"doc(Homogeneous typed list that can be stored as a property value.

Wraps a std::vector<T> together with a PropertyType tag describing the
element type. The element type must be one of the scalar/vector
property types (bool through float4), std::string, or ref<Object>. Any
elements are not supported.)doc";

static const char *__doc_falcor_detail_PropertyList_PropertyList = R"doc()doc";

static const char *__doc_falcor_detail_PropertyList_as_vector = R"doc(Retrieve the stored vector, throwing on element type mismatch.)doc";

static const char *__doc_falcor_detail_PropertyList_create = R"doc(Create a PropertyList from a typed vector.)doc";

static const char *__doc_falcor_detail_PropertyList_dispatch =
R"doc(Dispatch a callable on the typed vector. The callable receives a const
reference to std::vector<T> where T is the element type. Returns the
result of the callable.)doc";

static const char *__doc_falcor_detail_PropertyList_element_type = R"doc(Element type stored in this list.)doc";

static const char *__doc_falcor_detail_PropertyList_empty = R"doc(True if the list contains no elements.)doc";

static const char *__doc_falcor_detail_PropertyList_hash = R"doc(Compute a hash of the list contents.)doc";

static const char *__doc_falcor_detail_PropertyList_m_data = R"doc()doc";

static const char *__doc_falcor_detail_PropertyList_m_element_type = R"doc()doc";

static const char *__doc_falcor_detail_PropertyList_operator_eq = R"doc()doc";

static const char *__doc_falcor_detail_PropertyList_operator_ne = R"doc()doc";

static const char *__doc_falcor_detail_PropertyList_size = R"doc(Number of elements.)doc";

static const char *__doc_falcor_detail_PropertyList_to_string = R"doc(Human-readable string representation.)doc";

static const char *__doc_falcor_detail_PropertyValueTraits = R"doc(PropertyValueTraits template. Used to map types to PropertyType enum.)doc";

static const char *__doc_falcor_detail_PropertyValueTraits_2 = R"doc()doc";

static const char *__doc_falcor_detail_PropertyValueTraits_3 = R"doc()doc";

static const char *__doc_falcor_detail_PropertyValueTraits_4 = R"doc()doc";

static const char *__doc_falcor_detail_PropertyValueTraits_5 = R"doc()doc";

static const char *__doc_falcor_detail_PropertyValueTraits_6 = R"doc()doc";

static const char *__doc_falcor_detail_PropertyValueTraits_7 = R"doc()doc";

static const char *__doc_falcor_detail_PropertyValueTraits_8 = R"doc()doc";

static const char *__doc_falcor_detail_PropertyValueTraits_9 = R"doc()doc";

static const char *__doc_falcor_detail_PropertyValueTraits_10 = R"doc()doc";

static const char *__doc_falcor_detail_PropertyValueTraits_11 = R"doc()doc";

static const char *__doc_falcor_detail_PropertyValueTraits_12 = R"doc()doc";

static const char *__doc_falcor_detail_PropertyValueTraits_13 = R"doc()doc";

static const char *__doc_falcor_detail_PropertyValueTraits_14 = R"doc()doc";

static const char *__doc_falcor_detail_PropertyValueTraits_15 = R"doc()doc";

static const char *__doc_falcor_detail_PropertyValueTraits_16 = R"doc()doc";

static const char *__doc_falcor_detail_PropertyValueTraits_17 = R"doc()doc";

static const char *__doc_falcor_detail_PropertyValueTraits_18 = R"doc()doc";

static const char *__doc_falcor_detail_PropertyValueTraits_19 = R"doc()doc";

static const char *__doc_falcor_detail_PropertyValueTraits_20 = R"doc()doc";

static const char *__doc_falcor_detail_as_vector = R"doc(Retrieve the stored vector, throwing on element type mismatch.)doc";

static const char *__doc_falcor_detail_compare_typeid = R"doc()doc";

static const char *__doc_falcor_detail_create = R"doc(Create a PropertyList from a typed vector.)doc";

static const char *__doc_falcor_detail_dispatch =
R"doc(Dispatch a callable on the typed vector. The callable receives a const
reference to std::vector<T> where T is the element type. Returns the
result of the callable.)doc";

static const char *__doc_falcor_detail_from_storage_value =
R"doc(Convert a storage value back to the user-facing type. Handles range
checking for integers and static_cast for narrowing. Object types are
not handled here (they need dynamic_cast).)doc";

static const char *__doc_falcor_detail_from_storage_vector =
R"doc(Convert a vector of storage values back to user-facing type. Delegates
per-element conversion to from_storage_value for a single source of
truth.)doc";

static const char *__doc_falcor_detail_invoke_adl_archive = R"doc()doc";

static const char *__doc_falcor_detail_invoke_adl_archive_2 = R"doc()doc";

static const char *__doc_falcor_detail_object_type =
R"doc(Extracts the underlying Object-derived type from T or ref<T>. Used
internally by get() and try_get() for dynamic_cast on Object
properties.)doc";

static const char *__doc_falcor_detail_prop_map =
R"doc(Type mapping trait for the Properties container.

Maps user-facing C++ types to the internal storage types used by the
PropertyValue variant. The default template maps to void, which makes
the type ineligible for storage (rejected by the PropertyValueType
concept).)doc";

static const char *__doc_falcor_detail_prop_map_2 = R"doc()doc";

static const char *__doc_falcor_detail_prop_map_3 = R"doc()doc";

static const char *__doc_falcor_detail_prop_map_4 = R"doc()doc";

static const char *__doc_falcor_detail_prop_map_5 = R"doc()doc";

static const char *__doc_falcor_detail_prop_map_6 = R"doc()doc";

static const char *__doc_falcor_detail_prop_map_7 = R"doc()doc";

static const char *__doc_falcor_detail_prop_map_8 = R"doc()doc";

static const char *__doc_falcor_detail_prop_map_9 = R"doc()doc";

static const char *__doc_falcor_detail_prop_map_10 = R"doc()doc";

static const char *__doc_falcor_detail_prop_map_11 = R"doc()doc";

static const char *__doc_falcor_detail_prop_map_12 = R"doc()doc";

static const char *__doc_falcor_detail_prop_map_13 = R"doc()doc";

static const char *__doc_falcor_detail_prop_map_14 = R"doc()doc";

static const char *__doc_falcor_detail_prop_map_15 = R"doc()doc";

static const char *__doc_falcor_detail_prop_map_16 = R"doc()doc";

static const char *__doc_falcor_detail_prop_map_17 = R"doc()doc";

static const char *__doc_falcor_detail_prop_map_18 = R"doc()doc";

static const char *__doc_falcor_detail_prop_map_19 = R"doc()doc";

static const char *__doc_falcor_detail_prop_map_20 = R"doc()doc";

static const char *__doc_falcor_detail_prop_map_21 = R"doc()doc";

static const char *__doc_falcor_detail_skip_archive_payload = R"doc()doc";

static const char *__doc_falcor_detail_throw_enum_type_error = R"doc(Throw an exception when an enum property has an incompatible type.)doc";

static const char *__doc_falcor_detail_to_storage_value =
R"doc(Convert a single value to its property storage type. Handles
path->string, arithmetic widening, and identity forwarding.)doc";

static const char *__doc_falcor_detail_to_storage_vector =
R"doc(Convert a vector of values to property storage type. Delegates per-
element conversion to to_storage_value for a single source of truth.
When types match, the input is moved through. Otherwise elements are
converted individually.)doc";

static const char *__doc_falcor_detail_validate_list_element_count = R"doc(Validate a counted list size before reserving destination storage.)doc";

static const char *__doc_falcor_detail_write_field_header = R"doc()doc";

static const char *__doc_falcor_erase_invalid_objects =
R"doc(Helper function to erase invalid scene objects objects from a vector.

Template parameter ``TObject``:
    Type of scene object.

Parameter ``objects``:
    Vector of pointers to scene objects.)doc";

static const char *__doc_falcor_field =
R"doc(Write a field. Field IDs must be nonzero and strictly increasing
within this record.)doc";

static const char *__doc_falcor_field_2 =
R"doc(Read a field by ID. Returns false when missing and leaves value
unchanged.)doc";

static const char *__doc_falcor_find_enum_info_adl = R"doc()doc";

static const char *__doc_falcor_find_enum_info_adl_2 = R"doc()doc";

static const char *__doc_falcor_find_enum_info_adl_3 = R"doc()doc";

static const char *__doc_falcor_find_enum_info_adl_4 = R"doc()doc";

static const char *__doc_falcor_find_enum_info_adl_5 = R"doc()doc";

static const char *__doc_falcor_find_enum_info_adl_6 = R"doc()doc";

static const char *__doc_falcor_find_enum_info_adl_7 = R"doc()doc";

static const char *__doc_falcor_find_enum_info_adl_8 = R"doc()doc";

static const char *__doc_falcor_find_enum_info_adl_9 = R"doc()doc";

static const char *__doc_falcor_find_enum_info_adl_10 = R"doc()doc";

static const char *__doc_falcor_find_enum_info_adl_11 = R"doc()doc";

static const char *__doc_falcor_find_enum_info_adl_12 = R"doc()doc";

static const char *__doc_falcor_find_enum_info_adl_13 = R"doc()doc";

static const char *__doc_falcor_find_enum_info_adl_14 = R"doc()doc";

static const char *__doc_falcor_find_enum_info_adl_15 = R"doc()doc";

static const char *__doc_falcor_find_enum_info_adl_16 = R"doc()doc";

static const char *__doc_falcor_find_enum_info_adl_17 = R"doc()doc";

static const char *__doc_falcor_find_enum_info_adl_18 = R"doc()doc";

static const char *__doc_falcor_first = R"doc()doc";

static const char *__doc_falcor_first_2 = R"doc()doc";

static const char *__doc_falcor_first_3 = R"doc()doc";

static const char *__doc_falcor_flip_bit = R"doc()doc";

static const char *__doc_falcor_flip_bit_2 = R"doc()doc";

static const char *__doc_falcor_flip_bit_3 = R"doc()doc";

static const char *__doc_falcor_flip_bit_4 = R"doc()doc";

static const char *__doc_falcor_flip_bit_5 = R"doc()doc";

static const char *__doc_falcor_flip_bit_6 = R"doc()doc";

static const char *__doc_falcor_flip_bit_7 = R"doc()doc";

static const char *__doc_falcor_flip_bit_8 = R"doc()doc";

static const char *__doc_falcor_flip_bit_9 = R"doc()doc";

static const char *__doc_falcor_flip_bit_10 = R"doc()doc";

static const char *__doc_falcor_flip_bit_11 = R"doc()doc";

static const char *__doc_falcor_flip_bit_12 = R"doc()doc";

static const char *__doc_falcor_flip_bit_13 = R"doc()doc";

static const char *__doc_falcor_flip_bit_14 = R"doc()doc";

static const char *__doc_falcor_flip_bit_15 = R"doc()doc";

static const char *__doc_falcor_fnv_hash_array32 = R"doc()doc";

static const char *__doc_falcor_fnv_hash_array64 = R"doc()doc";

static const char *__doc_falcor_fourcc_to_string = R"doc(Convert a FourCC tag to a four-character string.)doc";

static const char *__doc_falcor_get =
R"doc(Retrieve a property value by index.

Parameter ``index``:
    Index of the property.

Returns:
    The value of the property.)doc";

static const char *__doc_falcor_get_scene_object_factory =
R"doc(Forward declaration of get function - defined in scene.cpp. This is
used to ensure a single instance of each factory across shared library
boundaries.)doc";

static const char *__doc_falcor_import_scene = R"doc()doc";

static const char *__doc_falcor_importer_cache =
R"doc(Get the global blob cache used by importers. May return nullptr if not
set.)doc";

static const char *__doc_falcor_importer_material_Node = R"doc()doc";

static const char *__doc_falcor_importer_material_NodeConnection = R"doc()doc";

static const char *__doc_falcor_importer_material_NodeConnection_NodeConnection = R"doc()doc";

static const char *__doc_falcor_importer_material_NodeConnection_NodeConnection_2 = R"doc()doc";

static const char *__doc_falcor_importer_material_NodeConnection_m_output = R"doc()doc";

static const char *__doc_falcor_importer_material_NodeConnection_m_source = R"doc()doc";

static const char *__doc_falcor_importer_material_NodeConnection_output = R"doc()doc";

static const char *__doc_falcor_importer_material_NodeConnection_source = R"doc()doc";

static const char *__doc_falcor_importer_material_NodeInput = R"doc()doc";

static const char *__doc_falcor_importer_material_NodeInput_2 = R"doc()doc";

static const char *__doc_falcor_importer_material_NodeInput_NodeInput = R"doc()doc";

static const char *__doc_falcor_importer_material_NodeInput_NodeInput_2 = R"doc()doc";

static const char *__doc_falcor_importer_material_NodeInput_colorspace = R"doc()doc";

static const char *__doc_falcor_importer_material_NodeInput_connection = R"doc()doc";

static const char *__doc_falcor_importer_material_NodeInput_is_valid = R"doc()doc";

static const char *__doc_falcor_importer_material_NodeInput_m_colorspace = R"doc()doc";

static const char *__doc_falcor_importer_material_NodeInput_m_connection = R"doc()doc";

static const char *__doc_falcor_importer_material_NodeInput_m_name = R"doc()doc";

static const char *__doc_falcor_importer_material_NodeInput_m_node = R"doc()doc";

static const char *__doc_falcor_importer_material_NodeInput_m_property = R"doc()doc";

static const char *__doc_falcor_importer_material_NodeInput_m_type = R"doc()doc";

static const char *__doc_falcor_importer_material_NodeInput_name = R"doc()doc";

static const char *__doc_falcor_importer_material_NodeInput_node = R"doc()doc";

static const char *__doc_falcor_importer_material_NodeInput_operator_bool = R"doc()doc";

static const char *__doc_falcor_importer_material_NodeInput_property = R"doc()doc";

static const char *__doc_falcor_importer_material_NodeInput_type = R"doc()doc";

static const char *__doc_falcor_importer_material_NodeNetwork = R"doc()doc";

static const char *__doc_falcor_importer_material_NodeNetwork_2 = R"doc()doc";

static const char *__doc_falcor_importer_material_NodeNetwork_NodeNetwork = R"doc()doc";

static const char *__doc_falcor_importer_material_NodeNetwork_get_node = R"doc()doc";

static const char *__doc_falcor_importer_material_NodeNetwork_m_path_cache = R"doc()doc";

static const char *__doc_falcor_importer_material_NodeNetwork_m_properties = R"doc()doc";

static const char *__doc_falcor_importer_material_NodeNetwork_terminal = R"doc()doc";

static const char *__doc_falcor_importer_material_Node_Node = R"doc()doc";

static const char *__doc_falcor_importer_material_Node_Node_2 = R"doc()doc";

static const char *__doc_falcor_importer_material_Node_begin = R"doc()doc";

static const char *__doc_falcor_importer_material_Node_end = R"doc()doc";

static const char *__doc_falcor_importer_material_Node_get_input = R"doc()doc";

static const char *__doc_falcor_importer_material_Node_is_valid = R"doc()doc";

static const char *__doc_falcor_importer_material_Node_iterator = R"doc()doc";

static const char *__doc_falcor_importer_material_Node_m_network = R"doc()doc";

static const char *__doc_falcor_importer_material_Node_m_props = R"doc()doc";

static const char *__doc_falcor_importer_material_Node_operator_bool = R"doc()doc";

static const char *__doc_falcor_importer_material_Node_props = R"doc()doc";

static const char *__doc_falcor_importer_material_iterator = R"doc()doc";

static const char *__doc_falcor_importer_material_iterator_advance_to_valid = R"doc()doc";

static const char *__doc_falcor_importer_material_iterator_iterator = R"doc()doc";

static const char *__doc_falcor_importer_material_iterator_iterator_2 = R"doc()doc";

static const char *__doc_falcor_importer_material_iterator_m_current = R"doc()doc";

static const char *__doc_falcor_importer_material_iterator_m_end = R"doc()doc";

static const char *__doc_falcor_importer_material_iterator_m_is_end = R"doc()doc";

static const char *__doc_falcor_importer_material_iterator_m_it = R"doc()doc";

static const char *__doc_falcor_importer_material_iterator_m_node = R"doc()doc";

static const char *__doc_falcor_importer_material_iterator_operator_eq = R"doc()doc";

static const char *__doc_falcor_importer_material_iterator_operator_inc = R"doc()doc";

static const char *__doc_falcor_importer_material_iterator_operator_mul = R"doc()doc";

static const char *__doc_falcor_importer_material_iterator_operator_ne = R"doc()doc";

static const char *__doc_falcor_importer_material_iterator_operator_sub = R"doc()doc";

static const char *__doc_falcor_is_set = R"doc()doc";

static const char *__doc_falcor_is_set_2 = R"doc()doc";

static const char *__doc_falcor_is_set_3 = R"doc()doc";

static const char *__doc_falcor_is_set_4 = R"doc()doc";

static const char *__doc_falcor_is_set_5 = R"doc()doc";

static const char *__doc_falcor_is_set_6 = R"doc()doc";

static const char *__doc_falcor_is_set_7 = R"doc()doc";

static const char *__doc_falcor_is_set_8 = R"doc()doc";

static const char *__doc_falcor_is_set_9 = R"doc()doc";

static const char *__doc_falcor_is_set_10 = R"doc()doc";

static const char *__doc_falcor_is_set_11 = R"doc()doc";

static const char *__doc_falcor_is_set_12 = R"doc()doc";

static const char *__doc_falcor_is_set_13 = R"doc()doc";

static const char *__doc_falcor_is_set_14 = R"doc()doc";

static const char *__doc_falcor_is_set_15 = R"doc()doc";

static const char *__doc_falcor_iterator =
R"doc(Forward iterator over the entries of a Properties container.

The iterator dereferences to itself, exposing name(), type(), and
get<T>() accessors on the current property. Tombstone entries (deleted
properties) are automatically skipped.

It is legal to mutate the container while iterating (e.g. removing
entries), but newly added entries may or may not be visited.)doc";

static const char *__doc_falcor_iterator_get = R"doc(Retrieve the current property's value, cast to the requested type.)doc";

static const char *__doc_falcor_iterator_index = R"doc(Internal index of the current property.)doc";

static const char *__doc_falcor_iterator_iterator = R"doc()doc";

static const char *__doc_falcor_iterator_iterator_2 = R"doc()doc";

static const char *__doc_falcor_iterator_m_index = R"doc()doc";

static const char *__doc_falcor_iterator_m_name = R"doc()doc";

static const char *__doc_falcor_iterator_m_props = R"doc()doc";

static const char *__doc_falcor_iterator_m_type = R"doc()doc";

static const char *__doc_falcor_iterator_name = R"doc(Name of the current property.)doc";

static const char *__doc_falcor_iterator_operator_eq = R"doc()doc";

static const char *__doc_falcor_iterator_operator_inc = R"doc()doc";

static const char *__doc_falcor_iterator_operator_inc_2 = R"doc()doc";

static const char *__doc_falcor_iterator_operator_mul = R"doc()doc";

static const char *__doc_falcor_iterator_operator_ne = R"doc()doc";

static const char *__doc_falcor_iterator_operator_sub = R"doc()doc";

static const char *__doc_falcor_iterator_skip_invalid_entries = R"doc()doc";

static const char *__doc_falcor_iterator_to_string = R"doc(String representation of the current property entry.)doc";

static const char *__doc_falcor_iterator_type = R"doc(Type of the current property.)doc";

static const char *__doc_falcor_levenshtein_distance = R"doc()doc";

static const char *__doc_falcor_list =
R"doc(Write a counted list of codec-backed element payloads as a sized
field.)doc";

static const char *__doc_falcor_list_2 = R"doc(Read a counted list of codec-backed element payloads.)doc";

static const char *__doc_falcor_load_importer_scene =
R"doc(Load an importer scene.

Parameter ``scene``:
    Scene to populate.

Parameter ``importer_scene``:
    Importer scene to load from.)doc";

static const char *__doc_falcor_load_scene =
R"doc(Load a scene from disk.

Parameter ``scene``:
    Scene to populate.

Parameter ``path``:
    Path to scene file.)doc";

static const char *__doc_falcor_make_error_code =
R"doc(Make std::error_code from BlobCache::Error.

Parameter ``e``:
    Blob cache error value.

Returns:
    std::error_code using blob_cache_category().)doc";

static const char *__doc_falcor_make_fourcc = R"doc(Create a little-endian FourCC tag from four characters.)doc";

static const char *__doc_falcor_materialx_CodeGen = R"doc()doc";

static const char *__doc_falcor_materialx_CodeGenDesc = R"doc()doc";

static const char *__doc_falcor_materialx_CodeGenDesc_asset_resolver =
R"doc(Optional resolver used to resolve the document and MaterialX search
paths.)doc";

static const char *__doc_falcor_materialx_CodeGenDesc_auto_gamma_image =
R"doc(When true, image nodes with colorN output ask the texture loader for
srgb conversion. Some Houdini-authored assets rely on this, but
MaterialX files with explicit colorspace transforms such as
srgb_texture -> lin_rec709 will be double-decoded if this is also
enabled.)doc";

static const char *__doc_falcor_materialx_CodeGenDesc_auto_transmission =
R"doc(When true, infer transmissive BSDFs from MaterialX node
implementations and scatter modes.)doc";

static const char *__doc_falcor_materialx_CodeGenDesc_compensation_mode =
R"doc(Mx139 microfacet energy-compensation policy used by the MaterialX 1.39
genslangpt backend.)doc";

static const char *__doc_falcor_materialx_CodeGenDesc_document = R"doc(Either the mtlx directly, or a path to it.)doc";

static const char *__doc_falcor_materialx_CodeGenDesc_flip_v_texcoord =
R"doc(Whether to flip the (fractional) part of UVs. True for USD, false for
glTF.)doc";

static const char *__doc_falcor_materialx_CodeGenDesc_geomprop_id_callback =
R"doc(Optional mapping from MaterialX geomprop string/type pairs to stable
shader IDs. Missing mappings intentionally fall back to the legacy
built-in codegen expressions.)doc";

static const char *__doc_falcor_materialx_CodeGenDesc_layering_mode =
R"doc(Mx139 layering implementation used by the MaterialX 1.39 genslangpt
backend.

closure_tree emits the closure tree directly, and bsdf_mix emits a
generated BSDF-mixture root.)doc";

static const char *__doc_falcor_materialx_CodeGenDesc_make_editable = R"doc(When set to true, all unbound parameters of the material are exposed.)doc";

static const char *__doc_falcor_materialx_CodeGenDesc_node_name = R"doc(Name of the node to build from. Will try auto-detect if left empty.)doc";

static const char *__doc_falcor_materialx_CodeGenDesc_optimize_graph_flags = R"doc(Mx139 graph optimizations for the genslangpt backend.)doc";

static const char *__doc_falcor_materialx_CodeGenDesc_payload_size =
R"doc(Size of the payload we can fit into the MaterialPayload. If the data
does not fit, we use a buffer instead.)doc";

static const char *__doc_falcor_materialx_CodeGenDesc_phase_function_params =
R"doc(Phase function: Isotropic, 1 HenyeyGreenstain with a float parameter,
or 2 HG with g0, g1, w0 (param, param, weight of the first one))doc";

static const char *__doc_falcor_materialx_CodeGenDesc_positionfree_layering = R"doc(When true, use position free layering instead of the standard one.)doc";

static const char *__doc_falcor_materialx_CodeGenDesc_sigma_a = R"doc(Optional homogeneous volume properties.)doc";

static const char *__doc_falcor_materialx_CodeGenDesc_sigma_s = R"doc()doc";

static const char *__doc_falcor_materialx_CodeGenDesc_target_color_space_override =
R"doc(Optional target colorspace for generated MaterialX color transforms.
Empty leaves MaterialX to use the active document colorspace.
TODO(mx139): Default MX139 to lin_rec709. Wrapper documents can omit a
root colorspace and then lose explicit srgb_texture -> lin_rec709
shader transforms. Keep auto_gamma_image false for MaterialX files
that emit shader-side transforms.)doc";

static const char *__doc_falcor_materialx_CodeGenDesc_transmissive_bsdfs =
R"doc(List of BSDFs that can transmit through the material.
TODO(@tdavidovic/@aweidlich): Investigate the use the R/T/RT on bsdfs
instead.)doc";

static const char *__doc_falcor_materialx_CodeGenResult = R"doc()doc";

static const char *__doc_falcor_materialx_CodeGenResult_2 = R"doc()doc";

static const char *__doc_falcor_materialx_CodeGenResult_GeometryStreamRequest = R"doc()doc";

static const char *__doc_falcor_materialx_CodeGenResult_GeometryStreamRequest_id = R"doc()doc";

static const char *__doc_falcor_materialx_CodeGenResult_GeometryStreamRequest_name = R"doc()doc";

static const char *__doc_falcor_materialx_CodeGenResult_GeometryStreamRequest_type = R"doc()doc";

static const char *__doc_falcor_materialx_CodeGenResult_all_material_params = R"doc()doc";

static const char *__doc_falcor_materialx_CodeGenResult_codegen_metadata = R"doc()doc";

static const char *__doc_falcor_materialx_CodeGenResult_data_struct_size = R"doc()doc";

static const char *__doc_falcor_materialx_CodeGenResult_has_entry_point_volume_properties = R"doc()doc";

static const char *__doc_falcor_materialx_CodeGenResult_material_data_name = R"doc()doc";

static const char *__doc_falcor_materialx_CodeGenResult_material_instance_byte_size = R"doc()doc";

static const char *__doc_falcor_materialx_CodeGenResult_material_instance_name = R"doc()doc";

static const char *__doc_falcor_materialx_CodeGenResult_material_name = R"doc()doc";

static const char *__doc_falcor_materialx_CodeGenResult_module_name = R"doc()doc";

static const char *__doc_falcor_materialx_CodeGenResult_module_source = R"doc()doc";

static const char *__doc_falcor_materialx_CodeGenResult_needs_mx139_lut_scene_globals = R"doc()doc";

static const char *__doc_falcor_materialx_CodeGenResult_required_geometry_streams = R"doc()doc";

static const char *__doc_falcor_materialx_CodeGenResult_required_geomprop_streams = R"doc()doc";

static const char *__doc_falcor_materialx_CodeGen_discover_material_node_names = R"doc()doc";

static const char *__doc_falcor_materialx_CodeGen_generate = R"doc()doc";

static const char *__doc_falcor_materialx_CompensationMode = R"doc()doc";

static const char *__doc_falcor_materialx_CompensationMode_info = R"doc()doc";

static const char *__doc_falcor_materialx_CompensationMode_no_compensation = R"doc()doc";

static const char *__doc_falcor_materialx_CompensationMode_turquin_analytic = R"doc()doc";

static const char *__doc_falcor_materialx_CompensationMode_turquin_lut = R"doc()doc";

static const char *__doc_falcor_materialx_LayeringMode = R"doc()doc";

static const char *__doc_falcor_materialx_LayeringMode_bsdf_mix = R"doc()doc";

static const char *__doc_falcor_materialx_LayeringMode_closure_tree = R"doc()doc";

static const char *__doc_falcor_materialx_LayeringMode_info = R"doc()doc";

static const char *__doc_falcor_materialx_MxParamInfo = R"doc(Reprensents the MaterialX public parameter.)doc";

static const char *__doc_falcor_materialx_MxParamInfo_MxMaterialInformation = R"doc()doc";

static const char *__doc_falcor_materialx_MxParamInfo_MxMaterialInformation_hasEntryPointVolumeProperties = R"doc()doc";

static const char *__doc_falcor_materialx_MxParamInfo_extract_value = R"doc()doc";

static const char *__doc_falcor_materialx_MxParamInfo_extract_value_2 = R"doc()doc";

static const char *__doc_falcor_materialx_MxParamInfo_holds_value = R"doc()doc";

static const char *__doc_falcor_materialx_MxParamInfo_interface = R"doc()doc";

static const char *__doc_falcor_materialx_MxParamInfo_is_editable =
R"doc(True when the parameter can be changed from outside. False if it is a
compile-time constant.)doc";

static const char *__doc_falcor_materialx_MxParamInfo_mx_type_string = R"doc(Parameter's type in MaterialX system.)doc";

static const char *__doc_falcor_materialx_MxParamInfo_param_name = R"doc(Name of the parameter in the generated Slang code.)doc";

static const char *__doc_falcor_materialx_MxParamInfo_param_type = R"doc(Type of the parameter, if support.)doc";

static const char *__doc_falcor_materialx_MxParamInfo_param_value =
R"doc(Current value of the parameter, only valid for editable parameters
with supported param_type;)doc";

static const char *__doc_falcor_materialx_MxParamInfo_set_cursor = R"doc()doc";

static const char *__doc_falcor_materialx_MxParamInfo_set_value = R"doc()doc";

static const char *__doc_falcor_materialx_MxParamInfo_set_value_2 = R"doc()doc";

static const char *__doc_falcor_materialx_MxParamInfo_set_value_3 = R"doc()doc";

static const char *__doc_falcor_materialx_MxParamInfo_set_value_4 = R"doc()doc";

static const char *__doc_falcor_materialx_MxParamInfo_to_dictionary = R"doc()doc";

static const char *__doc_falcor_materialx_MxParamList = R"doc()doc";

static const char *__doc_falcor_materialx_MxParamList_2 = R"doc()doc";

static const char *__doc_falcor_materialx_MxParamList_add_param_info = R"doc()doc";

static const char *__doc_falcor_materialx_MxParamList_m_editable_count = R"doc()doc";

static const char *__doc_falcor_materialx_MxParamList_m_params = R"doc()doc";

static const char *__doc_falcor_materialx_OptimizeGraphFlags = R"doc()doc";

static const char *__doc_falcor_materialx_OptimizeGraphFlags_all =
R"doc(Set IMxFresnelPolicy to the simplest mode needed (Airy, Color82,
standard).)doc";

static const char *__doc_falcor_materialx_OptimizeGraphFlags_closure_pruning = R"doc(Remove BSDFs, mixes, and layers with provably zero contribution.)doc";

static const char *__doc_falcor_materialx_OptimizeGraphFlags_fresnel_policy =
R"doc(Set IMxFresnelPolicy to the simplest mode needed (Airy, Color82,
standard).)doc";

static const char *__doc_falcor_materialx_OptimizeGraphFlags_info = R"doc()doc";

static const char *__doc_falcor_materialx_OptimizeGraphFlags_none = R"doc()doc";

static const char *__doc_falcor_materialx_OptimizeGraphFlags_static_scatter_mode = R"doc(Set IMxScatterModePolicy as Static rather than Runtime if possible.)doc";

static const char *__doc_falcor_materialx_find_enum_info_adl = R"doc()doc";

static const char *__doc_falcor_materialx_find_enum_info_adl_2 = R"doc()doc";

static const char *__doc_falcor_materialx_find_enum_info_adl_3 = R"doc()doc";

static const char *__doc_falcor_materialx_flip_bit = R"doc()doc";

static const char *__doc_falcor_materialx_is_set = R"doc()doc";

static const char *__doc_falcor_materialx_materialx_1_39_CodeGen_1_39 = R"doc()doc";

static const char *__doc_falcor_materialx_materialx_1_39_CodeGen_1_39_discover_material_node_names = R"doc()doc";

static const char *__doc_falcor_materialx_materialx_1_39_CodeGen_1_39_generate = R"doc()doc";

static const char *__doc_falcor_materialx_materialx_1_39_CodeGen_1_39_load_document = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_AnalysisState = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_AnalysisState_closure_pruning = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_AnalysisState_graph_has_emission = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_AnalysisState_layering = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_AsymmetricMixFoldPlan = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_AsymmetricMixFoldPlan_KeptSide = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_AsymmetricMixFoldPlan_KeptSide_Bg = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_AsymmetricMixFoldPlan_KeptSide_Fg = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_AsymmetricMixFoldPlan_kept_side = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_CallSite = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_CallSiteIndex = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_CallSiteIndex_call_sites_of = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_CallSiteIndex_m_reachable_graphs = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_CallSiteIndex_m_reachable_node_count = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_CallSiteIndex_reachable_graph_count = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_CallSiteIndex_reachable_graphs = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_CallSiteIndex_reachable_node_count = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_CallSite_callee_subgraph = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_CallSite_caller_input = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_CallSite_caller_node = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_CallSite_caller_subgraph = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_ClosurePruningAnalysis = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_ClosurePruningAnalysisResult = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_ClosurePruningAnalysisResult_pruning = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_ClosurePruningAnalysisResult_static_values = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_ClosurePruningAnalysis_asymmetric_mix_fold_plans = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_ClosurePruningAnalysis_classification = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_ClosureRef = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_ClosureRef_ClosureRef = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_ClosureRef_ClosureRef_2 = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_ClosureRef_bsdf = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_ClosureRef_bsdf_index = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_ClosureRef_combiner = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_ClosureRef_combiner_index = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_ClosureRef_from_raw_layering_index = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_ClosureRef_is_bsdf = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_ClosureRef_is_combiner = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_ClosureRef_is_none = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_ClosureRef_m_value = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_ClosureRef_none = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_ClosureRef_operator_le = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_ClosureRef_raw_layering_index = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_ClosureStaticValue = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_ClosureStaticValue_KnownEmpty = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_ClosureStaticValue_KnownNonEmpty = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_ClosureStaticValue_Unknown = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_CodegenInputs = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_CodegenInputs_desc = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_CodegenInputs_document = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_CodegenInputs_result = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_CodegenUserData = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_CodegenUserData_2 = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_CodegenUserData_3 = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_CodegenUserData_analysis = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_CodegenUserData_emit_state = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_CodegenUserData_inputs = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_CombinerMode = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_CombinerMode_add = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_CombinerMode_layer = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_CombinerMode_mix = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_CombinerMode_weighted_layer = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_LayeringDesc = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_LayeringDescBuilder = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_LayeringDescBuilder_build = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_LayeringDesc_BsdfDesc = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_LayeringDesc_BsdfDesc_bsdf_type = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_LayeringDesc_BsdfDesc_copy_source_index = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_LayeringDesc_BsdfDesc_curve_scattering = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_LayeringDesc_BsdfDesc_fresnel_selection = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_LayeringDesc_BsdfDesc_has_mx139_microfacet_selection = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_LayeringDesc_BsdfDesc_index = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_LayeringDesc_BsdfDesc_lobe_types = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_LayeringDesc_BsdfDesc_node_impl = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_LayeringDesc_BsdfDesc_node_path = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_LayeringDesc_BsdfDesc_preserve_tangent_frame = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_LayeringDesc_BsdfDesc_scatter_mode = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_LayeringDesc_BsdfDesc_scatter_selection = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_LayeringDesc_BsdfDesc_shader_node = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_LayeringDesc_BsdfDesc_through_material_transmissive = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_LayeringDesc_BsdfDesc_transmissive = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_LayeringDesc_CombinerDesc = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_LayeringDesc_CombinerDesc_children = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_LayeringDesc_CombinerDesc_copy_source_index = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_LayeringDesc_CombinerDesc_index = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_LayeringDesc_CombinerDesc_mode = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_LayeringDesc_CombinerDesc_node_impl = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_LayeringDesc_CombinerDesc_node_path = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_LayeringDesc_bsdfs = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_LayeringDesc_combiners = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_LayeringDesc_main_layer = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_LutBuffers = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_LutBuffers_dielectric_bothback_energy = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_LutBuffers_dielectric_bothfront_energy = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_LutBuffers_dielectric_reflfront_energy = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_LutBuffers_mini_microfacet_ggx_energy = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_LutBuffers_zeltner_sheen_ltc_param = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_MxDielectricLutBuffers = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_MxDielectricLutBuffers_bothback_energy = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_MxDielectricLutBuffers_bothback_fit_energy = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_MxDielectricLutBuffers_bothfront_energy = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_MxDielectricLutBuffers_reflfront_energy = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_MxFlatBsdfDesc = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_MxFlatBsdfDesc_field_name = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_MxFlatBsdfDesc_layer_pass_through_albedo_mode = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_MxFlatBsdfDesc_layer_pass_through_transmission_mode = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_MxFlatBsdfDesc_type_name = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_MxFlatBsdfDesc_uses_transmission_tint_predicate = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_MxFlatCombinerDesc = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_MxFlatCombinerDesc_branch0_ref = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_MxFlatCombinerDesc_branch1_ref = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_MxFlatCombinerDesc_index = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_MxFlatCombinerDesc_mode = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_MxFlatCombinerMode = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_MxFlatCombinerMode_add = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_MxFlatCombinerMode_layer = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_MxFlatCombinerMode_mix = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_MxFlatLayerPassThroughAlbedoMode = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_MxFlatLayerPassThroughAlbedoMode_reflection = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_MxFlatLayerPassThroughAlbedoMode_scattering = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_MxFlatLayerPassThroughAlbedoMode_transmissive_unity = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_MxFlatLayerPassThroughTransmissionMode = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_MxFlatLayerPassThroughTransmissionMode_absorption = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_MxFlatLayerPassThroughTransmissionMode_absorption_unless_transmissive = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_MxFlatLayerPassThroughTransmissionMode_one = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_MxFlatLayerPassThroughTransmissionMode_physical = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_MxFlatLayerPassThroughTransmissionMode_physical_unless_transmissive = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_MxFlatLayerPassThroughTransmissionMode_zero = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_MxFlatRootDesc = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_MxFlatRootDesc_bsdfs = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_MxFlatRootDesc_combiners = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_MxFlatRootDesc_root_ref = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_MxFlatRootDesc_type_name = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_MxFresnelSelection = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_MxFresnelSelection_airy = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_MxFresnelSelection_color82 = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_MxFresnelSelection_standard = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_MxMicrofacetKind = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_MxMicrofacetKind_conductor = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_MxMicrofacetKind_dielectric = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_MxMicrofacetKind_generalized_schlick = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_MxMicrofacetKind_none = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_MxMicrofacetSelection = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_MxMicrofacetSelection_fresnel = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_MxMicrofacetSelection_kind = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_MxMicrofacetSelection_scatter = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_MxScatterSelection = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_MxScatterSelection_runtime = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_MxScatterSelection_static_r = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_MxScatterSelection_static_rt = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_MxScatterSelection_static_t = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_NodeClassification = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_NodeClassification_Kind = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_NodeClassification_Kind_Keep = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_NodeClassification_Kind_RemoveAsEmpty = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_NodeClassification_Kind_RewireToInput = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_NodeClassification_kind = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_NodeClassification_rewire_input = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_OutputStaticValue = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_OutputStaticValue_closure = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_OutputStaticValue_operator_eq = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_OutputStaticValue_operator_ne = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_OutputStaticValue_scalar = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_PreparedCodegenGraph = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_PreparedCodegenGraph_Impl = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_PreparedCodegenGraph_PreparedCodegenGraph = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_PreparedCodegenGraph_PreparedCodegenGraph_2 = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_PreparedCodegenGraph_PreparedCodegenGraph_3 = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_PreparedCodegenGraph_context = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_PreparedCodegenGraph_graph = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_PreparedCodegenGraph_m_impl = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_PreparedCodegenGraph_operator_assign = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_PreparedCodegenGraph_operator_assign_2 = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_PreparedCodegenGraph_result = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_PreparedCodegenGraph_shader_name = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_PreparedCodegenGraph_user_data = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_ProfileDesc = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_ProfileDesc_bsdf_prefix = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_ProfileDesc_bsdf_type = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_ProfileDesc_display_name = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_ProfileDesc_generated_module_prefix = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_ProfileDesc_generated_prefix = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_ProfileDesc_helper_type = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_ProfileDesc_library_import = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_ProfileDesc_stack_data_type = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_ScalarStaticConstant = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_ScalarStaticConstant_operator_eq = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_ScalarStaticConstant_operator_ne = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_ScalarStaticConstant_size = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_ScalarStaticConstant_values = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_ScalarStaticValue = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_ScalarStaticValueKind = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_ScalarStaticValueKind_KnownConstant = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_ScalarStaticValueKind_KnownOne = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_ScalarStaticValueKind_KnownZero = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_ScalarStaticValueKind_Unknown = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_ScalarStaticValue_constant = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_ScalarStaticValue_kind = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_ScalarStaticValue_one = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_ScalarStaticValue_operator_eq = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_ScalarStaticValue_operator_ne = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_ScalarStaticValue_unknown = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_ScalarStaticValue_value = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_ScalarStaticValue_zero = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_StaticInputQuery = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_StaticInputQuery_2 = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_StaticInputQuery_3 = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_StaticInputQuery_StaticInputQuery = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_StaticInputQuery_StaticInputQuery_2 = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_StaticInputQuery_constant_value = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_StaticInputQuery_input_value = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_StaticInputQuery_is_exact = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_StaticInputQuery_m_constants = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_StaticInputQuery_m_output_static_values = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_StaticValueAnalysisResult = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_StaticValueAnalysisResult_output_static_values = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_analyze_closure_pruning = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_analyze_static_values = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_analyze_static_values_to_fixpoint = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_append_synthetic_opacity_mix = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_apply_closure_simplification = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_bind_lut_globals = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_build_call_site_index = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_classify_layering = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_closure_of = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_codegen_support_SnippetEmitState = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_codegen_support_SnippetEmitState_direct_microfacet_helpers_emitted = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_codegen_support_SnippetEmitState_emitted_snippets = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_codegen_support_ensure_snippet_emitted = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_codegen_support_load_snippet_source = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_codegen_support_mark_generated_void_methods_mutating = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_codegen_support_materialx_element_opacity_is_statically_opaque = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_codegen_support_materialx_element_static_opacity = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_codegen_support_materialx_node_static_opacity = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_codegen_support_materialx_static_float_input_value = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_codegen_support_optional_shader_input_value_string = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_codegen_support_parse_float_list = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_codegen_support_render_snippet = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_codegen_support_replace_all = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_codegen_support_replace_compensation_factory_calls = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_codegen_support_replace_fresnel_factory_calls = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_codegen_support_replace_glsl_compatibility_tokens = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_codegen_support_replace_hw_geom_tokens = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_codegen_support_required_shader_input_value_string = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_codegen_support_sanitize_identifier = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_codegen_support_snake_case_identifier = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_codegen_support_strip_slang_snippet_file_header = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_codegen_support_strip_slang_snippet_module_marker = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_codegen_support_trim = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_codegen_support_trim_trailing_whitespace_lines = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_create_lut_buffers = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_create_lut_scene_globals = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_create_mx_dielectric_lut_buffers = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_effective_optimize_graph_flags = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_emit_bsdf_mix_root_bsdf = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_emitter_ClosureTreeMaterialStructsDesc = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_emitter_ClosureTreeMaterialStructsDesc_codegen_desc = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_emitter_ClosureTreeMaterialStructsDesc_graph_has_emission = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_emitter_ClosureTreeMaterialStructsDesc_graph_name = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_emitter_ClosureTreeMaterialStructsDesc_layering = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_emitter_ClosureTreeMaterialStructsDesc_material_data_name = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_emitter_ClosureTreeMaterialStructsDesc_material_instance_name = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_emitter_ClosureTreeMaterialStructsDesc_material_name = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_emitter_ClosureTreeMaterialStructsDesc_materialx_payload_size = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_emitter_ClosureTreeMaterialStructsDesc_profile = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_emitter_CompositionAliases = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_emitter_CompositionAliases_AliasDesc = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_emitter_CompositionAliases_AliasDesc_name = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_emitter_CompositionAliases_AliasDesc_type = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_emitter_CompositionAliases_aliases = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_emitter_CompositionAliases_core_aliases = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_emitter_CompositionAliases_namespace_name = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_emitter_CompositionAliases_uses_weighted_helpers = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_emitter_CompositionAliases_weighted_aliases = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_emitter_MaterialStructsDesc = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_emitter_MaterialStructsDesc_codegen_desc = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_emitter_MaterialStructsDesc_graph_has_emission = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_emitter_MaterialStructsDesc_graph_name = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_emitter_MaterialStructsDesc_layering = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_emitter_MaterialStructsDesc_material_data_name = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_emitter_MaterialStructsDesc_material_instance_name = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_emitter_MaterialStructsDesc_material_name = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_emitter_MaterialStructsDesc_materialx_payload_size = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_emitter_RootStrategy = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_emitter_RootStrategy_RootValueBlock = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_emitter_RootStrategy_RootValueBlock_lines = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_emitter_RootStrategy_RootValueBlock_root_expr = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_emitter_RootStrategy_StackParams = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_emitter_RootStrategy_StackParams_use_stack_frames = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_emitter_RootStrategy_build_aliases = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_emitter_RootStrategy_build_collect_extra_text = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_emitter_RootStrategy_build_root_value_text = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_emitter_RootStrategy_emit_root_types = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_emitter_RootStrategy_emit_root_value = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_emitter_RootStrategy_stack_params = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_emitter_alias_reference = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_emitter_build_bsdf_mix_composition_aliases = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_emitter_build_bsdf_mix_composition_value_text = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_emitter_build_closure_tree_composition_value_text = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_emitter_build_composition_aliases = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_emitter_build_synthetic_opacity_stack_setup_text = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_emitter_combiner_value_prefix = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_emitter_emit_basic_stack_data = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_emitter_emit_bsdf_mix_composition_value = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_emitter_emit_bsdf_mix_root_bsdf_type = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_emitter_emit_closure_tree_composition_aliases = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_emitter_emit_closure_tree_composition_value = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_emitter_emit_closure_tree_material_structs = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_emitter_emit_material_data = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_emitter_emit_material_structs = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_emitter_emit_module_header = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_emitter_emit_synthetic_opacity_stack_setup = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_emitter_graph_emission_expr = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_emitter_graph_opacity_expr = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_emitter_graph_output_can_emit = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_emitter_graph_thin_walled_expr = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_emitter_make_root_strategy = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_emitter_root_curve_scattering = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_emitter_root_lobes = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_emitter_root_transmissive = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_fresnel_bsdf_type = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_generate_code = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_genslangpt_GenslangPtStringTypeSyntax = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_genslangpt_GenslangPtStringTypeSyntax_GenslangPtStringTypeSyntax = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_genslangpt_GenslangPtStringTypeSyntax_getValue = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_genslangpt_GenslangPtSyntax = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_genslangpt_GenslangPtSyntax_GenslangPtSyntax = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_genslangpt_GenslangPtSyntax_remapEnumeration = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_genslangpt_create_geomcolor_node = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_genslangpt_create_geomprop_value_node = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_genslangpt_create_microfacet_conductor_node = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_genslangpt_create_microfacet_dielectric_node = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_genslangpt_create_microfacet_generalized_schlick_node = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_genslangpt_create_texcoord_node = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_genslangpt_geomprop_id_const_name = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_genslangpt_is_builtin_geomprop = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_genslangpt_resolve_geomprop_id = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_genslangpt_shader_output_type_name = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_glossy_scatter_lobe_types = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_graph_prepare_ClosurePruningCounters = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_graph_prepare_ClosurePruningCounters_asymmetric_mix_rewrites = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_graph_prepare_ClosurePruningCounters_effective_enabled = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_graph_prepare_ClosurePruningCounters_iterations = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_graph_prepare_ClosurePruningCounters_nodes_pruned = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_graph_prepare_ClosurePruningCounters_output_rewrites = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_graph_prepare_ClosurePruningCounters_requested_enabled = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_graph_prepare_closure_pruning_enabled_to_string = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_graph_prepare_collect_reachable_core_nodes = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_graph_prepare_materialize_reachable_default_geomprop_inputs = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_graph_prepare_prepare_graph = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_graph_prepare_write_closure_pruning_report_fields = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_graph_prepare_write_closure_pruning_report_tsv = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_implementation_name_starts_with = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_input_static_value = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_input_static_values_detail_all_components_equal = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_input_static_values_detail_all_components_equal_2 = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_input_static_values_detail_all_components_equal_3 = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_input_static_values_detail_all_components_equal_4 = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_input_static_values_detail_all_components_equal_5 = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_input_static_values_detail_all_components_equal_6 = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_input_static_values_detail_lift_typed_static_value = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_input_static_values_detail_pack_scalar_static_constant = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_input_static_values_detail_pack_scalar_static_constant_2 = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_input_static_values_detail_pack_scalar_static_constant_3 = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_input_static_values_detail_pack_scalar_static_constant_4 = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_input_static_values_detail_pack_scalar_static_constant_5 = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_input_static_values_detail_pack_scalar_static_constant_6 = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_is_add_float_node = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_is_any_closure_type = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_is_connected_layer_vdf_boundary = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_is_constant_node = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_is_dot_node = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_is_empty_closure = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_is_invert_float_node = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_is_known_non_empty = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_is_layer_vdf_node = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_is_one_static_value = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_is_scalar_or_color_multiply_node = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_is_scatter_mode_bsdf = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_is_supported_weight_value_type = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_is_surface_closure_type = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_is_unity_layer_pass_through_transmission_bsdf = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_is_unknown_static_value = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_is_weighted_layer_node = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_is_zero_static_value = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_lift_static_value = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_lower_weighted_layers_to_layers_for_emission = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_matches_surface_closure_op = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_materialx_test_geomprop_id = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_merge_closure_values = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_merge_output_static_values = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_merge_scalar_values = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_microfacet_bsdf_type = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_microfacet_kind = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_node_implementation_starts_with = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_operator_eq = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_operator_ne = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_output_static_value_or_unknown = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_profile_desc = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_root_prepare_RootPrepareResult = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_root_prepare_RootPrepareResult_root_element = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_root_prepare_prepare_root = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_run_closure_pruning = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_scalar_of = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_scalar_of_2 = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_scatter_mode_includes_transmission = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_scatter_selection_mode_string = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_select_mx139_microfacet = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_shader_node_implementation_name = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_socket_default_static_value = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_string_starts_with = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_synthetic_opacity_bsdf_index = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_synthetic_opacity_combiner_index = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_use_synthetic_opacity_mix = R"doc()doc";

static const char *__doc_falcor_materialx_mx139_validate_weighted_layer_top_refs_are_not_shared = R"doc()doc";

static const char *__doc_falcor_materialx_operator_band = R"doc()doc";

static const char *__doc_falcor_materialx_operator_bnot = R"doc()doc";

static const char *__doc_falcor_materialx_operator_bor = R"doc()doc";

static const char *__doc_falcor_materialx_operator_iand = R"doc()doc";

static const char *__doc_falcor_materialx_operator_ior = R"doc()doc";

static const char *__doc_falcor_materialx_params_TextureInfo =
R"doc(Contains all the settings MaterialX uses to describe desired texture
loading.)doc";

static const char *__doc_falcor_materialx_params_TextureInfo_border_color =
R"doc(MaterialX image default value, used as sampler border color for
constant address mode.)doc";

static const char *__doc_falcor_materialx_params_TextureInfo_file_name = R"doc(< Filename of the texture to load)doc";

static const char *__doc_falcor_materialx_params_TextureInfo_filtertype = R"doc(< Filter mode used for texture sampler creation.)doc";

static const char *__doc_falcor_materialx_params_TextureInfo_frameendaction = R"doc(< Used for per-frame (animated) textures. IGNORED.)doc";

static const char *__doc_falcor_materialx_params_TextureInfo_frameoffset = R"doc(< Used for per-frame (animated) textures. IGNORED.)doc";

static const char *__doc_falcor_materialx_params_TextureInfo_framerange = R"doc(< Used for per-frame (animated) textures. IGNORED.)doc";

static const char *__doc_falcor_materialx_params_TextureInfo_is_color_output_type =
R"doc(ColorX outputs should read as srgb for LDR images, while the rest read
as is in file.)doc";

static const char *__doc_falcor_materialx_params_TextureInfo_layer = R"doc(< Layered within the texture. IGNORED)doc";

static const char *__doc_falcor_materialx_params_TextureInfo_mx_input_name = R"doc(< Name of the materialx input)doc";

static const char *__doc_falcor_materialx_params_TextureInfo_texture_handle = R"doc(< Value of the handle)doc";

static const char *__doc_falcor_materialx_params_TextureInfo_uaddressmode = R"doc(< U-direction wrap mode used for texture sampler creation.)doc";

static const char *__doc_falcor_materialx_params_TextureInfo_vaddressmode = R"doc(< V-direction wrap mode used for texture sampler creation.)doc";

static const char *__doc_falcor_materialx_params_Type = R"doc(The type of the parameter data)doc";

static const char *__doc_falcor_materialx_params_TypeTraits = R"doc()doc";

static const char *__doc_falcor_materialx_params_TypeTraits_2 = R"doc()doc";

static const char *__doc_falcor_materialx_params_TypeTraits_3 = R"doc()doc";

static const char *__doc_falcor_materialx_params_TypeTraits_4 = R"doc()doc";

static const char *__doc_falcor_materialx_params_TypeTraits_5 = R"doc()doc";

static const char *__doc_falcor_materialx_params_TypeTraits_6 = R"doc()doc";

static const char *__doc_falcor_materialx_params_TypeTraits_7 = R"doc()doc";

static const char *__doc_falcor_materialx_params_TypeTraits_8 = R"doc()doc";

static const char *__doc_falcor_materialx_params_TypeTraits_9 = R"doc()doc";

static const char *__doc_falcor_materialx_params_TypeTraits_10 = R"doc()doc";

static const char *__doc_falcor_materialx_params_TypeTraits_extract_value = R"doc()doc";

static const char *__doc_falcor_materialx_params_TypeTraits_extract_value_2 = R"doc()doc";

static const char *__doc_falcor_materialx_params_TypeTraits_extract_value_3 = R"doc()doc";

static const char *__doc_falcor_materialx_params_TypeTraits_extract_value_4 = R"doc()doc";

static const char *__doc_falcor_materialx_params_TypeTraits_extract_value_5 = R"doc()doc";

static const char *__doc_falcor_materialx_params_TypeTraits_extract_value_6 = R"doc()doc";

static const char *__doc_falcor_materialx_params_TypeTraits_extract_value_7 = R"doc()doc";

static const char *__doc_falcor_materialx_params_TypeTraits_extract_value_8 = R"doc()doc";

static const char *__doc_falcor_materialx_params_TypeTraits_extract_value_9 = R"doc()doc";

static const char *__doc_falcor_materialx_params_TypeTraits_extract_value_10 = R"doc()doc";

static const char *__doc_falcor_materialx_params_TypeTraits_extract_value_11 = R"doc()doc";

static const char *__doc_falcor_materialx_params_TypeTraits_extract_value_12 = R"doc()doc";

static const char *__doc_falcor_materialx_params_TypeTraits_extract_value_13 = R"doc()doc";

static const char *__doc_falcor_materialx_params_TypeTraits_extract_value_14 = R"doc()doc";

static const char *__doc_falcor_materialx_params_TypeTraits_extract_value_15 = R"doc()doc";

static const char *__doc_falcor_materialx_params_TypeTraits_extract_value_16 = R"doc()doc";

static const char *__doc_falcor_materialx_params_TypeTraits_extract_value_17 = R"doc()doc";

static const char *__doc_falcor_materialx_params_TypeTraits_extract_value_18 = R"doc()doc";

static const char *__doc_falcor_materialx_params_Type_bool = R"doc()doc";

static const char *__doc_falcor_materialx_params_Type_float = R"doc()doc";

static const char *__doc_falcor_materialx_params_Type_float2 = R"doc()doc";

static const char *__doc_falcor_materialx_params_Type_float3 = R"doc()doc";

static const char *__doc_falcor_materialx_params_Type_float4 = R"doc()doc";

static const char *__doc_falcor_materialx_params_Type_float4x4 = R"doc()doc";

static const char *__doc_falcor_materialx_params_Type_int = R"doc()doc";

static const char *__doc_falcor_materialx_params_Type_texture = R"doc()doc";

static const char *__doc_falcor_materialx_params_Type_unsupported = R"doc(< Type that is not supported by the UI (strings, arrays))doc";

static const char *__doc_falcor_materialx_params_UserInterfaceInfo = R"doc(@Describe user interface information for a parameter)doc";

static const char *__doc_falcor_materialx_params_UserInterfaceInfo_doc = R"doc(< Doc string)doc";

static const char *__doc_falcor_materialx_params_UserInterfaceInfo_folders = R"doc(< Folder nesting of the parameter in the UI, always starts with Params)doc";

static const char *__doc_falcor_materialx_params_UserInterfaceInfo_name = R"doc(< UI name of the parameter (prefer to display this))doc";

static const char *__doc_falcor_materialx_params_UserInterfaceInfo_operator_eq = R"doc()doc";

static const char *__doc_falcor_materialx_params_UserInterfaceInfo_operator_lt = R"doc()doc";

static const char *__doc_falcor_mikkt_generate_tangent_space = R"doc(Generate tangents for mesh vertices using MikkTSpace algorithm.)doc";

static const char *__doc_falcor_mikkt_generate_tangent_space_2 =
R"doc(Generate tangents with caching. On cache hit, tangents are read from
cache. On cache miss, tangents are computed and stored in cache.)doc";

static const char *__doc_falcor_ndir_to_oct_snorm =
R"doc(Converts normalized direction to the octahedral map (non-equal area,
signed normalized).

Parameter ``n``:
    Normalized direction.

Returns:
    Position in octahedral map in [-1,1] for each component.)doc";

static const char *__doc_falcor_ndir_to_oct_unorm =
R"doc(Converts normalized direction to the octahedral map (non-equal area,
unsigned normalized).

Parameter ``n``:
    Normalized direction.

Returns:
    Position in octahedral map in [0,1] for each component.)doc";

static const char *__doc_falcor_ngx_DLSSDOptimalSettings = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSDOptimalSettings_max_render_height = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSDOptimalSettings_max_render_width = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSDOptimalSettings_min_render_height = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSDOptimalSettings_min_render_width = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSDOptimalSettings_render_height = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSDOptimalSettings_render_width = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSDOptimalSettings_sharpness = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSGCameraDesc = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSGCameraDesc_automode_override_reset = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSGCameraDesc_camera_aspect_ratio = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSGCameraDesc_camera_far = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSGCameraDesc_camera_fov = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSGCameraDesc_camera_fwd = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSGCameraDesc_camera_motion_included = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSGCameraDesc_camera_near = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSGCameraDesc_camera_pinhole_offset = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSGCameraDesc_camera_pos = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSGCameraDesc_camera_right = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSGCameraDesc_camera_up = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSGCameraDesc_camera_view_to_clip = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSGCameraDesc_clip_to_camera_view = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSGCameraDesc_clip_to_lens_clip = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSGCameraDesc_clip_to_prev_clip = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSGCameraDesc_color_buffers_hdr = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSGCameraDesc_depth_inverted = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSGCameraDesc_jitter_offset = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSGCameraDesc_menu_detection_enabled = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSGCameraDesc_motion_vector_scale = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSGCameraDesc_motion_vectors_dilated = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSGCameraDesc_motion_vectors_invalid_value = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSGCameraDesc_multi_frame_count = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSGCameraDesc_multi_frame_index = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSGCameraDesc_not_rendering_game_frames = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSGCameraDesc_ortho_projection = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSGCameraDesc_prev_clip_to_clip = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSGCameraDesc_reset = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSGEvaluateDesc = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSGEvaluateDesc_backbuffer = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSGEvaluateDesc_camera = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSGEvaluateDesc_depth = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSGEvaluateDesc_motion_vectors = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSGEvaluateDesc_output_interpolated_frame = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSGEvaluateResources = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSGEvaluateResources_backbuffer = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSGEvaluateResources_camera = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSGEvaluateResources_depth = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSGEvaluateResources_motion_vectors = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSGEvaluateResources_output_interpolated_frame = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSGFeature = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSGFeature_2 = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSGFeatureDesc = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSGFeatureDesc_backbuffer_format = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSGFeatureDesc_height = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSGFeatureDesc_render_height = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSGFeatureDesc_render_width = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSGFeatureDesc_width = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSGFeature_DLSSGFeature =
R"doc(Create a DLSS Frame Generation feature from an initialized NGX
context.)doc";

static const char *__doc_falcor_ngx_DLSSGFeature_Impl = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSGFeature_class_name = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSGFeature_desc = R"doc(Return the feature creation descriptor.)doc";

static const char *__doc_falcor_ngx_DLSSGFeature_device = R"doc(Return the SGL device owned by the parent NGX context.)doc";

static const char *__doc_falcor_ngx_DLSSGFeature_evaluate =
R"doc(Submit DLSS Frame Generation evaluation on a short-lived command
buffer.)doc";

static const char *__doc_falcor_ngx_DLSSGFeature_evaluate_2 =
R"doc(Record a DLSS Frame Generation evaluation into an existing command
encoder.)doc";

static const char *__doc_falcor_ngx_DLSSGFeature_last_evaluate_result =
R"doc(Return and clear the NGX result from the most recently executed
evaluation callback, or Result::none if none has executed.)doc";

static const char *__doc_falcor_ngx_DLSSGFeature_m_desc = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSGFeature_m_impl = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSGFeature_m_ngx = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSGFeature_ngx = R"doc(Return the parent NGX context kept alive by this feature.)doc";

static const char *__doc_falcor_ngx_DLSSGFeature_throw_if_last_evaluate_failed = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSOptimalSettings = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSOptimalSettings_max_render_height = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSOptimalSettings_max_render_width = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSOptimalSettings_min_render_height = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSOptimalSettings_min_render_width = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSOptimalSettings_render_height = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSOptimalSettings_render_width = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSOptimalSettings_sharpness = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSRREvaluateDesc = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSRREvaluateDesc_clip_from_view = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSRREvaluateDesc_color = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSRREvaluateDesc_depth = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSRREvaluateDesc_diffuse_albedo = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSRREvaluateDesc_exposure_scale = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSRREvaluateDesc_frame_time_delta_ms = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSRREvaluateDesc_jitter_offset_x = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSRREvaluateDesc_jitter_offset_y = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSRREvaluateDesc_motion_vector_scale_x = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSRREvaluateDesc_motion_vector_scale_y = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSRREvaluateDesc_motion_vectors = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSRREvaluateDesc_motion_vectors_reflections = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSRREvaluateDesc_normals = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSRREvaluateDesc_output = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSRREvaluateDesc_pre_exposure = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSRREvaluateDesc_reset = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSRREvaluateDesc_roughness = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSRREvaluateDesc_specular_albedo = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSRREvaluateDesc_specular_hit_distance = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSRREvaluateDesc_view_from_world = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSRREvaluateResources = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSRREvaluateResources_clip_from_view = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSRREvaluateResources_color = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSRREvaluateResources_depth = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSRREvaluateResources_diffuse_albedo = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSRREvaluateResources_exposure_scale = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSRREvaluateResources_frame_time_delta_ms = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSRREvaluateResources_jitter_offset_x = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSRREvaluateResources_jitter_offset_y = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSRREvaluateResources_motion_vector_scale_x = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSRREvaluateResources_motion_vector_scale_y = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSRREvaluateResources_motion_vectors = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSRREvaluateResources_motion_vectors_reflections = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSRREvaluateResources_normals = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSRREvaluateResources_output = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSRREvaluateResources_pre_exposure = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSRREvaluateResources_reset = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSRREvaluateResources_roughness = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSRREvaluateResources_specular_albedo = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSRREvaluateResources_specular_hit_distance = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSRREvaluateResources_view_from_world = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSRRFeature = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSRRFeature_2 = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSRRFeatureDesc = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSRRFeatureDesc_quality = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSRRFeatureDesc_render_height = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSRRFeatureDesc_render_width = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSRRFeatureDesc_target_height = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSRRFeatureDesc_target_width = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSRRFeature_DLSSRRFeature =
R"doc(Create a DLSS Ray Reconstruction feature from an initialized NGX
context.)doc";

static const char *__doc_falcor_ngx_DLSSRRFeature_Impl = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSRRFeature_class_name = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSRRFeature_desc = R"doc(Return the feature creation descriptor.)doc";

static const char *__doc_falcor_ngx_DLSSRRFeature_device = R"doc(Return the SGL device owned by the parent NGX context.)doc";

static const char *__doc_falcor_ngx_DLSSRRFeature_evaluate =
R"doc(Submit DLSS Ray Reconstruction evaluation on a short-lived command
buffer.)doc";

static const char *__doc_falcor_ngx_DLSSRRFeature_evaluate_2 =
R"doc(Record a DLSS Ray Reconstruction evaluation into an existing command
encoder.)doc";

static const char *__doc_falcor_ngx_DLSSRRFeature_last_evaluate_result =
R"doc(Return and clear the NGX result from the most recently executed
evaluation callback, or Result::none if none has executed.)doc";

static const char *__doc_falcor_ngx_DLSSRRFeature_m_desc = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSRRFeature_m_impl = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSRRFeature_m_ngx = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSRRFeature_ngx = R"doc(Return the parent NGX context kept alive by this feature.)doc";

static const char *__doc_falcor_ngx_DLSSRRFeature_throw_if_last_evaluate_failed = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSSREvaluateDesc = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSSREvaluateDesc_color = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSSREvaluateDesc_depth = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSSREvaluateDesc_exposure = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSSREvaluateDesc_exposure_scale = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSSREvaluateDesc_jitter_offset_x = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSSREvaluateDesc_jitter_offset_y = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSSREvaluateDesc_motion_vector_scale_x = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSSREvaluateDesc_motion_vector_scale_y = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSSREvaluateDesc_motion_vectors = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSSREvaluateDesc_output = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSSREvaluateDesc_pre_exposure = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSSREvaluateDesc_render_subrect_height = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSSREvaluateDesc_render_subrect_width = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSSREvaluateDesc_reset = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSSREvaluateResources = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSSREvaluateResources_color = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSSREvaluateResources_depth = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSSREvaluateResources_exposure = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSSREvaluateResources_exposure_scale = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSSREvaluateResources_jitter_offset_x = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSSREvaluateResources_jitter_offset_y = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSSREvaluateResources_motion_vector_scale_x = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSSREvaluateResources_motion_vector_scale_y = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSSREvaluateResources_motion_vectors = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSSREvaluateResources_output = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSSREvaluateResources_pre_exposure = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSSREvaluateResources_render_subrect_height = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSSREvaluateResources_render_subrect_width = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSSREvaluateResources_reset = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSSRFeature = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSSRFeature_2 = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSSRFeatureDesc = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSSRFeatureDesc_auto_exposure = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSSRFeatureDesc_depth_inverted = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSSRFeatureDesc_enable_output_subrects = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSSRFeatureDesc_is_hdr = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSSRFeatureDesc_motion_vectors_jittered = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSSRFeatureDesc_quality = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSSRFeatureDesc_render_height = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSSRFeatureDesc_render_width = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSSRFeatureDesc_sharpness = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSSRFeatureDesc_target_height = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSSRFeatureDesc_target_width = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSSRFeature_DLSSSRFeature =
R"doc(Create a DLSS Super Resolution feature from an initialized NGX
context.)doc";

static const char *__doc_falcor_ngx_DLSSSRFeature_Impl = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSSRFeature_class_name = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSSRFeature_desc = R"doc(Return the feature creation descriptor.)doc";

static const char *__doc_falcor_ngx_DLSSSRFeature_device = R"doc(Return the SGL device owned by the parent NGX context.)doc";

static const char *__doc_falcor_ngx_DLSSSRFeature_evaluate =
R"doc(Submit DLSS Super Resolution evaluation on a short-lived command
buffer.)doc";

static const char *__doc_falcor_ngx_DLSSSRFeature_evaluate_2 =
R"doc(Record a DLSS Super Resolution evaluation into an existing command
encoder.)doc";

static const char *__doc_falcor_ngx_DLSSSRFeature_last_evaluate_result =
R"doc(Return and clear the NGX result from the most recently executed
evaluation callback, or Result::none if none has executed.)doc";

static const char *__doc_falcor_ngx_DLSSSRFeature_m_desc = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSSRFeature_m_impl = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSSRFeature_m_ngx = R"doc()doc";

static const char *__doc_falcor_ngx_DLSSSRFeature_ngx = R"doc(Return the parent NGX context kept alive by this feature.)doc";

static const char *__doc_falcor_ngx_DLSSSRFeature_throw_if_last_evaluate_failed = R"doc()doc";

static const char *__doc_falcor_ngx_Impl = R"doc()doc";

static const char *__doc_falcor_ngx_Impl_2 = R"doc()doc";

static const char *__doc_falcor_ngx_Impl_3 = R"doc()doc";

static const char *__doc_falcor_ngx_Impl_4 = R"doc()doc";

static const char *__doc_falcor_ngx_Impl_Impl = R"doc(Store the parent NGX context and immutable creation descriptor.)doc";

static const char *__doc_falcor_ngx_Impl_Impl_2 = R"doc(Store the parent NGX context and immutable creation descriptor.)doc";

static const char *__doc_falcor_ngx_Impl_Impl_3 = R"doc(Store the parent NGX context and immutable creation descriptor.)doc";

static const char *__doc_falcor_ngx_Impl_Impl_4 = R"doc(Store the parent device for use by backend-specific implementations.)doc";

static const char *__doc_falcor_ngx_Impl_app_data_path = R"doc(Return the writable app-data directory used by NGX.)doc";

static const char *__doc_falcor_ngx_Impl_create = R"doc(Create the backend-specific implementation for the device type.)doc";

static const char *__doc_falcor_ngx_Impl_device = R"doc(Return the SGL device that owns the native graphics/CUDA context.)doc";

static const char *__doc_falcor_ngx_Impl_effective_render_height = R"doc()doc";

static const char *__doc_falcor_ngx_Impl_effective_render_width = R"doc()doc";

static const char *__doc_falcor_ngx_Impl_evaluate_render_subrect_height = R"doc()doc";

static const char *__doc_falcor_ngx_Impl_evaluate_render_subrect_width = R"doc()doc";

static const char *__doc_falcor_ngx_Impl_get_dlss_optimal_settings =
R"doc(Validate support and query backend-specific DLSS Super Resolution
optimal settings.)doc";

static const char *__doc_falcor_ngx_Impl_get_dlssd_optimal_settings = R"doc(Validate support and query backend-specific DLSS-D optimal settings.)doc";

static const char *__doc_falcor_ngx_Impl_get_last_evaluate_result =
R"doc(Return and clear the result from the most recently executed evaluation
callback.)doc";

static const char *__doc_falcor_ngx_Impl_get_last_evaluate_result_2 =
R"doc(Return and clear the result from the most recently executed evaluation
callback.)doc";

static const char *__doc_falcor_ngx_Impl_get_last_evaluate_result_3 =
R"doc(Return and clear the result from the most recently executed evaluation
callback.)doc";

static const char *__doc_falcor_ngx_Impl_info = R"doc(Return capability information populated during backend initialization.)doc";

static const char *__doc_falcor_ngx_Impl_last_evaluate_result = R"doc(Return the result from the most recently executed evaluation callback.)doc";

static const char *__doc_falcor_ngx_Impl_last_evaluate_result_2 = R"doc(Return the result from the most recently executed evaluation callback.)doc";

static const char *__doc_falcor_ngx_Impl_last_evaluate_result_3 = R"doc(Return the result from the most recently executed evaluation callback.)doc";

static const char *__doc_falcor_ngx_Impl_m_desc = R"doc()doc";

static const char *__doc_falcor_ngx_Impl_m_desc_2 = R"doc()doc";

static const char *__doc_falcor_ngx_Impl_m_desc_3 = R"doc()doc";

static const char *__doc_falcor_ngx_Impl_m_device = R"doc()doc";

static const char *__doc_falcor_ngx_Impl_m_info = R"doc()doc";

static const char *__doc_falcor_ngx_Impl_m_last_evaluate_result = R"doc()doc";

static const char *__doc_falcor_ngx_Impl_m_last_evaluate_result_2 = R"doc()doc";

static const char *__doc_falcor_ngx_Impl_m_last_evaluate_result_3 = R"doc()doc";

static const char *__doc_falcor_ngx_Impl_m_ngx = R"doc()doc";

static const char *__doc_falcor_ngx_Impl_m_ngx_2 = R"doc()doc";

static const char *__doc_falcor_ngx_Impl_m_ngx_3 = R"doc()doc";

static const char *__doc_falcor_ngx_Impl_m_ngx_device = R"doc()doc";

static const char *__doc_falcor_ngx_Impl_make_create_params = R"doc(Build the common DLSS Frame Generation feature creation parameters.)doc";

static const char *__doc_falcor_ngx_Impl_make_create_params_2 = R"doc(Build the common DLSS-D/RR feature creation parameters.)doc";

static const char *__doc_falcor_ngx_Impl_make_create_params_3 = R"doc(Build the common DLSS Super Resolution feature creation parameters.)doc";

static const char *__doc_falcor_ngx_Impl_make_feature_common_info =
R"doc(Build shared NGX feature metadata, including the runtime search path
list.)doc";

static const char *__doc_falcor_ngx_Impl_make_opt_eval_params = R"doc(Build the optional DLSS Frame Generation camera/evaluation parameters.)doc";

static const char *__doc_falcor_ngx_Impl_ngx_device = R"doc(Return the backend-native NGX device handle.)doc";

static const char *__doc_falcor_ngx_Impl_prepare_evaluate_resources = R"doc(Transition resources into the states expected by NGX evaluation.)doc";

static const char *__doc_falcor_ngx_Impl_prepare_evaluate_resources_2 = R"doc(Transition resources into the states expected by NGX evaluation.)doc";

static const char *__doc_falcor_ngx_Impl_prepare_evaluate_resources_3 = R"doc(Transition resources into the states expected by NGX evaluation.)doc";

static const char *__doc_falcor_ngx_Impl_query_dlss_optimal_settings = R"doc(Backend hook for querying DLSS Super Resolution optimal settings.)doc";

static const char *__doc_falcor_ngx_Impl_query_dlss_optimal_settings_from_params =
R"doc(Query DLSS Super Resolution optimal settings from a valid NGX
capability parameter block.)doc";

static const char *__doc_falcor_ngx_Impl_query_dlssd_optimal_settings = R"doc(Backend hook for querying DLSS-D optimal settings.)doc";

static const char *__doc_falcor_ngx_Impl_query_dlssd_optimal_settings_from_params =
R"doc(Query DLSS-D optimal settings from a valid NGX capability parameter
block.)doc";

static const char *__doc_falcor_ngx_Impl_read_dlss_capability_params =
R"doc(Read DLSS Super Resolution capability values from an NGX parameter
block.)doc";

static const char *__doc_falcor_ngx_Impl_read_dlssd_capability_params = R"doc(Read DLSSD capability values from an NGX parameter block.)doc";

static const char *__doc_falcor_ngx_Impl_read_frame_generation_capability_params =
R"doc(Read DLSS Frame Generation capability values from an NGX parameter
block.)doc";

static const char *__doc_falcor_ngx_Impl_record_evaluate =
R"doc(Record backend-specific NGX evaluation work into an existing command
encoder.)doc";

static const char *__doc_falcor_ngx_Impl_record_evaluate_2 =
R"doc(Record backend-specific NGX evaluation work into an existing command
encoder.)doc";

static const char *__doc_falcor_ngx_Impl_record_evaluate_3 =
R"doc(Record backend-specific NGX evaluation work into an existing command
encoder.)doc";

static const char *__doc_falcor_ngx_Impl_reset_last_evaluate_result = R"doc()doc";

static const char *__doc_falcor_ngx_Impl_reset_last_evaluate_result_2 = R"doc()doc";

static const char *__doc_falcor_ngx_Impl_reset_last_evaluate_result_3 = R"doc()doc";

static const char *__doc_falcor_ngx_Impl_retain_evaluate_resources =
R"doc(Retain all resources referenced by an evaluation descriptor until the
callback retires.)doc";

static const char *__doc_falcor_ngx_Impl_retain_evaluate_resources_2 =
R"doc(Retain all resources referenced by an evaluation descriptor until the
callback retires.)doc";

static const char *__doc_falcor_ngx_Impl_retain_evaluate_resources_3 =
R"doc(Retain all resources referenced by an evaluation descriptor until the
callback retires.)doc";

static const char *__doc_falcor_ngx_Impl_set_last_evaluate_result = R"doc()doc";

static const char *__doc_falcor_ngx_Impl_set_last_evaluate_result_2 = R"doc()doc";

static const char *__doc_falcor_ngx_Impl_set_last_evaluate_result_3 = R"doc()doc";

static const char *__doc_falcor_ngx_Impl_to_wstring_path = R"doc(Convert a filesystem path to the wide string form expected by NGX.)doc";

static const char *__doc_falcor_ngx_Impl_validate_evaluate_desc =
R"doc(Validate that an evaluation descriptor is complete and matches the
feature/device.)doc";

static const char *__doc_falcor_ngx_Impl_validate_evaluate_desc_2 =
R"doc(Validate that an evaluation descriptor is complete and matches the
feature/device.)doc";

static const char *__doc_falcor_ngx_Impl_validate_evaluate_desc_3 =
R"doc(Validate that an evaluation descriptor is complete and matches the
feature/device.)doc";

static const char *__doc_falcor_ngx_Impl_validate_feature_desc =
R"doc(Validate feature dimensions and capability state before creating an
NGX feature.)doc";

static const char *__doc_falcor_ngx_Impl_validate_feature_desc_2 =
R"doc(Validate feature dimensions and capability state before creating an
NGX feature.)doc";

static const char *__doc_falcor_ngx_Impl_validate_feature_desc_3 =
R"doc(Validate feature dimensions and capability state before creating an
NGX feature.)doc";

static const char *__doc_falcor_ngx_NGX = R"doc()doc";

static const char *__doc_falcor_ngx_NGX_2 = R"doc()doc";

static const char *__doc_falcor_ngx_NGX_3 = R"doc()doc";

static const char *__doc_falcor_ngx_NGX_4 = R"doc()doc";

static const char *__doc_falcor_ngx_NGXInfo = R"doc()doc";

static const char *__doc_falcor_ngx_NGXInfo_dlss_available = R"doc()doc";

static const char *__doc_falcor_ngx_NGXInfo_dlss_feature_init_result = R"doc()doc";

static const char *__doc_falcor_ngx_NGXInfo_dlss_minimum_driver_version_major = R"doc()doc";

static const char *__doc_falcor_ngx_NGXInfo_dlss_minimum_driver_version_minor = R"doc()doc";

static const char *__doc_falcor_ngx_NGXInfo_dlss_needs_updated_driver = R"doc()doc";

static const char *__doc_falcor_ngx_NGXInfo_dlss_supported = R"doc()doc";

static const char *__doc_falcor_ngx_NGXInfo_dlssd_available = R"doc()doc";

static const char *__doc_falcor_ngx_NGXInfo_dlssd_feature_init_result = R"doc()doc";

static const char *__doc_falcor_ngx_NGXInfo_dlssd_minimum_driver_version_major = R"doc()doc";

static const char *__doc_falcor_ngx_NGXInfo_dlssd_minimum_driver_version_minor = R"doc()doc";

static const char *__doc_falcor_ngx_NGXInfo_dlssd_needs_updated_driver = R"doc()doc";

static const char *__doc_falcor_ngx_NGXInfo_dlssd_supported = R"doc()doc";

static const char *__doc_falcor_ngx_NGXInfo_frame_generation_available = R"doc()doc";

static const char *__doc_falcor_ngx_NGXInfo_frame_generation_feature_init_result = R"doc()doc";

static const char *__doc_falcor_ngx_NGXInfo_frame_generation_minimum_driver_version_major = R"doc()doc";

static const char *__doc_falcor_ngx_NGXInfo_frame_generation_minimum_driver_version_minor = R"doc()doc";

static const char *__doc_falcor_ngx_NGXInfo_frame_generation_multi_frame_count_max = R"doc()doc";

static const char *__doc_falcor_ngx_NGXInfo_frame_generation_needs_updated_driver = R"doc()doc";

static const char *__doc_falcor_ngx_NGXInfo_frame_generation_supported = R"doc()doc";

static const char *__doc_falcor_ngx_NGXInfo_required_vulkan_device_extensions = R"doc()doc";

static const char *__doc_falcor_ngx_NGXInfo_required_vulkan_instance_extensions = R"doc()doc";

static const char *__doc_falcor_ngx_NGX_Impl = R"doc()doc";

static const char *__doc_falcor_ngx_NGX_NGX = R"doc(Create and initialize an NGX context for the supplied SGL device.)doc";

static const char *__doc_falcor_ngx_NGX_class_name = R"doc()doc";

static const char *__doc_falcor_ngx_NGX_create_dlss_g_feature = R"doc(Create a DLSS Frame Generation feature owned by this NGX context.)doc";

static const char *__doc_falcor_ngx_NGX_create_dlss_rr_feature = R"doc(Create a DLSS Ray Reconstruction feature owned by this NGX context.)doc";

static const char *__doc_falcor_ngx_NGX_create_dlss_sr_feature = R"doc(Create a DLSS Super Resolution feature owned by this NGX context.)doc";

static const char *__doc_falcor_ngx_NGX_device = R"doc(Return the SGL device used to initialize NGX.)doc";

static const char *__doc_falcor_ngx_NGX_get =
R"doc(Return the cached NGX context for the supplied SGL device, creating it
if needed.)doc";

static const char *__doc_falcor_ngx_NGX_get_dlss_optimal_settings =
R"doc(Query NGX for the recommended DLSS Super Resolution render size for an
output size and quality mode.)doc";

static const char *__doc_falcor_ngx_NGX_get_dlssd_optimal_settings =
R"doc(Query NGX for the recommended DLSS-D render size for an output size
and quality mode.)doc";

static const char *__doc_falcor_ngx_NGX_info =
R"doc(Return NGX capability information that can be queried without creating
a feature.)doc";

static const char *__doc_falcor_ngx_NGX_m_device = R"doc()doc";

static const char *__doc_falcor_ngx_NGX_m_impl = R"doc()doc";

static const char *__doc_falcor_ngx_QualityMode = R"doc()doc";

static const char *__doc_falcor_ngx_QualityMode_balanced = R"doc()doc";

static const char *__doc_falcor_ngx_QualityMode_dlaa = R"doc()doc";

static const char *__doc_falcor_ngx_QualityMode_info = R"doc()doc";

static const char *__doc_falcor_ngx_QualityMode_performance = R"doc()doc";

static const char *__doc_falcor_ngx_QualityMode_quality = R"doc()doc";

static const char *__doc_falcor_ngx_QualityMode_ultra_performance = R"doc()doc";

static const char *__doc_falcor_ngx_Result = R"doc()doc";

static const char *__doc_falcor_ngx_Result_fail = R"doc()doc";

static const char *__doc_falcor_ngx_Result_fail_denied = R"doc()doc";

static const char *__doc_falcor_ngx_Result_fail_feature_already_exists = R"doc()doc";

static const char *__doc_falcor_ngx_Result_fail_feature_not_found = R"doc()doc";

static const char *__doc_falcor_ngx_Result_fail_feature_not_supported = R"doc()doc";

static const char *__doc_falcor_ngx_Result_fail_invalid_parameter = R"doc()doc";

static const char *__doc_falcor_ngx_Result_fail_missing_input = R"doc()doc";

static const char *__doc_falcor_ngx_Result_fail_not_implemented = R"doc()doc";

static const char *__doc_falcor_ngx_Result_fail_not_initialized = R"doc()doc";

static const char *__doc_falcor_ngx_Result_fail_out_of_date = R"doc()doc";

static const char *__doc_falcor_ngx_Result_fail_out_of_gpu_memory = R"doc()doc";

static const char *__doc_falcor_ngx_Result_fail_platform_error = R"doc()doc";

static const char *__doc_falcor_ngx_Result_fail_rw_flag_missing = R"doc()doc";

static const char *__doc_falcor_ngx_Result_fail_scratch_buffer_too_small = R"doc()doc";

static const char *__doc_falcor_ngx_Result_fail_unable_to_initialize_feature = R"doc()doc";

static const char *__doc_falcor_ngx_Result_fail_unable_to_write_to_app_data_path = R"doc()doc";

static const char *__doc_falcor_ngx_Result_fail_unsupported_format = R"doc()doc";

static const char *__doc_falcor_ngx_Result_fail_unsupported_input_format = R"doc()doc";

static const char *__doc_falcor_ngx_Result_fail_unsupported_parameter = R"doc()doc";

static const char *__doc_falcor_ngx_Result_info = R"doc()doc";

static const char *__doc_falcor_ngx_Result_none = R"doc()doc";

static const char *__doc_falcor_ngx_Result_success = R"doc()doc";

static const char *__doc_falcor_ngx_find_enum_info_adl = R"doc()doc";

static const char *__doc_falcor_ngx_find_enum_info_adl_2 = R"doc()doc";

static const char *__doc_falcor_ngx_get_ngx_impl = R"doc(Return the backend implementation owned by an NGX context.)doc";

static const char *__doc_falcor_ngx_get_vulkan_pre_device_info =
R"doc(Query Vulkan extensions that NGX requires before an SGL Vulkan device
is created.)doc";

static const char *__doc_falcor_ngx_make_dlss_g_feature_impl = R"doc()doc";

static const char *__doc_falcor_ngx_make_dlss_rr_feature_impl = R"doc()doc";

static const char *__doc_falcor_ngx_make_dlss_sr_feature_impl = R"doc()doc";

static const char *__doc_falcor_ngx_ngx_result_to_string = R"doc(Convert common NGX result codes to readable diagnostic strings.)doc";

static const char *__doc_falcor_ngx_query_vulkan_required_extensions =
R"doc(Query Vulkan instance/device extensions required before creating a
Vulkan NGX context.)doc";

static const char *__doc_falcor_ngx_to_ngx_perf_quality = R"doc(Convert the public Falcor2 quality mode to the NGX quality enum.)doc";

static const char *__doc_falcor_ngx_to_result = R"doc(Convert an NGX result code to the public Falcor2 result enum.)doc";

static const char *__doc_falcor_oct_to_ndir_snorm =
R"doc(Converts point in the octahedral map to normalized direction (non-
equal area, signed normalized).

Parameter ``p``:
    Position in octahedral map in [-1,1] for each component.

Returns:
    Normalized direction.)doc";

static const char *__doc_falcor_oct_to_ndir_unorm =
R"doc(Converts point in the octahedral map to normalized direction (non-
equal area, unsigned normalized).

Parameter ``p``:
    Position in octahedral map in [0,1] for each component.

Returns:
    Normalized direction.)doc";

static const char *__doc_falcor_oct_wrap =
R"doc(Helper function to reflect the folds of the lower hemisphere over the
diagonals in the octahedral map.)doc";

static const char *__doc_falcor_operator_RenderHandle = R"doc()doc";

static const char *__doc_falcor_operator_array = R"doc()doc";

static const char *__doc_falcor_operator_array_2 = R"doc()doc";

static const char *__doc_falcor_operator_array_3 = R"doc()doc";

static const char *__doc_falcor_operator_band = R"doc()doc";

static const char *__doc_falcor_operator_band_2 = R"doc()doc";

static const char *__doc_falcor_operator_band_3 = R"doc()doc";

static const char *__doc_falcor_operator_band_4 = R"doc()doc";

static const char *__doc_falcor_operator_band_5 = R"doc()doc";

static const char *__doc_falcor_operator_band_6 = R"doc()doc";

static const char *__doc_falcor_operator_band_7 = R"doc()doc";

static const char *__doc_falcor_operator_band_8 = R"doc()doc";

static const char *__doc_falcor_operator_band_9 = R"doc()doc";

static const char *__doc_falcor_operator_band_10 = R"doc()doc";

static const char *__doc_falcor_operator_band_11 = R"doc()doc";

static const char *__doc_falcor_operator_band_12 = R"doc()doc";

static const char *__doc_falcor_operator_band_13 = R"doc()doc";

static const char *__doc_falcor_operator_band_14 = R"doc()doc";

static const char *__doc_falcor_operator_band_15 = R"doc()doc";

static const char *__doc_falcor_operator_bnot = R"doc()doc";

static const char *__doc_falcor_operator_bnot_2 = R"doc()doc";

static const char *__doc_falcor_operator_bnot_3 = R"doc()doc";

static const char *__doc_falcor_operator_bnot_4 = R"doc()doc";

static const char *__doc_falcor_operator_bnot_5 = R"doc()doc";

static const char *__doc_falcor_operator_bnot_6 = R"doc()doc";

static const char *__doc_falcor_operator_bnot_7 = R"doc()doc";

static const char *__doc_falcor_operator_bnot_8 = R"doc()doc";

static const char *__doc_falcor_operator_bnot_9 = R"doc()doc";

static const char *__doc_falcor_operator_bnot_10 = R"doc()doc";

static const char *__doc_falcor_operator_bnot_11 = R"doc()doc";

static const char *__doc_falcor_operator_bnot_12 = R"doc()doc";

static const char *__doc_falcor_operator_bnot_13 = R"doc()doc";

static const char *__doc_falcor_operator_bnot_14 = R"doc()doc";

static const char *__doc_falcor_operator_bnot_15 = R"doc()doc";

static const char *__doc_falcor_operator_bor = R"doc()doc";

static const char *__doc_falcor_operator_bor_2 = R"doc()doc";

static const char *__doc_falcor_operator_bor_3 = R"doc()doc";

static const char *__doc_falcor_operator_bor_4 = R"doc()doc";

static const char *__doc_falcor_operator_bor_5 = R"doc()doc";

static const char *__doc_falcor_operator_bor_6 = R"doc()doc";

static const char *__doc_falcor_operator_bor_7 = R"doc()doc";

static const char *__doc_falcor_operator_bor_8 = R"doc()doc";

static const char *__doc_falcor_operator_bor_9 = R"doc()doc";

static const char *__doc_falcor_operator_bor_10 = R"doc()doc";

static const char *__doc_falcor_operator_bor_11 = R"doc()doc";

static const char *__doc_falcor_operator_bor_12 = R"doc()doc";

static const char *__doc_falcor_operator_bor_13 = R"doc()doc";

static const char *__doc_falcor_operator_bor_14 = R"doc()doc";

static const char *__doc_falcor_operator_bor_15 = R"doc()doc";

static const char *__doc_falcor_operator_iand = R"doc()doc";

static const char *__doc_falcor_operator_iand_2 = R"doc()doc";

static const char *__doc_falcor_operator_iand_3 = R"doc()doc";

static const char *__doc_falcor_operator_iand_4 = R"doc()doc";

static const char *__doc_falcor_operator_iand_5 = R"doc()doc";

static const char *__doc_falcor_operator_iand_6 = R"doc()doc";

static const char *__doc_falcor_operator_iand_7 = R"doc()doc";

static const char *__doc_falcor_operator_iand_8 = R"doc()doc";

static const char *__doc_falcor_operator_iand_9 = R"doc()doc";

static const char *__doc_falcor_operator_iand_10 = R"doc()doc";

static const char *__doc_falcor_operator_iand_11 = R"doc()doc";

static const char *__doc_falcor_operator_iand_12 = R"doc()doc";

static const char *__doc_falcor_operator_iand_13 = R"doc()doc";

static const char *__doc_falcor_operator_iand_14 = R"doc()doc";

static const char *__doc_falcor_operator_iand_15 = R"doc()doc";

static const char *__doc_falcor_operator_ior = R"doc()doc";

static const char *__doc_falcor_operator_ior_2 = R"doc()doc";

static const char *__doc_falcor_operator_ior_3 = R"doc()doc";

static const char *__doc_falcor_operator_ior_4 = R"doc()doc";

static const char *__doc_falcor_operator_ior_5 = R"doc()doc";

static const char *__doc_falcor_operator_ior_6 = R"doc()doc";

static const char *__doc_falcor_operator_ior_7 = R"doc()doc";

static const char *__doc_falcor_operator_ior_8 = R"doc()doc";

static const char *__doc_falcor_operator_ior_9 = R"doc()doc";

static const char *__doc_falcor_operator_ior_10 = R"doc()doc";

static const char *__doc_falcor_operator_ior_11 = R"doc()doc";

static const char *__doc_falcor_operator_ior_12 = R"doc()doc";

static const char *__doc_falcor_operator_ior_13 = R"doc()doc";

static const char *__doc_falcor_operator_ior_14 = R"doc()doc";

static const char *__doc_falcor_operator_ior_15 = R"doc()doc";

static const char *__doc_falcor_optix_AlphaMode = R"doc()doc";

static const char *__doc_falcor_optix_AlphaMode_copy = R"doc(Copy alpha (if present) from input layer, no denoising.)doc";

static const char *__doc_falcor_optix_AlphaMode_denoise = R"doc(Denoise alpha.)doc";

static const char *__doc_falcor_optix_AlphaMode_info = R"doc()doc";

static const char *__doc_falcor_optix_DenoiserAOVType =
R"doc(Type of AOV, used in temporal AOV modes as a hint to improve image
quality.)doc";

static const char *__doc_falcor_optix_DenoiserAOVType_beauty = R"doc()doc";

static const char *__doc_falcor_optix_DenoiserAOVType_diffuse = R"doc()doc";

static const char *__doc_falcor_optix_DenoiserAOVType_info = R"doc()doc";

static const char *__doc_falcor_optix_DenoiserAOVType_none = R"doc()doc";

static const char *__doc_falcor_optix_DenoiserAOVType_reflection = R"doc()doc";

static const char *__doc_falcor_optix_DenoiserAOVType_refraction = R"doc()doc";

static const char *__doc_falcor_optix_DenoiserAOVType_specular = R"doc()doc";

static const char *__doc_falcor_optix_DenoiserGuideLayer = R"doc(Guide layers to assist denoising.)doc";

static const char *__doc_falcor_optix_DenoiserGuideLayer_albedo = R"doc(Albedo image with three components: R, G, B.)doc";

static const char *__doc_falcor_optix_DenoiserGuideLayer_flow =
R"doc(Flow image with two components: X, Y. Pixel movement from previous to
current frame for each pixel in screen space.)doc";

static const char *__doc_falcor_optix_DenoiserGuideLayer_flow_trustworthiness =
R"doc(Image with a single component value that specifies how trustworthy the
flow vector at x,y position in OptixDenoiserGuideLayer::flow is. Range
0..1 (low->high trustworthiness). Ignored if data pointer in the image
is zero.)doc";

static const char *__doc_falcor_optix_DenoiserGuideLayer_normal =
R"doc(Normal image with two or three components: X, Y, Z. (X, Y) camera
space for OPTIX_DENOISER_MODEL_KIND_LDR, OPTIX_DENOISER_MODEL_KIND_HDR
models. (X, Y, Z) world space, all other models.)doc";

static const char *__doc_falcor_optix_DenoiserGuideLayer_output_internal_guide_layer =
R"doc(Internal image used in temporal AOV denoising modes, pixel format
OPTIX_PIXEL_FORMAT_INTERNAL_GUIDE_LAYER.)doc";

static const char *__doc_falcor_optix_DenoiserGuideLayer_previous_output_internal_guide_layer =
R"doc(Internal image used in temporal AOV denoising modes, pixel format
OPTIX_PIXEL_FORMAT_INTERNAL_GUIDE_LAYER.)doc";

static const char *__doc_falcor_optix_DenoiserLayer = R"doc(A layer to pass to the denoiser.)doc";

static const char *__doc_falcor_optix_DenoiserLayer_input = R"doc(Input image (beauty or AOV).)doc";

static const char *__doc_falcor_optix_DenoiserLayer_output = R"doc(Denoised output for given input.)doc";

static const char *__doc_falcor_optix_DenoiserLayer_previous_output =
R"doc(Denoised output image from previous frame if temporal model kind
selected.)doc";

static const char *__doc_falcor_optix_DenoiserLayer_type =
R"doc(Type of AOV, used in temporal AOV modes as a hint to improve image
quality.)doc";

static const char *__doc_falcor_optix_DenoiserParams = R"doc(Denoiser parameters.)doc";

static const char *__doc_falcor_optix_DenoiserParams_blend_factor =
R"doc(Blend factor. If set to 0 the output is 100% of the denoised input. If
set to 1, the output is 100% of the unmodified input. Values between 0
and 1 will linearly interpolate between the denoised and unmodified
input.)doc";

static const char *__doc_falcor_optix_DenoiserParams_hdr_average_color =
R"doc(Average log color of input image, separate for RGB channels
(optional). This parameter is used when the
OPTIX_DENOISER_MODEL_KIND_AOV model kind is set. If not set, average
log color will be calculated automatically. See hdr_intensity for
tiling, this also applies here.)doc";

static const char *__doc_falcor_optix_DenoiserParams_hdr_intensity =
R"doc(Average log intensity of input image (optional). If not set,
autoexposure will be calculated automatically for the input image.
Should be set to average log intensity of the entire image at least if
tiling is used to get consistent autoexposure for all tiles.)doc";

static const char *__doc_falcor_optix_DenoiserParams_temporal_mode_use_previous_layers =
R"doc(In temporal modes this parameter must be set to 1 if previous layers
(e.g. previous_output_internal_guide_layer) contain valid data. This
is the case in the second and subsequent frames of a sequence (for
example after a change of camera angle). In the first frame of such a
sequence this parameter must be set to 0.)doc";

static const char *__doc_falcor_optix_Image2D = R"doc(2D image contained in a buffer.)doc";

static const char *__doc_falcor_optix_Image2D_address =
R"doc(Raw pointer to CUDA memory containing image data (CUDA only, instead
of using buffer).)doc";

static const char *__doc_falcor_optix_Image2D_buffer = R"doc(Buffer containing the image data.)doc";

static const char *__doc_falcor_optix_Image2D_format = R"doc(Pixel format.)doc";

static const char *__doc_falcor_optix_Image2D_height = R"doc(Image height in pixels.)doc";

static const char *__doc_falcor_optix_Image2D_pixel_stride_in_bytes = R"doc(Pixel stride in bytes.)doc";

static const char *__doc_falcor_optix_Image2D_row_stride_in_bytes = R"doc(Row stride in bytes.)doc";

static const char *__doc_falcor_optix_Image2D_width = R"doc(Image width in pixels.)doc";

static const char *__doc_falcor_optix_ModelKind = R"doc()doc";

static const char *__doc_falcor_optix_ModelKind_aov = R"doc(Built-in model for denoising single image.)doc";

static const char *__doc_falcor_optix_ModelKind_hdr = R"doc(LDR/HDR denoisers. Deprecated on Optix9 in favour of aov.)doc";

static const char *__doc_falcor_optix_ModelKind_info = R"doc()doc";

static const char *__doc_falcor_optix_ModelKind_ldr = R"doc(LDR/HDR denoisers. Deprecated on Optix9 in favour of aov.)doc";

static const char *__doc_falcor_optix_ModelKind_temporal = R"doc(Temporal mode. Deprecated on Optix9 in favour of temporal_aov.)doc";

static const char *__doc_falcor_optix_ModelKind_temporal_aov = R"doc(Built-in model for denoising image sequence, temporally stable.)doc";

static const char *__doc_falcor_optix_ModelKind_temporal_upscale2x =
R"doc(Built-in model for denoising image sequence upscaling, temporally
stable (supports AOVs).)doc";

static const char *__doc_falcor_optix_ModelKind_upscale2x = R"doc(Built-in model for denoising single image upscaling (supports AOVs).)doc";

static const char *__doc_falcor_optix_PixelFormat = R"doc(Pixel format for an image.)doc";

static const char *__doc_falcor_optix_PixelFormat_float1 = R"doc()doc";

static const char *__doc_falcor_optix_PixelFormat_float2 = R"doc()doc";

static const char *__doc_falcor_optix_PixelFormat_float3 = R"doc()doc";

static const char *__doc_falcor_optix_PixelFormat_float4 = R"doc()doc";

static const char *__doc_falcor_optix_PixelFormat_guide_layer = R"doc()doc";

static const char *__doc_falcor_optix_PixelFormat_half1 = R"doc()doc";

static const char *__doc_falcor_optix_PixelFormat_half2 = R"doc()doc";

static const char *__doc_falcor_optix_PixelFormat_half3 = R"doc()doc";

static const char *__doc_falcor_optix_PixelFormat_half4 = R"doc()doc";

static const char *__doc_falcor_optix_PixelFormat_info = R"doc()doc";

static const char *__doc_falcor_optix_PixelFormat_uchar3 = R"doc()doc";

static const char *__doc_falcor_optix_PixelFormat_uchar4 = R"doc()doc";

static const char *__doc_falcor_optix_PixelFormat_unknown = R"doc()doc";

static const char *__doc_falcor_optix_find_enum_info_adl = R"doc()doc";

static const char *__doc_falcor_optix_find_enum_info_adl_2 = R"doc()doc";

static const char *__doc_falcor_optix_find_enum_info_adl_3 = R"doc()doc";

static const char *__doc_falcor_optix_find_enum_info_adl_4 = R"doc()doc";

static const char *__doc_falcor_pack_normal_oct_snorm2x16 =
R"doc(Pack a normal as 2x 16-bit snorms in the octahedral mapping.

Parameter ``normal``:
    Normal vector to encode.

Returns:
    Packed normal in 2x 16-bit snorms.)doc";

static const char *__doc_falcor_pack_normal_oct_snorm2x8 =
R"doc(Pack a normal as 2x 8-bit snorms in the octahedral mapping.

Parameter ``normal``:
    Normal vector to encode.

Returns:
    Packed normal in 2x 8-bit snorms.)doc";

static const char *__doc_falcor_pack_snorm16 =
R"doc(Pack single float value to 16-bit snorm.

Parameter ``v``:
    Float value in [-1,1].

Parameter ``options``:
    Packing options.

Returns:
    16-bit snorm value in low bits, high bits all zero.)doc";

static const char *__doc_falcor_pack_snorm2x16 =
R"doc(Pack two float values to 16-bit snorm.

Parameter ``v``:
    Float values in [-1,1].

Parameter ``options``:
    Packing options.

Returns:
    16-bit snorm values.)doc";

static const char *__doc_falcor_pack_snorm2x8 =
R"doc(Pack two float values to 8-bit snorm.

Parameter ``v``:
    Float values in [-1,1].

Parameter ``options``:
    Packing options.

Returns:
    8-bit snorm values in low bits, high bits all zero.)doc";

static const char *__doc_falcor_pack_snorm3x8 =
R"doc(Pack three float values to 8-bit snorm.

Parameter ``v``:
    Float values in [-1,1].

Parameter ``options``:
    Packing options.

Returns:
    8-bit snorm values in low bits, high bits all zero.)doc";

static const char *__doc_falcor_pack_snorm4x8 =
R"doc(Pack four float values to 8-bit snorm.

Parameter ``v``:
    Float values in [-1,1].

Parameter ``options``:
    Packing options.

Returns:
    8-bit snorm values in low bits, high bits all zero.)doc";

static const char *__doc_falcor_pack_snorm8 =
R"doc(Pack single float value to 8-bit snorm.

Parameter ``v``:
    Float value in [-1,1].

Parameter ``options``:
    Packing options.

Returns:
    8-bit snorm value in low bits, high bits all zero.)doc";

static const char *__doc_falcor_pack_unorm16 =
R"doc(Pack single float value to 16-bit unorm.

Parameter ``v``:
    Float value in [0,1].

Parameter ``options``:
    Packing options.

Returns:
    16-bit unorm value in low bits, high bits all zero.)doc";

static const char *__doc_falcor_pack_unorm2x16 =
R"doc(Pack two float values to 16-bit unorm.

Parameter ``v``:
    Float values in [0,1].

Parameter ``options``:
    Packing options.

Returns:
    16-bit unorm values.)doc";

static const char *__doc_falcor_pack_unorm2x8 =
R"doc(Pack two float values to 8-bit unorm.

Parameter ``v``:
    Float values in [0,1].

Parameter ``options``:
    Packing options.

Returns:
    8-bit unorm values in low bits, high bits all zero.)doc";

static const char *__doc_falcor_pack_unorm3x8 =
R"doc(Pack three float values to 8-bit unorm.

Parameter ``v``:
    Float values in [0,1].

Parameter ``options``:
    Packing options.

Returns:
    8-bit unorm values in low bits, high bits all zero.)doc";

static const char *__doc_falcor_pack_unorm4x8 =
R"doc(Pack four float values to 8-bit unorm.

Parameter ``v``:
    Float values in [0,1].

Parameter ``options``:
    Packing options.

Returns:
    8-bit unorm values in low bits, high bits all zero.)doc";

static const char *__doc_falcor_pack_unorm8 =
R"doc(Pack single float value to 8-bit unorm.

Parameter ``v``:
    Float value in [0,1].

Parameter ``options``:
    Packing options.

Returns:
    8-bit unorm value in low bits, high bits all zero.)doc";

static const char *__doc_falcor_parallel_reduce =
R"doc(Computes parallel reduction over a buffer of scalar elements. Supports
int32, uint32, int64, uint64, float32 types. element_count must be
greater than zero. Each provided output writes one scalar of the
requested type at the given offset. int32, uint32, and float32 outputs
reserve 4 bytes and require 4-byte aligned offsets. int64 and uint64
outputs reserve 8 bytes and require 8-byte aligned offsets. At least
one of sum/min/max must be provided.

Parameter ``command_encoder``:
    Command encoder.

Parameter ``type``:
    Data type of the elements.

Parameter ``data``:
    Data buffer and byte offset to reduce. The offset must be aligned
    to the element size.

Parameter ``element_count``:
    Number of elements to reduce.

Parameter ``sum``:
    (Optional) Buffer and offset to which the sum is written.

Parameter ``min``:
    (Optional) Buffer and offset to which the min is written.

Parameter ``max``:
    (Optional) Buffer and offset to which the max is written.

Returns:
    Nothing.)doc";

static const char *__doc_falcor_parallel_reduce_2 =
R"doc(Computes parallel reduction over a 2D texture. Results are vec4
(float4 for float/unorm/snorm, int4 for sint, uint4 for uint). Each
provided output writes one 16-byte vec4 and requires a 4-byte aligned
offset. At least one of sum/min/max must be provided.

Parameter ``command_encoder``:
    Command encoder.

Parameter ``input``:
    Input non-empty 2D texture (non-owning).

Parameter ``sum``:
    (Optional) Buffer and offset to which the sum is written (16
    bytes).

Parameter ``min``:
    (Optional) Buffer and offset to which the min is written (16
    bytes).

Parameter ``max``:
    (Optional) Buffer and offset to which the max is written (16
    bytes).

Returns:
    Nothing.)doc";

static const char *__doc_falcor_platform_FileLock =
R"doc(RAII file lock for cross-process synchronization. Uses LockFileEx on
Windows and flock on Linux/macOS.)doc";

static const char *__doc_falcor_platform_FileLock_FileLock = R"doc()doc";

static const char *__doc_falcor_platform_FileLock_FileLock_2 = R"doc()doc";

static const char *__doc_falcor_platform_FileLock_FileLock_3 = R"doc()doc";

static const char *__doc_falcor_platform_FileLock_is_locked = R"doc(Returns true if the lock is currently held.)doc";

static const char *__doc_falcor_platform_FileLock_m_handle = R"doc()doc";

static const char *__doc_falcor_platform_FileLock_m_locked = R"doc()doc";

static const char *__doc_falcor_platform_FileLock_operator_assign = R"doc()doc";

static const char *__doc_falcor_platform_FileLock_operator_assign_2 = R"doc()doc";

static const char *__doc_falcor_platform_FileLock_try_lock =
R"doc(Try to acquire the lock within the given timeout. Returns true if the
lock was acquired, false on timeout.)doc";

static const char *__doc_falcor_platform_FileLock_unlock = R"doc(Release the lock. Safe to call even if not locked.)doc";

static const char *__doc_falcor_platform_app_data_directory =
R"doc(The application data directory. Windows: %LOCALAPPDATA%/falcor2
Linux/macOS: ~/.falcor2 Creates the directory if it doesn't exist.)doc";

static const char *__doc_falcor_platform_demangle_type_name = R"doc(Return a readable string representation of a C++ type.)doc";

static const char *__doc_falcor_platform_get_proc_address = R"doc(Get a function pointer from a library.)doc";

static const char *__doc_falcor_platform_load_shared_library = R"doc(Load a shared library.)doc";

static const char *__doc_falcor_platform_project_directory =
R"doc(The project source directory. Note that this is only valid during
development.)doc";

static const char *__doc_falcor_platform_release_shared_library = R"doc(Release a shared library.)doc";

static const char *__doc_falcor_platform_runtime_directory =
R"doc(The runtime directory. This is the path where the falcor2 runtime
library (falcor2.dll or libfalcor2.so) resides.)doc";

static const char *__doc_falcor_prefix_sum =
R"doc(Computes the parallel prefix sum over an array of elements. This
supports int32, uint32, int64, uint64, float32 types.

Parameter ``command_encoder``:
    Command encoder.

Parameter ``type``:
    Data type of the elements.

Parameter ``data``:
    Data buffer and offset to compute prefix sum over.

Parameter ``element_count``:
    Number of elements to compute prefix sum over.

Parameter ``total_sum``:
    (Optional) Buffer and offset to which the total sum is copied.)doc";

static const char *__doc_falcor_record_list = R"doc(Write a counted list of nested archive records as a sized field.)doc";

static const char *__doc_falcor_record_list_2 = R"doc(Read a counted list of nested archive records.)doc";

static const char *__doc_falcor_reflection_ClassDescriptor =
R"doc(Describes a reflected class: its name, type, base class, and
properties.

ClassDescriptor owns its PropertyDescriptor instances (raw pointer
ownership so that properties() can return std::span directly over the
storage).

The base() pointer is resolved lazily on first access via
TypeRegistry::get().find() and cached. Returns null when the base
class is not registered.)doc";

static const char *__doc_falcor_reflection_ClassDescriptor_2 = R"doc()doc";

static const char *__doc_falcor_reflection_ClassDescriptor_ClassDescriptor =
R"doc(Construct a ClassDescriptor.

Parameter ``name``:
    Human-readable class name (from class_<T, Base>("Name")).

Parameter ``type_info``:
    typeid(T) of the described class.

Parameter ``base_type_info``:
    typeid(Base) of the base class, or typeid(void) if none.)doc";

static const char *__doc_falcor_reflection_ClassDescriptor_ClassDescriptor_2 = R"doc()doc";

static const char *__doc_falcor_reflection_ClassDescriptor_ClassDescriptor_3 = R"doc()doc";

static const char *__doc_falcor_reflection_ClassDescriptor_add_property =
R"doc(Append a property descriptor. Takes ownership. Called by
RegistryClassReflector during construction.)doc";

static const char *__doc_falcor_reflection_ClassDescriptor_all_property_range =
R"doc(Range over all properties in the registered inheritance chain (base
first).)doc";

static const char *__doc_falcor_reflection_ClassDescriptor_base =
R"doc(Pointer to the base class's ClassDescriptor, or null if unregistered.
Resolved lazily on first access via TypeRegistry.)doc";

static const char *__doc_falcor_reflection_ClassDescriptor_base_type = R"doc(typeid of the base class (typeid(void) if none).)doc";

static const char *__doc_falcor_reflection_ClassDescriptor_find_property = R"doc(Find a property by name across own + registered base chain.)doc";

static const char *__doc_falcor_reflection_ClassDescriptor_get_any = R"doc(Type-erased get: look up property by name and return its value.)doc";

static const char *__doc_falcor_reflection_ClassDescriptor_m_base = R"doc(Lazily resolved base class descriptor.)doc";

static const char *__doc_falcor_reflection_ClassDescriptor_m_base_resolved = R"doc()doc";

static const char *__doc_falcor_reflection_ClassDescriptor_m_base_type_info = R"doc()doc";

static const char *__doc_falcor_reflection_ClassDescriptor_m_name = R"doc()doc";

static const char *__doc_falcor_reflection_ClassDescriptor_m_properties = R"doc(Owned property descriptors.)doc";

static const char *__doc_falcor_reflection_ClassDescriptor_m_property_index =
R"doc(Name -> index into m_properties. @warning Key points into
PropertyDescriptor::m_name and expects them not to be copied/moved.)doc";

static const char *__doc_falcor_reflection_ClassDescriptor_m_type_info = R"doc()doc";

static const char *__doc_falcor_reflection_ClassDescriptor_name = R"doc(Class name as provided to class_<T, Base>("Name").)doc";

static const char *__doc_falcor_reflection_ClassDescriptor_operator_assign = R"doc()doc";

static const char *__doc_falcor_reflection_ClassDescriptor_operator_assign_2 = R"doc()doc";

static const char *__doc_falcor_reflection_ClassDescriptor_properties = R"doc(Own properties only (not including base class properties).)doc";

static const char *__doc_falcor_reflection_ClassDescriptor_property_range = R"doc(Range over just own properties.)doc";

static const char *__doc_falcor_reflection_ClassDescriptor_read_from_properties = R"doc(Read all properties (base first) from a Properties object.)doc";

static const char *__doc_falcor_reflection_ClassDescriptor_reset = R"doc(Reset all properties that have defaults (base first).)doc";

static const char *__doc_falcor_reflection_ClassDescriptor_resolve_base = R"doc(Resolve the base pointer (called once, then cached).)doc";

static const char *__doc_falcor_reflection_ClassDescriptor_set_any = R"doc(Type-erased set: look up property by name and set its value.)doc";

static const char *__doc_falcor_reflection_ClassDescriptor_type = R"doc(typeid of the described class.)doc";

static const char *__doc_falcor_reflection_ClassDescriptor_write_to_properties = R"doc(Write all properties (base first) into a Properties object.)doc";

static const char *__doc_falcor_reflection_DefaultValue =
R"doc(Explicit default value for a property. Used to reset a property back
to its default value.)doc";

static const char *__doc_falcor_reflection_DefaultValue_DefaultValue = R"doc()doc";

static const char *__doc_falcor_reflection_DefaultValue_DefaultValue_2 = R"doc()doc";

static const char *__doc_falcor_reflection_DefaultValue_value = R"doc()doc";

static const char *__doc_falcor_reflection_DynamicPropertySet =
R"doc(Per-instance storage for dynamic properties, using the same
PropertyDescriptor interface as static (TypedPropertyDescriptor)
properties.

Internally stores StoredPropertyDescriptor<T> instances via the
PropertyDescriptor* base pointer. Consumer code (UI, serialization)
treats all descriptors uniformly.)doc";

static const char *__doc_falcor_reflection_DynamicPropertySet_DynamicPropertySet = R"doc()doc";

static const char *__doc_falcor_reflection_DynamicPropertySet_DynamicPropertySet_2 = R"doc()doc";

static const char *__doc_falcor_reflection_DynamicPropertySet_DynamicPropertySet_3 = R"doc()doc";

static const char *__doc_falcor_reflection_DynamicPropertySet_add_property =
R"doc(Add a dynamic property with an initial value and optional metadata.
Throws if a property with the same name already exists in this set.)doc";

static const char *__doc_falcor_reflection_DynamicPropertySet_add_property_checked =
R"doc(Add a dynamic property, also checking for name collisions with the
owning class's static ClassDescriptor properties. Throws if the name
collides with either set.)doc";

static const char *__doc_falcor_reflection_DynamicPropertySet_clear = R"doc(Remove all dynamic properties, freeing all descriptors.)doc";

static const char *__doc_falcor_reflection_DynamicPropertySet_descriptors = R"doc(All dynamic property descriptors.)doc";

static const char *__doc_falcor_reflection_DynamicPropertySet_empty = R"doc(True if no dynamic properties.)doc";

static const char *__doc_falcor_reflection_DynamicPropertySet_find_property = R"doc(Find a property by name. Returns null if not found.)doc";

static const char *__doc_falcor_reflection_DynamicPropertySet_generation =
R"doc(Generation counter, incremented on every structural change
(add_property, remove_property, clear). Can be used to invalidate
caches.)doc";

static const char *__doc_falcor_reflection_DynamicPropertySet_get =
R"doc(Typed get by name. Returns a copy of the stored value. Throws if the
property does not exist or has a different type.)doc";

static const char *__doc_falcor_reflection_DynamicPropertySet_get_any = R"doc(Type-erased get by name.)doc";

static const char *__doc_falcor_reflection_DynamicPropertySet_get_property = R"doc(Get a property by name. Throws if not found.)doc";

static const char *__doc_falcor_reflection_DynamicPropertySet_get_property_2 = R"doc(Get a mutable property by name. Throws if not found.)doc";

static const char *__doc_falcor_reflection_DynamicPropertySet_get_ref =
R"doc(Typed get by name returning a const reference to the stored value.
Only valid for DynamicPropertySet (StoredPropertyDescriptor)
properties. Throws if the property does not exist or has a different
type.)doc";

static const char *__doc_falcor_reflection_DynamicPropertySet_has_property = R"doc(Check if a property with the given name exists.)doc";

static const char *__doc_falcor_reflection_DynamicPropertySet_is_default =
R"doc(Check if a property is at its default value. Returns false if the
property does not exist.)doc";

static const char *__doc_falcor_reflection_DynamicPropertySet_is_equal_property =
R"doc(Check if the named property's current value equals the same-named
property in another Properties object. Returns false if the property
does not exist in this set or in other.)doc";

static const char *__doc_falcor_reflection_DynamicPropertySet_m_generation = R"doc(Generation counter incremented on structural changes.)doc";

static const char *__doc_falcor_reflection_DynamicPropertySet_m_instance = R"doc(Owner instance pointer forwarded to OnChange and UIEnableIf callbacks.)doc";

static const char *__doc_falcor_reflection_DynamicPropertySet_m_properties = R"doc(Owned property descriptors (StoredPropertyDescriptor<T> instances).)doc";

static const char *__doc_falcor_reflection_DynamicPropertySet_m_property_index =
R"doc(Name -> index into m_properties. @warning Key points into
PropertyDescriptor::m_name and expects them not to be copied/moved.)doc";

static const char *__doc_falcor_reflection_DynamicPropertySet_operator_assign = R"doc()doc";

static const char *__doc_falcor_reflection_DynamicPropertySet_operator_assign_2 = R"doc()doc";

static const char *__doc_falcor_reflection_DynamicPropertySet_read_from_properties = R"doc(Read dynamic properties from a Properties object.)doc";

static const char *__doc_falcor_reflection_DynamicPropertySet_remove_property =
R"doc(Remove a dynamic property by name. No-op if the property does not
exist. Preserves insertion order of remaining properties.)doc";

static const char *__doc_falcor_reflection_DynamicPropertySet_reset = R"doc(Reset all dynamic properties to their default values.)doc";

static const char *__doc_falcor_reflection_DynamicPropertySet_reset_2 =
R"doc(Reset a single property to its default value by name. Throws if the
property does not exist.)doc";

static const char *__doc_falcor_reflection_DynamicPropertySet_set = R"doc(Typed set by name.)doc";

static const char *__doc_falcor_reflection_DynamicPropertySet_set_any = R"doc(Type-erased set by name.)doc";

static const char *__doc_falcor_reflection_DynamicPropertySet_set_instance =
R"doc(Set the owner instance pointer, forwarded to OnChange and UIEnableIf
callbacks.)doc";

static const char *__doc_falcor_reflection_DynamicPropertySet_size = R"doc(Number of dynamic properties.)doc";

static const char *__doc_falcor_reflection_DynamicPropertySet_write_to_properties = R"doc(Write all dynamic properties into a Properties object.)doc";

static const char *__doc_falcor_reflection_EnumDescriptor = R"doc(Descriptor for an enum type.)doc";

static const char *__doc_falcor_reflection_EnumDescriptor_is_flags = R"doc(True for flag/bitmask enums.)doc";

static const char *__doc_falcor_reflection_EnumDescriptor_items = R"doc(All registered values.)doc";

static const char *__doc_falcor_reflection_EnumItem = R"doc(A single enum value and its name.)doc";

static const char *__doc_falcor_reflection_EnumItem_name = R"doc()doc";

static const char *__doc_falcor_reflection_EnumItem_value = R"doc()doc";

static const char *__doc_falcor_reflection_OnChange =
R"doc(Change notification callback for a property. Extracted during property
construction (like DefaultValue) and used to wrap the setter. NOT
stored in the metadata bag. Constructed via on_change<T>() which hides
the void* cast.)doc";

static const char *__doc_falcor_reflection_OnChange_callback = R"doc()doc";

static const char *__doc_falcor_reflection_PropertyDescriptor =
R"doc(Abstract base class for property descriptors.

Provides a type-erased interface for accessing property values on
instances, querying metadata (default value, value range, doc string),
and serializing/deserializing via the Properties system.

Two concrete subclasses exist: - TypedPropertyDescriptor<T, OwnerT>:
for static C++ properties (member-function-pointer based) -
StoredPropertyDescriptor<T>: for dynamic per-instance properties
(value-storing))doc";

static const char *__doc_falcor_reflection_PropertyDescriptor_2 = R"doc()doc";

static const char *__doc_falcor_reflection_PropertyDescriptor_PropertyDescriptor = R"doc()doc";

static const char *__doc_falcor_reflection_PropertyDescriptor_doc = R"doc(Convenience: returns the doc string, or empty view if absent.)doc";

static const char *__doc_falcor_reflection_PropertyDescriptor_enum_descriptor = R"doc(Convenience: returns the EnumDescriptor metadata, or nullptr.)doc";

static const char *__doc_falcor_reflection_PropertyDescriptor_find_metadata =
R"doc(Search the metadata list for an entry whose typeid matches T. Returns
null if not found. This is the generic extensible metadata accessor --
new metadata kinds can be added without modifying PropertyDescriptor.)doc";

static const char *__doc_falcor_reflection_PropertyDescriptor_get =
R"doc(Typed read: returns the current value, bypassing Any. Throws if the
property holds a different type.)doc";

static const char *__doc_falcor_reflection_PropertyDescriptor_get_any = R"doc(Type-erased read: returns the current value wrapped in Any.)doc";

static const char *__doc_falcor_reflection_PropertyDescriptor_get_enum_as_int64 =
R"doc(Read an enum property as int64_t. Throws if the property is not an
enum.)doc";

static const char *__doc_falcor_reflection_PropertyDescriptor_get_value =
R"doc(Copy the current value into caller-provided storage. The caller must
ensure out points to valid memory for the property's type.)doc";

static const char *__doc_falcor_reflection_PropertyDescriptor_has_default_value = R"doc(True if the property has a default value.)doc";

static const char *__doc_falcor_reflection_PropertyDescriptor_is_default =
R"doc(True if the property has a default value and the current value equals
it. Returns false if no default is set.)doc";

static const char *__doc_falcor_reflection_PropertyDescriptor_is_read_only = R"doc(True if the property has no setter.)doc";

static const char *__doc_falcor_reflection_PropertyDescriptor_is_serializable_to_properties =
R"doc(True if the property's value type can be written to / read from
Properties.)doc";

static const char *__doc_falcor_reflection_PropertyDescriptor_m_metadata = R"doc()doc";

static const char *__doc_falcor_reflection_PropertyDescriptor_m_name = R"doc()doc";

static const char *__doc_falcor_reflection_PropertyDescriptor_m_read_only = R"doc()doc";

static const char *__doc_falcor_reflection_PropertyDescriptor_name = R"doc(Property name.)doc";

static const char *__doc_falcor_reflection_PropertyDescriptor_read_from_properties =
R"doc(Read the property value from a Properties object if the key exists.
Returns true if the property was found and applied. Returns false if
the property type is not serializable or the key is absent.)doc";

static const char *__doc_falcor_reflection_PropertyDescriptor_reset =
R"doc(Reset the property to its default value if one was provided. No-op if
no default or if read-only.)doc";

static const char *__doc_falcor_reflection_PropertyDescriptor_set =
R"doc(Typed write: sets the property value directly, bypassing Any. Throws
if read-only or if the property holds a different type.)doc";

static const char *__doc_falcor_reflection_PropertyDescriptor_set_any =
R"doc(Type-erased write: sets the property value from an Any. Throws if
read-only or if Any holds wrong type.)doc";

static const char *__doc_falcor_reflection_PropertyDescriptor_set_enum_from_int64 =
R"doc(Set an enum property from int64_t. Throws if read-only or if the
property is not an enum.)doc";

static const char *__doc_falcor_reflection_PropertyDescriptor_set_value =
R"doc(Set the property value from caller-provided storage. The caller must
ensure value points to valid memory for the property's type. Throws if
read-only.)doc";

static const char *__doc_falcor_reflection_PropertyDescriptor_type = R"doc(C++ typeid of the property value type.)doc";

static const char *__doc_falcor_reflection_PropertyDescriptor_ui_drag_speed = R"doc(Convenience: returns the UIDragSpeed metadata, or nullptr.)doc";

static const char *__doc_falcor_reflection_PropertyDescriptor_ui_editor = R"doc(Convenience: returns the UIEditor metadata, or nullptr.)doc";

static const char *__doc_falcor_reflection_PropertyDescriptor_ui_enable_if = R"doc(Convenience: returns the UIEnableIf metadata, or nullptr.)doc";

static const char *__doc_falcor_reflection_PropertyDescriptor_ui_flags = R"doc(Convenience: returns combined UIFlags, or UIFlags::none if absent.)doc";

static const char *__doc_falcor_reflection_PropertyDescriptor_ui_group = R"doc(Convenience: returns the UIGroup metadata, or nullptr.)doc";

static const char *__doc_falcor_reflection_PropertyDescriptor_ui_label = R"doc(Convenience: returns the UILabel metadata, or nullptr.)doc";

static const char *__doc_falcor_reflection_PropertyDescriptor_value_range = R"doc(Convenience: returns the ValueRange metadata, or nullptr.)doc";

static const char *__doc_falcor_reflection_PropertyDescriptor_write_to_properties =
R"doc(Write the property value into a Properties object using the property's
name as key. Throws if the property type is not serializable.)doc";

static const char *__doc_falcor_reflection_PropertyRange =
R"doc(A lightweight, non-owning range that iterates over PropertyDescriptor
pointers from one or more ClassDescriptors in the inheritance chain.

When constructed with `include_base = true`, properties are yielded in
base-first order (root of the registered chain first, most-derived
last).

This is a value type -- no shared mutable state, no mutex, no caching.
Each construction walks the base() chain to collect ClassDescriptor
pointers (typically 1-3 levels deep).)doc";

static const char *__doc_falcor_reflection_PropertyRange_Iterator = R"doc(Forward iterator over const PropertyDescriptor pointers.)doc";

static const char *__doc_falcor_reflection_PropertyRange_Iterator_Iterator = R"doc()doc";

static const char *__doc_falcor_reflection_PropertyRange_Iterator_Iterator_2 = R"doc()doc";

static const char *__doc_falcor_reflection_PropertyRange_Iterator_advance_to_valid = R"doc(Advance past any empty spans to the next valid property.)doc";

static const char *__doc_falcor_reflection_PropertyRange_Iterator_m_prop_idx = R"doc()doc";

static const char *__doc_falcor_reflection_PropertyRange_Iterator_m_range = R"doc()doc";

static const char *__doc_falcor_reflection_PropertyRange_Iterator_m_span_idx = R"doc()doc";

static const char *__doc_falcor_reflection_PropertyRange_Iterator_operator_eq = R"doc()doc";

static const char *__doc_falcor_reflection_PropertyRange_Iterator_operator_inc = R"doc()doc";

static const char *__doc_falcor_reflection_PropertyRange_Iterator_operator_inc_2 = R"doc()doc";

static const char *__doc_falcor_reflection_PropertyRange_Iterator_operator_mul = R"doc()doc";

static const char *__doc_falcor_reflection_PropertyRange_Iterator_operator_ne = R"doc()doc";

static const char *__doc_falcor_reflection_PropertyRange_all =
R"doc(Construct a range over all properties in the inheritance chain (base
first).)doc";

static const char *__doc_falcor_reflection_PropertyRange_begin = R"doc()doc";

static const char *__doc_falcor_reflection_PropertyRange_empty = R"doc(Returns true if the range contains no properties.)doc";

static const char *__doc_falcor_reflection_PropertyRange_end = R"doc()doc";

static const char *__doc_falcor_reflection_PropertyRange_m_spans =
R"doc(Spans of property pointers, one per ClassDescriptor in the chain (base
first). For own-only ranges, this contains a single span.)doc";

static const char *__doc_falcor_reflection_PropertyRange_own =
R"doc(Construct a range over just the own properties of a single
ClassDescriptor.)doc";

static const char *__doc_falcor_reflection_PropertyRange_size =
R"doc(Count the number of properties in the range. Note: this is O(n) where
n is the number of spans (i.e. depth of inheritance chain).)doc";

static const char *__doc_falcor_reflection_StoredPropertyDescriptor =
R"doc(Concrete property descriptor for dynamic per-instance properties.

Stores the property value directly as a T member. The instance pointer
passed to get/set is ignored (asserted to be nullptr). This allows
DynamicPropertySet to use the same PropertyDescriptor* interface as
static properties.)doc";

static const char *__doc_falcor_reflection_StoredPropertyDescriptor_StoredPropertyDescriptor = R"doc(Construct with a name, initial value, and metadata.)doc";

static const char *__doc_falcor_reflection_StoredPropertyDescriptor_StoredPropertyDescriptor_2 =
R"doc(Delegating constructor: receives pre-extracted metadata and default
value.)doc";

static const char *__doc_falcor_reflection_StoredPropertyDescriptor_StoredPropertyDescriptor_3 = R"doc()doc";

static const char *__doc_falcor_reflection_StoredPropertyDescriptor_StoredPropertyDescriptor_4 = R"doc()doc";

static const char *__doc_falcor_reflection_StoredPropertyDescriptor_get_any = R"doc()doc";

static const char *__doc_falcor_reflection_StoredPropertyDescriptor_get_enum_as_int64 = R"doc()doc";

static const char *__doc_falcor_reflection_StoredPropertyDescriptor_get_value = R"doc()doc";

static const char *__doc_falcor_reflection_StoredPropertyDescriptor_has_default_value = R"doc()doc";

static const char *__doc_falcor_reflection_StoredPropertyDescriptor_is_default = R"doc()doc";

static const char *__doc_falcor_reflection_StoredPropertyDescriptor_is_serializable_to_properties = R"doc()doc";

static const char *__doc_falcor_reflection_StoredPropertyDescriptor_m_default = R"doc()doc";

static const char *__doc_falcor_reflection_StoredPropertyDescriptor_m_on_change = R"doc()doc";

static const char *__doc_falcor_reflection_StoredPropertyDescriptor_m_value =
R"doc(Mutable because StoredPropertyDescriptor owns the value and
set()/read_from_properties()/reset() need to mutate it through the
const virtual interface inherited from PropertyDescriptor.)doc";

static const char *__doc_falcor_reflection_StoredPropertyDescriptor_operator_assign = R"doc()doc";

static const char *__doc_falcor_reflection_StoredPropertyDescriptor_operator_assign_2 = R"doc()doc";

static const char *__doc_falcor_reflection_StoredPropertyDescriptor_read_from_properties = R"doc()doc";

static const char *__doc_falcor_reflection_StoredPropertyDescriptor_reset = R"doc()doc";

static const char *__doc_falcor_reflection_StoredPropertyDescriptor_set_any = R"doc()doc";

static const char *__doc_falcor_reflection_StoredPropertyDescriptor_set_enum_from_int64 = R"doc()doc";

static const char *__doc_falcor_reflection_StoredPropertyDescriptor_set_value = R"doc()doc";

static const char *__doc_falcor_reflection_StoredPropertyDescriptor_type = R"doc()doc";

static const char *__doc_falcor_reflection_StoredPropertyDescriptor_value = R"doc(Direct access to the stored value (for DynamicPropertySet internals).)doc";

static const char *__doc_falcor_reflection_StoredPropertyDescriptor_write_to_properties = R"doc()doc";

static const char *__doc_falcor_reflection_TypeRegistry = R"doc()doc";

static const char *__doc_falcor_reflection_TypeRegistry_2 =
R"doc(Singleton registry of all reflected class descriptors.

Thread safety: not thread-safe for concurrent registration. All
register_class() calls must happen during static initialization on a
single thread before any concurrent lookups. After initialization,
find() and all() are safe to call concurrently from multiple threads
since the registry is then read-only.)doc";

static const char *__doc_falcor_reflection_TypeRegistry_TypeRegistry = R"doc()doc";

static const char *__doc_falcor_reflection_TypeRegistry_TypeRegistry_2 = R"doc()doc";

static const char *__doc_falcor_reflection_TypeRegistry_TypeRegistry_3 = R"doc()doc";

static const char *__doc_falcor_reflection_TypeRegistry_all = R"doc(All registered class descriptors.)doc";

static const char *__doc_falcor_reflection_TypeRegistry_find = R"doc(Find a class descriptor by type_info. Returns null if not registered.)doc";

static const char *__doc_falcor_reflection_TypeRegistry_find_2 = R"doc(Find a class descriptor by name. Returns null if not registered.)doc";

static const char *__doc_falcor_reflection_TypeRegistry_get = R"doc(Returns the singleton instance.)doc";

static const char *__doc_falcor_reflection_TypeRegistry_m_by_name = R"doc(name -> descriptor.)doc";

static const char *__doc_falcor_reflection_TypeRegistry_m_by_type = R"doc(type_index -> descriptor.)doc";

static const char *__doc_falcor_reflection_TypeRegistry_m_descs = R"doc(Descriptors (owned).)doc";

static const char *__doc_falcor_reflection_TypeRegistry_operator_assign = R"doc()doc";

static const char *__doc_falcor_reflection_TypeRegistry_operator_assign_2 = R"doc()doc";

static const char *__doc_falcor_reflection_TypeRegistry_register_class =
R"doc(Register a class descriptor. Takes ownership. Throws if a class with
the same std::type_info is already registered.)doc";

static const char *__doc_falcor_reflection_TypedPropertyDescriptor =
R"doc(Concrete property descriptor for static C++ properties.

Stores getter/setter as std::function, invoked on the owner instance
(cast from void* to OwnerT&). Metadata is stored in the base class's
metadata bag via extract_all.)doc";

static const char *__doc_falcor_reflection_TypedPropertyDescriptor_TypedPropertyDescriptor = R"doc(Construct from getter, optional setter, and metadata.)doc";

static const char *__doc_falcor_reflection_TypedPropertyDescriptor_TypedPropertyDescriptor_2 = R"doc(Construct from getter, optional setter, and pre-extracted metadata.)doc";

static const char *__doc_falcor_reflection_TypedPropertyDescriptor_TypedPropertyDescriptor_3 = R"doc()doc";

static const char *__doc_falcor_reflection_TypedPropertyDescriptor_TypedPropertyDescriptor_4 = R"doc()doc";

static const char *__doc_falcor_reflection_TypedPropertyDescriptor_cast_instance = R"doc()doc";

static const char *__doc_falcor_reflection_TypedPropertyDescriptor_cast_mutable_instance = R"doc()doc";

static const char *__doc_falcor_reflection_TypedPropertyDescriptor_get_any = R"doc()doc";

static const char *__doc_falcor_reflection_TypedPropertyDescriptor_get_enum_as_int64 = R"doc()doc";

static const char *__doc_falcor_reflection_TypedPropertyDescriptor_get_value = R"doc()doc";

static const char *__doc_falcor_reflection_TypedPropertyDescriptor_has_default_value = R"doc()doc";

static const char *__doc_falcor_reflection_TypedPropertyDescriptor_is_default = R"doc()doc";

static const char *__doc_falcor_reflection_TypedPropertyDescriptor_is_serializable_to_properties = R"doc()doc";

static const char *__doc_falcor_reflection_TypedPropertyDescriptor_m_default = R"doc()doc";

static const char *__doc_falcor_reflection_TypedPropertyDescriptor_m_getter = R"doc()doc";

static const char *__doc_falcor_reflection_TypedPropertyDescriptor_m_setter = R"doc()doc";

static const char *__doc_falcor_reflection_TypedPropertyDescriptor_operator_assign = R"doc()doc";

static const char *__doc_falcor_reflection_TypedPropertyDescriptor_operator_assign_2 = R"doc()doc";

static const char *__doc_falcor_reflection_TypedPropertyDescriptor_read_from_properties = R"doc()doc";

static const char *__doc_falcor_reflection_TypedPropertyDescriptor_reset = R"doc()doc";

static const char *__doc_falcor_reflection_TypedPropertyDescriptor_set_any = R"doc()doc";

static const char *__doc_falcor_reflection_TypedPropertyDescriptor_set_enum_from_int64 = R"doc()doc";

static const char *__doc_falcor_reflection_TypedPropertyDescriptor_set_value = R"doc()doc";

static const char *__doc_falcor_reflection_TypedPropertyDescriptor_type = R"doc()doc";

static const char *__doc_falcor_reflection_TypedPropertyDescriptor_write_to_properties = R"doc()doc";

static const char *__doc_falcor_reflection_UIDragSpeed = R"doc(Override default drag speed for numeric properties.)doc";

static const char *__doc_falcor_reflection_UIDragSpeed_speed = R"doc()doc";

static const char *__doc_falcor_reflection_UIEditor =
R"doc(Fully custom callback for a property editor. Uses a plain function
pointer to avoid heap allocation inside Any storage. Returns true if
property value changed.)doc";

static const char *__doc_falcor_reflection_UIEditor_PropertyDescriptor = R"doc()doc";

static const char *__doc_falcor_reflection_UIEditor_func = R"doc()doc";

static const char *__doc_falcor_reflection_UIEnableIf =
R"doc(Conditional enable/disable for a property in the UI. Constructed via
ui_enable_if<T>() which hides the void* cast.)doc";

static const char *__doc_falcor_reflection_UIEnableIf_predicate = R"doc()doc";

static const char *__doc_falcor_reflection_UIFlags =
R"doc(UI hint flags for property rendering. Multiple flags can be combined
and are OR-accumulated during property registration.)doc";

static const char *__doc_falcor_reflection_UIFlags_advanced = R"doc(Hide unless "show advanced" is enabled.)doc";

static const char *__doc_falcor_reflection_UIFlags_display_as_color =
R"doc(Display float3/float4 as a color picker instead of three/four separate
drags.)doc";

static const char *__doc_falcor_reflection_UIFlags_hidden = R"doc(Never show the property in the UI.)doc";

static const char *__doc_falcor_reflection_UIFlags_info = R"doc()doc";

static const char *__doc_falcor_reflection_UIFlags_none = R"doc()doc";

static const char *__doc_falcor_reflection_UIGroup =
R"doc(Hierarchical grouping using '/' delimiter (e.g., "Appearance/Color").
Rendered as collapsing headers in the UI.)doc";

static const char *__doc_falcor_reflection_UIGroup_path = R"doc()doc";

static const char *__doc_falcor_reflection_UILabel = R"doc(Override display name for a property (default: property name).)doc";

static const char *__doc_falcor_reflection_UILabel_text = R"doc()doc";

static const char *__doc_falcor_reflection_ValueRange =
R"doc(Numeric range constraint for property values. Used as a hint for UI
sliders. Not enforced by property setters. Stores min/max as double so
a single non-templated struct works for all numeric types.)doc";

static const char *__doc_falcor_reflection_ValueRange_ValueRange = R"doc()doc";

static const char *__doc_falcor_reflection_ValueRange_ValueRange_2 = R"doc()doc";

static const char *__doc_falcor_reflection_ValueRange_max = R"doc()doc";

static const char *__doc_falcor_reflection_ValueRange_min = R"doc()doc";

static const char *__doc_falcor_reflection_default_value = R"doc(Convenience factory for DefaultValue.)doc";

static const char *__doc_falcor_reflection_deserialize_properties =
R"doc(Deserialize the properties of a reflected object from a Properties
object. @warning This does not handle properties defined in the base
class.)doc";

static const char *__doc_falcor_reflection_detail_ClassReflectorProbe =
R"doc(Minimal reflector satisfying the ClassReflector concept, used only for
ReflectedClass trait checking.)doc";

static const char *__doc_falcor_reflection_detail_ClassReflectorProbe_def_prop_ro = R"doc()doc";

static const char *__doc_falcor_reflection_detail_ClassReflectorProbe_def_prop_rw = R"doc()doc";

static const char *__doc_falcor_reflection_detail_ClassReflectorProbe_def_ro = R"doc()doc";

static const char *__doc_falcor_reflection_detail_ClassReflectorProbe_def_rw = R"doc()doc";

static const char *__doc_falcor_reflection_detail_ExtraProbe =
R"doc(Sentinel type used in concept checks to verify that reflector methods
accept an arbitrary parameter pack of extras (metadata). If a method
accepts this unknown type it must be using a template parameter pack.)doc";

static const char *__doc_falcor_reflection_detail_ExtractedMetadata =
R"doc(Result of extract_all: both the metadata bag and the optional default
value, extracted from the variadic extras pack in a single pass.)doc";

static const char *__doc_falcor_reflection_detail_ExtractedMetadata_default_value = R"doc()doc";

static const char *__doc_falcor_reflection_detail_ExtractedMetadata_metadata = R"doc()doc";

static const char *__doc_falcor_reflection_detail_ExtractedMetadata_on_change = R"doc()doc";

static const char *__doc_falcor_reflection_detail_PropertiesDeserializer =
R"doc(ClassReflector that deserializes properties from a Properties object
onto an object. @warning This does not handle properties defined in
the base class.)doc";

static const char *__doc_falcor_reflection_detail_PropertiesDeserializer_PropertiesDeserializer = R"doc()doc";

static const char *__doc_falcor_reflection_detail_PropertiesDeserializer_def_prop_ro =
R"doc(Read-only properties are skipped during deserialization. Extra
arguments (doc strings, metadata) are silently ignored.)doc";

static const char *__doc_falcor_reflection_detail_PropertiesDeserializer_def_prop_rw =
R"doc(Deserialize a read-write property: reads from Properties (if present)
and applies via setter. If OnChange metadata is present, fires the
callback after assignment.)doc";

static const char *__doc_falcor_reflection_detail_PropertiesDeserializer_def_ro =
R"doc(Read-only properties from member pointers are skipped during
deserialization.)doc";

static const char *__doc_falcor_reflection_detail_PropertiesDeserializer_def_rw =
R"doc(Deserialize a read-write property from a member pointer. If OnChange
metadata is present, fires the callback after assignment.)doc";

static const char *__doc_falcor_reflection_detail_PropertiesDeserializer_m_obj = R"doc()doc";

static const char *__doc_falcor_reflection_detail_PropertiesDeserializer_m_props = R"doc()doc";

static const char *__doc_falcor_reflection_detail_PropertiesSerializer =
R"doc(ClassReflector that serializes an object's properties into a
Properties object. @warning This does not handle properties defined in
the base class.)doc";

static const char *__doc_falcor_reflection_detail_PropertiesSerializer_PropertiesSerializer = R"doc()doc";

static const char *__doc_falcor_reflection_detail_PropertiesSerializer_def_prop_ro =
R"doc(Serialize a read-only property: reads via getter and stores into
Properties. Extra arguments (doc strings, metadata) are silently
ignored.)doc";

static const char *__doc_falcor_reflection_detail_PropertiesSerializer_def_prop_rw =
R"doc(Serialize a read-write property: reads via getter and stores into
Properties. Extra arguments (doc strings, metadata) are silently
ignored.)doc";

static const char *__doc_falcor_reflection_detail_PropertiesSerializer_def_ro =
R"doc(Serialize a read-only property from a member pointer. Extra arguments
(doc strings, metadata) are silently ignored.)doc";

static const char *__doc_falcor_reflection_detail_PropertiesSerializer_def_rw =
R"doc(Serialize a read-write property from a member pointer. Extra arguments
(doc strings, metadata) are silently ignored.)doc";

static const char *__doc_falcor_reflection_detail_PropertiesSerializer_m_obj = R"doc()doc";

static const char *__doc_falcor_reflection_detail_PropertiesSerializer_m_props = R"doc()doc";

static const char *__doc_falcor_reflection_detail_ReflectorProbeTarget =
R"doc(Helper target type providing canonical getter, setter, and member
pointer for concept checking without depending on any real reflected
class. Uses ValueProbe as the property type to ensure reflector
methods are generic.)doc";

static const char *__doc_falcor_reflection_detail_ReflectorProbeTarget_get_value = R"doc()doc";

static const char *__doc_falcor_reflection_detail_ReflectorProbeTarget_set_value = R"doc()doc";

static const char *__doc_falcor_reflection_detail_ReflectorProbeTarget_value = R"doc()doc";

static const char *__doc_falcor_reflection_detail_RegistryClassReflector =
R"doc(Reflector that satisfies the ClassReflector concept and accumulates
properties into a ClassDescriptor. The descriptor is registered with
the TypeRegistry when the reflector is destroyed (RAII via
finalize()).)doc";

static const char *__doc_falcor_reflection_detail_RegistryClassReflector_RegistryClassReflector = R"doc()doc";

static const char *__doc_falcor_reflection_detail_RegistryClassReflector_RegistryClassReflector_2 = R"doc()doc";

static const char *__doc_falcor_reflection_detail_RegistryClassReflector_RegistryClassReflector_3 = R"doc()doc";

static const char *__doc_falcor_reflection_detail_RegistryClassReflector_def_prop_ro = R"doc(Define a read-only property with getter and optional metadata.)doc";

static const char *__doc_falcor_reflection_detail_RegistryClassReflector_def_prop_rw =
R"doc(Define a read-write property with getter, setter, and optional
metadata. If OnChange metadata is present, wraps the setter to call
the callback after assignment.)doc";

static const char *__doc_falcor_reflection_detail_RegistryClassReflector_def_ro =
R"doc(Define a read-only property from a member pointer, with optional
metadata.)doc";

static const char *__doc_falcor_reflection_detail_RegistryClassReflector_def_rw =
R"doc(Define a read-write property from a member pointer, with optional
metadata. If OnChange metadata is present, wraps the setter to call
the callback after assignment.)doc";

static const char *__doc_falcor_reflection_detail_RegistryClassReflector_finalize =
R"doc(Register the completed ClassDescriptor with the TypeRegistry. Called
once from the destructor.)doc";

static const char *__doc_falcor_reflection_detail_RegistryClassReflector_m_desc = R"doc()doc";

static const char *__doc_falcor_reflection_detail_RegistryClassReflector_operator_assign = R"doc()doc";

static const char *__doc_falcor_reflection_detail_RegistryClassReflector_operator_assign_2 = R"doc()doc";

static const char *__doc_falcor_reflection_detail_ValueProbe = R"doc(Opaque value type used as the property type in concept checks.)doc";

static const char *__doc_falcor_reflection_detail_accumulate_flags =
R"doc(OR-accumulate a flags value into the metadata bag. If an entry of the
same flags type already exists, OR it in; otherwise push a new entry.)doc";

static const char *__doc_falcor_reflection_detail_enum_descriptor_from_sgl_enum = R"doc(Create an EnumDescriptor from SGL's EnumInfo<T>.)doc";

static const char *__doc_falcor_reflection_detail_extract_all =
R"doc(Walk the variadic metadata pack once, extracting both the metadata bag
and the optional default value.

- DefaultValue<V> entries populate default_value. - Doc strings (const
char*) are stored as std::string in metadata. - Flags types
(is_reflection_flags_v) are OR-accumulated into a single entry per
flags type. - All other types satisfying is_reflection_metadata_v are
stored as-is. - Unrecognized non-metadata types cause a compile-time
error.

Each non-flags metadata type (ValueRange, UILabel, UIGroup,
UIDragSpeed, UIEnableIf, etc.) should appear at most once per
property. If duplicated, only the first will be found by
find_metadata<T>().)doc";

static const char *__doc_falcor_reflection_detail_is_default_value = R"doc(Trait to detect DefaultValue<T> specializations.)doc";

static const char *__doc_falcor_reflection_enum_descriptor = R"doc(Convenience factory for EnumDescriptor.)doc";

static const char *__doc_falcor_reflection_find_enum_info_adl = R"doc()doc";

static const char *__doc_falcor_reflection_flip_bit = R"doc()doc";

static const char *__doc_falcor_reflection_get_class_descriptor = R"doc()doc";

static const char *__doc_falcor_reflection_get_class_descriptor_2 =
R"doc(Look up a class descriptor by type_info. Throws if the type is not
registered.)doc";

static const char *__doc_falcor_reflection_is_reflection_flags =
R"doc(Trait to check if a type is a reflection flags enum (OR-accumulated).
Flags types are also valid reflection metadata. During
extract_metadata, multiple values of the same flags type are OR'd
together into a single entry.)doc";

static const char *__doc_falcor_reflection_is_reflection_flags_2 = R"doc()doc";

static const char *__doc_falcor_reflection_is_reflection_metadata = R"doc(Trait to check if a type is a falcor reflection metadata type.)doc";

static const char *__doc_falcor_reflection_is_reflection_metadata_2 = R"doc()doc";

static const char *__doc_falcor_reflection_is_reflection_metadata_3 = R"doc()doc";

static const char *__doc_falcor_reflection_is_reflection_metadata_4 = R"doc()doc";

static const char *__doc_falcor_reflection_is_reflection_metadata_5 = R"doc()doc";

static const char *__doc_falcor_reflection_is_reflection_metadata_6 = R"doc()doc";

static const char *__doc_falcor_reflection_is_reflection_metadata_7 = R"doc()doc";

static const char *__doc_falcor_reflection_is_reflection_metadata_8 = R"doc()doc";

static const char *__doc_falcor_reflection_is_reflection_metadata_9 = R"doc()doc";

static const char *__doc_falcor_reflection_is_reflection_metadata_10 = R"doc()doc";

static const char *__doc_falcor_reflection_is_set = R"doc()doc";

static const char *__doc_falcor_reflection_on_change =
R"doc(Convenience factory for OnChange. Hides the void* cast from the
caller.)doc";

static const char *__doc_falcor_reflection_on_change_2 =
R"doc(Convenience factory for OnChange from a member function pointer.
Usage: on_change<MyClass>(&MyClass::mark_dirty))doc";

static const char *__doc_falcor_reflection_operator_band = R"doc()doc";

static const char *__doc_falcor_reflection_operator_bnot = R"doc()doc";

static const char *__doc_falcor_reflection_operator_bor = R"doc()doc";

static const char *__doc_falcor_reflection_operator_iand = R"doc()doc";

static const char *__doc_falcor_reflection_operator_ior = R"doc()doc";

static const char *__doc_falcor_reflection_register_type =
R"doc(Register a ReflectedClass type with the TypeRegistry. Creates a
ClassDescriptor using T::reflected_class_name() and T::ReflectedBase,
then calls T::reflect() with a RegistryClassReflector to populate
properties.)doc";

static const char *__doc_falcor_reflection_serialize_properties =
R"doc(Serialize the properties of a reflected object into a Properties
object. @warning This does not handle properties defined in the base
class.)doc";

static const char *__doc_falcor_reflection_ui_drag_speed = R"doc(Convenience factory for UIDragSpeed.)doc";

static const char *__doc_falcor_reflection_ui_editor = R"doc(Convenience factory for UIEditor.)doc";

static const char *__doc_falcor_reflection_ui_enable_if =
R"doc(Convenience factory for UIEnableIf. Hides the void* cast from the
caller.)doc";

static const char *__doc_falcor_reflection_ui_group = R"doc(Convenience factory for UIGroup.)doc";

static const char *__doc_falcor_reflection_ui_label = R"doc(Convenience factory for UILabel.)doc";

static const char *__doc_falcor_reflection_value_range = R"doc(Convenience factory for ValueRange.)doc";

static const char *__doc_falcor_register_shutdown_callback = R"doc()doc";

static const char *__doc_falcor_replace_substring = R"doc()doc";

static const char *__doc_falcor_replace_substrings = R"doc()doc";

static const char *__doc_falcor_set =
R"doc(Set a property value.

Parameter ``name``:
    Property name.

Parameter ``value``:
    Value to set.

Parameter ``throw_if_exist``:
    Throw an exception if a property with the same name already
    exists.)doc";

static const char *__doc_falcor_set_2 = R"doc(@overload)doc";

static const char *__doc_falcor_set_importer_cache =
R"doc(Set the global blob cache used by importers (e.g. for MikkTSpace
caching).)doc";

static const char *__doc_falcor_static_init = R"doc()doc";

static const char *__doc_falcor_static_shutdown = R"doc()doc";

static const char *__doc_falcor_to_handle = R"doc()doc";

static const char *__doc_falcor_to_handle_2 = R"doc()doc";

static const char *__doc_falcor_try_get =
R"doc(Try to get a property value by index without implicit conversions.

Parameter ``index``:
    Index of the property.

Returns:
    A pointer to the stored value, or nullptr on type mismatch.)doc";

static const char *__doc_falcor_ui_CameraController =
R"doc(Controls camera navigation in a 3D viewport.

Provides Unreal Editor-style camera controls with multiple interaction
modes:

## Navigation Modes

- **First-person (RMB held):** WASD/QE movement with mouse-look
rotation. Shift for fast movement, Ctrl for slow. Scroll wheel adjusts
move speed. - **Pan (MMB held):** Pans the camera in the view plane. -
**Orbit (Alt+LMB held):** Tumbles the camera around the pivot point.
Vertical rotation is clamped to prevent gimbal lock. - **Dolly
(Alt+RMB held):** Zooms toward or away from the pivot point. - **Track
(Alt+MMB held):** Pans the camera and pivot together in the view
plane. Pan speed scales with orbit distance for a consistent feel at
any zoom level. - **Scroll (idle):** Dollies toward or away from the
pivot point.

## Pivot Point

The pivot is the point around which orbit and dolly operations
revolve. It is automatically kept at `orbit_distance` ahead of the
camera along its forward vector after first-person movement, and can
be set explicitly via `set_pivot()` or `focus()`. Drag and track
operations translate the pivot along with the camera.

## Usage

Feed keyboard and mouse events via `handle_keyboard_event()` /
`handle_mouse_event()`, then call `update(dt)` once per frame. The
controller updates its internal transform; the caller is responsible
for applying it back to the camera entity.)doc";

static const char *__doc_falcor_ui_CameraController_2 = R"doc()doc";

static const char *__doc_falcor_ui_CameraController_CameraController = R"doc()doc";

static const char *__doc_falcor_ui_CameraController_State = R"doc(Interaction state of the controller.)doc";

static const char *__doc_falcor_ui_CameraController_State_dolly = R"doc(Alt+RMB held: zoom toward/away from pivot.)doc";

static const char *__doc_falcor_ui_CameraController_State_first_person = R"doc(RMB held: fly-mode with WASD/QE.)doc";

static const char *__doc_falcor_ui_CameraController_State_idle = R"doc(No active interaction.)doc";

static const char *__doc_falcor_ui_CameraController_State_info = R"doc()doc";

static const char *__doc_falcor_ui_CameraController_State_orbit = R"doc(Alt+LMB held: tumble around pivot.)doc";

static const char *__doc_falcor_ui_CameraController_State_pan = R"doc(MMB held: pan in view plane.)doc";

static const char *__doc_falcor_ui_CameraController_State_track = R"doc(Alt+MMB held: pan camera and pivot together.)doc";

static const char *__doc_falcor_ui_CameraController_apply_orbit =
R"doc(Apply orbit at the given yaw/pitch angular velocities
(radians/second). Convenience wrapper over apply_orbit_delta for
inertia coast.)doc";

static const char *__doc_falcor_ui_CameraController_apply_orbit_delta =
R"doc(Apply an orbit by the given yaw/pitch angular deltas (radians). Core
implementation shared by all orbit paths.)doc";

static const char *__doc_falcor_ui_CameraController_apply_scroll_dolly =
R"doc(Apply a scroll dolly by the given amount. Positive scrolls toward the
pivot. Uses multiplicative zoom so the camera can never reach the
pivot.)doc";

static const char *__doc_falcor_ui_CameraController_capture_callback = R"doc(Capture callback.)doc";

static const char *__doc_falcor_ui_CameraController_focus =
R"doc(Focus the camera on a target point. Sets the pivot to `target` and
repositions the camera at `distance` along its current backward
direction so that it looks at the target.

Parameter ``target``:
    World-space point to focus on.

Parameter ``distance``:
    Distance from target to place the camera.)doc";

static const char *__doc_falcor_ui_CameraController_handle_keyboard_event = R"doc(Process a keyboard event. Updates modifier and movement key state.)doc";

static const char *__doc_falcor_ui_CameraController_handle_mouse_event =
R"doc(Process a mouse event. Manages mode transitions and accumulates mouse
delta/scroll.)doc";

static const char *__doc_falcor_ui_CameraController_is_captured =
R"doc(Whether the controller currently has the mouse captured (i.e. is in an
active mode).)doc";

static const char *__doc_falcor_ui_CameraController_m_alt_down = R"doc()doc";

static const char *__doc_falcor_ui_CameraController_m_capture_callback = R"doc(Callback for cursor capture/release.)doc";

static const char *__doc_falcor_ui_CameraController_m_captured = R"doc(Whether the controller currently has the mouse captured.)doc";

static const char *__doc_falcor_ui_CameraController_m_ctrl_down = R"doc()doc";

static const char *__doc_falcor_ui_CameraController_m_first_mouse_delta = R"doc()doc";

static const char *__doc_falcor_ui_CameraController_m_look_velocity = R"doc(Smoothed first-person look velocity (yaw, pitch) in radians/second.)doc";

static const char *__doc_falcor_ui_CameraController_m_mouse_delta = R"doc()doc";

static const char *__doc_falcor_ui_CameraController_m_mouse_pos = R"doc()doc";

static const char *__doc_falcor_ui_CameraController_m_mouse_scroll_delta = R"doc()doc";

static const char *__doc_falcor_ui_CameraController_m_move_key_down = R"doc()doc";

static const char *__doc_falcor_ui_CameraController_m_move_speed = R"doc(First-person movement speed in world units per second.)doc";

static const char *__doc_falcor_ui_CameraController_m_orbit_distance = R"doc(Distance from camera to pivot.)doc";

static const char *__doc_falcor_ui_CameraController_m_orbit_velocity = R"doc(Orbit angular velocity in radians/second (yaw, pitch) for inertia.)doc";

static const char *__doc_falcor_ui_CameraController_m_pivot = R"doc(World-space pivot point for orbit/dolly operations.)doc";

static const char *__doc_falcor_ui_CameraController_m_rotate_speed = R"doc(Mouse rotation sensitivity in radians per pixel.)doc";

static const char *__doc_falcor_ui_CameraController_m_scroll_velocity = R"doc(Scroll dolly velocity in units/second for smooth scrolling.)doc";

static const char *__doc_falcor_ui_CameraController_m_shift_down = R"doc()doc";

static const char *__doc_falcor_ui_CameraController_m_smoothing_enabled = R"doc(Whether smoothing/inertia is enabled.)doc";

static const char *__doc_falcor_ui_CameraController_m_state = R"doc(Current interaction state.)doc";

static const char *__doc_falcor_ui_CameraController_m_transform = R"doc(Camera transform (position, rotation, scale).)doc";

static const char *__doc_falcor_ui_CameraController_move_speed = R"doc(Movement speed in world units per second (first-person mode).)doc";

static const char *__doc_falcor_ui_CameraController_orbit_distance = R"doc(Distance from the camera to the pivot point.)doc";

static const char *__doc_falcor_ui_CameraController_pivot = R"doc(Pivot point used as the center of orbit/dolly operations.)doc";

static const char *__doc_falcor_ui_CameraController_set_capture = R"doc(Invoke the capture callback.)doc";

static const char *__doc_falcor_ui_CameraController_set_capture_callback =
R"doc(Set the capture callback. Called with `true` when entering an active
mode (e.g. first-person, orbit) and `false` when returning to idle.)doc";

static const char *__doc_falcor_ui_CameraController_set_move_speed = R"doc(Set the movement speed, clamped to [MIN_MOVE_SPEED, MAX_MOVE_SPEED].)doc";

static const char *__doc_falcor_ui_CameraController_set_orbit_distance =
R"doc(Set the orbit distance. Recomputes the pivot along the camera's
forward vector.)doc";

static const char *__doc_falcor_ui_CameraController_set_pivot =
R"doc(Set the pivot point. Updates orbit distance to match the camera-to-
pivot distance.)doc";

static const char *__doc_falcor_ui_CameraController_set_smoothing = R"doc(Enable or disable smoothing/inertia for orbit and scroll.)doc";

static const char *__doc_falcor_ui_CameraController_set_transform =
R"doc(Set the camera transform. Also recomputes the pivot from the current
orbit distance.)doc";

static const char *__doc_falcor_ui_CameraController_smoothing = R"doc(Whether smoothing/inertia is enabled for orbit and scroll.)doc";

static const char *__doc_falcor_ui_CameraController_state = R"doc(Current interaction mode state.)doc";

static const char *__doc_falcor_ui_CameraController_transform = R"doc(Camera transform.)doc";

static const char *__doc_falcor_ui_CameraController_update =
R"doc(Advance the controller by `dt` seconds. Applies movement, rotation,
orbit, etc. based on the current mode and accumulated input.

Returns:
    True if the transform changed this frame.)doc";

static const char *__doc_falcor_ui_CameraController_update_dolly = R"doc()doc";

static const char *__doc_falcor_ui_CameraController_update_first_person = R"doc()doc";

static const char *__doc_falcor_ui_CameraController_update_orbit = R"doc()doc";

static const char *__doc_falcor_ui_CameraController_update_pan = R"doc()doc";

static const char *__doc_falcor_ui_CameraController_update_pivot_from_camera =
R"doc(Recompute the pivot point from the camera position and forward
direction.)doc";

static const char *__doc_falcor_ui_CameraController_update_scroll =
R"doc(Per-mode update functions. Each returns true if the transform was
modified.)doc";

static const char *__doc_falcor_ui_CameraController_update_track = R"doc()doc";

static const char *__doc_falcor_ui_PropertyEditorContext = R"doc()doc";

static const char *__doc_falcor_ui_PropertyEditorContext_2 =
R"doc(Context passed through to property editor functions. Provides options
and caches layout information across frames.)doc";

static const char *__doc_falcor_ui_PropertyEditorContext_PropertyEditorContext = R"doc()doc";

static const char *__doc_falcor_ui_PropertyEditorContext_layout_cache = R"doc(Cached layout data, persists across frames.)doc";

static const char *__doc_falcor_ui_PropertyEditorContext_show_advanced = R"doc(Whether to show Advanced-tagged properties.)doc";

static const char *__doc_falcor_ui_PropertyEditorContext_show_read_only = R"doc(Whether to show read-only properties.)doc";

static const char *__doc_falcor_ui_PropertyLayout =
R"doc(Cached tree structure representing the grouped, ordered property list
for a given class + dynamic property set combination.)doc";

static const char *__doc_falcor_ui_PropertyLayoutCache =
R"doc(Bounded LRU cache of PropertyLayout objects, keyed by instance
pointer.)doc";

static const char *__doc_falcor_ui_PropertyLayoutCache_Entry = R"doc()doc";

static const char *__doc_falcor_ui_PropertyLayoutCache_Entry_key = R"doc()doc";

static const char *__doc_falcor_ui_PropertyLayoutCache_Entry_layout = R"doc()doc";

static const char *__doc_falcor_ui_PropertyLayoutCache_PropertyLayoutCache = R"doc(Constructor.)doc";

static const char *__doc_falcor_ui_PropertyLayoutCache_clear = R"doc(Remove all cached entries.)doc";

static const char *__doc_falcor_ui_PropertyLayoutCache_find =
R"doc(Look up a layout for the given instance. Returns nullptr if not
cached.)doc";

static const char *__doc_falcor_ui_PropertyLayoutCache_insert =
R"doc(Insert or replace a layout for the given instance, evicting LRU if at
capacity.)doc";

static const char *__doc_falcor_ui_PropertyLayoutCache_m_capacity = R"doc()doc";

static const char *__doc_falcor_ui_PropertyLayoutCache_m_entries = R"doc(LRU list: most recently used at front.)doc";

static const char *__doc_falcor_ui_PropertyLayoutCache_m_index = R"doc(Map from instance pointer to list iterator for O(1) lookup.)doc";

static const char *__doc_falcor_ui_PropertyLayoutNode = R"doc(A node in the grouped property layout tree.)doc";

static const char *__doc_falcor_ui_PropertyLayoutNode_children = R"doc(Nested subgroups, ordered by first appearance.)doc";

static const char *__doc_falcor_ui_PropertyLayoutNode_name = R"doc(Group name (empty for the root node).)doc";

static const char *__doc_falcor_ui_PropertyLayoutNode_properties = R"doc(Properties in this group, in registration order.)doc";

static const char *__doc_falcor_ui_PropertyLayout_class_desc = R"doc(ClassDescriptor used to build this layout.)doc";

static const char *__doc_falcor_ui_PropertyLayout_dynamic_generation =
R"doc(Generation of DynamicPropertySet when layout was built (0 if no
dynamic props).)doc";

static const char *__doc_falcor_ui_PropertyLayout_dynamic_set =
R"doc(Pointer to the DynamicPropertySet used to build this layout (nullptr
if none).)doc";

static const char *__doc_falcor_ui_PropertyLayout_root = R"doc(Top-level node; root.properties = ungrouped properties.)doc";

static const char *__doc_falcor_ui_SceneEditor = R"doc()doc";

static const char *__doc_falcor_ui_SceneEditor_2 = R"doc()doc";

static const char *__doc_falcor_ui_SceneEditor_HelpEntry = R"doc(A single key/description pair shown in the help popup.)doc";

static const char *__doc_falcor_ui_SceneEditor_HelpEntry_description = R"doc()doc";

static const char *__doc_falcor_ui_SceneEditor_HelpEntry_key = R"doc()doc";

static const char *__doc_falcor_ui_SceneEditor_HelpSection = R"doc(A titled group of help entries shown in the help popup.)doc";

static const char *__doc_falcor_ui_SceneEditor_HelpSection_entries = R"doc()doc";

static const char *__doc_falcor_ui_SceneEditor_HelpSection_title = R"doc()doc";

static const char *__doc_falcor_ui_SceneEditor_SceneEditor = R"doc()doc";

static const char *__doc_falcor_ui_SceneEditor_ToolMode = R"doc()doc";

static const char *__doc_falcor_ui_SceneEditor_ToolMode_info = R"doc()doc";

static const char *__doc_falcor_ui_SceneEditor_ToolMode_move = R"doc()doc";

static const char *__doc_falcor_ui_SceneEditor_ToolMode_rotate = R"doc()doc";

static const char *__doc_falcor_ui_SceneEditor_ToolMode_scale = R"doc()doc";

static const char *__doc_falcor_ui_SceneEditor_ToolMode_select = R"doc()doc";

static const char *__doc_falcor_ui_SceneEditor_TransformSpace = R"doc()doc";

static const char *__doc_falcor_ui_SceneEditor_TransformSpace_info = R"doc()doc";

static const char *__doc_falcor_ui_SceneEditor_TransformSpace_local = R"doc()doc";

static const char *__doc_falcor_ui_SceneEditor_TransformSpace_world = R"doc()doc";

static const char *__doc_falcor_ui_SceneEditor_ViewportState = R"doc()doc";

static const char *__doc_falcor_ui_SceneEditor_ViewportState_focused =
R"doc(Whether the viewport was focused in the last rendered viewport
snapshot.)doc";

static const char *__doc_falcor_ui_SceneEditor_ViewportState_hovered =
R"doc(Whether the viewport was hovered in the last rendered viewport
snapshot.)doc";

static const char *__doc_falcor_ui_SceneEditor_ViewportState_pos = R"doc(Viewport position in screen coordinates (top-left corner).)doc";

static const char *__doc_falcor_ui_SceneEditor_ViewportState_size = R"doc(Viewport size in pixels.)doc";

static const char *__doc_falcor_ui_SceneEditor_ViewportState_valid = R"doc(Whether the last rendered viewport snapshot has a non-zero size.)doc";

static const char *__doc_falcor_ui_SceneEditor_add_component_popup = R"doc()doc";

static const char *__doc_falcor_ui_SceneEditor_add_help_section =
R"doc(Add an extra section of help entries (e.g. application-level
shortcuts). These are displayed after the built-in camera and editor
sections.)doc";

static const char *__doc_falcor_ui_SceneEditor_add_scene_object_popup = R"doc()doc";

static const char *__doc_falcor_ui_SceneEditor_begin_frame =
R"doc(Optional explicit ImGuizmo initialization for the current ImGui frame.
viewport_ui() will initialize ImGuizmo automatically if this was not
called.)doc";

static const char *__doc_falcor_ui_SceneEditor_camera_controller = R"doc(Optional camera controller whose move speed is shown in the toolbar.)doc";

static const char *__doc_falcor_ui_SceneEditor_can_pick_at =
R"doc(Test whether a screen-space position can be used for viewport picking.
Returns true if the position is inside the viewport and the gizmo is
not active. On success, writes the viewport-local pixel coordinates to
@p local_pos.)doc";

static const char *__doc_falcor_ui_SceneEditor_class_name = R"doc()doc";

static const char *__doc_falcor_ui_SceneEditor_component_ui = R"doc()doc";

static const char *__doc_falcor_ui_SceneEditor_editor_ui = R"doc(Draw the scene editor UI.)doc";

static const char *__doc_falcor_ui_SceneEditor_entity_node = R"doc()doc";

static const char *__doc_falcor_ui_SceneEditor_entity_ui = R"doc()doc";

static const char *__doc_falcor_ui_SceneEditor_geometry_instance_ui = R"doc()doc";

static const char *__doc_falcor_ui_SceneEditor_geometry_ui = R"doc()doc";

static const char *__doc_falcor_ui_SceneEditor_help_ui = R"doc()doc";

static const char *__doc_falcor_ui_SceneEditor_inspector_ui = R"doc()doc";

static const char *__doc_falcor_ui_SceneEditor_is_gizmo_active = R"doc(Returns true if the gizmo is currently hovered or being manipulated.)doc";

static const char *__doc_falcor_ui_SceneEditor_is_viewport_interactive =
R"doc(Whether the viewport is currently interactive (valid and hovered).
Uses the viewport state from the last rendered frame.)doc";

static const char *__doc_falcor_ui_SceneEditor_looping = R"doc(Whether animation looping is enabled.)doc";

static const char *__doc_falcor_ui_SceneEditor_m_camera_controller = R"doc()doc";

static const char *__doc_falcor_ui_SceneEditor_m_gizmo_frame_initialized = R"doc()doc";

static const char *__doc_falcor_ui_SceneEditor_m_help_sections = R"doc()doc";

static const char *__doc_falcor_ui_SceneEditor_m_layout_initialized = R"doc()doc";

static const char *__doc_falcor_ui_SceneEditor_m_looping = R"doc()doc";

static const char *__doc_falcor_ui_SceneEditor_m_playing = R"doc(Animation transport state.)doc";

static const char *__doc_falcor_ui_SceneEditor_m_proj_matrix = R"doc()doc";

static const char *__doc_falcor_ui_SceneEditor_m_property_editor_ctx = R"doc()doc";

static const char *__doc_falcor_ui_SceneEditor_m_scene = R"doc()doc";

static const char *__doc_falcor_ui_SceneEditor_m_selected_object = R"doc()doc";

static const char *__doc_falcor_ui_SceneEditor_m_selection_version = R"doc()doc";

static const char *__doc_falcor_ui_SceneEditor_m_show_help = R"doc()doc";

static const char *__doc_falcor_ui_SceneEditor_m_tool_mode = R"doc()doc";

static const char *__doc_falcor_ui_SceneEditor_m_transform_space = R"doc()doc";

static const char *__doc_falcor_ui_SceneEditor_m_view_matrix = R"doc()doc";

static const char *__doc_falcor_ui_SceneEditor_m_view_matrix_no_scale = R"doc()doc";

static const char *__doc_falcor_ui_SceneEditor_m_viewport_state = R"doc()doc";

static const char *__doc_falcor_ui_SceneEditor_m_visible = R"doc()doc";

static const char *__doc_falcor_ui_SceneEditor_material_ui = R"doc()doc";

static const char *__doc_falcor_ui_SceneEditor_outliner_ui = R"doc()doc";

static const char *__doc_falcor_ui_SceneEditor_playing = R"doc(Whether animation playback is active.)doc";

static const char *__doc_falcor_ui_SceneEditor_remove_selected_object =
R"doc(Remove the currently selected object from the scene and clear the
selection.)doc";

static const char *__doc_falcor_ui_SceneEditor_scene = R"doc(The scene being edited, or nullptr if no scene is loaded.)doc";

static const char *__doc_falcor_ui_SceneEditor_scene_collection_combo = R"doc()doc";

static const char *__doc_falcor_ui_SceneEditor_scene_object_node = R"doc()doc";

static const char *__doc_falcor_ui_SceneEditor_scene_object_ui = R"doc()doc";

static const char *__doc_falcor_ui_SceneEditor_selected_object =
R"doc(The currently selected scene object, or nullptr if no object is
selected.)doc";

static const char *__doc_falcor_ui_SceneEditor_selection_version =
R"doc(Monotonically increasing version counter that changes whenever the
selection changes. Consumers can cache the value and compare to detect
changes without consuming state.)doc";

static const char *__doc_falcor_ui_SceneEditor_set_active_camera = R"doc(Set the active camera used for gizmo rendering.)doc";

static const char *__doc_falcor_ui_SceneEditor_set_camera_controller = R"doc()doc";

static const char *__doc_falcor_ui_SceneEditor_set_camera_matrices =
R"doc(Set the camera view and projection matrices used for rendering the
gizmo.)doc";

static const char *__doc_falcor_ui_SceneEditor_set_looping = R"doc(Enable or disable animation looping.)doc";

static const char *__doc_falcor_ui_SceneEditor_set_playing = R"doc(Start or stop animation playback.)doc";

static const char *__doc_falcor_ui_SceneEditor_set_scene = R"doc()doc";

static const char *__doc_falcor_ui_SceneEditor_set_selected_object = R"doc()doc";

static const char *__doc_falcor_ui_SceneEditor_set_tool_mode = R"doc()doc";

static const char *__doc_falcor_ui_SceneEditor_set_transform_space = R"doc()doc";

static const char *__doc_falcor_ui_SceneEditor_set_visible = R"doc()doc";

static const char *__doc_falcor_ui_SceneEditor_setup_default_layout = R"doc()doc";

static const char *__doc_falcor_ui_SceneEditor_setup_dockspace = R"doc()doc";

static const char *__doc_falcor_ui_SceneEditor_tool_mode = R"doc(Current tool mode (select, move, rotate, scale).)doc";

static const char *__doc_falcor_ui_SceneEditor_toolbar_ui = R"doc()doc";

static const char *__doc_falcor_ui_SceneEditor_transform_space = R"doc(Current transform space for the gizmo (world or local).)doc";

static const char *__doc_falcor_ui_SceneEditor_transport_ui = R"doc()doc";

static const char *__doc_falcor_ui_SceneEditor_update_playback =
R"doc(Advance animation playback by @p dt seconds (call once per frame).
When playing, advances the scene time and optionally loops.)doc";

static const char *__doc_falcor_ui_SceneEditor_viewport_gizmo = R"doc()doc";

static const char *__doc_falcor_ui_SceneEditor_viewport_state = R"doc(Last rendered viewport state.)doc";

static const char *__doc_falcor_ui_SceneEditor_viewport_ui = R"doc(Draw the viewport with the rendered scene and the gizmo.)doc";

static const char *__doc_falcor_ui_SceneEditor_visible =
R"doc(Whether the editor UI is visible. Used by SceneInteractionController
to gate viewport interactions.)doc";

static const char *__doc_falcor_ui_SceneInteractionController =
R"doc(Coordinates input routing between the scene editor, camera controller,
picker, and selection overlay.

Centralizes the input routing order, viewport-local coordinate
remapping, gizmo suppression, camera capture side effects, and
selection picking that would otherwise be scattered across the
application.

The controller does not own any of its services; they are injected by
the application and can be null (in which case the corresponding
feature is disabled).)doc";

static const char *__doc_falcor_ui_SceneInteractionController_SceneInteractionController = R"doc()doc";

static const char *__doc_falcor_ui_SceneInteractionController_camera_controller = R"doc(Camera controller used for navigation.)doc";

static const char *__doc_falcor_ui_SceneInteractionController_class_name = R"doc()doc";

static const char *__doc_falcor_ui_SceneInteractionController_focus_on_selection =
R"doc(Focus the camera on the currently selected entity.

Parameter ``camera``:
    The camera to update, or nullptr if no camera is active.

Returns:
    True if the camera was moved.)doc";

static const char *__doc_falcor_ui_SceneInteractionController_handle_keyboard_event =
R"doc(Process a keyboard event. Routes through capture state, UI, and camera
controller.

Returns:
    True if the event was consumed.)doc";

static const char *__doc_falcor_ui_SceneInteractionController_handle_mouse_event =
R"doc(Process a mouse event. Routes through capture state, UI, picker, and
camera controller.

Returns:
    True if the event was consumed.)doc";

static const char *__doc_falcor_ui_SceneInteractionController_m_camera_controller = R"doc()doc";

static const char *__doc_falcor_ui_SceneInteractionController_m_last_selection_version = R"doc()doc";

static const char *__doc_falcor_ui_SceneInteractionController_m_reset_callback = R"doc()doc";

static const char *__doc_falcor_ui_SceneInteractionController_m_scene = R"doc()doc";

static const char *__doc_falcor_ui_SceneInteractionController_m_scene_editor = R"doc()doc";

static const char *__doc_falcor_ui_SceneInteractionController_m_scene_picker = R"doc()doc";

static const char *__doc_falcor_ui_SceneInteractionController_m_selection_overlay = R"doc()doc";

static const char *__doc_falcor_ui_SceneInteractionController_request_reset = R"doc()doc";

static const char *__doc_falcor_ui_SceneInteractionController_reset_callback =
R"doc(Callback invoked when interaction requires accumulation reset (e.g.
selection change, focus on selection).)doc";

static const char *__doc_falcor_ui_SceneInteractionController_scene = R"doc(Scene for picking operations.)doc";

static const char *__doc_falcor_ui_SceneInteractionController_scene_editor = R"doc(Scene editor used for gizmos and selection.)doc";

static const char *__doc_falcor_ui_SceneInteractionController_scene_picker = R"doc(Scene picker used for viewport picking.)doc";

static const char *__doc_falcor_ui_SceneInteractionController_selection_overlay = R"doc(Selection overlay used for selection highlighting.)doc";

static const char *__doc_falcor_ui_SceneInteractionController_set_reset_callback = R"doc()doc";

static const char *__doc_falcor_ui_SceneInteractionController_set_scene = R"doc()doc";

static const char *__doc_falcor_ui_SceneInteractionController_update_selection_overlay = R"doc(Update selection overlay if the editor's selection changed.)doc";

static const char *__doc_falcor_ui_ScenePicker = R"doc()doc";

static const char *__doc_falcor_ui_ScenePicker_2 =
R"doc(GPU-based object picking service. Renders geometry instance IDs to an
internal texture using a standalone ray-tracing compute pass, then
performs single-pixel readback to determine which entity was clicked.)doc";

static const char *__doc_falcor_ui_ScenePicker_RenderIdsCS = R"doc()doc";

static const char *__doc_falcor_ui_ScenePicker_RenderIdsCS_kernel = R"doc()doc";

static const char *__doc_falcor_ui_ScenePicker_RenderIdsRT = R"doc()doc";

static const char *__doc_falcor_ui_ScenePicker_RenderIdsRT_pipeline = R"doc()doc";

static const char *__doc_falcor_ui_ScenePicker_RenderIdsRT_program = R"doc()doc";

static const char *__doc_falcor_ui_ScenePicker_RenderIdsRT_shader_table = R"doc()doc";

static const char *__doc_falcor_ui_ScenePicker_ScenePicker = R"doc()doc";

static const char *__doc_falcor_ui_ScenePicker_class_name = R"doc()doc";

static const char *__doc_falcor_ui_ScenePicker_create_pipelines = R"doc()doc";

static const char *__doc_falcor_ui_ScenePicker_find_entity_by_geometry_instance_id =
R"doc(Find the entity that owns the given geometry instance ID. Searches
through all GeometryInstance components in the scene.

Returns:
    The owning entity, or nullptr if the ID is invalid.)doc";

static const char *__doc_falcor_ui_ScenePicker_geometry_instance_id_texture =
R"doc(The internal geometry instance ID texture. Returns nullptr if render()
has not been called yet.)doc";

static const char *__doc_falcor_ui_ScenePicker_m_device = R"doc()doc";

static const char *__doc_falcor_ui_ScenePicker_m_geometry_instance_id_texture = R"doc()doc";

static const char *__doc_falcor_ui_ScenePicker_m_readback_buffer = R"doc()doc";

static const char *__doc_falcor_ui_ScenePicker_m_render_ids_cs = R"doc()doc";

static const char *__doc_falcor_ui_ScenePicker_m_render_ids_rt = R"doc()doc";

static const char *__doc_falcor_ui_ScenePicker_m_requirements_generation = R"doc()doc";

static const char *__doc_falcor_ui_ScenePicker_m_use_raytracing_pipeline = R"doc()doc";

static const char *__doc_falcor_ui_ScenePicker_pick =
R"doc(Pick a geometry instance ID at the given pixel position using either
the internal texture or a user-provided external texture.

Parameter ``position``:
    Pixel coordinates in the texture (viewport-local).

Parameter ``geometry_instance_id_texture``:
    Optional texture containing geometry instance IDs.

Returns:
    The geometry instance ID, or GeometryInstanceID::invalid if
    nothing was hit.)doc";

static const char *__doc_falcor_ui_ScenePicker_pick_entity =
R"doc(Pick an entity at the given pixel position using either the internal
texture or a user-provided external texture.

Parameter ``scene``:
    The scene to search for entities.

Parameter ``position``:
    Pixel coordinates in the texture (viewport-local).

Parameter ``geometry_instance_id_texture``:
    Optional texture containing geometry instance IDs.

Returns:
    The entity at the position, or nullptr if nothing was hit.)doc";

static const char *__doc_falcor_ui_ScenePicker_render =
R"doc(Render geometry instance IDs into the internal texture.
Creates/resizes the texture as needed based on camera dimensions.

Parameter ``command_encoder``:
    The command encoder to record GPU commands into.

Parameter ``scene``:
    The scene to render for picking.

Parameter ``camera``:
    The camera to render from (used for dimensions and picking rays).)doc";

static const char *__doc_falcor_ui_SelectionOverlay = R"doc()doc";

static const char *__doc_falcor_ui_SelectionOverlay_2 =
R"doc(GPU compute pass that renders selection outlines/highlights onto an
output texture. Uses a 5x5 neighborhood sampling approach to detect
silhouette edges of selected entities. Supports rendering outlines for
both visible and occluded selected objects using a selection probe ray
pass that traces through non-selected geometry.)doc";

static const char *__doc_falcor_ui_SelectionOverlay_Options = R"doc(Options for SelectionOverlay.)doc";

static const char *__doc_falcor_ui_SelectionOverlay_Options_fill_opacity = R"doc(Opacity of the fill overlay (0 = no fill).)doc";

static const char *__doc_falcor_ui_SelectionOverlay_Options_selection_color = R"doc(Color used for outlines and fill.)doc";

static const char *__doc_falcor_ui_SelectionOverlay_Options_show_occluded =
R"doc(Whether occluded selections are shown. This comes with a performance
cost due an additional probe ray pass.)doc";

static const char *__doc_falcor_ui_SelectionOverlay_ProbeRT = R"doc()doc";

static const char *__doc_falcor_ui_SelectionOverlay_ProbeRT_pipeline = R"doc()doc";

static const char *__doc_falcor_ui_SelectionOverlay_ProbeRT_program = R"doc()doc";

static const char *__doc_falcor_ui_SelectionOverlay_ProbeRT_shader_table = R"doc()doc";

static const char *__doc_falcor_ui_SelectionOverlay_SelectionOverlay = R"doc()doc";

static const char *__doc_falcor_ui_SelectionOverlay_class_name = R"doc()doc";

static const char *__doc_falcor_ui_SelectionOverlay_clear_selection = R"doc(Clear the selection.)doc";

static const char *__doc_falcor_ui_SelectionOverlay_create_probe_kernel = R"doc()doc";

static const char *__doc_falcor_ui_SelectionOverlay_draw_overlay =
R"doc(Execute the selection overlay pass. Reads selected geometry instances
from geometry_instance_id_texture and writes outlines/highlights onto
output_texture. When Options::show_occluded is true, an additional
probe ray tracing pass is dispatched against the scene to detect
selected objects behind occluders; when it is false, probe tracing is
skipped and only the geometry_instance_id_texture is used for outline
detection.

Parameter ``command_encoder``:
    Command encoder to record commands into.

Parameter ``output_texture``:
    Texture to write the selection overlay output to.

Parameter ``geometry_instance_id_texture``:
    Texture containing geometry instance IDs.

Parameter ``scene``:
    The scene to trace probe rays against.

Parameter ``camera``:
    The camera to generate probe rays from.)doc";

static const char *__doc_falcor_ui_SelectionOverlay_m_aabb_valid = R"doc()doc";

static const char *__doc_falcor_ui_SelectionOverlay_m_device = R"doc()doc";

static const char *__doc_falcor_ui_SelectionOverlay_m_kernel = R"doc()doc";

static const char *__doc_falcor_ui_SelectionOverlay_m_options = R"doc()doc";

static const char *__doc_falcor_ui_SelectionOverlay_m_probe_kernel = R"doc()doc";

static const char *__doc_falcor_ui_SelectionOverlay_m_probe_rt = R"doc()doc";

static const char *__doc_falcor_ui_SelectionOverlay_m_requirements_generation = R"doc()doc";

static const char *__doc_falcor_ui_SelectionOverlay_m_scene_requirements = R"doc()doc";

static const char *__doc_falcor_ui_SelectionOverlay_m_selected_entities = R"doc()doc";

static const char *__doc_falcor_ui_SelectionOverlay_m_selected_hit_texture = R"doc()doc";

static const char *__doc_falcor_ui_SelectionOverlay_m_selection_aabb = R"doc()doc";

static const char *__doc_falcor_ui_SelectionOverlay_m_selection_bitmap = R"doc()doc";

static const char *__doc_falcor_ui_SelectionOverlay_m_selection_bitmap_buffer = R"doc()doc";

static const char *__doc_falcor_ui_SelectionOverlay_m_selection_bitmap_dirty = R"doc()doc";

static const char *__doc_falcor_ui_SelectionOverlay_m_use_raytracing_pipeline = R"doc()doc";

static const char *__doc_falcor_ui_SelectionOverlay_options = R"doc(Options.)doc";

static const char *__doc_falcor_ui_SelectionOverlay_selection_bitmap_clear = R"doc()doc";

static const char *__doc_falcor_ui_SelectionOverlay_selection_bitmap_set = R"doc()doc";

static const char *__doc_falcor_ui_SelectionOverlay_set_options = R"doc(Set options.)doc";

static const char *__doc_falcor_ui_SelectionOverlay_set_selected_entities =
R"doc(Set the selection from a list of entities. Clears the previous
selection.

Parameter ``entities``:
    Entites to select.)doc";

static const char *__doc_falcor_ui_SelectionOverlay_set_selected_entity =
R"doc(Set the selection from a single entity. Clears the previous selection.

Parameter ``entity``:
    Entity to select, or nullptr to clear selection.)doc";

static const char *__doc_falcor_ui_SelectionOverlay_set_selected_geometry_instance_ids =
R"doc(Set the selection from selected geometry instance IDs directly. Clears
the previous selection.

Parameter ``ids``:
    Geometry instance IDs to select.)doc";

static const char *__doc_falcor_ui_SelectionOverlay_update_aabb = R"doc()doc";

static const char *__doc_falcor_ui_ViewportOverlayDesc = R"doc()doc";

static const char *__doc_falcor_ui_ViewportOverlayDesc_fps = R"doc()doc";

static const char *__doc_falcor_ui_ViewportOverlayDesc_screen_pos = R"doc()doc";

static const char *__doc_falcor_ui_ViewportOverlayDesc_screen_size = R"doc()doc";

static const char *__doc_falcor_ui_build_property_layout =
R"doc(Build a PropertyLayout from a class descriptor and optional dynamic
property set.)doc";

static const char *__doc_falcor_ui_build_property_layout_2 =
R"doc(Build a PropertyLayout from a flat span of property descriptors. This
overload is source-agnostic: it works for Python reflected properties,
dynamic property sets, or any other collection of descriptors.)doc";

static const char *__doc_falcor_ui_draw_viewport_overlay = R"doc(Draw a viewport overlay containing an FPS counter.)doc";

static const char *__doc_falcor_ui_find_enum_info_adl = R"doc()doc";

static const char *__doc_falcor_ui_find_enum_info_adl_2 = R"doc()doc";

static const char *__doc_falcor_ui_find_enum_info_adl_3 = R"doc()doc";

static const char *__doc_falcor_ui_properties_editor =
R"doc(Property editor for all properties of a class (including base classes
and dynamic properties). Returns true if any value changed.)doc";

static const char *__doc_falcor_ui_properties_editor_2 =
R"doc(Property editor for a flat span of property descriptors. The cache_key
is used to cache (and invalidate) the layout. The generation counter
detects when the layout needs rebuilding. Returns true if any value
changed.)doc";

static const char *__doc_falcor_ui_properties_editor_3 =
R"doc(Property editor for a Properties objects. Returns true if any value
changed.)doc";

static const char *__doc_falcor_ui_property_editor = R"doc(Editor for a single property. Returns true if value changed.)doc";

static const char *__doc_falcor_ui_toggle_button = R"doc()doc";

static const char *__doc_falcor_ui_transform_editor =
R"doc(ImGui widget for editing a Transform (translation, rotation in
degrees, scale). Returns true if the transform was modified.)doc";

static const char *__doc_falcor_unpack_normal_oct_snorm2x16 =
R"doc(Unpack normal packed 2x 16-bit snorms in the octahedral mapping.

Parameter ``packed``:
    Packed normal in 2x 16-bit snorms.

Returns:
    Unpacked normal vector.)doc";

static const char *__doc_falcor_unpack_normal_oct_snorm2x8 =
R"doc(Unpack normal packed 2x 8-bit snorms in the octahedral mapping.

Parameter ``packed``:
    Packed normal in 2x 8-bit snorms.

Returns:
    Unpacked normal vector.)doc";

static const char *__doc_falcor_unpack_snorm16 =
R"doc(Unpack single 16-bit snorm value to float.

Parameter ``packed``:
    16-bit snorm value in low bits.

Returns:
    Float value in [-1,1].)doc";

static const char *__doc_falcor_unpack_snorm2x16 =
R"doc(Unpack two 16-bit snorm values to float.

Parameter ``packed``:
    16-bit snorm values.

Returns:
    Float value in [-1,1].)doc";

static const char *__doc_falcor_unpack_snorm2x8 =
R"doc(Unpack two 8-bit snorm values to float.

Parameter ``packed``:
    8-bit snorm values in low bits.

Returns:
    Float values in [-1,1].)doc";

static const char *__doc_falcor_unpack_snorm3x8 =
R"doc(Unpack three 8-bit snorm values to float.

Parameter ``packed``:
    8-bit snorm values in low bits.

Returns:
    Float values in [-1,1].)doc";

static const char *__doc_falcor_unpack_snorm4x8 =
R"doc(Unpack four 8-bit snorm values to float.

Parameter ``packed``:
    8-bit snorm values in low bits.

Returns:
    Float values in [-1,1].)doc";

static const char *__doc_falcor_unpack_snorm8 =
R"doc(Unpack single 8-bit snorm value to float.

Parameter ``packed``:
    8-bit snorm value in low bits.

Returns:
    Float value in [-1,1].)doc";

static const char *__doc_falcor_unpack_unorm16 =
R"doc(Unpack single 16-bit unorm value to float.

Parameter ``packed``:
    16-bit unorm value in low bits.

Returns:
    Float value in [0,1].)doc";

static const char *__doc_falcor_unpack_unorm2x16 =
R"doc(Unpack two 16-bit unorm values to float.

Parameter ``packed``:
    16-bit unorm values.

Returns:
    Float value in [0,1].)doc";

static const char *__doc_falcor_unpack_unorm2x8 =
R"doc(Unpack two 8-bit unorm values to float.

Parameter ``packed``:
    8-bit unorm values in low bits.

Returns:
    Float values in [0,1].)doc";

static const char *__doc_falcor_unpack_unorm3x8 =
R"doc(Unpack three 8-bit unorm values to float.

Parameter ``packed``:
    8-bit unorm values in low bits.

Returns:
    Float values in [0,1].)doc";

static const char *__doc_falcor_unpack_unorm4x8 =
R"doc(Unpack four 8-bit unorm values to float.

Parameter ``packed``:
    8-bit unorm values in low bits.

Returns:
    Float values in [0,1].)doc";

static const char *__doc_falcor_unpack_unorm8 =
R"doc(Unpack single 8-bit unorm value to float.

Parameter ``packed``:
    8-bit unorm value in low bits.

Returns:
    Float value in [0,1].)doc";

static const char *__doc_falcor_usd_importer_TesselatorInputCurve = R"doc()doc";

static const char *__doc_falcor_usd_importer_TesselatorInputCurve_Basis = R"doc()doc";

static const char *__doc_falcor_usd_importer_TesselatorInputCurve_Basis_bezier = R"doc()doc";

static const char *__doc_falcor_usd_importer_TesselatorInputCurve_Basis_bspline = R"doc()doc";

static const char *__doc_falcor_usd_importer_TesselatorInputCurve_Basis_catmull_rom = R"doc()doc";

static const char *__doc_falcor_usd_importer_TesselatorInputCurve_Type = R"doc()doc";

static const char *__doc_falcor_usd_importer_TesselatorInputCurve_Type_cubic = R"doc()doc";

static const char *__doc_falcor_usd_importer_TesselatorInputCurve_Type_linear = R"doc()doc";

static const char *__doc_falcor_usd_importer_TesselatorInputCurve_basis = R"doc()doc";

static const char *__doc_falcor_usd_importer_TesselatorInputCurve_default_radius = R"doc(Default radius when widths are not provided.)doc";

static const char *__doc_falcor_usd_importer_TesselatorInputCurve_name = R"doc()doc";

static const char *__doc_falcor_usd_importer_TesselatorInputCurve_points = R"doc()doc";

static const char *__doc_falcor_usd_importer_TesselatorInputCurve_segments_per_span = R"doc(Number of linear segments to generate per cubic span.)doc";

static const char *__doc_falcor_usd_importer_TesselatorInputCurve_type = R"doc()doc";

static const char *__doc_falcor_usd_importer_TesselatorInputCurve_vertex_counts = R"doc()doc";

static const char *__doc_falcor_usd_importer_TesselatorInputCurve_widths = R"doc()doc";

static const char *__doc_falcor_usd_importer_TesselatorInputMesh = R"doc()doc";

static const char *__doc_falcor_usd_importer_TesselatorInputMesh_Subgeometry = R"doc()doc";

static const char *__doc_falcor_usd_importer_TesselatorInputMesh_Subgeometry_face_indices = R"doc()doc";

static const char *__doc_falcor_usd_importer_TesselatorInputMesh_Subgeometry_material_name = R"doc()doc";

static const char *__doc_falcor_usd_importer_TesselatorInputMesh_Subgeometry_name = R"doc()doc";

static const char *__doc_falcor_usd_importer_TesselatorInputMesh_face_indices = R"doc()doc";

static const char *__doc_falcor_usd_importer_TesselatorInputMesh_face_vertex_counts = R"doc()doc";

static const char *__doc_falcor_usd_importer_TesselatorInputMesh_facevarying_interpolation = R"doc()doc";

static const char *__doc_falcor_usd_importer_TesselatorInputMesh_hole_indices = R"doc()doc";

static const char *__doc_falcor_usd_importer_TesselatorInputMesh_name = R"doc()doc";

static const char *__doc_falcor_usd_importer_TesselatorInputMesh_normals = R"doc()doc";

static const char *__doc_falcor_usd_importer_TesselatorInputMesh_normals_interpolation = R"doc()doc";

static const char *__doc_falcor_usd_importer_TesselatorInputMesh_orientation = R"doc()doc";

static const char *__doc_falcor_usd_importer_TesselatorInputMesh_positions = R"doc()doc";

static const char *__doc_falcor_usd_importer_TesselatorInputMesh_refinement_level = R"doc()doc";

static const char *__doc_falcor_usd_importer_TesselatorInputMesh_subdiv_scheme = R"doc()doc";

static const char *__doc_falcor_usd_importer_TesselatorInputMesh_subgeometries = R"doc()doc";

static const char *__doc_falcor_usd_importer_TesselatorInputMesh_uv_interpolation = R"doc()doc";

static const char *__doc_falcor_usd_importer_TesselatorInputMesh_uvs = R"doc()doc";

static const char *__doc_falcor_usd_importer_TesselatorInputMesh_vertex_boundary_interpolation = R"doc()doc";

static const char *__doc_falcor_usd_importer_UsdImporterContext = R"doc()doc";

static const char *__doc_falcor_usd_importer_UsdImporterContext_UsdImporterContext = R"doc()doc";

static const char *__doc_falcor_usd_importer_UsdImporterContext_UsdSceneMutexes = R"doc()doc";

static const char *__doc_falcor_usd_importer_UsdImporterContext_UsdSceneMutexes_2 = R"doc()doc";

static const char *__doc_falcor_usd_importer_UsdImporterContext_UsdSceneMutexes_camera = R"doc()doc";

static const char *__doc_falcor_usd_importer_UsdImporterContext_UsdSceneMutexes_curve = R"doc()doc";

static const char *__doc_falcor_usd_importer_UsdImporterContext_UsdSceneMutexes_light = R"doc()doc";

static const char *__doc_falcor_usd_importer_UsdImporterContext_UsdSceneMutexes_material = R"doc()doc";

static const char *__doc_falcor_usd_importer_UsdImporterContext_UsdSceneMutexes_mesh = R"doc()doc";

static const char *__doc_falcor_usd_importer_UsdImporterContext_UsdSceneMutexes_nodes = R"doc()doc";

static const char *__doc_falcor_usd_importer_UsdImporterContext_UsdSceneMutexes_texture = R"doc()doc";

static const char *__doc_falcor_usd_importer_UsdImporterContext_add_camera = R"doc()doc";

static const char *__doc_falcor_usd_importer_UsdImporterContext_add_camera_2 =
R"doc(Thread safe, scene access is locked. Separated out to be easy to
package into a task. The `ws_from_local` must be captured by value,
rather than referene to m_scene_mutexes.)doc";

static const char *__doc_falcor_usd_importer_UsdImporterContext_add_curve = R"doc()doc";

static const char *__doc_falcor_usd_importer_UsdImporterContext_add_curve_2 = R"doc(Thread safe, scene access is locked.)doc";

static const char *__doc_falcor_usd_importer_UsdImporterContext_add_instance = R"doc()doc";

static const char *__doc_falcor_usd_importer_UsdImporterContext_add_light = R"doc()doc";

static const char *__doc_falcor_usd_importer_UsdImporterContext_add_light_2 =
R"doc(Thread safe, scene access is locked. Separated out to be easy to
package into a task. The `ws_from_local` must be captured by value,
rather than referene to m_scene_mutexes.)doc";

static const char *__doc_falcor_usd_importer_UsdImporterContext_add_mesh = R"doc()doc";

static const char *__doc_falcor_usd_importer_UsdImporterContext_add_mesh_2 =
R"doc(Thread safe, scene access is locked, resolve_material_func is
resolve_material. Separated out to be easy to package into a task. The
`ws_from_local` must be captured by value, rather than referene to
m_scene_mutexes.)doc";

static const char *__doc_falcor_usd_importer_UsdImporterContext_add_point_instancer = R"doc()doc";

static const char *__doc_falcor_usd_importer_UsdImporterContext_add_prototype = R"doc(Returns an index of prototype, creating it if necessary.)doc";

static const char *__doc_falcor_usd_importer_UsdImporterContext_add_root_node =
R"doc(Adds root node to the current scope (scene or prototype in the
future).)doc";

static const char *__doc_falcor_usd_importer_UsdImporterContext_add_texture_to_scene = R"doc(Add a texture to the scene and return its index)doc";

static const char *__doc_falcor_usd_importer_UsdImporterContext_create_light = R"doc()doc";

static const char *__doc_falcor_usd_importer_UsdImporterContext_extract_animations =
R"doc(Extract animation data from USD time samples and populate
ImporterAnimation.)doc";

static const char *__doc_falcor_usd_importer_UsdImporterContext_finalize = R"doc()doc";

static const char *__doc_falcor_usd_importer_UsdImporterContext_get_bound_material = R"doc(Thread safe, locks access to caches.)doc";

static const char *__doc_falcor_usd_importer_UsdImporterContext_get_meters_per_unit =
R"doc(How many meters is a single scene unit. Multiply scene units with this
to get meters.)doc";

static const char *__doc_falcor_usd_importer_UsdImporterContext_get_name = R"doc()doc";

static const char *__doc_falcor_usd_importer_UsdImporterContext_get_scene = R"doc()doc";

static const char *__doc_falcor_usd_importer_UsdImporterContext_is_prototype = R"doc()doc";

static const char *__doc_falcor_usd_importer_UsdImporterContext_m_bindings_cache = R"doc(< Material binding cache)doc";

static const char *__doc_falcor_usd_importer_UsdImporterContext_m_coll_query_cache = R"doc(< Material collection binding cache)doc";

static const char *__doc_falcor_usd_importer_UsdImporterContext_m_material_name_to_index = R"doc()doc";

static const char *__doc_falcor_usd_importer_UsdImporterContext_m_meters_per_unit = R"doc()doc";

static const char *__doc_falcor_usd_importer_UsdImporterContext_m_node_idx_stacks = R"doc()doc";

static const char *__doc_falcor_usd_importer_UsdImporterContext_m_parallel_adds = R"doc()doc";

static const char *__doc_falcor_usd_importer_UsdImporterContext_m_prim_path_to_node_idx =
R"doc(Maps Usd prim paths to their node index in the ImporterScene (for
animation).)doc";

static const char *__doc_falcor_usd_importer_UsdImporterContext_m_prototype_to_index = R"doc(Maps Usd prototype prims to their index in the ImporterScene.)doc";

static const char *__doc_falcor_usd_importer_UsdImporterContext_m_scene = R"doc()doc";

static const char *__doc_falcor_usd_importer_UsdImporterContext_m_scene_mutexes = R"doc()doc";

static const char *__doc_falcor_usd_importer_UsdImporterContext_m_task_group = R"doc()doc";

static const char *__doc_falcor_usd_importer_UsdImporterContext_m_texture_to_index = R"doc()doc";

static const char *__doc_falcor_usd_importer_UsdImporterContext_m_usd_stage = R"doc()doc";

static const char *__doc_falcor_usd_importer_UsdImporterContext_node_idx_stack = R"doc()doc";

static const char *__doc_falcor_usd_importer_UsdImporterContext_node_idx_stack_2 = R"doc()doc";

static const char *__doc_falcor_usd_importer_UsdImporterContext_parent_node_idx = R"doc(Get the current parent node index, -1 if none is present.)doc";

static const char *__doc_falcor_usd_importer_UsdImporterContext_pop_prototype_stack = R"doc()doc";

static const char *__doc_falcor_usd_importer_UsdImporterContext_pop_transform = R"doc()doc";

static const char *__doc_falcor_usd_importer_UsdImporterContext_process_task = R"doc(Run or enqeue tasks, depending on whether we do parallism.)doc";

static const char *__doc_falcor_usd_importer_UsdImporterContext_push_prototype_stack = R"doc()doc";

static const char *__doc_falcor_usd_importer_UsdImporterContext_push_transform = R"doc()doc";

static const char *__doc_falcor_usd_importer_UsdImporterContext_push_transform_2 = R"doc()doc";

static const char *__doc_falcor_usd_importer_UsdImporterContext_push_transform_3 = R"doc()doc";

static const char *__doc_falcor_usd_importer_UsdImporterContext_resolve_material =
R"doc(Thread safe, locks access to m_material_name_to_index and
m_scene.materials.)doc";

static const char *__doc_falcor_usd_importer_UsdImporterContext_root_node_idx =
R"doc(Returns the current root node, used when `GetLocalTransformation` asks
for reset. The reset ignores inheritance in the Usd hierarchy, but we
still want to inherit from the UsdImporter provided transform that
scales and rotates the scene.)doc";

static const char *__doc_falcor_usd_importer_UsdImporterContext_should_drop_prim = R"doc()doc";

static const char *__doc_falcor_usd_importer_UsdImporterContext_traverse_scene = R"doc()doc";

static const char *__doc_falcor_usd_importer_build_material =
R"doc(NOTE: Material graph extraction in build_material is adapted in part
from OpenUSD's UsdImaging material graph utilities
(pxr/usdImaging/usdImaging/materialParamUtils.cpp). Adds material with
the given terminal_identifier (surface, volume, displacement) to the
material.)doc";

static const char *__doc_falcor_usd_importer_get_attribute = R"doc()doc";

static const char *__doc_falcor_usd_importer_get_attribute_2 = R"doc()doc";

static const char *__doc_falcor_usd_importer_get_authored_attribute = R"doc()doc";

static const char *__doc_falcor_usd_importer_get_float2_value = R"doc()doc";

static const char *__doc_falcor_usd_importer_get_float4_value = R"doc()doc";

static const char *__doc_falcor_usd_importer_get_purpose = R"doc()doc";

static const char *__doc_falcor_usd_importer_get_source_input = R"doc()doc";

static const char *__doc_falcor_usd_importer_get_uv_primvar = R"doc()doc";

static const char *__doc_falcor_usd_importer_is_renderable = R"doc()doc";

static const char *__doc_falcor_usd_importer_process_mtlx_references =
R"doc(Expand Scope prims that reference .mtlx files into UsdShade materials
and shaders. Returns number of materials generated.)doc";

static const char *__doc_falcor_usd_importer_set_from_value = R"doc()doc";

static const char *__doc_falcor_usd_importer_set_from_value_2 = R"doc()doc";

static const char *__doc_falcor_usd_importer_tessellate = R"doc()doc";

static const char *__doc_falcor_usd_importer_tessellate_curves = R"doc()doc";

static const char *__doc_falcor_usd_importer_to_falcor = R"doc()doc";

static const char *__doc_falcor_usd_importer_to_falcor_2 = R"doc()doc";

static const char *__doc_falcor_usd_importer_to_falcor_3 = R"doc()doc";

static const char *__doc_falcor_usd_importer_to_falcor_4 = R"doc()doc";

static const char *__doc_falcor_usd_to_mdl = R"doc()doc";

static const char *__doc_falcor_usd_to_mtlx = R"doc()doc";

static const char *__doc_falcor_usdpreviewsurface_to_flattened = R"doc()doc";

static const char *__doc_falcor_usdpreviewsurface_to_mtlx = R"doc()doc";

static const char *__doc_falcor_usdpreviewsurface_to_standardmaterial = R"doc()doc";

static const char *__doc_falcor_usdshade_to_mtlx =
R"doc(Convert a UsdShade-derived MaterialX node network
(implementationSource=id) into a generated .mtlx document reference
consumable by MaterialXMaterial.)doc";

static const char *__doc_falcor_ustring =
R"doc(Class representing an unique (interned) string, see
https://en.wikipedia.org/wiki/String_interning for details.

When a new ustring is created, the string is compared to all
previously created strings stored in a global registry. If found, the
new ustring points to the same existing global string. If not found, a
new string is inserted into the registry. As a result, all ustrings
containing the same string point to the same memory, turning ustring
comparison into pointer comparison.)doc";

static const char *__doc_falcor_ustring_Header = R"doc()doc";

static const char *__doc_falcor_ustring_Header_hash = R"doc()doc";

static const char *__doc_falcor_ustring_Header_length = R"doc()doc";

static const char *__doc_falcor_ustring_Registry = R"doc()doc";

static const char *__doc_falcor_ustring_c_str = R"doc()doc";

static const char *__doc_falcor_ustring_data = R"doc()doc";

static const char *__doc_falcor_ustring_empty = R"doc()doc";

static const char *__doc_falcor_ustring_hash = R"doc()doc";

static const char *__doc_falcor_ustring_header = R"doc()doc";

static const char *__doc_falcor_ustring_length = R"doc()doc";

static const char *__doc_falcor_ustring_m_unique = R"doc()doc";

static const char *__doc_falcor_ustring_make_unique = R"doc()doc";

static const char *__doc_falcor_ustring_operator_assign = R"doc()doc";

static const char *__doc_falcor_ustring_operator_assign_2 = R"doc()doc";

static const char *__doc_falcor_ustring_operator_basic_string = R"doc(Conversion to std::string.)doc";

static const char *__doc_falcor_ustring_operator_basic_string_view = R"doc(Conversion to std::string_view.)doc";

static const char *__doc_falcor_ustring_operator_eq = R"doc()doc";

static const char *__doc_falcor_ustring_operator_eq_2 = R"doc()doc";

static const char *__doc_falcor_ustring_operator_le = R"doc()doc";

static const char *__doc_falcor_ustring_operator_ne = R"doc()doc";

static const char *__doc_falcor_ustring_operator_ne_2 = R"doc()doc";

static const char *__doc_falcor_ustring_reset = R"doc()doc";

static const char *__doc_falcor_ustring_reset_2 = R"doc()doc";

static const char *__doc_falcor_ustring_reset_3 = R"doc()doc";

static const char *__doc_falcor_ustring_size = R"doc()doc";

static const char *__doc_falcor_ustring_string = R"doc()doc";

static const char *__doc_falcor_ustring_ustring = R"doc()doc";

static const char *__doc_falcor_ustring_ustring_2 = R"doc()doc";

static const char *__doc_falcor_ustring_ustring_3 = R"doc()doc";

static const char *__doc_falcor_ustring_ustring_4 = R"doc()doc";

static const char *__doc_falcor_ustring_ustring_5 = R"doc()doc";

static const char *__doc_falcor_ustring_ustring_6 = R"doc()doc";

static const char *__doc_formatter = R"doc()doc";

static const char *__doc_formatter_format = R"doc()doc";

static const char *__doc_is_error_code_enum = R"doc()doc";

static const char *__doc_sgl_Buffer = R"doc()doc";

static const char *__doc_sgl_Buffer_2 = R"doc()doc";

static const char *__doc_sgl_Device = R"doc()doc";

static const char *__doc_sgl_Device_2 = R"doc()doc";

static const char *__doc_sgl_ShaderCursor = R"doc()doc";

static const char *__doc_sgl_SignatureBuffer =
R"doc(Stack-allocated signature buffer. Used in hot paths that need cheap,
append-only signature construction.)doc";

static const char *__doc_sgl_SignatureBuffer_2 =
R"doc(Stack-allocated signature buffer. Used in hot paths that need cheap,
append-only signature construction.)doc";

static const char *__doc_tinygltf_Model = R"doc()doc";

#if defined(__GNUG__)
#pragma GCC diagnostic pop
#endif

