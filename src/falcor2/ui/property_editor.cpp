// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "falcor2/ui/property_editor.h"

#include "falcor2/core/reflected_object.h"
#include "falcor2/core/properties.h"
#include "falcor2/core/types.h"
#include "falcor2/core/reflection/metadata.h"
#include "falcor2/core/reflection/property_range.h"

#include <imgui.h>

#include <array>
#include <limits>
#include <string>
#include <string_view>
#include <type_traits>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>

namespace falcor::ui {

// ----------------------------------------------------------------------------
// PropertyLayoutCache
// ----------------------------------------------------------------------------

PropertyLayoutCache::PropertyLayoutCache(size_t capacity)
    : m_capacity(capacity)
{
}

PropertyLayout* PropertyLayoutCache::find(const void* instance)
{
    auto it = m_index.find(instance);
    if (it == m_index.end())
        return nullptr;
    // Move to front (most recently used).
    m_entries.splice(m_entries.begin(), m_entries, it->second);
    return &it->second->layout;
}

PropertyLayout& PropertyLayoutCache::insert(const void* instance, PropertyLayout layout)
{
    // If already present, update in place.
    auto it = m_index.find(instance);
    if (it != m_index.end()) {
        it->second->layout = std::move(layout);
        m_entries.splice(m_entries.begin(), m_entries, it->second);
        return it->second->layout;
    }

    // Evict LRU if at capacity.
    if (m_entries.size() >= m_capacity) {
        auto& back = m_entries.back();
        m_index.erase(back.key);
        m_entries.pop_back();
    }

    // Insert at front.
    m_entries.push_front(Entry{instance, std::move(layout)});
    m_index[instance] = m_entries.begin();
    return m_entries.front().layout;
}

void PropertyLayoutCache::clear()
{
    m_entries.clear();
    m_index.clear();
}

// ----------------------------------------------------------------------------
// Layout building
// ----------------------------------------------------------------------------

namespace {

/// Find or create a child group node by name within a parent node.
PropertyLayoutNode& find_or_create_child(PropertyLayoutNode& parent, std::string_view name)
{
    for (auto& child : parent.children) {
        if (child.name == name)
            return child;
    }
    parent.children.push_back(PropertyLayoutNode{std::string(name), {}, {}});
    return parent.children.back();
}

/// Assign a property to the correct group node, creating intermediate nodes as needed.
/// Path is split on '/'.
void assign_to_group(PropertyLayoutNode& root, const reflection::PropertyDescriptor* prop)
{
    const reflection::UIGroup* group = prop->ui_group();
    if (!group || group->path.empty()) {
        root.properties.push_back(prop);
        return;
    }

    PropertyLayoutNode* current = &root;
    std::string_view path = group->path;
    while (!path.empty()) {
        size_t sep = path.find('/');
        std::string_view segment = path.substr(0, sep);
        current = &find_or_create_child(*current, segment);
        path = (sep == std::string_view::npos) ? std::string_view{} : path.substr(sep + 1);
    }
    current->properties.push_back(prop);
}

} // namespace

PropertyLayout
build_property_layout(const reflection::ClassDescriptor& class_desc, const reflection::DynamicPropertySet* dynamic_set)
{
    PropertyLayout layout;
    layout.class_desc = &class_desc;
    layout.dynamic_set = dynamic_set;
    layout.dynamic_generation = dynamic_set ? dynamic_set->generation() : 0;

    // Static properties (base-first via all_property_range).
    for (const auto* prop : class_desc.all_property_range()) {
        assign_to_group(layout.root, prop);
    }

    // Dynamic properties.
    if (dynamic_set) {
        for (const auto* prop : dynamic_set->descriptors()) {
            assign_to_group(layout.root, prop);
        }
    }

    return layout;
}

PropertyLayout build_property_layout(std::span<const reflection::PropertyDescriptor* const> descriptors)
{
    PropertyLayout layout;
    for (const auto* prop : descriptors)
        assign_to_group(layout.root, prop);
    return layout;
}

// ----------------------------------------------------------------------------
// Type-dispatch table
// ----------------------------------------------------------------------------

namespace {

using EditorFn = bool (*)(
    const char* label,
    const reflection::PropertyDescriptor& desc,
    void* instance,
    PropertyEditorContext& ctx
);

/// Helper: get drag speed from UIDragSpeed metadata or return default.
float get_drag_speed(const reflection::PropertyDescriptor& desc, float fallback = 1.0f)
{
    const reflection::UIDragSpeed* ds = desc.ui_drag_speed();
    return ds ? ds->speed : fallback;
}

/// Helper: get min/max from ValueRange metadata.
template<typename T>
T range_cast(double value)
{
    double min_value = static_cast<double>(std::numeric_limits<T>::lowest());
    double max_value = static_cast<double>(std::numeric_limits<T>::max());
    if (value <= min_value)
        return std::numeric_limits<T>::lowest();
    if (value >= max_value)
        return std::numeric_limits<T>::max();
    return static_cast<T>(value);
}

template<typename T>
void get_range(const reflection::PropertyDescriptor& desc, T& out_min, T& out_max)
{
    const reflection::ValueRange* vr = desc.value_range();
    if (vr) {
        out_min = range_cast<T>(vr->min);
        out_max = range_cast<T>(vr->max);
    } else {
        out_min = T{};
        out_max = T{};
    }
}

// ----------------------------------------------------------------------------
// Type traits
// ----------------------------------------------------------------------------

template<typename T>
struct ImGuiDataTypeMap;
// clang-format off
template<> struct ImGuiDataTypeMap<int8_t> { static constexpr ImGuiDataType value = ImGuiDataType_S8; };
template<> struct ImGuiDataTypeMap<int16_t> { static constexpr ImGuiDataType value = ImGuiDataType_S16; };
template<> struct ImGuiDataTypeMap<int32_t> { static constexpr ImGuiDataType value = ImGuiDataType_S32; };
template<> struct ImGuiDataTypeMap<int64_t> { static constexpr ImGuiDataType value = ImGuiDataType_S64; };
template<> struct ImGuiDataTypeMap<uint8_t> { static constexpr ImGuiDataType value = ImGuiDataType_U8; };
template<> struct ImGuiDataTypeMap<uint16_t> { static constexpr ImGuiDataType value = ImGuiDataType_U16; };
template<> struct ImGuiDataTypeMap<uint32_t> { static constexpr ImGuiDataType value = ImGuiDataType_U32; };
template<> struct ImGuiDataTypeMap<uint64_t> { static constexpr ImGuiDataType value = ImGuiDataType_U64; };
template<> struct ImGuiDataTypeMap<float> { static constexpr ImGuiDataType value = ImGuiDataType_Float; };
template<> struct ImGuiDataTypeMap<double> { static constexpr ImGuiDataType value = ImGuiDataType_Double; };
// clang-format on

template<typename T>
struct DragEditScalar {
    using type = T;
};

template<>
struct DragEditScalar<float16_t> {
    using type = float;
};

/// Primary template: scalar types (int, float, double, int64_t, ...).
template<typename T, typename = void>
struct DragTraits {
    using scalar_type = T;
    using edit_scalar_type = typename DragEditScalar<scalar_type>::type;
    static constexpr int count = 1;

    static void load(const T& src, std::array<edit_scalar_type, count>& dst)
    {
        dst[0] = static_cast<edit_scalar_type>(src);
    }

    static void store(T& dst, const std::array<edit_scalar_type, count>& src) { dst = T(src[0]); }
};

/// Specialization for SGL vector types (have value_type and dimension).
template<typename T>
struct DragTraits<T, std::void_t<typename T::value_type, decltype(T::dimension)>> {
    using scalar_type = typename T::value_type;
    using edit_scalar_type = typename DragEditScalar<scalar_type>::type;
    static constexpr int count = T::dimension;

    static void load(const T& src, std::array<edit_scalar_type, count>& dst)
    {
        for (int i = 0; i < count; ++i)
            dst[i] = static_cast<edit_scalar_type>(src[i]);
    }

    static void store(T& dst, const std::array<edit_scalar_type, count>& src)
    {
        for (int i = 0; i < count; ++i)
            dst[i] = scalar_type(src[i]);
    }
};

// ----------------------------------------------------------------------------
// Editor functions
// ----------------------------------------------------------------------------

/// Generic drag editor for scalars and vectors.
template<typename T>
bool drag_editor(
    const char* label,
    const reflection::PropertyDescriptor& desc,
    void* instance,
    PropertyEditorContext& /*ctx*/
)
{
    using Traits = DragTraits<T>;
    using EditScalar = typename Traits::edit_scalar_type;
    T value = desc.get<T>(instance);
    std::array<EditScalar, Traits::count> edit_value;
    Traits::load(value, edit_value);

    float speed = get_drag_speed(desc, std::is_floating_point_v<EditScalar> ? 0.01f : 1.0f);
    EditScalar vmin, vmax;
    get_range<EditScalar>(desc, vmin, vmax);
    if (ImGui::DragScalarN(
            label,
            ImGuiDataTypeMap<EditScalar>::value,
            edit_value.data(),
            Traits::count,
            speed,
            &vmin,
            &vmax
        )) {
        Traits::store(value, edit_value);
        desc.set<T>(instance, value);
        return true;
    }
    return false;
}

/// Drag editor with color-edit support for float3/float4 and half3/half4.
template<typename T>
bool drag_or_color_editor(
    const char* label,
    const reflection::PropertyDescriptor& desc,
    void* instance,
    PropertyEditorContext& ctx
)
{
    if ((desc.ui_flags() & reflection::UIFlags::display_as_color) != reflection::UIFlags::none) {
        using Traits = DragTraits<T>;
        using EditScalar = typename Traits::edit_scalar_type;
        static_assert(std::is_same_v<EditScalar, float>);
        static_assert(Traits::count == 3 || Traits::count == 4);

        T value = desc.get<T>(instance);
        std::array<EditScalar, Traits::count> edit_value;
        Traits::load(value, edit_value);

        bool changed = false;
        if constexpr (Traits::count == 3)
            changed = ImGui::ColorEdit3(label, edit_value.data(), ImGuiColorEditFlags_Float);
        else if constexpr (Traits::count == 4)
            changed = ImGui::ColorEdit4(label, edit_value.data(), ImGuiColorEditFlags_Float);

        if (changed) {
            Traits::store(value, edit_value);
            desc.set<T>(instance, value);
            return true;
        }
        return false;
    }
    return drag_editor<T>(label, desc, instance, ctx);
}

/// Matrix editor.
template<typename T>
bool matrix_editor(
    const char* label,
    const reflection::PropertyDescriptor& desc,
    void* instance,
    PropertyEditorContext& /*ctx*/
)
{
    using Scalar = typename T::value_type;
    T value = desc.get<T>(instance);
    float speed = get_drag_speed(desc, 0.01f);
    bool changed = false;

    ImGui::Text("%s", label);
    ImGui::Indent();
    for (int r = 0; r < T::rows; ++r) {
        char row_label[16];
        snprintf(row_label, sizeof(row_label), "Row %d", r);
        if (ImGui::DragScalarN(row_label, ImGuiDataTypeMap<Scalar>::value, &value[r].x, T::cols, speed))
            changed = true;
    }
    ImGui::Unindent();

    if (changed) {
        desc.set<T>(instance, value);
        return true;
    }
    return false;
}

/// Checkbox editor.
bool checkbox_editor(
    const char* label,
    const reflection::PropertyDescriptor& desc,
    void* instance,
    PropertyEditorContext& /*ctx*/
)
{
    bool value = desc.get<bool>(instance);
    if (ImGui::Checkbox(label, &value)) {
        desc.set<bool>(instance, value);
        return true;
    }
    return false;
}

/// String editor.
bool string_editor(
    const char* label,
    const reflection::PropertyDescriptor& desc,
    void* instance,
    PropertyEditorContext& /*ctx*/
)
{
    std::string value = desc.get<std::string>(instance);

    auto resize_callback = [](ImGuiInputTextCallbackData* data) -> int
    {
        if (data->EventFlag == ImGuiInputTextFlags_CallbackResize) {
            auto* str = static_cast<std::string*>(data->UserData);
            str->resize(data->BufTextLen);
            data->Buf = str->data();
        }
        return 0;
    };

    if (ImGui::InputText(
            label,
            value.data(),
            value.capacity() + 1,
            ImGuiInputTextFlags_CallbackResize,
            resize_callback,
            &value
        )) {
        desc.set<std::string>(instance, value);
        return true;
    }
    return false;
}

/// Enum editor.
bool enum_editor(
    const char* label,
    const reflection::PropertyDescriptor& desc,
    void* instance,
    PropertyEditorContext& /*ctx*/
)
{
    FALCOR_ASSERT(desc.enum_descriptor() != nullptr);
    const reflection::EnumDescriptor& enum_desc = *desc.enum_descriptor();

    int64_t current = desc.get_enum_as_int64(instance);

    // Find current item name and index.
    const char* preview = "?";
    int current_idx = -1;
    for (int i = 0; i < static_cast<int>(enum_desc.items.size()); ++i) {
        if (enum_desc.items[i].value == current) {
            preview = enum_desc.items[i].name.c_str();
            current_idx = i;
            break;
        }
    }

    bool changed = false;
    if (ImGui::BeginCombo(label, preview)) {
        for (int i = 0; i < static_cast<int>(enum_desc.items.size()); ++i) {
            bool is_selected = (i == current_idx);
            if (ImGui::Selectable(enum_desc.items[i].name.c_str(), is_selected)) {
                desc.set_enum_from_int64(instance, enum_desc.items[i].value);
                changed = true;
            }
            if (is_selected)
                ImGui::SetItemDefaultFocus();
        }
        ImGui::EndCombo();
    }
    return changed;
}

/// Enum flags editor (bitmask).
bool enum_flags_editor(
    const char* label,
    const reflection::PropertyDescriptor& desc,
    void* instance,
    PropertyEditorContext& /*ctx*/
)
{
    FALCOR_ASSERT(desc.enum_descriptor() != nullptr);
    const reflection::EnumDescriptor& enum_desc = *desc.enum_descriptor();

    int64_t flags = desc.get_enum_as_int64(instance);

    // Build preview string from active flags.
    std::string preview;
    for (const auto& item : enum_desc.items) {
        if (item.value != 0 && (flags & item.value) == item.value) {
            if (!preview.empty())
                preview += " | ";
            preview += item.name;
        }
    }

    bool changed = false;
    if (ImGui::BeginCombo(label, preview.c_str())) {
        for (const auto& item : enum_desc.items) {
            if (item.value == 0)
                continue;
            bool is_active = (flags & item.value) == item.value;
            if (ImGui::Checkbox(item.name.c_str(), &is_active)) {
                if (is_active)
                    flags |= item.value;
                else
                    flags &= ~item.value;
                desc.set_enum_from_int64(instance, flags);
                changed = true;
            }
        }
        ImGui::EndCombo();
    }
    return changed;
}


/// Editor dispatch table mapping type_index to editor functions.
using DispatchTable = std::unordered_map<std::type_index, EditorFn>;

const DispatchTable& get_dispatch_table()
{
    static const DispatchTable table = []
    {
        DispatchTable t;

        t[std::type_index(typeid(bool))] = checkbox_editor;

        t[std::type_index(typeid(int8_t))] = drag_editor<int8_t>;
        t[std::type_index(typeid(int16_t))] = drag_editor<int16_t>;
        t[std::type_index(typeid(int32_t))] = drag_editor<int32_t>;
        t[std::type_index(typeid(int64_t))] = drag_editor<int64_t>;

        t[std::type_index(typeid(uint8_t))] = drag_editor<uint8_t>;
        t[std::type_index(typeid(uint16_t))] = drag_editor<uint16_t>;
        t[std::type_index(typeid(uint32_t))] = drag_editor<uint32_t>;
        t[std::type_index(typeid(uint64_t))] = drag_editor<uint64_t>;

        t[std::type_index(typeid(float))] = drag_editor<float>;
        t[std::type_index(typeid(double))] = drag_editor<double>;
        t[std::type_index(typeid(float16_t))] = drag_editor<float16_t>;

        t[std::type_index(typeid(int2))] = drag_editor<int2>;
        t[std::type_index(typeid(int3))] = drag_editor<int3>;
        t[std::type_index(typeid(int4))] = drag_editor<int4>;

        t[std::type_index(typeid(uint2))] = drag_editor<uint2>;
        t[std::type_index(typeid(uint3))] = drag_editor<uint3>;
        t[std::type_index(typeid(uint4))] = drag_editor<uint4>;

        t[std::type_index(typeid(float2))] = drag_editor<float2>;
        t[std::type_index(typeid(float3))] = drag_or_color_editor<float3>;
        t[std::type_index(typeid(float4))] = drag_or_color_editor<float4>;

        t[std::type_index(typeid(float16_t2))] = drag_editor<float16_t2>;
        t[std::type_index(typeid(float16_t3))] = drag_or_color_editor<float16_t3>;
        t[std::type_index(typeid(float16_t4))] = drag_or_color_editor<float16_t4>;

        t[std::type_index(typeid(float3x3))] = matrix_editor<float3x3>;
        t[std::type_index(typeid(float4x4))] = matrix_editor<float4x4>;

        t[std::type_index(typeid(std::string))] = string_editor;

        return t;
    }();
    return table;
}

/// Look up an editor function for the given type. Returns nullptr if unsupported.
EditorFn find_editor_fn(const reflection::PropertyDescriptor& desc)
{
    if (desc.enum_descriptor())
        return desc.enum_descriptor()->is_flags ? enum_flags_editor : enum_editor;

    const auto& table = get_dispatch_table();
    auto it = table.find(std::type_index(desc.type()));
    return it != table.end() ? it->second : nullptr;
}

} // anonymous namespace

namespace {

/// Check if a property should be visible given the current context.
bool is_property_visible(const reflection::PropertyDescriptor& desc, void* /*instance*/, PropertyEditorContext& ctx)
{
    reflection::UIFlags flags = desc.ui_flags();

    if ((flags & reflection::UIFlags::hidden) != reflection::UIFlags::none)
        return false;

    if ((flags & reflection::UIFlags::advanced) != reflection::UIFlags::none && !ctx.show_advanced)
        return false;

    if (desc.is_read_only() && !ctx.show_read_only)
        return false;

    return true;
}

/// Check if any property in a layout node (recursively) is visible.
bool has_visible_properties(const PropertyLayoutNode& node, void* instance, PropertyEditorContext& ctx)
{
    for (const auto* prop : node.properties) {
        if (is_property_visible(*prop, instance, ctx))
            return true;
    }
    for (const auto& child : node.children) {
        if (has_visible_properties(child, instance, ctx))
            return true;
    }
    return false;
}

bool visit_layout_node(const PropertyLayoutNode& node, void* instance, PropertyEditorContext& ctx)
{
    bool changed = false;

    // Render properties at this level.
    for (const auto* prop : node.properties) {
        if (property_editor(*prop, instance, ctx))
            changed = true;
    }

    // Render child groups.
    for (const auto& child : node.children) {
        if (!has_visible_properties(child, instance, ctx))
            continue;

        if (ImGui::CollapsingHeader(child.name.c_str())) {
            ImGui::Indent();
            if (visit_layout_node(child, instance, ctx))
                changed = true;
            ImGui::Unindent();
        }
    }

    return changed;
}

} // anonymous namespace

// ----------------------------------------------------------------------------
// property_editor
// ----------------------------------------------------------------------------

bool property_editor(const reflection::PropertyDescriptor& desc, void* instance, PropertyEditorContext& ctx)
{
    if (!is_property_visible(desc, instance, ctx))
        return false;

    ImGui::PushID(desc.name().data());

    bool changed = false;

    // Check for custom editor first.
    const reflection::UIEditor* ui_editor = desc.ui_editor();
    if (ui_editor && ui_editor->func) {
        changed = ui_editor->func(instance, desc, ctx);
        ImGui::PopID();
        return changed;
    }

    // Determine label.
    const char* label;
    const reflection::UILabel* ui_label = desc.ui_label();
    if (ui_label && !ui_label->text.empty()) {
        label = ui_label->text.c_str();
    } else {
        label = desc.name().data();
    }

    // Handle read-only and enable-if disabling.
    bool disabled = desc.is_read_only();
    if (!disabled) {
        const reflection::UIEnableIf* enable_if = desc.ui_enable_if();
        if (enable_if && !enable_if->predicate(instance))
            disabled = true;
    }

    if (disabled)
        ImGui::BeginDisabled();

    // Look up type dispatch.
    EditorFn fn = find_editor_fn(desc);
    if (fn) {
        changed = fn(label, desc, instance, ctx);
    } else {
        // Unsupported type: show a disabled text label.
        ImGui::BeginDisabled();
        ImGui::Text("%s (%s)", label, desc.type().name());
        ImGui::EndDisabled();
    }

    if (disabled)
        ImGui::EndDisabled();

    // Show tooltip with docstring if available.
    std::string_view doc = desc.doc();
    if (!doc.empty() && ImGui::IsItemHovered(ImGuiHoveredFlags_DelayNormal))
        ImGui::SetTooltip("%.*s", static_cast<int>(doc.size()), doc.data());

    ImGui::PopID();
    return changed;
}

// ----------------------------------------------------------------------------
// properties_editor
// ----------------------------------------------------------------------------

bool properties_editor(ReflectedObject& object, PropertyEditorContext& ctx)
{
    void* instance = &object;
    const reflection::ClassDescriptor& class_desc = object.class_descriptor();
    const reflection::DynamicPropertySet* dyn = object.dynamic_properties();

    // Look up or build cached layout.
    PropertyLayout* cached = ctx.layout_cache.find(instance);
    bool need_rebuild = !cached;

    if (cached) {
        // Verify class descriptor matches.
        if (cached->class_desc != &class_desc)
            need_rebuild = true;
        // Verify dynamic set pointer matches.
        else if (cached->dynamic_set != dyn)
            need_rebuild = true;
        // Verify dynamic generation is current.
        else if (dyn && cached->dynamic_generation != dyn->generation())
            need_rebuild = true;
    }

    if (need_rebuild) {
        PropertyLayout layout = build_property_layout(class_desc, dyn);
        cached = &ctx.layout_cache.insert(instance, std::move(layout));
    }

    return visit_layout_node(cached->root, instance, ctx);
}

bool properties_editor(
    std::span<const reflection::PropertyDescriptor* const> descriptors,
    void* instance,
    const void* cache_key,
    uint64_t generation,
    PropertyEditorContext& ctx
)
{
    // Look up or build cached layout.
    PropertyLayout* cached = ctx.layout_cache.find(cache_key);
    bool need_rebuild = !cached || cached->dynamic_generation != generation;

    if (need_rebuild) {
        PropertyLayout layout = build_property_layout(descriptors);
        layout.dynamic_generation = generation;
        cached = &ctx.layout_cache.insert(cache_key, std::move(layout));
    }

    return visit_layout_node(cached->root, instance, ctx);
}

bool properties_editor(Properties& properties)
{
    bool changed = false;
    std::vector<std::string_view> keys = properties.keys();
    for (const auto& key : keys) {
        switch (properties.type(key)) {
        case PropertyType::bool_: {
            bool value = properties.get<bool>(key);
            if (ImGui::Checkbox(key.data(), &value)) {
                properties.set(key, value);
                changed = true;
            }
            break;
        }
        case PropertyType::int_: {
            int64_t value = properties.get<int64_t>(key);
            if (ImGui::DragScalarN(key.data(), ImGuiDataType_S64, &value, 1)) {
                properties.set(key, value);
                changed = true;
            }
            break;
        }
        case PropertyType::float_: {
            double value = properties.get<double>(key);
            if (ImGui::DragScalarN(key.data(), ImGuiDataType_Double, &value, 1, 0.1f)) {
                properties.set(key, value);
                changed = true;
            }
            break;
        }
        case PropertyType::int2: {
            int2 value = properties.get<int2>(key);
            if (ImGui::DragInt2(key.data(), &value.x)) {
                properties.set(key, value);
                changed = true;
            }
            break;
        }
        case PropertyType::int3: {
            int3 value = properties.get<int3>(key);
            if (ImGui::DragInt3(key.data(), &value.x)) {
                properties.set(key, value);
                changed = true;
            }
            break;
        }
        case PropertyType::int4: {
            int4 value = properties.get<int4>(key);
            if (ImGui::DragInt4(key.data(), &value.x)) {
                properties.set(key, value);
                changed = true;
            }
            break;
        }
        case PropertyType::uint2: {
            uint2 value = properties.get<uint2>(key);
            if (ImGui::DragScalarN(key.data(), ImGuiDataType_U32, &value.x, 2)) {
                properties.set(key, value);
                changed = true;
            }
            break;
        }
        case PropertyType::uint3: {
            uint3 value = properties.get<uint3>(key);
            if (ImGui::DragScalarN(key.data(), ImGuiDataType_U32, &value.x, 3)) {
                properties.set(key, value);
                changed = true;
            }
            break;
        }
        case PropertyType::uint4: {
            uint4 value = properties.get<uint4>(key);
            if (ImGui::DragScalarN(key.data(), ImGuiDataType_U32, &value.x, 4)) {
                properties.set(key, value);
                changed = true;
            }
            break;
        }
        case PropertyType::float2: {
            float2 value = properties.get<float2>(key);
            if (ImGui::DragFloat2(key.data(), &value.x, 0.1f)) {
                properties.set(key, value);
                changed = true;
            }
            break;
        }
        case PropertyType::float3: {
            float3 value = properties.get<float3>(key);
            if (ImGui::DragFloat3(key.data(), &value.x, 0.1f)) {
                properties.set(key, value);
                changed = true;
            }
            break;
        }
        case PropertyType::float4: {
            float4 value = properties.get<float4>(key);
            if (ImGui::DragFloat4(key.data(), &value.x, 0.1f)) {
                properties.set(key, value);
                changed = true;
            }
            break;
        }
        case PropertyType::float3x3: {
            float3x3 value = properties.get<float3x3>(key);
            ImGui::Text("%s", key.data());
            ImGui::Indent();
            for (int r = 0; r < 3; ++r) {
                char row_label[16];
                snprintf(row_label, sizeof(row_label), "Row %d", r);
                if (ImGui::DragFloat3(row_label, &value[r].x, 0.01f)) {
                    properties.set(key, value);
                    changed = true;
                }
            }
            ImGui::Unindent();
            break;
        }
        case PropertyType::float4x4: {
            float4x4 value = properties.get<float4x4>(key);
            ImGui::Text("%s", key.data());
            ImGui::Indent();
            for (int r = 0; r < 4; ++r) {
                char row_label[16];
                snprintf(row_label, sizeof(row_label), "Row %d", r);
                if (ImGui::DragFloat4(row_label, &value[r].x, 0.01f)) {
                    properties.set(key, value);
                    changed = true;
                }
            }
            ImGui::Unindent();
            break;
        }
        case PropertyType::string: {
            std::string value = properties.get<std::string>(key);
            auto resize_callback = [](ImGuiInputTextCallbackData* data) -> int
            {
                if (data->EventFlag == ImGuiInputTextFlags_CallbackResize) {
                    auto* str = static_cast<std::string*>(data->UserData);
                    str->resize(data->BufTextLen);
                    data->Buf = str->data();
                }
                return 0;
            };
            if (ImGui::InputText(
                    key.data(),
                    value.data(),
                    value.capacity() + 1,
                    ImGuiInputTextFlags_CallbackResize,
                    resize_callback,
                    &value
                )) {
                properties.set(key, value);
                changed = true;
            }
            break;
        }
        case PropertyType::enum_: {
            detail::PropertyEnumValue ev = properties.get<detail::PropertyEnumValue>(key);
            ImGui::Text("%s: %lld", key.data(), static_cast<long long>(ev.value));
            break;
        }
        default:
            ImGui::Text("%s:", key.data());
            break;
        }
    }
    return changed;
}

} // namespace falcor::ui
