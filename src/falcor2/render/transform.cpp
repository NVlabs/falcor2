// SPDX-License-Identifier: Apache-2.0

#include "transform.h"

namespace falcor {

Transform::Transform(const float3& translation, const quatf& rotation, const float3& scale)
{
    m_translation = translation;
    m_rotation = rotation;
    m_scale = scale;
    mark_dirty();
}

Transform::Transform(const float4x4& matrix)
{
    float3 skew = {};
    float4 perspective = {};
    if (!sgl::math::decompose(matrix, m_scale, m_rotation, m_translation, skew, perspective)) {
        reset();
    }
    mark_dirty();
}

void Transform::reset()
{
    m_translation = float3(0.f);
    m_rotation = quatf::identity();
    m_scale = float3(1.f);
    mark_dirty();
}

void Transform::set_composition_order(CompositionOrder order)
{
    m_composition_order = order;
    mark_dirty();
}

void Transform::set_translation(const float3& translation)
{
    m_translation = translation;
    mark_dirty();
}

void Transform::set_rotation(const quatf& rotation)
{
    m_rotation = rotation;
    mark_dirty();
}

float3 Transform::rotation_xyz() const
{
    return sgl::math::euler_angles(m_rotation);
}

void Transform::set_rotation_xyz(const float3& rotation_xyz)
{
    m_rotation = sgl::math::quat_from_euler_angles(rotation_xyz);
    mark_dirty();
}

void Transform::set_scale(const float3& scale)
{
    m_scale = scale;
    mark_dirty();
}

const float4x4& Transform::matrix() const
{
    if (m_dirty)
        update_matrix();
    return m_matrix;
}

Transform::IdentityFlags Transform::identity_flags() const
{
    if (m_dirty)
        update_matrix();
    return m_identity_flags;
}

Transform& Transform::translate(const float3& delta)
{
    m_translation += delta;
    mark_dirty();
    return *this;
}

Transform& Transform::rotate(const quatf& delta)
{
    m_rotation = sgl::math::normalize(sgl::math::mul(delta, m_rotation));
    mark_dirty();
    return *this;
}

Transform& Transform::scale_by(const float3& factor)
{
    m_scale *= factor;
    mark_dirty();
    return *this;
}

std::string Transform::to_string() const
{
    return fmt::format(
        "Transform(\n"
        "  translation = {},\n"
        "  rotation = {},\n"
        "  scale = {}\n"
        ")",
        m_translation,
        m_rotation,
        m_scale
    );
}

void Transform::update_matrix() const
{
    m_identity_flags = IdentityFlags::none;
    if (m_translation == float3(0.f))
        m_identity_flags |= IdentityFlags::translation;
    if (m_rotation == quatf::identity())
        m_identity_flags |= IdentityFlags::rotation;
    if (m_scale == float3(1.f))
        m_identity_flags |= IdentityFlags::scale;

    if (m_identity_flags == IdentityFlags::all) {
        m_matrix = float4x4::identity();
    } else {
        float4x4 t = sgl::math::matrix_from_translation(m_translation);
        float4x4 r = sgl::math::matrix_from_quat(m_rotation);
        float4x4 s = sgl::math::matrix_from_scaling(m_scale);

        switch (m_composition_order) {
        case CompositionOrder::srt:
            m_matrix = mul(t, mul(r, s));
            break;
        case CompositionOrder::str:
            m_matrix = mul(r, mul(t, s));
            break;
        case CompositionOrder::rst:
            m_matrix = mul(t, mul(s, r));
            break;
        case CompositionOrder::rts:
            m_matrix = mul(s, mul(t, r));
            break;
        case CompositionOrder::trs:
            m_matrix = mul(s, mul(r, t));
            break;
        case CompositionOrder::tsr:
            m_matrix = mul(r, mul(s, t));
            break;
        }
    }

    m_dirty = false;
}

} // namespace falcor
