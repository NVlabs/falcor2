// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/core/macros.h"
#include "falcor2/core/enum.h"
#include "falcor2/core/types.h"

namespace falcor {

/// Represents a 3D transformation with separate translation, rotation, and scale components.
/// The transformation matrix is computed on-demand and cached until any component changes.
/// By default, the composition order is Scale-Rotate-Translate (SRT).
class FALCOR_API Transform {
public:
    /// Order in which transformation components are composed into the final matrix.
    /// For example, 'srt' means: first scale, then rotate, then translate.
    enum class CompositionOrder {
        srt, ///< scale -> rotate -> translate
        str, ///< scale -> translate -> rotate
        rst, ///< rotate -> scale -> translate
        rts, ///< rotate -> translate -> scale
        trs, ///< translate -> rotate -> scale
        tsr, ///< translate -> scale -> rotate
    };
    SGL_ENUM_INFO(
        CompositionOrder,
        {
            {CompositionOrder::srt, "srt"},
            {CompositionOrder::str, "str"},
            {CompositionOrder::rst, "rst"},
            {CompositionOrder::rts, "rts"},
            {CompositionOrder::trs, "trs"},
            {CompositionOrder::tsr, "tsr"},
        }
    );

    /// Flags indicating which transformation components are identity (no effect).
    enum class IdentityFlags : uint8_t {
        none = 0,
        translation = (1u << 0),
        rotation = (1u << 1),
        scale = (1u << 2),
        all = translation | rotation | scale
    };

    /// Default constructor. Creates an identity transform.
    Transform() = default;

    /// Copy constructor.
    Transform(const Transform&) = default;

    /// Constructor with explicit translation, rotation, and scale components.
    Transform(const float3& translation, const quatf& rotation, const float3& scale);

    /// Constructor from a 4x4 transformation matrix (decomposes into TRS components).
    /// Note: Skew and perspective components are not supported, the result will be undefined.
    Transform(const float4x4& matrix);

    /// Copy assignment operator.
    Transform& operator=(const Transform&) = default;

    /// Reset the transform to identity.
    void reset();

    /// Translation component.
    float3 translation() const { return m_translation; }

    /// Set the translation component.
    void set_translation(const float3& translation);

    /// Rotation component (quaternion).
    quatf rotation() const { return m_rotation; }

    /// Set the rotation component (quaternion).
    void set_rotation(const quatf& rotation);

    /// Rotation as euler angles in radians, XYZ order.
    float3 rotation_xyz() const;

    /// Set the rotation from euler angles in radians, XYZ order.
    void set_rotation_xyz(const float3& rotation_xyz);

    /// Scale component.
    float3 scale() const { return m_scale; }

    /// Set the scale component.
    void set_scale(const float3& scale);

    /// Composition order for building the matrix.
    CompositionOrder composition_order() const { return m_composition_order; }

    /// Set the composition order for building the matrix.
    void set_composition_order(CompositionOrder order);

    /// Transformation matrix (cached and recomputed when dirty).
    const float4x4& matrix() const;

    /// Flags indicating which components are identity.
    IdentityFlags identity_flags() const;

    /// Check if this is an identity transform.
    bool is_identity() const { return identity_flags() == IdentityFlags::all; }

    /// Apply a translation delta (accumulates with existing translation).
    Transform& translate(const float3& delta);

    /// Apply a rotation delta (accumulates with existing rotation).
    Transform& rotate(const quatf& delta);

    /// Apply a scale factor (multiplies with existing scale).
    Transform& scale_by(const float3& factor);

    /// Get a string representation of the transform.
    std::string to_string() const;

private:
    void mark_dirty() const { m_dirty = true; }
    void update_matrix() const;

    float3 m_translation{0.f};
    quatf m_rotation{quatf::identity()};
    float3 m_scale{1.f};
    CompositionOrder m_composition_order{CompositionOrder::srt};

    mutable bool m_dirty{false};
    mutable float4x4 m_matrix{float4x4::identity()};
    mutable IdentityFlags m_identity_flags{IdentityFlags::all};
};

SGL_ENUM_REGISTER(Transform::CompositionOrder);

SGL_ENUM_CLASS_OPERATORS(Transform::IdentityFlags);

} // namespace falcor
