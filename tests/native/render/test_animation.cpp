// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "testing.h"

#include "falcor2/render/animation.h"
#include "falcor2/render/component/transform_animation.h"
#include "falcor2/render/scene.h"
#include "falcor2/render/scene_update.h"
#include "falcor2/ui/scene_editor.h"

#include <sgl/math/vector.h>
#include <sgl/math/quaternion_math.h>

#include <cmath>

using namespace falcor;

TEST_SUITE_BEGIN("Animation");

// ---------------------------------------------------------------------------
// AnimationChannel basic tests
// ---------------------------------------------------------------------------

TEST_CASE("AnimationChannel - components_per_value")
{
    AnimationChannel ch;

    ch.set_value_type(AnimationValueType::float_);
    CHECK_EQ(ch.components_per_value(), 1);

    ch.set_value_type(AnimationValueType::float3);
    CHECK_EQ(ch.components_per_value(), 3);

    ch.set_value_type(AnimationValueType::quatf);
    CHECK_EQ(ch.components_per_value(), 4);
}

TEST_CASE("AnimationChannel - keyframe_count")
{
    AnimationChannel ch;
    CHECK_EQ(ch.keyframe_count(), 0);

    ch.set_times({0.f, 1.f, 2.f});
    CHECK_EQ(ch.keyframe_count(), 3);
}

// ---------------------------------------------------------------------------
// STEP interpolation
// ---------------------------------------------------------------------------

TEST_CASE("AnimationChannel - step float3")
{
    AnimationChannel ch;
    ch.set_value_type(AnimationValueType::float3);
    ch.set_interpolation_mode(AnimationInterpolationMode::step);
    ch.set_times({0.f, 1.f, 2.f});
    ch.set_values({
        0.f,
        0.f,
        0.f,
        1.f,
        2.f,
        3.f,
        4.f,
        5.f,
        6.f,
    });

    float3 out;

    // Before first keyframe
    out = ch.evaluate_float3(-1.f);
    CHECK_EQ(out.x, doctest::Approx(0.f));
    CHECK_EQ(out.y, doctest::Approx(0.f));
    CHECK_EQ(out.z, doctest::Approx(0.f));

    // At first keyframe
    out = ch.evaluate_float3(0.f);
    CHECK_EQ(out.x, doctest::Approx(0.f));

    // Between first and second
    out = ch.evaluate_float3(0.5f);
    CHECK_EQ(out.x, doctest::Approx(0.f));
    CHECK_EQ(out.y, doctest::Approx(0.f));
    CHECK_EQ(out.z, doctest::Approx(0.f));

    // At second keyframe
    out = ch.evaluate_float3(1.f);
    CHECK_EQ(out.x, doctest::Approx(1.f));
    CHECK_EQ(out.y, doctest::Approx(2.f));
    CHECK_EQ(out.z, doctest::Approx(3.f));

    // Between second and third
    out = ch.evaluate_float3(1.5f);
    CHECK_EQ(out.x, doctest::Approx(1.f));
    CHECK_EQ(out.y, doctest::Approx(2.f));
    CHECK_EQ(out.z, doctest::Approx(3.f));

    // After last keyframe
    out = ch.evaluate_float3(5.f);
    CHECK_EQ(out.x, doctest::Approx(4.f));
    CHECK_EQ(out.y, doctest::Approx(5.f));
    CHECK_EQ(out.z, doctest::Approx(6.f));
}

// ---------------------------------------------------------------------------
// LINEAR interpolation
// ---------------------------------------------------------------------------

TEST_CASE("AnimationChannel - linear float3")
{
    AnimationChannel ch;
    ch.set_value_type(AnimationValueType::float3);
    ch.set_interpolation_mode(AnimationInterpolationMode::linear);
    ch.set_times({0.f, 1.f});
    ch.set_values({
        0.f,
        0.f,
        0.f,
        2.f,
        4.f,
        6.f,
    });

    float3 out;

    // Midpoint
    out = ch.evaluate_float3(0.5f);
    CHECK_EQ(out.x, doctest::Approx(1.f));
    CHECK_EQ(out.y, doctest::Approx(2.f));
    CHECK_EQ(out.z, doctest::Approx(3.f));

    // Quarter
    out = ch.evaluate_float3(0.25f);
    CHECK_EQ(out.x, doctest::Approx(0.5f));
    CHECK_EQ(out.y, doctest::Approx(1.f));
    CHECK_EQ(out.z, doctest::Approx(1.5f));
}

TEST_CASE("AnimationChannel - linear float")
{
    AnimationChannel ch;
    ch.set_value_type(AnimationValueType::float_);
    ch.set_interpolation_mode(AnimationInterpolationMode::linear);
    ch.set_times({0.f, 1.f});
    ch.set_values({10.f, 20.f});

    float out = ch.evaluate_float(0.5f);
    CHECK_EQ(out, doctest::Approx(15.f));
}

TEST_CASE("AnimationChannel - linear quatf slerp")
{
    AnimationChannel ch;
    ch.set_value_type(AnimationValueType::quatf);
    ch.set_interpolation_mode(AnimationInterpolationMode::linear);
    ch.set_times({0.f, 1.f});
    // Identity quaternion and 90-degree rotation around Z
    quatf q0 = quatf::identity();
    float angle = 3.14159265f / 2.f; // 90 degrees
    quatf q1(0.f, 0.f, std::sin(angle / 2.f), std::cos(angle / 2.f));
    ch.set_values({q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w});

    quatf out = ch.evaluate_quatf(0.5f);

    // Midpoint should be ~45 degree rotation
    quatf expected = sgl::math::slerp(q0, q1, 0.5f);
    CHECK_EQ(out.x, doctest::Approx(expected.x).epsilon(1e-5f));
    CHECK_EQ(out.y, doctest::Approx(expected.y).epsilon(1e-5f));
    CHECK_EQ(out.z, doctest::Approx(expected.z).epsilon(1e-5f));
    CHECK_EQ(out.w, doctest::Approx(expected.w).epsilon(1e-5f));
}

// ---------------------------------------------------------------------------
// CUBIC_HERMITE interpolation (Catmull-Rom)
// ---------------------------------------------------------------------------

TEST_CASE("AnimationChannel - cubic_hermite float3")
{
    AnimationChannel ch;
    ch.set_value_type(AnimationValueType::float3);
    ch.set_interpolation_mode(AnimationInterpolationMode::cubic_hermite);
    ch.set_times({0.f, 1.f, 2.f, 3.f});
    ch.set_values({
        0.f,
        0.f,
        0.f,
        1.f,
        0.f,
        0.f,
        2.f,
        0.f,
        0.f,
        3.f,
        0.f,
        0.f,
    });

    float3 out;

    // For a perfectly linear curve, cubic hermite should give linear results
    out = ch.evaluate_float3(0.5f);
    CHECK_EQ(out.x, doctest::Approx(0.5f).epsilon(1e-4f));

    out = ch.evaluate_float3(1.5f);
    CHECK_EQ(out.x, doctest::Approx(1.5f).epsilon(1e-4f));

    // At exact keyframes
    out = ch.evaluate_float3(0.f);
    CHECK_EQ(out.x, doctest::Approx(0.f));

    out = ch.evaluate_float3(2.f);
    CHECK_EQ(out.x, doctest::Approx(2.f));
}

// ---------------------------------------------------------------------------
// CUBIC_SPLINE interpolation (glTF explicit tangents)
// ---------------------------------------------------------------------------

TEST_CASE("AnimationChannel - cubic_spline float3")
{
    AnimationChannel ch;
    ch.set_value_type(AnimationValueType::float3);
    ch.set_interpolation_mode(AnimationInterpolationMode::cubic_spline);
    ch.set_times({0.f, 1.f});
    ch.set_values({
        0.f,
        0.f,
        0.f,
        1.f,
        0.f,
        0.f,
    });
    // Zero tangents: should degenerate to a smooth curve that still hits endpoints
    ch.set_in_tangents({
        0.f,
        0.f,
        0.f,
        0.f,
        0.f,
        0.f,
    });
    ch.set_out_tangents({
        0.f,
        0.f,
        0.f,
        0.f,
        0.f,
        0.f,
    });

    float3 out;

    // At endpoints
    out = ch.evaluate_float3(0.f);
    CHECK_EQ(out.x, doctest::Approx(0.f));

    out = ch.evaluate_float3(1.f);
    CHECK_EQ(out.x, doctest::Approx(1.f));

    // With zero tangents, midpoint should be 0.5 (Hermite with zero tangents = lerp-like)
    // Actually h00(0.5) = 0.5, h01(0.5) = 0.5, h10 = h11 = 0 for zero tangents
    out = ch.evaluate_float3(0.5f);
    CHECK_EQ(out.x, doctest::Approx(0.5f).epsilon(1e-4f));
}

// ---------------------------------------------------------------------------
// Edge cases
// ---------------------------------------------------------------------------

TEST_CASE("AnimationChannel - single keyframe")
{
    AnimationChannel ch;
    ch.set_value_type(AnimationValueType::float3);
    ch.set_interpolation_mode(AnimationInterpolationMode::linear);
    ch.set_times({0.f});
    ch.set_values({5.f, 6.f, 7.f});

    float3 out;
    out = ch.evaluate_float3(0.f);
    CHECK_EQ(out.x, doctest::Approx(5.f));
    CHECK_EQ(out.y, doctest::Approx(6.f));
    CHECK_EQ(out.z, doctest::Approx(7.f));

    out = ch.evaluate_float3(100.f);
    CHECK_EQ(out.x, doctest::Approx(5.f));
}

TEST_CASE("AnimationChannel - exact keyframe times")
{
    AnimationChannel ch;
    ch.set_value_type(AnimationValueType::float3);
    ch.set_interpolation_mode(AnimationInterpolationMode::linear);
    ch.set_times({0.f, 1.f, 2.f});
    ch.set_values({
        0.f,
        0.f,
        0.f,
        1.f,
        1.f,
        1.f,
        2.f,
        2.f,
        2.f,
    });

    float3 out;

    out = ch.evaluate_float3(1.f);
    CHECK_EQ(out.x, doctest::Approx(1.f));
    CHECK_EQ(out.y, doctest::Approx(1.f));
    CHECK_EQ(out.z, doctest::Approx(1.f));

    out = ch.evaluate_float3(2.f);
    CHECK_EQ(out.x, doctest::Approx(2.f));
}

// ---------------------------------------------------------------------------
// Animation key range
// ---------------------------------------------------------------------------

TEST_CASE("Animation - key range")
{
    ref<Animation> anim = ref<Animation>(new Animation());
    CHECK_FALSE(anim->key_range().is_valid());

    auto empty = make_ref<AnimationChannel>();
    anim->set_channels({empty});
    CHECK_FALSE(anim->key_range().is_valid());

    auto ch1 = make_ref<AnimationChannel>();
    ch1->set_times({0.f, 1.f});
    anim->set_channels({ch1});
    CHECK_EQ(anim->key_range().end, doctest::Approx(1.f));

    auto ch2 = make_ref<AnimationChannel>();
    ch2->set_times({0.f, 3.f});
    anim->set_channels({empty, ch1, ch2});
    CHECK_EQ(anim->key_range().end, doctest::Approx(3.f));
    ch1->set_times({-3.f, -1.f});
    ch2->set_times({-2.f, -0.5f});
    CHECK_EQ(anim->key_range().start, -3.0);
    CHECK_EQ(anim->key_range().end, -0.5);
}

TEST_CASE("Animation - channel ownership and timestamp replacement")
{
    auto channel = make_ref<AnimationChannel>();
    channel->set_value_type(AnimationValueType::float_);
    channel->set_times({0.f, 1.f, 2.f, 3.f});
    channel->set_values({0.f, 1.f, 2.f, 3.f});
    CHECK_EQ(channel->evaluate_float(2.5f), doctest::Approx(2.5f));
    channel->set_times({0.f, 4.f});
    channel->set_values({0.f, 8.f});
    CHECK_EQ(channel->evaluate_float(2.5f), doctest::Approx(5.f));

    {
        auto animation = make_ref<Animation>();
        animation->set_channels({channel});
        animation->set_channels(animation->channels());
        CHECK_THROWS_WITH(animation->set_channels({channel, nullptr}), doctest::Contains("must not be null"));
        REQUIRE_EQ(animation->channels().size(), 1);
        CHECK_EQ(animation->channel(0), channel.get());

        auto other = make_ref<Animation>();
        auto standalone = make_ref<AnimationChannel>();
        CHECK_THROWS_WITH(other->set_channels({standalone, channel}), doctest::Contains("already belongs"));
        CHECK(other->channels().empty());
        CHECK_EQ(animation->channel(0), channel.get());
        // A failed replacement must not attach any of the proposed channels.
        CHECK_NOTHROW(animation->set_channels({channel, standalone}));
    }

    // Destruction detaches without needing a Scene.
    channel->set_values({2.f, 6.f});
    auto animation = make_ref<Animation>();
    CHECK_NOTHROW(animation->set_channels({channel}));
}

// ---------------------------------------------------------------------------
// TransformAnimation component
// ---------------------------------------------------------------------------

TEST_CASE_GPU("TransformAnimation - has_channels")
{
    auto scene = Scene::create(ref(ctx.device));
    auto* entity = scene->create_entity();
    auto* ta = entity->create_component<TransformAnimation>();

    CHECK_FALSE(ta->has_channels());

    ta->set_translation_channel(0);
    CHECK(ta->has_channels());

    ta->set_translation_channel(-1);
    CHECK_FALSE(ta->has_channels());

    ta->set_rotation_channel(1);
    CHECK(ta->has_channels());

    ta->set_rotation_channel(-1);
    ta->set_scale_channel(2);
    CHECK(ta->has_channels());
}

TEST_CASE_GPU("TransformAnimation - binding lifecycle")
{
    // Backlinks and ownership are independent of the GPU backend.
    if (ctx.device->type() != testing::get_device_types().front())
        return;
    auto scene = Scene::create(ref(ctx.device));
    ref<Animation> first(scene->create_animation());
    ref<Animation> second(scene->create_animation());
    auto* entity = scene->create_entity();
    ref<TransformAnimation> a(entity->create_component<TransformAnimation>());
    ref<TransformAnimation> b(entity->create_component<TransformAnimation>());

    a->set_animation(first);
    a->set_animation(first);
    REQUIRE_EQ(first->transform_animations().size(), 1);
    CHECK_EQ(first->transform_animations()[0], a.get());
    b->set_animation(first);
    REQUIRE_EQ(first->transform_animations().size(), 2);

    a->set_animation(second);
    REQUIRE_EQ(first->transform_animations().size(), 1);
    CHECK_EQ(first->transform_animations()[0], b.get());
    REQUIRE_EQ(second->transform_animations().size(), 1);
    CHECK_EQ(second->transform_animations()[0], a.get());

    {
        auto foreign_scene = Scene::create(ref(ctx.device));
        auto* foreign = foreign_scene->create_animation();
        CHECK_THROWS(a->set_animation(foreign));
        CHECK(foreign->transform_animations().empty());
        CHECK_EQ(a->animation(), second.get());
    }

    b->set_animation(nullptr);
    CHECK(first->transform_animations().empty());
    b->set_animation(second);
    b->remove();
    b->remove();
    CHECK_THROWS(b->set_animation(first));
    CHECK_EQ(b->animation(), second.get());
    REQUIRE_EQ(second->transform_animations().size(), 1);
    CHECK_EQ(second->transform_animations()[0], a.get());
    scene->update();
    second->remove();
    CHECK_THROWS(a->set_animation(second));
    scene->update();
    CHECK_EQ(a->animation(), nullptr);
    CHECK(second->transform_animations().empty());
    CHECK_EQ(b->animation(), second.get());
    CHECK_EQ(second->ref_count(), 2); // Local reference and removed component.
    b = nullptr;
    CHECK_EQ(second->ref_count(), 1);

    a->set_animation(first);
    scene = nullptr;
    CHECK_EQ(a->animation(), first.get());
    CHECK_EQ(first->ref_count(), 2);
    a = nullptr;
    CHECK(first->transform_animations().empty());
    CHECK_EQ(first->ref_count(), 1);
}

// ---------------------------------------------------------------------------
// Scene animation integration
// ---------------------------------------------------------------------------

TEST_CASE_GPU("Scene - set_time updates entity transforms")
{
    auto scene = Scene::create(ref(ctx.device));

    // Create animation with a translation channel
    auto* animation = scene->create_animation();
    auto trans_ch = make_ref<AnimationChannel>();
    trans_ch->set_value_type(AnimationValueType::float3);
    trans_ch->set_interpolation_mode(AnimationInterpolationMode::linear);
    trans_ch->set_times({0.f, 1.f});
    trans_ch->set_values({0.f, 0.f, 0.f, 10.f, 0.f, 0.f});
    animation->set_channels({trans_ch});

    // Create entity with TransformAnimation component
    auto* entity = scene->create_entity();
    auto* ta = entity->create_component<TransformAnimation>();
    ta->set_animation(animation);
    ta->set_translation_channel(0);

    // Evaluate at t=0
    scene->set_time(0.f);
    scene->update();
    CHECK_EQ(entity->transform().translation().x, doctest::Approx(0.f));
    CHECK_EQ(entity->transform().translation().y, doctest::Approx(0.f));
    CHECK_EQ(entity->transform().translation().z, doctest::Approx(0.f));

    // Evaluate at t=0.5
    scene->set_time(0.5f);
    scene->update();
    CHECK_EQ(entity->transform().translation().x, doctest::Approx(5.f));

    // Evaluate at t=1
    scene->set_time(1.f);
    scene->update();
    CHECK_EQ(entity->transform().translation().x, doctest::Approx(10.f));

    auto override_ch = make_ref<AnimationChannel>();
    override_ch->set_times({0.f});
    override_ch->set_values({20.f, 0.f, 0.f});
    animation->set_channels({trans_ch, override_ch});
    auto* override_ta = entity->create_component<TransformAnimation>();
    override_ta->set_animation(animation);
    override_ta->set_translation_channel(1);
    scene->update();
    CHECK_EQ(entity->transform().translation().x, 20.f);
    override_ta->remove();
    scene->update();
    CHECK_EQ(entity->transform().translation().x, 10.f);

    animation->remove();
    scene->update();
    CHECK_EQ(ta->animation(), nullptr);
    CHECK(scene->animation_collection().empty());
}

TEST_CASE_GPU("Scene - set_time with rotation")
{
    auto scene = Scene::create(ref(ctx.device));

    auto* animation = scene->create_animation();
    auto rot_ch = make_ref<AnimationChannel>();
    rot_ch->set_value_type(AnimationValueType::quatf);
    rot_ch->set_interpolation_mode(AnimationInterpolationMode::linear);
    rot_ch->set_times({0.f, 1.f});
    quatf q0 = quatf::identity();
    float angle = 3.14159265f / 2.f;
    quatf q1(0.f, 0.f, std::sin(angle / 2.f), std::cos(angle / 2.f));
    rot_ch->set_values({q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w});
    animation->set_channels({rot_ch});

    auto* entity = scene->create_entity();
    auto* ta = entity->create_component<TransformAnimation>();
    ta->set_animation(animation);
    ta->set_rotation_channel(0);

    scene->set_time(0.f);
    scene->update();
    CHECK_EQ(entity->transform().rotation().x, doctest::Approx(q0.x).epsilon(1e-5f));
    CHECK_EQ(entity->transform().rotation().w, doctest::Approx(q0.w).epsilon(1e-5f));

    scene->set_time(1.f);
    scene->update();
    CHECK_EQ(entity->transform().rotation().z, doctest::Approx(q1.z).epsilon(1e-5f));
    CHECK_EQ(entity->transform().rotation().w, doctest::Approx(q1.w).epsilon(1e-5f));
}

TEST_CASE_GPU("Scene - set_time with scale")
{
    auto scene = Scene::create(ref(ctx.device));

    auto* animation = scene->create_animation();
    auto scale_ch = make_ref<AnimationChannel>();
    scale_ch->set_value_type(AnimationValueType::float3);
    scale_ch->set_interpolation_mode(AnimationInterpolationMode::linear);
    scale_ch->set_times({0.f, 1.f});
    scale_ch->set_values({1.f, 1.f, 1.f, 2.f, 3.f, 4.f});
    animation->set_channels({scale_ch});

    auto* entity = scene->create_entity();
    auto* ta = entity->create_component<TransformAnimation>();
    ta->set_animation(animation);
    ta->set_scale_channel(0);

    scene->set_time(0.5f);
    scene->update();
    CHECK_EQ(entity->transform().scale().x, doctest::Approx(1.5f));
    CHECK_EQ(entity->transform().scale().y, doctest::Approx(2.f));
    CHECK_EQ(entity->transform().scale().z, doctest::Approx(2.5f));

    animation->channels()[0]->set_times({});
    animation->channels()[0]->set_values({});
    scene->set_time(1.f);
    scene->update();
    CHECK_EQ(entity->transform().scale(), float3(1.5f, 2.f, 2.5f));
}

TEST_CASE_GPU("Scene - animation_range and has_animation")
{
    auto scene = Scene::create(ref(ctx.device));

    CHECK_FALSE(scene->has_animation());
    CHECK_FALSE(scene->animation_range().is_valid());

    auto* animation = scene->create_animation();
    auto ch = make_ref<AnimationChannel>();
    ch->set_times({0.f, 2.5f});
    ch->set_values({0.f, 0.f, 0.f, 1.f, 1.f, 1.f});
    animation->set_channels({ch});

    CHECK_FALSE(scene->has_animation());
    CHECK_EQ(scene->animation_range().end, doctest::Approx(2.5f));
    auto* entity = scene->create_entity();
    auto* binding = entity->create_component<TransformAnimation>();
    binding->set_animation(animation);
    CHECK_FALSE(scene->has_animation());
    binding->set_translation_channel(0);
    CHECK(scene->has_animation());
    binding->set_translation_channel(1);
    CHECK_FALSE(scene->has_animation());
    binding->set_translation_channel(-1);
    binding->set_scale_channel(0);
    CHECK(scene->has_animation());
    binding->set_scale_channel(-1);
    binding->set_rotation_channel(0);
    CHECK(scene->has_animation());
    binding->set_rotation_channel(-1);
    binding->set_translation_channel(0);
    ch->set_times({});
    CHECK_FALSE(scene->has_animation());
    ch->set_times({0.f, 2.5f});
    CHECK(scene->has_animation());
    binding->set_animation(scene->create_animation());
    CHECK_FALSE(scene->has_animation());
    binding->set_animation(animation);
    CHECK(scene->has_animation());
    binding->set_animation(nullptr);
    CHECK_FALSE(scene->has_animation());
    binding->set_animation(animation);
    entity->remove();
    CHECK_FALSE(scene->has_animation());
    CHECK(animation->transform_animations().empty());
    scene->update();
    CHECK_FALSE(scene->has_animation());

    // Empty animations must not pull positive or negative ranges toward zero.
    auto* empty = scene->create_animation();
    empty->set_channels({make_ref<AnimationChannel>()});
    for (float start : {-10.f, 10.f}) {
        ch->set_times({start, start + 2.f});
        CHECK_EQ(scene->animation_range().start, start);
        CHECK_EQ(scene->animation_range().end, start + 2.f);
    }
    ch->set_times({0.f});
    CHECK(scene->animation_range().is_valid());
    CHECK_EQ(scene->animation_range().start, 0.f);
    CHECK_EQ(scene->animation_range().end, 0.f);
    ch->set_times({});
    CHECK_FALSE(scene->animation_range().is_valid());
}

TEST_CASE_GPU("Scene - multiple animations")
{
    auto scene = Scene::create(ref(ctx.device));

    // First animation with translation
    auto* anim1 = scene->create_animation();
    anim1->set_name("anim1");
    auto ch1 = make_ref<AnimationChannel>();
    ch1->set_value_type(AnimationValueType::float3);
    ch1->set_interpolation_mode(AnimationInterpolationMode::linear);
    ch1->set_times({0.f, 1.f});
    ch1->set_values({0.f, 0.f, 0.f, 10.f, 0.f, 0.f});
    anim1->set_channels({ch1});

    // Second animation with longer duration
    auto* anim2 = scene->create_animation();
    anim2->set_name("anim2");
    auto ch2 = make_ref<AnimationChannel>();
    ch2->set_value_type(AnimationValueType::float3);
    ch2->set_interpolation_mode(AnimationInterpolationMode::linear);
    ch2->set_times({0.f, 3.f});
    ch2->set_values({0.f, 0.f, 0.f, 0.f, 30.f, 0.f});
    anim2->set_channels({ch2});

    // Entity 1 uses animation 1
    auto* entity1 = scene->create_entity();
    auto* ta1 = entity1->create_component<TransformAnimation>();
    ta1->set_animation(anim1);
    ta1->set_translation_channel(0);

    // Entity 2 uses animation 2
    auto* entity2 = scene->create_entity();
    auto* ta2 = entity2->create_component<TransformAnimation>();
    ta2->set_animation(anim2);
    ta2->set_translation_channel(0);

    // Duration is max across all animations
    CHECK_EQ(scene->animation_range().end, doctest::Approx(3.f));

    // Evaluate at t=0.5
    scene->set_time(0.5f);
    scene->update();
    CHECK_EQ(entity1->transform().translation().x, doctest::Approx(5.f));
    CHECK_EQ(entity2->transform().translation().y, doctest::Approx(5.f));
}

TEST_CASE_GPU("Scene - non-animated entities unchanged by set_time")
{
    auto scene = Scene::create(ref(ctx.device));

    auto* animation = scene->create_animation();
    auto ch = make_ref<AnimationChannel>();
    ch->set_value_type(AnimationValueType::float3);
    ch->set_interpolation_mode(AnimationInterpolationMode::linear);
    ch->set_times({0.f, 1.f});
    ch->set_values({0.f, 0.f, 0.f, 10.f, 0.f, 0.f});
    animation->set_channels({ch});

    // Create an animated entity
    auto* animated = scene->create_entity();
    auto* ta = animated->create_component<TransformAnimation>();
    ta->set_animation(animation);
    ta->set_translation_channel(0);

    // Create a non-animated entity with a custom transform
    auto* static_entity = scene->create_entity();
    static_entity->set_transform(Transform(float3(100.f, 200.f, 300.f), quatf::identity(), float3(1.f)));

    scene->set_time(0.5f);
    scene->update();

    // Non-animated entity should be unchanged
    CHECK_EQ(static_entity->transform().translation().x, doctest::Approx(100.f));
    CHECK_EQ(static_entity->transform().translation().y, doctest::Approx(200.f));
    CHECK_EQ(static_entity->transform().translation().z, doctest::Approx(300.f));
}

// ---------------------------------------------------------------------------
// Scene update flags for animation
// ---------------------------------------------------------------------------

TEST_CASE_GPU("Scene - set_time sets animation update flag")
{
    auto scene = Scene::create(ref(ctx.device));

    auto* animation = scene->create_animation();
    auto ch = make_ref<AnimationChannel>();
    ch->set_value_type(AnimationValueType::float3);
    ch->set_interpolation_mode(AnimationInterpolationMode::linear);
    ch->set_times({0.f, 1.f});
    ch->set_values({0.f, 0.f, 0.f, 10.f, 0.f, 0.f});
    animation->set_channels({ch});

    auto* entity = scene->create_entity();
    auto* ta = entity->create_component<TransformAnimation>();
    ta->set_animation(animation);
    ta->set_translation_channel(0);

    // Initial update to process entity creation.
    scene->update();

    // Now animate and update.
    scene->set_time(0.5f);
    SceneUpdateFlags flags = scene->update();

    CHECK(is_set(flags, SceneUpdateFlags::animation));
    CHECK(is_set(flags, SceneUpdateFlags::transforms));
}

TEST_CASE_GPU("Scene - update without animation has no animation flag")
{
    auto scene = Scene::create(ref(ctx.device));
    auto* entity = scene->create_entity();

    // Initial update.
    scene->update();

    // Update again without animation.
    SceneUpdateFlags flags = scene->update();
    CHECK_FALSE(is_set(flags, SceneUpdateFlags::animation));
}

TEST_CASE_GPU("Scene - transform-only fast path with animation")
{
    auto scene = Scene::create(ref(ctx.device));

    auto* animation = scene->create_animation();
    auto ch = make_ref<AnimationChannel>();
    ch->set_value_type(AnimationValueType::float3);
    ch->set_interpolation_mode(AnimationInterpolationMode::linear);
    ch->set_times({0.f, 1.f, 2.f});
    ch->set_values({
        0.f,
        0.f,
        0.f,
        5.f,
        0.f,
        0.f,
        10.f,
        0.f,
        0.f,
    });
    animation->set_channels({ch});

    auto* entity = scene->create_entity();
    auto* ta = entity->create_component<TransformAnimation>();
    ta->set_animation(animation);
    ta->set_translation_channel(0);

    // Initial update to process entity creation.
    scene->update();

    // Animate at different times and verify transforms propagate correctly.
    scene->set_time(0.f);
    scene->update();
    CHECK_EQ(entity->transform().translation().x, doctest::Approx(0.f));

    scene->set_time(0.5f);
    scene->update();
    CHECK_EQ(entity->transform().translation().x, doctest::Approx(2.5f));

    scene->set_time(1.f);
    scene->update();
    CHECK_EQ(entity->transform().translation().x, doctest::Approx(5.f));

    scene->set_time(2.f);
    scene->update();
    CHECK_EQ(entity->transform().translation().x, doctest::Approx(10.f));
}

TEST_CASE_GPU("Animation - editor playback with nonzero starts")
{
    if (ctx.device->type() != testing::get_device_types().front())
        return;
    auto scene = Scene::create(ref(ctx.device));
    auto* animation = scene->create_animation();
    auto ch = make_ref<AnimationChannel>();
    ch->set_values({0.f, 0.f, 0.f, 1.f, 1.f, 1.f});
    animation->set_channels({ch});
    auto* binding = scene->create_entity()->create_component<TransformAnimation>();
    binding->set_animation(animation);
    binding->set_translation_channel(0);
    ui::SceneEditor editor;
    editor.set_scene(scene);
    scene->set_time(1.0);
    editor.set_playing(true);
    editor.update_playback(0.5);
    CHECK_EQ(scene->time(), 1.0);
    for (float start : {-10.f, 10.f}) {
        ch->set_times({start, start + 2.f});
        CHECK_EQ(scene->animation_range().start, start);
        CHECK_EQ(scene->animation_range().end, start + 2.f);
        scene->set_time(start + 1.75);
        editor.set_playing(true);
        editor.set_looping(true);
        editor.update_playback(0.5);
        CHECK_EQ(scene->time(), start + 0.25);
        editor.set_looping(false);
        editor.update_playback(2.0);
        CHECK_EQ(scene->time(), start + 2.0);
        CHECK_FALSE(editor.playing());
    }
}

TEST_SUITE_END();
