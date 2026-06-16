// SPDX-License-Identifier: Apache-2.0

#include "testing.h"

#include "falcor2/render/animation.h"
#include "falcor2/render/component/transform_animation.h"
#include "falcor2/render/scene.h"
#include "falcor2/render/scene_update.h"

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

    ch.value_type = AnimationValueType::float_;
    CHECK_EQ(ch.components_per_value(), 1);

    ch.value_type = AnimationValueType::float3;
    CHECK_EQ(ch.components_per_value(), 3);

    ch.value_type = AnimationValueType::quatf;
    CHECK_EQ(ch.components_per_value(), 4);
}

TEST_CASE("AnimationChannel - keyframe_count")
{
    AnimationChannel ch;
    CHECK_EQ(ch.keyframe_count(), 0);

    ch.times = {0.f, 1.f, 2.f};
    CHECK_EQ(ch.keyframe_count(), 3);
}

// ---------------------------------------------------------------------------
// STEP interpolation
// ---------------------------------------------------------------------------

TEST_CASE("AnimationChannel - step float3")
{
    AnimationChannel ch;
    ch.value_type = AnimationValueType::float3;
    ch.interpolation_mode = AnimationInterpolationMode::step;
    ch.times = {0.f, 1.f, 2.f};
    ch.values = {
        0.f,
        0.f,
        0.f,
        1.f,
        2.f,
        3.f,
        4.f,
        5.f,
        6.f,
    };

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
    ch.value_type = AnimationValueType::float3;
    ch.interpolation_mode = AnimationInterpolationMode::linear;
    ch.times = {0.f, 1.f};
    ch.values = {
        0.f,
        0.f,
        0.f,
        2.f,
        4.f,
        6.f,
    };

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
    ch.value_type = AnimationValueType::float_;
    ch.interpolation_mode = AnimationInterpolationMode::linear;
    ch.times = {0.f, 1.f};
    ch.values = {10.f, 20.f};

    float out = ch.evaluate_float(0.5f);
    CHECK_EQ(out, doctest::Approx(15.f));
}

TEST_CASE("AnimationChannel - linear quatf slerp")
{
    AnimationChannel ch;
    ch.value_type = AnimationValueType::quatf;
    ch.interpolation_mode = AnimationInterpolationMode::linear;
    ch.times = {0.f, 1.f};
    // Identity quaternion and 90-degree rotation around Z
    quatf q0 = quatf::identity();
    float angle = 3.14159265f / 2.f; // 90 degrees
    quatf q1(0.f, 0.f, std::sin(angle / 2.f), std::cos(angle / 2.f));
    ch.values = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};

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
    ch.value_type = AnimationValueType::float3;
    ch.interpolation_mode = AnimationInterpolationMode::cubic_hermite;
    ch.times = {0.f, 1.f, 2.f, 3.f};
    ch.values = {
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
    };

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
    ch.value_type = AnimationValueType::float3;
    ch.interpolation_mode = AnimationInterpolationMode::cubic_spline;
    ch.times = {0.f, 1.f};
    ch.values = {
        0.f,
        0.f,
        0.f,
        1.f,
        0.f,
        0.f,
    };
    // Zero tangents: should degenerate to a smooth curve that still hits endpoints
    ch.in_tangents = {
        0.f,
        0.f,
        0.f,
        0.f,
        0.f,
        0.f,
    };
    ch.out_tangents = {
        0.f,
        0.f,
        0.f,
        0.f,
        0.f,
        0.f,
    };

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
    ch.value_type = AnimationValueType::float3;
    ch.interpolation_mode = AnimationInterpolationMode::linear;
    ch.times = {0.f};
    ch.values = {5.f, 6.f, 7.f};

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
    ch.value_type = AnimationValueType::float3;
    ch.interpolation_mode = AnimationInterpolationMode::linear;
    ch.times = {0.f, 1.f, 2.f};
    ch.values = {
        0.f,
        0.f,
        0.f,
        1.f,
        1.f,
        1.f,
        2.f,
        2.f,
        2.f,
    };

    float3 out;

    out = ch.evaluate_float3(1.f);
    CHECK_EQ(out.x, doctest::Approx(1.f));
    CHECK_EQ(out.y, doctest::Approx(1.f));
    CHECK_EQ(out.z, doctest::Approx(1.f));

    out = ch.evaluate_float3(2.f);
    CHECK_EQ(out.x, doctest::Approx(2.f));
}

// ---------------------------------------------------------------------------
// Animation duration
// ---------------------------------------------------------------------------

TEST_CASE("Animation - duration")
{
    ref<Animation> anim = ref<Animation>(new Animation());
    CHECK_EQ(anim->duration(), 0.f);

    AnimationChannel ch1;
    ch1.times = {0.f, 1.f};
    anim->channels.push_back(ch1);
    CHECK_EQ(anim->duration(), doctest::Approx(1.f));

    AnimationChannel ch2;
    ch2.times = {0.f, 3.f};
    anim->channels.push_back(ch2);
    CHECK_EQ(anim->duration(), doctest::Approx(3.f));
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

// ---------------------------------------------------------------------------
// Scene animation integration
// ---------------------------------------------------------------------------

TEST_CASE_GPU("Scene - set_time updates entity transforms")
{
    auto scene = Scene::create(ref(ctx.device));

    // Create animation with a translation channel
    auto* animation = scene->create_animation();
    AnimationChannel trans_ch;
    trans_ch.value_type = AnimationValueType::float3;
    trans_ch.interpolation_mode = AnimationInterpolationMode::linear;
    trans_ch.times = {0.f, 1.f};
    trans_ch.values = {0.f, 0.f, 0.f, 10.f, 0.f, 0.f};
    animation->channels.push_back(std::move(trans_ch));

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
}

TEST_CASE_GPU("Scene - set_time with rotation")
{
    auto scene = Scene::create(ref(ctx.device));

    auto* animation = scene->create_animation();
    AnimationChannel rot_ch;
    rot_ch.value_type = AnimationValueType::quatf;
    rot_ch.interpolation_mode = AnimationInterpolationMode::linear;
    rot_ch.times = {0.f, 1.f};
    quatf q0 = quatf::identity();
    float angle = 3.14159265f / 2.f;
    quatf q1(0.f, 0.f, std::sin(angle / 2.f), std::cos(angle / 2.f));
    rot_ch.values = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
    animation->channels.push_back(std::move(rot_ch));

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
    AnimationChannel scale_ch;
    scale_ch.value_type = AnimationValueType::float3;
    scale_ch.interpolation_mode = AnimationInterpolationMode::linear;
    scale_ch.times = {0.f, 1.f};
    scale_ch.values = {1.f, 1.f, 1.f, 2.f, 3.f, 4.f};
    animation->channels.push_back(std::move(scale_ch));

    auto* entity = scene->create_entity();
    auto* ta = entity->create_component<TransformAnimation>();
    ta->set_animation(animation);
    ta->set_scale_channel(0);

    scene->set_time(0.5f);
    scene->update();
    CHECK_EQ(entity->transform().scale().x, doctest::Approx(1.5f));
    CHECK_EQ(entity->transform().scale().y, doctest::Approx(2.f));
    CHECK_EQ(entity->transform().scale().z, doctest::Approx(2.5f));
}

TEST_CASE_GPU("Scene - animation_duration and has_animation")
{
    auto scene = Scene::create(ref(ctx.device));

    CHECK_FALSE(scene->has_animation());
    CHECK_EQ(scene->animation_duration(), 0.f);

    auto* animation = scene->create_animation();
    AnimationChannel ch;
    ch.times = {0.f, 2.5f};
    ch.values = {0.f, 0.f, 0.f, 1.f, 1.f, 1.f};
    animation->channels.push_back(std::move(ch));

    CHECK(scene->has_animation());
    CHECK_EQ(scene->animation_duration(), doctest::Approx(2.5f));
}

TEST_CASE_GPU("Scene - multiple animations")
{
    auto scene = Scene::create(ref(ctx.device));

    // First animation with translation
    auto* anim1 = scene->create_animation();
    anim1->set_name("anim1");
    AnimationChannel ch1;
    ch1.value_type = AnimationValueType::float3;
    ch1.interpolation_mode = AnimationInterpolationMode::linear;
    ch1.times = {0.f, 1.f};
    ch1.values = {0.f, 0.f, 0.f, 10.f, 0.f, 0.f};
    anim1->channels.push_back(std::move(ch1));

    // Second animation with longer duration
    auto* anim2 = scene->create_animation();
    anim2->set_name("anim2");
    AnimationChannel ch2;
    ch2.value_type = AnimationValueType::float3;
    ch2.interpolation_mode = AnimationInterpolationMode::linear;
    ch2.times = {0.f, 3.f};
    ch2.values = {0.f, 0.f, 0.f, 0.f, 30.f, 0.f};
    anim2->channels.push_back(std::move(ch2));

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
    CHECK_EQ(scene->animation_duration(), doctest::Approx(3.f));

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
    AnimationChannel ch;
    ch.value_type = AnimationValueType::float3;
    ch.interpolation_mode = AnimationInterpolationMode::linear;
    ch.times = {0.f, 1.f};
    ch.values = {0.f, 0.f, 0.f, 10.f, 0.f, 0.f};
    animation->channels.push_back(std::move(ch));

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
    AnimationChannel ch;
    ch.value_type = AnimationValueType::float3;
    ch.interpolation_mode = AnimationInterpolationMode::linear;
    ch.times = {0.f, 1.f};
    ch.values = {0.f, 0.f, 0.f, 10.f, 0.f, 0.f};
    animation->channels.push_back(std::move(ch));

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
    AnimationChannel ch;
    ch.value_type = AnimationValueType::float3;
    ch.interpolation_mode = AnimationInterpolationMode::linear;
    ch.times = {0.f, 1.f, 2.f};
    ch.values = {
        0.f,
        0.f,
        0.f,
        5.f,
        0.f,
        0.f,
        10.f,
        0.f,
        0.f,
    };
    animation->channels.push_back(std::move(ch));

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

TEST_SUITE_END();
