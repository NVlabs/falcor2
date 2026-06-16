// SPDX-License-Identifier: Apache-2.0

#include "testing.h"
#include "falcor2/importers/importer_types.h"

using namespace falcor;

TEST_SUITE_BEGIN("ImporterAnimation");

TEST_CASE("ImporterAnimationChannel - components_per_value")
{
    ImporterAnimationChannel ch;

    ch.value_type = AnimationValueType::float_;
    CHECK_EQ(ch.components_per_value(), 1);

    ch.value_type = AnimationValueType::float3;
    CHECK_EQ(ch.components_per_value(), 3);

    ch.value_type = AnimationValueType::quatf;
    CHECK_EQ(ch.components_per_value(), 4);
}

TEST_CASE("ImporterAnimationChannel - keyframe_count")
{
    ImporterAnimationChannel ch;
    CHECK_EQ(ch.keyframe_count(), 0);

    ch.times = {0.f, 1.f, 2.f};
    CHECK_EQ(ch.keyframe_count(), 3);
}

TEST_CASE("ImporterAnimation - duration empty")
{
    ImporterAnimation anim;
    CHECK_EQ(anim.duration(), 0.f);
}

TEST_CASE("ImporterAnimation - duration single channel")
{
    ImporterAnimation anim;
    ImporterAnimationChannel ch;
    ch.times = {0.f, 0.5f, 1.5f};
    anim.channels.push_back(ch);
    CHECK_EQ(anim.duration(), doctest::Approx(1.5f));
}

TEST_CASE("ImporterAnimation - duration multiple channels")
{
    ImporterAnimation anim;

    ImporterAnimationChannel ch1;
    ch1.times = {0.f, 1.f};

    ImporterAnimationChannel ch2;
    ch2.times = {0.f, 2.f, 3.f};

    ImporterAnimationChannel ch3;
    ch3.times = {0.f, 0.5f};

    anim.channels.push_back(ch1);
    anim.channels.push_back(ch2);
    anim.channels.push_back(ch3);

    CHECK_EQ(anim.duration(), doctest::Approx(3.f));
}

TEST_SUITE_END();
