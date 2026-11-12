// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

TEST_CASE("ImporterAnimation - key range empty")
{
    ImporterAnimation anim;
    CHECK_FALSE(anim.key_range().is_valid());
    anim.channels.emplace_back();
    CHECK_FALSE(anim.key_range().is_valid());
    const auto scene = make_ref<ImporterScene>();
    CHECK_FALSE(scene->animation_range().is_valid());
}

TEST_CASE("ImporterAnimation - key range single channel")
{
    ImporterAnimation anim;
    ImporterAnimationChannel ch;
    ch.times = {0.f, 0.5f, 1.5f};
    anim.channels.push_back(ch);
    CHECK_EQ(anim.key_range().end, doctest::Approx(1.5f));
    anim.channels[0].times = {-3.f, -1.f};
    CHECK_EQ(anim.key_range().start, -3.0);
    CHECK_EQ(anim.key_range().end, -1.0);
    anim.channels[0].times = {0.f};
    CHECK(anim.key_range().is_valid());
    CHECK_EQ(anim.key_range().start, 0.f);
    CHECK_EQ(anim.key_range().end, 0.f);
}

TEST_CASE("ImporterAnimation - key range multiple channels")
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

    CHECK_EQ(anim.key_range().end, doctest::Approx(3.f));
    anim.channels.insert(anim.channels.begin(), ImporterAnimationChannel{});
    for (float start : {-10.f, 10.f}) {
        anim.channels[1].times = {start, start + 1.f};
        anim.channels[2].times = {start + 0.5f, start + 3.f};
        anim.channels[3].times.clear();
        CHECK_EQ(anim.key_range().start, start);
        CHECK_EQ(anim.key_range().end, start + 3.f);
    }

    const auto scene = make_ref<ImporterScene>();
    scene->animation = anim;
    CHECK_EQ(scene->animation_range().start, 10.f);
    CHECK_EQ(scene->animation_range().end, 13.f);
    scene->meshes.emplace_back().motion_times = {-5.f, 20.f};
    scene->curves.emplace_back().motion_times = {-6.f, 25.f};
    CHECK_EQ(scene->animation_range().start, -6.f);
    CHECK_EQ(scene->animation_range().end, 25.f);
    scene->playback_range = {0.f, 6.f};
    CHECK_EQ(scene->animation_range().start, 0.f);
    CHECK_EQ(scene->animation_range().end, 6.f);
    CHECK_EQ(scene->animation.key_range().start, 10.f);
    CHECK_EQ(scene->animation.key_range().end, 13.f);
}

TEST_SUITE_END();
