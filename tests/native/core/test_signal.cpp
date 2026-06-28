// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "testing.h"
#include "falcor2/core/signal.h"

#include <stdexcept>
#include <string>
#include <vector>

using namespace falcor;

// Verify that an unconnected Signal is only the size of a pointer (lazy allocation).
static_assert(sizeof(Signal<void()>) == sizeof(void*));
static_assert(sizeof(Signal<void(int)>) == sizeof(void*));
static_assert(sizeof(Signal<void(int, float, double)>) == sizeof(void*));

TEST_SUITE_BEGIN("Signal");

// ----------------------------------------------------------------------------
// Basic emit and connect
// ----------------------------------------------------------------------------

TEST_CASE("basic connect and emit")
{
    Signal<void(int)> signal;
    int received = 0;

    auto conn = signal.connect(
        [&](int v)
        {
            received = v;
        }
    );

    signal.emit(42);
    CHECK(received == 42);

    signal.emit(7);
    CHECK(received == 7);
}

TEST_CASE("multiple slots")
{
    Signal<void(int)> signal;
    int sum = 0;

    auto c1 = signal.connect(
        [&](int v)
        {
            sum += v;
        }
    );
    auto c2 = signal.connect(
        [&](int v)
        {
            sum += v * 10;
        }
    );

    signal.emit(3);
    CHECK(sum == 33); // 3 + 30
}

TEST_CASE("operator() syntax")
{
    Signal<void()> signal;
    int count = 0;

    auto conn = signal.connect(
        [&]()
        {
            count++;
        }
    );

    signal();
    CHECK(count == 1);

    signal();
    CHECK(count == 2);
}

TEST_CASE("multiple arguments")
{
    Signal<void(int, const std::string&, float)> signal;
    int ri = 0;
    std::string rs;
    float rf = 0.f;

    auto conn = signal.connect(
        [&](int i, const std::string& s, float f)
        {
            ri = i;
            rs = s;
            rf = f;
        }
    );

    signal.emit(42, "hello", 3.14f);
    CHECK(ri == 42);
    CHECK(rs == "hello");
    CHECK(rf == doctest::Approx(3.14f));
}

TEST_CASE("emit with no connections is safe")
{
    Signal<void(int, float)> signal;
    signal.emit(1, 2.0f); // Should not crash
}

TEST_CASE("empty and size")
{
    Signal<void()> signal;

    CHECK(signal.empty());
    CHECK(signal.size() == 0);

    auto c1 = signal.connect(
        []()
        {
        }
    );
    CHECK_FALSE(signal.empty());
    CHECK(signal.size() == 1);

    auto c2 = signal.connect(
        []()
        {
        }
    );
    CHECK(signal.size() == 2);

    c1.disconnect();
    CHECK(signal.size() == 1);

    c2.disconnect();
    CHECK(signal.empty());
}

// ----------------------------------------------------------------------------
// Disconnect
// ----------------------------------------------------------------------------

TEST_CASE("disconnect")
{
    Signal<void()> signal;
    int count = 0;

    auto conn = signal.connect(
        [&]()
        {
            count++;
        }
    );

    signal.emit();
    CHECK(count == 1);

    conn.disconnect();
    signal.emit();
    CHECK(count == 1); // Should not have incremented

    CHECK_FALSE(conn.connected());
}

TEST_CASE("disconnect idempotent")
{
    Signal<void()> signal;
    auto conn = signal.connect(
        []()
        {
        }
    );

    conn.disconnect();
    conn.disconnect(); // Should be safe to call again
    CHECK_FALSE(conn.connected());
}

TEST_CASE("disconnect_all")
{
    Signal<void()> signal;
    int count = 0;

    auto c1 = signal.connect(
        [&]()
        {
            count++;
        }
    );
    auto c2 = signal.connect(
        [&]()
        {
            count++;
        }
    );

    signal.emit();
    CHECK(count == 2);

    signal.disconnect_all();
    signal.emit();
    CHECK(count == 2); // No change

    CHECK_FALSE(c1.connected());
    CHECK_FALSE(c2.connected());
}

TEST_CASE("disconnect_all on empty signal")
{
    Signal<void()> signal;
    CHECK(signal.empty());

    signal.disconnect_all(); // Should not crash
    CHECK(signal.empty());

    // Signal is still usable
    int count = 0;
    auto conn = signal.connect(
        [&]()
        {
            count++;
        }
    );
    signal.emit();
    CHECK(count == 1);
}

TEST_CASE("re-connect after disconnect")
{
    Signal<void()> signal;
    int count_a = 0;
    int count_b = 0;

    auto conn = signal.connect(
        [&]()
        {
            count_a++;
        }
    );
    signal.emit();
    CHECK(count_a == 1);

    conn.disconnect();

    auto conn2 = signal.connect(
        [&]()
        {
            count_b++;
        }
    );
    signal.emit();
    CHECK(count_a == 1); // Old slot didn't fire
    CHECK(count_b == 1); // New slot fired
}

// ----------------------------------------------------------------------------
// SignalConnection
// ----------------------------------------------------------------------------

TEST_CASE("default-constructed connection")
{
    SignalConnection conn;
    CHECK_FALSE(conn.connected());
    conn.disconnect(); // Should be safe
}

TEST_CASE("SignalConnection move semantics")
{
    Signal<void()> signal;
    int count = 0;

    SignalConnection c1 = signal.connect(
        [&]()
        {
            count++;
        }
    );
    SignalConnection c2 = std::move(c1);

    CHECK_FALSE(c1.connected()); // Moved-from
    CHECK(c2.connected());

    signal.emit();
    CHECK(count == 1);

    c2.disconnect();
    signal.emit();
    CHECK(count == 1);
}

TEST_CASE("share connection - disconnect from original")
{
    Signal<void()> signal;
    int count = 0;

    auto conn = signal.connect(
        [&]()
        {
            count++;
        }
    );
    auto clone = conn.share();

    CHECK(conn.connected());
    CHECK(clone.connected());

    conn.disconnect();
    CHECK_FALSE(conn.connected());
    CHECK_FALSE(clone.connected());

    signal.emit();
    CHECK(count == 0);
}

TEST_CASE("share connection - disconnect from shared")
{
    Signal<void()> signal;
    int count = 0;

    auto conn = signal.connect(
        [&]()
        {
            count++;
        }
    );
    auto clone = conn.share();

    clone.disconnect();
    CHECK_FALSE(clone.connected());
    CHECK_FALSE(conn.connected());

    signal.emit();
    CHECK(count == 0);
}

TEST_CASE("share connection - both disconnect is idempotent")
{
    Signal<void()> signal;
    auto conn = signal.connect(
        []()
        {
        }
    );
    auto clone = conn.share();

    conn.disconnect();
    clone.disconnect(); // Second disconnect on same underlying connection
    CHECK_FALSE(conn.connected());
    CHECK_FALSE(clone.connected());
}

TEST_CASE("share of connection to destroyed signal")
{
    SignalConnection conn;
    SignalConnection clone;
    {
        Signal<void()> signal;
        conn = signal.connect(
            []()
            {
            }
        );
        clone = conn.share();
        CHECK(conn.connected());
        CHECK(clone.connected());
    }
    // Signal destroyed
    CHECK_FALSE(conn.connected());
    CHECK_FALSE(clone.connected());
}

// ----------------------------------------------------------------------------
// SignalConnection RAII
// ----------------------------------------------------------------------------

TEST_CASE("SignalConnection auto-disconnect on destruction")
{
    Signal<void()> signal;
    int count = 0;

    {
        SignalConnection conn = signal.connect(
            [&]()
            {
                count++;
            }
        );
        signal.emit();
        CHECK(count == 1);
    }
    // SignalConnection destroyed, slot disconnected

    signal.emit();
    CHECK(count == 1);
}

TEST_CASE("SignalConnection move preserves connection")
{
    Signal<void()> signal;
    int count = 0;

    SignalConnection sc1;
    {
        SignalConnection sc2 = signal.connect(
            [&]()
            {
                count++;
            }
        );
        sc1 = std::move(sc2);
    }
    // sc2 destroyed but connection moved to sc1

    signal.emit();
    CHECK(count == 1); // Still connected via sc1

    sc1.disconnect();
    signal.emit();
    CHECK(count == 1); // Disconnected
}

TEST_CASE("SignalConnection move-assign disconnects old connection")
{
    Signal<void()> signal;
    int count_a = 0;
    int count_b = 0;

    SignalConnection conn = signal.connect(
        [&]()
        {
            count_a++;
        }
    );

    // Move-assign overwrites old connection, which should auto-disconnect
    conn = signal.connect(
        [&]()
        {
            count_b++;
        }
    );

    signal.emit();
    CHECK(count_a == 0); // Old connection disconnected by move-assign
    CHECK(count_b == 1); // New connection fires
}

TEST_CASE("discard connection auto-disconnects")
{
    Signal<void()> signal;
    int count = 0;

    // Discard the returned connection (temporary destroyed immediately)
    std::ignore = signal.connect(
        [&]()
        {
            count++;
        }
    );

    signal.emit();
    CHECK(count == 0); // Auto-disconnected because connection was discarded
    CHECK(signal.empty());
}

// ----------------------------------------------------------------------------
// Signal lifetime and move semantics
// ----------------------------------------------------------------------------

TEST_CASE("signal destruction invalidates connections")
{
    SignalConnection conn;
    {
        Signal<void()> signal;
        conn = signal.connect(
            [&]()
            {
            }
        );
        CHECK(conn.connected());
    }
    // Signal destroyed
    CHECK_FALSE(conn.connected());
}

TEST_CASE("Signal move constructor")
{
    Signal<void()> sig1;
    int count = 0;

    auto conn = sig1.connect(
        [&]()
        {
            count++;
        }
    );
    Signal<void()> sig2 = std::move(sig1);

    sig2.emit();
    CHECK(count == 1);

    conn.disconnect();
    sig2.emit();
    CHECK(count == 1);
}

TEST_CASE("Signal move assignment")
{
    Signal<void()> sig1;
    Signal<void()> sig2;
    int count1 = 0;
    int count2 = 0;

    auto c1 = sig1.connect(
        [&]()
        {
            count1++;
        }
    );
    auto c2 = sig2.connect(
        [&]()
        {
            count2++;
        }
    );

    sig2 = std::move(sig1);

    // sig2 now has sig1's slot; old sig2 slot should be invalidated.
    CHECK_FALSE(c2.connected());
    CHECK(c1.connected());

    sig2.emit();
    CHECK(count1 == 1);
    CHECK(count2 == 0);
}

// ----------------------------------------------------------------------------
// Swap
// ----------------------------------------------------------------------------

TEST_CASE("swap empty signals")
{
    Signal<void()> sig1;
    Signal<void()> sig2;

    swap(sig1, sig2);

    CHECK(sig1.empty());
    CHECK(sig2.empty());
}

TEST_CASE("swap populated signal with empty")
{
    Signal<void()> sig1;
    Signal<void()> sig2;
    int count1 = 0;

    auto conn = sig1.connect(
        [&]()
        {
            count1++;
        }
    );

    swap(sig1, sig2);

    // sig1 is now empty, sig2 has the slot
    CHECK(sig1.empty());
    CHECK_FALSE(sig2.empty());

    sig2.emit();
    CHECK(count1 == 1);

    sig1.emit();
    CHECK(count1 == 1); // No change
}

TEST_CASE("swap preserves connection identity")
{
    Signal<void()> sig1;
    Signal<void()> sig2;
    int count1 = 0;
    int count2 = 0;

    auto c1 = sig1.connect(
        [&]()
        {
            count1++;
        }
    );
    auto c2 = sig2.connect(
        [&]()
        {
            count2++;
        }
    );

    CHECK(c1.connected());
    CHECK(c2.connected());

    swap(sig1, sig2);

    // Connections still valid
    CHECK(c1.connected());
    CHECK(c2.connected());

    // Slots moved: sig1 now has c2's slot, sig2 has c1's slot
    sig1.emit();
    CHECK(count2 == 1);
    CHECK(count1 == 0);

    sig2.emit();
    CHECK(count1 == 1);
    CHECK(count2 == 1);

    // Disconnect still works correctly after swap
    c1.disconnect();
    CHECK_FALSE(c1.connected());
    sig2.emit();
    CHECK(count1 == 1); // Didn't fire

    c2.disconnect();
    CHECK_FALSE(c2.connected());
    sig1.emit();
    CHECK(count2 == 1); // Didn't fire
}

TEST_CASE("swap with self is safe")
{
    Signal<void()> sig;
    int count = 0;

    auto conn = sig.connect(
        [&]()
        {
            count++;
        }
    );
    swap(sig, sig);

    CHECK(conn.connected());
    sig.emit();
    CHECK(count == 1);
}

TEST_CASE("swap signals with argument forwarding")
{
    Signal<void(int)> sig1;
    Signal<void(int)> sig2;
    std::vector<int> log1;
    std::vector<int> log2;

    auto c1 = sig1.connect(
        [&](int v)
        {
            log1.push_back(v);
        }
    );
    auto c2 = sig2.connect(
        [&](int v)
        {
            log2.push_back(v);
        }
    );

    swap(sig1, sig2);

    // sig1 now has c2's slot, sig2 has c1's slot
    sig1.emit(10);
    sig2.emit(20);

    CHECK(log1 == std::vector<int>{20}); // c1's slot received from sig2
    CHECK(log2 == std::vector<int>{10}); // c2's slot received from sig1

    // Disconnect via original handles
    c1.disconnect();
    sig2.emit(30);
    CHECK(log1 == std::vector<int>{20}); // No longer fires

    c2.disconnect();
    sig1.emit(40);
    CHECK(log2 == std::vector<int>{10}); // No longer fires
}

// ----------------------------------------------------------------------------
// Reentrancy: connect during emission
// ----------------------------------------------------------------------------

TEST_CASE("connect during emit fires on next emit only")
{
    Signal<void()> signal;
    int main_count = 0;
    int extra_count = 0;
    SignalConnection extra_conn;

    auto conn = signal.connect(
        [&]()
        {
            main_count++;
            if (main_count == 1) {
                extra_conn = signal.connect(
                    [&]()
                    {
                        extra_count++;
                    }
                );
            }
        }
    );

    signal.emit();
    CHECK(main_count == 1);
    CHECK(extra_count == 0); // Deferred, not fired this emission

    signal.emit();
    CHECK(main_count == 2);
    CHECK(extra_count == 1); // Now fires
}

TEST_CASE("connect and disconnect new connection during same emission")
{
    Signal<void()> signal;
    int count = 0;
    SignalConnection new_conn;

    auto conn = signal.connect(
        [&]()
        {
            count++;
            new_conn = signal.connect(
                [&]()
                {
                    count += 100;
                }
            );
            new_conn.disconnect(); // Disconnect immediately
        }
    );

    signal.emit();
    CHECK(count == 1);
    CHECK_FALSE(new_conn.connected());

    // The new slot was connected then disconnected during emission,
    // it should never fire.
    signal.emit();
    CHECK(count == 2); // Only original fires
}

TEST_CASE("deferred connect then deferred disconnect of same slot")
{
    Signal<void()> signal;
    int count = 0;
    SignalConnection new_conn;

    auto c1 = signal.connect(
        [&]()
        {
            count++;
            // First slot connects a new one during emission
            new_conn = signal.connect(
                [&]()
                {
                    count += 100;
                }
            );
        }
    );
    auto c2 = signal.connect(
        [&]()
        {
            count += 10;
            // Second slot disconnects the newly connected one
            new_conn.disconnect();
        }
    );

    signal.emit();
    CHECK(count == 11); // c1 (1) + c2 (10), new slot deferred
    CHECK_FALSE(new_conn.connected());

    // New slot was deferred-connected then deferred-disconnected
    signal.emit();
    CHECK(count == 22); // Only c1 and c2 fire again
}

// ----------------------------------------------------------------------------
// Reentrancy: disconnect during emission
// ----------------------------------------------------------------------------

TEST_CASE("disconnect during emit suppresses later callbacks")
{
    Signal<void()> signal;
    int count_a = 0;
    int count_b = 0;
    SignalConnection conn_b;

    auto conn_a = signal.connect(
        [&]()
        {
            count_a++;
            conn_b.disconnect();
        }
    );
    conn_b = signal.connect(
        [&]()
        {
            count_b++;
        }
    );

    signal.emit();
    CHECK(count_a == 1);
    CHECK(count_b == 0); // Tombstoned by conn_a's callback

    signal.emit();
    CHECK(count_a == 2);
    CHECK(count_b == 0); // Permanently disconnected
}

TEST_CASE("self-disconnect during emission")
{
    Signal<void()> signal;
    int count = 0;
    SignalConnection conn;

    conn = signal.connect(
        [&]()
        {
            count++;
            conn.disconnect();
        }
    );

    signal.emit();
    CHECK(count == 1);
    CHECK_FALSE(conn.connected());

    // Should not fire again
    signal.emit();
    CHECK(count == 1);
}

TEST_CASE("cross-signal disconnect during emission")
{
    Signal<void()> sig_a;
    Signal<void()> sig_b;
    int count_a = 0;
    int count_b = 0;
    SignalConnection conn_b;

    auto conn_a = sig_a.connect(
        [&]()
        {
            count_a++;
            conn_b.disconnect(); // Disconnect a slot on sig_b
        }
    );
    conn_b = sig_b.connect(
        [&]()
        {
            count_b++;
        }
    );

    sig_a.emit();
    CHECK(count_a == 1);
    CHECK_FALSE(conn_b.connected());

    sig_b.emit();
    CHECK(count_b == 0); // Was disconnected by sig_a's slot
}

// ----------------------------------------------------------------------------
// Reentrancy: disconnect_all during emission
// ----------------------------------------------------------------------------

TEST_CASE("disconnect_all during emit suppresses remaining")
{
    Signal<void()> signal;
    int count_a = 0;
    int count_b = 0;

    auto c1 = signal.connect(
        [&]()
        {
            count_a++;
            signal.disconnect_all();
        }
    );
    auto c2 = signal.connect(
        [&]()
        {
            count_b++;
        }
    );

    signal.emit();
    CHECK(count_a == 1);
    CHECK(count_b == 0); // Suppressed

    signal.emit();
    CHECK(count_a == 1); // All disconnected
    CHECK(count_b == 0);
    CHECK_FALSE(c1.connected());
    CHECK_FALSE(c2.connected());
}

TEST_CASE("disconnect_all called multiple times during same emission")
{
    Signal<void()> signal;
    int count_a = 0;
    int count_b = 0;
    int count_c = 0;

    auto c1 = signal.connect(
        [&]()
        {
            count_a++;
            signal.disconnect_all(); // First disconnect_all
        }
    );
    auto c2 = signal.connect(
        [&]()
        {
            count_b++;
            signal.disconnect_all(); // Second disconnect_all (redundant)
        }
    );
    auto c3 = signal.connect(
        [&]()
        {
            count_c++;
        }
    );

    signal.emit();
    CHECK(count_a == 1);
    CHECK(count_b == 0); // Suppressed by first disconnect_all
    CHECK(count_c == 0); // Suppressed
    CHECK_FALSE(c1.connected());
    CHECK_FALSE(c2.connected());
    CHECK_FALSE(c3.connected());
    CHECK(signal.empty());

    // Signal should still be usable
    int count_new = 0;
    auto c4 = signal.connect(
        [&]()
        {
            count_new++;
        }
    );
    signal.emit();
    CHECK(count_new == 1);
}

TEST_CASE("disconnect_all during emit invalidates deferred connects")
{
    Signal<void()> signal;
    int count = 0;
    SignalConnection deferred_conn;

    auto c1 = signal.connect(
        [&]()
        {
            count++;
            // Connect a new slot during emission.
            deferred_conn = signal.connect(
                [&]()
                {
                    count += 100;
                }
            );
            // Then disconnect everything.
            signal.disconnect_all();
        }
    );

    signal.emit();
    CHECK(count == 1);
    // The deferred connection should be invalidated by disconnect_all.
    CHECK_FALSE(deferred_conn.connected());
    CHECK(signal.empty());

    signal.emit();
    CHECK(count == 1); // Nothing fires
}

// ----------------------------------------------------------------------------
// Reentrancy: nested emit
// ----------------------------------------------------------------------------

TEST_CASE("nested emit")
{
    Signal<void(int)> signal;
    std::vector<int> log;

    auto conn = signal.connect(
        [&](int depth)
        {
            log.push_back(depth);
            if (depth > 0) {
                signal.emit(depth - 1);
            }
        }
    );

    signal.emit(3);
    CHECK(log == std::vector<int>{3, 2, 1, 0});
}

TEST_CASE("empty and size during emission")
{
    Signal<void()> signal;
    bool was_empty = true;
    size_t observed_size = 0;
    SignalConnection c2;

    auto c1 = signal.connect(
        [&]()
        {
            c2.disconnect(); // Tombstone c2
            was_empty = signal.empty();
            observed_size = signal.size();
        }
    );
    c2 = signal.connect(
        [&]()
        {
        }
    );

    signal.emit();
    // After c2 is tombstoned, only c1 is active
    CHECK_FALSE(was_empty);
    CHECK(observed_size == 1);
}

// ----------------------------------------------------------------------------
// Exception safety
// ----------------------------------------------------------------------------

TEST_CASE("slot exception leaves signal in consistent state")
{
    Signal<void()> signal;
    int count = 0;

    auto c1 = signal.connect(
        [&]()
        {
            count++;
            throw std::runtime_error("boom");
        }
    );
    auto c2 = signal.connect(
        [&]()
        {
            count += 10;
        }
    );

    CHECK_THROWS_AS(signal.emit(), std::runtime_error);
    CHECK(count == 1); // c1 fired, c2 skipped due to exception

    // Signal should still be usable after the exception.
    // Disconnect the throwing slot, then verify c2 still works.
    c1.disconnect();
    count = 0;
    CHECK_NOTHROW(signal.emit());
    CHECK(count == 10); // Only c2 fires
}

TEST_CASE("slot exception processes deferred operations")
{
    Signal<void()> signal;
    int count = 0;
    SignalConnection extra_conn;

    auto c1 = signal.connect(
        [&]()
        {
            count++;
            // Connect during emission, then throw.
            extra_conn = signal.connect(
                [&]()
                {
                    count += 100;
                }
            );
            throw std::runtime_error("boom");
        }
    );

    CHECK_THROWS(signal.emit());
    CHECK(count == 1);
    // The deferred connection should have been promoted despite the exception.
    CHECK(extra_conn.connected());
    CHECK(signal.size() == 2);

    // Disconnect the throwing slot before verifying the deferred one fires.
    c1.disconnect();
    count = 0;
    signal.emit();
    CHECK(count == 100); // Only the deferred slot fires
}

TEST_CASE("exception in nested emit")
{
    Signal<void(int)> signal;
    std::vector<int> log;

    auto conn = signal.connect(
        [&](int depth)
        {
            log.push_back(depth);
            if (depth == 1) {
                throw std::runtime_error("inner boom");
            }
            if (depth > 0) {
                signal.emit(depth - 1);
            }
        }
    );

    // Emit depth 3: fires 3, 2, 1 (throws at 1)
    CHECK_THROWS_AS(signal.emit(3), std::runtime_error);
    CHECK(log == std::vector<int>{3, 2, 1});

    // Signal should still be usable
    log.clear();
    signal.emit(0);
    CHECK(log == std::vector<int>{0});
}

TEST_CASE("multiple sequential exceptions")
{
    Signal<void()> signal;
    int count = 0;

    auto c1 = signal.connect(
        [&]()
        {
            count++;
            throw std::runtime_error("first");
        }
    );

    CHECK_THROWS(signal.emit());
    CHECK(count == 1);

    CHECK_THROWS(signal.emit());
    CHECK(count == 2);

    // Replace with non-throwing slot
    c1.disconnect();
    auto c2 = signal.connect(
        [&]()
        {
            count += 10;
        }
    );

    CHECK_NOTHROW(signal.emit());
    CHECK(count == 12);
}

// ----------------------------------------------------------------------------
// SignalConnectionGroup
// ----------------------------------------------------------------------------

TEST_CASE("SignalConnectionGroup disconnect_all")
{
    Signal<void()> sig1;
    Signal<void()> sig2;
    int count = 0;

    SignalConnectionGroup group;
    group.add(sig1.connect(
        [&]()
        {
            count++;
        }
    ));
    group.add(sig2.connect(
        [&]()
        {
            count++;
        }
    ));

    sig1.emit();
    sig2.emit();
    CHECK(count == 2);

    group.disconnect_all();

    sig1.emit();
    sig2.emit();
    CHECK(count == 2); // No change
    CHECK(group.empty());
}

TEST_CASE("SignalConnectionGroup empty and size")
{
    Signal<void()> signal;

    SignalConnectionGroup group;
    CHECK(group.empty());
    CHECK(group.size() == 0);

    group.add(signal.connect(
        []()
        {
        }
    ));
    CHECK_FALSE(group.empty());
    CHECK(group.size() == 1);

    group.add(signal.connect(
        []()
        {
        }
    ));
    CHECK(group.size() == 2);

    group.disconnect_all();
    CHECK(group.empty());
    CHECK(group.size() == 0);
}

TEST_CASE("SignalConnectionGroup prunes dead connections on add")
{
    Signal<void()> sig1;
    Signal<void()> sig2;
    Signal<void()> sig3;
    int count = 0;

    SignalConnectionGroup group;
    auto c1 = sig1.connect(
        [&]()
        {
            count++;
        }
    );
    group.add(c1.share());

    group.add(sig2.connect(
        [&]()
        {
            count++;
        }
    ));

    // Disconnect c1 externally
    c1.disconnect();

    // Add another connection - should prune the dead c1 entry
    group.add(sig3.connect(
        [&]()
        {
            count++;
        }
    ));

    // Emit all - only sig2 and sig3 should fire
    sig1.emit();
    sig2.emit();
    sig3.emit();
    CHECK(count == 2);
}

TEST_CASE("SignalConnectionGroup move constructor")
{
    Signal<void()> signal;
    int count = 0;

    SignalConnectionGroup group1;
    group1.add(signal.connect(
        [&]()
        {
            count++;
        }
    ));
    CHECK(group1.size() == 1);

    SignalConnectionGroup group2(std::move(group1));
    CHECK(group2.size() == 1);

    signal.emit();
    CHECK(count == 1);

    group2.disconnect_all();
    signal.emit();
    CHECK(count == 1);
}

TEST_CASE("SignalConnectionGroup move assignment")
{
    Signal<void()> sig1;
    Signal<void()> sig2;
    int count1 = 0;
    int count2 = 0;

    SignalConnectionGroup group1;
    group1.add(sig1.connect(
        [&]()
        {
            count1++;
        }
    ));

    SignalConnectionGroup group2;
    group2.add(sig2.connect(
        [&]()
        {
            count2++;
        }
    ));

    group2 = std::move(group1);
    // group2 now holds sig1's connection; old sig2 connection is disconnected by RAII

    group2.disconnect_all();
    sig1.emit();
    sig2.emit();
    CHECK(count1 == 0); // Disconnected via group2
    CHECK(count2 == 0); // Disconnected by move-assignment RAII
}

// ----------------------------------------------------------------------------
// SignalConnectionGroup RAII
// ----------------------------------------------------------------------------

TEST_CASE("SignalConnectionGroup auto-disconnect on destruction")
{
    Signal<void()> signal;
    int count = 0;

    {
        SignalConnectionGroup group;
        group.add(signal.connect(
            [&]()
            {
                count++;
            }
        ));
        signal.emit();
        CHECK(count == 1);
    }
    // Group destroyed, all tracked connections disconnected

    signal.emit();
    CHECK(count == 1);
}

TEST_CASE("SignalConnectionGroup RAII multiple connections")
{
    Signal<void()> sig1;
    Signal<void()> sig2;
    int count1 = 0;
    int count2 = 0;

    {
        SignalConnectionGroup group;
        group.add(sig1.connect(
            [&]()
            {
                count1++;
            }
        ));
        group.add(sig2.connect(
            [&]()
            {
                count2++;
            }
        ));

        sig1.emit();
        sig2.emit();
        CHECK(count1 == 1);
        CHECK(count2 == 1);
    }

    sig1.emit();
    sig2.emit();
    CHECK(count1 == 1);
    CHECK(count2 == 1);
}

TEST_CASE("SignalConnectionGroup move assignment disconnects existing")
{
    Signal<void()> sig1;
    Signal<void()> sig2;
    int count1 = 0;
    int count2 = 0;

    SignalConnectionGroup scoped1;
    scoped1.add(sig1.connect(
        [&]()
        {
            count1++;
        }
    ));

    SignalConnectionGroup scoped2;
    scoped2.add(sig2.connect(
        [&]()
        {
            count2++;
        }
    ));

    // Move-assign: scoped2's old connections should be disconnected
    scoped2 = std::move(scoped1);

    sig1.emit();
    sig2.emit();
    CHECK(count1 == 1); // Now owned by scoped2
    CHECK(count2 == 0); // Was disconnected by move-assignment
}

// ----------------------------------------------------------------------------
// Stress
// ----------------------------------------------------------------------------

TEST_CASE("stress test many connections")
{
    Signal<void(int)> signal;
    std::vector<int> counts(1000, 0);

    std::vector<SignalConnection> connections;
    for (int i = 0; i < 1000; ++i) {
        connections.push_back(signal.connect(
            [&counts, i](int v)
            {
                counts[i] += v;
            }
        ));
    }

    signal.emit(1);
    for (int i = 0; i < 1000; ++i) {
        CHECK(counts[i] == 1);
    }

    // Disconnect the first half
    for (int i = 0; i < 500; ++i) {
        connections[i].disconnect();
    }

    signal.emit(2);
    for (int i = 0; i < 500; ++i) {
        CHECK(counts[i] == 1); // Didn't fire
    }
    for (int i = 500; i < 1000; ++i) {
        CHECK(counts[i] == 3); // 1 + 2
    }

    CHECK(signal.size() == 500);
}

// ----------------------------------------------------------------------------
// Additional coverage
// ----------------------------------------------------------------------------

TEST_CASE("operator() with arguments")
{
    Signal<void(int, float)> signal;
    int ri = 0;
    float rf = 0.f;

    auto conn = signal.connect(
        [&](int i, float f)
        {
            ri = i;
            rf = f;
        }
    );

    signal(99, 1.5f);
    CHECK(ri == 99);
    CHECK(rf == doctest::Approx(1.5f));
}

TEST_CASE("connecting same callable twice creates independent slots")
{
    Signal<void()> signal;
    int count = 0;

    auto fn = [&]()
    {
        count++;
    };

    auto c1 = signal.connect(fn);
    auto c2 = signal.connect(fn);

    signal.emit();
    CHECK(count == 2); // Both fire independently

    c1.disconnect();
    signal.emit();
    CHECK(count == 3); // Only c2 fires

    c2.disconnect();
    signal.emit();
    CHECK(count == 3); // Neither fires
}

TEST_CASE("SignalConnectionGroup with shared connection")
{
    Signal<void()> signal;
    int count = 0;

    auto conn = signal.connect(
        [&]()
        {
            count++;
        }
    );

    SignalConnectionGroup group;
    group.add(conn.share());

    signal.emit();
    CHECK(count == 1);

    // Disconnect via group disconnects the shared handle (and the underlying slot)
    group.disconnect_all();
    CHECK_FALSE(conn.connected());

    signal.emit();
    CHECK(count == 1); // Slot no longer fires
}

TEST_SUITE_END();
