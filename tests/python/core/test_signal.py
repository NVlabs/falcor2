# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gc

import pytest
import falcor2
from typing import Any


# =============================================================================
# Signal - connect and emit
# =============================================================================


def test_connect_and_emit():
    signal = falcor2.Signal()
    received = []
    conn = signal.connect(lambda v: received.append(v))
    signal.emit(42)
    assert received == [42]


def test_multiple_slots():
    signal = falcor2.Signal()
    results = []
    c1 = signal.connect(lambda v: results.append(("a", v)))
    c2 = signal.connect(lambda v: results.append(("b", v)))
    signal.emit(10)
    assert results == [("a", 10), ("b", 10)]


# =============================================================================
# Signal - properties
# =============================================================================


def test_empty_and_size():
    signal = falcor2.Signal()
    assert signal.empty
    assert signal.size == 0

    c1 = signal.connect(lambda v: None)
    assert not signal.empty
    assert signal.size == 1

    c2 = signal.connect(lambda v: None)
    assert signal.size == 2

    c1.disconnect()
    assert signal.size == 1

    c2.disconnect()
    assert signal.empty


# =============================================================================
# Signal - disconnect
# =============================================================================


def test_disconnect():
    signal = falcor2.Signal()
    received = []
    conn = signal.connect(lambda v: received.append(v))
    signal.emit(1)
    conn.disconnect()
    signal.emit(2)
    assert received == [1]
    assert not conn.connected


def test_share_connection():
    signal = falcor2.Signal()
    received = []
    conn = signal.connect(lambda v: received.append(v))

    clone = conn.share()
    assert clone.connected

    # Disconnecting via shared handle disconnects the same slot
    clone.disconnect()
    assert not conn.connected
    assert not clone.connected

    signal.emit(1)
    assert received == []


def test_disconnect_all():
    signal = falcor2.Signal()
    received = []
    c1 = signal.connect(lambda v: received.append(("a", v)))
    c2 = signal.connect(lambda v: received.append(("b", v)))

    signal.emit(1)
    assert len(received) == 2

    signal.disconnect_all()
    assert not c1.connected
    assert not c2.connected
    assert signal.empty

    signal.emit(2)
    assert len(received) == 2  # No change


# =============================================================================
# Signal - connect with owner (auto-disconnect on GC)
# =============================================================================


def test_connect_with_owner_fires_normally():
    """Callback fires normally while owner is alive."""
    signal = falcor2.Signal()
    received = []

    class Owner:
        pass

    owner = Owner()
    conn = signal.connect(lambda v: received.append(v), owner=owner)

    signal.emit(1)
    signal.emit(2)
    assert received == [1, 2]
    assert conn.connected


def test_connect_with_owner_gc_disconnects():
    """Owner GC auto-disconnects the connection."""
    signal = falcor2.Signal()
    received = []

    class Owner:
        pass

    owner = Owner()
    conn = signal.connect(lambda v: received.append(v), owner=owner)

    signal.emit(1)
    assert received == [1]

    del owner
    gc.collect()

    assert not conn.connected
    signal.emit(2)
    assert received == [1]


def test_connect_with_owner_manual_disconnect_before_gc():
    """Manual disconnect before owner GC is safe."""
    signal = falcor2.Signal()
    received = []

    class Owner:
        pass

    owner = Owner()
    conn = signal.connect(lambda v: received.append(v), owner=owner)
    conn.disconnect()

    signal.emit(1)
    assert received == []

    # GC should still be safe (finalizer disconnects already-disconnected conn)
    del owner
    gc.collect()


def test_connect_with_owner_signal_destroyed_before_owner():
    """Signal destroyed before owner. Finalizer should not crash."""

    class Owner:
        pass

    owner = Owner()
    signal = falcor2.Signal()
    conn = signal.connect(lambda v: None, owner=owner)

    del signal
    gc.collect()

    assert not conn.connected

    # Owner GC should not crash
    del owner
    gc.collect()


def test_connect_with_owner_multiple_connections():
    """Multiple connections with the same owner all disconnect on GC."""
    sig1 = falcor2.Signal()
    sig2 = falcor2.Signal()
    received = []

    class Owner:
        pass

    owner = Owner()
    c1 = sig1.connect(lambda v: received.append(("a", v)), owner=owner)
    c2 = sig2.connect(lambda v: received.append(("b", v)), owner=owner)

    sig1.emit(1)
    sig2.emit(2)
    assert received == [("a", 1), ("b", 2)]

    del owner
    gc.collect()

    assert not c1.connected
    assert not c2.connected
    sig1.emit(3)
    sig2.emit(4)
    assert received == [("a", 1), ("b", 2)]


def test_connect_with_owner_independent_owners():
    """Connections with different owners disconnect independently."""
    signal = falcor2.Signal()
    received = []

    class Owner:
        pass

    owner1 = Owner()
    owner2 = Owner()
    c1 = signal.connect(lambda v: received.append(("a", v)), owner=owner1)
    c2 = signal.connect(lambda v: received.append(("b", v)), owner=owner2)

    signal.emit(1)
    assert received == [("a", 1), ("b", 1)]

    # Kill owner1 only
    del owner1
    gc.collect()

    assert not c1.connected
    assert c2.connected

    signal.emit(2)
    assert received == [("a", 1), ("b", 1), ("b", 2)]


# =============================================================================
# SignalConnectionGroup - basic usage
# =============================================================================


def test_group_add_and_disconnect_all():
    signal = falcor2.Signal()
    received = []

    group = falcor2.SignalConnectionGroup()
    group.add(signal.connect(lambda v: received.append(v)))

    signal.emit(1)
    assert received == [1]

    group.disconnect_all()
    signal.emit(2)
    assert received == [1]


def test_group_multiple_signals():
    sig1 = falcor2.Signal()
    sig2 = falcor2.Signal()
    results = []

    group = falcor2.SignalConnectionGroup()
    group.add(sig1.connect(lambda v: results.append(("a", v))))
    group.add(sig2.connect(lambda v: results.append(("b", v))))

    sig1.emit(1)
    sig2.emit(2)
    assert results == [("a", 1), ("b", 2)]

    group.disconnect_all()
    sig1.emit(3)
    sig2.emit(4)
    assert results == [("a", 1), ("b", 2)]


# =============================================================================
# SignalConnectionGroup - properties
# =============================================================================


def test_group_empty_and_size():
    signal = falcor2.Signal()
    group = falcor2.SignalConnectionGroup()

    assert group.empty
    assert group.size == 0

    group.add(signal.connect(lambda v: None))
    assert not group.empty
    assert group.size == 1

    group.add(signal.connect(lambda v: None))
    assert group.size == 2

    group.disconnect_all()
    assert group.empty
    assert group.size == 0


def test_group_empty_and_size_reflect_liveness():
    """add() moves ownership to the group; original handle becomes empty."""
    signal = falcor2.Signal()
    group = falcor2.SignalConnectionGroup()

    c1 = signal.connect(lambda v: None)
    c2 = signal.connect(lambda v: None)
    group.add(c1)
    group.add(c2)
    assert group.size == 2

    # After add(), the original handles are moved-from (empty)
    assert not c1.connected
    assert not c2.connected

    # Disconnect via group
    group.disconnect_all()
    assert group.size == 0
    assert group.empty


# =============================================================================
# Signal - edge cases and re-entrancy
# =============================================================================


def test_emit_with_no_connections():
    """Emitting with no connections does not raise."""
    signal = falcor2.Signal()
    signal.emit(42)  # Should not crash or raise


def test_disconnect_during_emit_suppresses_later_callbacks():
    """Disconnecting a later slot during emission prevents it from firing."""
    signal = falcor2.Signal()
    received = []
    conn_b: falcor2.SignalConnection = None  # type: ignore[assignment]

    def slot_a(v: Any):
        nonlocal conn_b
        received.append(("a", v))
        conn_b.disconnect()

    def slot_b(v: Any):
        received.append(("b", v))

    conn_a = signal.connect(slot_a)
    conn_b = signal.connect(slot_b)

    signal.emit(1)
    assert received == [("a", 1)]
    assert not conn_b.connected


def test_connect_during_emit_fires_on_next_emit():
    """A slot connected during emission is deferred to the next emit."""
    signal = falcor2.Signal()
    received = []
    extra_conn = None

    def slot_main(v: Any):
        nonlocal extra_conn
        received.append(("main", v))
        if extra_conn is None:
            extra_conn = signal.connect(lambda v: received.append(("extra", v)))

    conn = signal.connect(slot_main)

    signal.emit(1)
    assert received == [("main", 1)]

    signal.emit(2)
    assert received == [("main", 1), ("main", 2), ("extra", 2)]


def test_exception_in_slot_propagates():
    """An exception in a slot propagates and skips remaining slots."""
    signal = falcor2.Signal()
    received = []

    def slot_throw(v: Any):
        raise RuntimeError("boom")

    def slot_after(v: Any):
        received.append(v)

    c1 = signal.connect(slot_throw)
    c2 = signal.connect(slot_after)

    with pytest.raises(RuntimeError, match="boom"):
        signal.emit(1)

    # slot_after was skipped
    assert received == []

    # Signal is still usable after exception
    c1.disconnect()
    signal.emit(2)
    assert received == [2]


def test_signal_usable_after_disconnect_all():
    """Signal can be reconnected and used after disconnect_all."""
    signal = falcor2.Signal()
    received = []

    c1 = signal.connect(lambda v: received.append(("old", v)))
    signal.emit(1)
    assert received == [("old", 1)]

    signal.disconnect_all()

    c2 = signal.connect(lambda v: received.append(("new", v)))
    signal.emit(2)
    assert received == [("old", 1), ("new", 2)]


def test_connection_raii_auto_disconnect():
    """Letting a connection go out of scope disconnects the slot."""
    signal = falcor2.Signal()
    received = []

    def connect_in_scope():
        conn = signal.connect(lambda v: received.append(v))
        signal.emit(1)
        assert received == [1]
        # conn goes out of scope here

    connect_in_scope()
    signal.emit(2)
    # Slot was disconnected when conn went out of scope
    assert received == [1]


def test_discarded_connection_auto_disconnects():
    """Not storing the connection return value auto-disconnects immediately."""
    signal = falcor2.Signal()
    received = []

    signal.connect(lambda v: received.append(v))  # Connection discarded

    signal.emit(1)
    assert received == []
    assert signal.empty


if __name__ == "__main__":
    pytest.main([__file__, "-vs"])
