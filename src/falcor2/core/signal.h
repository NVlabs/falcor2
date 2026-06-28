// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @brief Single-threaded signal/slot event system.
///
/// Provides a type-safe publish/subscribe mechanism: a Signal<void(Args...)>
/// holds a list of callable slots and invokes them all when emitted.
///
/// Architecture
/// ------------
/// Signal<void(Args...)>     Core event source. Connecting a slot returns a
///                           SignalConnection handle. Emitting calls every
///                           active slot with the supplied arguments.
///
/// SignalConnection          RAII move-only handle to a single connection.
///                           Auto-disconnects on destruction. Use share()
///                           to create a second handle that shares the same
///                           connection (first dtor disconnects, second is
///                           no-op).
///
/// SignalConnectionGroup     Collects multiple SignalConnection handles for
///                           bulk disconnect_all(). Auto-disconnects all on
///                           destruction. Dead connections are pruned lazily
///                           on add().
///
/// Memory model
/// ------------
/// An unconnected Signal is a single pointer (unique_ptr<Impl>, initially null).
/// Internal state is heap-allocated on the first connect() and released when
/// the signal is next observed to be empty (i.e. after an emit() or
/// disconnect_all() finds no remaining slots). Individual disconnects via
/// SignalConnection do not trigger release because the disconnect function
/// has no back-pointer to the owning Signal. This keeps idle signals cheap
/// in objects that declare many of them.
///
/// Re-entrancy and deferred operations
/// ------------------------------------
/// Signals track an emission depth counter so they are safe to use re-entrantly
/// (emitting a signal from within a slot callback). During emission:
///
///  - New connect() calls are deferred: the slot is recorded with active=false
///    and promoted into the live slot list once the outermost emission finishes.
///    The returned SignalConnection is valid immediately.
///
///  - disconnect() calls tombstone the slot right away (active=false) so it
///    won't fire again in the current emission, but actual removal from the
///    slot vector is deferred until the outermost emission completes.
///
///  - disconnect_all() tombstones every slot immediately and sets a flag so
///    that deferred processing clears everything (including pending connects).
///
/// Exception safety
/// ----------------
/// If a slot throws, the remaining slots in that emission are skipped.
/// The emit depth is decremented and deferred operations are processed,
/// leaving the signal in a consistent state. The exception then propagates.
///
/// Connection lifetime
/// -------------------
/// Each connection is backed by a shared ConnectionControl block referenced
/// by both the Signal's slot list and the external SignalConnection handle(s).
/// When the Signal is destroyed it nullifies every control block's disconnect
/// function, so a SignalConnection that outlives its Signal can safely call
/// disconnect() (it becomes a no-op) or query connected() (returns false).

#pragma once

#include "falcor2/core/macros.h"
#include "falcor2/core/error.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

namespace falcor {

// ----------------------------------------------------------------------------
// Forward declarations
// ----------------------------------------------------------------------------

class SignalConnectionGroup;

// ----------------------------------------------------------------------------
// SignalConnection
// ----------------------------------------------------------------------------

namespace detail {

/// Shared control block for a connection.
/// Allows SignalConnection objects to outlive the Signal they were connected to.
struct ConnectionControl {
    /// Pointer to the disconnect function. Null if signal was destroyed.
    std::function<void(uint64_t)> disconnect_fn;
    uint64_t slot_id{0};
    bool connected{true};
    /// Flag indicating this slot is active for emission purposes.
    bool active{true};
};

} // namespace detail

/// RAII handle representing a signal-slot connection.
/// Auto-disconnects on destruction. Move-only.
class SignalConnection {
public:
    SignalConnection() = default;

    ~SignalConnection() { disconnect(); }

    SignalConnection(SignalConnection&& other) noexcept
        : m_control(std::move(other.m_control))
    {
    }

    SignalConnection& operator=(SignalConnection&& other) noexcept
    {
        if (this != &other) {
            disconnect();
            m_control = std::move(other.m_control);
        }
        return *this;
    }

    /// Disconnect the slot from the signal.
    void disconnect()
    {
        if (!m_control)
            return;
        if (!m_control->connected)
            return;
        m_control->connected = false;
        auto fn = m_control->disconnect_fn;
        m_control->disconnect_fn = nullptr;
        if (fn) {
            fn(m_control->slot_id);
        }
    }

    /// Check if this connection is still active.
    bool connected() const { return m_control && m_control->connected; }

    /// Create a shared duplicate that shares the same control block.
    /// Both SignalConnection objects can independently call disconnect().
    /// First destructor disconnects; second is a no-op.
    [[nodiscard]] SignalConnection share() const { return SignalConnection(m_control); }

private:
    template<typename>
    friend class Signal;
    friend class SignalConnectionGroup;

    explicit SignalConnection(std::shared_ptr<detail::ConnectionControl> control)
        : m_control(std::move(control))
    {
    }

    std::shared_ptr<detail::ConnectionControl> m_control;

    FALCOR_NON_COPYABLE(SignalConnection);
};

// ----------------------------------------------------------------------------
// Signal
// ----------------------------------------------------------------------------

/// Primary template (undefined). Use Signal<void(Args...)>.
template<typename Signature>
class Signal;

/// Signal that emits void(Args...) to connected slots.
/// Single-threaded. Safe to connect/disconnect during emission via deferred operations.
///
/// Uses lazy allocation: an unconnected signal is only sizeof(void*).
/// Internal state is heap-allocated on first connect() and freed when
/// the last slot disconnects (outside of emission).
template<typename... Args>
class Signal<void(Args...)> {
public:
    using SlotFunction = std::function<void(Args...)>;

    Signal() = default;

    ~Signal()
    {
        if (m_impl)
            m_impl->destroy();
    }

    Signal(Signal&& other) noexcept
        : m_impl(std::move(other.m_impl))
    {
        FALCOR_ASSERT(!m_impl || m_impl->m_emit_depth == 0);
    }

    Signal& operator=(Signal&& other) noexcept
    {
        if (this != &other) {
            FALCOR_ASSERT(!m_impl || m_impl->m_emit_depth == 0);
            FALCOR_ASSERT(!other.m_impl || other.m_impl->m_emit_depth == 0);
            if (m_impl)
                m_impl->destroy();
            m_impl = std::move(other.m_impl);
        }
        return *this;
    }

    /// Connect a callable slot. Returns a SignalConnection handle.
    /// The returned connection auto-disconnects when destroyed.
    [[nodiscard]] SignalConnection connect(SlotFunction slot) { return ensure_impl().connect_impl(std::move(slot)); }

    /// Emit the signal, calling all connected slots.
    /// Disconnect during emit suppresses later callbacks in the same emission.
    /// New connections created during emit are deferred until the next emission.
    /// If a slot throws, remaining slots are skipped but the signal is left in a
    /// consistent state (emit depth is decremented and deferred ops are processed).
    void emit(Args... args)
    {
        if (!m_impl)
            return;
        m_impl->emit(args...);
        maybe_release_impl();
    }

    /// Emit the signal (operator() syntax).
    void operator()(Args... args) { emit(args...); }

    /// Disconnect all slots.
    void disconnect_all()
    {
        if (!m_impl)
            return;
        m_impl->disconnect_all();
        maybe_release_impl();
    }

    /// Returns true if no slots are connected.
    bool empty() const
    {
        if (!m_impl)
            return true;
        return m_impl->empty();
    }

    /// Returns the number of connected slots.
    size_t size() const
    {
        if (!m_impl)
            return 0;
        return m_impl->size();
    }

    /// Swap the contents of two signals.
    /// Both signals must not be emitting. After swap, existing SignalConnection handles
    /// continue to target the signal instance now owning their slot.
    friend void swap(Signal& a, Signal& b) noexcept
    {
        if (&a == &b)
            return;
        FALCOR_ASSERT(!a.m_impl || a.m_impl->m_emit_depth == 0);
        FALCOR_ASSERT(!b.m_impl || b.m_impl->m_emit_depth == 0);
        std::swap(a.m_impl, b.m_impl);
    }

private:
    struct Impl {
        struct SlotEntry {
            uint64_t id{0};
            SlotFunction fn;
            std::shared_ptr<detail::ConnectionControl> control;
        };

        enum class DeferredOp { Connect, Disconnect };

        struct DeferredEntry {
            DeferredOp op;
            uint64_t id{0};       // For disconnect
            SlotEntry slot_entry; // For connect
        };

        SignalConnection connect_impl(SlotFunction slot)
        {
            uint64_t id = m_next_id++;
            auto control = std::make_shared<detail::ConnectionControl>();
            control->slot_id = id;
            control->connected = true;
            control->active = true;
            // Capture raw Impl* -- the Impl has a stable heap address and
            // outlives all connections (disconnect_fn is nulled on destroy).
            control->disconnect_fn = [this](uint64_t slot_id)
            {
                request_disconnect(slot_id);
            };

            SlotEntry entry;
            entry.id = id;
            entry.fn = std::move(slot);
            entry.control = control;

            if (m_emit_depth > 0) {
                // Mark as inactive until deferred processing promotes it
                control->active = false;
                DeferredEntry deferred;
                deferred.op = DeferredOp::Connect;
                deferred.slot_entry = std::move(entry);
                m_deferred.push_back(std::move(deferred));
                return SignalConnection(std::move(control));
            }

            m_slots.push_back(std::move(entry));
            return SignalConnection(std::move(control));
        }

        void emit(Args... args)
        {
            ++m_emit_depth;

            // Iterate by index. m_slots is never resized during emission:
            // new connections go to m_deferred, disconnects only tombstone.
            try {
                size_t count = m_slots.size();
                for (size_t i = 0; i < count; ++i) {
                    if (m_slots[i].control->active) {
                        m_slots[i].fn(args...);
                    }
                }
            } catch (...) {
                --m_emit_depth;
                if (m_emit_depth == 0) {
                    process_deferred();
                }
                throw;
            }

            // Process deferred operations.
            --m_emit_depth;
            if (m_emit_depth == 0) {
                process_deferred();
            }
        }

        void disconnect_all()
        {
            if (m_emit_depth > 0) {
                // Tombstone all slots immediately so they won't fire
                for (auto& slot : m_slots) {
                    slot.control->active = false;
                }
                m_deferred_disconnect_all = true;
                return;
            }
            invalidate_all();
        }

        bool empty() const
        {
            for (const auto& slot : m_slots) {
                if (slot.control->active)
                    return false;
            }
            return true;
        }

        size_t size() const
        {
            size_t count = 0;
            for (const auto& slot : m_slots) {
                if (slot.control->active)
                    ++count;
            }
            return count;
        }

        /// Returns true if impl can be released (no slots, not emitting).
        bool can_release() const { return m_slots.empty() && m_emit_depth == 0; }

        void request_disconnect(uint64_t id)
        {
            if (m_emit_depth > 0) {
                // Tombstone the slot immediately so it won't fire again
                for (auto& slot : m_slots) {
                    if (slot.id == id) {
                        slot.control->active = false;
                        break;
                    }
                }
                DeferredEntry deferred;
                deferred.op = DeferredOp::Disconnect;
                deferred.id = id;
                m_deferred.push_back(std::move(deferred));
                return;
            }
            remove_slot(id);
        }

        void remove_slot(uint64_t id)
        {
            for (auto it = m_slots.begin(); it != m_slots.end(); ++it) {
                if (it->id == id) {
                    invalidate_control(it->control);
                    m_slots.erase(it);
                    return;
                }
            }
        }

        void process_deferred()
        {
            if (m_deferred_disconnect_all) {
                m_deferred_disconnect_all = false;
                // Invalidate controls of any deferred connects so their
                // SignalConnection handles correctly report connected() == false.
                for (auto& entry : m_deferred) {
                    if (entry.op == DeferredOp::Connect) {
                        invalidate_control(entry.slot_entry.control);
                    }
                }
                m_deferred.clear();
                invalidate_all();
                return;
            }

            for (auto& entry : m_deferred) {
                switch (entry.op) {
                case DeferredOp::Connect:
                    // Skip slots that were already disconnected during emission
                    if (!entry.slot_entry.control->connected)
                        break;
                    // Activate the slot now that emission is complete
                    entry.slot_entry.control->active = true;
                    m_slots.push_back(std::move(entry.slot_entry));
                    break;
                case DeferredOp::Disconnect:
                    remove_slot(entry.id);
                    break;
                }
            }
            m_deferred.clear();
        }

        void invalidate_control(const std::shared_ptr<detail::ConnectionControl>& control)
        {
            if (control) {
                control->connected = false;
                control->active = false;
                control->disconnect_fn = nullptr;
            }
        }

        void invalidate_all()
        {
            for (auto& slot : m_slots) {
                invalidate_control(slot.control);
            }
            m_slots.clear();
        }

        void destroy()
        {
            for (auto& entry : m_deferred) {
                if (entry.op == DeferredOp::Connect) {
                    invalidate_control(entry.slot_entry.control);
                }
            }
            m_deferred.clear();
            invalidate_all();
        }

        std::vector<SlotEntry> m_slots;
        std::vector<DeferredEntry> m_deferred;
        uint64_t m_next_id{1};
        int m_emit_depth{0};
        bool m_deferred_disconnect_all{false};
    };

    Impl& ensure_impl()
    {
        if (!m_impl)
            m_impl = std::make_unique<Impl>();
        return *m_impl;
    }

    void maybe_release_impl()
    {
        if (m_impl && m_impl->can_release())
            m_impl.reset();
    }

    std::unique_ptr<Impl> m_impl;

    FALCOR_NON_COPYABLE(Signal);
};

// ----------------------------------------------------------------------------
// SignalConnectionGroup
// ----------------------------------------------------------------------------

/// Manages a group of signal connections with bulk disconnect.
/// RAII: auto-disconnects all on destruction and on move-assignment.
class SignalConnectionGroup {
public:
    SignalConnectionGroup() = default;

    ~SignalConnectionGroup() { disconnect_all(); }

    SignalConnectionGroup(SignalConnectionGroup&& other) noexcept
        : m_connections(std::move(other.m_connections))
    {
    }

    SignalConnectionGroup& operator=(SignalConnectionGroup&& other) noexcept
    {
        if (this != &other) {
            disconnect_all();
            m_connections = std::move(other.m_connections);
        }
        return *this;
    }

    /// Add a connection to the group.
    void add(SignalConnection conn)
    {
        prune();
        m_connections.push_back(std::move(conn));
    }

    /// Disconnect all tracked connections.
    void disconnect_all()
    {
        for (auto& conn : m_connections) {
            conn.disconnect();
        }
        m_connections.clear();
    }

    /// Returns true if no connections are tracked.
    bool empty() const
    {
        for (const auto& conn : m_connections) {
            if (conn.connected())
                return false;
        }
        return true;
    }

    /// Returns the number of live (connected) connections.
    size_t size() const
    {
        size_t count = 0;
        for (const auto& conn : m_connections) {
            if (conn.connected())
                ++count;
        }
        return count;
    }

private:
    /// Remove dead connections to keep the vector from growing unboundedly.
    void prune()
    {
        m_connections.erase(
            std::remove_if(
                m_connections.begin(),
                m_connections.end(),
                [](const SignalConnection& c)
                {
                    return !c.connected();
                }
            ),
            m_connections.end()
        );
    }

    std::vector<SignalConnection> m_connections;

    FALCOR_NON_COPYABLE(SignalConnectionGroup);
};

} // namespace falcor
