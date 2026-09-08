// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing render;

/// Entry stored for one active closed material interior.
/// A default-initialized entry must be empty; its other property values are unspecified while empty.
public interface IInteriorListEntry
{
    /// Constructs an occupied entry. `material_id` must be valid.
    public __init(
        MaterialID material_id,
        uint nested_priority,
        float interior_ior,
        HomogeneousMediumProperties interior_medium
    );

    /// Returns the material identity. The value is unspecified while the entry is empty.
    public property MaterialID material_id;
    /// Returns the nested dielectric priority. The value is unspecified while the entry is empty.
    public property uint nested_priority;
    /// Returns the resolved homogeneous interior IOR. The value is unspecified while the entry is empty.
    public property float interior_ior;
    /// Returns the resolved homogeneous medium. The value is unspecified while the entry is empty.
    public property HomogeneousMediumProperties interior_medium;

    /// Returns true if the entry does not represent an active interior.
    public bool is_empty();
    /// Returns true if the entry is occupied by `material_id`. The argument must be valid.
    public bool matches(MaterialID material_id);

    /// Resets the entry to its empty state.
    [mutating]
    public void clear();
}

/// Compact surface-only interior entry storing material identity, priority, and IOR.
/// The IOR must be finite and in [1, 65504].
public struct BasicInteriorListEntry : IInteriorListEntry
{
    private uint data0 = uint(MaterialID::invalid);
    // Bits 0..7 store nested priority and bits 16..31 store the binary16 IOR.
    private uint data1 = 0x3c000000;

    public property MaterialID material_id
    {
        get {
            return MaterialID(data0);
        }
    }

    public property uint nested_priority
    {
        get {
            return data1 & 0xff;
        }
    }

    public property float interior_ior
    {
        get {
            return f16tof32(data1 >> 16);
        }
    }

    public property HomogeneousMediumProperties interior_medium
    {
        get {
            return {};
        }
    }

    public __init(
        MaterialID material_id,
        uint nested_priority,
        float interior_ior,
        HomogeneousMediumProperties interior_medium
    )
    {
        data0 = uint(material_id);
        data1 = (f32tof16(interior_ior) << 16) | (nested_priority & 0xff);
    }

    public bool is_empty() { return data0 == uint(MaterialID::invalid); }
    public bool matches(MaterialID candidate_material_id) { return data0 == uint(candidate_material_id); }

    [mutating]
    public void clear() { data0 = uint(MaterialID::invalid); }
}

/// Interior entry storing an IOR and homogeneous medium in six 32-bit words.
/// Medium coefficients and anisotropy are stored as binary16 values.
/// The IOR must be finite and in [1, 65504], medium coefficients must be finite
/// and in [0, 65504], and anisotropy must be finite and in [-0.999, 0.999].
public struct HomogeneousInteriorListEntry : IInteriorListEntry
{
    private uint data0 = uint(MaterialID::invalid);
    // Bits 0..7 store nested priority and bits 16..31 store the binary16 IOR.
    private uint data1 = 0x3c000000;
    // Binary16 sigma_a.x/y.
    private uint data2 = 0;
    // Binary16 sigma_a.z/sigma_s.x.
    private uint data3 = 0;
    // Binary16 sigma_s.y/z.
    private uint data4 = 0;
    // Bits 0..15 store binary16 anisotropy. Bits 16..31 are reserved.
    private uint data5 = 0;

    public property MaterialID material_id
    {
        get {
            return MaterialID(data0);
        }
    }

    public property uint nested_priority
    {
        get {
            return data1 & 0xff;
        }
    }

    public property float interior_ior
    {
        get {
            return f16tof32(data1 >> 16);
        }
    }

    public property HomogeneousMediumProperties interior_medium
    {
        get {
            HomogeneousMediumProperties result = {};
            result.sigma_a = float3(f16tof32(data2 & 0xffff), f16tof32(data2 >> 16), f16tof32(data3 & 0xffff));
            result.sigma_s = float3(f16tof32(data3 >> 16), f16tof32(data4 & 0xffff), f16tof32(data4 >> 16));
            result.anisotropy = f16tof32(data5 & 0xffff);
            return result;
        }
    }

    public __init(
        MaterialID material_id,
        uint nested_priority,
        float interior_ior,
        HomogeneousMediumProperties interior_medium
    )
    {
        data0 = uint(material_id);
        data1 = (f32tof16(interior_ior) << 16) | (nested_priority & 0xff);
        data2 = f32tof16(interior_medium.sigma_a.x) | (f32tof16(interior_medium.sigma_a.y) << 16);
        data3 = f32tof16(interior_medium.sigma_a.z) | (f32tof16(interior_medium.sigma_s.x) << 16);
        data4 = f32tof16(interior_medium.sigma_s.y) | (f32tof16(interior_medium.sigma_s.z) << 16);
        data5 = f32tof16(interior_medium.anisotropy);
    }

    public bool is_empty() { return data0 == uint(MaterialID::invalid); }
    public bool matches(MaterialID candidate_material_id) { return data0 == uint(candidate_material_id); }

    [mutating]
    public void clear() { data0 = uint(MaterialID::invalid); }
}

public enum class InteriorListUpdateResult : uint {
    success,
    entry_not_found,
    capacity_exceeded,
};

/// Operations required by the path tracer's link-time-selected interior list.
/// Material IDs passed to these operations and stored in entries must be valid.
/// Repeated entries with the same material ID are valid only when they describe equivalent resolved medium state.
/// Such entries are interchangeable because exits identify an occurrence by material ID alone. Callers must use
/// distinct material IDs when simultaneously active occurrences can resolve to different medium properties.
public interface IInteriorList
{
    /// Concrete entry representation stored by the list.
    associatedtype Entry : IInteriorListEntry;

    /// Constructs an occupied entry. `material_id` must be valid.
    public Entry make_entry(
        MaterialID material_id,
        uint nested_priority,
        float interior_ior,
        HomogeneousMediumProperties interior_medium
    );
    /// Returns true if any active entry matches `material_id`. The argument must be valid.
    public bool contains(MaterialID material_id);
    /// Returns true if exactly one material interior is active and it matches `material_id`.
    public bool contains_only(MaterialID material_id);
    /// Returns true if no material interior is active.
    public bool is_empty();
    /// Returns true if a boundary with `nested_priority` is optically active relative to the current top entry.
    public bool is_true_interface(uint nested_priority);
    /// Inserts one occupied entry or reports that the fixed-capacity list is full.
    [mutating]
    public InteriorListUpdateResult enter(Entry entry);
    /// Removes one active occurrence of `material_id`, or reports that none exists. The argument must be valid.
    [mutating]
    public InteriorListUpdateResult exit(MaterialID material_id);
    /// Returns the incident-side IOR before crossing the specified valid material boundary.
    public float exterior_ior(MaterialID material_id, bool is_entering);
    /// Returns the active top-priority medium, or vacuum when the list is empty.
    public HomogeneousMediumProperties active_medium();
}

/// Priority-sorted multiset of active closed material interiors.
/// Each boundary entry occupies one slot, including overlapping boundaries that share a material.
/// Count must be one of the supported sorting-network sizes 1, 2, 3, or 4.
public struct InteriorListImpl<TEntry : IInteriorListEntry, let Count : uint> : IInteriorList
{
    public typealias Entry = TEntry;

    public TEntry entries[Count] = {};

    public Entry make_entry(
        MaterialID material_id,
        uint nested_priority,
        float interior_ior,
        HomogeneousMediumProperties interior_medium
    )
    {
        return Entry(material_id, nested_priority, interior_ior, interior_medium);
    }

    public bool contains(MaterialID material_id)
    {
        for (uint i = 0; i < Count; ++i) {
            if (entries[i].matches(material_id))
                return true;
        }
        return false;
    }

    public bool contains_only(MaterialID material_id)
    {
        if (!entries[0].matches(material_id))
            return false;
        if (Count >= 2)
            return entries[1].is_empty();
        return true;
    }

    public MaterialID top_material_id() { return entries[0].is_empty() ? MaterialID::invalid : entries[0].material_id; }

    public bool is_empty() { return entries[0].is_empty(); }

    private static uint effective_priority(uint nested_priority)
    {
        return nested_priority == 0 ? 256 : nested_priority;
    }

    private static bool precedes(TEntry a, TEntry b)
    {
        if (a.is_empty())
            return false;
        if (b.is_empty())
            return true;

        uint a_priority = effective_priority(a.nested_priority);
        uint b_priority = effective_priority(b.nested_priority);
        return a_priority > b_priority || (a_priority == b_priority && uint(a.material_id) > uint(b.material_id));
    }

    [ForceInline]
    private static void compare_swap(inout TEntry a, inout TEntry b)
    {
        if (precedes(b, a)) {
            TEntry temp = a;
            a = b;
            b = temp;
        }
    }

    [mutating]
    private void sort()
    {
        // Sorting networks specialized for the supported entry counts.
        if (Count >= 2)
            compare_swap(entries[0], entries[1]);
        if (Count >= 4)
            compare_swap(entries[2], entries[3]);
        if (Count >= 3)
            compare_swap(entries[0], entries[2]);
        if (Count >= 4)
            compare_swap(entries[1], entries[3]);
        if (Count >= 3)
            compare_swap(entries[1], entries[2]);
    }

    public bool is_true_interface(uint nested_priority)
    {
        if (entries[0].is_empty() || nested_priority == 0)
            return true;
        return effective_priority(nested_priority) >= effective_priority(entries[0].nested_priority);
    }

    [mutating]
    public InteriorListUpdateResult enter(Entry entry)
    {
        bool inserted = false;
        [ForceUnroll]
        for (uint i = 0; i < Count; ++i) {
            if (entries[i].is_empty()) {
                entries[i] = entry;
                inserted = true;
                break;
            }
        }
        if (!inserted)
            return InteriorListUpdateResult::capacity_exceeded;

        sort();
        return InteriorListUpdateResult::success;
    }

    [mutating]
    public InteriorListUpdateResult exit(MaterialID material_id)
    {
        // Repeated entries for one material represent the same homogeneous medium, so any matching occurrence may
        // be removed. One occurrence remains active until each corresponding boundary has been exited.
        // Keep the sorting network outside the scan so there is only one sort call site.
        bool removed = false;
        [ForceUnroll]
        for (uint i = 0; i < Count; ++i) {
            if (entries[i].matches(material_id)) {
                entries[i].clear();
                removed = true;
                break;
            }
        }
        if (!removed)
            return InteriorListUpdateResult::entry_not_found;

        sort();
        return InteriorListUpdateResult::success;
    }

    public float exterior_ior(MaterialID material_id, bool is_entering)
    {
        if (!is_entering && entries[0].matches(material_id)) {
            if (Count >= 2)
                return entries[1].is_empty() ? 1.0 : entries[1].interior_ior;
            return 1.0;
        }
        return entries[0].is_empty() ? 1.0 : entries[0].interior_ior;
    }

    public HomogeneousMediumProperties active_medium()
    {
        return entries[0].is_empty() ? HomogeneousMediumProperties() : entries[0].interior_medium;
    }
}

/// Default interior-list specialization used by the path tracer.
public typealias InteriorList = InteriorListImpl<HomogeneousInteriorListEntry, 1>;
