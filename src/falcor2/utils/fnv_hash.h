// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/core/macros.h"
#include "falcor2/core/error.h"

#include <cstdint>

namespace falcor {

namespace detail {
template<typename TUInt>
struct FNVHashConstants { };

template<>
struct FNVHashConstants<uint64_t> {
    static constexpr uint64_t OFFSET_BASIS = UINT64_C(4695981039346656037);
    static constexpr uint64_t PRIME = UINT64_C(1099511628211);
};

template<>
struct FNVHashConstants<uint32_t> {
    static constexpr uint32_t OFFSET_BASIS = UINT32_C(2166136261);
    static constexpr uint32_t PRIME = UINT32_C(16777619);
};
} // namespace detail

/// Accumulates Fowler-Noll-Vo hash for inserted data.
/// To hash multiple items, create one Hash and insert all the items into it if at all possible.
/// This is superior to hashing the items individually and combining the hashes.
///
/// @tparam TUInt Type of the storage for the hash, either 32 or 64 unsigned integer.
template<typename TUInt>
class FNVHash {
public:
    static constexpr TUInt OFFSET_BASIS = detail::FNVHashConstants<TUInt>::OFFSET_BASIS;
    static constexpr TUInt PRIME = detail::FNVHashConstants<TUInt>::PRIME;

    /// Inserts all data between [begin,end) into the hash.
    /// @param begin
    /// @param end
    void insert(const void* begin, const void* end)
    {
        FALCOR_ASSERT(begin <= end);
        const uint8_t* bytes = reinterpret_cast<const uint8_t*>(begin);

        for (; bytes != end; ++bytes) {
            m_state *= PRIME;
            m_state ^= *bytes;
        }
    }

    /// Inserts all data starting at data and going for size bytes into the hash
    /// @param data
    /// @param size
    void insert(const void* data, size_t size)
    {
        const uint8_t* bytes = reinterpret_cast<const uint8_t*>(data);
        insert(bytes, bytes + size);
    }

    template<typename T>
    void insert(const T& data)
    {
        insert(&data, sizeof(T));
    }

    TUInt get() const { return m_state; }

    auto operator<=>(const FNVHash&) const = default;

private:
    TUInt m_state = OFFSET_BASIS;
};

using FNVHash64 = FNVHash<uint64_t>;
using FNVHash32 = FNVHash<uint32_t>;

inline uint64_t fnv_hash_array64(const void* data, size_t size)
{
    FNVHash64 hash;
    hash.insert(data, size);
    return hash.get();
}

inline uint32_t fnv_hash_array32(const void* data, size_t size)
{
    FNVHash32 hash;
    hash.insert(data, size);
    return hash.get();
}

} // namespace falcor
