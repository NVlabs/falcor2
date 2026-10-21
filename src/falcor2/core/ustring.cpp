// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "ustring.h"

#include <array>
#include <atomic>
#include <cstring>
#include <memory>
#include <mutex>
#include <new>
#include <type_traits>

namespace falcor {

/// Append-only interner: each distinct string has one immutable record containing its hash,
/// length, and null-terminated bytes. A ustring points directly at those bytes, so copying and
/// comparing tokens needs no registry access. Records never move or change once published.
///
/// Lookup hashes the input once. The low bits select one of 16 independent shards; the remaining
/// bits select a starting slot in that shard's flat pointer table. Collisions advance through
/// slots with steps of 1, 2, 3, ... (quadratic probing), comparing hash, length, then bytes.
/// Readers use atomic loads without taking a mutex. The empty string uses a cached static record.
///
/// On a miss, the caller locks only the selected shard and searches its current table again.
/// This handles concurrent insertions and misses in an outdated table, ensuring that competing
/// callers return the same record. The writer initializes a new record before publishing its
/// pointer; release stores and acquire loads make its contents visible to readers.
///
/// Before an insertion would make a table more than half full, the writer doubles its capacity,
/// rehashes existing record pointers, and atomically publishes the new table. Existing readers
/// can finish using the old table because replaced arrays remain allocated until shutdown.
/// Their combined slot capacity stays below that of the current table, trading memory for
/// simple reader lifetimes without reference counts or reclamation callbacks.
///
/// Each shard owns an arena shared by string records and table arrays. Allocations advance a
/// cursor within pages that grow from 512 bytes to 64 KiB; larger requests get dedicated pages.
/// This avoids a separate heap allocation for each string. Registry destruction releases all
/// pages and requires that registry users have stopped.
///
/// Design sources are credited beside the relevant structures below.
/// Junction informed the atomic-read design; per-shard writer locks and retaining old tables
/// until shutdown are local choices that avoid its reader-registration/reclamation protocol.
/// https://github.com/preshing/junction/tree/fa76568b3bf6665965a8281d1765db8608633ee8#rules-and-behavior
class ustring::Registry {
private:
    // Shard count and arena growth sizes were selected by Falcor's interning benchmarks.
    static constexpr size_t SHARD_COUNT = 16;
    static constexpr size_t PAGE_SIZE = 64 * 1024;
    static constexpr size_t ALIGNMENT = alignof(Header);
    using Entry = std::atomic<const Header*>;
    static_assert(Entry::is_always_lock_free);

    struct Query {
        std::string_view text;
        size_t hash;
    };

    // Inspired by OpenImageIO's TableRepMap: flat pointer slots, quadratic probing,
    // power-of-two growth, and a maximum load factor of 0.5. Atomic slots are our addition.
    // https://github.com/AcademySoftwareFoundation/OpenImageIO/blob/v3.1.13.0/src/libutil/ustring.cpp
    struct Table {
        size_t mask;

        Entry* entries() { return reinterpret_cast<Entry*>(this + 1); }

        const Header* find(const Query& query)
        {
            // The low hash bits select the shard; use the remaining bits for the bucket.
            size_t pos = (query.hash / SHARD_COUNT) & mask;
            size_t step = 1;
            while (const Header* entry = entries()[pos].load(std::memory_order_acquire)) {
                if (entry->hash == query.hash && entry->length == query.text.size()
                    && std::memcmp(entry + 1, query.text.data(), entry->length) == 0)
                    return entry;
                pos = (pos + step++) & mask;
            }
            return nullptr;
        }

        // Called only by the shard's writer. Records are immutable once published.
        void insert(const Header* entry)
        {
            size_t pos = (entry->hash / SHARD_COUNT) & mask;
            size_t step = 1;
            while (entries()[pos].load(std::memory_order_relaxed))
                pos = (pos + step++) & mask;
            entries()[pos].store(entry, std::memory_order_release);
        }
    };

    // OpenUSD's Tf_TokenRegistry demonstrates hash-selected sets with separate locks and
    // cache-line alignment to avoid false sharing. OIIO's UstringTable also partitions by hash.
    // https://github.com/PixarAnimationStudios/OpenUSD/blob/v25.02/pxr/base/tf/token.cpp
    struct alignas(64) Shard {
        struct Page {
            Page* previous;
        };
        static_assert(sizeof(Page) % ALIGNMENT == 0);
        static_assert(sizeof(Table) % alignof(Entry) == 0);
        static_assert(ALIGNMENT >= alignof(Entry));
        static_assert(std::is_trivially_destructible_v<Entry>);

        std::atomic<Table*> current{nullptr};
        std::mutex mutex;
        size_t count{0};
        Page* pages{nullptr};
        uint8_t* cursor{nullptr};
        size_t remaining{0};
        size_t next_page_size{512};

        ~Shard()
        {
            // All users must have stopped before registry destruction.
            while (pages) {
                Page* previous = pages->previous;
                delete[] reinterpret_cast<uint8_t*>(pages);
                pages = previous;
            }
        }

        void* new_page(size_t size)
        {
            auto* memory = new uint8_t[sizeof(Page) + size];
            pages = new (memory) Page{pages};
            return pages + 1;
        }

        // OIIO's TableRepMap::pool_alloc motivates pooling immutable string records per shard.
        // Sharing the arena with table arrays and growing pages from 512 bytes to 64 KiB are local choices.
        void* allocate(size_t size)
        {
            if (size > std::numeric_limits<size_t>::max() - sizeof(Page) - (ALIGNMENT - 1))
                throw std::bad_array_new_length();
            size = (size + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
            if (size > PAGE_SIZE)
                return new_page(size);
            if (size > remaining) {
                size_t capacity = next_page_size;
                while (capacity < size)
                    capacity *= 2;
                cursor = static_cast<uint8_t*>(new_page(capacity));
                remaining = capacity;
                next_page_size = capacity < PAGE_SIZE ? capacity * 2 : PAGE_SIZE;
            }
            void* result = cursor;
            cursor += size;
            remaining -= size;
            return result;
        }

        const Header* insert(const Query& query)
        {
            if (Table* table = current.load(std::memory_order_acquire)) {
                if (const Header* entry = table->find(query))
                    return entry;
            }

            std::lock_guard lock(mutex);
            Table* table = current.load(std::memory_order_relaxed);
            // A reader can miss an insertion or observe an old table. Recheck under the writer lock.
            if (table) {
                if (const Header* entry = table->find(query))
                    return entry;
            }
            if (!table || count + 1 > (table->mask + 1) / 2) {
                const size_t capacity = table ? (table->mask + 1) * 2 : 16;
                auto* next = new (allocate(sizeof(Table) + capacity * sizeof(Entry))) Table{capacity - 1};
                std::uninitialized_value_construct_n(next->entries(), capacity);
                if (table) {
                    for (size_t i = 0; i <= table->mask; ++i) {
                        if (const Header* entry = table->entries()[i].load(std::memory_order_relaxed))
                            next->insert(entry);
                    }
                }
                table = next;
                // Retain old arrays in the arena so readers never need reclamation bookkeeping.
                // With doubling, their total capacity is smaller than the current array's capacity.
                current.store(table, std::memory_order_release);
            }

            auto* entry = new (allocate(sizeof(Header) + query.text.size() + 1)) Header{query.hash, query.text.size()};
            std::memcpy(entry + 1, query.text.data(), query.text.size());
            reinterpret_cast<char*>(entry + 1)[query.text.size()] = 0;
            table->insert(entry);
            ++count;
            return entry;
        }
    };

    std::array<Shard, SHARD_COUNT> m_shards;

public:
    static Registry& get()
    {
        static Registry instance;
        return instance;
    }

    const Header* insert(std::string_view text)
    {
        if (text.empty()) {
            struct Empty {
                Header header;
                char data[1];
            };
            static const Empty empty{{std::hash<std::string_view>()({}), 0}, {0}};
            return &empty.header;
        }
        if (text.size() > std::numeric_limits<size_t>::max() - sizeof(Header) - 1)
            throw std::bad_array_new_length();
        const Query query{text, std::hash<std::string_view>()(text)};
        return m_shards[query.hash & (SHARD_COUNT - 1)].insert(query);
    }
};

const char* ustring::make_unique(const std::string_view str)
{
    const Header* header = Registry::get().insert(str);
    return reinterpret_cast<const char*>(header + 1);
}

} // namespace falcor
