// SPDX-License-Identifier: Apache-2.0

#include "ustring.h"

#include <cstring>
#include <set>
#include <shared_mutex>
#include <vector>
#include <memory>
#include <mutex>

namespace falcor {

class ustring::Registry {
private:
    /// Allocation page size.
    static constexpr size_t PAGE_SIZE = 64 * 1024;
    /// Alignment of entries.
    static constexpr size_t ALIGNMENT = 16;

    class Query {
    public:
        Query(std::string_view str_)
            : str(str_)
            , hash(std::hash<std::string_view>()(str))
        {
        }

        std::string_view str;
        size_t hash{0};
    };

    void* allocate(size_t size)
    {
        size_t addr = m_addr;
        m_addr += (size + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
        size_t page = addr / PAGE_SIZE;
        while (page >= m_memory.size())
            m_memory.emplace_back(new uint8_t[PAGE_SIZE]);
        return m_memory[page].get() + (addr % PAGE_SIZE);
    }

public:
    static Registry& get()
    {
        static Registry instance;
        return instance;
    }

    const Header* insert(std::string_view str)
    {
        Query q(str);

        // Check if string is alread interned.
        {
            std::shared_lock shared_lock(m_mutex);
            auto it = m_entries.find(q);
            if (it != m_entries.end())
                return *it;
        }

        // If not, intern the string.
        {
            std::unique_lock lock(m_mutex);

            Header* header = reinterpret_cast<Header*>(allocate(sizeof(Header) + str.size() + 1));
            header->hash = q.hash;
            header->length = q.str.length();
            // Copy string and add null-termination.
            char* payload = reinterpret_cast<char*>(header + 1);
            std::memcpy(payload, str.data(), str.size());
            payload[str.size()] = 0;

            auto result = m_entries.insert(header);
            return *result.first;
        }
    }

private:
    static inline std::string_view to_str(const Header* header)
    {
        return std::string_view(reinterpret_cast<const char*>(header + 1), header->length);
    }

    struct Comparator {
        using is_transparent = int;

        bool operator()(const Header* lhs, const Header* rhs) const
        {
            if (lhs->hash != rhs->hash)
                return lhs->hash < rhs->hash;
            return to_str(lhs) < to_str(rhs);
        }

        bool operator()(const Header* lhs, const Query& rhs) const
        {
            if (lhs->hash != rhs.hash)
                return lhs->hash < rhs.hash;
            return to_str(lhs) < rhs.str;
        }

        bool operator()(const Query& lhs, const Header* rhs) const
        {
            if (lhs.hash != rhs->hash)
                return lhs.hash < rhs->hash;
            return lhs.str < to_str(rhs);
        }
    };

    // Global mutex is not idea, but good enough for now.
    // For improved performance, consider using a lock-free data structure.
    // https://github.com/preshing/junction
    std::shared_mutex m_mutex;
    std::set<Header*, Comparator> m_entries;

    size_t m_addr{0};
    std::vector<std::unique_ptr<uint8_t[]>> m_memory;
};

const char* ustring::make_unique(const std::string_view str)
{
    const Header* header = Registry::get().insert(str);
    return reinterpret_cast<const char*>(header + 1);
}

} // namespace falcor
