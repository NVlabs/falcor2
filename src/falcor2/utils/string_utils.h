// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <string_view>
#include <vector>
#include <map>

namespace falcor {

inline size_t levenshtein_distance(std::string_view s1, std::string_view s2)
{
    size_t m = s1.size(), n = s2.size();
    std::vector<std::vector<size_t>> dp(m + 1, std::vector<size_t>(n + 1));
    for (size_t i = 0; i <= m; ++i)
        dp[i][0] = i;
    for (size_t j = 0; j <= n; ++j)
        dp[0][j] = j;
    for (size_t i = 1; i <= m; ++i)
        for (size_t j = 1; j <= n; ++j)
            dp[i][j] = std::min({dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + (s1[i - 1] != s2[j - 1])});
    return dp[m][n];
}

inline std::string replace_substring(std::string_view input, std::string_view src, std::string_view dst)
{
    std::string res(input);
    size_t offset = res.find(src);
    while (offset != std::string::npos) {
        res.replace(offset, src.length(), dst);
        offset += dst.length();
        offset = res.find(src, offset);
    }
    return res;
}

template<typename StringMap>
std::string replace_substrings(std::string_view input, const StringMap& string_map)
{
    std::string res(input);
    for (const auto& pair : string_map) {
        if (pair.first.empty())
            continue;

        size_t pos = 0;
        while ((pos = res.find(pair.first, pos)) != std::string::npos) {
            res.replace(pos, pair.first.length(), pair.second);
            pos += pair.second.length();
        }
    }
    return res;
}


} // namespace falcor
