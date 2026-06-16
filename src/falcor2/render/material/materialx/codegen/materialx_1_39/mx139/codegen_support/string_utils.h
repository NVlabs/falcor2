// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <vector>

namespace falcor {
namespace materialx {
namespace mx139 {
namespace codegen_support {

std::string trim(std::string value);
std::string trim_trailing_whitespace_lines(const std::string& value);
std::string sanitize_identifier(std::string value);
std::string snake_case_identifier(std::string value);
void replace_all(std::string& text, const std::string& needle, const std::string& replacement);
std::vector<float> parse_float_list(std::string value);

} // namespace codegen_support
} // namespace mx139
} // namespace materialx
} // namespace falcor
