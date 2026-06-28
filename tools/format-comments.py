#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Comment formatter for C++ and Slang shading language files.

This utility parses source files and reformats block comments to use
consistent /// style documentation comments.
"""

import argparse
import fnmatch
import os
import re
import sys
from pathlib import Path


# =============================================================================
# CONFIGURATION - Edit these settings as needed
# =============================================================================

# File extensions to process
FILE_EXTENSIONS = [".cpp", ".hpp", ".h", ".c", ".slang", ".slangh"]

# Folders to include (relative to root, supports glob patterns)
# Empty list means include all folders
INCLUDE_FOLDERS = [
    "src/**",
    "slang/**",
]

# Folders to exclude (relative to root, supports glob patterns)
EXCLUDE_FOLDERS = [
    "build/**",
    "external/**",
    ".git/**",
    "__pycache__/**",
]

# File patterns to exclude
EXCLUDE_PATTERNS = [
    "*.generated.*",
    "*.auto.*",
]

# Doxygen attribute replacements: backslash style -> at-sign style
# Set to True to enable \xxx -> @xxx conversion
CONVERT_BACKSLASH_TO_AT = True

# Common doxygen commands to convert (only used if CONVERT_BACKSLASH_TO_AT is True)
DOXYGEN_COMMANDS = [
    "param",
    "return",
    "returns",
    "brief",
    "details",
    "note",
    "warning",
    "todo",
    "bug",
    "see",
    "ref",
    "file",
    "class",
    "struct",
    "enum",
    "var",
    "def",
    "fn",
    "namespace",
    "example",
    "code",
    "endcode",
    "verbatim",
    "endverbatim",
    "deprecated",
    "since",
    "version",
    "author",
    "date",
    "pre",
    "post",
    "invariant",
    "tparam",
    "throws",
    "exception",
    "retval",
    "ingroup",
    "addtogroup",
    "defgroup",
    "name",
    "copydoc",
    "copybrief",
    "copydetails",
    "internal",
    "endinternal",
    "private",
    "public",
    "protected",
]

# Add blank line before first @param/@return
ADD_SPACING_BEFORE_PARAMS = False

# Add blank line after last @param/@return
ADD_SPACING_AFTER_PARAMS = False


# =============================================================================
# IMPLEMENTATION
# =============================================================================


def matches_any_pattern(path: str, patterns: list[str]) -> bool:
    """Check if a path matches any of the given glob patterns."""
    path = path.replace("\\", "/")
    for pattern in patterns:
        pattern = pattern.replace("\\", "/")
        if fnmatch.fnmatch(path, pattern):
            return True
        # Also check if the pattern matches as a prefix for directories
        if pattern.endswith("/**") and path.startswith(pattern[:-3]):
            return True
    return False


def should_process_file(filepath: Path, root: Path) -> bool:
    """Determine if a file should be processed based on configuration."""
    # Check extension
    if filepath.suffix.lower() not in FILE_EXTENSIONS:
        return False

    # Get relative path for pattern matching (always use forward slashes)
    try:
        rel_path = str(filepath.relative_to(root)).replace("\\", "/")
    except ValueError:
        rel_path = str(filepath).replace("\\", "/")

    # Check exclude patterns for filename
    if matches_any_pattern(filepath.name, EXCLUDE_PATTERNS):
        return False

    # Check exclude folders
    if matches_any_pattern(rel_path, EXCLUDE_FOLDERS):
        return False

    # Check include folders (if specified)
    if INCLUDE_FOLDERS:
        if not matches_any_pattern(rel_path, INCLUDE_FOLDERS):
            return False

    return True


def find_files(root: Path) -> list[Path]:
    """Find all files that should be processed."""
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        # Skip hidden directories
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]

        for filename in filenames:
            filepath = Path(dirpath) / filename
            if should_process_file(filepath, root):
                files.append(filepath)

    return sorted(files)


def convert_doxygen_backslash(text: str) -> str:
    """Convert \\xxx doxygen commands to @xxx."""
    if not CONVERT_BACKSLASH_TO_AT:
        return text

    for cmd in DOXYGEN_COMMANDS:
        # Match \cmd followed by word boundary (space, newline, [, etc.)
        pattern = rf"\\{cmd}(?=\s|\[|$)"
        text = re.sub(pattern, f"@{cmd}", text)

    return text


def normalize_param_annotations(text: str) -> str:
    """Normalize @param annotations.

    - @param[in] -> @param (input is the default)
    - @param[out] -> keep as is
    - @param[in,out] -> @param[inout]
    - @param[inout] -> keep as is
    """
    # Replace @param[in] with @param (but not @param[in,out] or @param[inout])
    text = re.sub(r"@param\[in\](?!\s*,)", "@param", text)
    # Replace @param[in,out] with @param[inout]
    text = re.sub(r"@param\[in\s*,\s*out\]", "@param[inout]", text)
    return text


def add_param_return_spacing(lines: list[str]) -> list[str]:
    """Add blank lines before first @param/@return and after last one."""
    if (not ADD_SPACING_BEFORE_PARAMS and not ADD_SPACING_AFTER_PARAMS) or not lines:
        return lines

    # Find indices of @param and @return lines
    param_return_indices = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if re.match(r"@(param|return|returns|retval|tparam|throws|exception)\b", stripped):
            param_return_indices.append(i)

    if not param_return_indices:
        return lines

    first_idx = param_return_indices[0]
    last_idx = param_return_indices[-1]

    result = []
    for i, line in enumerate(lines):
        # Add blank line before first param/return (if not already blank and not first line)
        if ADD_SPACING_BEFORE_PARAMS and i == first_idx and i > 0:
            prev_line = lines[i - 1].strip()
            if prev_line:  # Previous line is not blank
                result.append("")

        result.append(line)

        # Add blank line after last param/return
        if ADD_SPACING_AFTER_PARAMS and i == last_idx:
            if i < len(lines) - 1:
                # There's more content after - add blank line if next line isn't blank
                next_line = lines[i + 1].strip()
                if next_line:
                    result.append("")
            else:
                # This is the last line - always add trailing blank line
                result.append("")

    return result


def extract_comment_content(comment: str, comment_type: str) -> list[str]:
    """Extract the content lines from a comment block."""
    lines = comment.split("\n")
    content_lines = []

    if comment_type == "block":
        # Handle /* ... */ or /** ... */
        for i, line in enumerate(lines):
            line = line.rstrip()

            if i == 0:
                # First line: remove /* or /**
                line = re.sub(r"^\s*/\*\*?\s?", "", line)
                if line.strip() == "":
                    continue
            elif i == len(lines) - 1:
                # Last line: remove */
                line = re.sub(r"\s*\*/\s*$", "", line)
                if line.strip() == "":
                    continue

            # Remove leading * with optional space (common in block comments)
            line = re.sub(r"^\s*\*\s?", "", line)
            content_lines.append(line)

        # Dedent block comment content: find minimum leading whitespace and remove it
        # This handles cases where block comments have extra indentation
        # Skip the first line when calculating min indent since it starts right after /**
        if len(content_lines) > 1:
            # Find minimum indentation among non-empty lines (excluding first line)
            min_indent = None
            for line in content_lines[1:]:
                if line.strip():  # Only consider non-empty lines
                    leading_spaces = len(line) - len(line.lstrip())
                    if min_indent is None or leading_spaces < min_indent:
                        min_indent = leading_spaces

            # Remove the common indentation from continuation lines
            if min_indent and min_indent > 0:
                content_lines = [content_lines[0]] + [
                    line[min_indent:] if len(line) >= min_indent else line
                    for line in content_lines[1:]
                ]

    elif comment_type == "triple_slash":
        # Handle /// comments - preserve existing indentation as it may be intentional
        for line in lines:
            line = line.rstrip()
            # Remove /// with optional space
            line = re.sub(r"^\s*///\s?", "", line)
            content_lines.append(line)

    # Remove leading/trailing empty lines
    while content_lines and content_lines[0].strip() == "":
        content_lines.pop(0)
    while content_lines and content_lines[-1].strip() == "":
        content_lines.pop()

    return content_lines


def format_as_triple_slash(content_lines: list[str], indent: str) -> str:
    """Format content lines as /// comments with proper indentation."""
    if not content_lines:
        return ""

    result_lines = []
    for line in content_lines:
        if line.strip():
            result_lines.append(f"{indent}/// {line}")
        else:
            result_lines.append(f"{indent}///")

    return "\n".join(result_lines)


def is_documentation_comment(text_after: str) -> bool:
    """Check if the comment precedes actual code (not just whitespace or another comment)."""
    # Skip any blank lines
    lines = text_after.split("\n")
    for line in lines:
        stripped = line.strip()
        if stripped == "":
            continue
        # Check if it's another comment
        if stripped.startswith("//") or stripped.startswith("/*"):
            return False
        # It's actual code
        return True
    return False


def process_file_content(content: str) -> tuple[str, int]:
    """
    Process file content and reformat comments.
    Returns the modified content and number of changes made.
    """
    changes = 0
    result = content

    # Pattern for block comments (/* ... */ or /** ... */)
    # This matches block comments that:
    # 1. Start at the beginning of a line (after optional whitespace)
    # 2. End with */ followed only by whitespace and newline (not inline comments)
    # The pattern [^*]|(\*+[^*/]) matches any char except * OR a sequence of *s not followed by /
    # This prevents matching across separate block comments
    block_comment_pattern = re.compile(
        r"^([ \t]*)/\*\*?((?:[^*]|\*+(?!/))*)\*/([ \t]*\n)",
        re.MULTILINE,
    )

    # Pattern for consecutive /// comments
    triple_slash_pattern = re.compile(
        r"^([ \t]*///.*(?:\n[ \t]*///.*)*)([ \t]*\n)",
        re.MULTILINE,
    )

    def replace_block_comment(match: re.Match) -> str:
        nonlocal changes
        indent = match.group(1)
        comment_body = match.group(2)
        trailing = match.group(3)

        # Check what comes after
        end_pos = match.end()
        text_after = result[end_pos : end_pos + 500]  # Look ahead

        if not is_documentation_comment(text_after):
            return match.group(0)

        # Check if it's actually multi-line or has documentation content
        full_comment = f"/*{comment_body}*/"
        if "\n" not in full_comment and not re.search(r"@\w+|\\[a-z]+", comment_body):
            # Single line comment without doxygen - might be a regular comment, leave it
            # But still convert doxygen backslash if present
            if CONVERT_BACKSLASH_TO_AT:
                new_body = convert_doxygen_backslash(comment_body)
                new_body = normalize_param_annotations(new_body)
                if new_body != comment_body:
                    changes += 1
                    return f"{indent}/*{new_body}*/{trailing}"
            return match.group(0)

        # Extract content
        content_lines = extract_comment_content(full_comment, "block")

        if not content_lines:
            return match.group(0)

        # Convert doxygen backslash to at-sign and normalize param annotations
        content_lines = [convert_doxygen_backslash(line) for line in content_lines]
        content_lines = [normalize_param_annotations(line) for line in content_lines]

        # Add param/return spacing
        content_lines = add_param_return_spacing(content_lines)

        # Format as ///
        new_comment = format_as_triple_slash(content_lines, indent)
        changes += 1

        return new_comment + trailing

    def replace_triple_slash(match: re.Match) -> str:
        nonlocal changes
        comment_block = match.group(1)
        trailing = match.group(2)

        # Get the indentation from the first line
        first_line = comment_block.split("\n")[0]
        indent_match = re.match(r"^([ \t]*)", first_line)
        indent = indent_match.group(1) if indent_match else ""

        # Check what comes after
        end_pos = match.end()
        text_after = result[end_pos : end_pos + 500]

        if not is_documentation_comment(text_after):
            return match.group(0)

        # Extract content
        content_lines = extract_comment_content(comment_block, "triple_slash")

        if not content_lines:
            return match.group(0)

        # Convert doxygen backslash to at-sign and normalize param annotations
        new_content_lines = [convert_doxygen_backslash(line) for line in content_lines]
        new_content_lines = [normalize_param_annotations(line) for line in new_content_lines]

        # Add param/return spacing
        new_content_lines = add_param_return_spacing(new_content_lines)

        # Check if anything changed
        if new_content_lines == content_lines:
            return match.group(0)

        # Format as ///
        new_comment = format_as_triple_slash(new_content_lines, indent)
        changes += 1

        return new_comment + trailing

    # Process block comments first
    # We need to track positions of comments we've already processed to avoid infinite loops
    processed_positions = set()
    search_start = 0

    while True:
        match = block_comment_pattern.search(result, search_start)
        if not match:
            break

        # Create a key for this match to detect if we've seen it before
        match_key = (match.start(), match.group(0))
        if match_key in processed_positions:
            search_start = match.end()
            continue

        old_result = result
        old_len = len(result)
        replacement = replace_block_comment(match)
        result = result[: match.start()] + replacement + result[match.end() :]

        # If no actual replacement happened, mark as processed and move past
        if result == old_result:
            processed_positions.add(match_key)
            search_start = match.end()
        else:
            # Content changed, adjust search position based on length difference
            len_diff = len(result) - old_len
            search_start = match.start() + len(replacement)
            # Clear processed positions since content changed
            processed_positions.clear()

    # Process /// comments
    processed_positions.clear()
    search_start = 0

    while True:
        match = triple_slash_pattern.search(result, search_start)
        if not match:
            break

        match_key = (match.start(), match.group(0))
        if match_key in processed_positions:
            search_start = match.end()
            continue

        old_result = result
        old_len = len(result)
        replacement = replace_triple_slash(match)
        result = result[: match.start()] + replacement + result[match.end() :]

        if result == old_result:
            processed_positions.add(match_key)
            search_start = match.end()
        else:
            len_diff = len(result) - old_len
            search_start = match.start() + len(replacement)
            processed_positions.clear()

    return result, changes


def process_file(filepath: Path, dry_run: bool = False) -> tuple[bool, int]:
    """
    Process a single file.
    Returns (was_modified, num_changes).
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
    except UnicodeDecodeError:
        try:
            with open(filepath, "r", encoding="latin-1") as f:
                content = f.read()
        except Exception as e:
            print(f"Error reading {filepath}: {e}", file=sys.stderr)
            return False, 0

    new_content, changes = process_file_content(content)

    if changes > 0 and new_content != content:
        if not dry_run:
            with open(filepath, "w", encoding="utf-8", newline="\n") as f:
                f.write(new_content)
        return True, changes

    return False, 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Format C++ and Slang comments to use consistent /// style.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                     Process all matching files in current directory
  %(prog)s path/to/file.cpp    Process a specific file
  %(prog)s -n                  Dry run - show what would be changed
  %(prog)s -v                  Verbose output
        """,
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="Specific files to process. If not provided, searches for files in the current directory.",
    )
    parser.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="Show what would be changed without modifying files.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print verbose output.",
    )
    parser.add_argument(
        "-r",
        "--root",
        type=Path,
        default=None,
        help="Root directory to search for files. Defaults to current directory.",
    )

    args = parser.parse_args()

    root = args.root or Path.cwd()

    if args.files:
        files = [Path(f) for f in args.files]
    else:
        files = find_files(root)

    if not files:
        print("No files to process.")
        return 0

    total_files = 0
    total_changes = 0

    for filepath in files:
        if args.verbose:
            print(f"Processing: {filepath}")

        modified, changes = process_file(filepath, dry_run=args.dry_run)

        if modified:
            total_files += 1
            total_changes += changes
            action = "Would modify" if args.dry_run else "Modified"
            print(f"{action}: {filepath} ({changes} changes)")

    print(f"\nSummary: {total_changes} changes in {total_files} files")
    if args.dry_run:
        print("(dry run - no files were modified)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
