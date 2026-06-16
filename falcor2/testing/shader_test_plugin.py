# SPDX-License-Identifier: Apache-2.0

"""
Shader test plugin for pytest.

Auto-discovers test_*.slang files that import the shader_test.slangh framework,
uses Slang reflection to find functions with the _test_case_ prefix, generates
a single dispatcher compute entry point per file, and runs each test case as a
standalone pytest item.
"""

from pathlib import Path
import ast
import codeop
from dataclasses import dataclass
import re
from typing import Any, Optional

import numpy as np
import pytest
import slangpy as spy

import falcor2.testing.helpers as helpers


@dataclass(frozen=True)
class ShaderTestDirective:
    marker_name: str
    marker_args: tuple[Any, ...]
    marker_kwargs: dict[str, Any]
    line: int


_DIRECTIVE_RE = re.compile(r"^\s*//\s*PYTEST_MARK\b(.*)$")
_TEST_CASE_RE = re.compile(r"^\s*TEST_CASE\s*\(\s*([A-Za-z_]\w*)\s*\)")
_COMMENT_RE = re.compile(r"^\s*//\s?(.*)$")


def _collect_error(path: Path, line: int, message: str) -> Exception:
    return pytest.Collector.CollectError(f"{path}:{line}: {message}")


def _extract_pytest_mark_name(path: Path, line_no: int, node: ast.expr) -> str:
    if not isinstance(node, ast.Attribute):
        raise _collect_error(path, line_no, "PYTEST_MARK must call pytest.mark.<name>(...)")

    mark_attr = node.value
    if not isinstance(mark_attr, ast.Attribute) or mark_attr.attr != "mark":
        raise _collect_error(path, line_no, "PYTEST_MARK must call pytest.mark.<name>(...)")

    pytest_name = mark_attr.value
    if not isinstance(pytest_name, ast.Name) or pytest_name.id != "pytest":
        raise _collect_error(path, line_no, "PYTEST_MARK must call pytest.mark.<name>(...)")

    return node.attr


def _is_wrapped_directive_complete(payload: str) -> bool:
    source = f"PYTEST_MARK{payload}"
    try:
        compiled = codeop.compile_command(source, symbol="eval")
    except (SyntaxError, ValueError, OverflowError):
        # Syntactically complete (but potentially invalid) expressions should stop accumulation.
        return True
    return compiled is not None


def _parse_directive(path: Path, line_no: int, payload: str) -> ShaderTestDirective:
    directive_text = payload.strip()
    if not directive_text:
        raise _collect_error(path, line_no, "directive requires PYTEST_MARK(...)")

    if not directive_text.startswith("("):
        raise _collect_error(path, line_no, "directive must use PYTEST_MARK(...)")

    try:
        expression = ast.parse(f"PYTEST_MARK{directive_text}", mode="eval")
    except (ValueError, SyntaxError) as ex:
        raise _collect_error(path, line_no, f"invalid PYTEST_MARK(...): {ex}") from ex

    wrapper_call = expression.body
    if not isinstance(wrapper_call, ast.Call):
        raise _collect_error(
            path, line_no, "directive must use PYTEST_MARK(...) with one pytest mark call"
        )

    if not isinstance(wrapper_call.func, ast.Name) or wrapper_call.func.id != "PYTEST_MARK":
        raise _collect_error(path, line_no, "directive must use PYTEST_MARK(...)")

    if wrapper_call.keywords:
        raise _collect_error(path, line_no, "PYTEST_MARK(...) does not accept keyword arguments")

    if len(wrapper_call.args) != 1:
        raise _collect_error(path, line_no, "PYTEST_MARK(...) must have exactly one argument")

    call = wrapper_call.args[0]
    if not isinstance(call, ast.Call):
        raise _collect_error(path, line_no, "PYTEST_MARK must be a call to pytest.mark.<name>(...)")

    marker_name = _extract_pytest_mark_name(path, line_no, call.func)

    if not hasattr(pytest.mark, marker_name):
        raise _collect_error(path, line_no, f"unknown pytest mark '{marker_name}'")

    try:
        marker_args = tuple(ast.literal_eval(arg) for arg in call.args)
    except (ValueError, SyntaxError) as ex:
        raise _collect_error(
            path, line_no, f"positional mark arguments must be Python literals: {ex}"
        ) from ex

    marker_kwargs: dict[str, Any] = {}
    for kw in call.keywords:
        if kw.arg is None:
            raise _collect_error(
                path, line_no, "**kwargs expansion is not supported in PYTEST_MARK"
            )
        if kw.arg in marker_kwargs:
            raise _collect_error(path, line_no, f"duplicate keyword argument '{kw.arg}'")
        try:
            marker_kwargs[kw.arg] = ast.literal_eval(kw.value)
        except (ValueError, SyntaxError) as ex:
            raise _collect_error(
                path, line_no, f"keyword argument '{kw.arg}' must be a Python literal: {ex}"
            ) from ex

    return ShaderTestDirective(
        marker_name=marker_name,
        marker_args=marker_args,
        marker_kwargs=marker_kwargs,
        line=line_no,
    )


def _parse_shader_test_directives(path: Path, content: str) -> dict[str, ShaderTestDirective]:
    directives: dict[str, ShaderTestDirective] = {}
    pending: Optional[ShaderTestDirective] = None

    lines = content.splitlines()
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        line_no = idx + 1
        directive_match = _DIRECTIVE_RE.match(line)
        if directive_match:
            if pending is not None:
                raise _collect_error(
                    path,
                    line_no,
                    "multiple directives before TEST_CASE are not allowed",
                )

            # A directive may be wrapped by formatters across multiple comment lines.
            # Keep consuming comment-only continuation lines until the wrapped
            # PYTEST_MARK(...) expression is syntactically complete.
            expression_lines = [directive_match.group(1).strip()]
            next_idx = idx + 1
            while not _is_wrapped_directive_complete(
                "\n".join(expression_lines)
            ) and next_idx < len(lines):
                next_line = lines[next_idx]
                if _TEST_CASE_RE.match(next_line):
                    break

                comment_match = _COMMENT_RE.match(next_line)
                if not comment_match:
                    break

                expression_lines.append(comment_match.group(1).strip())
                next_idx += 1

            pending = _parse_directive(path, line_no, "\n".join(expression_lines))
            idx = next_idx
            continue

        test_case_match = _TEST_CASE_RE.match(line)
        if test_case_match:
            test_name = test_case_match.group(1)
            if pending is not None:
                if test_name in directives:
                    raise _collect_error(
                        path,
                        line_no,
                        f"duplicate directive binding for TEST_CASE('{test_name}')",
                    )
                directives[test_name] = pending
                pending = None

        idx += 1

    if pending is not None:
        raise _collect_error(
            path,
            pending.line,
            "directive is not followed by a TEST_CASE",
        )

    return directives


# ---------------------------------------------------------------------------
# Dispatcher generation & caching
# ---------------------------------------------------------------------------

_DISPATCHER_ENTRY_POINT = "_shader_test_dispatch"
_TEST_CASE_PREFIX = "_test_case_"


def _discover_test_functions(device: spy.Device, path: Path) -> list[str]:
    """Load a module and return all function names starting with _test_case_."""
    module = device.load_module(str(path))
    funcs = module.module_decl.children_of_kind(spy.DeclReflection.Kind.func)
    return [f.name for f in funcs if f.name.startswith(_TEST_CASE_PREFIX)]


def _strip_test_prefix(func_name: str) -> str:
    """Strip the _test_case_ prefix to get the user-facing test name."""
    return func_name[len(_TEST_CASE_PREFIX) :]


def _generate_dispatcher_source(func_names: list[str]) -> str:
    """Generate a compute entry point that dispatches to individual test functions."""
    cases = "\n".join(f"        case {i}: {name}(); break;" for i, name in enumerate(func_names))
    return (
        f"\n// Auto-generated dispatcher entry point\n"
        f"uniform uint _test_case_index;\n"
        f'[shader("compute")]\n'
        f"[numthreads(1, 1, 1)]\n"
        f"void {_DISPATCHER_ENTRY_POINT}("
        f"uint3 _tid: SV_DispatchThreadID)\n"
        f"{{\n"
        f"    switch (_test_case_index)\n"
        f"    {{\n"
        f"{cases}\n"
        f"    }}\n"
        f"}}\n"
    )


@dataclass
class _CachedShaderTest:
    session: Any  # spy.SlangSession -- prevent GC
    kernel: spy.ComputeKernel


_dispatcher_cache: dict[tuple[Path, spy.DeviceType], _CachedShaderTest] = {}


def _create_slang_session(device: spy.Device) -> Any:
    return device.create_slang_session(
        {
            "include_paths": device.slang_session.desc.compiler_options.include_paths,
            "debug_info": (
                spy.SlangDebugInfoLevel.standard
                if device.info.type != spy.DeviceType.vulkan
                else spy.SlangDebugInfoLevel.none
            ),
        }
    )


def _get_or_compile_dispatcher(
    device: spy.Device,
    device_type: spy.DeviceType,
    path: Path,
    func_names: list[str],
) -> _CachedShaderTest:
    cache_key = (path, device_type)
    cached = _dispatcher_cache.get(cache_key)
    if cached is not None:
        return cached

    source = path.read_text(encoding="utf-8")
    dispatcher = _generate_dispatcher_source(func_names)
    combined = source + "\n" + dispatcher

    session = _create_slang_session(device)
    module = session.load_module_from_source(str(path), combined, path=path)
    ep = module.entry_point(_DISPATCHER_ENTRY_POINT)
    program = session.link_program([module], [ep])
    kernel = device.create_compute_kernel(program)

    result = _CachedShaderTest(session=session, kernel=kernel)
    _dispatcher_cache[cache_key] = result
    return result


def clear_dispatcher_cache() -> None:
    """Release all cached dispatcher sessions and kernels."""
    _dispatcher_cache.clear()


# ---------------------------------------------------------------------------
# Test execution
# ---------------------------------------------------------------------------


def _dispatch_and_check(
    device: spy.Device,
    kernel: spy.ComputeKernel,
    test_name: str,
    extra_vars: Optional[dict[str, Any]] = None,
) -> None:
    """Dispatch a kernel, flush prints, and assert the failure counter is zero."""
    counter_buffer = device.create_buffer(
        element_count=1,
        resource_type_layout=kernel.reflection["g_test_failure_count"],
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
        data=np.zeros(1, dtype=np.uint32),
    )

    dispatch_vars: dict[str, Any] = {"g_test_failure_count": counter_buffer}
    if extra_vars:
        dispatch_vars.update(extra_vars)

    kernel.dispatch(thread_count=[1, 1, 1], vars=dispatch_vars)

    print_output = device.flush_print_to_string()
    if print_output:
        print(print_output, end="")

    failures = int(counter_buffer.to_numpy().view(np.uint32)[0])
    if failures > 0:
        pytest.fail(f"{failures} shader CHECK(s) failed in {test_name}")


def run_shader_test_batched(
    device: spy.Device,
    device_type: spy.DeviceType,
    path: Path,
    func_names: list[str],
    test_case_index: int,
) -> None:
    """Run a single test case using the shared dispatcher kernel."""
    cached = _get_or_compile_dispatcher(device, device_type, path, func_names)
    _dispatch_and_check(
        device,
        cached.kernel,
        _strip_test_prefix(func_names[test_case_index]),
        extra_vars={"_test_case_index": test_case_index},
    )


class SlangTestFile(pytest.File):
    """Collector that discovers TEST_CASE functions in a .slang file."""

    def collect(self):
        # Only collect files that import the test framework.
        content = self.path.read_text(encoding="utf-8", errors="replace")
        if "falcor2/testing/shader_test.slangh" not in content:
            return

        directives = _parse_shader_test_directives(self.path, content)

        # Discover test functions via Slang reflection -- correctly handles
        # commented-out or #if-0 guarded TEST_CASE blocks.
        device = helpers.get_device(helpers.DEFAULT_DEVICE_TYPES[0])
        func_names = _discover_test_functions(device, self.path)

        for idx, func_name in enumerate(func_names):
            display_name = _strip_test_prefix(func_name)
            for device_type in helpers.DEFAULT_DEVICE_TYPES:
                item = SlangTestItem.from_parent(
                    self,
                    name=f"{display_name}[{device_type.name}]",
                    entry_point_name=display_name,
                    device_type=device_type,
                    test_case_index=idx,
                    all_func_names=func_names,
                )
                directive = directives.get(display_name)
                if directive is not None:
                    marker_factory = getattr(pytest.mark, directive.marker_name)
                    item.add_marker(
                        marker_factory(*directive.marker_args, **directive.marker_kwargs)
                    )
                yield item


class SlangTestItem(pytest.Item):
    """A single shader test case (one test function x one device type)."""

    def __init__(
        self,
        name: str,
        parent: SlangTestFile,
        entry_point_name: str,
        device_type: spy.DeviceType,
        test_case_index: int,
        all_func_names: list[str],
    ):
        super().__init__(name, parent)
        self.entry_point_name = entry_point_name
        self.device_type = device_type
        self.test_case_index = test_case_index
        self.all_func_names = all_func_names

    def runtest(self):
        device = helpers.get_device(self.device_type, enable_print=True)
        run_shader_test_batched(
            device,
            self.device_type,
            self.path,
            self.all_func_names,
            self.test_case_index,
        )

    def repr_failure(self, excinfo: Any, style: Any = None):
        return str(excinfo.value)

    def reportinfo(self):
        return self.path, None, f"{self.entry_point_name}[{self.device_type.name}]"
