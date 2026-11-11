# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("private_first", [False, True])
def test_private_usd_coexists_with_usd_core(private_first: bool, tmp_path: Path) -> None:
    if importlib.util.find_spec("pxr") is None:
        pytest.skip("usd-core is not installed")

    if private_first:
        imports = [
            "from falcor2.pxr import Usd as private_usd",
            "assert 'pxr' not in sys.modules",
            "from pxr import Usd as system_usd",
        ]
    else:
        imports = [
            "from pxr import Usd as system_usd",
            "system_pxr = sys.modules['pxr']",
            "from falcor2.pxr import Usd as private_usd",
            "assert sys.modules['pxr'] is system_pxr",
        ]

    runtime_checks = []
    if sys.platform == "win32":
        runtime_checks = [
            "import ctypes",
            "kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)",
            "kernel32.GetModuleHandleW.argtypes = [ctypes.c_wchar_p]",
            "kernel32.GetModuleHandleW.restype = ctypes.c_void_p",
            "kernel32.GetModuleFileNameW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_uint]",
            "private_tbb_names = ('falcor2_tbb_2021_13.dll', 'falcor2_tbb_2021_13_debug.dll')",
            "private_tbb_name = next((name for name in private_tbb_names if kernel32.GetModuleHandleW(name)), None)",
            "assert private_tbb_name",
            "private_tbb_handle = kernel32.GetModuleHandleW(private_tbb_name)",
            "assert private_tbb_handle",
            "private_tbb_path = ctypes.create_unicode_buffer(32768)",
            "assert kernel32.GetModuleFileNameW(private_tbb_handle, private_tbb_path, len(private_tbb_path))",
            "private_tbb = ctypes.CDLL(private_tbb_path.value)",
            "allocate = getattr(private_tbb, '?allocate_memory@r1@detail@tbb@@YAPEAX_K@Z')",
            "allocate.argtypes = [ctypes.c_size_t]",
            "allocate.restype = ctypes.c_void_p",
            "deallocate = getattr(private_tbb, '?deallocate_memory@r1@detail@tbb@@YAXPEAX@Z')",
            "deallocate.argtypes = [ctypes.c_void_p]",
            "deallocate.restype = None",
            "allocation = allocate(64)",
            "assert allocation",
            "deallocate(allocation)",
            "private_tbbmalloc_name = 'falcor2_tbbmalloc_2021_13_debug.dll' if private_tbb_name.endswith('_debug.dll') else 'falcor2_tbbmalloc_2021_13.dll'",
            "assert kernel32.GetModuleHandleW(private_tbbmalloc_name)",
        ]

    code = "\n".join(
        [
            "import sys",
            *imports,
            "system_stage = system_usd.Stage.CreateInMemory()",
            "private_stage = private_usd.Stage.CreateInMemory()",
            "assert type(system_stage) is not type(private_stage)",
            "assert private_usd.GetVersion() == (0, 26, 8)",
            "for provider in (system_usd, private_usd):",
            "    for iteration in range(16):",
            "        stage = provider.Stage.CreateInMemory()",
            "        for index in range(128):",
            "            stage.DefinePrim(f'/Root/Prim{index}', 'Xform')",
            "        assert len(list(stage.Traverse())) == 129",
            "        assert stage.GetRootLayer().ExportToString()",
            "        del stage",
            *runtime_checks,
        ]
    )
    script = tmp_path / "test_usd_import_order.py"
    script.write_text(code, encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert (
        result.returncode == 0
    ), f"exit code {result.returncode}\n{result.stdout}\n{result.stderr}"


def test_private_tbb_helper_lookup_names() -> None:
    import falcor2.pxr as private_pxr

    runtime_root = Path(private_pxr.__file__).resolve().parent.parent
    runtimes = {
        path.resolve()
        for path in runtime_root.glob("*falcor2_tbb_2021_13*")
        if path.suffix in (".dll", ".dylib") or ".so." in path.name
    }
    assert runtimes, f"Private oneTBB runtime not found in {runtime_root}"

    # Inspect the deployed binary, not the overlay source: build-time hwloc
    # discovery being disabled does not disable oneTBB's runtime helper probes.
    # Check all fallback names, including configuration and platform suffixes.
    for runtime in runtimes:
        contents = runtime.read_bytes()
        debug_suffix = "_debug" if "_debug" in runtime.name else ""
        if sys.platform == "win32":
            prefix, suffix = "", ".dll"
        elif sys.platform == "darwin":
            prefix, suffix = "lib", ".3.dylib"
        else:
            prefix, suffix = "lib", ".so.3"
        for helper in ("tbbbind", "tbbbind_2_0", "tbbbind_2_5"):
            private_name = f"{prefix}falcor2_{helper}_2021_13{debug_suffix}{suffix}"
            public_name = f"{prefix}{helper}{debug_suffix}{suffix}"
            assert private_name.encode() + b"\0" in contents, (runtime, private_name)
            assert b"\0" + public_name.encode() + b"\0" not in contents, (
                runtime,
                public_name,
            )


def test_private_usd_exports_are_importable() -> None:
    import falcor2.pxr as private_pxr
    from falcor2.pxr import Usd

    assert Usd.GetVersion() == (0, 26, 8)

    for module_name in private_pxr.__all__:
        importlib.import_module(f"falcor2.pxr.{module_name}")


def test_private_usd_has_generated_docstrings() -> None:
    from falcor2.pxr import Usd

    docstring = Usd.Stage.Open.__doc__
    if docstring is None or "Attempt to find a matching existing stage" not in docstring:
        pytest.skip("OpenUSD was built without optional Doxygen documentation")
    assert "Open(filePath, load=...) -> Stage" in docstring


def test_private_usd_has_generated_stubs() -> None:
    import falcor2.pxr as private_pxr
    from falcor2.pxr import Sdf, Usd, UsdGeom

    package_root = Path(private_pxr.__file__).parent
    if not (package_root / "Usd" / "__init__.pyi").is_file():
        pytest.skip("OpenUSD was built without optional generated stubs")
    for module_name in private_pxr.__all__:
        stub_path = package_root / module_name / "__init__.pyi"
        assert stub_path.is_file()
        ast.parse(stub_path.read_text(encoding="utf-8"), filename=str(stub_path))

    usd_stub = (package_root / "Usd" / "__init__.pyi").read_text(encoding="utf-8")
    sdf_stub = (package_root / "Sdf" / "__init__.pyi").read_text(encoding="utf-8")
    usd_geom_stub = (package_root / "UsdGeom" / "__init__.pyi").read_text(encoding="utf-8")
    assert "def Open(cls, filePath: str" in usd_stub
    assert "def GetPrimAtPath(self, path: _Sdf.Path) -> Prim" in usd_stub
    assert "def AppendChild(self, childName: str) -> Path" in sdf_stub
    assert "def CreatePointsAttr(self, defaultValue: Any" in usd_geom_stub
    assert "@hidden.setter" in sdf_stub
    assert "class ErrorException(RuntimeError):" in (
        package_root / "Tf" / "__init__.pyi"
    ).read_text(encoding="utf-8")

    usd_tree = ast.parse(usd_stub)
    stage = next(
        node for node in usd_tree.body if isinstance(node, ast.ClassDef) and node.name == "Stage"
    )
    open_overloads = [
        node
        for node in stage.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "Open"
    ]
    assert any(
        "Attempt to find a matching existing stage" in (ast.get_docstring(overload) or "")
        for overload in open_overloads
    )

    assert Usd.Stage.Open is not None
    assert Sdf.Path.AppendChild is not None
    assert UsdGeom.Mesh.CreatePointsAttr is not None


def test_private_usd_stubs_are_mirrored_for_editors() -> None:
    import falcor2.pxr as private_pxr

    package_root = Path(private_pxr.__file__).parent
    source_stub_root = Path(__file__).resolve().parents[3] / "falcor2" / "pxr"
    relative_stub_paths = {
        stub_path.relative_to(package_root) for stub_path in package_root.rglob("*.pyi")
    }

    if not relative_stub_paths:
        pytest.skip("OpenUSD was built without optional generated stubs")
    for relative_stub_path in relative_stub_paths:
        deployed_stub = package_root / relative_stub_path
        editor_stub = source_stub_root / relative_stub_path
        assert editor_stub.read_bytes() == deployed_stub.read_bytes()
