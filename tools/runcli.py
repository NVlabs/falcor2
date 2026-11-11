# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Launch f2 without executing falcor2's native-library imports."""

from functools import lru_cache
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType


@lru_cache(maxsize=1)
def load_cli() -> ModuleType:
    directory = Path(__file__).resolve().parent
    for root in (directory, directory.parent):
        path = root / "falcor2/tools/cli.py"
        if path.is_file():
            spec = spec_from_file_location("_falcor2_setup", path)
            assert spec is not None and spec.loader is not None
            module = module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    raise ImportError("Cannot locate falcor2/tools/cli.py; reinstall Falcor2.")


def main(argv: list[str] | None = None) -> int:
    return load_cli().main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
