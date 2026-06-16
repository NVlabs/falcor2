# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

# pyright: reportMissingImports=false
from fix_license import process_file

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fix_license_hook.py <files>")
        sys.exit(1)
    for fn in sys.argv[1:]:
        process_file(fn)
