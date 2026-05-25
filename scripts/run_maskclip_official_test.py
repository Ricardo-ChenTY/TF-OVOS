#!/usr/bin/env python3
from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.maskclip_mmcv_compat import apply

REPO = ROOT / "third_party" / "official_methods" / "maskclip"

if __name__ == "__main__":
    apply()
    sys.path.insert(0, str(REPO))
    sys.argv = [str(REPO / "tools" / "test.py"), *sys.argv[1:]]
    runpy.run_path(str(REPO / "tools" / "test.py"), run_name="__main__")
