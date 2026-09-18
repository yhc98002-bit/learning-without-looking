#!/usr/bin/env python3
"""Merge a sharded trainer checkpoint into HuggingFace format.

Runs the trainer's own merger with DeepSpeed reported as unavailable, which is what an
environment where DeepSpeed is installed but CUDA_HOME is unset needs. Arguments are
passed straight through to that merger, for example:

    python scripts/merge_checkpoint.py --local_dir <run>/global_step_100/actor

The merger comes from the trainer tree that scripts/setup_easyr1.sh prepares
(artifacts/repos/EasyR1, or LWL_EASYR1).
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def merger_path() -> Path:
    tree = os.environ.get("LWL_EASYR1")
    root = Path(tree).expanduser() if tree else REPO_ROOT / "artifacts" / "repos" / "EasyR1"
    return root / "scripts" / "model_merger.py"


def main() -> int:
    merger = merger_path()
    if not merger.is_file():
        if any(arg in ("-h", "--help") for arg in sys.argv[1:]):
            print(__doc__.strip())
            print(f"\nThe merger is not present at {merger}, so its own options cannot be listed.")
            return 0
        print(f"trainer merger is absent: {merger} (run scripts/setup_easyr1.sh)", file=sys.stderr)
        return 2

    import accelerate.utils.other as accelerate_other

    accelerate_other.is_deepspeed_available = lambda: False
    runpy.run_path(str(merger), run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main())
