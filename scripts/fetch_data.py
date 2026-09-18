#!/usr/bin/env python3
"""Download the released data into the local data directory.

    python scripts/fetch_data.py                        # predictions part, about 22 MB
    python scripts/fetch_data.py --part instruments --part corpora --extract
    python scripts/fetch_data.py --part all --extract

The predictions part holds everything scripts/reproduce.py reads: the per-item outputs, the
realized training streams and the audit rows; its small archives are always unpacked.
The dataset's data/ goes to the data directory (LWL_DATA), its predictions/ to the predictions
directory (LWL_PREDICTIONS) and its checkpoints/ into the repository.
"""
import argparse
import shutil
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lwl.paths import data_root, predictions_root, repo_root

DATASET = "Despaireyes613/learning-without-looking"

# Download patterns per part: a trailing slash takes a whole folder.
PARTS = {
    "predictions": [
        "predictions/",
        "data/training/mixtures/",
        "data/training/constructed.tar.gz",
        "data/audit/*.jsonl",
    ],
    "instruments": ["data/scenes/", "data/grounding/", "data/audit/"],
    "corpora": ["data/training/", "data/captions/", "data/benchmarks/"],
    "checkpoint": ["checkpoints/"],
}
# Parts whose archives are unpacked without --extract.
ALWAYS_EXTRACT = {"predictions"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--part",
        action="append",
        choices=sorted(PARTS) + ["all"],
        help="which part to download; repeatable (default: predictions)",
    )
    parser.add_argument("--dataset", default=DATASET, help="dataset repository to download from")
    parser.add_argument("--dest", type=Path, default=None,
                        help="put the dataset's folders under this directory instead "
                             "(default: the data, predictions and repository directories)")
    parser.add_argument("--extract", action="store_true",
                        help="also unpack the archives of the instruments and corpora parts")
    return parser.parse_args()


def local_roots(dest):
    """Where each top-level folder of the dataset goes."""
    if dest is not None:
        return {top: dest / top for top in ("data", "predictions", "checkpoints")}
    return {"data": data_root(), "predictions": predictions_root(), "checkpoints": repo_root() / "checkpoints"}


def top_of(pattern):
    return pattern.split("/")[0]


def download(snapshot_download, dataset, top, patterns, root):
    """Fetch the files of the dataset's `top` folder that match `patterns` into `root`."""
    if root.name == top:
        root.parent.mkdir(parents=True, exist_ok=True)
        snapshot_download(repo_id=dataset, repo_type="dataset", local_dir=root.parent, allow_patterns=patterns)
        return
    # snapshot_download keeps the dataset's folder names, so a root under another name is
    # filled from a staging directory beside it.
    staging = root.parent / f".{root.name}.download"
    snapshot_download(repo_id=dataset, repo_type="dataset", local_dir=staging, allow_patterns=patterns)
    source = staging / top
    for path in sorted(source.rglob("*")):
        if path.is_file():
            target = root / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), str(target))
    shutil.rmtree(staging)


def unpack(archive):
    """Unpack one archive beside itself: scenes/training.tar.gz becomes scenes/training/."""
    with tarfile.open(archive) as tar:
        if hasattr(tarfile, "data_filter"):
            tar.extractall(archive.parent, filter="data")
        else:
            tar.extractall(archive.parent)
    print(f"unpacked {archive}")


def archives_of(root, pattern):
    """The archives a download pattern brought in under `root`."""
    local = root.joinpath(*pattern.rstrip("/").split("/")[1:])
    if pattern.endswith("/"):
        return sorted(local.rglob("*.tar.gz")) if local.is_dir() else []
    return sorted(path for path in local.parent.glob(local.name) if path.name.endswith(".tar.gz"))


def main():
    args = parse_args()
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit("huggingface_hub is required: pip install -r requirements-cpu.txt")

    parts = args.part or ["predictions"]
    if "all" in parts:
        parts = sorted(PARTS)
    parts = list(dict.fromkeys(parts))
    patterns = list(dict.fromkeys(pattern for part in parts for pattern in PARTS[part]))
    unpacked = [pattern for part in parts for pattern in PARTS[part]
                if args.extract or part in ALWAYS_EXTRACT]
    roots = local_roots(args.dest)

    for top in sorted({top_of(pattern) for pattern in patterns}):
        root = roots[top]
        download(snapshot_download, args.dataset, top, [p for p in patterns if top_of(p) == top], root)
        print(f"downloaded {top}/ into {root}")
        archives = [archive for p in unpacked if top_of(p) == top for archive in archives_of(root, p)]
        for archive in dict.fromkeys(archives):
            unpack(archive)


if __name__ == "__main__":
    main()
