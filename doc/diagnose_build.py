#!/usr/bin/env python3
"""Render the doc build manifest for CI/local diagnostics.

Usage:
    python doc/diagnose_build.py [manifest_path]

If manifest_path is omitted, defaults to doc/_build/doc_build_manifest.json
relative to this script's location.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

DOC_ROOT = Path(__file__).resolve().parent
REPO_ROOT = DOC_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from seaborn._doc_config import get_manifest_path, get_data_cache_path  # noqa: E402


def main() -> int:
    if len(sys.argv) > 1:
        manifest_path = Path(sys.argv[1])
    else:
        manifest_path = get_manifest_path(DOC_ROOT)

    print("=" * 60)
    print("DOC BUILD FAILURE DIAGNOSTICS")
    print("=" * 60)
    print()

    print("-- Build manifest --")
    if manifest_path.exists():
        print(f"Manifest file: {manifest_path}")
        print(f"Manifest size: {manifest_path.stat().st_size} bytes")
        print()
        try:
            with open(manifest_path) as f:
                d = json.load(f)
        except Exception as e:
            print(f"  <manifest unreadable: {e}>")
            return 1

        print("Stages:")
        for s in d.get("stages", []):
            extra = {k: s[k] for k in s.keys()
                     if k not in ("name", "status", "timestamp")}
            print(f"  [{s.get('status')}] {s.get('name')} "
                  f"extra={extra if extra else '<none>'}")
        print()

        total = len(d.get("examples", []))
        succ = sum(1 for e in d["examples"] if e.get("status") == "success")
        fail = sum(1 for e in d["examples"] if e.get("status") == "failure")
        print(f"Examples: total={total} successes={succ} failures={fail}")
        failed = [e for e in d.get("examples", []) if e.get("status") == "failure"]
        for e in failed[:20]:
            print(f"  FAIL: {e.get('name')} "
                  f"src={e.get('source_file')} "
                  f"datasets={e.get('datasets')} "
                  f"err={str(e.get('error', ''))[:100]}")
        print()

        datasets = d.get("datasets", {})
        print(f"Datasets referenced: {len(datasets)}")
        print(f"Cache hits:        {d.get('cache_hits', [])}")
        print(f"Cache misses:      {d.get('cache_misses', [])}")
        for name, info in list(datasets.items())[:30]:
            hit = "HIT" if info.get("cache_hit") else "MISS"
            print(f"  [{hit}] {name}: cache={info.get('cache_path')} "
                  f"source={info.get('source_url')}")
        print()

        imgs = d.get("images", [])
        print(f"Images recorded: {len(imgs)}")
        for i in imgs[-30:]:
            st = "OK" if i.get("exists") else "MISSING"
            print(f"  [{st}] {i.get('produced_by', '<unknown>')}: "
                  f"{i.get('path')} ({i.get('size_bytes')}B)")
        print()

        fails = d.get("failures", [])
        print(f"Failures logged: {len(fails)}")
        for fl in fails[:10]:
            print(f"  *** {fl.get('example')} ***")
            print(f"    src={fl.get('source_file')}")
            print(f"    datasets={fl.get('datasets')}")
            print(f"    images={fl.get('images')}")
            print(f"    error={fl.get('error_type')}: {fl.get('error_message')}")
            tb = fl.get("traceback", "").splitlines()
            for line in tb[-15:]:
                print(f"        {line}")
            print()
    else:
        print(f"  <manifest file missing at {manifest_path}>")
        print()

    print("-- Dataset cache dir --")
    cache_dir = Path(get_data_cache_path())
    print(f"SEABORN_DATA={cache_dir}")
    if cache_dir.is_dir():
        csv_count = len(list(cache_dir.glob("*.csv")))
        print(f"CSV count: {csv_count}")
        sample = sorted(cache_dir.iterdir())[:5]
        for p in sample:
            try:
                print(f"  {p.name:30s} {p.stat().st_size:>10d} bytes")
            except OSError:
                print(f"  {p.name}")
    else:
        print("  <cache dir missing>")
    print()

    print("-- Example images on disk --")
    roots = [DOC_ROOT / d for d in
             ["_build", "examples", "tutorial", "docstrings", "example_thumbs"]]
    count = 0
    for root in roots:
        if not root.exists():
            continue
        for ext in ("*.png", "*.svg"):
            for p in sorted(root.rglob(ext))[:80]:
                print(f"  {p}")
                count += 1
                if count >= 80:
                    break
            if count >= 80:
                break
        if count >= 80:
            break
    if count == 0:
        print("  <no image files found>")
    print()

    print("-- Doc config environment --")
    for v in [
        "SEABORN_DATA",
        "SEABORN_DOC_BUILD",
        "SEABORN_DOC_MPL_BACKEND",
        "SEABORN_DOC_RANDOM_SEED",
        "SEABORN_DOC_IMAGE_FORMAT",
        "SEABORN_DOC_IMAGE_DPI",
        "SEABORN_DOC_SAVEFIG_BBOX",
        "SEABORN_DOC_NB_TIMEOUT",
        "NB_KERNEL",
        "MPLBACKEND",
    ]:
        print(f"  {v}={os.environ.get(v, '<unset>')}")
    print()

    print("-- Sphinx build artifacts --")
    html_dir = DOC_ROOT / "_build" / "html"
    if html_dir.is_dir():
        entries = sorted(html_dir.iterdir())[:20]
        for e in entries:
            try:
                print(f"  {e.name:30s} "
                      f"{'DIR' if e.is_dir() else 'FILE':>4s} "
                      f"{e.stat().st_size if e.is_file() else '-':>10s}")
            except OSError:
                print(f"  {e.name}")
    else:
        print("  <no html dir>")
    buildinfo = DOC_ROOT / "_build" / "html" / ".buildinfo"
    if buildinfo.is_file():
        print()
        print(buildinfo.read_text())
    else:
        print("  <no buildinfo>")

    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
