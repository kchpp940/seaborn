#!/usr/bin/env python3
"""Unified seaborn documentation build wrapper.

This script is the single entry point for both local and CI doc builds.
It ensures the same environment (data cache path, random seed, matplotlib
backend, image params) is used everywhere, and writes a build manifest.

Usage:
    python doc/build.py --help
    python doc/build.py cache-datasets
    python doc/build.py notebooks
    python doc/build.py html
    python doc/build.py all
    python doc/build.py summary
    python doc/build.py clean
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

DOC_ROOT = Path(__file__).resolve().parent
REPO_ROOT = DOC_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from seaborn._doc_config import (
    BuildManifest,
    get_manifest,
    get_manifest_path,
    get_data_cache_path,
    init_doc_env,
    apply_random_seed,
    apply_mpl_backend,
    apply_mpl_rc,
    print_manifest_summary,
    DATASET_SOURCE_BASE,
)


SEABORN_DATA_REPO = "https://github.com/mwaskom/seaborn-data.git"


def _run(cmd, **kwargs):
    if isinstance(cmd, str):
        print(f"[doc_build] $ {cmd}", flush=True)
        kwargs.setdefault("shell", True)
    else:
        print(f"[doc_build] $ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, **kwargs)


def _ensure_env():
    init_doc_env()
    apply_mpl_backend()
    apply_random_seed()
    apply_mpl_rc()
    os.environ.setdefault("SEABORN_DOC_BASEDIR", str(DOC_ROOT))
    cache_path = get_data_cache_path()
    Path(cache_path).mkdir(parents=True, exist_ok=True)


def _stage(m, name, fn, **extra):
    m.add_stage(name, "started", **extra)
    t0 = time.time()
    try:
        rc = fn()
        if rc is None:
            rc = 0
        m.add_stage(
            name,
            "completed",
            duration_sec=round(time.time() - t0, 2),
            **extra,
        )
        return rc
    except SystemExit as e:
        m.add_stage(
            name,
            "failed",
            duration_sec=round(time.time() - t0, 2),
            error=str(e),
            **extra,
        )
        raise
    except Exception as e:
        m.add_stage(
            name,
            "failed",
            duration_sec=round(time.time() - t0, 2),
            error=f"{type(e).__name__}: {e}",
            **extra,
        )
        raise


def cmd_cache_datasets(_args):
    """Populate SEABORN_DATA cache directory."""
    cache_path = get_data_cache_path()
    m = get_manifest()
    cache_dir = Path(cache_path)
    status = {}
    if cache_dir.exists() and any(cache_dir.iterdir()):
        status["mode"] = "existing"
        csvs = sorted(cache_dir.glob("*.csv"))
        status["count"] = len(csvs)
        print(
            f"[doc_build] Dataset cache already populated at {cache_dir} "
            f"({len(csvs)} .csv files)"
        )
    else:
        status["mode"] = "clone"
        cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"[doc_build] Cloning seaborn-data repo into {cache_dir} ...")
        res = _run(
            ["git", "clone", "--depth", "1", SEABORN_DATA_REPO, str(cache_dir)]
        )
        if res.returncode != 0:
            print(
                f"[doc_build] WARNING: git clone failed (rc={res.returncode}). "
                f"Datasets will be fetched on demand from {DATASET_SOURCE_BASE}.",
                file=sys.stderr,
            )
            status["clone_failed"] = True
        csvs = sorted(cache_dir.glob("*.csv"))
        status["count"] = len(csvs)
    for p in cache_dir.glob("*.csv"):
        m.add_dataset(p.stem, cache_hit=True)
    m.add_stage("cache_datasets", "completed", **status)
    return 0


def _run_notebook_convert(input_pattern, outdir_label):
    outdir_abs = DOC_ROOT / outdir_label
    outdir_abs.mkdir(parents=True, exist_ok=True)
    (DOC_ROOT / "generated").mkdir(exist_ok=True)

    nb_tool = DOC_ROOT / "tools" / "nb_to_doc.py"
    pattern = str(DOC_ROOT / input_pattern)
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"[doc_build] No notebooks matching {pattern}")
        return 0

    for f in files:
        res = _run([sys.executable, str(nb_tool), f, str(outdir_abs)])
        if res.returncode != 0:
            raise SystemExit(res.returncode)

        stem = Path(f).stem
        files_dir = outdir_abs / f"{stem}_files"
        if outdir_label == "docstrings" and files_dir.exists():
            target = DOC_ROOT / "generated" / f"{stem}_files"
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(files_dir, target)
            gen_rst = DOC_ROOT / "generated" / f"seaborn.{stem}.rst"
            if gen_rst.exists():
                gen_rst.touch()
    return 0


def cmd_tutorials(_args):
    return _run_notebook_convert("_tutorial/*.ipynb", "tutorial")


def cmd_docstrings(_args):
    return _run_notebook_convert("_docstrings/*.ipynb", "docstrings")


def cmd_notebooks(_args):
    rc = cmd_tutorials(_args)
    if rc != 0:
        return rc
    return cmd_docstrings(_args)


def cmd_html(args):
    m = get_manifest()
    os.chdir(DOC_ROOT)
    builddir = "_build"
    doctreedir = f"{builddir}/doctrees"

    cmd = [sys.executable, "-m", "sphinx", "-b", "html", "-d", doctreedir]
    if getattr(args, "jobs", None):
        cmd += ["-j", str(args.jobs)]
    if getattr(args, "sphinx_opts", ""):
        cmd.extend(args.sphinx_opts.split())
    cmd += [".", f"{builddir}/html"]

    res = _run(cmd)
    m.add_stage(
        "sphinx_html",
        "completed" if res.returncode == 0 else "failed",
        returncode=res.returncode,
    )
    if res.returncode != 0:
        raise SystemExit(res.returncode)
    return 0


def cmd_summary(_args):
    return print_manifest_summary()


def cmd_clean(_args):
    build_dir = DOC_ROOT / "_build"
    if build_dir.exists():
        shutil.rmtree(build_dir)
        print(f"[doc_build] Removed {build_dir}")

    for sub in ("examples", "example_thumbs", "generated", "tutorial", "docstrings"):
        d = DOC_ROOT / sub
        if d.exists():
            shutil.rmtree(d)
            print(f"[doc_build] Removed {d}")

    tutorial_rst = DOC_ROOT / "tutorial.rst"
    if tutorial_rst.exists():
        tutorial_rst.unlink()
        print(f"[doc_build] Removed {tutorial_rst}")

    manifest_path = get_manifest_path(DOC_ROOT)
    if manifest_path.exists():
        manifest_path.unlink()
        print(f"[doc_build] Removed {manifest_path}")
    return 0


def cmd_all(args):
    m = get_manifest()
    m.add_stage("full_build", "started")
    rc = 0
    steps = [
        ("cache_datasets", cmd_cache_datasets),
        ("notebooks", cmd_notebooks),
        ("html", cmd_html),
    ]
    for step_name, step_fn in steps:
        rc = _stage(m, step_name, lambda s=step_fn, a=args: s(a))
        if rc != 0:
            break
    final_status = "completed" if rc == 0 else "failed"
    m.add_stage("full_build", final_status, returncode=rc)
    _ = print_manifest_summary()
    return rc


def build_parser():
    p = argparse.ArgumentParser(
        prog="doc/build.py",
        description="Unified documentation build wrapper for seaborn.",
    )

    # Shared parent parser: --jobs and --sphinx-opts can be passed to any
    # subcommand that drives sphinx or parallel builds. Using `parents` ensures
    # all these subcommands have the EXACT same parameter contract.
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument(
        "--jobs", "-j", type=int,
        help="Parallel jobs (passed to sphinx).",
    )
    shared.add_argument(
        "--sphinx-opts", default="",
        help="Extra options to pass to sphinx.",
    )

    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("cache-datasets", help="Populate SEABORN_DATA cache dir.")
    sub.add_parser(
        "tutorials",
        help="Build tutorial .ipynb -> .rst.",
        parents=[shared],
    )
    sub.add_parser(
        "docstrings",
        help="Build docstring .ipynb -> .rst.",
        parents=[shared],
    )
    sub.add_parser(
        "notebooks",
        help="tutorials + docstrings.",
        parents=[shared],
    )
    sub.add_parser(
        "html",
        help="Run sphinx html builder.",
        parents=[shared],
    )
    sub.add_parser(
        "all",
        help="cache-datasets -> notebooks -> html.",
        parents=[shared],
    )
    sub.add_parser("summary", help="Print build manifest summary.")
    sub.add_parser("clean", help="Remove all build output.")
    return p


def main():
    _ensure_env()
    m = get_manifest()

    parser = build_parser()
    args = parser.parse_args()

    dispatch = {
        "cache-datasets": cmd_cache_datasets,
        "tutorials": cmd_tutorials,
        "docstrings": cmd_docstrings,
        "notebooks": cmd_notebooks,
        "html": cmd_html,
        "all": cmd_all,
        "summary": cmd_summary,
        "clean": cmd_clean,
    }
    fn = dispatch[args.command]

    if not hasattr(args, "jobs") or args.jobs is None:
        args.jobs = None
    if not hasattr(args, "sphinx_opts"):
        args.sphinx_opts = ""

    try:
        rc = fn(args)
    finally:
        m.save()

    if rc is None:
        rc = 0
    sys.exit(rc)


if __name__ == "__main__":
    main()
