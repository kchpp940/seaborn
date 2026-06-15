import os
import sys
import traceback
from contextlib import contextmanager

_DOC_BUILDING = os.environ.get("SEABORN_DOC_BUILD", "").lower() in ("1", "true", "yes")


def get_data_cache_path():
    return os.environ.get(
        "SEABORN_DATA",
        os.path.join(os.path.expanduser("~"), ".cache", "seaborn-data"),
    )


RANDOM_SEED = int(os.environ.get("SEABORN_DOC_RANDOM_SEED", "0"))

MPL_BACKEND = os.environ.get("SEABORN_DOC_MPL_BACKEND", "Agg")

IMAGE_FORMAT = os.environ.get("SEABORN_DOC_IMAGE_FORMAT", "png")

IMAGE_DPI = int(os.environ.get("SEABORN_DOC_IMAGE_DPI", "90"))

SAVEFIG_BBOX = os.environ.get("SEABORN_DOC_SAVEFIG_BBOX", "tight")

NB_EXEC_TIMEOUT = int(os.environ.get("SEABORN_DOC_NB_TIMEOUT", "600"))

NB_KERNEL = os.environ.get("NB_KERNEL", "python")


def apply_random_seed():
    import numpy as np

    np.random.seed(RANDOM_SEED)
    try:
        import random

        random.seed(RANDOM_SEED)
    except Exception:
        pass


def apply_mpl_backend():
    import matplotlib

    matplotlib.use(MPL_BACKEND)


def apply_mpl_rc():
    import matplotlib as mpl

    mpl.rcParams["savefig.bbox"] = SAVEFIG_BBOX
    mpl.rcParams["savefig.dpi"] = IMAGE_DPI


_CURRENT_CONTEXT = {}


@contextmanager
def doc_example_context(example_name, dataset_names=None):
    global _CURRENT_CONTEXT
    _CURRENT_CONTEXT = {
        "example": example_name,
        "datasets": dataset_names or [],
        "backend": MPL_BACKEND,
        "seed": RANDOM_SEED,
        "image_format": IMAGE_FORMAT,
        "image_dpi": IMAGE_DPI,
    }
    try:
        yield _CURRENT_CONTEXT
    except Exception:
        _report_failure()
        raise
    finally:
        _CURRENT_CONTEXT = {}


def _report_failure():
    ctx = _CURRENT_CONTEXT
    if not ctx:
        return
    parts = [
        "=" * 60,
        "SEABORN DOC BUILD FAILURE DIAGNOSTICS",
        "=" * 60,
        f"  Example:       {ctx.get('example', '<unknown>')}",
        f"  Datasets:      {', '.join(ctx.get('datasets', [])) or '<none detected>'}",
        f"  MPL backend:   {ctx.get('backend', '<unknown>')}",
        f"  Random seed:   {ctx.get('seed', '<unknown>')}",
        f"  Image format:  {ctx.get('image_format', '<unknown>')}",
        f"  Image DPI:     {ctx.get('image_dpi', '<unknown>')}",
    ]
    data_cache = get_data_cache_path()
    parts.append(f"  Data cache:    {data_cache}")
    if os.path.isdir(data_cache):
        cached = [
            f
            for f in os.listdir(data_cache)
            if f.endswith(".csv")
        ]
        parts.append(f"  Cached data:   {', '.join(sorted(cached)[:10])}")
        if len(cached) > 10:
            parts.append(f"                  ... and {len(cached) - 10} more")
    else:
        parts.append("  Cached data:   <cache directory missing>")
    parts.append("-" * 60)
    parts.append(traceback.format_exc().strip())
    parts.append("=" * 60)
    msg = "\n".join(parts)
    print(msg, file=sys.stderr)


def extract_dataset_names(source_text):
    import re

    return sorted(set(re.findall(r"load_dataset\(['\"](\w+)['\"]", source_text)))
