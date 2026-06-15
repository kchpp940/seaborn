import os
import sys
import json
import atexit
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

_DOC_BUILDING = os.environ.get("SEABORN_DOC_BUILD", "").lower() in ("1", "true", "yes")

MANIFEST_PATH_ENV = "SEABORN_DOC_MANIFEST"
DEFAULT_MANIFEST_FILENAME = "_build/doc_build_manifest.json"


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

DATASET_SOURCE_BASE = os.environ.get(
    "SEABORN_DATASET_SOURCE",
    "https://raw.githubusercontent.com/mwaskom/seaborn-data/master",
)


def get_manifest_path(base_dir=None):
    explicit = os.environ.get(MANIFEST_PATH_ENV)
    if explicit:
        return Path(explicit)
    if base_dir is None:
        base_dir = os.environ.get("SEABORN_DOC_BASEDIR", Path.cwd())
    return Path(base_dir) / DEFAULT_MANIFEST_FILENAME


class BuildManifest:
    def __init__(self, path=None):
        self.path = Path(path) if path else get_manifest_path()
        self.data = {
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "config": {
                "random_seed": RANDOM_SEED,
                "mpl_backend": MPL_BACKEND,
                "image_format": IMAGE_FORMAT,
                "image_dpi": IMAGE_DPI,
                "savefig_bbox": SAVEFIG_BBOX,
                "nb_exec_timeout": NB_EXEC_TIMEOUT,
                "nb_kernel": NB_KERNEL,
                "data_cache_path": get_data_cache_path(),
                "dataset_source_base": DATASET_SOURCE_BASE,
            },
            "stages": [],
            "examples": [],
            "datasets": {},
            "images": [],
            "cache_hits": [],
            "cache_misses": [],
            "failures": [],
        }
        self._loaded = False

    def load(self):
        if self._loaded:
            return
        if self.path.exists():
            try:
                with open(self.path) as f:
                    existing = json.load(f)
                existing.setdefault("stages", [])
                existing.setdefault("examples", [])
                existing.setdefault("datasets", {})
                existing.setdefault("images", [])
                existing.setdefault("cache_hits", [])
                existing.setdefault("cache_misses", [])
                existing.setdefault("failures", [])
                self.data = existing
            except Exception:
                pass
        self._loaded = True
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def save(self):
        self.data["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "w") as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[doc_manifest] WARNING: could not write {self.path}: {e}",
                  file=sys.stderr)

    def add_stage(self, name, status="started", **extra):
        self.load()
        entry = {
            "name": name,
            "status": status,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        entry.update(extra)
        self.data["stages"].append(entry)
        self.save()

    def add_example(self, example_name, source_file=None, datasets=None,
                    images=None, status="success", error=None):
        self.load()
        entry = {
            "name": example_name,
            "source_file": str(source_file) if source_file else None,
            "datasets": list(datasets or []),
            "images": list(images or []),
            "status": status,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        if error:
            entry["error"] = str(error)
        self.data["examples"].append(entry)
        if images:
            for img in images:
                self.add_image(img, example_name)
        self.save()

    def add_dataset(self, name, cache_hit=None, source_url=None):
        self.load()
        cache_path = os.path.join(get_data_cache_path(), f"{name}.csv")
        entry = self.data["datasets"].get(name, {})
        entry.setdefault("name", name)
        entry["cache_path"] = cache_path
        if source_url is None:
            source_url = f"{DATASET_SOURCE_BASE}/{name}.csv"
        entry["source_url"] = source_url
        if cache_hit is None:
            cache_hit = os.path.isfile(cache_path)
        entry["cache_hit"] = cache_hit
        entry["last_accessed"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.data["datasets"][name] = entry
        if cache_hit:
            if name not in self.data["cache_hits"]:
                self.data["cache_hits"].append(name)
        else:
            if name not in self.data["cache_misses"]:
                self.data["cache_misses"].append(name)
        self.save()

    def add_image(self, image_path, produced_by=None):
        self.load()
        try:
            p = Path(image_path)
            abs_path = str(p.resolve()) if p.is_absolute() else str(image_path)
        except Exception:
            abs_path = str(image_path)
        entry = {
            "path": abs_path,
            "produced_by": produced_by,
            "exists": os.path.isfile(image_path),
            "size_bytes": os.path.getsize(image_path) if os.path.isfile(image_path) else None,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        self.data["images"].append(entry)
        self.save()

    def add_failure(self, example_name, error_type, error_message, traceback_str,
                    source_file=None, datasets=None, images=None):
        self.load()
        entry = {
            "example": example_name,
            "error_type": error_type,
            "error_message": error_message,
            "traceback": traceback_str,
            "source_file": str(source_file) if source_file else None,
            "datasets": list(datasets or []),
            "images": list(images or []),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        self.data["failures"].append(entry)
        self.save()


_manifest_singleton = None


def get_manifest():
    global _manifest_singleton
    if _manifest_singleton is None:
        _manifest_singleton = BuildManifest()
        atexit.register(_manifest_singleton.save)
    return _manifest_singleton


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


def init_doc_env():
    os.environ.setdefault("SEABORN_DOC_BUILD", "1")
    os.environ.setdefault("SEABORN_DATA", get_data_cache_path())
    os.environ.setdefault("MPLBACKEND", MPL_BACKEND)
    os.environ.setdefault("SEABORN_DOC_MPL_BACKEND", MPL_BACKEND)
    os.environ.setdefault("SEABORN_DOC_RANDOM_SEED", str(RANDOM_SEED))
    os.environ.setdefault("SEABORN_DOC_IMAGE_FORMAT", IMAGE_FORMAT)
    os.environ.setdefault("SEABORN_DOC_IMAGE_DPI", str(IMAGE_DPI))
    os.environ.setdefault("SEABORN_DOC_SAVEFIG_BBOX", SAVEFIG_BBOX)
    os.environ.setdefault("SEABORN_DOC_NB_TIMEOUT", str(NB_EXEC_TIMEOUT))
    os.environ.setdefault("NB_KERNEL", NB_KERNEL)


_CURRENT_CONTEXT = {}


@contextmanager
def doc_example_context(example_name, dataset_names=None, source_file=None):
    global _CURRENT_CONTEXT
    m = get_manifest()
    datasets = list(dataset_names or [])
    for ds in datasets:
        m.add_dataset(ds)

    _CURRENT_CONTEXT = {
        "example": example_name,
        "source_file": source_file,
        "datasets": datasets,
        "backend": MPL_BACKEND,
        "seed": RANDOM_SEED,
        "image_format": IMAGE_FORMAT,
        "image_dpi": IMAGE_DPI,
        "images": [],
    }
    try:
        yield _CURRENT_CONTEXT
        m.add_example(
            example_name,
            source_file=source_file,
            datasets=datasets,
            images=_CURRENT_CONTEXT.get("images", []),
            status="success",
        )
    except Exception as e:
        tb_str = traceback.format_exc()
        m.add_example(
            example_name,
            source_file=source_file,
            datasets=datasets,
            images=_CURRENT_CONTEXT.get("images", []),
            status="failure",
            error=str(e),
        )
        m.add_failure(
            example_name,
            error_type=type(e).__name__,
            error_message=str(e),
            traceback_str=tb_str,
            source_file=source_file,
            datasets=datasets,
            images=_CURRENT_CONTEXT.get("images", []),
        )
        _report_failure()
        raise
    finally:
        _CURRENT_CONTEXT = {}


def record_image(image_path):
    if _CURRENT_CONTEXT:
        _CURRENT_CONTEXT.setdefault("images", []).append(str(image_path))
    try:
        m = get_manifest()
        m.add_image(image_path, produced_by=_CURRENT_CONTEXT.get("example") if _CURRENT_CONTEXT else None)
    except Exception:
        pass


def record_dataset(name, cache_hit=None):
    try:
        m = get_manifest()
        m.add_dataset(name, cache_hit=cache_hit)
    except Exception:
        pass


def _report_failure():
    ctx = _CURRENT_CONTEXT
    if not ctx:
        return
    m = get_manifest()
    parts = [
        "=" * 60,
        "SEABORN DOC BUILD FAILURE DIAGNOSTICS",
        "=" * 60,
        f"  Example:       {ctx.get('example', '<unknown>')}",
    ]
    if ctx.get("source_file"):
        parts.append(f"  Source file:   {ctx['source_file']}")
    parts.append(f"  Datasets:      {', '.join(ctx.get('datasets', [])) or '<none detected>'}")
    parts.append(f"  MPL backend:   {ctx.get('backend', '<unknown>')}")
    parts.append(f"  Random seed:   {ctx.get('seed', '<unknown>')}")
    parts.append(f"  Image format:  {ctx.get('image_format', '<unknown>')}")
    parts.append(f"  Image DPI:     {ctx.get('image_dpi', '<unknown>')}")

    images = ctx.get("images", [])
    if images:
        parts.append(f"  Output images:")
        for img in images:
            exists = "EXISTS" if os.path.isfile(img) else "MISSING"
            sz = os.path.getsize(img) if os.path.isfile(img) else 0
            parts.append(f"    [{exists}] {img} ({sz} bytes)")
    else:
        parts.append("  Output images: <none recorded>")

    data_cache = get_data_cache_path()
    parts.append(f"  Data cache:    {data_cache}")
    parts.append(f"  Manifest:      {m.path}")
    if os.path.isdir(data_cache):
        cached = [f for f in os.listdir(data_cache) if f.endswith(".csv")]
        missing_ds = [d for d in ctx.get("datasets", []) if f"{d}.csv" not in cached]
        if missing_ds:
            parts.append(f"  Missing cache: {', '.join(missing_ds)}")
            for md in missing_ds:
                parts.append(f"    -> expected: {os.path.join(data_cache, md + '.csv')}")
                parts.append(f"    -> source:   {DATASET_SOURCE_BASE}/{md}.csv")
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


def print_manifest_summary(path=None):
    if path is None:
        path = get_manifest_path()
    p = Path(path)
    if not p.exists():
        print(f"No manifest found at {p}")
        return 1
    with open(p) as f:
        data = json.load(f)
    total = len(data.get("examples", []))
    successes = [e for e in data.get("examples", []) if e.get("status") == "success"]
    failures = data.get("failures", [])
    datasets = data.get("datasets", {})
    cache_hits = data.get("cache_hits", [])
    cache_misses = data.get("cache_misses", [])
    images = data.get("images", [])

    lines = [
        "=" * 60,
        "SEABORN DOC BUILD MANIFEST SUMMARY",
        "=" * 60,
        f"Manifest file:    {p}",
        f"Started at:       {data.get('started_at', '<unknown>')}",
        f"Finished at:      {data.get('finished_at', '<still running>')}",
        "",
        "-- Config --",
    ]
    cfg = data.get("config", {})
    for k, v in cfg.items():
        lines.append(f"  {k:24s} = {v}")
    lines.extend([
        "",
        "-- Stages --",
    ])
    for s in data.get("stages", []):
        lines.append(f"  [{s.get('status')}] {s.get('name')} ({s.get('timestamp')})")
    lines.extend([
        "",
        "-- Examples --",
        f"  Total:           {total}",
        f"  Success:         {len(successes)}",
        f"  Failures:        {len(failures)}",
    ])
    if failures:
        lines.append("  Failed examples:")
        for fl in failures:
            lines.append(f"    - {fl.get('example')}: {fl.get('error_type')}: {fl.get('error_message')[:120]}")
    lines.extend([
        "",
        "-- Datasets --",
        f"  Referenced:      {len(datasets)}",
        f"  Cache hits:      {len(cache_hits)} ({', '.join(cache_hits[:10])}{'...' if len(cache_hits) > 10 else ''})",
        f"  Cache misses:    {len(cache_misses)} ({', '.join(cache_misses) or 'none'})",
        "",
        "-- Images --",
        f"  Produced:        {len(images)}",
    ])
    existing_imgs = [i for i in images if i.get("exists")]
    total_bytes = sum(i.get("size_bytes") or 0 for i in existing_imgs)
    lines.append(f"  Existing:        {len(existing_imgs)} ({total_bytes} bytes)")
    if failures:
        lines.extend([
            "",
            "-- Failures (detail) --",
        ])
        for fl in failures:
            lines.extend([
                "",
                f"*** {fl.get('example')} ***",
                f"  Source:    {fl.get('source_file')}",
                f"  Datasets:  {', '.join(fl.get('datasets', [])) or '<none>'}",
                f"  Images:    {', '.join(fl.get('images', [])) or '<none>'}",
                f"  Error:     {fl.get('error_type')}: {fl.get('error_message')}",
                f"  Traceback:",
            ])
            for tb_line in fl.get("traceback", "").splitlines()[:40]:
                lines.append(f"    {tb_line}")
    lines.append("=" * 60)
    print("\n".join(lines))
    return 0 if not failures else 2
