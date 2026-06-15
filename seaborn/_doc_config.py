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
        deduped_images = []
        seen = set()
        for img in (images or []):
            norm = str(img)
            if norm not in seen:
                seen.add(norm)
                deduped_images.append(norm)
        deduped_datasets = list(dict.fromkeys(datasets or []))
        entry = {
            "name": example_name,
            "source_file": str(source_file) if source_file else None,
            "datasets": deduped_datasets,
            "images": deduped_images,
            "status": status,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        if error:
            entry["error"] = str(error)
        self.data["examples"].append(entry)
        for img in deduped_images:
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
def doc_example_context(example_name, dataset_names=None, source_file=None,
                        exec_globals=None):
    """Context manager wrapping a single doc example execution.

    Image recording single write point: producers call record_image() which
    populates this context's images list; when the context exits (success or
    failure) those images are flushed ONCE to the manifest via add_example().

    Dataset recording:
      1. `dataset_names` = best-effort hints from regex pre-scans of source.
      2. Context ENTERS → seaborn.load_dataset is monkey-patched AND
         if `exec_globals` is provided, any pre-existing `load_dataset`
         reference in it is also swapped. This covers both:
             import seaborn; seaborn.load_dataset("x")
             from seaborn import load_dataset; load_dataset("x")
      3. Context EXITS → original references restored, no permanent change.
    """
    global _CURRENT_CONTEXT
    m = get_manifest()
    hinted_datasets = list(dict.fromkeys(dataset_names or []))
    for ds in hinted_datasets:
        m.add_dataset(ds)

    # --- Patch phase ---
    # Only patch when actually in a doc build. Outside doc builds the context
    # manager is fully transparent (no monkey-patching, restores nothing).
    seaborn_module = None
    seaborn_orig_ld = None
    exec_globals_orig_ld = None
    patched_ld = None

    if _DOC_BUILDING:
        try:
            import seaborn as _sns
            seaborn_module = _sns
            seaborn_orig_ld = _sns.load_dataset

            def _tracked_load_dataset(name, **kwargs):
                cache_path = os.path.join(get_data_cache_path(), f"{name}.csv")
                cache_hit_before = os.path.isfile(cache_path)
                try:
                    result = seaborn_orig_ld(name, **kwargs)
                except Exception:
                    try:
                        m.add_dataset(name, cache_hit=False)
                    except Exception:
                        pass
                    raise
                try:
                    m.add_dataset(
                        name,
                        cache_hit=cache_hit_before or os.path.isfile(cache_path),
                    )
                except Exception:
                    pass
                if _CURRENT_CONTEXT:
                    ds_list = _CURRENT_CONTEXT.setdefault("datasets", [])
                    if name not in ds_list:
                        ds_list.append(name)
                return result

            patched_ld = _tracked_load_dataset
            patched_ld.__name__ = seaborn_orig_ld.__name__
            patched_ld.__wrapped__ = seaborn_orig_ld
            seaborn_module.load_dataset = patched_ld

            if exec_globals is not None:
                if "load_dataset" in exec_globals:
                    exec_globals_orig_ld = exec_globals["load_dataset"]
                exec_globals["load_dataset"] = patched_ld
                if "seaborn" not in exec_globals:
                    exec_globals["seaborn"] = seaborn_module
                if "sns" not in exec_globals:
                    exec_globals["sns"] = seaborn_module
        except Exception as patch_err:
            print(f"[doc_config] WARNING: load_dataset patch failed for "
                  f"'{example_name}': {patch_err}", file=sys.stderr)
            seaborn_orig_ld = None

    _CURRENT_CONTEXT = {
        "example": example_name,
        "source_file": source_file,
        "datasets": list(hinted_datasets),
        "backend": MPL_BACKEND,
        "seed": RANDOM_SEED,
        "image_format": IMAGE_FORMAT,
        "image_dpi": IMAGE_DPI,
        "images": [],
        "_patched_ld": patched_ld,
        "_seaborn_module": seaborn_module,
        "_seaborn_orig_ld": seaborn_orig_ld,
        "_exec_globals": exec_globals,
        "_exec_globals_orig_ld": exec_globals_orig_ld,
    }
    try:
        yield _CURRENT_CONTEXT
        final_datasets = list(dict.fromkeys(_CURRENT_CONTEXT.get("datasets", [])))
        final_images = list(dict.fromkeys(_CURRENT_CONTEXT.get("images", [])))
        m.add_example(
            example_name,
            source_file=source_file,
            datasets=final_datasets,
            images=final_images,
            status="success",
        )
    except Exception as e:
        tb_str = traceback.format_exc()
        final_datasets = list(dict.fromkeys(_CURRENT_CONTEXT.get("datasets", [])))
        final_images = list(dict.fromkeys(_CURRENT_CONTEXT.get("images", [])))
        m.add_example(
            example_name,
            source_file=source_file,
            datasets=final_datasets,
            images=final_images,
            status="failure",
            error=str(e),
        )
        m.add_failure(
            example_name,
            error_type=type(e).__name__,
            error_message=str(e),
            traceback_str=tb_str,
            source_file=source_file,
            datasets=final_datasets,
            images=final_images,
        )
        _report_failure()
        raise
    finally:
        # --- Unpatch phase: restore all originals ---
        ctx = _CURRENT_CONTEXT
        _CURRENT_CONTEXT = {}
        if ctx.get("_seaborn_module") is not None and ctx.get("_seaborn_orig_ld") is not None:
            try:
                ctx["_seaborn_module"].load_dataset = ctx["_seaborn_orig_ld"]
            except Exception:
                pass
        eg = ctx.get("_exec_globals")
        if eg is not None:
            try:
                if "load_dataset" in eg:
                    orig = ctx.get("_exec_globals_orig_ld")
                    if orig is not None:
                        eg["load_dataset"] = orig
                    else:
                        eg.pop("load_dataset", None)
            except Exception:
                pass


def record_image(image_path):
    """Register an image produced by the currently executing example.

    Images are only recorded ONCE per example — they are stored in the
    active doc_example_context and flushed to the manifest ONLY when the
    context exits (success or failure). This is the single write point for
    image recording.
    """
    if not _CURRENT_CONTEXT:
        return
    img_str = str(image_path)
    imgs = _CURRENT_CONTEXT.setdefault("images", [])
    if img_str not in imgs:
        imgs.append(img_str)


def record_dataset(name, cache_hit=None):
    """Record a dataset access against the manifest.

    NOTE: During doc_example_context, prefer the context-scope patched
    load_dataset() which binds to REAL calls inside that example. Use this
    only for pre-declaration (e.g. from regex scans when the actual code
    path may not run, or for external tracking).
    """
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
