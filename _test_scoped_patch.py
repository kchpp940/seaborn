"""Tests for context-scoped load_dataset patching with clean restore."""
import os
import sys
import tempfile
import subprocess

os.environ["SEABORN_DOC_BUILD"] = "1"
os.environ["SEABORN_DOC_BASEDIR"] = tempfile.mkdtemp(prefix="seaborn-doc-test-")
os.environ["SEABORN_DATA"] = os.path.join(os.environ["SEABORN_DOC_BASEDIR"], "seaborn-data")
os.makedirs(os.environ["SEABORN_DATA"], exist_ok=True)

# Fake tips in cache
with open(os.path.join(os.environ["SEABORN_DATA"], "tips.csv"), "w") as f:
    f.write("total_bill,tip,sex,smoker,day,time,size\n16.99,1.01,Female,No,Sun,Dinner,2\n")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from seaborn._doc_config import (
    doc_example_context,
    record_image,
    get_manifest,
    init_doc_env,
)
import seaborn as sns

init_doc_env()

# --- Test 0: Before any context, load_dataset should be ORIGINAL
print("=== Test 0: No global patch after init_doc_env ===")
orig_ld = sns.load_dataset
is_patched = hasattr(orig_ld, "__wrapped__")
print(f"  sns.load_dataset has __wrapped__ (patched globally)? {is_patched}")
assert not is_patched, "load_dataset should NOT be patched globally after init_doc_env!"
print("  ✓ No global patch — init_doc_env is side-effect free")

# Clean manifest
m = get_manifest()
if m.path.exists():
    m.path.unlink()

tmpdir = tempfile.mkdtemp()
img1 = os.path.join(tmpdir, "fig1.png")
with open(img1, "wb") as f:
    f.write(b"\x89PNG\r\n\x1a\n" + b"x" * 100)

# --- Test 1: Context patches seaborn.load_dataset, restores on exit
print("\n=== Test 1: Patch + restore cycle on seaborn.load_dataset ===")
ld_before = sns.load_dataset
with doc_example_context("test:scope", source_file="t.py"):
    ld_during = sns.load_dataset
    assert ld_during is not ld_before, "Inside context, load_dataset MUST be patched"
    print(f"  Inside context: load_dataset changed ✓")
    # Also check via exec-style globals
    tips = sns.load_dataset("tips")
    print(f"  sns.load_dataset('tips') inside context: shape={tips.shape}")
    record_image(img1)

ld_after = sns.load_dataset
assert ld_after is ld_before, "After context, load_dataset MUST be restored"
print(f"  After context: load_dataset restored to original ✓")

m = get_manifest()
example = [e for e in m.data["examples"] if e["name"] == "test:scope"][0]
assert "tips" in example["datasets"], "tips must be in example datasets"
assert img1 in example["images"], "img1 must be in example images"
print(f"  Datasets in example: {example['datasets']}")
print(f"  Images in example: {len(example['images'])} unique")
print("  ✓ Patch + restore works correctly")

# --- Test 2: exec_globals injection covers `from seaborn import load_dataset`
print("\n=== Test 2: exec_globals injection for from-seaborn-import pattern ===")
exec_globals = {}
# Simulate code that does: from seaborn import load_dataset
exec("from seaborn import load_dataset", exec_globals)
# Also simulate: import seaborn as sns
exec("import seaborn as sns_alias", exec_globals)

pre_direct_ld = exec_globals["load_dataset"]
print(f"  Before context: load_dataset in globals = {type(pre_direct_ld).__name__} (original: {not hasattr(pre_direct_ld, '__wrapped__')})")

with doc_example_context("test:exec", source_file="exec.py", exec_globals=exec_globals):
    ctx_direct_ld = exec_globals["load_dataset"]
    assert ctx_direct_ld is not pre_direct_ld, "Direct load_dataset reference should be swapped in exec_globals"
    print(f"  Inside context: load_dataset in globals swapped ✓")
    # Test: call the direct one
    tips2 = exec_globals["load_dataset"]("tips")
    print(f"  load_dataset('tips') via globals: shape={tips2.shape}")
    # Test: sns alias also works (context injected sns)
    tips3 = exec_globals["sns"].load_dataset("tips")
    print(f"  sns.load_dataset via injected alias: shape={tips3.shape}")
    # Execute a script with both patterns
    source = """
tips_a = sns.load_dataset("tips")      # via module alias
tips_b = load_dataset("tips")           # direct imported name
"""
    exec(source, exec_globals)
    print("  exec() with both binding patterns worked")

# Verify restore
post_direct_ld = exec_globals.get("load_dataset")
print(f"  After context: load_dataset in globals = {type(post_direct_ld).__name__ if post_direct_ld else 'REMOVED'}")
# If there was an original, it should be restored; if not, it was added by context so should be removed
if pre_direct_ld is not None:
    assert post_direct_ld is pre_direct_ld or post_direct_ld is None, "exec_globals load_dataset must be restored or removed"
print("  ✓ exec_globals properly restored/cleaned")

m = get_manifest()
example = [e for e in m.data["examples"] if e["name"] == "test:exec"][0]
assert "tips" in example["datasets"], "tips must be tracked via exec_globals binding"
print(f"  datasets: {example['datasets']}")
print("  ✓ exec_globals injection covers all import patterns")

# --- Test 3: Exception during context still restores properly
print("\n=== Test 3: Exception path still restores load_dataset ===")
ld_before_exc = sns.load_dataset
try:
    with doc_example_context("test:exc", source_file="exc.py"):
        _ = sns.load_dataset("tips")
        raise RuntimeError("oops")
except RuntimeError:
    pass

ld_after_exc = sns.load_dataset
assert ld_after_exc is ld_before_exc, "Even on exception, load_dataset must be restored"
print("  ✓ Exception path restores correctly")

m = get_manifest()
failure = [f for f in m.data["failures"] if f["example"] == "test:exc"][0]
assert "tips" in failure["datasets"], "Failure entry should still include real dataset"
assert failure["error_type"] == "RuntimeError"
print(f"  Failure recorded: datasets={failure['datasets']}, error={failure['error_type']}")
print("  ✓ Failure diagnostics complete even on exception path")

# --- Test 4: Nested or multiple contexts don't conflict
print("\n=== Test 4: Multiple sequential contexts keep correct scoping ===")
all_ld = []
all_ld.append(sns.load_dataset)

with doc_example_context("test:seq1", source_file="s1.py"):
    all_ld.append(sns.load_dataset)
    _ = sns.load_dataset("tips")

all_ld.append(sns.load_dataset)

with doc_example_context("test:seq2", source_file="s2.py"):
    all_ld.append(sns.load_dataset)

all_ld.append(sns.load_dataset)

# Check: 0=orig, 1=patched, 2=orig, 3=patched, 4=orig
assert all_ld[0] is all_ld[2] is all_ld[4], "Outside contexts = always original"
assert all_ld[0] is not all_ld[1], "Inside first context = patched"
assert all_ld[0] is not all_ld[3], "Inside second context = patched"
print("  Sequential contexts correctly toggle patched/original ✓")
print("  ✓ Multiple contexts don't leak state")

# --- Test 5: Runtime isolation (outside doc build, no patch happens ever)
print("\n=== Test 5: No patch when SEABORN_DOC_BUILD is not set ===")
import os as _os_mod
saved_env = _os_mod.environ.get("SEABORN_DOC_BUILD")
try:
    _os_mod.environ.pop("SEABORN_DOC_BUILD", None)
    # Force _DOC_BUILDING to be re-evaluated by re-importing
    import importlib
    import seaborn._doc_config as dc_mod
    importlib.reload(dc_mod)
    from seaborn._doc_config import _DOC_BUILDING
    print(f"  _DOC_BUILDING flag = {_DOC_BUILDING}")
    assert _DOC_BUILDING is False, "_DOC_BUILDING should be False when env not set"
    import seaborn as sns2
    orig = sns2.load_dataset
    with dc_mod.doc_example_context("t", source_file="x.py"):
        during = sns2.load_dataset
        assert orig is during, "Outside doc build, no patching should happen!"
    after = sns2.load_dataset
    assert after is orig, "Restored correctly"
    print("  ✓ Outside SEABORN_DOC_BUILD=1, context is side-effect free")
finally:
    if saved_env is not None:
        _os_mod.environ["SEABORN_DOC_BUILD"] = saved_env
    importlib.reload(importlib.import_module("seaborn._doc_config"))

print("\n" + "=" * 60)
print("ALL TESTS PASSED")
