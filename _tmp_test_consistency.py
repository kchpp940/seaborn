import sys
import shutil
from pathlib import Path

sys.path.insert(0, 'ci')
from check_extras_consistency import (
    check_consistency,
    fix_pyproject,
    load_extras,
    _find_extra_block,
)

PYPROJECT = Path('pyproject.toml')
BACKUP = Path('/tmp/pyproject_test_bak.toml')

print("=== Test 1: check mode on clean file (should pass) ===")
extras = load_extras()
errors = check_consistency(extras)
assert len(errors) == 0, f"Expected 0 errors, got {len(errors)}: {errors}"
print("PASS")

print("\n=== Test 2: find_extra_block works ===")
text = PYPROJECT.read_text()
for name in ["test", "lint", "devtools", "docs", "stats", "dev", "build"]:
    pos = _find_extra_block(text, name)
    assert pos is not None, f"Could not find {name!r} block"
    assert pos[0] < pos[1], f"Invalid range for {name!r}: {pos}"
print("PASS (all 7 extras found)")

print("\n=== Test 3: corrupt test extra, check detects drift ===")
shutil.copy(PYPROJECT, BACKUP)
text = text.replace('    "pytest-xdist",', '    "pytest-xdist",\n    "coverage",')
PYPROJECT.write_text(text)
extras = load_extras()
errors = check_consistency(extras)
assert len(errors) >= 1, f"Expected errors, got {len(errors)}"
print(f"PASS ({len(errors)} errors detected)")
print(f"  First error preview: {errors[0].split(chr(10))[0]}")

print("\n=== Test 4: --fix auto-repairs dev and build ===")
extras = load_extras()
rc = fix_pyproject(extras)
assert rc == 0, f"fix_pyproject returned {rc}"
extras_new = load_extras()
assert "coverage" in extras_new["test"], "coverage should still be in test"
assert "coverage" in extras_new["dev"], "coverage should have been propagated to dev"
assert "coverage" in extras_new["build"], "coverage should have been propagated to build"
print("PASS (coverage propagated to dev and build)")

print("\n=== Test 5: TOML still valid after fix ===")
import tomllib
tomllib.loads(PYPROJECT.read_text())
print("PASS")

shutil.copy(BACKUP, PYPROJECT)
print("\n=== All tests passed, original file restored ===")
