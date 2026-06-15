"""
Tests that enforce the import boundary rules for seaborn._core.

These tests ensure that:

1. Upper-layer modules (_base.py, axisgrid.py, categorical.py, etc.) and
   objects interface modules (_marks/*, _stats/*) only import from
   seaborn._core (the package entry point), not from individual
   seaborn._core.* sub-modules.

2. seaborn._core sub-modules use relative imports among themselves,
   not absolute imports from seaborn._core.

3. seaborn._core sub-modules do not reverse-import from seaborn._core
   (the package root), avoiding circular dependencies.

4. objects.py is the only module exempted from the deep-import ban,
   as it re-exports public API classes from their defining sub-modules.
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import pytest


SEABORN_ROOT = Path(__file__).resolve().parent.parent / "seaborn"

UPPER_LAYER_PATTERNS = [
    "_base.py",
    "axisgrid.py",
    "categorical.py",
    "distributions.py",
    "matrix.py",
    "miscplot.py",
    "relational.py",
    "regression.py",
    "rcmod.py",
    "utils.py",
    "palettes.py",
]

OBJECTS_INTERFACE_PATTERNS = [
    "_marks/",
    "_stats/",
]

# These modules are exempt from the deep-import ban
DEEP_IMPORT_EXEMPT = [
    "_core/__init__.py",
    "objects.py",
]

# These modules are exempt from the internal reverse-import ban
REVERSE_IMPORT_EXEMPT = [
    "_core/__init__.py",
]


def iter_seaborn_modules():
    """Yield (module_path, module_name) for all .py files in seaborn/."""
    for py_file in SEABORN_ROOT.rglob("*.py"):
        rel_path = py_file.relative_to(SEABORN_ROOT).as_posix()
        if rel_path.startswith("external/"):
            continue
        if rel_path.startswith("tests/"):
            continue
        yield py_file, rel_path


def collect_imports(py_file):
    """Parse a Python file and return all import statements."""
    with open(py_file, "r", encoding="utf-8") as f:
        source = f.read()
    tree = ast.parse(source, filename=str(py_file))

    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append((node.module, node.level))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imports.append((alias.name, 0))
    return imports


class TestCoreImportBoundaries:
    """Test suite enforcing seaborn._core import boundaries."""

    @pytest.mark.parametrize("py_file,rel_path", list(iter_seaborn_modules()))
    def test_no_deep_imports_from_core_submodules(
        self, py_file: Path, rel_path: str
    ):
        """
        Verify that upper-layer and objects-interface modules do not
        import directly from seaborn._core.* sub-modules.

        Only objects.py and _core/__init__.py are exempted.
        """
        if any(exempt in rel_path for exempt in DEEP_IMPORT_EXEMPT):
            pytest.skip(f"{rel_path} is exempt from deep-import ban")

        if rel_path.startswith("_core/") and rel_path != "_core/__init__.py":
            pytest.skip(f"{rel_path} is internal to _core")

        imports = collect_imports(py_file)

        deep_imports = []
        for module, _level in imports:
            if isinstance(module, str) and re.match(
                r"^seaborn\._core\.[a-z_]+", module
            ):
                deep_imports.append(module)

        assert not deep_imports, (
            f"{rel_path} imports directly from _core sub-module(s): "
            f"{', '.join(deep_imports)}. "
            f"Import from 'seaborn._core' (the package entry point) instead."
        )

    @pytest.mark.parametrize("py_file,rel_path", list(iter_seaborn_modules()))
    def test_core_submodules_use_relative_imports(
        self, py_file: Path, rel_path: str
    ):
        """
        Verify that seaborn._core sub-modules use relative imports
        (e.g. `from .rules import X`) rather than absolute imports
        (e.g. `from seaborn._core.rules import X`) when importing
        from other _core sub-modules.
        """
        if not rel_path.startswith("_core/") or rel_path == "_core/__init__.py":
            pytest.skip(f"{rel_path} is not a _core sub-module")

        imports = collect_imports(py_file)

        absolute_imports = []
        for module, level in imports:
            if isinstance(module, str) and module.startswith("seaborn._core."):
                absolute_imports.append(module)

        assert not absolute_imports, (
            f"{rel_path} uses absolute import(s) from other _core sub-module(s): "
            f"{', '.join(absolute_imports)}. "
            f"Use relative imports (e.g. 'from .rules import X') instead."
        )

    @pytest.mark.parametrize("py_file,rel_path", list(iter_seaborn_modules()))
    def test_core_submodules_no_reverse_import_from_root(
        self, py_file: Path, rel_path: str
    ):
        """
        Verify that seaborn._core sub-modules do not reverse-import from
        seaborn._core (the package root), which would create circular
        dependencies during package initialization.
        """
        if not rel_path.startswith("_core/") or rel_path == "_core/__init__.py":
            pytest.skip(f"{rel_path} is not a _core sub-module")

        imports = collect_imports(py_file)

        reverse_imports = []
        for module, level in imports:
            if isinstance(module, str) and module == "seaborn._core":
                reverse_imports.append(module)
            # Also check relative imports from .. (but _core is top-level)

        assert not reverse_imports, (
            f"{rel_path} reverse-imports from seaborn._core (package root): "
            f"{', '.join(reverse_imports)}. "
            f"This creates circular dependencies during package initialization. "
            f"Import from the specific sub-module using relative imports instead."
        )
