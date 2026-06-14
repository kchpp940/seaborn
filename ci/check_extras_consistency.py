from __future__ import annotations

import argparse
import re
import sys
import tomllib
from pathlib import Path


COMPOSITIONS: dict[str, list[str]] = {
    "dev": ["test", "lint", "devtools"],
    "build": ["stats", "dev", "docs"],
}

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT_PATH = ROOT / "pyproject.toml"


def load_extras() -> dict[str, list[str]]:
    data = tomllib.loads(PYPROJECT_PATH.read_text())
    return data["project"]["optional-dependencies"]


def expand_packages(extras: dict[str, list[str]], name: str, seen: set[str] | None = None) -> set[str]:
    if seen is None:
        seen = set()
    if name in seen:
        return set()
    seen.add(name)
    packages: set[str] = set()
    for entry in extras.get(name, []):
        packages.add(entry)
    return packages


def expected_packages_for(extras: dict[str, list[str]], components: list[str]) -> set[str]:
    result: set[str] = set()
    for comp in components:
        result |= expand_packages(extras, comp)
    return result


def _sort_key(pkg: str) -> str:
    return pkg.lower().lstrip("!<>=")


def check_consistency(extras: dict[str, list[str]]) -> list[str]:
    errors: list[str] = []
    for name, components in COMPOSITIONS.items():
        if name not in extras:
            errors.append(f"Missing extra: {name!r}")
            continue
        expected = expected_packages_for(extras, components)
        actual = set(extras[name])
        if actual != expected:
            missing = expected - actual
            extra = actual - expected
            msg_parts = [f"Extra {name!r} is out of sync with its components {components}:"]
            if missing:
                msg_parts.append(f"  Missing: {sorted(missing, key=_sort_key)}")
            if extra:
                msg_parts.append(f"  Unexpected: {sorted(extra, key=_sort_key)}")
            errors.append("\n".join(msg_parts))
    return errors


def _find_extra_block(text: str, name: str) -> tuple[int, int] | None:
    pattern = re.compile(
        "^" + re.escape(name) + r" = \[" "\n"
        r"((?:    \"[^\"]*\",?\n)+)"
        r"\]",
        re.MULTILINE,
    )
    m = pattern.search(text)
    if m:
        return m.start(), m.end()
    return None


def _format_extra_block(name: str, packages: list[str]) -> str:
    lines = [f"{name} = ["]
    for pkg in packages:
        lines.append(f'    "{pkg}",')
    lines.append("]")
    return "\n".join(lines)


def fix_pyproject(extras: dict[str, list[str]]) -> int:
    text = PYPROJECT_PATH.read_text()
    changes = 0

    for name, components in COMPOSITIONS.items():
        expected = sorted(expected_packages_for(extras, components), key=_sort_key)
        pos = _find_extra_block(text, name)
        if pos is None:
            print(f"ERROR: Could not locate extra block for {name!r} in pyproject.toml")
            return 1
        start, end = pos
        new_block = _format_extra_block(name, expected)
        old_block = text[start:end]
        if old_block != new_block:
            text = text[:start] + new_block + text[end:]
            changes += 1
            print(f"Updated extra {name!r} synced with {components}")

    if changes:
        PYPROJECT_PATH.write_text(text)
        print(f"\n{changes} extra(s) updated in pyproject.toml")
    else:
        print("All composition extras already in sync.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check or fix consistency of composition extras in pyproject.toml"
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Rewrite composition extras (dev, build) to match their atomic components",
    )
    args = parser.parse_args()

    extras = load_extras()

    if args.fix:
        return fix_pyproject(extras)

    errors = check_consistency(extras)
    if errors:
        print("ERROR: Composition extras are out of sync with atomic extras:\n")
        for e in errors:
            print(f"  {e}\n")
        print(
            "Run `python3 ci/check_extras_consistency.py --fix` to automatically update them.\n"
            "Remember: add/modify packages in ONE place — the atomic extras (test/lint/devtools/stats/docs.\n"
            "Never edit dev or build directly; they are generated compositions."
        )
        return 1

    print("OK: All composition extras are in sync with their atomic components.")
    for name, components in COMPOSITIONS.items():
        pkgs = expand_packages(extras, name)
        print(f"  {name}: {len(components)} components -> {len(pkgs)} packages")
    return 0


if __name__ == "__main__":
    sys.exit(main())
