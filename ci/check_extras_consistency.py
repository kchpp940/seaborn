from __future__ import annotations

import sys
import tomllib
from pathlib import Path


EXPECTED_COMPOSITIONS: dict[str, list[str]] = {
    "dev": ["test", "lint", "devtools"],
    "build": ["stats", "dev", "docs"],
}

ROOT = Path(__file__).resolve().parent.parent


def load_extras() -> dict[str, list[str]]:
    pyproject = tomllib.load((ROOT / "pyproject.toml").open("rb"))
    return pyproject["project"]["optional-dependencies"]


def _parse_self_ref(entry: str) -> list[str] | None:
    prefix = "seaborn["
    suffix = "]"
    entry = entry.strip()
    if not (entry.startswith(prefix) and entry.endswith(suffix)):
        return None
    inner = entry[len(prefix) : -len(suffix)]
    return [e.strip() for e in inner.split(",") if e.strip()]


def check_composition_uses_self_ref(
    extras: dict[str, list[str]], name: str, parts: list[str]
) -> list[str]:
    errors: list[str] = []
    if name not in extras:
        errors.append(f"Missing extra: {name!r}")
        return errors

    entries = extras[name]
    parsed = [_parse_self_ref(e) for e in entries]

    if len(entries) != 1 or parsed[0] is None:
        errors.append(
            f"Extra {name!r} must be a single self-reference "
            f"(e.g. 'seaborn[{','.join(parts)}]'), not a manual package list."
        )
        return errors

    if sorted(parsed[0]) != sorted(parts):
        errors.append(
            f"Extra {name!r} should reference {sorted(parts)!r}, "
            f"got {sorted(parsed[0])!r}."
        )

    return errors


def expand_extras(extras: dict[str, list[str]], name: str, seen: set[str] | None = None) -> set[str]:
    if seen is None:
        seen = set()
    if name in seen:
        return set()
    seen.add(name)
    packages: set[str] = set()
    for entry in extras.get(name, []):
        refs = _parse_self_ref(entry)
        if refs is not None:
            for ref in refs:
                packages |= expand_extras(extras, ref, seen)
        else:
            packages.add(entry)
    return packages


def check_no_duplicate_maintenance(extras: dict[str, list[str]]) -> list[str]:
    errors: list[str] = []
    for name, parts in EXPECTED_COMPOSITIONS.items():
        composed_pkgs = set()
        for p in parts:
            composed_pkgs |= expand_extras(extras, p)

        if name in extras:
            actual_pkgs = expand_extras(extras, name)
            if actual_pkgs != composed_pkgs:
                missing = composed_pkgs - actual_pkgs
                extra = actual_pkgs - composed_pkgs
                msg = [f"Package mismatch between {name!r} and its composition {parts}:"]
                if missing:
                    msg.append(f"  Missing in {name!r}: {sorted(missing)}")
                if extra:
                    msg.append(f"  Unexpected in {name!r}: {sorted(extra)}")
                errors.append("\n".join(msg))
    return errors


def main() -> int:
    extras = load_extras()
    errors: list[str] = []

    for name, parts in EXPECTED_COMPOSITIONS.items():
        errors.extend(check_composition_uses_self_ref(extras, name, parts))

    errors.extend(check_no_duplicate_maintenance(extras))

    if errors:
        print("ERROR: pyproject.toml extras consistency check failed:\n")
        for e in errors:
            print(f"  - {e}\n")
        print(
            "HINT: Composition extras (dev/build) must use self-references like "
            "'seaborn[test,lint,devtools]' so that package lists are maintained "
            "in ONE place only — the atomic extras."
        )
        return 1

    print("OK: All composition extras use self-references and resolve consistently.")
    for name, parts in EXPECTED_COMPOSITIONS.items():
        pkgs = expand_extras(extras, name)
        print(f"  {name}: {len(parts)} atomic extras -> {len(pkgs)} packages total")
    return 0


if __name__ == "__main__":
    sys.exit(main())
