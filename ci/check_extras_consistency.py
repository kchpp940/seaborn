from __future__ import annotations

import argparse
import email
import re
import subprocess
import sys
import tempfile
import tomllib
import zipfile
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


def _build_wheel(dist_dir: Path) -> Path:
    dist_dir.mkdir(parents=True, exist_ok=True)
    existing = list(dist_dir.glob("seaborn-*.whl"))
    for w in existing:
        w.unlink()

    result = subprocess.run(
        [sys.executable, "-m", "pip", "wheel", "--no-deps", "--wheel-dir", str(dist_dir), "."],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print("ERROR: Failed to build wheel.")
        print(result.stderr)
        sys.exit(1)

    wheels = list(dist_dir.glob("seaborn-*.whl"))
    if len(wheels) != 1:
        print(f"ERROR: Expected exactly 1 wheel, found {len(wheels)}: {wheels}")
        sys.exit(1)
    return wheels[0]


def _extract_metadata_from_wheel(wheel_path: Path) -> email.message.Message:
    with zipfile.ZipFile(wheel_path) as zf:
        metadata_names = [n for n in zf.namelist() if n.endswith(".dist-info/METADATA")]
        if len(metadata_names) != 1:
            raise RuntimeError(f"Expected 1 METADATA, found {len(metadata_names)}")
        with zf.open(metadata_names[0]) as f:
            return email.message_from_binary_file(f)


def _parse_provides_extras(metadata: email.message.Message) -> list[str]:
    return sorted(metadata.get_all("Provides-Extra", []))


def _parse_requires_dist(metadata: email.message.Message) -> list[tuple[str, str | None]]:
    """Returns list of (package_spec, extra_marker) tuples."""
    result: list[tuple[str, str | None]] = []
    for req in metadata.get_all("Requires-Dist", []):
        extra = None
        if ";" in req:
            spec_part, marker_part = req.split(";", 1)
            marker = marker_part.strip()
            m = re.search(r"extra\s*==\s*['\"]([^'\"]+)['\"]", marker)
            if m:
                extra = m.group(1)
            spec_part = spec_part.strip()
        else:
            spec_part = req.strip()
        result.append((spec_part, extra))
    return result


def check_wheel_metadata() -> int:
    errors: list[str] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        dist_dir = Path(tmpdir) / "dist"
        wheel_path = _build_wheel(dist_dir)
        print(f"Built wheel: {wheel_path.name}")

        metadata = _extract_metadata_from_wheel(wheel_path)
        extras = _parse_provides_extras(metadata)
        print(f"Provides-Extra: {extras}")

        requires_dist = _parse_requires_dist(metadata)
        self_refs = [
            (spec, extra) for spec, extra in requires_dist
            if spec.lower().startswith("seaborn")
        ]
        if self_refs:
            errors.append(
                "Wheel metadata contains self-referencing dependencies (seaborn depending on seaborn):\n"
                + "\n".join(f"  - {spec} (extra={extra!r})" for spec, extra in self_refs)
            )

        source_extras = set(load_extras().keys())
        wheel_extras = set(extras)
        if source_extras != wheel_extras:
            missing = source_extras - wheel_extras
            extra_wheel = wheel_extras - source_extras
            msg = ["Wheel extras do not match pyproject.toml extras:"]
            if missing:
                msg.append(f"  Missing in wheel: {sorted(missing)}")
            if extra_wheel:
                msg.append(f"  Unexpected in wheel: {sorted(extra_wheel)}")
            errors.append("\n".join(msg))

        for extra_name in extras:
            wheel_pkgs = {spec for spec, e in requires_dist if e == extra_name}
            source_pkgs = set(load_extras().get(extra_name, []))
            if wheel_pkgs != source_pkgs:
                missing = source_pkgs - wheel_pkgs
                extra_wheel = wheel_pkgs - source_pkgs
                msg = [f"Extra {extra_name!r} mismatch between wheel and pyproject.toml:"]
                if missing:
                    msg.append(f"  Missing in wheel: {sorted(missing)}")
                if extra_wheel:
                    msg.append(f"  Unexpected in wheel: {sorted(extra_wheel)}")
                errors.append("\n".join(msg))

    if errors:
        print("\nERROR: Wheel metadata validation failed:\n")
        for e in errors:
            print(f"  {e}\n")
        return 1

    print("OK: Wheel metadata is clean — no self-references and all extras match pyproject.toml.")
    total = sum(1 for _, e in requires_dist if e is not None)
    print(f"  {len(requires_dist)} Requires-Dist entries ({total} with extra markers)")
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
    parser.add_argument(
        "--check-wheel",
        action="store_true",
        help="Build a wheel and validate its metadata (no self-refs, extras match pyproject.toml)",
    )
    args = parser.parse_args()

    if args.check_wheel:
        return check_wheel_metadata()

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
            "Remember: add/modify packages in ONE place — the atomic extras (test/lint/devtools/stats/docs).\n"
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
