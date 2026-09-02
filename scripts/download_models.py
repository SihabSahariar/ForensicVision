"""Command-line model weight installer.

The GUI's Model Manager is the normal route. This script exists for headless
installs, CI images and air-gapped staging, and applies the same policy: it
lists licences and sources, never downloads without being asked, and verifies
digests where upstream publishes them.

Usage::

    python scripts/download_models.py --list
    python scripts/download_models.py --install realesrgan_x4plus
    python scripts/download_models.py --install-task super_resolution
    python scripts/download_models.py --verify
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.constants import MODEL_STATUS_LABELS, ModelKind, ModelStatus  # noqa: E402
from app.logging_setup import configure_logging  # noqa: E402
from app.paths import weights_dir  # noqa: E402
from restoration import register_all_models  # noqa: E402
from restoration.base import ModelInfo, WeightSpec  # noqa: E402
from restoration.registry import ModelRegistry  # noqa: E402
from restoration.weights import (  # noqa: E402
    WeightInstallError,
    download_weight,
    installed_size,
    probe_url,
    verify_weight,
)


def _primary_spec(info: ModelInfo) -> Optional[WeightSpec]:
    """Return the first required weight specification."""
    for spec in info.weights:
        if spec.required:
            return spec
    return None


def list_models(show_all: bool = True) -> int:
    """Print the model table."""
    rows = ModelRegistry.status_table()
    print(f"{'NAME':30s} {'TASK':22s} {'KIND':10s} {'SIZE':>9s}  STATUS")
    print("-" * 92)
    for row in rows:
        info: ModelInfo = row["info"]
        if not show_all and info.kind == ModelKind.CLASSICAL.value:
            continue
        spec = _primary_spec(info)
        size = spec.size_human() if spec else "-"
        print(
            f"{info.name:30s} {info.task_label[:21]:22s} "
            f"{info.kind:10s} {size:>9s}  {row['status_label']}"
        )
    print("-" * 92)
    ready = sum(1 for r in rows if r["status"] == ModelStatus.INSTALLED.value)
    print(f"{ready} of {len(rows)} models ready")
    print(f"weights folder: {weights_dir()}  "
          f"({installed_size() / (1024 * 1024):.0f} MiB)")
    return 0


def describe(name: str) -> int:
    """Print the full record for one model."""
    info = ModelRegistry.info(name)
    if info is None:
        print(f"No model named '{name}'. Use --list to see the available names.")
        return 1
    model = ModelRegistry.try_get(name)
    state = model.availability() if model else None

    print(f"{info.display_name}  ({info.name})")
    print("=" * 72)
    print(f"Task       : {info.task_label}")
    print(f"Kind       : {info.kind}")
    print(f"Version    : {info.version}")
    print(f"Status     : {state.label if state else 'unknown'}")
    if state and state.reason:
        print(f"Detail     : {state.reason}")
    print(f"Authors    : {info.authors or 'n/a'}")
    print(f"Licence    : {info.license_name or 'n/a'}")
    print(f"Repository : {info.repository or 'n/a'}")
    print(f"Paper      : {info.paper or 'n/a'}")
    print(f"Synthesises: {'YES' if info.may_synthesise else 'no'}")
    print()
    print("Description:")
    print(f"  {info.description}")
    print()
    print("What it does to the pixels:")
    print(f"  {info.method}")
    if info.notes:
        print()
        print(f"Notes: {info.notes}")
    for spec in info.weights:
        path = weights_dir() / spec.filename
        print()
        print(f"Weight file: {spec.filename}")
        print(f"  installed : {'yes' if path.is_file() else 'no'}")
        print(f"  size      : {spec.size_human()}")
        print(f"  licence   : {spec.license_name or 'see model licence'}")
        print(f"  url       : {spec.url or '(no direct download published)'}")
        print(f"  source    : {spec.source or 'n/a'}")
        print(f"  sha256    : {spec.sha256 or '(not published upstream)'}")
    return 0


def install(names: List[str], assume_yes: bool = False) -> int:
    """Download the named models' weights after showing their terms."""
    failures = 0
    for name in names:
        info = ModelRegistry.info(name)
        if info is None:
            print(f"[skip] no model named '{name}'")
            failures += 1
            continue

        spec = _primary_spec(info)
        if spec is None:
            print(f"[skip] {info.display_name} needs no weight file")
            continue
        if (weights_dir() / spec.filename).is_file():
            print(f"[ok]   {info.display_name} is already installed")
            continue
        if not spec.url:
            print(
                f"[skip] {info.display_name} has no direct download URL.\n"
                f"       Obtain '{spec.filename}' from {spec.source}\n"
                f"       and place it in {weights_dir()}"
            )
            failures += 1
            continue

        print(f"\n{info.display_name}")
        print(f"  file    : {spec.filename} ({spec.size_human()})")
        print(f"  licence : {spec.license_name or info.license_name}")
        print(f"  source  : {spec.url}")
        if spec.sha256:
            print(f"  sha256  : {spec.sha256}  (will be verified)")
        else:
            print("  sha256  : not published upstream - cannot be verified")

        if not assume_yes:
            answer = input("  Download? [y/N] ").strip().lower()
            if answer not in ("y", "yes"):
                print("  skipped")
                continue

        reachable, _, message = probe_url(spec.url)
        if not reachable:
            print(f"  [error] source unreachable: {message}")
            failures += 1
            continue

        state = {"last": -1}

        def progress(done: int, total: int) -> None:
            if total <= 0:
                return
            percent = int(done * 100 / total)
            if percent >= state["last"] + 5:
                state["last"] = percent
                print(
                    f"\r  {percent:3d}%  "
                    f"{done / (1024 * 1024):.1f} / {total / (1024 * 1024):.1f} MiB",
                    end="", flush=True,
                )

        try:
            result = download_weight(spec, progress=progress)
            print(f"\r  installed: {result.summary()}")
        except WeightInstallError as exc:
            print(f"\r  [error] {exc}")
            failures += 1

    return 1 if failures else 0


def verify_all() -> int:
    """Re-hash every installed weight file."""
    problems = 0
    for info in ModelRegistry.infos():
        for spec in info.weights:
            if not (weights_dir() / spec.filename).is_file():
                continue
            ok, message = verify_weight(spec)
            print(f"[{'ok ' if ok else 'FAIL'}] {message}")
            if not ok:
                problems += 1
    if problems:
        print(f"\n{problems} file(s) failed verification.")
    return 1 if problems else 0


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="List every model")
    parser.add_argument(
        "--neural-only", action="store_true",
        help="With --list, hide the classical operators",
    )
    parser.add_argument("--describe", metavar="NAME", help="Show one model in full")
    parser.add_argument(
        "--install", metavar="NAME", nargs="+", help="Install these models"
    )
    parser.add_argument(
        "--install-task", metavar="TASK",
        help="Install the preferred model for a task (e.g. super_resolution)",
    )
    parser.add_argument("--verify", action="store_true", help="Verify installed weights")
    parser.add_argument(
        "-y", "--yes", action="store_true", help="Do not prompt for confirmation"
    )
    args = parser.parse_args(argv)

    configure_logging(console=False)
    register_all_models()

    if args.describe:
        return describe(args.describe)
    if args.install:
        return install(args.install, args.yes)
    if args.install_task:
        names = [i.name for i in ModelRegistry.by_task(args.install_task)
                 if i.kind == ModelKind.NEURAL.value]
        if not names:
            print(f"No neural model is registered for task '{args.install_task}'.")
            return 1
        return install(names[:1], args.yes)
    if args.verify:
        return verify_all()
    return list_models(show_all=not args.neural_only)


if __name__ == "__main__":
    raise SystemExit(main())
