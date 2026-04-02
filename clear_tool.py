import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from utils.paths import PROJECT_ROOT


@dataclass
class CleanupStats:
    deleted: int = 0
    failed: int = 0


def _iter_files(base_dir: Path) -> Iterable[Path]:
    if not base_dir.exists():
        return []
    return (path for path in base_dir.rglob("*") if path.is_file() or path.is_symlink())


def _safe_unlink(path: Path, dry_run: bool) -> bool:
    if dry_run:
        return True
    try:
        path.unlink()
        return True
    except Exception:
        return False


def _clean_by_suffixes(base_dir: Path, suffixes: tuple[str, ...], dry_run: bool) -> CleanupStats:
    stats = CleanupStats()
    for file_path in _iter_files(base_dir):
        if not file_path.name.lower().endswith(suffixes):
            continue
        if _safe_unlink(file_path, dry_run):
            stats.deleted += 1
        else:
            stats.failed += 1
    return stats


def _clean_python_cache(dry_run: bool) -> CleanupStats:
    stats = CleanupStats()
    cache_dirs = [
        p for p in PROJECT_ROOT.rglob("__pycache__") if p.is_dir()
    ]

    for cache_dir in cache_dirs:
        for file_path in _iter_files(cache_dir):
            if _safe_unlink(file_path, dry_run):
                stats.deleted += 1
            else:
                stats.failed += 1
        try:
            if not dry_run and cache_dir.exists():
                cache_dir.rmdir()
        except Exception:
            pass
    return stats


def run_cleanup(dry_run: bool) -> dict[str, CleanupStats]:
    return {
        "tmp_images": _clean_by_suffixes(
            PROJECT_ROOT / "tmp_imgs",
            (".png", ".jpg", ".jpeg", ".webp", ".gif", ".html", ".svg"),
            dry_run,
        ),
        "tmp_images_alt": _clean_by_suffixes(
            PROJECT_ROOT / "tmp_img",
            (".png", ".jpg", ".jpeg", ".webp", ".gif", ".html", ".svg"),
            dry_run,
        ),
        "tmp_jpg": _clean_by_suffixes(
            PROJECT_ROOT / "tmp_jpg",
            (".png", ".jpg", ".jpeg", ".webp", ".gif", ".html", ".svg"),
            dry_run,
        ),
        "runtime_log": _clean_by_suffixes(
            PROJECT_ROOT,
            (".log",),
            dry_run,
        ),
        "exports": _clean_by_suffixes(
            PROJECT_ROOT / "exports",
            (".csv", ".txt", ".json", ".png", ".jpg", ".jpeg", ".html"),
            dry_run,
        ),
        "python_cache": _clean_python_cache(dry_run),
    }


def _print_summary(results: dict[str, CleanupStats], dry_run: bool) -> None:
    mode = "DRY-RUN" if dry_run else "EXECUTE"
    print(f"Cleanup mode: {mode}")

    total_deleted = 0
    total_failed = 0
    for name, stats in results.items():
        total_deleted += stats.deleted
        total_failed += stats.failed
        print(f"[{name}] deleted={stats.deleted}, failed={stats.failed}")

    print(f"Total: deleted={total_deleted}, failed={total_failed}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean temporary and generated files")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be cleaned without deleting files",
    )
    args = parser.parse_args()

    results = run_cleanup(dry_run=args.dry_run)
    _print_summary(results, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
