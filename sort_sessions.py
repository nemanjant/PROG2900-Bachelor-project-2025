"""
Moves collected session data from questionnaire_sessions/ into data/truthful/
and data/deceitful/, deduplicating filenames across sessions.

Usage (run from project root):
    python sort_sessions.py
    python sort_sessions.py --dry-run   # preview without copying
"""

import argparse
import json
import os
import shutil

SESSIONS_DIR = "questionnaire_sessions"
TRUTHFUL_DIR = os.path.join("data", "truthful")
DECEITFUL_DIR = os.path.join("data", "deceitful")

PART_MAP = {
    "part_1": TRUTHFUL_DIR,
    "part_2": DECEITFUL_DIR,
}


def unique_dest(dest_dir, filename):
    """Return a non-colliding destination path, appending _0, _1, … as needed."""
    base, ext = os.path.splitext(filename)
    candidate = os.path.join(dest_dir, filename)
    counter = 0
    while os.path.exists(candidate):
        candidate = os.path.join(dest_dir, f"{base}_{counter}{ext}")
        counter += 1
    return candidate


def sort_sessions(dry_run=False):
    if not os.path.isdir(SESSIONS_DIR):
        print(f"No '{SESSIONS_DIR}' directory found — nothing to do.")
        return

    for dest in [TRUTHFUL_DIR, DECEITFUL_DIR]:
        if not dry_run:
            os.makedirs(dest, exist_ok=True)

    copied = skipped = 0

    for session in sorted(os.listdir(SESSIONS_DIR)):
        session_path = os.path.join(SESSIONS_DIR, session)
        if not os.path.isdir(session_path):
            continue

        for part, dest_dir in PART_MAP.items():
            part_path = os.path.join(session_path, part)
            if not os.path.isdir(part_path):
                continue

            for fname in sorted(os.listdir(part_path)):
                if not fname.endswith(".json"):
                    continue

                src = os.path.join(part_path, fname)
                dst = unique_dest(dest_dir, fname)

                if dry_run:
                    print(f"[dry-run] {src} -> {dst}")
                else:
                    shutil.copy2(src, dst)
                    print(f"Copied: {src} -> {dst}")
                copied += 1

    label = "Would copy" if dry_run else "Copied"
    print(f"\nDone. {label} {copied} files, skipped {skipped}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sort session data into truthful/deceitful folders.")
    parser.add_argument("--dry-run", action="store_true", help="Preview without copying files.")
    args = parser.parse_args()
    sort_sessions(dry_run=args.dry_run)
