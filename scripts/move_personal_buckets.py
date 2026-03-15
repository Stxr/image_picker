from __future__ import annotations

import csv
import shutil
from pathlib import Path


def main() -> None:
    matches = list(Path("D:/").glob("*/Capture/scores.personal.csv"))
    if len(matches) != 1:
        raise SystemExit(f"Expected exactly one scores.personal.csv match, got {len(matches)}: {matches}")

    scores_csv = matches[0]
    capture_dir = scores_csv.parent

    moved = 0
    deduped = 0
    skipped = 0

    with scores_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            source = Path(str(row["filepath"]))
            bucket = str(row["personal_bucket"])
            target_dir = capture_dir / bucket
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / source.name

            if source.exists():
                if target.exists():
                    source.unlink()
                    deduped += 1
                else:
                    shutil.move(str(source), str(target))
                    moved += 1
            else:
                skipped += 1

    print(f"moved={moved} deduped={deduped} skipped={skipped}")


if __name__ == "__main__":
    main()
