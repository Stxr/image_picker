from __future__ import annotations

import csv
import shutil
from pathlib import Path


def main() -> None:
    capture_dir = Path("D:/新疆/Capture")
    scores_csv = capture_dir / "scores.personal.csv"
    log_path = Path("C:/Users/txr/Desktop/project/image_picker/artifacts/move_xinjiang_log.txt")

    moved = 0
    deduped = 0
    skipped = 0
    processed = 0

    with scores_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            processed += 1
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

    message = f"processed={processed} moved={moved} deduped={deduped} skipped={skipped}\n"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(message, encoding="utf-8")
    print(message, end="")


if __name__ == "__main__":
    main()
