from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, ImageOps

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".rw2",
    ".webp",
    ".tif",
    ".tiff",
    ".bmp",
}


@dataclass(frozen=True)
class ImageInfo:
    path: Path
    width: int
    height: int
    size_bytes: int
    mtime_ns: int

    @property
    def filename(self) -> str:
        return self.path.name


def scan_images(input_dir: Path, recursive: bool) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    return sorted(
        path
        for path in input_dir.glob(pattern)
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def load_image_info(path: Path) -> ImageInfo:
    stat = path.stat()
    width, height = get_image_dimensions(path)
    return ImageInfo(
        path=path,
        width=width,
        height=height,
        size_bytes=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
    )


def make_cache_key(info: ImageInfo) -> str:
    payload = f"{info.path.resolve()}|{info.size_bytes}|{info.mtime_ns}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def default_cache_path(output_csv: Path) -> Path:
    return output_csv.with_suffix(".cache.json")


def default_errors_path(output_csv: Path) -> Path:
    return output_csv.with_name(f"{output_csv.stem}.errors.csv")


def load_cache(cache_path: Path | None) -> dict[str, dict[str, Any]]:
    if cache_path is None or not cache_path.exists():
        return {}
    data = json.loads(cache_path.read_text(encoding="utf-8"))
    entries = data.get("entries", {})
    if isinstance(entries, dict):
        return entries
    return {}


def save_cache(cache_path: Path | None, entries: dict[str, dict[str, Any]]) -> None:
    if cache_path is None:
        return
    payload = {"version": 1, "entries": entries}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def write_results_csv(output_csv: Path, records: list[dict[str, Any]]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(output_csv, index=False)


def write_errors_csv(errors_csv: Path, errors: list[dict[str, Any]]) -> None:
    if not errors:
        return
    errors_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(errors).to_csv(errors_csv, index=False)


def get_image_dimensions(path: Path) -> tuple[int, int]:
    suffix = path.suffix.lower()
    if suffix == ".rw2":
        try:
            import rawpy
        except ImportError as exc:
            raise RuntimeError("rawpy is required to read RW2 files.") from exc
        with rawpy.imread(str(path)) as raw:
            return int(raw.sizes.width), int(raw.sizes.height)

    with Image.open(path) as image:
        normalized = ImageOps.exif_transpose(image)
        return normalized.size


def load_rgb_image(path: Path, max_image_side: int = 0) -> tuple[Image.Image, tuple[int, int]]:
    suffix = path.suffix.lower()
    if suffix == ".rw2":
        image = load_raw_image(path)
    else:
        with Image.open(path) as pil_image:
            image = ImageOps.exif_transpose(pil_image).convert("RGB")

    original_size = image.size
    if max_image_side > 0:
        image = resize_for_scoring(image, max_image_side=max_image_side)
    return image, original_size


def load_raw_image(path: Path) -> Image.Image:
    try:
        import rawpy
    except ImportError as exc:
        raise RuntimeError("rawpy is required to read RW2 files.") from exc

    with rawpy.imread(str(path)) as raw:
        rgb = raw.postprocess(
            use_camera_wb=True,
            no_auto_bright=True,
            output_bps=8,
            demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,
        )
    return Image.fromarray(rgb).convert("RGB")


def resize_for_scoring(image: Image.Image, max_image_side: int) -> Image.Image:
    width, height = image.size
    longest_side = max(width, height)
    if longest_side <= max_image_side:
        return image

    scale = max_image_side / float(longest_side)
    target_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return image.resize(target_size, Image.Resampling.LANCZOS)


def copy_bucket_files(
    csv_path: Path,
    destination_dir: Path,
    overwrite: bool = False,
    bucket_column: str = "bucket",
    move_files: bool = False,
) -> tuple[int, int]:
    frame = pd.read_csv(csv_path)
    copied = 0
    skipped = 0
    destination_dir.mkdir(parents=True, exist_ok=True)

    for row in frame.to_dict(orient="records"):
        source = Path(str(row["filepath"]))
        bucket = str(row.get(bucket_column, "unbucketed"))
        target_dir = destination_dir / bucket
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / source.name

        if target_path.exists() and not overwrite:
            skipped += 1
            continue

        if move_files:
            shutil.move(str(source), str(target_path))
        else:
            shutil.copy2(source, target_path)
        copied += 1

    return copied, skipped
