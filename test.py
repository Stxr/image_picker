from __future__ import annotations

import csv
from pathlib import Path

import torch
from aesthetic_predictor_v2_5 import convert_v2_5_from_siglip
from PIL import Image, UnidentifiedImageError

IMAGE_DIR = Path(r"D:\漫展\Capture")
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg"}
TOP_K = 20
OUTPUT_CSV = Path("aesthetic_scores.csv")


def main() -> None:
    image_paths = sorted(
        path
        for path in IMAGE_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not image_paths:
        raise SystemExit(f"No supported images found in {IMAGE_DIR}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    model, preprocessor = convert_v2_5_from_siglip(
        predictor_name_or_path=Path("aesthetic-predictor-v2-5-main/models/aesthetic_predictor_v2_5.pth"),
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model = model.to(device=device, dtype=dtype)

    results: list[tuple[Path, float]] = []

    for image_path in image_paths:
        try:
            image = Image.open(image_path).convert("RGB")
        except (OSError, UnidentifiedImageError) as exc:
            print(f"Skip {image_path.name}: {exc}")
            continue

        pixel_values = preprocessor(images=image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device=device, dtype=dtype)

        with torch.inference_mode():
            score = model(pixel_values).logits.squeeze().float().cpu().item()

        results.append((image_path, score))
        print(f"{image_path.name}\t{score:.3f}")

    results.sort(key=lambda item: item[1], reverse=True)

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["score", "filename", "path"])
        for image_path, score in results:
            writer.writerow([f"{score:.6f}", image_path.name, str(image_path)])

    print()
    print(f"Saved full results to {OUTPUT_CSV.resolve()}")
    print(f"Top {min(TOP_K, len(results))} images:")
    for image_path, score in results[:TOP_K]:
        print(f"{score:.3f}\t{image_path.name}")


if __name__ == "__main__":
    main()
