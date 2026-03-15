from pathlib import Path

from PIL import Image

from image_picker.io_utils import (
    copy_bucket_files,
    load_image_info,
    load_rgb_image,
    make_cache_key,
    resize_for_scoring,
    scan_images,
)


def test_scan_images_supports_common_extensions(tmp_path: Path) -> None:
    image = Image.new("RGB", (8, 6), color="white")
    image.save(tmp_path / "a.JPG")
    image.save(tmp_path / "b.png")
    image.save(tmp_path / "nested.webp")
    (tmp_path / "raw.RW2").write_bytes(b"raw")
    (tmp_path / "notes.txt").write_text("x", encoding="utf-8")

    found = scan_images(tmp_path, recursive=False)
    assert [path.name for path in found] == ["a.JPG", "b.png", "nested.webp", "raw.RW2"]


def test_load_image_info_applies_exif_transpose(monkeypatch, tmp_path: Path) -> None:
    image_path = tmp_path / "photo.jpg"
    Image.new("RGB", (10, 20), color="white").save(image_path)

    def fake_exif_transpose(image):
        return image.rotate(90, expand=True)

    monkeypatch.setattr("image_picker.io_utils.ImageOps.exif_transpose", fake_exif_transpose)
    info = load_image_info(image_path)
    assert (info.width, info.height) == (20, 10)


def test_cache_key_changes_when_file_changes(tmp_path: Path) -> None:
    image_path = tmp_path / "photo.jpg"
    Image.new("RGB", (4, 4), color="white").save(image_path)

    info_a = load_image_info(image_path)
    key_a = make_cache_key(info_a)

    Image.new("RGB", (6, 6), color="black").save(image_path)
    info_b = load_image_info(image_path)
    key_b = make_cache_key(info_b)

    assert key_a != key_b


def test_copy_bucket_files_groups_by_bucket(tmp_path: Path) -> None:
    image_a = tmp_path / "a.jpg"
    image_b = tmp_path / "b.jpg"
    Image.new("RGB", (4, 4), color="white").save(image_a)
    Image.new("RGB", (4, 4), color="black").save(image_b)

    csv_path = tmp_path / "scores.csv"
    csv_path.write_text(
        "filepath,bucket\n"
        f"{image_a},top\n"
        f"{image_b},low\n",
        encoding="utf-8",
    )

    destination = tmp_path / "out"
    copied, skipped = copy_bucket_files(csv_path, destination)

    assert copied == 2
    assert skipped == 0
    assert (destination / "top" / "a.jpg").exists()
    assert (destination / "low" / "b.jpg").exists()


def test_copy_bucket_files_can_use_custom_bucket_and_move(tmp_path: Path) -> None:
    image_a = tmp_path / "a.jpg"
    image_b = tmp_path / "b.jpg"
    Image.new("RGB", (4, 4), color="white").save(image_a)
    Image.new("RGB", (4, 4), color="black").save(image_b)

    csv_path = tmp_path / "scores.csv"
    csv_path.write_text(
        "filepath,personal_bucket\n"
        f"{image_a},top\n"
        f"{image_b},low\n",
        encoding="utf-8",
    )

    destination = tmp_path / "out"
    copied, skipped = copy_bucket_files(
        csv_path,
        destination,
        bucket_column="personal_bucket",
        move_files=True,
    )

    assert copied == 2
    assert skipped == 0
    assert (destination / "top" / "a.jpg").exists()
    assert (destination / "low" / "b.jpg").exists()
    assert not image_a.exists()
    assert not image_b.exists()


def test_resize_for_scoring_limits_longest_side() -> None:
    image = Image.new("RGB", (6000, 4000), color="white")
    resized = resize_for_scoring(image, max_image_side=2048)
    assert resized.size == (2048, 1365)


def test_load_rgb_image_resizes_jpg(tmp_path: Path) -> None:
    image_path = tmp_path / "photo.jpg"
    Image.new("RGB", (4000, 3000), color="white").save(image_path)

    image, original_size = load_rgb_image(image_path, max_image_side=2000)
    assert original_size == (4000, 3000)
    assert image.size == (2000, 1500)
