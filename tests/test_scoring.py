from pathlib import Path

from image_picker.scoring import ScoreConfig, ScoreRow, annotate_comments, assign_buckets, merge_scores, process_images


def test_assign_buckets_uses_quantiles() -> None:
    buckets = assign_buckets([0.1, 0.2, 0.3, 0.4, 0.9])
    assert buckets == ["low", "mid", "mid", "mid", "top"]


def test_merge_scores_keeps_formula_stable() -> None:
    rows = [
        ScoreRow("a", "a.jpg", 10, 10, 1.0, 2.0),
        ScoreRow("b", "b.jpg", 10, 10, 3.0, 1.0),
    ]
    merged = merge_scores(rows, ScoreConfig(device="cpu"))
    assert merged[0]["final_score"] == 0.4
    assert merged[1]["final_score"] == 0.6
    assert merged[0]["final_score_batch"] == 0.4
    assert merged[1]["final_score_batch"] == 0.6


def test_merge_scores_adds_global_score_with_fixed_ranges() -> None:
    rows = [
        ScoreRow("a", "a.jpg", 10, 10, 5.5, 0.25),
        ScoreRow("b", "b.jpg", 10, 10, 7.0, 0.75),
    ]
    merged = merge_scores(rows, ScoreConfig(device="cpu"))
    assert merged[0]["aesthetic_score_global_norm"] == 0.5
    assert merged[0]["quality_score_global_norm"] == 0.25
    assert merged[0]["final_score_global"] == 0.4
    assert round(merged[1]["final_score_global"], 6) == 0.7


def test_annotate_comments_does_not_change_scores_when_stub_enabled() -> None:
    rows = [
        {
            "filepath": "a.jpg",
            "filename": "a.jpg",
            "width": 1,
            "height": 1,
            "aesthetic_score_raw": 1.0,
            "quality_score_raw": 1.0,
            "aesthetic_score_norm": 1.0,
            "quality_score_norm": 1.0,
            "final_score": 0.7,
            "bucket": "top",
            "feedback": "",
            "comment": "",
        }
    ]
    updated = annotate_comments(rows, enabled=True, comment_limit=1)
    assert updated[0]["final_score"] == 0.7
    assert updated[0]["comment"] == ""


def test_process_images_uses_cache_without_loading_models(monkeypatch, tmp_path: Path) -> None:
    image_path = tmp_path / "cached.jpg"
    image_path.write_bytes(b"placeholder")

    class FakeInfo:
        path = image_path
        width = 10
        height = 20
        size_bytes = 123
        mtime_ns = 999
        filename = "cached.jpg"

    monkeypatch.setattr("image_picker.scoring.load_image_info", lambda _: FakeInfo)
    monkeypatch.setattr("image_picker.scoring.make_cache_key", lambda _: "cache-key")

    def fail_model_init(_):
        raise AssertionError("ModelBundle should not be initialized when cache is valid")

    monkeypatch.setattr("image_picker.scoring.ModelBundle", fail_model_init)

    records, errors, cache_entries = process_images(
        [image_path],
        cache={
            "cache-key": {
                "filepath": str(image_path),
                "filename": "cached.jpg",
                "width": 10,
                "height": 20,
                "aesthetic_score_raw": 2.0,
                "quality_score_raw": 3.0,
            }
        },
        config=ScoreConfig(device="cpu"),
        progress=False,
    )

    assert not errors
    assert records[0]["filename"] == "cached.jpg"
    assert "cache-key" in cache_entries
