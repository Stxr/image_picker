import pickle
from pathlib import Path

import pandas as pd

from image_picker.calibrator import (
    CALIBRATOR_FEATURES,
    apply_calibrator_to_records,
    records_to_feature_frame,
    train_calibrator_from_dataset,
)


def test_records_to_feature_frame_adds_derived_columns() -> None:
    frame = records_to_feature_frame(
        [
            {
                "filepath": "a.RW2",
                "width": 6000,
                "height": 4000,
                "aesthetic_score_raw": 5.0,
                "quality_score_raw": 0.5,
                "aesthetic_score_global_norm": 0.4,
                "quality_score_global_norm": 0.5,
                "final_score_global": 0.44,
            }
        ]
    )
    assert frame.loc[0, "is_raw"] == 1.0
    assert frame.loc[0, "megapixels"] == 24.0
    assert frame.loc[0, "aspect_ratio"] == 1.5


def test_train_and_apply_calibrator() -> None:
    dataset = pd.DataFrame(
        [
            {
                "filepath": "good1.jpg",
                "width": 6000,
                "height": 4000,
                "aesthetic_score_raw": 6.0,
                "quality_score_raw": 0.8,
                "aesthetic_score_global_norm": 0.6,
                "quality_score_global_norm": 0.8,
                "final_score_global": 0.68,
                "label": 1,
                "aspect_ratio": 1.5,
                "megapixels": 24.0,
                "is_raw": 0.0,
            },
            {
                "filepath": "good2.jpg",
                "width": 6000,
                "height": 4000,
                "aesthetic_score_raw": 6.5,
                "quality_score_raw": 0.85,
                "aesthetic_score_global_norm": 0.65,
                "quality_score_global_norm": 0.85,
                "final_score_global": 0.73,
                "label": 1,
                "aspect_ratio": 1.5,
                "megapixels": 24.0,
                "is_raw": 0.0,
            },
            {
                "filepath": "bad1.rw2",
                "width": 6000,
                "height": 4000,
                "aesthetic_score_raw": 4.0,
                "quality_score_raw": 0.3,
                "aesthetic_score_global_norm": 0.33,
                "quality_score_global_norm": 0.3,
                "final_score_global": 0.318,
                "label": 0,
                "aspect_ratio": 1.5,
                "megapixels": 24.0,
                "is_raw": 1.0,
            },
            {
                "filepath": "bad2.rw2",
                "width": 6000,
                "height": 4000,
                "aesthetic_score_raw": 4.3,
                "quality_score_raw": 0.35,
                "aesthetic_score_global_norm": 0.37,
                "quality_score_global_norm": 0.35,
                "final_score_global": 0.362,
                "label": 0,
                "aspect_ratio": 1.5,
                "megapixels": 24.0,
                "is_raw": 1.0,
            },
        ]
    )
    artifacts = train_calibrator_from_dataset(dataset)
    assert artifacts.metadata["train_rows"] == 4
    assert CALIBRATOR_FEATURES == artifacts.feature_names

    records = dataset.drop(columns=["label"]).to_dict(orient="records")
    updated = apply_calibrator_to_records(records, artifacts)
    assert "personal_score" in updated[0]
    assert "personal_bucket" in updated[0]
    assert updated[0]["personal_score"] > updated[-1]["personal_score"]
