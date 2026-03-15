from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from image_picker.io_utils import default_cache_path, load_cache, save_cache, scan_images
from image_picker.scoring import ScoreConfig, assign_buckets, process_images

CALIBRATOR_FEATURES = [
    "aesthetic_score_raw",
    "quality_score_raw",
    "aesthetic_score_global_norm",
    "quality_score_global_norm",
    "final_score_global",
    "width",
    "height",
    "aspect_ratio",
    "megapixels",
    "is_raw",
]


@dataclass(frozen=True)
class CalibrationArtifacts:
    pipeline: Pipeline
    feature_names: list[str]
    metadata: dict[str, Any]


def records_to_feature_frame(records: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(records).copy()
    if frame.empty:
        return frame
    frame["filepath"] = frame["filepath"].astype(str)
    frame["is_raw"] = frame["filepath"].str.lower().str.endswith(".rw2").astype(float)
    frame["width"] = frame["width"].astype(float)
    frame["height"] = frame["height"].astype(float)
    frame["aspect_ratio"] = frame["width"] / frame["height"].replace(0, np.nan)
    frame["aspect_ratio"] = frame["aspect_ratio"].fillna(1.0)
    frame["megapixels"] = (frame["width"] * frame["height"]) / 1_000_000.0
    return frame


def build_labeled_dataset(
    positive_dir: Path,
    negative_dir: Path,
    config: ScoreConfig,
    cache_dir: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    positive_paths = scan_images(positive_dir, recursive=True)
    negative_paths = scan_images(negative_dir, recursive=True)

    if not positive_paths:
        raise RuntimeError(f"No supported images found in positive_dir: {positive_dir}")
    if not negative_paths:
        raise RuntimeError(f"No supported images found in negative_dir: {negative_dir}")

    positive_cache_path = None
    negative_cache_path = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        positive_cache_path = cache_dir / "positive.cache.json"
        negative_cache_path = cache_dir / "negative.cache.json"

    positive_records, positive_errors, positive_cache = process_images(
        positive_paths,
        cache=load_cache(positive_cache_path),
        config=config,
    )
    save_cache(positive_cache_path, positive_cache)

    negative_records, negative_errors, negative_cache = process_images(
        negative_paths,
        cache=load_cache(negative_cache_path),
        config=config,
    )
    save_cache(negative_cache_path, negative_cache)

    positive_frame = records_to_feature_frame(positive_records)
    positive_frame["label"] = 1
    negative_frame = records_to_feature_frame(negative_records)
    negative_frame["label"] = 0

    dataset = pd.concat([positive_frame, negative_frame], ignore_index=True)
    metadata = {
        "positive_dir": str(positive_dir),
        "negative_dir": str(negative_dir),
        "positive_count": int(len(positive_frame)),
        "negative_count": int(len(negative_frame)),
        "positive_errors": len(positive_errors),
        "negative_errors": len(negative_errors),
    }
    return dataset, metadata


def train_calibrator_from_dataset(dataset: pd.DataFrame) -> CalibrationArtifacts:
    if dataset.empty:
        raise RuntimeError("Training dataset is empty.")
    if dataset["label"].nunique() < 2:
        raise RuntimeError("Training dataset must contain both positive and negative samples.")

    x = dataset[CALIBRATOR_FEATURES].astype(float)
    y = dataset["label"].astype(int)

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )
    pipeline.fit(x, y)

    probabilities = pipeline.predict_proba(x)[:, 1]
    metadata = {
        "train_rows": int(len(dataset)),
        "positive_ratio": float(y.mean()),
        "roc_auc_train": float(roc_auc_score(y, probabilities)),
    }
    return CalibrationArtifacts(
        pipeline=pipeline,
        feature_names=list(CALIBRATOR_FEATURES),
        metadata=metadata,
    )


def save_calibrator(output_path: Path, artifacts: CalibrationArtifacts) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "pipeline": artifacts.pipeline,
        "feature_names": artifacts.feature_names,
        "metadata": artifacts.metadata,
    }
    with output_path.open("wb") as handle:
        pickle.dump(payload, handle)


def load_calibrator(model_path: Path) -> CalibrationArtifacts:
    with model_path.open("rb") as handle:
        payload = pickle.load(handle)
    return CalibrationArtifacts(
        pipeline=payload["pipeline"],
        feature_names=list(payload["feature_names"]),
        metadata=dict(payload["metadata"]),
    )


def apply_calibrator_to_records(
    records: list[dict[str, object]],
    artifacts: CalibrationArtifacts,
) -> list[dict[str, object]]:
    frame = records_to_feature_frame(records)
    if frame.empty:
        return records

    probabilities = artifacts.pipeline.predict_proba(
        frame[artifacts.feature_names].astype(float)
    )[:, 1]
    for index, probability in enumerate(probabilities):
        records[index]["personal_score"] = float(probability)
    personal_buckets = assign_buckets(probabilities.tolist())
    for index, bucket in enumerate(personal_buckets):
        records[index]["personal_bucket"] = bucket
    return records
