from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from image_picker.io_utils import ImageInfo, load_image_info, load_rgb_image, make_cache_key
from image_picker.vlm import CommentRequest, build_comment_generator

AESTHETIC_HEAD_URL = (
    "https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/"
    "sac%2Blogos%2Bava1-l14-linearMSE.pth"
)


@dataclass(frozen=True)
class ScoreConfig:
    device: str = "cuda"
    batch_size: int = 8
    workers: int = 0
    quality_metric: str = "topiq_nr"
    aesthetic_model_name: str = "ViT-L-14"
    aesthetic_pretrained: str = "openai"
    max_image_side: int = 2048
    enable_vlm_comments: bool = False
    comment_limit: int = 50
    aesthetic_global_min: float = 1.0
    aesthetic_global_max: float = 10.0
    quality_global_min: float = 0.0
    quality_global_max: float = 1.0


@dataclass(frozen=True)
class ScoreRow:
    filepath: str
    filename: str
    width: int
    height: int
    aesthetic_score_raw: float
    quality_score_raw: float


def ensure_device(device: str) -> str:
    if device == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return device


class AestheticPredictor(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.layers = model

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)


def build_aesthetic_predictor(state_dict: dict[str, torch.Tensor]) -> AestheticPredictor:
    if "layers.0.weight" in state_dict:
        model = torch.nn.Sequential(
            torch.nn.Linear(768, 1024),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(1024, 128),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, 64),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(64, 16),
            torch.nn.Linear(16, 1),
        )
        predictor = AestheticPredictor(model)
        predictor.load_state_dict(state_dict)
        return predictor

    if "head.weight" in state_dict:
        input_size = int(state_dict["head.weight"].shape[1])
        model = torch.nn.Linear(input_size, 1)
        predictor = AestheticPredictor(model)
        remapped = {
            key.replace("head.", "layers."): value for key, value in state_dict.items()
        }
        predictor.load_state_dict(remapped)
        return predictor

    raise RuntimeError("Unsupported aesthetic predictor checkpoint structure.")


class ModelBundle:
    def __init__(self, config: ScoreConfig) -> None:
        self.device = ensure_device(config.device)
        self.max_image_side = config.max_image_side
        if self.device == "cpu" and config.workers > 0:
            torch.set_num_threads(config.workers)
        self.aesthetic_model = self._load_aesthetic_model(config)
        self.quality_metric = self._load_quality_metric(config)

    def _load_aesthetic_model(self, config: ScoreConfig):
        try:
            import open_clip
        except ImportError as exc:
            raise RuntimeError(
                "open-clip-torch is required for aesthetic scoring."
            ) from exc

        model, _, preprocess = open_clip.create_model_and_transforms(
            config.aesthetic_model_name,
            pretrained=config.aesthetic_pretrained,
            device=self.device,
        )
        model.eval()

        weights_path = Path(torch.hub.get_dir()) / "checkpoints" / "laion-aesthetic-head.pth"
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        if not weights_path.exists():
            torch.hub.download_url_to_file(AESTHETIC_HEAD_URL, str(weights_path), progress=True)
        state_dict = torch.load(weights_path, map_location=self.device)
        predictor = build_aesthetic_predictor(state_dict)
        predictor.to(self.device)
        predictor.eval()
        return (model, preprocess, predictor)

    def _load_quality_metric(self, config: ScoreConfig):
        try:
            import pyiqa
        except ImportError as exc:
            raise RuntimeError("pyiqa is required for quality scoring.") from exc
        return pyiqa.create_metric(config.quality_metric, device=self.device)

    def score_image(self, info: ImageInfo) -> ScoreRow:
        return self.score_batch([info])[0]

    def score_batch(self, infos: list[ImageInfo]) -> list[ScoreRow]:
        if not infos:
            return []

        rgb_images: list[Image.Image] = []
        for info in infos:
            image, _ = load_rgb_image(info.path, max_image_side=self.max_image_side)
            rgb_images.append(image)

        aesthetic_scores = self._score_aesthetic_batch(rgb_images)
        quality_scores = self._score_quality_batch(rgb_images)
        rows: list[ScoreRow] = []
        for index, info in enumerate(infos):
            rows.append(
                ScoreRow(
                    filepath=str(info.path.resolve()),
                    filename=info.filename,
                    width=info.width,
                    height=info.height,
                    aesthetic_score_raw=aesthetic_scores[index],
                    quality_score_raw=quality_scores[index],
                )
            )
        return rows

    def _score_aesthetic_batch(self, images: list[Image.Image]) -> list[float]:
        clip_model, preprocess, predictor = self.aesthetic_model
        image_tensor = torch.stack([preprocess(image) for image in images]).to(self.device)
        with torch.inference_mode():
            features = clip_model.encode_image(image_tensor)
            features = torch.nn.functional.normalize(features, dim=-1)
            scores = predictor(features)
        return [float(score) for score in scores.squeeze(-1).detach().cpu().tolist()]

    def _score_quality_batch(self, images: list[Image.Image]) -> list[float]:
        tensors = []
        for image in images:
            array = np.asarray(image, dtype=np.float32) / 255.0
            tensors.append(torch.from_numpy(array).permute(2, 0, 1))
        batch = torch.stack(tensors).to(self.device)
        with torch.inference_mode():
            scores = self.quality_metric(batch)
        if isinstance(scores, torch.Tensor):
            flat_scores = scores.reshape(-1).detach().cpu().tolist()
            return [float(score) for score in flat_scores]
        return [float(score) for score in scores]


def normalize_series(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return array
    min_value = np.min(array)
    max_value = np.max(array)
    if np.isclose(min_value, max_value):
        return np.full_like(array, 0.5, dtype=np.float64)
    return (array - min_value) / (max_value - min_value)


def normalize_with_fixed_range(
    values: Iterable[float],
    min_value: float,
    max_value: float,
) -> np.ndarray:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return array
    if np.isclose(min_value, max_value):
        return np.full_like(array, 0.5, dtype=np.float64)
    normalized = (array - min_value) / (max_value - min_value)
    return np.clip(normalized, 0.0, 1.0)


def assign_buckets(final_scores: Iterable[float]) -> list[str]:
    series = pd.Series(list(final_scores), dtype="float64")
    if series.empty:
        return []
    low_threshold = float(series.quantile(0.2))
    high_threshold = float(series.quantile(0.8))
    buckets: list[str] = []
    for value in series:
        if value >= high_threshold:
            buckets.append("top")
        elif value <= low_threshold:
            buckets.append("low")
        else:
            buckets.append("mid")
    return buckets


def merge_scores(rows: list[ScoreRow], config: ScoreConfig) -> list[dict[str, object]]:
    if not rows:
        return []
    aesthetic_norm = normalize_series(row.aesthetic_score_raw for row in rows)
    quality_norm = normalize_series(row.quality_score_raw for row in rows)
    aesthetic_global = normalize_with_fixed_range(
        (row.aesthetic_score_raw for row in rows),
        min_value=config.aesthetic_global_min,
        max_value=config.aesthetic_global_max,
    )
    quality_global = normalize_with_fixed_range(
        (row.quality_score_raw for row in rows),
        min_value=config.quality_global_min,
        max_value=config.quality_global_max,
    )
    final_scores_batch = (0.6 * aesthetic_norm) + (0.4 * quality_norm)
    final_scores_global = (0.6 * aesthetic_global) + (0.4 * quality_global)
    buckets = assign_buckets(final_scores_batch.tolist())

    merged: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        merged.append(
            {
                "filepath": row.filepath,
                "filename": row.filename,
                "width": row.width,
                "height": row.height,
                "aesthetic_score_raw": row.aesthetic_score_raw,
                "quality_score_raw": row.quality_score_raw,
                "aesthetic_score_norm": float(aesthetic_norm[index]),
                "quality_score_norm": float(quality_norm[index]),
                "aesthetic_score_global_norm": float(aesthetic_global[index]),
                "quality_score_global_norm": float(quality_global[index]),
                "final_score": float(final_scores_batch[index]),
                "final_score_batch": float(final_scores_batch[index]),
                "final_score_global": float(final_scores_global[index]),
                "bucket": buckets[index],
                "feedback": "",
                "comment": "",
            }
        )
    return merged


def annotate_comments(
    records: list[dict[str, object]],
    enabled: bool,
    comment_limit: int,
) -> list[dict[str, object]]:
    generator = build_comment_generator(enabled)
    if not enabled or comment_limit <= 0:
        return records

    ordered = sorted(
        range(len(records)),
        key=lambda idx: abs(float(records[idx]["final_score"]) - 0.5),
    )
    for idx in ordered[:comment_limit]:
        request = CommentRequest(
            image_path=Path(str(records[idx]["filepath"])),
            final_score=float(records[idx]["final_score"]),
            bucket=str(records[idx]["bucket"]),
        )
        records[idx]["comment"] = generator.generate(request)
    return records


def process_images(
    image_paths: list[Path],
    cache: dict[str, dict[str, object]],
    config: ScoreConfig,
    progress: bool = True,
) -> tuple[list[dict[str, object]], list[dict[str, str]], dict[str, dict[str, object]]]:
    model_bundle: ModelBundle | None = None
    rows: list[ScoreRow] = []
    errors: list[dict[str, str]] = []
    cache_entries = dict(cache)
    uncached_infos: list[ImageInfo] = []

    iterator = tqdm(image_paths, desc="Scoring images") if progress else image_paths
    for image_path in iterator:
        try:
            info = load_image_info(image_path)
            cache_key = make_cache_key(info)
            cached = cache_entries.get(cache_key)
            if cached is not None:
                rows.append(
                    ScoreRow(
                        filepath=str(cached["filepath"]),
                        filename=str(cached["filename"]),
                        width=int(cached["width"]),
                        height=int(cached["height"]),
                        aesthetic_score_raw=float(cached["aesthetic_score_raw"]),
                        quality_score_raw=float(cached["quality_score_raw"]),
                    )
                )
                continue

            uncached_infos.append(info)
            if len(uncached_infos) >= config.batch_size:
                batch_rows, model_bundle = _score_uncached_batch(
                    uncached_infos,
                    cache_entries,
                    config,
                    errors,
                    model_bundle,
                )
                rows.extend(batch_rows)
                uncached_infos = []
        except Exception as exc:  # pragma: no cover - intentional runtime guard
            errors.append({"filepath": str(image_path.resolve()), "error": str(exc)})

    if uncached_infos:
        batch_rows, model_bundle = _score_uncached_batch(
            uncached_infos,
            cache_entries,
            config,
            errors,
            model_bundle,
        )
        rows.extend(batch_rows)

    records = merge_scores(rows, config)
    records = annotate_comments(records, config.enable_vlm_comments, config.comment_limit)
    return records, errors, cache_entries


def _score_uncached_batch(
    infos: list[ImageInfo],
    cache_entries: dict[str, dict[str, object]],
    config: ScoreConfig,
    errors: list[dict[str, str]],
    model_bundle: ModelBundle | None,
) -> tuple[list[ScoreRow], ModelBundle]:
    if not infos:
        return [], model_bundle or ModelBundle(config)

    bundle = model_bundle or ModelBundle(config)
    try:
        rows = bundle.score_batch(infos)
    except Exception:
        rows = []
        for info in infos:
            try:
                rows.append(bundle.score_image(info))
            except Exception as exc:  # pragma: no cover - intentional runtime guard
                errors.append({"filepath": str(info.path.resolve()), "error": str(exc)})

    for info, row in zip(infos, rows):
        cache_entries[make_cache_key(info)] = {
            "filepath": row.filepath,
            "filename": row.filename,
            "width": row.width,
            "height": row.height,
            "aesthetic_score_raw": row.aesthetic_score_raw,
            "quality_score_raw": row.quality_score_raw,
        }
    return rows, bundle
