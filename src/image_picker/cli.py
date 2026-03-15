from __future__ import annotations

from pathlib import Path
import sys
from typing import Annotated

import pandas as pd
import typer

from image_picker.io_utils import (
    copy_bucket_files,
    default_cache_path,
    default_errors_path,
    load_cache,
    save_cache,
    scan_images,
    write_errors_csv,
    write_results_csv,
)
from image_picker.calibrator import (
    CalibrationArtifacts,
    apply_calibrator_to_records,
    build_labeled_dataset,
    load_calibrator,
    save_calibrator,
    train_calibrator_from_dataset,
)
from image_picker.scoring import ScoreConfig, process_images


def run_score_images(
    input_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False, dir_okay=True)],
    output_csv: Annotated[Path | None, typer.Option("--output")] = None,
    recursive: Annotated[bool, typer.Option("--recursive")] = False,
    batch_size: Annotated[int, typer.Option("--batch-size", min=1)] = 8,
    device: Annotated[str, typer.Option("--device")] = "cuda",
    workers: Annotated[int, typer.Option("--workers", min=0)] = 0,
    max_image_side: Annotated[int, typer.Option("--max-image-side", min=0)] = 2048,
    enable_vlm_comments: Annotated[bool, typer.Option("--enable-vlm-comments")] = False,
    comment_limit: Annotated[int, typer.Option("--comment-limit", min=0)] = 50,
    cache_file: Annotated[Path | None, typer.Option("--cache-file")] = None,
    quality_metric: Annotated[str, typer.Option("--quality-metric")] = "topiq_nr",
    aesthetic_model_name: Annotated[str, typer.Option("--aesthetic-model-name")] = "ViT-L-14",
    aesthetic_pretrained: Annotated[str, typer.Option("--aesthetic-pretrained")] = "openai",
    aesthetic_global_min: Annotated[float, typer.Option("--aesthetic-global-min")] = 1.0,
    aesthetic_global_max: Annotated[float, typer.Option("--aesthetic-global-max")] = 10.0,
    quality_global_min: Annotated[float, typer.Option("--quality-global-min")] = 0.0,
    quality_global_max: Annotated[float, typer.Option("--quality-global-max")] = 1.0,
) -> None:
    image_paths = scan_images(input_dir, recursive=recursive)
    if not image_paths:
        raise typer.BadParameter("No supported images were found in the input directory.")

    resolved_output = output_csv or (input_dir / "scores.csv")
    resolved_cache = cache_file or default_cache_path(resolved_output)
    errors_csv = default_errors_path(resolved_output)
    cache = load_cache(resolved_cache)

    config = ScoreConfig(
        device=device,
        batch_size=batch_size,
        workers=workers,
        quality_metric=quality_metric,
        aesthetic_model_name=aesthetic_model_name,
        aesthetic_pretrained=aesthetic_pretrained,
        max_image_side=max_image_side,
        enable_vlm_comments=enable_vlm_comments,
        comment_limit=comment_limit,
        aesthetic_global_min=aesthetic_global_min,
        aesthetic_global_max=aesthetic_global_max,
        quality_global_min=quality_global_min,
        quality_global_max=quality_global_max,
    )
    records, errors, cache_entries = process_images(image_paths, cache=cache, config=config)
    if not records:
        raise typer.Exit(code=1)

    write_results_csv(resolved_output, records)
    write_errors_csv(errors_csv, errors)
    save_cache(resolved_cache, cache_entries)

    typer.echo(f"Wrote {len(records)} rows to {resolved_output}")
    if errors:
        typer.echo(f"Skipped {len(errors)} files. Details: {errors_csv}")


def export_buckets(
    scores_csv: Annotated[Path, typer.Option("--scores-csv", exists=True, dir_okay=False)],
    destination_dir: Annotated[Path, typer.Option("--destination")],
    bucket_column: Annotated[str, typer.Option("--bucket-column")] = "bucket",
    move_files: Annotated[bool, typer.Option("--move")] = False,
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
) -> None:
    copied, skipped = copy_bucket_files(
        scores_csv,
        destination_dir,
        overwrite=overwrite,
        bucket_column=bucket_column,
        move_files=move_files,
    )
    action = "Moved" if move_files else "Copied"
    typer.echo(f"{action} {copied} files to {destination_dir}")
    if skipped:
        typer.echo(f"Skipped {skipped} existing files. Use --overwrite to replace them.")


def train_calibrator(
    positive_dir: Annotated[Path, typer.Option("--positive-dir", exists=True, file_okay=False, dir_okay=True)],
    negative_dir: Annotated[Path, typer.Option("--negative-dir", exists=True, file_okay=False, dir_okay=True)],
    output_model: Annotated[Path, typer.Option("--output-model")] = Path("personal_calibrator.pkl"),
    output_dataset: Annotated[Path | None, typer.Option("--output-dataset")] = None,
    device: Annotated[str, typer.Option("--device")] = "cuda",
    batch_size: Annotated[int, typer.Option("--batch-size", min=1)] = 4,
    max_image_side: Annotated[int, typer.Option("--max-image-side", min=0)] = 2048,
    quality_metric: Annotated[str, typer.Option("--quality-metric")] = "topiq_nr",
) -> None:
    config = ScoreConfig(
        device=device,
        batch_size=batch_size,
        max_image_side=max_image_side,
        quality_metric=quality_metric,
    )
    cache_dir = output_model.with_suffix("")
    dataset, dataset_metadata = build_labeled_dataset(
        positive_dir=positive_dir,
        negative_dir=negative_dir,
        config=config,
        cache_dir=cache_dir,
    )
    artifacts = train_calibrator_from_dataset(dataset)
    metadata = dict(artifacts.metadata)
    metadata.update(dataset_metadata)
    artifacts = CalibrationArtifacts(
        pipeline=artifacts.pipeline,
        feature_names=artifacts.feature_names,
        metadata=metadata,
    )
    save_calibrator(output_model, artifacts)
    if output_dataset is not None:
        output_dataset.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_csv(output_dataset, index=False)
    typer.echo(
        "Trained calibrator "
        f"with {metadata['positive_count']} positives and {metadata['negative_count']} negatives. "
        f"Train ROC-AUC: {metadata['roc_auc_train']:.4f}"
    )
    typer.echo(f"Saved model to {output_model}")


def apply_calibrator(
    scores_csv: Annotated[Path, typer.Option("--scores-csv", exists=True, dir_okay=False)],
    model_path: Annotated[Path, typer.Option("--model", exists=True, dir_okay=False)],
    output_csv: Annotated[Path | None, typer.Option("--output")] = None,
) -> None:
    records = pd.read_csv(scores_csv).to_dict(orient="records")
    artifacts = load_calibrator(model_path)
    updated_records = apply_calibrator_to_records(records, artifacts)
    resolved_output = output_csv or scores_csv.with_name(f"{scores_csv.stem}.personal.csv")
    write_results_csv(resolved_output, updated_records)
    typer.echo(f"Wrote calibrated scores to {resolved_output}")


def main() -> None:
    typer.run(run_score_images)


def export_main() -> None:
    typer.run(export_buckets)


def train_calibrator_main() -> None:
    typer.run(train_calibrator)


def apply_calibrator_main() -> None:
    typer.run(apply_calibrator)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "export-buckets":
        sys.argv.pop(1)
        export_main()
    else:
        main()
