from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def _validate_dataset(dataset_dir: Path) -> dict[str, object]:
    summary_path = dataset_dir / "dataset_summary.json"
    if not summary_path.exists():
        raise ValueError(f"Missing dataset summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    splits = summary.get("splits", {})
    train = splits.get("train", {})
    valid = splits.get("valid", {})
    train_sources = set(train.get("source_videos", []))
    valid_sources = set(valid.get("source_videos", []))
    if not train_sources or not valid_sources:
        raise ValueError("Both train and valid must contain source videos")
    overlap = train_sources & valid_sources
    if overlap:
        raise ValueError(f"Source-video leakage between train and valid: {sorted(overlap)}")
    for split in ("train", "valid"):
        annotations = dataset_dir / split / "_annotations.coco.json"
        if not annotations.exists():
            raise ValueError(f"Missing COCO annotations: {annotations}")
    return summary


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune RF-DETR Medium on a clip-separated ball dataset."
    )
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"))
    parser.add_argument(
        "--disable-early-stopping",
        action="store_true",
        help="Run every requested epoch; useful when a newly initialized head learns slowly.",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Permit a full CPU run; normally refused because it is very slow.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate data and print the resolved training configuration.",
    )
    args = parser.parse_args()

    dataset = args.dataset.resolve()
    output = args.output.resolve()
    summary = _validate_dataset(dataset)
    device = args.device or _default_device()
    config = {
        "dataset_dir": str(dataset),
        "output_dir": str(output),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "device": device,
        "early_stopping": not args.disable_early_stopping,
        "validation_sources": summary["validation_sources"],
    }
    print(json.dumps(config, indent=2))
    if args.dry_run:
        return
    if device == "cpu" and not args.allow_cpu:
        raise SystemExit(
            "Refusing slow RF-DETR Medium CPU training; use a GPU/MPS device or pass --allow-cpu."
        )
    from rfdetr import RFDETRMedium

    output.mkdir(parents=True, exist_ok=True)
    model = RFDETRMedium()
    model.train(
        dataset_dir=str(dataset),
        output_dir=str(output),
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        device=device,
        checkpoint_interval=5,
        early_stopping=not args.disable_early_stopping,
        early_stopping_patience=10,
        tensorboard=True,
        wandb=False,
        notes={
            "purpose": "football ball detector keeper holdout experiment",
            "validation_sources": summary["validation_sources"],
        },
    )


if __name__ == "__main__":
    main()
