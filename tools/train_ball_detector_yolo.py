from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune YOLO26 for tiny football detection.")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default="yolo26s.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument(
        "--augmentation-profile",
        choices=("default", "low-light-tiny-ball"),
        default="default",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    dataset = args.dataset.resolve()
    summary_path = dataset / "dataset_summary.json"
    yolo_summary_path = dataset / "yolo_summary.json"
    if not summary_path.exists() or not yolo_summary_path.exists():
        raise ValueError("Dataset needs dataset_summary.json and exported YOLO labels")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    train_sources = set(summary["splits"]["train"]["source_videos"])
    valid_sources = set(summary["splits"]["valid"]["source_videos"])
    if not train_sources or not valid_sources or train_sources & valid_sources:
        raise ValueError("YOLO train and validation sources must be non-empty and disjoint")
    output = args.output.resolve()
    config = {
        "dataset": str(dataset),
        "output": str(output),
        "model": args.model,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch_size": args.batch_size,
        "device": args.device,
        "augmentation_profile": args.augmentation_profile,
        "train_sources": sorted(train_sources),
        "validation_sources": sorted(valid_sources),
    }
    print(json.dumps(config, indent=2))
    if args.dry_run:
        return

    import albumentations as A
    from ultralytics import YOLO

    output.mkdir(parents=True, exist_ok=True)
    data_yaml = output / "dataset.yaml"
    data_yaml.write_text(
        f"path: {dataset}\ntrain: train/images\nval: valid/images\nnames:\n  0: sports ball\n",
        encoding="utf-8",
    )
    blur_probability = 0.10 if args.augmentation_profile == "low-light-tiny-ball" else 0.20
    blur_augmentations = [
        A.OneOf(
            [
                A.MotionBlur(blur_limit=(3, 7), p=1.0),
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            ],
            p=blur_probability,
        )
    ]
    low_light_profile = args.augmentation_profile == "low-light-tiny-ball"
    model = YOLO(args.model)
    model.train(
        data=str(data_yaml),
        project=str(output.parent),
        name=output.name,
        exist_ok=True,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch_size,
        device=args.device,
        workers=4,
        patience=max(5, args.epochs),
        optimizer="AdamW",
        lr0=0.001,
        warmup_epochs=1.0,
        seed=0,
        deterministic=True,
        mosaic=0.0 if low_light_profile else 0.5,
        mixup=0.0,
        copy_paste=0.0,
        hsv_h=0.005 if low_light_profile else 0.015,
        hsv_s=0.25 if low_light_profile else 0.7,
        hsv_v=0.20 if low_light_profile else 0.4,
        translate=0.05 if low_light_profile else 0.1,
        scale=0.20 if low_light_profile else 0.5,
        fliplr=0.5,
        augmentations=blur_augmentations,
        plots=True,
        verbose=True,
    )


if __name__ == "__main__":
    main()
