from __future__ import annotations

import json
from pathlib import Path


def export_yolo_labels_from_coco_dataset(dataset_dir: str | Path) -> dict[str, object]:
    """Add YOLO labels beside an existing clip-separated COCO image dataset."""

    dataset = Path(dataset_dir)
    summary = json.loads((dataset / "dataset_summary.json").read_text(encoding="utf-8"))
    split_summary: dict[str, dict[str, int]] = {}
    for split in ("train", "valid"):
        annotation_path = dataset / split / "_annotations.coco.json"
        payload = json.loads(annotation_path.read_text(encoding="utf-8"))
        categories = payload.get("categories", [])
        ball_categories = [item for item in categories if item.get("name") == "sports ball"]
        if len(ball_categories) != 1:
            raise ValueError(f"Expected exactly one sports-ball category in {annotation_path}")
        ball_category_id = int(ball_categories[0]["id"])
        images = {int(item["id"]): item for item in payload.get("images", [])}
        labels_by_image: dict[int, list[str]] = {image_id: [] for image_id in images}
        for annotation in payload.get("annotations", []):
            if int(annotation["category_id"]) != ball_category_id:
                continue
            image_id = int(annotation["image_id"])
            image = images[image_id]
            width, height = float(image["width"]), float(image["height"])
            x, y, box_width, box_height = (float(value) for value in annotation["bbox"])
            if width <= 0 or height <= 0 or box_width <= 0 or box_height <= 0:
                raise ValueError(f"Invalid image or box dimensions for image_id {image_id}")
            center_x = (x + box_width / 2.0) / width
            center_y = (y + box_height / 2.0) / height
            normalized_width = box_width / width
            normalized_height = box_height / height
            values = (center_x, center_y, normalized_width, normalized_height)
            if any(not 0.0 <= value <= 1.0 for value in values):
                raise ValueError(f"Ball box falls outside image_id {image_id}")
            labels_by_image[image_id].append(
                "0 " + " ".join(f"{value:.8f}" for value in values)
            )

        labels_dir = dataset / split / "labels"
        labels_dir.mkdir(parents=True, exist_ok=True)
        for image_id, image in images.items():
            label_name = Path(str(image["file_name"])).with_suffix(".txt").name
            lines = labels_by_image[image_id]
            (labels_dir / label_name).write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
            )
        split_summary[split] = {
            "images": len(images),
            "positive_images": sum(bool(lines) for lines in labels_by_image.values()),
            "negative_images": sum(not lines for lines in labels_by_image.values()),
            "annotations": sum(len(lines) for lines in labels_by_image.values()),
        }

    yolo_summary = {
        "schema_version": 1,
        "format": "yolo_detection",
        "source_dataset": str(dataset.resolve()),
        "validation_sources": summary.get("validation_sources", []),
        "splits": split_summary,
    }
    (dataset / "yolo_summary.json").write_text(
        json.dumps(yolo_summary, indent=2) + "\n", encoding="utf-8"
    )
    return yolo_summary
