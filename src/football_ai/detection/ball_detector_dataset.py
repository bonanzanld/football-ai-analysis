from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
from typing import Iterable

import cv2
import numpy as np


COCO_HEAD_CLASS_COUNT = 90
COCO_SPORTS_BALL_SLOT = 37


def detector_category_schema(
    preserve_coco_head: bool,
) -> tuple[list[dict[str, object]], int]:
    """Return categories and ball ID, optionally aligned to pretrained COCO logits."""

    if not preserve_coco_head:
        return [{"id": 1, "name": "sports ball"}], 1
    categories = [
        {"id": slot, "name": f"coco_slot_{slot}"}
        for slot in range(COCO_HEAD_CLASS_COUNT)
    ]
    categories[COCO_SPORTS_BALL_SLOT] = {
        "id": COCO_SPORTS_BALL_SLOT,
        "name": "sports ball",
    }
    return categories, COCO_SPORTS_BALL_SLOT


@dataclass(frozen=True)
class DetectorFrame:
    source_video: str
    frame_number: int
    image_path: Path
    visibility: str
    ball_box: tuple[float, float, float, float] | None


def square_tile_origins(
    width: int,
    height: int,
    *,
    tile_size: int,
    overlap: float,
) -> tuple[tuple[int, int], ...]:
    """Return deterministic overlapping square-tile origins covering an image."""

    if width < 1 or height < 1 or tile_size < 1:
        raise ValueError("Image dimensions and tile_size must be positive")
    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must be at least zero and below one")
    step = max(1, int(round(tile_size * (1.0 - overlap))))

    def axis_origins(length: int) -> list[int]:
        maximum = max(0, length - tile_size)
        values = list(range(0, maximum + 1, step))
        if not values or values[-1] != maximum:
            values.append(maximum)
        return values

    return tuple(
        (x, y) for y in axis_origins(height) for x in axis_origins(width)
    )


def extract_square_tile(
    image: np.ndarray,
    origin: tuple[int, int],
    *,
    tile_size: int,
) -> np.ndarray:
    """Extract a square tile, padding only beyond the image boundary."""

    if image.ndim not in (2, 3):
        raise ValueError("Expected a two- or three-dimensional image")
    if tile_size < 1:
        raise ValueError("tile_size must be positive")
    x, y = origin
    height, width = image.shape[:2]
    if x < 0 or y < 0 or x >= width or y >= height:
        raise ValueError("Tile origin falls outside the image")
    shape = (tile_size, tile_size, *image.shape[2:])
    tile = np.zeros(shape, dtype=image.dtype)
    crop = image[y:min(y + tile_size, height), x:min(x + tile_size, width)]
    tile[: crop.shape[0], : crop.shape[1]] = crop
    return tile


def load_human_detector_frames(
    manifest_paths: Iterable[str | Path],
) -> tuple[DetectorFrame, ...]:
    """Load unambiguous human detector truth from review manifests.

    Visible annotations are positive detector truth and not-visible frames are
    useful negative images. Occluded annotations are intentionally excluded:
    their boxes describe an inferred ball location, not necessarily visible
    pixels a detector can learn from.
    """

    frames: dict[tuple[str, int], DetectorFrame] = {}
    for raw_path in manifest_paths:
        manifest_path = Path(raw_path)
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        source_video = str(payload.get("source_video", ""))
        if not source_video:
            raise ValueError(f"Manifest has no source_video: {manifest_path}")
        for item in payload.get("annotations", []):
            if item.get("review_status") != "human_reviewed":
                continue
            visibility = str(item.get("visibility", "unreviewed"))
            if visibility not in {"visible", "not_visible"}:
                continue
            frame_number = int(item["frame_number"])
            raw_box = item.get("ball_box")
            box = (
                None
                if raw_box is None
                else tuple(float(value) for value in raw_box)
            )
            if visibility == "visible" and box is None:
                raise ValueError(
                    f"Visible frame {frame_number} has no ball_box in {manifest_path}"
                )
            if visibility == "not_visible" and box is not None:
                raise ValueError(
                    f"Not-visible frame {frame_number} has a ball_box in {manifest_path}"
                )
            image_value = item.get("image")
            if not image_value:
                raise ValueError(
                    f"Frame {frame_number} has no image path in {manifest_path}"
                )
            frame = DetectorFrame(
                source_video=source_video,
                frame_number=frame_number,
                image_path=manifest_path.parent / str(image_value),
                visibility=visibility,
                ball_box=box,
            )
            key = (source_video, frame_number)
            previous = frames.get(key)
            if previous is not None and (
                previous.visibility != frame.visibility
                or previous.ball_box != frame.ball_box
            ):
                raise ValueError(
                    f"Conflicting human annotations for {source_video} frame {frame_number}"
                )
            frames[key] = frame
    return tuple(frames[key] for key in sorted(frames))


def export_coco_ball_detector_dataset(
    frames: Iterable[DetectorFrame],
    output_dir: str | Path,
    *,
    validation_sources: Iterable[str],
    preserve_coco_head: bool = False,
) -> dict[str, object]:
    """Export clip-separated COCO train/validation data for one ball class."""

    output = Path(output_dir)
    validation = set(validation_sources)
    all_frames = tuple(frames)
    known_sources = {item.source_video for item in all_frames}
    unknown = validation - known_sources
    if unknown:
        raise ValueError(f"Unknown validation source(s): {sorted(unknown)}")
    if not validation:
        raise ValueError("At least one validation source is required")
    if validation == known_sources:
        raise ValueError("Validation sources cannot contain every source clip")

    split_frames = {
        "train": [item for item in all_frames if item.source_video not in validation],
        "valid": [item for item in all_frames if item.source_video in validation],
    }
    categories, ball_category_id = detector_category_schema(preserve_coco_head)
    summary: dict[str, object] = {
        "schema_version": 1,
        "format": "coco_detection",
        "category": {"id": ball_category_id, "name": "sports ball"},
        "class_layout": "coco_90_preserved" if preserve_coco_head else "single_class",
        "validation_sources": sorted(validation),
        "excluded_visibility": ["occluded", "unreviewed"],
        "splits": {},
    }
    for split, items in split_frames.items():
        image_dir = output / split / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        images: list[dict[str, object]] = []
        annotations: list[dict[str, object]] = []
        for image_id, item in enumerate(items, start=1):
            if not item.image_path.exists():
                raise FileNotFoundError(item.image_path)
            image = cv2.imread(str(item.image_path))
            if image is None:
                raise ValueError(f"Cannot read image: {item.image_path}")
            height, width = image.shape[:2]
            source_slug = Path(item.source_video).stem
            filename = f"{source_slug}_frame_{item.frame_number:06d}.jpg"
            shutil.copy2(item.image_path, image_dir / filename)
            images.append(
                {
                    "id": image_id,
                    "file_name": f"images/{filename}",
                    "width": width,
                    "height": height,
                    "source_video": item.source_video,
                    "frame_number": item.frame_number,
                }
            )
            if item.ball_box is not None:
                x1, y1, x2, y2 = item.ball_box
                x1 = min(max(x1, 0.0), float(width))
                y1 = min(max(y1, 0.0), float(height))
                x2 = min(max(x2, x1), float(width))
                y2 = min(max(y2, y1), float(height))
                box_width = x2 - x1
                box_height = y2 - y1
                if box_width <= 0.0 or box_height <= 0.0:
                    raise ValueError(
                        f"Invalid ball box for {item.source_video} frame {item.frame_number}"
                    )
                annotations.append(
                    {
                        "id": len(annotations) + 1,
                        "image_id": image_id,
                        "category_id": ball_category_id,
                        "bbox": [x1, y1, box_width, box_height],
                        "area": box_width * box_height,
                        "iscrowd": 0,
                    }
                )
        payload = {
            "images": images,
            "annotations": annotations,
            "categories": categories,
        }
        annotation_path = output / split / "_annotations.coco.json"
        annotation_path.parent.mkdir(parents=True, exist_ok=True)
        annotation_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        summary["splits"][split] = {
            "source_videos": sorted({item.source_video for item in items}),
            "images": len(images),
            "positive_images": len(annotations),
            "negative_images": len(images) - len(annotations),
        }
    (output / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def export_tiled_coco_ball_detector_dataset(
    frames: Iterable[DetectorFrame],
    output_dir: str | Path,
    *,
    validation_sources: Iterable[str],
    tile_size: int = 960,
    overlap: float = 0.25,
    preserve_coco_head: bool = False,
) -> dict[str, object]:
    """Export source-separated tiles without cutting through labelled balls."""

    output = Path(output_dir)
    validation = set(validation_sources)
    all_frames = tuple(frames)
    known_sources = {item.source_video for item in all_frames}
    unknown = validation - known_sources
    if unknown:
        raise ValueError(f"Unknown validation source(s): {sorted(unknown)}")
    if not validation or validation == known_sources:
        raise ValueError("Validation sources must be a non-empty proper subset")
    split_frames = {
        "train": [item for item in all_frames if item.source_video not in validation],
        "valid": [item for item in all_frames if item.source_video in validation],
    }
    categories, ball_category_id = detector_category_schema(preserve_coco_head)
    summary: dict[str, object] = {
        "schema_version": 1,
        "format": "coco_detection_tiled",
        "category": {"id": ball_category_id, "name": "sports ball"},
        "class_layout": "coco_90_preserved" if preserve_coco_head else "single_class",
        "validation_sources": sorted(validation),
        "excluded_visibility": ["occluded", "unreviewed"],
        "tiling": {"tile_size": tile_size, "overlap": overlap},
        "splits": {},
    }
    for split, items in split_frames.items():
        image_dir = output / split / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        images: list[dict[str, object]] = []
        annotations: list[dict[str, object]] = []
        skipped_partial_tiles = 0
        for item in items:
            image = cv2.imread(str(item.image_path))
            if image is None:
                raise ValueError(f"Cannot read image: {item.image_path}")
            height, width = image.shape[:2]
            source_slug = Path(item.source_video).stem
            for tile_x, tile_y in square_tile_origins(
                width, height, tile_size=tile_size, overlap=overlap
            ):
                translated_box: tuple[float, float, float, float] | None = None
                if item.ball_box is not None:
                    x1, y1, x2, y2 = item.ball_box
                    tile_right = tile_x + tile_size
                    tile_bottom = tile_y + tile_size
                    intersects = (
                        x2 > tile_x
                        and x1 < tile_right
                        and y2 > tile_y
                        and y1 < tile_bottom
                    )
                    fully_inside = (
                        x1 >= tile_x
                        and y1 >= tile_y
                        and x2 <= tile_right
                        and y2 <= tile_bottom
                    )
                    if intersects and not fully_inside:
                        skipped_partial_tiles += 1
                        continue
                    if fully_inside:
                        translated_box = (
                            x1 - tile_x,
                            y1 - tile_y,
                            x2 - tile_x,
                            y2 - tile_y,
                        )
                image_id = len(images) + 1
                filename = (
                    f"{source_slug}_frame_{item.frame_number:06d}"
                    f"_x{tile_x:04d}_y{tile_y:04d}.jpg"
                )
                tile = extract_square_tile(
                    image, (tile_x, tile_y), tile_size=tile_size
                )
                if not cv2.imwrite(str(image_dir / filename), tile):
                    raise RuntimeError(f"Cannot write tile: {image_dir / filename}")
                images.append(
                    {
                        "id": image_id,
                        "file_name": f"images/{filename}",
                        "width": tile_size,
                        "height": tile_size,
                        "source_video": item.source_video,
                        "frame_number": item.frame_number,
                        "tile_origin": [tile_x, tile_y],
                        "source_width": width,
                        "source_height": height,
                    }
                )
                if translated_box is not None:
                    x1, y1, x2, y2 = translated_box
                    annotations.append(
                        {
                            "id": len(annotations) + 1,
                            "image_id": image_id,
                            "category_id": ball_category_id,
                            "bbox": [x1, y1, x2 - x1, y2 - y1],
                            "area": (x2 - x1) * (y2 - y1),
                            "iscrowd": 0,
                        }
                    )
        payload = {
            "images": images,
            "annotations": annotations,
            "categories": categories,
        }
        annotation_path = output / split / "_annotations.coco.json"
        annotation_path.parent.mkdir(parents=True, exist_ok=True)
        annotation_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        positive_ids = {item["image_id"] for item in annotations}
        summary["splits"][split] = {
            "source_videos": sorted({item.source_video for item in items}),
            "source_images": len(items),
            "images": len(images),
            "positive_images": len(positive_ids),
            "negative_images": len(images) - len(positive_ids),
            "annotations": len(annotations),
            "skipped_partial_ball_tiles": skipped_partial_tiles,
        }
    (output / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary
