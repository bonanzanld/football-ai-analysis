from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from football_ai.detection.active_ball_dataset import (
    active_ball_dataset_report,
    label_active_ball_candidates,
)
from football_ai.detection.ball_evaluation import BallGroundTruth, load_ball_annotations
from football_ai.tracking.online_camera_motion import transform_box


def _merge_annotations(
    paths: list[Path],
    *,
    allow_conflicting_overrides: bool = False,
) -> tuple[str, list[BallGroundTruth], list[int]]:
    source_video: str | None = None
    by_frame: dict[int, BallGroundTruth] = {}
    overridden_frames: list[int] = []
    for path in paths:
        manifest, annotations = load_ball_annotations(path)
        current_source = str(manifest.get("source_video", ""))
        if not current_source:
            raise ValueError(f"Annotation manifest has no source_video: {path}")
        if source_video is None:
            source_video = current_source
        elif current_source != source_video:
            raise ValueError("All annotation manifests must refer to the same video")
        for annotation in annotations:
            existing = by_frame.get(annotation.frame_number)
            if existing is not None:
                existing_reviewed = existing.review_status == "human_reviewed"
                annotation_reviewed = annotation.review_status == "human_reviewed"
                if existing_reviewed and not annotation_reviewed:
                    # Dense manifests commonly overlap an earlier sparse review.
                    # An open placeholder must never erase human-confirmed truth.
                    continue
                if annotation_reviewed and not existing_reviewed:
                    by_frame[annotation.frame_number] = annotation
                    continue
            if existing is not None and existing != annotation:
                if not allow_conflicting_overrides:
                    raise ValueError(
                        "Conflicting annotations for frame "
                        f"{annotation.frame_number}; pass "
                        "--allow-conflicting-overrides only when later manifests "
                        "are intentional corrections"
                    )
                overridden_frames.append(annotation.frame_number)
            by_frame[annotation.frame_number] = annotation
    return (
        source_video or "",
        [by_frame[frame] for frame in sorted(by_frame)],
        sorted(set(overridden_frames)),
    )


def _load_image_space_candidates(
    path: Path,
) -> tuple[str, dict[int, list[tuple[tuple[float, float, float, float], float]]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    frames = payload.get("frames")
    if payload.get("schema_version") not in (1, 2) or not isinstance(frames, list):
        raise ValueError("Unsupported or empty candidate cache")
    by_frame = {}
    for frame_number, frame in enumerate(frames):
        current_to_reference = np.asarray(frame["transform"], dtype=np.float64)
        try:
            reference_to_image = np.linalg.inv(current_to_reference)
        except np.linalg.LinAlgError:
            reference_to_image = np.eye(3, dtype=np.float64)
        by_frame[frame_number] = [
            (
                transform_box(
                    tuple(float(value) for value in candidate["box"]),
                    reference_to_image,
                ),
                float(candidate["confidence"]),
            )
            for candidate in frame.get("candidates", [])
        ]
    return str(payload.get("source_video", "")), by_frame


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build conservative active-ball candidate labels from human review."
    )
    parser.add_argument("--annotations", nargs="+", required=True, type=Path)
    parser.add_argument("--candidate-cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--center-distance", type=float, default=20.0)
    parser.add_argument("--minimum-iou", type=float, default=0.10)
    parser.add_argument(
        "--negative-center-distance",
        type=float,
        default=60.0,
        help="Mark non-matching candidates at least this many pixels away as negative.",
    )
    parser.add_argument(
        "--allow-conflicting-overrides",
        action="store_true",
        help=(
            "Let a later annotation manifest replace a conflicting earlier "
            "annotation for the same frame; overridden frames are recorded"
        ),
    )
    args = parser.parse_args()

    source_video, annotations, overridden_frames = _merge_annotations(
        args.annotations,
        allow_conflicting_overrides=args.allow_conflicting_overrides,
    )
    cached_source, candidates_by_frame = _load_image_space_candidates(
        args.candidate_cache
    )
    if Path(cached_source).resolve() != (PROJECT_ROOT / source_video).resolve():
        raise ValueError("Candidate cache belongs to a different source video")
    examples = label_active_ball_candidates(
        annotations,
        candidates_by_frame,
        maximum_center_distance=args.center_distance,
        minimum_iou=args.minimum_iou,
        negative_center_distance=args.negative_center_distance,
    )
    report = active_ball_dataset_report(
        annotations,
        examples,
        source_clip_count=1,
    )
    payload = {
        "schema_version": 1,
        "source_video": source_video,
        "candidate_cache": str(args.candidate_cache),
        "annotation_manifests": [str(path) for path in args.annotations],
        "overridden_annotation_frames": overridden_frames,
        **report,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    counts = payload["label_counts"]
    print(
        f"examples={payload['candidate_examples']} | "
        f"positive={counts.get('positive', 0)} | "
        f"negative={counts.get('negative', 0)} | "
        f"ambiguous={counts.get('ambiguous', 0)}"
    )
    print(
        f"positive_frames={payload['positive_frames']}/"
        f"{payload['reviewed_ball_frames']} | "
        f"ready_for_training={payload['ready_for_training']}"
    )
    for reason in payload["blocking_reasons"]:
        print(f"BLOCKED: {reason}")
    print(f"Dataset report: {args.output}")


if __name__ == "__main__":
    main()
