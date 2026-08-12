from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import math
from typing import Iterable, Mapping, Sequence

from football_ai.detection.ball_evaluation import BallGroundTruth, boxes_match


VALID_CANDIDATE_LABELS = {"positive", "negative", "ambiguous"}


@dataclass(frozen=True)
class ActiveBallCandidateExample:
    frame_number: int
    candidate_index: int
    box: tuple[float, float, float, float]
    confidence: float
    label: str
    visibility: str
    occlusion: str

    def __post_init__(self) -> None:
        if self.label not in VALID_CANDIDATE_LABELS:
            raise ValueError(f"Unknown active-ball candidate label: {self.label}")


def label_active_ball_candidates(
    annotations: Iterable[BallGroundTruth],
    candidates_by_frame: Mapping[
        int,
        Sequence[tuple[tuple[float, float, float, float], float]],
    ],
    *,
    maximum_center_distance: float = 20.0,
    minimum_iou: float = 0.10,
    negative_center_distance: float = 60.0,
) -> list[ActiveBallCandidateExample]:
    """Derive conservative candidate labels from human-reviewed ball boxes.

    A matching candidate is positive. Every candidate on a reviewed
    ``not_visible`` frame is negative. On ball frames, candidates far outside
    the reviewed location are safe negatives; nearby non-matches remain
    ambiguous so duplicate or slightly mislocalized ball detections are not
    penalized.
    """

    examples: list[ActiveBallCandidateExample] = []
    for annotation in annotations:
        if (
            annotation.review_status != "human_reviewed"
            or annotation.visibility == "unreviewed"
        ):
            continue
        for index, (box, confidence) in enumerate(
            candidates_by_frame.get(annotation.frame_number, ())
        ):
            if annotation.visibility == "not_visible":
                label = "negative"
            elif annotation.box is not None and boxes_match(
                annotation.box,
                box,
                maximum_center_distance,
                minimum_iou,
            ):
                label = "positive"
            elif annotation.box is not None and math.dist(
                (
                    (annotation.box[0] + annotation.box[2]) / 2.0,
                    (annotation.box[1] + annotation.box[3]) / 2.0,
                ),
                ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0),
            ) >= negative_center_distance:
                label = "negative"
            else:
                label = "ambiguous"
            examples.append(
                ActiveBallCandidateExample(
                    frame_number=annotation.frame_number,
                    candidate_index=index,
                    box=box,
                    confidence=float(confidence),
                    label=label,
                    visibility=annotation.visibility,
                    occlusion=annotation.occlusion,
                )
            )
    return examples


def active_ball_dataset_report(
    annotations: Iterable[BallGroundTruth],
    examples: Iterable[ActiveBallCandidateExample],
    *,
    source_clip_count: int,
    minimum_positive_frames: int = 100,
    minimum_source_clips: int = 3,
) -> dict[str, object]:
    reviewed = [
        item
        for item in annotations
        if item.review_status == "human_reviewed" and item.visibility != "unreviewed"
    ]
    materialized = list(examples)
    label_counts = Counter(item.label for item in materialized)
    positive_frames = {item.frame_number for item in materialized if item.label == "positive"}
    visible_ball_frames = {
        item.frame_number
        for item in reviewed
        if item.visibility == "visible"
    }
    occluded_ball_frames = {
        item.frame_number
        for item in reviewed
        if item.visibility == "occluded"
    }
    missing_positive_frames = sorted(visible_ball_frames - positive_frames)
    occluded_without_positive_frames = sorted(
        occluded_ball_frames - positive_frames
    )
    reasons = []
    if len(positive_frames) < minimum_positive_frames:
        reasons.append(
            f"only {len(positive_frames)} positive frames; require at least "
            f"{minimum_positive_frames}"
        )
    if source_clip_count < minimum_source_clips:
        reasons.append(
            f"only {source_clip_count} source clip; require at least "
            f"{minimum_source_clips}"
        )
    if missing_positive_frames:
        reasons.append(
            f"detector has no matching candidate on {len(missing_positive_frames)} "
            "reviewed ball frames"
        )
    return {
        "reviewed_frames": len(reviewed),
        "candidate_examples": len(materialized),
        "label_counts": dict(sorted(label_counts.items())),
        "positive_frames": len(positive_frames),
        "reviewed_ball_frames": len(visible_ball_frames | occluded_ball_frames),
        "reviewed_visible_ball_frames": len(visible_ball_frames),
        "reviewed_occluded_ball_frames": len(occluded_ball_frames),
        "missing_positive_frames": missing_positive_frames,
        "occluded_without_positive_frames": occluded_without_positive_frames,
        "source_clips": int(source_clip_count),
        "ready_for_training": not reasons,
        "blocking_reasons": reasons,
        "examples": [asdict(item) for item in materialized],
    }


def aggregate_active_ball_dataset_reports(
    reports: Iterable[Mapping[str, object]],
    *,
    minimum_positive_frames: int = 100,
    minimum_source_clips: int = 3,
) -> dict[str, object]:
    """Combine independently built clip reports without losing provenance."""

    materialized = list(reports)
    sources: list[str] = []
    positive_sources: list[str] = []
    label_counts: Counter[str] = Counter()
    positive_frames = 0
    reviewed_frames = 0
    candidate_examples = 0
    missing_by_source: dict[str, list[int]] = {}
    occluded_without_positive_by_source: dict[str, list[int]] = {}

    for report in materialized:
        source = str(report.get("source_video", ""))
        if not source:
            raise ValueError("Every dataset report must identify source_video")
        if source in sources:
            raise ValueError(f"Duplicate source video in dataset reports: {source}")
        sources.append(source)
        report_positive_frames = int(report.get("positive_frames", 0))
        positive_frames += report_positive_frames
        if report_positive_frames > 0:
            positive_sources.append(source)
        reviewed_frames += int(report.get("reviewed_frames", 0))
        candidate_examples += int(report.get("candidate_examples", 0))
        counts = report.get("label_counts", {})
        if not isinstance(counts, Mapping):
            raise ValueError(f"Invalid label_counts for source video: {source}")
        for label, count in counts.items():
            if str(label) not in VALID_CANDIDATE_LABELS:
                raise ValueError(f"Unknown candidate label in report: {label}")
            label_counts[str(label)] += int(count)
        missing = [int(frame) for frame in report.get("missing_positive_frames", [])]
        if missing:
            missing_by_source[source] = missing
        occluded_missing = [
            int(frame)
            for frame in report.get("occluded_without_positive_frames", [])
        ]
        if occluded_missing:
            occluded_without_positive_by_source[source] = occluded_missing

    reasons = []
    if positive_frames < minimum_positive_frames:
        reasons.append(
            f"only {positive_frames} positive frames; require at least "
            f"{minimum_positive_frames}"
        )
    if len(positive_sources) < minimum_source_clips:
        reasons.append(
            f"only {len(positive_sources)} source clips with positive frames; "
            f"require at least {minimum_source_clips}"
        )
    if missing_by_source:
        reasons.append(
            "detector has no matching candidate on "
            f"{sum(len(frames) for frames in missing_by_source.values())} "
            "reviewed ball frames"
        )

    return {
        "source_clips": len(sources),
        "source_videos": sources,
        "positive_source_clips": len(positive_sources),
        "positive_source_videos": positive_sources,
        "reviewed_frames": reviewed_frames,
        "candidate_examples": candidate_examples,
        "label_counts": dict(sorted(label_counts.items())),
        "positive_frames": positive_frames,
        "missing_positive_frames_by_source": missing_by_source,
        "occluded_without_positive_frames_by_source": (
            occluded_without_positive_by_source
        ),
        "ready_for_training": not reasons,
        "blocking_reasons": reasons,
    }
