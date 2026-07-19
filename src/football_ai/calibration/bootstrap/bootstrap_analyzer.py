from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import cv2
import numpy as np

from football_ai.calibration.bootstrap.camera_state_clustering import (
    CameraStateClustering,
    cluster_camera_states,
)
from football_ai.calibration.bootstrap.frame_sampler import (
    BootstrapFrameSample,
    sample_bootstrap_frames,
)


@dataclass(frozen=True, slots=True)
class PitchBootstrapReport:
    source_video: str
    fps: float
    total_frames: int
    requested_duration_seconds: float
    analyzed_duration_seconds: float
    sample_interval_seconds: float
    dense_sample_count: int
    coverage_scan_interval_seconds: float
    samples: tuple[BootstrapFrameSample, ...]
    clustering: CameraStateClustering

    def to_dict(self) -> dict:
        cluster_by_sample = {
            sample_index: cluster.cluster_id
            for cluster in self.clustering.clusters
            for sample_index in cluster.sample_indices
        }
        return {
            "schema_version": 1,
            "source_video": self.source_video,
            "fps": self.fps,
            "total_frames": self.total_frames,
            "requested_duration_seconds": self.requested_duration_seconds,
            "analyzed_duration_seconds": self.analyzed_duration_seconds,
            "sample_interval_seconds": self.sample_interval_seconds,
            "dense_sample_count": self.dense_sample_count,
            "coverage_scan_interval_seconds": self.coverage_scan_interval_seconds,
            "sample_count": len(self.samples),
            "camera_state_count": len(self.clustering.clusters),
            "separation_score": self.clustering.separation_score,
            "samples": [
                {
                    "sample_index": item.sample_index,
                    "frame_number": item.frame_number,
                    "time_seconds": item.time_seconds,
                    "camera_state": cluster_by_sample[item.sample_index],
                }
                for item in self.samples
            ],
            "camera_states": [
                {
                    "camera_state": cluster.cluster_id,
                    "sample_count": len(cluster.sample_indices),
                    "representative_sample_index": cluster.representative_sample_index,
                    "representative_frame_number": self.samples[
                        cluster.representative_sample_index
                    ].frame_number,
                    "representative_time_seconds": self.samples[
                        cluster.representative_sample_index
                    ].time_seconds,
                    "mean_distance": cluster.mean_distance,
                    "maximum_distance": cluster.maximum_distance,
                    "support_ratio": cluster.support_ratio,
                    "stable": cluster.stable,
                }
                for cluster in self.clustering.clusters
            ],
        }

    def save_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


class PitchBootstrapAnalyzer:
    def __init__(
        self,
        duration_seconds: float = 60.0,
        sample_interval_seconds: float = 0.5,
        camera_state_count: int = 5,
        coverage_scan_interval_seconds: float = 5.0,
    ) -> None:
        self.duration_seconds = duration_seconds
        self.sample_interval_seconds = sample_interval_seconds
        self.camera_state_count = camera_state_count
        self.coverage_scan_interval_seconds = coverage_scan_interval_seconds

    def analyze(self, video_path: Path) -> PitchBootstrapReport:
        dense_samples, fps, total_frames = sample_bootstrap_frames(
            video_path,
            duration_seconds=self.duration_seconds,
            interval_seconds=self.sample_interval_seconds,
        )
        video_duration = (total_frames - 1) / fps
        coverage_samples, _coverage_fps, _coverage_total = sample_bootstrap_frames(
            video_path,
            duration_seconds=video_duration,
            interval_seconds=self.coverage_scan_interval_seconds,
        )
        combined = [*dense_samples]
        combined.extend(
            item for item in coverage_samples
            if item.time_seconds > dense_samples[-1].time_seconds + 0.01
        )
        samples = [
            BootstrapFrameSample(
                sample_index=index,
                frame_number=item.frame_number,
                time_seconds=item.time_seconds,
                frame=item.frame,
            )
            for index, item in enumerate(combined)
        ]
        clustering = cluster_camera_states(samples, self.camera_state_count)
        return PitchBootstrapReport(
            source_video=str(video_path),
            fps=fps,
            total_frames=total_frames,
            requested_duration_seconds=self.duration_seconds,
            analyzed_duration_seconds=dense_samples[-1].time_seconds,
            sample_interval_seconds=self.sample_interval_seconds,
            dense_sample_count=len(dense_samples),
            coverage_scan_interval_seconds=self.coverage_scan_interval_seconds,
            samples=tuple(samples),
            clustering=clustering,
        )

    @staticmethod
    def create_contact_sheet(report: PitchBootstrapReport) -> np.ndarray:
        tile_width, tile_height = 480, 270
        columns = min(3, len(report.clustering.clusters))
        rows = int(np.ceil(len(report.clustering.clusters) / columns))
        header_height = 82
        sheet = np.zeros(
            (header_height + rows * tile_height, columns * tile_width, 3),
            dtype=np.uint8,
        )
        _text(sheet, "PITCH BOOTSTRAP - CAMERASTANDEN", (18, 30), 0.75, 2)
        _text(
            sheet,
            f"{len(report.samples)} samples | {len(report.clustering.clusters)} "
            f"groepen | scheiding {report.clustering.separation_score:.1%}",
            (18, 61),
            0.55,
            1,
        )
        for index, cluster in enumerate(report.clustering.clusters):
            sample = report.samples[cluster.representative_sample_index]
            tile = cv2.resize(sample.frame, (tile_width, tile_height))
            cv2.rectangle(tile, (0, 0), (tile_width, 48), (18, 18, 18), -1)
            state_label = "STABIEL" if cluster.stable else "ZELDZAAM"
            _text(
                tile,
                f"Stand {cluster.cluster_id} {state_label} | "
                f"{len(cluster.sample_indices)} samples | {sample.time_seconds:.1f}s",
                (14, 31),
                0.58,
                2,
            )
            row, column = divmod(index, columns)
            y = header_height + row * tile_height
            x = column * tile_width
            sheet[y:y + tile_height, x:x + tile_width] = tile
        return sheet


def _text(
    image: np.ndarray,
    value: str,
    origin: tuple[int, int],
    scale: float,
    thickness: int,
) -> None:
    cv2.putText(
        image,
        value,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (245, 245, 245),
        thickness,
        cv2.LINE_AA,
    )
