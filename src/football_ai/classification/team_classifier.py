from __future__ import annotations

from collections import defaultdict, deque

import cv2
import numpy as np
import supervision as sv

from football_ai.classification.color_features import (
    extract_shirt_feature,
)


class TeamClassifier:
    def __init__(
        self,
        samples_per_player: int = 30,
        minimum_players: int = 4,
        refit_interval: int = 30,
        minimum_samples_per_player: int = 3,
    ) -> None:
        self.samples_per_player = (
            samples_per_player
        )
        self.minimum_players = (
            minimum_players
        )
        self.refit_interval = (
            refit_interval
        )
        self.minimum_samples_per_player = (
            minimum_samples_per_player
        )

        self.player_features: dict[
            int,
            deque[np.ndarray],
        ] = defaultdict(
            lambda: deque(
                maxlen=self.samples_per_player
            )
        )

        self.team_by_tracker_id: dict[
            int,
            int,
        ] = {}

        self.cluster_centers: (
            np.ndarray | None
        ) = None

        self.frame_counter = 0

    def update(
        self,
        frame: np.ndarray,
        tracked_players: sv.Detections,
    ) -> dict[int, int]:
        self.frame_counter += 1

        if tracked_players.tracker_id is None:
            return dict(
                self.team_by_tracker_id
            )

        for index in range(
            len(tracked_players)
        ):
            tracker_id = int(
                tracked_players.tracker_id[index]
            )

            bounding_box = (
                tracked_players.xyxy[index]
            )

            feature = extract_shirt_feature(
                frame=frame,
                bounding_box=bounding_box,
            )

            if feature is None:
                continue

            self.player_features[
                tracker_id
            ].append(feature)

        should_fit = (
            self.cluster_centers is None
            or (
                self.frame_counter
                % self.refit_interval
                == 0
            )
        )

        if should_fit:
            self._fit_clusters()

        self._assign_unclassified_players()

        return dict(
            self.team_by_tracker_id
        )

    def _get_average_features(
        self,
    ) -> tuple[
        list[int],
        np.ndarray | None,
    ]:
        tracker_ids: list[int] = []
        average_features: list[
            np.ndarray
        ] = []

        for (
            tracker_id,
            features,
        ) in self.player_features.items():
            if (
                len(features)
                < self.minimum_samples_per_player
            ):
                continue

            tracker_ids.append(
                tracker_id
            )

            average_feature = np.mean(
                np.stack(features),
                axis=0,
            ).astype(np.float32)

            average_features.append(
                average_feature
            )

        if not average_features:
            return tracker_ids, None

        feature_matrix = np.stack(
            average_features
        ).astype(np.float32)

        return (
            tracker_ids,
            feature_matrix,
        )

    def _fit_clusters(self) -> None:
        (
            tracker_ids,
            feature_matrix,
        ) = self._get_average_features()

        if feature_matrix is None:
            return

        if (
            len(tracker_ids)
            < self.minimum_players
        ):
            return

        criteria = (
            cv2.TERM_CRITERIA_EPS
            + cv2.TERM_CRITERIA_MAX_ITER,
            100,
            0.001,
        )

        (
            _compactness,
            labels,
            centers,
        ) = cv2.kmeans(
            feature_matrix,
            2,
            None,
            criteria,
            10,
            cv2.KMEANS_PP_CENTERS,
        )

        centers = centers.astype(
            np.float32
        )

        labels = labels.flatten()

        if self.cluster_centers is not None:
            normal_distance = (
                np.linalg.norm(
                    centers[0]
                    - self.cluster_centers[0]
                )
                + np.linalg.norm(
                    centers[1]
                    - self.cluster_centers[1]
                )
            )

            swapped_distance = (
                np.linalg.norm(
                    centers[0]
                    - self.cluster_centers[1]
                )
                + np.linalg.norm(
                    centers[1]
                    - self.cluster_centers[0]
                )
            )

            if (
                swapped_distance
                < normal_distance
            ):
                centers = centers[
                    [1, 0]
                ]

                labels = (
                    1 - labels
                )

        self.cluster_centers = centers

        for tracker_id, label in zip(
            tracker_ids,
            labels,
            strict=True,
        ):
            self.team_by_tracker_id[
                tracker_id
            ] = int(label)

    def _assign_unclassified_players(
        self,
    ) -> None:
        if self.cluster_centers is None:
            return

        for (
            tracker_id,
            features,
        ) in self.player_features.items():
            if (
                tracker_id
                in self.team_by_tracker_id
            ):
                continue

            if (
                len(features)
                < self.minimum_samples_per_player
            ):
                continue

            average_feature = np.mean(
                np.stack(features),
                axis=0,
            ).astype(np.float32)

            distances = np.linalg.norm(
                self.cluster_centers
                - average_feature,
                axis=1,
            )

            team_id = int(
                np.argmin(distances)
            )

            self.team_by_tracker_id[
                tracker_id
            ] = team_id