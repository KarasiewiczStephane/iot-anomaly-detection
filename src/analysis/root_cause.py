"""Root cause analysis identifying top contributing sensors per anomaly."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class RootCauseResult:
    """Analysis output for a single detected anomaly.

    Attributes:
        timestamp: When the anomaly occurred.
        anomaly_score: Overall ensemble anomaly score.
        top_contributors: Ranked ``(sensor_id, contribution_score)`` pairs.
        all_contributions: Full contribution mapping.
        explanation: Human-readable summary.
    """

    timestamp: Any
    anomaly_score: float
    top_contributors: list[tuple[str, float]] = field(default_factory=list)
    all_contributions: dict[str, float] = field(default_factory=dict)
    explanation: str = ""


class RootCauseAnalyzer:
    """Identify which sensors contributed most to each detected anomaly.

    Args:
        sensor_names: List of known sensor identifiers.
    """

    def __init__(self, sensor_names: list[str]) -> None:
        self.sensor_names = sensor_names

    def analyze(
        self,
        feature_contributions: dict[str, np.ndarray],
        anomaly_scores: np.ndarray,
        timestamps: np.ndarray | None,
        top_k: int = 3,
    ) -> list[RootCauseResult]:
        """Produce root-cause results for all anomalous points.

        Args:
            feature_contributions: Per-sensor score arrays keyed by sensor name.
            anomaly_scores: Ensemble scores for every sample.
            timestamps: Aligned timestamp array (may be ``None``).
            top_k: Number of top contributors to include per anomaly.

        Returns:
            List of :class:`RootCauseResult` for anomalous samples only.
        """
        results: list[RootCauseResult] = []
        anomaly_indices = np.where(anomaly_scores > 0.5)[0]

        for idx in anomaly_indices:
            contributions = {
                sensor: float(scores[idx]) for sensor, scores in feature_contributions.items()
            }

            sorted_contrib = sorted(contributions.items(), key=lambda x: x[1], reverse=True)
            top_contributors = sorted_contrib[:top_k]
            explanation = self._generate_explanation(top_contributors, anomaly_scores[idx])

            results.append(
                RootCauseResult(
                    timestamp=timestamps[idx] if timestamps is not None else idx,
                    anomaly_score=float(anomaly_scores[idx]),
                    top_contributors=top_contributors,
                    all_contributions=contributions,
                    explanation=explanation,
                )
            )

        logger.info("Root-cause analysis produced %d results", len(results))
        return results

    def _generate_explanation(
        self,
        top_contributors: list[tuple[str, float]],
        anomaly_score: float,
    ) -> str:
        """Build a human-readable explanation string.

        Args:
            top_contributors: Ranked contributor list.
            anomaly_score: Overall anomaly score.

        Returns:
            Descriptive sentence.
        """
        if not top_contributors:
            return "No significant contributors identified."

        parts: list[str] = []
        for sensor, score in top_contributors:
            sensor_type = sensor.split("_")[0] if "_" in sensor else sensor
            if score > 0.8:
                parts.append(f"{sensor_type} ({sensor}) showing critical deviation")
            elif score > 0.5:
                parts.append(f"{sensor_type} ({sensor}) showing moderate deviation")
            else:
                parts.append(f"{sensor_type} ({sensor}) showing minor deviation")

        severity = (
            "Critical" if anomaly_score > 0.8 else "Moderate" if anomaly_score > 0.5 else "Minor"
        )
        return f"{severity} anomaly detected. Primary contributors: {'; '.join(parts)}."

    def get_sensor_anomaly_rates(
        self,
        feature_contributions: dict[str, np.ndarray],
        threshold: float = 0.5,
    ) -> dict[str, float]:
        """Fraction of readings above *threshold* per sensor.

        Args:
            feature_contributions: Per-sensor score arrays.
            threshold: Score above which a reading counts as anomalous.

        Returns:
            Dict mapping sensor name to anomaly rate in [0, 1].
        """
        rates: dict[str, float] = {}
        for sensor, scores in feature_contributions.items():
            anomaly_count = int(np.sum(scores > threshold))
            rates[sensor] = anomaly_count / len(scores)
        return rates

    def get_temporal_patterns(
        self,
        feature_contributions: dict[str, np.ndarray],
        timestamps: np.ndarray,
        window_hours: int = 24,
    ) -> dict[str, dict[int, float]]:
        """Average anomaly score by hour-of-day for each sensor.

        Args:
            feature_contributions: Per-sensor score arrays.
            timestamps: Aligned timestamp array.
            window_hours: Unused (kept for interface compatibility).

        Returns:
            Nested dict ``{sensor: {hour: mean_score}}``.
        """
        hours = pd.to_datetime(timestamps).hour

        patterns: dict[str, dict[int, float]] = {}
        for sensor, scores in feature_contributions.items():
            hourly_rates: dict[int, float] = {}
            for hour in range(24):
                mask = hours == hour
                if mask.sum() > 0:
                    hourly_rates[hour] = float(np.mean(scores[mask]))
            patterns[sensor] = hourly_rates

        return patterns
