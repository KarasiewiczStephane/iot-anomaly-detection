"""Synthetic IoT sensor data generator with controllable anomaly injection."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class SensorConfig:
    """Configuration for a single sensor type.

    Attributes:
        name: Human-readable sensor name.
        normal_min: Lower bound of the normal operating range.
        normal_max: Upper bound of the normal operating range.
        noise_std: Standard deviation of Gaussian noise added to readings.
    """

    name: str
    normal_min: float
    normal_max: float
    noise_std: float = 0.02


SENSOR_CONFIGS: dict[str, SensorConfig] = {
    "temperature": SensorConfig("temperature", 60, 80, 0.5),
    "vibration": SensorConfig("vibration", 0.1, 0.5, 0.02),
    "pressure": SensorConfig("pressure", 2, 4, 0.1),
    "power": SensorConfig("power", 100, 150, 2.0),
}

AnomalyType = Literal["point", "contextual", "collective"]


class IoTDataGenerator:
    """Generate synthetic IoT sensor readings with injected anomalies.

    Args:
        seed: Random seed for reproducibility.
    """

    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.default_rng(seed)

    def _generate_base_signal(self, config: SensorConfig, n_points: int) -> np.ndarray:
        """Generate normal sensor readings with slight temporal correlation.

        Args:
            config: Sensor configuration with range and noise parameters.
            n_points: Number of data points to generate.

        Returns:
            Array of simulated sensor values.
        """
        mean = (config.normal_min + config.normal_max) / 2
        amplitude = (config.normal_max - config.normal_min) / 4
        t = np.linspace(0, 4 * np.pi, n_points)
        signal = mean + amplitude * np.sin(t + self.rng.random() * np.pi)
        noise = self.rng.normal(0, config.noise_std, n_points)
        return signal + noise

    def _inject_point_anomaly(self, data: np.ndarray, idx: int, config: SensorConfig) -> float:
        """Inject a sudden spike or drop at a single index.

        Args:
            data: Sensor data array.
            idx: Index to inject at.
            config: Sensor configuration for magnitude scaling.

        Returns:
            The anomalous value.
        """
        direction = self.rng.choice([-1, 1])
        magnitude = (config.normal_max - config.normal_min) * self.rng.uniform(0.5, 1.5)
        return float(data[idx] + direction * magnitude)

    def _inject_contextual_anomaly(
        self, data: np.ndarray, idx: int, config: SensorConfig
    ) -> float:
        """Inject a contextual anomaly (normal value at the wrong time).

        Args:
            data: Sensor data array.
            idx: Index to inject at.
            config: Sensor configuration (unused but kept for API consistency).

        Returns:
            The anomalous value swapped from a distant time index.
        """
        distant_idx = (idx + len(data) // 2) % len(data)
        return float(data[distant_idx])

    def _inject_collective_anomaly(
        self,
        data: np.ndarray,
        start_idx: int,
        duration: int,
        config: SensorConfig,
    ) -> np.ndarray:
        """Inject a gradual drift anomaly over a range of indices.

        Args:
            data: Sensor data array (modified in place).
            start_idx: Starting index for the drift.
            duration: Number of consecutive points affected.
            config: Sensor configuration for drift magnitude scaling.

        Returns:
            The modified data array.
        """
        drift = np.linspace(0, (config.normal_max - config.normal_min) * 0.8, duration)
        data[start_idx : start_idx + duration] += drift * self.rng.choice([-1, 1])
        return data

    def generate(
        self,
        days: int = 30,
        sampling_interval_minutes: int = 1,
        anomaly_rate: float = 0.05,
        sensors_per_type: int = 1,
    ) -> pd.DataFrame:
        """Generate a full synthetic IoT dataset.

        Args:
            days: Number of days of data to generate.
            sampling_interval_minutes: Minutes between consecutive readings.
            anomaly_rate: Fraction of readings to mark as anomalous.
            sensors_per_type: Number of individual sensors per sensor type.

        Returns:
            Long-format DataFrame with columns:
            ``timestamp``, ``sensor_type``, ``sensor_id``, ``value``, ``is_anomaly``.
        """
        n_points = days * 24 * 60 // sampling_interval_minutes
        start_time = datetime.now() - timedelta(days=days)
        timestamps = [
            start_time + timedelta(minutes=i * sampling_interval_minutes) for i in range(n_points)
        ]

        records: list[dict] = []
        for sensor_type, config in SENSOR_CONFIGS.items():
            for sensor_num in range(sensors_per_type):
                sensor_id = f"{sensor_type}_{sensor_num:02d}"
                values = self._generate_base_signal(config, n_points)
                labels = np.zeros(n_points, dtype=int)

                n_anomalies = int(n_points * anomaly_rate)
                anomaly_indices = self.rng.choice(n_points, n_anomalies, replace=False)

                for idx in anomaly_indices:
                    anomaly_type: AnomalyType = self.rng.choice(
                        ["point", "contextual", "collective"]
                    )
                    if anomaly_type == "point":
                        values[idx] = self._inject_point_anomaly(values, idx, config)
                        labels[idx] = 1
                    elif anomaly_type == "contextual":
                        values[idx] = self._inject_contextual_anomaly(values, idx, config)
                        labels[idx] = 1
                    else:
                        duration = min(60, n_points - idx)
                        values = self._inject_collective_anomaly(values, idx, duration, config)
                        labels[idx : idx + duration] = 1

                for i in range(n_points):
                    records.append(
                        {
                            "timestamp": timestamps[i],
                            "sensor_type": sensor_type,
                            "sensor_id": sensor_id,
                            "value": values[i],
                            "is_anomaly": labels[i],
                        }
                    )

        logger.info(
            "Generated %d records across %d sensor types", len(records), len(SENSOR_CONFIGS)
        )
        return pd.DataFrame(records)

    def save_to_csv(self, df: pd.DataFrame, path: str) -> None:
        """Persist the generated DataFrame to CSV.

        Args:
            df: The generated sensor DataFrame.
            path: Output file path (parent dirs created automatically).
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        logger.info("Saved %d rows to %s", len(df), path)
