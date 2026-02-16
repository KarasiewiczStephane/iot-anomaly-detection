"""Tests for synthetic IoT data generator."""

import numpy as np
import pandas as pd
import pytest

from src.data.generator import SENSOR_CONFIGS, IoTDataGenerator, SensorConfig


@pytest.fixture()
def generator() -> IoTDataGenerator:
    return IoTDataGenerator(seed=42)


class TestSensorConfig:
    def test_default_noise_std(self) -> None:
        cfg = SensorConfig("test", 0.0, 1.0)
        assert cfg.noise_std == 0.02

    def test_custom_noise_std(self) -> None:
        cfg = SensorConfig("test", 0.0, 1.0, noise_std=0.5)
        assert cfg.noise_std == 0.5

    def test_all_sensor_types_defined(self) -> None:
        expected = {"temperature", "vibration", "pressure", "power"}
        assert set(SENSOR_CONFIGS.keys()) == expected


class TestBaseSignal:
    def test_output_length(self, generator: IoTDataGenerator) -> None:
        config = SENSOR_CONFIGS["temperature"]
        signal = generator._generate_base_signal(config, 100)
        assert len(signal) == 100

    def test_values_within_reasonable_range(self, generator: IoTDataGenerator) -> None:
        config = SENSOR_CONFIGS["temperature"]
        signal = generator._generate_base_signal(config, 1000)
        # With noise, values should roughly be within range ± some margin
        assert signal.min() > config.normal_min - 10 * config.noise_std
        assert signal.max() < config.normal_max + 10 * config.noise_std


class TestAnomalyInjection:
    def test_point_anomaly_deviates(self, generator: IoTDataGenerator) -> None:
        config = SENSOR_CONFIGS["temperature"]
        data = generator._generate_base_signal(config, 100)
        original = data[50]
        anomalous = generator._inject_point_anomaly(data, 50, config)
        assert anomalous != original

    def test_contextual_anomaly_swaps(self, generator: IoTDataGenerator) -> None:
        config = SENSOR_CONFIGS["temperature"]
        data = np.arange(100, dtype=float)
        result = generator._inject_contextual_anomaly(data, 10, config)
        assert result == data[60]  # idx + len//2

    def test_collective_anomaly_affects_range(self, generator: IoTDataGenerator) -> None:
        config = SENSOR_CONFIGS["temperature"]
        data = np.ones(100) * 70.0
        original = data.copy()
        generator._inject_collective_anomaly(data, 20, 10, config)
        # At least some points in the range should differ from original
        assert not np.array_equal(data[20:30], original[20:30])


class TestGenerate:
    def test_row_count_1_day(self, generator: IoTDataGenerator) -> None:
        df = generator.generate(days=1, sensors_per_type=1)
        expected = 1 * 24 * 60 * len(SENSOR_CONFIGS)  # 1 day, 4 sensors
        assert len(df) == expected

    def test_columns_present(self, generator: IoTDataGenerator) -> None:
        df = generator.generate(days=1)
        expected_cols = {"timestamp", "sensor_type", "sensor_id", "value", "is_anomaly"}
        assert set(df.columns) == expected_cols

    def test_anomaly_rate_approximate(self, generator: IoTDataGenerator) -> None:
        df = generator.generate(days=7, anomaly_rate=0.05)
        actual_rate = df["is_anomaly"].mean()
        # Collective anomalies expand labels, so rate will be >= configured
        assert actual_rate > 0.01

    def test_sensor_ids_format(self, generator: IoTDataGenerator) -> None:
        df = generator.generate(days=1, sensors_per_type=2)
        ids = df["sensor_id"].unique()
        assert "temperature_00" in ids
        assert "temperature_01" in ids

    def test_timestamps_sequential(self, generator: IoTDataGenerator) -> None:
        df = generator.generate(days=1)
        temp_df = df[df["sensor_id"] == "temperature_00"]
        timestamps = pd.to_datetime(temp_df["timestamp"])
        diffs = timestamps.diff().dropna()
        # All diffs should be exactly 1 minute
        assert (diffs == pd.Timedelta(minutes=1)).all()

    def test_reproducibility(self) -> None:
        gen1 = IoTDataGenerator(seed=123)
        gen2 = IoTDataGenerator(seed=123)
        df1 = gen1.generate(days=1)
        df2 = gen2.generate(days=1)
        # Timestamps differ due to datetime.now(); compare values and labels
        np.testing.assert_array_equal(df1["value"].values, df2["value"].values)
        np.testing.assert_array_equal(df1["is_anomaly"].values, df2["is_anomaly"].values)
        np.testing.assert_array_equal(df1["sensor_id"].values, df2["sensor_id"].values)

    def test_multiple_sensors_per_type(self, generator: IoTDataGenerator) -> None:
        df = generator.generate(days=1, sensors_per_type=3)
        for sensor_type in SENSOR_CONFIGS:
            type_ids = df[df["sensor_type"] == sensor_type]["sensor_id"].nunique()
            assert type_ids == 3


class TestSaveToCsv:
    def test_save_creates_file(self, generator: IoTDataGenerator, tmp_path) -> None:
        df = generator.generate(days=1)
        out = tmp_path / "output.csv"
        generator.save_to_csv(df, str(out))
        assert out.exists()

    def test_save_creates_parent_dirs(self, generator: IoTDataGenerator, tmp_path) -> None:
        df = generator.generate(days=1)
        out = tmp_path / "a" / "b" / "output.csv"
        generator.save_to_csv(df, str(out))
        assert out.exists()

    def test_saved_csv_roundtrip(self, generator: IoTDataGenerator, tmp_path) -> None:
        df = generator.generate(days=1)
        out = tmp_path / "output.csv"
        generator.save_to_csv(df, str(out))
        loaded = pd.read_csv(str(out))
        assert len(loaded) == len(df)
        assert set(loaded.columns) == set(df.columns)
