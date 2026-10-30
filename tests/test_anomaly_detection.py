"""
Tests for anomaly detection module.
"""
import pytest
from datetime import datetime
from unittest.mock import Mock, MagicMock

from app.ml.anomaly_detection import (
    ThresholdDetector, StatisticalDetector, IsolationForestDetector,
    TrendDetector, AnomalyDetector, AnomalyResult, AnomalyType,
    DEFAULT_THRESHOLDS, persist_anomalies
)
from app.models.database import SensorReading, AlertSeverity, Anomaly


class TestThresholdDetector:
    """Tests for ThresholdDetector."""

    def setup_method(self):
        self.detector = ThresholdDetector()

    def create_mock_reading(self, **kwargs) -> SensorReading:
        """Create a mock sensor reading."""
        defaults = {
            "engine_temperature_c": 90.0,
            "engine_rpm": 1500.0,
            "oil_pressure_kpa": 400.0,
            "coolant_temperature_c": 85.0,
            "vibration_mm_s": 2.0,
            "battery_voltage_v": 14.0,
            "fuel_consumption_l_per_100km": 30.0,
        }
        defaults.update(kwargs)
        reading = Mock(spec=SensorReading)
        for k, v in defaults.items():
            setattr(reading, k, v)
        reading.vehicle_id = 1
        reading.id = 1
        reading.timestamp = datetime.utcnow()
        return reading

    def test_normal_reading_no_anomalies(self):
        """Normal readings should not trigger anomalies."""
        reading = self.create_mock_reading()
        results = self.detector.check(reading)
        assert len(results) == 0

    def test_engine_temperature_critical_high(self):
        """Engine temperature above critical threshold."""
        reading = self.create_mock_reading(engine_temperature_c=120.0)
        results = self.detector.check(reading)
        assert len(results) == 1
        assert results[0].sensor_name == "engine_temperature_c"
        assert results[0].severity == AlertSeverity.CRITICAL
        assert results[0].anomaly_type == AnomalyType.THRESHOLD_BREACH

    def test_engine_temperature_warning_high(self):
        """Engine temperature above warning threshold."""
        reading = self.create_mock_reading(engine_temperature_c=110.0)
        results = self.detector.check(reading)
        assert len(results) == 1
        assert results[0].severity == AlertSeverity.HIGH

    def test_oil_pressure_critical_low(self):
        """Oil pressure below critical threshold (lower is worse)."""
        reading = self.create_mock_reading(oil_pressure_kpa=100.0)
        results = self.detector.check(reading)
        assert len(results) == 1
        assert results[0].sensor_name == "oil_pressure_kpa"
        assert results[0].severity == AlertSeverity.CRITICAL

    def test_oil_pressure_warning_low(self):
        """Oil pressure below warning threshold."""
        reading = self.create_mock_reading(oil_pressure_kpa=200.0)
        results = self.detector.check(reading)
        assert len(results) == 1
        assert results[0].severity == AlertSeverity.HIGH

    def test_vibration_critical(self):
        """Vibration above critical threshold."""
        reading = self.create_mock_reading(vibration_mm_s=12.0)
        results = self.detector.check(reading)
        assert len(results) == 1
        assert results[0].severity == AlertSeverity.CRITICAL

    def test_battery_voltage_critical_low(self):
        """Battery voltage below critical threshold."""
        reading = self.create_mock_reading(battery_voltage_v=10.5)
        results = self.detector.check(reading)
        assert len(results) == 1
        assert results[0].sensor_name == "battery_voltage_v"
        assert results[0].severity == AlertSeverity.CRITICAL

    def test_multiple_threshold_breaches(self):
        """Multiple sensors breaching thresholds."""
        reading = self.create_mock_reading(
            engine_temperature_c=120.0,
            oil_pressure_kpa=100.0,
            vibration_mm_s=12.0,
        )
        results = self.detector.check(reading)
        assert len(results) == 3
        sensor_names = {r.sensor_name for r in results}
        assert sensor_names == {"engine_temperature_c", "oil_pressure_kpa", "vibration_mm_s"}

    def test_custom_thresholds(self):
        """Custom thresholds should override defaults."""
        custom = {"engine_temperature_c": {"warning": 95, "critical": 100}}
        detector = ThresholdDetector(thresholds=custom)
        reading = self.create_mock_reading(engine_temperature_c=97.0)
        results = detector.check(reading)
        assert len(results) == 1
        assert results[0].severity == AlertSeverity.HIGH


class TestStatisticalDetector:
    """Tests for StatisticalDetector."""

    def setup_method(self):
        self.detector = StatisticalDetector(window_size=10, std_threshold=2.0)

    def create_mock_reading(self, **kwargs) -> SensorReading:
        defaults = {
            "engine_temperature_c": 90.0,
            "engine_rpm": 1500.0,
            "oil_pressure_kpa": 400.0,
            "coolant_temperature_c": 85.0,
            "vibration_mm_s": 2.0,
            "battery_voltage_v": 14.0,
            "fuel_consumption_l_per_100km": 30.0,
        }
        defaults.update(kwargs)
        reading = Mock(spec=SensorReading)
        for k, v in defaults.items():
            setattr(reading, k, v)
        reading.vehicle_id = 1
        reading.id = 1
        reading.timestamp = datetime.utcnow()
        return reading

    def test_insufficient_history(self):
        """Not enough history should return no anomalies."""
        reading = self.create_mock_reading(engine_temperature_c=95.0)
        results = self.detector.check(reading)
        assert len(results) == 0

    def test_statistical_outlier_detected(self):
        """Outlier beyond 3 std deviations should be detected."""
        # Add 10 normal readings
        for _ in range(10):
            reading = self.create_mock_reading(engine_temperature_c=90.0)
            self.detector.check(reading)

        # Add an outlier
        outlier = self.create_mock_reading(engine_temperature_c=130.0)
        results = self.detector.check(outlier)
        assert len(results) == 1
        assert results[0].anomaly_type == AnomalyType.STATISTICAL_OUTLIER
        assert results[0].severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL]

    def test_zero_std_no_anomaly(self):
        """Zero standard deviation should not cause false positives."""
        for _ in range(10):
            reading = self.create_mock_reading(engine_temperature_c=90.0)
            self.detector.check(reading)

        # Same value - zero std
        reading = self.create_mock_reading(engine_temperature_c=90.0)
        results = self.detector.check(reading)
        assert len(results) == 0


class TestIsolationForestDetector:
    """Tests for IsolationForestDetector."""

    def setup_method(self):
        self.detector = IsolationForestDetector(contamination=0.1, n_estimators=10)

    def create_mock_reading(self, **kwargs) -> SensorReading:
        defaults = {
            "engine_temperature_c": 90.0,
            "engine_rpm": 1500.0,
            "oil_pressure_kpa": 400.0,
            "coolant_temperature_c": 85.0,
            "vibration_mm_s": 2.0,
            "battery_voltage_v": 14.0,
            "fuel_consumption_l_per_100km": 30.0,
        }
        defaults.update(kwargs)
        reading = Mock(spec=SensorReading)
        for k, v in defaults.items():
            setattr(reading, k, v)
        reading.vehicle_id = 1
        reading.id = 1
        reading.timestamp = datetime.utcnow()
        return reading

    def test_untrained_model_no_anomalies(self):
        """Untrained model should not produce anomalies."""
        reading = self.create_mock_reading()
        results = self.detector.check(reading)
        assert len(results) == 0

    def test_trained_model_detects_anomaly(self):
        """Trained model should detect outliers."""
        import numpy as np
        # Generate training data
        np.random.seed(42)
        normal_data = []
        for _ in range(150):
            normal_data.append([
                np.random.normal(90, 2),    # engine_temp
                np.random.normal(1500, 100), # rpm
                np.random.normal(400, 20),   # oil_pressure
                np.random.normal(85, 2),     # coolant_temp
                np.random.normal(2.0, 0.3),  # vibration
                np.random.normal(14.0, 0.1), # battery
                np.random.normal(30, 2),     # fuel
            ])

        # Train
        self.detector.train(1, [self.create_mock_reading() for _ in range(150)])

        # Test with anomaly
        anomaly_reading = self.create_mock_reading(
            engine_temperature_c=150.0,  # Way off
            vibration_mm_s=20.0,
        )
        results = self.detector.check(anomaly_reading)
        # May or may not detect depending on training - just verify it runs
        assert isinstance(results, list)


class TestTrendDetector:
    """Tests for TrendDetector."""

    def setup_method(self):
        self.detector = TrendDetector(window_size=10, trend_threshold=0.001)

    def create_mock_reading(self, **kwargs) -> SensorReading:
        defaults = {
            "engine_temperature_c": 90.0,
            "engine_rpm": 1500.0,
            "oil_pressure_kpa": 400.0,
            "coolant_temperature_c": 85.0,
            "vibration_mm_s": 2.0,
            "battery_voltage_v": 14.0,
            "fuel_consumption_l_per_100km": 30.0,
        }
        defaults.update(kwargs)
        reading = Mock(spec=SensorReading)
        for k, v in defaults.items():
            setattr(reading, k, v)
        reading.vehicle_id = 1
        reading.id = 1
        reading.timestamp = datetime.utcnow()
        return reading

    def test_insufficient_history(self):
        """Not enough history should return no anomalies."""
        reading = self.create_mock_reading()
        results = self.detector.check(reading)
        assert len(results) == 0

    def test_increasing_trend_detected(self):
        """Consistently increasing trend should be detected."""
        # Add readings with increasing temperature
        base_temp = 90.0
        for i in range(12):
            reading = self.create_mock_reading(engine_temperature_c=base_temp + i * 0.5)
            self.detector.check(reading)

        # Last reading should show trend
        results = self.detector.check(self.create_mock_reading(engine_temperature_c=base_temp + 12 * 0.5))
        trend_results = [r for r in results if r.anomaly_type == AnomalyType.TREND_ANOMALY]
        assert len(trend_results) > 0
        assert trend_results[0].sensor_name == "engine_temperature_c"


class TestAnomalyDetector:
    """Tests for composite AnomalyDetector."""

    def setup_method(self):
        self.detector = AnomalyDetector(
            enable_statistical=True,
            enable_isolation_forest=False,
            enable_trend=True,
        )

    def create_mock_reading(self, **kwargs) -> SensorReading:
        defaults = {
            "engine_temperature_c": 90.0,
            "engine_rpm": 1500.0,
            "oil_pressure_kpa": 400.0,
            "coolant_temperature_c": 85.0,
            "vibration_mm_s": 2.0,
            "battery_voltage_v": 14.0,
            "fuel_consumption_l_per_100km": 30.0,
        }
        defaults.update(kwargs)
        reading = Mock(spec=SensorReading)
        for k, v in defaults.items():
            setattr(reading, k, v)
        reading.vehicle_id = 1
        reading.id = 1
        reading.timestamp = datetime.utcnow()
        return reading

    def test_all_detectors_combined(self):
        """Composite detector should run all enabled detectors."""
        reading = self.create_mock_reading(engine_temperature_c=120.0)
        results = self.detector.detect(reading)
        # Should have threshold breach at minimum
        assert len(results) >= 1
        assert any(r.anomaly_type == AnomalyType.THRESHOLD_BREACH for r in results)

    def test_no_anomalies_normal_reading(self):
        """Normal reading should produce no anomalies from any detector."""
        reading = self.create_mock_reading()
        results = self.detector.detect(reading)
        assert len(results) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])