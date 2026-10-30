"""
Anomaly Detection for FleetGuard AI.

Implements multiple anomaly detection approaches:
1. Threshold-based detection (rule-based)
2. Statistical anomaly detection (rolling statistics)
3. Isolation Forest (ML-based)
"""
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from collections import deque

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sqlalchemy.orm import Session

from app.models.database import (
    SensorReading, Anomaly, Vehicle, AlertSeverity, utcnow
)
from app.models.connection import get_db_session


class AnomalyType(Enum):
    THRESHOLD_BREACH = "threshold_breach"
    TREND_ANOMALY = "trend_anomaly"
    ISOLATION_FOREST = "isolation_forest"
    STATISTICAL_OUTLIER = "statistical_outlier"


# Default thresholds for commercial vehicles (demo values)
DEFAULT_THRESHOLDS = {
    "engine_temperature_c": {"warning": 105, "critical": 115},
    "engine_rpm": {"warning": 2500, "critical": 3000},
    "oil_pressure_kpa": {"warning": 250, "critical": 150},  # Lower is worse
    "coolant_temperature_c": {"warning": 100, "critical": 110},
    "vibration_mm_s": {"warning": 6.0, "critical": 10.0},
    "battery_voltage_v": {"warning": 12.0, "critical": 11.0},  # Lower is worse
    "fuel_consumption_l_per_100km": {"warning": 50, "critical": 60},
}


@dataclass
class AnomalyResult:
    """Result of anomaly detection."""
    sensor_name: str
    sensor_value: float
    anomaly_type: AnomalyType
    severity: AlertSeverity
    expected_min: Optional[float]
    expected_max: Optional[float]
    anomaly_score: Optional[float]
    description: str


class ThresholdDetector:
    """Rule-based threshold anomaly detector."""

    def __init__(self, thresholds: Dict = None):
        self.thresholds = thresholds or DEFAULT_THRESHOLDS

    def check(self, reading: SensorReading) -> List[AnomalyResult]:
        """Check a single reading against thresholds."""
        results = []
        sensor_map = {
            "engine_temperature_c": reading.engine_temperature_c,
            "engine_rpm": reading.engine_rpm,
            "oil_pressure_kpa": reading.oil_pressure_kpa,
            "coolant_temperature_c": reading.coolant_temperature_c,
            "vibration_mm_s": reading.vibration_mm_s,
            "battery_voltage_v": reading.battery_voltage_v,
            "fuel_consumption_l_per_100km": reading.fuel_consumption_l_per_100km,
        }

        for sensor_name, value in sensor_map.items():
            if sensor_name not in self.thresholds:
                continue

            thresh = self.thresholds[sensor_name]
            is_lower_better = sensor_name in ["oil_pressure_kpa", "battery_voltage_v"]

            if is_lower_better:
                if value <= thresh["critical"]:
                    severity = AlertSeverity.CRITICAL
                    desc = f"{sensor_name} critically low: {value:.1f} (threshold: {thresh['critical']})"
                elif value <= thresh["warning"]:
                    severity = AlertSeverity.HIGH
                    desc = f"{sensor_name} below warning threshold: {value:.1f} (threshold: {thresh['warning']})"
                else:
                    continue
            else:
                if value >= thresh["critical"]:
                    severity = AlertSeverity.CRITICAL
                    desc = f"{sensor_name} critically high: {value:.1f} (threshold: {thresh['critical']})"
                elif value >= thresh["warning"]:
                    severity = AlertSeverity.HIGH
                    desc = f"{sensor_name} above warning threshold: {value:.1f} (threshold: {thresh['warning']})"
                else:
                    continue

            results.append(AnomalyResult(
                sensor_name=sensor_name,
                sensor_value=value,
                anomaly_type=AnomalyType.THRESHOLD_BREACH,
                severity=severity,
                expected_min=thresh["warning"] if is_lower_better else None,
                expected_max=thresh["warning"] if not is_lower_better else None,
                anomaly_score=None,
                description=desc,
            ))

        return results


class StatisticalDetector:
    """Statistical anomaly detector using rolling statistics."""

    def __init__(self, window_size: int = 50, std_threshold: float = 3.0):
        self.window_size = window_size
        self.std_threshold = std_threshold
        self._history: Dict[int, deque] = {}  # vehicle_id -> deque of readings

    def _get_history(self, vehicle_id: int) -> deque:
        if vehicle_id not in self._history:
            self._history[vehicle_id] = deque(maxlen=self.window_size)
        return self._history[vehicle_id]

    def check(self, reading: SensorReading) -> List[AnomalyResult]:
        """Check reading against rolling statistics."""
        results = []
        history = self._get_history(reading.vehicle_id)

        # Convert reading to dict
        current = {
            "engine_temperature_c": reading.engine_temperature_c,
            "engine_rpm": reading.engine_rpm,
            "oil_pressure_kpa": reading.oil_pressure_kpa,
            "coolant_temperature_c": reading.coolant_temperature_c,
            "vibration_mm_s": reading.vibration_mm_s,
            "battery_voltage_v": reading.battery_voltage_v,
            "fuel_consumption_l_per_100km": reading.fuel_consumption_l_per_100km,
        }

        # Add to history
        history.append(current)

        # Need minimum samples
        if len(history) < 10:
            return results

        # Compute rolling stats
        df = pd.DataFrame(list(history))
        means = df.mean()
        stds = df.std()

        for sensor_name, value in current.items():
            if stds[sensor_name] == 0:
                continue

            z_score = abs(value - means[sensor_name]) / stds[sensor_name]

            if z_score > self.std_threshold:
                severity = AlertSeverity.HIGH if z_score < 4 else AlertSeverity.CRITICAL
                expected_min = means[sensor_name] - 3 * stds[sensor_name]
                expected_max = means[sensor_name] + 3 * stds[sensor_name]

                results.append(AnomalyResult(
                    sensor_name=sensor_name,
                    sensor_value=value,
                    anomaly_type=AnomalyType.STATISTICAL_OUTLIER,
                    severity=severity,
                    expected_min=expected_min,
                    expected_max=expected_max,
                    anomaly_score=z_score,
                    description=f"{sensor_name} statistical outlier: z-score={z_score:.2f} (mean={means[sensor_name]:.2f}, std={stds[sensor_name]:.2f})",
                ))

        return results


class IsolationForestDetector:
    """ML-based anomaly detector using Isolation Forest."""

    def __init__(self, contamination: float = 0.05, n_estimators: int = 100):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self._models: Dict[int, Pipeline] = {}
        self._feature_names = [
            "engine_temperature_c", "engine_rpm", "oil_pressure_kpa",
            "coolant_temperature_c", "vibration_mm_s", "battery_voltage_v",
            "fuel_consumption_l_per_100km"
        ]
        self._min_training_samples = 100

    def _get_model(self, vehicle_id: int) -> Pipeline:
        if vehicle_id not in self._models:
            self._models[vehicle_id] = Pipeline([
                ("scaler", StandardScaler()),
                ("iso_forest", IsolationForest(
                    contamination=self.contamination,
                    n_estimators=self.n_estimators,
                    random_state=42,
                    n_jobs=-1
                ))
            ])
        return self._models[vehicle_id]

    def train(self, vehicle_id: int, readings: List[SensorReading]) -> bool:
        """Train the model on historical data."""
        if len(readings) < self._min_training_samples:
            return False

        data = []
        for r in readings:
            data.append([
                r.engine_temperature_c, r.engine_rpm, r.oil_pressure_kpa,
                r.coolant_temperature_c, r.vibration_mm_s, r.battery_voltage_v,
                r.fuel_consumption_l_per_100km
            ])

        X = np.array(data)
        model = self._get_model(vehicle_id)
        model.fit(X)
        return True

    def check(self, reading: SensorReading) -> List[AnomalyResult]:
        """Check a reading using the trained model."""
        results = []
        vehicle_id = reading.vehicle_id

        if vehicle_id not in self._models:
            return results

        model = self._models[vehicle_id]
        X = np.array([[
            reading.engine_temperature_c, reading.engine_rpm, reading.oil_pressure_kpa,
            reading.coolant_temperature_c, reading.vibration_mm_s, reading.battery_voltage_v,
            reading.fuel_consumption_l_per_100km
        ]])

        try:
            # Get anomaly score (negative = more anomalous)
            score = model.decision_function(X)[0]
            prediction = model.predict(X)[0]

            if prediction == -1:  # Anomaly detected
                # Convert score to 0-1 scale (approximate)
                anomaly_score = 1 / (1 + np.exp(score))

                severity = AlertSeverity.HIGH if anomaly_score < 0.7 else AlertSeverity.CRITICAL

                results.append(AnomalyResult(
                    sensor_name="multi_sensor",
                    sensor_value=float(score),
                    anomaly_type=AnomalyType.ISOLATION_FOREST,
                    severity=severity,
                    expected_min=None,
                    expected_max=None,
                    anomaly_score=anomaly_score,
                    description=f"Isolation Forest anomaly detected: score={score:.4f}, anomaly_score={anomaly_score:.4f}",
                ))
        except Exception:
            pass

        return results


class TrendDetector:
    """Detects anomalous trends in sensor data."""

    def __init__(self, window_size: int = 20, trend_threshold: float = 0.05):
        self.window_size = window_size
        self.trend_threshold = trend_threshold
        self._history: Dict[int, deque] = {}

    def _get_history(self, vehicle_id: int) -> deque:
        if vehicle_id not in self._history:
            self._history[vehicle_id] = deque(maxlen=self.window_size)
        return self._history[vehicle_id]

    def check(self, reading: SensorReading) -> List[AnomalyResult]:
        """Check for anomalous trends."""
        results = []
        history = self._get_history(reading.vehicle_id)

        current = {
            "engine_temperature_c": reading.engine_temperature_c,
            "engine_rpm": reading.engine_rpm,
            "oil_pressure_kpa": reading.oil_pressure_kpa,
            "coolant_temperature_c": reading.coolant_temperature_c,
            "vibration_mm_s": reading.vibration_mm_s,
            "battery_voltage_v": reading.battery_voltage_v,
            "fuel_consumption_l_per_100km": reading.fuel_consumption_l_per_100km,
        }

        history.append(current)

        if len(history) < self.window_size:
            return results

        df = pd.DataFrame(list(history))
        x = np.arange(len(df))

        for sensor_name in current.keys():
            y = df[sensor_name].values
            if np.std(y) == 0:
                continue

            # Linear regression slope
            slope = np.polyfit(x, y, 1)[0]
            normalized_slope = slope / (np.mean(y) + 1e-6)

            if abs(normalized_slope) > self.trend_threshold:
                direction = "increasing" if normalized_slope > 0 else "decreasing"

                # Determine severity based on sensor and direction
                severity = AlertSeverity.MEDIUM
                if sensor_name in ["oil_pressure_kpa", "battery_voltage_v"] and direction == "decreasing":
                    severity = AlertSeverity.HIGH
                elif sensor_name in ["engine_temperature_c", "coolant_temperature_c", "vibration_mm_s"] and direction == "increasing":
                    severity = AlertSeverity.HIGH

                results.append(AnomalyResult(
                    sensor_name=sensor_name,
                    sensor_value=current[sensor_name],
                    anomaly_type=AnomalyType.TREND_ANOMALY,
                    severity=severity,
                    expected_min=None,
                    expected_max=None,
                    anomaly_score=abs(normalized_slope),
                    description=f"{sensor_name} showing {direction} trend: slope={normalized_slope:.6f} per reading",
                ))

        return results


class AnomalyDetector:
    """
    Composite anomaly detector combining multiple detection strategies.
    """

    def __init__(
        self,
        thresholds: Dict = None,
        enable_statistical: bool = True,
        enable_isolation_forest: bool = True,
        enable_trend: bool = True,
    ):
        self.threshold_detector = ThresholdDetector(thresholds)
        self.statistical_detector = StatisticalDetector() if enable_statistical else None
        self.isolation_forest_detector = IsolationForestDetector() if enable_isolation_forest else None
        self.trend_detector = TrendDetector() if enable_trend else None

        self._training_sessions: Dict[int, int] = {}  # vehicle_id -> samples trained

    def detect(self, reading: SensorReading) -> List[AnomalyResult]:
        """Run all detectors on a reading."""
        all_results = []

        # Threshold detection (always runs)
        all_results.extend(self.threshold_detector.check(reading))

        # Statistical detection
        if self.statistical_detector:
            all_results.extend(self.statistical_detector.check(reading))

        # Trend detection
        if self.trend_detector:
            all_results.extend(self.trend_detector.check(reading))

        # Isolation Forest (if trained)
        if self.isolation_forest_detector:
            all_results.extend(self.isolation_forest_detector.check(reading))

        return all_results

    def train_isolation_forest(self, vehicle_id: int, session: Session) -> bool:
        """Train Isolation Forest on historical data for a vehicle."""
        if not self.isolation_forest_detector:
            return False

        readings = session.query(SensorReading).filter(
            SensorReading.vehicle_id == vehicle_id
        ).order_by(SensorReading.timestamp.desc()).limit(500).all()

        if len(readings) < 100:
            return False

        readings.reverse()  # Chronological order
        return self.isolation_forest_detector.train(vehicle_id, readings)

    def train_all_vehicles(self, session: Session) -> Dict[int, bool]:
        """Train models for all vehicles."""
        vehicles = session.query(Vehicle).filter(Vehicle.is_active == True).all()
        results = {}
        for vehicle in vehicles:
            results[vehicle.id] = self.train_isolation_forest(vehicle.id, session)
        return results


def persist_anomalies(session: Session, reading: SensorReading, anomalies: List[AnomalyResult]) -> List[Anomaly]:
    """Persist detected anomalies to database."""
    persisted = []
    for anomaly_result in anomalies:
        anomaly = Anomaly(
            vehicle_id=reading.vehicle_id,
            sensor_reading_id=reading.id,
            timestamp=reading.timestamp,
            anomaly_type=anomaly_result.anomaly_type.value,
            sensor_name=anomaly_result.sensor_name,
            sensor_value=anomaly_result.sensor_value,
            expected_range_min=anomaly_result.expected_min,
            expected_range_max=anomaly_result.expected_max,
            anomaly_score=anomaly_result.anomaly_score,
            severity=anomaly_result.severity,
            description=anomaly_result.description,
        )
        session.add(anomaly)
        persisted.append(anomaly)

    session.flush()
    return persisted


def get_recent_anomalies(
    session: Session,
    vehicle_id: Optional[int] = None,
    hours: int = 24,
    severity: Optional[AlertSeverity] = None,
    limit: int = 100
) -> List[Anomaly]:
    """Get recent anomalies from database."""
    query = session.query(Anomaly).filter(
        Anomaly.timestamp >= utcnow() - timedelta(hours=hours)
    )

    if vehicle_id:
        query = query.filter(Anomaly.vehicle_id == vehicle_id)

    if severity:
        query = query.filter(Anomaly.severity == severity)

    return query.order_by(Anomaly.timestamp.desc()).limit(limit).all()