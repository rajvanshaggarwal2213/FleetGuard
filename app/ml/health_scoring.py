"""
Vehicle Health Scoring and Rule-Based Maintenance Triage for FleetGuard AI.

Implements:
1. Transparent health score calculation (0-100)
2. Rule-based maintenance triage (no LLM)
3. Alert generation and work order creation
"""
import json
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from collections import defaultdict

from sqlalchemy.orm import Session

from app.models.database import (
    Vehicle, SensorReading, VehicleHealthHistory, HealthStatus,
    MaintenanceAlert, WorkOrder, WorkOrderStatus, WorkOrderPriority,
    AlertSeverity, Anomaly, SimulationEvent, Technician, utcnow
)
from app.models.connection import get_db_session
from app.ml.anomaly_detection import AnomalyDetector, AnomalyResult, AnomalyType, DEFAULT_THRESHOLDS


# Health score weights
HEALTH_WEIGHTS = {
    "engine_temperature": 0.20,
    "oil_pressure": 0.20,
    "vibration": 0.15,
    "coolant_temperature": 0.10,
    "battery_voltage": 0.15,
    "fuel_consumption": 0.10,
    "maintenance_interval": 0.10,
}

# Healthy ranges for scoring (min, max) - values within range get full points
HEALTHY_RANGES = {
    "engine_temperature_c": (75, 100),
    "oil_pressure_kpa": (300, 500),
    "vibration_mm_s": (0, 4.0),
    "coolant_temperature_c": (70, 95),
    "battery_voltage_v": (12.5, 14.5),
    "fuel_consumption_l_per_100km": (20, 40),
}


@dataclass
class HealthFactor:
    """Individual factor contributing to health score."""
    name: str
    score: float  # 0-100
    weight: float
    status: str  # "normal", "warning", "critical"
    value: float
    expected_range: Tuple[float, float]
    description: str


@dataclass
class HealthAssessment:
    """Complete vehicle health assessment."""
    vehicle_id: int
    overall_score: float
    health_status: HealthStatus
    factors: List[HealthFactor]
    anomalies: List[AnomalyResult]
    timestamp: datetime
    recommendations: List[str]


class HealthScorer:
    """Calculate vehicle health score from sensor readings and anomalies."""

    def __init__(
        self,
        weights: Dict[str, float] = None,
        healthy_ranges: Dict[str, Tuple[float, float]] = None,
        anomaly_detector: AnomalyDetector = None
    ):
        self.weights = weights or HEALTH_WEIGHTS
        self.healthy_ranges = healthy_ranges or HEALTHY_RANGES
        self.anomaly_detector = anomaly_detector or AnomalyDetector()

    def calculate_sensor_score(self, sensor_name: str, value: float) -> Tuple[float, str]:
        """Calculate 0-100 score for a single sensor."""
        if sensor_name not in self.healthy_ranges:
            return 50.0, "unknown"

        min_val, max_val = self.healthy_ranges[sensor_name]
        range_span = max_val - min_val

        if min_val <= value <= max_val:
            # Within healthy range
            return 100.0, "normal"

        # Outside range - calculate penalty
        if value < min_val:
            deviation = (min_val - value) / range_span
        else:
            deviation = (value - max_val) / range_span

        # Exponential penalty
        score = max(0, 100 * (1 - deviation * 2))

        if deviation > 0.5:
            status = "critical"
        elif deviation > 0.2:
            status = "warning"
        else:
            status = "normal"

        return score, status

    def calculate_maintenance_interval_score(self, vehicle: Vehicle) -> Tuple[float, str]:
        """Calculate score based on time since last service."""
        if not vehicle.last_service_date:
            return 50.0, "unknown"

        days_since_service = (utcnow() - vehicle.last_service_date).days

        # Assume 90-day service interval
        if days_since_service <= 60:
            return 100.0, "normal"
        elif days_since_service <= 90:
            return 75.0, "warning"
        elif days_since_service <= 120:
            return 50.0, "warning"
        else:
            return 20.0, "critical"

    def assess_vehicle(self, vehicle: Vehicle, session: Session) -> HealthAssessment:
        """Perform complete health assessment for a vehicle."""
        # Get latest sensor reading
        latest_reading = session.query(SensorReading).filter(
            SensorReading.vehicle_id == vehicle.id
        ).order_by(SensorReading.timestamp.desc()).first()

        if not latest_reading:
            return HealthAssessment(
                vehicle_id=vehicle.id,
                overall_score=0.0,
                health_status=HealthStatus.CRITICAL,
                factors=[],
                anomalies=[],
                timestamp=utcnow(),
                recommendations=["No sensor data available"]
            )

        # Run anomaly detection
        anomalies = self.anomaly_detector.detect(latest_reading)

        # Calculate factor scores
        factors = []
        sensor_map = {
            "engine_temperature": ("engine_temperature_c", latest_reading.engine_temperature_c),
            "oil_pressure": ("oil_pressure_kpa", latest_reading.oil_pressure_kpa),
            "vibration": ("vibration_mm_s", latest_reading.vibration_mm_s),
            "coolant_temperature": ("coolant_temperature_c", latest_reading.coolant_temperature_c),
            "battery_voltage": ("battery_voltage_v", latest_reading.battery_voltage_v),
            "fuel_consumption": ("fuel_consumption_l_per_100km", latest_reading.fuel_consumption_l_per_100km),
        }

        for factor_name, (sensor_name, value) in sensor_map.items():
            weight = self.weights.get(factor_name, 0.1)
            score, status = self.calculate_sensor_score(sensor_name, value)
            expected_range = self.healthy_ranges.get(sensor_name, (0, 100))

            factor = HealthFactor(
                name=factor_name,
                score=score,
                weight=weight,
                status=status,
                value=value,
                expected_range=expected_range,
                description=f"{sensor_name}: {value:.1f} (healthy: {expected_range[0]}-{expected_range[1]})"
            )
            factors.append(factor)

        # Maintenance interval factor
        maint_score, maint_status = self.calculate_maintenance_interval_score(vehicle)
        maint_factor = HealthFactor(
            name="maintenance_interval",
            score=maint_score,
            weight=self.weights.get("maintenance_interval", 0.1),
            status=maint_status,
            value=(utcnow() - vehicle.last_service_date).days if vehicle.last_service_date else 0,
            expected_range=(0, 90),
            description=f"Days since last service: {(utcnow() - vehicle.last_service_date).days if vehicle.last_service_date else 'Unknown'}"
        )
        factors.append(maint_factor)

        # Calculate weighted overall score
        total_weight = sum(f.weight for f in factors)
        weighted_score = sum(f.score * f.weight for f in factors) / total_weight if total_weight > 0 else 0

        # Apply anomaly penalty
        critical_anomalies = [a for a in anomalies if a.severity == AlertSeverity.CRITICAL]
        high_anomalies = [a for a in anomalies if a.severity == AlertSeverity.HIGH]

        anomaly_penalty = len(critical_anomalies) * 15 + len(high_anomalies) * 8
        final_score = max(0, weighted_score - anomaly_penalty)

        # Determine health status
        if final_score >= 80:
            health_status = HealthStatus.HEALTHY
        elif final_score >= 50:
            health_status = HealthStatus.MAINTENANCE_APPROACHING
        else:
            health_status = HealthStatus.CRITICAL

        # Generate recommendations
        recommendations = self._generate_recommendations(factors, anomalies)

        return HealthAssessment(
            vehicle_id=vehicle.id,
            overall_score=round(final_score, 1),
            health_status=health_status,
            factors=factors,
            anomalies=anomalies,
            timestamp=utcnow(),
            recommendations=recommendations
        )

    def _generate_recommendations(
        self,
        factors: List[HealthFactor],
        anomalies: List[AnomalyResult]
    ) -> List[str]:
        """Generate maintenance recommendations."""
        recommendations = []

        # Factor-based recommendations
        for factor in factors:
            if factor.status == "critical":
                recommendations.append(f"URGENT: {factor.name} is critical ({factor.value:.1f}, expected {factor.expected_range[0]}-{factor.expected_range[1]})")
            elif factor.status == "warning":
                recommendations.append(f"ATTENTION: {factor.name} approaching limits ({factor.value:.1f}, expected {factor.expected_range[0]}-{factor.expected_range[1]})")

        # Anomaly-based recommendations
        for anomaly in anomalies:
            if anomaly.severity in [AlertSeverity.CRITICAL, AlertSeverity.HIGH]:
                recommendations.append(f"ANOMALY: {anomaly.description}")

        if not recommendations:
            recommendations.append("All systems operating normally. Continue routine monitoring.")

        return recommendations


class MaintenanceTriageEngine:
    """
    Rule-based maintenance triage engine.

    Deterministic rules for generating maintenance alerts and work orders.
    No LLM - purely rule-based for auditability and safety.
    """

    def __init__(self, session: Session):
        self.session = session
        self.health_scorer = HealthScorer()
        self.anomaly_detector = AnomalyDetector()

        # Alert deduplication window (hours)
        self.dedup_window_hours = 24

        # Rule configurations
        self.rules = self._initialize_rules()

    def _initialize_rules(self) -> List[Dict]:
        """Define maintenance triage rules."""
        return [
            {
                "id": "RULE_001",
                "name": "Engine Overheating Critical",
                "condition": lambda a: a.sensor_name == "engine_temperature_c" and a.severity == AlertSeverity.CRITICAL,
                "component": "Engine Cooling System",
                "severity": AlertSeverity.CRITICAL,
                "action": "Immediate engine shutdown and cooling system inspection required",
                "priority": WorkOrderPriority.URGENT,
            },
            {
                "id": "RULE_002",
                "name": "Engine Overheating Warning",
                "condition": lambda a: a.sensor_name == "engine_temperature_c" and a.severity == AlertSeverity.HIGH,
                "component": "Engine Cooling System",
                "severity": AlertSeverity.HIGH,
                "action": "Schedule cooling system inspection within 24 hours",
                "priority": WorkOrderPriority.HIGH,
            },
            {
                "id": "RULE_003",
                "name": "Low Oil Pressure Critical",
                "condition": lambda a: a.sensor_name == "oil_pressure_kpa" and a.severity == AlertSeverity.CRITICAL,
                "component": "Engine Lubrication System",
                "severity": AlertSeverity.CRITICAL,
                "action": "Immediate engine shutdown - risk of catastrophic damage",
                "priority": WorkOrderPriority.URGENT,
            },
            {
                "id": "RULE_004",
                "name": "Low Oil Pressure Warning",
                "condition": lambda a: a.sensor_name == "oil_pressure_kpa" and a.severity == AlertSeverity.HIGH,
                "component": "Engine Lubrication System",
                "severity": AlertSeverity.HIGH,
                "action": "Check oil level and pressure sensor, schedule inspection",
                "priority": WorkOrderPriority.HIGH,
            },
            {
                "id": "RULE_005",
                "name": "Abnormal Vibration",
                "condition": lambda a: a.sensor_name == "vibration_mm_s" and a.severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL],
                "component": "Engine/Transmission Mounts",
                "severity": AlertSeverity.HIGH,
                "action": "Inspect engine mounts, transmission, and rotating components",
                "priority": WorkOrderPriority.HIGH,
            },
            {
                "id": "RULE_006",
                "name": "Coolant Temperature High",
                "condition": lambda a: a.sensor_name == "coolant_temperature_c" and a.severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL],
                "component": "Cooling System",
                "severity": AlertSeverity.HIGH,
                "action": "Check coolant level, radiator, and thermostat",
                "priority": WorkOrderPriority.HIGH,
            },
            {
                "id": "RULE_007",
                "name": "Battery Voltage Low",
                "condition": lambda a: a.sensor_name == "battery_voltage_v" and a.severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL],
                "component": "Electrical System",
                "severity": AlertSeverity.HIGH,
                "action": "Test battery and charging system, replace if needed",
                "priority": WorkOrderPriority.MEDIUM,
            },
            {
                "id": "RULE_008",
                "name": "High Fuel Consumption",
                "condition": lambda a: a.sensor_name == "fuel_consumption_l_per_100km" and a.severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL],
                "component": "Fuel System/Engine",
                "severity": AlertSeverity.MEDIUM,
                "action": "Check fuel injectors, air filter, and engine tuning",
                "priority": WorkOrderPriority.MEDIUM,
            },
            {
                "id": "RULE_009",
                "name": "Statistical Outlier",
                "condition": lambda a: a.anomaly_type == AnomalyType.STATISTICAL_OUTLIER and a.severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL],
                "component": "Multiple Systems",
                "severity": AlertSeverity.MEDIUM,
                "action": "Investigate anomalous sensor pattern",
                "priority": WorkOrderPriority.MEDIUM,
            },
            {
                "id": "RULE_010",
                "name": "Isolation Forest Anomaly",
                "condition": lambda a: a.anomaly_type == AnomalyType.ISOLATION_FOREST and a.severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL],
                "component": "Multiple Systems",
                "severity": AlertSeverity.HIGH,
                "action": "ML model detected unusual multi-sensor pattern - comprehensive inspection recommended",
                "priority": WorkOrderPriority.HIGH,
            },
            {
                "id": "RULE_011",
                "name": "Degradation Trend",
                "condition": lambda a: a.anomaly_type == AnomalyType.TREND_ANOMALY and a.severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL],
                "component": "Multiple Systems",
                "severity": AlertSeverity.MEDIUM,
                "action": "Monitor trend - schedule preventive maintenance if continues",
                "priority": WorkOrderPriority.MEDIUM,
            },
            {
                "id": "RULE_012",
                "name": "Multiple Simultaneous Anomalies",
                "condition": lambda anomalies: len([a for a in anomalies if a.severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL]]) >= 3,
                "component": "Multiple Systems",
                "severity": AlertSeverity.CRITICAL,
                "action": "Multiple critical anomalies detected - immediate comprehensive inspection",
                "priority": WorkOrderPriority.URGENT,
            },
        ]

    def process_vehicle(self, vehicle: Vehicle) -> List[MaintenanceAlert]:
        """Process a vehicle through the triage engine."""
        # Get latest reading
        latest_reading = self.session.query(SensorReading).filter(
            SensorReading.vehicle_id == vehicle.id
        ).order_by(SensorReading.timestamp.desc()).first()

        if not latest_reading:
            return []

        # Detect anomalies
        anomalies = self.anomaly_detector.detect(latest_reading)

        # Run health assessment
        assessment = self.health_scorer.assess_vehicle(vehicle, self.session)

        # Update vehicle health in database
        vehicle.health_score = assessment.overall_score
        vehicle.current_health_status = assessment.health_status
        vehicle.updated_at = utcnow()

        # Save health history
        health_history = VehicleHealthHistory(
            vehicle_id=vehicle.id,
            timestamp=assessment.timestamp,
            health_score=assessment.overall_score,
            health_status=assessment.health_status,
            contributing_factors=json.dumps([
                {
                    "name": f.name,
                    "score": f.score,
                    "status": f.status,
                    "value": f.value,
                } for f in assessment.factors
            ])
        )
        self.session.add(health_history)

        # Persist anomalies
        for anomaly_result in anomalies:
            anomaly = Anomaly(
                vehicle_id=vehicle.id,
                sensor_reading_id=latest_reading.id,
                timestamp=latest_reading.timestamp,
                anomaly_type=anomaly_result.anomaly_type.value,
                sensor_name=anomaly_result.sensor_name,
                sensor_value=anomaly_result.sensor_value,
                expected_range_min=anomaly_result.expected_min,
                expected_range_max=anomaly_result.expected_max,
                anomaly_score=anomaly_result.anomaly_score,
                severity=anomaly_result.severity,
                description=anomaly_result.description,
            )
            self.session.add(anomaly)
        self.session.flush()

        # Apply rules to generate alerts
        new_alerts = self._apply_rules(vehicle, anomalies, assessment, latest_reading)

        # Save alerts
        for alert in new_alerts:
            self.session.add(alert)
        self.session.commit()

        return new_alerts

    def _apply_rules(
        self,
        vehicle: Vehicle,
        anomalies: List[AnomalyResult],
        assessment: HealthAssessment,
        reading: SensorReading
    ) -> List[MaintenanceAlert]:
        """Apply triage rules to generate alerts."""
        new_alerts = []

        # Check single-anomaly rules
        for rule in self.rules:
            if rule["id"] == "RULE_012":  # Multi-anomaly rule handled separately
                continue

            for anomaly in anomalies:
                try:
                    if rule["condition"](anomaly):
                        # Check deduplication
                        if not self._is_duplicate_alert(vehicle.id, rule["id"]):
                            alert = self._create_alert(vehicle, anomaly, rule, reading)
                            new_alerts.append(alert)
                except Exception:
                    continue

        # Check multi-anomaly rule
        multi_rule = next(r for r in self.rules if r["id"] == "RULE_012")
        try:
            if multi_rule["condition"](anomalies):
                if not self._is_duplicate_alert(vehicle.id, multi_rule["id"]):
                    # Create composite alert
                    alert = MaintenanceAlert(
                        vehicle_id=vehicle.id,
                        timestamp=utcnow(),
                        alert_type="multiple_anomalies",
                        component=multi_rule["component"],
                        severity=multi_rule["severity"],
                        title=f"Multiple Critical Anomalies - {vehicle.vehicle_id}",
                        description=multi_rule["action"],
                        recommended_action=multi_rule["action"],
                        evidence=json.dumps([a.description for a in anomalies if a.severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL]]),
                    )
                    new_alerts.append(alert)
        except Exception:
            pass

        return new_alerts

    def _is_duplicate_alert(self, vehicle_id: int, rule_id: str) -> bool:
        """Check if similar alert was recently created."""
        cutoff = utcnow() - timedelta(hours=self.dedup_window_hours)
        existing = self.session.query(MaintenanceAlert).filter(
            MaintenanceAlert.vehicle_id == vehicle_id,
            MaintenanceAlert.timestamp >= cutoff,
            MaintenanceAlert.alert_type.like(f"%{rule_id}%")
        ).first()
        return existing is not None

    def _create_alert(
        self,
        vehicle: Vehicle,
        anomaly: AnomalyResult,
        rule: Dict,
        reading: SensorReading
    ) -> MaintenanceAlert:
        """Create maintenance alert from rule match."""
        alert = MaintenanceAlert(
            vehicle_id=vehicle.id,
            anomaly_id=None,  # Will be set after flush
            timestamp=utcnow(),
            alert_type=rule["id"],
            component=rule["component"],
            severity=rule["severity"],
            title=f"{rule['name']} - {vehicle.vehicle_id}",
            description=rule["action"],
            recommended_action=rule["action"],
            evidence=json.dumps({
                "sensor": anomaly.sensor_name,
                "value": anomaly.sensor_value,
                "anomaly_type": anomaly.anomaly_type.value,
                "anomaly_score": anomaly.anomaly_score,
                "rule_id": rule["id"],
            }),
        )
        return alert


def auto_generate_work_orders(session: Session) -> List[WorkOrder]:
    """Automatically generate work orders from unresolved critical/high alerts.

    Idempotent: the query filter ``MaintenanceAlert.work_order == None`` excludes
    alerts that already have a linked work order, and the unique constraint on
    ``alert_id`` (one work order per alert) is enforced at the database level.
    Duplicate work orders are never created for the same alert.
    """
    new_work_orders = []

    # Find alerts needing work orders (no existing work order, unresolved, critical/high)
    alerts = session.query(MaintenanceAlert).filter(
        MaintenanceAlert.is_resolved == False,
        MaintenanceAlert.severity.in_([AlertSeverity.CRITICAL, AlertSeverity.HIGH]),
        MaintenanceAlert.work_order == None  # No existing work order
    ).all()

    for alert in alerts:
        # Determine priority from severity
        priority_map = {
            AlertSeverity.CRITICAL: WorkOrderPriority.URGENT,
            AlertSeverity.HIGH: WorkOrderPriority.HIGH,
            AlertSeverity.MEDIUM: WorkOrderPriority.MEDIUM,
            AlertSeverity.LOW: WorkOrderPriority.LOW,
        }

        # Auto-assign technician based on component (simplified)
        technician = session.query(Technician).filter(
            Technician.is_active == True,
            Technician.specialization.ilike(f"%{alert.component.split()[0]}%")
        ).first()

        work_order = WorkOrder(
            work_order_id=f"WO-{utcnow().strftime('%Y%m%d%H%M%S')}-{alert.vehicle_id}",
            vehicle_id=alert.vehicle_id,
            alert_id=alert.id,
            technician_id=technician.id if technician else None,
            component=alert.component,
            issue_description=alert.description,
            priority=priority_map.get(alert.severity, WorkOrderPriority.MEDIUM),
            recommended_inspection=alert.recommended_action,
            status=WorkOrderStatus.OPEN,
        )
        session.add(work_order)
        new_work_orders.append(work_order)

    session.commit()
    return new_work_orders


def process_fleet_health(session: Session) -> Dict[str, Any]:
    """Process health for all active vehicles."""
    vehicles = session.query(Vehicle).filter(Vehicle.is_active == True).all()

    results = {
        "processed": 0,
        "alerts_generated": 0,
        "work_orders_created": 0,
        "vehicles_by_status": defaultdict(int),
    }

    triage_engine = MaintenanceTriageEngine(session)

    for vehicle in vehicles:
        alerts = triage_engine.process_vehicle(vehicle)
        results["processed"] += 1
        results["alerts_generated"] += len(alerts)
        results["vehicles_by_status"][vehicle.current_health_status.value] += 1

    # Generate work orders
    work_orders = auto_generate_work_orders(session)
    results["work_orders_created"] = len(work_orders)

    session.commit()
    return results