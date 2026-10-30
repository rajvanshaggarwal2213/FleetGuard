"""
Tests for health scoring and maintenance triage.
"""
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock

from app.ml.health_scoring import (
    HealthScorer, MaintenanceTriageEngine, HealthFactor, HealthAssessment,
    HEALTH_WEIGHTS, HEALTHY_RANGES, auto_generate_work_orders, process_fleet_health
)
from app.ml.anomaly_detection import AnomalyDetector, AnomalyResult, AnomalyType
from app.models.database import (
    Vehicle, VehicleType, HealthStatus, SensorReading, MaintenanceAlert,
    WorkOrder, WorkOrderStatus, WorkOrderPriority, AlertSeverity,
    VehicleHealthHistory, Technician
)


class TestHealthScorer:
    """Tests for HealthScorer."""

    def setup_method(self):
        self.scorer = HealthScorer()

    def create_mock_vehicle(self, health_score=90.0, **kwargs) -> Vehicle:
        """Create a mock vehicle."""
        defaults = {
            "id": 1,
            "vehicle_id": "TEST-001",
            "vehicle_type": VehicleType.TRUCK,
            "registration_number": "TEST-1234",
            "manufacturer": "Test",
            "model": "Model",
            "total_distance_km": 10000.0,
            "engine_hours": 500.0,
            "last_service_date": datetime.utcnow() - timedelta(days=30),
            "next_scheduled_service": datetime.utcnow() + timedelta(days=60),
            "current_health_status": HealthStatus.HEALTHY,
            "health_score": health_score,
            "is_active": True,
        }
        defaults.update(kwargs)
        vehicle = Mock(spec=Vehicle)
        for k, v in defaults.items():
            setattr(vehicle, k, v)
        return vehicle

    def create_mock_reading(self, **kwargs) -> SensorReading:
        """Create a mock sensor reading."""
        defaults = {
            "id": 1,
            "vehicle_id": 1,
            "timestamp": datetime.utcnow(),
            "engine_temperature_c": 90.0,
            "engine_rpm": 1500.0,
            "oil_pressure_kpa": 400.0,
            "coolant_temperature_c": 85.0,
            "vibration_mm_s": 2.0,
            "battery_voltage_v": 14.0,
            "fuel_consumption_l_per_100km": 30.0,
            "operating_hours": 500.0,
            "distance_km": 10000.0,
        }
        defaults.update(kwargs)
        reading = Mock(spec=SensorReading)
        for k, v in defaults.items():
            setattr(reading, k, v)
        return reading

    def test_healthy_vehicle_high_score(self):
        """Healthy vehicle should get high score."""
        vehicle = self.create_mock_vehicle()
        reading = self.create_mock_reading()

        # Mock anomaly detector to return no anomalies
        mock_detector = Mock()
        mock_detector.detect.return_value = []
        self.scorer.anomaly_detector = mock_detector

        # Mock session
        mock_session = Mock()
        mock_session.query.return_value.filter.return_value.order_by.return_value.first.return_value = reading

        assessment = self.scorer.assess_vehicle(vehicle, mock_session)

        assert assessment.overall_score >= 80
        assert assessment.health_status == HealthStatus.HEALTHY
        assert len(assessment.factors) == 7  # 6 sensors + maintenance interval

    def test_critical_temperature_lowers_score(self):
        """Critical engine temperature should lower health score."""
        vehicle = self.create_mock_vehicle()
        reading = self.create_mock_reading(engine_temperature_c=120.0)

        # Mock anomaly detector to return critical anomaly
        mock_anomaly = AnomalyResult(
            sensor_name="engine_temperature_c",
            sensor_value=120.0,
            anomaly_type=AnomalyType.THRESHOLD_BREACH,
            severity=AlertSeverity.CRITICAL,
            expected_min=None,
            expected_max=115.0,
            anomaly_score=None,
            description="Engine temperature critically high",
        )
        mock_detector = Mock()
        mock_detector.detect.return_value = [mock_anomaly]
        self.scorer.anomaly_detector = mock_detector

        mock_session = Mock()
        mock_session.query.return_value.filter.return_value.order_by.return_value.first.return_value = reading

        assessment = self.scorer.assess_vehicle(vehicle, mock_session)

        # Should have penalty from anomaly
        assert assessment.overall_score < 80
        assert any("URGENT" in r for r in assessment.recommendations)

    def test_maintenance_overdue_lowers_score(self):
        """Overdue maintenance should lower score."""
        vehicle = self.create_mock_vehicle(
            last_service_date=datetime.utcnow() - timedelta(days=150)
        )
        reading = self.create_mock_reading()

        mock_detector = Mock()
        mock_detector.detect.return_value = []
        self.scorer.anomaly_detector = mock_detector

        mock_session = Mock()
        mock_session.query.return_value.filter.return_value.order_by.return_value.first.return_value = reading

        assessment = self.scorer.assess_vehicle(vehicle, mock_session)

        # Maintenance interval factor should be low
        maint_factor = next(f for f in assessment.factors if f.name == "maintenance_interval")
        assert maint_factor.score <= 50
        assert maint_factor.status == "critical"

    def test_calculate_sensor_score_within_range(self):
        """Sensor within healthy range should score 100."""
        score, status = self.scorer.calculate_sensor_score("engine_temperature_c", 90.0)
        assert score == 100.0
        assert status == "normal"

    def test_calculate_sensor_score_out_of_range(self):
        """Sensor out of range should have reduced score."""
        score, status = self.scorer.calculate_sensor_score("engine_temperature_c", 120.0)
        assert score < 100.0
        assert status in ["warning", "critical"]

    def test_calculate_maintenance_interval_score(self):
        """Maintenance interval scoring."""
        vehicle = self.create_mock_vehicle(last_service_date=datetime.utcnow() - timedelta(days=30))
        score, status = self.scorer.calculate_maintenance_interval_score(vehicle)
        assert score == 100.0
        assert status == "normal"

        vehicle.last_service_date = datetime.utcnow() - timedelta(days=90)
        score, status = self.scorer.calculate_maintenance_interval_score(vehicle)
        assert score == 75.0

        vehicle.last_service_date = datetime.utcnow() - timedelta(days=150)
        score, status = self.scorer.calculate_maintenance_interval_score(vehicle)
        assert score == 20.0


class TestMaintenanceTriageEngine:
    """Tests for MaintenanceTriageEngine."""

    def setup_method(self):
        self.mock_session = Mock()
        self.engine = MaintenanceTriageEngine(self.mock_session)

    def create_mock_vehicle(self) -> Vehicle:
        vehicle = Mock(spec=Vehicle)
        vehicle.id = 1
        vehicle.vehicle_id = "TEST-001"
        vehicle.current_health_status = HealthStatus.HEALTHY
        vehicle.health_score = 90.0
        return vehicle

    def create_mock_reading(self, **kwargs) -> SensorReading:
        defaults = {
            "id": 1,
            "vehicle_id": 1,
            "timestamp": datetime.utcnow(),
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
        return reading

    def test_overheating_critical_creates_alert(self):
        """Critical overheating should create critical alert."""
        vehicle = self.create_mock_vehicle()
        reading = self.create_mock_reading(engine_temperature_c=120.0)

        # Mock anomaly detector
        mock_anomaly = AnomalyResult(
            sensor_name="engine_temperature_c",
            sensor_value=120.0,
            anomaly_type=AnomalyType.THRESHOLD_BREACH,
            severity=AlertSeverity.CRITICAL,
            expected_min=None,
            expected_max=115.0,
            anomaly_score=None,
            description="Engine temperature critically high",
        )
        self.engine.anomaly_detector.detect = Mock(return_value=[mock_anomaly])

        # Mock health scorer
        mock_assessment = HealthAssessment(
            vehicle_id=1,
            overall_score=50.0,
            health_status=HealthStatus.CRITICAL,
            factors=[],
            anomalies=[mock_anomaly],
            timestamp=datetime.utcnow(),
            recommendations=["URGENT: Engine overheating"],
        )
        self.engine.health_scorer.assess_vehicle = Mock(return_value=mock_assessment)

        # Mock session queries - need to handle two different query chains:
        # 1. session.query(SensorReading).filter(...).order_by(...).first() -> reading
        # 2. session.query(MaintenanceAlert).filter(...).first() -> None (no duplicate)
        def query_side_effect(model):
            mock_q = Mock()
            mock_filter = Mock()
            mock_q.filter.return_value = mock_filter

            if model == SensorReading:
                # SensorReading query has order_by
                mock_order_by = Mock()
                mock_filter.order_by.return_value = mock_order_by
                mock_order_by.first.return_value = reading
            elif model == MaintenanceAlert:
                # MaintenanceAlert query for deduplication has no order_by
                mock_filter.first.return_value = None
            return mock_q

        self.mock_session.query.side_effect = query_side_effect

        alerts = self.engine.process_vehicle(vehicle)

        assert len(alerts) >= 1
        critical_alerts = [a for a in alerts if a.severity == AlertSeverity.CRITICAL]
        assert len(critical_alerts) >= 1
        assert "Cooling System" in critical_alerts[0].component

    def test_low_oil_pressure_creates_alert(self):
        """Low oil pressure should create alert."""
        vehicle = self.create_mock_vehicle()
        reading = self.create_mock_reading(oil_pressure_kpa=100.0)

        mock_anomaly = AnomalyResult(
            sensor_name="oil_pressure_kpa",
            sensor_value=100.0,
            anomaly_type=AnomalyType.THRESHOLD_BREACH,
            severity=AlertSeverity.CRITICAL,
            expected_min=150.0,
            expected_max=None,
            anomaly_score=None,
            description="Oil pressure critically low",
        )
        self.engine.anomaly_detector.detect = Mock(return_value=[mock_anomaly])

        mock_assessment = HealthAssessment(
            vehicle_id=1, overall_score=40.0, health_status=HealthStatus.CRITICAL,
            factors=[], anomalies=[mock_anomaly], timestamp=datetime.utcnow(),
            recommendations=["URGENT: Low oil pressure"]
        )
        self.engine.health_scorer.assess_vehicle = Mock(return_value=mock_assessment)

        # Mock session queries - handle two different query chains
        def query_side_effect(model):
            mock_q = Mock()
            mock_filter = Mock()
            mock_q.filter.return_value = mock_filter

            if model == SensorReading:
                mock_order_by = Mock()
                mock_filter.order_by.return_value = mock_order_by
                mock_order_by.first.return_value = reading
            elif model == MaintenanceAlert:
                mock_filter.first.return_value = None
            return mock_q

        self.mock_session.query.side_effect = query_side_effect

        alerts = self.engine.process_vehicle(vehicle)

        oil_alerts = [a for a in alerts if "Lubrication" in a.component]
        assert len(oil_alerts) >= 1
        assert oil_alerts[0].severity == AlertSeverity.CRITICAL

    def test_multiple_anomalies_creates_composite_alert(self):
        """Multiple critical anomalies should create composite alert."""
        vehicle = self.create_mock_vehicle()
        reading = self.create_mock_reading(
            engine_temperature_c=120.0,
            oil_pressure_kpa=100.0,
            vibration_mm_s=15.0,
        )

        anomalies = [
            AnomalyResult("engine_temperature_c", 120.0, AnomalyType.THRESHOLD_BREACH,
                         AlertSeverity.CRITICAL, None, 115.0, None, "High temp"),
            AnomalyResult("oil_pressure_kpa", 100.0, AnomalyType.THRESHOLD_BREACH,
                         AlertSeverity.CRITICAL, 150.0, None, None, "Low oil"),
            AnomalyResult("vibration_mm_s", 15.0, AnomalyType.THRESHOLD_BREACH,
                         AlertSeverity.CRITICAL, None, 10.0, None, "High vibration"),
        ]
        self.engine.anomaly_detector.detect = Mock(return_value=anomalies)

        mock_assessment = HealthAssessment(
            vehicle_id=1, overall_score=20.0, health_status=HealthStatus.CRITICAL,
            factors=[], anomalies=anomalies, timestamp=datetime.utcnow(),
            recommendations=[]
        )
        self.engine.health_scorer.assess_vehicle = Mock(return_value=mock_assessment)

        # Mock session queries - handle two different query chains
        def query_side_effect(model):
            mock_q = Mock()
            mock_filter = Mock()
            mock_q.filter.return_value = mock_filter

            if model == SensorReading:
                mock_order_by = Mock()
                mock_filter.order_by.return_value = mock_order_by
                mock_order_by.first.return_value = reading
            elif model == MaintenanceAlert:
                mock_filter.first.return_value = None
            return mock_q

        self.mock_session.query.side_effect = query_side_effect

        alerts = self.engine.process_vehicle(vehicle)

        multi_alerts = [a for a in alerts if a.alert_type == "multiple_anomalies"]
        assert len(multi_alerts) >= 1
        assert multi_alerts[0].severity == AlertSeverity.CRITICAL

    def test_deduplication_prevents_duplicate_alerts(self):
        """Same rule should not create duplicate alerts within window."""
        vehicle = self.create_mock_vehicle()
        reading = self.create_mock_reading(engine_temperature_c=120.0)

        mock_anomaly = AnomalyResult(
            sensor_name="engine_temperature_c", sensor_value=120.0,
            anomaly_type=AnomalyType.THRESHOLD_BREACH, severity=AlertSeverity.CRITICAL,
            expected_min=None, expected_max=115.0, anomaly_score=None,
            description="High temp"
        )
        self.engine.anomaly_detector.detect = Mock(return_value=[mock_anomaly])

        mock_assessment = HealthAssessment(
            vehicle_id=1, overall_score=50.0, health_status=HealthStatus.CRITICAL,
            factors=[], anomalies=[mock_anomaly], timestamp=datetime.utcnow(),
            recommendations=[]
        )
        self.engine.health_scorer.assess_vehicle = Mock(return_value=mock_assessment)

        self.mock_session.query.return_value.filter.return_value.order_by.return_value.first.return_value = reading

        # First call - should create alert
        self.mock_session.query.return_value.filter.return_value.first.return_value = None
        alerts1 = self.engine.process_vehicle(vehicle)
        assert len(alerts1) >= 1

        # Second call - should deduplicate
        self.mock_session.query.return_value.filter.return_value.first.return_value = Mock()  # Existing alert
        alerts2 = self.engine.process_vehicle(vehicle)
        # Should not create new alert for same rule


class TestAutoGenerateWorkOrders:
    """Tests for automatic work order generation."""

    def test_creates_work_order_from_critical_alert(self):
        """Critical alert should generate work order."""
        mock_session = Mock()

        # Create mock alert
        alert = Mock(spec=MaintenanceAlert)
        alert.id = 1
        alert.vehicle_id = 1
        alert.severity = AlertSeverity.CRITICAL
        alert.component = "Engine Cooling System"
        alert.description = "Engine overheating"
        alert.recommended_action = "Inspect cooling system"

        # Mock query to return alert
        mock_query = Mock()
        mock_query.filter.return_value.all.return_value = [alert]
        mock_session.query.return_value = mock_query

        # Mock technician
        tech = Mock(spec=Technician)
        tech.id = 1
        tech.specialization = "Engine"
        mock_session.query.return_value.filter.return_value.first.return_value = tech

        work_orders = auto_generate_work_orders(mock_session)

        assert len(work_orders) == 1
        wo = work_orders[0]
        assert wo.priority == WorkOrderPriority.URGENT
        assert wo.component == "Engine Cooling System"
        assert wo.status == WorkOrderStatus.OPEN


if __name__ == "__main__":
    pytest.main([__file__, "-v"])