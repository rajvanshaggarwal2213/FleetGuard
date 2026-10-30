"""
Tests for database models and operations.
"""
import pytest
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
import uuid

from app.models.database import (
    Base, Vehicle, VehicleType, HealthStatus, SensorReading,
    VehicleHealthHistory, Anomaly, MaintenanceAlert, AlertSeverity,
    WorkOrder, WorkOrderStatus, WorkOrderPriority, MaintenanceHistory,
    Technician, MLModel, ModelExperiment, DatasetUpload,
    SimulationEvent
)
from app.models.connection import DatabaseManager, init_database


def unique_id(prefix: str = "TEST") -> str:
    """Generate a unique test ID."""
    return f"{prefix}-{uuid.uuid4().hex[:8].upper()}"


@pytest.fixture(scope="session")
def db_manager():
    """Create test database manager with file-based SQLite."""
    import os
    import tempfile
    # Use a temporary file for SQLite to persist across connections
    temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
    temp_db.close()
    db_path = temp_db.name
    os.environ["DATABASE_URL"] = f"sqlite:///{db_path}"
    manager = DatabaseManager()
    manager.create_tables()
    yield manager
    manager.drop_tables()
    # Cleanup
    try:
        os.unlink(db_path)
    except:
        pass


@pytest.fixture
def db_session(db_manager):
    """Provide a clean database session for each test with automatic rollback."""
    session = db_manager.SessionLocal()
    try:
        yield session
        session.rollback()  # Rollback to clean state
    finally:
        session.close()


class TestVehicleModel:
    """Tests for Vehicle model."""

    def test_create_vehicle(self, db_session: Session):
        vehicle = Vehicle(
            vehicle_id=unique_id("VEH"),
            vehicle_type=VehicleType.TRUCK,
            registration_number="TEST-1234",
            manufacturer="Test Motors",
            model="Test Model",
            total_distance_km=10000.0,
            engine_hours=500.0,
            current_health_status=HealthStatus.HEALTHY,
            health_score=95.0,
        )
        db_session.add(vehicle)
        db_session.commit()

        assert vehicle.id is not None
        assert vehicle.vehicle_id.startswith("VEH-")
        assert vehicle.health_score == 95.0

    def test_vehicle_unique_constraint(self, db_session: Session):
        uid = unique_id("UNIQ")
        v1 = Vehicle(vehicle_id=uid, vehicle_type=VehicleType.TRUCK,
                     registration_number="TEST-1", manufacturer="M", model="M")
        v2 = Vehicle(vehicle_id=uid, vehicle_type=VehicleType.BUS,
                     registration_number="TEST-2", manufacturer="M", model="M")
        db_session.add(v1)
        db_session.commit()

        db_session.add(v2)
        with pytest.raises(Exception):  # IntegrityError
            db_session.commit()


class TestSensorReadingModel:
    """Tests for SensorReading model."""

    def test_create_sensor_reading(self, db_session: Session):
        vehicle = Vehicle(vehicle_id=unique_id("SR"), vehicle_type=VehicleType.TRUCK,
                          registration_number="TEST-1", manufacturer="M", model="M")
        db_session.add(vehicle)
        db_session.commit()

        reading = SensorReading(
            vehicle_id=vehicle.id,
            timestamp=datetime.utcnow(),
            engine_temperature_c=90.0,
            engine_rpm=1500.0,
            oil_pressure_kpa=400.0,
            coolant_temperature_c=85.0,
            vibration_mm_s=2.0,
            battery_voltage_v=14.0,
            fuel_consumption_l_per_100km=30.0,
            operating_hours=500.0,
            distance_km=10000.0,
        )
        db_session.add(reading)
        db_session.commit()

        assert reading.id is not None
        assert reading.engine_temperature_c == 90.0


class TestHealthHistoryModel:
    """Tests for VehicleHealthHistory model."""

    def test_create_health_history(self, db_session: Session):
        vehicle = Vehicle(vehicle_id=unique_id("HH"), vehicle_type=VehicleType.TRUCK,
                          registration_number="TEST-1", manufacturer="M", model="M")
        db_session.add(vehicle)
        db_session.commit()

        history = VehicleHealthHistory(
            vehicle_id=vehicle.id,
            timestamp=datetime.utcnow(),
            health_score=85.0,
            health_status=HealthStatus.MAINTENANCE_APPROACHING,
            contributing_factors='{"engine_temperature": "normal"}',
        )
        db_session.add(history)
        db_session.commit()

        assert history.id is not None
        assert history.health_score == 85.0


class TestAnomalyModel:
    """Tests for Anomaly model."""

    def test_create_anomaly(self, db_session: Session):
        vehicle = Vehicle(vehicle_id=unique_id("ANOM"), vehicle_type=VehicleType.TRUCK,
                          registration_number="TEST-1", manufacturer="M", model="M")
        db_session.add(vehicle)
        db_session.commit()

        anomaly = Anomaly(
            vehicle_id=vehicle.id,
            timestamp=datetime.utcnow(),
            anomaly_type="threshold_breach",
            sensor_name="engine_temperature_c",
            sensor_value=120.0,
            expected_range_min=75.0,
            expected_range_max=100.0,
            severity=AlertSeverity.CRITICAL,
            description="Engine temperature critically high",
        )
        db_session.add(anomaly)
        db_session.commit()

        assert anomaly.id is not None
        assert anomaly.severity == AlertSeverity.CRITICAL


class TestMaintenanceAlertModel:
    """Tests for MaintenanceAlert model."""

    def test_create_alert(self, db_session: Session):
        vehicle = Vehicle(vehicle_id=unique_id("ALERT"), vehicle_type=VehicleType.TRUCK,
                          registration_number="TEST-1", manufacturer="M", model="M")
        db_session.add(vehicle)
        db_session.commit()

        alert = MaintenanceAlert(
            vehicle_id=vehicle.id,
            timestamp=datetime.utcnow(),
            alert_type="RULE_001",
            component="Engine Cooling System",
            severity=AlertSeverity.CRITICAL,
            title="Engine Overheating - TEST-001",
            description="Immediate engine shutdown required",
            recommended_action="Shutdown and inspect cooling system",
        )
        db_session.add(alert)
        db_session.commit()

        assert alert.id is not None
        assert alert.severity == AlertSeverity.CRITICAL


class TestWorkOrderModel:
    """Tests for WorkOrder model."""

    def test_create_work_order(self, db_session: Session):
        vehicle = Vehicle(vehicle_id=unique_id("WO"), vehicle_type=VehicleType.TRUCK,
                          registration_number="TEST-1", manufacturer="M", model="M")
        tech = Technician(employee_id=unique_id("TECH"), name="Test Tech", is_active=True)
        db_session.add_all([vehicle, tech])
        db_session.commit()

        wo = WorkOrder(
            work_order_id=f"WO-{unique_id('WOID')}",
            vehicle_id=vehicle.id,
            technician_id=tech.id,
            component="Engine Cooling System",
            issue_description="Engine overheating detected",
            priority=WorkOrderPriority.URGENT,
            recommended_inspection="Inspect radiator and thermostat",
            status=WorkOrderStatus.OPEN,
        )
        db_session.add(wo)
        db_session.commit()

        assert wo.id is not None
        assert wo.status == WorkOrderStatus.OPEN
        assert wo.priority == WorkOrderPriority.URGENT


class TestRelationships:
    """Tests for model relationships."""

    def test_vehicle_sensor_readings_relationship(self, db_session: Session):
        vehicle = Vehicle(vehicle_id=unique_id("REL1"), vehicle_type=VehicleType.TRUCK,
                          registration_number="TEST-1", manufacturer="M", model="M")
        db_session.add(vehicle)
        db_session.commit()

        reading = SensorReading(
            vehicle_id=vehicle.id,
            timestamp=datetime.utcnow(),
            engine_temperature_c=90.0, engine_rpm=1500.0, oil_pressure_kpa=400.0,
            coolant_temperature_c=85.0, vibration_mm_s=2.0, battery_voltage_v=14.0,
            fuel_consumption_l_per_100km=30.0,
        )
        db_session.add(reading)
        db_session.commit()

        # Reload vehicle
        db_session.refresh(vehicle)
        assert len(vehicle.sensor_readings) == 1

    def test_vehicle_alerts_relationship(self, db_session: Session):
        vehicle = Vehicle(vehicle_id=unique_id("REL2"), vehicle_type=VehicleType.TRUCK,
                          registration_number="TEST-1", manufacturer="M", model="M")
        db_session.add(vehicle)
        db_session.commit()

        alert = MaintenanceAlert(
            vehicle_id=vehicle.id,
            timestamp=datetime.utcnow(),
            alert_type="TEST", component="Test", severity=AlertSeverity.HIGH,
            title="Test Alert", description="Test", recommended_action="Test",
        )
        db_session.add(alert)
        db_session.commit()

        db_session.refresh(vehicle)
        assert len(vehicle.alerts) == 1


class TestDatabaseInitialization:
    """Tests for database initialization script."""

    def test_init_database_creates_tables(self, db_manager: DatabaseManager):
        # Tables should already exist from fixture
        with db_manager.get_session() as session:
            # Should be able to query all tables without error
            count = session.query(Vehicle).count()
            # Just verify the query works (data from other tests may exist)
            assert count >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])