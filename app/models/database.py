"""
Database models for FleetGuard AI.
"""
from datetime import datetime
from enum import Enum as PyEnum
from typing import Optional

from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey, Enum, Index, UniqueConstraint
)
from sqlalchemy.orm import relationship, declarative_base

Base = declarative_base()


class VehicleType(PyEnum):
    TRUCK = "truck"
    BUS = "bus"
    VAN = "van"
    OTHER = "other"


class HealthStatus(PyEnum):
    HEALTHY = "healthy"          # GREEN
    MAINTENANCE_APPROACHING = "maintenance_approaching"  # YELLOW
    CRITICAL = "critical"        # RED


class AlertSeverity(PyEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class WorkOrderStatus(PyEnum):
    OPEN = "open"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class WorkOrderPriority(PyEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class Vehicle(Base):
    __tablename__ = "vehicles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    vehicle_id = Column(String(50), unique=True, nullable=False, index=True)
    vehicle_type = Column(Enum(VehicleType), nullable=False)
    registration_number = Column(String(50), nullable=False)
    manufacturer = Column(String(100), nullable=False)
    model = Column(String(100), nullable=False)
    total_distance_km = Column(Float, default=0.0)
    engine_hours = Column(Float, default=0.0)
    last_service_date = Column(DateTime, nullable=True)
    next_scheduled_service = Column(DateTime, nullable=True)
    current_health_status = Column(Enum(HealthStatus), default=HealthStatus.HEALTHY)
    health_score = Column(Float, default=100.0)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    sensor_readings = relationship("SensorReading", back_populates="vehicle", cascade="all, delete-orphan")
    health_history = relationship("VehicleHealthHistory", back_populates="vehicle", cascade="all, delete-orphan")
    anomalies = relationship("Anomaly", back_populates="vehicle", cascade="all, delete-orphan")
    alerts = relationship("MaintenanceAlert", back_populates="vehicle", cascade="all, delete-orphan")
    work_orders = relationship("WorkOrder", back_populates="vehicle", cascade="all, delete-orphan")
    maintenance_history = relationship("MaintenanceHistory", back_populates="vehicle", cascade="all, delete-orphan")
    simulation_events = relationship("SimulationEvent", back_populates="vehicle", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Vehicle {self.vehicle_id}>"


class SensorReading(Base):
    __tablename__ = "sensor_readings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    vehicle_id = Column(Integer, ForeignKey("vehicles.id", ondelete="CASCADE"), nullable=False, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    # Engine sensors
    engine_temperature_c = Column(Float, nullable=False)
    engine_rpm = Column(Float, nullable=False)
    oil_pressure_kpa = Column(Float, nullable=False)
    coolant_temperature_c = Column(Float, nullable=False)
    vibration_mm_s = Column(Float, nullable=False)
    battery_voltage_v = Column(Float, nullable=False)
    fuel_consumption_l_per_100km = Column(Float, nullable=False)

    # Derived
    operating_hours = Column(Float, default=0.0)
    distance_km = Column(Float, default=0.0)

    # Simulation metadata
    is_simulated = Column(Boolean, default=True)
    simulation_event_id = Column(Integer, ForeignKey("simulation_events.id", ondelete="SET NULL"), nullable=True)

    # Relationships
    vehicle = relationship("Vehicle", back_populates="sensor_readings")

    __table_args__ = (
        Index("ix_sensor_readings_vehicle_timestamp", "vehicle_id", "timestamp"),
    )

    def __repr__(self):
        return f"<SensorReading vehicle={self.vehicle_id} time={self.timestamp}>"


class VehicleHealthHistory(Base):
    __tablename__ = "vehicle_health_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    vehicle_id = Column(Integer, ForeignKey("vehicles.id", ondelete="CASCADE"), nullable=False, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    health_score = Column(Float, nullable=False)
    health_status = Column(Enum(HealthStatus), nullable=False)
    contributing_factors = Column(Text, nullable=True)  # JSON string

    # Relationships
    vehicle = relationship("Vehicle", back_populates="health_history")

    __table_args__ = (
        Index("ix_health_history_vehicle_timestamp", "vehicle_id", "timestamp"),
    )

    def __repr__(self):
        return f"<VehicleHealthHistory vehicle={self.vehicle_id} score={self.health_score}>"


class Anomaly(Base):
    __tablename__ = "anomalies"

    id = Column(Integer, primary_key=True, autoincrement=True)
    vehicle_id = Column(Integer, ForeignKey("vehicles.id", ondelete="CASCADE"), nullable=False, index=True)
    sensor_reading_id = Column(Integer, ForeignKey("sensor_readings.id", ondelete="SET NULL"), nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    anomaly_type = Column(String(100), nullable=False)  # e.g., "threshold_breach", "trend_anomaly", "isolation_forest"
    sensor_name = Column(String(50), nullable=False)
    sensor_value = Column(Float, nullable=False)
    expected_range_min = Column(Float, nullable=True)
    expected_range_max = Column(Float, nullable=True)
    anomaly_score = Column(Float, nullable=True)  # For ML-based anomalies
    severity = Column(Enum(AlertSeverity), nullable=False)
    description = Column(Text, nullable=False)
    is_resolved = Column(Boolean, default=False)
    resolved_at = Column(DateTime, nullable=True)

    # Relationships
    vehicle = relationship("Vehicle", back_populates="anomalies")

    __table_args__ = (
        Index("ix_anomalies_vehicle_timestamp", "vehicle_id", "timestamp"),
    )

    def __repr__(self):
        return f"<Anomaly vehicle={self.vehicle_id} type={self.anomaly_type} severity={self.severity}>"


class MaintenanceAlert(Base):
    __tablename__ = "maintenance_alerts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    vehicle_id = Column(Integer, ForeignKey("vehicles.id", ondelete="CASCADE"), nullable=False, index=True)
    anomaly_id = Column(Integer, ForeignKey("anomalies.id", ondelete="SET NULL"), nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    alert_type = Column(String(100), nullable=False)
    component = Column(String(100), nullable=False)
    severity = Column(Enum(AlertSeverity), nullable=False)
    title = Column(String(200), nullable=False)
    description = Column(Text, nullable=False)
    recommended_action = Column(Text, nullable=False)
    evidence = Column(Text, nullable=True)  # JSON string with supporting data

    is_acknowledged = Column(Boolean, default=False)
    acknowledged_by = Column(Integer, ForeignKey("technicians.id", ondelete="SET NULL"), nullable=True)
    acknowledged_at = Column(DateTime, nullable=True)
    is_resolved = Column(Boolean, default=False)
    resolved_at = Column(DateTime, nullable=True)
    resolution_notes = Column(Text, nullable=True)

    # Relationships
    vehicle = relationship("Vehicle", back_populates="alerts")
    work_order = relationship("WorkOrder", back_populates="alert", uselist=False)
    acknowledged_by_technician = relationship("Technician", back_populates="acknowledged_alerts")

    __table_args__ = (
        Index("ix_alerts_vehicle_timestamp", "vehicle_id", "timestamp"),
        Index("ix_alerts_severity_resolved", "severity", "is_resolved"),
    )

    def __repr__(self):
        return f"<MaintenanceAlert {self.title} severity={self.severity}>"


class WorkOrder(Base):
    __tablename__ = "work_orders"

    id = Column(Integer, primary_key=True, autoincrement=True)
    work_order_id = Column(String(50), unique=True, nullable=False, index=True)
    vehicle_id = Column(Integer, ForeignKey("vehicles.id", ondelete="CASCADE"), nullable=False, index=True)
    alert_id = Column(Integer, ForeignKey("maintenance_alerts.id", ondelete="SET NULL"), nullable=True, unique=True)
    technician_id = Column(Integer, ForeignKey("technicians.id", ondelete="SET NULL"), nullable=True)

    component = Column(String(100), nullable=False)
    issue_description = Column(Text, nullable=False)
    priority = Column(Enum(WorkOrderPriority), nullable=False)
    recommended_inspection = Column(Text, nullable=False)

    status = Column(Enum(WorkOrderStatus), default=WorkOrderStatus.OPEN, nullable=False)
    creation_timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    assigned_timestamp = Column(DateTime, nullable=True)
    started_timestamp = Column(DateTime, nullable=True)
    completed_timestamp = Column(DateTime, nullable=True)
    cancelled_timestamp = Column(DateTime, nullable=True)

    completion_notes = Column(Text, nullable=True)
    resolution_timestamp = Column(DateTime, nullable=True)

    # Relationships
    vehicle = relationship("Vehicle", back_populates="work_orders")
    alert = relationship("MaintenanceAlert", back_populates="work_order")
    technician = relationship("Technician", back_populates="work_orders")

    def __repr__(self):
        return f"<WorkOrder {self.work_order_id} status={self.status}>"


class MaintenanceHistory(Base):
    __tablename__ = "maintenance_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    vehicle_id = Column(Integer, ForeignKey("vehicles.id", ondelete="CASCADE"), nullable=False, index=True)
    work_order_id = Column(Integer, ForeignKey("work_orders.id", ondelete="SET NULL"), nullable=True)

    service_date = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    service_type = Column(String(100), nullable=False)  # preventive, corrective, inspection
    component = Column(String(100), nullable=False)
    description = Column(Text, nullable=False)
    parts_replaced = Column(Text, nullable=True)  # JSON string
    cost = Column(Float, nullable=True)
    mileage_at_service = Column(Float, nullable=True)
    engine_hours_at_service = Column(Float, nullable=True)
    technician_name = Column(String(100), nullable=True)
    notes = Column(Text, nullable=True)

    # Relationships
    vehicle = relationship("Vehicle", back_populates="maintenance_history")

    __table_args__ = (
        Index("ix_maintenance_history_vehicle_date", "vehicle_id", "service_date"),
    )

    def __repr__(self):
        return f"<MaintenanceHistory vehicle={self.vehicle_id} component={self.component}>"


class MLModel(Base):
    __tablename__ = "ml_models"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(100), nullable=False)
    model_version = Column(String(50), nullable=False)
    model_type = Column(String(50), nullable=False)  # rul_regressor, anomaly_detector, baseline
    dataset_name = Column(String(100), nullable=True)  # e.g., "NASA FD001"

    # Model artifacts
    model_path = Column(String(500), nullable=True)
    scaler_path = Column(String(500), nullable=True)
    feature_names = Column(Text, nullable=True)  # JSON string

    # Metrics
    mae = Column(Float, nullable=True)
    rmse = Column(Float, nullable=True)
    r2_score = Column(Float, nullable=True)
    training_samples = Column(Integer, nullable=True)
    validation_samples = Column(Integer, nullable=True)

    # Metadata
    training_config = Column(Text, nullable=True)  # JSON string
    is_active = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    trained_by = Column(String(100), nullable=True)

    # Relationships
    experiments = relationship("ModelExperiment", back_populates="model")

    __table_args__ = (
        UniqueConstraint("model_name", "model_version", name="uq_model_name_version"),
    )

    def __repr__(self):
        return f"<MLModel {self.model_name} v{self.model_version}>"


class ModelExperiment(Base):
    __tablename__ = "model_experiments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_id = Column(Integer, ForeignKey("ml_models.id", ondelete="CASCADE"), nullable=False, index=True)
    experiment_name = Column(String(200), nullable=False)
    run_id = Column(String(100), nullable=True)  # MLflow run ID

    # Parameters
    hyperparameters = Column(Text, nullable=True)  # JSON string
    training_data_hash = Column(String(64), nullable=True)

    # Metrics
    metrics = Column(Text, nullable=True)  # JSON string

    # Artifacts
    artifact_uri = Column(String(500), nullable=True)

    # Status
    status = Column(String(50), default="running")  # running, completed, failed
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    error_message = Column(Text, nullable=True)

    # Relationships
    model = relationship("MLModel", back_populates="experiments")

    def __repr__(self):
        return f"<ModelExperiment {self.experiment_name} status={self.status}>"


class DatasetUpload(Base):
    __tablename__ = "dataset_uploads"

    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String(255), nullable=False)
    dataset_type = Column(String(50), nullable=False)  # nasa_fd001, nasa_fd002, etc.
    file_size_bytes = Column(Integer, nullable=False)
    row_count = Column(Integer, nullable=True)
    column_count = Column(Integer, nullable=True)

    # Validation
    is_valid = Column(Boolean, default=False)
    validation_errors = Column(Text, nullable=True)  # JSON string
    validation_warnings = Column(Text, nullable=True)  # JSON string

    # Processing
    processed_at = Column(DateTime, nullable=True)
    processing_duration_seconds = Column(Float, nullable=True)

    uploaded_by = Column(String(100), nullable=True)
    uploaded_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<DatasetUpload {self.filename} valid={self.is_valid}>"


class SimulationEvent(Base):
    __tablename__ = "simulation_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    vehicle_id = Column(Integer, ForeignKey("vehicles.id", ondelete="CASCADE"), nullable=False, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    event_type = Column(String(100), nullable=False)  # normal, overheating, vibration, low_oil_pressure, battery_deterioration, degradation
    description = Column(Text, nullable=False)
    parameters = Column(Text, nullable=True)  # JSON string with event parameters
    triggered_by = Column(String(100), nullable=True)  # user, auto, demo

    # Relationships
    vehicle = relationship("Vehicle", back_populates="simulation_events")

    __table_args__ = (
        Index("ix_simulation_events_vehicle_timestamp", "vehicle_id", "timestamp"),
    )

    def __repr__(self):
        return f"<SimulationEvent vehicle={self.vehicle_id} type={self.event_type}>"


class Technician(Base):
    __tablename__ = "technicians"

    id = Column(Integer, primary_key=True, autoincrement=True)
    employee_id = Column(String(50), unique=True, nullable=False)
    name = Column(String(100), nullable=False)
    email = Column(String(100), nullable=True)
    phone = Column(String(50), nullable=True)
    specialization = Column(String(100), nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    work_orders = relationship("WorkOrder", back_populates="technician")
    acknowledged_alerts = relationship("MaintenanceAlert", back_populates="acknowledged_by_technician")

    def __repr__(self):
        return f"<Technician {self.employee_id} - {self.name}>"