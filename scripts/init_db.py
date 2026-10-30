"""
Database initialization and seed data for FleetGuard AI.
"""
import json
from datetime import datetime, timedelta
from typing import List

from sqlalchemy.orm import Session

from app.models.database import (
    Base, Vehicle, VehicleType, HealthStatus, SensorReading,
    VehicleHealthHistory, Technician, WorkOrderStatus, WorkOrderPriority,
    AlertSeverity, utcnow
)
from app.models.connection import DatabaseManager


def create_seed_vehicles() -> List[Vehicle]:
    """Create initial fleet of vehicles."""
    vehicles = [
        Vehicle(
            vehicle_id="TRK-001",
            vehicle_type=VehicleType.TRUCK,
            registration_number="MH-12-AB-1234",
            manufacturer="Tata Motors",
            model="Signa 5525.S",
            total_distance_km=125000.0,
            engine_hours=4200.0,
            last_service_date=datetime.utcnow() - timedelta(days=45),
            next_scheduled_service=datetime.utcnow() + timedelta(days=15),
            current_health_status=HealthStatus.HEALTHY,
            health_score=92.5,
        ),
        Vehicle(
            vehicle_id="TRK-002",
            vehicle_type=VehicleType.TRUCK,
            registration_number="MH-12-CD-5678",
            manufacturer="Ashok Leyland",
            model="AVTR 5520",
            total_distance_km=98000.0,
            engine_hours=3500.0,
            last_service_date=datetime.utcnow() - timedelta(days=30),
            next_scheduled_service=datetime.utcnow() + timedelta(days=30),
            current_health_status=HealthStatus.HEALTHY,
            health_score=88.0,
        ),
        Vehicle(
            vehicle_id="TRK-003",
            vehicle_type=VehicleType.TRUCK,
            registration_number="MH-14-EF-9012",
            manufacturer="Mahindra",
            model="Blazo X 55",
            total_distance_km=156000.0,
            engine_hours=5100.0,
            last_service_date=datetime.utcnow() - timedelta(days=60),
            next_scheduled_service=datetime.utcnow() + timedelta(days=5),
            current_health_status=HealthStatus.MAINTENANCE_APPROACHING,
            health_score=72.0,
        ),
        Vehicle(
            vehicle_id="BUS-001",
            vehicle_type=VehicleType.BUS,
            registration_number="MH-01-GH-3456",
            manufacturer="Tata Motors",
            model="Starbus Ultra",
            total_distance_km=85000.0,
            engine_hours=3800.0,
            last_service_date=datetime.utcnow() - timedelta(days=25),
            next_scheduled_service=datetime.utcnow() + timedelta(days=35),
            current_health_status=HealthStatus.HEALTHY,
            health_score=90.5,
        ),
        Vehicle(
            vehicle_id="BUS-002",
            vehicle_type=VehicleType.BUS,
            registration_number="MH-01-IJ-7890",
            manufacturer="Ashok Leyland",
            model="Viking 220",
            total_distance_km=112000.0,
            engine_hours=4500.0,
            last_service_date=datetime.utcnow() - timedelta(days=50),
            next_scheduled_service=datetime.utcnow() + timedelta(days=10),
            current_health_status=HealthStatus.MAINTENANCE_APPROACHING,
            health_score=68.0,
        ),
        Vehicle(
            vehicle_id="TRK-004",
            vehicle_type=VehicleType.TRUCK,
            registration_number="MH-12-KL-1357",
            manufacturer="Eicher",
            model="Pro 6042",
            total_distance_km=67000.0,
            engine_hours=2800.0,
            last_service_date=datetime.utcnow() - timedelta(days=20),
            next_scheduled_service=datetime.utcnow() + timedelta(days=40),
            current_health_status=HealthStatus.HEALTHY,
            health_score=94.0,
        ),
        Vehicle(
            vehicle_id="TRK-005",
            vehicle_type=VehicleType.TRUCK,
            registration_number="MH-12-MN-2468",
            manufacturer="BharatBenz",
            model="4228R",
            total_distance_km=143000.0,
            engine_hours=4800.0,
            last_service_date=datetime.utcnow() - timedelta(days=55),
            next_scheduled_service=datetime.utcnow() + timedelta(days=8),
            current_health_status=HealthStatus.MAINTENANCE_APPROACHING,
            health_score=65.0,
        ),
        Vehicle(
            vehicle_id="TRK-006",
            vehicle_type=VehicleType.TRUCK,
            registration_number="MH-14-OP-3579",
            manufacturer="Volvo",
            model="FMX 460",
            total_distance_km=210000.0,
            engine_hours=6200.0,
            last_service_date=datetime.utcnow() - timedelta(days=70),
            next_scheduled_service=datetime.utcnow() + timedelta(days=2),
            current_health_status=HealthStatus.CRITICAL,
            health_score=42.0,
        ),
        Vehicle(
            vehicle_id="TRK-007",
            vehicle_type=VehicleType.TRUCK,
            registration_number="MH-12-QR-4680",
            manufacturer="Scania",
            model="G 500",
            total_distance_km=89000.0,
            engine_hours=3200.0,
            last_service_date=datetime.utcnow() - timedelta(days=15),
            next_scheduled_service=datetime.utcnow() + timedelta(days=45),
            current_health_status=HealthStatus.HEALTHY,
            health_score=95.0,
        ),
        Vehicle(
            vehicle_id="TRK-008",
            vehicle_type=VehicleType.TRUCK,
            registration_number="MH-14-ST-5791",
            manufacturer="Tata Motors",
            model="LPT 3723",
            total_distance_km=178000.0,
            engine_hours=5500.0,
            last_service_date=datetime.utcnow() - timedelta(days=65),
            next_scheduled_service=datetime.utcnow() + timedelta(days=3),
            current_health_status=HealthStatus.CRITICAL,
            health_score=38.0,
        ),
    ]
    return vehicles


def create_seed_technicians(session: Session) -> List[Technician]:
    """Create initial technicians."""
    technicians = [
        Technician(
            employee_id="TECH-001",
            name="Rajesh Kumar",
            email="rajesh.kumar@fleetguard.ai",
            phone="+91-9876543210",
            specialization="Engine & Transmission",
            is_active=True,
        ),
        Technician(
            employee_id="TECH-002",
            name="Priya Sharma",
            email="priya.sharma@fleetguard.ai",
            phone="+91-9876543211",
            specialization="Electrical & Diagnostics",
            is_active=True,
        ),
        Technician(
            employee_id="TECH-003",
            name="Amit Singh",
            email="amit.singh@fleetguard.ai",
            phone="+91-9876543212",
            specialization="Brakes & Suspension",
            is_active=True,
        ),
        Technician(
            employee_id="TECH-004",
            name="Sunita Patel",
            email="sunita.patel@fleetguard.ai",
            phone="+91-9876543213",
            specialization="Preventive Maintenance",
            is_active=True,
        ),
        Technician(
            employee_id="TECH-005",
            name="Vikram Reddy",
            email="vikram.reddy@fleetguard.ai",
            phone="+91-9876543214",
            specialization="Heavy Vehicle Systems",
            is_active=True,
        ),
    ]
    session.add_all(technicians)
    session.flush()
    return technicians


def create_initial_sensor_readings(session: Session, vehicles: List[Vehicle]) -> None:
    """Create initial sensor readings for all vehicles."""
    import random

    base_readings = {
        VehicleType.TRUCK: {
            "engine_temperature_c": (85, 95),
            "engine_rpm": (1200, 1800),
            "oil_pressure_kpa": (350, 450),
            "coolant_temperature_c": (80, 90),
            "vibration_mm_s": (1.5, 3.0),
            "battery_voltage_v": (13.5, 14.2),
            "fuel_consumption_l_per_100km": (28, 35),
        },
        VehicleType.BUS: {
            "engine_temperature_c": (82, 92),
            "engine_rpm": (1000, 1600),
            "oil_pressure_kpa": (380, 480),
            "coolant_temperature_c": (78, 88),
            "vibration_mm_s": (1.2, 2.5),
            "battery_voltage_v": (13.8, 14.4),
            "fuel_consumption_l_per_100km": (22, 30),
        },
    }

    readings = []
    for vehicle in vehicles:
        # Create a few historical readings
        for i in range(5):
            timestamp = datetime.utcnow() - timedelta(hours=4-i)
            ranges = base_readings.get(vehicle.vehicle_type, base_readings[VehicleType.TRUCK])

            # Adjust for health status
            health_factor = vehicle.health_score / 100.0

            reading = SensorReading(
                vehicle_id=vehicle.id,
                timestamp=timestamp,
                engine_temperature_c=random.uniform(*ranges["engine_temperature_c"]) * (2 - health_factor) * 0.1 + random.uniform(*ranges["engine_temperature_c"]) * health_factor,
                engine_rpm=random.uniform(*ranges["engine_rpm"]),
                oil_pressure_kpa=random.uniform(*ranges["oil_pressure_kpa"]) * health_factor,
                coolant_temperature_c=random.uniform(*ranges["coolant_temperature_c"]) * (2 - health_factor) * 0.1 + random.uniform(*ranges["coolant_temperature_c"]) * health_factor,
                vibration_mm_s=random.uniform(*ranges["vibration_mm_s"]) * (2 - health_factor),
                battery_voltage_v=random.uniform(*ranges["battery_voltage_v"]) * health_factor,
                fuel_consumption_l_per_100km=random.uniform(*ranges["fuel_consumption_l_per_100km"]) * (2 - health_factor),
                operating_hours=vehicle.engine_hours - (4-i) * 10,
                distance_km=vehicle.total_distance_km - (4-i) * 50,
                is_simulated=True,
            )
            readings.append(reading)

        # Add latest reading
        timestamp = datetime.utcnow()
        ranges = base_readings.get(vehicle.vehicle_type, base_readings[VehicleType.TRUCK])
        health_factor = vehicle.health_score / 100.0

        reading = SensorReading(
            vehicle_id=vehicle.id,
            timestamp=timestamp,
            engine_temperature_c=random.uniform(*ranges["engine_temperature_c"]) * (2 - health_factor) * 0.1 + random.uniform(*ranges["engine_temperature_c"]) * health_factor,
            engine_rpm=random.uniform(*ranges["engine_rpm"]),
            oil_pressure_kpa=random.uniform(*ranges["oil_pressure_kpa"]) * health_factor,
            coolant_temperature_c=random.uniform(*ranges["coolant_temperature_c"]) * (2 - health_factor) * 0.1 + random.uniform(*ranges["coolant_temperature_c"]) * health_factor,
            vibration_mm_s=random.uniform(*ranges["vibration_mm_s"]) * (2 - health_factor),
            battery_voltage_v=random.uniform(*ranges["battery_voltage_v"]) * health_factor,
            fuel_consumption_l_per_100km=random.uniform(*ranges["fuel_consumption_l_per_100km"]) * (2 - health_factor),
            operating_hours=vehicle.engine_hours,
            distance_km=vehicle.total_distance_km,
            is_simulated=True,
        )
        readings.append(reading)

    session.add_all(readings)


def create_initial_health_history(session: Session, vehicles: List[Vehicle]) -> None:
    """Create initial health history for vehicles."""
    history = []
    for vehicle in vehicles:
        for i in range(10):
            timestamp = datetime.utcnow() - timedelta(days=9-i)
            # Slight variation in health score over time
            base_score = vehicle.health_score
            variation = (i - 5) * 1.5
            score = max(0, min(100, base_score + variation))

            if score >= 80:
                status = HealthStatus.HEALTHY
            elif score >= 50:
                status = HealthStatus.MAINTENANCE_APPROACHING
            else:
                status = HealthStatus.CRITICAL

            factors = {
                "engine_temperature": "normal" if score > 70 else "elevated",
                "oil_pressure": "normal" if score > 60 else "low",
                "vibration": "normal" if score > 65 else "elevated",
                "battery": "normal" if score > 55 else "degrading",
            }

            health_record = VehicleHealthHistory(
                vehicle_id=vehicle.id,
                timestamp=timestamp,
                health_score=score,
                health_status=status,
                contributing_factors=json.dumps(factors),
            )
            history.append(health_record)

    session.add_all(history)


def initialize_database(database_url: str = None) -> None:
    """Initialize database with schema and seed data."""
    manager = DatabaseManager(database_url)
    manager.create_tables()

    with manager.get_session() as session:
        # Check if already initialized
        existing_vehicles = session.query(Vehicle).count()
        if existing_vehicles > 0:
            print(f"Database already initialized with {existing_vehicles} vehicles. Skipping seed data.")
            return

        print("Creating seed vehicles...")
        vehicles = create_seed_vehicles()
        session.add_all(vehicles)
        session.flush()

        print("Creating seed technicians...")
        create_seed_technicians(session)

        print("Creating initial sensor readings...")
        create_initial_sensor_readings(session, vehicles)

        print("Creating initial health history...")
        create_initial_health_history(session, vehicles)

        session.commit()
        print("Database initialization complete!")


if __name__ == "__main__":
    import sys
    db_url = sys.argv[1] if len(sys.argv) > 1 else None
    initialize_database(db_url)