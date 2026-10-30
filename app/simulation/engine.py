"""
Fleet Simulation Engine for FleetGuard AI.

Generates realistic sensor data for simulated vehicles with configurable
fault injection for demonstration purposes.
"""
import random
import json
import threading
import time
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Callable
from collections import deque

from sqlalchemy.orm import Session

from app.models.database import (
    Vehicle, VehicleType, SensorReading, SimulationEvent, HealthStatus, utcnow
)
from app.models.connection import get_db_session


class SimulationEventType(Enum):
    NORMAL = "normal"
    OVERHEATING = "overheating"
    ABNORMAL_VIBRATION = "abnormal_vibration"
    LOW_OIL_PRESSURE = "low_oil_pressure"
    BATTERY_DETERIORATION = "battery_deterioration"
    GRADUAL_DEGRADATION = "gradual_degradation"


@dataclass
class VehicleSimState:
    """Runtime simulation state for a single vehicle."""
    vehicle_id: int
    vehicle_type: VehicleType

    # Base parameters (healthy state)
    base_engine_temp: float = 90.0
    base_rpm: float = 1500.0
    base_oil_pressure: float = 400.0
    base_coolant_temp: float = 85.0
    base_vibration: float = 2.0
    base_battery_voltage: float = 14.0
    base_fuel_consumption: float = 30.0

    # Current values
    engine_temp: float = 90.0
    rpm: float = 1500.0
    oil_pressure: float = 400.0
    coolant_temp: float = 85.0
    vibration: float = 2.0
    battery_voltage: float = 14.0
    fuel_consumption: float = 30.0

    # Operating state
    operating_hours: float = 0.0
    distance_km: float = 0.0

    # Fault state
    active_faults: Dict[str, Dict] = field(default_factory=dict)
    fault_history: List[Dict] = field(default_factory=list)

    # Degradation
    degradation_factor: float = 0.0  # 0.0 = new, 1.0 = end of life

    # Smoothing
    _prev_values: Dict[str, float] = field(default_factory=dict)


class SimulationEngine:
    """
    Deterministic fleet simulation engine.

    Generates continuous sensor readings for all vehicles with support for
    fault injection and gradual degradation modeling.
    """

    # Healthy ranges by vehicle type
    HEALTHY_RANGES = {
        VehicleType.TRUCK: {
            "engine_temp": (85, 95),
            "rpm": (1200, 1800),
            "oil_pressure": (350, 450),
            "coolant_temp": (80, 90),
            "vibration": (1.5, 3.0),
            "battery_voltage": (13.5, 14.2),
            "fuel_consumption": (28, 35),
        },
        VehicleType.BUS: {
            "engine_temp": (82, 92),
            "rpm": (1000, 1600),
            "oil_pressure": (380, 480),
            "coolant_temp": (78, 88),
            "vibration": (1.2, 2.5),
            "battery_voltage": (13.8, 14.4),
            "fuel_consumption": (22, 30),
        },
        VehicleType.VAN: {
            "engine_temp": (80, 90),
            "rpm": (1500, 2200),
            "oil_pressure": (300, 400),
            "coolant_temp": (75, 85),
            "vibration": (1.0, 2.0),
            "battery_voltage": (13.5, 14.0),
            "fuel_consumption": (15, 22),
        },
    }

    # Fault configurations
    FAULT_CONFIGS = {
        SimulationEventType.OVERHEATING: {
            "engine_temp_multiplier": 1.35,
            "coolant_temp_multiplier": 1.25,
            "duration_cycles": 20,
            "severity_progression": "linear",
        },
        SimulationEventType.ABNORMAL_VIBRATION: {
            "vibration_multiplier": 4.0,
            "duration_cycles": 15,
            "severity_progression": "step",
        },
        SimulationEventType.LOW_OIL_PRESSURE: {
            "oil_pressure_multiplier": 0.4,
            "duration_cycles": 10,
            "severity_progression": "linear",
        },
        SimulationEventType.BATTERY_DETERIORATION: {
            "battery_voltage_multiplier": 0.85,
            "duration_cycles": 50,
            "severity_progression": "gradual",
        },
        SimulationEventType.GRADUAL_DEGRADATION: {
            "degradation_rate": 0.02,  # per cycle
            "duration_cycles": 100,
            "severity_progression": "gradual",
        },
    }

    def __init__(
        self,
        speed: float = 1.0,
        update_interval: float = 2.0,
        session_factory: Optional[Callable] = None,
    ):
        """
        Initialize simulation engine.

        Args:
            speed: Simulation speed multiplier (1.0 = real-time)
            update_interval: Seconds between simulation updates
            session_factory: Callable returning a DB session
        """
        self.speed = speed
        self.update_interval = update_interval
        self.session_factory = session_factory or get_db_session

        self._vehicles: Dict[int, VehicleSimState] = {}
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        self._callbacks: List[Callable] = []

        # Deterministic random seed per vehicle for reproducibility
        self._vehicle_seeds: Dict[int, int] = {}

    def register_callback(self, callback: Callable) -> None:
        """Register a callback for simulation updates."""
        self._callbacks.append(callback)

    def _notify_callbacks(self, vehicle_id: int, reading: SensorReading) -> None:
        """Notify all registered callbacks."""
        for callback in self._callbacks:
            try:
                callback(vehicle_id, reading)
            except Exception:
                pass  # Silently ignore callback errors

    def load_vehicles(self, session: Session) -> None:
        """Load vehicle states from database."""
        with self._lock:
            vehicles = session.query(Vehicle).filter(Vehicle.is_active == True).all()

            for vehicle in vehicles:
                # Get latest sensor reading for current state
                latest = session.query(SensorReading).filter(
                    SensorReading.vehicle_id == vehicle.id
                ).order_by(SensorReading.timestamp.desc()).first()

                ranges = self.HEALTHY_RANGES.get(vehicle.vehicle_type, self.HEALTHY_RANGES[VehicleType.TRUCK])

                state = VehicleSimState(
                    vehicle_id=vehicle.id,
                    vehicle_type=vehicle.vehicle_type,
                    base_engine_temp=sum(ranges["engine_temp"]) / 2,
                    base_rpm=sum(ranges["rpm"]) / 2,
                    base_oil_pressure=sum(ranges["oil_pressure"]) / 2,
                    base_coolant_temp=sum(ranges["coolant_temp"]) / 2,
                    base_vibration=sum(ranges["vibration"]) / 2,
                    base_battery_voltage=sum(ranges["battery_voltage"]) / 2,
                    base_fuel_consumption=sum(ranges["fuel_consumption"]) / 2,
                    operating_hours=vehicle.engine_hours or 0,
                    distance_km=vehicle.total_distance_km or 0,
                    degradation_factor=max(0, min(1, 1 - (vehicle.health_score or 100) / 100)),
                )

                if latest:
                    state.engine_temp = latest.engine_temperature_c
                    state.rpm = latest.engine_rpm
                    state.oil_pressure = latest.oil_pressure_kpa
                    state.coolant_temp = latest.coolant_temperature_c
                    state.vibration = latest.vibration_mm_s
                    state.battery_voltage = latest.battery_voltage_v
                    state.fuel_consumption = latest.fuel_consumption_l_per_100km
                    state.operating_hours = latest.operating_hours or vehicle.engine_hours or 0
                    state.distance_km = latest.distance_km or vehicle.total_distance_km or 0

                self._vehicles[vehicle.id] = state
                self._vehicle_seeds[vehicle.id] = hash(vehicle.vehicle_id) % 2**32

    def start(self) -> None:
        """Start the simulation loop."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._simulation_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the simulation loop."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)

    def reset(self, session: Session) -> None:
        """Reset all vehicle states to healthy."""
        with self._lock:
            for state in self._vehicles.values():
                state.active_faults.clear()
                state.fault_history.clear()
                state.degradation_factor = 0.0
                ranges = self.HEALTHY_RANGES.get(state.vehicle_type, self.HEALTHY_RANGES[VehicleType.TRUCK])
                state.engine_temp = state.base_engine_temp
                state.rpm = state.base_rpm
                state.oil_pressure = state.base_oil_pressure
                state.coolant_temp = state.base_coolant_temp
                state.vibration = state.base_vibration
                state.battery_voltage = state.base_battery_voltage
                state.fuel_consumption = state.base_fuel_consumption

            # Clear simulation events from DB
            session.query(SimulationEvent).delete()
            session.query(SensorReading).delete()
            session.commit()

            # Reload vehicles
            self.load_vehicles(session)

    def trigger_fault(self, vehicle_id: int, fault_type: SimulationEventType, session: Session) -> bool:
        """Inject a fault into a specific vehicle."""
        with self._lock:
            if vehicle_id not in self._vehicles:
                return False

            state = self._vehicles[vehicle_id]
            config = self.FAULT_CONFIGS.get(fault_type)

            if not config:
                return False

            # Record fault
            fault_info = {
                "type": fault_type.value,
                "start_cycle": len(state.fault_history),
                "config": config,
                "active": True,
            }
            state.active_faults[fault_type.value] = fault_info

            # Log simulation event
            event = SimulationEvent(
                vehicle_id=vehicle_id,
                event_type=fault_type.value,
                description=f"Fault injected: {fault_type.value}",
                parameters=json.dumps(config),
                triggered_by="demo",
            )
            session.add(event)
            session.commit()

            return True

    def clear_fault(self, vehicle_id: int, fault_type: SimulationEventType, session: Session) -> bool:
        """Clear a specific fault from a vehicle."""
        with self._lock:
            if vehicle_id not in self._vehicles:
                return False

            state = self._vehicles[vehicle_id]
            fault_key = fault_type.value

            if fault_key in state.active_faults:
                state.active_faults[fault_key]["active"] = False
                state.fault_history.append(state.active_faults.pop(fault_key))

                event = SimulationEvent(
                    vehicle_id=vehicle_id,
                    event_type=f"clear_{fault_type.value}",
                    description=f"Fault cleared: {fault_type.value}",
                    triggered_by="demo",
                )
                session.add(event)
                session.commit()
                return True

            return False

    def get_vehicle_state(self, vehicle_id: int) -> Optional[VehicleSimState]:
        """Get current simulation state for a vehicle."""
        with self._lock:
            return self._vehicles.get(vehicle_id)

    def get_all_states(self) -> Dict[int, VehicleSimState]:
        """Get all vehicle states."""
        with self._lock:
            return dict(self._vehicles)

    def _simulation_loop(self) -> None:
        """Main simulation loop."""
        while self._running:
            try:
                with self.session_factory() as session:
                    self._update_all_vehicles(session)
            except Exception as e:
                print(f"Simulation error: {e}")

            time.sleep(self.update_interval / self.speed)

    def _update_all_vehicles(self, session: Session) -> None:
        """Update all vehicles and persist readings."""
        with self._lock:
            vehicle_ids = list(self._vehicles.keys())

        for vehicle_id in vehicle_ids:
            self._update_vehicle(vehicle_id, session)

    def _update_vehicle(self, vehicle_id: int, session: Session) -> None:
        """Update a single vehicle's sensor readings."""
        with self._lock:
            state = self._vehicles.get(vehicle_id)
            if not state:
                return

            # Get deterministic random for this vehicle
            random.seed(self._vehicle_seeds[vehicle_id] + int(state.operating_hours * 10))

            ranges = self.HEALTHY_RANGES.get(state.vehicle_type, self.HEALTHY_RANGES[VehicleType.TRUCK])

            # Apply active faults
            self._apply_faults(state)

            # Apply degradation
            self._apply_degradation(state)

            # Generate new readings with noise
            reading = self._generate_reading(state, ranges)

            # Smooth values
            reading = self._smooth_reading(state, reading)

            # Update operating state
            state.operating_hours += 0.1 * self.speed
            state.distance_km += (state.rpm / 1500) * 0.5 * self.speed

            # Persist to database
            sensor_reading = SensorReading(
                vehicle_id=vehicle_id,
                timestamp=utcnow(),
                engine_temperature_c=reading["engine_temp"],
                engine_rpm=reading["rpm"],
                oil_pressure_kpa=reading["oil_pressure"],
                coolant_temperature_c=reading["coolant_temp"],
                vibration_mm_s=reading["vibration"],
                battery_voltage_v=reading["battery_voltage"],
                fuel_consumption_l_per_100km=reading["fuel_consumption"],
                operating_hours=state.operating_hours,
                distance_km=state.distance_km,
                is_simulated=True,
            )
            session.add(sensor_reading)
            session.flush()

            # Update vehicle totals
            vehicle = session.query(Vehicle).filter(Vehicle.id == vehicle_id).first()
            if vehicle:
                vehicle.engine_hours = state.operating_hours
                vehicle.total_distance_km = state.distance_km
                vehicle.updated_at = utcnow()

            session.commit()

            # Notify callbacks
            self._notify_callbacks(vehicle_id, sensor_reading)

    def _apply_faults(self, state: VehicleSimState) -> None:
        """Apply active faults to vehicle state."""
        faults_to_remove = []

        for fault_key, fault_info in state.active_faults.items():
            if not fault_info["active"]:
                faults_to_remove.append(fault_key)
                continue

            fault_type = SimulationEventType(fault_key)
            config = fault_info["config"]
            cycle = len(state.fault_history) + list(state.active_faults.keys()).index(fault_key)

            if cycle >= config["duration_cycles"]:
                fault_info["active"] = False
                faults_to_remove.append(fault_key)
                continue

            progress = cycle / config["duration_cycles"]

            if fault_type == SimulationEventType.OVERHEATING:
                multiplier = 1 + (config["engine_temp_multiplier"] - 1) * progress
                state.engine_temp = state.base_engine_temp * multiplier
                state.coolant_temp = state.base_coolant_temp * (1 + (config["coolant_temp_multiplier"] - 1) * progress)

            elif fault_type == SimulationEventType.ABNORMAL_VIBRATION:
                if config["severity_progression"] == "step":
                    state.vibration = state.base_vibration * config["vibration_multiplier"]
                else:
                    multiplier = 1 + (config["vibration_multiplier"] - 1) * progress
                    state.vibration = state.base_vibration * multiplier

            elif fault_type == SimulationEventType.LOW_OIL_PRESSURE:
                multiplier = 1 - (1 - config["oil_pressure_multiplier"]) * progress
                state.oil_pressure = state.base_oil_pressure * multiplier

            elif fault_type == SimulationEventType.BATTERY_DETERIORATION:
                multiplier = 1 - (1 - config["battery_voltage_multiplier"]) * progress
                state.battery_voltage = state.base_battery_voltage * multiplier

            elif fault_type == SimulationEventType.GRADUAL_DEGRADATION:
                state.degradation_factor = min(1.0, state.degradation_factor + config["degradation_rate"])

        for key in faults_to_remove:
            if key in state.active_faults:
                state.fault_history.append(state.active_faults.pop(key))

    def _apply_degradation(self, state: VehicleSimState) -> None:
        """Apply gradual degradation to base parameters."""
        deg = state.degradation_factor

        if deg > 0:
            state.engine_temp = state.base_engine_temp * (1 + 0.15 * deg)
            state.coolant_temp = state.base_coolant_temp * (1 + 0.1 * deg)
            state.oil_pressure = state.base_oil_pressure * (1 - 0.2 * deg)
            state.vibration = state.base_vibration * (1 + 0.5 * deg)
            state.battery_voltage = state.base_battery_voltage * (1 - 0.1 * deg)
            state.fuel_consumption = state.base_fuel_consumption * (1 + 0.2 * deg)

    def _generate_reading(self, state: VehicleSimState, ranges: Dict) -> Dict[str, float]:
        """Generate a sensor reading with noise."""
        # Noise factors (standard deviation as fraction of range)
        noise_factors = {
            "engine_temp": 0.02,
            "rpm": 0.03,
            "oil_pressure": 0.025,
            "coolant_temp": 0.02,
            "vibration": 0.05,
            "battery_voltage": 0.01,
            "fuel_consumption": 0.04,
        }

        reading = {}
        for param, (min_val, max_val) in ranges.items():
            base_val = getattr(state, param)
            noise = random.gauss(0, (max_val - min_val) * noise_factors[param])
            reading[param] = base_val + noise

        return reading

    def _smooth_reading(self, state: VehicleSimState, reading: Dict[str, float]) -> Dict[str, float]:
        """Apply exponential smoothing to readings."""
        alpha = 0.3  # Smoothing factor
        smoothed = {}

        for key, value in reading.items():
            prev = state._prev_values.get(key, value)
            smoothed[key] = alpha * value + (1 - alpha) * prev
            state._prev_values[key] = smoothed[key]

        return smoothed


# Global simulation engine instance
_simulation_engine: Optional[SimulationEngine] = None


def get_simulation_engine() -> SimulationEngine:
    """Get or create the global simulation engine."""
    global _simulation_engine
    if _simulation_engine is None:
        _simulation_engine = SimulationEngine()
    return _simulation_engine


def init_simulation_engine(
    speed: float = 1.0,
    update_interval: float = 2.0,
    session_factory: Optional[Callable] = None
) -> SimulationEngine:
    """Initialize the global simulation engine."""
    global _simulation_engine
    _simulation_engine = SimulationEngine(speed, update_interval, session_factory)
    return _simulation_engine