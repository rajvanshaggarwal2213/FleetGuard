"""
Fleet Management Service for FleetGuard AI.

Orchestrates simulation, health monitoring, alerting, and work order generation.
"""
import threading
import time
from datetime import datetime
from typing import Optional, Callable
from contextlib import contextmanager

from sqlalchemy.orm import Session

from app.models.connection import get_db_session, DatabaseManager
from app.simulation.engine import SimulationEngine, SimulationEventType
from app.ml.anomaly_detection import AnomalyDetector
from app.ml.health_scoring import (
    HealthScorer, MaintenanceTriageEngine, process_fleet_health
)


class FleetService:
    """
    Main fleet management service.

    Coordinates:
    - Sensor simulation
    - Anomaly detection
    - Health scoring
    - Maintenance triage
    - Work order generation
    """

    def __init__(
        self,
        simulation_speed: float = 1.0,
        update_interval: float = 2.0,
        db_manager: Optional[DatabaseManager] = None,
    ):
        self.simulation_speed = simulation_speed
        self.update_interval = update_interval
        self.db_manager = db_manager or DatabaseManager()

        # Components
        self.simulation_engine = SimulationEngine(
            speed=simulation_speed,
            update_interval=update_interval,
            session_factory=self.db_manager.get_session_direct
        )
        self.anomaly_detector = AnomalyDetector()
        self.health_scorer = HealthScorer(anomaly_detector=self.anomaly_detector)

        # State
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        self._update_callbacks: list[Callable] = []

    def initialize(self) -> None:
        """Initialize the service - load vehicles, train models."""
        with self.db_manager.get_session() as session:
            # Load vehicles into simulation
            self.simulation_engine.load_vehicles(session)

            # Train anomaly detection models for each vehicle
            self.anomaly_detector.train_all_vehicles(session)

    def register_update_callback(self, callback: Callable) -> None:
        """Register a callback for simulation updates."""
        self._update_callbacks.append(callback)
        self.simulation_engine.register_callback(self._on_simulation_update)

    def _on_simulation_update(self, vehicle_id: int, reading) -> None:
        """Handle simulation update - run health assessment."""
        try:
            with self.db_manager.get_session() as session:
                # Process health for this vehicle
                vehicle = session.query(self.db_manager.SessionLocal().query.__self__.mapper.class_).filter(
                    # This is a bit hacky, better to import Vehicle
                ).filter_by(id=vehicle_id).first()

                if vehicle:
                    # Run triage engine
                    triage = MaintenanceTriageEngine(session)
                    triage.process_vehicle(vehicle)

            # Notify callbacks
            for callback in self._update_callbacks:
                try:
                    callback(vehicle_id, reading)
                except Exception:
                    pass
        except Exception as e:
            print(f"Error in simulation update callback: {e}")

    def start(self) -> None:
        """Start the fleet service."""
        with self._lock:
            if self._running:
                return
            self._running = True
            self.simulation_engine.start()
            self._thread = threading.Thread(target=self._processing_loop, daemon=True)
            self._thread.start()

    def stop(self) -> None:
        """Stop the fleet service."""
        with self._lock:
            self._running = False
            self.simulation_engine.stop()
            if self._thread:
                self._thread.join(timeout=5.0)

    def _processing_loop(self) -> None:
        """Background processing loop for periodic tasks."""
        while self._running:
            try:
                with self.db_manager.get_session() as session:
                    # Process fleet health periodically
                    process_fleet_health(session)
            except Exception as e:
                print(f"Error in processing loop: {e}")

            time.sleep(30)  # Run every 30 seconds

    def trigger_fault(self, vehicle_id: int, fault_type: SimulationEventType) -> bool:
        """Inject a fault into a vehicle."""
        with self.db_manager.get_session() as session:
            return self.simulation_engine.trigger_fault(vehicle_id, fault_type, session)

    def clear_fault(self, vehicle_id: int, fault_type: SimulationEventType) -> bool:
        """Clear a fault from a vehicle."""
        with self.db_manager.get_session() as session:
            return self.simulation_engine.clear_fault(vehicle_id, fault_type, session)

    def reset_simulation(self) -> None:
        """Reset all vehicle states."""
        with self.db_manager.get_session() as session:
            self.simulation_engine.reset(session)

    def get_vehicle_state(self, vehicle_id: int):
        """Get current simulation state for a vehicle."""
        return self.simulation_engine.get_vehicle_state(vehicle_id)

    def get_all_states(self):
        """Get all vehicle states."""
        return self.simulation_engine.get_all_states()

    @contextmanager
    def session(self):
        """Get a database session."""
        with self.db_manager.get_session() as session:
            yield session


# Global fleet service instance
_fleet_service: Optional[FleetService] = None


def get_fleet_service() -> FleetService:
    """Get or create the global fleet service."""
    global _fleet_service
    if _fleet_service is None:
        _fleet_service = FleetService()
    return _fleet_service


def init_fleet_service(
    simulation_speed: float = 1.0,
    update_interval: float = 2.0,
    database_url: Optional[str] = None
) -> FleetService:
    """Initialize the global fleet service."""
    global _fleet_service
    db_manager = DatabaseManager(database_url) if database_url else DatabaseManager()
    _fleet_service = FleetService(simulation_speed, update_interval, db_manager)
    _fleet_service.initialize()
    return _fleet_service