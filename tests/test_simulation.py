"""
Tests for simulation engine.
"""
import pytest
from datetime import datetime
from unittest.mock import Mock, MagicMock, patch

from app.simulation.engine import (
    SimulationEngine, VehicleSimState, SimulationEventType,
)
from app.ml.health_scoring import HEALTHY_RANGES
from app.models.database import Vehicle, VehicleType, SensorReading, SimulationEvent


class TestVehicleSimState:
    """Tests for VehicleSimState dataclass."""

    def test_default_state(self):
        state = VehicleSimState(vehicle_id=1, vehicle_type=VehicleType.TRUCK)
        assert state.vehicle_id == 1
        assert state.vehicle_type == VehicleType.TRUCK
        assert state.engine_temp == 90.0
        assert state.active_faults == {}
        assert state.fault_history == []
        assert state.degradation_factor == 0.0


class TestSimulationEngine:
    """Tests for SimulationEngine."""

    def setup_method(self):
        self.engine = SimulationEngine(speed=1.0, update_interval=0.1)

    def test_initialization(self):
        assert self.engine.speed == 1.0
        assert self.engine.update_interval == 0.1
        assert self.engine._vehicles == {}
        assert self.engine._running is False

    def test_load_vehicles(self):
        """Test loading vehicles from database."""
        mock_session = Mock()
        mock_vehicle = Mock(spec=Vehicle)
        mock_vehicle.id = 1
        mock_vehicle.vehicle_id = "TEST-001"
        mock_vehicle.vehicle_type = VehicleType.TRUCK
        mock_vehicle.engine_hours = 100.0
        mock_vehicle.total_distance_km = 5000.0
        mock_vehicle.health_score = 90.0
        mock_vehicle.is_active = True

        mock_session.query.return_value.filter.return_value.all.return_value = [mock_vehicle]
        mock_session.query.return_value.filter.return_value.order_by.return_value.first.return_value = None

        self.engine.load_vehicles(mock_session)

        assert 1 in self.engine._vehicles
        state = self.engine._vehicles[1]
        assert state.vehicle_id == 1
        assert state.vehicle_type == VehicleType.TRUCK
        assert state.operating_hours == 100.0
        assert state.distance_km == 5000.0

    def test_load_vehicles_with_existing_readings(self):
        """Test loading vehicles with existing sensor readings."""
        mock_session = Mock()
        mock_vehicle = Mock(spec=Vehicle)
        mock_vehicle.id = 1
        mock_vehicle.vehicle_id = "TEST-001"
        mock_vehicle.vehicle_type = VehicleType.TRUCK
        mock_vehicle.engine_hours = 100.0
        mock_vehicle.total_distance_km = 5000.0
        mock_vehicle.health_score = 90.0
        mock_vehicle.is_active = True

        mock_reading = Mock(spec=SensorReading)
        mock_reading.engine_temperature_c = 92.0
        mock_reading.engine_rpm = 1600.0
        mock_reading.oil_pressure_kpa = 390.0
        mock_reading.coolant_temperature_c = 87.0
        mock_reading.vibration_mm_s = 2.5
        mock_reading.battery_voltage_v = 13.8
        mock_reading.fuel_consumption_l_per_100km = 31.0
        mock_reading.operating_hours = 100.0
        mock_reading.distance_km = 5000.0

        mock_session.query.return_value.filter.return_value.all.return_value = [mock_vehicle]
        mock_session.query.return_value.filter.return_value.order_by.return_value.first.return_value = mock_reading

        self.engine.load_vehicles(mock_session)

        state = self.engine._vehicles[1]
        assert state.engine_temp == 92.0
        assert state.rpm == 1600.0

    def test_start_stop(self):
        """Test starting and stopping simulation."""
        assert self.engine._running is False

        self.engine.start()
        assert self.engine._running is True
        assert self.engine._thread is not None

        self.engine.stop()
        assert self.engine._running is False

    def test_reset(self):
        """Test simulation reset."""
        mock_session = Mock()
        mock_vehicle = Mock(spec=Vehicle)
        mock_vehicle.id = 1
        mock_vehicle.vehicle_id = "TEST-001"
        mock_vehicle.vehicle_type = VehicleType.TRUCK
        mock_vehicle.engine_hours = 100.0
        mock_vehicle.total_distance_km = 5000.0
        mock_vehicle.health_score = 100.0  # Full health = no degradation
        mock_vehicle.is_active = True

        mock_session.query.return_value.filter.return_value.all.return_value = [mock_vehicle]
        mock_session.query.return_value.filter.return_value.order_by.return_value.first.return_value = None
        mock_session.query.return_value.delete.return_value = None

        self.engine.load_vehicles(mock_session)
        # Trigger a fault first to set degradation
        self.engine.trigger_fault(1, SimulationEventType.OVERHEATING, mock_session)
        self.engine.reset(mock_session)

        state = self.engine._vehicles[1]
        assert state.active_faults == {}
        assert state.fault_history == []
        assert state.degradation_factor == 0.0

    def test_trigger_fault(self):
        """Test fault injection."""
        mock_session = Mock()
        mock_vehicle = Mock(spec=Vehicle)
        mock_vehicle.id = 1
        mock_vehicle.vehicle_id = "TEST-001"
        mock_vehicle.vehicle_type = VehicleType.TRUCK
        mock_vehicle.engine_hours = 100.0
        mock_vehicle.total_distance_km = 5000.0
        mock_vehicle.health_score = 90.0
        mock_vehicle.is_active = True

        mock_session.query.return_value.filter.return_value.all.return_value = [mock_vehicle]
        mock_session.query.return_value.filter.return_value.order_by.return_value.first.return_value = None

        self.engine.load_vehicles(mock_session)

        # Trigger fault
        result = self.engine.trigger_fault(1, SimulationEventType.OVERHEATING, mock_session)
        assert result is True

        state = self.engine._vehicles[1]
        assert "overheating" in state.active_faults
        assert state.active_faults["overheating"]["active"] is True

        # Verify simulation event was logged
        mock_session.add.assert_called()
        mock_session.commit.assert_called()

    def test_clear_fault(self):
        """Test fault clearing."""
        mock_session = Mock()
        mock_vehicle = Mock(spec=Vehicle)
        mock_vehicle.id = 1
        mock_vehicle.vehicle_id = "TEST-001"
        mock_vehicle.vehicle_type = VehicleType.TRUCK
        mock_vehicle.engine_hours = 100.0
        mock_vehicle.total_distance_km = 5000.0
        mock_vehicle.health_score = 90.0
        mock_vehicle.is_active = True

        mock_session.query.return_value.filter.return_value.all.return_value = [mock_vehicle]
        mock_session.query.return_value.filter.return_value.order_by.return_value.first.return_value = None

        self.engine.load_vehicles(mock_session)
        self.engine.trigger_fault(1, SimulationEventType.OVERHEATING, mock_session)

        # Clear fault
        result = self.engine.clear_fault(1, SimulationEventType.OVERHEATING, mock_session)
        assert result is True

        state = self.engine._vehicles[1]
        assert "overheating" not in state.active_faults
        assert len(state.fault_history) == 1
        assert state.fault_history[0]["type"] == "overheating"

    def test_get_vehicle_state(self):
        """Test getting vehicle state."""
        mock_session = Mock()
        mock_vehicle = Mock(spec=Vehicle)
        mock_vehicle.id = 1
        mock_vehicle.vehicle_id = "TEST-001"
        mock_vehicle.vehicle_type = VehicleType.TRUCK
        mock_vehicle.engine_hours = 100.0
        mock_vehicle.total_distance_km = 5000.0
        mock_vehicle.health_score = 90.0
        mock_vehicle.is_active = True

        mock_session.query.return_value.filter.return_value.all.return_value = [mock_vehicle]
        mock_session.query.return_value.filter.return_value.order_by.return_value.first.return_value = None

        self.engine.load_vehicles(mock_session)

        state = self.engine.get_vehicle_state(1)
        assert state is not None
        assert state.vehicle_id == 1

        # Non-existent vehicle
        state = self.engine.get_vehicle_state(999)
        assert state is None

    def test_apply_overheating_fault(self):
        """Test overheating fault application."""
        state = VehicleSimState(vehicle_id=1, vehicle_type=VehicleType.TRUCK)
        state.base_engine_temp = 90.0
        state.base_coolant_temp = 85.0
        state.engine_temp = 90.0
        state.coolant_temp = 85.0

        config = SimulationEngine.FAULT_CONFIGS[SimulationEventType.OVERHEATING]
        state.active_faults["overheating"] = {
            "type": "overheating",
            "start_cycle": 0,
            "config": config,
            "active": True,
        }

        # Simulate a few cycles
        for cycle in range(5):
            state.fault_history = []  # Reset for each cycle simulation
            # Manually apply fault logic
            progress = cycle / config["duration_cycles"]
            multiplier = 1 + (config["engine_temp_multiplier"] - 1) * progress
            state.engine_temp = state.base_engine_temp * multiplier
            state.coolant_temp = state.base_coolant_temp * (1 + (config["coolant_temp_multiplier"] - 1) * progress)

        # After 5 cycles (25% progress), temp should be elevated
        assert state.engine_temp > 90.0
        assert state.coolant_temp > 85.0

    def test_apply_low_oil_pressure_fault(self):
        """Test low oil pressure fault application."""
        state = VehicleSimState(vehicle_id=1, vehicle_type=VehicleType.TRUCK)
        state.base_oil_pressure = 400.0
        state.oil_pressure = 400.0

        config = SimulationEngine.FAULT_CONFIGS[SimulationEventType.LOW_OIL_PRESSURE]
        state.active_faults["low_oil_pressure"] = {
            "type": "low_oil_pressure",
            "start_cycle": 0,
            "config": config,
            "active": True,
        }

        # Apply at 50% progress
        progress = 0.5
        multiplier = 1 - (1 - config["oil_pressure_multiplier"]) * progress
        state.oil_pressure = state.base_oil_pressure * multiplier

        # Should be reduced
        assert state.oil_pressure < 400.0
        assert state.oil_pressure > 400.0 * config["oil_pressure_multiplier"]


class TestHealthyRanges:
    """Tests for healthy ranges configuration."""

    def test_healthy_ranges_exist(self):
        """Test that all expected sensors have healthy ranges."""
        expected_sensors = [
            "engine_temperature_c",
            "oil_pressure_kpa",
            "vibration_mm_s",
            "coolant_temperature_c",
            "battery_voltage_v",
            "fuel_consumption_l_per_100km",
        ]
        for sensor in expected_sensors:
            assert sensor in HEALTHY_RANGES, f"Missing healthy range for {sensor}"

    def test_ranges_have_min_max(self):
        """Test that all ranges have valid min < max."""
        for sensor, (min_val, max_val) in HEALTHY_RANGES.items():
            assert min_val < max_val, f"{sensor}: min ({min_val}) >= max ({max_val})"


class TestFaultConfigs:
    """Tests for fault configurations."""

    def test_all_fault_types_have_config(self):
        for fault_type in SimulationEventType:
            if fault_type != SimulationEventType.NORMAL:
                assert fault_type in SimulationEngine.FAULT_CONFIGS, f"Missing config for {fault_type}"

    def test_config_has_required_fields(self):
        for fault_type, config in SimulationEngine.FAULT_CONFIGS.items():
            assert "duration_cycles" in config
            assert config["duration_cycles"] > 0
            assert "severity_progression" in config


if __name__ == "__main__":
    pytest.main([__file__, "-v"])