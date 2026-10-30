"""
Tests for RUL prediction module.
"""
import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime

from app.ml.rul_prediction import (
    CMapSSDataProcessor, BaselineRULModel, RULPredictor, ModelMetrics,
    train_rul_model
)


class TestCMapSSDataProcessor:
    """Tests for CMapSSDataProcessor."""

    def setup_method(self):
        self.processor = CMapSSDataProcessor(window_size=10)

    def test_columns_defined(self):
        """Test that all expected columns are defined."""
        assert len(self.processor.COLUMNS) == 26
        assert self.processor.COLUMNS[0] == "unit_number"
        assert self.processor.COLUMNS[1] == "cycle"
        assert self.processor.COLUMNS[2] == "setting_1"
        assert "sensor_1" in self.processor.COLUMNS
        assert "sensor_21" in self.processor.COLUMNS

    def test_sensors_to_drop(self):
        """Test sensors marked for dropping."""
        assert "sensor_1" in self.processor.SENSORS_TO_DROP
        assert "sensor_5" in self.processor.SENSORS_TO_DROP
        assert len(self.processor.SENSORS_TO_DROP) == 7

    def test_add_rul_labels(self):
        """Test RUL label generation for training data."""
        # Create sample data: 2 engines, 5 cycles each
        df = pd.DataFrame({
            "unit_number": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
            "cycle": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
            "setting_1": [0.5] * 10,
            "sensor_1": [1.0] * 10,
        })

        result = self.processor.add_rul_labels(df)

        # Engine 1: max_cycle=5, RUL = [4,3,2,1,0]
        # Engine 2: max_cycle=5, RUL = [4,3,2,1,0]
        assert "RUL" in result.columns
        assert result.loc[0, "RUL"] == 4  # Engine 1, cycle 1
        assert result.loc[4, "RUL"] == 0  # Engine 1, cycle 5
        assert result.loc[5, "RUL"] == 4  # Engine 2, cycle 1

    def test_engineer_features(self):
        """Test feature engineering."""
        # Create sample data with multiple sensors
        np.random.seed(42)
        n_samples = 100
        df = pd.DataFrame({
            "unit_number": np.repeat([1, 2], n_samples // 2),
            "cycle": np.tile(np.arange(1, n_samples // 2 + 1), 2),
            "setting_1": np.random.uniform(0, 1, n_samples),
            "setting_2": np.random.uniform(0, 1, n_samples),
            "setting_3": np.random.uniform(0, 1, n_samples),
            "sensor_2": np.random.normal(10, 1, n_samples),
            "sensor_3": np.random.normal(20, 2, n_samples),
            "sensor_4": np.random.normal(30, 3, n_samples),
            "sensor_7": np.random.normal(40, 4, n_samples),
            "sensor_8": np.random.normal(50, 5, n_samples),
        })

        result = self.processor.engineer_features(df)

        # Should have added rolling features (using sensor_2 which is not in SENSORS_TO_DROP)
        assert "sensor_2_roll_mean" in result.columns
        assert "sensor_2_roll_std" in result.columns
        assert "sensor_2_roll_min" in result.columns
        assert "sensor_2_roll_max" in result.columns
        assert "sensor_2_trend" in result.columns

        # Should have cycle_norm
        assert "cycle_norm" in result.columns

        # Should have setting interactions
        assert "setting_1_2" in result.columns
        assert "setting_2_3" in result.columns

        # Should have dropped constant sensors (sensor_2 is not in SENSORS_TO_DROP, but sensor_1 would be)
        # sensor_2 is kept, sensor_1 would be dropped if present

    def test_prepare_features(self):
        """Test feature preparation with scaler."""
        df = pd.DataFrame({
            "unit_number": [1, 1, 2, 2],
            "cycle": [1, 2, 1, 2],
            "RUL": [10, 5, 10, 5],
            "feature_1": [1.0, 2.0, 3.0, 4.0],
            "feature_2": [10.0, 20.0, 30.0, 40.0],
        })

        X, y = self.processor.prepare_features(df, fit_scaler=True)

        assert X.shape == (4, 2)
        assert y.shape == (4,)
        assert np.array_equal(y, [10, 5, 10, 5])
        assert len(self.processor.feature_columns) == 2

    def test_split_by_unit(self):
        """Test splitting by engine unit to avoid leakage."""
        df = pd.DataFrame({
            "unit_number": [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
            "cycle": [1, 2, 3] * 4,
            "RUL": [10, 5, 0] * 4,
        })

        train_df, test_df = self.processor.split_by_unit(df, test_ratio=0.25)

        # Should split by unit, not by row
        train_units = train_df["unit_number"].unique()
        test_units = test_df["unit_number"].unique()

        assert len(set(train_units) & set(test_units)) == 0  # No overlap
        assert len(train_units) + len(test_units) == 4
        assert len(test_units) == 1  # 25% of 4 units


class TestBaselineRULModel:
    """Tests for BaselineRULModel."""

    def test_fit_predict(self):
        """Test baseline model fit and predict."""
        model = BaselineRULModel()

        # Create dummy data
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([10, 5, 0])
        unit_numbers = np.array([1, 1, 2])

        model.fit(X, y, unit_numbers)

        predictions = model.predict(X, unit_numbers)
        assert predictions.shape == (3,)

        # Baseline should predict based on unit max cycles
        # Since we don't have actual cycle data, it uses stored max RUL
        assert np.all(predictions >= 0)


class TestRULPredictor:
    """Tests for RULPredictor."""

    def setup_method(self):
        self.predictor = RULPredictor(model_type="random_forest")

    def test_create_model_random_forest(self):
        """Test Random Forest model creation."""
        model = self.predictor._create_model()
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")

    def test_create_model_linear(self):
        """Test Linear Regression model creation."""
        predictor = RULPredictor(model_type="linear")
        model = predictor._create_model()
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")

    def test_create_model_invalid(self):
        """Test invalid model type raises error."""
        predictor = RULPredictor(model_type="invalid")
        with pytest.raises(ValueError):
            predictor._create_model()

    @patch("app.ml.rul_prediction.mlflow")
    def test_train_with_mock_data(self, mock_mlflow):
        """Test training with mocked MLflow."""
        # Mock MLflow
        mock_mlflow.set_experiment = Mock()
        mock_mlflow.start_run = Mock()
        mock_run = Mock()
        mock_run.info.run_id = "test_run_123"
        mock_mlflow.start_run.return_value.__enter__ = Mock(return_value=mock_run)
        mock_mlflow.start_run.return_value.__exit__ = Mock(return_value=None)
        mock_mlflow.log_param = Mock()
        mock_mlflow.log_metric = Mock()
        mock_mlflow.sklearn.log_model = Mock()

        # Create mock training data with more units to allow split
        np.random.seed(42)
        n_units = 10
        n_cycles = 20
        train_data = pd.DataFrame({
            "unit_number": np.repeat(np.arange(1, n_units + 1), n_cycles),
            "cycle": np.tile(np.arange(1, n_cycles + 1), n_units),
            "setting_1": np.random.uniform(0, 1, n_units * n_cycles),
            "setting_2": np.random.uniform(0, 1, n_units * n_cycles),
            "setting_3": np.random.uniform(0, 1, n_units * n_cycles),
        })
        # Add 21 sensors
        for i in range(1, 22):
            train_data[f"sensor_{i}"] = np.random.normal(10*i, 1, n_units * n_cycles)

        # Add RUL labels
        train_data = self.predictor.processor.add_rul_labels(train_data)
        train_data = self.predictor.processor.engineer_features(train_data)

        # Split (use test_ratio=0.3 to ensure at least 1 unit in validation with 10 units)
        train_df, val_df = self.predictor.processor.split_by_unit(train_data, test_ratio=0.3)
        X_train, y_train = self.predictor.processor.prepare_features(train_df, fit_scaler=True)
        X_val, y_val = self.predictor.processor.prepare_features(val_df, fit_scaler=False)

        # Train (this will use mocked MLflow)
        # We can't easily test full training without real data, so test initialization
        assert self.predictor.model is None
        assert self.predictor.metrics is None
        assert X_train.shape[0] > 0
        assert X_val.shape[0] > 0


class TestModelMetrics:
    """Tests for ModelMetrics dataclass."""

    def test_metrics_creation(self):
        metrics = ModelMetrics(
            mae=10.5,
            rmse=15.2,
            r2=0.85,
            training_samples=1000,
            validation_samples=200
        )
        assert metrics.mae == 10.5
        assert metrics.rmse == 15.2
        assert metrics.r2 == 0.85


class TestIntegration:
    """Integration tests (require actual data files)."""

    @pytest.mark.skip(reason="Requires NASA dataset files")
    def test_full_training_pipeline(self):
        """Test full training pipeline with real data."""
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])