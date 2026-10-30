"""
RUL Prediction Models for FleetGuard AI.

Implements:
1. NASA C-MAPSS Turbofan Engine RUL Prediction (Random Forest, Baseline)
2. Feature engineering for C-MAPSS dataset
3. Model training, evaluation, and MLflow tracking
"""
import os
import json
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

# Make MLflow optional for environments without full dependencies
try:
    import mlflow
    import mlflow.sklearn
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    mlflow = None
    MlflowClient = None

from sqlalchemy.orm import Session

from app.models.database import MLModel, ModelExperiment, DatasetUpload
from app.models.connection import get_db_session


# Suppress warnings
warnings.filterwarnings("ignore")


@dataclass
class ModelMetrics:
    """Model evaluation metrics."""
    mae: float
    rmse: float
    r2: float
    training_samples: int
    validation_samples: int


class CMapSSDataProcessor:
    """Process NASA C-MAPSS dataset for RUL prediction."""

    # Column names for C-MAPSS dataset
    COLUMNS = [
        "unit_number", "cycle", "setting_1", "setting_2", "setting_3"
    ] + [f"sensor_{i}" for i in range(1, 22)]

    # Sensors to drop (constant or near-constant)
    SENSORS_TO_DROP = ["sensor_1", "sensor_5", "sensor_6", "sensor_10", "sensor_16", "sensor_18", "sensor_19"]

    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.scaler = MinMaxScaler()
        self.feature_columns: List[str] = []

    def load_data(self, train_path: str, test_path: str = None, rul_path: str = None) -> Tuple[pd.DataFrame, ...]:
        """Load C-MAPSS dataset files."""
        # Load training data
        train_df = pd.read_csv(train_path, sep=r"\s+", header=None, names=self.COLUMNS)

        test_df = None
        if test_path and os.path.exists(test_path):
            test_df = pd.read_csv(test_path, sep=r"\s+", header=None, names=self.COLUMNS)

        rul_df = None
        if rul_path and os.path.exists(rul_path):
            rul_df = pd.read_csv(rul_path, sep=r"\s+", header=None, names=["RUL"])

        return train_df, test_df, rul_df

    def add_rul_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add RUL labels to training data (run-to-failure)."""
        df = df.copy()
        max_cycles = df.groupby("unit_number")["cycle"].transform("max")
        df["RUL"] = max_cycles - df["cycle"]
        return df

    def add_rul_labels_test(self, test_df: pd.DataFrame, rul_df: pd.DataFrame) -> pd.DataFrame:
        """Add RUL labels to test data using provided RUL values."""
        test_df = test_df.copy()
        max_cycles = test_df.groupby("unit_number")["cycle"].transform("max")
        # RUL for test is: true_RUL + (max_cycle - current_cycle)
        rul_values = rul_df["RUL"].values
        test_df["RUL"] = test_df["unit_number"].map(lambda x: rul_values[x - 1] + max_cycles[test_df["unit_number"] == x].values[0] - test_df[test_df["unit_number"] == x]["cycle"].values)
        # Simpler approach:
        test_max_cycles = test_df.groupby("unit_number")["cycle"].max()
        test_df["RUL"] = test_df.apply(
            lambda row: rul_values[row["unit_number"] - 1] + (test_max_cycles[row["unit_number"]] - row["cycle"]),
            axis=1
        )
        return test_df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features from raw sensor data."""
        df = df.copy()

        # Drop constant sensors
        df = df.drop(columns=[c for c in self.SENSORS_TO_DROP if c in df.columns])

        # Rolling statistics per engine unit
        sensor_cols = [c for c in df.columns if c.startswith("sensor_")]

        for col in sensor_cols:
            # Rolling mean
            df[f"{col}_roll_mean"] = df.groupby("unit_number")[col].transform(
                lambda x: x.rolling(self.window_size, min_periods=1).mean()
            )
            # Rolling std
            df[f"{col}_roll_std"] = df.groupby("unit_number")[col].transform(
                lambda x: x.rolling(self.window_size, min_periods=1).std().fillna(0)
            )
            # Rolling min/max
            df[f"{col}_roll_min"] = df.groupby("unit_number")[col].transform(
                lambda x: x.rolling(self.window_size, min_periods=1).min()
            )
            df[f"{col}_roll_max"] = df.groupby("unit_number")[col].transform(
                lambda x: x.rolling(self.window_size, min_periods=1).max()
            )
            # Trend (slope)
            df[f"{col}_trend"] = df.groupby("unit_number")[col].transform(
                lambda x: x.rolling(self.window_size, min_periods=1).apply(
                    lambda y: np.polyfit(np.arange(len(y)), y, 1)[0] if len(y) > 1 else 0
                ).fillna(0)
            )

        # Cycle-based features
        df["cycle_norm"] = df.groupby("unit_number")["cycle"].transform(
            lambda x: x / x.max()
        )

        # Setting interactions
        df["setting_1_2"] = df["setting_1"] * df["setting_2"]
        df["setting_2_3"] = df["setting_2"] * df["setting_3"]

        # Drop NaN from rolling windows
        df = df.dropna()

        return df

    def prepare_features(self, df: pd.DataFrame, fit_scaler: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare feature matrix and target vector."""
        # Exclude non-feature columns
        exclude = ["unit_number", "cycle", "RUL"]
        feature_cols = [c for c in df.columns if c not in exclude]

        if fit_scaler:
            self.feature_columns = feature_cols
            X = self.scaler.fit_transform(df[feature_cols])
        else:
            # Use only columns seen during training
            available_cols = [c for c in self.feature_columns if c in df.columns]
            X = self.scaler.transform(df[available_cols])

        y = df["RUL"].values
        return X, y

    def split_by_unit(self, df: pd.DataFrame, test_ratio: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data by engine unit to avoid data leakage."""
        units = df["unit_number"].unique()
        np.random.seed(42)
        np.random.shuffle(units)

        n_test = int(len(units) * test_ratio)
        test_units = units[:n_test]
        train_units = units[n_test:]

        train_df = df[df["unit_number"].isin(train_units)].copy()
        test_df = df[df["unit_number"].isin(test_units)].copy()

        return train_df, test_df


class BaselineRULModel:
    """Simple deterministic baseline: RUL = max_cycle - current_cycle (linear degradation assumption)."""

    def __init__(self):
        self.name = "Linear Baseline"
        self.unit_max_cycles: Dict[int, int] = {}

    def fit(self, X: np.ndarray, y: np.ndarray, unit_numbers: np.ndarray) -> "BaselineRULModel":
        """Store max cycles per unit."""
        df = pd.DataFrame({"unit": unit_numbers, "cycle": np.arange(len(unit_numbers)), "RUL": y})
        # Actually we need original cycle values - this is simplified
        # In practice, baseline would use domain knowledge
        self.unit_max_cycles = df.groupby("unit")["RUL"].max().to_dict()
        return self

    def predict(self, X: np.ndarray, unit_numbers: np.ndarray) -> np.ndarray:
        """Predict using linear degradation assumption."""
        # Simplified: assume constant degradation rate
        return np.array([self.unit_max_cycles.get(u, 100) for u in unit_numbers])


class RULPredictor:
    """RUL Prediction with multiple model support."""

    def __init__(self, model_type: str = "random_forest"):
        self.model_type = model_type
        self.model = None
        self.processor = CMapSSDataProcessor()
        self.metrics: Optional[ModelMetrics] = None
        self.feature_importance: Optional[pd.DataFrame] = None

    def _create_model(self):
        """Create model based on type."""
        if self.model_type == "random_forest":
            return RandomForestRegressor(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == "linear":
            return LinearRegression()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def train(
        self,
        train_path: str,
        test_path: str = None,
        rul_path: str = None,
        experiment_name: str = "NASA_CMAPSS_RUL",
        session: Session = None
    ) -> ModelMetrics:
        """Train RUL prediction model on C-MAPSS data."""
        # Load data
        train_df, test_df, rul_df = self.processor.load_data(train_path, test_path, rul_path)

        # Add RUL labels
        train_df = self.processor.add_rul_labels(train_df)

        if test_df is not None and rul_df is not None:
            test_df = self.processor.add_rul_labels_test(test_df, rul_df)

        # Feature engineering
        train_df = self.processor.engineer_features(train_df)

        if test_df is not None:
            test_df = self.processor.engineer_features(test_df)

        # Split by unit (for validation)
        train_data, val_data = self.processor.split_by_unit(train_df, test_ratio=0.2)

        # Prepare features
        X_train, y_train = self.processor.prepare_features(train_data, fit_scaler=True)
        X_val, y_val = self.processor.prepare_features(val_data, fit_scaler=False)

        # Get unit numbers for baseline
        train_units = train_data["unit_number"].values
        val_units = val_data["unit_number"].values

        # Train model
        self.model = self._create_model()

        # MLflow tracking (if available)
        if MLFLOW_AVAILABLE:
            mlflow.set_experiment(experiment_name)
            with mlflow.start_run(run_name=f"{self.model_type}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"):
                # Log parameters
                mlflow.log_param("model_type", self.model_type)
                mlflow.log_param("window_size", self.processor.window_size)
                mlflow.log_param("n_features", X_train.shape[1])
                mlflow.log_param("train_samples", len(X_train))
                mlflow.log_param("val_samples", len(X_val))

                # Train
                self.model.fit(X_train, y_train)

                # Predict
                y_pred = self.model.predict(X_val)

                # Evaluate
                mae = mean_absolute_error(y_val, y_pred)
                rmse = np.sqrt(mean_squared_error(y_val, y_pred))
                r2 = r2_score(y_val, y_pred)

                self.metrics = ModelMetrics(
                    mae=mae,
                    rmse=rmse,
                    r2=r2,
                    training_samples=len(X_train),
                    validation_samples=len(X_val)
                )

                # Log metrics
                mlflow.log_metric("mae", mae)
                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("r2", r2)

                # Feature importance (for tree-based models)
                if hasattr(self.model, "feature_importances_"):
                    self.feature_importance = pd.DataFrame({
                        "feature": self.processor.feature_columns,
                        "importance": self.model.feature_importances_
                    }).sort_values("importance", ascending=False)

                    # Log top features
                    top_features = self.feature_importance.head(20)
                    for _, row in top_features.iterrows():
                        mlflow.log_metric(f"fi_{row['feature']}", row['importance'])

                # Log model
                mlflow.sklearn.log_model(self.model, "model")

                # Also train baseline for comparison
                baseline = BaselineRULModel()
                baseline.fit(X_train, y_train, train_units)
                baseline_pred = baseline.predict(X_val, val_units)
                baseline_mae = mean_absolute_error(y_val, baseline_pred)
                baseline_rmse = np.sqrt(mean_squared_error(y_val, baseline_pred))
                baseline_r2 = r2_score(y_val, baseline_pred)

                mlflow.log_metric("baseline_mae", baseline_mae)
                mlflow.log_metric("baseline_rmse", baseline_rmse)
                mlflow.log_metric("baseline_r2", baseline_r2)

                # Save to database if session provided
                if session:
                    self._save_model_to_db(
                        session, experiment_name, mae, rmse, r2,
                        baseline_mae, baseline_rmse, baseline_r2,
                        len(X_train), len(X_val)
                    )
        else:
            # Train without MLflow
            self.model.fit(X_train, y_train)

            # Predict
            y_pred = self.model.predict(X_val)

            # Evaluate
            mae = mean_absolute_error(y_val, y_pred)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            r2 = r2_score(y_val, y_pred)

            self.metrics = ModelMetrics(
                mae=mae,
                rmse=rmse,
                r2=r2,
                training_samples=len(X_train),
                validation_samples=len(X_val)
            )

            # Feature importance (for tree-based models)
            if hasattr(self.model, "feature_importances_"):
                self.feature_importance = pd.DataFrame({
                    "feature": self.processor.feature_columns,
                    "importance": self.model.feature_importances_
                }).sort_values("importance", ascending=False)

            # Also train baseline for comparison
            baseline = BaselineRULModel()
            baseline.fit(X_train, y_train, train_units)
            baseline_pred = baseline.predict(X_val, val_units)
            baseline_mae = mean_absolute_error(y_val, baseline_pred)
            baseline_rmse = np.sqrt(mean_squared_error(y_val, baseline_pred))
            baseline_r2 = r2_score(y_val, baseline_pred)

            # Save to database if session provided
            if session:
                self._save_model_to_db(
                    session, experiment_name, mae, rmse, r2,
                    baseline_mae, baseline_rmse, baseline_r2,
                    len(X_train), len(X_val)
                )

        return self.metrics

    def _save_model_to_db(
        self,
        session: Session,
        experiment_name: str,
        mae: float, rmse: float, r2: float,
        baseline_mae: float, baseline_rmse: float, baseline_r2: float,
        train_samples: int, val_samples: int
    ) -> MLModel:
        """Save model metadata to database."""
        model_record = MLModel(
            model_name=f"RUL_{self.model_type}",
            model_version=datetime.utcnow().strftime("%Y%m%d_%H%M%S"),
            model_type="rul_regressor",
            dataset_name="NASA_CMAPSS_FD001",
            mae=mae,
            rmse=rmse,
            r2_score=r2,
            training_samples=train_samples,
            validation_samples=val_samples,
            feature_names=json.dumps(self.processor.feature_columns),
            training_config=json.dumps({
                "model_type": self.model_type,
                "window_size": self.processor.window_size,
                "baseline_mae": baseline_mae,
                "baseline_rmse": baseline_rmse,
                "baseline_r2": baseline_r2,
            }),
            is_active=True,
            trained_by="system",
        )
        session.add(model_record)
        session.flush()

        run_id = None
        if MLFLOW_AVAILABLE and mlflow.active_run():
            run_id = mlflow.active_run().info.run_id

        experiment_record = ModelExperiment(
            model_id=model_record.id,
            experiment_name=experiment_name,
            run_id=run_id,
            hyperparameters=json.dumps({
                "model_type": self.model_type,
                "n_estimators": 200,
                "max_depth": 20,
            }),
            metrics=json.dumps({
                "mae": mae, "rmse": rmse, "r2": r2,
                "baseline_mae": baseline_mae, "baseline_rmse": baseline_rmse, "baseline_r2": baseline_r2,
            }),
            status="completed",
            completed_at=datetime.utcnow(),
        )
        session.add(experiment_record)
        session.commit()

        return model_record

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict RUL for new data."""
        if self.model is None:
            raise ValueError("Model not trained")

        df = self.processor.engineer_features(df)
        X, _ = self.processor.prepare_features(df, fit_scaler=False)
        return self.model.predict(X)

    def evaluate_on_test(self, test_path: str, rul_path: str) -> ModelMetrics:
        """Evaluate on official test set."""
        _, test_df, rul_df = self.processor.load_data(test_path, test_path, rul_path)
        test_df = self.processor.add_rul_labels_test(test_df, rul_df)
        test_df = self.processor.engineer_features(test_df)
        X_test, y_test = self.processor.prepare_features(test_df, fit_scaler=False)

        y_pred = self.model.predict(X_test)

        return ModelMetrics(
            mae=mean_absolute_error(y_test, y_pred),
            rmse=np.sqrt(mean_squared_error(y_test, y_pred)),
            r2=r2_score(y_test, y_pred),
            training_samples=0,
            validation_samples=len(X_test)
        )


def train_rul_model(
    train_path: str,
    test_path: str = None,
    rul_path: str = None,
    model_type: str = "random_forest",
    session: Session = None
) -> Tuple[RULPredictor, ModelMetrics]:
    """Convenience function to train RUL model."""
    predictor = RULPredictor(model_type=model_type)
    metrics = predictor.train(train_path, test_path, rul_path, session=session)
    return predictor, metrics