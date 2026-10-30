"""
RUL Prediction Models for FleetGuard AI.

Implements:
1. NASA C-MAPSS Turbofan Engine RUL Prediction (Random Forest, Baseline)
2. Feature engineering for C-MAPSS dataset
3. Model training, evaluation, and MLflow tracking
4. Dataset validation, RUL clipping, NASA asymmetric scoring
5. Model artifact persistence to disk
"""
import os
import json
import warnings
import hashlib
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

import numpy as np
import pandas as pd
import joblib
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

from app.models.database import MLModel, ModelExperiment, DatasetUpload, utcnow
from app.models.connection import get_db_session


# Suppress warnings
warnings.filterwarnings("ignore")


# Directory for persisted model artifacts (model, scaler, feature config)
ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "artifacts")


def _ensure_artifact_dir() -> str:
    """Create the artifact directory if it doesn't exist."""
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    return ARTIFACT_DIR


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

    def __init__(self, window_size: int = 30, rul_clip: Optional[int] = None):
        self.window_size = window_size
        self.rul_clip = rul_clip
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
        """Add RUL labels to training data (run-to-failure).

        Optionally clips RUL at ``self.rul_clip`` (piecewise target) so that
        engines with very large remaining useful life are treated the same as
        engines near the clip threshold -- a standard C-MAPSS preprocessing
        step that prevents extreme RUL values from dominating the loss.
        """
        df = df.copy()
        max_cycles = df.groupby("unit_number")["cycle"].transform("max")
        df["RUL"] = max_cycles - df["cycle"]
        if self.rul_clip is not None:
            df["RUL"] = df["RUL"].clip(upper=self.rul_clip)
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

    def validate_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate a C-MAPSS dataframe for structural and data-quality issues.

        Checks column schema, missing values, duplicate (unit, cycle) pairs,
        invalid sensor values, and sort order. Returns a dict with
        ``is_valid``, ``errors``, ``warnings`` and summary statistics --
        suitable for persisting to a ``DatasetUpload`` record.
        """
        errors: List[str] = []
        warnings_list: List[str] = []

        expected_cols = set(self.COLUMNS)
        actual_cols = set(df.columns)
        missing_cols = expected_cols - actual_cols
        extra_cols = actual_cols - expected_cols

        if missing_cols:
            errors.append(f"Missing required columns: {sorted(missing_cols)}")
        if extra_cols:
            warnings_list.append(f"Unexpected extra columns: {sorted(extra_cols)}")

        # Row count
        n_rows = len(df)
        if n_rows == 0:
            errors.append("Dataset is empty")
            return {
                "is_valid": False,
                "errors": errors,
                "warnings": warnings_list,
                "row_count": 0,
                "column_count": len(actual_cols),
                "n_units": 0,
                "missing_values": 0,
                "duplicate_unit_cycle_pairs": 0,
                "sorted_by_unit_cycle": False,
                "invalid_sensor_values": 0,
            }

        # Missing values
        missing_values = int(df.isna().sum().sum())
        if missing_values > 0:
            warnings_list.append(f"{missing_values} missing values found")

        # Duplicate (unit_number, cycle) pairs
        dup_pairs = 0
        is_sorted = True
        if "unit_number" in df.columns and "cycle" in df.columns:
            dup_pairs = int(df.duplicated(subset=["unit_number", "cycle"]).sum())
            if dup_pairs > 0:
                errors.append(f"{dup_pairs} duplicate (unit_number, cycle) pairs found")

            # Sort order check
            sorted_df = df.sort_values(["unit_number", "cycle"]).reset_index(drop=True)
            current_df = df.reset_index(drop=True)
            is_sorted = bool((sorted_df == current_df).all().all())
            if not is_sorted:
                warnings_list.append("Data is not sorted by (unit_number, cycle)")
        else:
            is_sorted = False

        # Invalid sensor values (physical plausibility ranges)
        invalid_count = 0
        sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
        plausible_ranges = {
            "sensor_2": (0, 1000),
            "sensor_3": (0, 1000),
            "sensor_4": (0, 1000),
            "sensor_7": (0, 100),
            "sensor_8": (0, 1000),
            "sensor_9": (0, 100),
            "sensor_11": (0, 100),
            "sensor_12": (0, 100),
            "sensor_13": (0, 100),
            "sensor_14": (0, 100),
            "sensor_15": (0, 100),
            "sensor_17": (0, 1000),
            "sensor_20": (0, 100),
            "sensor_21": (0, 100),
        }
        for col in sensor_cols:
            if col in plausible_ranges:
                lo, hi = plausible_ranges[col]
                invalid_count += int(((df[col] < lo) | (df[col] > hi)).sum())
        if invalid_count > 0:
            warnings_list.append(f"{invalid_count} sensor values outside plausible physical ranges")

        n_units = int(df["unit_number"].nunique()) if "unit_number" in df.columns else 0

        return {
            "is_valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings_list,
            "row_count": n_rows,
            "column_count": len(actual_cols),
            "n_units": n_units,
            "missing_values": missing_values,
            "duplicate_unit_cycle_pairs": dup_pairs,
            "sorted_by_unit_cycle": is_sorted,
            "invalid_sensor_values": int(invalid_count),
        }


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


def nasa_asymmetric_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """NASA C-MAPSS asymmetric scoring function.

    The NASA scoring metric penalizes over-prediction (predicting more life
    remains than actually exists) more heavily than under-prediction, because
    unexpected failure is costlier than unnecessary maintenance.

    For error e = y_true - y_pred:
      - e >= 0 (under-predicted):  score += exp(e / 13) - 1
      - e < 0  (over-predicted):   score += exp(-e / 10) - 1

    Lower total score is better. Returns the sum over all samples.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    errors = y_true - y_pred
    scores = np.where(
        errors >= 0,
        np.exp(errors / 13.0) - 1.0,
        np.exp(-errors / 10.0) - 1.0,
    )
    return float(np.sum(scores))


class RULPredictor:
    """RUL Prediction with multiple model support."""

    def __init__(self, model_type: str = "random_forest"):
        self.model_type = model_type
        self.model = None
        self.processor = CMapSSDataProcessor()
        self.metrics: Optional[ModelMetrics] = None
        self.feature_importance: Optional[pd.DataFrame] = None
        # Extended evaluation (populated by train())
        self.nasa_score: Optional[float] = None
        self.per_engine_errors: Dict[int, float] = {}
        self.rul_range_errors: Dict[str, Optional[float]] = {}
        self.training_duration: Optional[float] = None
        self.dataset_version: Optional[str] = None
        self.model_path: Optional[str] = None
        self.scaler_path: Optional[str] = None

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
        """Train RUL prediction model on C-MAPSS data.

        Steps:
          1. Load and validate raw C-MAPSS files
          2. Add RUL labels (with optional clipping)
          3. Engineer rolling-window + statistical features
          4. Split by engine unit (no data leakage)
          5. Fit scaler on train only, transform val
          6. Train model, evaluate with MAE/RMSE/R² + NASA asymmetric score
          7. Persist model + scaler artifacts to disk
          8. Save metadata to database
        """
        start_time = time.time()

        # Load data
        train_df, test_df, rul_df = self.processor.load_data(train_path, test_path, rul_path)

        # Validate training dataset before training
        validation = self.processor.validate_dataset(train_df)
        if not validation["is_valid"]:
            raise ValueError(
                f"Training data validation failed: {'; '.join(validation['errors'])}"
            )

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

        # Get unit numbers for baseline + per-engine error
        train_units = train_data["unit_number"].values
        val_units = val_data["unit_number"].values

        # Train model
        self.model = self._create_model()
        self.model.fit(X_train, y_train)

        # Predict
        y_pred = self.model.predict(X_val)

        # Core metrics
        mae = mean_absolute_error(y_val, y_pred)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        r2 = r2_score(y_val, y_pred)

        # NASA asymmetric scoring (lower is better)
        nasa_score = nasa_asymmetric_score(y_val, y_pred)

        # Per-engine error (MAE per engine unit on validation set)
        per_engine_errors: Dict[int, float] = {}
        val_df_with_units = val_data.copy()
        val_df_with_units["y_true"] = y_val
        val_df_with_units["y_pred"] = y_pred
        for unit_id in val_units:
            unit_mask = val_df_with_units["unit_number"] == unit_id
            if unit_mask.any():
                ue = mean_absolute_error(
                    val_df_with_units.loc[unit_mask, "y_true"],
                    val_df_with_units.loc[unit_mask, "y_pred"],
                )
                per_engine_errors[int(unit_id)] = float(ue)

        # RUL-range binned error: low (0-40), medium (40-80), high (80+)
        def _bin_error(y_t, y_p, lo, hi):
            mask = (y_t >= lo) & (y_t < hi)
            if not mask.any():
                return None
            return float(mean_absolute_error(y_t[mask], y_p[mask]))

        rul_range_errors = {
            "low_0_40": _bin_error(y_val, y_pred, 0, 40),
            "medium_40_80": _bin_error(y_val, y_pred, 40, 80),
            "high_80_plus": _bin_error(y_val, y_pred, 80, float("inf")),
        }

        # Baseline for comparison
        baseline = BaselineRULModel()
        baseline.fit(X_train, y_train, train_units)
        baseline_pred = baseline.predict(X_val, val_units)
        baseline_mae = mean_absolute_error(y_val, baseline_pred)
        baseline_rmse = np.sqrt(mean_squared_error(y_val, baseline_pred))
        baseline_r2 = r2_score(y_val, baseline_pred)

        training_duration = time.time() - start_time

        # Dataset version: hash of training file contents
        dataset_version = None
        if os.path.exists(train_path):
            with open(train_path, "rb") as f:
                dataset_version = hashlib.sha256(f.read()).hexdigest()[:16]

        # Persist model artifacts to disk (model + scaler + feature columns)
        model_path = scaler_path = None
        if session:
            model_record = self._save_model_to_db(
                session, experiment_name, mae, rmse, r2, nasa_score,
                baseline_mae, baseline_rmse, baseline_r2,
                len(X_train), len(X_val),
                train_units=train_units,
                per_engine_errors=per_engine_errors,
                rul_range_errors=rul_range_errors,
                training_duration=training_duration,
                dataset_version=dataset_version,
                validation=validation,
            )
            model_path = model_record.model_path
            scaler_path = model_record.scaler_path

        # MLflow tracking (if available)
        if MLFLOW_AVAILABLE:
            mlflow.set_experiment(experiment_name)
            with mlflow.start_run(run_name=f"{self.model_type}_{utcnow().strftime('%Y%m%d_%H%M%S')}"):
                # Log parameters
                mlflow.log_param("model_type", self.model_type)
                mlflow.log_param("window_size", self.processor.window_size)
                mlflow.log_param("rul_clip", self.processor.rul_clip)
                mlflow.log_param("n_features", X_train.shape[1])
                mlflow.log_param("train_samples", len(X_train))
                mlflow.log_param("val_samples", len(X_val))
                mlflow.log_param("dataset_version", dataset_version)

                # Log metrics
                mlflow.log_metric("mae", mae)
                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("r2", r2)
                mlflow.log_metric("nasa_asymmetric_score", nasa_score)
                mlflow.log_metric("baseline_mae", baseline_mae)
                mlflow.log_metric("baseline_rmse", baseline_rmse)
                mlflow.log_metric("baseline_r2", baseline_r2)
                mlflow.log_metric("training_duration_seconds", training_duration)

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

                # Log model artifact
                mlflow.sklearn.log_model(self.model, "model")

        # Store extended evaluation on the predictor for later inspection
        self.metrics = ModelMetrics(
            mae=mae,
            rmse=rmse,
            r2=r2,
            training_samples=len(X_train),
            validation_samples=len(X_val)
        )
        self.nasa_score = nasa_score
        self.per_engine_errors = per_engine_errors
        self.rul_range_errors = rul_range_errors
        self.training_duration = training_duration
        self.dataset_version = dataset_version
        self.model_path = model_path
        self.scaler_path = scaler_path

        return self.metrics

    def _save_model_to_db(
        self,
        session: Session,
        experiment_name: str,
        mae: float, rmse: float, r2: float, nasa_score: float,
        baseline_mae: float, baseline_rmse: float, baseline_r2: float,
        train_samples: int, val_samples: int,
        train_units: np.ndarray = None,
        per_engine_errors: Dict[int, float] = None,
        rul_range_errors: Dict[str, Optional[float]] = None,
        training_duration: float = None,
        dataset_version: str = None,
        validation: Dict[str, Any] = None
    ) -> MLModel:
        """Save model metadata to database and persist artifacts to disk.

        Persists the trained model, scaler, and feature-column config as
        joblib artifacts under ARTIFACT_DIR, and stores the file paths plus
        extended evaluation metrics on the MLModel / ModelExperiment records.
        """
        # Persist artifacts to disk
        _ensure_artifact_dir()
        version_tag = utcnow().strftime("%Y%m%d_%H%M%S")
        model_filename = f"rul_model_{self.model_type}_{version_tag}.joblib"
        scaler_filename = f"rul_scaler_{self.model_type}_{version_tag}.joblib"
        features_filename = f"rul_features_{self.model_type}_{version_tag}.json"

        model_path = os.path.join(ARTIFACT_DIR, model_filename)
        scaler_path = os.path.join(ARTIFACT_DIR, scaler_filename)
        features_path = os.path.join(ARTIFACT_DIR, features_filename)

        joblib.dump(self.model, model_path)
        joblib.dump(self.processor.scaler, scaler_path)
        with open(features_path, "w") as f:
            json.dump(self.processor.feature_columns, f)

        model_record = MLModel(
            model_name=f"RUL_{self.model_type}",
            model_version=version_tag,
            model_type="rul_regressor",
            dataset_name="NASA_CMAPSS_FD001",
            mae=mae,
            rmse=rmse,
            r2_score=r2,
            training_samples=train_samples,
            validation_samples=val_samples,
            feature_names=json.dumps(self.processor.feature_columns),
            model_path=model_path,
            scaler_path=scaler_path,
            training_config=json.dumps({
                "model_type": self.model_type,
                "window_size": self.processor.window_size,
                "rul_clip": self.processor.rul_clip,
                "nasa_asymmetric_score": nasa_score,
                "baseline_mae": baseline_mae,
                "baseline_rmse": baseline_rmse,
                "baseline_r2": baseline_r2,
                "dataset_version": dataset_version,
                "training_duration_seconds": training_duration,
                "per_engine_errors": {str(k): v for k, v in (per_engine_errors or {}).items()},
                "rul_range_errors": rul_range_errors or {},
                "validation": validation or {},
            }),
            is_active=True,
            trained_by="system",
        )
        session.add(model_record)
        session.flush()

        run_id = None
        if MLFLOW_AVAILABLE and mlflow.active_run():
            run_id = mlflow.active_run().info.run_id

        experiment_metrics = {
            "mae": mae, "rmse": rmse, "r2": r2,
            "nasa_asymmetric_score": nasa_score,
            "baseline_mae": baseline_mae, "baseline_rmse": baseline_rmse, "baseline_r2": baseline_r2,
            "training_duration_seconds": training_duration,
        }
        experiment_record = ModelExperiment(
            model_id=model_record.id,
            experiment_name=experiment_name,
            run_id=run_id,
            hyperparameters=json.dumps({
                "model_type": self.model_type,
                "n_estimators": 200,
                "max_depth": 20,
            }),
            metrics=json.dumps(experiment_metrics),
            status="completed",
            completed_at=utcnow(),
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
