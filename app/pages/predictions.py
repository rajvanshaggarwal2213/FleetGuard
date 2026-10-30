"""
AI/ML Predictions Page for FleetGuard AI.

Shows RUL predictions for NASA benchmark engines and simulated vehicle health assessments.
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta
import json
import os

from sqlalchemy.orm import Session
from sqlalchemy import func

from app.models.database import (
    Vehicle, MLModel, ModelExperiment, DatasetUpload,
    SensorReading, Anomaly, AlertSeverity
)
from app.models.connection import get_db_session
from app.ml.rul_prediction import RULPredictor, CMapSSDataProcessor


def render():
    """Render the AI/ML predictions page."""
    st.markdown('<div class="chart-title">🤖 AI/ML Predictions</div>', unsafe_allow_html=True)

    # Important notice
    st.warning("""
    ⚠️ **Important Distinction**: This page shows two separate prediction systems:
    1. **NASA C-MAPSS Benchmark** - Turbofan engine RUL predictions (validated on NASA dataset)
    2. **Fleet Simulator Health** - Simulated commercial vehicle health scores (demo rules only)

    Do NOT apply NASA turbofan models to truck sensor data. They are completely different systems.
    """)

    tab1, tab2, tab3, tab4 = st.tabs([
        "🚀 NASA Benchmark (FD001)", "🚛 Fleet Health Predictions", "📊 Model Performance", "🧪 Experiments"
    ])

    with tab1:
        _render_nasa_benchmark_tab()

    with tab2:
        _render_fleet_health_tab()

    with tab3:
        _render_model_performance_tab()

    with tab4:
        _render_experiments_tab()


def _render_nasa_benchmark_tab():
    """Render NASA C-MAPSS benchmark predictions."""
    st.markdown('<div class="chart-title">NASA C-MAPSS Turbofan Engine RUL Prediction (FD001)</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### Dataset Upload")
        uploaded_train = st.file_uploader("Training Data (train_FD001.txt)", type=["txt"], key="nasa_train")
        uploaded_test = st.file_uploader("Test Data (test_FD001.txt)", type=["txt"], key="nasa_test")
        uploaded_rul = st.file_uploader("RUL Labels (RUL_FD001.txt)", type=["txt"], key="nasa_rul")

        if st.button("🚀 Train RUL Model", type="primary", use_container_width=True):
            if uploaded_train:
                _train_nasa_model(uploaded_train, uploaded_test, uploaded_rul)
            else:
                st.error("Please upload training data file.")

        st.markdown("---")

        # Model selection
        with get_db_session() as session:
            models = session.query(MLModel).filter(
                MLModel.model_type == "rul_regressor",
                MLModel.is_active == True
            ).order_by(MLModel.created_at.desc()).all()

            if models:
                st.markdown("### Available Models")
                for m in models:
                    config = json.loads(m.training_config) if m.training_config else {}
                    baseline_mae = config.get("baseline_mae", "N/A")
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">{m.model_name} v{m.model_version}</div>
                        <div class="metric-value" style="font-size: 1rem;">MAE: {m.mae:.2f} | RMSE: {m.rmse:.2f} | R²: {m.r2_score:.3f}</div>
                        <div style="font-size: 0.75rem; color: #6c757d;">Baseline MAE: {baseline_mae} | Trained: {m.created_at.strftime('%Y-%m-%d')}</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No trained models yet. Upload NASA dataset and train a model.")

    with col2:
        # Show prediction visualization if model exists
        with get_db_session() as session:
            latest_model = session.query(MLModel).filter(
                MLModel.model_type == "rul_regressor",
                MLModel.is_active == True
            ).order_by(MLModel.created_at.desc()).first()

            if latest_model and uploaded_test and uploaded_rul:
                _show_nasa_predictions(latest_model, uploaded_test, uploaded_rul)
            elif latest_model:
                _show_model_info(latest_model)
            else:
                st.info("Train a model to see predictions on test data.")


def _train_nasa_model(train_file, test_file, rul_file):
    """Train NASA RUL model."""
    with st.spinner("Training RUL model on NASA C-MAPSS data..."):
        try:
            # Save uploaded files temporarily
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(train_file.getvalue().decode('utf-8'))
                train_path = f.name

            test_path = None
            if test_file:
                with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                    f.write(test_file.getvalue().decode('utf-8'))
                    test_path = f.name

            rul_path = None
            if rul_file:
                with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                    f.write(rul_file.getvalue().decode('utf-8'))
                    rul_path = f.name

            # Train model
            with get_db_session() as session:
                predictor = RULPredictor(model_type="random_forest")
                metrics = predictor.train(train_path, test_path, rul_path, session=session)

                # Save dataset upload record
                upload = DatasetUpload(
                    filename=train_file.name,
                    dataset_type="nasa_fd001",
                    file_size_bytes=len(train_file.getvalue()),
                    is_valid=True,
                    processed_at=datetime.utcnow(),
                    uploaded_by="user",
                )
                session.add(upload)
                session.commit()

            st.success(f"✅ Model trained successfully! MAE: {metrics.mae:.2f}, RMSE: {metrics.rmse:.2f}, R²: {metrics.r2:.3f}")
            st.rerun()

        except Exception as e:
            st.error(f"Training failed: {str(e)}")
        finally:
            # Cleanup temp files
            for p in [train_path, test_path, rul_path]:
                if p and os.path.exists(p):
                    os.unlink(p)


def _show_nasa_predictions(model_record: MLModel, test_file, rul_file):
    """Show NASA benchmark predictions vs actual."""
    st.markdown("### Test Set Predictions vs Actual RUL")

    try:
        # Load and process test data
        processor = CMapSSDataProcessor()
        test_df, _, rul_df = processor.load_data(test_file.name, test_file.name, rul_file.name)
        # Actually need to read from uploaded files properly
        test_content = test_file.getvalue().decode('utf-8')
        rul_content = rul_file.getvalue().decode('utf-8')

        import io
        test_df = pd.read_csv(io.StringIO(test_content), sep=r"\s+", header=None, names=processor.COLUMNS)
        rul_df = pd.read_csv(io.StringIO(rul_content), sep=r"\s+", header=None, names=["RUL"])

        test_df = processor.add_rul_labels_test(test_df, rul_df)
        test_df = processor.engineer_features(test_df)
        X_test, y_test = processor.prepare_features(test_df, fit_scaler=False)

        # Load model from MLflow
        import mlflow
        mlflow.set_tracking_uri("file:///app/mlruns")
        model_uri = f"runs:/{model_record.experiments[0].run_id}/model" if model_record.experiments else None

        if model_uri:
            ml_model = mlflow.sklearn.load_model(model_uri)
            y_pred = ml_model.predict(X_test)

            # Plot actual vs predicted
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=y_test, y=y_pred,
                mode="markers",
                name="Predictions",
                marker=dict(color="#00a8e8", size=8, opacity=0.6),
                hovertemplate="Actual: %{x}<br>Predicted: %{y}<extra></extra>",
            ))

            # Perfect prediction line
            max_val = max(y_test.max(), y_pred.max())
            fig.add_trace(go.Scatter(
                x=[0, max_val], y=[0, max_val],
                mode="lines",
                name="Perfect Prediction",
                line=dict(color="#dc3545", dash="dash", width=2),
            ))

            fig.update_layout(
                title="Actual vs Predicted RUL",
                xaxis_title="Actual RUL (cycles)",
                yaxis_title="Predicted RUL (cycles)",
                height=400,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#1e3a5f"),
            )

            st.plotly_chart(fig, use_container_width=True)

            # Residuals
            residuals = y_test - y_pred
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=y_pred, y=residuals,
                mode="markers",
                marker=dict(color="#6f42c1", size=6, opacity=0.6),
            ))
            fig2.add_hline(y=0, line_dash="dash", line_color="red")
            fig2.update_layout(
                title="Residual Plot",
                xaxis_title="Predicted RUL",
                yaxis_title="Residual (Actual - Predicted)",
                height=300,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#1e3a5f"),
            )
            st.plotly_chart(fig2, use_container_width=True)

            # Metrics
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            cols = st.columns(3)
            cols[0].metric("Test MAE", f"{mae:.2f}")
            cols[1].metric("Test RMSE", f"{rmse:.2f}")
            cols[2].metric("Test R²", f"{r2:.3f}")

    except Exception as e:
        st.error(f"Error generating predictions: {str(e)}")


def _show_model_info(model_record: MLModel):
    """Show model information."""
    st.markdown("### Model Information")

    config = json.loads(model_record.training_config) if model_record.training_config else {}

    st.json({
        "Model": model_record.model_name,
        "Version": model_record.model_version,
        "Type": model_record.model_type,
        "Dataset": model_record.dataset_name,
        "MAE": model_record.mae,
        "RMSE": model_record.rmse,
        "R²": model_record.r2_score,
        "Training Samples": model_record.training_samples,
        "Validation Samples": model_record.validation_samples,
        "Baseline MAE": config.get("baseline_mae", "N/A"),
        "Baseline RMSE": config.get("baseline_rmse", "N/A"),
        "Baseline R²": config.get("baseline_r2", "N/A"),
        "Features": len(json.loads(model_record.feature_names)) if model_record.feature_names else 0,
    })


def _render_fleet_health_tab():
    """Render fleet health predictions (simulated vehicles)."""
    st.markdown('<div class="chart-title">Simulated Fleet Vehicle Health Predictions</div>', unsafe_allow_html=True)

    st.info("""
    ℹ️ **Demo Mode**: These predictions are based on rule-based health scoring of SIMULATED sensor data.
    They do NOT use the NASA turbofan model. Real deployment would require validated models trained on
    actual commercial vehicle telematics data.
    """)

    with get_db_session() as session:
        vehicles = session.query(Vehicle).filter(Vehicle.is_active == True).all()

        if not vehicles:
            st.warning("No vehicles in fleet.")
            return

        # Health predictions table
        pred_data = []
        for v in vehicles:
            # Get latest health assessment
            latest_health = session.query(VehicleHealthHistory).filter(
                VehicleHealthHistory.vehicle_id == v.id
            ).order_by(VehicleHealthHistory.timestamp.desc()).first()

            # Count active anomalies
            active_anomalies = session.query(Anomaly).filter(
                Anomaly.vehicle_id == v.id,
                Anomaly.is_resolved == False
            ).count()

            # Estimate maintenance window (simplified)
            if v.current_health_status == HealthStatus.CRITICAL:
                maint_window = "Immediate"
            elif v.current_health_status == HealthStatus.MAINTENANCE_APPROACHING:
                maint_window = "Within 7 days"
            else:
                maint_window = "Normal schedule"

            pred_data.append({
                "Vehicle": v.vehicle_id,
                "Type": v.vehicle_type.value.title(),
                "Health Score": f"{v.health_score:.1f}%",
                "Status": v.current_health_status.value.replace("_", " ").title(),
                "Active Anomalies": active_anomalies,
                "Est. Maintenance": maint_window,
                "Last Service": v.last_service_date.strftime("%Y-%m-%d") if v.last_service_date else "Never",
            })

        df = pd.DataFrame(pred_data)

        def style_status(val):
            if val == "Healthy":
                return "background-color: rgba(40,167,69,0.15); color: #28a745; font-weight: 600;"
            elif val == "Maintenance Approaching":
                return "background-color: rgba(255,193,7,0.15); color: #b8860b; font-weight: 600;"
            elif val == "Critical":
                return "background-color: rgba(220,53,69,0.15); color: #dc3545; font-weight: 600;"
            return ""

        def style_maint(val):
            if val == "Immediate":
                return "color: #dc3545; font-weight: 700;"
            elif val == "Within 7 days":
                return "color: #ffc107; font-weight: 700;"
            return ""

        st.dataframe(
            df.style.map(style_status, subset=["Status"]).map(style_maint, subset=["Est. Maintenance"]),
            use_container_width=True,
            hide_index=True,
        )

        # Health score timeline for all vehicles
        st.markdown('<div class="chart-title">Fleet Health Score Timeline (Last 7 Days)</div>', unsafe_allow_html=True)

        cutoff = datetime.utcnow() - timedelta(days=7)
        history = session.query(VehicleHealthHistory).filter(
            VehicleHealthHistory.timestamp >= cutoff
        ).order_by(VehicleHealthHistory.timestamp).all()

        if history:
            df_hist = pd.DataFrame([{
                "timestamp": h.timestamp,
                "health_score": h.health_score,
                "vehicle_id": h.vehicle_id,
            } for h in history])

            # Get vehicle IDs
            vehicle_map = {v.id: v.vehicle_id for v in vehicles}
            df_hist["vehicle"] = df_hist["vehicle_id"].map(vehicle_map)

            fig = go.Figure()
            for vehicle_id in df_hist["vehicle"].unique():
                subset = df_hist[df_hist["vehicle"] == vehicle_id]
                fig.add_trace(go.Scatter(
                    x=subset["timestamp"],
                    y=subset["health_score"],
                    mode="lines",
                    name=vehicle_id,
                    line=dict(width=2),
                    opacity=0.7,
                ))

            fig.add_hline(y=80, line_dash="dash", line_color="#28a745", annotation_text="Healthy")
            fig.add_hline(y=50, line_dash="dash", line_color="#ffc107", annotation_text="Warning")

            fig.update_layout(
                height=400,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#1e3a5f"),
                xaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.05)"),
                yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.05)", range=[0, 105]),
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
            )

            st.plotly_chart(fig, use_container_width=True)


def _render_model_performance_tab():
    """Render model performance comparison."""
    st.markdown('<div class="chart-title">Model Performance Comparison</div>', unsafe_allow_html=True)

    with get_db_session() as session:
        models = session.query(MLModel).filter(
            MLModel.model_type == "rul_regressor"
        ).order_by(MLModel.created_at.desc()).all()

        if not models:
            st.info("No models trained yet.")
            return

        # Model comparison table
        model_data = []
        for m in models:
            config = json.loads(m.training_config) if m.training_config else {}
            model_data.append({
                "Model": f"{m.model_name} v{m.model_version}",
                "Dataset": m.dataset_name,
                "MAE": f"{m.mae:.2f}" if m.mae else "N/A",
                "RMSE": f"{m.rmse:.2f}" if m.rmse else "N/A",
                "R²": f"{m.r2_score:.3f}" if m.r2_score else "N/A",
                "Baseline MAE": f"{config.get('baseline_mae', 'N/A')}",
                "Improvement": f"{(config.get('baseline_mae', 0) - m.mae) / config.get('baseline_mae', 1) * 100:.1f}%" if config.get('baseline_mae') and m.mae else "N/A",
                "Status": "Active" if m.is_active else "Archived",
                "Created": m.created_at.strftime("%Y-%m-%d %H:%M"),
            })

        df = pd.DataFrame(model_data)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Performance chart
        active_models = [m for m in models if m.mae is not None]
        if len(active_models) > 1:
            fig = go.Figure()

            fig.add_trace(go.Bar(
                x=[f"{m.model_name} v{m.model_version}" for m in active_models],
                y=[m.mae for m in active_models],
                name="MAE",
                marker_color="#00a8e8",
                yaxis="y",
            ))

            fig.add_trace(go.Bar(
                x=[f"{m.model_name} v{m.model_version}" for m in active_models],
                y=[m.rmse for m in active_models],
                name="RMSE",
                marker_color="#ffc107",
                yaxis="y",
            ))

            fig.update_layout(
                title="Model Error Metrics Comparison",
                barmode="group",
                height=400,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#1e3a5f"),
                yaxis=dict(title="Error (cycles)"),
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
            )

            st.plotly_chart(fig, use_container_width=True)


def _render_experiments_tab():
    """Render ML experiment tracking."""
    st.markdown('<div class="chart-title">ML Experiment Tracking (MLflow)</div>', unsafe_allow_html=True)

    with get_db_session() as session:
        experiments = session.query(ModelExperiment).order_by(ModelExperiment.started_at.desc()).limit(20).all()

        if not experiments:
            st.info("No experiments recorded yet.")
            return

        exp_data = []
        for e in experiments:
            model = session.query(MLModel).filter(MLModel.id == e.model_id).first()
            metrics = json.loads(e.metrics) if e.metrics else {}
            exp_data.append({
                "Experiment": e.experiment_name,
                "Model": f"{model.model_name} v{model.model_version}" if model else "Unknown",
                "Status": e.status.title(),
                "Run ID": e.run_id[:8] + "..." if e.run_id else "N/A",
                "MAE": f"{metrics.get('mae', 'N/A')}",
                "RMSE": f"{metrics.get('rmse', 'N/A')}",
                "R²": f"{metrics.get('r2', 'N/A')}",
                "Started": e.started_at.strftime("%Y-%m-%d %H:%M"),
                "Completed": e.completed_at.strftime("%Y-%m-%d %H:%M") if e.completed_at else "Running",
            })

        df = pd.DataFrame(exp_data)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # MLflow link
        st.markdown("---")
        st.markdown("""
        **MLflow UI**: To view detailed experiment tracking, run:
        ```bash
        mlflow ui --backend-store-uri file:///app/mlruns --port 5000
        ```
        Then access http://localhost:5000
        """)


# Import needed for health status
from app.models.database import HealthStatus