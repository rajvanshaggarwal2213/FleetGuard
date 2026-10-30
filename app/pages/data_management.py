"""
Data & Model Management Page for FleetGuard AI.
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import json
import os

from sqlalchemy.orm import Session
from sqlalchemy import func

from app.models.database import (
    DatasetUpload, MLModel, ModelExperiment, Vehicle, SensorReading
)
from app.models.connection import get_db_session
from app.ml.rul_prediction import CMapSSDataProcessor


def render():
    """Render the data & model management page."""
    st.markdown('<div class="chart-title">📁 Data & Model Management</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "📤 NASA Dataset Upload", "🗄️ Dataset Registry", "🤖 Model Registry", "⚙️ Simulation Settings"
    ])

    with tab1:
        _render_dataset_upload_tab()

    with tab2:
        _render_dataset_registry_tab()

    with tab3:
        _render_model_registry_tab()

    with tab4:
        _render_simulation_settings_tab()


def _render_dataset_upload_tab():
    """Render NASA dataset upload interface."""
    st.markdown('<div class="chart-title">NASA C-MAPSS Dataset Upload</div>', unsafe_allow_html=True)

    st.info("""
    **Supported Dataset**: NASA C-MAPSS Turbofan Engine Degradation Dataset (FD001-FD004)

    Required files for training:
    - `train_FD001.txt` - Training data (run-to-failure)
    - `test_FD001.txt` - Test data (truncated before failure)
    - `RUL_FD001.txt` - True RUL values for test data

    Download from: https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Upload Training Data")
        train_file = st.file_uploader(
            "Train File (train_FD00*.txt)",
            type=["txt"],
            key="upload_train",
            help="Run-to-failure training data with 26 columns"
        )

        st.markdown("### Upload Test Data")
        test_file = st.file_uploader(
            "Test File (test_FD00*.txt)",
            type=["txt"],
            key="upload_test",
            help="Test data truncated before failure"
        )

    with col2:
        st.markdown("### Upload RUL Labels")
        rul_file = st.file_uploader(
            "RUL File (RUL_FD00*.txt)",
            type=["txt"],
            key="upload_rul",
            help="True RUL values for test engines"
        )

        st.markdown("### Dataset Type")
        dataset_type = st.selectbox(
            "Dataset Variant",
            ["FD001 (100 engines, 1 operating condition)", "FD002 (260 engines, 6 operating conditions)",
             "FD003 (100 engines, 1 operating condition)", "FD004 (248 engines, 6 operating conditions)"],
            index=0
        )

    # Validation preview
    if train_file:
        st.markdown("### Data Preview & Validation")
        _preview_nasa_data(train_file, dataset_type)

    # Train button
    if train_file and test_file and rul_file:
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Train RUL Model", type="primary", use_container_width=True):
                _train_model_from_upload(train_file, test_file, rul_file, dataset_type)


def _preview_nasa_data(train_file, dataset_type: str):
    """Preview and validate NASA dataset."""
    try:
        content = train_file.getvalue().decode('utf-8')
        processor = CMapSSDataProcessor()
        df = pd.read_csv(
            pd.io.common.StringIO(content),
            sep=r"\s+", header=None, names=processor.COLUMNS
        )

        st.success(f"✅ File loaded: {len(df):,} rows, {len(df.columns)} columns")

        # Basic stats
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Rows", f"{len(df):,}")
        col1.metric("Columns", len(df.columns))
        col2.metric("Unique Engines", df["unit_number"].nunique())
        col2.metric("Max Cycles", df["cycle"].max())
        col3.metric("Settings", 3)
        col3.metric("Sensors", 21)
        col4.metric("File Size", f"{len(content)/1024:.1f} KB")

        # Data quality checks
        st.markdown("#### Data Quality")
        checks = {
            "No missing values": df.isnull().sum().sum() == 0,
            "Correct column count": len(df.columns) == 26,
            "Unit numbers sequential": df["unit_number"].min() == 1,
            "Cycles start at 1": (df.groupby("unit_number")["cycle"].min() == 1).all(),
            "Monotonic cycles": (df.groupby("unit_number")["cycle"].apply(lambda x: x.is_monotonic_increasing)).all(),
        }

        for check, passed in checks.items():
            icon = "✅" if passed else "❌"
            st.write(f"{icon} {check}")

        # Sample data
        st.markdown("#### Sample Data (First 10 rows)")
        st.dataframe(df.head(10), use_container_width=True)

        # Sensor statistics
        st.markdown("#### Sensor Statistics")
        sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
        sensor_stats = df[sensor_cols].describe().T
        st.dataframe(sensor_stats.style.format("{:.3f}"), use_container_width=True)

    except Exception as e:
        st.error(f"Error reading file: {str(e)}")


def _train_model_from_upload(train_file, test_file, rul_file, dataset_type: str):
    """Train model from uploaded files."""
    import tempfile
    from app.ml.rul_prediction import RULPredictor

    with st.spinner("Training model... This may take a minute."):
        try:
            # Save temp files
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(train_file.getvalue().decode('utf-8'))
                train_path = f.name

            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(test_file.getvalue().decode('utf-8'))
                test_path = f.name

            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(rul_file.getvalue().decode('utf-8'))
                rul_path = f.name

            # Train
            with get_db_session() as session:
                predictor = RULPredictor(model_type="random_forest")
                metrics = predictor.train(train_path, test_path, rul_path, session=session)

                # Save upload record
                upload = DatasetUpload(
                    filename=train_file.name,
                    dataset_type=dataset_type.split(" ")[0].lower(),
                    file_size_bytes=len(train_file.getvalue()),
                    row_count=metrics.training_samples + metrics.validation_samples,
                    column_count=26,
                    is_valid=True,
                    processed_at=datetime.utcnow(),
                    uploaded_by="user",
                )
                session.add(upload)
                session.commit()

            st.success(f"✅ Model trained successfully!")
            st.balloons()
            st.rerun()

        except Exception as e:
            st.error(f"Training failed: {str(e)}")
        finally:
            for p in [train_path, test_path, rul_path]:
                if p and os.path.exists(p):
                    os.unlink(p)


def _render_dataset_registry_tab():
    """Render dataset registry."""
    st.markdown('<div class="chart-title">Dataset Registry</div>', unsafe_allow_html=True)

    with get_db_session() as session:
        uploads = session.query(DatasetUpload).order_by(DatasetUpload.uploaded_at.desc()).all()

        if not uploads:
            st.info("No datasets uploaded yet.")
            return

        for upload in uploads:
            with st.expander(f"{upload.filename} • {upload.dataset_type.upper()} • {upload.uploaded_at.strftime('%Y-%m-%d %H:%M')}", expanded=False):
                col1, col2, col3 = st.columns(3)
                col1.metric("Rows", f"{upload.row_count:,}" if upload.row_count else "—")
                col2.metric("Columns", upload.column_count or "—")
                col3.metric("Size", f"{upload.file_size_bytes/1024:.1f} KB")

                st.write(f"**Valid:** {'✅ Yes' if upload.is_valid else '❌ No'}")
                if upload.validation_errors:
                    st.error(f"Errors: {upload.validation_errors}")
                if upload.validation_warnings:
                    st.warning(f"Warnings: {upload.validation_warnings}")

                st.write(f"**Processed:** {upload.processed_at.strftime('%Y-%m-%d %H:%M') if upload.processed_at else 'Not processed'}")
                st.write(f"**Duration:** {upload.processing_duration_seconds:.1f}s" if upload.processing_duration_seconds else "**Duration:** —")


def _render_model_registry_tab():
    """Render model registry."""
    st.markdown('<div class="chart-title">Model Registry</div>', unsafe_allow_html=True)

    with get_db_session() as session:
        models = session.query(MLModel).order_by(MLModel.created_at.desc()).all()

        if not models:
            st.info("No models registered yet.")
            return

        # Model cards
        for model in models:
            config = json.loads(model.training_config) if model.training_config else {}
            exp_count = session.query(ModelExperiment).filter(ModelExperiment.model_id == model.id).count()

            status_badge = "✅ Active" if model.is_active else "📦 Archived"

            with st.expander(f"{model.model_name} v{model.model_version} • {model.model_type} • {status_badge}", expanded=False):
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("MAE", f"{model.mae:.2f}" if model.mae else "—")
                col2.metric("RMSE", f"{model.rmse:.2f}" if model.rmse else "—")
                col3.metric("R²", f"{model.r2_score:.3f}" if model.r2_score else "—")
                col4.metric("Experiments", exp_count)

                st.write(f"**Dataset:** {model.dataset_name or 'N/A'}")
                st.write(f"**Type:** {model.model_type}")
                st.write(f"**Trained:** {model.created_at.strftime('%Y-%m-%d %H:%M')}")
                st.write(f"**By:** {model.trained_by or 'Unknown'}")

                if config:
                    st.write(f"**Baseline MAE:** {config.get('baseline_mae', 'N/A')}")

                # Feature importance
                if model.feature_names:
                    features = json.loads(model.feature_names)
                    st.write(f"**Features:** {len(features)} total")
                    if st.checkbox(f"Show features", key=f"feat_{model.id}"):
                        st.json(features[:50])

                # Actions
                col_a, col_b = st.columns(2)
                with col_a:
                    if not model.is_active:
                        if st.button("✅ Activate", key=f"activate_{model.id}"):
                            # Deactivate others
                            session.query(MLModel).filter(MLModel.model_type == model.model_type).update({MLModel.is_active: False})
                            model.is_active = True
                            session.commit()
                            st.success("Model activated!")
                            st.rerun()
                    else:
                        st.success("Currently Active")

                with col_b:
                    if st.button("🗑️ Archive", key=f"archive_{model.id}", type="secondary"):
                        model.is_active = False
                        session.commit()
                        st.success("Model archived!")
                        st.rerun()


def _render_simulation_settings_tab():
    """Render simulation settings."""
    st.markdown('<div class="chart-title">Simulation Settings</div>', unsafe_allow_html=True)

    with get_db_session() as session:
        vehicles = session.query(Vehicle).filter(Vehicle.is_active == True).all()

        st.markdown("### Global Simulation Controls")

        col1, col2 = st.columns(2)
        with col1:
            st.session_state.simulation_speed = st.slider(
                "Simulation Speed",
                0.1, 5.0, st.session_state.simulation_speed, 0.1,
                help="Speed multiplier for simulation (1.0 = real-time)"
            )

        with col2:
            update_interval = st.slider(
                "Update Interval (seconds)",
                0.5, 10.0, 2.0, 0.5,
                help="How often to generate new sensor readings"
            )

        st.markdown("---")

        st.markdown("### Vehicle Fault Injection")

        for vehicle in vehicles:
            with st.expander(f"{vehicle.vehicle_id} ({vehicle.vehicle_type.value.title()})", expanded=False):
                col1, col2, col3 = st.columns(3)

                with col1:
                    if st.button("🌡️ Overheating", key=f"sim_overheat_{vehicle.id}", use_container_width=True):
                        _inject_fault(session, vehicle.id, "overheating")
                        st.rerun()

                    if st.button("📳 Vibration", key=f"sim_vib_{vehicle.id}", use_container_width=True):
                        _inject_fault(session, vehicle.id, "abnormal_vibration")
                        st.rerun()

                with col2:
                    if st.button("🛢️ Low Oil Pressure", key=f"sim_oil_{vehicle.id}", use_container_width=True):
                        _inject_fault(session, vehicle.id, "low_oil_pressure")
                        st.rerun()

                    if st.button("🔋 Battery Drain", key=f"sim_bat_{vehicle.id}", use_container_width=True):
                        _inject_fault(session, vehicle.id, "battery_deterioration")
                        st.rerun()

                with col3:
                    if st.button("📉 Gradual Degradation", key=f"sim_deg_{vehicle.id}", use_container_width=True):
                        _inject_fault(session, vehicle.id, "gradual_degradation")
                        st.rerun()

                    if st.button("✅ Clear All Faults", key=f"sim_clear_{vehicle.id}", use_container_width=True):
                        _clear_faults(session, vehicle.id)
                        st.rerun()

        st.markdown("---")

        st.markdown("### Simulation Data Management")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("🗑️ Clear All Sensor Data", use_container_width=True, type="secondary"):
                if st.checkbox("I confirm - this will delete ALL sensor readings"):
                    session.query(SensorReading).delete()
                    session.query(VehicleHealthHistory).delete()
                    session.query(Anomaly).delete()
                    session.query(SimulationEvent).delete()
                    session.commit()
                    st.success("All sensor data cleared!")
                    st.rerun()

        with col2:
            if st.button("🔄 Reset All Vehicles to Healthy", use_container_width=True):
                for v in vehicles:
                    v.health_score = 100.0
                    v.current_health_status = HealthStatus.HEALTHY
                    v.updated_at = datetime.utcnow()
                session.commit()
                st.success("All vehicles reset to healthy!")
                st.rerun()

        with col3:
            # Export data
            if st.button("📥 Export Sensor Data (CSV)", use_container_width=True):
                readings = session.query(SensorReading).limit(10000).all()
                if readings:
                    df = pd.DataFrame([{
                        "vehicle_id": r.vehicle_id,
                        "timestamp": r.timestamp,
                        "engine_temp": r.engine_temperature_c,
                        "rpm": r.engine_rpm,
                        "oil_pressure": r.oil_pressure_kpa,
                        "coolant_temp": r.coolant_temperature_c,
                        "vibration": r.vibration_mm_s,
                        "battery_voltage": r.battery_voltage_v,
                        "fuel_consumption": r.fuel_consumption_l_per_100km,
                        "operating_hours": r.operating_hours,
                        "distance_km": r.distance_km,
                    } for r in readings])
                    csv = df.to_csv(index=False)
                    st.download_button(
                        "Download CSV",
                        csv,
                        f"sensor_data_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv",
                        "text/csv",
                        use_container_width=True
                    )


def _inject_fault(session: Session, vehicle_id: int, fault_type: str):
    """Inject a fault into a vehicle."""
    from app.simulation.engine import get_simulation_engine, SimulationEventType

    engine = get_simulation_engine()
    event_type_map = {
        "overheating": SimulationEventType.OVERHEATING,
        "abnormal_vibration": SimulationEventType.ABNORMAL_VIBRATION,
        "low_oil_pressure": SimulationEventType.LOW_OIL_PRESSURE,
        "battery_deterioration": SimulationEventType.BATTERY_DETERIORATION,
        "gradual_degradation": SimulationEventType.GRADUAL_DEGRADATION,
    }

    if fault_type in event_type_map:
        engine.trigger_fault(vehicle_id, event_type_map[fault_type], session)
        st.success(f"Injected {fault_type} fault!")


def _clear_faults(session: Session, vehicle_id: int):
    """Clear all faults for a vehicle."""
    from app.simulation.engine import get_simulation_engine, SimulationEventType

    engine = get_simulation_engine()
    for fault_type in SimulationEventType:
        if fault_type != SimulationEventType.NORMAL:
            engine.clear_fault(vehicle_id, fault_type, session)
    st.success("All faults cleared!")