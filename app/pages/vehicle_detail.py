"""
Vehicle Detail Page for FleetGuard AI.
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

from sqlalchemy.orm import Session
from sqlalchemy import func

from app.models.database import (
    Vehicle, VehicleType, HealthStatus, SensorReading,
    VehicleHealthHistory, Anomaly, MaintenanceAlert, WorkOrder,
    MaintenanceHistory, AlertSeverity, WorkOrderStatus
)
from app.models.connection import get_db_session


def render():
    """Render the vehicle detail page."""
    vehicle_id = st.session_state.get("selected_vehicle")

    if not vehicle_id:
        st.warning("No vehicle selected. Go to Fleet Management and click on a vehicle.")
        if st.button("← Back to Fleet"):
            st.session_state.current_page = "fleet"
            st.rerun()
        return

    with get_db_session() as session:
        vehicle = session.query(Vehicle).filter(Vehicle.id == vehicle_id).first()

        if not vehicle:
            st.error("Vehicle not found.")
            if st.button("← Back to Fleet"):
                st.session_state.current_page = "fleet"
                st.rerun()
            return

        # Header
        _render_vehicle_header(vehicle)

        # Tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Overview", "📈 Sensor Trends", "⚠️ Anomalies", "🔧 Maintenance", "📋 History"
        ])

        with tab1:
            _render_overview_tab(vehicle, session)

        with tab2:
            _render_sensor_trends_tab(vehicle, session)

        with tab3:
            _render_anomalies_tab(vehicle, session)

        with tab4:
            _render_maintenance_tab(vehicle, session)

        with tab5:
            _render_history_tab(vehicle, session)


def _render_vehicle_header(vehicle: Vehicle):
    """Render vehicle header with key info."""
    status_class = {
        HealthStatus.HEALTHY: "status-healthy",
        HealthStatus.MAINTENANCE_APPROACHING: "status-warning",
        HealthStatus.CRITICAL: "status-critical",
    }[vehicle.current_health_status]

    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

    with col1:
        st.markdown(f"""
        <div style="margin-bottom: 1rem;">
            <h2 style="margin: 0;">{vehicle.vehicle_id}</h2>
            <div style="color: #6c757d;">{vehicle.manufacturer} {vehicle.model} • {vehicle.vehicle_type.value.title()}</div>
            <div style="color: #6c757d; font-size: 0.85rem;">Registration: {vehicle.registration_number}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card" style="text-align: center;">
            <div class="metric-label">Health Score</div>
            <div class="metric-value" style="font-size: 2.5rem;">{vehicle.health_score:.1f}%</div>
            <span class="status-badge {status_class}">{vehicle.current_health_status.value.replace('_', ' ').title()}</span>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card" style="text-align: center;">
            <div class="metric-label">Mileage</div>
            <div class="metric-value">{vehicle.total_distance_km:,.0f} km</div>
            <div class="metric-label">Engine Hours: {vehicle.engine_hours:,.1f}</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        next_service = "—"
        if vehicle.next_scheduled_service:
            days = (vehicle.next_scheduled_service - datetime.utcnow()).days
            if days > 0:
                next_service = f"{days} days"
            elif days == 0:
                next_service = "Today"
            else:
                next_service = f"{abs(days)} days overdue"

        st.markdown(f"""
        <div class="metric-card" style="text-align: center;">
            <div class="metric-label">Next Maintenance</div>
            <div class="metric-value" style="font-size: 1.5rem;">{next_service}</div>
            <div class="metric-label">Last: {vehicle.last_service_date.strftime('%Y-%m-%d') if vehicle.last_service_date else 'Never'}</div>
        </div>
        """, unsafe_allow_html=True)


def _render_overview_tab(vehicle: Vehicle, session: Session):
    """Render overview tab with current sensor readings."""
    st.markdown('<div class="chart-title">Current Sensor Readings</div>', unsafe_allow_html=True)

    latest = session.query(SensorReading).filter(
        SensorReading.vehicle_id == vehicle.id
    ).order_by(SensorReading.timestamp.desc()).first()

    if not latest:
        st.info("No sensor data available for this vehicle.")
        return

    # Sensor cards
    sensors = [
        ("🌡️ Engine Temp", f"{latest.engine_temperature_c:.1f} °C", "°C", (75, 100)),
        ("🔄 RPM", f"{latest.engine_rpm:.0f}", "RPM", (1000, 2500)),
        ("🛢️ Oil Pressure", f"{latest.oil_pressure_kpa:.0f} kPa", "kPa", (300, 500)),
        ("❄️ Coolant Temp", f"{latest.coolant_temperature_c:.1f} °C", "°C", (70, 95)),
        ("📳 Vibration", f"{latest.vibration_mm_s:.2f} mm/s", "mm/s", (0, 4)),
        ("🔋 Battery", f"{latest.battery_voltage_v:.2f} V", "V", (12.5, 14.5)),
        ("⛽ Fuel Consumption", f"{latest.fuel_consumption_l_per_100km:.1f} L/100km", "L/100km", (20, 40)),
        ("⏱️ Operating Hours", f"{latest.operating_hours:.1f} hrs", "hrs", (0, 10000)),
    ]

    cols = st.columns(4)
    for i, (label, value, unit, healthy_range) in enumerate(sensors):
        with cols[i % 4]:
            min_h, max_h = healthy_range
            # Determine if in healthy range
            try:
                num_val = float(value.split()[0])
                if min_h <= num_val <= max_h:
                    status_color = "#28a745"
                    status_text = "Normal"
                else:
                    status_color = "#dc3545"
                    status_text = "Alert"
            except:
                status_color = "#6c757d"
                status_text = "—"

            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value" style="font-size: 1.5rem;">{value}</div>
                <div style="font-size: 0.75rem; color: {status_color}; font-weight: 600;">{status_text} (Healthy: {min_h}-{max_h} {unit})</div>
            </div>
            """, unsafe_allow_html=True)

    # Health factors
    st.markdown('<div class="chart-title">Health Score Breakdown</div>', unsafe_allow_html=True)

    latest_history = session.query(VehicleHealthHistory).filter(
        VehicleHealthHistory.vehicle_id == vehicle.id
    ).order_by(VehicleHealthHistory.timestamp.desc()).first()

    if latest_history and latest_history.contributing_factors:
        import json
        factors = json.loads(latest_history.contributing_factors)

        factor_data = []
        for f in factors:
            factor_data.append({
                "Factor": f["name"].replace("_", " ").title(),
                "Score": f"{f['score']:.0f}%",
                "Status": f["status"].title(),
                "Value": f"{f['value']:.1f}" if isinstance(f['value'], float) else str(f['value']),
            })

        df = pd.DataFrame(factor_data)

        def style_status(val):
            if val == "Normal":
                return "color: #28a745; font-weight: 600;"
            elif val == "Warning":
                return "color: #ffc107; font-weight: 600;"
            elif val == "Critical":
                return "color: #dc3545; font-weight: 600;"
            return ""

        st.dataframe(
            df.style.applymap(style_status, subset=["Status"]),
            use_container_width=True,
            hide_index=True,
        )


def _render_sensor_trends_tab(vehicle: Vehicle, session: Session):
    """Render sensor trends with interactive charts."""
    st.markdown('<div class="chart-title">Sensor Trends (Last 24 Hours)</div>', unsafe_allow_html=True)

    cutoff = datetime.utcnow() - timedelta(hours=24)
    readings = session.query(SensorReading).filter(
        SensorReading.vehicle_id == vehicle.id,
        SensorReading.timestamp >= cutoff
    ).order_by(SensorReading.timestamp).all()

    if not readings:
        st.info("No sensor history available for the last 24 hours.")
        return

    df = pd.DataFrame([{
        "timestamp": r.timestamp,
        "engine_temperature_c": r.engine_temperature_c,
        "engine_rpm": r.engine_rpm,
        "oil_pressure_kpa": r.oil_pressure_kpa,
        "coolant_temperature_c": r.coolant_temperature_c,
        "vibration_mm_s": r.vibration_mm_s,
        "battery_voltage_v": r.battery_voltage_v,
        "fuel_consumption_l_per_100km": r.fuel_consumption_l_per_100km,
    } for r in readings])

    # Multi-panel chart
    fig = make_subplots(
        rows=4, cols=2,
        subplot_titles=(
            "Engine Temperature (°C)", "Engine RPM",
            "Oil Pressure (kPa)", "Coolant Temperature (°C)",
            "Vibration (mm/s)", "Battery Voltage (V)",
            "Fuel Consumption (L/100km)", "Operating Hours"
        ),
        vertical_spacing=0.08,
        shared_xaxes=True,
    )

    sensors = [
        ("engine_temperature_c", "#dc3545", 1, 1, (75, 100)),
        ("engine_rpm", "#00a8e8", 1, 2, (1000, 2500)),
        ("oil_pressure_kpa", "#28a745", 2, 1, (300, 500)),
        ("coolant_temperature_c", "#fd7e14", 2, 2, (70, 95)),
        ("vibration_mm_s", "#6f42c1", 3, 1, (0, 4)),
        ("battery_voltage_v", "#ffc107", 3, 2, (12.5, 14.5)),
        ("fuel_consumption_l_per_100km", "#20c997", 4, 1, (20, 40)),
    ]

    for sensor, color, row, col, healthy_range in sensors:
        fig.add_trace(go.Scatter(
            x=df["timestamp"],
            y=df[sensor],
            mode="lines",
            name=sensor,
            line=dict(color=color, width=2),
            hovertemplate=f"<b>{sensor}</b><br>Time: %{{x}}<br>Value: %{{y}}<extra></extra>",
        ), row=row, col=col)

        # Add healthy range band
        fig.add_hrect(
            y0=healthy_range[0], y1=healthy_range[1],
            fillcolor="rgba(40,167,69,0.1)", line_width=0,
            row=row, col=col
        )

    fig.update_layout(
        height=800,
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#1e3a5f"),
        margin=dict(t=50, b=20, l=20, r=20),
    )

    fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.05)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.05)")

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True})


def _render_anomalies_tab(vehicle: Vehicle, session: Session):
    """Render anomalies for the vehicle."""
    st.markdown('<div class="chart-title">Detected Anomalies</div>', unsafe_allow_html=True)

    cutoff = datetime.utcnow() - timedelta(hours=48)
    anomalies = session.query(Anomaly).filter(
        Anomaly.vehicle_id == vehicle.id,
        Anomaly.timestamp >= cutoff
    ).order_by(Anomaly.timestamp.desc()).limit(50).all()

    if not anomalies:
        st.success("✅ No anomalies detected in the last 48 hours.")
        return

    # Anomaly summary
    anomaly_data = []
    for a in anomalies:
        anomaly_data.append({
            "Time": a.timestamp.strftime("%H:%M:%S"),
            "Sensor": a.sensor_name,
            "Value": f"{a.sensor_value:.2f}",
            "Type": a.anomaly_type.replace("_", " ").title(),
            "Severity": a.severity.value.title(),
            "Description": a.description[:80] + "..." if len(a.description) > 80 else a.description,
        })

    df = pd.DataFrame(anomaly_data)

    def style_severity(val):
        colors = {
            "Critical": "color: #dc3545; font-weight: 700;",
            "High": "color: #fd7e14; font-weight: 700;",
            "Medium": "color: #ffc107; font-weight: 700;",
            "Low": "color: #00a8e8; font-weight: 700;",
        }
        return colors.get(val, "")

    st.dataframe(
        df.style.applymap(style_severity, subset=["Severity"]),
        use_container_width=True,
        hide_index=True,
    )

    # Anomaly timeline
    st.markdown('<div class="chart-title">Anomaly Timeline</div>', unsafe_allow_html=True)

    timeline_df = pd.DataFrame([{
        "timestamp": a.timestamp,
        "sensor": a.sensor_name,
        "severity": a.severity.value,
        "type": a.anomaly_type,
    } for a in anomalies])

    severity_colors = {
        "critical": "#dc3545",
        "high": "#fd7e14",
        "medium": "#ffc107",
        "low": "#00a8e8",
    }

    fig = go.Figure()

    for severity in ["critical", "high", "medium", "low"]:
        subset = timeline_df[timeline_df["severity"] == severity]
        if not subset.empty:
            fig.add_trace(go.Scatter(
                x=subset["timestamp"],
                y=subset["sensor"],
                mode="markers",
                name=severity.title(),
                marker=dict(
                    size=12,
                    color=severity_colors[severity],
                    symbol="diamond",
                    line=dict(width=1, color="white"),
                ),
                hovertemplate="<b>%{y}</b><br>Time: %{x}<br>Severity: %{text}<extra></extra>",
                text=[severity.title()] * len(subset),
            ))

    fig.update_layout(
        height=300,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#1e3a5f"),
        margin=dict(t=10, b=10, l=10, r=10),
        xaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.05)"),
        yaxis=dict(showgrid=False),
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
    )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def _render_maintenance_tab(vehicle: Vehicle, session: Session):
    """Render maintenance alerts and work orders for the vehicle."""
    st.markdown('<div class="chart-title">Active Alerts</div>', unsafe_allow_html=True)

    alerts = session.query(MaintenanceAlert).filter(
        MaintenanceAlert.vehicle_id == vehicle.id,
        MaintenanceAlert.is_resolved == False
    ).order_by(MaintenanceAlert.timestamp.desc()).all()

    if not alerts:
        st.success("✅ No active alerts for this vehicle.")
    else:
        for alert in alerts:
            severity_class = alert.severity.value
            st.markdown(f"""
            <div class="alert-item {severity_class}">
                <div class="alert-header">
                    <div class="alert-title">{alert.title}</div>
                    <span class="status-badge status-{severity_class}">{severity_class.upper()}</span>
                </div>
                <div class="alert-meta">
                    <strong>Component:</strong> {alert.component} |
                    <strong>Time:</strong> {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
                </div>
                <div style="margin-top: 0.5rem;">{alert.description}</div>
                <div style="margin-top: 0.5rem; padding: 0.5rem; background: rgba(0,168,232,0.1); border-radius: 4px;">
                    <strong>Recommended Action:</strong> {alert.recommended_action}
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="chart-title">Work Orders</div>', unsafe_allow_html=True)

    work_orders = session.query(WorkOrder).filter(
        WorkOrder.vehicle_id == vehicle.id
    ).order_by(WorkOrder.creation_timestamp.desc()).all()

    if not work_orders:
        st.info("No work orders for this vehicle.")
    else:
        for wo in work_orders:
            status_class = {
                WorkOrderStatus.OPEN: "status-open",
                WorkOrderStatus.ASSIGNED: "status-assigned",
                WorkOrderStatus.IN_PROGRESS: "status-in-progress",
                WorkOrderStatus.COMPLETED: "status-completed",
                WorkOrderStatus.CANCELLED: "status-cancelled",
            }[wo.status]

            technician = session.query(Technician).filter(Technician.id == wo.technician_id).first() if wo.technician_id else None

            st.markdown(f"""
            <div class="wo-card">
                <div class="wo-header">
                    <span class="wo-id">{wo.work_order_id}</span>
                    <span class="status-badge {status_class}">{wo.status.value.replace('_', ' ').title()}</span>
                </div>
                <div style="margin-bottom: 0.5rem;"><strong>Component:</strong> {wo.component}</div>
                <div style="margin-bottom: 0.5rem;"><strong>Issue:</strong> {wo.issue_description}</div>
                <div style="margin-bottom: 0.5rem;"><strong>Priority:</strong> {wo.priority.value.title()}</div>
                <div style="margin-bottom: 0.5rem;"><strong>Recommended:</strong> {wo.recommended_inspection}</div>
                <div style="margin-bottom: 0.5rem;"><strong>Technician:</strong> {technician.name if technician else 'Unassigned'}</div>
                <div style="margin-bottom: 0.5rem;"><strong>Created:</strong> {wo.creation_timestamp.strftime('%Y-%m-%d %H:%M')}</div>
                {f'<div style="margin-bottom: 0.5rem;"><strong>Completed:</strong> {wo.completion_notes}</div>' if wo.completion_notes else ''}
            </div>
            """, unsafe_allow_html=True)


def _render_history_tab(vehicle: Vehicle, session: Session):
    """Render maintenance history."""
    st.markdown('<div class="chart-title">Maintenance History</div>', unsafe_allow_html=True)

    history = session.query(MaintenanceHistory).filter(
        MaintenanceHistory.vehicle_id == vehicle.id
    ).order_by(MaintenanceHistory.service_date.desc()).all()

    if not history:
        st.info("No maintenance history recorded yet.")
        return

    hist_data = []
    for h in history:
        hist_data.append({
            "Date": h.service_date.strftime("%Y-%m-%d"),
            "Type": h.service_type.title(),
            "Component": h.component,
            "Description": h.description[:100] + "..." if len(h.description) > 100 else h.description,
            "Mileage": f"{h.mileage_at_service:,.0f} km" if h.mileage_at_service else "—",
            "Hours": f"{h.engine_hours_at_service:,.1f}" if h.engine_hours_at_service else "—",
            "Technician": h.technician_name or "—",
            "Cost": f"${h.cost:,.2f}" if h.cost else "—",
        })

    df = pd.DataFrame(hist_data)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Health score history
    st.markdown('<div class="chart-title">Health Score History</div>', unsafe_allow_html=True)

    health_history = session.query(VehicleHealthHistory).filter(
        VehicleHealthHistory.vehicle_id == vehicle.id
    ).order_by(VehicleHealthHistory.timestamp).all()

    if health_history:
        df_health = pd.DataFrame([{
            "timestamp": h.timestamp,
            "health_score": h.health_score,
            "status": h.health_status.value,
        } for h in health_history])

        fig = go.Figure()

        # Color by status
        for status, color in [
            (HealthStatus.HEALTHY, "#28a745"),
            (HealthStatus.MAINTENANCE_APPROACHING, "#ffc107"),
            (HealthStatus.CRITICAL, "#dc3545"),
        ]:
            subset = df_health[df_health["status"] == status.value]
            if not subset.empty:
                fig.add_trace(go.Scatter(
                    x=subset["timestamp"],
                    y=subset["health_score"],
                    mode="lines+markers",
                    name=status.value.replace("_", " ").title(),
                    line=dict(color=color, width=2),
                    marker=dict(size=6),
                ))

        fig.add_hline(y=80, line_dash="dash", line_color="#28a745")
        fig.add_hline(y=50, line_dash="dash", line_color="#ffc107")

        fig.update_layout(
            height=350,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#1e3a5f"),
            margin=dict(t=10, b=10, l=10, r=10),
            xaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.05)"),
            yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.05)", range=[0, 105]),
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        )

        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# Import Technician for use in maintenance tab
from app.models.database import Technician