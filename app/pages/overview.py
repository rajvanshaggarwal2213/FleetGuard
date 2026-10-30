"""
Overview Dashboard Page for FleetGuard AI.
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import List, Dict, Any

from sqlalchemy.orm import Session
from sqlalchemy import func

from app.models.database import (
    Vehicle, VehicleType, HealthStatus, SensorReading,
    MaintenanceAlert, WorkOrder, WorkOrderStatus, AlertSeverity,
    VehicleHealthHistory, SimulationEvent
)
from app.models.connection import get_db_session
from app.simulation.engine import get_simulation_engine, SimulationEventType


def render():
    """Render the overview dashboard."""
    st.markdown('<div class="chart-title">📊 Fleet Overview Dashboard</div>', unsafe_allow_html=True)

    # Get database session
    with get_db_session() as session:
        # Fetch fleet statistics
        stats = _get_fleet_stats(session)

        # Render KPI cards
        _render_kpi_cards(stats)

        st.markdown("---")

        # Main charts row
        col1, col2 = st.columns(2)

        with col1:
            _render_health_distribution(session, stats)
            _render_fleet_health_trend(session)

        with col2:
            _render_maintenance_events(session)
            _render_sensor_anomaly_trend(session)

        st.markdown("---")

        # Bottom row
        col3, col4 = st.columns(2)

        with col3:
            _render_maintenance_workload(session)

        with col4:
            _render_recent_alerts(session)

        # Live data notice
        _render_live_indicator()


def _get_fleet_stats(session: Session) -> Dict[str, Any]:
    """Calculate fleet statistics."""
    vehicles = session.query(Vehicle).filter(Vehicle.is_active == True).all()

    total = len(vehicles)
    healthy = sum(1 for v in vehicles if v.current_health_status == HealthStatus.HEALTHY)
    warning = sum(1 for v in vehicles if v.current_health_status == HealthStatus.MAINTENANCE_APPROACHING)
    critical = sum(1 for v in vehicles if v.current_health_status == HealthStatus.CRITICAL)

    open_wos = session.query(WorkOrder).filter(
        WorkOrder.status.in_([WorkOrderStatus.OPEN, WorkOrderStatus.ASSIGNED, WorkOrderStatus.IN_PROGRESS])
    ).count()

    completed_wos = session.query(WorkOrder).filter(
        WorkOrder.status == WorkOrderStatus.COMPLETED
    ).count()

    avg_health = sum(v.health_score for v in vehicles) / total if total > 0 else 0

    # Recent alerts (last 24h)
    cutoff = datetime.utcnow() - timedelta(hours=24)
    recent_alerts = session.query(MaintenanceAlert).filter(
        MaintenanceAlert.timestamp >= cutoff
    ).order_by(MaintenanceAlert.timestamp.desc()).limit(5).all()

    return {
        "total_vehicles": total,
        "healthy": healthy,
        "maintenance_approaching": warning,
        "critical": critical,
        "open_work_orders": open_wos,
        "completed_work_orders": completed_wos,
        "avg_health_score": round(avg_health, 1),
        "recent_alerts": recent_alerts,
        "vehicles": vehicles,
    }


def _render_kpi_cards(stats: Dict[str, Any]):
    """Render KPI metric cards."""
    cols = st.columns(5)

    metrics = [
        ("Total Vehicles", stats["total_vehicles"], "", ""),
        ("✅ Healthy", stats["healthy"], f"+{stats['healthy']}", "positive"),
        ("⚠️ Maintenance Due", stats["maintenance_approaching"], f"+{stats['maintenance_approaching']}", "negative" if stats['maintenance_approaching'] > 0 else "positive"),
        ("🔴 Critical", stats["critical"], f"+{stats['critical']}", "negative" if stats['critical'] > 0 else "positive"),
        ("📋 Open Work Orders", stats["open_work_orders"], "", ""),
    ]

    for i, (label, value, delta, delta_class) in enumerate(metrics):
        with cols[i]:
            delta_html = f'<div class="metric-delta {delta_class}">{delta}</div>' if delta else ""
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
                {delta_html}
            </div>
            """, unsafe_allow_html=True)

    # Second row for additional metrics
    cols2 = st.columns(3)
    with cols2[0]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Completed Work Orders</div>
            <div class="metric-value">{stats["completed_work_orders"]}</div>
        </div>
        """, unsafe_allow_html=True)
    with cols2[1]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Fleet Health Average</div>
            <div class="metric-value">{stats["avg_health_score"]}%</div>
        </div>
        """, unsafe_allow_html=True)
    with cols2[2]:
        sim_engine = get_simulation_engine()
        status = "🟢 Running" if st.session_state.simulation_running else "🔴 Paused"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Simulation Status</div>
            <div class="metric-value" style="font-size: 1.25rem;">{status}</div>
        </div>
        """, unsafe_allow_html=True)


def _render_health_distribution(session: Session, stats: Dict[str, Any]):
    """Render vehicle health distribution pie chart."""
    st.markdown("""
    <div class="chart-container">
        <div class="chart-title">Vehicle Health Distribution</div>
    </div>
    """, unsafe_allow_html=True)

    labels = ["Healthy", "Maintenance Approaching", "Critical"]
    values = [stats["healthy"], stats["maintenance_approaching"], stats["critical"]]
    colors = ["#28a745", "#ffc107", "#dc3545"]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.5,
        marker_colors=colors,
        textinfo="label+percent",
        textfont_size=12,
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>",
    )])

    fig.update_layout(
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        margin=dict(t=0, b=0, l=0, r=0),
        height=300,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#1e3a5f"),
    )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def _render_fleet_health_trend(session: Session):
    """Render fleet health trend over time."""
    st.markdown("""
    <div class="chart-container">
        <div class="chart-title">Fleet Health Trend (Last 7 Days)</div>
    </div>
    """, unsafe_allow_html=True)

    cutoff = datetime.utcnow() - timedelta(days=7)
    history = session.query(VehicleHealthHistory).filter(
        VehicleHealthHistory.timestamp >= cutoff
    ).order_by(VehicleHealthHistory.timestamp).all()

    if history:
        df = pd.DataFrame([{
            "timestamp": h.timestamp,
            "health_score": h.health_score,
            "vehicle_id": h.vehicle_id
        } for h in history])

        # Average per day
        df["date"] = df["timestamp"].dt.date
        daily_avg = df.groupby("date")["health_score"].mean().reset_index()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=daily_avg["date"],
            y=daily_avg["health_score"],
            mode="lines+markers",
            name="Avg Health",
            line=dict(color="#00a8e8", width=3),
            marker=dict(size=8),
            fill="tozeroy",
            fillcolor="rgba(0,168,232,0.1)",
        ))

        # Add threshold lines
        fig.add_hline(y=80, line_dash="dash", line_color="#28a745", annotation_text="Healthy Threshold")
        fig.add_hline(y=50, line_dash="dash", line_color="#ffc107", annotation_text="Warning Threshold")

        fig.update_layout(
            margin=dict(t=10, b=10, l=10, r=10),
            height=300,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#1e3a5f"),
            xaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.05)"),
            yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.05)", range=[0, 105]),
            hovermode="x unified",
        )

        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("No health history data available yet. Start simulation to generate data.")


def _render_maintenance_events(session: Session):
    """Render maintenance events timeline."""
    st.markdown("""
    <div class="chart-container">
        <div class="chart-title">Maintenance Events Timeline</div>
    </div>
    """, unsafe_allow_html=True)

    cutoff = datetime.utcnow() - timedelta(days=30)
    events = session.query(SimulationEvent).filter(
        SimulationEvent.timestamp >= cutoff
    ).order_by(SimulationEvent.timestamp).all()

    if events:
        df = pd.DataFrame([{
            "timestamp": e.timestamp,
            "event_type": e.event_type,
            "vehicle_id": e.vehicle_id,
        } for e in events])

        # Map event types to colors
        color_map = {
            "normal": "#28a745",
            "overheating": "#dc3545",
            "abnormal_vibration": "#fd7e14",
            "low_oil_pressure": "#dc3545",
            "battery_deterioration": "#6f42c1",
            "gradual_degradation": "#ffc107",
        }

        fig = go.Figure()

        for event_type in df["event_type"].unique():
            subset = df[df["event_type"] == event_type]
            fig.add_trace(go.Scatter(
                x=subset["timestamp"],
                y=[event_type] * len(subset),
                mode="markers",
                name=event_type.replace("_", " ").title(),
                marker=dict(
                    size=10,
                    color=color_map.get(event_type, "#6c757d"),
                    symbol="circle",
                ),
                hovertemplate="<b>%{y}</b><br>Time: %{x}<extra></extra>",
            ))

        fig.update_layout(
            margin=dict(t=10, b=10, l=10, r=10),
            height=300,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#1e3a5f"),
            xaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.05)"),
            yaxis=dict(showgrid=False),
            hovermode="closest",
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
        )

        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("No simulation events recorded yet. Use Demo Control Panel to inject faults.")


def _render_sensor_anomaly_trend(session: Session):
    """Render sensor anomaly trend."""
    st.markdown("""
    <div class="chart-container">
        <div class="chart-title">Sensor Anomaly Trend (Last 24h)</div>
    </div>
    """, unsafe_allow_html=True)

    cutoff = datetime.utcnow() - timedelta(hours=24)
    from app.models.database import Anomaly
    anomalies = session.query(Anomaly).filter(
        Anomaly.timestamp >= cutoff
    ).order_by(Anomaly.timestamp).all()

    if anomalies:
        df = pd.DataFrame([{
            "timestamp": a.timestamp,
            "sensor_name": a.sensor_name,
            "severity": a.severity.value,
            "anomaly_type": a.anomaly_type,
        } for a in anomalies])

        # Count by hour
        df["hour"] = df["timestamp"].dt.floor("H")
        hourly = df.groupby(["hour", "severity"]).size().reset_index(name="count")

        severity_colors = {
            "critical": "#dc3545",
            "high": "#fd7e14",
            "medium": "#ffc107",
            "low": "#00a8e8",
        }

        fig = go.Figure()

        for severity in ["critical", "high", "medium", "low"]:
            subset = hourly[hourly["severity"] == severity]
            if not subset.empty:
                fig.add_trace(go.Bar(
                    x=subset["hour"],
                    y=subset["count"],
                    name=severity.title(),
                    marker_color=severity_colors.get(severity, "#6c757d"),
                ))

        fig.update_layout(
            barmode="stack",
            margin=dict(t=10, b=10, l=10, r=10),
            height=300,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#1e3a5f"),
            xaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.05)"),
            yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.05)"),
            hovermode="x unified",
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
        )

        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("No anomalies detected in the last 24 hours.")


def _render_maintenance_workload(session: Session):
    """Render maintenance workload by vehicle."""
    st.markdown("""
    <div class="chart-container">
        <div class="chart-title">Maintenance Workload by Vehicle</div>
    </div>
    """, unsafe_allow_html=True)

    # Count work orders per vehicle
    wos = session.query(
        Vehicle.vehicle_id,
        Vehicle.vehicle_type,
        func.count(WorkOrder.id).label("wo_count")
    ).join(WorkOrder, Vehicle.id == WorkOrder.vehicle_id).filter(
        Vehicle.is_active == True
    ).group_by(Vehicle.id, Vehicle.vehicle_id, Vehicle.vehicle_type).all()

    if wos:
        df = pd.DataFrame(wos, columns=["vehicle_id", "vehicle_type", "wo_count"])

        fig = go.Figure(data=[go.Bar(
            x=df["vehicle_id"],
            y=df["wo_count"],
            marker_color="#00a8e8",
            text=df["wo_count"],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Work Orders: %{y}<extra></extra>",
        )]

        fig.update_layout(
            margin=dict(t=10, b=10, l=10, r=10),
            height=300,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#1e3a5f"),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.05)"),
        )

        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("No work orders created yet. Trigger faults to generate maintenance events.")


def _render_recent_alerts(session: Session):
    """Render recent alerts panel."""
    st.markdown("""
    <div class="chart-container">
        <div class="chart-title">Recent Alerts</div>
    </div>
    """, unsafe_allow_html=True)

    cutoff = datetime.utcnow() - timedelta(hours=24)
    alerts = session.query(MaintenanceAlert).filter(
        MaintenanceAlert.timestamp >= cutoff
    ).order_by(MaintenanceAlert.timestamp.desc()).limit(10).all()

    if alerts:
        for alert in alerts:
            vehicle = session.query(Vehicle).filter(Vehicle.id == alert.vehicle_id).first()
            vehicle_id = vehicle.vehicle_id if vehicle else "Unknown"

            severity_class = alert.severity.value
            st.markdown(f"""
            <div class="alert-item {severity_class}">
                <div class="alert-header">
                    <div class="alert-title">{alert.title}</div>
                    <span class="status-badge status-{severity_class}">{severity_class.upper()}</span>
                </div>
                <div class="alert-meta">
                    <strong>Vehicle:</strong> {vehicle_id} |
                    <strong>Component:</strong> {alert.component} |
                    <strong>Time:</strong> {alert.timestamp.strftime('%H:%M:%S')}
                </div>
                <div style="margin-top: 0.5rem; font-size: 0.85rem;">{alert.description}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No recent alerts in the last 24 hours.")


def _render_live_indicator():
    """Render live data indicator."""
    if st.session_state.simulation_running:
        st.markdown("""
        <div style="text-align: center; padding: 1rem; background: rgba(40,167,69,0.1); border-radius: 8px; margin-top: 1rem;">
            <strong style="color: #28a745;">🔴 LIVE DATA STREAMING</strong><br>
            <small style="color: #6c757d;">Dashboard auto-refreshes every 3 seconds during simulation</small>
        </div>
        """, unsafe_allow_html=True)
        # Auto-refresh
        st.empty()
        import time
        time.sleep(0.1)
        st.rerun()
    else:
        st.markdown("""
        <div style="text-align: center; padding: 1rem; background: rgba(108,117,125,0.1); border-radius: 8px; margin-top: 1rem;">
            <strong style="color: #6c757d;">⚫ SIMULATION PAUSED</strong><br>
            <small style="color: #6c757d;">Click "Start" in Demo Control Panel to begin live data streaming</small>
        </div>
        """, unsafe_allow_html=True)