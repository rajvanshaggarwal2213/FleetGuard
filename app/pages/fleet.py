"""
Fleet Management Page for FleetGuard AI.
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

from sqlalchemy.orm import Session
from sqlalchemy import func

from app.models.database import (
    Vehicle, VehicleType, HealthStatus, SensorReading, utcnow
)
from app.models.connection import get_db_session


def render():
    """Render the fleet management page."""
    st.markdown('<div class="chart-title">🚛 Fleet Management</div>', unsafe_allow_html=True)

    with get_db_session() as session:
        vehicles = session.query(Vehicle).filter(Vehicle.is_active == True).all()

        if not vehicles:
            st.warning("No vehicles in fleet. Initialize database to add demo vehicles.")
            return

        # Search and filter
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            search = st.text_input("🔍 Search vehicles", placeholder="Search by ID, type, registration...")
        with col2:
            type_filter = st.selectbox("Type", ["All", "Truck", "Bus", "Van"], index=0)
        with col3:
            status_filter = st.selectbox("Health Status", ["All", "Healthy", "Maintenance Approaching", "Critical"], index=0)

        # Filter vehicles
        filtered = vehicles
        if search:
            search_lower = search.lower()
            filtered = [v for v in filtered if
                search_lower in v.vehicle_id.lower() or
                search_lower in v.registration_number.lower() or
                search_lower in v.manufacturer.lower() or
                search_lower in v.model.lower()]
        if type_filter != "All":
            filtered = [v for v in filtered if v.vehicle_type.value == type_filter.lower()]
        if status_filter != "All":
            status_map = {"Healthy": HealthStatus.HEALTHY, "Maintenance Approaching": HealthStatus.MAINTENANCE_APPROACHING, "Critical": HealthStatus.CRITICAL}
            filtered = [v for v in filtered if v.current_health_status == status_map[status_filter]]

        st.markdown(f"**Showing {len(filtered)} of {len(vehicles)} vehicles**")

        # Fleet table
        _render_fleet_table(filtered, session)

        st.markdown("---")

        # Fleet summary charts
        col1, col2 = st.columns(2)
        with col1:
            _render_fleet_by_type(vehicles)
        with col2:
            _render_fleet_health_scores(vehicles)


def _render_fleet_table(vehicles: list, session: Session):
    """Render searchable, filterable fleet table."""
    if not vehicles:
        st.info("No vehicles match the current filters.")
        return

    # Get latest sensor readings for each vehicle
    vehicle_data = []
    for v in vehicles:
        latest = session.query(SensorReading).filter(
            SensorReading.vehicle_id == v.id
        ).order_by(SensorReading.timestamp.desc()).first()

        # Next maintenance estimate
        next_maint = "—"
        if v.next_scheduled_service:
            days = (v.next_scheduled_service - utcnow()).days
            if days > 0:
                next_maint = f"{days} days"
            elif days == 0:
                next_maint = "Today"
            else:
                next_maint = f"{abs(days)} days overdue"

        vehicle_data.append({
            "Vehicle ID": v.vehicle_id,
            "Type": v.vehicle_type.value.title(),
            "Registration": v.registration_number,
            "Manufacturer": v.manufacturer,
            "Model": v.model,
            "Health Score": f"{v.health_score:.1f}%",
            "Status": v.current_health_status.value.replace("_", " ").title(),
            "Mileage (km)": f"{v.total_distance_km:,.0f}",
            "Engine Temp (°C)": f"{latest.engine_temperature_c:.1f}" if latest else "—",
            "Oil Pressure (kPa)": f"{latest.oil_pressure_kpa:.0f}" if latest else "—",
            "Last Service": v.last_service_date.strftime("%Y-%m-%d") if v.last_service_date else "—",
            "Next Maintenance": next_maint,
        })

    df = pd.DataFrame(vehicle_data)

    # Style the dataframe
    def style_status(val):
        if val == "Healthy":
            return "background-color: rgba(40,167,69,0.15); color: #28a745; font-weight: 600;"
        elif val == "Maintenance Approaching":
            return "background-color: rgba(255,193,7,0.15); color: #b8860b; font-weight: 600;"
        elif val == "Critical":
            return "background-color: rgba(220,53,69,0.15); color: #dc3545; font-weight: 600;"
        return ""

    def style_health(val):
        try:
            score = float(val.replace("%", ""))
            if score >= 80:
                return "color: #28a745; font-weight: 700;"
            elif score >= 50:
                return "color: #ffc107; font-weight: 700;"
            else:
                return "color: #dc3545; font-weight: 700;"
        except:
            return ""

    styled = df.style.map(style_status, subset=["Status"]).map(style_health, subset=["Health Score"])

    # Display with clickable rows
    event = st.dataframe(
        styled,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Vehicle ID": st.column_config.TextColumn("Vehicle ID", width="small"),
            "Type": st.column_config.TextColumn("Type", width="small"),
            "Health Score": st.column_config.TextColumn("Health Score", width="small"),
            "Status": st.column_config.TextColumn("Status", width="medium"),
        },
        on_select="rerun",
        selection_mode="single-row",
    )

    # Handle row selection
    if event.selection.rows:
        selected_idx = event.selection.rows[0]
        selected_vehicle = vehicles[selected_idx]
        st.session_state.selected_vehicle = selected_vehicle.id
        st.session_state.current_page = "vehicle_detail"
        st.rerun()


def _render_fleet_by_type(vehicles: list):
    """Render fleet composition by vehicle type."""
    st.markdown("""
    <div class="chart-container">
        <div class="chart-title">Fleet Composition by Type</div>
    </div>
    """, unsafe_allow_html=True)

    type_counts = {}
    for v in vehicles:
        t = v.vehicle_type.value.title()
        type_counts[t] = type_counts.get(t, 0) + 1

    fig = go.Figure(data=[go.Pie(
        labels=list(type_counts.keys()),
        values=list(type_counts.values()),
        hole=0.4,
        marker_colors=["#00a8e8", "#28a745", "#ffc107", "#6f42c1"],
        textinfo="label+value+percent",
        textfont_size=12,
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


def _render_fleet_health_scores(vehicles: list):
    """Render health score distribution."""
    st.markdown("""
    <div class="chart-container">
        <div class="chart-title">Health Score Distribution</div>
    </div>
    """, unsafe_allow_html=True)

    scores = [v.health_score for v in vehicles]

    fig = go.Figure(data=[go.Histogram(
        x=scores,
        nbinsx=20,
        marker_color="#00a8e8",
        opacity=0.8,
    )])

    # Add threshold lines
    fig.add_vline(x=80, line_dash="dash", line_color="#28a745", line_width=2)
    fig.add_vline(x=50, line_dash="dash", line_color="#ffc107", line_width=2)

    fig.update_layout(
        margin=dict(t=10, b=10, l=10, r=10),
        height=300,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#1e3a5f"),
        xaxis=dict(title="Health Score", showgrid=True, gridcolor="rgba(0,0,0,0.05)"),
        yaxis=dict(title="Count", showgrid=True, gridcolor="rgba(0,0,0,0.05)"),
        showlegend=False,
    )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})