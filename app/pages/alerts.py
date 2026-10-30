"""
Alert Center Page for FleetGuard AI.
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

from sqlalchemy.orm import Session
from sqlalchemy import func

from app.models.database import (
    MaintenanceAlert, Vehicle, Anomaly, AlertSeverity, utcnow
)
from app.models.connection import get_db_session


def render():
    """Render the alert center page."""
    st.markdown('<div class="chart-title">🚨 Alert Center</div>', unsafe_allow_html=True)

    with get_db_session() as session:
        # Filters
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            severity_filter = st.selectbox("Severity", ["All", "Critical", "High", "Medium", "Low"], index=0)
        with col2:
            status_filter = st.selectbox("Status", ["All", "Active", "Resolved", "Acknowledged"], index=0)
        with col3:
            time_filter = st.selectbox("Time Range", ["Last 24h", "Last 7d", "Last 30d", "All Time"], index=0)
        with col4:
            if st.button("🔄 Refresh", use_container_width=True):
                st.rerun()

        # Build query
        query = session.query(MaintenanceAlert).join(Vehicle)

        # Apply severity filter
        if severity_filter != "All":
            query = query.filter(MaintenanceAlert.severity == AlertSeverity[severity_filter.upper()])

        # Apply status filter
        if status_filter == "Active":
            query = query.filter(MaintenanceAlert.is_resolved == False, MaintenanceAlert.is_acknowledged == False)
        elif status_filter == "Resolved":
            query = query.filter(MaintenanceAlert.is_resolved == True)
        elif status_filter == "Acknowledged":
            query = query.filter(MaintenanceAlert.is_acknowledged == True, MaintenanceAlert.is_resolved == False)

        # Apply time filter
        if time_filter == "Last 24h":
            query = query.filter(MaintenanceAlert.timestamp >= utcnow() - timedelta(hours=24))
        elif time_filter == "Last 7d":
            query = query.filter(MaintenanceAlert.timestamp >= utcnow() - timedelta(days=7))
        elif time_filter == "Last 30d":
            query = query.filter(MaintenanceAlert.timestamp >= utcnow() - timedelta(days=30))

        alerts = query.order_by(MaintenanceAlert.timestamp.desc()).limit(200).all()

        # Summary metrics
        _render_alert_summary(session, alerts)

        st.markdown("---")

        # Alert list
        _render_alert_list(alerts, session)

        st.markdown("---")

        # Alert analytics
        col1, col2 = st.columns(2)
        with col1:
            _render_alerts_by_severity(alerts)
        with col2:
            _render_alerts_by_component(alerts)

        # Alert timeline
        _render_alert_timeline(alerts)


def _render_alert_summary(session: Session, alerts: list):
    """Render alert summary metrics."""
    total = len(alerts)
    critical = sum(1 for a in alerts if a.severity == AlertSeverity.CRITICAL)
    high = sum(1 for a in alerts if a.severity == AlertSeverity.HIGH)
    medium = sum(1 for a in alerts if a.severity == AlertSeverity.MEDIUM)
    low = sum(1 for a in alerts if a.severity == AlertSeverity.LOW)
    resolved = sum(1 for a in alerts if a.is_resolved)
    acknowledged = sum(1 for a in alerts if a.is_acknowledged and not a.is_resolved)
    active = total - resolved

    cols = st.columns(6)
    cols[0].metric("Total Alerts", total)
    cols[1].metric("🔴 Critical", critical, delta_color="inverse")
    cols[2].metric("🟠 High", high, delta_color="inverse")
    cols[3].metric("🟡 Medium", medium)
    cols[4].metric("🔵 Low", low)
    cols[5].metric("✅ Resolved", resolved)


def _render_alert_list(alerts: list, session: Session):
    """Render detailed alert list."""
    if not alerts:
        st.info("No alerts match the current filters.")
        return

    st.markdown(f"### Alerts ({len(alerts)} total)")

    for alert in alerts:
        vehicle = session.query(Vehicle).filter(Vehicle.id == alert.vehicle_id).first()
        vehicle_id = vehicle.vehicle_id if vehicle else "Unknown"

        severity_class = alert.severity.value
        status_badge = ""
        if alert.is_resolved:
            status_badge = '<span class="status-badge status-completed">Resolved</span>'
        elif alert.is_acknowledged:
            status_badge = '<span class="status-badge status-assigned">Acknowledged</span>'
        else:
            status_badge = f'<span class="status-badge status-{severity_class}">Active</span>'

        with st.expander(f"{alert.title} • {vehicle_id} • {alert.timestamp.strftime('%Y-%m-%d %H:%M')} {status_badge}", expanded=False):
            col1, col2 = st.columns([2, 1])

            with col1:
                st.markdown(f"""
                **Component:** {alert.component}
                **Severity:** <span class="status-badge status-{severity_class}">{severity_class.upper()}</span>
                **Type:** {alert.alert_type}
                **Time:** {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
                **Description:** {alert.description}
                **Recommended Action:** {alert.recommended_action}
                """, unsafe_allow_html=True)

                if alert.evidence:
                    import json
                    try:
                        evidence = json.loads(alert.evidence)
                        st.json(evidence)
                    except:
                        st.text(alert.evidence)

            with col2:
                st.markdown("**Actions**")

                if not alert.is_acknowledged:
                    if st.button("✅ Acknowledge", key=f"ack_{alert.id}", use_container_width=True):
                        alert.is_acknowledged = True
                        alert.acknowledged_at = utcnow()
                        session.commit()
                        st.success("Alert acknowledged!")
                        st.rerun()
                else:
                    st.success("✅ Acknowledged")
                    if alert.acknowledged_at:
                        st.caption(f"By: {alert.acknowledged_at.strftime('%H:%M:%S')}")

                if not alert.is_resolved:
                    if st.button("✅ Resolve", key=f"resolve_{alert.id}", use_container_width=True, type="primary"):
                        alert.is_resolved = True
                        alert.resolved_at = utcnow()
                        session.commit()
                        st.success("Alert resolved!")
                        st.rerun()
                else:
                    st.success("✅ Resolved")
                    if alert.resolved_at:
                        st.caption(f"At: {alert.resolved_at.strftime('%H:%M:%S')}")

                if alert.resolution_notes:
                    st.caption(f"Notes: {alert.resolution_notes}")

                # Show linked work order
                if alert.work_order:
                    wo = alert.work_order
                    st.markdown(f"**Work Order:** [{wo.work_order_id}](/?page=maintenance&wo={wo.id})")
                    st.caption(f"Status: {wo.status.value.replace('_', ' ').title()}")


def _render_alerts_by_severity(alerts: list):
    """Render alerts by severity chart."""
    st.markdown("""
    <div class="chart-container">
        <div class="chart-title">Alerts by Severity</div>
    </div>
    """, unsafe_allow_html=True)

    severity_counts = {}
    for a in alerts:
        sev = a.severity.value
        severity_counts[sev] = severity_counts.get(sev, 0) + 1

    if severity_counts:
        colors = {"critical": "#dc3545", "high": "#fd7e14", "medium": "#ffc107", "low": "#00a8e8"}

        fig = go.Figure(data=[go.Pie(
            labels=[k.title() for k in severity_counts.keys()],
            values=list(severity_counts.values()),
            hole=0.4,
            marker_colors=[colors.get(k, "#6c757d") for k in severity_counts.keys()],
            textinfo="label+value+percent",
        )])

        fig.update_layout(
            showlegend=False,
            margin=dict(t=0, b=0, l=0, r=0),
            height=300,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#1e3a5f"),
        )

        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("No data")


def _render_alerts_by_component(alerts: list):
    """Render alerts by component chart."""
    st.markdown("""
    <div class="chart-container">
        <div class="chart-title">Alerts by Component</div>
    </div>
    """, unsafe_allow_html=True)

    component_counts = {}
    for a in alerts:
        component_counts[a.component] = component_counts.get(a.component, 0) + 1

    if component_counts:
        # Sort by count
        sorted_components = sorted(component_counts.items(), key=lambda x: x[1], reverse=True)[:10]

        fig = go.Figure(data=[go.Bar(
            y=[c[0] for c in sorted_components],
            x=[c[1] for c in sorted_components],
            orientation="h",
            marker_color="#00a8e8",
            text=[c[1] for c in sorted_components],
            textposition="outside",
        )])

        fig.update_layout(
            margin=dict(t=10, b=10, l=10, r=10),
            height=300,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#1e3a5f"),
            xaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.05)"),
            yaxis=dict(showgrid=False),
        )

        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("No data")


def _render_alert_timeline(alerts: list):
    """Render alert timeline."""
    st.markdown("""
    <div class="chart-container">
        <div class="chart-title">Alert Timeline</div>
    </div>
    """, unsafe_allow_html=True)

    if not alerts:
        st.info("No alerts to display.")
        return

    df = pd.DataFrame([{
        "timestamp": a.timestamp,
        "severity": a.severity.value,
        "component": a.component,
        "vehicle_id": a.vehicle_id,
    } for a in alerts])

    # Add vehicle ID lookup
    with get_db_session() as session:
        vehicles = session.query(Vehicle).filter(Vehicle.id.in_(df["vehicle_id"].unique())).all()
        vmap = {v.id: v.vehicle_id for v in vehicles}
        df["vehicle"] = df["vehicle_id"].map(vmap)

    severity_colors = {
        "critical": "#dc3545",
        "high": "#fd7e14",
        "medium": "#ffc107",
        "low": "#00a8e8",
    }

    fig = go.Figure()

    for severity in ["critical", "high", "medium", "low"]:
        subset = df[df["severity"] == severity]
        if not subset.empty:
            fig.add_trace(go.Scatter(
                x=subset["timestamp"],
                y=subset["vehicle"],
                mode="markers",
                name=severity.title(),
                marker=dict(
                    size=10,
                    color=severity_colors[severity],
                    symbol="circle",
                    line=dict(width=1, color="white"),
                ),
                hovertemplate="<b>%{y}</b><br>Component: %{text}<br>Time: %{x}<extra></extra>",
                text=subset["component"],
            ))

    fig.update_layout(
        height=350,
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