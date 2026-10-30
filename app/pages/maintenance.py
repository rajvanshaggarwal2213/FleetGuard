"""
Maintenance Management Page for FleetGuard AI.
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

from sqlalchemy.orm import Session
from sqlalchemy import func

from app.models.database import (
    WorkOrder, WorkOrderStatus, WorkOrderPriority, Vehicle,
    MaintenanceAlert, MaintenanceHistory, Technician, AlertSeverity
)
from app.models.connection import get_db_session


def render():
    """Render the maintenance management page."""
    st.markdown('<div class="chart-title">🔧 Maintenance Management</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Work Orders", "👨‍🔧 Technicians", "📅 Calendar", "📊 Analytics"
    ])

    with tab1:
        _render_work_orders_tab()

    with tab2:
        _render_technicians_tab()

    with tab3:
        _render_calendar_tab()

    with tab4:
        _render_maintenance_analytics_tab()


def _render_work_orders_tab():
    """Render work orders management."""
    with get_db_session() as session:
        # Filters
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            status_filter = st.selectbox("Status",
                ["All", "Open", "Assigned", "In Progress", "Completed", "Cancelled"], index=0)
        with col2:
            priority_filter = st.selectbox("Priority",
                ["All", "Urgent", "High", "Medium", "Low"], index=0)
        with col3:
            vehicle_filter = st.selectbox("Vehicle", ["All"] +
                [v.vehicle_id for v in session.query(Vehicle).filter(Vehicle.is_active==True).all()])
        with col4:
            if st.button("🔄 Refresh", use_container_width=True):
                st.rerun()

        # Build query
        query = session.query(WorkOrder).join(Vehicle)

        if status_filter != "All":
            query = query.filter(WorkOrder.status == WorkOrderStatus[status_filter.upper().replace(" ", "_")])
        if priority_filter != "All":
            query = query.filter(WorkOrder.priority == WorkOrderPriority[priority_filter.upper()])
        if vehicle_filter != "All":
            vehicle = session.query(Vehicle).filter(Vehicle.vehicle_id == vehicle_filter).first()
            if vehicle:
                query = query.filter(WorkOrder.vehicle_id == vehicle.id)

        work_orders = query.order_by(WorkOrder.creation_timestamp.desc()).limit(100).all()

        # Summary metrics
        _render_wo_summary(session, work_orders)

        st.markdown("---")

        # Work order list
        if not work_orders:
            st.info("No work orders match the current filters.")
        else:
            for wo in work_orders:
                _render_work_order_card(wo, session)


def _render_wo_summary(session: Session, work_orders: list):
    """Render work order summary metrics."""
    total = len(work_orders)
    open_count = sum(1 for wo in work_orders if wo.status == WorkOrderStatus.OPEN)
    assigned = sum(1 for wo in work_orders if wo.status == WorkOrderStatus.ASSIGNED)
    in_progress = sum(1 for wo in work_orders if wo.status == WorkOrderStatus.IN_PROGRESS)
    completed = sum(1 for wo in work_orders if wo.status == WorkOrderStatus.COMPLETED)
    urgent = sum(1 for wo in work_orders if wo.priority == WorkOrderPriority.URGENT)

    cols = st.columns(6)
    cols[0].metric("Total", total)
    cols[1].metric("📋 Open", open_count)
    cols[2].metric("👤 Assigned", assigned)
    cols[3].metric("⚙️ In Progress", in_progress)
    cols[4].metric("✅ Completed", completed)
    cols[5].metric("🚨 Urgent", urgent, delta_color="inverse")


def _render_work_order_card(wo: WorkOrder, session: Session):
    """Render a single work order card."""
    vehicle = session.query(Vehicle).filter(Vehicle.id == wo.vehicle_id).first()
    technician = session.query(Technician).filter(Technician.id == wo.technician_id).first() if wo.technician_id else None
    alert = session.query(MaintenanceAlert).filter(MaintenanceAlert.id == wo.alert_id).first() if wo.alert_id else None

    status_class = {
        WorkOrderStatus.OPEN: "status-open",
        WorkOrderStatus.ASSIGNED: "status-assigned",
        WorkOrderStatus.IN_PROGRESS: "status-in-progress",
        WorkOrderStatus.COMPLETED: "status-completed",
        WorkOrderStatus.CANCELLED: "status-cancelled",
    }[wo.status]

    priority_class = {
        WorkOrderPriority.URGENT: "status-critical",
        WorkOrderPriority.HIGH: "status-high",
        WorkOrderPriority.MEDIUM: "status-medium",
        WorkOrderPriority.LOW: "status-low",
    }[wo.priority]

    with st.expander(f"{wo.work_order_id} • {vehicle.vehicle_id if vehicle else 'Unknown'} • {wo.component}", expanded=False):
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown(f"""
            **Component:** {wo.component}
            **Vehicle:** {vehicle.vehicle_id if vehicle else 'Unknown'} ({vehicle.registration_number if vehicle else ''})
            **Issue:** {wo.issue_description}
            **Priority:** <span class="status-badge {priority_class}">{wo.priority.value.upper()}</span>
            **Status:** <span class="status-badge {status_class}">{wo.status.value.replace('_', ' ').title()}</span>
            **Recommended Inspection:** {wo.recommended_inspection}
            **Created:** {wo.creation_timestamp.strftime('%Y-%m-%d %H:%M')}
            {f'**Assigned:** {wo.assigned_timestamp.strftime("%Y-%m-%d %H:%M")}' if wo.assigned_timestamp else ''}
            {f'**Started:** {wo.started_timestamp.strftime("%Y-%m-%d %H:%M")}' if wo.started_timestamp else ''}
            {f'**Completed:** {wo.completed_timestamp.strftime("%Y-%m-%d %H:%M")}' if wo.completed_timestamp else ''}
            """, unsafe_allow_html=True)

            if wo.completion_notes:
                st.markdown(f"**Completion Notes:** {wo.completion_notes}")

        with col2:
            st.markdown("**Actions**")

            # Status transitions
            if wo.status == WorkOrderStatus.OPEN:
                tech_options = session.query(Technician).filter(Technician.is_active == True).all()
                tech_names = ["Unassigned"] + [f"{t.employee_id} - {t.name}" for t in tech_options]
                selected_tech = st.selectbox("Assign Technician", tech_names, key=f"assign_{wo.id}")

                if st.button("📋 Assign", key=f"btn_assign_{wo.id}", use_container_width=True):
                    if selected_tech != "Unassigned":
                        tech_id = int(selected_tech.split(" - ")[0].replace("TECH-", ""))
                        tech = session.query(Technician).filter(Technician.employee_id == f"TECH-{tech_id:03d}").first()
                        if tech:
                            wo.technician_id = tech.id
                            wo.status = WorkOrderStatus.ASSIGNED
                            wo.assigned_timestamp = datetime.utcnow()
                            session.commit()
                            st.success(f"Assigned to {tech.name}!")
                            st.rerun()
                    else:
                        st.warning("Please select a technician")

            elif wo.status == WorkOrderStatus.ASSIGNED:
                if st.button("▶️ Start Work", key=f"btn_start_{wo.id}", use_container_width=True, type="primary"):
                    wo.status = WorkOrderStatus.IN_PROGRESS
                    wo.started_timestamp = datetime.utcnow()
                    session.commit()
                    st.success("Work started!")
                    st.rerun()

            elif wo.status == WorkOrderStatus.IN_PROGRESS:
                completion_notes = st.text_area("Completion Notes", key=f"notes_{wo.id}",
                    placeholder="Describe work performed, parts replaced, etc.")
                if st.button("✅ Complete", key=f"btn_complete_{wo.id}", use_container_width=True, type="primary"):
                    wo.status = WorkOrderStatus.COMPLETED
                    wo.completed_timestamp = datetime.utcnow()
                    wo.completion_notes = completion_notes
                    wo.resolution_timestamp = datetime.utcnow()

                    # Create maintenance history record
                    history = MaintenanceHistory(
                        vehicle_id=wo.vehicle_id,
                        work_order_id=wo.id,
                        service_date=datetime.utcnow(),
                        service_type="corrective" if wo.priority in [WorkOrderPriority.URGENT, WorkOrderPriority.HIGH] else "preventive",
                        component=wo.component,
                        description=wo.issue_description,
                        technician_name=technician.name if technician else "Unknown",
                        notes=completion_notes,
                        mileage_at_service=vehicle.total_distance_km if vehicle else None,
                        engine_hours_at_service=vehicle.engine_hours if vehicle else None,
                    )
                    session.add(history)

                    # Update alert if linked
                    if alert:
                        alert.is_resolved = True
                        alert.resolved_at = datetime.utcnow()
                        alert.resolution_notes = completion_notes

                    session.commit()
                    st.success("Work order completed!")
                    st.rerun()

            elif wo.status == WorkOrderStatus.COMPLETED:
                st.success("✅ Completed")
                if st.button("🔄 Reopen", key=f"btn_reopen_{wo.id}", use_container_width=True):
                    wo.status = WorkOrderStatus.OPEN
                    wo.completed_timestamp = None
                    wo.resolution_timestamp = None
                    session.commit()
                    st.success("Work order reopened!")
                    st.rerun()

            # Cancel button for non-completed
            if wo.status not in [WorkOrderStatus.COMPLETED, WorkOrderStatus.CANCELLED]:
                if st.button("❌ Cancel", key=f"btn_cancel_{wo.id}", use_container_width=True):
                    wo.status = WorkOrderStatus.CANCELLED
                    wo.cancelled_timestamp = datetime.utcnow()
                    session.commit()
                    st.success("Work order cancelled!")
                    st.rerun()

            # Show linked alert
            if alert:
                st.markdown("---")
                st.caption(f"Linked Alert: {alert.title}")
                st.caption(f"Severity: {alert.severity.value.title()}")


def _render_technicians_tab():
    """Render technicians management."""
    with get_db_session() as session:
        technicians = session.query(Technician).filter(Technician.is_active == True).all()

        st.markdown('<div class="chart-title">Technician Workload</div>', unsafe_allow_html=True)

        if not technicians:
            st.info("No technicians registered.")
            return

        # Technician cards
        for tech in technicians:
            # Count work orders by status
            assigned = session.query(WorkOrder).filter(
                WorkOrder.technician_id == tech.id,
                WorkOrder.status.in_([WorkOrderStatus.ASSIGNED, WorkOrderStatus.IN_PROGRESS])
            ).count()
            completed = session.query(WorkOrder).filter(
                WorkOrder.technician_id == tech.id,
                WorkOrder.status == WorkOrderStatus.COMPLETED
            ).count()

            with st.expander(f"{tech.employee_id} - {tech.name} ({tech.specialization})", expanded=True):
                col1, col2, col3 = st.columns(3)
                col1.metric("Active", assigned)
                col2.metric("Completed", completed)
                col3.metric("Total", assigned + completed)

                st.caption(f"Email: {tech.email or 'N/A'} | Phone: {tech.phone or 'N/A'}")


def _render_calendar_tab():
    """Render maintenance calendar."""
    st.markdown('<div class="chart-title">Maintenance Calendar</div>', unsafe_allow_html=True)

    with get_db_session() as session:
        # Get upcoming scheduled maintenance
        vehicles = session.query(Vehicle).filter(
            Vehicle.is_active == True,
            Vehicle.next_scheduled_service.isnot(None)
        ).order_by(Vehicle.next_scheduled_service).all()

        if not vehicles:
            st.info("No scheduled maintenance.")
            return

        # Calendar view
        today = datetime.utcnow().date()
        next_30_days = today + timedelta(days=30)

        st.markdown("### Upcoming Scheduled Maintenance (Next 30 Days)")

        cal_data = []
        for v in vehicles:
            if v.next_scheduled_service:
                service_date = v.next_scheduled_service.date()
                if service_date <= next_30_days:
                    days_until = (service_date - today).days
                    cal_data.append({
                        "Date": service_date.strftime("%Y-%m-%d"),
                        "Days": days_until,
                        "Vehicle": v.vehicle_id,
                        "Type": v.vehicle_type.value.title(),
                        "Mileage": f"{v.total_distance_km:,.0f} km",
                        "Last Service": v.last_service_date.strftime("%Y-%m-%d") if v.last_service_date else "Never",
                    })

        if cal_data:
            df = pd.DataFrame(cal_data)

            def style_days(val):
                if val <= 0:
                    return "color: #dc3545; font-weight: 700;"
                elif val <= 3:
                    return "color: #fd7e14; font-weight: 700;"
                elif val <= 7:
                    return "color: #ffc107; font-weight: 700;"
                return ""

            st.dataframe(
                df.style.map(style_days, subset=["Days"]),
                use_container_width=True,
                hide_index=True,
            )

            # Timeline chart
            fig = go.Figure()
            for _, row in df.iterrows():
                color = "#dc3545" if row["Days"] <= 0 else ("#fd7e14" if row["Days"] <= 3 else ("#ffc107" if row["Days"] <= 7 else "#28a745"))
                fig.add_trace(go.Scatter(
                    x=[row["Date"]],
                    y=[row["Vehicle"]],
                    mode="markers+text",
                    text=[f"{row['Days']}d"],
                    textposition="middle right",
                    marker=dict(size=15, color=color, symbol="square"),
                    showlegend=False,
                    hovertemplate=f"<b>{row['Vehicle']}</b><br>Date: {row['Date']}<br>Days: {row['Days']}<extra></extra>",
                ))

            fig.update_layout(
                height=400,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#1e3a5f"),
                xaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.05)"),
                yaxis=dict(showgrid=False),
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        else:
            st.success("No maintenance due in the next 30 days!")


def _render_maintenance_analytics_tab():
    """Render maintenance analytics."""
    st.markdown('<div class="chart-title">Maintenance Analytics</div>', unsafe_allow_html=True)

    with get_db_session() as session:
        # Completed work orders in last 30 days
        cutoff = datetime.utcnow() - timedelta(days=30)
        completed = session.query(WorkOrder).filter(
            WorkOrder.status == WorkOrderStatus.COMPLETED,
            WorkOrder.completed_timestamp >= cutoff
        ).all()

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            <div class="chart-container">
                <div class="chart-title">Work Orders by Type (Last 30 Days)</div>
            </div>
            """, unsafe_allow_html=True)

            if completed:
                type_counts = {}
                for wo in completed:
                    # Determine type from priority
                    if wo.priority == WorkOrderPriority.URGENT:
                        t = "Emergency"
                    elif wo.priority == WorkOrderPriority.HIGH:
                        t = "Corrective"
                    else:
                        t = "Preventive"
                    type_counts[t] = type_counts.get(t, 0) + 1

                fig = go.Figure(data=[go.Pie(
                    labels=list(type_counts.keys()),
                    values=list(type_counts.values()),
                    hole=0.4,
                    marker_colors=["#dc3545", "#fd7e14", "#28a745"],
                    textinfo="label+value+percent",
                )])
                fig.update_layout(
                    showlegend=False, margin=dict(t=0,b=0,l=0,r=0), height=300,
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#1e3a5f"),
                )
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            else:
                st.info("No completed work orders in the last 30 days.")

        with col2:
            st.markdown("""
            <div class="chart-container">
                <div class="chart-title">Average Resolution Time by Priority</div>
            </div>
            """, unsafe_allow_html=True)

            if completed:
                priority_times = {}
                for wo in completed:
                    if wo.creation_timestamp and wo.completed_timestamp:
                        hours = (wo.completed_timestamp - wo.creation_timestamp).total_seconds() / 3600
                        priority_times.setdefault(wo.priority.value, []).append(hours)

                avg_times = {k: sum(v)/len(v) for k, v in priority_times.items()}

                fig = go.Figure(data=[go.Bar(
                    x=list(avg_times.keys()),
                    y=list(avg_times.values()),
                    marker_color=["#dc3545", "#fd7e14", "#ffc107", "#00a8e8"],
                    text=[f"{v:.1f}h" for v in avg_times.values()],
                    textposition="outside",
                )])
                fig.update_layout(
                    margin=dict(t=10,b=10,l=10,r=10), height=300,
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#1e3a5f"),
                    xaxis=dict(showgrid=False),
                    yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.05)", title="Hours"),
                )
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            else:
                st.info("No data")

        # Maintenance history
        st.markdown("""
        <div class="chart-container">
            <div class="chart-title">Recent Maintenance History</div>
        </div>
        """, unsafe_allow_html=True)

        history = session.query(MaintenanceHistory).order_by(
            MaintenanceHistory.service_date.desc()
        ).limit(20).all()

        if history:
            hist_data = []
            for h in history:
                vehicle = session.query(Vehicle).filter(Vehicle.id == h.vehicle_id).first()
                hist_data.append({
                    "Date": h.service_date.strftime("%Y-%m-%d"),
                    "Vehicle": vehicle.vehicle_id if vehicle else "Unknown",
                    "Type": h.service_type.title(),
                    "Component": h.component,
                    "Technician": h.technician_name or "—",
                    "Cost": f"${h.cost:,.2f}" if h.cost else "—",
                    "Mileage": f"{h.mileage_at_service:,.0f} km" if h.mileage_at_service else "—",
                })
            df = pd.DataFrame(hist_data)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No maintenance history recorded.")