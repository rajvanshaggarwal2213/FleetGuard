"""
FleetGuard AI - Main Streamlit Application Entry Point.

Predictive Maintenance Platform for Fleet Management.
"""
import streamlit as st
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Page config - MUST be first Streamlit command
st.set_page_config(
    page_title="FleetGuard AI - Predictive Maintenance Platform",
    page_icon="🚛",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "FleetGuard AI - Predictive Maintenance Platform for Connected Fleets"
    }
)

# Custom CSS for enterprise styling
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary: #1e3a5f;
        --secondary: #2c5f8a;
        --accent: #00a8e8;
        --success: #28a745;
        --warning: #ffc107;
        --danger: #dc3545;
        --light: #f8f9fa;
        --dark: #1e3a5f;
        --bg: #ffffff;
        --card-bg: #ffffff;
        --border: #dee2e6;
    }

    @media (prefers-color-scheme: dark) {
        :root {
            --primary: #2c5f8a;
            --secondary: #3d7ab8;
            --accent: #4db8ff;
            --bg: #1a1f2e;
            --card-bg: #23293a;
            --border: #3d4658;
            --light: #2d3345;
        }
    }

    /* Global styles */
    .stApp {
        background-color: var(--bg);
    }

    /* Header */
    .fleetguard-header {
        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
        padding: 1.5rem 2rem;
        border-radius: 0 0 16px 16px;
        margin: -1rem -1rem 2rem -1rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    }

    .fleetguard-header h1 {
        color: white;
        margin: 0;
        font-weight: 700;
        font-size: 1.75rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }

    .fleetguard-header .tagline {
        color: rgba(255,255,255,0.9);
        font-size: 0.9rem;
        font-weight: 400;
        margin-top: 0.25rem;
    }

    /* Metric cards */
    .metric-card {
        background: var(--card-bg);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.25rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        transition: transform 0.2s, box-shadow 0.2s;
    }

    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
    }

    .metric-label {
        font-size: 0.8rem;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--dark);
        line-height: 1.2;
    }

    .metric-delta {
        font-size: 0.8rem;
        margin-top: 0.25rem;
    }

    .metric-delta.positive { color: var(--success); }
    .metric-delta.negative { color: var(--danger); }

    /* Status badges */
    .status-badge {
        display: inline-flex;
        align-items: center;
        padding: 0.35rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .status-healthy { background: rgba(40,167,69,0.15); color: var(--success); }
    .status-warning { background: rgba(255,193,7,0.15); color: #b8860b; }
    .status-critical { background: rgba(220,53,69,0.15); color: var(--danger); }
    .status-open { background: rgba(0,168,232,0.15); color: var(--accent); }
    .status-assigned { background: rgba(111,66,193,0.15); color: #6f42c1; }
    .status-in-progress { background: rgba(255,193,7,0.15); color: #b8860b; }
    .status-completed { background: rgba(40,167,69,0.15); color: var(--success); }
    .status-cancelled { background: rgba(108,117,125,0.15); color: #6c757d; }

    /* Sidebar */
    .sidebar-section {
        padding: 1rem 0;
        border-bottom: 1px solid var(--border);
    }

    .sidebar-section:last-child {
        border-bottom: none;
    }

    .sidebar-title {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #6c757d;
        font-weight: 700;
        margin-bottom: 0.75rem;
    }

    /* Demo control panel */
    .demo-panel {
        background: var(--card-bg);
        border: 2px solid var(--accent);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }

    .demo-panel h3 {
        color: var(--accent);
        margin-top: 0;
        font-size: 1.1rem;
    }

    .demo-button {
        width: 100%;
        margin: 0.25rem 0;
        padding: 0.6rem 1rem;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s;
    }

    .demo-button.primary {
        background: var(--accent);
        color: white;
        border: none;
    }

    .demo-button.primary:hover {
        background: #008cc7;
    }

    .demo-button.danger {
        background: var(--danger);
        color: white;
        border: none;
    }

    .demo-button.warning {
        background: var(--warning);
        color: var(--dark);
        border: none;
    }

    /* Simulation indicator */
    .sim-indicator {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }

    .sim-running {
        background: rgba(40,167,69,0.15);
        color: var(--success);
        animation: pulse 2s infinite;
    }

    .sim-stopped {
        background: rgba(108,117,125,0.15);
        color: #6c757d;
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.6; }
    }

    /* Vehicle cards */
    .vehicle-card {
        background: var(--card-bg);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        transition: all 0.2s;
    }

    .vehicle-card:hover {
        border-color: var(--accent);
        box-shadow: 0 4px 16px rgba(0,168,232,0.1);
    }

    .vehicle-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        margin-bottom: 1rem;
    }

    .vehicle-id {
        font-size: 1.1rem;
        font-weight: 700;
        color: var(--dark);
    }

    .vehicle-type {
        font-size: 0.75rem;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        background: var(--light);
        color: var(--secondary);
        font-weight: 600;
    }

    .vehicle-metrics {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
        gap: 1rem;
        margin-top: 1rem;
    }

    .vehicle-metric {
        text-align: center;
        padding: 0.75rem;
        background: var(--light);
        border-radius: 8px;
    }

    .vehicle-metric-value {
        font-size: 1.25rem;
        font-weight: 700;
        color: var(--dark);
    }

    .vehicle-metric-label {
        font-size: 0.7rem;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Health score circle */
    .health-score {
        width: 80px;
        height: 80px;
        border-radius: 50%;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        font-weight: 700;
    }

    .health-score.healthy { background: conic-gradient(var(--success) calc(var(--score)*3.6deg), var(--border) 0deg); }
    .health-score.warning { background: conic-gradient(var(--warning) calc(var(--score)*3.6deg), var(--border) 0deg); }
    .health-score.critical { background: conic-gradient(var(--danger) calc(var(--score)*3.6deg), var(--border) 0deg); }

    .health-score-inner {
        width: 64px;
        height: 64px;
        border-radius: 50%;
        background: var(--card-bg);
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }

    .health-score-value { font-size: 1.25rem; line-height: 1; }
    .health-score-label { font-size: 0.6rem; text-transform: uppercase; }

    /* Tables */
    .dataframe {
        font-size: 0.85rem !important;
    }

    .dataframe th {
        background: var(--light) !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-size: 0.7rem !important;
        color: #6c757d !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 0.75rem 1.25rem;
        font-weight: 600;
        font-size: 0.85rem;
    }

    /* Alerts */
    .alert-item {
        background: var(--card-bg);
        border: 1px solid var(--border);
        border-left: 4px solid var(--accent);
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.75rem;
    }

    .alert-item.critical { border-left-color: var(--danger); }
    .alert-item.high { border-left-color: var(--danger); }
    .alert-item.medium { border-left-color: var(--warning); }
    .alert-item.low { border-left-color: var(--accent); }

    .alert-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        margin-bottom: 0.5rem;
    }

    .alert-title {
        font-weight: 700;
        color: var(--dark);
    }

    .alert-meta {
        font-size: 0.75rem;
        color: #6c757d;
        margin-top: 0.5rem;
    }

    /* Work order cards */
    .wo-card {
        background: var(--card-bg);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }

    .wo-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.75rem;
    }

    .wo-id {
        font-family: monospace;
        font-size: 0.85rem;
        font-weight: 600;
        color: var(--accent);
    }

    /* Chart containers */
    .chart-container {
        background: var(--card-bg);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1.5rem;
    }

    .chart-title {
        font-size: 1rem;
        font-weight: 700;
        color: var(--dark);
        margin-bottom: 1rem;
    }

    /* Footer */
    .fleetguard-footer {
        text-align: center;
        padding: 2rem;
        color: #6c757d;
        font-size: 0.8rem;
        border-top: 1px solid var(--border);
        margin-top: 3rem;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: var(--light);
    }
    ::-webkit-scrollbar-thumb {
        background: var(--border);
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #adb5bd;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
def init_session_state():
    """Initialize session state variables."""
    defaults = {
        'simulation_running': False,
        'simulation_speed': 1.0,
        'selected_vehicle': None,
        'current_page': 'overview',
        'demo_mode': True,
        'last_update': None,
        'alert_filter': 'all',
        'work_order_filter': 'all',
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# Initialize
init_session_state()

# Import pages
from app.pages import overview, fleet, vehicle_detail, predictions, alerts, maintenance, data_management


def render_header():
    """Render application header."""
    col1, col2, col3 = st.columns([3, 1, 1])

    with col1:
        st.markdown("""
        <div class="fleetguard-header">
            <h1>🚛 FleetGuard AI</h1>
            <div class="tagline">Predict Before Failure. Maintain Before Breakdown.</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        # Simulation status indicator
        if st.session_state.simulation_running:
            st.markdown("""
            <div class="sim-indicator sim-running">
                🔴 LIVE SIMULATION
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="sim-indicator sim-stopped">
                ⚫ SIMULATION PAUSED
            </div>
            """, unsafe_allow_html=True)

    with col3:
        # Demo mode badge
        if st.session_state.demo_mode:
            st.markdown("""
            <div style="text-align: right; padding-top: 0.5rem;">
                <span class="status-badge status-warning">DEMO MODE</span>
            </div>
            """, unsafe_allow_html=True)


def render_sidebar():
    """Render sidebar navigation and controls."""
    with st.sidebar:
        # Navigation
        st.markdown('<div class="sidebar-title">Navigation</div>', unsafe_allow_html=True)

        pages = {
            "📊 Overview": "overview",
            "🚛 Fleet Management": "fleet",
            "🔍 Vehicle Details": "vehicle_detail",
            "🤖 AI/ML Predictions": "predictions",
            "🚨 Alert Center": "alerts",
            "🔧 Maintenance": "maintenance",
            "📁 Data & Models": "data_management",
        }

        for label, key in pages.items():
            if st.button(label, key=f"nav_{key}", use_container_width=True,
                         type="primary" if st.session_state.current_page == key else "secondary"):
                st.session_state.current_page = key
                st.rerun()

        st.markdown('<div class="sidebar-section"></div>', unsafe_allow_html=True)

        # Demo Control Panel
        st.markdown("""
        <div class="demo-panel">
            <h3>🎮 Demo Control Panel</h3>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("▶️ Start", key="demo_start", use_container_width=True, type="primary"):
                st.session_state.simulation_running = True
                st.rerun()
        with col2:
            if st.button("⏹️ Stop", key="demo_stop", use_container_width=True):
                st.session_state.simulation_running = False
                st.rerun()

        if st.button("🔄 Reset Simulation", key="demo_reset", use_container_width=True):
            st.session_state.simulation_running = False
            # Reset would be handled by simulation engine
            st.success("Simulation reset!")
            st.rerun()

        st.markdown("---")

        # Fault injection buttons
        st.markdown("**Fault Injection**")
        fault_buttons = [
            ("🌡️ Overheating", "overheating"),
            ("📳 Vibration", "vibration"),
            ("🛢️ Low Oil Pressure", "low_oil"),
            ("🔋 Battery Drain", "battery"),
            ("📉 Gradual Degradation", "degradation"),
        ]

        for label, fault_key in fault_buttons:
            if st.button(label, key=f"fault_{fault_key}", use_container_width=True):
                st.session_state[f"trigger_{fault_key}"] = True
                st.rerun()

        if st.button("✅ Generate Work Order", key="fault_workorder", use_container_width=True):
            st.session_state["trigger_workorder"] = True
            st.rerun()

        st.markdown('<div class="sidebar-section"></div>', unsafe_allow_html=True)

        # Simulation speed
        st.markdown('<div class="sidebar-title">Simulation Speed</div>', unsafe_allow_html=True)
        speed = st.slider("Speed", 0.1, 5.0, st.session_state.simulation_speed, 0.1,
                          key="sim_speed", label_visibility="collapsed")
        if speed != st.session_state.simulation_speed:
            st.session_state.simulation_speed = speed
            st.rerun()

        st.markdown('<div class="sidebar-section"></div>', unsafe_allow_html=True)

        # Filters
        st.markdown('<div class="sidebar-title">Filters</div>', unsafe_allow_html=True)

        alert_filter = st.selectbox(
            "Alert Severity",
            ["All", "Critical", "High", "Medium", "Low", "Resolved"],
            index=0,
            key="alert_filter_select"
        )
        st.session_state.alert_filter = alert_filter.lower()

        wo_filter = st.selectbox(
            "Work Order Status",
            ["All", "Open", "Assigned", "In Progress", "Completed", "Cancelled"],
            index=0,
            key="wo_filter_select"
        )
        st.session_state.work_order_filter = wo_filter.lower().replace(" ", "_")

        st.markdown('<div class="sidebar-section"></div>', unsafe_allow_html=True)

        # Info
        st.markdown("""
        <div style="font-size: 0.75rem; color: #6c757d;">
            <strong>FleetGuard AI v1.0.0</strong><br>
            Teliport Season 4 Hackathon<br>
            Case Study 1: Self-Healing Machines
        </div>
        """, unsafe_allow_html=True)


def render_page():
    """Render the current page."""
    page = st.session_state.current_page

    if page == "overview":
        overview.render()
    elif page == "fleet":
        fleet.render()
    elif page == "vehicle_detail":
        vehicle_detail.render()
    elif page == "predictions":
        predictions.render()
    elif page == "alerts":
        alerts.render()
    elif page == "maintenance":
        maintenance.render()
    elif page == "data_management":
        data_management.render()
    else:
        overview.render()


def main():
    """Main application entry point."""
    render_header()
    render_sidebar()
    render_page()

    # Footer
    st.markdown("""
    <div class="fleetguard-footer">
        FleetGuard AI — Predictive Maintenance Platform |
        Built for Teliport Season 4 Hackathon |
        <em>Simulated data for demonstration purposes only</em>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()