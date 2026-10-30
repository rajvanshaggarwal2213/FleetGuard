"""
Utility functions for FleetGuard AI.
"""
import hashlib
import json
from datetime import datetime
from typing import Any, Dict, List, Optional
from functools import wraps


def generate_work_order_id() -> str:
    """Generate a unique work order ID."""
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    hash_suffix = hashlib.md5(str(datetime.utcnow().timestamp()).encode()).hexdigest()[:4].upper()
    return f"WO-{timestamp}-{hash_suffix}"


def format_timedelta(seconds: float) -> str:
    """Format a timedelta in human-readable form."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    elif seconds < 86400:
        return f"{seconds/3600:.1f}h"
    else:
        return f"{seconds/86400:.1f}d"


def safe_json_loads(data: str, default: Any = None) -> Any:
    """Safely parse JSON with fallback."""
    try:
        return json.loads(data)
    except (json.JSONDecodeError, TypeError):
        return default


def safe_json_dumps(data: Any, default: str = "{}") -> str:
    """Safely serialize to JSON with fallback."""
    try:
        return json.dumps(data, default=str)
    except (TypeError, ValueError):
        return default


def calculate_health_color(score: float) -> str:
    """Get color for health score."""
    if score >= 80:
        return "#28a745"  # Green
    elif score >= 50:
        return "#ffc107"  # Yellow
    else:
        return "#dc3545"  # Red


def calculate_severity_color(severity: str) -> str:
    """Get color for alert severity."""
    colors = {
        "critical": "#dc3545",
        "high": "#fd7e14",
        "medium": "#ffc107",
        "low": "#00a8e8",
    }
    return colors.get(severity.lower(), "#6c757d")


def calculate_status_color(status: str) -> str:
    """Get color for work order status."""
    colors = {
        "open": "#00a8e8",
        "assigned": "#6f42c1",
        "in_progress": "#ffc107",
        "completed": "#28a745",
        "cancelled": "#6c757d",
    }
    return colors.get(status.lower(), "#6c757d")


def paginate_query(query, page: int = 1, page_size: int = 20):
    """Paginate a SQLAlchemy query."""
    total = query.count()
    items = query.offset((page - 1) * page_size).limit(page_size).all()
    return {
        "items": items,
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": (total + page_size - 1) // page_size,
    }


def validate_sensor_reading(data: Dict) -> List[str]:
    """Validate sensor reading data."""
    errors = []
    required = [
        "engine_temperature_c", "engine_rpm", "oil_pressure_kpa",
        "coolant_temperature_c", "vibration_mm_s", "battery_voltage_v",
        "fuel_consumption_l_per_100km"
    ]

    for field in required:
        if field not in data:
            errors.append(f"Missing required field: {field}")
        elif not isinstance(data[field], (int, float)):
            errors.append(f"Field {field} must be numeric")

    # Range checks
    ranges = {
        "engine_temperature_c": (-50, 200),
        "engine_rpm": (0, 5000),
        "oil_pressure_kpa": (0, 1000),
        "coolant_temperature_c": (-50, 150),
        "vibration_mm_s": (0, 50),
        "battery_voltage_v": (0, 20),
        "fuel_consumption_l_per_100km": (0, 100),
    }

    for field, (min_val, max_val) in ranges.items():
        if field in data and isinstance(data[field], (int, float)):
            if not min_val <= data[field] <= max_val:
                errors.append(f"{field} out of range: {data[field]} (expected {min_val}-{max_val})")

    return errors


def retry_on_failure(max_retries: int = 3, delay: float = 1.0):
    """Decorator to retry a function on failure."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        time.sleep(delay * (attempt + 1))
            raise last_exception
        return wrapper
    return decorator


# Import time for retry decorator
import time