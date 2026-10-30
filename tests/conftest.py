"""
Pytest configuration for FleetGuard AI tests.
"""
import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture(scope="session", autouse=True)
def setup_test_env():
    """Set up test environment variables."""
    import os
    os.environ["DATABASE_URL"] = "sqlite:///:memory:"
    os.environ["MLFLOW_TRACKING_URI"] = "file:///tmp/mlruns"
    os.environ["DEMO_MODE"] = "true"
    yield
    # Cleanup


def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test requiring external resources"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )