"""
Pytest Configuration and Shared Fixtures
"""

import sys
import types
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def _make_streamlit_mock():
    """Build a minimal streamlit mock that satisfies all src imports."""
    st = MagicMock()

    # session_state: behave like a dict
    st.session_state = {}

    # cache_data / cache_resource: act as passthrough decorators
    def _passthrough_decorator(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]
        def decorator(fn):
            return fn
        return decorator

    st.cache_data = MagicMock(side_effect=_passthrough_decorator)
    st.cache_data.clear = MagicMock()
    st.cache_resource = MagicMock(side_effect=_passthrough_decorator)
    st.cache_resource.clear = MagicMock()

    return st


# Inject streamlit mock before any test module is imported
_st_mock = _make_streamlit_mock()
sys.modules.setdefault('streamlit', _st_mock)


@pytest.fixture(scope="session")
def project_root_fixture():
    return project_root


@pytest.fixture
def temp_data_dir(tmp_path):
    """Temporary directory with standard sub-structure."""
    (tmp_path / 'data' / 'processed').mkdir(parents=True)
    (tmp_path / 'data' / 'predictions').mkdir(parents=True)
    (tmp_path / 'data' / 'uploads').mkdir(parents=True)
    (tmp_path / 'models' / 'versions').mkdir(parents=True)
    return tmp_path


def pytest_configure(config):
    for marker in ('unit', 'integration', 'slow'):
        config.addinivalue_line("markers", f"{marker}: mark test as {marker}")
