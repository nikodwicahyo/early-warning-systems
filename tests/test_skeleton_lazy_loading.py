"""
Smoke tests for lazy loading + skeleton UI standardization.
Verifies new helpers exist, have correct signatures, and that
page modules import cleanly without calling load_models at import time.
"""
import sys
import os
import types
import unittest
from unittest.mock import MagicMock, patch

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def _make_st_stub():
    """Minimal Streamlit stub for unit tests."""
    st = MagicMock()
    st.session_state = {}
    # Make st.columns return a list of MagicMocks (not a single Mock)
    st.columns.side_effect = lambda n, **kw: [MagicMock() for _ in range(n if isinstance(n, int) else len(n))]
    st.empty.return_value = MagicMock()
    st.markdown = MagicMock()
    st.cache_data = MagicMock(return_value=lambda f: f)
    st.cache_resource = MagicMock(return_value=lambda f: f)
    st.rerun = MagicMock()
    st.secrets = {}
    return st


class TestSkeletonHelpers(unittest.TestCase):
    """Test show_page_skeleton, clear_page_skeleton, models_ready."""

    def setUp(self):
        self._st_stub = _make_st_stub()
        self._orig_st = sys.modules.get("streamlit")
        sys.modules["streamlit"] = self._st_stub
        # Remove cached src.utils so it re-imports with stub
        for key in list(sys.modules.keys()):
            if key in ("src.utils", "src.config"):
                del sys.modules[key]

    def tearDown(self):
        if self._orig_st is not None:
            sys.modules["streamlit"] = self._orig_st
        for key in list(sys.modules.keys()):
            if key in ("src.utils", "src.config"):
                del sys.modules[key]

    def _get_utils(self):
        # Stub src.config.LOGS_DIR so utils.py doesn't fail on import
        import src.config as cfg_real  # real module — just patch LOGS_DIR
        from pathlib import Path
        cfg_real.LOGS_DIR = Path(PROJECT_ROOT) / "logs"
        import src.utils as utils
        return utils

    def test_show_page_skeleton_exists(self):
        utils = self._get_utils()
        self.assertTrue(hasattr(utils, "show_page_skeleton"))

    def test_clear_page_skeleton_exists(self):
        utils = self._get_utils()
        self.assertTrue(hasattr(utils, "clear_page_skeleton"))

    def test_models_ready_exists(self):
        utils = self._get_utils()
        self.assertTrue(hasattr(utils, "models_ready"))

    def test_show_page_skeleton_returns_dict_with_header(self):
        utils = self._get_utils()
        for page in ("dashboard", "prediksi", "shap_analysis", "monitoring",
                     "detail_bank", "retraining", "model_management"):
            result = utils.show_page_skeleton(page)
            self.assertIsInstance(result, dict, f"show_page_skeleton('{page}') must return dict")
            self.assertIn("header", result, f"'{page}' skeleton must have 'header' key")

    def test_show_page_skeleton_metrics_is_list_for_data_pages(self):
        utils = self._get_utils()
        for page in ("dashboard", "prediksi", "monitoring", "detail_bank"):
            result = utils.show_page_skeleton(page)
            self.assertIn("metrics", result, f"'{page}' must have 'metrics'")
            self.assertIsInstance(result["metrics"], list, f"'{page}' metrics must be list")

    def test_clear_page_skeleton_does_not_raise(self):
        utils = self._get_utils()
        handles = utils.show_page_skeleton("dashboard")
        # Should not raise regardless of mock state
        try:
            utils.clear_page_skeleton(handles)
        except Exception as e:
            self.fail(f"clear_page_skeleton raised: {e}")

    def test_models_ready_returns_true_when_already_loaded(self):
        utils = self._get_utils()
        import streamlit as st
        st.session_state["models_loaded"] = True
        st.session_state.pop("_models_loading", None)
        result = utils.models_ready()
        self.assertTrue(result)

    def test_models_ready_loads_and_returns_true(self):
        utils = self._get_utils()
        import streamlit as st
        st.session_state["models_loaded"] = False
        st.session_state.pop("_models_loading", None)

        mock_models = {"lgbm_reg": {}, "feat_cols": []}
        mock_ds_instance = MagicMock()
        mock_ds_instance.load_models.return_value = mock_models
        mock_ds_cls = MagicMock(return_value=mock_ds_instance)

        with patch.dict(sys.modules, {"src.data_service": MagicMock(DataService=mock_ds_cls)}):
            result = utils.models_ready()

        self.assertIsInstance(result, bool)
        self.assertTrue(result)
        self.assertTrue(st.session_state.get("models_loaded"))

    def test_models_ready_handles_load_error_gracefully(self):
        """models_ready must not raise even if load_models returns error dict."""
        utils = self._get_utils()
        import streamlit as st
        st.session_state["models_loaded"] = False
        st.session_state.pop("_models_loading", None)

        mock_ds_instance = MagicMock()
        mock_ds_instance.load_models.return_value = {"load_error": "torch DLL failed"}
        mock_ds_cls = MagicMock(return_value=mock_ds_instance)

        with patch.dict(sys.modules, {"src.data_service": MagicMock(DataService=mock_ds_cls)}):
            result = utils.models_ready()

        self.assertIsInstance(result, bool)
        # models_loaded should still be set to True (graceful degradation)
        self.assertTrue(st.session_state.get("models_loaded"))


class TestCacheConfig(unittest.TestCase):
    """Verify cache TTL settings in data_loader."""

    def test_load_bank_metadata_has_86400_ttl(self):
        """load_bank_metadata should use ttl=86400 (static data)."""
        with open(os.path.join(PROJECT_ROOT, "src", "data_loader.py"), encoding='utf-8') as f:
            source = f.read()
        import re
        match = re.search(r'@st\.cache_data\([^)]*ttl=(\d+)[^)]*\)[^\n]*\n\s*def load_bank_metadata', source)
        self.assertIsNotNone(match, "load_bank_metadata must have @st.cache_data decorator")
        self.assertEqual(int(match.group(1)), 86400, "load_bank_metadata TTL should be 86400")

    def test_get_hf_space_info_has_300_ttl(self):
        """get_hf_space_info should use ttl=300."""
        with open(os.path.join(PROJECT_ROOT, "pages", "model_management.py"), encoding='utf-8') as f:
            source = f.read()
        import re
        match = re.search(r'@st\.cache_data\([^)]*ttl=(\d+)[^)]*\)[^\n]*\n\s*def get_hf_space_info', source)
        self.assertIsNotNone(match, "get_hf_space_info must have @st.cache_data decorator")
        self.assertEqual(int(match.group(1)), 300, "get_hf_space_info TTL should be 300")

    def test_load_models_not_called_in_load_application_data(self):
        """app.py load_application_data() must not call load_models."""
        with open(os.path.join(PROJECT_ROOT, "app.py"), encoding='utf-8') as f:
            source = f.read()
        import ast
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "load_application_data":
                func_src = ast.get_source_segment(source, node) or ""
                self.assertNotIn("load_models", func_src,
                    "load_application_data must NOT call load_models (lazy loading)")


class TestPageSkeletonIntegration(unittest.TestCase):
    """Verify each modified page imports the new helpers."""

    def _check_page_imports(self, page_file, expected_symbols):
        path = os.path.join(PROJECT_ROOT, "pages", page_file)
        with open(path, encoding='utf-8') as f:
            source = f.read()
        for sym in expected_symbols:
            self.assertIn(sym, source, f"{page_file} must import/use '{sym}'")

    def test_prediksi_uses_models_ready(self):
        self._check_page_imports("prediksi.py", ["models_ready", "show_page_skeleton", "clear_page_skeleton"])

    def test_shap_uses_models_ready(self):
        self._check_page_imports("shap_analysis.py", ["models_ready", "show_page_skeleton", "clear_page_skeleton"])

    def test_dashboard_uses_extended_skeletons(self):
        self._check_page_imports("dashboard.py", ["heatmap_placeholder", "chart_placeholder", "show_page_skeleton"])

    def test_monitoring_uses_skeleton(self):
        self._check_page_imports("monitoring.py", ["show_page_skeleton", "clear_page_skeleton"])

    def test_detail_bank_uses_skeleton(self):
        self._check_page_imports("detail_bank.py", ["show_page_skeleton", "clear_page_skeleton"])

    def test_model_management_uses_skeleton(self):
        self._check_page_imports("model_management.py", ["show_page_skeleton", "clear_page_skeleton"])

    def test_app_uses_nav_skeleton(self):
        path = os.path.join(PROJECT_ROOT, "app.py")
        with open(path, encoding='utf-8') as f:
            source = f.read()
        self.assertIn("is_navigating", source)
        self.assertIn("show_page_skeleton", source)
        self.assertIn("clear_page_skeleton", source)


if __name__ == "__main__":
    unittest.main()
