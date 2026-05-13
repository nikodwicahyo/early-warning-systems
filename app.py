"""
OJK RBBR Early Warning System - Streamlit Application
Main Entry Point dengan Multi-Page Navigation
"""

import streamlit as st
import sys
import os
import textwrap
from pathlib import Path
import warnings

# Windows encoding for emojis in logs/console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

# Suppress persistent library warnings
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")
warnings.filterwarnings("ignore", message=".*Attribute 'loss' is an instance of `nn.Module`.*")
warnings.filterwarnings("ignore", message=".*Attribute 'logging_metrics' is an instance of `nn.Module`.*")
warnings.filterwarnings("ignore", message=".*The loaded checkpoint was produced with Lightning v.*")
warnings.filterwarnings("ignore", category=UserWarning, module="lightning.pytorch.utilities.parsing")
warnings.filterwarnings("ignore", message=".*Importing plotly failed.*")

# NOTE: neuralprophet warmup import removed from top-level — it's deferred to
# load_models() inside DataLoader which runs inside st.cache_resource.

from src.config import AppConfig
from src.data_loader import DataLoader
from src.utils import setup_logging, check_system_health, show_nav_loader, hide_nav_loader, inject_nav_loader_elements, show_page_skeleton, clear_page_skeleton
from utils.hf_storage import HFStorageManager

# Initialize logging at top-level to prevent crashes in pre-initialization paths (e.g. force_reload)
logger = setup_logging()

# =============================================================================
# PAGE CONFIGURATION  (must be first Streamlit call)
# =============================================================================

st.set_page_config(
    page_title="OJK - Early Warning System",
    page_icon="assets/favicon.ico",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/nikodwicahyo/',
        'About': """
        # OJK RBBR Early Warning System v1.0
        Sistem prediksi kondisi keuangan bank berbasis machine learning
        untuk deteksi dini risiko perbankan.
        © 2026 OJK DPKP - Internal Use Only
        """
    }
)

# =============================================================================
# UTILS & ANIMATIONS
# =============================================================================

# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

def initialize_session_state():
    defaults = {
        'data_loaded': False,
        'models_loaded': False,
        'predictions': None,
        'master_data': None,
        'selected_bank': None,
        'models': {},
        'retraining_in_progress': False,
        'retrain_version': 0,
        '_last_page': None,
        '_welcomed': False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

    if 'storage_manager' not in st.session_state:
        try:
            repo_id = os.getenv('SPACE_ID') or st.secrets.get('SPACE_ID')
            if not repo_id:
                st.error("❌ **Konfigurasi SPACE_ID Tidak Ditemukan**")
                st.info("Pastikan `SPACE_ID` (e.g., 'username/repo-name') telah diset di Streamlit Secrets atau Environment Variable.")
                st.stop()
                
            st.session_state.storage_manager = HFStorageManager(repo_id=repo_id)
        except Exception as e:
            st.warning(f"HF Storage initialization failed: {str(e)}")
            st.session_state.storage_manager = None


initialize_session_state()


# =============================================================================
# PAGE MODULE CACHE  — avoids re-importing on every navigation
# =============================================================================

@st.cache_resource
def _import_page(page_name: str):
    """Import a page module once and cache it for the session lifetime."""
    import importlib
    return importlib.import_module(f"pages.{page_name}")


# =============================================================================
# DATA LOADING (CACHED)
# =============================================================================


def load_application_data():
    """Load all necessary data and models — no caching, always fresh."""
    try:
        loader = DataLoader()
        
        # Sync from Hugging Face on startup
        if st.session_state.get('storage_manager'):
            loader.sync_from_hf(st.session_state.storage_manager)
        
        predictions = loader.load_predictions()
        master_data = loader.load_master_data()
        return {
            'predictions': predictions,
            'master_data': master_data,
            'loader': loader,
        }
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None


def load_css():
    css_file = Path(__file__).parent / 'assets' / 'custom.css'
    if css_file.exists():
        with open(css_file) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


def inject_startup_overlay():
    """
    Inject a premium full-screen loading overlay.
    """
    import base64
    try:
        with open("assets/logo_ojk.png", "rb") as f:
            logo_b64 = base64.b64encode(f.read()).decode()
        logo_tag = f'<img src="data:image/png;base64,{logo_b64}" class="loader-logo" alt="OJK">'
    except Exception:
        logo_tag = '<div class="loader-logo-fallback">OJK</div>'

    st.markdown(textwrap.dedent(f"""
    <div id="ews-loader-overlay">
    <div class="loader-content">
    <div class="loader-visual-wrap">
    <div class="loader-ring"></div>
    <div class="loader-logo-wrap">
    {logo_tag}
    </div>
    </div>
    <div class="loader-info">
    <div class="loader-title">Early Warning System</div>
    <div class="loader-subtitle">DPKP - OJK</div>
    <div class="loader-progress-track">
    <div class="loader-progress-bar"></div>
    </div>
    </div>
    </div>
    </div>
    """).strip(), unsafe_allow_html=True)


def inject_overlay_dismiss():
    """
    Dismiss startup overlay.
    """
    # JS for dismissal only
    st.markdown(textwrap.dedent("""
    <script>
    (function(doc){
    // Dismiss startup overlay
    var loader = doc.getElementById('ews-loader-overlay');
    if (loader) {
    setTimeout(function() {
    loader.classList.add('hidden');
    setTimeout(function() { loader.style.display = 'none'; }, 600);
    }, 400);
    }
    })(window.parent.document);
    </script>
    """).strip(), unsafe_allow_html=True)


# Load CSS and show overlay before loading data
load_css()
if not st.session_state.get('data_loaded', False):
    inject_startup_overlay()

# Inject persistent nav loader elements on every run
inject_nav_loader_elements()

# Load data on first run or after retraining
if not st.session_state.data_loaded or st.session_state.get('force_reload', False):
    # Clear ALL caches when force_reload is set
    if st.session_state.get('force_reload', False):
        st.cache_data.clear()
        st.cache_resource.clear()
        # Also clear the cached page imports to force fresh module load
        _import_page.clear()
        st.session_state.force_reload = False
        logger.info("Force reload triggered - all caches cleared")

    app_data = load_application_data()

    if app_data:
        st.session_state.predictions = app_data['predictions']
        st.session_state.master_data = app_data['master_data']
        st.session_state.loader = app_data['loader']
        st.session_state.data_loaded = True

        # Cache metadata for lightweight health check
        preds = app_data['predictions']
        if not preds.empty:
            st.session_state.last_update = preds['periode'].max().strftime("%Y-%m-%d")
            for col in ['kode_bank', 'nama_bank', 'bank']:
                if col in preds.columns:
                    st.session_state.total_banks = preds[col].nunique()
                    break
        inject_overlay_dismiss()
    else:
        # Dismiss overlay on failure to show error state
        inject_overlay_dismiss()


# =============================================================================
# CUSTOM CSS + STARTUP OVERLAY
# =============================================================================




# =============================================================================
# SIDEBAR - NAVIGATION & INFO
# =============================================================================

with st.sidebar:
    logo_path = "assets/logo_ojk.png"
    try:
        st.image(logo_path, width=200)
    except Exception:
        st.title("🏦 OJK - DPKP")

    # Lazy import option_menu only when sidebar renders
    from streamlit_option_menu import option_menu

    selected = option_menu(
        menu_title="",
        options=[
            "Dashboard",
            "Detail Bank",
            "Prediksi",
            "SHAP Analysis",
            "Monitoring",
            "Retraining",
            "Model Management",
        ],
        icons=[
            "speedometer2",
            "bank",
            "graph-up-arrow",
            "lightbulb",
            "activity",
            "gear",
            "cpu",
        ],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {
                "padding": "0!important",
                "background-color": "transparent",
                "border-radius": "10px",
                "overflow": "hidden",
            },
            "icon": {"font-size": "20px"},
            "nav-link": {
                "font-size": "16px",
                "text-align": "left",
                "margin": "4px",
                "border-radius": "8px",
                "transition": "all 0.3s ease",
                "--hover-color": "#fee2e2",
            },
            "nav-link-selected": {
                "background-color": "#b91c1c",
                "color": "white",
                "font-weight": "bold",
                "box-shadow": "0 4px 8px rgba(185, 28, 28, 0.2)",
            },
        },
    )

    # Show nav loader on page change (not on initial load)
    if st.session_state.get('_last_page') is not None and selected != st.session_state.get('_last_page'):
        show_nav_loader(destination_label=selected)

    with st.expander("ℹ️ System Info"):
        health = check_system_health()
        st.metric("Data Status", "✅ Ready" if health['data_loaded'] else "⚠️ Not Loaded")

        if health['models_loaded']:
            models_label = "✅ Active"
            models_help = "Real ML models are loaded and operational in memory."
        elif health['models_available']:
            models_label = "⏳ Ready"
            models_help = "Model files found. Will be loaded on demand to save memory."
        else:
            models_label = "⚠️ Missing"
            models_help = health.get('load_error') or "Model files not found in workspace."
            
        st.metric("Models Status", models_label, help=models_help)

        if health.get('load_error'):
            st.error(f"Models failed to load: {health['load_error'][:50]}...")
            if st.button("Retry Loading Models"):
                st.cache_resource.clear()
                st.rerun()

        st.metric("Last Update", health['last_update'])

    st.warning(
        "**INTERNAL USE ONLY**  \n"
        "Data bersifat rahasia (Sangat Rahasia). Dilarang menyebarluaskan "
        "screenshot atau data dari sistem ini tanpa izin tertulis dari DPKP OJK."
    )




# Map menu labels → module names + display names for spinner
_PAGE_MAP = {
    "Dashboard":        ("dashboard",         "Dashboard"),
    "Detail Bank":      ("detail_bank",        "Detail Bank"),
    "Prediksi":         ("prediksi",           "Prediksi"),
    "SHAP Analysis":    ("shap_analysis",      "SHAP Analysis"),
    "Monitoring":       ("monitoring",         "Monitoring"),
    "Retraining":       ("retraining",         "Retraining"),
    "Model Management": ("model_management",   "Model Management"),
}

# =============================================================================
# MAIN CONTENT - ROUTE TO PAGES
# =============================================================================

def main():
    module_name, display_name = _PAGE_MAP.get(selected, ("dashboard", "Dashboard"))

    # Show toast on first successful load (once)
    if not st.session_state.get('_welcomed'):
        st.session_state._welcomed = True
        st.toast("✅ Sistem Siap: Data & Model berhasil dimuat!", icon="🚀")

    # Detect navigation (page change vs. re-render of same page)
    is_navigating = (
        st.session_state.get('_last_page') is not None
        and selected != st.session_state.get('_last_page')
    )

    # Track page for session context
    st.session_state._last_page = selected

    # Wrap content in fade-in container
    st.markdown('<div class="main-content-fade">', unsafe_allow_html=True)

    # Import (cached after first call) and render
    page_mod = _import_page(module_name)

    # Show page-level skeleton during navigation transitions.
    # Each page's show() clears its own internal skeletons; we clear the
    # nav-level skeleton immediately after show() returns so there's no
    # double-skeleton visible when both fire on the same render.
    if is_navigating:
        nav_sk = show_page_skeleton(module_name)
        page_mod.show(st.session_state)
        clear_page_skeleton(nav_sk)
    else:
        page_mod.show(st.session_state)

    # Footer
    try:
        import json
        from src.config import MODEL_DIR
        _app_ver = json.loads((MODEL_DIR / "model_metadata.json").read_text()).get("version", "N/A")
    except Exception:
        _app_ver = "N/A"
    st.markdown(textwrap.dedent(f"""
    <div class="app-footer">
    <p><strong>OJK - Early Warning System</strong> v{_app_ver}</p>
    <p>© 2026 Departemen Pengendalian Kualitas dan Pengembangan Pengawasan Perbankan (DPKP) - OJK</p>
    <p style="font-size:14px;">
    Powered by Machine Learning Models |
    Created By Niko Dwicahyo
    </p>
    </div>
    """).strip(), unsafe_allow_html=True)

    # Hide navigation loader after rendering completes
    hide_nav_loader()

    st.markdown('</div>', unsafe_allow_html=True)


# =============================================================================
# RUN APPLICATION
# =============================================================================

if __name__ == "__main__":
    logger.info(f"Application started - Page: {selected}")

    try:
        main()
    except Exception as e:
        st.error("❌ An unexpected application error occurred.")
        try:
            error_msg = str(e)
        except Exception:
            error_msg = "Could not convert exception to string"

        logger.error(f"Global application error: {error_msg}", exc_info=True)

        with st.expander("🔍 Technical Details (Support Only)"):
            st.info("Please provide the following info to the administrator:")
            st.code(f"Error Type: {type(e).__name__}\nMessage: {error_msg[:100]}")
            if st.checkbox("Show full traceback (caution: sensitive paths)"):
                st.exception(e)
