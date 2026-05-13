"""
Utility Functions
Helper functions untuk aplikasi
"""

import streamlit as st
import logging
import textwrap
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import sys
from contextlib import contextmanager

from .config import LOGS_DIR

# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(log_level='INFO'):
    """Setup logging configuration"""
    log_file = LOGS_DIR / f'app_{datetime.now().strftime("%Y%m%d")}.log'
    
    # Ensure stdout is UTF-8 for Windows
    import sys
    import io
    
    handlers = [
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
    
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    
    return logging.getLogger(__name__)

# =============================================================================
# FORMATTING FUNCTIONS
# =============================================================================

def format_currency(value, prefix='Rp'):
    """Format number as currency"""
    if pd.isna(value):
        return "N/A"
    
    if value >= 1e12:
        return f"{prefix} {value/1e12:.2f} T"
    elif value >= 1e9:
        return f"{prefix} {value/1e9:.2f} M"
    elif value >= 1e6:
        return f"{prefix} {value/1e6:.2f} jt"
    else:
        return f"{prefix} {value:,.0f}"

def format_percentage(value, decimals=2):
    """Format number as percentage"""
    if pd.isna(value):
        return "N/A"
    return f"{value:.{decimals}f}%"

def format_number(value, decimals=2):
    """Format number with thousand separators"""
    if pd.isna(value):
        return "N/A"
    return f"{value:,.{decimals}f}"


# =============================================================================
# BADGE HTML GENERATORS (Safe Implementation)
# =============================================================================

def get_pk_badge_html(pk: int) -> str:
    """Generate HTML badge for PK"""
    from .config import RBBRConfig
    
    # Ensure pk is an integer for safety
    try:
        pk_val = int(pk)
    except:
        pk_val = 3
        
    label = RBBRConfig.PK_LABELS.get(pk_val, f"PK {pk_val}")
    color = RBBRConfig.PK_COLORS.get(pk_val, "#6c757d")
    
    return f"""
    <span style="
        background-color: {color};
        color: white;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9em;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        display: inline-block;
    ">{label}</span>
    """

def get_sinyal_badge_html(sinyal: str) -> str:
    """Generate HTML badge for EWS signal"""
    from .config import RBBRConfig
    
    # Ensure sinyal is a clean string
    s_val = str(sinyal).lower().strip()
    
    emoji = {
        'hijau': '🟢',
        'kuning': '🟡',
        'merah': '🔴',
    }
    
    color = RBBRConfig.SINYAL_COLORS.get(s_val, "#6c757d")
    
    return f"""
    <span style="
        background-color: {color};
        color: white;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9em;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        display: inline-block;
    ">{emoji.get(s_val, '')} {s_val.upper()}</span>
    """

# =============================================================================
# CALCULATION HELPERS
# =============================================================================

def compute_change(current, previous):
    """Calculate percentage change"""
    if pd.isna(current) or pd.isna(previous) or previous == 0:
        return 0
    return ((current - previous) / abs(previous)) * 100

def get_trend_indicator(change):
    """Get trend indicator (↑ ↓ →)"""
    if change > 0.5:
        return "📈 ↑"
    elif change < -0.5:
        return "📉 ↓"
    else:
        return "→"

def get_sinyal_color(sinyal: str) -> str:
    """Get hex color for EWS signal"""
    from .config import RBBRConfig
    return RBBRConfig.SINYAL_COLORS.get(sinyal.lower(), "#6c757d")

# =============================================================================
# SYSTEM HEALTH CHECK
# =============================================================================

def check_system_health():
    """Check system health status"""
    from .data_loader import get_data_loader
    
    # Try to get status from session state first for maximum speed
    if st.session_state.get('data_loaded', False) and 'last_update' in st.session_state:
        models_loaded = st.session_state.get('models_loaded', False)
        return {
            'data_loaded': True,
            'models_available': True,
            'models_loaded': models_loaded,
            'last_update': st.session_state.get('last_update', 'Unknown'),
            'total_banks': st.session_state.get('total_banks', 0),
        }

    loader = get_data_loader()
    
    try:
        # If not in session state, we still check predictions but loader.load_predictions is cached
        predictions = loader.load_predictions()
        model_status = loader.get_model_status()
        
        last_update = "Unknown"
        total_banks = 0
        if not predictions.empty:
            if 'periode' in predictions.columns:
                last_update = predictions['periode'].max().strftime("%Y-%m-%d")
            
            bank_id_col = None
            for col in ['kode_bank', 'nama_bank', 'bank']:
                if col in predictions.columns:
                    bank_id_col = col
                    break
            if bank_id_col:
                total_banks = predictions[bank_id_col].nunique()
        
        # Determine if models are available (lightweight check)
        # If models are already in session_state, we prefer that status
        models_loaded = st.session_state.get('models_loaded', False)
        
        return {
            'data_loaded': not predictions.empty,
            'models_available': model_status.get('available', False),
            'models_loaded': models_loaded,
            'load_error': model_status.get('error'),
            'last_update': last_update,
            'total_banks': total_banks,
        }
    except Exception as e:
        return {
            'data_loaded': False,
            'models_loaded': False,
            'last_update': 'Error',
            'total_banks': 0,
        }

# =============================================================================
# DATA VALIDATION
# =============================================================================

def validate_data_quality(df: pd.DataFrame) -> dict:
    """Validate data quality and return report"""
    report = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_pct': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
        'duplicates': df.duplicated().sum(),
        'issues': []
    }
    
    # Check for high missing values
    high_missing = df.columns[df.isnull().sum() / len(df) > 0.5]
    if len(high_missing) > 0:
        report['issues'].append(f"High missing values in: {', '.join(high_missing)}")
    
    # Check for duplicates
    if report['duplicates'] > 0:
        report['issues'].append(f"{report['duplicates']} duplicate rows found")
    
    return report

# =============================================================================
# EXPORT HELPERS
# =============================================================================

def export_dataframe(df: pd.DataFrame, filename: str, format='csv'):
    """Export DataFrame to various formats"""
    if format.lower() == 'csv':
        return df.to_csv(index=False).encode('utf-8')
    elif format.lower() == 'excel':
        from io import BytesIO
        buffer = BytesIO()
        df.to_excel(buffer, index=False, engine='openpyxl')
        return buffer.getvalue()
    elif format.lower() == 'parquet':
        return df.to_parquet(index=False)
    else:
        raise ValueError(f"Unsupported format: {format}")

# =============================================================================
# SHARED LOADING HELPERS
# =============================================================================

@contextmanager
def loading_spinner(message="Loading...", subtitle=""):
    """
    Shared loader wrapper that shows spinner with status message and optional subtitle.
    """
    with st.spinner(message):
        if subtitle:
            st.caption(subtitle)
        yield

def show_skeleton(variant="metric", count=1):
    """
    Renders skeleton placeholder HTML using Phase 1 CSS classes.
    Variants: "metric", "chart", "row"
    Returns an st.empty() placeholder.
    """
    html = ""
    for _ in range(count):
        html += f'<div class="skeleton skeleton-{variant}"></div>'
        if variant == "row":
            html += '<div style="height: 8px"></div>'
    
    placeholder = st.empty()
    placeholder.markdown(html, unsafe_allow_html=True)
    return placeholder

def show_operation_progress(title, steps=None):
    """
    Creates a standardized multi-step progress UI.
    Returns a dict of st.empty() handles.
    """
    st.markdown(f"### {title}")
    
    container = st.container()
    with container:
        handles = {
            "bar": st.progress(0.0),
            "status": st.empty(),
            "time": st.empty(),
            "detail": st.empty()
        }
        
        if steps:
            st.markdown("#### 📋 Status Detail")
            handles["step_cols"] = st.columns(len(steps))
            handles["steps"] = {}
            for i, step in enumerate(steps):
                handles["steps"][step] = handles["step_cols"][i].empty()
                handles["steps"][step].markdown(f"⏳ {step}")
                
    return handles

def show_page_skeleton(page_name: str) -> dict:
    """
    Render a page-level skeleton layout matching the given page structure.
    Returns dict of st.empty() handles keyed by slot name.
    Call handle.empty() on each to dismiss when real content is ready.
    """
    handles = {}

    # Header skeleton — all pages
    handles['header'] = st.empty()
    handles['header'].markdown(
        '<div class="skeleton skeleton-title" style="height:80px;border-radius:12px;margin-bottom:1.5rem"></div>',
        unsafe_allow_html=True,
    )

    if page_name in ('dashboard',):
        cols = st.columns(4)
        handles['metrics'] = [c.empty() for c in cols]
        for h in handles['metrics']:
            h.markdown('<div class="skeleton skeleton-metric"></div>', unsafe_allow_html=True)
        handles['heatmap'] = st.empty()
        handles['heatmap'].markdown('<div class="skeleton skeleton-chart" style="height:360px"></div>', unsafe_allow_html=True)
        handles['chart'] = st.empty()
        handles['chart'].markdown('<div class="skeleton skeleton-chart"></div>', unsafe_allow_html=True)

    elif page_name in ('prediksi', 'shap_analysis', 'shap'):
        cols = st.columns(3)
        handles['metrics'] = [c.empty() for c in cols]
        for h in handles['metrics']:
            h.markdown('<div class="skeleton skeleton-metric"></div>', unsafe_allow_html=True)
        handles['chart'] = st.empty()
        handles['chart'].markdown('<div class="skeleton skeleton-chart"></div>', unsafe_allow_html=True)

    elif page_name in ('monitoring',):
        cols = st.columns(3)
        handles['metrics'] = [c.empty() for c in cols]
        for h in handles['metrics']:
            h.markdown('<div class="skeleton skeleton-metric"></div>', unsafe_allow_html=True)
        handles['chart'] = st.empty()
        handles['chart'].markdown('<div class="skeleton skeleton-chart"></div>', unsafe_allow_html=True)

    elif page_name in ('detail_bank',):
        cols = st.columns(4)
        handles['metrics'] = [c.empty() for c in cols]
        for h in handles['metrics']:
            h.markdown('<div class="skeleton skeleton-metric"></div>', unsafe_allow_html=True)
        handles['chart'] = st.empty()
        handles['chart'].markdown('<div class="skeleton skeleton-chart"></div>', unsafe_allow_html=True)

    else:
        # Generic: header + chart
        handles['chart'] = st.empty()
        handles['chart'].markdown('<div class="skeleton skeleton-chart"></div>', unsafe_allow_html=True)

    return handles


def clear_page_skeleton(handles: dict) -> None:
    """Clear all skeleton placeholders returned by show_page_skeleton()."""
    for key, val in handles.items():
        if key == 'metrics' and isinstance(val, list):
            for h in val:
                h.empty()
        elif hasattr(val, 'empty'):
            val.empty()


def models_ready() -> bool:
    """
    Lazy model-load gate.
    Returns True immediately if models are already cached in session state.
    On first call (models not loaded), triggers load_models() via DataService
    (hits @st.cache_resource so subsequent calls are instant), updates session
    state, and returns True.  Never blocks — always returns bool synchronously.
    """
    if st.session_state.get('models_loaded', False):
        return True

    # Avoid double-triggering within the same run
    if st.session_state.get('_models_loading', False):
        return False

    st.session_state['_models_loading'] = True
    try:
        from src.data_service import DataService
        models = DataService().load_models()
        if models and 'load_error' not in models:
            st.session_state['models'] = models
            st.session_state['models_loaded'] = True
        else:
            # Models unavailable — don't loop forever
            st.session_state['models_loaded'] = True
    except Exception:
        st.session_state['models_loaded'] = True
    finally:
        st.session_state['_models_loading'] = False

    return True


def show_nav_loader(destination_label=""):
    """
    Show the navigation loader and set destination text.
    """
    label_text = f"Memuat {destination_label}..." if destination_label else "Mohon Tunggu..."
    
    st.markdown(textwrap.dedent(f"""
    <script>
    (function(){{
    var bar = window.parent.document.getElementById('ews-nav-bar');
    var spinner = window.parent.document.getElementById('ews-nav-spinner');
    var label = window.parent.document.getElementById('ews-nav-label');
    if (bar) bar.classList.add('active');
    if (spinner) spinner.classList.add('active');
    if (label) {{
        label.innerText = '{label_text}';
        label.classList.add('active');
    }}
    }})();
    </script>
    """).strip(), unsafe_allow_html=True)

def hide_nav_loader():
    """Hide the navigation loader by deactivating elements"""
    st.markdown(textwrap.dedent("""
    <script>
    (function(){
    var bar = window.parent.document.getElementById('ews-nav-bar');
    var spinner = window.parent.document.getElementById('ews-nav-spinner');
    var label = window.parent.document.getElementById('ews-nav-label');
    if (bar) bar.classList.remove('active');
    if (spinner) spinner.classList.remove('active');
    if (label) label.classList.remove('active');
    })();
    </script>
    """).strip(), unsafe_allow_html=True)

def inject_nav_loader_elements():
    """Inject persistent navigation loader elements"""
    st.markdown(textwrap.dedent("""
    <div id="ews-nav-bar"></div>
    <div id="ews-nav-spinner">
    <div class="loader-ring"></div>
    </div>
    <div id="ews-nav-label">Mohon Tunggu...</div>
    """).strip(), unsafe_allow_html=True)
