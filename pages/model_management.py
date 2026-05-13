import streamlit as st
import os
import json
import psutil
import pandas as pd
from datetime import datetime
from pathlib import Path
import gc
from utils.hf_storage import HFStorageManager
from src.data_service import DataService
from src.utils import loading_spinner, show_page_skeleton, clear_page_skeleton

# Custom CSS for Premium Look - REMOVED: Styles now in assets/custom.css

@st.cache_resource
def get_storage_manager():
    """Initialize HF Storage Manager (cached)"""
    try:
        repo_id = st.secrets.get('SPACE_ID', os.getenv('SPACE_ID'))
        if not repo_id:
            return None
        return HFStorageManager(repo_id=repo_id)
    except Exception:
        return None

def get_system_metrics():
    """Get real-time system resource usage from the container (psutil).
    When deployed on HF Spaces this reads the actual container metrics."""
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    cpu = psutil.cpu_percent(interval=0.5)
    return {
        "cpu_percent": cpu,
        "mem_percent": mem.percent,
        "mem_used_gb": mem.used / (1024**3),
        "mem_total_gb": mem.total / (1024**3),
        "disk_free_gb": disk.free / (1024**3),
        "disk_used_gb": disk.used / (1024**3),
        "disk_total_gb": disk.total / (1024**3),
    }

@st.cache_data(ttl=300, show_spinner=False)
def get_hf_space_info() -> dict:
    """Fetch HF Space runtime info and repo storage size.
    Compatible with huggingface_hub v0.23.x.
    Uses space_info(files_metadata=True) — siblings carry per-file sizes."""
    result = {
        "connected": False,
        "hardware": "N/A",
        "stage": "N/A",
        "username": "N/A",
        "repo_size_mb": None,
        "error": None,
    }
    try:
        from huggingface_hub import HfApi
        repo_id = st.secrets.get('SPACE_ID') or os.getenv('SPACE_ID')
        token = st.secrets.get('HF_TOKEN') or os.getenv('HF_TOKEN')
        if not repo_id or not token:
            result["error"] = "SPACE_ID atau HF_TOKEN belum dikonfigurasi"
            return result

        api = HfApi(token=token)

        # 1. Space info with file sizes + runtime in one call
        info = api.space_info(repo_id=repo_id, files_metadata=True, token=token)

        # Storage: for LFS files use lfs.size (actual), else sibling.size (git blob)
        total_bytes = 0
        for s in (getattr(info, 'siblings', None) or []):
            lfs = getattr(s, 'lfs', None)
            if lfs:
                total_bytes += (getattr(lfs, 'size', 0) or 0)
            else:
                total_bytes += (s.size or 0)
        result["repo_size_mb"] = total_bytes / (1024 * 1024)

        # Runtime: hardware & stage from SpaceRuntime object
        runtime = getattr(info, 'runtime', None)
        if runtime:
            hw = getattr(runtime, 'hardware', None)
            result["hardware"] = str(hw) if hw else "cpu-basic"
            result["stage"] = str(getattr(runtime, 'stage', 'RUNNING'))
        else:
            result["hardware"] = "cpu-basic"
            result["stage"] = "UNKNOWN"

        # 2. Authenticated user
        try:
            result["username"] = api.whoami().get("name", "N/A")
        except Exception:
            pass

        result["connected"] = True
    except Exception as e:
        result["error"] = str(e)
    return result

def show_local_models():
    """Display information about local models currently in use"""
    st.subheader("🏠 Active Local Models")

    from src.config import MODEL_DIR
    metadata_path = MODEL_DIR / "model_metadata.json"
    if not metadata_path.exists():
        st.warning("⚠️ Local model metadata not found.")
        return

    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            
        last_trained = (metadata.get('last_trained') or metadata.get('last_train_date')
                        or metadata.get('trained_at') or 'N/A')
        if isinstance(last_trained, str) and 'T' in last_trained:
            last_trained = last_trained.split('T')[0]
        st.markdown(f"**Version:** `{metadata.get('version', 'N/A')}` | **Last Trained:** `{last_trained}`")
        
        models_data = []
        for model_id, info in metadata.get('models', {}).items():
            if isinstance(info, dict):
                file_name = info.get('file', '')
                description = info.get('description', '')
                file_path = Path('models') / file_name if file_name else None
            else:
                file_path = Path(str(info)) if info else None
                file_name = file_path.name if file_path else ''
                description = ''
            exists = file_path.exists() if file_path else False
            size_mb = file_path.stat().st_size / (1024*1024) if exists else 0
            models_data.append({
                "Model ID": model_id.upper(),
                "Description": description,
                "File": file_name,
                "Size (MB)": f"{size_mb:.1f}" if exists else "Missing",
                "Status": "✅ Ready" if exists else "❌ Missing",
            })
        df = pd.DataFrame(models_data)
        st.table(df)
        
    except Exception as e:
        st.error(f"Error loading local metadata: {str(e)}")

def show_cloud_versions():
    """Display model version history from local versions directory"""
    st.subheader("📦 Model Version History")
    
    from src.config import MODEL_DIR
    versions_dir = MODEL_DIR / 'versions'
    
    if not versions_dir.exists():
        st.info("💡 No version history available yet. Versions are created after retraining.")
        return
    
    # Get all version directories
    version_dirs = sorted(versions_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    
    if not version_dirs:
        st.info("💡 No versions found.")
        return
    
    # Load live metadata to identify the active version
    active_metadata = {}
    metadata_path = MODEL_DIR / "model_metadata.json"
    if metadata_path.exists():
        try:
            with open(metadata_path, 'r') as f:
                active_metadata = json.load(f)
        except Exception:
            pass

    # Show versions
    for idx, version_dir in enumerate(version_dirs[:12]):  # Show last 12 versions
        if not version_dir.is_dir():
            continue
        
        metadata_file = version_dir / 'model_metadata.json'
        if not metadata_file.exists():
            continue
        
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            is_staged = "_staged" in version_dir.name
            is_active = (
                not is_staged and 
                metadata.get('version') == active_metadata.get('version') and
                metadata.get('last_trained') == active_metadata.get('last_trained')
            )
            
            badge = ""
            if is_active:
                badge = " 🌟 (Active)"
            elif is_staged:
                badge = " 🛠️ (Staged)"
            
            version_name = version_dir.name
            
            with st.expander(f"📦 {version_name}{badge}", expanded=is_active):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Version", metadata.get('version', 'N/A'))
                with col2:
                    trained = metadata.get('last_trained', 'N/A')
                    if isinstance(trained, str) and 'T' in trained:
                        trained = trained.split('T')[0]
                    st.metric("Trained", trained)
                with col3:
                    new_periods = metadata.get('new_periods', [])
                    st.metric("New Periods", len(new_periods))
                
                # Performance metrics
                if metadata.get('performance'):
                    st.write("**Performance Metrics:**")
                    perf = metadata['performance']
                    
                    perf_data = []
                    if 'tft' in perf:
                        tft = perf['tft']
                        perf_data.append({
                            'Model': 'TFT',
                            'Metric': 'Val Loss',
                            'Value': f"{tft.get('val_loss', 0):.4f}"
                        })
                    
                    if 'lightgbm' in perf and 'metrics' in perf['lightgbm']:
                        for target, mae in list(perf['lightgbm']['metrics'].items())[:3]:
                            perf_data.append({
                                'Model': 'LightGBM',
                                'Metric': target,
                                'Value': f"{mae:.4f}"
                            })
                    
                    if perf_data:
                        st.table(pd.DataFrame(perf_data))
                
                # Models in this version
                if metadata.get('models'):
                    st.write("**Models:**")
                    model_list = []
                    for key, info in metadata['models'].items():
                        if isinstance(info, dict):
                            model_list.append(f"• {key}: {info.get('file', 'N/A')}")
                        else:
                            model_list.append(f"• {key}: {info}")
                    st.text('\n'.join(model_list[:5]))
                
                # Rollback button (only for non-active and non-staged)
                if not is_active and not is_staged:
                    # Fallback for streamlit < 1.32.0
                    popover_container = st.popover if hasattr(st, 'popover') else st.expander
                    with popover_container(f"🔄 Rollback to {version_name}", use_container_width=True):
                        st.markdown(f"**Konfirmasi Rollback**")
                        st.write(f"Apakah Anda yakin ingin mengembalikan model ke versi `{version_name}`?")
                        st.info("💡 Semua data model, metadata, dan prediksi saat ini akan digantikan dengan data dari versi ini.")
                        
                        if st.button(f"Ya, Konfirmasi Rollback", key=f"confirm_{version_name}", type="primary", use_container_width=True):
                            with loading_spinner("Rolling back model...", subtitle="Memulihkan versi model sebelumnya..."):
                                from scripts.auto_retraining_new import AutoRetrainer
                                retrainer = AutoRetrainer(storage_manager=get_storage_manager())
                                success, msg = retrainer.rollback_to_version(version_name)
                                if success:
                                    if msg:
                                        st.session_state['rollback_alert'] = {'type': 'warning', 'msg': msg}
                                    else:
                                        st.session_state['rollback_alert'] = {'type': 'success', 'msg': f"✅ Rolled back to {version_name}"}
                                    st.rerun()
                                else:
                                    st.error(f"❌ Rollback failed: {msg}")
        
        except Exception as e:
            st.warning(f"Could not load version {version_dir.name}: {str(e)}")

def show(session_state=None):
    """Main entry point for the page"""
    
    if 'rollback_alert' in st.session_state:
        alert = st.session_state.pop('rollback_alert')
        if alert['type'] == 'warning':
            st.warning(f"⚠️ {alert['msg']}")
        else:
            st.success(alert['msg'])
    
    # Header
    st.markdown("""
        <div class="main-header">
            <h1>🔧 Model Management</h1>
            <p>Monitor, analyze, and manage machine learning models and system resources.</p>
        </div>
    """, unsafe_allow_html=True)

    # ── Top System & HF Health Cards ─────────────────────────────────────────
    sk = show_page_skeleton('model_management')
    metrics = get_system_metrics()
    with loading_spinner("Fetching HuggingFace Space info…"):
        hf_info = get_hf_space_info()
    clear_page_skeleton(sk)

    now_str = datetime.now().strftime("%H:%M:%S")

    # ── Section 1: Local Container Resources ─────────────────────────────────
    st.markdown(
        f"**🖥️ Local Container Resources** "
        f"<span style='font-size:0.8em;color:gray;'>— data diperbarui {now_str}</span>",
        unsafe_allow_html=True,
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        cpu = metrics['cpu_percent']
        status_cpu = "🔴 High" if cpu > 80 else ("🟡 Moderate" if cpu > 50 else "🟢 Normal")
        st.metric(
            label="🖥️ CPU Usage (Local)",
            value=f"{cpu:.1f}%",
            delta=status_cpu,
            delta_color="off",
            help="CPU utilisation proses Python di container lokal/HF Space ini (psutil).",
        )
    with col2:
        ram = metrics['mem_percent']
        ram_used = metrics['mem_used_gb']
        ram_total = metrics['mem_total_gb']
        status_ram = "🔴 High" if ram > 85 else ("🟡 Moderate" if ram > 65 else "🟢 Normal")
        st.metric(
            label="💾 RAM Usage (Local)",
            value=f"{ram:.1f}%",
            delta=f"{ram_used:.1f} / {ram_total:.1f} GB  {status_ram}",
            delta_color="off",
            help="RAM container lokal/HF Space ini (psutil). HF free tier: 16 GB.",
        )
    with col3:
        disk_free = metrics['disk_free_gb']
        disk_total = metrics['disk_total_gb']
        disk_used_pct = (metrics['disk_used_gb'] / disk_total * 100) if disk_total > 0 else 0
        status_disk = "🔴 Low" if disk_free < 5 else ("🟡 Moderate" if disk_free < 20 else "🟢 OK")
        st.metric(
            label="💿 Local Disk Free",
            value=f"{disk_free:.1f} GB  {status_disk}",
            delta=f"{disk_used_pct:.1f}% used of {disk_total:.1f} GB",
            delta_color="off",
            help="Sisa ruang disk ephemeral container lokal/HF Space ini (psutil).",
        )

    st.divider()

    # ── Section 2: HuggingFace Space Info ────────────────────────────────────
    hf_label_suffix = (
        f"<span style='font-size:0.8em;color:gray;'>— data dari HF Hub API · cache 60 s</span>"
        if hf_info["connected"]
        else f"<span style='font-size:0.8em;color:red;'>— tidak terhubung ke HF Hub</span>"
    )
    st.markdown(
        f"**☁️ HuggingFace Space Info** {hf_label_suffix}",
        unsafe_allow_html=True,
    )
    col4, col5, col6 = st.columns(3)
    with col4:
        if hf_info["repo_size_mb"] is not None:
            used_mb = hf_info["repo_size_mb"]
            limit_mb = 1024.0
            pct = min(used_mb / limit_mb * 100, 100)
            status_storage = "🔴 Near limit" if pct > 85 else ("🟡 Moderate" if pct > 60 else "🟢 OK")
            st.metric(
                label="☁️ HF Repo Storage",
                value=f"{used_mb:.1f} MB",
                delta=f"{pct:.1f}% of 1 GB limit  {status_storage}",
                delta_color="off",
                help="Total ukuran semua file di HF Space repo (via HF Hub API list_repo_tree). Limit free tier: 1 GB.",
            )
        else:
            st.metric(
                label="☁️ HF Repo Storage",
                value="N/A",
                delta="Gagal fetch" if hf_info["connected"] else "Tidak terhubung",
                delta_color="off",
                help="Total ukuran file di HF Space repo. Pastikan SPACE_ID & HF_TOKEN dikonfigurasi.",
            )
    with col5:
        hw_raw = hf_info["hardware"]
        hw = hw_raw.replace("_", " ").title() if hw_raw not in ("N/A", "") else "N/A"
        st.metric(
            label="⚙️ Space Hardware (HF)",
            value=hw,
            delta=f"User: {hf_info['username']}",
            delta_color="off",
            help="Tier hardware HF Space (dari HF Hub API get_space_runtime). Free: cpu-basic (2 vCPU, 16 GB).",
        )
    with col6:
        stage = hf_info["stage"]
        stage_icon = "🟢" if stage in ("RUNNING", "RUNNING_BUILDING") else ("🔴" if stage in ("ERROR", "STOPPED") else "🟡")
        connected_str = "CONNECTED" if hf_info["connected"] else "DISCONNECTED"
        st.metric(
            label="🌐 HF Hub Status",
            value=connected_str,
            delta=f"{stage_icon} {stage}",
            delta_color="off",
            help="Konektivitas ke HF Hub dan status Space saat ini (dari HF Hub API).",
        )

    if hf_info.get("error") and not hf_info["connected"]:
        st.warning(f"⚠️ HuggingFace API error: {hf_info['error']}")

    st.divider()

    # Main Content
    left_col, right_col = st.columns([1, 1])
    
    with left_col:
        show_local_models()
        
    with right_col:
        show_cloud_versions()

    st.divider()
    
    # Bottom Actions
    st.subheader("🛠️ Maintenance Tools")

    # Show toast-style alerts from previous actions
    for key in ("_maint_success", "_maint_warning"):
        if key in st.session_state:
            msg = st.session_state.pop(key)
            if key == "_maint_success":
                st.success(msg)
            else:
                st.warning(msg)

    act_col1, act_col2, act_col3, act_col4 = st.columns(4)

    with act_col1:
        if st.button("🧹 Force Memory Cleanup", use_container_width=True,
                     help="Jalankan gc.collect() untuk membebaskan memori Python yang tidak terpakai."):
            gc.collect()
            mem_after = psutil.virtual_memory()
            st.session_state["_maint_success"] = (
                f"✅ Memory cleanup selesai. RAM saat ini: "
                f"{mem_after.used / (1024**3):.1f} / {mem_after.total / (1024**3):.1f} GB "
                f"({mem_after.percent:.1f}%)"
            )
            st.rerun()

    with act_col2:
        if st.button("🔄 Refresh Data Cache", use_container_width=True,
                     help="Hapus cache Streamlit (st.cache_data) agar data master, prediksi, dan makro di-reload ulang dari disk."):
            from src.data_loader import get_data_loader
            st.cache_data.clear()
            get_data_loader().reload_all()
            st.session_state.force_reload = True
            st.session_state["_maint_success"] = "✅ Data cache dihapus. Data akan di-reload dari disk pada akses berikutnya."
            st.rerun()

    with act_col3:
        if st.button("🛰️ Refresh HF Info", use_container_width=True,
                     help="Paksa fetch ulang info HF Space dari API (bypass cache 60 s). Data lokal (CPU/RAM/Disk) selalu real-time."):
            get_hf_space_info.clear()
            st.session_state["_maint_success"] = "✅ Cache HF info dihapus. Informasi Space akan diperbarui."
            st.rerun()

    with act_col4:
        # Read version from metadata, fallback to hardcoded
        from src.config import MODEL_DIR as _MODEL_DIR
        _meta_path = _MODEL_DIR / "model_metadata.json"
        try:
            with open(_meta_path) as _f:
                _ver = json.load(_f).get("version", "N/A")
        except Exception:
            _ver = "N/A"
        st.info(f"Model Version: **{_ver}**", icon="ℹ️")

if __name__ == "__main__":
    show()

