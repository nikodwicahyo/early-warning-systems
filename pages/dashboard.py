"""
Dashboard Page
Main overview of banking system health
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import traceback

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.visualization import plot_heatmap_pk, plot_pk_distribution
from src.utils import get_pk_badge_html, format_number, loading_spinner, show_skeleton, show_page_skeleton, clear_page_skeleton
from src.config import RBBRConfig
from src.data_service import DataService
from src.rbbr_scoring import batch_score_rbbr, pk_to_sinyal



def show(session_state):
    """Main dashboard page"""
    
    # Load data using centralized DataService
    data_service = DataService()
    
    # Check for cache invalidation and refresh if needed
    from pages.cache_listener import check_and_handle_cache_invalidation
    if check_and_handle_cache_invalidation(data_service):
        st.rerun()
    
    # Premium Header
    st.markdown("""
        <div class="main-header">
            <h1>🏠 Dashboard Early Warning System</h1>
            <p>Monitor kondisi kesehatan perbankan dengan dukungan Machine Learning Berbasis <b>RBBR</b>.</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Load data using centralized DataService
    try:
        # Pre-render full page skeleton: KPI + heatmap + chart
        st.subheader("📊 Ringkasan Kondisi Perbankan")
        kpi_cols = st.columns(4)
        kpi_placeholders = [col.empty() for col in kpi_cols]
        for ph in kpi_placeholders:
            ph.markdown('<div class="skeleton skeleton-metric"></div>', unsafe_allow_html=True)
        heatmap_placeholder = st.empty()
        heatmap_placeholder.markdown('<div class="skeleton skeleton-chart" style="height:360px;margin-top:1rem"></div>', unsafe_allow_html=True)
        chart_placeholder = st.empty()
        chart_placeholder.markdown('<div class="skeleton skeleton-chart" style="margin-top:0.5rem"></div>', unsafe_allow_html=True)

        with loading_spinner("Memuat dashboard..."):
            df_history = data_service.load_master_data()
            df_predictions_all = data_service.load_predictions()
            metadata = data_service.load_bank_metadata()
        
        # Dismiss heatmap + chart skeletons now that data is ready
        heatmap_placeholder.empty()
        chart_placeholder.empty()
        
        # Normalize metadata for join
        if not metadata.empty and 'nama_bank' in metadata.columns:
            metadata['nama_bank_norm'] = metadata['nama_bank'].astype(str).str.strip().str.upper()
        
        if df_predictions_all is None or df_predictions_all.empty:
            st.warning("⚠️ Data belum dimuat. Silakan refresh halaman.")
            return
            
        # ===== DATA PREPARATION FOR HEATMAP & KPI =====
        
        # Combine historical (from master) + forecast (from predictions)
        df_forecast = df_predictions_all[df_predictions_all['is_forecast'] == True].copy() if 'is_forecast' in df_predictions_all.columns else df_predictions_all.copy()
        
        # Get historical data from master_panel (already has pk_prediksi from RBBR scoring)
        hist_cols = ['kode_bank', 'periode', 'pk_prediksi']
        for col in ['nama_bank', 'bank', 'rbbr_score', 'sinyal_ews', 'skor_rbbr', 'sinyal']:
            if col in df_history.columns:
                hist_cols.append(col)
        
        df_historical = df_history[hist_cols].copy() if not df_history.empty else pd.DataFrame()
        
        # Ensure nama_bank column exists
        if 'bank' in df_historical.columns and 'nama_bank' not in df_historical.columns:
            df_historical = df_historical.rename(columns={'bank': 'nama_bank'})
        
        # Ensure forecast has required columns
        if 'pk_prediksi' not in df_forecast.columns and 'pk_pred' in df_forecast.columns:
            df_forecast['pk_prediksi'] = df_forecast['pk_pred']
        if 'nama_bank' not in df_forecast.columns and 'bank' in df_forecast.columns:
            df_forecast = df_forecast.rename(columns={'bank': 'nama_bank'})
        
        # Combine historical + forecast
        if not df_historical.empty:
            df_historical['is_forecast'] = False
        if not df_forecast.empty:
            df_forecast['is_forecast'] = True
        
        # Normalize bank names BEFORE concat for metadata join
        if not df_historical.empty and 'nama_bank' in df_historical.columns:
            df_historical['nama_bank_norm'] = df_historical['nama_bank'].astype(str).str.strip().str.upper()
        if not df_forecast.empty and 'nama_bank' in df_forecast.columns:
            df_forecast['nama_bank_norm'] = df_forecast['nama_bank'].astype(str).str.strip().str.upper()
        
        # Join metadata to BOTH historical and forecast before concat
        if not metadata.empty:
            meta_cols = ['kode_bank']
            if 'kategori_bank' in metadata.columns:
                meta_cols.append('kategori_bank')
            elif 'kategori_buku' in metadata.columns:
                meta_cols.append('kategori_buku')
            
            if len(meta_cols) > 1:
                if not df_historical.empty and 'kode_bank' in df_historical.columns:
                    df_historical = df_historical.merge(metadata[meta_cols], on='kode_bank', how='left')
                if not df_forecast.empty and 'kode_bank' in df_forecast.columns:
                    # Only merge if forecast doesn't already have the category column
                    has_cat = any(c in df_forecast.columns for c in ['kategori_bank', 'kategori_buku'])
                    if not has_cat:
                        df_forecast = df_forecast.merge(metadata[meta_cols], on='kode_bank', how='left')
        
        df_all_data = pd.concat([df_historical, df_forecast], ignore_index=True)
        
        # Calculate forecast start date as the first forecast period
        forecast_start_date = None
        if not df_historical.empty and not df_forecast.empty:
            latest_hist = df_historical['periode'].max()
            earliest_forecast = df_forecast['periode'].min()
            forecast_start_date = earliest_forecast
        elif not df_forecast.empty:
            forecast_start_date = df_forecast['periode'].min()
        
        # Use all data for heatmap
        df_final = df_all_data.copy()

        # 4. Sorting and Categorical setup
        # Use latest HISTORICAL period for KPI cards
        if not df_historical.empty and 'periode' in df_historical.columns:
            latest_hist_period = df_historical['periode'].max()
        elif 'periode' in df_final.columns:
            # Fallback: use period before forecast starts
            if forecast_start_date:
                historical_periods = df_final[df_final['periode'] < forecast_start_date]['periode']
                latest_hist_period = historical_periods.max() if not historical_periods.empty else df_final['periode'].min()
            else:
                latest_hist_period = df_final['periode'].max()
        else:
            latest_hist_period = pd.Timestamp.now()

        df_latest = df_final[df_final['periode'] == latest_hist_period].copy() \
            if 'periode' in df_final.columns else df_forecast.copy()
        
        # Fallback if df_latest is empty
        if df_latest.empty and 'periode' in df_final.columns:
            # Use the latest available period
            latest_available = df_final['periode'].max()
            df_latest = df_final[df_final['periode'] == latest_available].copy()
        
        if df_latest.empty:
            # Last resort: use any data from df_final
            df_latest = df_final.copy() if not df_final.empty else df_forecast.copy()

        # Rename history columns to match expected names
        hist_map = {'bank': 'nama_bank', 'pk_computed': 'pk_prediksi'}
        for old_col, new_col in hist_map.items():
            if old_col in df_latest.columns and new_col not in df_latest.columns:
                df_latest = df_latest.rename(columns={old_col: new_col})

        if not metadata.empty and 'nama_bank' in df_latest.columns:
            df_latest['nama_bank_norm'] = df_latest['nama_bank'].astype(str).str.strip().str.upper()

        latest_period = latest_hist_period

        # Ensure df_latest has metadata for KPI metrics
        if not metadata.empty and 'kode_bank' in df_latest.columns:
             df_latest = df_latest.merge(metadata, on='kode_bank', how='left', suffixes=('', '_meta'))
        
        # Ensure sinyal_ews and rbbr_score columns exist (data_loader should provide these)
        pk_col = next((c for c in ['pk_prediksi', 'pk_pred', 'pk_computed'] if c in df_latest.columns), None)
        
        # Ensure sinyal_ews exists and handle NaNs from concat
        if 'sinyal_ews' not in df_latest.columns:
            df_latest['sinyal_ews'] = pd.NA
        if 'sinyal' in df_latest.columns:
            df_latest['sinyal_ews'] = df_latest['sinyal_ews'].fillna(df_latest['sinyal'])
        if pk_col:
            missing_idx = df_latest['sinyal_ews'].isna()
            if missing_idx.any():
                df_latest.loc[missing_idx, 'sinyal_ews'] = df_latest.loc[missing_idx, pk_col].fillna(3).apply(pk_to_sinyal)
        df_latest['sinyal_ews'] = df_latest['sinyal_ews'].fillna('hijau')
        
        # Ensure rbbr_score exists and handle NaNs from concat
        if 'rbbr_score' not in df_latest.columns:
            df_latest['rbbr_score'] = pd.NA
        if 'rbbr_score_pred' in df_latest.columns:
            df_latest['rbbr_score'] = df_latest['rbbr_score'].fillna(df_latest['rbbr_score_pred'])
        if 'skor_rbbr' in df_latest.columns:
            df_latest['rbbr_score'] = df_latest['rbbr_score'].fillna(df_latest['skor_rbbr'])
        if pk_col:
            df_latest['rbbr_score'] = df_latest['rbbr_score'].fillna(df_latest[pk_col])
        df_latest['rbbr_score'] = df_latest['rbbr_score'].fillna(3.0)

        # Add bank_label for heatmap and display
        if 'kode_bank' in df_final.columns and 'nama_bank' in df_final.columns:
            df_final['bank_label'] = df_final['kode_bank'].astype(str) + ' - ' + df_final['nama_bank'].astype(str)
        else:
            df_final['bank_label'] = df_final.get('nama_bank', df_final.get('kode_bank', 'Unknown'))

        # Sort banks by latest HISTORICAL PK score
        latest_pk_scores = df_final[df_final['periode'] == latest_hist_period]
        if latest_pk_scores.empty:
            # fallback to any available period
            latest_pk_scores = df_final[df_final['periode'] == df_final['periode'].max()]
        if not latest_pk_scores.empty:
            # Higher PK = Higher Risk = Top of heatmap
            bank_risk = latest_pk_scores.groupby('bank_label')['pk_prediksi'].max().sort_values(ascending=False)
            sorted_banks = [b for b in bank_risk.index.tolist() if b is not None and str(b) != 'nan']
            
            # Add any banks not in latest_period to end
            all_banks = df_final['bank_label'].dropna().unique().tolist()
            for b in all_banks:
                if b not in sorted_banks:
                    sorted_banks.append(b)
            
            df_final['bank_label'] = pd.Categorical(df_final['bank_label'], categories=sorted_banks, ordered=True)
            if 'periode' in df_final.columns:
                df_final['periode'] = pd.to_datetime(df_final['periode'], errors='coerce')
            df_final = df_final.sort_values(['bank_label', 'periode'])
        
        df = df_final # Alias for following code
        

        # ===== KPI CARDS =====
        # Data is ready, clear skeletons and show real metrics
        for ph in kpi_placeholders:
            ph.empty()
        
        # Determine the best column to use for bank identification
        bank_id_col = None
        if 'kode_bank' in df_latest.columns:
            bank_id_col = 'kode_bank'
        elif 'nama_bank' in df_latest.columns:
            bank_id_col = 'nama_bank'
        elif 'bank' in df_latest.columns:
            bank_id_col = 'bank'
            
        if bank_id_col is None:
            st.error(f"❌ Kolom identitas bank tidak ditemukan di data. Kolom tersedia: {df_latest.columns.tolist()}")
            total_banks = 0
        else:
            total_banks = df_latest[bank_id_col].nunique()

        with kpi_placeholders[0]:
            st.metric("Total Bank", total_banks)
        
        with kpi_placeholders[1]:
            red_count = 0
            if 'sinyal_ews' in df_latest.columns:
                red_count = len(df_latest[df_latest['sinyal_ews'].astype(str).str.lower() == 'merah'])
            
            red_pct = (red_count / total_banks * 100) if total_banks > 0 else 0
            st.metric(
                "🔴 Sinyal Merah",
                red_count,
                delta=f"{red_pct:.1f}%",
                delta_color="inverse"
            )
        
        with kpi_placeholders[2]:
            yellow_count = 0
            if 'sinyal_ews' in df_latest.columns:
                yellow_count = len(df_latest[df_latest['sinyal_ews'].astype(str).str.lower() == 'kuning'])
            st.metric("🟡 Sinyal Kuning", yellow_count)
        
        with kpi_placeholders[3]:
            green_count = 0
            if 'sinyal_ews' in df_latest.columns:
                green_count = len(df_latest[df_latest['sinyal_ews'].astype(str).str.lower() == 'hijau'])
            st.metric("🟢 Sinyal Hijau", green_count)
        
        # ===== HEATMAP =====
        st.markdown("---")
        st.subheader("🗺️ Heatmap Kondisi Perbankan")
        
        # Filters - use kategori_bank if available (works for both historical and forecast)
        col_f1, col_f2 = st.columns([3, 1])
        with col_f1:
            cat_col = 'kategori_bank' if 'kategori_bank' in df.columns else next((c for c in ['kategori_buku', 'BUKU'] if c in df.columns), None)
            if cat_col:
                cat_options = sorted(df[cat_col].dropna().unique().tolist())
                if cat_options:
                    selected_cat = st.multiselect(f"Filter {cat_col}", cat_options, default=cat_options)
                else:
                    selected_cat = []
            else:
                selected_cat = []
        
        # Apply filter
        if selected_cat and cat_col:
            df_filtered = df[df[cat_col].isin(selected_cat)].copy()
        else:
            df_filtered = df.copy()
        
        if not df_filtered.empty:
            fig_heatmap = plot_heatmap_pk(df_filtered, "Heatmap Kondisi Perbankan", forecast_start_date)
            st.plotly_chart(fig_heatmap, use_container_width=True)
        else:
            st.warning("⚠️ Tidak ada data untuk ditampilkan")
        
        # ===== TWO COLUMNS =====
        st.markdown("---")
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.markdown("### 📊 Distribusi Peringkat Komposit")
            fig_dist = plot_pk_distribution(df_latest)
            st.plotly_chart(fig_dist, use_container_width=True, config={'displaylogo': False})
        
        with col_right:
            st.markdown("### 🚨 Top Bank Berisiko")
            
            pk_col = next((c for c in ['pk_prediksi', 'pk_pred', 'pk_computed'] if c in df_latest.columns), None)
            
            if pk_col and not df_latest.empty:
                sort_cols = [pk_col]
                if 'rbbr_score' in df_latest.columns:
                    sort_cols.append('rbbr_score')
                
                top_risk = df_latest.sort_values(by=sort_cols, ascending=False).head(10).copy()
                
                # Ensure required columns exist
                if 'rbbr_score' not in top_risk.columns:
                    if 'rbbr_score_pred' in top_risk.columns:
                        top_risk['rbbr_score'] = top_risk['rbbr_score_pred']
                    elif 'skor_rbbr' in top_risk.columns:
                        top_risk['rbbr_score'] = top_risk['skor_rbbr']
                    else:
                        top_risk['rbbr_score'] = top_risk[pk_col]
                
                if 'sinyal_ews' not in top_risk.columns:
                    if 'sinyal' in top_risk.columns:
                        top_risk['sinyal_ews'] = top_risk['sinyal']
                    else:
                        top_risk['sinyal_ews'] = top_risk[pk_col].apply(pk_to_sinyal)
                
                # Build clean display table
                display_data = {
                    'Bank': top_risk.get('nama_bank', top_risk.get('bank', pd.Series(['Unknown']*len(top_risk)))),
                    'PK': top_risk[pk_col].fillna(3).astype(int),
                    'Skor RBBR': top_risk['rbbr_score'].fillna(3.0).round(2),
                    'Sinyal': top_risk['sinyal_ews'].astype(str).str.title()
                }
                
                top_risk_display = pd.DataFrame(display_data).reset_index(drop=True)
                
                st.dataframe(
                    top_risk_display,
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        'PK': st.column_config.NumberColumn('PK', format='%d'),
                        'Skor RBBR': st.column_config.NumberColumn('Skor RBBR', format='%.2f')
                    }
                )
            else:
                st.info("Data peringkat tidak tersedia.")
        
        # ===== FOOTER INFO =====
        forecast_info = ""
        if forecast_start_date and 'periode' in df_forecast.columns and not df_forecast.empty:
            forecast_end = df_forecast['periode'].max()
            forecast_info = f" | 🔮 Proyeksi: {forecast_start_date.strftime('%B %Y')} - {forecast_end.strftime('%B %Y')}"
        else:
            forecast_info = " | ⚠️ Belum ada data forecast (silakan lakukan retraining)"
        
        st.caption(f"📅 Data historis terakhir: {latest_period.strftime('%B %Y')}{forecast_info} | 🔄 Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    except Exception as e:
        st.error(f"❌ Terjadi kesalahan pada Dashboard: {str(e)}")
        with st.expander("🔍 Detail Error"):
            st.code(traceback.format_exc())
            # Basic debug info
            cols_info = {}
            if 'df_forecast' in locals() and df_forecast is not None:
                cols_info['df_forecast'] = df_forecast.columns.tolist()
            if 'df_history' in locals() and df_history is not None:
                cols_info['df_history'] = df_history.columns.tolist()
            if 'metadata' in locals() and not metadata.empty:
                cols_info['metadata'] = metadata.columns.tolist()
            
            if cols_info:
                st.json(cols_info)
