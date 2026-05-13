"""
Detail Bank Page
In-depth analysis per bank
"""

import streamlit as st
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.visualization import plot_radar_rbbr, plot_trend_forecast
from src.utils import get_pk_badge_html, get_sinyal_badge_html, format_percentage, show_page_skeleton, clear_page_skeleton, loading_spinner
from src.data_service import DataService

def show(session_state):
    """Detail bank page"""
    
    # Load data using DataService
    data_service = DataService()
    
    # Check for cache invalidation and refresh if needed
    from pages.cache_listener import check_and_handle_cache_invalidation
    if check_and_handle_cache_invalidation(data_service):
        st.rerun()
    
    # Premium Header
    st.markdown("""
        <div class="main-header">
            <h1>🏦 Analisis Detail Bank</h1>
            <p>Informasi mendalam mengenai profil risiko dan kinerja keuangan bank.</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Skeleton placeholders before data loads
    sk = show_page_skeleton('detail_bank')

    # Load data using DataService
    with loading_spinner("Memuat data bank..."):
        master_data = data_service.load_master_data()
        predictions = data_service.load_predictions()

    clear_page_skeleton(sk)
    
    if master_data.empty:
        st.warning("⚠️ Data belum dimuat. Silakan refresh halaman.")
        return
    
    # Get latest historical period for splitting
    latest_hist_period = data_service.get_latest_historical_period()
    
    # Split predictions into historical and forecast
    from src.data_service import get_forecast_data, get_historical_data
    forecast_data = get_forecast_data(predictions, master_data)
    historical_predictions = get_historical_data(predictions, master_data)
    
    # Get bank list
    bank_list_dicts = data_service.get_bank_list()
    
    if not bank_list_dicts:
        st.warning("Data bank tidak tersedia")
        return
        
    bank_options = {f"{b['kode_bank']} - {b['nama_bank']}": b['kode_bank'] for b in bank_list_dicts}
    
    # Bank selection
    selected_label = st.selectbox("Pilih Bank:", options=list(bank_options.keys()))
    
    if not selected_label:
        return
        
    selected_kode_bank = bank_options[selected_label]
    selected_bank = selected_label.split(' - ')[1] if ' - ' in selected_label else selected_label # fallback
    
    # Load bank data
    bank_data = master_data[master_data['kode_bank'] == selected_kode_bank].copy()
    
    # Ensure periode is datetime
    if 'periode' in bank_data.columns:
        bank_data['periode'] = pd.to_datetime(bank_data['periode'], errors='coerce')
    
    bank_data = bank_data.sort_values('periode', ascending=False)
    
    if bank_data.empty:
        st.error(f"Data tidak ditemukan untuk bank {selected_bank}")
        return
    
    # Latest data
    latest = bank_data.iloc[0]
    kode_bank = latest.get('kode_bank', 'unknown')
    
    # ===== HEADER =====
    col_h1, col_h2, col_h3 = st.columns([2, 1, 1])
    
    with col_h1:
        bank_name = latest.get('nama_bank', latest.get('bank', 'Bank'))
        
        # Show latest historical period (use actual latest from master data)
        periode_str = ""
        if latest_hist_period:
            try:
                periode_str = f"Data per: {latest_hist_period.strftime('%B %Y')}"
            except:
                periode_str = f"Data per: {latest_hist_period}"
        elif 'periode' in latest and pd.notna(latest['periode']):
            try:
                periode_str = f"Data per: {pd.to_datetime(latest['periode']).strftime('%B %Y')}"
            except:
                periode_str = f"Data per: {latest['periode']}"
        
        # Show forecast coverage if available
        fc_bank_col = 'kode_bank' if 'kode_bank' in forecast_data.columns else ('nama_bank' if 'nama_bank' in forecast_data.columns else 'bank')
        fc_bank_val = selected_kode_bank if fc_bank_col == 'kode_bank' else selected_bank
        bank_forecast = forecast_data[forecast_data[fc_bank_col] == fc_bank_val] if not forecast_data.empty and fc_bank_col in forecast_data.columns else pd.DataFrame()
        
        forecast_str = ""
        if not bank_forecast.empty and 'periode' in bank_forecast.columns:
            fc_periods = sorted(bank_forecast['periode'].unique())
            if fc_periods:
                fc_start = pd.to_datetime(fc_periods[0]).strftime('%b %Y')
                fc_end = pd.to_datetime(fc_periods[-1]).strftime('%b %Y')
                forecast_str = f" | Proyeksi: {fc_start} - {fc_end}"
                
        st.subheader(f"{bank_name}")
        st.caption(f"{periode_str}{forecast_str} | Kategori: {latest.get('kategori_bank', 'N/A')}")
    
    with col_h2:
        pk = int(latest.get('pk_computed', latest.get('pk_prediksi', 3)))
        st.markdown(get_pk_badge_html(pk), unsafe_allow_html=True)
    
    with col_h3:
        sinyal = latest.get('sinyal_ews', latest.get('sinyal', 'kuning'))
        st.markdown(get_sinyal_badge_html(sinyal), unsafe_allow_html=True)
    
    # ===== TABS =====
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🎯 Radar RBBR", "📈 Proyeksi 6 Bulan", "📋 Data Historis"])
    
    with tab1:
        st.subheader("Rasio Kunci")
        
        # Row 1
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.metric("CAR", format_percentage(latest.get('CAR', 0)))
        with col_m2:
            st.metric("NPL Gross", format_percentage(latest.get('NPL_gross', 0)))
        with col_m3:
            st.metric("NPL Net", format_percentage(latest.get('NPL_net', 0)))
        with col_m4:
            st.metric("ROA", format_percentage(latest.get('ROA', 0)))
        st.write("")
        col_m5, col_m6, col_m7, col_m8 = st.columns(4)
        with col_m5:
            st.metric("ROE", format_percentage(latest.get('ROE', 0)))
        with col_m6:
            st.metric("LDR", format_percentage(latest.get('LDR', 0)))
        with col_m7:
            st.metric("NIM", format_percentage(latest.get('NIM', 0)))
        with col_m8:
            st.metric("BOPO", format_percentage(latest.get('BOPO', 0)))
    
    with tab2:
        st.subheader("RBBR 4 Pilar Assessment")
        
        fig_radar = plot_radar_rbbr(latest)
        st.plotly_chart(fig_radar, use_container_width=True)
    
    with tab3:
        st.subheader("Proyeksi 6 Bulan Ke Depan")
        
        # Get bank-specific forecast
        fc_bank_col = 'kode_bank' if 'kode_bank' in forecast_data.columns else ('nama_bank' if 'nama_bank' in forecast_data.columns else 'bank')
        fc_bank_val = selected_kode_bank if fc_bank_col == 'kode_bank' else selected_bank
        bank_forecast = forecast_data[forecast_data[fc_bank_col] == fc_bank_val].copy() if not forecast_data.empty and fc_bank_col in forecast_data.columns else pd.DataFrame()
        
        if bank_forecast.empty:
            st.info("📊 Proyeksi belum tersedia. Lakukan retraining untuk menghasilkan proyeksi terbaru.")
        else:
            # Sort by period
            bank_forecast = bank_forecast.sort_values('periode')
            
            # Show forecast metrics
            st.markdown("#### Proyeksi Rasio Kunci")
            
            # Key metrics trends
            metrics_to_plot = ['CAR', 'NPL_gross', 'NPL_net', 'ROA', 'ROE', 'LDR', 'NIM', 'BOPO']
            available_metrics = [m for m in metrics_to_plot if m in bank_data.columns]
            
            for metric in available_metrics:
                # Prepare all historical data in ascending order
                hist_data = bank_data[['periode', metric]].copy().sort_values('periode')
                
                # Check if forecast has this metric (with or without _pred suffix)
                has_pred = f'{metric}_pred' in bank_forecast.columns
                has_direct = metric in bank_forecast.columns
                
                if not (has_pred or has_direct):
                    continue  # Skip if metric not in forecast
                
                # Get last historical point for bridge
                last_hist = hist_data.iloc[-1]
                
                # Prepare forecast data
                fc_data = bank_forecast[['periode']].copy()
                
                # Use _pred column if available, otherwise use direct column
                if has_pred:
                    fc_data[f'{metric}_pred'] = bank_forecast[f'{metric}_pred']
                elif has_direct:
                    # Rename to _pred format for plot_trend_forecast
                    fc_data[f'{metric}_pred'] = bank_forecast[metric]
                
                # Add confidence intervals if available
                if f'{metric}_lower' in bank_forecast.columns:
                    fc_data[f'{metric}_lower'] = bank_forecast[f'{metric}_lower']
                if f'{metric}_upper' in bank_forecast.columns:
                    fc_data[f'{metric}_upper'] = bank_forecast[f'{metric}_upper']
                
                # Add bridge point to connect historical and forecast
                bridge = pd.DataFrame({
                    'periode': [last_hist['periode']],
                    f'{metric}_pred': [last_hist[metric]]
                })
                
                # Concatenate bridge with forecast
                full_fore = pd.concat([bridge, fc_data])
                if 'periode' in full_fore.columns:
                    full_fore['periode'] = pd.to_datetime(full_fore['periode'], errors='coerce')
                full_fore = full_fore.sort_values('periode')
                
                # Plot trend with forecast
                fig = plot_trend_forecast(hist_data, full_fore, metric=metric)
                st.plotly_chart(fig, use_container_width=True)
            
            # Show forecast table
            st.markdown("#### Tabel Proyeksi")
            display_fc_cols = ['periode', 'CAR', 'NPL_gross', 'NPL_net', 'ROA', 'ROE', 'LDR', 'NIM', 'BOPO', 'pk_prediksi', 'sinyal_ews']
            # Map to prediction columns if they exist
            fc_display_data = bank_forecast.copy()
            for col in ['CAR', 'NPL_gross', 'NPL_net', 'ROA', 'ROE', 'LDR', 'NIM', 'BOPO']:
                if f'{col}_pred' in fc_display_data.columns and col not in fc_display_data.columns:
                    fc_display_data[col] = fc_display_data[f'{col}_pred']
            
            available_fc_cols = [c for c in display_fc_cols if c in fc_display_data.columns]
            if available_fc_cols:
                st.dataframe(
                    fc_display_data[available_fc_cols],
                    hide_index=True,
                    use_container_width=True
                )
    
    with tab4:
        st.subheader(f"Data Historis ({len(bank_data)} bulan)")
        
        # Display table
        display_cols = ['periode', 'CAR', 'NPL_gross', 'NPL_net', 'ROA', 'ROE', 'NIM', 'BOPO', 'LDR', 'LCR', 'pk_computed', 'total_aset']
        available_cols = [c for c in display_cols if c in bank_data.columns]
        
        st.dataframe(
            bank_data[available_cols],
            hide_index=True,
            use_container_width=True
        )
        
        # Download button
        csv = bank_data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"bank_{kode_bank}_historical.csv",
            mime="text/csv"
        )