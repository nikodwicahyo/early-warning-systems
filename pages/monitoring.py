"""
Monitoring Page - System-wide Trends & Alerts
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.visualization import plot_trend_forecast
from src.data_service import DataService
from src.utils import loading_spinner, show_page_skeleton, clear_page_skeleton

def show(session_state):
    # Load data using DataService
    data_service = DataService()
    
    # Check for cache invalidation and refresh if needed
    from pages.cache_listener import check_and_handle_cache_invalidation
    if check_and_handle_cache_invalidation(data_service):
        st.rerun()
    
    # Premium Header
    st.markdown("""
        <div class="main-header">
            <h1>📈 Monitoring Sistem & Tren Agregat</h1>
            <p>Pemantauan kondisi sistem perbankan secara makro dan tren agregat industri.</p>
        </div>
    """, unsafe_allow_html=True)

    # Skeleton placeholders before data loads
    sk = show_page_skeleton('monitoring')

    # Load data using DataService
    with loading_spinner("Memuat data monitoring..."):
        master_data = data_service.load_master_data()
        macro_data = data_service.load_macro_variables()

    clear_page_skeleton(sk)
    
    if master_data.empty:
        st.warning("⚠️ Data belum dimuat.")
        return

    # 1. Macroeconomic Indicators
    st.subheader("🌐 Indikator Makroekonomi")
    
    if not macro_data.empty:
        col1, col2, col3 = st.columns(3)
        
        # Ensure periode is datetime
        if 'periode' in macro_data.columns:
            macro_data['periode'] = pd.to_datetime(macro_data['periode'], errors='coerce')
        
        latest_macro = macro_data.sort_values('periode').iloc[-1]
        prev_macro = macro_data.sort_values('periode').iloc[-2] if len(macro_data) > 1 else latest_macro
        
        # Column mapping for macro variables
        cols = {
            'rate': 'bi_rate',
            'inflasi': 'inflasi',
            'kurs': 'kurs_usd'
        }
        
        with col1:
            val = latest_macro.get(cols['rate'], 0)
            delta = val - prev_macro.get(cols['rate'], val) if prev_macro is not None else 0
            st.metric("BI Rate", f"{val:.2f}%", delta=f"{delta:.2f}%", delta_color="inverse")
        with col2:
            val = latest_macro.get(cols['inflasi'], 0)
            delta = val - prev_macro.get(cols['inflasi'], val) if prev_macro is not None else 0
            st.metric("Inflasi (YoY)", f"{val:.2f}%", delta=f"{delta:.2f}%", delta_color="inverse")
        with col3:
            val = latest_macro.get(cols['kurs'], 0)
            delta = val - prev_macro.get(cols['kurs'], val) if prev_macro is not None else 0
            st.metric("Kurs USD/IDR", f"Rp {val:,.0f}", delta=f"{delta:,.0f}", delta_color="inverse")
            
        # Macro Chart
        macro_metrics = [v for v in cols.values() if v in macro_data.columns]
        fig_macro = None
        
        if macro_metrics:
            selected_macro = st.selectbox("Pilih Indikator Makro untuk Grafik:", macro_metrics)
            
            # Check if we have forecast data to add a vertical line
            forecast_start = None
            try:
                latest_hist = data_service.get_latest_historical_period()
                if latest_hist:
                    forecast_start = latest_hist.strftime('%Y-%m-%d')
            except:
                pass

            fig_macro = px.line(macro_data, x='periode', y=selected_macro, 
                              title=f"Tren {selected_macro.replace('_', ' ').upper()}",
                              template="plotly_white")
            
            if forecast_start:
                fig_macro.add_vline(x=forecast_start, line_dash="dash", line_color="red")
                # Add annotation separately
                fig_macro.add_annotation(
                    x=forecast_start,
                    y=1,
                    yref="paper",
                    text="Forecast Start",
                    showarrow=False,
                    xanchor="left",
                    font=dict(color="red")
                )
            
            st.plotly_chart(fig_macro, use_container_width=True)
        else:
            st.info("💡 Data historis makroekonomi tidak tersedia untuk ditampilkan dalam grafik.")

    st.markdown("---")
    
    # 2. System-wide Financial Metrics
    st.subheader("📊 Tren Agregat Perbankan")
    
    # Define targets to monitor
    targets = ['CAR', 'NPL_gross', 'NPL_net', 'ROA', 'ROE', 'NIM', 'BOPO', 'LDR']
    
    # Historical Aggregate
    # Only aggregate columns that exist
    available_hist = [c for c in targets if c in master_data.columns]
    if available_hist:
        agg_metrics_hist = master_data.groupby('periode').agg({c: 'mean' for c in available_hist}).reset_index()
    else:
        agg_metrics_hist = pd.DataFrame(columns=['periode'] + targets)
    agg_metrics_hist['type'] = 'Historis'
    
    # Forecast Aggregate
    predictions_all = data_service.load_predictions()
    
    # Use data_service to get forecast-only data
    from src.data_service import get_forecast_data
    predictions = get_forecast_data(predictions_all, master_data)
    
    agg_metrics_fc = pd.DataFrame()
    
    if not predictions.empty:
        # Check for columns, handling potential '_pred' suffix
        pred_cols = {}
        for t in targets:
            if t in predictions.columns:
                pred_cols[t] = t
            elif f"{t}_pred" in predictions.columns:
                pred_cols[f"{t}_pred"] = t
        
        if 'ALL_BANKS' in predictions['kode_bank'].values:
            agg_metrics_fc = predictions[predictions['kode_bank'] == 'ALL_BANKS'].copy()
            # Rename if necessary
            agg_metrics_fc = agg_metrics_fc.rename(columns={k: v for k, v in pred_cols.items() if k != v})
        elif pred_cols:
            agg_metrics_fc = predictions.groupby('periode').agg({k: 'mean' for k in pred_cols.keys()}).reset_index()
            # Rename to standard names
            agg_metrics_fc = agg_metrics_fc.rename(columns={k: v for k, v in pred_cols.items() if k != v})
            
        if not agg_metrics_fc.empty:
            agg_metrics_fc['type'] = 'Proyeksi'
        
    # Combine
    if not agg_metrics_fc.empty:
        full_agg = pd.concat([agg_metrics_hist, agg_metrics_fc], ignore_index=True)
        if 'periode' in full_agg.columns:
            full_agg['periode'] = pd.to_datetime(full_agg['periode'], errors='coerce')
        full_agg = full_agg.sort_values('periode')
    else:
        full_agg = agg_metrics_hist.copy()
    
    col_sel1, col_sel2 = st.columns([1, 3])
    with col_sel1:
        metric_agg = st.radio("Pilih Rasio:", ['CAR', 'NPL_gross', 'NPL_net', 'ROA', 'ROE', 'NIM', 'BOPO', 'LDR'])
    
    with col_sel2:
        fig_agg = px.line(full_agg, x='periode', y=metric_agg, color='type', 
                          color_discrete_map={'Historis': '#1f77b4', 'Proyeksi': '#ff7f0e'},
                          title=f"Tren Agregat {metric_agg} Perbankan")
        
        # Add transition line
        if not agg_metrics_fc.empty:
            fc_start = agg_metrics_fc['periode'].min()
            fig_agg.add_vline(x=fc_start, line_dash="dash", line_color="gray")
            
        st.plotly_chart(fig_agg, use_container_width=True)

    st.markdown("---")
    
    # 3. Alert History
    st.subheader("🚨 Alert & Perubahan Status")
    
    # Identify banks with high risk (NPL > 5% or CAR < 12% or PK >= 4)
    # Use .get() or check columns to avoid KeyError
    mask = pd.Series([False] * len(master_data))
    if 'NPL_gross' in master_data.columns:
        mask |= (master_data['NPL_gross'] > 5.0)
    if 'CAR' in master_data.columns:
        mask |= (master_data['CAR'] < 12.0)
    if 'pk_computed' in master_data.columns:
        mask |= (master_data['pk_computed'] >= 4)
    
    risky_banks = master_data[mask].copy()
    if not risky_banks.empty:
        sort_cols = [c for c in ['periode', 'nama_bank'] if c in risky_banks.columns]
        if sort_cols:
            risky_banks = risky_banks.sort_values(sort_cols, ascending=[False, True])
    
    if not risky_banks.empty:
        st.warning(f"Ditemukan {len(risky_banks[risky_banks['periode'] == risky_banks['periode'].max()])} bank dengan indikator di luar threshold normal pada periode terakhir.")
        
        display_risky = risky_banks[risky_banks['periode'] == risky_banks['periode'].max()][
            ['nama_bank', 'kode_bank', 'kategori_bank', 'CAR', 'NPL_gross', 'LDR', 'ROA', 'pk_computed']
        ]
        st.dataframe(display_risky.style.format({
            'CAR': '{:.2f}%',
            'NPL_gross': '{:.2f}%',
            'LDR': '{:.2f}%',
            'ROA': '{:.2f}%',
            'pk_computed': '{:.0f}'
        }), use_container_width=True)
    else:
        st.success("✅ Tidak ada bank yang melewati threshold kritis pada periode terakhir.")
