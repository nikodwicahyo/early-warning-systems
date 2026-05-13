"""
Prediction Page - 6-Month Forecasting
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from src.visualization import plot_trend_forecast
from src.utils import get_pk_badge_html, get_sinyal_color, show_page_skeleton, clear_page_skeleton, models_ready
from src.data_service import DataService

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
            <h1>🔮 Prediksi & Proyeksi Kondisi Bank</h1>
            <p>Fitur <b>What-If Analysis</b> untuk simulasi risiko dan proyeksi kondisi bank 6 bulan ke depan.</p>
        </div>
    """, unsafe_allow_html=True)

    # Lazy model loading: show skeleton, trigger load, auto-rerun until ready
    if not session_state.get('models_loaded', False):
        sk = show_page_skeleton('prediksi')
        if not models_ready():
            st.rerun()
        clear_page_skeleton(sk)

    master_data = data_service.load_master_data()
    predictions_all = data_service.load_predictions()

    # models already loaded via models_ready(); retrieve from session state
    models = session_state.get('models', {})
    
    if master_data.empty or predictions_all.empty:
        st.warning("⚠️ Data belum dimuat. Silakan tunggu atau refresh halaman.")
        return
    
    # Use data_service to get forecast-only data
    from src.data_service import get_forecast_data
    predictions = get_forecast_data(predictions_all, master_data)

    # Bank Selection
    bank_list_dicts = data_service.get_bank_list()
    
    if not bank_list_dicts:
        st.error("❌ Tidak ada data bank tersedia.")
        return
        
    bank_options = {f"{b['kode_bank']} - {b['nama_bank']}": b['kode_bank'] for b in bank_list_dicts}
    selected_label = st.selectbox("Pilih Bank untuk Analisis:", options=list(bank_options.keys()), index=0)

    if selected_label:
        selected_kode_bank = bank_options[selected_label]
        
        # Filter data
        df_bank_hist = master_data[master_data['kode_bank'] == selected_kode_bank].copy()
        
        if 'kode_bank' in predictions.columns:
            df_bank_pred = predictions[predictions['kode_bank'] == selected_kode_bank].copy()
        else:
            bank_col = 'nama_bank' if 'nama_bank' in predictions.columns else 'bank'
            df_bank_pred = predictions[predictions[bank_col] == selected_label.split(' - ')[1]].copy()
            
        selected_bank = selected_kode_bank # Pass kode_bank to model_inference
        
        # Ensure periode is datetime
        if 'periode' in df_bank_hist.columns:
            df_bank_hist['periode'] = pd.to_datetime(df_bank_hist['periode'], errors='coerce')
        if 'periode' in df_bank_pred.columns:
            df_bank_pred['periode'] = pd.to_datetime(df_bank_pred['periode'], errors='coerce')
        
        df_bank_hist = df_bank_hist.sort_values('periode')
        df_bank_pred = df_bank_pred.sort_values('periode')
        
        if df_bank_hist.empty:
            st.warning(f"Data historis untuk {selected_bank} tidak ditemukan.")
            return
            
        latest_actual = df_bank_hist.iloc[-1]
        
        # Get forecast periods
        forecast_periods = data_service.get_forecast_periods()
        
        # Initialize Simulation Session State
        sim_metrics = ['CAR', 'NPL_gross', 'NPL_net', 'ROA', 'ROE', 'LDR', 'BOPO', 'NIM']
        if 'last_selected_bank' not in st.session_state or st.session_state.last_selected_bank != selected_bank:
            st.session_state.last_selected_bank = selected_bank
            for m in sim_metrics:
                st.session_state[f"sim_{m}"] = float(latest_actual.get(m, 0.0))
        
        # Calculation based on simulation values from session state
        # (We use session state directly to avoid instantiation order issues)
        sim_data = latest_actual.copy()
        for m in sim_metrics:
            sim_data[m] = st.session_state.get(f"sim_{m}", float(latest_actual.get(m, 0.0)))
        
        from src.rbbr_scoring import compute_rbbr_score, score_to_pk, pk_to_sinyal
        from src.utils import get_pk_badge_html
        
        sim_score = compute_rbbr_score(sim_data)
        sim_pk = score_to_pk(sim_score)
        sim_sinyal = pk_to_sinyal(sim_pk)

        # Simulation Result Area & Reset Logic (Must be before sliders to allow session_state modification)
        # Check if simulation is active
        is_simulating = False
        for m in sim_metrics:
            if abs(st.session_state.get(f"sim_{m}", 0.0) - float(latest_actual.get(m, 0.0))) > 0.001:
                is_simulating = True
                break
        
        if is_simulating:
            st.info("✨ **Mode Simulasi Aktif**: Rasio telah dimodifikasi. Model AI sedang menghitung proyeksi berdasarkan skenario ini.")
            if st.button("🔄 Reset ke Data Aktual"):
                for m in sim_metrics:
                    st.session_state[f"sim_{m}"] = float(latest_actual.get(m, 0.0))
                st.rerun() 
        else:
            st.success("✅ **Mode Standar**: Menggunakan data historis riil dan model AI yang telah dilatih.")

        res_col1, res_col2, res_col3 = st.columns(3)
        with res_col1:
            prev_score = latest_actual.get('rbbr_score', latest_actual.get('rbbr_final', 0))
            st.metric("Skor Komposit", f"{sim_score:.3f}", delta=f"{sim_score - prev_score:.3f}", delta_color="inverse")
        with res_col2:
            st.markdown(f"**Peringkat Komposit:**")
            st.markdown(get_pk_badge_html(sim_pk), unsafe_allow_html=True)
        with res_col3:
            st.markdown(f"**Sinyal EWS:**")
            from src.config import RBBRConfig
            color = RBBRConfig.SINYAL_COLORS.get(sim_sinyal, '#6c757d')
            st.markdown(f'<div style="background-color:{color}; color:white; padding:5px 10px; border-radius:10px; text-align:center; font-weight:bold;">{sim_sinyal.upper()}</div>', unsafe_allow_html=True)
        
        # Simulation Inputs (Sliders)
        st.markdown("#### ⚙️ Simulator (*What-If Analysis*)")
        col_sim1, col_sim2, col_sim3, col_sim4 = st.columns(4)
        
        with col_sim1:
            sim_car = st.slider("CAR (%)", 0.0, 50.0, key="sim_CAR", help="Rasio Permodalan")
            sim_npl = st.slider("NPL Gross (%)", 0.0, 25.0, key="sim_NPL_gross", help="Rasio Kredit Bermasalah")
            
        with col_sim2:
            sim_npl_nett = st.slider("NPL Net (%)", 0.0, 25.0, key="sim_NPL_net", help="Rasio Kredit Bermasalah Net")
            sim_roa = st.slider("ROA (%)", -5.0, 10.0, key="sim_ROA", help="Rasio Rentabilitas (Aset)")
 
        with col_sim3:
            sim_roe = st.slider("ROE (%)", -10.0, 30.0, key="sim_ROE", help="Rasio Rentabilitas (Ekuitas)")
            sim_ldr = st.slider("LDR (%)", 0.0, 150.0, key="sim_LDR", help="Rasio Likuiditas")
            
        with col_sim4:
            sim_bopo = st.slider("BOPO (%)", 40.0, 120.0, key="sim_BOPO", help="Efisiensi Operasional")
            sim_nim = st.slider("NIM (%)", 0.0, 15.0, key="sim_NIM", help="Net Interest Margin")
        
        # Default Forecast
        st.markdown("### 📊 Proyeksi 6 Bulan ke Depan (AI Models)")
        
        from src.model_inference import predict_bank_6m
        
        # Cache key to avoid re-running inference on every slider interaction
        pred_cache_key = f"pred_{selected_bank}_{'sim' if is_simulating else 'std'}"
        if is_simulating:
            sim_vals = tuple(st.session_state.get(f"sim_{m}", 0.0) for m in sim_metrics)
            pred_cache_key += f"_{hash(sim_vals)}"
        
        if pred_cache_key not in st.session_state:
            if is_simulating:
                sim_master_data = master_data.copy()
                idx = df_bank_hist.index[-1]
                for col in ['CAR', 'NPL_gross', 'NPL_net', 'ROA', 'ROE', 'LDR', 'BOPO', 'NIM', 'rbbr_score']:
                    if col in sim_master_data.columns:
                        if col == 'rbbr_score':
                            sim_master_data.at[idx, col] = sim_score
                        else:
                            sim_master_data.at[idx, col] = sim_data[col]
                st.session_state[pred_cache_key] = predict_bank_6m(selected_bank, sim_master_data, models, is_simulation=True)
            else:
                st.session_state[pred_cache_key] = predict_bank_6m(selected_bank, master_data, models, is_simulation=False)

        res = st.session_state[pred_cache_key]

        if is_simulating:
            st.caption("✨ **Mode Simulasi**: Proyeksi di bawah ini berdasarkan skenario *What-If* yang Anda atur pada slider.")
        else:
            st.caption("📈 **Mode Standar**: Proyeksi di bawah ini berdasarkan data historis riil dan model AI yang telah dilatih.")
        
        if res and 'error' not in res:
            if res.get('is_simulated'):
                st.warning("⚠️ **Proyeksi Estimasi**: Menggunakan proyeksi tren linear berdasarkan performa historis (Model AI Spesifik tidak tersedia).")
            
            metrics_to_plot = ['rbbr_score', 'CAR', 'NPL_gross', 'NPL_net', 'ROA', 'ROE', 'LDR', 'BOPO', 'NIM']
            inner_tabs = st.tabs(metrics_to_plot)
            
            # Simple wrapper to show forecast from result
            df_plot_pred = df_bank_pred.copy() if (df_bank_pred is not None and not df_bank_pred.empty) else pd.DataFrame()
            
            # Check if we have valid predictions in session state
            # IMPORTANT: If simulating, we ignore pre-loaded predictions and use fresh ones from predict_bank_6m
            has_valid_predictions = not df_plot_pred.empty
            
            if has_valid_predictions and not is_simulating:
                if 'periode' in df_plot_pred.columns:
                    df_plot_pred['periode'] = pd.to_datetime(df_plot_pred['periode'], errors='coerce')
                df_plot_pred = df_plot_pred.sort_values('periode')
            else:
                # Construct df_plot_pred from ML result
                f_dates = res.get('forecast_dates')
                projs = res.get('projections', {})
                if f_dates is not None and projs:
                    df_plot_pred = pd.DataFrame({'periode': f_dates})
                    for m, vals in projs.items():
                        df_plot_pred[f'{m}_pred'] = vals
                    # For detail table, also add metrics without _pred suffix for compatibility
                    for m, vals in projs.items():
                        df_plot_pred[m] = vals

            for i, metric in enumerate(metrics_to_plot):
                with inner_tabs[i]:
                    hist_metric = df_bank_hist[['periode', metric]].copy()
                    
                    # If simulating, update the LATEST historical point to show the SIMULATED value
                    if is_simulating:
                        hist_metric.iloc[-1, hist_metric.columns.get_loc(metric)] = sim_data[metric]
                    
                    last_hist = hist_metric.iloc[-1]
                    
                    # Determine prediction columns
                    pred_col = f'{metric}_pred' if f'{metric}_pred' in df_plot_pred.columns else metric
                    lower_col = f'{metric}_lower' if f'{metric}_lower' in df_plot_pred.columns else None
                    upper_col = f'{metric}_upper' if f'{metric}_upper' in df_plot_pred.columns else None
                    
                    if pred_col in df_plot_pred.columns:
                         fore_df = df_plot_pred[['periode', pred_col]].copy()
                         fore_df = fore_df.rename(columns={pred_col: f'{metric}_pred'})
                         if lower_col: fore_df[f'{metric}_lower'] = df_plot_pred[lower_col]
                         if upper_col: fore_df[f'{metric}_upper'] = df_plot_pred[upper_col]
                    else:
                         # Last resort: linear trend if not in df_bank_pred
                         y = hist_metric[metric].values
                         x = np.arange(len(y))
                         try:
                             slope, intercept = np.polyfit(x, y, 1)
                         except:
                             slope, intercept = 0, y[-1]
                         f_vals = [y[-1] + (slope * j) for j in range(1, 7)]
                         fore_df = pd.DataFrame({'periode': res.get('forecast_dates'), f'{metric}_pred': f_vals})
                    
                    bridge = pd.DataFrame({
                        'periode': [last_hist['periode']],
                        f'{metric}_pred': [last_hist[metric]]
                    })
                    
                    full_fore = pd.concat([bridge, fore_df])
                    if 'periode' in full_fore.columns:
                        full_fore['periode'] = pd.to_datetime(full_fore['periode'], errors='coerce')
                    full_fore = full_fore.sort_values('periode')
                    fig = plot_trend_forecast(hist_metric, full_fore, metric=metric)
                    if is_simulating:
                        fig.update_layout(
                            title=f'Simulasi Tren & Proyeksi {metric} (Based on What-If)',
                            xaxis_title='Periode (Simulated Latest)'
                        )
                    st.plotly_chart(fig, use_container_width=True)
                    
            # Detailed Table
            st.subheader("📋 Detail Proyeksi Bulanan")
            if not df_plot_pred.empty:
                # Clean up for display
                display_df = df_plot_pred.copy()
                
                if 'periode' in display_df.columns:
                    display_df['periode'] = pd.to_datetime(display_df['periode']).dt.strftime('%Y-%m')
                
                # Construct columns to display
                cols_to_show = ['periode']
                for m in metrics_to_plot:
                    if f'{m}_pred' in display_df.columns:
                        cols_to_show.append(f'{m}_pred')
                    elif m in display_df.columns:
                        cols_to_show.append(m)
                        
                clean_df = display_df[cols_to_show].copy()
                
                # Rename columns for cleaner display
                rename_dict = {'periode': 'Periode'}
                for m in metrics_to_plot:
                    if f'{m}_pred' in clean_df.columns:
                        rename_dict[f'{m}_pred'] = m
                clean_df = clean_df.rename(columns=rename_dict)
                
                # Add Peringkat & Sinyal if score is available
                if 'rbbr_score' in clean_df.columns:
                    from src.rbbr_scoring import score_to_pk, pk_to_sinyal
                    clean_df['Peringkat'] = clean_df['rbbr_score'].apply(score_to_pk)
                    clean_df['Sinyal'] = clean_df['Peringkat'].apply(pk_to_sinyal).str.upper()
                
                st.dataframe(clean_df, use_container_width=True)
        elif res and 'error' in res:
            st.error(f"⚠️ **Gagal memuat proyeksi**: {res['error']}")
            st.info("Pastikan data historis untuk bank ini tersedia di database.")
        else:
            st.info("ℹ️ Data proyeksi temporal tidak tersedia untuk bank ini. Menampilkan hasil perhitungan statis di atas.")
