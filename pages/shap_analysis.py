"""
SHAP Explainability Page
Model Interpretation
"""

import streamlit as st
import pandas as pd
import numpy as np
from src.model_inference import get_shap_values
from src.data_service import DataService
from src.utils import show_page_skeleton, clear_page_skeleton, models_ready

def show(session_state):
    import matplotlib.pyplot as plt
    import shap

    data_service = DataService()

    from pages.cache_listener import check_and_handle_cache_invalidation
    if check_and_handle_cache_invalidation(data_service):
        st.rerun()

    st.markdown("""
        <div class="main-header">
            <h1>🔍 Model Explainability (SHAP Analysis)</h1>
            <p>Memahami kontribusi indikator keuangan terhadap skor risiko bank menggunakan SHAP Values.</p>
        </div>
    """, unsafe_allow_html=True)

    # Lazy model loading: show skeleton, trigger load, auto-rerun until ready
    if not session_state.get('models_loaded', False):
        sk = show_page_skeleton('shap_analysis')
        if not models_ready():
            st.rerun()
        clear_page_skeleton(sk)

    master_data = data_service.load_master_data()
    models = session_state.get('models', {})

    if master_data.empty or not models:
        st.warning("⚠️ Data atau Model belum dimuat.")
        return

    bank_list_dicts = data_service.get_bank_list()
    if not bank_list_dicts:
        st.error("❌ Data bank tidak tersedia.")
        return

    bank_options = {f"{b['kode_bank']} - {b['nama_bank']}": b['kode_bank'] for b in bank_list_dicts}

    col_sel1, col_sel2, col_sel3 = st.columns(3)
    with col_sel1:
        selected_label = st.selectbox("Pilih Bank:", options=list(bank_options.keys()))
    with col_sel2:
        period_type = st.radio("Tipe Periode:", ["Historis", "Forecast"], horizontal=True)
    with col_sel3:
        if period_type == "Historis":
            available_periods = data_service.get_available_periods()
            period_list = sorted(available_periods, reverse=True)
        else:
            period_list = data_service.get_forecast_periods()

        selected_period = st.selectbox(
            "Pilih Periode:",
            period_list,
            format_func=lambda x: x.strftime('%B %Y') if hasattr(x, 'strftime') else str(x)
        )

    if st.button("🔬 Analisis Faktor Prediksi", use_container_width=True):
        shap_skeleton = st.empty()
        shap_skeleton.markdown('<div class="skeleton skeleton-chart"></div>', unsafe_allow_html=True)

        try:
            bank_id = bank_options[selected_label]

            if period_type == "Forecast":
                from src.model_inference import get_shap_values_forecast
                shap_values, expected_value, X, is_simulated = get_shap_values_forecast(
                    bank_id, selected_period, models, master_data, data_service
                )
            else:
                shap_values, expected_value, X, is_simulated = get_shap_values(
                    bank_id, selected_period, models, master_data
                )

            shap_skeleton.empty()

            if shap_values is None:
                st.error("Gagal menghitung SHAP values. Data untuk bank/periode tersebut mungkin tidak lengkap.")
                return

            st.markdown("---")
            st.subheader(f"📊 Kontribusi Fitur: {selected_label}")

            if is_simulated:
                st.warning("⚠️ **Hasil Simulasi**: Model asli tidak tersedia for bank/periode ini. Menampilkan estimasi berdasarkan bobot pilar standar.")

            st.caption(f"Periode: {selected_period.strftime('%B %Y')} | Baseline RBBR Netral: {expected_value:.2f} | Sumber: Formula RBBR OJK")

            fig, ax = plt.subplots(figsize=(10, 8))

            s_vals = shap_values[0]
            if isinstance(s_vals, list):
                s_vals = np.array(s_vals)

            explanation = shap.Explanation(
                values=s_vals,
                base_values=expected_value,
                data=X.iloc[0].values,
                feature_names=X.columns.tolist()
            )

            shap.plots.waterfall(explanation, max_display=12, show=False)
            fig = plt.gcf()
            fig.set_size_inches(10, 8)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.markdown("---")
            st.subheader("📝 Penjelasan Hasil Analisis RBBR")

            feature_names  = X.columns.tolist()
            feature_values = X.iloc[0].values
            final_score    = expected_value + float(np.sum(s_vals))

            # Compute real RBBR pillar scores from actual data
            from src.rbbr_scoring import score_car, score_npl, score_ldr, score_roa, score_bopo, score_to_pk, pk_to_sinyal
            from src.config import RBBRConfig

            row_data = X.iloc[0]
            s_car_val  = score_car(row_data['CAR'])
            s_npl_val  = score_npl(row_data['NPL_gross'])
            s_ldr_val  = score_ldr(row_data['LDR'])
            s_roa_val  = score_roa(row_data['ROA'])
            s_bopo_val = score_bopo(row_data['BOPO'])

            W = RBBRConfig.WEIGHTS
            risk_score  = (s_npl_val + s_ldr_val + s_car_val) / 3.0
            rent_score  = (s_roa_val + s_bopo_val) / 2.0
            perm_score  = float(s_car_val)
            gcg_score   = 2.0

            pk_val  = score_to_pk(final_score)
            pk_label = RBBRConfig.PK_LABELS.get(pk_val, '-')
            sinyal  = pk_to_sinyal(pk_val)
            sinyal_emoji = {'hijau': '🟢', 'kuning': '🟡', 'merah': '🔴'}.get(sinyal, '⚪')

            # Feature display names
            FEAT_LABELS = {
                'CAR':       'CAR (Kecukupan Modal)',
                'NPL_gross': 'NPL Gross (Kredit Bermasalah)',
                'LDR':       'LDR (Rasio Pinjaman-Simpanan)',
                'ROA':       'ROA (Return on Assets)',
                'BOPO':      'BOPO (Efisiensi Operasional)',
            }
            PK_SCORE_LABELS = {1: 'Sangat Sehat', 2: 'Sehat', 3: 'Cukup Sehat', 4: 'Kurang Sehat', 5: 'Tidak Sehat'}

            st.markdown(
                f"Analisis ini menggunakan metode **RBBR (Risk-Based Bank Rating)** sesuai ketentuan OJK. "
                f"Setiap rasio keuangan dikonversi ke **skor 1–5** berdasarkan threshold OJK, lalu digabungkan "
                f"dengan bobot per pilar menjadi **Skor Komposit RBBR**."
            )

            # Pillar breakdown table
            st.markdown("#### Skor Per Pilar RBBR")
            pillar_data = {
                'Pilar': ['Profil Risiko (NPL, LDR, CAR)', 'GCG', 'Rentabilitas (ROA, BOPO)', 'Permodalan (CAR)'],
                'Bobot': [f"{W['risk_profile']:.0%}", f"{W['gcg']:.0%}", f"{W['rentabilitas']:.0%}", f"{W['permodalan']:.0%}"],
                'Skor Pilar': [f"{risk_score:.2f}", f"{gcg_score:.2f}", f"{rent_score:.2f}", f"{perm_score:.2f}"],
                'Kontribusi': [
                    f"{W['risk_profile'] * risk_score:.4f}",
                    f"{W['gcg'] * gcg_score:.4f}",
                    f"{W['rentabilitas'] * rent_score:.4f}",
                    f"{W['permodalan'] * perm_score:.4f}",
                ],
            }
            st.dataframe(pd.DataFrame(pillar_data), use_container_width=True, hide_index=True)

            # Individual indicator scores
            st.markdown("#### Skor Indikator Individual")
            ind_rows = [
                ('CAR',       row_data['CAR'],       s_car_val,  'Profil Risiko + Permodalan'),
                ('NPL_gross', row_data['NPL_gross'],  s_npl_val,  'Profil Risiko'),
                ('LDR',       row_data['LDR'],        s_ldr_val,  'Profil Risiko'),
                ('ROA',       row_data['ROA'],        s_roa_val,  'Rentabilitas'),
                ('BOPO',      row_data['BOPO'],       s_bopo_val, 'Rentabilitas'),
            ]
            ind_data = {
                'Indikator': [FEAT_LABELS[r[0]] for r in ind_rows],
                'Nilai Aktual': [f"{r[1]:.2f}" for r in ind_rows],
                'Skor OJK (1–5)': [str(r[2]) for r in ind_rows],
                'Kategori': [PK_SCORE_LABELS[r[2]] for r in ind_rows],
                'Pilar': [r[3] for r in ind_rows],
                'Kontribusi SHAP': [f"{s_vals[i]:+.4f}" for i, r in enumerate(ind_rows)],
            }
            st.dataframe(pd.DataFrame(ind_data), use_container_width=True, hide_index=True)

            # Summary
            st.info(
                f"**Skor Dasar (Baseline):** `{expected_value:.2f}` — titik netral formula RBBR "
                f"(semua skor indikator = 3, GCG = 2.0)  \n"
                f"**Total Kontribusi Fitur:** `{np.sum(s_vals):+.4f}`  \n"
                f"**Skor RBBR Akhir:** `{final_score:.4f}` → **PK {pk_val} ({pk_label})** {sinyal_emoji}"
            )

            # Protective vs risk factors
            col_exp1, col_exp2 = st.columns(2)
            feat_names_display = [FEAT_LABELS[f] for f in feature_names]

            with col_exp1:
                st.success("🛡️ **Faktor Pelindung (Menurunkan Risiko)**")
                neg_factors = [(i, s_vals[i]) for i in range(len(s_vals)) if s_vals[i] < -0.001]
                neg_factors.sort(key=lambda x: x[1])
                if not neg_factors:
                    st.write("*Tidak ada faktor pelindung yang signifikan.*")
                else:
                    for i, sv in neg_factors:
                        st.write(
                            f"- **{feat_names_display[i]}**: nilai `{feature_values[i]:.2f}` → "
                            f"skor OJK `{ind_rows[i][2]}` ({PK_SCORE_LABELS[ind_rows[i][2]]}) → "
                            f"kontribusi `{sv:+.4f}`"
                        )

            with col_exp2:
                st.error("⚠️ **Faktor Pemicu (Meningkatkan Risiko)**")
                pos_factors = [(i, s_vals[i]) for i in range(len(s_vals)) if s_vals[i] > 0.001]
                pos_factors.sort(key=lambda x: x[1], reverse=True)
                if not pos_factors:
                    st.write("*Tidak ada faktor pemicu risiko yang signifikan.*")
                else:
                    for i, sv in pos_factors:
                        st.write(
                            f"- **{feat_names_display[i]}**: nilai `{feature_values[i]:.2f}` → "
                            f"skor OJK `{ind_rows[i][2]}` ({PK_SCORE_LABELS[ind_rows[i][2]]}) → "
                            f"kontribusi `{sv:+.4f}`"
                        )

            st.markdown("---")
            st.markdown(
                f"**💡 Kesimpulan:** Berdasarkan formula RBBR OJK, Bank **{selected_label}** pada periode **{selected_period.strftime('%B %Y')}** memperoleh "
                f"**Skor Komposit {final_score:.4f}** yang setara dengan **Peringkat Komposit {pk_val} — {pk_label}** "
                f"{sinyal_emoji}. "
                f"Skor dasar formula adalah **{expected_value:.2f}** (kondisi netral semua indikator). "
                f"Deviasi dari skor dasar ditentukan oleh kinerja aktual tiap rasio: "
                f"skor OJK < 3 menurunkan risiko (kontribusi negatif), skor OJK > 3 meningkatkan risiko (kontribusi positif)."
            )

        except Exception as e:
            shap_skeleton.empty()
            st.error(f"Terjadi kesalahan saat analisis: {str(e)}")
            st.exception(e)
