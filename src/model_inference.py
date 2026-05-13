"""
Model Inference Module
Handles real-time prediction using trained models (LightGBM, etc.)
"""

import pandas as pd
import numpy as np
import pickle
import logging
from pathlib import Path
from typing import Dict, Optional, List, Tuple
import streamlit as st

# Import config using absolute path
try:
    from src.config import MODEL_DIR
except ImportError:
    from .config import MODEL_DIR

logger = logging.getLogger(__name__)

def predict_bank_6m(kode_bank: str, df: pd.DataFrame, models: Dict, is_simulation: bool = True) -> Dict:
    """
    Predict bank financial metrics for the next 6 months using trained ML models.
    Supports both Standard Mode (Pre-trained) and Simulation Mode (Iterative).
    """
    try:
        # 1. Robust Data Selection
        kb_str = str(kode_bank).replace('.0', '').zfill(3)
        bank_data = df[df['kode_bank'] == kb_str].copy()
        
        if bank_data.empty:
            logger.error(f"Bank data empty for {kode_bank}")
            return {'error': f"Data historis untuk {kode_bank} tidak ditemukan."}

        # Sort ascending for time series logic
        bank_data = bank_data.sort_values('periode')
        latest = bank_data.iloc[-1:]
        last_date = latest['periode'].iloc[0]
        
        # Metrics to predict
        metrics = ['CAR', 'NPL_gross', 'NPL_net', 'ROA', 'ROE', 'NIM', 'BOPO', 'LDR']
        projections = {}
        
        # AI Models
        lgbm_models = models.get('lgbm_reg', {})
        feat_cols = models.get('feat_cols', [])
        
        forecast_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=6, freq='MS')
        is_simulated_inference = True
        
        # 2. AI Inference Engine
        if isinstance(lgbm_models, dict) and len(lgbm_models) > 0 and len(bank_data) >= 3:
            try:
                from src.models.feature_engineering import create_lag_features, create_rolling_features
                
                # Identify base features needed
                base_cols = set(metrics)
                if feat_cols:
                    for c in feat_cols:
                        base = c.split('_lag')[0].split('_ma')[0].split('_roll')[0]
                        base_cols.add(base)
                base_cols_present = [c for c in base_cols if c in bank_data.columns]
                
                if is_simulation:
                    # ==========================================
                    # A. ITERATIVE MODE (Simulation)
                    # ==========================================
                    current_bank_data = bank_data.copy()
                    metric_predictions = {m: [] for m in metrics}
                    
                    for step in range(1, 7):
                        next_date = last_date + pd.DateOffset(months=step)
                        next_row = current_bank_data.iloc[-1:].copy()
                        next_row['periode'] = next_date
                        
                        temp_extended = pd.concat([current_bank_data, next_row], ignore_index=True)
                        df_feat = create_lag_features(temp_extended, columns=base_cols_present)
                        df_feat = create_rolling_features(df_feat, columns=base_cols_present)
                        
                        X_latest = df_feat.tail(1).copy()
                        if feat_cols:
                            missing_data = {}
                            for c in feat_cols:
                                if c not in X_latest.columns:
                                    missing_data[c] = [temp_extended[c].iloc[-1] if c in temp_extended.columns else 0]
                            if missing_data:
                                X_latest = pd.concat([X_latest, pd.DataFrame(missing_data, index=X_latest.index)], axis=1)
                            X_input = X_latest[feat_cols]
                        else:
                            X_input = X_latest.select_dtypes(include=[np.number])
                        
                        step_results = {}
                        for metric in metrics:
                            model = lgbm_models.get(metric)
                            if not model:
                                model_key = next((k for k in lgbm_models.keys() if k.lower() == metric.lower()), None)
                                model = lgbm_models.get(model_key) if model_key else None
                            
                            if model and hasattr(model, 'predict'):
                                pred_val = float(model.predict(X_input)[0])
                                # Robust clipping based on Section 12 logic
                                if metric == 'CAR': pred_val = max(0, min(100, pred_val))
                                if 'NPL' in metric: pred_val = max(0, min(100, pred_val))
                                if metric == 'ROE': pred_val = max(-100, min(100, pred_val))
                                if metric == 'LDR': pred_val = max(0, min(200, pred_val))
                                step_results[metric] = pred_val
                                metric_predictions[metric].append(pred_val)
                            else:
                                prev_val = float(current_bank_data[metric].iloc[-1])
                                step_results[metric] = prev_val
                                metric_predictions[metric].append(prev_val)
                        
                        for m, val in step_results.items():
                            next_row[m] = val
                        current_bank_data = pd.concat([current_bank_data, next_row], ignore_index=True)
                    
                    for m in metrics: projections[m] = metric_predictions[m]
                    is_simulated_inference = False
                    logger.info(f"Iterative ML Forecast (Simulation) completed for {kode_bank}")
                
                else:
                    # ==========================================
                    # B. STANDARD MODE (Pre-trained logic)
                    # ==========================================
                    # Old logic: 1-step prediction interpolated over 6 months
                    future_date = last_date + pd.DateOffset(months=1)
                    future_row = latest.copy()
                    future_row['periode'] = future_date
                    
                    bank_data_extended = pd.concat([bank_data, future_row], ignore_index=True)
                    df_feat = create_lag_features(bank_data_extended, columns=base_cols_present)
                    df_feat = create_rolling_features(df_feat, columns=base_cols_present)
                    
                    X_latest = df_feat.tail(1).copy()
                    if feat_cols:
                        missing_data = {}
                        for c in feat_cols:
                            if c not in X_latest.columns:
                                missing_data[c] = [bank_data[c].iloc[-1] if c in bank_data.columns else 0]
                        if missing_data:
                            X_latest = pd.concat([X_latest, pd.DataFrame(missing_data, index=X_latest.index)], axis=1)
                        X_input = X_latest[feat_cols]
                    else:
                        X_input = X_latest.select_dtypes(include=[np.number])
                    
                    for metric in metrics:
                        model = lgbm_models.get(metric)
                        if not model:
                            model_key = next((k for k in lgbm_models.keys() if k.lower() == metric.lower()), None)
                            model = lgbm_models.get(model_key) if model_key else None
                        
                        if model and hasattr(model, 'predict'):
                            base_pred = float(model.predict(X_input)[0])
                            last_val = float(latest[metric].iloc[0])
                            projections[metric] = [float(last_val + (base_pred - last_val) * (i/6)) for i in range(1, 7)]
                            is_simulated_inference = False
                    
                    logger.info(f"Standard ML Forecast (Pre-trained) completed for {kode_bank}")
                            
            except Exception as e:
                logger.error(f"Inference Engine failed: {e}")
                import traceback
                logger.error(traceback.format_exc())

        # 3. Dynamic Fallback (Responsive Trend)
        for metric in metrics:
            if metric not in projections:
                try:
                    y = bank_data[metric].values
                    last_val_sim = float(y[-1])
                    
                    if len(y) >= 3:
                        # Slope from REAL history (excluding simulated point)
                        y_hist = y[:-1].astype(float)
                        x_hist = np.arange(len(y_hist))
                        slope, _ = np.polyfit(x_hist, y_hist, 1)
                    else:
                        slope = 0
                    
                    projections[metric] = [float(last_val_sim + (slope * i)) for i in range(1, 7)]
                    logger.info(f"Fallback Trend {metric} for {kode_bank}: Start {last_val_sim:.2f}, Slope {slope:.4f}")
                except Exception as fall_e:
                    val = float(latest[metric].iloc[0] if metric in latest.columns else 2.5)
                    projections[metric] = [val] * 6

        # 4. Global RBBR Calculation
        from src.rbbr_scoring import compute_rbbr_score, score_to_pk, pk_to_sinyal
        rbbr_proj = []
        for i in range(6):
            row_sim = latest.iloc[0].copy()
            for metric in metrics:
                if metric in projections:
                    row_sim[metric] = projections[metric][i]
            # Use real rbbr scoring logic on the simulated metrics
            score = compute_rbbr_score(row_sim)
            rbbr_proj.append(float(score))
        
        projections['rbbr_score'] = rbbr_proj
        
        # Latest PK/Sinyal from forecast
        pk = score_to_pk(rbbr_proj[0])
        sinyal = pk_to_sinyal(pk)
            
        return {
            'kode_bank': kb_str,
            'nama_bank': latest['nama_bank'].iloc[0] if 'nama_bank' in latest.columns else kode_bank,
            'periode_latest': last_date,
            'skor_rbbr': round(float(latest['rbbr_score'].iloc[0] if 'rbbr_score' in latest.columns else 2.5), 3),
            'pk_prediksi': int(pk),
            'sinyal_ews': sinyal,
            'projections': projections,
            'forecast_dates': forecast_dates.tolist(),
            'is_simulated': is_simulated_inference
        }
    except Exception as global_e:
        logger.error(f"Global Inference Error: {global_e}")
        return {'error': f"Inference failed: {str(global_e)}"}

def get_exact_rbbr_contributions(row: pd.Series):
    """
    Calculate exact additive SHAP-like contributions to RBBR score based on real OJK RBBR formula.

    RBBR formula (from compute_rbbr_score):
        risk_profile  = mean(score_npl, score_ldr, score_car)   weight=0.50
        gcg           = 2.0 (fixed)                              weight=0.15
        rentabilitas  = mean(score_roa, score_bopo)              weight=0.15
        permodalan    = score_car                                 weight=0.20

    Per-feature contribution = weight_of_pillar / n_features_in_pillar * (score_feature - baseline_score)
    where baseline_score = 3 (neutral midpoint on 1-5 scale).

    expected_value = formula baseline (all variable scores=3, gcg=2.0)
    Guarantees: expected_value + sum(contributions) == compute_rbbr_score(row)
    """
    from src.rbbr_scoring import score_npl, score_car, score_ldr, score_roa, score_bopo
    from src.config import RBBRConfig

    W = RBBRConfig.WEIGHTS  # risk_profile=0.50, gcg=0.15, rentabilitas=0.15, permodalan=0.20
    BASELINE = 3.0  # neutral score

    def _get(col):
        """Get value, falling back to col_pred variant. Handles None, float NaN, numpy NaN."""
        v = row.get(col)
        if v is None or (not isinstance(v, str) and pd.isna(v)):
            v = row.get(f'{col}_pred')
        if v is None or (not isinstance(v, str) and pd.isna(v)):
            return BASELINE
        return float(v)

    s_car  = float(score_car(_get('CAR')))
    s_npl  = float(score_npl(_get('NPL_gross')))
    s_ldr  = float(score_ldr(_get('LDR')))
    s_roa  = float(score_roa(_get('ROA')))
    s_bopo = float(score_bopo(_get('BOPO')))

    # risk_profile pillar: NPL, LDR, CAR each get 1/3 of weight 0.50
    w_risk_each = W['risk_profile'] / 3.0
    # rentabilitas pillar: ROA, BOPO each get 1/2 of weight 0.15
    w_rent_each = W['rentabilitas'] / 2.0
    # permodalan pillar: CAR alone gets full weight 0.20
    w_perm = W['permodalan']

    # CAR contributes to both risk_profile and permodalan
    contrib_car   = w_risk_each * (s_car  - BASELINE) + w_perm * (s_car  - BASELINE)
    contrib_npl   = w_risk_each * (s_npl  - BASELINE)
    contrib_ldr   = w_risk_each * (s_ldr  - BASELINE)
    contrib_roa   = w_rent_each * (s_roa  - BASELINE)
    contrib_bopo  = w_rent_each * (s_bopo - BASELINE)

    # expected_value: formula baseline (all variable scores = 3, gcg fixed = 2.0)
    # Guarantees: expected_value + sum(contributions) == compute_rbbr_score(row)
    expected_value = (
        W['risk_profile'] * BASELINE +
        W['gcg'] * 2.0 +
        W['rentabilitas'] * BASELINE +
        W['permodalan'] * BASELINE
    )

    features = ['CAR', 'NPL_gross', 'LDR', 'ROA', 'BOPO']
    shap_values = [contrib_car, contrib_npl, contrib_ldr, contrib_roa, contrib_bopo]
    X = pd.DataFrame([{
        'CAR': _get('CAR'), 'NPL_gross': _get('NPL_gross'),
        'LDR': _get('LDR'), 'ROA': _get('ROA'), 'BOPO': _get('BOPO'),
    }])

    return np.array([shap_values]), expected_value, X

def get_shap_values(bank_name: str, periode: pd.Timestamp, models: Dict, df: pd.DataFrame):
    """
    Calculate SHAP values for a specific bank and period using exact RBBR rules.
    """
    try:
        kb_str = str(bank_name).replace('.0', '').zfill(3)
        bank_data = df[(df['kode_bank'] == kb_str) & (df['periode'] == periode)].copy()
        if bank_data.empty: return None, None, None, False

        row = bank_data.iloc[0]
        shap_values, expected_value, X = get_exact_rbbr_contributions(row)

        return shap_values, expected_value, X, False
    except Exception as e:
        logger.warning(f"Exact SHAP calculation failed: {e}")
        return None, None, None, False

def get_shap_values_forecast(bank_name: str, periode: pd.Timestamp, models: Dict, df: pd.DataFrame, data_service):
    """
    Calculate SHAP values for forecast period using exact RBBR rules.
    """
    try:
        predictions_df = data_service.load_predictions()
        if predictions_df.empty:
            logger.warning("No predictions data available")
            return None, None, None, True

        from src.data_service import get_forecast_data
        forecast_df = get_forecast_data(predictions_df, df)

        if forecast_df.empty:
            logger.warning("No forecast data available")
            return None, None, None, True

        kb_str = str(bank_name).replace('.0', '').zfill(3)
        bank_row = forecast_df[(forecast_df['kode_bank'] == kb_str) & (forecast_df['periode'] == periode)]

        if bank_row.empty:
            logger.warning(f"No forecast row found for {kb_str} at {periode}")
            return None, None, None, True

        row = bank_row.iloc[0]
        # Use master data for expected_value baseline (real historical distribution)
        shap_values, expected_value, X = get_exact_rbbr_contributions(row)

        return shap_values, expected_value, X, False

    except Exception as e:
        logger.error(f"SHAP forecast calculation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None, None, None, True
