"""
Ensemble Predictor
Loads all 4 model types and generates weighted ensemble predictions.
NeuralProphet: macro-only (bi_rate, inflasi, kurs_usd) — used as exogenous signal
SARIMAX: per-bank per-rasio with macro exogenous (sarimax_fc.pkl)
Dynamic weights: stable (CAR, LDR) → heavier SARIMAX; volatile (NPL, ROA, NIM) → heavier LightGBM+TFT
"""

import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from training.ensemble_trainer import TARGET_COLS, MACRO_COLS
from src.models.feature_engineering import create_lag_features, create_rolling_features

_STABLE = {'CAR', 'LDR'}
_VOLATILE = {'NPL_gross', 'NPL_net', 'ROA', 'NIM'}

_WEIGHTS: Dict[str, Dict[str, float]] = {
    'stable':   {'tft': 0.15, 'lightgbm': 0.20, 'sarimax': 0.65},
    'volatile': {'tft': 0.30, 'lightgbm': 0.45, 'sarimax': 0.25},
    'balanced': {'tft': 0.25, 'lightgbm': 0.35, 'sarimax': 0.40},
}


def _weight_profile(target: str) -> Dict[str, float]:
    if target in _STABLE:
        return _WEIGHTS['stable']
    if target in _VOLATILE:
        return _WEIGHTS['volatile']
    return _WEIGHTS['balanced']


class EnsemblePredictor:
    """
    model_paths keys:
      'tft'        → path to .ckpt
      'lgbm_reg'   → path to lgbm_reg_models.pkl  (dict[str, Booster])
      'lgbm_clf'   → path to lgbm_clf.pkl
      'feat_cols'  → path to feat_cols.pkl
      'np_bi_rate' → path to np_bi_rate.pkl
      'np_inflasi' → path to np_inflasi.pkl
      'np_kurs_usd'→ path to np_kurs_usd.pkl
      'sarimax_fc' → path to sarimax_fc.pkl  (dict[bank][rasio] = SARIMAXResults)
    """

    def __init__(self, model_paths: Dict, config: Dict):
        self.model_paths = model_paths
        self.config = config
        self.tft_model = None
        self.lgbm_reg: Dict = {}
        self.lgbm_clf = None
        self.feat_cols: List[str] = []
        self.np_macro: Dict = {}       # {var: NeuralProphet}
        self.sarimax_fc: Dict = {}     # {bank: {rasio: SARIMAXResults}}
        self._load_models()

    # -------------------------------------------------------------------------
    # Model loading
    # -------------------------------------------------------------------------

    def _load_models(self):
        # TFT
        tft_path = self.model_paths.get('tft', '')
        if tft_path and Path(tft_path).exists():
            try:
                import torch as _torch, inspect as _inspect
                from pytorch_forecasting import TemporalFusionTransformer
                _ckpt = _torch.load(tft_path, map_location='cpu')
                _hp = _ckpt.get('hyper_parameters', {})
                _valid = set(_inspect.signature(TemporalFusionTransformer.__init__).parameters.keys())
                for k in [k for k in list(_hp.keys()) if k not in _valid and k != 'self']:
                    _hp.pop(k)
                _torch.save(_ckpt, tft_path)
                self.tft_model = TemporalFusionTransformer.load_from_checkpoint(tft_path)
                self.tft_model.eval()
            except Exception:
                self.tft_model = None

        # LightGBM reg dict
        lgbm_reg_path = self.model_paths.get('lgbm_reg', '')
        if lgbm_reg_path and Path(lgbm_reg_path).exists():
            try:
                with open(lgbm_reg_path, 'rb') as f:
                    obj = pickle.load(f)
                if isinstance(obj, dict):
                    self.lgbm_reg = obj
            except Exception:
                pass

        # LightGBM clf
        lgbm_clf_path = self.model_paths.get('lgbm_clf', '')
        if lgbm_clf_path and Path(lgbm_clf_path).exists():
            try:
                with open(lgbm_clf_path, 'rb') as f:
                    self.lgbm_clf = pickle.load(f)
            except Exception:
                pass

        # feat_cols
        feat_path = self.model_paths.get('feat_cols', '')
        if feat_path and Path(feat_path).exists():
            try:
                with open(feat_path, 'rb') as f:
                    self.feat_cols = pickle.load(f)
            except Exception:
                pass

        # NeuralProphet macro models (flat files)
        for var in MACRO_COLS:
            key = f'np_{var}'
            np_path = self.model_paths.get(key, '')
            if np_path and Path(np_path).exists():
                try:
                    import torch as _torch
                    # Use weights_only=False for NeuralProphet models (they contain custom objects)
                    loaded_model = _torch.load(np_path, map_location='cpu', weights_only=False)
                    if hasattr(loaded_model, 'trainer'):
                        loaded_model.trainer = None
                    self.np_macro[var] = loaded_model
                except Exception as e:
                    import logging
                    logging.getLogger(__name__).warning(f"Failed to load NP model {var}: {e}")
                    pass

        # SARIMAX per-bank per-rasio (flat sarimax_fc.pkl)
        sx_path = self.model_paths.get('sarimax_fc', '')
        if sx_path and Path(sx_path).exists():
            try:
                with open(sx_path, 'rb') as f:
                    obj = pickle.load(f)
                if isinstance(obj, dict):
                    self.sarimax_fc = obj
            except Exception:
                pass

    # -------------------------------------------------------------------------
    # Macro forecast helper (NeuralProphet → future exogenous for SARIMAX)
    # -------------------------------------------------------------------------

    def _get_macro_future(self, df: pd.DataFrame, horizon: int) -> Dict[str, np.ndarray]:
        """Generate macro forecasts from NP models; fallback to last-value."""
        macro_future: Dict[str, np.ndarray] = {}
        for var, model in self.np_macro.items():
            if var not in df.columns:
                continue
            try:
                df_np = (
                    df[['periode', var]].copy()
                    .assign(ds=lambda x: pd.to_datetime(x['periode']), y=lambda x: x[var])
                    .groupby('ds')['y'].mean().reset_index().sort_values('ds').dropna(subset=['y'])
                )
                future = model.make_future_dataframe(df_np, periods=horizon, n_historic_predictions=False)
                forecast = model.predict(future)
                last_hist = df_np['ds'].max()
                fc = forecast[forecast['ds'] > last_hist].head(horizon)
                yhat_col = 'yhat1' if 'yhat1' in fc.columns else 'yhat'
                macro_future[var] = fc[yhat_col].values.astype(float)
            except Exception:
                last_val = df[var].dropna().iloc[-1] if not df[var].dropna().empty else 0.0
                macro_future[var] = np.full(horizon, float(last_val))

        # Fallback for any macro var not covered by NP
        for var in MACRO_COLS:
            if var not in macro_future and var in df.columns:
                last_val = df[var].dropna().iloc[-1] if not df[var].dropna().empty else 0.0
                macro_future[var] = np.full(horizon, float(last_val))

        return macro_future

    # -------------------------------------------------------------------------
    # Per-model prediction helpers
    # -------------------------------------------------------------------------

    def _predict_lgbm(self, df: pd.DataFrame, target: str, horizon: int) -> Optional[pd.DataFrame]:
        model = self.lgbm_reg.get(target)
        if model is None:
            return None
        try:
            df_f = df.copy().sort_values(['kode_bank', 'periode'])
            base_cols = (
                list({c.split('_lag')[0].split('_ma')[0].split('_roll')[0] for c in self.feat_cols})
                if self.feat_cols else [target]
            )
            base_cols = [c for c in base_cols if c in df_f.columns]
            df_f = create_lag_features(df_f, columns=base_cols)
            df_f = create_rolling_features(df_f, columns=base_cols)

            rows = []
            for bank in df_f['kode_bank'].unique():
                bdf = df_f[df_f['kode_bank'] == bank].sort_values('periode')
                if bdf.empty:
                    continue
                last_row = bdf.tail(1).copy()
                last_date = pd.to_datetime(last_row['periode'].iloc[0])
                X = (last_row.reindex(columns=self.feat_cols, fill_value=0)
                     if self.feat_cols else last_row.select_dtypes(include=[np.number]))
                pred_val = float(model.predict(X)[0])
                for h in range(1, horizon + 1):
                    rows.append({'kode_bank': bank,
                                 'periode': last_date + pd.DateOffset(months=h),
                                 'lightgbm': pred_val})
            return pd.DataFrame(rows) if rows else None
        except Exception:
            return None

    def _predict_sarimax(self, df: pd.DataFrame, target: str, horizon: int,
                         macro_future: Dict[str, np.ndarray]) -> Optional[pd.DataFrame]:
        """
        Per-bank SARIMAX forecast using macro exogenous.
        Looks up sarimax_fc[bank][target]; falls back to linear drift.
        """
        if not self.sarimax_fc:
            return None

        exog_cols = [c for c in MACRO_COLS if c in df.columns]
        rows = []

        for bank in df['kode_bank'].unique():
            bdf = df[df['kode_bank'] == bank].sort_values('periode')
            if bdf.empty or target not in bdf.columns:
                continue
            last_date = pd.to_datetime(bdf['periode'].iloc[-1])

            bank_models = self.sarimax_fc.get(str(bank), {})
            results = bank_models.get(target)

            if results is not None:
                try:
                    # Build future exogenous matrix
                    if exog_cols:
                        exog_future = np.column_stack(
                            [macro_future.get(c, np.full(horizon, 0.0)) for c in exog_cols]
                        )
                    else:
                        exog_future = None

                    forecast = results.forecast(steps=horizon, exog=exog_future)
                    for h, val in enumerate(forecast, start=1):
                        rows.append({'kode_bank': bank,
                                     'periode': last_date + pd.DateOffset(months=h),
                                     'sarimax': float(val)})
                    continue
                except Exception:
                    pass

            # Fallback: linear drift
            y = bdf[target].values.astype(float)
            slope = float(np.polyfit(np.arange(len(y)), y, 1)[0]) if len(y) >= 2 else 0.0
            for h in range(1, horizon + 1):
                rows.append({'kode_bank': bank,
                             'periode': last_date + pd.DateOffset(months=h),
                             'sarimax': float(y[-1] + slope * h)})

        return pd.DataFrame(rows) if rows else None

    def _predict_tft(self, df: pd.DataFrame, target: str, horizon: int) -> Optional[pd.DataFrame]:
        if self.tft_model is None:
            return None
        try:
            import torch
            from pytorch_forecasting import TimeSeriesDataSet
            from pytorch_forecasting.data import GroupNormalizer
            from training.ensemble_trainer import TFT_PRIMARY_TARGET, TFT_UNKNOWN_REALS

            df_t = df.copy().loc[:, ~df.columns.duplicated()]
            df_t['periode'] = pd.to_datetime(df_t['periode'])
            df_t = df_t.sort_values(['kode_bank', 'periode'])
            df_t['time_idx'] = df_t.groupby('kode_bank').cumcount()
            df_t = df_t.rename(columns={'kode_bank': 'group'})
            df_t['group'] = df_t['group'].astype(str)

            if TFT_PRIMARY_TARGET not in df_t.columns:
                df_t[TFT_PRIMARY_TARGET] = df_t[[c for c in TFT_UNKNOWN_REALS if c in df_t.columns]].mean(axis=1)

            unknown_reals = [c for c in TFT_UNKNOWN_REALS if c in df_t.columns]
            if TFT_PRIMARY_TARGET not in unknown_reals:
                unknown_reals = [TFT_PRIMARY_TARGET] + unknown_reals

            known_reals = ['time_idx'] + [c for c in MACRO_COLS if c in df_t.columns]
            keep = list({'time_idx', 'group', 'periode', TFT_PRIMARY_TARGET} | set(unknown_reals) | set(known_reals))
            df_t = df_t[[c for c in keep if c in df_t.columns]].ffill().fillna(0)
            df_t = df_t.loc[:, ~df_t.columns.duplicated()]

            max_enc = self.config.get('max_encoder_length', 12)
            max_pred = horizon
            min_len = max_enc + max_pred
            
            # Filter groups with sufficient data
            group_counts = df_t.groupby('group')['time_idx'].count()
            valid_groups = group_counts[group_counts >= min_len].index.tolist()
            
            if not valid_groups:
                return None
            
            df_enc = df_t[df_t['group'].isin(valid_groups)].groupby('group').tail(max_enc).copy()

            dataset = TimeSeriesDataSet(
                df_enc, time_idx='time_idx', target=TFT_PRIMARY_TARGET, group_ids=['group'],
                max_encoder_length=max_enc, max_prediction_length=horizon,
                time_varying_known_reals=known_reals, time_varying_unknown_reals=unknown_reals,
                target_normalizer=GroupNormalizer(groups=['group'], transformation='softplus'),
                add_relative_time_idx=True, add_target_scales=True, add_encoder_length=True,
                allow_missing_timesteps=True,
            )
            dl = dataset.to_dataloader(train=False, batch_size=32, num_workers=0)

            with torch.no_grad():
                preds = self.tft_model.predict(dl, mode='prediction')

            rows = []
            for i, group in enumerate(valid_groups):
                last_date = df_enc[df_enc['group'] == group]['periode'].max()
                group_preds = preds[i].cpu().numpy().flatten()
                for h, val in enumerate(group_preds[:horizon], start=1):
                    rows.append({'kode_bank': group,
                                 'periode': last_date + pd.DateOffset(months=h),
                                 'tft': float(val)})
            return pd.DataFrame(rows) if rows else None
        except Exception:
            return None

    # -------------------------------------------------------------------------
    # Ensemble combination
    # -------------------------------------------------------------------------

    def _smooth_anchor_blend(self, target: str, historical_last_series: pd.Series, 
                             model_preds: pd.DataFrame, horizon: int = 6) -> pd.Series:
        """
        Applies smooth anchor blend matching Section 17 of base training:
        H+1: 0.85 history + 0.15 model
        H+2: 0.65 + 0.35
        H+3: 0.45 + 0.55
        H+4: 0.25 + 0.75
        H+5: 0.10 + 0.90
        H+6: 0.00 + 1.00
        """
        weights = {1: 0.85, 2: 0.65, 3: 0.45, 4: 0.25, 5: 0.10, 6: 0.00}
        
        model_preds = model_preds.sort_values(['kode_bank', 'periode']).copy()
        model_preds['h'] = model_preds.groupby('kode_bank').cumcount() + 1
        
        def blend_row(row):
            bank = str(row['kode_bank'])
            h = int(row['h'])
            w_hist = weights.get(h, 0.0)
            w_model = 1.0 - w_hist
            
            # Get last known historical value for this bank/target
            hist_val = historical_last_series.get(bank, row[target])
            return (w_hist * hist_val) + (w_model * row[target])
            
        return model_preds.apply(blend_row, axis=1)

    def predict_target(self, df: pd.DataFrame, target: str, horizon: int,
                       macro_future: Optional[Dict] = None) -> pd.DataFrame:
        if macro_future is None:
            macro_future = self._get_macro_future(df, horizon)

        weights = _weight_profile(target)
        
        # Collect predictions from all models
        lgbm_df = self._predict_lgbm(df, target, horizon)
        sx_df = self._predict_sarimax(df, target, horizon, macro_future)
        tft_df = self._predict_tft(df, target, horizon)
        
        # Merge predictions using outer join to preserve all banks
        all_dfs = []
        if lgbm_df is not None and not lgbm_df.empty:
            all_dfs.append(lgbm_df)
        if sx_df is not None and not sx_df.empty:
            all_dfs.append(sx_df)
        if tft_df is not None and not tft_df.empty:
            all_dfs.append(tft_df)
        
        if not all_dfs:
            return self._fallback_drift(df, target, horizon)
        
        # Merge all predictions
        combined = all_dfs[0]
        for other_df in all_dfs[1:]:
            combined = pd.merge(combined, other_df, on=['kode_bank', 'periode'], how='outer')
        
        # Get available model columns
        available = [c for c in ['lightgbm', 'sarimax', 'tft'] if c in combined.columns]
        
        if not available:
            return self._fallback_drift(df, target, horizon)
        
        # Calculate weighted ensemble
        total_w = sum(weights.get(m, 0) for m in available)
        if total_w == 0:
            w_vec = {m: 1.0 / len(available) for m in available}
        else:
            w_vec = {m: weights.get(m, 0) / total_w for m in available}
        
        combined['ensemble'] = combined[available].fillna(0).apply(
            lambda row: sum(row[m] * w_vec[m] for m in available if pd.notna(row[m])), axis=1
        )
        
        # Apply smooth anchor blend
        historical_last = df.sort_values('periode').groupby('kode_bank')[target].last()
        historical_last.index = historical_last.index.astype(str)
        
        combined['ensemble_smoothed'] = self._smooth_anchor_blend(
            'ensemble', historical_last, combined[['kode_bank', 'periode', 'ensemble']], horizon
        )
        
        # Calculate uncertainty bounds
        std = combined[available].std(axis=1).fillna(0)
        combined['lower_bound'] = combined['ensemble_smoothed'] - 1.96 * std
        combined['upper_bound'] = combined['ensemble_smoothed'] + 1.96 * std

        return combined[['kode_bank', 'periode', 'ensemble_smoothed', 'lower_bound', 'upper_bound']].rename(columns={
            'ensemble_smoothed': target,
            'lower_bound': f'{target}_lower',
            'upper_bound': f'{target}_upper',
        })

    def _fallback_drift(self, df: pd.DataFrame, target: str, horizon: int) -> pd.DataFrame:
        rows = []
        for bank in df['kode_bank'].unique():
            bdf = df[df['kode_bank'] == bank].sort_values('periode')
            if target not in bdf.columns or bdf.empty:
                continue
            y = bdf[target].values.astype(float)
            last_date = pd.to_datetime(bdf['periode'].iloc[-1])
            slope = float(np.polyfit(np.arange(len(y)), y, 1)[0]) if len(y) >= 2 else 0.0
            for h in range(1, horizon + 1):
                val = float(y[-1] + slope * h)
                rows.append({'kode_bank': bank,
                             'periode': last_date + pd.DateOffset(months=h),
                             target: val, f'{target}_lower': val, f'{target}_upper': val})
        return pd.DataFrame(rows)

    def predict_all_targets(self, df: pd.DataFrame, targets: List[str],
                            horizon: int = 6, callback=None) -> pd.DataFrame:
        """Generate predictions for all targets using ensemble models with progress tracking"""
        import logging
        import sys
        logger = logging.getLogger(__name__)
        
        # Normalize kode_bank immediately
        from src.data_utils import normalize_kode_bank
        df = normalize_kode_bank(df)
        
        _cb = callback or (lambda p, m: None)
        logger.info(f"Starting predict_all_targets for {len(targets)} targets, horizon={horizon}")
        
        # Pre-compute macro forecasts once for all targets
        try:
            macro_future = self._get_macro_future(df, horizon)
            logger.info(f"✓ Generated macro forecasts: {list(macro_future.keys())}")
        except Exception as e:
            logger.error(f"Failed to generate macro forecasts: {e}")
            macro_future = {}

        all_dfs = []
        for target in targets:
            if target not in df.columns:
                logger.warning(f"Target {target} not in dataframe, skipping")
                continue
            try:
                pred_df = self.predict_target(df, target, horizon, macro_future)
                if not pred_df.empty:
                    all_dfs.append(pred_df)
                    msg = f"✓ Predicted {target}: {len(pred_df)} rows"
                    logger.info(msg)
                    _cb(None, msg)
                else:
                    msg = f"⚠️ Empty prediction for {target}"
                    logger.warning(msg)
                    _cb(None, msg)
            except Exception as e:
                msg = f"❌ Failed to predict {target}: {str(e)}"
                logger.error(msg)
                _cb(None, msg)

        if not all_dfs:
            logger.error("❌ No predictions generated for any target!")
            return pd.DataFrame()

        logger.info(f"Merging {len(all_dfs)} prediction dataframes...")
        merged = all_dfs[0]
        for other in all_dfs[1:]:
            merged = pd.merge(merged, other, on=['kode_bank', 'periode'], how='outer')

        merged['kode_bank'] = merged['kode_bank'].astype(str)
        merged['periode'] = pd.to_datetime(merged['periode'])
        result = merged.sort_values(['kode_bank', 'periode']).reset_index(drop=True)
        logger.info(f"✓ Final merged predictions: {len(result)} rows")
        return result
