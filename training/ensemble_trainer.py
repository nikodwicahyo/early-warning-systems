"""
Ensemble Model Trainer
Incremental fine-tuning for TFT + LightGBM + NeuralProphet (macro-only) + SARIMAX (per-bank per-rasio with exogenous)
Matches base training approach in OJK_RBBR_EWS_v4_2.ipynb
"""

import gc
import pickle
import shutil
import time
import warnings
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore", message=".*does not have many workers.*")
warnings.filterwarnings("ignore", message=".*The behavior of DataFrame concatenation with empty or all-NA entries is deprecated.*")

import numpy as np
import pandas as pd

from src.models.feature_engineering import create_full_feature_set
from src.models.data_augmentation import augment_and_recompute

logger = logging.getLogger(__name__)

TARGET_COLS = ['CAR', 'NPL_gross', 'NPL_net', 'ROA', 'ROE', 'NIM', 'BOPO', 'LDR']
MACRO_COLS = ['bi_rate', 'inflasi', 'kurs_usd']
TFT_PRIMARY_TARGET = 'rbbr_score'
TFT_UNKNOWN_REALS = ['CAR', 'NPL_gross', 'NPL_net', 'ROA', 'ROE', 'NIM', 'BOPO', 'LDR']

# NeuralProphet config per macro variable (matches Section 11 of base training)
_NP_CONFIG = {
    'bi_rate':   {'n_lags': 6,  'n_changepoints': 10, 'epochs': 300},
    'inflasi':   {'n_lags': 12, 'n_changepoints': 10, 'epochs': 300},
    'kurs_usd':  {'n_lags': 6,  'n_changepoints': 10, 'epochs': 300},
}


class EnsembleTrainer:
    def __init__(self, config: Dict):
        self.config = config
        self.models_dir = Path(config.get('models_dir', 'models'))
        self.tft_best_model_path: Optional[str] = None

    def _prepare_data_full(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Runs the full feature engineering and data augmentation pipeline.
        Returns enriched and augmented DataFrame.
        """
        # 1. Full Feature Engineering
        df_feat = create_full_feature_set(df, include_forward_targets=True)
        
        # 2. Data Augmentation (3x data)
        df_aug = augment_and_recompute(df_feat, max_multiplier=3)
        
        return df_aug

    # =========================================================================
    # TFT
    # =========================================================================

    def prepare_data_tft(self, df: pd.DataFrame) -> Tuple:
        from pytorch_forecasting import TimeSeriesDataSet
        from pytorch_forecasting.data import GroupNormalizer
        
        df_t = df.copy().loc[:, lambda x: ~x.columns.duplicated()]
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

        max_pred = self.config.get('max_prediction_length', 6)
        max_enc = self.config.get('max_encoder_length', 12)
        min_len = max_enc + max_pred
        # Drop groups that don't have enough timesteps to form even one sample
        group_counts = df_t.groupby('group')['time_idx'].count()
        valid_groups = group_counts[group_counts >= min_len].index
        df_t = df_t[df_t['group'].isin(valid_groups)].copy()

        cutoff = df_t['time_idx'].max() - max_pred

        training = TimeSeriesDataSet(
            df_t[df_t['time_idx'] <= cutoff],
            time_idx='time_idx', target=TFT_PRIMARY_TARGET, group_ids=['group'],
            max_encoder_length=self.config.get('max_encoder_length', 12),
            max_prediction_length=max_pred,
            time_varying_known_reals=known_reals,
            time_varying_unknown_reals=unknown_reals,
            target_normalizer=GroupNormalizer(groups=['group'], transformation='softplus'),
            add_relative_time_idx=True, add_target_scales=True, add_encoder_length=True,
            allow_missing_timesteps=True,
        )
        validation = TimeSeriesDataSet.from_dataset(training, df_t, predict=True, stop_randomization=True)
        return training, validation

    def train_ensemble(self, df: pd.DataFrame, progress_callback=None) -> Dict:
        """
        Master method to run full retraining pipeline with detailed progress
        """
        _cb = progress_callback or (lambda p, m: None)
        
        # Normalize kode_bank immediately
        from src.data_utils import normalize_kode_bank
        df = normalize_kode_bank(df)
        
        # Phase 1: Data Prep
        _cb(0.05, "Phase 1/4 — Mempersiapkan data (FE & Augmentasi)...")
        df_full = self._prepare_data_full(df)
        _cb(0.10, "✓ Data preparation selesai")
        
        # Phase 2: NP Macro
        _cb(0.15, "Phase 2/4 — Training NeuralProphet (Makro)...")
        existing_np = {
            'bi_rate': str(self.models_dir / 'np_bi_rate.pkl'),
            'inflasi': str(self.models_dir / 'np_inflasi.pkl'),
            'kurs_usd': str(self.models_dir / 'np_kurs_usd.pkl'),
        }
        np_models = self.train_neuralprophet_macro(df_full, existing_np, _cb)
        np_paths = self.save_neuralprophet_macro(np_models)
        macro_forecasts = self.get_macro_forecasts(np_models, df_full)
        _cb(0.30, "✓ NeuralProphet selesai")
        
        # Phase 3: TFT
        _cb(0.35, "Phase 3/4 — Training TFT (Deep Learning)...")
        tft_ckpts = sorted(self.models_dir.glob('tft-retrained-*.ckpt'))
        base_tft = self.models_dir / 'tft.ckpt'
        alt_base = self.models_dir / 'tft-epoch=00-val_loss=0.0507.ckpt'
        
        if tft_ckpts:
            tft_ckpt = str(tft_ckpts[-1])
        elif base_tft.exists():
            tft_ckpt = str(base_tft)
        elif alt_base.exists():
            tft_ckpt = str(alt_base)
        else:
            any_ckpt = sorted(self.models_dir.glob('*.ckpt'))
            tft_ckpt = str(any_ckpt[-1]) if any_ckpt else None

        tft_res = self.train_tft(df_full, tft_ckpt if tft_ckpt and Path(tft_ckpt).exists() else None, _cb)
        tft_path = self.save_tft(datetime.now().strftime('%Y%m%d_%H%M'))
        _cb(0.55, "✓ TFT selesai")
        
        # Phase 4: LightGBM
        _cb(0.60, "Phase 3/4 — Training LightGBM (Regression)...")
        lgbm_reg_res = self.train_lightgbm_reg(df_full, str(self.models_dir / 'lgbm_reg_models.pkl'), 
                                               str(self.models_dir / 'feat_cols.pkl'), _cb)
        _cb(0.70, "✓ LightGBM Regression selesai")
        
        _cb(0.72, "Training LightGBM (Classification)...")
        lgbm_clf = self.train_lightgbm_clf(df_full, str(self.models_dir / 'lgbm_clf.pkl'),
                                           str(self.models_dir / 'feat_cols.pkl'), _cb)
        lgbm_paths = self.save_lightgbm(lgbm_reg_res['models'], lgbm_clf)
        _cb(0.75, "✓ LightGBM selesai")
        
        # Phase 5: SARIMAX
        _cb(0.78, "Phase 3/4 — Training SARIMAX (Per-bank)...")
        sarimax_fc, sarimax_agg = self.train_sarimax_per_bank(df_full, macro_forecasts, _cb)
        sarimax_paths = self.save_sarimax(sarimax_fc, sarimax_agg)
        sarimax_metrics = {r: float(m.aic) for r, m in sarimax_agg.items() if hasattr(m, 'aic')}
        _cb(0.83, "✓ SARIMAX selesai")
        
        _cb(0.85, "✓ Semua model berhasil dilatih")
        
        return {
            'timestamp': datetime.now().isoformat(),
            'tft': tft_res,
            'lgbm': lgbm_reg_res['metrics'],
            'neuralprophet': getattr(self, '_np_metrics', {}),
            'sarimax': sarimax_metrics,
            'paths': {
                'tft': tft_path,
                **lgbm_paths,
                **np_paths,
                **sarimax_paths
            }
        }

    def train_tft(self, df: pd.DataFrame, existing_ckpt_path: Optional[str] = None,
                  progress_callback=None) -> Dict:
        import torch
        from pytorch_forecasting import TemporalFusionTransformer
        from pytorch_forecasting.metrics import QuantileLoss
        from lightning.pytorch import Trainer
        from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
        
        _cb = progress_callback or (lambda p, m: None)
        _cb(0.02, "TFT: menyiapkan dataset...")
        training, validation = self.prepare_data_tft(df)
        batch_size = self.config.get('batch_size', 32)
        train_dl = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
        val_dl = validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=0)

        _cb(0.06, "TFT: memuat model...")
        # Always train from scratch to avoid categorical/feature mismatch
        _cb(0.08, "TFT: memulai dari awal (scratch)...")
        tft = TemporalFusionTransformer.from_dataset(
            training,
            learning_rate=self.config.get('learning_rate', 3e-3),
            hidden_size=self.config.get('hidden_size', 64),
            attention_head_size=self.config.get('attention_head_size', 4),
            dropout=self.config.get('dropout', 0.15),
            hidden_continuous_size=self.config.get('hidden_continuous_size', 32),
            output_size=7,
            loss=QuantileLoss([0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]),
            lstm_layers=2, reduce_on_plateau_patience=4,
        )
        max_epochs = self.config.get('max_epochs', 30)

        _cb(0.10, f"TFT: training from scratch ({max_epochs} epochs)...")
        checkpoint = ModelCheckpoint(monitor='val_loss', mode='min', save_top_k=1)
        trainer = Trainer(
            max_epochs=max_epochs, accelerator='auto', gradient_clip_val=0.15,
            callbacks=[EarlyStopping(monitor='val_loss', patience=5, mode='min'), checkpoint],
            enable_progress_bar=False, enable_model_summary=False, logger=False,
        )
        t0 = time.time()
        trainer.fit(tft, train_dataloaders=train_dl, val_dataloaders=val_dl)
        elapsed = time.time() - t0
        self.tft_best_model_path = checkpoint.best_model_path
        val_loss = trainer.callback_metrics.get('val_loss', torch.tensor(0)).item()
        _cb(0.24, f"TFT selesai — val_loss={val_loss:.4f}, waktu={elapsed:.0f}s")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {'best_model_path': self.tft_best_model_path, 'val_loss': val_loss,
                'training_time': elapsed, 'fine_tuned': False}

    def save_tft(self, timestamp: str) -> Optional[str]:
        if not self.tft_best_model_path or not Path(self.tft_best_model_path).exists():
            return None
        dest = self.models_dir / f"tft-retrained-{timestamp}.ckpt"
        shutil.copy(self.tft_best_model_path, dest)
        return str(dest)

    # =========================================================================
    # LightGBM
    # =========================================================================

    def _build_lgbm_features(self, df: pd.DataFrame, target_col: str,
                             feat_cols: List[str] = None) -> Tuple:
        df_f = df.copy().sort_values(['kode_bank', 'periode'])
        
        # If feat_cols not provided, use all numeric columns except targets/identifiers
        if not feat_cols:
            exclude = set(TARGET_COLS) | {target_col, 'kode_bank', 'bank', 'periode', 'time_idx', 'group'}
            feat_cols = [c for c in df_f.select_dtypes(include=[np.number]).columns 
                        if c not in exclude and not any(t in c for t in ['_fwd', 'pk_'])]
        
        X = df_f[feat_cols].fillna(0)
        y = df_f[target_col]
        
        # Split: 80% train+val, 20% test
        split_idx = int(len(df_f) * 0.8)
        X_train_val, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train_val, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Further split train+val for Optuna: 75% train, 25% val
        tv_split = int(len(X_train_val) * 0.75)
        X_train, X_val = X_train_val.iloc[:tv_split], X_train_val.iloc[tv_split:]
        y_train, y_val = y_train_val.iloc[:tv_split], y_train_val.iloc[tv_split:]
        
        return X_train, X_val, X_test, y_train, y_val, y_test, feat_cols

    def _tune_lgbm_params(self, X_train, y_train, X_val, y_val, is_clf=False) -> Dict:
        """Optuna TPE tuning for LightGBM"""
        import optuna
        import lightgbm as lgb
        from sklearn.metrics import mean_absolute_error
        
        n_trials = self.config.get('optuna_trials', 20)
        
        def objective(trial):
            params = {
                'objective': 'multiclass' if is_clf else 'regression',
                'metric': 'multi_logloss' if is_clf else 'mae',
                'verbosity': -1,
                'boosting_type': 'gbdt',
                'num_class': 5 if is_clf else None,
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 31, 256),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
                'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
            }
            
            train_data = lgb.Dataset(X_train, label=y_train)
            valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            
            model = lgb.train(
                params, train_data, num_boost_round=1000,
                valid_sets=[valid_data],
                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
            )
            
            preds = model.predict(X_val)
            if is_clf:
                from sklearn.metrics import log_loss
                return log_loss(y_val, preds)
            else:
                return mean_absolute_error(y_val, preds)

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        return study.best_params


    def train_lightgbm_reg(self, df: pd.DataFrame, existing_models_path: Optional[str] = None,
                           feat_cols_path: Optional[str] = None, progress_callback=None) -> Dict:
        import lightgbm as lgb
        from sklearn.metrics import mean_absolute_error
        
        _cb = progress_callback or (lambda p, m: None)
        _cb(0.26, "LightGBM reg: memuat model dan fitur...")

        existing: Dict[str, lgb.Booster] = {}
        if existing_models_path and Path(existing_models_path).exists():
            with open(existing_models_path, 'rb') as f:
                existing = pickle.load(f)

        feat_cols: List[str] = []
        if feat_cols_path and Path(feat_cols_path).exists():
            with open(feat_cols_path, 'rb') as f:
                feat_cols = pickle.load(f)

        updated_models: Dict[str, lgb.Booster] = dict(existing)
        metrics: Dict[str, float] = {}
        all_feat_cols = []
        n = len(TARGET_COLS)

        for i, target in enumerate(TARGET_COLS):
            if target not in df.columns:
                continue
            _cb(0.26 + 0.12 * i / n, f"LightGBM reg: tuning & training {target}...")
            try:
                # Build features
                X_tr, X_val, X_te, y_tr, y_val, y_te, current_feats = self._build_lgbm_features(df, target, feat_cols)
                if not all_feat_cols:
                    all_feat_cols = current_feats
                
                # Optuna Tuning
                best_params = self._tune_lgbm_params(X_tr, y_tr, X_val, y_val, is_clf=False)
                best_params.update({'objective': 'regression', 'metric': 'mae', 'verbosity': -1})
                
                # Training on full train_val
                X_full = pd.concat([X_tr, X_val])
                y_full = pd.concat([y_tr, y_val])
                train_data = lgb.Dataset(X_full, label=y_full)
                
                model = lgb.train(
                    best_params, train_data, num_boost_round=500,
                    init_model=existing.get(target),
                    callbacks=[lgb.log_evaluation(0)],
                )
                updated_models[target] = model
                
                # Evaluation
                if len(y_te) > 0:
                    y_pred = model.predict(X_te)
                    mae_val = mean_absolute_error(y_te, y_pred)
                    metrics[target] = float(mae_val)
                else:
                    # Fallback to validation MAE if test is empty
                    y_pred_val = model.predict(X_val)
                    metrics[target] = float(mean_absolute_error(y_val, y_pred_val))
                    
            except Exception as e:
                print(f"Error training LightGBM for {target}: {str(e)}")

        # Save regenerated feature columns
        if all_feat_cols:
            with open(self.models_dir / 'feat_cols.pkl', 'wb') as f:
                pickle.dump(all_feat_cols, f)

        _cb(0.38, "LightGBM reg selesai.")
        gc.collect()
        return {'models': updated_models, 'metrics': metrics}

    def train_lightgbm_clf(self, df: pd.DataFrame, existing_clf_path: Optional[str] = None,
                           feat_cols_path: Optional[str] = None, progress_callback=None) -> Optional[object]:
        import lightgbm as lgb
        
        _cb = progress_callback or (lambda p, m: None)
        _cb(0.38, "LightGBM clf: tuning & training PK classifier...")

        target_col = 'pk_computed' if 'pk_computed' in df.columns else ('pk' if 'pk' in df.columns else None)
        if target_col is None:
            return None

        feat_cols: List[str] = []
        if feat_cols_path and Path(feat_cols_path).exists():
            with open(feat_cols_path, 'rb') as f:
                feat_cols = pickle.load(f)

        try:
            X_tr, X_val, X_te, y_tr, y_val, y_te, _ = self._build_lgbm_features(df, target_col, feat_cols)
            
            # Label shift for LightGBM multiclass (1-5 -> 0-4)
            y_tr_0 = (y_tr - 1).clip(0, 4).astype(int)
            y_val_0 = (y_val - 1).clip(0, 4).astype(int)
            y_te_0 = (y_te - 1).clip(0, 4).astype(int)
            y_full_0 = pd.concat([y_tr_0, y_val_0])
            X_full = pd.concat([X_tr, X_val])

            # Optuna Tuning
            best_params = self._tune_lgbm_params(X_tr, y_tr_0, X_val, y_val_0, is_clf=True)
            best_params.update({'objective': 'multiclass', 'num_class': 5, 'metric': 'multi_logloss', 'verbosity': -1})

            init_model = None
            if existing_clf_path and Path(existing_clf_path).exists():
                with open(existing_clf_path, 'rb') as f:
                    init_model = pickle.load(f)

            clf = lgb.train(
                best_params, lgb.Dataset(X_full, label=y_full_0), num_boost_round=500,
                init_model=init_model,
                callbacks=[lgb.log_evaluation(0)],
            )
            _cb(0.40, "LightGBM clf selesai.")
            return clf
        except Exception as e:
            print(f"Error training LightGBM CLF: {e}")
            return None

    def save_lightgbm(self, reg_models: Dict, clf_model: Optional[object]) -> Dict[str, str]:
        paths = {}
        reg_path = self.models_dir / 'lgbm_reg_models.pkl'
        if reg_models:
            if reg_path.exists():
                shutil.copy(reg_path, reg_path.with_suffix('.pkl.bak'))
            with open(reg_path, 'wb') as f:
                pickle.dump(reg_models, f)
            paths['lgbm_reg'] = str(reg_path)

        clf_path = self.models_dir / 'lgbm_clf.pkl'
        if clf_model is not None:
            if clf_path.exists():
                shutil.copy(clf_path, clf_path.with_suffix('.pkl.bak'))
            with open(clf_path, 'wb') as f:
                pickle.dump(clf_model, f)
            paths['lgbm_clf'] = str(clf_path)
        
        # Include feat_cols.pkl path
        feat_cols_path = self.models_dir / 'feat_cols.pkl'
        if feat_cols_path.exists():
            paths['feat_cols'] = str(feat_cols_path)
        
        return paths

    # =========================================================================
    # NeuralProphet — macro variables only (bi_rate, inflasi, kurs_usd)
    # Matches Section 11 of base training: one NP model per macro variable
    # =========================================================================

    def train_neuralprophet_macro(
        self,
        df: pd.DataFrame,
        existing_np_paths: Dict[str, str],
        progress_callback=None,
    ) -> Dict[str, object]:
        """
        Train NeuralProphet for each macro variable independently.
        Fine-tunes existing model if path provided, else trains from scratch.
        Returns {var_name: fitted_NeuralProphet_model}
        """
        from neuralprophet import NeuralProphet
        
        _cb = progress_callback or (lambda p, m: None)
        np_models: Dict[str, object] = {}
        self._np_metrics = {}

        for i, var in enumerate(MACRO_COLS):
            if var not in df.columns:
                continue
            pct = 0.40 + 0.05 * i
            _cb(pct, f"NeuralProphet: training {var}...")
            
            try:
                # Prepare data
                df_np = df[['periode', var]].copy().rename(columns={'periode': 'ds', var: 'y'})
                df_np['ds'] = pd.to_datetime(df_np['ds'])
                df_np = df_np.groupby('ds').mean().reset_index().sort_values('ds').dropna()
                
                if len(df_np) < 12:
                    continue
                
                # Always create fresh model (NeuralProphet doesn't support refit)
                conf = _NP_CONFIG.get(var, {'n_lags': 6, 'epochs': 100})
                m = NeuralProphet(**conf)
                
                metrics = m.fit(df_np, freq='MS', progress=None)
                np_models[var] = m
                
                # Track last MAE
                if not metrics.empty:
                    mae_col = 'MAE' if 'MAE' in metrics.columns else (
                        'MAE_val' if 'MAE_val' in metrics.columns else None
                    )
                if mae_col:
                        self._np_metrics[var] = float(metrics[mae_col].iloc[-1])
                _cb(pct + 0.04, f"NeuralProphet {var} selesai.")
            except Exception as e:
                print(f"Error training NeuralProphet for {var}: {e}")
                _cb(pct + 0.04, f"NeuralProphet {var} gagal.")

        return np_models

    def save_neuralprophet_macro(self, np_models: Dict[str, object]) -> Dict[str, str]:
        """
        Save NeuralProphet macro models as flat files:
        np_bi_rate.pkl, np_inflasi.pkl, np_kurs_usd.pkl
        Uses torch.save for proper serialization
        """
        paths: Dict[str, str] = {}
        var_to_filename = {
            'bi_rate': 'np_bi_rate.pkl',
            'inflasi': 'np_inflasi.pkl',
            'kurs_usd': 'np_kurs_usd.pkl',
        }
        for var, model in np_models.items():
            filename = var_to_filename.get(var)
            if filename is None:
                continue
            dest = self.models_dir / filename
            if dest.exists():
                shutil.copy(dest, dest.with_suffix('.pkl.bak'))
            try:
                # Use torch.save for NeuralProphet models (they contain torch components)
                import torch
                if hasattr(model, 'trainer'):
                    model.trainer = None
                torch.save(model, str(dest))
                paths[f'np_{var}'] = str(dest)
                logger.info(f"Saved NeuralProphet model: {var} to {dest}")
            except Exception as e:
                logger.error(f"Failed to save NP model {var}: {e}")
        return paths

    def get_macro_forecasts(self, np_models: Dict[str, object], df: pd.DataFrame,
                            horizon: int = 6) -> Dict[str, np.ndarray]:
        """
        Generate 6-step macro forecasts from trained NP models.
        Returns {var: array of shape (horizon,)}
        Used as exogenous input for SARIMAX.
        """
        macro_fc: Dict[str, np.ndarray] = {}
        for var, model in np_models.items():
            if var not in df.columns:
                continue
            try:
                df_np = (
                    df[['periode', var]].copy()
                    .assign(ds=lambda x: pd.to_datetime(x['periode']), y=lambda x: x[var])
                    .groupby('ds')['y'].mean()
                    .reset_index()
                    .sort_values('ds')
                    .dropna(subset=['y'])
                )
                future = model.make_future_dataframe(df_np, periods=horizon, n_historic_predictions=False)
                forecast = model.predict(future)
                last_hist = df_np['ds'].max()
                fc_rows = forecast[forecast['ds'] > last_hist].head(horizon)
                yhat_col = 'yhat1' if 'yhat1' in fc_rows.columns else 'yhat'
                macro_fc[var] = fc_rows[yhat_col].values.astype(float)
            except Exception:
                # Fallback: last known value repeated
                last_val = df[var].dropna().iloc[-1] if var in df.columns and not df[var].dropna().empty else 0.0
                macro_fc[var] = np.full(horizon, float(last_val))
        return macro_fc

    # =========================================================================
    # SARIMAX — per-rasio per-bank with macro exogenous
    # Matches Section 15 of base training
    # =========================================================================

    def _fit_best_sarimax(self, y, exog):
        """Try candidate SARIMAX orders, return the fit with lowest AIC."""
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        
        candidates = [
            ((1, 1, 1), (1, 1, 1, 12)),
            ((2, 1, 1), (1, 1, 0, 12)),
            ((1, 1, 2), (0, 1, 1, 12)),
            ((1, 1, 1), (0, 0, 0, 0)),
            ((1, 1, 0), (0, 0, 0, 0)),
            ((0, 1, 1), (0, 0, 0, 0)),
            ((0, 1, 0), (0, 0, 0, 0)),
        ]
        best, best_aic = None, float('inf')
        for order, seasonal_order in candidates:
            try:
                kwargs = dict(enforce_stationarity=False, enforce_invertibility=False)
                if seasonal_order[3] > 0 and len(y) < seasonal_order[3] * 2 + 10:
                    continue  # not enough data for this seasonal period
                m = SARIMAX(y, exog=exog, order=order, seasonal_order=seasonal_order, **kwargs)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    r = m.fit(disp=False)
                if r.aic < best_aic:
                    best_aic, best = r.aic, r
            except Exception:
                continue
        return best

    def train_sarimax_per_bank(
        self,
        df: pd.DataFrame,
        macro_forecasts: Dict[str, np.ndarray],
        progress_callback=None,
    ) -> Tuple[Dict, Dict]:
        """
        Train SARIMAX for each bank × each rasio with macro exogenous variables.
        Returns:
          sarimax_fc: {bank: {rasio: SARIMAXResults}}  — for forecasting
          sarimax_agg: {rasio: SARIMAXResults}          — aggregated (all banks), for sarimax.pkl
        """
        _cb = progress_callback or (lambda p, m: None)
        sarimax_fc: Dict[str, Dict] = {}
        sarimax_agg: Dict[str, object] = {}

        banks = df['kode_bank'].unique().tolist()
        n_total = len(banks) * len(TARGET_COLS)
        done = 0

        for bank in banks:
            sarimax_fc[bank] = {}
            bdf = df[df['kode_bank'] == bank].sort_values('periode').copy()

            for rasio in TARGET_COLS:
                if rasio not in bdf.columns:
                    done += 1
                    continue

                pct = 0.55 + 0.25 * done / max(n_total, 1)
                _cb(pct, f"SARIMAX: {bank} — {rasio}...")
                done += 1

                y = bdf[rasio].ffill().bfill().values.astype(float)
                if len(y) < 6:
                    continue

                # Build exogenous matrix from macro columns available in data
                exog_cols = [c for c in MACRO_COLS if c in bdf.columns]
                exog_hist = bdf[exog_cols].ffill().bfill().values if exog_cols else None

                results = self._fit_best_sarimax(y, exog_hist)

                sarimax_fc[bank][rasio] = results

        # Aggregated SARIMAX (all banks pooled) — matches sarimax.pkl in base training
        _cb(0.80, "SARIMAX: training aggregated model...")
        for rasio in TARGET_COLS:
            if rasio not in df.columns:
                continue
            df_agg = (
                df[['periode', rasio] + [c for c in MACRO_COLS if c in df.columns]]
                .copy()
                .groupby('periode')
                .mean()
                .reset_index()
                .sort_values('periode')
            )
            y_agg = df_agg[rasio].ffill().bfill().values.astype(float)
            exog_cols = [c for c in MACRO_COLS if c in df_agg.columns]
            exog_agg = df_agg[exog_cols].ffill().bfill().values if exog_cols else None
            res = self._fit_best_sarimax(y_agg, exog_agg)
            if res is not None:
                sarimax_agg[rasio] = res

        _cb(0.83, "SARIMAX selesai.")
        return sarimax_fc, sarimax_agg

    def save_sarimax(self, sarimax_fc: Dict, sarimax_agg: Dict) -> Dict[str, str]:
        """
        Save SARIMAX models as flat files matching base training artifacts:
          models/sarimax_fc.pkl  — per-bank per-rasio dict
          models/sarimax.pkl     — aggregated dict
        """
        paths: Dict[str, str] = {}

        fc_path = self.models_dir / 'sarimax_fc.pkl'
        if fc_path.exists():
            shutil.copy(fc_path, fc_path.with_suffix('.pkl.bak'))
        with open(fc_path, 'wb') as f:
            pickle.dump(sarimax_fc, f)
        paths['sarimax_fc'] = str(fc_path)

        agg_path = self.models_dir / 'sarimax.pkl'
        if agg_path.exists():
            shutil.copy(agg_path, agg_path.with_suffix('.pkl.bak'))
        with open(agg_path, 'wb') as f:
            pickle.dump(sarimax_agg, f)
        paths['sarimax'] = str(agg_path)

        return paths
