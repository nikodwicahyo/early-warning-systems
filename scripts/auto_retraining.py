"""
Automated Retraining Pipeline
4-phase: TFT → LightGBM → NeuralProphet macro + SARIMAX per-bank → Predictions
Post-success: semantic version bump, clear uploads folder, version management with rollback
"""

import json
import shutil
import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from training.ensemble_trainer import TARGET_COLS, MACRO_COLS, EnsembleTrainer
from training.ensemble_predictor import EnsemblePredictor

logger = logging.getLogger(__name__)


def _bump_minor_version(version_str: str) -> str:
    """
    Bump minor version following semantic versioning (major.minor.patch).
    Handles legacy formats: "1.0" → "1.1.0", "1.1.0" → "1.2.0"
    """
    parts = str(version_str).strip().split('.')
    # Normalize to 3 parts
    while len(parts) < 3:
        parts.append('0')
    try:
        major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])
    except (ValueError, IndexError):
        major, minor, patch = 1, 0, 0
    return f"{major}.{minor + 1}.0"


class AutoRetrainer:
    def __init__(self, config_path: str = None, storage_manager=None):
        self.storage_manager = storage_manager
        self.config = self._default_config()
        if config_path and Path(config_path).exists():
            with open(config_path) as f:
                self.config.update(json.load(f))

        self.models_dir = Path(self.config['models_dir'])
        self.data_dir = Path(self.config['data_dir'])
        self.uploads_dir = self.data_dir / 'uploads'
        self.metadata_file = self.models_dir / 'model_metadata.json'
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _default_config(self) -> Dict:
        return {
            'target_columns': TARGET_COLS,
            'macro_columns': MACRO_COLS,
            'prediction_horizon': 6,
            'retrain_threshold_days': 30,
            'max_encoder_length': 12,
            'max_prediction_length': 6,
            'batch_size': 32,
            'learning_rate': 3e-3,
            'hidden_size': 64,
            'attention_head_size': 4,
            'dropout': 0.15,
            'hidden_continuous_size': 32,
            'max_epochs': 30,
            'fine_tune_epochs': 10,
            'fine_tune_lr': 1e-4,
            'optuna_trials': 20,
            'models_dir': 'models',
            'data_dir': 'data',
            'processed_data_file': 'data/processed/master_panel.csv',
            'output_prediction_file': 'data/predictions/forecast_6m.parquet',
        }

    # -------------------------------------------------------------------------
    # Data / Metadata helpers
    # -------------------------------------------------------------------------

    def load_latest_data(self) -> pd.DataFrame:
        path = Path(self.config['processed_data_file'])
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")
        df = pd.read_csv(path)
        df = df.loc[:, ~df.columns.duplicated()]
        df['periode'] = pd.to_datetime(df['periode'])
        df['kode_bank'] = df['kode_bank'].astype(str).str.strip()
        return df

    def load_model_metadata(self) -> Dict:
        if self.metadata_file.exists():
            with open(self.metadata_file) as f:
                meta = json.load(f)
        else:
            meta = {}

        # Normalize version to semver
        raw_ver = meta.get('version', '1.0.0')
        parts = str(raw_ver).split('.')
        while len(parts) < 3:
            parts.append('0')
        try:
            meta['version'] = f"{int(parts[0])}.{int(parts[1])}.{int(parts[2])}"
        except ValueError:
            meta['version'] = '1.0.0'

        meta.setdefault('last_trained', meta.get('trained_at'))  # normalize legacy field
        meta.setdefault('models', {})
        meta.setdefault('performance', {})
        return meta

    def save_model_metadata(self, metadata: Dict):
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

    def _get_existing_np_paths(self, metadata: Dict) -> Dict[str, str]:
        """Return existing flat NP model paths from metadata or models/ dir."""
        var_to_file = {
            'bi_rate': 'np_bi_rate.pkl',
            'inflasi': 'np_inflasi.pkl',
            'kurs_usd': 'np_kurs_usd.pkl',
        }
        paths: Dict[str, str] = {}
        for var, filename in var_to_file.items():
            # Check metadata first, then flat file
            meta_path = metadata.get('models', {}).get(f'np_{var}', '')
            flat_path = str(self.models_dir / filename)
            for candidate in [meta_path, flat_path]:
                if candidate and Path(candidate).exists():
                    paths[var] = candidate
                    break
        return paths

    def _clear_uploads(self):
        """Delete all data files in data/uploads/ after successful retraining."""
        if not self.uploads_dir.exists():
            return
        deleted_count = 0
        for f in self.uploads_dir.iterdir():
            if f.suffix.lower() in ('.csv', '.xlsx', '.parquet', '.xls'):
                try:
                    f.unlink()
                    deleted_count += 1
                    logger.info(f"Deleted upload file: {f.name}")
                except Exception as e:
                    logger.warning(f"Failed to delete {f.name}: {e}")
        if deleted_count > 0:
            logger.info(f"Cleared {deleted_count} upload files")
    
    def _archive_current_models(self, metadata: Dict) -> str:
        """Archive current models before retraining for rollback capability"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_dir = self.models_dir / 'versions' / f"v{metadata.get('version', '1.0.0')}_{timestamp}"
            archive_dir.mkdir(parents=True, exist_ok=True)
            
            # Archive key model files
            model_files = ['tft-epoch=00-val_loss=0.0507.ckpt', 'lgbm_reg_models.pkl', 
                          'lgbm_clf.pkl', 'np_bi_rate.pkl', 'np_inflasi.pkl', 'np_kurs_usd.pkl']
            for fname in model_files:
                src = self.models_dir / fname
                if src.exists():
                    shutil.copy2(src, archive_dir / fname)
            
            # Archive metadata
            shutil.copy2(self.metadata_file, archive_dir / 'metadata.json')
            
            logger.info(f"Archived models to {archive_dir}")
            return str(archive_dir)
        except Exception as e:
            logger.error(f"Failed to archive models: {e}")
            return None
    
    def _cleanup_old_versions(self, metadata: Dict):
        """Keep only last 3 versions + original, delete older versions"""
        try:
            versions_dir = self.models_dir / 'versions'
            if not versions_dir.exists():
                return
            
            # Get all version directories sorted by modification time
            version_dirs = sorted(versions_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
            
            # Keep last 3 versions (plus original is handled separately)
            if len(version_dirs) > 3:
                for old_dir in version_dirs[3:]:
                    try:
                        shutil.rmtree(old_dir)
                        logger.info(f"Deleted old version: {old_dir.name}")
                    except Exception as e:
                        logger.warning(f"Failed to delete {old_dir.name}: {e}")
        except Exception as e:
            logger.error(f"Error cleaning up old versions: {e}")
    
    def rollback_to_version(self, version: str) -> bool:
        """Rollback to a specific model version"""
        try:
            version_dir = self.models_dir / 'versions' / version
            if not version_dir.exists():
                logger.error(f"Version directory not found: {version_dir}")
                return False
            
            # Restore model files
            model_files = ['tft-epoch=00-val_loss=0.0507.ckpt', 'lgbm_reg_models.pkl', 
                          'lgbm_clf.pkl', 'np_bi_rate.pkl', 'np_inflasi.pkl', 'np_kurs_usd.pkl']
            for fname in model_files:
                src = version_dir / fname
                if src.exists():
                    shutil.copy2(src, self.models_dir / fname)
            
            # Restore metadata
            metadata_src = version_dir / 'metadata.json'
            if metadata_src.exists():
                shutil.copy2(metadata_src, self.metadata_file)
            
            logger.info(f"Rolled back to version: {version}")
            return True
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False

    # -------------------------------------------------------------------------
    # Main pipeline
    # -------------------------------------------------------------------------

    def _merge_uploads_to_master(self):
        """
        Merges new data from data/uploads/ into the master panel CSV.
        Deduplicates by bank code and period.
        """
        master_path = Path(self.config['processed_data_file'])
        if not master_path.exists():
            return
            
        try:
            df_master = pd.read_csv(master_path)
            df_master['periode'] = pd.to_datetime(df_master['periode'])
            df_master['kode_bank'] = df_master['kode_bank'].astype(str).str.strip()
            
            new_data_list = []
            if not self.uploads_dir.exists():
                return
                
            for f in self.uploads_dir.iterdir():
                if f.suffix.lower() == '.csv':
                    new_data_list.append(pd.read_csv(f))
                elif f.suffix.lower() in ('.xlsx', '.xls'):
                    new_data_list.append(pd.read_excel(f))
                elif f.suffix.lower() == '.parquet':
                    new_data_list.append(pd.read_parquet(f))
            
            if not new_data_list:
                return
                
            df_new = pd.concat(new_data_list, ignore_index=True)
            df_new['periode'] = pd.to_datetime(df_new['periode'])
            df_new['kode_bank'] = df_new['kode_bank'].astype(str).str.strip()
            
            # Combine and dedup
            df_combined = pd.concat([df_master, df_new], ignore_index=True)
            # Keep most recent version if duplicates exist (new data overwrites old)
            df_combined = df_combined.drop_duplicates(subset=['kode_bank', 'periode'], keep='last')
            df_combined = df_combined.sort_values(['kode_bank', 'periode'])
            
            df_combined.to_csv(master_path, index=False)
            return True
        except Exception as e:
            print(f"Error merging uploads: {e}")
            return False

    # -------------------------------------------------------------------------
    # Main pipeline
    # -------------------------------------------------------------------------

    def run_retraining_pipeline(self, force_retrain: bool = False, progress_callback=None):
        _cb = progress_callback or (lambda p, m: None)
        
        previous_version = None
        archive_path = None

        try:
            _cb(0.01, "Memuat data...")
            df = self.load_latest_data()
            metadata = self.load_model_metadata()
            previous_version = metadata.get('version', '1.0.0')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Archive current models before retraining
            _cb(0.02, "Mengarsipkan model saat ini untuk rollback...")
            archive_path = self._archive_current_models(metadata)
            
            trainer = EnsembleTrainer(self.config)

            # ── Unified Ensemble Training (All Phases) ────────────────────────────
            _cb(0.05, "Memulai pipeline retraining ensemble...")
            results = trainer.train_ensemble(df, progress_callback=_cb)
            
            # Phase 4: Generate predictions
            _cb(0.86, "Phase 4/4 — Membuat prediksi ensemble...")
            np_paths = {k.replace('np_', ''): v for k, v in results['paths'].items() if k.startswith('np_')}
            sarimax_paths = {k: v for k, v in results['paths'].items() if k.startswith('sarimax')}
            
            logger.info("Starting prediction generation...")
            
            # Create a wrapper callback to track target predictions
            target_count = [0]
            def pred_callback(msg):
                if '✓ Predicted' in msg:
                    target_count[0] += 1
                    _cb(0.86 + (target_count[0] * 0.01), msg)
            
            predictions = self._generate_all_predictions(
                df, results['paths'].get('tft'), np_paths, sarimax_paths, metadata, pred_callback
            )
            
            if predictions.empty:
                logger.error("❌ Prediction generation returned empty DataFrame!")
                _cb(0.90, "⚠️ Prediksi kosong, melanjutkan tanpa update prediksi...")
            else:
                logger.info(f"✓ Generated {len(predictions)} predictions, saving...")
                self._save_predictions(predictions, timestamp)
                _cb(0.93, "✓ Prediksi bank tersimpan.")
                
                # Generate and save macro forecasts
                logger.info("Generating macro forecasts...")
                self._save_macro_forecasts(df, timestamp)
                _cb(0.96, "✓ Prediksi makro tersimpan.")
                
            # Update metadata + bump version
            _cb(0.98, "Menyimpan metadata dan bump versi...")
            new_version = _bump_minor_version(metadata.get('version', '1.0.0'))
            
            self._update_metadata(metadata, timestamp, new_version, results)
            self.save_model_metadata(metadata)
            
            # ── Data Post-Processing ──────────────────────────────────────────
            _cb(0.99, "Merging data upload ke master panel...")
            self._merge_uploads_to_master()
            
            # Clear uploads ONLY after successful retraining
            self._clear_uploads()
            
            # Cleanup old versions (keep last 3)
            self._cleanup_old_versions(metadata)

            _cb(1.0, f"✓ Retraining selesai! Model versi {new_version} aktif.")
            logger.info(f"Retraining completed successfully. Version: {previous_version} → {new_version}")
            return True
            
        except Exception as e:
            import traceback
            err_msg = f"❌ Retraining gagal: {str(e)}"
            logger.error(f"Retraining failed: {err_msg}\n{traceback.format_exc()}")
            _cb(1.0, err_msg)
            
            # Automatic rollback on failure
            if archive_path and previous_version:
                try:
                    _cb(0.95, f"Melakukan rollback ke versi {previous_version}...")
                    if self.rollback_to_version(Path(archive_path).name):
                        logger.info(f"Automatic rollback to {previous_version} successful")
                        _cb(1.0, f"⚠️ Retraining gagal. Rollback ke versi {previous_version} berhasil.")
                    else:
                        logger.error("Automatic rollback failed")
                        _cb(1.0, f"❌ Retraining gagal dan rollback gagal. Hubungi administrator.")
                except Exception as rollback_err:
                    logger.error(f"Rollback error: {rollback_err}")
                    _cb(1.0, f"❌ Retraining gagal dan rollback error: {str(rollback_err)}")
            
            return False

    # -------------------------------------------------------------------------
    # Prediction generation
    # -------------------------------------------------------------------------

    def _generate_all_predictions(
        self,
        df: pd.DataFrame,
        new_tft_path: Optional[str],
        np_paths: Dict[str, str],
        sarimax_paths: Dict[str, str],
        metadata: Dict,
        callback=None,
    ) -> pd.DataFrame:
        """Generate ensemble predictions for all targets with progress tracking"""
        from src.rbbr_scoring import batch_score_rbbr
        
        _cb = callback or (lambda m: None)
        logger.info(f"Generating predictions from {len(df)} rows of training data...")

        model_paths: Dict = {
            'lgbm_reg': str(self.models_dir / 'lgbm_reg_models.pkl'),
            'lgbm_clf': str(self.models_dir / 'lgbm_clf.pkl'),
            'feat_cols': str(self.models_dir / 'feat_cols.pkl'),
            'np_bi_rate': np_paths.get('bi_rate', str(self.models_dir / 'np_bi_rate.pkl')),
            'np_inflasi': np_paths.get('inflasi', str(self.models_dir / 'np_inflasi.pkl')),
            'np_kurs_usd': np_paths.get('kurs_usd', str(self.models_dir / 'np_kurs_usd.pkl')),
            'sarimax_fc': sarimax_paths.get('sarimax_fc', str(self.models_dir / 'sarimax_fc.pkl')),
        }

        if new_tft_path and Path(new_tft_path).exists():
            model_paths['tft'] = new_tft_path
            logger.info(f"Using new TFT: {new_tft_path}")
        else:
            ckpts = sorted(self.models_dir.glob('tft-retrained-*.ckpt'))
            base_ckpt = self.models_dir / 'tft-epoch=00-val_loss=0.0507.ckpt'
            if ckpts:
                model_paths['tft'] = str(ckpts[-1])
                logger.info(f"Using existing TFT: {ckpts[-1]}")
            elif base_ckpt.exists():
                model_paths['tft'] = str(base_ckpt)
                logger.info(f"Using base TFT: {base_ckpt}")

        predictor = EnsemblePredictor(model_paths, self.config)
        
        # Pass callback to predictor
        all_preds = predictor.predict_all_targets(df, TARGET_COLS, self.config['prediction_horizon'], callback=_cb)

        if all_preds.empty:
            logger.warning("⚠️ Ensemble predictor returned empty predictions!")
            return pd.DataFrame()
        
        logger.info(f"✓ Generated {len(all_preds)} prediction rows")

        try:
            all_preds = batch_score_rbbr(all_preds)
            logger.info("✓ Applied RBBR scoring")
        except Exception as e:
            logger.warning(f"RBBR scoring failed: {e}")

        if 'bank' in df.columns:
            bank_meta = df[['kode_bank', 'bank']].drop_duplicates()
            bank_meta = bank_meta.rename(columns={'bank': 'nama_bank'})
            all_preds = pd.merge(all_preds, bank_meta, on='kode_bank', how='left')
            logger.info("✓ Merged bank metadata")

        return all_preds

    def _save_predictions(self, predictions: pd.DataFrame, timestamp: str):
        """Save predictions to multiple locations with logging"""
        logger.info(f"Saving {len(predictions)} prediction rows...")
        
        # Save to models/predictions directory
        pred_dir = self.models_dir / 'predictions'
        pred_dir.mkdir(exist_ok=True)
        csv_path = pred_dir / f"ensemble_forecast_{timestamp}.csv"
        predictions.to_csv(csv_path, index=False)
        logger.info(f"Saved CSV: {csv_path}")

        # Save to data/predictions/forecast_6m.parquet (main file)
        prod_path = Path(self.config['output_prediction_file'])
        prod_path.parent.mkdir(parents=True, exist_ok=True)
        
        predictions.to_parquet(prod_path, index=False)
        logger.info(f"✓ Saved forecast_6m.parquet: {prod_path} ({len(predictions)} rows)")
        
        # Also save as latest_predictions.parquet for backward compatibility
        latest_path = prod_path.parent / 'latest_predictions.parquet'
        predictions.to_parquet(latest_path, index=False)
        logger.info(f"✓ Saved latest_predictions.parquet: {latest_path}")

        # Upload to HuggingFace if available
        if self.storage_manager:
            try:
                self.storage_manager.upload_dataset(
                    file_path=str(prod_path),
                    destination_path=f"data/predictions/{prod_path.name}",
                    commit_message=f"Update predictions {timestamp}",
                )
                logger.info("✓ Uploaded to HuggingFace")
            except Exception as e:
                logger.warning(f"HF upload failed: {e}")
    
    def _save_macro_forecasts(self, df: pd.DataFrame, timestamp: str):
        """Generate and save macro forecasts to macro_fc.parquet"""
        logger.info("Generating macro forecasts...")
        
        try:
            # Load NeuralProphet models
            import torch
            np_models = {}
            for var in ['bi_rate', 'inflasi', 'kurs_usd']:
                model_path = self.models_dir / f'np_{var}.pkl'
                if model_path.exists():
                    try:
                        np_models[var] = torch.load(model_path, map_location='cpu', weights_only=False)
                        logger.info(f"Loaded NP model: {var}")
                    except Exception as e:
                        logger.warning(f"torch.load failed for {var} ({model_path}), trying pickle fallback: {e}")
                        try:
                            import pickle
                            with open(model_path, 'rb') as f:
                                np_models[var] = pickle.load(f)
                            logger.info(f"Loaded NP model via pickle: {var}")
                        except Exception as e2:
                            logger.error(f"Both torch and pickle failed to load NP model {var}: {e2}")
            
            if not np_models:
                logger.warning("No NeuralProphet models found, skipping macro forecasts")
                return
            
            # Generate forecasts
            horizon = self.config.get('prediction_horizon', 6)
            macro_forecasts = []
            
            for var, model in np_models.items():
                if var not in df.columns:
                    continue
                
                try:
                    # Prepare data
                    df_np = (
                        df[['periode', var]].copy()
                        .assign(ds=lambda x: pd.to_datetime(x['periode']), y=lambda x: x[var])
                        .groupby('ds')['y'].mean().reset_index().sort_values('ds').dropna(subset=['y'])
                    )
                    
                    # Generate forecast
                    future = model.make_future_dataframe(df_np, periods=horizon, n_historic_predictions=False)
                    forecast = model.predict(future)
                    last_hist = df_np['ds'].max()
                    fc = forecast[forecast['ds'] > last_hist].head(horizon)
                    
                    yhat_col = 'yhat1' if 'yhat1' in fc.columns else 'yhat'
                    
                    # Create forecast dataframe
                    for _, row in fc.iterrows():
                        macro_forecasts.append({
                            'periode': row['ds'],
                            'variable': var,
                            'forecast': row[yhat_col],
                            'lower': row.get('yhat_lower', row[yhat_col] * 0.95),
                            'upper': row.get('yhat_upper', row[yhat_col] * 1.05)
                        })
                    
                    logger.info(f"✓ Generated {var} forecast: {len(fc)} periods")
                    
                except Exception as e:
                    logger.error(f"Failed to forecast {var}: {e}")
            
            if macro_forecasts:
                macro_df = pd.DataFrame(macro_forecasts)
                
                # Save to data/predictions/macro_fc.parquet
                macro_path = Path('data/predictions/macro_fc.parquet')
                macro_path.parent.mkdir(parents=True, exist_ok=True)
                macro_df.to_parquet(macro_path, index=False)
                logger.info(f"✓ Saved macro_fc.parquet: {macro_path} ({len(macro_df)} rows)")
                
                # Also save CSV for inspection
                macro_df.to_csv(self.models_dir / 'predictions' / f'macro_forecast_{timestamp}.csv', index=False)
            else:
                logger.warning("No macro forecasts generated")
                
        except Exception as e:
            logger.error(f"Failed to save macro forecasts: {e}")

    # -------------------------------------------------------------------------
    # Metadata update
    # -------------------------------------------------------------------------

    def _update_metadata(
        self,
        metadata: Dict,
        timestamp: str,
        new_version: str,
        results: Dict,
    ):
        metadata['version'] = new_version
        metadata['last_trained'] = datetime.now().isoformat()
        metadata.setdefault('models', {})
        metadata.setdefault('performance', {})

        # Extract paths from results
        np_paths = {k.replace('np_', ''): v for k, v in results['paths'].items() if k.startswith('np_')}
        sarimax_paths = {k: v for k, v in results['paths'].items() if k.startswith('sarimax')}

        tft_path = results['paths'].get('tft')
        tft_result = results.get('tft', {})
        if tft_path:
            metadata['models']['tft'] = tft_path
            metadata['performance']['tft'] = {
                'timestamp': timestamp,
                'val_loss': float(tft_result.get('val_loss', 0)),
                'training_time': float(tft_result.get('training_time', 0)),
                'fine_tuned': tft_result.get('fine_tuned', False),
            }

        lgbm_paths = {k: v for k, v in results['paths'].items() if k.startswith('lgbm')}
        if lgbm_paths:
            metadata['models']['lgbm_reg'] = lgbm_paths.get('lgbm_reg', '')
            metadata['models']['lgbm_clf'] = lgbm_paths.get('lgbm_clf', '')
            
            lgbm_metrics = results.get('lgbm', {})
            if lgbm_metrics:
                metadata['performance']['lightgbm'] = {
                    'timestamp': timestamp,
                    'metrics': {t: float(v) for t, v in lgbm_metrics.items()},
                }

        np_perf = results.get('neuralprophet', {})
        if np_perf:
            metadata['performance']['neuralprophet'] = {
                'timestamp': timestamp,
                'metrics': {k: float(v) for k, v in np_perf.items()}
            }

        sarimax_perf = results.get('sarimax', {})
        if sarimax_perf:
            metadata['performance']['sarimax'] = {
                'timestamp': timestamp,
                'metrics': {k: float(v) for k, v in sarimax_perf.items()}
            }

        var_to_key = {'bi_rate': 'np_bi_rate', 'inflasi': 'np_inflasi', 'kurs_usd': 'np_kurs_usd'}
        for var, key in var_to_key.items():
            if var in np_paths:
                metadata['models'][key] = np_paths[var]

        if 'sarimax_fc' in sarimax_paths:
            metadata['models']['sarimax_fc'] = sarimax_paths['sarimax_fc']
        if 'sarimax' in sarimax_paths:
            metadata['models']['sarimax'] = sarimax_paths['sarimax']
