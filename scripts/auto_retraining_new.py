"""
Automated Retraining Pipeline - FIXED VERSION
Incremental training with smart data merge, prediction generation from master panel,
verification, and cache invalidation
"""

import json
import shutil
import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data_merge import SmartDataMerger
from src.cache_manager import CacheManager

logger = logging.getLogger(__name__)


def _bump_minor_version(version_str: str) -> str:
    """Bump minor version following semantic versioning"""
    parts = str(version_str).strip().split('.')
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
        
        self.merger = SmartDataMerger(self.config['processed_data_file'])
        self.retraining_start_time = None

    def _default_config(self) -> Dict:
        from training.ensemble_trainer import TARGET_COLS, MACRO_COLS
        return {
            'target_columns': TARGET_COLS,
            'macro_columns': MACRO_COLS,
            'prediction_horizon': 6,
            'max_epochs': 30,
            'fine_tune_epochs': 10,
            'fine_tune_lr': 1e-4,
            'models_dir': 'models',
            'data_dir': 'data',
            'processed_data_file': 'data/processed/master_panel.csv',
            'output_prediction_file': 'data/predictions/forecast_6m.parquet',
        }

    # =========================================================================
    # Data & Metadata Management
    # =========================================================================

    def load_latest_data(self) -> pd.DataFrame:
        """Load master panel data using shared resolution logic"""
        from src.data_utils import load_master_panel, normalize_kode_bank
        
        df = load_master_panel(self.data_dir)
        if df.empty:
            raise FileNotFoundError(f"Master panel data not found in {self.data_dir}")
        
        df = df.loc[:, ~df.columns.duplicated()]
        df = normalize_kode_bank(df)
        return df

    def load_model_metadata(self) -> Dict:
        """Load model metadata"""
        if self.metadata_file.exists():
            with open(self.metadata_file) as f:
                meta = json.load(f)
        else:
            meta = {}

        raw_ver = meta.get('version', '1.0.0')
        parts = str(raw_ver).split('.')
        while len(parts) < 3:
            parts.append('0')
        try:
            meta['version'] = f"{int(parts[0])}.{int(parts[1])}.{int(parts[2])}"
        except ValueError:
            meta['version'] = '1.0.0'

        meta.setdefault('last_trained', meta.get('trained_at'))
        meta.setdefault('models', {})
        meta.setdefault('performance', {})
        return meta

    def save_model_metadata(self, metadata: Dict):
        """Save model metadata"""
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

    def _archive_current_artifacts(self, metadata: Dict) -> Optional[str]:
        """Archive all artifacts (models + data + metadata) before retraining"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_dir = self.models_dir / 'versions' / f"v{metadata.get('version', '1.0.0')}_{timestamp}"
            archive_dir.mkdir(parents=True, exist_ok=True)
            
            # Archive models from manifest
            model_manifest = metadata.get('models', {})
            for key, model_info in model_manifest.items():
                if isinstance(model_info, dict):
                    filename = model_info.get('file')
                else:
                    filename = model_info
                
                if filename:
                    src = self.models_dir / filename
                    if src.exists():
                        shutil.copy2(src, archive_dir / filename)
            
            # Archive metadata
            if self.metadata_file.exists():
                shutil.copy2(self.metadata_file, archive_dir / 'model_metadata.json')
            
            # Archive data artifacts
            data_artifacts = [
                (Path(self.config['processed_data_file']), 'master_panel.csv'),
                (self.data_dir / 'processed' / 'master_panel.parquet', 'master_panel.parquet'),
                (self.data_dir / 'predictions' / 'forecast_6m.parquet', 'forecast_6m.parquet'),
                (self.data_dir / 'macro_forecast' / 'macro_fc.parquet', 'macro_fc.parquet'),
            ]
            
            for src, dest_name in data_artifacts:
                if src.exists():
                    shutil.copy2(src, archive_dir / dest_name)
            
            logger.info(f"Archived all artifacts to {archive_dir}")
            return str(archive_dir)
        except Exception as e:
            logger.error(f"Failed to archive artifacts: {e}")
            return None

    def _cleanup_old_versions(self):
        """Keep only last 3 versions"""
        try:
            versions_dir = self.models_dir / 'versions'
            if not versions_dir.exists():
                return
            
            version_dirs = sorted(versions_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
            if len(version_dirs) > 3:
                for old_dir in version_dirs[3:]:
                    try:
                        shutil.rmtree(old_dir)
                        logger.info(f"Deleted old version: {old_dir.name}")
                    except Exception as e:
                        logger.warning(f"Failed to delete {old_dir.name}: {e}")
        except Exception as e:
            logger.error(f"Error cleaning up old versions: {e}")

    def rollback_to_version(self, version_dir_name: str) -> Tuple[bool, str]:
        """Rollback to a specific version (models + data + metadata)"""
        try:
            version_dir = self.models_dir / 'versions' / version_dir_name
            if not version_dir.exists():
                logger.error(f"Version directory not found: {version_dir}")
                return False, "Version directory not found"
            
            # Load metadata from version to get model manifest
            metadata_src = version_dir / 'model_metadata.json'
            if metadata_src.exists():
                with open(metadata_src, 'r') as f:
                    metadata = json.load(f)
                
                # Restore models from manifest
                model_manifest = metadata.get('models', {})
                for key, model_info in model_manifest.items():
                    if isinstance(model_info, dict):
                        filename = model_info.get('file')
                    else:
                        filename = model_info
                    
                    if filename:
                        src = version_dir / filename
                        if src.exists():
                            shutil.copy2(src, self.models_dir / filename)
                
                # Restore metadata
                shutil.copy2(metadata_src, self.metadata_file)
            
            # Restore data artifacts
            data_artifacts = [
                ('master_panel.csv', Path(self.config['processed_data_file'])),
                ('master_panel.parquet', self.data_dir / 'processed' / 'master_panel.parquet'),
                ('forecast_6m.parquet', self.data_dir / 'predictions' / 'forecast_6m.parquet'),
                ('macro_fc.parquet', self.data_dir / 'macro_forecast' / 'macro_fc.parquet'),
            ]
            
            for src_name, dest in data_artifacts:
                src = version_dir / src_name
                if src.exists():
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dest)
            
            logger.info(f"Rolled back all artifacts to version: {version_dir_name}")
            
            # Post-rollback activation flow
            try:
                import streamlit as st
                from src.cache_manager import CacheManager
                from src.data_loader import get_data_loader
                
                CacheManager.invalidate_all()
                loader = get_data_loader()
                loader.reload_all()
                
                if hasattr(st, 'session_state'):
                    st.session_state.force_reload = True
                    
                logger.info("Triggered post-rollback cache invalidation and app reload flags")
            except Exception as cache_e:
                logger.warning(f"Could not trigger cache invalidation: {cache_e}")

            # Post-rollback upload step
            upload_success, upload_msg = self._upload_artifacts_to_hf(version_dir_name)
            
            if not upload_success:
                warning_msg = f"Rollback succeeded locally, but cloud sync failed: {upload_msg}. This rollback is local-only and may be overwritten on the next sync."
                logger.warning(warning_msg)
                return True, warning_msg

            return True, ""
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False, f"Rollback failed locally: {str(e)}"

    def _clear_uploads(self):
        """Delete all data files in data/uploads/ locally and remotely"""
        if not self.uploads_dir.exists():
            return
        
        # Local cleanup
        deleted_count = 0
        for f in self.uploads_dir.iterdir():
            if f.suffix.lower() in ('.csv', '.xlsx', '.parquet', '.xls'):
                try:
                    f.unlink()
                    deleted_count += 1
                    logger.info(f"Deleted local upload file: {f.name}")
                except Exception as e:
                    logger.warning(f"Failed to delete local {f.name}: {e}")
        
        if deleted_count > 0:
            logger.info(f"Cleared {deleted_count} local upload files")
            
        # Remote cleanup
        self._cleanup_remote_uploads()

    def _cleanup_remote_uploads(self):
        """Delete remote staging directory on HF to prevent re-processing on restart"""
        if not self.storage_manager:
            return
            
        try:
            logger.info("Cleaning up remote staged uploads on Hugging Face...")
            success = self.storage_manager.delete_folder(
                "data/uploads/",
                commit_message="Cleanup processed staged uploads after successful retraining"
            )
            if success:
                logger.info("Successfully cleared remote data/uploads/ directory")
            else:
                logger.warning("Failed to clear remote data/uploads/ directory")
        except Exception as e:
            logger.error(f"Error cleaning up remote uploads: {e}")

    # =========================================================================
    # Data Merge & Validation
    # =========================================================================

    def _merge_and_validate_uploads(self, callback=None) -> Tuple[pd.DataFrame, List[str], Dict]:
        """
        Merge uploads using SmartDataMerger
        Returns: (merged_df, new_periods, gap_info)
        """
        _cb = callback or (lambda p, m: None)
        _cb(None, "Merging uploaded data with smart validation...")
        
        try:
            df_merged, new_periods, gap_info = self.merger.merge_uploads(self.uploads_dir)
            
            if new_periods:
                _cb(None, f"✓ Detected new periods: {new_periods[0]} to {new_periods[-1]}")
                logger.info(f"New periods detected: {new_periods}")
            
            if gap_info['gaps']:
                _cb(None, f"⚠️ Gaps detected: {gap_info['gaps']}")
                logger.warning(f"Gaps: {gap_info['gaps']}")
            
            return df_merged, new_periods, gap_info
            
        except Exception as e:
            logger.error(f"Error merging uploads: {e}")
            raise

    # =========================================================================
    # Prediction Generation (FROM MASTER PANEL, NOT TRAINING DATA)
    # =========================================================================

    def _generate_predictions_from_master_panel(
        self,
        df_master: pd.DataFrame,
        new_tft_path: Optional[str],
        np_paths: Dict[str, str],
        sarimax_paths: Dict[str, str],
        callback=None,
    ) -> pd.DataFrame:
        """
        Generate predictions from COMPLETE master panel
        Generates BOTH historical validation predictions AND 6-month forecasts
        """
        _cb = callback or (lambda p, m: None)
        _cb(None, "Generating predictions from master panel...")
        
        try:
            from training.ensemble_trainer import TARGET_COLS
            from training.ensemble_predictor import EnsemblePredictor
            
            # Get latest historical period
            latest_period = df_master['periode'].max()
            logger.info(f"Latest historical period: {latest_period}")
            
            # Build model paths
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

            # Generate predictions
            predictor = EnsemblePredictor(model_paths, self.config)
            
            # Generate predictions for 6 months ahead (forecast only, not historical)
            all_preds = predictor.predict_all_targets(
                df_master, TARGET_COLS, self.config['prediction_horizon'], callback=_cb
            )

            if all_preds.empty:
                logger.warning("⚠️ Ensemble predictor returned empty predictions!")
                return pd.DataFrame()
            
            logger.info(f"✓ Generated {len(all_preds)} prediction rows")

            # Apply RBBR scoring
            try:
                from src.rbbr_scoring import batch_score_rbbr
                all_preds = batch_score_rbbr(all_preds)
                logger.info("✓ Applied RBBR scoring")
            except Exception as e:
                logger.warning(f"RBBR scoring failed: {e}")

            # Merge bank metadata
            if 'bank' in df_master.columns or 'nama_bank' in df_master.columns:
                from src.data_utils import resolve_canonical_bank_identity
                df_master = resolve_canonical_bank_identity(df_master)
                bank_col = 'nama_bank' if 'nama_bank' in df_master.columns else 'bank'
                bank_meta = df_master[['kode_bank', bank_col]].drop_duplicates(subset=['kode_bank'])
                bank_meta = bank_meta.rename(columns={bank_col: 'nama_bank'})
                all_preds = pd.merge(all_preds, bank_meta, on='kode_bank', how='left')
                logger.info("✓ Merged bank metadata")

            # IMPORTANT: Also include historical data from master panel in predictions
            # This ensures dashboard shows continuous data from historical to forecast
            historical_preds = df_master[['kode_bank', 'periode', 'CAR', 'NPL_gross', 'NPL_net', 'ROA', 'ROE', 'NIM', 'BOPO', 'LDR']].copy()
            
            # Add required columns for consistency
            if 'nama_bank' in df_master.columns:
                historical_preds['nama_bank'] = df_master['nama_bank']
            
            # Apply RBBR scoring to historical data
            try:
                historical_preds = batch_score_rbbr(historical_preds)
            except Exception as e:
                logger.warning(f"RBBR scoring on historical data failed: {e}")
            
            # Add is_forecast discriminator
            latest_period = df_master['periode'].max()
            historical_preds['is_forecast'] = False
            all_preds['is_forecast'] = True
            
            # Combine historical + forecast predictions
            combined_preds = pd.concat([historical_preds, all_preds], ignore_index=True)
            combined_preds = combined_preds.drop_duplicates(subset=['kode_bank', 'periode'], keep='last')
            combined_preds = combined_preds.sort_values(['kode_bank', 'periode'])
            
            logger.info(f"✓ Combined {len(historical_preds)} historical + {len(all_preds)} forecast = {len(combined_preds)} total predictions")
            logger.info(f"✓ Latest historical period: {latest_period.strftime('%Y-%m')}")

            return combined_preds

        except Exception as e:
            logger.error(f"Error generating predictions: {e}")
            raise

    def _save_predictions(self, predictions: pd.DataFrame, timestamp: str) -> Path:
        """Save predictions to temporary path, returns temp path for transactional commit"""
        try:
            temp_file = self.data_dir / 'predictions' / f'forecast_6m_{timestamp}.tmp.parquet'
            temp_file.parent.mkdir(parents=True, exist_ok=True)
            
            predictions.to_parquet(temp_file, index=False)
            logger.info(f"Saved {len(predictions)} predictions to temp: {temp_file}")
            return temp_file
            
        except Exception as e:
            logger.error(f"Error saving predictions: {e}")
            raise

    def _save_macro_forecasts(self, df_master: pd.DataFrame, np_paths: Dict[str, str], timestamp: str) -> Path:
        """Generate and save macro forecasts to temporary path, returns temp path for transactional commit"""
        try:
            temp_file = self.data_dir / 'macro_forecast' / f'macro_fc_{timestamp}.tmp.parquet'
            temp_file.parent.mkdir(parents=True, exist_ok=True)
            
            import torch
            np_models = {}
            for var in ['bi_rate', 'inflasi', 'kurs_usd']:
                np_path = np_paths.get(var)
                if np_path and Path(np_path).exists():
                    try:
                        loaded_model = torch.load(np_path, map_location='cpu', weights_only=False)
                        if hasattr(loaded_model, 'trainer'):
                            loaded_model.trainer = None
                        np_models[var] = loaded_model
                        logger.info(f"Loaded NeuralProphet model for {var}")
                    except Exception as e:
                        logger.warning(f"Failed to load NP model for {var}: {e}")
            
            if not np_models:
                logger.warning("No NeuralProphet models loaded, cannot generate macro forecasts")
                raise ValueError("No NeuralProphet models available")
            
            horizon = self.config['prediction_horizon']
            latest_period = df_master['periode'].max()
            forecast_periods = pd.date_range(
                start=latest_period + pd.DateOffset(months=1),
                periods=horizon,
                freq='MS'
            )
            
            macro_forecasts = {'periode': forecast_periods}
            
            for var, model in np_models.items():
                if var not in df_master.columns:
                    logger.warning(f"Variable {var} not in master data")
                    continue
                
                try:
                    df_np = (
                        df_master[['periode', var]].copy()
                        .assign(ds=lambda x: pd.to_datetime(x['periode']), y=lambda x: x[var])
                        .groupby('ds')['y'].mean().reset_index()
                        .sort_values('ds').dropna(subset=['y'])
                    )
                    
                    future = model.make_future_dataframe(df_np, periods=horizon, n_historic_predictions=False)
                    forecast = model.predict(future)
                    
                    last_hist = df_np['ds'].max()
                    fc = forecast[forecast['ds'] > last_hist].head(horizon)
                    yhat_col = 'yhat1' if 'yhat1' in fc.columns else 'yhat'
                    
                    macro_forecasts[f'{var}_fc'] = fc[yhat_col].values[:horizon]
                    logger.info(f"Generated {len(fc)} forecasts for {var}")
                    
                except Exception as e:
                    logger.warning(f"Failed to generate forecast for {var}: {e}")
                    last_val = df_master[var].dropna().iloc[-1] if not df_master[var].dropna().empty else 0.0
                    macro_forecasts[f'{var}_fc'] = [float(last_val)] * horizon
            
            df_macro_fc = pd.DataFrame(macro_forecasts)
            df_macro_fc.to_parquet(temp_file, index=False)
            logger.info(f"Saved macro forecasts to temp: {temp_file} ({len(df_macro_fc)} periods)")
            return temp_file
            
        except Exception as e:
            logger.error(f"Error generating macro forecasts: {e}")
            raise

    # =========================================================================
    # Verification & Validation
    # =========================================================================

    def _commit_temp_files(
        self,
        temp_master_path: Path,
        temp_forecast_path: Path,
        temp_macro_path: Path,
        callback=None
    ):
        """Atomically replace live files with validated temporary files"""
        _cb = callback or (lambda p, m: None)
        _cb(None, "Committing validated artifacts...")
        
        try:
            # Master panel
            master_file = Path(self.config['processed_data_file'])
            master_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(temp_master_path), str(master_file))
            logger.info(f"Committed master panel: {master_file}")
            
            # Forecast
            forecast_file = self.data_dir / 'predictions' / 'forecast_6m.parquet'
            forecast_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(temp_forecast_path), str(forecast_file))
            logger.info(f"Committed forecast: {forecast_file}")
            
            # Macro forecast
            macro_file = self.data_dir / 'macro_forecast' / 'macro_fc.parquet'
            macro_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(temp_macro_path), str(macro_file))
            logger.info(f"Committed macro forecast: {macro_file}")
            
            _cb(None, "✓ All artifacts committed")
            
        except Exception as e:
            logger.error(f"Error committing temp files: {e}")
            raise

    def _stage_version(
        self,
        version: str,
        temp_master_path: Path,
        temp_forecast_path: Path,
        temp_macro_path: Path,
        metadata: Dict
    ) -> Path:
        """Stage all outputs into version directory before committing to live paths"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            stage_dir = self.models_dir / 'versions' / f"v{version}_{timestamp}_staged"
            stage_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy models from manifest to stage
            model_manifest = metadata.get('models', {})
            for key, model_info in model_manifest.items():
                if isinstance(model_info, dict):
                    filename = model_info.get('file')
                else:
                    filename = model_info
                
                if filename:
                    src = self.models_dir / filename
                    if src.exists():
                        shutil.copy2(src, stage_dir / filename)
                        logger.info(f"Staged model: {filename}")
            
            # Copy temp data artifacts to stage
            shutil.copy2(temp_master_path, stage_dir / 'master_panel.csv')
            shutil.copy2(temp_forecast_path, stage_dir / 'forecast_6m.parquet')
            shutil.copy2(temp_macro_path, stage_dir / 'macro_fc.parquet')
            
            # Save metadata to stage
            with open(stage_dir / 'model_metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            logger.info(f"Staged version to {stage_dir}")
            return stage_dir
            
        except Exception as e:
            logger.error(f"Error staging version: {e}")
            raise

    def _promote_staged_version(self, stage_dir: Path, callback=None):
        """Promote staged version to live paths atomically"""
        _cb = callback or (lambda p, m: None)
        _cb(None, "Promoting staged version to live...")
        
        try:
            # Load metadata from staged version to get model manifest
            metadata_src = stage_dir / 'model_metadata.json'
            if metadata_src.exists():
                with open(metadata_src, 'r') as f:
                    metadata = json.load(f)
                
                # Promote models from manifest
                model_manifest = metadata.get('models', {})
                for key, model_info in model_manifest.items():
                    if isinstance(model_info, dict):
                        filename = model_info.get('file')
                    else:
                        filename = model_info
                    
                    if filename:
                        src = stage_dir / filename
                        if src.exists():
                            shutil.copy2(src, self.models_dir / filename)
                            logger.info(f"Promoted model: {filename}")
                
                # Promote metadata
                shutil.copy2(metadata_src, self.metadata_file)
            else:
                logger.error("Metadata not found in staged version")
            
            # Promote data artifacts
            data_artifacts = [
                ('master_panel.csv', Path(self.config['processed_data_file'])),
                ('forecast_6m.parquet', self.data_dir / 'predictions' / 'forecast_6m.parquet'),
                ('macro_fc.parquet', self.data_dir / 'macro_forecast' / 'macro_fc.parquet'),
            ]
            
            for src_name, dest in data_artifacts:
                src = stage_dir / src_name
                if src.exists():
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dest)
            
            # Also create and promote Parquet format from CSV for consistency
            master_csv = Path(self.config['processed_data_file'])
            if master_csv.exists():
                try:
                    import pandas as pd
                    df = pd.read_csv(master_csv)
                    parquet_dest = master_csv.parent / 'master_panel.parquet'
                    df.to_parquet(parquet_dest, index=False)
                    logger.info(f"Created and promoted master_panel.parquet")
                except Exception as e:
                    logger.warning(f"Could not create Parquet format locally: {e}")
            
            logger.info(f"Promoted staged version from {stage_dir}")
            _cb(None, "✓ Version promoted to live")
            
        except Exception as e:
            logger.error(f"Error promoting staged version: {e}")
            raise

    def _verify_retraining_success(
        self, 
        merged_max_period: pd.Timestamp,
        temp_master_path: Path,
        temp_forecast_path: Path,
        temp_macro_path: Path,
        callback=None
    ) -> Tuple[bool, str]:
        """
        Verify retraining success with strict checks:
        - Master panel advanced to merged_max_period
        - Forecast rows strictly after merged_max_period
        - Exactly prediction_horizon future months
        - All artifacts newer than retraining_start_time
        """
        _cb = callback or (lambda p, m: None)
        _cb(None, "Verifying retraining success...")
        
        try:
            checks = []
            
            # Check 1: Temp master panel exists, non-empty, and advanced to merged_max_period
            if not temp_master_path.exists():
                return False, f"Master panel temp file missing: {temp_master_path}"
            
            master = pd.read_csv(temp_master_path)
            master['periode'] = pd.to_datetime(master['periode'])
            
            # Normalize kode_bank for consistent comparison
            from src.data_utils import normalize_kode_bank
            master = normalize_kode_bank(master)
            
            master_max = master['periode'].max()
            
            if len(master) == 0:
                return False, "Master panel is empty"
            
            if master_max < merged_max_period:
                return False, f"Master panel not advanced: max={master_max}, expected={merged_max_period}"
            
            checks.append(("Master panel advanced", True))
            _cb(None, f"✓ Master panel advanced to {master_max.strftime('%Y-%m')}")
            
            # Check 2: Temp forecast exists, non-empty, with rows strictly after merged_max_period
            if not temp_forecast_path.exists():
                return False, f"Forecast temp file missing: {temp_forecast_path}"
            
            forecast = pd.read_parquet(temp_forecast_path)
            forecast['periode'] = pd.to_datetime(forecast['periode'])
            forecast = normalize_kode_bank(forecast)
            
            if len(forecast) == 0:
                return False, "Forecast is empty"
            
            # Filter to forecast-only rows
            forecast_only = forecast[forecast['is_forecast'] == True] if 'is_forecast' in forecast.columns else forecast
            
            if len(forecast_only) == 0:
                return False, "No forecast rows generated"
            
            forecast_min = forecast_only['periode'].min()
            forecast_max = forecast_only['periode'].max()
            
            if forecast_min <= merged_max_period:
                return False, f"Forecast not strictly after master: forecast_min={forecast_min}, master_max={merged_max_period}"
            
            checks.append(("Forecast after master", True))
            _cb(None, f"✓ Forecast periods: {forecast_min.strftime('%Y-%m')} to {forecast_max.strftime('%Y-%m')}")
            
            # Check 3: Exactly prediction_horizon future months
            expected_periods = pd.date_range(
                start=merged_max_period + pd.DateOffset(months=1),
                periods=self.config['prediction_horizon'],
                freq='MS'
            )
            
            forecast_periods = forecast_only['periode'].unique()
            forecast_periods = pd.to_datetime(forecast_periods)
            
            if len(forecast_periods) != self.config['prediction_horizon']:
                return False, f"Wrong forecast horizon: got {len(forecast_periods)}, expected {self.config['prediction_horizon']}"
            
            # Check periods match expected
            missing = set(expected_periods) - set(forecast_periods)
            if missing:
                return False, f"Missing forecast periods: {[p.strftime('%Y-%m') for p in missing]}"
            
            checks.append(("Correct forecast horizon", True))
            _cb(None, f"✓ Forecast horizon: {self.config['prediction_horizon']} months")
            
            # Check 4: Per-bank forecast completeness
            _cb(None, "Checking per-bank forecast completeness...")
            
            # Get unique banks from master panel
            master_banks = set(master['kode_bank'].unique())
            forecast_banks = set(forecast_only['kode_bank'].unique())
            
            # Check for missing banks
            missing_banks = master_banks - forecast_banks
            if missing_banks:
                missing_list = sorted(list(missing_banks))
                return False, f"Banks missing from forecast: {missing_list} ({len(missing_list)} banks)"
            
            # Check each bank has all forecast periods
            banks_missing_periods = []
            for bank in master_banks:
                bank_forecast = forecast_only[forecast_only['kode_bank'] == bank]
                bank_periods = set(pd.to_datetime(bank_forecast['periode'].unique()))
                missing_periods = set(expected_periods) - bank_periods
                if missing_periods:
                    missing_str = [p.strftime('%Y-%m') for p in sorted(missing_periods)]
                    banks_missing_periods.append(f"{bank}: {missing_str}")
            
            if banks_missing_periods:
                return False, f"Banks with incomplete forecast periods:\n" + "\n".join(banks_missing_periods[:10])
            
            # Check required targets are not null for any bank-period
            from training.ensemble_trainer import TARGET_COLS
            required_targets = [t for t in TARGET_COLS if t in forecast_only.columns]
            
            banks_with_nulls = []
            for bank in master_banks:
                bank_forecast = forecast_only[forecast_only['kode_bank'] == bank]
                for target in required_targets:
                    null_periods = bank_forecast[bank_forecast[target].isnull()]['periode']
                    if len(null_periods) > 0:
                        null_str = [p.strftime('%Y-%m') for p in null_periods]
                        banks_with_nulls.append(f"{bank}/{target}: {null_str}")
            
            if banks_with_nulls:
                return False, f"Banks with null forecast targets:\n" + "\n".join(banks_with_nulls[:10])
            
            checks.append(("Per-bank forecast complete", True))
            _cb(None, f"✓ All {len(master_banks)} banks have complete forecasts")
            
            # Check 5: Temp macro forecast exists with exactly prediction_horizon periods
            if not temp_macro_path.exists():
                return False, f"Macro forecast temp file missing: {temp_macro_path}"
            
            macro_fc = pd.read_parquet(temp_macro_path)
            
            if len(macro_fc) != self.config['prediction_horizon']:
                return False, f"Wrong macro forecast periods: got {len(macro_fc)}, expected {self.config['prediction_horizon']}"
            
            required_cols = ['periode', 'bi_rate_fc', 'inflasi_fc', 'kurs_usd_fc']
            missing_cols = set(required_cols) - set(macro_fc.columns)
            if missing_cols:
                return False, f"Macro forecast missing columns: {missing_cols}"
            
            checks.append(("Macro forecast complete", True))
            _cb(None, f"✓ Macro forecast: {len(macro_fc)} periods")
            
            # Check 6: All artifacts newer than retraining_start_time
            if self.retraining_start_time:
                for path, name in [
                    (temp_master_path, "master panel"),
                    (temp_forecast_path, "forecast"),
                    (temp_macro_path, "macro forecast")
                ]:
                    mtime = datetime.fromtimestamp(path.stat().st_mtime)
                    if mtime < self.retraining_start_time:
                        return False, f"{name} is stale: modified {mtime}, retraining started {self.retraining_start_time}"
                
                checks.append(("Artifacts fresh", True))
                _cb(None, "✓ All artifacts are fresh")
            
            return True, "All verification checks passed"
                
        except Exception as e:
            logger.error(f"Error verifying retraining: {e}")
            return False, f"Verification error: {str(e)}"

    # =========================================================================
    # Hugging Face Synchronization
    # =========================================================================

    def _upload_artifacts_to_hf(self, version: str, callback=None) -> Tuple[bool, str]:
        """Upload finalized artifacts to Hugging Face"""
        _cb = callback or (lambda p, m: None)
        
        if not self.storage_manager:
            logger.warning("No storage_manager available, skipping HF upload")
            return False, "No storage manager available"
        
        try:
            # Upload master panel (CSV)
            master_file = Path(self.config['processed_data_file'])
            if master_file.exists():
                self.storage_manager.upload_file(
                    str(master_file),
                    f"data/processed/{master_file.name}",
                    commit_message=f"Update master panel - version {version}"
                )
                logger.info(f"Uploaded {master_file.name} to HF")
                
                # Also save and upload Parquet format for consistency
                try:
                    import pandas as pd
                    df = pd.read_csv(master_file)
                    parquet_file = master_file.parent / 'master_panel.parquet'
                    df.to_parquet(parquet_file, index=False)
                    
                    self.storage_manager.upload_file(
                        str(parquet_file),
                        f"data/processed/{parquet_file.name}",
                        commit_message=f"Update master panel (parquet) - version {version}"
                    )
                    logger.info(f"Uploaded {parquet_file.name} to HF")
                except Exception as e:
                    logger.warning(f"Could not create/upload Parquet format: {e}")
            
            # Upload forecast
            forecast_file = self.data_dir / 'predictions' / 'forecast_6m.parquet'
            if forecast_file.exists():
                self.storage_manager.upload_file(
                    str(forecast_file),
                    f"data/predictions/{forecast_file.name}",
                    commit_message=f"Update forecast - version {version}"
                )
                logger.info(f"Uploaded {forecast_file.name} to HF")
            
            # Upload macro forecast
            macro_file = self.data_dir / 'macro_forecast' / 'macro_fc.parquet'
            if macro_file.exists():
                self.storage_manager.upload_file(
                    str(macro_file),
                    f"data/macro_forecast/{macro_file.name}",
                    commit_message=f"Update macro forecast - version {version}"
                )
                logger.info(f"Uploaded {macro_file.name} to HF")
            
            # Upload model metadata
            if self.metadata_file.exists():
                self.storage_manager.upload_file(
                    str(self.metadata_file),
                    f"models/{self.metadata_file.name}",
                    commit_message=f"Update model metadata - version {version}"
                )
                logger.info(f"Uploaded {self.metadata_file.name} to HF")
            
            # Upload model binaries from manifest
            try:
                with open(self.metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                model_manifest = metadata.get('models', {})
                for key, model_info in model_manifest.items():
                    if isinstance(model_info, dict):
                        filename = model_info.get('file')
                    else:
                        filename = model_info
                    
                    if filename:
                        model_file = self.models_dir / filename
                        if model_file.exists():
                            self.storage_manager.upload_file(
                                str(model_file),
                                f"models/{filename}",
                                commit_message=f"Update {key} model - version {version}"
                            )
                            logger.info(f"Uploaded model binary: {filename}")
            except Exception as e:
                logger.warning(f"Could not upload model binaries: {e}")
            
            _cb(None, "✓ Artifacts uploaded to Hugging Face")
            return True, ""
            
        except Exception as e:
            logger.error(f"Error uploading to HF: {e}")
            # Don't fail the retraining if upload fails
            _cb(None, f"⚠️ HF upload failed: {str(e)}")
            return False, f"HF upload failed: {str(e)}"

    # =========================================================================
    # Main Retraining Pipeline
    # =========================================================================

    def run_retraining_pipeline(self, force_retrain: bool = False, progress_callback=None) -> bool:
        """
        Main retraining pipeline with incremental training, smart merge, and verification
        """
        _cb = progress_callback or (lambda p, m: None)
        
        previous_version = None
        archive_path = None
        new_periods = []
        temp_master_path = None
        temp_forecast_path = None
        temp_macro_path = None
        stage_dir = None

        try:
            self.retraining_start_time = datetime.now()
            
            _cb(0.01, "Memuat data...")
            df_master = self.load_latest_data()
            metadata = self.load_model_metadata()
            previous_version = metadata.get('version', '1.0.0')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Archive current artifacts (models + data + metadata)
            _cb(0.02, "Mengarsipkan artifacts saat ini untuk rollback...")
            archive_path = self._archive_current_artifacts(metadata)
            
            # Merge uploads with smart validation
            _cb(0.05, "Merging data upload dengan validasi...")
            df_merged, new_periods, gap_info = self._merge_and_validate_uploads(
                callback=lambda p, m: _cb(0.10, m)
            )
            
            if not new_periods:
                _cb(0.15, "⚠️ Tidak ada data baru terdeteksi")
                logger.warning("No new periods detected")
            
            merged_max_period = df_merged['periode'].max()
            logger.info(f"Merged data max period: {merged_max_period}")
            
            # Save merged data to temporary path
            _cb(0.15, "Menyimpan data yang sudah dimerge...")
            temp_master_path = Path(self.config['processed_data_file']).parent / f'master_panel_{timestamp}.tmp.csv'
            temp_master_path.parent.mkdir(parents=True, exist_ok=True)
            df_merged.to_csv(temp_master_path, index=False)
            logger.info(f"Saved merged data to temp: {temp_master_path}")
            
            # Train ensemble (incremental)
            _cb(0.20, "Memulai pelatihan ensemble (incremental)...")
            from training.ensemble_trainer import EnsembleTrainer
            trainer = EnsembleTrainer(self.config)
            results = trainer.train_ensemble(df_merged, progress_callback=_cb)
            
            # Generate predictions from MASTER PANEL (not training data)
            _cb(0.85, "Membuat prediksi dari master panel...")
            np_paths = {k.replace('np_', ''): v for k, v in results['paths'].items() if k.startswith('np_')}
            sarimax_paths = {k: v for k, v in results['paths'].items() if k.startswith('sarimax')}
            
            predictions = self._generate_predictions_from_master_panel(
                df_merged, results['paths'].get('tft'), np_paths, sarimax_paths,
                callback=lambda p, m: _cb(0.90, m)
            )
            
            if predictions.empty:
                raise Exception("Prediction generation returned empty DataFrame")
            
            _cb(0.92, f"✓ Generated {len(predictions)} predictions")
            
            # Save to temporary paths
            temp_forecast_path = self._save_predictions(predictions, timestamp)
            temp_macro_path = self._save_macro_forecasts(df_merged, np_paths, timestamp)
            
            # Verify retraining success with strict checks
            _cb(0.95, "Verifying retraining success...")
            success, verify_msg = self._verify_retraining_success(
                merged_max_period,
                temp_master_path,
                temp_forecast_path,
                temp_macro_path,
                callback=lambda p, m: _cb(0.97, m)
            )
            
            if not success:
                logger.error(f"Verification failed: {verify_msg}")
                raise Exception(verify_msg)
            
            # Update metadata
            new_version = _bump_minor_version(metadata.get('version', '1.0.0'))
            metadata['version'] = new_version
            metadata['last_trained'] = datetime.now().isoformat()
            metadata['new_periods'] = new_periods
            metadata['forecast_periods'] = self._get_forecast_periods(df_merged)
            
            # Update model manifest with active filenames
            metadata['models'] = self._build_model_manifest(results['paths'])
            metadata['performance'] = {
                'tft': results.get('tft', {}),
                'lightgbm': {'metrics': results.get('lgbm', {})},
                'neuralprophet': {'metrics': results.get('neuralprophet', {})},
                'sarimax': {'metrics': results.get('sarimax', {})}
            }
            
            # Stage complete version (models + data + metadata)
            _cb(0.98, "Staging complete version...")
            stage_dir = self._stage_version(
                new_version,
                temp_master_path,
                temp_forecast_path,
                temp_macro_path,
                metadata
            )
            
            # Promote staged version to live paths atomically
            _cb(0.98, "Promoting to live...")
            self._promote_staged_version(stage_dir, callback=_cb)
            
            # Invalidate cache to force reload of new data
            _cb(0.98, "Invalidating cache...")
            CacheManager.invalidate_all()
            
            # Also clear data loader cache
            try:
                from src.data_loader import get_data_loader
                loader = get_data_loader()
                loader.reload_all()
                logger.info("Data loader cache cleared")
            except Exception as cache_err:
                logger.warning(f"Could not clear data loader cache: {cache_err}")
            
            # Upload artifacts to Hugging Face if storage_manager available
            if self.storage_manager:
                _cb(0.99, "Uploading artifacts to Hugging Face...")
                self._upload_artifacts_to_hf(new_version, callback=_cb)
            
            # Clear uploads
            self._clear_uploads()
            
            # Cleanup old versions
            self._cleanup_old_versions()
            
            _cb(1.0, f"✓ Retraining selesai! Model versi {new_version} aktif.")
            logger.info(f"Retraining completed successfully. Version: {previous_version} → {new_version}")
            
            return True
            
        except Exception as e:
            import traceback
            err_msg = f"❌ Retraining gagal: {str(e)}"
            logger.error(f"Retraining failed: {err_msg}\n{traceback.format_exc()}")
            _cb(1.0, err_msg)
            
            # Clean up temporary files
            for temp_path in [temp_master_path, temp_forecast_path, temp_macro_path]:
                if temp_path and temp_path.exists():
                    try:
                        temp_path.unlink()
                        logger.info(f"Cleaned up temp file: {temp_path}")
                    except Exception as cleanup_err:
                        logger.warning(f"Failed to clean up {temp_path}: {cleanup_err}")
            
            # Clean up staged version
            if stage_dir and stage_dir.exists():
                try:
                    shutil.rmtree(stage_dir)
                    logger.info(f"Cleaned up staged version: {stage_dir}")
                except Exception as cleanup_err:
                    logger.warning(f"Failed to clean up {stage_dir}: {cleanup_err}")
            
            # Automatic rollback on failure (restores models + data + metadata)
            if archive_path and previous_version:
                try:
                    _cb(0.95, f"Melakukan rollback ke versi {previous_version}...")
                    archive_name = Path(archive_path).name
                    success, rb_msg = self.rollback_to_version(archive_name)
                    if success:
                        logger.info(f"Automatic rollback to {previous_version} successful. {rb_msg}")
                        msg = f"⚠️ Retraining gagal. Rollback ke versi {previous_version} berhasil."
                        if rb_msg:
                            msg += f" ({rb_msg})"
                        _cb(1.0, msg)
                    else:
                        logger.error(f"Automatic rollback failed: {rb_msg}")
                        _cb(1.0, f"❌ Retraining gagal dan rollback gagal: {rb_msg}. Hubungi administrator.")
                except Exception as rollback_err:
                    logger.error(f"Rollback error: {rollback_err}")
                    _cb(1.0, f"❌ Retraining gagal dan rollback error: {str(rollback_err)}")
            
            return False

    def _build_model_manifest(self, model_paths: Dict[str, str]) -> Dict:
        """Build model manifest from training results"""
        manifest = {}
        
        # TFT
        if 'tft' in model_paths and model_paths['tft']:
            tft_file = Path(model_paths['tft']).name
            manifest['tft'] = {
                'file': tft_file,
                'description': 'Temporal Fusion Transformer — time-series forecasting'
            }
        
        # LightGBM
        if 'lgbm_reg' in model_paths:
            manifest['lgbm_reg'] = {
                'file': Path(model_paths['lgbm_reg']).name,
                'description': 'LightGBM regressor — prediksi skor RBBR kontinu',
                'features': 'feat_cols.pkl'
            }
        
        if 'lgbm_clf' in model_paths:
            manifest['lgbm_clf'] = {
                'file': Path(model_paths['lgbm_clf']).name,
                'description': 'LightGBM classifier — probabilitas per Peringkat Komposit'
            }
        
        if 'feat_cols' in model_paths:
            manifest['feat_cols'] = {
                'file': Path(model_paths['feat_cols']).name,
                'description': 'Feature columns for LightGBM'
            }
        
        # NeuralProphet
        for var in ['bi_rate', 'inflasi', 'kurs_usd']:
            key = f'np_{var}'
            if key in model_paths:
                manifest[key] = {
                    'file': Path(model_paths[key]).name,
                    'description': f'NeuralProphet — forecast {var}'
                }
        
        # SARIMAX
        if 'sarimax_fc' in model_paths:
            manifest['sarimax_fc'] = {
                'file': Path(model_paths['sarimax_fc']).name,
                'description': 'SARIMAX — per-bank per-rasio forecasting'
            }
        
        if 'sarimax' in model_paths:
            manifest['sarimax'] = {
                'file': Path(model_paths['sarimax']).name,
                'description': 'SARIMAX — aggregated forecasting'
            }
        
        return manifest

    def _get_forecast_periods(self, df_master: pd.DataFrame) -> List[str]:
        """Get forecast periods (6 months from latest historical)"""
        try:
            latest = df_master['periode'].max()
            periods = []
            for i in range(1, 7):
                if latest.month + i <= 12:
                    next_period = latest.replace(month=latest.month + i)
                else:
                    next_period = latest.replace(year=latest.year + 1, month=(latest.month + i) % 12)
                periods.append(next_period.strftime('%Y-%m'))
            return periods
        except Exception as e:
            logger.error(f"Error getting forecast periods: {e}")
            return []
