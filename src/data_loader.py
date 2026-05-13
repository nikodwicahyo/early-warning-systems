"""
Data Loader Module
Handles loading of data, predictions, and models with caching
"""

import pandas as pd
import pickle
import logging
from pathlib import Path
from typing import Dict, Optional, List
import streamlit as st

from .config import DATA_DIR, MODEL_DIR, ModelConfig

logger = logging.getLogger(__name__)

# =============================================================================
# DATA LOADER CLASS
# =============================================================================

class DataLoader:
    """Centralized data loading with caching"""
    
    def __init__(self):
        self.data_dir = DATA_DIR
        self.model_dir = MODEL_DIR
        self.config = ModelConfig()
        self._synced = False  # Track if HF sync has been done

    def sync_from_hf(self, storage_manager):
        """
        Sync runtime artifacts from Hugging Face on startup
        Downloads master panel, predictions, macro forecast, and model binaries from manifest
        """
        if self._synced or not storage_manager:
            return
        
        try:
            logger.info("Syncing artifacts from Hugging Face...")
            
            # Sync data files
            sync_files = [
                ('data/processed/master_panel.csv', self.data_dir / 'processed' / 'master_panel.csv'),
                ('data/processed/master_panel.parquet', self.data_dir / 'processed' / 'master_panel.parquet'),
                ('data/predictions/forecast_6m.parquet', self.data_dir / 'predictions' / 'forecast_6m.parquet'),
                ('data/macro_forecast/macro_fc.parquet', self.data_dir / 'macro_forecast' / 'macro_fc.parquet'),
                ('models/model_metadata.json', self.model_dir / 'model_metadata.json'),
            ]
            
            for remote_path, local_path in sync_files:
                try:
                    success = storage_manager.download_to_path(remote_path, str(local_path))
                    if success:
                        logger.info(f"Synced {remote_path} to {local_path}")
                except Exception as e:
                    logger.warning(f"Could not sync {remote_path}: {e}")
            
            # Sync model binaries from manifest
            metadata_file = self.model_dir / 'model_metadata.json'
            if metadata_file.exists():
                try:
                    import json
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    model_manifest = metadata.get('models', {})
                    for key, model_info in model_manifest.items():
                        if isinstance(model_info, dict):
                            filename = model_info.get('file')
                        else:
                            filename = model_info
                        
                        if filename:
                            remote_path = f"models/{filename}"
                            local_path = self.model_dir / filename
                            try:
                                success = storage_manager.download_to_path(remote_path, str(local_path))
                                if success:
                                    logger.info(f"Synced model: {filename}")
                            except Exception as e:
                                logger.warning(f"Could not sync model {filename}: {e}")
                    
                    # Also sync feat_cols from lgbm_reg.features for backward compatibility
                    if 'lgbm_reg' in model_manifest:
                        lgbm_info = model_manifest['lgbm_reg']
                        if isinstance(lgbm_info, dict) and 'features' in lgbm_info:
                            feat_file = lgbm_info['features']
                            remote_path = f"models/{feat_file}"
                            local_path = self.model_dir / feat_file
                            try:
                                success = storage_manager.download_to_path(remote_path, str(local_path))
                                if success:
                                    logger.info(f"Synced feat_cols from lgbm_reg.features: {feat_file}")
                            except Exception as e:
                                logger.warning(f"Could not sync {feat_file}: {e}")
                
                except Exception as e:
                    logger.warning(f"Could not sync models from manifest: {e}")
            
            # Sync pending uploads (staging area)
            try:
                remote_uploads = storage_manager.list_files("data/uploads/")
                remote_filenames = {Path(f).name for f in remote_uploads}
                
                # 1. Download missing pending uploads
                for remote_path in remote_uploads:
                    local_path = self.data_dir / "uploads" / Path(remote_path).name
                    if not local_path.exists():
                        success = storage_manager.download_to_path(remote_path, str(local_path))
                        if success:
                            logger.info(f"Synced pending upload: {remote_path}")
                
                # 2. Purge local uploads that are no longer in remote (genuinely pending)
                local_uploads_dir = self.data_dir / "uploads"
                if local_uploads_dir.exists():
                    for local_file in local_uploads_dir.iterdir():
                        if local_file.is_file() and local_file.suffix.lower() in ('.csv', '.xlsx', '.parquet', '.xls'):
                            if local_file.name not in remote_filenames:
                                try:
                                    local_file.unlink()
                                    logger.info(f"Purged processed/stale local upload: {local_file.name}")
                                except Exception as e:
                                    logger.warning(f"Could not purge local file {local_file.name}: {e}")
                                    
            except Exception as e:
                logger.warning(f"Could not sync pending uploads: {e}")

            self._synced = True
            logger.info("HF sync completed")
            
            # CRITICAL: Clear caches after sync so subsequent loads see the new files
            # instead of the sample data cached during the initial health check.
            st.cache_data.clear()
            st.cache_resource.clear()
            logger.info("Caches cleared after HF sync")
            
        except Exception as e:
            logger.error(f"Error syncing from HF: {e}")
            # Don't fail startup if sync fails

    def reload_all(self):
        """Force clear all Streamlit caches and local references"""
        st.cache_data.clear()
        st.cache_resource.clear()
        logger.info("DataLoader: All caches cleared.")
    
    @st.cache_data(ttl=3600, show_spinner="")
    def load_predictions(_self) -> pd.DataFrame:
        """
        Load latest predictions from parquet
        
        Returns:
            DataFrame with columns: kode_bank, nama_bank, periode, pk_prediksi, 
                                   skor_rbbr, sinyal_ews, proyeksi_6bln
        """
        try:
            from .data_utils import normalize_kode_bank
            
            # Prefer forecast_6m.parquet
            pred_file = _self.data_dir / 'predictions' / 'forecast_6m.parquet'
            
            if not pred_file.exists():
                # Fallback to pattern search (handles timestamps/temp files)
                pred_dir = _self.data_dir / 'predictions'
                if pred_dir.exists():
                    matches = list(pred_dir.glob('forecast_6m*.parquet'))
                    if not matches:
                        matches = list(pred_dir.glob('latest_predictions*.parquet'))
                    
                    if matches:
                        pred_file = max(matches, key=lambda p: p.stat().st_mtime)
            
            if not pred_file.exists():
                logger.warning(f"Predictions file not found in candidates or patterns")
                # Return sample data untuk demo
                return _self._generate_sample_predictions()
            
            df = pd.read_parquet(pred_file)
            df = normalize_kode_bank(df)
            logger.info(f"Loaded {len(df)} prediction records from {pred_file}")

            # --- Build kode_bank from master bank→kode_bank mapping if missing ---
            if 'kode_bank' not in df.columns and 'bank' in df.columns:
                try:
                    master_raw = _self.load_master_data()
                    if not master_raw.empty and 'bank' in master_raw.columns and 'kode_bank' in master_raw.columns:
                        bank_map = master_raw.drop_duplicates('bank').set_index('bank')['kode_bank'].to_dict()
                        df['kode_bank'] = df['bank'].map(bank_map)
                        logger.info(f"Mapped bank→kode_bank: {bank_map}")
                except Exception as _e:
                    logger.warning(f"Could not map bank→kode_bank: {_e}")

            # --- Alias *_pred metric columns to base names for SHAP/scoring ---
            pred_metric_aliases = {
                'CAR_pred': 'CAR', 'NPL_gross_pred': 'NPL_gross', 'NPL_net_pred': 'NPL_net',
                'ROA_pred': 'ROA', 'ROE_pred': 'ROE', 'NIM_pred': 'NIM',
                'BOPO_pred': 'BOPO', 'LDR_pred': 'LDR',
            }
            for src, dst in pred_metric_aliases.items():
                if src in df.columns and dst not in df.columns:
                    df[dst] = df[src]

            # Normalize column names
            rename_map = {
                'bank': 'nama_bank',
                'pk_pred': 'pk_prediksi',
                'pk_computed': 'pk_prediksi',
                'rbbr_final': 'rbbr_score_pred',
                'skor_rbbr': 'rbbr_score_pred',
                'rbbr_score': 'rbbr_score_pred',
                'sinyal': 'sinyal_ews',
                's_risk_pred': 'score_risk_profile_pred',
                's_rent_pred': 'score_rentabilitas_pred',
                's_perm_pred': 'score_permodalan_pred',
                'rbbr_lower': 'rbbr_score_lower',
                'rbbr_upper': 'rbbr_score_upper'
            }
            # Only rename if columns exist
            rename_map = {k: v for k, v in rename_map.items() if k in df.columns}
            df = df.rename(columns=rename_map)
            
            # Backwards compatibility / alias
            if 'rbbr_score_pred' in df.columns and 'skor_rbbr' not in df.columns:
                df['skor_rbbr'] = df['rbbr_score_pred']
            
            # Ensure rbbr_score is also present for compatibility with UI expecting metric name
            if 'rbbr_score_pred' in df.columns and 'rbbr_score' not in df.columns:
                df['rbbr_score'] = df['rbbr_score_pred']
            
            # Strip whitespace from nama_bank for clean join
            if 'nama_bank' in df.columns:
                df['nama_bank'] = df['nama_bank'].astype(str).str.strip()
            
            # Ensure proper dtypes
            if 'periode' in df.columns:
                df['periode'] = pd.to_datetime(df['periode'])
            
            # Add is_forecast discriminator if missing
            if 'is_forecast' not in df.columns and 'periode' in df.columns:
                try:
                    master = _self.load_master_data()
                    if not master.empty and 'periode' in master.columns:
                        latest_hist = master['periode'].max()
                        df['is_forecast'] = df['periode'] > latest_hist
                        logger.info(f"Added is_forecast column: latest_hist={latest_hist}, forecast_count={df['is_forecast'].sum()}")
                    else:
                        df['is_forecast'] = False
                except Exception as e:
                    logger.warning(f"Could not add is_forecast column: {e}")
                    df['is_forecast'] = False

            # If predictions are aggregate (kode_bank == 'ALL_BANKS'),
            # expand per real bank using master data so dashboard shows real banks
            if 'kode_bank' in df.columns and (df['kode_bank'] == 'ALL_BANKS').any():
                try:
                    master = _self.load_master_data()
                    if not master.empty:
                        banks = master[['kode_bank', 'nama_bank']].drop_duplicates()
                        # Cross-join: each bank gets all forecast periods
                        df_agg = df[df['kode_bank'] == 'ALL_BANKS'].drop(
                            columns=['kode_bank', 'nama_bank'], errors='ignore'
                        )
                        expanded = banks.merge(df_agg.assign(_key=1), how='cross') \
                            if hasattr(pd.DataFrame, 'merge') else \
                            banks.assign(_key=1).merge(df_agg.assign(_key=1), on='_key').drop('_key', axis=1)
                        expanded = expanded.drop(columns=['_key'], errors='ignore')
                        # Keep non-aggregate rows if any, append expanded
                        df_real = df[df['kode_bank'] != 'ALL_BANKS']
                        df = pd.concat([df_real, expanded], ignore_index=True)
                        logger.info(f'Expanded ALL_BANKS predictions to {len(banks)} banks')
                except Exception as _e:
                    logger.warning(f'Could not expand ALL_BANKS predictions: {_e}')
            
            # Join with metadata for additional info (category, etc.)
            try:
                metadata = _self.load_bank_metadata()
                if not metadata.empty:
                    # Rename columns for joining if necessary
                    # Rename columns for joining if necessary
                    if 'kode_bank' in metadata.columns and 'kode_bank' in df.columns:
                        meta_cols = [c for c in metadata.columns if c not in df.columns or c == 'kode_bank']
                        df = df.merge(metadata[meta_cols], on='kode_bank', how='left')
                    elif 'nama_bank' in metadata.columns and 'nama_bank' in df.columns:
                        # Drop columns that might conflict except the join key
                        meta_cols = [c for c in metadata.columns if c not in df.columns or c == 'nama_bank']
                        df = df.merge(metadata[meta_cols], on='nama_bank', how='left')
                
                # Check if kode_bank is now present, if not use placeholder or names
                if 'kode_bank' not in df.columns:
                    if 'nama_bank' in df.columns:
                        df['kode_bank'] = df['nama_bank']
                    else:
                        df['kode_bank'] = "UNKNOWN"
                    logger.warning("kode_bank was missing, using nama_bank as identifier")
                    
                logger.info("Joined predictions with bank metadata")
            except Exception as e:
                logger.warning(f"Could not join metadata: {e}")
                if 'kode_bank' not in df.columns:
                     if 'nama_bank' in df.columns:
                         df['kode_bank'] = df['nama_bank']
                     else:
                         df['kode_bank'] = "UNKNOWN"
            
            return df
        
        except Exception as e:
            logger.error(f"Error loading predictions: {e}")
            return _self._generate_sample_predictions()

    @st.cache_data(ttl=3600, show_spinner="")
    def load_master_data(_self) -> pd.DataFrame:
        """
        Load master panel data (24 months)
        
        Returns:
            DataFrame with all financial metrics
        """
        try:
            from .data_utils import normalize_kode_bank
            
            # Select by modification time (freshest file wins)
            candidates = [
                _self.data_dir / 'processed' / 'master_panel.parquet',
                _self.data_dir / 'processed' / 'master_panel.csv',
                _self.data_dir / 'processed' / 'master_panel_lite.parquet',
            ]
            existing = [(p, p.stat().st_mtime) for p in candidates if p.exists()]
            if not existing:
                logger.warning('Master data file not found')
                return _self._generate_sample_master()
            master_file = max(existing, key=lambda x: x[1])[0]
            logger.info(f"Selected master file by mtime: {master_file.name}")

            df = pd.read_csv(master_file) if master_file.suffix == '.csv' else pd.read_parquet(master_file)
            df = normalize_kode_bank(df)
            logger.info(f"Loaded master data from {master_file}: {len(df)} rows, {df['kode_bank'].nunique()} banks")
            
            # Normalize column names
            if 'bank' in df.columns and 'nama_bank' not in df.columns:
                df = df.rename(columns={'bank': 'nama_bank'})
            
            # Apply RBBR scoring
            try:
                from .rbbr_scoring import batch_score_rbbr
                df = batch_score_rbbr(df)
                # detail_bank.py expects pk_computed
                # visualization.py often expects pk_prediksi
                df['pk_prediksi'] = df['pk_computed']
                logger.info("Applied RBBR scoring to master data")
            except Exception as e:
                logger.warning(f"Could not apply RBBR scoring: {e}")
            
            # Ensure proper dtypes
            if 'periode' in df.columns:
                df['periode'] = pd.to_datetime(df['periode'])
            
            return df
        
        except Exception as e:
            logger.error(f"Error loading master data: {e}")
            return _self._generate_sample_master()
    
    @st.cache_data(ttl=86400, show_spinner="")  # static data — 24h TTL
    def load_bank_metadata(_self) -> pd.DataFrame:
        """Load bank metadata (code, name, category, ownership)"""
        try:
            from .data_utils import normalize_kode_bank
            
            # Try both processed and static folders
            meta_file_processed = _self.data_dir / 'processed' / 'bank_metadata.csv'
            meta_file_static = _self.data_dir / 'static' / 'bank_metadata.csv'
            
            df_meta = pd.DataFrame()
            if meta_file_static.exists():
                df_meta = pd.read_csv(meta_file_static, skipinitialspace=True)
            elif meta_file_processed.exists():
                df_meta = pd.read_csv(meta_file_processed, skipinitialspace=True)
            
            if not df_meta.empty:
                # Clean column names and string values
                df_meta.columns = df_meta.columns.str.strip()
                for col in df_meta.select_dtypes(['object']).columns:
                    df_meta[col] = df_meta[col].str.strip()
                df_meta = normalize_kode_bank(df_meta)
                # Ensure nama_bank has no null/None strings
                if 'nama_bank' in df_meta.columns:
                    df_meta['nama_bank'] = df_meta['nama_bank'].astype(str).str.strip()
                return df_meta
            else:
                # Generate from master data
                master = _self.load_master_data()
                if master is not None and not master.empty:
                    from .data_utils import resolve_canonical_bank_identity
                    master = resolve_canonical_bank_identity(master)
                    metadata = master.groupby('kode_bank').agg({
                        'nama_bank': 'first',
                        'kategori_bank': 'first',
                        'jenis_bank': 'first',
                        'kepemilikan': 'first',
                    }).reset_index()
                    return metadata
                
                return pd.DataFrame()
        
        except Exception as e:
            logger.error(f"Error loading bank metadata: {e}")
            return pd.DataFrame()
    @st.cache_data(ttl=3600, show_spinner="")
    def load_forecast_data(_self) -> pd.DataFrame:
        """Load forecast data with predictions for future periods"""
        try:
            forecast_file = _self.data_dir / 'predictions' / 'forecast_6m.parquet'
            if not forecast_file.exists():
                logger.warning(f"Forecast file not found: {forecast_file}")
                return pd.DataFrame()
            
            df = pd.read_parquet(forecast_file)
            if 'periode' in df.columns:
                df['periode'] = pd.to_datetime(df['periode'])
            
            logger.info(f"Loaded {len(df)} forecast records")
            return df
        except Exception as e:
            logger.error(f"Error loading forecast data: {e}")
            return pd.DataFrame()
    

    
    @st.cache_data(ttl=3600, show_spinner="")
    def load_macro_data(_self) -> pd.DataFrame:
        """Load macroeconomic data (historical + forecast)"""
        try:
            # 1. Load Forecast Data
            macro_file = _self.data_dir / 'macro_forecast' / 'macro_fc.parquet'
            if not macro_file.exists():
                # Fallback to pattern search
                macro_dir = _self.data_dir / 'macro_forecast'
                if macro_dir.exists():
                    matches = list(macro_dir.glob('macro_fc*.parquet'))
                    if matches:
                        macro_file = max(matches, key=lambda p: p.stat().st_mtime)

            df_fc = pd.DataFrame()
            if macro_file and macro_file.exists():
                df_fc = pd.read_parquet(macro_file)
                if 'periode' in df_fc.columns:
                    df_fc['periode'] = pd.to_datetime(df_fc['periode'])
                
                logger.info(f'Loaded macro forecast: {macro_file.name}, {len(df_fc)} periods')
            
            # 2. Load Historical Data from Master Panel
            from .data_utils import resolve_master_panel
            master_file, _ = resolve_master_panel(_self.data_dir)
            
            df_hist = pd.DataFrame()
            if master_file is not None and master_file.exists():
                df_master = pd.read_csv(master_file) if master_file.suffix == '.csv' else pd.read_parquet(master_file)
                if 'periode' in df_master.columns:
                    df_master['periode'] = pd.to_datetime(df_master['periode'])
                
                # Extract and normalize macro columns
                alias_map = {
                    'bi_rate': 'bi_rate',
                    'suku_bunga_bi_rate': 'bi_rate',
                    'inflasi': 'inflasi',
                    'inflasi_yoy': 'inflasi',
                    'kurs_usd': 'kurs_usd',
                    'kurs_usd_idr': 'kurs_usd'
                }
                
                # Check which aliases are available in master data
                actual_rename = {k: v for k, v in alias_map.items() if k in df_master.columns}
                if actual_rename:
                    df_master = df_master.rename(columns=actual_rename)
                
                macro_cols = ['periode', 'bi_rate', 'inflasi', 'kurs_usd']
                available_cols = [c for c in macro_cols if c in df_master.columns]
                df_hist = df_master[available_cols].drop_duplicates().sort_values('periode')
                

            
            # 3. Combine Historical and Forecast
            if not df_hist.empty and not df_fc.empty:
                fc_start = df_fc['periode'].min()
                df_hist = df_hist[df_hist['periode'] < fc_start]
                
                # Rename forecast columns to match historical
                df_fc_renamed = df_fc.rename(columns={
                    'bi_rate_fc': 'bi_rate',
                    'inflasi_fc': 'inflasi',
                    'kurs_usd_fc': 'kurs_usd'
                })
                
                df_combined = pd.concat([df_hist, df_fc_renamed], ignore_index=True).sort_values('periode')
                logger.info(f'Combined macro data: {len(df_hist)} historical + {len(df_fc)} forecast = {len(df_combined)} total')
                return df_combined
            elif not df_fc.empty:
                df_fc_renamed = df_fc.rename(columns={
                    'bi_rate_fc': 'bi_rate',
                    'inflasi_fc': 'inflasi',
                    'kurs_usd_fc': 'kurs_usd'
                })
                return df_fc_renamed
            elif not df_hist.empty:
                return df_hist
            else:
                return _self._generate_sample_macro()
        except Exception as e:
            logger.error(f"Error loading macro data: {e}")
            return _self._generate_sample_macro()
    
    def get_model_status(_self) -> Dict:
        """
        Lightweight check of model availability without loading into memory.
        Checks manifest and presence of physical files.
        """
        metadata_file = _self.model_dir / 'model_metadata.json'
        if not metadata_file.exists():
            return {'available': False, 'error': 'Manifest missing'}
            
        try:
            import json
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            model_manifest = metadata.get('models', {})
            if not model_manifest:
                return {'available': False, 'error': 'Manifest empty'}
                
            missing = []
            found_count = 0
            for key, model_info in model_manifest.items():
                filename = model_info.get('file') if isinstance(model_info, dict) else model_info
                if filename:
                    if not (_self.model_dir / filename).exists():
                        missing.append(filename)
                    else:
                        found_count += 1
            
            return {
                'available': found_count > 0,
                'total': len(model_manifest),
                'found': found_count,
                'missing': missing,
                'last_update': metadata.get('updated_at', 'Unknown')
            }
        except Exception as e:
            return {'available': False, 'error': str(e)}

    @st.cache_resource(show_spinner="")  # lazy — called only by Prediksi/SHAP via models_ready()
    def load_models(_self) -> Dict:
        """
        Load all trained models from manifest in model_metadata.json
        Includes a DLL sanity check for torch/Windows.
        """
        models = {}
        
        # 1. DLL Sanity Check (Prevent WinError 1114)
        try:
            import torch
            _ = torch.zeros(1)
        except Exception as e:
            error_msg = f"Torch DLL Error: {str(e)}"
            if "1114" in error_msg:
                error_msg = "Critical: Torch DLL failed to init (Error 1114). Check Visual C++ Redistributables."
            logger.error(error_msg)
            return {'load_error': error_msg}

        # 2. Load manifest from model_metadata.json
        metadata_file = _self.model_dir / 'model_metadata.json'
        if not metadata_file.exists():
            logger.error("model_metadata.json not found")
            return {'load_error': 'Model manifest not found'}
        
        try:
            import json
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            model_manifest = metadata.get('models', {})
            if not model_manifest:
                logger.error("No models section in metadata")
                return {'load_error': 'Model manifest empty'}
            
            logger.info(f"Loaded model manifest with {len(model_manifest)} entries")
        except Exception as e:
            logger.error(f"Failed to load model manifest: {e}")
            return {'load_error': f'Failed to load manifest: {str(e)}'}

        # 3. Load models from manifest
        for key, model_info in model_manifest.items():
            if isinstance(model_info, dict):
                filename = model_info.get('file')
            else:
                filename = model_info
            
            if not filename:
                logger.warning(f"No file specified for {key}")
                continue
            
            filepath = _self.model_dir / filename
            if not filepath.exists():
                logger.warning(f"Model file missing: {filename}")
                continue

            try:
                if filename.endswith('.ckpt'):
                    from pytorch_forecasting.models.temporal_fusion_transformer import TemporalFusionTransformer
                    import torch as _torch
                    import inspect
                    try:
                        _ckpt = _torch.load(filepath, map_location='cpu')
                        _hp = _ckpt.get('hyper_parameters', {})
                        _valid = set(inspect.signature(TemporalFusionTransformer.__init__).parameters.keys())
                        _filtered = {k: v for k, v in _hp.items() if k in _valid}
                        _ckpt['hyper_parameters'] = _filtered
                        import tempfile
                        with tempfile.NamedTemporaryFile(suffix='.ckpt', delete=False) as tmp:
                            tmp_path = tmp.name
                        _torch.save(_ckpt, tmp_path)
                        models[key] = TemporalFusionTransformer.load_from_checkpoint(tmp_path)
                        import os
                        os.unlink(tmp_path)
                        logger.info(f"Loaded TFT checkpoint: {key} from {filename}")
                    except Exception as ckpt_err:
                        logger.error(f"Failed to load checkpoint {filename}: {ckpt_err}. Skipping.")
                        continue
                elif filename.endswith('.pkl'):
                    try:
                        # NeuralProphet models are saved with torch.save
                        # Use weights_only=False as they contain custom objects
                        models[key] = torch.load(filepath, map_location='cpu', weights_only=False)
                        
                        # Cleanup trainer for NeuralProphet models if present
                        if hasattr(models[key], 'trainer'):
                            models[key].trainer = None
                            
                        logger.info(f"Loaded model (torch): {key} from {filename}")
                    except Exception as _torch_err:
                        # torch.load failed → file is plain-pickle (e.g. LightGBM, SARIMAX)
                        logger.debug(f"torch.load failed for {filename} ({_torch_err}), falling back to pickle")
                        with open(filepath, 'rb') as f:
                            models[key] = pickle.load(f)
                        logger.info(f"Loaded model (pickle): {key} from {filename}")
                elif filename.endswith('.json'):
                    import json
                    with open(filepath, 'r') as f:
                        models[key] = json.load(f)
                    logger.info(f"Loaded config: {key} from {filename}")
            except Exception as e:
                logger.error(f"Failed to load {key} ({filename}): {str(e)}")
                continue
        
        # 4. Backward compatibility: load feat_cols from lgbm_reg.features if not already loaded
        if 'feat_cols' not in models and 'lgbm_reg' in model_manifest:
            lgbm_info = model_manifest['lgbm_reg']
            if isinstance(lgbm_info, dict) and 'features' in lgbm_info:
                feat_file = lgbm_info['features']
                feat_path = _self.model_dir / feat_file
                if feat_path.exists():
                    try:
                        with open(feat_path, 'rb') as f:
                            models['feat_cols'] = pickle.load(f)
                        logger.info(f"Loaded feat_cols from lgbm_reg.features: {feat_file}")
                    except Exception as e:
                        logger.warning(f"Failed to load feat_cols from {feat_file}: {e}")
        
        # 5. Store metadata for reference
        models['metadata'] = metadata
        
        return models
    
    def get_bank_list(_self) -> List[Dict]:
        """
        Get list of banks for dropdown
        
        Returns:
            List of dicts with kode_bank, nama_bank
        """
        master = _self.load_master_data()
        
        if not master.empty and 'kode_bank' in master.columns:
            # Extract unique banks directly from master data to ensure we don't miss any
            # Get the most recent name for each bank
            if 'nama_bank' in master.columns:
                # Sort to get latest, then drop duplicates
                if 'periode' in master.columns:
                    master_sorted = master.sort_values('periode', ascending=False)
                else:
                    master_sorted = master
                bank_info = master_sorted.drop_duplicates('kode_bank')[['kode_bank', 'nama_bank']]
                return bank_info.to_dict('records')
            else:
                available_banks = master['kode_bank'].unique()
                # Try to get names from metadata
                metadata = _self.load_bank_metadata()
                if not metadata.empty:
                    metadata = metadata[metadata['kode_bank'].isin(available_banks)]
                    # Add any missing banks with their code as name
                    found_codes = metadata['kode_bank'].tolist()
                    missing_codes = [c for c in available_banks if c not in found_codes]
                    missing_df = pd.DataFrame({'kode_bank': missing_codes, 'nama_bank': [f"Bank {c}" for c in missing_codes]})
                    metadata = pd.concat([metadata[['kode_bank', 'nama_bank']], missing_df], ignore_index=True)
                    return metadata.to_dict('records')
                else:
                    return [{'kode_bank': c, 'nama_bank': f"Bank {c}"} for c in available_banks]
        else:
            # Fallback to metadata
            metadata = _self.load_bank_metadata()
            if not metadata.empty:
                return metadata[['kode_bank', 'nama_bank']].to_dict('records')
            
        return []
    
    def get_bank_data(_self, kode_bank: str, months: int = None) -> pd.DataFrame:
        """
        Get historical data for specific bank — returns ALL available periods.
        months param kept for backward compatibility but ignored.
        """
        from .data_utils import normalize_kode_bank
        
        master = _self.load_master_data()
        
        if master.empty:
            return pd.DataFrame()
        
        # Normalize the search key
        kode_bank_normalized = normalize_kode_bank(pd.DataFrame({'kode_bank': [kode_bank]}))['kode_bank'].iloc[0]
        
        bank_data = master[master['kode_bank'] == kode_bank_normalized].copy()
            
        # Merge with metadata to ensure we have kategori_bank, jenis_bank etc
        try:
            metadata = _self.load_bank_metadata()
            if not metadata.empty:
                # Only bring in columns that don't already exist to avoid conflicts
                meta_cols = [c for c in metadata.columns if c not in bank_data.columns or c == 'kode_bank']
                if len(meta_cols) > 1: # if there is more than just kode_bank
                    bank_data = bank_data.merge(metadata[meta_cols], on='kode_bank', how='left')
        except Exception as e:
            logger.warning(f"Could not merge metadata in get_bank_data: {e}")
        
        # Sort by period descending
        if 'periode' in bank_data.columns:
            bank_data = bank_data.sort_values('periode', ascending=False)
        
        return bank_data
    
    # =========================================================================
    # SAMPLE DATA GENERATORS (for demo/fallback)
    # =========================================================================
    
    def _generate_sample_predictions(_self) -> pd.DataFrame:
        """Generate sample predictions for demo"""
        import numpy as np
        from datetime import datetime, timedelta
        
        logger.info("Generating sample prediction data")
        
        # Sample banks
        banks = [
            ('001', 'Bank Mandiri'),
            ('002', 'Bank BRI'),
            ('003', 'Bank BNI'),
            ('008', 'Bank BCA'),
            ('011', 'Bank Danamon'),
        ]
        
        # Generate 6 months of data
        periods = pd.date_range(
            end=datetime.now(),
            periods=6,
            freq='MS'
        )
        
        data = []
        for kode, nama in banks:
            for periode in periods:
                # Random PK (weighted towards healthy banks)
                pk = np.random.choice([1, 2, 3, 4, 5], p=[0.3, 0.4, 0.2, 0.08, 0.02])
                skor = pk + np.random.uniform(-0.3, 0.3)
                
                sinyal = 'hijau' if pk <= 2 else 'kuning' if pk == 3 else 'merah'
                
                data.append({
                    'kode_bank': kode,
                    'nama_bank': nama,
                    'periode': periode,
                    'pk_prediksi': pk,
                    'skor_rbbr': round(skor, 3),
                    'sinyal_ews': sinyal,
                    'CAR': round(np.random.uniform(12, 20), 2),
                    'NPL_gross': round(np.random.uniform(1, 5), 2),
                    'ROA': round(np.random.uniform(1, 3), 2),
                    'LDR': round(np.random.uniform(75, 95), 2),
                })
        
        return pd.DataFrame(data)
    
    def _generate_sample_master(_self) -> pd.DataFrame:
        """Generate sample master data for demo"""
        import numpy as np
        from datetime import datetime, timedelta
        
        logger.info("Generating sample master data")
        
        banks = [
            ('008', 'Mandiri', 'BUKU 1', 'konvensional', 'PERSERO'),
            ('002', 'BRI', 'BUKU 1', 'konvensional', 'PERSERO'),
            ('009', 'BNI', 'BUKU 1', 'konvensional', 'PERSERO'),
            ('200', 'BTN', 'BUKU 1', 'konvensional', 'PERSERO')
        ]
        
        periods = pd.date_range(
            end=datetime.now(),
            periods=24,
            freq='MS'
        )
        
        data = []
        for kode, nama, buku, jenis, kepemilikan in banks:
            for periode in periods:
                data.append({
                    'kode_bank': kode,
                    'nama_bank': nama,
                    'kategori_bank': buku,
                    'jenis_bank': jenis,
                    'kepemilikan': kepemilikan,
                    'periode': periode,
                    'CAR': round(np.random.uniform(12, 25), 2),
                    'NPL_gross': round(np.random.uniform(1, 6), 2),
                    'NPL_net': round(np.random.uniform(0.5, 3), 2),
                    'NIM': round(np.random.uniform(4, 7), 2),
                    'ROA': round(np.random.uniform(1, 3.5), 2),
                    'ROE': round(np.random.uniform(8, 18), 2),
                    'BOPO': round(np.random.uniform(60, 85), 2),
                    'LDR': round(np.random.uniform(70, 100), 2),
                    'LCR': round(np.random.uniform(100, 200), 2),
                    'total_aset': round(np.random.uniform(500000, 1500000), 0),
                })
        
        return pd.DataFrame(data)
    
    def _generate_sample_macro(_self) -> pd.DataFrame:
        """Generate sample macro data with standardized columns"""
        import numpy as np
        from datetime import datetime
        
        periods = pd.date_range(
            end=datetime.now(),
            periods=24,
            freq='MS'
        )
        
        data = []
        base_rate = 4.75
        base_inflasi = 3.0
        base_kurs = 15500
        
        for periode in periods:
            data.append({
                'periode': periode,
                'bi_rate': round(base_rate + np.random.uniform(-0.5, 0.5), 2),
                'inflasi': round(base_inflasi + np.random.uniform(-1, 1), 2),
                'kurs_usd': round(base_kurs + np.random.uniform(-500, 500), 0),
            })
        
        return pd.DataFrame(data)

# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_loader_instance = None

def get_data_loader() -> DataLoader:
    """Get singleton DataLoader instance"""
    global _loader_instance
    if _loader_instance is None:
        _loader_instance = DataLoader()
    return _loader_instance
