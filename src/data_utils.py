"""
Shared data utilities for consistent file resolution across UI and retraining
"""

import pandas as pd
import logging
from pathlib import Path
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


def normalize_kode_bank(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize kode_bank column to 3-character string with leading zeros preserved.
    Handles numeric types (8 → '008'), floats (8.0 → '008'), and strings ('8' → '008').
    
    Args:
        df: DataFrame with 'kode_bank' column
    
    Returns:
        DataFrame with normalized kode_bank
    """
    if 'kode_bank' not in df.columns:
        return df
    
    df['kode_bank'] = (
        df['kode_bank']
        .astype(str)
        .str.strip()
        .str.replace(r'\.0+$', '', regex=True)
        .str.zfill(3)
    )
    return df


def resolve_canonical_bank_identity(df: pd.DataFrame, reference_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Resolves exactly one display name for each kode_bank.
    If reference_df is provided, uses its mapping first to preserve existing canonical names.
    """
    name_col = 'nama_bank' if 'nama_bank' in df.columns else ('bank' if 'bank' in df.columns else None)
    if 'kode_bank' not in df.columns or not name_col:
        return df

    df = normalize_kode_bank(df.copy())
    mapping = {}

    if reference_df is not None and not reference_df.empty and 'kode_bank' in reference_df.columns:
        ref_name_col = 'nama_bank' if 'nama_bank' in reference_df.columns else ('bank' if 'bank' in reference_df.columns else None)
        if ref_name_col:
            valid_ref = reference_df.dropna(subset=['kode_bank', ref_name_col])
            mapping = valid_ref.groupby('kode_bank')[ref_name_col].first().to_dict()

    valid_df = df.dropna(subset=['kode_bank', name_col])
    current_mapping = valid_df.groupby('kode_bank')[name_col].first().to_dict()

    for k, v in current_mapping.items():
        if k not in mapping:
            mapping[k] = v

    if mapping:
        df[name_col] = df['kode_bank'].map(mapping).fillna(df[name_col])

    return df


def resolve_master_panel(data_dir: Path) -> Tuple[Optional[Path], Optional[str]]:
    """
    Resolve master panel file by selecting the freshest (newest mtime) among candidates.
    
    Args:
        data_dir: Base data directory (e.g., Path('data'))
    
    Returns:
        Tuple of (selected_path, format) where format is 'csv' or 'parquet'
        Returns (None, None) if no candidates exist
    """
    candidates = [
        data_dir / 'processed' / 'master_panel.parquet',
        data_dir / 'processed' / 'master_panel.csv',
        data_dir / 'processed' / 'master_panel_lite.parquet',
    ]
    
    # 1. Check candidates first
    existing = [(p, p.stat().st_mtime) for p in candidates if p.exists()]
    
    # 2. If no candidates, scan directory for patterns (handles timestamped/temp files)
    if not existing:
        proc_dir = data_dir / 'processed'
        if proc_dir.exists():
            patterns = ['master_panel*.parquet', 'master_panel*.csv']
            for pattern in patterns:
                for p in proc_dir.glob(pattern):
                    existing.append((p, p.stat().st_mtime))
    
    if not existing:
        logger.warning('No master panel files found in candidates or patterns')
        return None, None
    
    # Select newest
    selected_path = max(existing, key=lambda x: x[1])[0]
    file_format = 'csv' if selected_path.suffix == '.csv' else 'parquet'
    
    logger.info(f"Resolved master panel: {selected_path.name} (mtime: {selected_path.stat().st_mtime})")
    return selected_path, file_format


def load_master_panel(data_dir: Path) -> pd.DataFrame:
    """
    Load master panel using freshest-file resolution.
    
    Args:
        data_dir: Base data directory
    
    Returns:
        DataFrame with master panel data
    """
    master_path, file_format = resolve_master_panel(data_dir)
    
    if master_path is None:
        logger.error("No master panel file found")
        return pd.DataFrame()
    
    try:
        if file_format == 'csv':
            df = pd.read_csv(master_path)
        else:
            df = pd.read_parquet(master_path)
        
        # Normalize kode_bank
        df = normalize_kode_bank(df)
        
        # Normalize periode
        if 'periode' in df.columns:
            df['periode'] = pd.to_datetime(df['periode'])
        
        logger.info(f"Loaded master panel from {master_path.name}: {len(df)} rows")
        return df
        
    except Exception as e:
        logger.error(f"Error loading master panel from {master_path}: {e}")
        return pd.DataFrame()
