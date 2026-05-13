"""
Smart Data Merge Module
Intelligent data merging with gap detection and timeline validation
"""

import logging
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import pandas as pd

logger = logging.getLogger(__name__)


class SmartDataMerger:
    """Intelligent data merging with gap detection and timeline validation"""
    
    def __init__(self, master_panel_path: str = 'data/processed/master_panel.csv'):
        """
        Initialize SmartDataMerger
        Note: master_panel_path is kept for backward compatibility but actual resolution
        uses freshest-file logic from data_utils
        """
        self.master_panel_path = Path(master_panel_path)
        self.data_dir = self.master_panel_path.parent.parent  # data/processed -> data
    
    def merge_uploads(self, uploads_dir: Path) -> Tuple[pd.DataFrame, List[str], Dict]:
        """
        Merge all uploaded files into master panel
        
        Args:
            uploads_dir: Directory containing uploaded files
            
        Returns:
            Tuple of (merged_df, new_periods, gap_info)
        """
        try:
            # Load existing master panel using shared resolution logic
            from .data_utils import load_master_panel, normalize_kode_bank, resolve_canonical_bank_identity
            df_master = load_master_panel(self.data_dir)
            
            if df_master.empty:
                logger.warning(f"Master panel not found in {self.data_dir}")
            
            # Load all uploaded files
            new_data_list = []
            if not uploads_dir.exists():
                logger.warning(f"Uploads directory not found: {uploads_dir}")
                return df_master, [], {}
            
            for f in uploads_dir.iterdir():
                if f.suffix.lower() == '.csv':
                    new_data_list.append(pd.read_csv(f))
                    logger.info(f"Loaded CSV: {f.name}")
                elif f.suffix.lower() in ('.xlsx', '.xls'):
                    new_data_list.append(pd.read_excel(f))
                    logger.info(f"Loaded XLSX: {f.name}")
                elif f.suffix.lower() == '.parquet':
                    new_data_list.append(pd.read_parquet(f))
                    logger.info(f"Loaded Parquet: {f.name}")
            
            if not new_data_list:
                logger.info("No new data files found")
                return df_master, [], {}
            
            # Combine new data
            df_new = pd.concat(new_data_list, ignore_index=True)
            df_new['periode'] = pd.to_datetime(df_new['periode'])
            df_new = normalize_kode_bank(df_new)
            
            # Detect new periods
            new_periods = self._detect_new_periods(df_master, df_new)
            
            # Validate continuity and detect gaps
            gap_info = self._validate_continuity(df_master, df_new, new_periods)
            
            # Merge: keep latest values for overlaps
            if df_master.empty:
                df_merged = df_new.copy()
            else:
                df_merged = pd.concat([df_master, df_new], ignore_index=True)
                df_merged = df_merged.drop_duplicates(subset=['kode_bank', 'periode'], keep='last')
            
            df_merged = resolve_canonical_bank_identity(df_merged, df_master)
            df_merged = df_merged.sort_values(['kode_bank', 'periode']).reset_index(drop=True)
            
            # Log merge results
            logger.info(f"Merged data: {len(df_merged)} rows, {len(new_periods)} new periods")
            logger.info(f"New periods: {new_periods}")
            if gap_info['gaps']:
                logger.warning(f"Gaps detected: {gap_info['gaps']}")
            
            return df_merged, new_periods, gap_info
            
        except Exception as e:
            logger.error(f"Error merging uploads: {e}")
            raise
    
    def _detect_new_periods(self, df_master: pd.DataFrame, df_new: pd.DataFrame) -> List[str]:
        """Detect new periods in uploaded data - ALL periods from new data"""
        try:
            if df_master.empty:
                periods = sorted(df_new['periode'].unique())
                return [p.strftime('%Y-%m') for p in periods]
            
            # Get ALL unique periods from new data (not just those >= max_master)
            # This ensures all uploaded data is treated as new/updated
            all_new_periods = sorted(df_new['periode'].unique())
            
            return [p.strftime('%Y-%m') for p in all_new_periods]
        except Exception as e:
            logger.error(f"Error detecting new periods: {e}")
            return []
    
    def _validate_continuity(self, df_master: pd.DataFrame, df_new: pd.DataFrame, new_periods: List[str]) -> Dict:
        """Validate data continuity and detect gaps"""
        gap_info = {'gaps': [], 'overlaps': [], 'warnings': []}
        
        try:
            if df_master.empty or df_new.empty:
                return gap_info
            
            # Check for gaps in new data
            for bank in df_new['kode_bank'].unique():
                bank_new = df_new[df_new['kode_bank'] == bank]['periode'].sort_values()
                if len(bank_new) > 1:
                    try:
                        periods_diff = (bank_new.diff().dt.days / 30).fillna(0).astype(int)
                        gaps = periods_diff[periods_diff > 1]
                        if len(gaps) > 0:
                            gap_info['gaps'].append(f"Bank {bank}: {len(gaps)} gap(s)")
                    except Exception as e:
                        logger.warning(f"Could not check gaps for bank {bank}: {e}")
            
            # Overlaps are handled automatically (keep='last'), no need to warn
            
            # Check if new data extends timeline
            if new_periods:
                gap_info['new_timeline'] = f"{new_periods[0]} to {new_periods[-1]}"
            
            return gap_info
            
        except Exception as e:
            logger.error(f"Error validating continuity: {e}")
            return gap_info
    
    def save_merged_data(self, df_merged: pd.DataFrame) -> bool:
        """Save merged data to master panel in both CSV and Parquet"""
        try:
            self.master_panel_path.parent.mkdir(parents=True, exist_ok=True)
            df_merged.to_csv(self.master_panel_path, index=False)
            df_merged.to_parquet(self.master_panel_path.with_suffix('.parquet'), index=False)
            logger.info(f"Saved merged data to {self.master_panel_path} and .parquet")
            return True
        except Exception as e:
            logger.error(f"Error saving merged data: {e}")
            return False
