"""
Regression test for master panel freshness resolution
Tests that both UI and retraining use the same freshest-file logic
"""

import pytest
import pandas as pd
import tempfile
import time
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_utils import resolve_master_panel, load_master_panel
from src.data_merge import SmartDataMerger


def create_sample_master_data(periods=3):
    """Create sample master panel data"""
    data = []
    for i in range(periods):
        data.append({
            'kode_bank': '001',
            'periode': f'2024-0{i+1}-01',
            'CAR': 15.0 + i,
            'NPL_gross': 2.0,
            'ROA': 2.5,
        })
    return pd.DataFrame(data)


class TestMasterPanelFreshness:
    """Test that freshest file is selected consistently"""
    
    def test_resolve_master_panel_prefers_newest_mtime(self):
        """Test that resolve_master_panel selects file with newest mtime"""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            processed_dir = data_dir / 'processed'
            processed_dir.mkdir()
            
            # Create CSV first
            csv_path = processed_dir / 'master_panel.csv'
            df = create_sample_master_data()
            df.to_csv(csv_path, index=False)
            time.sleep(0.1)  # Ensure different mtime
            
            # Create Parquet later (should be selected)
            parquet_path = processed_dir / 'master_panel.parquet'
            df.to_parquet(parquet_path, index=False)
            
            # Verify parquet is newer
            assert parquet_path.stat().st_mtime > csv_path.stat().st_mtime
            
            # Test resolution
            selected_path, file_format = resolve_master_panel(data_dir)
            
            assert selected_path == parquet_path
            assert file_format == 'parquet'
    
    def test_resolve_master_panel_csv_newer_than_parquet(self):
        """Test case where CSV is newer than Parquet (retraining output)"""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            processed_dir = data_dir / 'processed'
            processed_dir.mkdir()
            
            # Create Parquet first
            parquet_path = processed_dir / 'master_panel.parquet'
            df = create_sample_master_data()
            df.to_parquet(parquet_path, index=False)
            time.sleep(0.1)  # Ensure different mtime
            
            # Create CSV later (should be selected)
            csv_path = processed_dir / 'master_panel.csv'
            df.to_csv(csv_path, index=False)
            
            # Verify CSV is newer
            assert csv_path.stat().st_mtime > parquet_path.stat().st_mtime
            
            # Test resolution
            selected_path, file_format = resolve_master_panel(data_dir)
            
            assert selected_path == csv_path
            assert file_format == 'csv'
    
    def test_load_master_panel_uses_freshest(self):
        """Test that load_master_panel loads from freshest file"""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            processed_dir = data_dir / 'processed'
            processed_dir.mkdir()
            
            # Create old CSV with 3 periods
            csv_path = processed_dir / 'master_panel.csv'
            df_old = create_sample_master_data(periods=3)
            df_old.to_csv(csv_path, index=False)
            time.sleep(0.1)
            
            # Create new Parquet with 5 periods
            parquet_path = processed_dir / 'master_panel.parquet'
            df_new = create_sample_master_data(periods=5)
            df_new.to_parquet(parquet_path, index=False)
            
            # Load should return 5 periods (from parquet)
            df_loaded = load_master_panel(data_dir)
            
            assert len(df_loaded) == 5
            assert 'periode' in df_loaded.columns
            assert df_loaded['periode'].dtype == 'datetime64[ns]'
    
    def test_smart_data_merger_uses_shared_resolution(self):
        """Test that SmartDataMerger uses shared resolution logic"""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            processed_dir = data_dir / 'processed'
            uploads_dir = data_dir / 'uploads'
            processed_dir.mkdir()
            uploads_dir.mkdir()
            
            # Create old CSV
            csv_path = processed_dir / 'master_panel.csv'
            df_old = create_sample_master_data(periods=2)
            df_old.to_csv(csv_path, index=False)
            time.sleep(0.1)
            
            # Create newer Parquet
            parquet_path = processed_dir / 'master_panel.parquet'
            df_newer = create_sample_master_data(periods=3)
            df_newer.to_parquet(parquet_path, index=False)
            
            # Create upload with new period
            upload_path = uploads_dir / 'new_data.csv'
            df_upload = pd.DataFrame([{
                'kode_bank': '001',
                'periode': '2024-04-01',
                'CAR': 18.0,
                'NPL_gross': 2.0,
                'ROA': 2.5,
            }])
            df_upload.to_csv(upload_path, index=False)
            
            # Merge should use parquet (3 periods) + upload (1 period) = 4 periods
            merger = SmartDataMerger(str(csv_path))  # Path points to CSV but should use parquet
            df_merged, new_periods, gap_info = merger.merge_uploads(uploads_dir)
            
            assert len(df_merged) == 4  # 3 from parquet + 1 from upload
            assert '2024-04' in new_periods
    
    def test_no_master_panel_files(self):
        """Test behavior when no master panel files exist"""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            processed_dir = data_dir / 'processed'
            processed_dir.mkdir()
            
            selected_path, file_format = resolve_master_panel(data_dir)
            
            assert selected_path is None
            assert file_format is None
            
            df = load_master_panel(data_dir)
            assert df.empty


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
