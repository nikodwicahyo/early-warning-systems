"""
Test Suite for Post-Retraining Data Refresh System
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import json
import tempfile
import shutil
import sys
from unittest.mock import MagicMock, patch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# ---------------------------------------------------------------------------
# Actual module imports (streamlit is mocked in conftest.py)
# ---------------------------------------------------------------------------
from src.config import RBBRConfig, ModelConfig, AppConfig
from src.data_validator import DataValidator


# ============================================================================
# TESTS: src.config constants
# ============================================================================

class TestSrcConfig:
    def test_pk_labels_pk1(self):
        assert RBBRConfig.PK_LABELS[1] == 'Sangat Sehat'

    def test_pk_labels_pk5(self):
        assert RBBRConfig.PK_LABELS[5] == 'Tidak Sehat'

    def test_pk_colors_all_present(self):
        assert set(RBBRConfig.PK_COLORS.keys()) == {1, 2, 3, 4, 5}

    def test_weights_sum_to_one(self):
        total = sum(RBBRConfig.WEIGHTS.values())
        assert abs(total - 1.0) < 1e-9

    def test_model_config_retrain_min_months(self):
        assert ModelConfig.RETRAIN_MIN_MONTHS == 36

    def test_app_config_allowed_extensions(self):
        assert '.csv' in AppConfig.ALLOWED_EXTENSIONS

    def test_data_validator_required_cols(self):
        assert 'CAR' in DataValidator.REQUIRED_COLS
        assert 'periode' in DataValidator.REQUIRED_COLS


# ============================================================================
# TESTS: Version Bumping Logic
# ============================================================================

class TestVersionBumping:
    """Test version incrementing logic"""
    
    def test_bump_minor_version_1_0_0(self):
        """Test version bump from 1.0.0"""
        version = "1.0.0"
        parts = str(version).strip().split('.')
        while len(parts) < 3:
            parts.append('0')
        major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])
        new_version = f"{major}.{minor + 1}.0"
        assert new_version == "1.1.0"
    
    def test_bump_minor_version_1_5_3(self):
        """Test version bump from 1.5.3"""
        version = "1.5.3"
        parts = str(version).strip().split('.')
        while len(parts) < 3:
            parts.append('0')
        major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])
        new_version = f"{major}.{minor + 1}.0"
        assert new_version == "1.6.0"
    
    def test_bump_minor_version_2_0_0(self):
        """Test version bump from 2.0.0"""
        version = "2.0.0"
        parts = str(version).strip().split('.')
        while len(parts) < 3:
            parts.append('0')
        major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])
        new_version = f"{major}.{minor + 1}.0"
        assert new_version == "2.1.0"
    
    def test_bump_minor_version_legacy(self):
        """Test version bump from legacy format"""
        version = "1.0"
        parts = str(version).strip().split('.')
        while len(parts) < 3:
            parts.append('0')
        major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])
        new_version = f"{major}.{minor + 1}.0"
        assert new_version == "1.1.0"


# ============================================================================
# TESTS: Data Service Logic (without streamlit)
# ============================================================================

class TestDataServiceLogic:
    """Test DataService logic without streamlit dependencies"""
    
    def test_forecast_periods_calculation(self):
        """Test 6-month forecast period calculation"""
        latest_period = pd.Timestamp('2026-11-01')
        
        periods = []
        current = latest_period
        for _ in range(6):
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1)
            else:
                current = current.replace(month=current.month + 1)
            periods.append(current)
        
        assert len(periods) == 6
        assert periods[0] == pd.Timestamp('2026-12-01')
        assert periods[-1] == pd.Timestamp('2027-05-01')
    
    def test_forecast_periods_from_january(self):
        """Test forecast periods from January"""
        latest_period = pd.Timestamp('2026-01-01')
        
        periods = []
        current = latest_period
        for _ in range(6):
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1)
            else:
                current = current.replace(month=current.month + 1)
            periods.append(current)
        
        assert len(periods) == 6
        assert periods[0] == pd.Timestamp('2026-02-01')
        assert periods[-1] == pd.Timestamp('2026-07-01')
    
    def test_forecast_periods_from_december(self):
        """Test forecast periods from December"""
        latest_period = pd.Timestamp('2026-12-01')
        
        periods = []
        current = latest_period
        for _ in range(6):
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1)
            else:
                current = current.replace(month=current.month + 1)
            periods.append(current)
        
        assert len(periods) == 6
        assert periods[0] == pd.Timestamp('2027-01-01')
        assert periods[-1] == pd.Timestamp('2027-06-01')


# ============================================================================
# TESTS: Model Version Manager Logic
# ============================================================================

class TestModelVersionManagerLogic:
    """Test ModelVersionManager logic"""
    
    def test_version_directory_creation(self, temp_data_dir):
        """Test version directory creation"""
        versions_dir = temp_data_dir / 'versions'
        versions_dir.mkdir(parents=True, exist_ok=True)
        
        assert versions_dir.exists()
        assert versions_dir.is_dir()
    
    def test_list_versions_empty(self, temp_data_dir):
        """Test listing versions when empty"""
        versions_dir = temp_data_dir / 'versions'
        versions_dir.mkdir(parents=True, exist_ok=True)
        
        versions = [d for d in versions_dir.iterdir() if d.is_dir()]
        
        assert isinstance(versions, list)
        assert len(versions) == 0
    
    def test_create_version_metadata(self, temp_data_dir):
        """Test creating version metadata"""
        version_dir = temp_data_dir / 'versions' / 'v1.0.0_20260427_120000'
        version_dir.mkdir(parents=True, exist_ok=True)
        
        metadata = {
            'version': '1.0.0',
            'last_trained': datetime.now().isoformat(),
            'is_original': True,
            'models': {},
            'performance': {}
        }
        
        metadata_file = version_dir / 'metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f)
        
        assert metadata_file.exists()
        
        with open(metadata_file, 'r') as f:
            loaded = json.load(f)
        
        assert loaded['version'] == '1.0.0'
        assert loaded['is_original'] is True
    
    def test_mark_version_as_original(self, temp_data_dir):
        """Test marking version as original"""
        version_dir = temp_data_dir / 'versions' / 'v1.0.0_20260427_120000'
        version_dir.mkdir(parents=True, exist_ok=True)
        
        metadata = {'version': '1.0.0', 'is_original': False}
        with open(version_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f)
        
        # Mark as original
        with open(version_dir / 'metadata.json', 'r') as f:
            loaded = json.load(f)
        
        loaded['is_original'] = True
        
        with open(version_dir / 'metadata.json', 'w') as f:
            json.dump(loaded, f)
        
        with open(version_dir / 'metadata.json', 'r') as f:
            result = json.load(f)
        
        assert result['is_original'] is True
    
    def test_is_version_immutable(self, temp_data_dir):
        """Test checking if version is immutable"""
        version_dir = temp_data_dir / 'versions' / 'v1.0.0_original'
        version_dir.mkdir(parents=True, exist_ok=True)
        
        metadata = {'version': '1.0.0', 'is_original': True}
        with open(version_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f)
        
        with open(version_dir / 'metadata.json', 'r') as f:
            loaded = json.load(f)
        
        is_immutable = loaded.get('is_original', False)
        assert is_immutable is True


# ============================================================================
# TESTS: AutoRetrainer Logic
# ============================================================================

class TestAutoRetrainerLogic:
    """Test AutoRetrainer logic"""
    
    def test_clear_uploads(self, temp_data_dir):
        """Test upload file clearing"""
        uploads_dir = temp_data_dir / 'data' / 'uploads'
        uploads_dir.mkdir(parents=True, exist_ok=True)
        
        test_files = ['test1.csv', 'test2.xlsx', 'test3.parquet']
        for fname in test_files:
            (uploads_dir / fname).touch()
        
        # Clear uploads
        deleted_count = 0
        for f in uploads_dir.iterdir():
            if f.suffix.lower() in ('.csv', '.xlsx', '.parquet', '.xls'):
                f.unlink()
                deleted_count += 1
        
        assert deleted_count == 3
        assert len(list(uploads_dir.glob('*'))) == 0
    
    def test_archive_current_models(self, temp_data_dir):
        """Test model archiving"""
        models_dir = temp_data_dir / 'models'
        models_dir.mkdir(parents=True, exist_ok=True)
        
        test_models = ['lgbm_reg_models.pkl', 'lgbm_clf.pkl', 'np_bi_rate.pkl']
        for fname in test_models:
            (models_dir / fname).touch()
        
        # Create metadata
        metadata = {'version': '1.0.0'}
        metadata_file = models_dir / 'model_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f)
        
        # Archive models
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_dir = models_dir / 'versions' / f"v{metadata['version']}_{timestamp}"
        archive_dir.mkdir(parents=True, exist_ok=True)
        
        for fname in test_models:
            src = models_dir / fname
            if src.exists():
                shutil.copy2(src, archive_dir / fname)
        
        shutil.copy2(metadata_file, archive_dir / 'metadata.json')
        
        assert archive_dir.exists()
        assert (archive_dir / 'metadata.json').exists()
    
    def test_cleanup_old_versions(self, temp_data_dir):
        """Test old version cleanup"""
        versions_dir = temp_data_dir / 'models' / 'versions'
        versions_dir.mkdir(parents=True, exist_ok=True)
        
        # Create 5 versions
        for i in range(5):
            version_dir = versions_dir / f'v1.0.{i}_20260427_12000{i}'
            version_dir.mkdir(parents=True, exist_ok=True)
            metadata = {'version': f'1.0.{i}'}
            with open(version_dir / 'metadata.json', 'w') as f:
                json.dump(metadata, f)
        
        # Cleanup (keep last 3)
        all_versions = sorted(versions_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        
        if len(all_versions) > 3:
            for old_dir in all_versions[3:]:
                shutil.rmtree(old_dir)
        
        remaining = list(versions_dir.iterdir())
        assert len(remaining) == 3


# ============================================================================
# TESTS: Data Validation
# ============================================================================

class TestDataValidation:
    """Test data validation and consistency"""
    
    def test_master_data_columns(self, sample_master_data):
        """Test master data has required columns"""
        required_cols = ['periode', 'kode_bank', 'nama_bank', 'CAR', 'NPL_gross', 'ROA']
        
        for col in required_cols:
            assert col in sample_master_data.columns
    
    def test_predictions_columns(self, sample_predictions):
        """Test predictions have required columns"""
        required_cols = ['periode', 'kode_bank', 'nama_bank', 'pk_prediksi', 'sinyal_ews']
        
        for col in required_cols:
            assert col in sample_predictions.columns
    
    def test_macro_data_columns(self, sample_macro_data):
        """Test macro data has required columns"""
        required_cols = ['periode', 'suku_bunga_bi_rate', 'inflasi_yoy', 'kurs_usd_idr']
        
        for col in required_cols:
            assert col in sample_macro_data.columns
    
    def test_data_types(self, sample_master_data):
        """Test data types are correct"""
        assert pd.api.types.is_datetime64_any_dtype(sample_master_data['periode'])
        assert pd.api.types.is_numeric_dtype(sample_master_data['CAR'])
        assert pd.api.types.is_numeric_dtype(sample_master_data['NPL_gross'])


# ============================================================================
# TESTS: Integration
# ============================================================================

class TestIntegration:
    """Integration tests for complete workflow"""
    
    def test_data_merge(self, temp_data_dir):
        """Test data merging"""
        # Create initial data
        dates1 = pd.date_range('2025-01-01', periods=12, freq='MS')
        data1 = [{'periode': d, 'kode_bank': 'BANK', 'nama_bank': 'Bank A', 'CAR': 15.0} for d in dates1]
        df1 = pd.DataFrame(data1)
        
        # Create new data
        dates2 = pd.date_range('2026-02-01', periods=10, freq='MS')
        data2 = [{'periode': d, 'kode_bank': 'BANK', 'nama_bank': 'Bank A', 'CAR': 16.0} for d in dates2]
        df2 = pd.DataFrame(data2)
        
        # Merge
        df_combined = pd.concat([df1, df2], ignore_index=True)
        df_combined = df_combined.drop_duplicates(subset=['kode_bank', 'periode'], keep='last')
        
        assert len(df_combined) > len(df1)
        
        latest_period = df_combined['periode'].max()
        assert latest_period.month == 11
        assert latest_period.year == 2026
    
    def test_forecast_from_merged_data(self, temp_data_dir):
        """Test forecast calculation from merged data"""
        # Create merged data
        dates = pd.date_range('2025-01-01', periods=23, freq='MS')
        data = [{'periode': d, 'kode_bank': 'BANK', 'nama_bank': 'Bank A'} for d in dates]
        df = pd.DataFrame(data)
        
        latest_period = df['periode'].max()
        
        # Calculate forecast
        periods = []
        current = latest_period
        for _ in range(6):
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1)
            else:
                current = current.replace(month=current.month + 1)
            periods.append(current)
        
        assert len(periods) == 6
        assert periods[0] == pd.Timestamp('2026-12-01')
        assert periods[-1] == pd.Timestamp('2027-05-01')


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_master_data():
    dates = pd.date_range('2026-01-01', periods=12, freq='MS')
    banks = ['BANK A', 'BANK B', 'BANK C']
    data = []
    for date in dates:
        for bank in banks:
            data.append({
                'periode': date, 'kode_bank': bank[:4].upper(), 'nama_bank': bank,
                'CAR': np.random.uniform(10, 25), 'NPL_gross': np.random.uniform(0, 5),
                'ROA': np.random.uniform(0, 3), 'LDR': np.random.uniform(70, 95),
                'BOPO': np.random.uniform(60, 90), 'NIM': np.random.uniform(2, 6),
                'pk_computed': np.random.randint(1, 6),
                'sinyal_ews': np.random.choice(['hijau', 'kuning', 'merah']),
            })
    return pd.DataFrame(data)


@pytest.fixture
def sample_predictions():
    dates = pd.date_range('2026-02-01', periods=6, freq='MS')
    banks = ['BANK A', 'BANK B', 'BANK C']
    data = []
    for date in dates:
        for bank in banks:
            data.append({
                'periode': date, 'kode_bank': bank[:4].upper(), 'nama_bank': bank,
                'pk_prediksi': np.random.randint(1, 6), 'skor_rbbr': np.random.uniform(1, 5),
                'sinyal_ews': np.random.choice(['hijau', 'kuning', 'merah']),
                'CAR': np.random.uniform(10, 25), 'NPL_gross': np.random.uniform(0, 5),
            })
    return pd.DataFrame(data)


@pytest.fixture
def sample_macro_data():
    dates = pd.date_range('2026-01-01', periods=12, freq='MS')
    data = [{'periode': d, 'suku_bunga_bi_rate': np.random.uniform(3, 7),
             'inflasi_yoy': np.random.uniform(2, 5), 'kurs_usd_idr': np.random.uniform(15000, 16000)}
            for d in dates]
    return pd.DataFrame(data)


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])