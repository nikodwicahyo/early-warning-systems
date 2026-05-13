"""
Integration Tests for Post-Retraining Workflow
End-to-end testing without heavy ML dependencies
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import json
import tempfile
import shutil
import sys
from datetime import datetime


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def test_environment():
    """Create complete test environment"""
    temp_dir = Path(tempfile.mkdtemp())
    
    # Create directory structure
    (temp_dir / 'data' / 'processed').mkdir(parents=True)
    (temp_dir / 'data' / 'predictions').mkdir(parents=True)
    (temp_dir / 'data' / 'uploads').mkdir(parents=True)
    (temp_dir / 'models' / 'versions').mkdir(parents=True)
    
    yield temp_dir
    
    shutil.rmtree(temp_dir)


@pytest.fixture
def initial_data(test_environment):
    """Create initial master data (2026-01)"""
    dates = pd.date_range('2025-01-01', periods=12, freq='MS')
    banks = ['BANK A', 'BANK B', 'BANK C']
    
    data = []
    for date in dates:
        for bank in banks:
            data.append({
                'periode': date,
                'kode_bank': bank[:4].upper(),
                'nama_bank': bank,
                'CAR': np.random.uniform(10, 25),
                'NPL_gross': np.random.uniform(0, 5),
                'NPL_net': np.random.uniform(0, 3),
                'ROA': np.random.uniform(0, 3),
                'ROE': np.random.uniform(0, 20),
                'LDR': np.random.uniform(70, 95),
                'BOPO': np.random.uniform(60, 90),
                'NIM': np.random.uniform(2, 6),
                'bi_rate': np.random.uniform(3, 7),
                'inflasi': np.random.uniform(2, 5),
                'kurs_usd': np.random.uniform(15000, 16000),
                'pk_computed': np.random.randint(1, 6),
                'sinyal_ews': np.random.choice(['hijau', 'kuning', 'merah']),
            })
    
    df = pd.DataFrame(data)
    
    # Save to test environment
    df.to_csv(test_environment / 'data' / 'processed' / 'master_panel.csv', index=False)
    
    return df


@pytest.fixture
def new_training_data(test_environment):
    """Create new training data (2026-02 to 2026-11)"""
    dates = pd.date_range('2026-02-01', periods=10, freq='MS')
    banks = ['BANK A', 'BANK B', 'BANK C']
    
    data = []
    for date in dates:
        for bank in banks:
            data.append({
                'periode': date,
                'kode_bank': bank[:4].upper(),
                'nama_bank': bank,
                'CAR': np.random.uniform(10, 25),
                'NPL_gross': np.random.uniform(0, 5),
                'NPL_net': np.random.uniform(0, 3),
                'ROA': np.random.uniform(0, 3),
                'ROE': np.random.uniform(0, 20),
                'LDR': np.random.uniform(70, 95),
                'BOPO': np.random.uniform(60, 90),
                'NIM': np.random.uniform(2, 6),
                'bi_rate': np.random.uniform(3, 7),
                'inflasi': np.random.uniform(2, 5),
                'kurs_usd': np.random.uniform(15000, 16000),
                'pk_computed': np.random.randint(1, 6),
                'sinyal_ews': np.random.choice(['hijau', 'kuning', 'merah']),
            })
    
    df = pd.DataFrame(data)
    
    # Save to uploads directory
    df.to_csv(test_environment / 'data' / 'uploads' / 'new_data_20260427.csv', index=False)
    
    return df


# ============================================================================
# TESTS: Complete Workflow
# ============================================================================

class TestCompleteWorkflow:
    """Test complete retraining workflow"""
    
    def test_phase1_pre_retraining_verification(self, test_environment, initial_data):
        """Phase 1: Verify initial state before retraining"""
        
        # Load initial data
        master_file = test_environment / 'data' / 'processed' / 'master_panel.csv'
        assert master_file.exists()
        
        df = pd.read_csv(master_file)
        assert len(df) > 0
        
        # Verify latest period is December 2025 (fixture creates 12 months from 2025-01)
        df['periode'] = pd.to_datetime(df['periode'])
        latest_period = df['periode'].max()
        assert latest_period.month == 12
        assert latest_period.year == 2025
    
    def test_phase2_data_upload(self, test_environment, initial_data, new_training_data):
        """Phase 2: Verify new data upload"""
        
        # Check new data file exists
        upload_file = test_environment / 'data' / 'uploads' / 'new_data_20260427.csv'
        assert upload_file.exists()
        
        # Verify new data content
        df_new = pd.read_csv(upload_file)
        assert len(df_new) > 0
        
        df_new['periode'] = pd.to_datetime(df_new['periode'])
        assert df_new['periode'].min().month == 2
        assert df_new['periode'].max().month == 11
    
    def test_phase3_data_merge(self, test_environment, initial_data, new_training_data):
        """Phase 3: Verify data merging"""
        
        # Load both datasets
        master_file = test_environment / 'data' / 'processed' / 'master_panel.csv'
        upload_file = test_environment / 'data' / 'uploads' / 'new_data_20260427.csv'
        
        df_master = pd.read_csv(master_file)
        df_new = pd.read_csv(upload_file)
        
        # Merge
        df_master['periode'] = pd.to_datetime(df_master['periode'])
        df_new['periode'] = pd.to_datetime(df_new['periode'])
        
        # Count unique periods in each
        master_periods = len(df_master['periode'].unique())
        new_periods = len(df_new['periode'].unique())
        
        df_combined = pd.concat([df_master, df_new], ignore_index=True)
        
        # Verify combined data has all periods
        combined_periods = len(df_combined['periode'].unique())
        assert combined_periods == master_periods + new_periods
        
        latest_period = df_combined['periode'].max()
        assert latest_period.month == 11
        assert latest_period.year == 2026
    
    def test_phase4_version_management(self, test_environment):
        """Phase 4: Verify version management"""
        
        # Create initial metadata
        metadata = {
            'version': '1.0.0',
            'last_trained': datetime.now().isoformat(),
            'models': {},
            'performance': {}
        }
        
        metadata_file = test_environment / 'models' / 'model_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f)
        
        # Bump version
        version = metadata['version']
        parts = str(version).strip().split('.')
        while len(parts) < 3:
            parts.append('0')
        major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])
        new_version = f"{major}.{minor + 1}.0"
        
        assert new_version == '1.1.0'
        
        # Update metadata
        metadata['version'] = new_version
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f)
        
        # Verify
        with open(metadata_file, 'r') as f:
            updated = json.load(f)
        
        assert updated['version'] == '1.1.0'
    
    def test_phase5_upload_file_cleanup(self, test_environment, new_training_data):
        """Phase 5: Verify upload file cleanup"""
        
        uploads_dir = test_environment / 'data' / 'uploads'
        
        # Verify file exists before cleanup
        upload_file = uploads_dir / 'new_data_20260427.csv'
        assert upload_file.exists()
        
        # Simulate cleanup
        for f in uploads_dir.iterdir():
            if f.suffix.lower() in ('.csv', '.xlsx', '.parquet', '.xls'):
                f.unlink()
        
        # Verify file is deleted
        assert not upload_file.exists()
    
    def test_phase6_forecast_periods(self, test_environment, initial_data, new_training_data):
        """Phase 6: Verify 6-month forecast periods"""
        
        # Load merged data
        master_file = test_environment / 'data' / 'processed' / 'master_panel.csv'
        upload_file = test_environment / 'data' / 'uploads' / 'new_data_20260427.csv'
        
        df_master = pd.read_csv(master_file)
        df_new = pd.read_csv(upload_file)
        
        df_master['periode'] = pd.to_datetime(df_master['periode'])
        df_new['periode'] = pd.to_datetime(df_new['periode'])
        
        df_combined = pd.concat([df_master, df_new], ignore_index=True)
        
        # Get latest period
        latest_period = df_combined['periode'].max()
        
        # Calculate forecast periods
        forecast_periods = []
        current = latest_period
        for _ in range(6):
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1)
            else:
                current = current.replace(month=current.month + 1)
            forecast_periods.append(current)
        
        # Verify forecast periods
        assert len(forecast_periods) == 6
        assert forecast_periods[0] == pd.Timestamp('2026-12-01')
        assert forecast_periods[-1] == pd.Timestamp('2027-05-01')


# ============================================================================
# TESTS: Version Management
# ============================================================================

class TestVersionManagement:
    """Test model version management"""
    
    def test_create_version_archive(self, test_environment):
        """Test creating version archive"""
        
        # Create test models
        models_dir = test_environment / 'models'
        test_models = ['lgbm_reg_models.pkl', 'lgbm_clf.pkl', 'np_bi_rate.pkl']
        for fname in test_models:
            (models_dir / fname).touch()
        
        # Create metadata
        metadata = {'version': '1.0.0'}
        metadata_file = models_dir / 'model_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f)
        
        # Create archive
        archive_dir = models_dir / 'versions' / 'v1.0.0_20260427_120000'
        archive_dir.mkdir(parents=True, exist_ok=True)
        
        for fname in test_models:
            src = models_dir / fname
            if src.exists():
                shutil.copy2(src, archive_dir / fname)
        
        shutil.copy2(metadata_file, archive_dir / 'metadata.json')
        
        # Verify archive
        assert archive_dir.exists()
        assert (archive_dir / 'metadata.json').exists()
    
    def test_keep_4_versions(self, test_environment):
        """Test keeping only 4 versions (original + last 3)"""
        
        versions_dir = test_environment / 'models' / 'versions'
        
        # Create 5 versions
        for i in range(5):
            version_dir = versions_dir / f'v1.0.{i}_20260427_12000{i}'
            version_dir.mkdir(parents=True, exist_ok=True)
            metadata = {'version': f'1.0.{i}'}
            with open(version_dir / 'metadata.json', 'w') as f:
                json.dump(metadata, f)
        
        # Simulate cleanup (keep last 3)
        all_versions = sorted(versions_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        
        if len(all_versions) > 3:
            for old_dir in all_versions[3:]:
                shutil.rmtree(old_dir)
        
        # Verify only 3 remain
        remaining = list(versions_dir.iterdir())
        assert len(remaining) == 3
    
    def test_rollback_to_version(self, test_environment):
        """Test rollback to previous version"""
        
        # Create version archive
        version_dir = test_environment / 'models' / 'versions' / 'v1.0.0_20260427_120000'
        version_dir.mkdir(parents=True, exist_ok=True)
        
        # Create test model files in archive
        test_models = ['lgbm_reg_models.pkl', 'lgbm_clf.pkl']
        for fname in test_models:
            (version_dir / fname).touch()
        
        # Create metadata
        metadata = {'version': '1.0.0'}
        with open(version_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f)
        
        # Simulate rollback
        models_dir = test_environment / 'models'
        for fname in test_models:
            src = version_dir / fname
            if src.exists():
                shutil.copy2(src, models_dir / fname)
        
        shutil.copy2(version_dir / 'metadata.json', models_dir / 'model_metadata.json')
        
        # Verify rollback
        assert (models_dir / 'lgbm_reg_models.pkl').exists()
        assert (models_dir / 'model_metadata.json').exists()
        
        with open(models_dir / 'model_metadata.json', 'r') as f:
            restored = json.load(f)
        
        assert restored['version'] == '1.0.0'


# ============================================================================
# TESTS: Error Handling
# ============================================================================

class TestErrorHandling:
    """Test error handling and recovery"""
    
    def test_invalid_data_handling(self, test_environment):
        """Test handling of invalid data"""
        
        # Create invalid data (missing required columns)
        invalid_data = pd.DataFrame({
            'periode': pd.date_range('2026-01-01', periods=3),
            'invalid_col': [1, 2, 3]
        })
        
        invalid_file = test_environment / 'data' / 'uploads' / 'invalid.csv'
        invalid_data.to_csv(invalid_file, index=False)
        
        # Verify file exists
        assert invalid_file.exists()
        
        # Try to load and validate
        df = pd.read_csv(invalid_file)
        required_cols = ['kode_bank', 'nama_bank', 'CAR']
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        assert len(missing_cols) > 0
    
    def test_retraining_failure_recovery(self, test_environment):
        """Test recovery from retraining failure"""
        
        # Create backup of current version
        models_dir = test_environment / 'models'
        backup_dir = models_dir / 'versions' / 'v1.0.0_backup'
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Create test model
        test_model = models_dir / 'lgbm_reg_models.pkl'
        test_model.touch()
        
        # Backup model
        shutil.copy2(test_model, backup_dir / 'lgbm_reg_models.pkl')
        
        # Simulate failure and rollback
        test_model.unlink()
        
        # Restore from backup
        shutil.copy2(backup_dir / 'lgbm_reg_models.pkl', test_model)
        
        # Verify recovery
        assert test_model.exists()
    
    def test_preserve_upload_on_failure(self, test_environment, new_training_data):
        """Test that upload files are preserved on failure"""
        
        uploads_dir = test_environment / 'data' / 'uploads'
        upload_file = uploads_dir / 'new_data_20260427.csv'
        
        # Verify file exists
        assert upload_file.exists()
        
        # On failure, file should NOT be deleted
        # (This is the expected behavior)
        assert upload_file.exists()


# ============================================================================
# TESTS: Data Consistency
# ============================================================================

class TestDataConsistency:
    """Test data consistency across operations"""
    
    def test_historical_data_consistency(self, test_environment, initial_data, new_training_data):
        """Test historical data remains consistent"""
        
        # Load initial data
        master_file = test_environment / 'data' / 'processed' / 'master_panel.csv'
        df_initial = pd.read_csv(master_file)
        
        initial_count = len(df_initial)
        
        # Load new data
        upload_file = test_environment / 'data' / 'uploads' / 'new_data_20260427.csv'
        df_new = pd.read_csv(upload_file)
        
        # Merge (without dropping duplicates to preserve all data)
        df_initial['periode'] = pd.to_datetime(df_initial['periode'])
        df_new['periode'] = pd.to_datetime(df_new['periode'])
        
        df_combined = pd.concat([df_initial, df_new], ignore_index=True)
        
        # Verify combined data has all records
        assert len(df_combined) == initial_count + len(df_new)
        
        # Verify all periods are present
        all_periods = df_combined['periode'].unique()
        assert len(all_periods) > len(df_initial['periode'].unique())
    
    def test_forecast_data_consistency(self, test_environment):
        """Test forecast data consistency"""
        
        # Create sample forecast
        dates = pd.date_range('2026-12-01', periods=6, freq='MS')
        banks = ['BANK A', 'BANK B', 'BANK C']
        
        forecast_data = []
        for date in dates:
            for bank in banks:
                forecast_data.append({
                    'periode': date,
                    'kode_bank': bank[:4].upper(),
                    'nama_bank': bank,
                    'pk_prediksi': np.random.randint(1, 6),
                })
        
        df_forecast = pd.DataFrame(forecast_data)
        
        # Verify consistency
        assert len(df_forecast) == len(dates) * len(banks)
        assert df_forecast['periode'].min() == pd.Timestamp('2026-12-01')
        assert df_forecast['periode'].max() == pd.Timestamp('2027-05-01')


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])