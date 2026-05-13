"""Tests for scripts/auto_retraining_new.py — validates staged pipeline with mocked ML dependencies."""

import json
import sys
import types
import pytest
import shutil
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock, patch
from datetime import datetime


# ---------------------------------------------------------------------------
# Inject lightweight mocks for training.* before importing auto_retraining_new
# ---------------------------------------------------------------------------

def _inject_training_mocks():
    """Inject mock modules so auto_retraining_new can be imported without torch/neuralprophet."""
    for mod_name in ('training', 'training.ensemble_trainer', 'training.ensemble_predictor'):
        if mod_name not in sys.modules:
            sys.modules[mod_name] = types.ModuleType(mod_name)

    trainer_mock = sys.modules['training.ensemble_trainer']
    trainer_mock.TARGET_COLS = ['CAR', 'NPL_gross', 'ROA', 'LDR', 'BOPO']
    trainer_mock.MACRO_COLS = ['bi_rate', 'inflasi', 'kurs_usd']
    trainer_mock.EnsembleTrainer = MagicMock()

    predictor_mock = sys.modules['training.ensemble_predictor']
    predictor_mock.EnsemblePredictor = MagicMock()


_inject_training_mocks()

from scripts.auto_retraining_new import _bump_minor_version, AutoRetrainer  # noqa: E402


# ---------------------------------------------------------------------------
# _bump_minor_version
# ---------------------------------------------------------------------------

class TestBumpMinorVersion:
    def test_standard(self):        assert _bump_minor_version('1.0.0') == '1.1.0'
    def test_non_zero_minor(self):  assert _bump_minor_version('1.5.3') == '1.6.0'
    def test_major_2(self):         assert _bump_minor_version('2.0.0') == '2.1.0'
    def test_legacy_two_parts(self): assert _bump_minor_version('1.0') == '1.1.0'
    def test_legacy_one_part(self): assert _bump_minor_version('1') == '1.1.0'
    def test_bad_string_fallback(self): assert _bump_minor_version('bad') == '1.1.0'
    def test_resets_patch(self):    assert _bump_minor_version('1.3.7') == '1.4.0'


# ---------------------------------------------------------------------------
# AutoRetrainer helpers (no actual training)
# ---------------------------------------------------------------------------

@pytest.fixture
def retrainer(tmp_path):
    config = {
        'models_dir': str(tmp_path / 'models'),
        'data_dir': str(tmp_path / 'data'),
        'processed_data_file': str(tmp_path / 'data' / 'processed' / 'master_panel.csv'),
        'output_prediction_file': str(tmp_path / 'data' / 'predictions' / 'forecast_6m.parquet'),
        'target_columns': ['CAR'],
        'macro_columns': ['bi_rate'],
        'prediction_horizon': 6,
        'max_epochs': 30,
        'fine_tune_epochs': 10,
        'fine_tune_lr': 1e-4,
    }
    import tempfile, json as _json
    cfg_file = tmp_path / 'config.json'
    cfg_file.write_text(_json.dumps(config))
    return AutoRetrainer(config_path=str(cfg_file))


class TestAutoRetrainerLoadMetadata:
    def test_no_file_returns_empty_dict(self, retrainer):
        meta = retrainer.load_model_metadata()
        assert isinstance(meta, dict)

    def test_with_valid_json(self, retrainer):
        retrainer.metadata_file.parent.mkdir(parents=True, exist_ok=True)
        retrainer.metadata_file.write_text(json.dumps({'version': '1.2.0'}))
        meta = retrainer.load_model_metadata()
        assert meta['version'] == '1.2.0'

    def test_normalizes_version_to_semver(self, retrainer):
        retrainer.metadata_file.parent.mkdir(parents=True, exist_ok=True)
        retrainer.metadata_file.write_text(json.dumps({'version': '1.0'}))
        meta = retrainer.load_model_metadata()
        assert meta['version'] == '1.0.0'

    def test_sets_default_keys(self, retrainer):
        retrainer.metadata_file.parent.mkdir(parents=True, exist_ok=True)
        retrainer.metadata_file.write_text(json.dumps({'version': '1.0.0'}))
        meta = retrainer.load_model_metadata()
        assert 'models' in meta
        assert 'performance' in meta


class TestAutoRetrainerClearUploads:
    def test_clears_csv_files(self, retrainer):
        retrainer.uploads_dir.mkdir(parents=True, exist_ok=True)
        (retrainer.uploads_dir / 'data.csv').touch()
        (retrainer.uploads_dir / 'data.xlsx').touch()
        retrainer._clear_uploads()
        assert list(retrainer.uploads_dir.iterdir()) == []

    def test_no_uploads_dir_does_not_raise(self, retrainer):
        retrainer._clear_uploads()

    def test_non_data_files_preserved(self, retrainer):
        retrainer.uploads_dir.mkdir(parents=True, exist_ok=True)
        (retrainer.uploads_dir / 'readme.txt').touch()
        retrainer._clear_uploads()
        assert (retrainer.uploads_dir / 'readme.txt').exists()


class TestAutoRetrainerArchiveArtifacts:
    """Test _archive_current_artifacts (models + data + metadata)"""
    
    def test_creates_archive_dir(self, retrainer):
        retrainer.models_dir.mkdir(parents=True, exist_ok=True)
        meta = {'version': '1.0.0'}
        retrainer.metadata_file.write_text(json.dumps(meta))
        archive_path = retrainer._archive_current_artifacts(meta)
        assert archive_path is not None
        assert Path(archive_path).exists()

    def test_archives_models_and_data(self, retrainer):
        retrainer.models_dir.mkdir(parents=True, exist_ok=True)
        retrainer.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Create model files
        (retrainer.models_dir / 'lgbm_reg_models.pkl').touch()
        
        # Create data artifacts
        processed_dir = retrainer.data_dir / 'processed'
        processed_dir.mkdir(parents=True, exist_ok=True)
        (processed_dir / 'master_panel.csv').touch()
        
        predictions_dir = retrainer.data_dir / 'predictions'
        predictions_dir.mkdir(parents=True, exist_ok=True)
        (predictions_dir / 'forecast_6m.parquet').touch()
        
        meta = {
            'version': '1.0.0',
            'models': {
                'lgbm_reg': 'lgbm_reg_models.pkl'
            }
        }
        retrainer.metadata_file.write_text(json.dumps(meta))
        
        archive_path = retrainer._archive_current_artifacts(meta)
        archive_dir = Path(archive_path)
        
        # Verify models archived
        assert (archive_dir / 'lgbm_reg_models.pkl').exists()
        
        # Verify data artifacts archived
        assert (archive_dir / 'master_panel.csv').exists()
        assert (archive_dir / 'forecast_6m.parquet').exists()


class TestAutoRetrainerRollback:
    """Test rollback restores all artifacts (models + data + metadata)"""
    
    def test_nonexistent_version_returns_false(self, retrainer):
        retrainer.models_dir.mkdir(parents=True, exist_ok=True)
        success, msg = retrainer.rollback_to_version('v_does_not_exist')
        assert success is False
        assert "not found" in msg.lower()

    def test_rollback_restores_models_and_data(self, retrainer):
        retrainer.models_dir.mkdir(parents=True, exist_ok=True)
        retrainer.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Create version directory with models and data
        vdir = retrainer.models_dir / 'versions' / 'v1.0.0_20260101_120000'
        vdir.mkdir(parents=True)
        
        # Add model files
        (vdir / 'lgbm_clf.pkl').touch()
        metadata = {
            'version': '1.0.0',
            'models': {
                'lgbm_clf': 'lgbm_clf.pkl'
            }
        }
        (vdir / 'model_metadata.json').write_text(json.dumps(metadata))
        
        # Add data artifacts
        (vdir / 'master_panel.csv').touch()
        (vdir / 'forecast_6m.parquet').touch()
        (vdir / 'macro_fc.parquet').touch()
        
        with patch.object(retrainer, '_upload_artifacts_to_hf', return_value=(True, "")):
            success, msg = retrainer.rollback_to_version('v1.0.0_20260101_120000')
            assert success is True
            assert msg == ""
        
        # Verify models restored
        assert (retrainer.models_dir / 'lgbm_clf.pkl').exists()
        assert (retrainer.models_dir / 'model_metadata.json').exists()
        
        # Verify data artifacts restored
        assert Path(retrainer.config['processed_data_file']).exists()
        assert (retrainer.data_dir / 'predictions' / 'forecast_6m.parquet').exists()
        assert (retrainer.data_dir / 'macro_forecast' / 'macro_fc.parquet').exists()


class TestRollbackFailureHandling:
    def test_run_retraining_pipeline_handles_rollback_failure(self, retrainer):
        """Regression test: verify pipeline reports rollback failure correctly"""
        retrainer.models_dir.mkdir(parents=True, exist_ok=True)
        retrainer.metadata_file.write_text(json.dumps({'version': '1.0.0'}))
        
        # Mock load_latest_data to succeed, but train_ensemble to fail
        retrainer.load_latest_data = MagicMock(return_value=pd.DataFrame({'kode_bank':['001'], 'periode':[pd.Timestamp('2024-01-01')]}))
        
        with patch('scripts.auto_retraining_new.EnsembleTrainer') as mock_trainer_cls:
            mock_trainer = mock_trainer_cls.return_value
            mock_trainer.train_ensemble.side_effect = Exception("Training Failed")
            
            # Mock rollback_to_version to return failure
            retrainer.rollback_to_version = MagicMock(return_value=(False, "Rollback Failed"))
            
            callback_msgs = []
            def cb(p, m): callback_msgs.append(m)
            
            # We need an archive_path for rollback to be triggered
            retrainer._archive_current_artifacts = MagicMock(return_value='v1.0.0_archive')
            
            result = retrainer.run_retraining_pipeline(progress_callback=cb)
            
            assert result is False
            # Check if failure message is in callback
            assert any("Rollback Failed" in m for m in callback_msgs)
            assert any("Retraining gagal dan rollback gagal" in m for m in callback_msgs)

    def test_rollback_to_version_warns_local_only_no_storage(self, retrainer):
        """Verify warning message for local-only rollback when storage manager is missing"""
        retrainer.models_dir.mkdir(parents=True, exist_ok=True)
        retrainer.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Create version directory
        vdir = retrainer.models_dir / 'versions' / 'v1.0.0_local'
        vdir.mkdir(parents=True)
        metadata = {
            'version': '1.0.0',
            'models': {
                'lgbm_clf': 'lgbm_clf.pkl'
            }
        }
        (vdir / 'model_metadata.json').write_text(json.dumps(metadata))
        
        # Ensure storage_manager is None
        retrainer.storage_manager = None
        
        success, msg = retrainer.rollback_to_version('v1.0.0_local')
        
        assert success is True
        assert "local-only" in msg.lower()
        assert "No storage manager available" in msg


class TestAutoRetrainerStaging:
    """Test staged version promotion"""
    
    def test_stage_version_creates_complete_snapshot(self, retrainer):
        retrainer.models_dir.mkdir(parents=True, exist_ok=True)
        retrainer.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Create temp files
        temp_dir = retrainer.data_dir / 'temp'
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_master = temp_dir / 'master_panel.csv'
        temp_forecast = temp_dir / 'forecast_6m.parquet'
        temp_macro = temp_dir / 'macro_fc.parquet'
        
        temp_master.write_text('kode_bank,periode,CAR\n001,2024-01-01,15.0')
        temp_forecast.touch()
        temp_macro.touch()
        
        # Create model files
        (retrainer.models_dir / 'lgbm_reg_models.pkl').touch()
        
        metadata = {
            'version': '1.1.0', 
            'last_trained': datetime.now().isoformat(),
            'models': {
                'lgbm_reg': 'lgbm_reg_models.pkl'
            }
        }
        
        stage_dir = retrainer._stage_version(
            '1.1.0',
            temp_master,
            temp_forecast,
            temp_macro,
            metadata
        )
        
        # Verify staged directory contains everything
        assert (stage_dir / 'master_panel.csv').exists()
        assert (stage_dir / 'forecast_6m.parquet').exists()
        assert (stage_dir / 'macro_fc.parquet').exists()
        assert (stage_dir / 'lgbm_reg_models.pkl').exists()
        assert (stage_dir / 'model_metadata.json').exists()

    def test_promote_staged_version_creates_parquet(self, retrainer):
        retrainer.models_dir.mkdir(parents=True, exist_ok=True)
        retrainer.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Create staged version
        stage_dir = retrainer.models_dir / 'versions' / 'v1.1.0_staged'
        stage_dir.mkdir(parents=True)
        
        # Add CSV to staged
        csv_content = 'kode_bank,periode,CAR\n001,2024-01-01,15.0\n'
        (stage_dir / 'master_panel.csv').write_text(csv_content)
        (stage_dir / 'forecast_6m.parquet').touch()
        (stage_dir / 'macro_fc.parquet').touch()
        (stage_dir / 'model_metadata.json').write_text(json.dumps({'version': '1.1.0'}))
        
        # Promote
        retrainer._promote_staged_version(stage_dir)
        
        # Verify CSV promoted
        assert Path(retrainer.config['processed_data_file']).exists()
        
        # Verify Parquet created from CSV
        parquet_path = Path(retrainer.config['processed_data_file']).parent / 'master_panel.parquet'
        assert parquet_path.exists()
        
        # Verify both have same data
        df_csv = pd.read_csv(retrainer.config['processed_data_file'])
        df_parquet = pd.read_parquet(parquet_path)
        assert len(df_csv) == len(df_parquet)


class TestAutoRetrainerVerification:
    """Test strict verification checks"""
    
    def test_verification_fails_if_temp_files_missing(self, retrainer):
        retrainer.data_dir.mkdir(parents=True, exist_ok=True)
        
        merged_max = pd.Timestamp('2024-06-01')
        temp_master = retrainer.data_dir / 'master_panel_temp.csv'
        temp_forecast = retrainer.data_dir / 'forecast_temp.parquet'
        temp_macro = retrainer.data_dir / 'macro_temp.parquet'
        
        success, msg = retrainer._verify_retraining_success(
            merged_max, temp_master, temp_forecast, temp_macro
        )
        
        assert success is False
        assert 'missing' in msg.lower()

    def test_verification_checks_forecast_after_master(self, retrainer):
        retrainer.data_dir.mkdir(parents=True, exist_ok=True)
        
        merged_max = pd.Timestamp('2024-06-01')
        
        # Create temp master
        temp_master = retrainer.data_dir / 'master_panel_temp.csv'
        df_master = pd.DataFrame({
            'kode_bank': ['001'],
            'periode': [pd.Timestamp('2024-06-01')],
            'CAR': [15.0]
        })
        df_master.to_csv(temp_master, index=False)
        
        # Create temp forecast with wrong periods (not after master)
        temp_forecast = retrainer.data_dir / 'forecast_temp.parquet'
        df_forecast = pd.DataFrame({
            'kode_bank': ['001'],
            'periode': [pd.Timestamp('2024-05-01')],  # Before master!
            'CAR': [15.0],
            'is_forecast': [True]
        })
        df_forecast.to_parquet(temp_forecast, index=False)
        
        # Create temp macro
        temp_macro = retrainer.data_dir / 'macro_temp.parquet'
        df_macro = pd.DataFrame({
            'periode': pd.date_range('2024-07-01', periods=6, freq='MS'),
            'bi_rate_fc': [5.0] * 6,
            'inflasi_fc': [3.0] * 6,
            'kurs_usd_fc': [15000] * 6
        })
        df_macro.to_parquet(temp_macro, index=False)
        
        success, msg = retrainer._verify_retraining_success(
            merged_max, temp_master, temp_forecast, temp_macro
        )
        
        assert success is False
        assert 'not strictly after' in msg.lower()


class TestAutoRetrainerHFUpload:
    """Test Hugging Face artifact upload"""
    
    def test_upload_skipped_if_no_storage_manager(self, retrainer):
        # No storage_manager set
        retrainer.storage_manager = None
        
        # Should not raise, just log warning
        retrainer._upload_artifacts_to_hf('1.1.0')

    def test_upload_calls_storage_manager(self, retrainer):
        retrainer.models_dir.mkdir(parents=True, exist_ok=True)
        retrainer.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Create files to upload
        processed_dir = retrainer.data_dir / 'processed'
        processed_dir.mkdir(parents=True, exist_ok=True)
        master_csv = processed_dir / 'master_panel.csv'
        master_csv.write_text('kode_bank,periode,CAR\n001,2024-01-01,15.0')
        
        predictions_dir = retrainer.data_dir / 'predictions'
        predictions_dir.mkdir(parents=True, exist_ok=True)
        (predictions_dir / 'forecast_6m.parquet').touch()
        
        # Mock storage manager
        mock_storage = MagicMock()
        retrainer.storage_manager = mock_storage
        
        retrainer._upload_artifacts_to_hf('1.1.0')
        
        # Verify upload_file was called
        assert mock_storage.upload_file.call_count >= 2  # At least CSV and forecast


class TestAutoRetrainerCleanupOldVersions:
    def test_keeps_only_3_versions(self, retrainer):
        versions_dir = retrainer.models_dir / 'versions'
        versions_dir.mkdir(parents=True, exist_ok=True)
        for i in range(5):
            vdir = versions_dir / f'v1.0.{i}_20260101_12000{i}'
            vdir.mkdir()
            (vdir / 'model_metadata.json').write_text(json.dumps({'version': f'1.0.{i}'}))
        retrainer._cleanup_old_versions()
        assert len(list(versions_dir.iterdir())) == 3

    def test_fewer_than_3_untouched(self, retrainer):
        versions_dir = retrainer.models_dir / 'versions'
        versions_dir.mkdir(parents=True, exist_ok=True)
        for i in range(2):
            vdir = versions_dir / f'v1.0.{i}_20260101_12000{i}'
            vdir.mkdir()
        retrainer._cleanup_old_versions()
        assert len(list(versions_dir.iterdir())) == 2


class TestBankIdentityResolution:
    def test_canonical_identity(self, retrainer):
        retrainer.models_dir.mkdir(parents=True, exist_ok=True)
        retrainer.data_dir.mkdir(parents=True, exist_ok=True)
        retrainer.uploads_dir.mkdir(parents=True, exist_ok=True)
        
        # Initial master
        master_path = retrainer.data_dir / 'processed' / 'master_panel.csv'
        master_path.parent.mkdir(parents=True, exist_ok=True)
        
        df_master = pd.DataFrame({
            'kode_bank': ['008'],
            'bank': ['MANDIRI'],
            'periode': ['2023-01-01'],
            'CAR': [15.0]
        })
        df_master.to_csv(master_path, index=False)
        
        # New upload with mixed casing
        df_upload = pd.DataFrame({
            'kode_bank': ['008', '008'],
            'nama_bank': ['mandiri', 'Mandiri'],
            'periode': ['2023-02-01', '2023-03-01'],
            'CAR': [16.0, 17.0]
        })
        df_upload.to_csv(retrainer.uploads_dir / 'data.csv', index=False)
        
        # Merge
        merged_df, new_periods, gap_info = retrainer._merge_and_validate_uploads()
        
        # Assert canonical identity
        assert len(merged_df['kode_bank'].unique()) == 1
        assert merged_df['kode_bank'].iloc[0] == '008'
        name_col = 'nama_bank' if 'nama_bank' in merged_df.columns else 'bank'
        assert len(merged_df[name_col].unique()) == 1
        assert merged_df[name_col].iloc[0] == 'MANDIRI'


class TestUIDataGrouping:
    def test_dashboard_and_selector_grouping(self, tmp_path):
        import pandas as pd
        from src.data_loader import get_data_loader
        
        # Prepare mixed-case data
        df_mixed = pd.DataFrame({
            'kode_bank': ['008', '008', '008'],
            'nama_bank': ['MANDIRI', 'Mandiri', 'mandiri'],
            'periode': ['2023-01-01', '2023-02-01', '2023-03-01'],
            'CAR': [15.0, 16.0, 17.0],
            'pk_prediksi': [1, 2, 3]
        })
        
        # Mock load_master_data and load_predictions
        loader = get_data_loader()
        
        # Test get_bank_list returns exactly 1 item for kode_bank '008'
        with patch.object(loader, 'load_master_data', return_value=df_mixed):
            bank_list = loader.get_bank_list()
            assert len(bank_list) == 1
            assert bank_list[0]['kode_bank'] == '008'
        
        # Test dashboard logic (group by kode_bank)
        df_latest = df_mixed[df_mixed['periode'] == '2023-03-01'].copy()
        bank_id_col = 'kode_bank' if 'kode_bank' in df_latest.columns else 'nama_bank'
        total_banks = df_latest[bank_id_col].nunique()
        assert total_banks == 1


class TestDataServiceBankList:
    def test_get_bank_list_delegation(self):
        from src.data_service import DataService
        from unittest.mock import patch
        
        service = DataService()
        expected_list = [{'kode_bank': '001', 'nama_bank': 'Bank Test'}]
        
        with patch.object(service.loader, 'get_bank_list', return_value=expected_list) as mock_get:
            result = service.get_bank_list()
            mock_get.assert_called_once()
            assert result == expected_list


class TestPrediksiPageSavedForecasts:
    def test_prediksi_uses_saved_predictions_in_standard_mode(self):
        from pages import prediksi
        import pandas as pd
        from unittest.mock import patch, MagicMock
        
        mock_master = pd.DataFrame({
            'kode_bank': ['001'],
            'nama_bank': ['Bank Test'],
            'periode': [pd.Timestamp('2023-01-01')],
            'CAR': [15.0], 'NPL_gross': [1.0], 'NPL_net': [0.5], 'ROA': [2.0],
            'ROE': [10.0], 'LDR': [90.0], 'BOPO': [70.0], 'NIM': [5.0],
            'rbbr_score': [2.5]
        })
        
        mock_predictions = pd.DataFrame({
            'kode_bank': ['001'],
            'nama_bank': ['Bank Test'],
            'periode': [pd.Timestamp('2023-02-01')],
            'CAR_pred': [99.99],
            'is_forecast': [True]
        })
        
        mock_ds = MagicMock()
        mock_ds.load_master_data.return_value = mock_master
        mock_ds.load_predictions.return_value = mock_predictions
        mock_ds.get_bank_list.return_value = [{'kode_bank': '001', 'nama_bank': 'Bank Test'}]
        mock_ds.get_forecast_periods.return_value = [pd.Timestamp('2023-02-01')]
        
        # We need a mock res for predict_bank_6m
        mock_res = {
            'forecast_dates': pd.date_range(start='2023-02-01', periods=6, freq='MS'),
            'projections': {'CAR': [1.0] * 6} # A different value to ensure we use mock_predictions
        }
        
        with patch('pages.prediksi.DataService', return_value=mock_ds), \
             patch('pages.prediksi.st') as mock_st, \
             patch('src.model_inference.predict_bank_6m', return_value=mock_res), \
             patch('pages.prediksi.plot_trend_forecast') as mock_plot, \
             patch('pages.cache_listener.check_and_handle_cache_invalidation', return_value=False):
            
            mock_st.selectbox.return_value = '001 - Bank Test'
            
            mock_st.columns.side_effect = lambda n: [MagicMock() for _ in range(n)]
            
            mock_st.session_state = MagicMock()
            def mock_get(k, default=0.0):
                base = k.replace('sim_', '')
                if base in mock_master.columns:
                    return float(mock_master.iloc[-1][base])
                return default
            mock_st.session_state.get.side_effect = mock_get
            
            prediksi.show(None)
            
            assert mock_plot.call_count > 0
            
            # Find the call for 'CAR'
            car_fore = None
            for call_args in mock_plot.call_args_list:
                args, kwargs = call_args
                if kwargs.get('metric') == 'CAR':
                    car_fore = args[1]
                    break
            
            assert car_fore is not None
            assert 99.99 in car_fore['CAR_pred'].values




