"""
Test per-bank forecast verification in retraining pipeline
"""

import sys
from pathlib import Path
import pandas as pd
import tempfile
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.auto_retraining_new import AutoRetrainer


def test_verify_missing_bank():
    """Test that verification fails when a bank is missing from forecast"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create master panel with 3 banks
        master = pd.DataFrame({
            'kode_bank': ['001', '002', '003'] * 6,
            'periode': pd.date_range('2024-01-01', periods=6, freq='MS').tolist() * 3,
            'CAR': [15.0] * 18,
            'NPL_gross': [2.0] * 18
        })
        master_path = tmpdir / 'master.csv'
        master.to_csv(master_path, index=False)
        
        # Create forecast with only 2 banks (missing '003')
        forecast = pd.DataFrame({
            'kode_bank': ['001', '002'] * 6,
            'periode': pd.date_range('2024-07-01', periods=6, freq='MS').tolist() * 2,
            'CAR': [15.0] * 12,
            'NPL_gross': [2.0] * 12,
            'is_forecast': [True] * 12
        })
        forecast_path = tmpdir / 'forecast.parquet'
        forecast.to_parquet(forecast_path, index=False)
        
        # Create macro forecast
        macro = pd.DataFrame({
            'periode': pd.date_range('2024-07-01', periods=6, freq='MS'),
            'bi_rate_fc': [5.0] * 6,
            'inflasi_fc': [3.0] * 6,
            'kurs_usd_fc': [15000] * 6
        })
        macro_path = tmpdir / 'macro.parquet'
        macro.to_parquet(macro_path, index=False)
        
        # Run verification
        retrainer = AutoRetrainer()
        retrainer.config['prediction_horizon'] = 6
        
        success, msg = retrainer._verify_retraining_success(
            pd.Timestamp('2024-06-01'),
            master_path,
            forecast_path,
            macro_path
        )
        
        assert not success
        assert 'missing from forecast' in msg.lower()
        assert '003' in msg


def test_verify_incomplete_periods():
    """Test that verification fails when a bank has incomplete forecast periods"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create master panel
        master = pd.DataFrame({
            'kode_bank': ['001', '002'] * 6,
            'periode': pd.date_range('2024-01-01', periods=6, freq='MS').tolist() * 2,
            'CAR': [15.0] * 12,
            'NPL_gross': [2.0] * 12
        })
        master_path = tmpdir / 'master.csv'
        master.to_csv(master_path, index=False)
        
        # Create forecast with bank '002' missing some periods
        forecast_001 = pd.DataFrame({
            'kode_bank': ['001'] * 6,
            'periode': pd.date_range('2024-07-01', periods=6, freq='MS'),
            'CAR': [15.0] * 6,
            'NPL_gross': [2.0] * 6,
            'is_forecast': [True] * 6
        })
        forecast_002 = pd.DataFrame({
            'kode_bank': ['002'] * 4,  # Only 4 periods instead of 6
            'periode': pd.date_range('2024-07-01', periods=4, freq='MS'),
            'CAR': [15.0] * 4,
            'NPL_gross': [2.0] * 4,
            'is_forecast': [True] * 4
        })
        forecast = pd.concat([forecast_001, forecast_002], ignore_index=True)
        forecast_path = tmpdir / 'forecast.parquet'
        forecast.to_parquet(forecast_path, index=False)
        
        # Create macro forecast
        macro = pd.DataFrame({
            'periode': pd.date_range('2024-07-01', periods=6, freq='MS'),
            'bi_rate_fc': [5.0] * 6,
            'inflasi_fc': [3.0] * 6,
            'kurs_usd_fc': [15000] * 6
        })
        macro_path = tmpdir / 'macro.parquet'
        macro.to_parquet(macro_path, index=False)
        
        # Run verification
        retrainer = AutoRetrainer()
        retrainer.config['prediction_horizon'] = 6
        
        success, msg = retrainer._verify_retraining_success(
            pd.Timestamp('2024-06-01'),
            master_path,
            forecast_path,
            macro_path
        )
        
        assert not success
        assert 'incomplete forecast periods' in msg.lower()
        assert '002' in msg


def test_verify_null_targets():
    """Test that verification fails when forecast has null target values"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create master panel
        master = pd.DataFrame({
            'kode_bank': ['001', '002'] * 6,
            'periode': pd.date_range('2024-01-01', periods=6, freq='MS').tolist() * 2,
            'CAR': [15.0] * 12,
            'NPL_gross': [2.0] * 12
        })
        master_path = tmpdir / 'master.csv'
        master.to_csv(master_path, index=False)
        
        # Create forecast with null values for bank '002'
        forecast_001 = pd.DataFrame({
            'kode_bank': ['001'] * 6,
            'periode': pd.date_range('2024-07-01', periods=6, freq='MS'),
            'CAR': [15.0] * 6,
            'NPL_gross': [2.0] * 6,
            'is_forecast': [True] * 6
        })
        forecast_002 = pd.DataFrame({
            'kode_bank': ['002'] * 6,
            'periode': pd.date_range('2024-07-01', periods=6, freq='MS'),
            'CAR': [None, 15.0, 15.0, None, 15.0, 15.0],  # Some nulls
            'NPL_gross': [2.0] * 6,
            'is_forecast': [True] * 6
        })
        forecast = pd.concat([forecast_001, forecast_002], ignore_index=True)
        forecast_path = tmpdir / 'forecast.parquet'
        forecast.to_parquet(forecast_path, index=False)
        
        # Create macro forecast
        macro = pd.DataFrame({
            'periode': pd.date_range('2024-07-01', periods=6, freq='MS'),
            'bi_rate_fc': [5.0] * 6,
            'inflasi_fc': [3.0] * 6,
            'kurs_usd_fc': [15000] * 6
        })
        macro_path = tmpdir / 'macro.parquet'
        macro.to_parquet(macro_path, index=False)
        
        # Run verification
        retrainer = AutoRetrainer()
        retrainer.config['prediction_horizon'] = 6
        
        success, msg = retrainer._verify_retraining_success(
            pd.Timestamp('2024-06-01'),
            master_path,
            forecast_path,
            macro_path
        )
        
        assert not success
        assert 'null forecast targets' in msg.lower()
        assert '002' in msg
        assert 'CAR' in msg


def test_verify_success_all_banks_complete():
    """Test that verification passes when all banks have complete forecasts"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create master panel
        master = pd.DataFrame({
            'kode_bank': ['001', '002'] * 6,
            'periode': pd.date_range('2024-01-01', periods=6, freq='MS').tolist() * 2,
            'CAR': [15.0] * 12,
            'NPL_gross': [2.0] * 12
        })
        master_path = tmpdir / 'master.csv'
        master.to_csv(master_path, index=False)
        
        # Create complete forecast for all banks
        forecast_data = []
        for bank in ['001', '002']:
            for period in pd.date_range('2024-07-01', periods=6, freq='MS'):
                forecast_data.append({
                    'kode_bank': bank,
                    'periode': period,
                    'CAR': 15.0,
                    'NPL_gross': 2.0,
                    'NPL_net': 1.0,
                    'ROA': 2.0,
                    'ROE': 15.0,
                    'NIM': 5.0,
                    'BOPO': 70.0,
                    'LDR': 85.0,
                    'is_forecast': True
                })
        forecast = pd.DataFrame(forecast_data)
        forecast_path = tmpdir / 'forecast.parquet'
        forecast.to_parquet(forecast_path, index=False)
        
        # Create macro forecast
        macro = pd.DataFrame({
            'periode': pd.date_range('2024-07-01', periods=6, freq='MS'),
            'bi_rate_fc': [5.0] * 6,
            'inflasi_fc': [3.0] * 6,
            'kurs_usd_fc': [15000] * 6
        })
        macro_path = tmpdir / 'macro.parquet'
        macro.to_parquet(macro_path, index=False)
        
        # Run verification
        retrainer = AutoRetrainer()
        retrainer.config['prediction_horizon'] = 6
        
        success, msg = retrainer._verify_retraining_success(
            pd.Timestamp('2024-06-01'),
            master_path,
            forecast_path,
            macro_path
        )
        
        if not success:
            print(f"Verification failed: {msg}")
        
        assert success
        assert 'all verification checks passed' in msg.lower()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
