"""Tests for src/data_validator.py"""

import json
import pytest
import numpy as np
import pandas as pd

from src.data_validator import DataValidator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _valid_df(n=10):
    """Minimal valid DataFrame that satisfies REQUIRED_COLS."""
    dates = pd.date_range('2026-01-01', periods=n, freq='MS')
    return pd.DataFrame({
        'periode':   dates,
        'kode_bank': [f'B{i:02d}' for i in range(n)],
        'bank':      [f'Bank {i}' for i in range(n)],
        'CAR':       [15.0] * n,
        'NPL_gross': [2.0] * n,
        'LDR':       [85.0] * n,
        'ROA':       [1.5] * n,
        'BOPO':      [70.0] * n,
    })


# ---------------------------------------------------------------------------
# validate_upload
# ---------------------------------------------------------------------------

class TestValidateUpload:

    def test_valid_data_passes(self):
        v = DataValidator()
        ok, report = v.validate_upload(_valid_df())
        assert ok is True
        assert report['errors'] == []

    def test_empty_dataframe_fails(self):
        v = DataValidator()
        ok, report = v.validate_upload(pd.DataFrame())
        assert ok is False
        assert len(report['errors']) > 0

    def test_missing_required_columns_fails(self):
        df = _valid_df().drop(columns=['CAR', 'NPL_gross'])
        v = DataValidator()
        ok, report = v.validate_upload(df)
        assert ok is False
        assert any('CAR' in e or 'NPL_gross' in e for e in report['errors'])

    def test_out_of_range_values_produce_warnings(self):
        df = _valid_df()
        df['CAR'] = 999.0  # way above max 200
        v = DataValidator()
        ok, report = v.validate_upload(df)
        # Out-of-range → warning, not error
        assert len(report['warnings']) > 0

    def test_sql_injection_rejected(self):
        df = _valid_df()
        df['bank'] = 'DROP TABLE banks;'
        v = DataValidator()
        ok, report = v.validate_upload(df)
        assert ok is False

    def test_invalid_periode_fails(self):
        df = _valid_df()
        df['periode'] = 'not-a-date'
        v = DataValidator()
        ok, report = v.validate_upload(df)
        assert ok is False

    def test_summary_contains_row_count(self):
        df = _valid_df(5)
        v = DataValidator()
        _, report = v.validate_upload(df)
        assert report['summary']['total_rows'] == 5

    def test_continuity_gap_too_large_fails(self):
        existing = _valid_df(12)  # up to 2026-12-01
        new_df = _valid_df(3)
        # Shift new data 6 months ahead of existing max
        new_df['periode'] = pd.date_range('2027-07-01', periods=3, freq='MS')
        v = DataValidator()
        ok, report = v.validate_upload(new_df, existing_data=existing)
        assert ok is False
        assert any('Gap' in e or 'gap' in e for e in report['errors'])

    def test_continuity_overlap_produces_warning(self):
        existing = _valid_df(12)
        new_df = _valid_df(3)
        # new data overlaps with existing
        new_df['periode'] = pd.date_range('2026-06-01', periods=3, freq='MS')
        v = DataValidator()
        ok, report = v.validate_upload(new_df, existing_data=existing)
        assert len(report['warnings']) > 0

    def test_high_null_critical_col_fails(self):
        df = _valid_df(20)
        df.loc[:15, 'CAR'] = np.nan  # >50% null
        v = DataValidator()
        ok, report = v.validate_upload(df)
        assert ok is False


# ---------------------------------------------------------------------------
# check_retraining_readiness
# ---------------------------------------------------------------------------

class TestCheckRetrainingReadiness:

    def test_no_master_file_not_ready(self, tmp_path):
        (tmp_path / 'processed').mkdir()
        (tmp_path / 'uploads').mkdir()
        v = DataValidator()
        ready, info = v.check_retraining_readiness(tmp_path)
        assert ready is False
        assert info['checks']['master_data_exists'] is False

    def test_master_exists_no_upload_not_ready(self, tmp_path):
        proc = tmp_path / 'processed'
        proc.mkdir()
        (tmp_path / 'uploads').mkdir()
        # Write a valid master panel
        df = _valid_df(120)
        df.to_csv(proc / 'master_panel.csv', index=False)
        v = DataValidator()
        ready, info = v.check_retraining_readiness(tmp_path)
        assert ready is False
        assert info['checks']['master_data_exists'] is True
        assert info['checks']['has_new_upload'] is False

    def test_master_and_upload_ready(self, tmp_path):
        proc = tmp_path / 'processed'
        proc.mkdir()
        uploads = tmp_path / 'uploads'
        uploads.mkdir()
        # Master: 2025-01 to 2025-12
        master = _valid_df(12)
        master.to_csv(proc / 'master_panel.csv', index=False)
        # Upload: 2026-01 to 2026-03 (continuous)
        upload = _valid_df(3)
        upload['periode'] = pd.date_range('2026-01-01', periods=3, freq='MS')
        upload.to_csv(uploads / 'new_data.csv', index=False)
        v = DataValidator()
        ready, info = v.check_retraining_readiness(tmp_path)
        assert info['checks']['master_data_exists'] is True
        assert info['checks']['has_new_upload'] is True

    def test_master_too_small_not_ready(self, tmp_path):
        proc = tmp_path / 'processed'
        proc.mkdir()
        (tmp_path / 'uploads').mkdir()
        # Only 5 rows — below 100 minimum
        _valid_df(5).to_csv(proc / 'master_panel.csv', index=False)
        v = DataValidator()
        ready, info = v.check_retraining_readiness(tmp_path)
        assert ready is False
        assert any('100' in e for e in info['errors'])

    def test_upload_gap_too_large_not_ready(self, tmp_path):
        proc = tmp_path / 'processed'
        proc.mkdir()
        uploads = tmp_path / 'uploads'
        uploads.mkdir()
        # Master: 2025-01 to 2025-12 (12 months)
        master = _valid_df(12)
        master.to_csv(proc / 'master_panel.csv', index=False)
        # Upload starts 6 months after master max (2025-12 → gap to 2026-07)
        upload = _valid_df(3)
        upload['periode'] = pd.date_range('2026-07-01', periods=3, freq='MS')
        upload.to_csv(uploads / 'new_data.csv', index=False)
        v = DataValidator()
        ready, info = v.check_retraining_readiness(tmp_path)
        assert ready is False
