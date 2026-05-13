"""
Regression tests for kode_bank normalization
Tests that numeric CSV uploads with codes 008, 009, 002, 200 are properly normalized
"""

import sys
from pathlib import Path
import pandas as pd
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data_utils import normalize_kode_bank


def test_normalize_kode_bank_numeric_int():
    """Test normalization of integer bank codes"""
    df = pd.DataFrame({
        'kode_bank': [8, 9, 2, 200, 1, 11],
        'value': [1, 2, 3, 4, 5, 6]
    })
    
    result = normalize_kode_bank(df)
    
    assert result['kode_bank'].tolist() == ['008', '009', '002', '200', '001', '011']
    assert result['kode_bank'].dtype == object


def test_normalize_kode_bank_numeric_float():
    """Test normalization of float bank codes (from CSV numeric columns)"""
    df = pd.DataFrame({
        'kode_bank': [8.0, 9.0, 2.0, 200.0, 1.0, 11.0],
        'value': [1, 2, 3, 4, 5, 6]
    })
    
    result = normalize_kode_bank(df)
    
    assert result['kode_bank'].tolist() == ['008', '009', '002', '200', '001', '011']
    assert result['kode_bank'].dtype == object


def test_normalize_kode_bank_string():
    """Test normalization of string bank codes"""
    df = pd.DataFrame({
        'kode_bank': ['8', '9', '2', '200', '1', '11'],
        'value': [1, 2, 3, 4, 5, 6]
    })
    
    result = normalize_kode_bank(df)
    
    assert result['kode_bank'].tolist() == ['008', '009', '002', '200', '001', '011']


def test_normalize_kode_bank_mixed():
    """Test normalization of mixed types"""
    df = pd.DataFrame({
        'kode_bank': [8, '9', 2.0, '200', 1, '11'],
        'value': [1, 2, 3, 4, 5, 6]
    })
    
    result = normalize_kode_bank(df)
    
    assert result['kode_bank'].tolist() == ['008', '009', '002', '200', '001', '011']


def test_normalize_kode_bank_with_whitespace():
    """Test normalization strips whitespace"""
    df = pd.DataFrame({
        'kode_bank': [' 8 ', '9  ', '  2', '200', ' 1', '11 '],
        'value': [1, 2, 3, 4, 5, 6]
    })
    
    result = normalize_kode_bank(df)
    
    assert result['kode_bank'].tolist() == ['008', '009', '002', '200', '001', '011']


def test_normalize_kode_bank_already_normalized():
    """Test that already normalized codes remain unchanged"""
    df = pd.DataFrame({
        'kode_bank': ['008', '009', '002', '200', '001', '011'],
        'value': [1, 2, 3, 4, 5, 6]
    })
    
    result = normalize_kode_bank(df)
    
    assert result['kode_bank'].tolist() == ['008', '009', '002', '200', '001', '011']


def test_normalize_kode_bank_no_column():
    """Test that DataFrames without kode_bank column are returned unchanged"""
    df = pd.DataFrame({
        'other_column': [1, 2, 3],
        'value': [4, 5, 6]
    })
    
    result = normalize_kode_bank(df)
    
    assert 'kode_bank' not in result.columns
    assert result.equals(df)


def test_normalize_kode_bank_preserves_other_columns():
    """Test that other columns are preserved"""
    df = pd.DataFrame({
        'kode_bank': [8, 9, 2],
        'nama_bank': ['Bank A', 'Bank B', 'Bank C'],
        'CAR': [15.5, 16.2, 14.8],
        'periode': pd.to_datetime(['2024-01-01', '2024-01-01', '2024-01-01'])
    })
    
    result = normalize_kode_bank(df)
    
    assert result['kode_bank'].tolist() == ['008', '009', '002']
    assert result['nama_bank'].tolist() == ['Bank A', 'Bank B', 'Bank C']
    assert result['CAR'].tolist() == [15.5, 16.2, 14.8]
    assert result['periode'].tolist() == df['periode'].tolist()


def test_normalize_kode_bank_deduplication():
    """Test that normalized codes can be used for deduplication"""
    df = pd.DataFrame({
        'kode_bank': [8, 8.0, '008', 9, '009'],
        'periode': pd.to_datetime(['2024-01-01', '2024-01-01', '2024-01-01', '2024-01-01', '2024-01-01']),
        'value': [1, 2, 3, 4, 5]
    })
    
    result = normalize_kode_bank(df)
    deduped = result.drop_duplicates(subset=['kode_bank', 'periode'], keep='last')
    
    # Should have only 2 unique banks after normalization and deduplication
    assert len(deduped) == 2
    assert set(deduped['kode_bank'].unique()) == {'008', '009'}
    # Should keep last values (3 for 008, 5 for 009)
    assert deduped[deduped['kode_bank'] == '008']['value'].iloc[0] == 3
    assert deduped[deduped['kode_bank'] == '009']['value'].iloc[0] == 5


def test_normalize_kode_bank_groupby():
    """Test that normalized codes work correctly with groupby"""
    df = pd.DataFrame({
        'kode_bank': [8, 8.0, '008', 9, '009', 2],
        'value': [10, 20, 30, 40, 50, 60]
    })
    
    result = normalize_kode_bank(df)
    grouped = result.groupby('kode_bank')['value'].sum()
    
    # All variants of 008 should be grouped together
    assert grouped['008'] == 60  # 10 + 20 + 30
    assert grouped['009'] == 90  # 40 + 50
    assert grouped['002'] == 60


def test_normalize_kode_bank_merge():
    """Test that normalized codes work correctly with merge/join"""
    df1 = pd.DataFrame({
        'kode_bank': [8, 9, 2],
        'value1': [1, 2, 3]
    })
    
    df2 = pd.DataFrame({
        'kode_bank': ['008', '009', '002'],
        'value2': [10, 20, 30]
    })
    
    df1_norm = normalize_kode_bank(df1)
    df2_norm = normalize_kode_bank(df2)
    
    merged = pd.merge(df1_norm, df2_norm, on='kode_bank')
    
    assert len(merged) == 3
    assert merged['kode_bank'].tolist() == ['008', '009', '002']
    assert merged['value1'].tolist() == [1, 2, 3]
    assert merged['value2'].tolist() == [10, 20, 30]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
