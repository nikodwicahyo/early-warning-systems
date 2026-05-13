"""
Regression test: Verify is_forecast discriminator correctly separates historical from forecast
Test case: latest_history=2026-08, forecast=2026-09..2027-02
"""

import pandas as pd
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_service import (
    get_latest_historical_period,
    get_forecast_data,
    get_historical_data,
    add_forecast_discriminator
)


def test_forecast_discriminator_exact_case():
    """Test exact case: latest_history=2026-08, forecast=2026-09..2027-02"""
    
    # Create master data with historical periods up to 2026-08
    df_master = pd.DataFrame({
        'kode_bank': ['001'] * 8,
        'periode': pd.to_datetime([
            '2026-01-01', '2026-02-01', '2026-03-01', '2026-04-01',
            '2026-05-01', '2026-06-01', '2026-07-01', '2026-08-01'
        ]),
        'CAR': [15.0] * 8,
        'NPL_gross': [2.0] * 8,
    })
    
    # Create predictions with both historical (2026-08) and forecast (2026-09..2027-02)
    df_predictions = pd.DataFrame({
        'kode_bank': ['001'] * 7,
        'periode': pd.to_datetime([
            '2026-08-01',  # Historical (last month)
            '2026-09-01', '2026-10-01', '2026-11-01',  # Forecast
            '2026-12-01', '2027-01-01', '2027-02-01'   # Forecast
        ]),
        'CAR': [15.0, 15.1, 15.2, 15.3, 15.4, 15.5, 15.6],
        'NPL_gross': [2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6],
        'is_forecast': [False, True, True, True, True, True, True]
    })
    
    # Test 1: get_latest_historical_period
    latest = get_latest_historical_period(df_master)
    assert latest == pd.Timestamp('2026-08-01'), \
        f"Expected latest historical period 2026-08-01, got {latest}"
    print("[PASS] Latest historical period correctly identified as 2026-08")
    
    # Test 2: get_forecast_data should return only 2026-09..2027-02
    df_forecast = get_forecast_data(df_predictions, df_master)
    assert len(df_forecast) == 6, \
        f"Expected 6 forecast rows (2026-09 to 2027-02), got {len(df_forecast)}"
    assert df_forecast['periode'].min() == pd.Timestamp('2026-09-01'), \
        f"Expected forecast start 2026-09-01, got {df_forecast['periode'].min()}"
    assert df_forecast['periode'].max() == pd.Timestamp('2027-02-01'), \
        f"Expected forecast end 2027-02-01, got {df_forecast['periode'].max()}"
    print("[PASS] Forecast data correctly filtered to 2026-09..2027-02 (6 months)")
    
    # Test 3: get_historical_data should return only 2026-08
    df_historical = get_historical_data(df_predictions, df_master)
    assert len(df_historical) == 1, \
        f"Expected 1 historical row (2026-08), got {len(df_historical)}"
    assert df_historical['periode'].iloc[0] == pd.Timestamp('2026-08-01'), \
        f"Expected historical period 2026-08-01, got {df_historical['periode'].iloc[0]}"
    print("[PASS] Historical data correctly filtered to 2026-08 only")
    
    # Test 4: Verify is_forecast flag is correct
    assert all(df_forecast['is_forecast'] == True), \
        "All forecast rows should have is_forecast=True"
    assert all(df_historical['is_forecast'] == False), \
        "All historical rows should have is_forecast=False"
    print("[PASS] is_forecast flag correctly set for all rows")


def test_add_forecast_discriminator():
    """Test add_forecast_discriminator function"""
    
    df_predictions = pd.DataFrame({
        'kode_bank': ['001'] * 7,
        'periode': pd.to_datetime([
            '2026-08-01',  # Historical
            '2026-09-01', '2026-10-01', '2026-11-01',  # Forecast
            '2026-12-01', '2027-01-01', '2027-02-01'   # Forecast
        ]),
        'CAR': [15.0, 15.1, 15.2, 15.3, 15.4, 15.5, 15.6],
    })
    
    latest_historical = pd.Timestamp('2026-08-01')
    df_with_flag = add_forecast_discriminator(df_predictions, latest_historical)
    
    # Check that is_forecast column was added
    assert 'is_forecast' in df_with_flag.columns, \
        "is_forecast column should be added"
    
    # Check that only periods after 2026-08 are marked as forecast
    forecast_count = df_with_flag['is_forecast'].sum()
    assert forecast_count == 6, \
        f"Expected 6 forecast rows, got {forecast_count}"
    
    historical_count = (~df_with_flag['is_forecast']).sum()
    assert historical_count == 1, \
        f"Expected 1 historical row, got {historical_count}"
    
    print("[PASS] add_forecast_discriminator correctly adds is_forecast flag")


def test_fallback_without_is_forecast_column():
    """Test that helpers work even without is_forecast column (fallback mode)"""
    
    df_master = pd.DataFrame({
        'kode_bank': ['001'] * 8,
        'periode': pd.to_datetime([
            '2026-01-01', '2026-02-01', '2026-03-01', '2026-04-01',
            '2026-05-01', '2026-06-01', '2026-07-01', '2026-08-01'
        ]),
        'CAR': [15.0] * 8,
    })
    
    # Predictions WITHOUT is_forecast column
    df_predictions = pd.DataFrame({
        'kode_bank': ['001'] * 7,
        'periode': pd.to_datetime([
            '2026-08-01',  # Historical
            '2026-09-01', '2026-10-01', '2026-11-01',  # Forecast
            '2026-12-01', '2027-01-01', '2027-02-01'   # Forecast
        ]),
        'CAR': [15.0, 15.1, 15.2, 15.3, 15.4, 15.5, 15.6],
    })
    
    # Should still work using periode comparison
    df_forecast = get_forecast_data(df_predictions, df_master)
    assert len(df_forecast) == 6, \
        f"Fallback mode: Expected 6 forecast rows, got {len(df_forecast)}"
    
    df_historical = get_historical_data(df_predictions, df_master)
    assert len(df_historical) == 1, \
        f"Fallback mode: Expected 1 historical row, got {len(df_historical)}"
    
    print("[PASS] Fallback mode works without is_forecast column")


def test_edge_case_no_overlap():
    """Test case where predictions start after historical data ends"""
    
    df_master = pd.DataFrame({
        'kode_bank': ['001'] * 3,
        'periode': pd.to_datetime(['2026-06-01', '2026-07-01', '2026-08-01']),
        'CAR': [15.0] * 3,
    })
    
    # Predictions start from 2026-09 (no overlap)
    df_predictions = pd.DataFrame({
        'kode_bank': ['001'] * 6,
        'periode': pd.to_datetime([
            '2026-09-01', '2026-10-01', '2026-11-01',
            '2026-12-01', '2027-01-01', '2027-02-01'
        ]),
        'CAR': [15.1, 15.2, 15.3, 15.4, 15.5, 15.6],
        'is_forecast': [True] * 6
    })
    
    df_forecast = get_forecast_data(df_predictions, df_master)
    assert len(df_forecast) == 6, \
        f"No overlap case: Expected 6 forecast rows, got {len(df_forecast)}"
    
    df_historical = get_historical_data(df_predictions, df_master)
    assert len(df_historical) == 0, \
        f"No overlap case: Expected 0 historical rows, got {len(df_historical)}"
    
    print("[PASS] Edge case with no overlap handled correctly")


if __name__ == '__main__':
    print("Running regression tests for forecast discriminator...")
    print()
    
    test_forecast_discriminator_exact_case()
    test_add_forecast_discriminator()
    test_fallback_without_is_forecast_column()
    test_edge_case_no_overlap()
    
    print()
    print("[SUCCESS] All forecast discriminator tests passed!")
