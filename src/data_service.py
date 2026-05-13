"""
Data Service - Shared helpers for historical vs forecast data separation
"""

import pandas as pd
import logging
from typing import Optional, Tuple, List
from src.data_loader import get_data_loader

logger = logging.getLogger(__name__)


class DataService:
    """Wrapper around DataLoader with forecast/historical separation helpers"""
    
    def __init__(self):
        self.loader = get_data_loader()
    
    def load_master_data(self) -> pd.DataFrame:
        return self.loader.load_master_data()
    
    def load_predictions(self) -> pd.DataFrame:
        return self.loader.load_predictions()
    
    def load_models(self):
        return self.loader.load_models()
    
    def load_macro_variables(self) -> pd.DataFrame:
        return self.loader.load_macro_data()
    
    def load_bank_metadata(self) -> pd.DataFrame:
        return self.loader.load_bank_metadata()
    
    def get_available_periods(self) -> List[pd.Timestamp]:
        """Get list of historical periods"""
        master = self.load_master_data()
        if master.empty or 'periode' not in master.columns:
            return []
        return sorted(master['periode'].unique())
    
    def get_forecast_periods(self) -> List[pd.Timestamp]:
        """Get list of forecast periods"""
        predictions = self.load_predictions()
        if predictions.empty:
            return []
        
        # Filter forecast periods
        if 'is_forecast' in predictions.columns:
            forecast = predictions[predictions['is_forecast'] == True]
        else:
            master = self.load_master_data()
            latest_hist = get_latest_historical_period(master)
            if latest_hist:
                forecast = predictions[predictions['periode'] > latest_hist]
            else:
                forecast = predictions
        
        if forecast.empty or 'periode' not in forecast.columns:
            return []
        return sorted(forecast['periode'].unique())
    
    def get_latest_historical_period(self) -> Optional[pd.Timestamp]:
        """Get latest historical period"""
        master = self.load_master_data()
        return get_latest_historical_period(master)

    def get_bank_list(self) -> List[dict]:
        """Get list of banks delegating to DataLoader"""
        return self.loader.get_bank_list()


def get_latest_historical_period(df_master: pd.DataFrame) -> Optional[pd.Timestamp]:
    """
    Get the latest historical period from master data
    
    Args:
        df_master: Master panel DataFrame with 'periode' column
        
    Returns:
        Latest historical period as Timestamp, or None if empty
    """
    if df_master.empty or 'periode' not in df_master.columns:
        return None
    return df_master['periode'].max()


def get_forecast_data(df_predictions: pd.DataFrame, df_master: pd.DataFrame) -> pd.DataFrame:
    """
    Extract forecast-only data (periods after latest historical)
    
    Args:
        df_predictions: Predictions DataFrame with 'periode' and optional 'is_forecast'
        df_master: Master panel DataFrame to determine latest historical period
        
    Returns:
        DataFrame containing only forecast rows
    """
    if df_predictions.empty:
        return pd.DataFrame()
    
    # If is_forecast column exists, use it
    if 'is_forecast' in df_predictions.columns:
        return df_predictions[df_predictions['is_forecast'] == True].copy()
    
    # Fallback: filter by latest historical period
    latest_hist = get_latest_historical_period(df_master)
    if latest_hist is None:
        logger.warning("No historical data found, returning all predictions as forecast")
        return df_predictions.copy()
    
    return df_predictions[df_predictions['periode'] > latest_hist].copy()


def get_historical_data(df_predictions: pd.DataFrame, df_master: pd.DataFrame) -> pd.DataFrame:
    """
    Extract historical data (periods up to and including latest historical)
    
    Args:
        df_predictions: Predictions DataFrame with 'periode' and optional 'is_forecast'
        df_master: Master panel DataFrame to determine latest historical period
        
    Returns:
        DataFrame containing only historical rows
    """
    if df_predictions.empty:
        return pd.DataFrame()
    
    # If is_forecast column exists, use it
    if 'is_forecast' in df_predictions.columns:
        return df_predictions[df_predictions['is_forecast'] == False].copy()
    
    # Fallback: filter by latest historical period
    latest_hist = get_latest_historical_period(df_master)
    if latest_hist is None:
        logger.warning("No historical data found, returning empty DataFrame")
        return pd.DataFrame()
    
    return df_predictions[df_predictions['periode'] <= latest_hist].copy()


def add_forecast_discriminator(df_predictions: pd.DataFrame, latest_historical_period: pd.Timestamp) -> pd.DataFrame:
    """
    Add is_forecast column to predictions DataFrame
    
    Args:
        df_predictions: Predictions DataFrame with 'periode' column
        latest_historical_period: Latest historical period timestamp
        
    Returns:
        DataFrame with is_forecast column added
    """
    df = df_predictions.copy()
    df['is_forecast'] = df['periode'] > latest_historical_period
    return df
