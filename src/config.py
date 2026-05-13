"""
Application Configuration
Centralized settings and constants
"""

from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List
import json

# =============================================================================
# PATHS
# =============================================================================

ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / 'data'
MODEL_DIR = ROOT_DIR / 'models'
ASSETS_DIR = ROOT_DIR / 'assets'
LOGS_DIR = ROOT_DIR / 'logs'

# Create directories if not exist
for dir_path in [DATA_DIR, MODEL_DIR, ASSETS_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# =============================================================================
# RBBR CONFIGURATION
# =============================================================================

@dataclass
class RBBRConfig:
    """RBBR scoring thresholds and weights"""
    
    # Thresholds untuk NPL
    NPL_THRESHOLDS = {
        1: (0, 2),      # Sangat Sehat
        2: (2, 5),      # Sehat
        3: (5, 8),      # Cukup Sehat
        4: (8, 12),     # Kurang Sehat
        5: (12, 100),   # Tidak Sehat
    }
    
    # Thresholds untuk CAR
    CAR_THRESHOLDS = {
        1: (15, 100),   # Sangat Sehat
        2: (12, 15),    # Sehat
        3: (10, 12),    # Cukup Sehat
        4: (8, 10),     # Kurang Sehat
        5: (0, 8),      # Tidak Sehat
    }
    
    # Thresholds untuk LDR
    LDR_THRESHOLDS = {
        1: (78, 92),    # Sangat Sehat
        2: [(60, 78), (92, 100)],  # Sehat
        3: [(50, 60), (100, 110)], # Cukup Sehat
        4: [(40, 50), (110, 120)], # Kurang Sehat
        5: [(0, 40), (120, 200)],  # Tidak Sehat
    }
    
    # Thresholds untuk ROA
    ROA_THRESHOLDS = {
        1: (1.5, 100),   # Sangat Sehat
        2: (1.25, 1.5),  # Sehat
        3: (0.5, 1.25),  # Cukup Sehat
        4: (0.0, 0.5),   # Kurang Sehat
        5: (-100, 0.0),  # Tidak Sehat
    }
    
    # Thresholds untuk BOPO
    BOPO_THRESHOLDS = {
        1: (0, 70),      # Sangat Sehat
        2: (70, 80),     # Sehat
        3: (80, 85),     # Cukup Sehat
        4: (85, 90),     # Kurang Sehat
        5: (90, 200),    # Tidak Sehat
    }
    
    # Bobot per pilar
    WEIGHTS = {
        'risk_profile': 0.50,
        'gcg': 0.15,
        'rentabilitas': 0.15,
        'permodalan': 0.20,
    }
    
    # PK Labels
    PK_LABELS = {
        1: 'Sangat Sehat',
        2: 'Sehat',
        3: 'Cukup Sehat',
        4: 'Kurang Sehat',
        5: 'Tidak Sehat',
    }
    
    # PK Colors (untuk visualisasi)
    PK_COLORS = {
        1: '#2d6a4f',  # Dark Green
        2: '#52b788',  # Green
        3: '#f9c74f',  # Yellow
        4: '#f4845f',  # Orange
        5: '#d62828',  # Red
    }
    
    # Sinyal EWS
    SINYAL_COLORS = {
        'hijau': '#2d6a4f',
        'kuning': '#f9c74f',
        'merah': '#d62828',
    }

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

@dataclass
class ModelConfig:
    """Model training and inference configuration"""
    
    # Feature columns
    LAG_FEATURES = [
        'CAR', 'NPL_gross', 'NPL_net', 'NIM', 'ROA', 'ROE',
        'BOPO', 'LDR', 'LCR', 'total_aset', 'kredit_yang_diberikan',
        'suku_bunga_bi_rate', 'inflasi_yoy', 'kurs_usd_idr'
    ]
    
    # Rolling windows
    ROLLING_WINDOWS = [3, 6]
    
    # TFT targets
    TFT_TARGETS = ['CAR', 'NPL_gross', 'NPL_net', 'NIM', 'ROA', 'ROE', 'BOPO', 'LDR', 'rbbr_score']
    
    # Retraining parameters
    RETRAIN_MIN_MONTHS = 36  # Minimum 3 tahun data
    RETRAIN_VALIDATION_SPLIT = 0.2
    RETRAIN_TEST_SPLIT = 0.1

# =============================================================================
# APP CONFIGURATION
# =============================================================================

@dataclass
class AppConfig:
    """Main application configuration"""
    
    # App metadata
    APP_NAME = "OJK RBBR Early Warning System"
    VERSION = "1.0.0"
    
    # Data refresh
    CACHE_TTL = 3600  # 1 hour in seconds
    AUTO_REFRESH = True
    
    # Visualization
    PLOTLY_TEMPLATE = "plotly_white"
    DEFAULT_HEIGHT = 500
    
    # Upload settings
    MAX_UPLOAD_SIZE_MB = 10
    ALLOWED_EXTENSIONS = ['.xlsx', '.csv', '.parquet']
    
    # Logging
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Export settings
    EXPORT_FORMATS = ['CSV', 'Excel', 'Parquet']
    
    # Pagination
    ROWS_PER_PAGE = 50
    
    # Monitoring thresholds
    ALERT_NPL_THRESHOLD = 5.0  # %
    ALERT_CAR_THRESHOLD = 10.0  # %
    ALERT_LDR_THRESHOLD = 100.0  # %

# =============================================================================
# LOAD CONFIG FROM FILE (if exists)
# =============================================================================

def load_config_from_file():
    """Load configuration from JSON file if exists"""
    config_file = ROOT_DIR / 'config.json'
    
    if config_file.exists():
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load config.json: {e}")
            return {}
    return {}

# Load custom config
CUSTOM_CONFIG = load_config_from_file()

# =============================================================================
# EXPORT
# =============================================================================

__all__ = [
    'ROOT_DIR',
    'DATA_DIR',
    'MODEL_DIR',
    'ASSETS_DIR',
    'LOGS_DIR',
    'RBBRConfig',
    'ModelConfig',
    'AppConfig',
    'CUSTOM_CONFIG',
]
