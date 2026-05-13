"""
Pages module
Multi-page Streamlit application
"""

from . import dashboard
from . import detail_bank
from . import prediksi
from . import shap_analysis
from . import monitoring
from . import retraining

__all__ = [
    'dashboard',
    'detail_bank',
    'prediksi',
    'shap_analysis',
    'monitoring',
    'retraining',
]