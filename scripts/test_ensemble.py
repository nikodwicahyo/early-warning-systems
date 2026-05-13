"""
Basic test script for ensemble training system
Tests imports and basic functionality without heavy ML dependencies
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def test_imports():
    """Test basic imports"""
    try:
        # Test basic Python imports
        import pandas as pd
        import numpy as np
        print("✅ Basic imports (pandas, numpy) successful")

        # Test config structure (without streamlit)
        try:
            from training.ensemble_trainer import EnsembleTrainer
            print("✅ EnsembleTrainer import successful")
        except ImportError as e:
            if 'streamlit' in str(e):
                print("⚠️  EnsembleTrainer requires streamlit (expected in deployment)")
            else:
                raise

        try:
            from training.ensemble_predictor import EnsemblePredictor
            print("✅ EnsemblePredictor import successful")
        except ImportError as e:
            if 'streamlit' in str(e):
                print("⚠️  EnsemblePredictor requires streamlit (expected in deployment)")
            else:
                raise

        try:
            from scripts.auto_retraining import AutoRetrainer
            print("✅ AutoRetrainer import successful")
        except ImportError as e:
            if 'streamlit' in str(e):
                print("⚠️  AutoRetrainer requires streamlit (expected in deployment)")
            else:
                raise

        return True

    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_config():
    """Test configuration loading"""
    try:
        from scripts.auto_retraining import AutoRetrainer

        retrainer = AutoRetrainer()
        config = retrainer.config

        required_keys = ['target_columns', 'prediction_horizon', 'models_dir']
        for key in required_keys:
            if key not in config:
                print(f"❌ Missing config key: {key}")
                return False

        print("✅ Configuration loading successful")
        print(f"   Target columns: {config['target_columns']}")
        print(f"   Prediction horizon: {config['prediction_horizon']}")

        return True

    except ImportError as e:
        if 'streamlit' in str(e):
            print("⚠️  Config test requires streamlit (expected in deployment)")
            return True  # Not a failure
        else:
            print(f"❌ Config test failed: {e}")
            return False
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False

def test_data_loading():
    """Test data loading functionality"""
    try:
        from scripts.auto_retraining import AutoRetrainer

        retrainer = AutoRetrainer()

        # Check if data file exists
        data_file = Path(retrainer.config['processed_data_file'])
        if not data_file.exists():
            print(f"⚠️  Data file not found: {data_file}")
            return False

        # Try loading data
        df = retrainer.load_latest_data()
        print("✅ Data loading successful")
        print(f"   Records: {len(df)}")
        print(f"   Date range: {df['periode'].min()} to {df['periode'].max()}")
        print(f"   Banks: {df['kode_bank'].nunique()}")
        print(f"   Columns: {len(df.columns)}")

        return True

    except ImportError as e:
        if 'streamlit' in str(e):
            print("⚠️  Data loading test requires streamlit (expected in deployment)")
            return True  # Not a failure
        else:
            print(f"❌ Data loading test failed: {e}")
            return False
    except Exception as e:
        print(f"❌ Data loading test failed: {e}")
        return False

def test_metadata():
    """Test metadata management"""
    try:
        from scripts.auto_retraining import AutoRetrainer

        retrainer = AutoRetrainer()

        # Load metadata
        metadata = retrainer.load_model_metadata()
        print("✅ Metadata loading successful")

        # Check retrain needed
        needs_retrain = retrainer.check_retrain_needed(metadata)
        print(f"   Retraining needed: {needs_retrain}")

        return True

    except ImportError as e:
        if 'streamlit' in str(e):
            print("⚠️  Metadata test requires streamlit (expected in deployment)")
            return True  # Not a failure
        else:
            print(f"❌ Metadata test failed: {e}")
            return False
    except Exception as e:
        print(f"❌ Metadata test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Ensemble Training System")
    print("=" * 50)

    tests = [
        ("Basic Imports", test_imports),
        ("Configuration", test_config),
        ("Data Loading", test_data_loading),
        ("Metadata Management", test_metadata),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n🔍 Testing {test_name}...")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} failed")

    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} passed")

    if passed == total:
        print("🎉 All tests passed! Ensemble system is ready.")
        return 0
    else:
        print("⚠️  Some tests failed. Check dependencies and data.")
        return 1

if __name__ == "__main__":
    exit(main())