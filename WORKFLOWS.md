# WORKFLOWS.md

# EWS Bank Persero (RBBR Method) - Migration & Retraining Guide

# End-to-End Implementation on Hugging Face Spaces

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture Design](#architecture-design)
3. [Pre-Migration Checklist](#pre-migration-checklist)
4. [Step 1: Refactor Storage Logic](#step-1-refactor-storage-logic)
5. [Step 2: Implement Retraining UI](#step-2-implement-retraining-ui)
6. [Step 3: Model Management](#step-3-model-management)
7. [Step 4: Deployment Configuration](#step-4-deployment-configuration)
8. [Step 5: Testing & Validation](#step-5-testing-validation)
9. [Troubleshooting Guide](#troubleshooting-guide)
10. [Performance Optimization](#performance-optimization)

---

## 🎯 Project Overview

**Objective**: Migrate EWS (Early Warning System) for Bank Persero financial health monitoring from Streamlit Cloud to Hugging Face Spaces with enhanced retraining capabilities.

**Key Features**:

- RBBR (Risk-Based Bank Rating) Method Implementation
- Ensemble Forecasting: TFT (Temporal Fusion Transformer) + NeuralProphet + LightGBM + SARIMAX
- Manual Trigger Retraining with Progress Tracking
- Persistent Storage on Hugging Face Infrastructure
- Incremental Learning for faster model updates

**Technology Stack**:

- Frontend: Streamlit
- ML Framework: PyTorch Forecasting, NeuralProphet, LightGBM
- Storage: Hugging Face Hub API
- Hosting: Hugging Face Spaces (SDK: Streamlit)

---

## 🏗️ Architecture Design

````
┌─────────────────────────────────────────────────────────────┐
│                    Hugging Face Space                       │
│  ┌───────────────────────────────────────────────────────┐  │
│  │              Streamlit Application                    │  │
│  │  ┌─────────────┐  ┌──────────────┐  ┌─────────────┐   │  │
│  │  │ Data Upload │→ │ Data Storage │→ │ HF Dataset  │   │  │
│  │  │   (UI)      │  │   Handler    │  │  /Repo      │   │  │
│  │  └─────────────┘  └──────────────┘  └─────────────┘   │  │
│  │         ↓                                             │  │
│  │  ┌─────────────────────────────────────────────────┐  │  │
│  │  │       Retraining Trigger Button                 │  │  │
│  │  │  • Progress Bar (st.progress)                   │  │  │
│  │  │  • ETA Display                                  │  │  │
│  │  │  • Status Messages                              │  │  │
│  │  └─────────────────────────────────────────────────┘  │  │
│  │         ↓                                             │  │
│  │  ┌─────────────────────────────────────────────────┐  │  │
│  │  │     Ensemble Training Pipeline                  │  │  │
│  │  │  ┌────────────----------------──────────────┐   |  |  |
│  │  │  │ TFT + NeuralProphet + LightGBM + SARIMAX │   │  │  |
│  │  │  └────────────----------------──────────────┘   │  │  │
│  │  │         ↓                      ↓                │  │  │
│  │  │  ┌──────────────────────────────────┐           │  │  │
│  │  │  │    Incremental Learning          │           │  │  │
│  │  │  │    (Fine-tuning Strategy)        │           │  │  │
│  │  │  └──────────────────────────────────┘           │  │  │
│  │  └─────────────────────────────────────────────────┘  │  │
│  │         ↓                                             │  │
│  │  ┌─────────────────────────────────────────────────┐  │  │
│  │  │       Model Upload to HF Repository             │  │  │
│  │  │  • model_tft.pth                                │  │  │
│  │  │  • model_neuralprophet.pkl
|  |  |  • model_lgbm_clf.pkl                           │  │  │
|  |  |  • model_lgbm_reg.pkl                           │  │  │
|  |  |  • model_sarimax.pkl                            │  │  │
│  │  │  • metadata.json (versioning)                   │  │  │
│  │  └─────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────┘  │
---

## 🤖 Ensemble Model Implementation

### Architecture Overview

The system implements a **4-model ensemble** for robust financial forecasting:

1. **Temporal Fusion Transformer (TFT)** - Deep learning for complex temporal patterns
2. **NeuralProphet** - Facebook's forecasting model with trend decomposition
3. **LightGBM** - Gradient boosting for tabular financial data
4. **SARIMAX** - Statistical model for seasonal patterns

### Key Components

#### `training/ensemble_trainer.py`
```python
class EnsembleTrainer:
    def train_ensemble(df, target_col, existing_models=None):
        """Train all 4 models with incremental learning"""

    def prepare_data_tft(df, target_col):
        """Convert banking panel data to TFT format"""

    def prepare_data_neuralprophet(df, target_col):
        """Aggregate data for univariate forecasting"""
````

#### `training/ensemble_predictor.py`

```python
class EnsemblePredictor:
    def predict_ensemble(df, target_col, weights=None):
        """Generate weighted ensemble predictions"""

    def calculate_prediction_intervals(predictions):
        """Add uncertainty quantification"""
```

#### `scripts/auto_retraining.py`

```python
class AutoRetrainer:
    def run_retraining_pipeline(force_retrain=False):
        """End-to-end retraining with progress tracking"""
```

### Data Preparation

The system adapts real banking data structure:

```python
# Real data columns: periode, bank, kode_bank, ROA, CAR, NPL_net, etc.
# Converted to:
# TFT: time_idx, group, target, features...
# NeuralProphet: ds, y (aggregated)
# LightGBM: lag features + bank indicators
# SARIMAX: ds, y (aggregated)
```

### Incremental Learning Strategy

- **Fine-tuning**: Load existing models and continue training
- **Model versioning**: Timestamp-based model storage
- **Performance tracking**: Validation metrics and training time
- **Fallback handling**: Continue with available models if some fail

### Prediction Ensemble

```python
# Equal weights by default
weights = {'tft': 0.25, 'neuralprophet': 0.25,
          'lightgbm': 0.25, 'sarimax': 0.25}

# Weighted average predictions
ensemble_pred = sum(w * model_pred for w, model_pred in zip(weights, predictions))
```

---

## ✅ Pre-Migration Checklist

### 1. Hugging Face Setup

- [x] Create Hugging Face account (https://huggingface.co/)
- [x] Generate Access Token with **WRITE** permission
  - Go to: https://huggingface.co/settings/tokens
  - Create token with `write` scope
- [x] Create new Space
  - Navigate to: https://huggingface.co/new-space
  - Select SDK: **Streamlit**
  - Choose visibility: Public/Private
  - Note your Space name: `username/space-name`

### 2. Local Development Setup

```bash
# Clone your existing project
git clone <your-current-repo>
cd ews-bank-persero

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install streamlit pytorch-forecasting neuralprophet lightgbm huggingface_hub statsmodels
```

### 3. Backup Current State

```bash
# Backup existing models
mkdir -p backup/models
cp -r models/* backup/models/

# Backup data
mkdir -p backup/data
cp -r data/* backup/data/
```

---

## � Step 0: Implement Ensemble Training System

### 0.1 Create Training Module Structure

```bash
# Create training directory
mkdir -p training
touch training/__init__.py
```

### 0.2 Implement Ensemble Trainer (`training/ensemble_trainer.py`)

**Key Features:**

- TFT (Temporal Fusion Transformer) for deep temporal learning
- NeuralProphet for trend decomposition
- LightGBM for gradient boosting on financial features
- SARIMAX for statistical seasonal modeling
- Incremental learning with fine-tuning
- Progress tracking for UI integration

**Data Adaptation:**

```python
# Convert banking panel data to model formats
def prepare_data_tft(df, target_col='ROA'):
    # periode -> time_idx, kode_bank -> group, ROA -> target
    df_tft = df.rename(columns={'kode_bank': 'group', target_col: 'target'})
    df_tft['time_idx'] = df_tft.groupby('group').cumcount()
    return df_tft
```

### 0.3 Implement Ensemble Predictor (`training/ensemble_predictor.py`)

**Features:**

- Multi-model prediction aggregation
- Uncertainty quantification
- Model fallback handling
- Real-time prediction intervals

### 0.4 Create Retraining Script (`scripts/auto_retraining.py`)

**Capabilities:**

- Automated model retraining pipeline
- Progress tracking with Streamlit UI
- Model versioning and metadata management
- Performance monitoring

### 0.5 Test Ensemble System

```bash
# Test basic imports
python -c "from training.ensemble_trainer import EnsembleTrainer; print('OK')"

# Run retraining (if dependencies installed)
python scripts/auto_retraining.py
```

### 0.6 Integration Points

**Streamlit UI Integration:**

```python
# In retraining.py page
if st.button("🔄 Retrain Models"):
    with st.spinner("Retraining ensemble models..."):
        progress_bar = st.progress(0)

        def update_progress(pct, msg):
            progress_bar.progress(pct)
            st.write(msg)

        trainer = EnsembleTrainer(config)
        results = trainer.train_ensemble(df, target_col='ROA',
                                       progress_callback=update_progress)
```

---

## �📦 Step 1: Refactor Storage Logic

### 1.1 Create Storage Module

Create file: `utils/hf_storage.py`

```python
"""
Hugging Face Storage Handler
Manages persistent storage for uploaded datasets and trained models
"""

import os
import gc
import pandas as pd
from typing import Optional, Dict, Any
from pathlib import Path
from huggingface_hub import HfApi, login, hf_hub_download
import streamlit as st

class HFStorageManager:
    """
    Manages data and model storage on Hugging Face Hub
    """

    def __init__(self, repo_id: str, token: Optional[str] = None):
        """
        Initialize HF Storage Manager

        Args:
            repo_id: HuggingFace repository ID (format: username/repo-name)
            token: HF token (if None, will use st.secrets or env variable)
        """
        self.repo_id = repo_id

        # Get token from multiple sources
        if token is None:
            token = self._get_token()

        self.token = token
        self.api = HfApi()

        # Login to HF
        try:
            login(token=self.token, add_to_git_credential=False)
            st.success("✅ Connected to Hugging Face Hub")
        except Exception as e:
            st.error(f"❌ HF Authentication failed: {str(e)}")
            raise

    def _get_token(self) -> str:
        """
        Retrieve HF token from secrets or environment

        Priority:
        1. Streamlit secrets
        2. Environment variable
        3. Raise error if not found
        """
        # Try Streamlit secrets first
        if hasattr(st, 'secrets') and 'HF_TOKEN' in st.secrets:
            return st.secrets['HF_TOKEN']

        # Try environment variable
        token = os.getenv('HF_TOKEN')
        if token:
            return token

        raise ValueError(
            "HF_TOKEN not found. Please set it in:\n"
            "1. Streamlit secrets (.streamlit/secrets.toml), or\n"
            "2. Environment variable (HF_TOKEN=...)"
        )

    def upload_dataset(
        self,
        file_path: str,
        destination_path: str,
        commit_message: Optional[str] = None
    ) -> str:
        """
        Upload dataset file to HF repository

        Args:
            file_path: Local file path
            destination_path: Path in HF repo (e.g., 'data/bank_data.csv')
            commit_message: Git commit message

        Returns:
            URL of uploaded file
        """
        try:
            if commit_message is None:
                commit_message = f"Upload dataset: {Path(file_path).name}"

            # Upload file
            url = self.api.upload_file(
                path_or_fileobj=file_path,
                path_in_repo=destination_path,
                repo_id=self.repo_id,
                token=self.token,
                commit_message=commit_message,
                repo_type="space"
            )

            st.success(f"✅ Dataset uploaded: {destination_path}")
            return url

        except Exception as e:
            st.error(f"❌ Upload failed: {str(e)}")
            raise

    def upload_model(
        self,
        model_path: str,
        model_name: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Upload trained model to HF repository

        Args:
            model_path: Local model file path
            model_name: Name for model in repo (e.g., 'model_tft_v1.pth')
            metadata: Model metadata (version, metrics, etc.)

        Returns:
            URL of uploaded model
        """
        try:
            destination = f"models/{model_name}"

            # Upload model file
            url = self.api.upload_file(
                path_or_fileobj=model_path,
                path_in_repo=destination,
                repo_id=self.repo_id,
                token=self.token,
                commit_message=f"Update model: {model_name}",
                repo_type="space"
            )

            # Upload metadata if provided
            if metadata:
                import json
                metadata_path = f"models/{Path(model_name).stem}_metadata.json"

                # Save metadata locally first
                temp_meta = f"/tmp/{Path(model_name).stem}_metadata.json"
                with open(temp_meta, 'w') as f:
                    json.dump(metadata, f, indent=2)

                # Upload metadata
                self.api.upload_file(
                    path_or_fileobj=temp_meta,
                    path_in_repo=metadata_path,
                    repo_id=self.repo_id,
                    token=self.token,
                    commit_message=f"Update metadata: {model_name}",
                    repo_type="space"
                )

                # Cleanup
                os.remove(temp_meta)

            st.success(f"✅ Model uploaded: {model_name}")
            return url

        except Exception as e:
            st.error(f"❌ Model upload failed: {str(e)}")
            raise

    def download_file(
        self,
        repo_file_path: str,
        local_path: Optional[str] = None
    ) -> str:
        """
        Download file from HF repository

        Args:
            repo_file_path: Path in HF repo
            local_path: Local destination path (if None, downloads to cache)

        Returns:
            Local file path
        """
        try:
            downloaded_path = hf_hub_download(
                repo_id=self.repo_id,
                filename=repo_file_path,
                token=self.token,
                repo_type="space",
                local_dir=local_path
            )

            return downloaded_path

        except Exception as e:
            st.warning(f"⚠️ Download failed: {str(e)}")
            return None

    def list_files(self, path_prefix: str = "") -> list:
        """
        List files in HF repository

        Args:
            path_prefix: Filter by path prefix (e.g., 'models/')

        Returns:
            List of file paths
        """
        try:
            files = self.api.list_repo_files(
                repo_id=self.repo_id,
                token=self.token,
                repo_type="space"
            )

            if path_prefix:
                files = [f for f in files if f.startswith(path_prefix)]

            return files

        except Exception as e:
            st.error(f"❌ Failed to list files: {str(e)}")
            return []

    def cleanup_memory(self):
        """
        Force garbage collection to free memory
        """
        gc.collect()
        if hasattr(gc, 'collect'):
            gc.collect(generation=2)
```

### 1.2 Update Data Upload Handler

Modify your existing data upload function in `app.py`:

```python
import streamlit as st
import pandas as pd
from utils.hf_storage import HFStorageManager
import tempfile
import os

# Initialize HF Storage (place at top of app.py)
@st.cache_resource
def get_storage_manager():
    """Initialize HF Storage Manager (cached)"""
    # Get Space name from environment or secrets
    repo_id = os.getenv('SPACE_ID') or st.secrets.get('SPACE_ID', 'your-username/ews-bank-persero')
    return HFStorageManager(repo_id=repo_id)

storage = get_storage_manager()

# Modified upload function
def handle_file_upload(uploaded_file):
    """
    Handle uploaded CSV/Excel file
    Saves to HF repository for persistence
    """
    try:
        # Read file
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("❌ Format tidak didukung. Gunakan CSV atau Excel.")
            return None

        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        # Upload to HF repository
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        dest_path = f"data/{timestamp}_{uploaded_file.name}"

        with st.spinner("📤 Mengunggah data ke Hugging Face..."):
            storage.upload_dataset(
                file_path=tmp_path,
                destination_path=dest_path,
                commit_message=f"User upload: {uploaded_file.name}"
            )

        # Cleanup
        os.remove(tmp_path)

        st.success(f"✅ Data berhasil diunggah: {uploaded_file.name}")
        st.info(f"📁 Disimpan di: {dest_path}")

        return df

    except Exception as e:
        st.error(f"❌ Error saat upload: {str(e)}")
        return None
```

### 1.3 Configure Secrets

Create `.streamlit/secrets.toml` (for local development):

```toml
# Hugging Face Configuration
HF_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxx"  # Your HF token
SPACE_ID = "username/ews-bank-persero"  # Your Space ID
```

**For Hugging Face Spaces Deployment:**

- Go to your Space Settings
- Add secrets in "Repository secrets" section:
  - `HF_TOKEN`: Your HF token
  - `SPACE_ID`: Your Space ID

---

## 🔄 Step 2: Implement Retraining UI

### 2.1 Create Training Module

Create file: `training/ensemble_trainer.py`

```python
"""
Ensemble Model Trainer
Implements incremental learning for TFT + NeuralProphet
"""

import time
import gc
import torch
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from pathlib import Path
import streamlit as st

# PyTorch Forecasting
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

# NeuralProphet
from neuralprophet import NeuralProphet

class EnsembleTrainer:
    """
    Ensemble trainer for TFT + NeuralProphet with incremental learning
    """

    def __init__(self, config: Dict):
        """
        Initialize trainer

        Args:
            config: Training configuration
        """
        self.config = config
        self.models = {
            'tft': None,
            'neuralprophet': None
        }
        self.metrics = {}

    def prepare_data_tft(self, df: pd.DataFrame) -> Tuple[TimeSeriesDataSet, TimeSeriesDataSet]:
        """
        Prepare data for TFT training

        Args:
            df: Input dataframe with time series data

        Returns:
            Training and validation datasets
        """
        # Ensure required columns
        required_cols = ['time_idx', 'group', 'target']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"DataFrame must contain: {required_cols}")

        # Split train/val
        max_prediction_length = self.config.get('max_prediction_length', 12)
        max_encoder_length = self.config.get('max_encoder_length', 24)

        training_cutoff = df['time_idx'].max() - max_prediction_length

        # Create dataset
        training = TimeSeriesDataSet(
            df[df['time_idx'] <= training_cutoff],
            time_idx='time_idx',
            target='target',
            group_ids=['group'],
            max_encoder_length=max_encoder_length,
            max_prediction_length=max_prediction_length,
            time_varying_known_reals=['time_idx'],
            time_varying_unknown_reals=['target'],
            target_normalizer=GroupNormalizer(
                groups=['group'],
                transformation='softplus'
            ),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )

        # Validation dataset
        validation = TimeSeriesDataSet.from_dataset(
            training,
            df,
            predict=True,
            stop_randomization=True
        )

        return training, validation

    def train_tft(
        self,
        df: pd.DataFrame,
        existing_model_path: Optional[str] = None,
        progress_callback=None
    ) -> Dict:
        """
        Train TFT model with optional fine-tuning

        Args:
            df: Training data
            existing_model_path: Path to existing model for fine-tuning
            progress_callback: Callback function for progress updates

        Returns:
            Training metrics
        """
        try:
            st.info("🔄 Preparing TFT data...")
            if progress_callback:
                progress_callback(0.1, "Preparing TFT dataset")

            training, validation = self.prepare_data_tft(df)

            # Create dataloaders
            batch_size = self.config.get('batch_size', 64)
            train_dataloader = training.to_dataloader(
                train=True,
                batch_size=batch_size,
                num_workers=0
            )
            val_dataloader = validation.to_dataloader(
                train=False,
                batch_size=batch_size * 10,
                num_workers=0
            )

            if progress_callback:
                progress_callback(0.2, "Initializing TFT model")

            # Load existing model for fine-tuning or create new
            if existing_model_path and Path(existing_model_path).exists():
                st.info(f"🔁 Loading existing model for fine-tuning: {existing_model_path}")
                tft = TemporalFusionTransformer.load_from_checkpoint(existing_model_path)

                # Adjust learning rate for fine-tuning
                fine_tune_lr = self.config.get('fine_tune_lr', 1e-4)
                for param_group in tft.configure_optimizers()['optimizer'].param_groups:
                    param_group['lr'] = fine_tune_lr
            else:
                st.info("🆕 Creating new TFT model")
                tft = TemporalFusionTransformer.from_dataset(
                    training,
                    learning_rate=self.config.get('learning_rate', 3e-4),
                    hidden_size=self.config.get('hidden_size', 32),
                    attention_head_size=self.config.get('attention_head_size', 4),
                    dropout=self.config.get('dropout', 0.1),
                    hidden_continuous_size=self.config.get('hidden_continuous_size', 16),
                    loss=QuantileLoss(),
                    reduce_on_plateau_patience=4,
                )

            if progress_callback:
                progress_callback(0.3, "Setting up trainer")

            # Setup trainer
            max_epochs = self.config.get('max_epochs', 50) if not existing_model_path else self.config.get('fine_tune_epochs', 20)

            early_stop = EarlyStopping(
                monitor='val_loss',
                patience=5,
                mode='min'
            )

            checkpoint = ModelCheckpoint(
                monitor='val_loss',
                mode='min',
                save_top_k=1
            )

            trainer = Trainer(
                max_epochs=max_epochs,
                accelerator='auto',
                enable_model_summary=True,
                gradient_clip_val=0.1,
                callbacks=[early_stop, checkpoint],
                enable_progress_bar=False,  # We'll use custom progress
                enable_checkpointing=True,
            )

            if progress_callback:
                progress_callback(0.4, f"Training TFT ({max_epochs} epochs)")

            # Train with progress tracking
            start_time = time.time()

            # Custom training loop for progress tracking
            for epoch in range(max_epochs):
                trainer.fit(
                    tft,
                    train_dataloaders=train_dataloader,
                    val_dataloaders=val_dataloader,
                )

                # Update progress
                if progress_callback:
                    progress = 0.4 + (0.3 * (epoch + 1) / max_epochs)
                    elapsed = time.time() - start_time
                    eta = (elapsed / (epoch + 1)) * (max_epochs - epoch - 1)
                    progress_callback(
                        progress,
                        f"TFT Epoch {epoch+1}/{max_epochs} (ETA: {eta:.0f}s)"
                    )

                # Early stopping check
                if trainer.should_stop:
                    break

            training_time = time.time() - start_time

            self.models['tft'] = tft

            # Get best model path
            best_model_path = checkpoint.best_model_path

            if progress_callback:
                progress_callback(0.7, "TFT training complete")

            # Cleanup
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            return {
                'model': tft,
                'best_model_path': best_model_path,
                'training_time': training_time,
                'val_loss': trainer.callback_metrics.get('val_loss', 0).item()
            }

        except Exception as e:
            st.error(f"❌ TFT training failed: {str(e)}")
            raise

    def train_neuralprophet(
        self,
        df: pd.DataFrame,
        existing_model_path: Optional[str] = None,
        progress_callback=None
    ) -> Dict:
        """
        Train NeuralProphet model with optional fine-tuning

        Args:
            df: Training data (must have 'ds' and 'y' columns)
            existing_model_path: Path to existing model for fine-tuning
            progress_callback: Callback function for progress updates

        Returns:
            Training metrics
        """
        try:
            st.info("🔄 Preparing NeuralProphet data...")
            if progress_callback:
                progress_callback(0.75, "Preparing NeuralProphet data")

            # Ensure correct columns
            if 'ds' not in df.columns or 'y' not in df.columns:
                raise ValueError("DataFrame must contain 'ds' (datetime) and 'y' (target) columns")

            # Initialize or load model
            if existing_model_path and Path(existing_model_path).exists():
                st.info(f"🔁 Loading existing NeuralProphet for fine-tuning")
                from neuralprophet import load
                m = load(existing_model_path)
            else:
                st.info("🆕 Creating new NeuralProphet model")
                m = NeuralProphet(
                    growth=self.config.get('growth', 'linear'),
                    n_changepoints=self.config.get('n_changepoints', 10),
                    changepoints_range=self.config.get('changepoints_range', 0.8),
                    trend_reg=self.config.get('trend_reg', 0),
                    yearly_seasonality=self.config.get('yearly_seasonality', 'auto'),
                    weekly_seasonality=self.config.get('weekly_seasonality', 'auto'),
                    daily_seasonality=self.config.get('daily_seasonality', 'auto'),
                    seasonality_mode=self.config.get('seasonality_mode', 'additive'),
                    epochs=self.config.get('epochs', 100),
                    learning_rate=self.config.get('learning_rate', None),
                    batch_size=self.config.get('batch_size', None),
                )

            if progress_callback:
                progress_callback(0.8, "Training NeuralProphet")

            # Train model
            start_time = time.time()

            metrics = m.fit(
                df,
                freq='D',
                progress=None  # Disable default progress bar
            )

            training_time = time.time() - start_time

            if progress_callback:
                progress_callback(0.95, "NeuralProphet training complete")

            self.models['neuralprophet'] = m

            # Cleanup
            gc.collect()

            return {
                'model': m,
                'training_time': training_time,
                'metrics': metrics
            }

        except Exception as e:
            st.error(f"❌ NeuralProphet training failed: {str(e)}")
            raise

    def save_models(self, output_dir: str) -> Dict[str, str]:
        """
        Save trained models

        Args:
            output_dir: Directory to save models

        Returns:
            Dictionary of model paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        model_paths = {}

        # Save TFT
        if self.models['tft']:
            tft_path = output_path / "model_tft.pth"
            torch.save(self.models['tft'].state_dict(), tft_path)
            model_paths['tft'] = str(tft_path)

        # Save NeuralProphet
        if self.models['neuralprophet']:
            np_path = output_path / "model_neuralprophet.pkl"
            self.models['neuralprophet'].save(str(np_path))
            model_paths['neuralprophet'] = str(np_path)

        return model_paths
```

### 2.2 Create Retraining UI Component

Add to `app.py`:

```python
import streamlit as st
import time
from datetime import datetime
from training.ensemble_trainer import EnsembleTrainer
from utils.hf_storage import HFStorageManager

def retraining_interface():
    """
    Retraining UI with progress tracking
    """
    st.header("🔄 Manual Model Retraining")

    st.markdown("""
    **Sistem Retraining Incremental:**
    - **TFT (Temporal Fusion Transformer)**: Deep learning forecasting
    - **NeuralProphet**: Prophet++ with neural network
    - **Mode**: Fine-tuning (lebih cepat dari training from scratch)
    """)

    # Configuration
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("⚙️ Training Configuration")

        max_epochs = st.number_input(
            "Max Epochs (TFT)",
            min_value=10,
            max_value=200,
            value=50,
            step=10
        )

        fine_tune_epochs = st.number_input(
            "Fine-tune Epochs (jika model exist)",
            min_value=5,
            max_value=50,
            value=20,
            step=5
        )

        batch_size = st.selectbox(
            "Batch Size",
            options=[32, 64, 128],
            index=1
        )

    with col2:
        st.subheader("📊 Dataset Info")

        # Check available datasets
        storage = get_storage_manager()
        datasets = storage.list_files("data/")

        if datasets:
            st.success(f"✅ {len(datasets)} dataset(s) tersedia")
            latest_dataset = datasets[-1]
            st.info(f"📁 Latest: {latest_dataset}")
        else:
            st.warning("⚠️ Belum ada dataset. Upload data terlebih dahulu.")
            return

    st.divider()

    # Retraining button
    if st.button("🚀 Jalankan Retraining", type="primary", use_container_width=True):

        # Initialize progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        eta_text = st.empty()

        def update_progress(progress: float, message: str):
            """Update progress UI"""
            progress_bar.progress(min(progress, 1.0))
            status_text.text(f"📍 {message}")

        try:
            start_time = time.time()

            # Step 1: Load data
            update_progress(0.05, "Loading dataset...")

            # Download latest dataset
            local_data_path = storage.download_file(latest_dataset)

            if local_data_path.endswith('.csv'):
                df = pd.read_csv(local_data_path)
            else:
                df = pd.read_excel(local_data_path)

            st.success(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

            # Step 2: Initialize trainer
            update_progress(0.1, "Initializing trainer...")

            config = {
                'max_epochs': max_epochs,
                'fine_tune_epochs': fine_tune_epochs,
                'batch_size': batch_size,
                'learning_rate': 3e-4,
                'fine_tune_lr': 1e-4,
                'max_prediction_length': 12,
                'max_encoder_length': 24,
                'hidden_size': 32,
                'attention_head_size': 4,
                'dropout': 0.1,
                'hidden_continuous_size': 16,
            }

            trainer = EnsembleTrainer(config)

            # Step 3: Check for existing models
            update_progress(0.15, "Checking for existing models...")

            existing_models = storage.list_files("models/")
            tft_model_path = None
            np_model_path = None

            for model in existing_models:
                if 'tft' in model.lower() and model.endswith('.pth'):
                    tft_model_path = storage.download_file(model)
                elif 'neuralprophet' in model.lower() and model.endswith('.pkl'):
                    np_model_path = storage.download_file(model)

            if tft_model_path or np_model_path:
                st.info("🔁 Existing models found. Will perform fine-tuning.")
            else:
                st.info("🆕 No existing models. Training from scratch.")

            # Step 4: Train TFT
            update_progress(0.2, "Training TFT model...")

            tft_results = trainer.train_tft(
                df,
                existing_model_path=tft_model_path,
                progress_callback=update_progress
            )

            elapsed = time.time() - start_time
            eta_text.text(f"⏱️ Elapsed: {elapsed:.0f}s | ETA: ~{elapsed:.0f}s remaining")

            # Step 5: Train NeuralProphet
            update_progress(0.7, "Training NeuralProphet...")

            # Prepare data for NeuralProphet (convert to ds/y format)
            df_np = df.rename(columns={'date': 'ds', 'target': 'y'})

            np_results = trainer.train_neuralprophet(
                df_np,
                existing_model_path=np_model_path,
                progress_callback=update_progress
            )

            # Step 6: Save models
            update_progress(0.95, "Saving models...")

            model_paths = trainer.save_models("/tmp/trained_models")

            # Step 7: Upload to HF
            update_progress(0.97, "Uploading models to Hugging Face...")

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Upload TFT
            if 'tft' in model_paths:
                tft_metadata = {
                    'timestamp': timestamp,
                    'model_type': 'TFT',
                    'val_loss': tft_results.get('val_loss', 0),
                    'training_time': tft_results.get('training_time', 0),
                    'config': config
                }

                storage.upload_model(
                    model_path=model_paths['tft'],
                    model_name=f"model_tft_{timestamp}.pth",
                    metadata=tft_metadata
                )

            # Upload NeuralProphet
            if 'neuralprophet' in model_paths:
                np_metadata = {
                    'timestamp': timestamp,
                    'model_type': 'NeuralProphet',
                    'training_time': np_results.get('training_time', 0),
                    'config': config
                }

                storage.upload_model(
                    model_path=model_paths['neuralprophet'],
                    model_name=f"model_neuralprophet_{timestamp}.pkl",
                    metadata=np_metadata
                )

            # Complete
            update_progress(1.0, "Training complete! ✅")

            total_time = time.time() - start_time

            st.success(f"""
            ### ✅ Retraining Berhasil!

            **Training Summary:**
            - **Total Time**: {total_time:.2f} seconds
            - **TFT Val Loss**: {tft_results.get('val_loss', 'N/A')}
            - **Models Uploaded**: {len(model_paths)}
            - **Timestamp**: {timestamp}

            Model siap digunakan untuk prediksi!
            """)

            # Cleanup
            storage.cleanup_memory()

        except Exception as e:
            st.error(f"""
            ### ❌ Training Failed

            **Error**: {str(e)}

            Silakan cek:
            1. Format dataset sudah benar
            2. Memory cukup (gunakan batch size lebih kecil jika perlu)
            3. Token HF masih valid
            """)

            # Cleanup on error
            storage.cleanup_memory()

# Add to main app
def main():
    st.set_page_config(
        page_title="EWS Bank Persero",
        page_icon="🏦",
        layout="wide"
    )

    st.title("🏦 Early Warning System - Bank Persero")
    st.caption("RBBR Method | Powered by HuggingFace Spaces")

    # Sidebar
    page = st.sidebar.selectbox(
        "📍 Navigation",
        ["Dashboard", "Upload Data", "Retraining", "Predictions", "About"]
    )

    if page == "Retraining":
        retraining_interface()
    # ... other pages
```

---

## 💾 Step 3: Model Management

### 3.1 Model Versioning Strategy

Create file: `utils/model_manager.py`

```python
"""
Model Version Manager
Handles model versioning, loading, and selection
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import streamlit as st

class ModelVersionManager:
    """
    Manages model versions and metadata
    """

    def __init__(self, storage_manager):
        """
        Initialize model manager

        Args:
            storage_manager: HFStorageManager instance
        """
        self.storage = storage_manager

    def list_model_versions(self, model_type: str = None) -> List[Dict]:
        """
        List all model versions

        Args:
            model_type: Filter by model type ('tft' or 'neuralprophet')

        Returns:
            List of model metadata dictionaries
        """
        # Get all model files
        model_files = self.storage.list_files("models/")

        # Filter metadata files
        metadata_files = [f for f in model_files if f.endswith('_metadata.json')]

        versions = []

        for meta_file in metadata_files:
            try:
                # Download metadata
                local_path = self.storage.download_file(meta_file)

                with open(local_path, 'r') as f:
                    metadata = json.load(f)

                # Add file path
                metadata['metadata_path'] = meta_file
                metadata['model_path'] = meta_file.replace('_metadata.json', '.pth' if 'tft' in meta_file else '.pkl')

                # Filter by type if specified
                if model_type is None or metadata.get('model_type', '').lower() == model_type.lower():
                    versions.append(metadata)

            except Exception as e:
                st.warning(f"⚠️ Could not load metadata: {meta_file} ({str(e)})")

        # Sort by timestamp (newest first)
        versions.sort(key=lambda x: x.get('timestamp', ''), reverse=True)

        return versions

    def get_latest_model(self, model_type: str) -> Optional[Dict]:
        """
        Get latest model of specific type

        Args:
            model_type: 'tft' or 'neuralprophet'

        Returns:
            Latest model metadata
        """
        versions = self.list_model_versions(model_type)

        if versions:
            return versions[0]
        else:
            return None

    def load_model(self, model_metadata: Dict):
        """
        Load model from HF repository

        Args:
            model_metadata: Model metadata dictionary

        Returns:
            Loaded model
        """
        model_path = model_metadata.get('model_path')
        model_type = model_metadata.get('model_type', '').lower()

        if not model_path:
            raise ValueError("Model path not found in metadata")

        # Download model
        local_path = self.storage.download_file(model_path)

        # Load based on type
        if model_type == 'tft':
            import torch
            from pytorch_forecasting import TemporalFusionTransformer

            model = TemporalFusionTransformer.load_from_checkpoint(local_path)
            return model

        elif model_type == 'neuralprophet':
            from neuralprophet import load

            model = load(local_path)
            return model

        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def delete_old_versions(self, keep_n: int = 5):
        """
        Delete old model versions, keeping only latest N

        Args:
            keep_n: Number of versions to keep
        """
        for model_type in ['tft', 'neuralprophet']:
            versions = self.list_model_versions(model_type)

            if len(versions) > keep_n:
                # Delete old versions
                to_delete = versions[keep_n:]

                for version in to_delete:
                    try:
                        # This would require HF API delete functionality
                        # Currently not implemented in basic version
                        st.info(f"Would delete: {version.get('model_path')}")
                    except Exception as e:
                        st.warning(f"⚠️ Could not delete: {str(e)}")
```

### 3.2 Memory Management

Add to `utils/hf_storage.py`:

```python
def check_memory_usage():
    """
    Check current memory usage and warn if high
    """
    import psutil

    memory = psutil.virtual_memory()

    usage_percent = memory.percent
    available_gb = memory.available / (1024**3)

    if usage_percent > 80:
        st.warning(f"""
        ⚠️ **High Memory Usage**: {usage_percent:.1f}%

        Available: {available_gb:.2f} GB

        **Recommendations**:
        - Reduce batch size
        - Close other applications
        - Consider upgrading Space to higher tier
        """)

        # Force garbage collection
        gc.collect()

    return {
        'percent': usage_percent,
        'available_gb': available_gb
    }
```

---

## 🚀 Step 4: Deployment Configuration

### 4.1 Create requirements.txt

```txt
# Core Framework
streamlit>=1.31.0

# Data Processing
pandas>=2.0.0
numpy>=1.24.0
openpyxl>=3.1.0  # Excel support

# Machine Learning - PyTorch Ecosystem
torch>=2.0.0
pytorch-lightning>=2.0.0
pytorch-forecasting>=1.0.0

# Machine Learning - Prophet
neuralprophet>=0.7.0

# Machine Learning - Additional
lightgbm>=4.0.0
statsmodels>=0.14.0
scikit-learn>=1.3.0

# Hugging Face Integration
huggingface_hub>=0.20.0

# Visualization
plotly>=5.18.0
matplotlib>=3.7.0
seaborn>=0.13.0

# Utilities
python-dateutil>=2.8.0
pytz>=2023.3
psutil>=5.9.0  # Memory monitoring

# Optional: Acceleration
# accelerate>=0.25.0  # Uncomment if using GPU
```

### 4.2 Create packages.txt (System Dependencies)

Create file: `packages.txt`

```txt
# Required for LightGBM and some numerical libraries
libomp-dev
libgomp1

# Optional: For better performance
build-essential
```

### 4.3 Create README.md for HF Space

````markdown
---
title: EWS Bank Persero
emoji: 🏦
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.31.0
app_file: app.py
pinned: false
license: mit
---

# 🏦 Early Warning System - Bank Persero

## Overview

Sistem Early Warning untuk monitoring kesehatan keuangan Bank Persero menggunakan metode RBBR (Risk-Based Bank Rating).

## Features

- 📊 **Real-time Monitoring**: Dashboard kesehatan finansial
- 🤖 **AI Forecasting**: Ensemble model (TFT + NeuralProphet)
- 🔄 **Manual Retraining**: Trigger retraining dengan progress tracking
- 💾 **Persistent Storage**: Data dan model tersimpan di HF Repository
- 📈 **Advanced Analytics**: Analisis risiko komprehensif

## Technology Stack

- **Frontend**: Streamlit
- **ML Models**:
  - Temporal Fusion Transformer (PyTorch)
  - NeuralProphet
  - LightGBM
  - SARIMAX
- **Storage**: Hugging Face Hub
- **Deployment**: Hugging Face Spaces

## Setup

1. **Clone Space**:
   ```bash
   git clone https://huggingface.co/spaces/YOUR_USERNAME/ews-bank-persero
   cd ews-bank-persero
   ```
````

2. **Set Secrets**:
   - `HF_TOKEN`: Your HuggingFace token (with write access)
   - `SPACE_ID`: Your Space ID (format: username/space-name)

3. **Upload Initial Data**:
   - Navigate to "Upload Data" tab
   - Upload CSV/Excel with required columns

4. **Train Models**:
   - Go to "Retraining" tab
   - Configure parameters
   - Click "Jalankan Retraining"

## Usage

### Data Format

Required columns:

- `date`: Timestamp
- `target`: Financial metric to forecast
- `group`: Bank identifier
- Additional features as needed

### Retraining

Models support incremental learning:

- **From Scratch**: First training (~5-10 minutes)
- **Fine-tuning**: Update with new data (~2-5 minutes)

## License

MIT License

## Contact

For questions: [Your Contact]

````

### 4.4 Create .gitignore

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
build/
dist/
*.egg-info/

# Streamlit
.streamlit/secrets.toml

# Models (stored in HF repo, not git)
models/*.pth
models/*.pkl
models/*.pt

# Data (stored in HF repo, not git)
data/*.csv
data/*.xlsx

# Temporary files
/tmp/
*.tmp
*.log

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Jupyter
.ipynb_checkpoints/
*.ipynb
````

### 4.5 Create app.py (Main Application)

```python
"""
EWS Bank Persero - Main Application
Early Warning System using RBBR Method
"""

import streamlit as st
import pandas as pd
import os
from pathlib import Path

# Import custom modules
from utils.hf_storage import HFStorageManager
from utils.model_manager import ModelVersionManager
from training.ensemble_trainer import EnsembleTrainer

# Page Configuration
st.set_page_config(
    page_title="EWS Bank Persero",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Storage Manager
@st.cache_resource
def get_storage_manager():
    """Initialize HF Storage Manager (cached)"""
    repo_id = os.getenv('SPACE_ID') or st.secrets.get('SPACE_ID', 'your-username/ews-bank-persero')
    return HFStorageManager(repo_id=repo_id)

# Initialize Model Manager
@st.cache_resource
def get_model_manager():
    """Initialize Model Version Manager (cached)"""
    storage = get_storage_manager()
    return ModelVersionManager(storage)

# Main Application
def main():
    # Header
    st.title("🏦 Early Warning System - Bank Persero")
    st.caption("RBBR Method | Powered by Hugging Face Spaces")

    # Sidebar Navigation
    with st.sidebar:
        st.image("https://huggingface.co/front/assets/huggingface_logo.svg", width=100)
        st.divider()

        page = st.radio(
            "📍 Navigation",
            [
                "🏠 Dashboard",
                "📤 Upload Data",
                "🔄 Retraining",
                "📊 Predictions",
                "🔧 Model Management",
                "ℹ️ About"
            ]
        )

        st.divider()

        # System Status
        st.subheader("System Status")

        try:
            storage = get_storage_manager()
            st.success("✅ HF Connected")

            # Show data count
            datasets = storage.list_files("data/")
            st.metric("Datasets", len(datasets))

            # Show model count
            models = storage.list_files("models/")
            model_count = len([m for m in models if m.endswith(('.pth', '.pkl'))])
            st.metric("Models", model_count)

        except Exception as e:
            st.error(f"❌ Connection Error")

    # Route to pages
    if "Dashboard" in page:
        show_dashboard()
    elif "Upload Data" in page:
        show_upload_page()
    elif "Retraining" in page:
        retraining_interface()
    elif "Predictions" in page:
        show_predictions()
    elif "Model Management" in page:
        show_model_management()
    elif "About" in page:
        show_about()

def show_dashboard():
    """Dashboard page"""
    st.header("📊 Dashboard")
    st.info("Dashboard implementation here...")

def show_upload_page():
    """Data upload page"""
    st.header("📤 Upload Data")

    uploaded_file = st.file_uploader(
        "Upload CSV atau Excel",
        type=['csv', 'xlsx', 'xls']
    )

    if uploaded_file:
        # Use the upload handler from Step 1.2
        df = handle_file_upload(uploaded_file)

        if df is not None:
            st.subheader("Preview Data")
            st.dataframe(df.head(10))

            st.subheader("Data Info")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", df.shape[0])
            with col2:
                st.metric("Columns", df.shape[1])
            with col3:
                st.metric("Memory", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

def show_predictions():
    """Predictions page"""
    st.header("📊 Predictions")
    st.info("Predictions implementation here...")

def show_model_management():
    """Model management page"""
    st.header("🔧 Model Management")

    model_manager = get_model_manager()

    # Show TFT versions
    st.subheader("TFT Models")
    tft_versions = model_manager.list_model_versions('tft')

    if tft_versions:
        for i, version in enumerate(tft_versions[:5]):  # Show latest 5
            with st.expander(f"Version {i+1}: {version.get('timestamp', 'Unknown')}"):
                st.json(version)
    else:
        st.info("No TFT models found")

    st.divider()

    # Show NeuralProphet versions
    st.subheader("NeuralProphet Models")
    np_versions = model_manager.list_model_versions('neuralprophet')

    if np_versions:
        for i, version in enumerate(np_versions[:5]):
            with st.expander(f"Version {i+1}: {version.get('timestamp', 'Unknown')}"):
                st.json(version)
    else:
        st.info("No NeuralProphet models found")

def show_about():
    """About page"""
    st.header("ℹ️ About")

    st.markdown("""
    ## Early Warning System - Bank Persero

    ### Overview
    Sistem monitoring dan prediksi kesehatan finansial Bank Persero menggunakan metode RBBR.

    ### Features
    - ✅ Real-time data monitoring
    - ✅ AI-powered forecasting (TFT + NeuralProphet)
    - ✅ Manual trigger retraining
    - ✅ Incremental learning support
    - ✅ Model versioning
    - ✅ Persistent storage on Hugging Face

    ### Technology Stack
    - **Frontend**: Streamlit
    - **ML Models**: PyTorch Forecasting (TFT), NeuralProphet, LightGBM
    - **Storage**: Hugging Face Hub
    - **Deployment**: Hugging Face Spaces

    ### Version
    - **Version**: 1.0.0
    - **Last Update**: 2024

    ### License
    MIT License
    """)

# Run the application
if __name__ == "__main__":
    main()
```

---

## ✅ Step 5: Testing & Validation

### 5.1 Local Testing Checklist

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set environment variables
export HF_TOKEN="hf_your_token_here"
export SPACE_ID="username/ews-bank-persero"

# 3. Run locally
streamlit run app.py

# 4. Test upload
# - Navigate to Upload Data
# - Upload sample CSV
# - Verify file appears in HF repo

# 5. Test retraining
# - Navigate to Retraining
# - Configure parameters
# - Run training
# - Monitor progress
# - Verify models uploaded

# 6. Check memory usage
# - Monitor during training
# - Ensure no OOM errors
```

### 5.2 Deployment Steps

```bash
# 1. Create new HF Space
# Go to: https://huggingface.co/new-space
# Select: SDK = Streamlit

# 2. Clone Space repository
git clone https://huggingface.co/spaces/USERNAME/SPACE_NAME
cd SPACE_NAME

# 3. Copy all files
cp -r /path/to/your/project/* .

# 4. Commit and push
git add .
git commit -m "Initial deployment"
git push

# 5. Set secrets in Space settings
# Go to Space Settings > Repository secrets
# Add:
# - HF_TOKEN
# - SPACE_ID

# 6. Monitor build
# Check build logs in Space page
# Wait for "Running" status
```

### 5.3 Post-Deployment Validation

**Test Checklist**:

- [ ] Space loads successfully
- [ ] HF authentication works
- [ ] File upload persists to repo
- [ ] Retraining completes without errors
- [ ] Models upload successfully
- [ ] Progress bar updates correctly
- [ ] ETA calculation is accurate
- [ ] Memory usage is acceptable
- [ ] No timeout errors

---

## 🔧 Troubleshooting Guide

### Common Issues and Solutions

#### 1. **Authentication Error**

**Problem**: `HF_TOKEN not found`

**Solution**:

```python
# Check secrets configuration
# In Space Settings > Repository secrets:
# - HF_TOKEN = your_token_here
# - SPACE_ID = username/space-name

# Verify in code:
import os
print(os.getenv('HF_TOKEN'))  # Should not be None
```

#### 2. **Out of Memory (OOM)**

**Problem**: Training crashes with memory error

**Solutions**:

```python
# Option 1: Reduce batch size
config = {
    'batch_size': 32,  # Reduce from 64
}

# Option 2: Use gradient accumulation
config = {
    'accumulate_grad_batches': 2,
}

# Option 3: Clear cache regularly
import gc
import torch

gc.collect()
torch.cuda.empty_cache()
```

#### 3. **Slow Training**

**Problem**: Training takes too long

**Solutions**:

```python
# Option 1: Reduce epochs for fine-tuning
config = {
    'fine_tune_epochs': 10,  # Instead of 20
}

# Option 2: Use smaller model
config = {
    'hidden_size': 16,  # Instead of 32
    'attention_head_size': 2,  # Instead of 4
}

# Option 3: Upgrade Space tier
# Go to Space Settings > Hardware
# Select: CPU Upgrade or GPU
```

#### 4. **File Upload Fails**

**Problem**: Dataset upload returns error

**Solution**:

```python
# Check file size limits
# HF Spaces: 50GB total storage

# Verify file format
accepted_formats = ['.csv', '.xlsx', '.xls']

# Check permissions
# Token must have 'write' scope
```

#### 5. **Model Download Fails**

**Problem**: Cannot load existing models

**Solution**:

```python
# Check model exists
storage = get_storage_manager()
models = storage.list_files("models/")
print(models)

# Verify file path
model_path = "models/model_tft_20240101.pth"
local_path = storage.download_file(model_path)
print(f"Downloaded to: {local_path}")

# Check file integrity
import os
if os.path.exists(local_path):
    file_size = os.path.getsize(local_path)
    print(f"File size: {file_size} bytes")
```

---

## ⚡ Performance Optimization

### 1. Caching Strategy

```python
# Cache storage manager
@st.cache_resource
def get_storage_manager():
    return HFStorageManager(repo_id=SPACE_ID)

# Cache data loading
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_dataset(file_path):
    return pd.read_csv(file_path)

# Cache model predictions
@st.cache_data(ttl=1800)  # Cache for 30 minutes
def get_predictions(_model, data):
    return _model.predict(data)
```

### 2. Batch Processing

```python
# Process data in batches
def batch_process(data, batch_size=1000):
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        yield process_batch(batch)
```

### 3. Memory Optimization

```python
# Use efficient data types
df['category'] = df['category'].astype('category')
df['value'] = pd.to_numeric(df['value'], downcast='float')

# Delete unused variables
del large_dataframe
gc.collect()

# Use generators instead of lists
def data_generator():
    for chunk in pd.read_csv('large_file.csv', chunksize=1000):
        yield chunk
```

### 4. Model Optimization

```python
# Use mixed precision training (if GPU available)
from pytorch_lightning import Trainer

trainer = Trainer(
    precision=16,  # Use FP16
    accelerator='gpu'
)

# Quantize models for inference
import torch.quantization

model_quantized = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8
)
```

---

## 📚 Additional Resources

### Documentation Links

- **Hugging Face Spaces**: https://huggingface.co/docs/hub/spaces
- **Streamlit Docs**: https://docs.streamlit.io
- **PyTorch Forecasting**: https://pytorch-forecasting.readthedocs.io
- **NeuralProphet**: https://neuralprophet.com

### Support

For issues:

1. Check this WORKFLOWS.md
2. Review HF Spaces documentation
3. Check Space build logs
4. Create issue in repository

---

## 🎯 Next Steps

After successful deployment:

1. **Monitor Performance**
   - Check Space analytics
   - Monitor memory usage
   - Track training times

2. **Optimize Models**
   - Fine-tune hyperparameters
   - Experiment with architectures
   - Add more features

3. **Enhance UI**
   - Add visualizations
   - Improve UX
   - Add more analytics

4. **Scale**
   - Consider GPU upgrade for faster training
   - Implement automated retraining schedule
   - Add CI/CD pipeline

---

## ✅ Completion Checklist

- [ ] Storage logic refactored with HF Hub
- [ ] Retraining UI implemented with progress tracking
- [ ] Model management system working
- [ ] All dependencies in requirements.txt
- [ ] packages.txt created (if needed)
- [ ] Secrets configured in HF Space
- [ ] Local testing passed
- [ ] Deployed to HF Spaces
- [ ] Post-deployment validation complete
- [ ] Documentation updated

---

**End of WORKFLOWS.md**

_Last Updated: 2024_
_Version: 1.0.0_
