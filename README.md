---
title: EWS Bank Persero
emoji: 🏦
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.34.0
app_file: app.py
pinned: false
license: mit
---

# 🏦 OJK RBBR Early Warning System

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/streamlit-1.34.0-red)
![License](https://img.shields.io/badge/license-Internal%20OJK-orange)

Sistem prediksi kondisi keuangan bank berbasis machine learning untuk deteksi dini risiko perbankan menggunakan metode **RBBR (Risk-Based Bank Rating)**.

---

## 📋 Features

- **Dashboard Interaktif** — Heatmap kondisi bank dengan color-coded Peringkat Komposit (PK 1–5)
- **Detail Bank Analysis** — Analisis mendalam per bank dengan 4 pilar RBBR
- **Prediksi 6 Bulan** — Forecasting ensemble: TFT + LightGBM + NeuralProphet + SARIMAX
- **SHAP Explainability** — Interpretasi model dengan SHAP waterfall & summary plots
- **Monitoring System** — Tracking tren makro & alert bank berisiko tinggi
- **Retraining UI** — Upload data baru → trigger retraining dengan progress tracking
- **Model Management** — Versioning model dengan timestamp, upload otomatis ke HF Hub
- **Persistent Storage** — Data & model tersimpan di Hugging Face repository

---

## 🚀 Quick Start (Local)

```bash
# 1. Clone / extract project
cd prediksi-bank-v4

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure secrets
# Edit .streamlit/secrets.toml:
# HF_TOKEN = "hf_xxxxxxxxxxxx"
# SPACE_ID  = "username/ews-bank-persero"

# 5. Run
streamlit run app.py
```

App tersedia di `http://localhost:8501`

---

## 🚢 Deploy ke Hugging Face Spaces

### 1. Buat Space baru

- Buka https://huggingface.co/new-space
- SDK: **Streamlit**, visibility: Public/Private
- Catat Space ID: `username/ews-bank-persero`

### 2. Generate HF Token

- Buka https://huggingface.co/settings/tokens
- Buat token dengan scope **write**

### 3. Set Repository Secrets

Di halaman Space → **Settings → Repository secrets**, tambahkan:

| Key | Value |
|-----|-------|
| `HF_TOKEN` | Token HF dengan write access |
| `SPACE_ID` | `username/ews-bank-persero` |

### 4. Push ke Space

```bash
# Clone Space repository
git clone https://huggingface.co/spaces/USERNAME/SPACE_NAME
cd SPACE_NAME

# Copy semua file project (kecuali venv, __pycache__, .coverage)
# Pastikan .gitignore sudah benar

git add .
git commit -m "Initial deployment"
git push
```

### 5. Upload Models & Data

Setelah Space running, gunakan halaman **Model Management** atau **Upload Data** di UI untuk mengunggah:

- `models/tft-retrained-*.ckpt`
- `models/lgbm_reg_models.pkl`
- `models/feat_cols.pkl`
- `models/np_bi_rate.pkl`, `np_inflasi.pkl`, `np_kurs_usd.pkl`
- `models/sarimax_fc.pkl`
- `data/processed/master_panel.parquet`
- `data/predictions/forecast_6m.parquet`

> **Catatan:** File `sarimax_fc.pkl` (~312 MB) dan `sarimax.pkl` (~26 MB) cukup besar. Pastikan Space tier memiliki storage yang cukup, atau gunakan HF Dataset repository terpisah untuk file besar.

---

## 📁 Project Structure

```
prediksi-bank-v4/
│
├── app.py                          # Main entry point
├── requirements.txt                # Python dependencies
├── packages.txt                    # System dependencies (libomp-dev, libgomp1)
├── README.md
├── .gitignore
│
├── .streamlit/
│   ├── config.toml                # Theme & server config
│   └── secrets.toml               # HF_TOKEN, SPACE_ID (jangan di-commit!)
│
├── src/                            # Backend logic
│   ├── config.py                  # App configuration & constants
│   ├── data_loader.py             # Data loading dengan caching
│   ├── data_service.py            # Data service layer
│   ├── data_validator.py          # Validasi format data upload
│   ├── data_merge.py              # Merge data baru ke master panel
│   ├── data_utils.py              # Helper data utilities
│   ├── model_inference.py         # Inference pipeline ensemble
│   ├── rbbr_scoring.py            # RBBR scoring logic (PK 1–5)
│   ├── visualization.py           # Plotly chart builders
│   ├── utils.py                   # General helpers
│   ├── cache_manager.py           # Cache management
│   └── models/
│       ├── feature_engineering.py # Feature generation (lag, rolling, RBBR pillars)
│       └── data_augmentation.py   # Data augmentation utilities
│
├── pages/                          # Streamlit multi-page app
│   ├── dashboard.py               # Heatmap & distribusi PK
│   ├── detail_bank.py             # Analisis per bank
│   ├── prediksi.py                # Interface prediksi 6 bulan
│   ├── shap_analysis.py           # SHAP explainability
│   ├── monitoring.py              # Tren makro & alert
│   ├── retraining.py              # Retraining UI dengan progress bar
│   ├── model_management.py        # Model versioning & management
│   └── cache_listener.py          # Cache invalidation listener
│
├── training/                       # Ensemble training system
│   ├── ensemble_trainer.py        # TFT + LightGBM + NeuralProphet + SARIMAX trainer
│   └── ensemble_predictor.py      # Ensemble prediction & uncertainty
│
├── utils/
│   └── hf_storage.py              # Hugging Face Hub storage manager
│
├── scripts/
│   ├── auto_retraining.py         # Automated retraining pipeline
│   ├── auto_retraining_new.py     # Updated retraining pipeline
│   └── retrain_model.py           # Legacy retraining script
│
├── models/                         # Trained models (di-sync ke HF Hub)
│   ├── tft-retrained-*.ckpt       # TFT checkpoint (PyTorch Lightning)
│   ├── lgbm_reg_models.pkl        # LightGBM regressor (per-horizon)
│   ├── lgbm_clf.pkl               # LightGBM classifier (PK 1–5)
│   ├── feat_cols.pkl              # Feature columns untuk LightGBM
│   ├── np_bi_rate.pkl             # NeuralProphet — BI Rate forecast
│   ├── np_inflasi.pkl             # NeuralProphet — Inflasi forecast
│   ├── np_kurs_usd.pkl            # NeuralProphet — Kurs USD forecast
│   ├── sarimax_fc.pkl             # SARIMAX per-bank per-rasio (~312 MB)
│   ├── sarimax.pkl                # SARIMAX aggregated (~26 MB)
│   ├── model_metadata.json        # Versioning & performance info
│   └── versions/                  # Versioned model snapshots
│
├── data/
│   ├── processed/
│   │   └── master_panel.parquet   # Panel data utama (bank × bulan)
│   ├── predictions/
│   │   └── forecast_6m.parquet    # Hasil prediksi 6 bulan ke depan
│   ├── macro_forecast/
│   │   └── macro_fc.parquet       # Proyeksi makro (BI rate, inflasi, kurs)
│   ├── static/
│   │   ├── bank_metadata.csv      # Metadata bank (kode, nama, kategori)
│   │   └── thresholds_rbbr.json   # Threshold RBBR per rasio
│   ├── uploads/                   # File yang diunggah user
│   └── template_ews.csv           # Template format data upload
│
├── assets/
│   ├── logo_ojk.png
│   ├── favicon.ico
│   └── custom.css
│
├── tests/                          # Unit & integration tests
└── docs/
```

---

## 🤖 Model Architecture

**4-Model Ensemble:**

| Model | Peran | File |
|-------|-------|------|
| **TFT** (Temporal Fusion Transformer) | Multi-horizon forecasting utama | `tft-retrained-*.ckpt` |
| **LightGBM** | Regresi skor RBBR & klasifikasi PK | `lgbm_reg_models.pkl` |
| **NeuralProphet** | Forecast variabel makro (BI rate, inflasi, kurs) | `np_*.pkl` |
| **SARIMAX** | Per-bank per-rasio statistical forecasting | `sarimax_fc.pkl` |

**Current model version:** `1.1.0` (trained 2026-05-12)  
**Forecast horizon:** 6 bulan ke depan (Sep 2026 – Feb 2027)

---

## 📊 Data Requirements

### Format Upload Data

File CSV/Excel dengan kolom minimal:

```
kode_bank, nama_bank, periode (YYYY-MM), kategori_bank,
CAR, NPL_gross, NPL_net, NIM, ROA, ROE, BOPO, LDR, LCR,
total_aset, kredit_yang_diberikan,
suku_bunga_bi_rate, inflasi_yoy, kurs_usd_idr
```

Lihat `data/template_ews.csv` untuk format lengkap.

### Data yang Dibutuhkan untuk Inference

- `data/processed/master_panel.parquet` — panel data historis
- `data/predictions/forecast_6m.parquet` — hasil prediksi terkini
- `data/macro_forecast/macro_fc.parquet` — proyeksi makro

---

## 🔧 Configuration

### `.streamlit/config.toml`

```toml
[theme]
primaryColor = "#003d82"
backgroundColor = "#f8f9fa"
textColor = "#212529"

[server]
maxUploadSize = 200
port = 8501
```

### `.streamlit/secrets.toml` (lokal saja, jangan di-commit)

```toml
HF_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxx"
SPACE_ID  = "username/ews-bank-persero"
```

### `src/config.py`

Konfigurasi threshold alert, warna PK, dan parameter model dapat diubah di sini.

---

## 🐛 Troubleshooting

**Models tidak ter-load**  
→ Pastikan semua file `.pkl` dan `.ckpt` ada di `models/`. Cek `model_metadata.json` untuk nama file yang diharapkan.

**`sarimax_fc.pkl` terlalu besar untuk di-push ke git**  
→ Upload manual via HF Hub UI atau gunakan `huggingface-cli upload`. File ini di-exclude dari git via `.gitignore`.

**HF authentication error**  
→ Pastikan `HF_TOKEN` sudah di-set di Space secrets (bukan di `.streamlit/secrets.toml` yang tidak di-commit).

**OOM saat retraining**  
→ Kurangi batch size di halaman Retraining UI, atau upgrade Space hardware ke CPU Upgrade / GPU.

**SHAP plots tidak muncul**  
→ `shap==0.44.1` sudah ada di `requirements.txt`. Pastikan `lgbm_reg_models.pkl` ter-load dengan benar.

**Data upload gagal**  
→ Pastikan format kolom sesuai template. Gunakan `data/template_ews.csv` sebagai referensi.

---

## 📈 Performance (Model v1.1.0)

| Model | Metrik | Nilai |
|-------|--------|-------|
| TFT | Val Loss | 0.0833 |
| NeuralProphet BI Rate | MAE | 0.121 |
| NeuralProphet Inflasi | MAE | 0.420 |

---

## 🚢 Docker (Opsional)

```bash
docker build -t ojk-rbbr-ews .
docker run -p 8501:8501 \
  -e HF_TOKEN=hf_xxx \
  -e SPACE_ID=username/ews-bank-persero \
  ojk-rbbr-ews
```

---

## ✅ Changelog

### v1.1.0 (2026-05-12)
- 🔄 Retraining dengan data terbaru (Feb–Apr 2026)
- 🤖 Fine-tuning TFT, LightGBM, NeuralProphet, SARIMAX
- 📊 Forecast horizon: Sep 2026 – Feb 2027

### v1.0.0 (2026-04-21)
- ✨ Initial release
- 🎨 Multi-page dashboard
- 📈 SHAP explainability
- 🔄 Retraining UI dengan HF Hub integration

---

## 📝 License

**Internal Use Only — OJK DPKP**  
Dilarang disebarluaskan tanpa izin.

---

**Built with ❤️ for OJK DPKP** — _Protecting Indonesia's Banking System_
