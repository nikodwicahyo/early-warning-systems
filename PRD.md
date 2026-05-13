# PRD — Sistem Prediksi Kondisi Keuangan Bank Berbasis RBBR
**Direktorat Pengawasan Kepatuhan Perbankan (DPKP) — OJK**
**Versi:** 1.0.0 | **Status:** Draft | **Tanggal:** April 2026

---

## 1. Latar Belakang & Konteks

### 1.1 Problem Statement
Proses analisis kondisi keuangan bank di DPKP saat ini dilakukan secara manual — analis membaca laporan keuangan bulanan per bank, menghitung rasio, lalu memetakannya ke Peringkat Komposit (PK) RBBR. Proses ini memiliki tiga kelemahan utama:

1. **Reaktif, bukan preventif** — analisis baru dilakukan setelah laporan diterima; tidak ada proyeksi ke depan.
2. **Tidak skalabel** — dengan jumlah bank yang besar, kedalaman analisis per bank terbatas oleh kapasitas analis.
3. **Tidak konsisten** — penilaian antar analis dapat berbeda untuk kasus ambang batas yang sama.

### 1.2 Solusi yang Diusulkan
Membangun sistem *Early Warning System* (EWS) berbasis machine learning yang:
- Memprediksi kondisi keuangan bank **6 bulan ke depan** secara otomatis.
- Menghasilkan **Peringkat Komposit RBBR** (1–5) yang dapat diaudit dan dijelaskan.
- Memberikan **sinyal peringatan dini** (hijau/kuning/merah) per bank per bulan.
- Mempercepat proses analisis analis dengan visualisasi dan penjelasan berbasis SHAP.

### 1.3 Ruang Lingkup
- **In scope:** Pipeline data, feature engineering RBBR, pelatihan model TFT + LightGBM + NeuralProphet + Stacking Ensemble, skoring RBBR, SHAP explainability, dashboard visualisasi, training di Google Colab.
- **Out of scope:** Integrasi real-time ke sistem core OJK, otentikasi pengguna multi-level, model GCG (karena data GCG bersifat kualitatif dan memerlukan perlakuan khusus di luar cakupan ini).

---

## 2. Tujuan & Keberhasilan

### 2.1 Tujuan Bisnis
| # | Tujuan | Metrik Keberhasilan |
|---|--------|---------------------|
| 1 | Deteksi dini penurunan kondisi bank | Recall sinyal merah ≥ 80% pada window 3 bulan sebelum penurunan PK |
| 2 | Otomatisasi skoring RBBR | Waktu skoring per bank turun dari ~4 jam → <5 menit |
| 3 | Konsistensi penilaian | Akurasi klasifikasi PK ≥ 75% dibanding penilaian pengawas |
| 4 | Explainability | Setiap prediksi memiliki penjelasan SHAP top-5 faktor |

### 2.2 Key Performance Indicators (KPI) Model
| Metrik | Target | Keterangan |
|--------|--------|------------|
| MAE (skor komposit) | ≤ 0.3 poin | Dari skala 1–5 |
| RMSE (per rasio) | ≤ 10% | Relatif terhadap nilai aktual |
| AUC-ROC (klasifikasi PK turun) | ≥ 0.80 | Binary: PK memburuk vs tidak |
| F1-Score (alarm merah) | ≥ 0.75 | Prioritas recall atas precision |
| MAPE (proyeksi 6 bulan) | ≤ 15% | Untuk variabel utama (CAR, NPL, NIM) |

---

## 3. Arsitektur Sistem

```
┌─────────────────────────────────────────────────────────────────────┐
│                        SUMBER DATA                                  │
│  Neraca │ Laba Rugi │ Kom. & Kontinjensi │ Rasio Keuangan │ Makro  │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   DATA PIPELINE (Google Colab)                      │
│  Ingestion → Validasi → Normalisasi → Feature Engineering           │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
              ┌────────────┼────────────────┐
              ▼            ▼                ▼
        ┌──────────┐ ┌──────────┐   ┌──────────────┐
        │   TFT    │ │LightGBM  │   │NeuralProphet │
        │ (primer) │ │(sekunder)│   │ (pendukung)  │
        └────┬─────┘ └────┬─────┘   └──────┬───────┘
             │             │                │
             └─────────────┼────────────────┘
                           ▼
                  ┌──────────────────┐
                  │  Meta-Learner    │
                  │ Ridge Regression │
                  │ (Stacking)       │
                  └────────┬─────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │  Mesin Skoring   │
                  │  RBBR (PK 1–5)  │
                  └────────┬─────────┘
                           │
                           ▼
          ┌────────────────────────────────┐
          │  Output: Skor + SHAP + Sinyal  │
          │  EWS + Dashboard Visualisasi   │
          └────────────────────────────────┘
```

---

## 4. Spesifikasi Data

### 4.1 Sumber Data Input

#### A. Laporan Keuangan Bank (Bulanan)

**Neraca (Balance Sheet)**
```
Aset:
- kas_dan_setara_kas
- penempatan_bi
- penempatan_bank_lain
- surat_berharga
- kredit_yang_diberikan
- kredit_bermasalah (NPL gross)
- penyisihan_kerugian_kredit (CKPN)
- aset_tetap
- total_aset

Liabilitas:
- giro
- tabungan
- deposito_berjangka
- pinjaman_diterima
- total_liabilitas

Ekuitas:
- modal_disetor
- laba_ditahan
- total_ekuitas
```

**Laba Rugi (P&L)**
```
- pendapatan_bunga
- beban_bunga
- pendapatan_bunga_bersih (NII)
- pendapatan_non_bunga (fee based)
- beban_operasional
- BOPO
- laba_sebelum_pajak
- laba_bersih
```

**Komitmen & Kontinjensi**
```
- LC_dan_SKBDN
- garansi_bank
- fasilitas_kredit_belum_digunakan
- total_komitmen_dan_kontinjensi
- rasio_komitmen_terhadap_aset
```

**Rasio Keuangan (sudah dihitung)**
```
Permodalan:
- CAR (KPMM)
- tier1_capital_ratio
- leverage_ratio

Kualitas Aset:
- NPL_gross
- NPL_net
- LAR (Loan at Risk)
- rasio_CKPN_terhadap_NPL

Profitabilitas:
- ROA
- ROE
- NIM
- BOPO

Likuiditas:
- LDR (Loan to Deposit Ratio)
- LCR (Liquidity Coverage Ratio)
- NSFR (Net Stable Funding Ratio)
- AL_DPK (Aset Likuid / DPK)

Efisiensi:
- CIR (Cost to Income Ratio)
- beban_per_aset
```

#### B. Data Makroekonomi (Bulanan)
```
- suku_bunga_bi_rate
- inflasi_mom
- kurs_usd_idr (JISDOR)
```

### 4.2 Format Data Input
```
Format file  : CSV atau Excel (.xlsx)
Granularitas : Bank × Bulan (panel data)
Identifier   : kode_bank (sesuai kode OJK), nama_bank, periode (YYYY-MM)
Histori min  : 36 bulan (3 tahun) untuk training awal
Histori ideal: 60–84 bulan (5–7 tahun) untuk menangkap siklus kredit
```

### 4.3 Skema Kolom Master
```python
# Skema DataFrame utama
df.columns = [
    'kode_bank',        # str  — identifier unik bank
    'nama_bank',        # str  — nama bank
    'kategori_bank',    # str  — BUKU 1/2/3/4
    'periode',          # datetime — YYYY-MM-01
    # ... semua variabel di atas ...
    'pk_rbbr_aktual',   # int  — label: PK 1-5 (dari penilaian pengawas)
    'pk_rbbr_3m_lalu',  # int  — PK 3 bulan sebelumnya (untuk change detection)
]
```

---

## 5. Feature Engineering

### 5.1 Temporal Features (Lag & Lead)

```python
# Untuk setiap variabel numerik kunci, buat lag 1–6 bulan
LAG_FEATURES = ['CAR', 'NPL_gross', 'NPL_net', 'NIM', 'ROA', 'ROE',
                'LDR', 'LCR', 'BOPO', 'total_aset', 'kredit_yang_diberikan',
                'suku_bunga_bi_rate', 'inflasi_mom', 'kurs_usd_idr']

for feat in LAG_FEATURES:
    for lag in range(1, 7):
        df[f'{feat}_lag{lag}'] = df.groupby('kode_bank')[feat].shift(lag)
```

### 5.2 Rolling Statistics

```python
ROLLING_WINDOWS = [3, 6]  # bulan

for feat in LAG_FEATURES:
    for window in ROLLING_WINDOWS:
        grp = df.groupby('kode_bank')[feat]
        df[f'{feat}_ma{window}']  = grp.transform(lambda x: x.rolling(window).mean())
        df[f'{feat}_std{window}'] = grp.transform(lambda x: x.rolling(window).std())
        df[f'{feat}_trend{window}'] = grp.transform(
            lambda x: x.rolling(window).apply(
                lambda v: np.polyfit(range(len(v)), v, 1)[0]  # slope
            )
        )
```

### 5.3 RBBR Pillar Features

```python
# Risk Profile Score (berbasis threshold OJK)
def score_npl(npl):
    if npl < 2:    return 1
    elif npl < 5:  return 2
    elif npl < 8:  return 3
    elif npl < 12: return 4
    else:          return 5

def score_car(car):
    if car >= 15:   return 1
    elif car >= 12: return 2
    elif car >= 10: return 3
    elif car >= 8:  return 4
    else:           return 5

def score_ldr(ldr):
    if 78 <= ldr <= 92:   return 1
    elif 60 <= ldr < 78:  return 2
    elif 92 < ldr <= 100: return 2
    elif ldr < 60:        return 3
    elif ldr <= 110:      return 3
    else:                 return 5

def score_roa(roa):
    if roa >= 1.5:    return 1
    elif roa >= 1.25: return 2
    elif roa >= 0.5:  return 3
    elif roa >= 0.0:  return 4
    else:             return 5

def score_bopo(bopo):
    if bopo <= 70:   return 1
    elif bopo <= 80: return 2
    elif bopo <= 85: return 3
    elif bopo <= 90: return 4
    else:            return 5

# Weighted RBBR score (bobot indikatif, dapat dikalibrasi)
RBBR_WEIGHTS = {
    'risk_profile': 0.50,   # 8 risiko utama
    'gcg':          0.15,   # tata kelola (placeholder)
    'rentabilitas': 0.15,   # ROA, ROE, NIM, BOPO
    'permodalan':   0.20,   # CAR, tier1
}

def compute_rbbr_score(row):
    risk_score = np.mean([score_npl(row['NPL_gross']),
                          score_ldr(row['LDR']),
                          score_car(row['CAR'])])
    rent_score = np.mean([score_roa(row['ROA']),
                          score_bopo(row['BOPO'])])
    perm_score = score_car(row['CAR'])
    gcg_score  = 2.0  # default; ganti dengan data GCG jika tersedia

    composite = (
        RBBR_WEIGHTS['risk_profile'] * risk_score +
        RBBR_WEIGHTS['gcg']          * gcg_score  +
        RBBR_WEIGHTS['rentabilitas'] * rent_score +
        RBBR_WEIGHTS['permodalan']   * perm_score
    )
    return composite
```

### 5.4 Interaction Features

```python
# Interaksi kondisi bank × makroekonomi
df['npl_x_birate']    = df['NPL_gross'] * df['suku_bunga_bi_rate']
df['car_x_kurs']      = df['CAR'] * df['kurs_usd_idr'] / 1e4
df['ldr_x_inflasi']   = df['LDR'] * df['inflasi_yoy']
df['nim_x_birate']    = df['NIM'] * df['suku_bunga_bi_rate']
df['bopo_x_inflasi']  = df['BOPO'] * df['inflasi_yoy']
```

### 5.5 Static (Time-invariant) Features

```python
STATIC_FEATURES = [
    'kategori_bank',    # BUKU 1/2/3/4 (one-hot encoded)
    'jenis_bank',       # konvensional / syariah
    'kepemilikan',      # BUMN / swasta nasional / asing / campuran
]
```

---

## 6. Spesifikasi Model

### 6.1 Model 1 — Temporal Fusion Transformer (TFT)

**Peran:** Model utama untuk multi-horizon forecasting (t+1 s.d. t+6).

**Arsitektur:**
```
Input layer:
  - Past time series     : variabel keuangan + makro (T=24 timesteps)
  - Static covariates    : kategori bank, jenis bank, kepemilikan
  - Known future inputs  : skenario makro (BI rate, kurs proyeksi)

TFT Blocks:
  - Gated Residual Network (GRN) untuk variable selection
  - LSTM encoder-decoder untuk temporal processing
  - Multi-head self-attention untuk long-range dependencies
  - Quantile output (10th, 50th, 90th percentile)

Output:
  - Prediksi 6 kuantil untuk setiap variabel target
  - Attention weights (interpretable)
  - Variable importance scores
```

**Hyperparameter awal (tuning via Optuna):**
```python
tft_config = {
    'hidden_size':           64,    # 32–128
    'lstm_layers':           2,     # 1–4
    'attention_head_size':   4,     # 1–8
    'dropout':               0.1,   # 0.0–0.3
    'hidden_continuous_size':32,    # 16–64
    'output_size':           7,     # 7 quantiles
    'loss':                  'QuantileLoss([0.1, 0.25, 0.5, 0.75, 0.9])',
    'learning_rate':         1e-3,
    'batch_size':            64,
    'max_epochs':            50,
    'gradient_clip_val':     0.1,
}
```

**Library:** `pytorch-forecasting` (di atas PyTorch + PyTorch Lightning)

**Target variabel:**
```python
TFT_TARGETS = ['CAR', 'NPL_gross', 'NIM', 'ROA', 'BOPO', 'LDR', 'rbbr_score_computed']
```

---

### 6.2 Model 2 — LightGBM

**Peran:** Model sekunder untuk klasifikasi kondisi bank dan feature importance.

**Task:** Multi-class classification (PK 1–5) + Regression (skor RBBR)

**Dua sub-model:**
```python
# Sub-model A: Regresi skor RBBR (kontinu)
lgbm_reg = LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    num_leaves=31,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
)

# Sub-model B: Klasifikasi PK (1–5)
lgbm_clf = LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    num_leaves=31,
    class_weight='balanced',  # penting: PK 4–5 jarang
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
)
```

**Input features:**
```python
# Semua fitur lag, rolling, interaksi, dan RBBR pillar scores
# Total estimasi: ~150–200 fitur
LGBM_FEATURES = (
    LAG_FEATURES_ALL +          # lag 1–6 semua variabel kunci
    ROLLING_FEATURES_ALL +      # MA3, MA6, std3, std6, trend3, trend6
    RBBR_PILLAR_SCORES +        # skor tiap pilar per bulan
    INTERACTION_FEATURES +      # fitur interaksi makro × bank
    STATIC_FEATURES_ENCODED     # one-hot kategori bank
)
```

**Optimasi:** Optuna dengan 100 trials, cross-validation time-series split (5 fold).

---

### 6.3 Model 3 — NeuralProphet

**Peran:** Model pendukung untuk forecasting variabel makroekonomi dan tren jangka menengah.

**Output yang digunakan sebagai input TFT & LightGBM (future covariates):**
```python
NEURALPROPHET_TARGETS = [
    'suku_bunga_bi_rate',
    'inflasi_yoy',
    'kurs_usd_idr',
    'ihsg',
]
```

**Konfigurasi:**
```python
from neuralprophet import NeuralProphet

# Satu model per variabel makro
np_model = NeuralProphet(
    n_forecasts=6,           # 6 bulan ke depan
    n_lags=12,               # AR dengan 12 lag
    yearly_seasonality=True,
    weekly_seasonality=False, # data bulanan
    daily_seasonality=False,
    learning_rate=0.01,
    epochs=200,
    batch_size=32,
    changepoints_range=0.9,
    n_changepoints=10,
    trend_reg=0.5,
    seasonality_reg=1.0,
)

# Tambahkan lagged regressors jika ada kausalitas
np_model.add_lagged_regressor('fed_funds_rate', n_lags=3)  # jika tersedia
```

**Output NeuralProphet** disimpan dan digunakan sebagai `known_future_covariates` pada TFT.

---

### 6.4 Model 4 — Stacking Ensemble (Meta-Learner: Ridge Regression)

**Arsitektur stacking:**
```
Level 0 (Base Learners):
  - TFT          → prediksi skor RBBR [t+1, t+2, ..., t+6] — kuantil median
  - LightGBM Reg → prediksi skor RBBR [t+1, ..., t+6]
  - LightGBM Clf → probabilitas tiap PK [t+1, ..., t+6]
  - NeuralProphet→ proyeksi makro [t+1, ..., t+6] (sebagai fitur tambahan)

Level 1 (Meta-Learner):
  - Ridge Regression (alpha dikalibrasi via cross-validation)
  - Input  : output semua base learner + fitur kontekstual (kategori bank)
  - Output : skor RBBR final (kontinu) → dibulatkan ke PK 1–5
```

**Implementasi cross-validation untuk stacking:**
```python
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import RidgeCV

# Out-of-fold predictions untuk menghindari data leakage
tscv = TimeSeriesSplit(n_splits=5, gap=6)  # gap=6 bulan untuk menghindari lookahead

oof_tft   = np.zeros((len(X_train), 6))  # 6 horizon
oof_lgbm  = np.zeros((len(X_train), 6))
# ... dst untuk setiap base learner

# Meta-learner
meta_X_train = np.hstack([oof_tft, oof_lgbm, oof_lgbm_clf_proba])
ridge_meta = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0], cv=5)
ridge_meta.fit(meta_X_train, y_train_composite)
```

---

## 7. Target Variabel & Label

### 7.1 Target Regresi
```python
# Skor RBBR komposit (kontinu, 1.0–5.0)
y_regression = df['rbbr_score_computed']  # dari formula section 5.3

# Per variabel kunci untuk 6 horizon
y_multivariate = df[['CAR', 'NPL_gross', 'NIM', 'ROA', 'BOPO', 'LDR']]
```

### 7.2 Target Klasifikasi
```python
# Peringkat Komposit (1–5)
y_pk = df['pk_rbbr_aktual']  # label dari penilaian pengawas (jika tersedia)

# Early Warning Binary (penurunan PK ≥ 1 dalam 6 bulan ke depan)
y_ews = (df['pk_rbbr_aktual'].shift(-6) > df['pk_rbbr_aktual']).astype(int)
```

### 7.3 Konversi Skor ke PK
```python
def score_to_pk(score: float) -> int:
    if score < 1.5:  return 1
    elif score < 2.5: return 2
    elif score < 3.5: return 3
    elif score < 4.5: return 4
    else:             return 5
```

---

## 8. Pipeline Training (Google Colab)

### 8.1 Struktur Direktori Proyek
```
/content/drive/MyDrive/OJK_EWS/
│
├── data/
│   ├── raw/
│   │   ├── neraca_YYYY.xlsx
│   │   ├── labarugi_YYYY.xlsx 
│   │   ├── komkont_YYYY.xlsx
│   │   ├── rasio_YYYY.xlsx
│   │   └── makro_YYYY.xlsx
│   ├── processed/
│   │   ├── master_panel.parquet
│   │   ├── features_engineered.parquet
│   │   ├── train_set.parquet
│   │   ├── val_set.parquet
│   │   └── test_set.parquet
│   └── macro_forecast/
│       └── neuralprophet_forecasts.parquet
│
├── notebooks/
│   ├── 01_data_ingestion.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_neuralprophet_macro.ipynb
│   ├── 05_lightgbm_training.ipynb
│   ├── 06_tft_training.ipynb
│   ├── 07_stacking_ensemble.ipynb
│   ├── 08_evaluation.ipynb
│   └── 09_dashboard.ipynb
│
├── src/
│   ├── data_pipeline.py
│   ├── feature_engineering.py
│   ├── rbbr_scoring.py
│   ├── models/
│   │   ├── tft_model.py
│   │   ├── lgbm_model.py
│   │   ├── neuralprophet_model.py
│   │   └── stacking.py
│   ├── evaluation.py
│   ├── explainability.py
│   └── visualization.py
│
├── models/
│   ├── tft_best/           # checkpoint PyTorch Lightning
│   ├── lgbm_reg.pkl
│   ├── lgbm_clf.pkl
│   ├── neuralprophet_*.pkl
│   └── ridge_meta.pkl
│
├── outputs/
│   ├── predictions/
│   │   └── forecast_YYYY_MM.parquet
│   ├── shap/
│   │   └── shap_values_YYYY_MM.parquet
│   └── reports/
│       └── ews_report_YYYY_MM.html
│
└── requirements.txt
```

### 8.2 Setup Google Colab (Notebook 00 — Setup)

```python
# ============================================================
# CELL 1: Mount Google Drive
# ============================================================
from google.colab import drive
drive.mount('/content/drive')

PROJECT_PATH = '/content/drive/MyDrive/OJK_RBBR_EWS'
import sys
sys.path.insert(0, f'{PROJECT_PATH}/src')

# ============================================================
# CELL 2: Install dependencies
# ============================================================
%%capture
!pip install pytorch-forecasting pytorch-lightning lightning
!pip install neuralprophet[live]
!pip install lightgbm optuna shap
!pip install plotly kaleido pandas numpy scikit-learn
!pip install pyarrow fastparquet openpyxl
!pip install mlflow  # opsional: experiment tracking

# ============================================================
# CELL 3: Verifikasi GPU
# ============================================================
import torch
print(f"CUDA tersedia: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
# Target: Tesla T4 (15 GB) — cukup untuk TFT dengan hidden_size=64
```

### 8.3 Notebook 01 — Data Ingestion

```python
# ============================================================
# CELL: Load dan merge semua laporan
# ============================================================
import pandas as pd
import numpy as np
from pathlib import Path

DATA_RAW = Path(f'{PROJECT_PATH}/data/raw')

def load_laporan(nama_file_pattern: str, sheet: str = 'Sheet1') -> pd.DataFrame:
    """Load semua file Excel untuk satu jenis laporan."""
    files = sorted(DATA_RAW.glob(nama_file_pattern))
    dfs = []
    for f in files:
        df = pd.read_excel(f, sheet_name=sheet)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

neraca    = load_laporan('neraca_*.xlsx')
labarugi  = load_laporan('labarugi_*.xlsx')
komkont   = load_laporan('komkont_*.xlsx')
rasio     = load_laporan('rasio_*.xlsx')
makro     = load_laporan('makro_*.xlsx')

# Standarisasi kolom identifier
for df in [neraca, labarugi, komkont, rasio]:
    df['periode'] = pd.to_datetime(df['periode'])
    df['kode_bank'] = df['kode_bank'].astype(str).str.zfill(3)

# Merge semua laporan pada kode_bank × periode
master = neraca.merge(labarugi, on=['kode_bank', 'periode'], how='outer', suffixes=('', '_pl'))
master = master.merge(komkont,  on=['kode_bank', 'periode'], how='outer')
master = master.merge(rasio,    on=['kode_bank', 'periode'], how='outer')

# Join makroekonomi (bukan per bank, tapi per periode)
master = master.merge(makro, on='periode', how='left')

# ============================================================
# CELL: Validasi kualitas data
# ============================================================
def validate_data(df: pd.DataFrame) -> None:
    print("=== VALIDASI DATA ===")
    print(f"Total baris   : {len(df):,}")
    print(f"Jumlah bank   : {df['kode_bank'].nunique()}")
    print(f"Periode       : {df['periode'].min()} s.d. {df['periode'].max()}")
    print(f"\nMissing values (top 15):")
    missing = df.isnull().sum().sort_values(ascending=False)
    print(missing[missing > 0].head(15).to_string())
    print(f"\nDuplikat: {df.duplicated(['kode_bank', 'periode']).sum()}")

validate_data(master)

# Simpan
master.to_parquet(f'{PROJECT_PATH}/data/processed/master_panel.parquet', index=False)
print("✓ master_panel.parquet tersimpan")
```

### 8.4 Notebook 03 — Feature Engineering

```python
# ============================================================
# CELL: Lag & rolling features
# ============================================================
df = pd.read_parquet(f'{PROJECT_PATH}/data/processed/master_panel.parquet')
df = df.sort_values(['kode_bank', 'periode']).reset_index(drop=True)

KEY_VARS = ['CAR', 'NPL_gross', 'NPL_net', 'NIM', 'ROA', 'ROE',
            'BOPO', 'LDR', 'LCR', 'total_aset',
            'suku_bunga_bi_rate', 'inflasi_yoy', 'kurs_usd_idr']

# Lag features
for var in KEY_VARS:
    if var not in df.columns:
        continue
    for lag in range(1, 7):
        df[f'{var}_lag{lag}'] = df.groupby('kode_bank')[var].shift(lag)

# Rolling features
for var in KEY_VARS:
    if var not in df.columns:
        continue
    grp = df.groupby('kode_bank')[var]
    for w in [3, 6]:
        df[f'{var}_ma{w}']    = grp.transform(lambda x: x.rolling(w, min_periods=1).mean())
        df[f'{var}_std{w}']   = grp.transform(lambda x: x.rolling(w, min_periods=2).std())
        df[f'{var}_trend{w}'] = grp.transform(
            lambda x: x.rolling(w, min_periods=w).apply(
                lambda v: np.polyfit(range(len(v)), v, 1)[0] if len(v) == w else np.nan
            )
        )

# RBBR pillar scores
df['score_npl']  = df['NPL_gross'].apply(score_npl)
df['score_car']  = df['CAR'].apply(score_car)
df['score_ldr']  = df['LDR'].apply(score_ldr)
df['score_roa']  = df['ROA'].apply(score_roa)
df['score_bopo'] = df['BOPO'].apply(score_bopo)
df['rbbr_score'] = df.apply(compute_rbbr_score, axis=1)
df['pk_computed'] = df['rbbr_score'].apply(score_to_pk)

# Interaction features
df['npl_x_birate']  = df['NPL_gross'] * df['suku_bunga_bi_rate']
df['car_x_kurs']    = df['CAR'] * df['kurs_usd_idr'] / 15000
df['ldr_x_inflasi'] = df['LDR'] * df['inflasi_yoy']
df['nim_x_birate']  = df['NIM'] * df['suku_bunga_bi_rate']

# One-hot encoding static features
df = pd.get_dummies(df, columns=['kategori_bank', 'jenis_bank', 'kepemilikan'],
                    drop_first=True)

# Simpan
df.to_parquet(f'{PROJECT_PATH}/data/processed/features_engineered.parquet', index=False)
print(f"✓ Features: {len(df.columns)} kolom, {len(df):,} baris")
```

### 8.5 Notebook 04 — NeuralProphet (Makro Forecasting)

```python
# ============================================================
# CELL: Train NeuralProphet untuk setiap variabel makro
# ============================================================
from neuralprophet import NeuralProphet
import pickle

makro_vars  = ['suku_bunga_bi_rate', 'inflasi_yoy', 'kurs_usd_idr', 'ihsg']
np_forecasts = {}

for var in makro_vars:
    print(f"\n--- Training NeuralProphet: {var} ---")

    # NeuralProphet menggunakan kolom 'ds' dan 'y'
    df_np = makro[['periode', var]].rename(columns={'periode': 'ds', var: 'y'}).dropna()
    df_np = df_np.sort_values('ds').reset_index(drop=True)

    model = NeuralProphet(
        n_forecasts=6,
        n_lags=12,
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        epochs=200,
        batch_size=32,
        learning_rate=0.01,
        trend_reg=0.5,
    )

    # Train / val split (80/20 temporal)
    split_idx = int(len(df_np) * 0.8)
    df_train, df_val = df_np.iloc[:split_idx], df_np.iloc[split_idx:]

    metrics = model.fit(df_train, validation_df=df_val, freq='MS', progress='plot')

    # Forecast 6 bulan ke depan dari data terkini
    future = model.make_future_dataframe(df_np, n_historic_predictions=True)
    forecast = model.predict(future)

    np_forecasts[var] = forecast[['ds', 'yhat1', 'yhat2', 'yhat3', 'yhat4', 'yhat5', 'yhat6']]

    # Simpan model
    with open(f'{PROJECT_PATH}/models/neuralprophet_{var}.pkl', 'wb') as f:
        pickle.dump(model, f)
    print(f"✓ Model {var} tersimpan")

# Gabungkan semua forecast makro
macro_future = pd.DataFrame()
# ... (merge semua np_forecasts ke satu DataFrame dengan periode sebagai index)

macro_future.to_parquet(f'{PROJECT_PATH}/data/macro_forecast/neuralprophet_forecasts.parquet')
```

### 8.6 Notebook 05 — LightGBM Training

```python
# ============================================================
# CELL: Train/val/test split (time-based)
# ============================================================
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, roc_auc_score
import lightgbm as lgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

df = pd.read_parquet(f'{PROJECT_PATH}/data/processed/features_engineered.parquet')
df = df.sort_values(['periode', 'kode_bank']).reset_index(drop=True)

# Split: train s.d. T-9bln, val T-9 s.d. T-3bln, test T-3bln s.d. T
cutoff_test = df['periode'].max() - pd.DateOffset(months=3)
cutoff_val  = cutoff_test - pd.DateOffset(months=6)

train = df[df['periode'] < cutoff_val]
val   = df[(df['periode'] >= cutoff_val) & (df['periode'] < cutoff_test)]
test  = df[df['periode'] >= cutoff_test]

# Buat target horizon 6 bulan ke depan
for h in range(1, 7):
    df[f'target_pk_h{h}'] = df.groupby('kode_bank')['pk_computed'].shift(-h)

FEATURE_COLS = [c for c in df.columns if any(c.endswith(s) for s in
    ['_lag1','_lag2','_lag3','_lag4','_lag5','_lag6',
     '_ma3','_ma6','_std3','_std6','_trend3','_trend6',
     '_x_','score_','_BUKU','_syariah','_swasta'])]

# ============================================================
# CELL: Optuna hyperparameter tuning
# ============================================================
def objective(trial):
    params = {
        'n_estimators':      trial.suggest_int('n_estimators', 200, 1000),
        'learning_rate':     trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'max_depth':         trial.suggest_int('max_depth', 3, 8),
        'num_leaves':        trial.suggest_int('num_leaves', 15, 63),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
        'subsample':         trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree':  trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha':         trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
        'reg_lambda':        trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
        'random_state': 42, 'verbose': -1,
    }
    model = lgb.LGBMRegressor(**params)
    model.fit(train[FEATURE_COLS], train['target_pk_h1'],
              eval_set=[(val[FEATURE_COLS], val['target_pk_h1'])],
              callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
    pred = model.predict(val[FEATURE_COLS])
    return mean_absolute_error(val['target_pk_h1'].dropna(), pred[:len(val['target_pk_h1'].dropna())])

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100, show_progress_bar=True)
best_params = study.best_params
print(f"Best MAE: {study.best_value:.4f}")
print(f"Best params: {best_params}")

# ============================================================
# CELL: Train final LightGBM dengan best params
# ============================================================
import pickle

# Regresi (skor RBBR)
lgbm_reg = lgb.LGBMRegressor(**best_params, random_state=42, verbose=-1)
lgbm_reg.fit(
    pd.concat([train, val])[FEATURE_COLS],
    pd.concat([train, val])['rbbr_score'],
)

# Klasifikasi (PK 1–5)
lgbm_clf = lgb.LGBMClassifier(**{**best_params, 'class_weight': 'balanced'},
                               random_state=42, verbose=-1)
lgbm_clf.fit(
    pd.concat([train, val])[FEATURE_COLS],
    pd.concat([train, val])['pk_computed'],
)

pickle.dump(lgbm_reg, open(f'{PROJECT_PATH}/models/lgbm_reg.pkl', 'wb'))
pickle.dump(lgbm_clf, open(f'{PROJECT_PATH}/models/lgbm_clf.pkl', 'wb'))
print("✓ LightGBM models tersimpan")
```

### 8.7 Notebook 06 — TFT Training

```python
# ============================================================
# CELL: Persiapan dataset untuk pytorch-forecasting
# ============================================================
import torch
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

df = pd.read_parquet(f'{PROJECT_PATH}/data/processed/features_engineered.parquet')
df = df.sort_values(['kode_bank', 'periode']).reset_index(drop=True)

# TFT memerlukan integer time_idx
df['time_idx'] = df.groupby('kode_bank')['periode'].rank(method='dense').astype(int)

MAX_ENCODER_LENGTH = 24  # 24 bulan histori
MAX_PREDICTION_LENGTH = 6  # prediksi 6 bulan ke depan

# Definisi TimeSeriesDataSet
training_cutoff = df['time_idx'].max() - MAX_PREDICTION_LENGTH

STATIC_CATS  = ['kategori_bank', 'jenis_bank', 'kepemilikan']
STATIC_REALS = []  # dapat ditambahkan jika ada numerik statis
TIME_VARYING_KNOWN_REALS = [
    'suku_bunga_bi_rate', 'inflasi_yoy', 'kurs_usd_idr',  # makro (diketahui dari NeuralProphet)
    'time_idx',
]
TIME_VARYING_UNKNOWN_REALS = [
    'CAR', 'NPL_gross', 'NIM', 'ROA', 'BOPO', 'LDR',
    'rbbr_score', 'score_npl', 'score_car', 'score_roa',
    # lag features
    'CAR_lag1', 'NPL_gross_lag1', 'NIM_lag1',
    'CAR_ma3', 'NPL_gross_ma3',
]

dataset_train = TimeSeriesDataSet(
    df[df['time_idx'] <= training_cutoff],
    time_idx             = 'time_idx',
    target               = 'rbbr_score',
    group_ids            = ['kode_bank'],
    min_encoder_length   = MAX_ENCODER_LENGTH // 2,
    max_encoder_length   = MAX_ENCODER_LENGTH,
    min_prediction_length= 1,
    max_prediction_length= MAX_PREDICTION_LENGTH,
    static_categoricals  = STATIC_CATS,
    static_reals         = STATIC_REALS,
    time_varying_known_reals     = TIME_VARYING_KNOWN_REALS,
    time_varying_unknown_reals   = TIME_VARYING_UNKNOWN_REALS,
    target_normalizer    = GroupNormalizer(groups=['kode_bank'], transformation='softplus'),
    add_relative_time_idx= True,
    add_target_scales    = True,
    add_encoder_length   = True,
)

dataset_val = TimeSeriesDataSet.from_dataset(
    dataset_train, df, predict=True, stop_randomization=True
)

train_loader = dataset_train.to_dataloader(train=True,  batch_size=64,  num_workers=2)
val_loader   = dataset_val.to_dataloader(  train=False, batch_size=128, num_workers=2)

# ============================================================
# CELL: Definisi dan training TFT
# ============================================================
pl.seed_everything(42)

tft = TemporalFusionTransformer.from_dataset(
    dataset_train,
    learning_rate           = 1e-3,
    hidden_size             = 64,
    attention_head_size     = 4,
    dropout                 = 0.1,
    hidden_continuous_size  = 32,
    output_size             = 7,
    loss                    = QuantileLoss(),
    log_interval            = 10,
    reduce_on_plateau_patience = 4,
)
print(f"Parameter TFT: {tft.size() / 1e3:.1f}k")

early_stop = EarlyStopping(monitor='val_loss', patience=10, mode='min')
checkpoint = ModelCheckpoint(
    dirpath     = f'{PROJECT_PATH}/models/tft_best',
    filename    = 'tft-{epoch:02d}-{val_loss:.4f}',
    save_top_k  = 1,
    monitor     = 'val_loss',
)

trainer = pl.Trainer(
    max_epochs          = 50,
    accelerator         = 'gpu' if torch.cuda.is_available() else 'cpu',
    gradient_clip_val   = 0.1,
    callbacks           = [early_stop, checkpoint],
    enable_progress_bar = True,
    log_every_n_steps   = 5,
)

trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=val_loader)
print(f"✓ TFT training selesai. Best val_loss: {trainer.callback_metrics['val_loss']:.4f}")

# Load best checkpoint
best_tft = TemporalFusionTransformer.load_from_checkpoint(checkpoint.best_model_path)
print(f"✓ Best model dimuat dari: {checkpoint.best_model_path}")
```

### 8.8 Notebook 07 — Stacking Ensemble

```python
# ============================================================
# CELL: Generate out-of-fold predictions untuk stacking
# ============================================================
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import pickle

# Load semua base models
lgbm_reg = pickle.load(open(f'{PROJECT_PATH}/models/lgbm_reg.pkl', 'rb'))
lgbm_clf = pickle.load(open(f'{PROJECT_PATH}/models/lgbm_clf.pkl', 'rb'))
best_tft  = TemporalFusionTransformer.load_from_checkpoint(
    sorted(Path(f'{PROJECT_PATH}/models/tft_best').glob('*.ckpt'))[-1]
)

# Collect test set predictions dari semua base models
pred_lgbm_reg  = lgbm_reg.predict(test[FEATURE_COLS])
pred_lgbm_proba = lgbm_clf.predict_proba(test[FEATURE_COLS])  # shape: [N, 5]

# TFT predictions (median quantile)
raw_preds = best_tft.predict(val_loader, mode='quantiles', return_index=True)
pred_tft_median = raw_preds.output[:, :, 3].numpy()  # quantile ke-4 = median (50th)

# ============================================================
# CELL: Train meta-learner Ridge Regression
# ============================================================
# Meta features: [tft_score, lgbm_score, lgbm_pk_proba_1..5, kategori_bank_encoded]
meta_X = np.column_stack([
    pred_tft_median.mean(axis=1),   # rata-rata 6 horizon TFT
    pred_lgbm_reg,
    pred_lgbm_proba,
])

meta_y = test['rbbr_score'].values[:len(meta_X)]

scaler_meta = StandardScaler()
meta_X_scaled = scaler_meta.fit_transform(meta_X)

ridge_meta = RidgeCV(alphas=np.logspace(-3, 3, 50), cv=5, scoring='neg_mean_absolute_error')
ridge_meta.fit(meta_X_scaled, meta_y)

print(f"Ridge alpha terpilih : {ridge_meta.alpha_:.4f}")
print(f"Meta-learner R²      : {ridge_meta.score(meta_X_scaled, meta_y):.4f}")

pickle.dump(ridge_meta,  open(f'{PROJECT_PATH}/models/ridge_meta.pkl', 'wb'))
pickle.dump(scaler_meta, open(f'{PROJECT_PATH}/models/scaler_meta.pkl', 'wb'))
print("✓ Stacking ensemble tersimpan")

# ============================================================
# CELL: Fungsi inference end-to-end
# ============================================================
def predict_bank_condition(kode_bank: str, periode_terkini: str) -> dict:
    """
    Prediksi kondisi keuangan bank untuk 6 bulan ke depan.

    Returns: dict berisi skor per horizon, PK, sinyal EWS, dan SHAP values.
    """
    # 1. Ambil data bank (24 bulan terakhir)
    bank_data = df[df['kode_bank'] == kode_bank].tail(24)

    # 2. Prediksi TFT
    # ... (inference loop)

    # 3. Prediksi LightGBM
    lgbm_score = lgbm_reg.predict(bank_data[FEATURE_COLS].tail(1))
    lgbm_proba = lgbm_clf.predict_proba(bank_data[FEATURE_COLS].tail(1))

    # 4. Ensemble
    meta_input = np.column_stack([tft_score, lgbm_score, lgbm_proba])
    meta_scaled = scaler_meta.transform(meta_input)
    final_score = ridge_meta.predict(meta_scaled)[0]

    # 5. Konversi ke PK dan sinyal EWS
    pk = score_to_pk(final_score)
    sinyal = 'hijau' if pk <= 2 else 'kuning' if pk == 3 else 'merah'

    return {
        'kode_bank':   kode_bank,
        'periode':     periode_terkini,
        'skor_final':  round(final_score, 3),
        'pk_prediksi': pk,
        'sinyal_ews':  sinyal,
        'proyeksi_6bln': [...]  # skor per horizon
    }
```

---

## 9. Evaluasi & Backtesting

### 9.1 Metrik Evaluasi

```python
# ============================================================
# CELL: Evaluasi komprehensif
# ============================================================
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                              classification_report, confusion_matrix,
                              roc_auc_score, f1_score)

def evaluate_model(y_true_score, y_pred_score, y_true_pk, y_pred_pk):
    print("=== EVALUASI REGRESI ===")
    mae  = mean_absolute_error(y_true_score, y_pred_score)
    rmse = np.sqrt(mean_squared_error(y_true_score, y_pred_score))
    mape = np.mean(np.abs((y_true_score - y_pred_score) / (y_true_score + 1e-6))) * 100
    print(f"MAE  : {mae:.4f}  (target ≤ 0.30)")
    print(f"RMSE : {rmse:.4f}")
    print(f"MAPE : {mape:.2f}% (target ≤ 15%)")

    print("\n=== EVALUASI KLASIFIKASI PK ===")
    print(classification_report(y_true_pk, y_pred_pk,
          target_names=['PK1','PK2','PK3','PK4','PK5']))

    print("\n=== EARLY WARNING (deteksi penurunan PK) ===")
    y_ews_true = (y_true_pk > y_true_pk.shift(1)).astype(int)
    y_ews_pred = (y_pred_pk > y_true_pk).astype(int)
    f1  = f1_score(y_ews_true, y_ews_pred, zero_division=0)
    auc = roc_auc_score(y_ews_true, y_ews_pred)
    print(f"F1-Score EWS : {f1:.4f}  (target ≥ 0.75)")
    print(f"AUC-ROC EWS  : {auc:.4f}  (target ≥ 0.80)")
```

### 9.2 Backtesting pada Periode Krisis

```python
# Walk-forward backtesting
def walk_forward_backtest(df, start_date, end_date, retrain_freq=6):
    """
    Walk-forward validation:
    - Retrain setiap 6 bulan
    - Evaluasi prediksi H+1 s.d. H+6
    - Bandingkan dengan PK aktual (jika tersedia)
    """
    periods = pd.date_range(start_date, end_date, freq='6MS')
    results = []

    for cutoff in periods:
        train_data = df[df['periode'] < cutoff]
        test_data  = df[(df['periode'] >= cutoff) &
                        (df['periode'] < cutoff + pd.DateOffset(months=6))]
        # ... retrain + predict + evaluate
        results.append({'cutoff': cutoff, 'metrics': {...}})

    return pd.DataFrame(results)

# Backtest periode pandemi COVID-19 (stres test penting)
backtest_covid = walk_forward_backtest(df, '2019-01-01', '2021-12-31')
```

---

## 10. SHAP Explainability

```python
# ============================================================
# CELL: SHAP untuk LightGBM
# ============================================================
import shap
import matplotlib.pyplot as plt

explainer_lgbm = shap.TreeExplainer(lgbm_reg)
shap_values    = explainer_lgbm.shap_values(test[FEATURE_COLS])

# Summary plot (global importance)
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, test[FEATURE_COLS], max_display=20, show=False)
plt.title('SHAP Feature Importance — LightGBM Regresi RBBR')
plt.tight_layout()
plt.savefig(f'{PROJECT_PATH}/outputs/shap/shap_summary_lgbm.png', dpi=150, bbox_inches='tight')
plt.show()

# Waterfall plot untuk satu bank spesifik
def explain_bank_prediction(kode_bank: str, periode: str):
    """Tampilkan penjelasan SHAP untuk prediksi satu bank."""
    bank_row = test[(test['kode_bank'] == kode_bank) &
                    (test['periode'] == pd.to_datetime(periode))]
    if bank_row.empty:
        print(f"Data tidak ditemukan untuk {kode_bank} periode {periode}")
        return

    sv = explainer_lgbm(bank_row[FEATURE_COLS])
    plt.figure(figsize=(10, 5))
    shap.plots.waterfall(sv[0], max_display=10, show=False)
    plt.title(f'Penjelasan Prediksi: Bank {kode_bank} — {periode}')
    plt.tight_layout()
    plt.show()

    # Top 5 faktor risiko
    feat_impact = pd.Series(sv[0].values, index=FEATURE_COLS).abs().sort_values(ascending=False)
    print(f"\nTop 5 faktor risiko untuk {kode_bank} ({periode}):")
    for i, (feat, val) in enumerate(feat_impact.head(5).items()):
        direction = '▲ naik' if sv[0].values[FEATURE_COLS.index(feat)] > 0 else '▼ turun'
        print(f"  {i+1}. {feat}: {direction} (impact: {val:.3f})")

# Contoh penggunaan
explain_bank_prediction('001', '2025-06-01')

# TFT Attention weights (interpretasi temporal)
interpretation = best_tft.interpret_output(
    best_tft.predict(val_loader, mode='raw', return_x=True),
    reduction='sum'
)
best_tft.plot_interpretation(interpretation)
```

---

## 11. Dashboard & Visualisasi

### 11.1 Komponen Dashboard

```python
# ============================================================
# CELL: Dashboard menggunakan Plotly (export ke HTML)
# ============================================================
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

def create_ews_dashboard(predictions_df: pd.DataFrame, periode: str) -> go.Figure:
    """
    Buat dashboard early warning interaktif untuk satu periode.
    """

    # --- 1. Heatmap kondisi bank (bank × bulan) ---
    fig_heatmap = px.imshow(
        pivot_pk,                          # DataFrame: bank × bulan, nilai PK
        color_continuous_scale=[[0, '#2d6a4f'], [0.4, '#f9c74f'], [1, '#d62828']],
        zmin=1, zmax=5,
        title=f'Heatmap Peringkat Komposit — Per Bank Per Bulan',
        labels={'color': 'PK RBBR'},
    )

    # --- 2. Bar chart distribusi PK ---
    pk_counts = predictions_df['pk_prediksi'].value_counts().sort_index()
    fig_dist = go.Figure(go.Bar(
        x=[f'PK {i}' for i in pk_counts.index],
        y=pk_counts.values,
        marker_color=['#2d6a4f','#52b788','#f9c74f','#f4845f','#d62828'],
    ))

    # --- 3. Radar chart RBBR 4 pilar untuk satu bank ---
    def radar_rbbr(kode_bank):
        bank = predictions_df[predictions_df['kode_bank'] == kode_bank].iloc[-1]
        fig_radar = go.Figure(go.Scatterpolar(
            r=[bank['score_risk_profile'], bank['score_gcg'],
               bank['score_rentabilitas'], bank['score_permodalan'],
               bank['score_risk_profile']],
            theta=['Risk Profile','GCG','Rentabilitas','Permodalan','Risk Profile'],
            fill='toself', name=kode_bank,
        ))
        fig_radar.update_layout(polar=dict(radialaxis=dict(range=[0, 5])))
        return fig_radar

    # --- 4. Tren proyeksi 6 bulan dengan confidence interval ---
    def trend_forecast(kode_bank):
        bank_pred = predictions_df[predictions_df['kode_bank'] == kode_bank]
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=bank_pred['periode'], y=bank_pred['skor_q50'],
            name='Proyeksi (median)', line=dict(color='#4361ee', width=2),
        ))
        fig_trend.add_trace(go.Scatter(
            x=pd.concat([bank_pred['periode'], bank_pred['periode'][::-1]]),
            y=pd.concat([bank_pred['skor_q90'], bank_pred['skor_q10'][::-1]]),
            fill='toself', fillcolor='rgba(67,97,238,0.15)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Confidence interval (10–90%)',
        ))
        # Garis threshold PK
        for pk, val, color in [(2.5, 2.5, '#f9c74f'), (3.5, 3.5, '#d62828')]:
            fig_trend.add_hline(y=val, line_dash='dash', line_color=color,
                                annotation_text=f'Threshold PK {int(pk+0.5)}')
        return fig_trend

    return {'heatmap': fig_heatmap, 'dist': fig_dist}

# Export ke HTML
dashboard = create_ews_dashboard(predictions_df, '2025-06')
for name, fig in dashboard.items():
    fig.write_html(f'{PROJECT_PATH}/outputs/reports/dashboard_{name}.html')
    print(f"✓ Dashboard {name} tersimpan")
```

---

## 12. Manajemen Eksperimen & Reproducibility

```python
# ============================================================
# CELL: MLflow tracking (opsional, direkomendasikan)
# ============================================================
import mlflow
import mlflow.lightgbm

EXPERIMENT_NAME = 'OJK_RBBR_EWS'
mlflow.set_experiment(EXPERIMENT_NAME)

with mlflow.start_run(run_name='LightGBM_Optuna_v1'):
    mlflow.log_params(best_params)
    mlflow.log_metrics({
        'val_mae':  val_mae,
        'val_rmse': val_rmse,
        'val_auc':  val_auc,
        'val_f1':   val_f1,
    })
    mlflow.lightgbm.log_model(lgbm_reg, 'lgbm_reg')
    mlflow.log_artifact(f'{PROJECT_PATH}/outputs/shap/shap_summary_lgbm.png')
    print("✓ Eksperimen dicatat di MLflow")

# Seed management untuk reproducibility
import random
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
pl.seed_everything(SEED, workers=True)
```

---

## 13. Deployment & Production System (Streamlit + Hugging Face)

### 13.1 Arsitektur Deployment
Aplikasi EWS di-deploy sebagai web application interaktif menggunakan **Streamlit** dan di-host secara permanen di **Hugging Face Spaces**.

- **Frontend:** Streamlit UI untuk navigasi multi-page (Dashboard, Detail Bank, Prediksi, Upload Data, Retraining, Monitoring).
- **Backend Storage:** Hugging Face Hub digunakan sebagai *Persistent Storage* untuk mengatasi sifat *ephemeral* dari Streamlit Spaces.
- **Environment:** Containerized environment dengan `libomp-dev` untuk optimasi LightGBM.

### 13.2 Hugging Face Storage Integration
Implementasi `HFStorageManager` untuk sinkronisasi data secara dua arah antara memori lokal dan repository HF:
- **Upload Data:** File CSV/Excel yang diunggah pengguna divalidasi dan langsung disimpan ke direktori `data/` di HF.
- **Model Versioning:** Metadata pelatihan model disimpan di `models/` menggunakan `ModelVersionManager` dengan cap waktu (*timestamp*) untuk pelacakan versi.

### 13.3 Automated Retraining UI
Sistem dilengkapi dengan antarmuka untuk melakukan *retraining* model secara langsung dari aplikasi tanpa perlu menjalankan pipeline Colab:
- **Full Ensemble Retraining:** Melatih ulang 4 model utama (TFT, LightGBM, NeuralProphet, SARIMAX) menggunakan antarmuka UI.
- **Progress Tracking:** Indikator real-time untuk status *training* per model, menghindari kebingungan pengguna (*timeout prevention*).
- **Auto-Push to HF:** Model yang selesai di-*fine-tune* akan otomatis diunggah kembali ke repository Hugging Face.

### 13.4 Manajemen Memori (OOM Prevention)
Mengingat beban komputasi ensemble yang tinggi (TFT + LightGBM), fitur *garbage collection* (`gc.collect()`) dan monitoring RAM (berbasis `psutil`) diimplementasikan secara dinamis untuk memberi *warning* bila penggunaan memori melebihi 80%.

---

## 14. Requirements & Dependensi

```txt
# requirements.txt

# Core
pandas==2.2.0
numpy==1.26.4
scikit-learn==1.4.0
scipy==1.12.0

# Deep Learning
torch==2.2.0
pytorch-lightning==2.2.0
lightning==2.2.0

# Forecasting
pytorch-forecasting==1.0.0
neuralprophet==0.9.0

# Gradient Boosting
lightgbm==4.3.0
optuna==3.6.0

# Explainability
shap==0.45.0

# Visualization
plotly==5.20.0
matplotlib==3.8.3
seaborn==0.13.2
kaleido==0.2.1

# Data I/O
pyarrow==15.0.0
fastparquet==2024.2.0
openpyxl==3.1.2
xlrd==2.0.1

# Experiment Tracking
mlflow==2.11.0

# Utils
tqdm==4.66.2
python-dotenv==1.0.1
```

---

## 15. Jadwal & Estimasi Waktu Training (Google Colab)

| Tahap | Notebook | Estimasi Waktu | Runtime |
|-------|----------|----------------|---------|
| Data ingestion & validasi | 01 | 15–30 mnt | CPU |
| EDA | 02 | 30 mnt | CPU |
| Feature engineering | 03 | 20–45 mnt | CPU |
| NeuralProphet (4 variabel) | 04 | 30–60 mnt | CPU/GPU |
| LightGBM + Optuna (100 trials) | 05 | 60–90 mnt | CPU |
| TFT training (50 epoch) | 06 | 90–180 mnt | **GPU (T4)** |
| Stacking ensemble | 07 | 15–30 mnt | CPU |
| Evaluasi & SHAP | 08 | 30–60 mnt | CPU/GPU |
| Dashboard | 09 | 30 mnt | CPU |
| **Total** | | **~7–8 jam** | |

> **Catatan Colab:** Gunakan Colab Pro atau Pro+ untuk runtime GPU yang tidak terputus. Simpan checkpoint secara berkala ke Google Drive setiap selesai satu notebook.

---

## 16. Risiko & Mitigasi

| Risiko | Dampak | Probabilitas | Mitigasi |
|--------|--------|--------------|----------|
| Data tidak lengkap / banyak missing | Tinggi | Sedang | Imputation strategy; validasi minimum 24 bulan histori per bank |
| TFT overfitting pada bank kecil | Sedang | Tinggi | Dropout + early stopping; group normalization; augmentasi data |
| Distribusi PK tidak seimbang (PK 4–5 jarang) | Tinggi | Tinggi | `class_weight='balanced'` di LightGBM; SMOTE untuk upsampling |
| Colab session timeout saat training TFT | Sedang | Sedang | Checkpoint setiap epoch; resuming dari checkpoint |
| Data snooping / look-ahead bias | Tinggi | Sedang | TimeSeriesSplit ketat; gap=6 bulan di stacking CV |
| Perubahan metodologi RBBR OJK | Tinggi | Rendah | Modularisasi mesin skoring; parameter bobot di config file |

---

## 17. Glosarium

| Istilah | Definisi |
|---------|----------|
| RBBR | Risk-Based Bank Rating — kerangka penilaian tingkat kesehatan bank berbasis risiko |
| PK | Peringkat Komposit (1=sangat sehat, 5=tidak sehat) |
| TFT | Temporal Fusion Transformer — arsitektur deep learning untuk multi-horizon forecasting |
| EWS | Early Warning System — sistem peringatan dini |
| SHAP | SHapley Additive exPlanations — metode explainability model ML |
| CAR | Capital Adequacy Ratio (KPMM) |
| NPL | Non-Performing Loan |
| NIM | Net Interest Margin |
| LDR | Loan to Deposit Ratio |
| BOPO | Beban Operasional / Pendapatan Operasional |
| GRN | Gated Residual Network — komponen dalam TFT |
| OOF | Out-of-Fold — prediksi yang dihasilkan di luar data training untuk stacking |

---

*Dokumen ini bersifat internal OJK DPKP. Dilarang disebarluaskan tanpa izin.*
*Versi berikutnya (v1.1) akan mencakup: integrasi GCG kualitatif, stress testing skenario makro, dan API endpoint untuk integrasi sistem pengawasan.*
