"""
Upload models & data ke nikodwicahyo/EWS-DPKP-storage
Jalankan: python scripts/upload_to_hf.py
"""

import os
import sys
from pathlib import Path

# Tambah root ke path
sys.path.insert(0, str(Path(__file__).parent.parent))

from huggingface_hub import HfApi

# ── CONFIG ────────────────────────────────────────────────────────────────────
REPO_ID    = "nikodwicahyo/EWS-DPKP-storage"
REPO_TYPE  = "dataset"
HF_TOKEN   = os.getenv("HF_TOKEN") or input("Masukkan HF Token: ").strip()

ROOT = Path(__file__).parent.parent

# File yang akan diupload: (local_path, repo_path)
FILES = [
    # Data
    ("data/processed/master_panel.parquet",   "data/processed/master_panel.parquet"),
    ("data/predictions/forecast_6m.parquet",  "data/predictions/forecast_6m.parquet"),
    ("data/macro_forecast/macro_fc.parquet",  "data/macro_forecast/macro_fc.parquet"),
    ("data/static/bank_metadata.csv",         "data/static/bank_metadata.csv"),
    ("data/static/thresholds_rbbr.json",      "data/static/thresholds_rbbr.json"),
    # Models
    ("models/lgbm_reg_models.pkl",  "models/lgbm_reg_models.pkl"),
    ("models/lgbm_clf.pkl",         "models/lgbm_clf.pkl"),
    ("models/feat_cols.pkl",        "models/feat_cols.pkl"),
    ("models/np_bi_rate.pkl",       "models/np_bi_rate.pkl"),
    ("models/np_inflasi.pkl",       "models/np_inflasi.pkl"),
    ("models/np_kurs_usd.pkl",      "models/np_kurs_usd.pkl"),
    ("models/sarimax_fc.pkl",       "models/sarimax_fc.pkl"),
    ("models/sarimax.pkl",          "models/sarimax.pkl"),
    ("models/model_metadata.json",  "models/model_metadata.json"),
]

# ── UPLOAD ────────────────────────────────────────────────────────────────────
api = HfApi(token=HF_TOKEN)

# Pastikan repo ada (buat jika belum)
try:
    api.repo_info(repo_id=REPO_ID, repo_type=REPO_TYPE)
    print(f"✅ Repo ditemukan: {REPO_ID}")
except Exception:
    print(f"📦 Membuat repo baru: {REPO_ID}")
    api.create_repo(repo_id=REPO_ID, repo_type=REPO_TYPE, private=True)

# Upload satu per satu
for local_rel, repo_path in FILES:
    local_path = ROOT / local_rel
    if not local_path.exists():
        print(f"⚠️  Skip (tidak ada): {local_rel}")
        continue

    size_mb = local_path.stat().st_size / 1_048_576
    print(f"⬆️  Uploading {local_rel} ({size_mb:.1f} MB) → {repo_path} ...", end=" ", flush=True)
    try:
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=repo_path,
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            commit_message=f"sync: {repo_path}",
        )
        print("✅")
    except Exception as e:
        print(f"❌ Error: {e}")

print("\n🎉 Upload selesai!")
print(f"   Lihat di: https://huggingface.co/datasets/{REPO_ID}")
