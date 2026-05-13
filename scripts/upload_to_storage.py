"""
upload_to_storage.py
--------------------
Upload model binaries & data files ke Hugging Face Dataset repository:
    nikodwicahyo/EWS-DPKP-storage

Usage:
    python scripts/upload_to_storage.py
    python scripts/upload_to_storage.py --token hf_xxxx
    python scripts/upload_to_storage.py --dry-run

Struktur di HF repo:
    models/   ← semua .pkl, .ckpt, .json (model_metadata.json)
    data/     ← parquet files (master_panel, forecast_6m, macro_fc)
"""

import os
import sys
import argparse
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).parent.parent
MODEL_DIR  = ROOT_DIR / "models"
DATA_DIR   = ROOT_DIR / "data"

STORAGE_REPO = "nikodwicahyo/EWS-DPKP-storage"

# ─── Files to upload ──────────────────────────────────────────────────────────
# Format: (local_path, remote_path_in_repo)
UPLOAD_MANIFEST = [
    # ── Models ─────────────────────────────────────────────────────────────
    (MODEL_DIR / "model_metadata.json",              "models/model_metadata.json"),
    (MODEL_DIR / "lgbm_reg_models.pkl",              "models/lgbm_reg_models.pkl"),
    (MODEL_DIR / "lgbm_clf.pkl",                     "models/lgbm_clf.pkl"),
    (MODEL_DIR / "feat_cols.pkl",                    "models/feat_cols.pkl"),
    (MODEL_DIR / "np_bi_rate.pkl",                   "models/np_bi_rate.pkl"),
    (MODEL_DIR / "np_inflasi.pkl",                   "models/np_inflasi.pkl"),
    (MODEL_DIR / "np_kurs_usd.pkl",                  "models/np_kurs_usd.pkl"),
    (MODEL_DIR / "sarimax_fc.pkl",                   "models/sarimax_fc.pkl"),
    (MODEL_DIR / "sarimax.pkl",                      "models/sarimax.pkl"),
    # ── Data ───────────────────────────────────────────────────────────────
    (DATA_DIR / "processed"    / "master_panel.parquet",  "data/processed/master_panel.parquet"),
    (DATA_DIR / "predictions"  / "forecast_6m.parquet",   "data/predictions/forecast_6m.parquet"),
    (DATA_DIR / "macro_forecast" / "macro_fc.parquet",    "data/macro_forecast/macro_fc.parquet"),
]

# TFT checkpoints (glob pattern)
TFT_CKPTS = list(MODEL_DIR.glob("tft-retrained-*.ckpt"))


def get_token(cli_token: str | None) -> str:
    """Resolve HF token: CLI → env var → secrets.toml"""
    if cli_token:
        return cli_token

    token = os.getenv("HF_TOKEN")
    if token:
        return token

    # Try .streamlit/secrets.toml
    secrets_path = ROOT_DIR / ".streamlit" / "secrets.toml"
    if secrets_path.exists():
        try:
            import tomllib  # Python 3.11+
        except ImportError:
            try:
                import tomli as tomllib
            except ImportError:
                tomllib = None

        if tomllib:
            with open(secrets_path, "rb") as f:
                secrets = tomllib.load(f)
            if "HF_TOKEN" in secrets:
                return secrets["HF_TOKEN"]
        else:
            # Fallback manual parse
            for line in secrets_path.read_text().splitlines():
                if line.strip().startswith("HF_TOKEN"):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")

    raise ValueError(
        "HF_TOKEN tidak ditemukan.\n"
        "Set dengan: --token hf_xxx  ATAU  env var HF_TOKEN=hf_xxx"
    )


def upload_all(token: str, dry_run: bool = False):
    from huggingface_hub import HfApi

    api = HfApi(token=token)

    # Pastikan repo dataset ada, buat jika belum ada
    try:
        api.repo_info(repo_id=STORAGE_REPO, repo_type="dataset")
        print(f"✅ Repo ditemukan: {STORAGE_REPO}")
    except Exception:
        if not dry_run:
            print(f"📦 Membuat repo dataset baru: {STORAGE_REPO} ...")
            api.create_repo(
                repo_id=STORAGE_REPO,
                repo_type="dataset",
                private=True,
                exist_ok=True,
            )
            print(f"✅ Repo berhasil dibuat: {STORAGE_REPO}")
        else:
            print(f"[DRY-RUN] Akan membuat repo: {STORAGE_REPO}")

    # Build final manifest (static + TFT ckpts)
    manifest = list(UPLOAD_MANIFEST)
    for ckpt in TFT_CKPTS:
        manifest.append((ckpt, f"models/{ckpt.name}"))

    print(f"\n📋 Total files dalam manifest: {len(manifest)}")
    print("-" * 60)

    success, skipped, failed = 0, 0, 0

    for local_path, remote_path in manifest:
        local_path = Path(local_path)

        if not local_path.exists():
            print(f"  ⚠️  SKIP  (tidak ada lokal): {local_path.name}")
            skipped += 1
            continue

        size_mb = local_path.stat().st_size / 1_048_576
        print(f"  ⬆️  {remote_path:<55} ({size_mb:.1f} MB)", end=" ", flush=True)

        if dry_run:
            print("[DRY-RUN]")
            success += 1
            continue

        try:
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=remote_path,
                repo_id=STORAGE_REPO,
                repo_type="dataset",
                commit_message=f"upload: {local_path.name}",
            )
            print("✅")
            success += 1
        except Exception as e:
            print(f"❌ GAGAL: {e}")
            failed += 1

    print("-" * 60)
    print(f"\n📊 Hasil: {success} berhasil | {skipped} dilewati | {failed} gagal")
    if not dry_run:
        print(f"\n🔗 Cek repo: https://huggingface.co/datasets/{STORAGE_REPO}")


def main():
    parser = argparse.ArgumentParser(
        description="Upload model & data ke HF Dataset repo EWS-DPKP-storage"
    )
    parser.add_argument("--token", help="HF access token (write)")
    parser.add_argument("--dry-run", action="store_true", help="Preview saja, tidak upload")
    args = parser.parse_args()

    try:
        token = get_token(args.token)
    except ValueError as e:
        print(f"❌ {e}")
        sys.exit(1)

    print(f"🚀 Upload ke: {STORAGE_REPO}")
    print(f"📁 Root dir: {ROOT_DIR}")
    if args.dry_run:
        print("⚡ Mode: DRY-RUN (tidak ada yang diupload)\n")

    upload_all(token=token, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
