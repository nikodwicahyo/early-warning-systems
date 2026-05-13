# TRAINING_DOCUMENTATION — OJK DPKP EWS Prediksi Kondisi Keuangan Bank v4.0

## 1. Ringkasan Proyek

Notebook `OJK_RBBR_EWS_v4_2.ipynb` membangun pipeline **end-to-end** untuk memprediksi kondisi keuangan bank dan peringatan dini (Early Warning System/EWS) berbasis data laporan OJK. Pipeline ini menggabungkan:

- **Laporan Neraca**
- **Laporan Laba Rugi**
- **Laporan Administratif (Komitmen & Kontinjensi)**
- **Laporan Rasio**
- **Faktor makro ekonomi**

Target utama yang dituliskan pada notebook adalah:

- **MAE ≤ 0.30**
- **Accuracy PK ≥ 90%**
- **F1 ≥ 0.75**
- **MAPE ≤ 15%**

Versi ini memakai pendekatan multi-model: `NeuralProphet`, `LightGBM`, `Temporal Fusion Transformer (TFT)`, `SARIMAX`, lalu digabung dengan **stacking ensemble** dan ditunjang **SHAP** untuk interpretabilitas.

---

## 2. Cakupan Data dan Bank yang Diproses

Notebook memproses 4 bank utama:

- **MANDIRI**
- **BNI**
- **BRI**
- **BTN**

Konfigurasi bank memuat:

- `file` Excel sumber data
- `kode_bank`
- `kategori` bank
- `kepemilikan`

Jika data riil tidak lengkap, notebook menyediakan **synthetic generator** sebagai fallback agar pipeline tetap dapat dijalankan.

---

## 3. Lingkungan Eksekusi dan Dependensi

Cell setup melakukan instalasi dan/atau penguncian versi paket penting, antara lain:

- `pytorch-lightning`
- `lightgbm`
- `optuna`
- `scikit-learn`
- `neuralprophet`
- `pytorch-forecasting`
- `statsmodels`
- `pmdarima`
- `shap`
- `plotly`
- `pyarrow`
- `openpyxl`

Notebook dirancang untuk berjalan di **Google Colab** dan melakukan mount Google Drive ke path proyek:

`/content/drive/MyDrive/OJK_RBBR_V4-2`

Struktur folder output yang dibuat meliputi:

- `data/raw`
- `data/processed`
- `data/macro_forecast`
- `models`
- `outputs/predictions`
- `outputs/shap`
- `outputs/reports`
- `outputs/plots`

---

## 4. Alur Besar Pipeline Training

Secara logis, notebook berjalan dalam urutan berikut:

1. **Setup environment**
2. **Parser universal** untuk semua sheet laporan OJK
3. **Load laporan riil**, lalu fallback ke synthetic jika perlu
4. **Merge data makro**
5. **Hitung RBBR score dan PK**
6. **Bangun derived features lintas laporan**
7. **Feature engineering**: lag, rolling, momentum, interaksi, kalender
8. **Augmentasi data realistis**
9. **Forecast faktor makro** dengan NeuralProphet
10. **Train LightGBM regresi** untuk semua rasio target
11. **Train LightGBM klasifikasi** untuk PK
12. **Train TFT** untuk multivariate multi-horizon forecasting
13. **Train SARIMAX** per rasio per bank
14. **Gabungkan semua model** dalam stacking ensemble
15. **Prediksi 6 bulan ke depan**
16. **Interpretasi SHAP**
17. **Visualisasi dashboard EWS**

---

## 5. Universal Parser Laporan OJK

Notebook menggunakan parser universal agar bisa membaca format laporan OJK yang sering berbeda antar file/sheet.

### 5.1 Fungsi utama

Fungsi `parse_sheet_auto(filepath, sheet_name)` melakukan:

- membaca sheet Excel tanpa asumsi header tetap
- mendeteksi baris header otomatis berdasarkan banyaknya kolom periode
- mengubah kolom header menjadi `Timestamp`
- mengekstrak baris data sebagai pasangan:
  - `nama item laporan`
  - `nilai per periode`

### 5.2 Normalisasi dan cleaning

Parser didukung oleh utilitas:

- `parse_period(s)` untuk mendeteksi format periode
- `clean_num(v)` untuk membersihkan angka dari format teks, pemisah ribuan, tanda kurung, dan nilai non-numerik

### 5.3 Pemetaan nama variabel

Notebook memakai mapping dari nama asli laporan OJK ke nama standar yang konsisten. Pemetaan dibagi menjadi:

- **NERACA**
- **LABA RUGI**
- **ADMINISTRATIF**
- **RASIO**

Hasil akhirnya adalah data panel yang berisi variabel standar seperti:

- `kas`
- `kredit_diberikan`
- `total_aset`
- `NII`
- `laba_bersih`
- `tagihan_komitmen`
- `CAR`
- `NPL_gross`
- `ROA`
- dan lainnya

### 5.4 Output parser

Fungsi `load_bank_all_statements(bank, cfg, data_path)` menggabungkan seluruh sheet ke format **wide panel**:

- baris = periode
- kolom = variabel keuangan
- metadata = `bank`, `kode_bank`, `kategori_bank`, `kepemilikan`

---

## 6. Fallback Synthetic Data

Jika data riil tidak lengkap, notebook membangkitkan data sintetis dengan `generate_bank_synthetic()`.

Karakteristik synthetic data:

- berbasis periode bulanan
- memakai dinamika mirip proses time series
- ada drift, shock, dan seasonal component
- dibatasi oleh batas bawah/atas (`BOUNDS`)

Synthetic data digunakan agar workflow tetap bisa berjalan, terutama saat ada bank atau periode yang tidak tersedia pada file sumber.

---

## 7. Perhitungan RBBR dan PK

Bagian inti notebook adalah **RBBR scoring engine**.

### 7.1 Rasio yang dipakai

Rasio target utama:

- `CAR`
- `NPL_gross`
- `NPL_net`
- `ROA`
- `ROE`
- `NIM`
- `BOPO`
- `LDR`

### 7.2 Fungsi skor per rasio

Notebook membuat fungsi scoring manual:

- `score_npl()`
- `score_car()`
- `score_ldr()`
- `score_roa()`
- `score_bopo()`
- `score_nim()`

Fungsi-fungsi ini mengubah angka rasio menjadi skor ordinal 1–5 berdasarkan threshold internal notebook.

### 7.3 Rumus RBBR

Fungsi `compute_rbbr(row)` menghitung komponen:

- `risk`
- `gcg`
- `rent`
- `perm`

Lalu menggabungkannya dengan bobot:

- `risk = 0.50`
- `gcg = 0.15`
- `rent = 0.15`
- `perm = 0.20`

Secara ringkas:

```text
RBBR Score = 0.50*risk + 0.15*gcg + 0.15*rent + 0.20*perm
```

### 7.4 PK (Peringkat Komposit)

Hasil `rbbr_score` dikonversi menjadi `pk` dengan skala 1–5:

- `pk 1` = sangat baik
- `pk 2` = baik
- `pk 3` = cukup
- `pk 4` = kurang baik
- `pk 5` = buruk

Notebook juga membentuk label warna:

- `hijau`
- `kuning`
- `merah`

### 7.5 Label EWS

Notebook mendefinisikan:

- `pk_fwd6` = PK 6 bulan ke depan
- `ews_alarm` = alarm jika kondisi memburuk atau mencapai level risiko tinggi

---

## 8. Cross-Statement Derived Features

Notebook tidak hanya memakai angka mentah, tetapi juga membuat fitur turunan dari hubungan antar laporan.

### 8.1 Contoh fitur turunan

Dibuat fitur seperti:

- `loan_to_asset`
- `equity_ratio`
- `leverage_ratio`
- `dpk_total`
- `current_account_ratio`
- `saving_ratio`
- `deposit_ratio`
- `casa_ratio`
- `cost_of_fund`
- `net_interest_margin_proxy`
- `credit_cost_ratio`
- `fee_to_income`
- `komitmen_to_aset`
- `off_balance_ratio`
- `ckpn_coverage_est`

### 8.2 Tujuan fitur turunan

Fitur ini penting karena membantu model menangkap:

- kualitas intermediasi bank
- struktur pendanaan
- efisiensi biaya
- tekanan risiko aset
- hubungan antar laporan yang sering lebih informatif daripada variabel tunggal

---

## 9. Feature Engineering Lengkap

Notebook melakukan engineering lanjutan untuk seluruh variabel numerik yang layak.

### 9.1 Lag features

Dibuat lag untuk variabel utama dan variabel numerik lain, misalnya:

- `lag1`
- `lag2`
- `lag3`
- `lag6`

### 9.2 Rolling statistics

Untuk variabel penting dibuat:

- moving average `ma3`, `ma6`
- rolling std `std3`, `std6`
- rolling slope `slp3`, `slp6`

### 9.3 Momentum / perubahan

Dibuat fitur:

- `diff1`
- `diff3`
- `pct_change1`

### 9.4 Interaksi antar laporan dan makro

Contoh:

- `npl_x_bir`
- `nim_x_bir`
- `ldr_x_bir`
- `bopo_x_inf`
- `car_x_kurs`
- `credit_cost_x_npl`
- `casa_x_nim`
- `roe_decomp`

### 9.5 Calendar features

Notebook juga menambahkan:

- `month`
- `quarter`
- `is_yearend`
- `is_q1`
- `is_mid_year`
- `month_sin`
- `month_cos`
- `time_idx`

### 9.6 Encoding fitur statik

Kolom kategorikal seperti:

- `kategori_bank`
- `kepemilikan`

diubah menjadi one-hot encoding.

### 9.7 Multi-step target

Notebook membentuk target prediksi ke depan:

- `CAR_fwd1` sampai `CAR_fwd6`
- ... untuk semua rasio target
- `rbbr_score_fwd1` sampai `rbbr_score_fwd6`

Label EWS juga dibentuk dari horizon 6 bulan.

---

## 10. Augmentasi Data

Notebook memperkaya data dengan tiga strategi augmentasi:

### 10.1 Jitter AR(1)

Menambahkan noise realistis berbasis proses autoregresif orde 1 agar variasi data lebih natural.

### 10.2 Stress scenario

Simulasi kondisi memburuk dengan pola seperti:

- `NPL_gross` naik
- `CAR` turun
- `ROA` turun
- `BOPO` naik
- `impairment` naik
- `kredit_diberikan` menurun
- `laba_bersih` menurun

### 10.3 Recovery scenario

Simulasi pemulihan setelah tekanan risiko terjadi.

### 10.4 Recompute setelah augmentasi

Setelah augmentasi, notebook menghitung ulang:

- RBBR
- PK
- forward targets
- EWS label
- beberapa derived features

Hasil akhirnya disimpan sebagai:

- `data_aug.parquet`
- `data_aug.csv`

---

## 11. Forecast Makro dengan NeuralProphet

Untuk variabel makro:

- `bi_rate`
- `inflasi`
- `kurs_usd`

Notebook melatih **NeuralProphet** secara terpisah per variabel.

### 11.1 Konfigurasi model

Parameter utama yang dipakai:

- `n_forecasts=6`
- `n_lags` bervariasi per variabel
- `yearly_seasonality=True`
- `weekly_seasonality=False`
- `daily_seasonality=False`
- `n_changepoints=10`
- `epochs=300`
- `loss_func='Huber'`
- `normalize='standardize'`

### 11.2 Tujuan

Model ini menghasilkan proyeksi makro 6 bulan ke depan yang kemudian dipakai sebagai:

- exogenous variable untuk SARIMAX
- known future regressors untuk TFT
- informasi proyeksi untuk ensemble

Jika model gagal, notebook menyiapkan fallback linear drift.

---

## 12. LightGBM Regresi Multi-Target

Notebook melatih **satu model LightGBM per rasio target**.

### 12.1 Target yang dilatih

- `CAR`
- `NPL_gross`
- `NPL_net`
- `ROA`
- `ROE`
- `NIM`
- `BOPO`
- `LDR`

Setiap target diprediksi dalam bentuk `fwd1` untuk horizon 1 bulan ke depan.

### 12.2 Feature selection

Fitur yang dipakai adalah `FEAT_COLS`, dengan aturan:

- tidak mengandung identifier
- tidak mengandung target masa depan
- hanya numerik
- missing ratio masih ditoleransi jika cukup informatif

### 12.3 Hyperparameter tuning

Notebook menggunakan **Optuna TPE sampler** untuk mencari parameter terbaik.

Ruang pencarian mencakup:

- `n_estimators`
- `learning_rate`
- `max_depth`
- `num_leaves`
- `min_child_samples`
- `subsample`
- `colsample_bytree`
- `reg_alpha`
- `reg_lambda`

### 12.4 Objective function

Tujuan optimisasi adalah meminimalkan **MAE** pada validation set.

### 12.5 Training final

Setelah hyperparameter terbaik ditemukan:

- model dilatih ulang pada train+validation
- dievaluasi pada test set
- metrik yang dihitung:
  - `MAE`
  - `MAPE`
  - `RMSE`

### 12.6 Output

Model tersimpan sebagai:

- `models/lgbm_reg_models.pkl`

---

## 13. LightGBM Klasifikasi PK

Notebook juga melatih **LightGBM multiclass classifier** untuk prediksi `pk`.

### 13.1 Setup target

Target PK memiliki kelas 1–5.

Sebelum fitting, label diubah ke 0–4 karena kebutuhan LightGBM multiclass.

### 13.2 Class weighting

Untuk mengatasi ketidakseimbangan kelas, notebook menghitung:

- `class_weight = balanced`

### 13.3 Tuning dan objective

Optuna dipakai untuk meminimalkan:

```text
1 - weighted F1 score
```

### 13.4 Evaluasi

Pada test set dihitung:

- `accuracy`
- `weighted F1`
- `classification report`

### 13.5 Output

Model disimpan sebagai:

- `models/lgbm_clf.pkl`

---

## 14. TFT — Temporal Fusion Transformer

Notebook menggunakan **Temporal Fusion Transformer** untuk pemodelan multivariate multi-horizon.

### 14.1 Tujuan TFT

TFT dipakai untuk belajar dinamika panel waktu yang kompleks pada:

- rasio
- variabel neraca
- laba rugi
- administratif
- makro

### 14.2 Dataset TFT

Notebook menyiapkan `TimeSeriesDataSet` dengan:

- `group_ids=['bank']`
- `time_idx`
- static categoricals
- known future reals
- unknown reals
- `allow_missing_timesteps=True`

### 14.3 Parameter utama

- `hidden_size=64`
- `attention_head_size=4`
- `dropout=0.15`
- `hidden_continuous_size=32`
- `output_size=7`
- `loss=QuantileLoss([0.02,0.1,0.25,0.5,0.75,0.9,0.98])`
- `lstm_layers=2`
- `learning_rate=3e-3`
- `max_epochs=100`
- `gradient_clip_val=0.15`

### 14.4 Training monitoring

Notebook memakai:

- `EarlyStopping`
- `ModelCheckpoint`

### 14.5 Interpretability

Setelah training, notebook menjalankan interpretasi output TFT dan menyimpan plot attention jika berhasil.

---

## 15. SARIMAX per Rasio per Bank

Notebook melatih **SARIMAX** untuk tiap rasio dan tiap bank.

### 15.1 Prosedur

Untuk setiap bank dan rasio:

1. ambil time series historis
2. tambahkan exogenous variable:
   - `bi_rate`
   - `inflasi`
   - `kurs_usd`
3. jalankan `auto_arima`
4. fit `SARIMAX` dengan parameter terbaik
5. forecast 6 bulan ke depan

### 15.2 Forecast exogenous

Jika tersedia, proyeksi makro dari NeuralProphet dipakai sebagai exogenous future.

Jika tidak tersedia, notebook menggunakan nilai terakhir sebagai fallback.

### 15.3 Fallback

Jika fitting gagal, notebook menggunakan pendekatan linear drift sebagai cadangan.

### 15.4 Output

Hasil disimpan dalam struktur `sarimax_fc` dan kemudian dipickle.

---

## 16. Stacking Ensemble

Notebook menggabungkan beberapa model dalam fungsi `predict_ensemble(df_input, bank)`.

### 16.1 Sumber prediksi

- **LightGBM regresi** untuk rasio
- **SARIMAX** untuk tren time series
- **TFT** untuk konteks multivariate
- **LightGBM classifier** untuk PK

### 16.2 Bobot dinamis

Notebook menerapkan bobot berbeda tergantung karakteristik variabel:

- **Stable (CAR, LDR)**  
  lebih berat ke SARIMAX

- **Volatile (NPL, ROA, NIM)**  
  lebih berat ke LightGBM + TFT

- **Balanced lainnya**  
  bobot merata

### 16.3 Output ensemble

Fungsi ensemble menghasilkan:

- prediksi rasio
- prediksi `rbbr`
- prediksi `pk`

---

## 17. Prediksi 6 Bulan ke Depan

Notebook membuat proyeksi 6 bulan ke depan dengan pendekatan **smooth anchor blend**.

### 17.1 Prinsip

- nilai historis terakhir dijadikan anchor
- prediksi model tetap dipakai sebagai arah tren
- hasil akhir dismooth supaya tidak terjadi loncatan ekstrem

### 17.2 Bobot anchor

Notebook menentukan bobot historis yang makin kecil untuk horizon yang lebih jauh:

- H+1: sangat dekat ke historis
- H+6: sepenuhnya ke model

### 17.3 Tujuan

Pendekatan ini membuat grafik proyeksi:

- lebih natural
- tetap konsisten dengan arah model
- tidak kaku atau patah di titik awal forecast

---

## 18. SHAP untuk Interpretabilitas

Notebook menggunakan `shap.TreeExplainer` pada model LightGBM.

### 18.1 Fokus analisis

Model representatif yang dipilih adalah `ROA` karena dianggap informatif untuk kondisi bank.

### 18.2 Hasil yang dihasilkan

- summary plot global
- daftar top 20 fitur terpenting
- kontribusi fitur per sumber laporan:
  - Neraca
  - Laba Rugi
  - Administratif
  - Rasio
  - Makro
  - Derived/FE
- waterfall plot per bank

### 18.3 Tujuan

SHAP dipakai untuk menjawab:

- variabel apa yang paling mendorong prediksi
- apakah sinyal datang dari neraca, laba rugi, makro, atau rasio
- bagaimana faktor tertentu memengaruhi kondisi bank secara individual

---

## 19. Dashboard EWS

Notebook menyiapkan dashboard visual yang menampilkan:

- `rbbr_score`
- `CAR`
- `NPL_gross`
- `ROA`
- `NIM`
- `BOPO`
- `LDR`
- sinyal EWS 6 bulan
- neraca: kredit & aset
- laba rugi: NII & fee
- komitmen & kontinjensi
- faktor makro

### 19.1 Heatmap PK

Terdapat heatmap yang memperlihatkan:

- histori PK
- proyeksi PK 6 bulan
- pemisahan visual historis vs prediksi

### 19.2 Komparasi antar bank

Notebook juga menampilkan perbandingan strategi antar bank dengan garis historis dan garis proyeksi.

---

## 20. File dan Artefak yang Dihasilkan

Notebook menyimpan banyak artefak penting, antara lain:

### 20.1 Data processed

- `data/processed/master_panel.parquet`
- `data/processed/master_panel.csv`
- `data/processed/panel_with_derived.parquet`
- `data/processed/panel_with_derived.csv`
- `data/processed/features.parquet`
- `data/processed/features.csv`
- `data/processed/data_aug.parquet`
- `data/processed/data_aug.csv`

### 20.2 Model

- `models/np_*.pkl`
- `models/lgbm_reg_models.pkl`
- `models/lgbm_clf.pkl`
- `models/sarimax_fc.pkl`
- checkpoint TFT di folder `models/` dengan ekstensi `.ckpt`
- `models/feat_cols.pkl`

### 20.3 Output visual

- `outputs/shap/shap_summary_*.png`
- `outputs/shap/waterfall_*.png`
- `outputs/plots/tft_attention.png`
- `outputs/predictions/forecast_6m.parquet`
- laporan dan visualisasi lain di `outputs/reports`

---

## 21. Evaluasi yang Dipakai

Notebook mengevaluasi model menggunakan metrik yang sesuai dengan jenis tugas:

### 21.1 Regresi

- `MAE`
- `MAPE`
- `RMSE`

### 21.2 Klasifikasi

- `Accuracy`
- `Weighted F1`
- `Classification report`

### 21.3 Time series / probabilistic

- `val_loss`
- `QuantileLoss`
- interpretasi attention TFT

---

## 22. Hal Penting yang Perlu Diperhatikan

### 22.1 Pipeline bersifat hybrid

Notebook tidak mengandalkan satu model tunggal. Hasil akhir adalah gabungan dari beberapa pendekatan yang saling melengkapi.

### 22.2 Ada fallback logic

Jika bagian tertentu gagal, notebook tetap mencoba melanjutkan dengan pendekatan cadangan, misalnya:

- synthetic data generator
- drift forecast
- last-value fallback

### 22.3 Data bersifat panel dan time-aware

Seluruh proses menjaga konteks waktu per bank, sehingga prediksi lebih sesuai dengan dinamika historis.

### 22.4 Interpretabilitas cukup kuat

Penggunaan SHAP dan dashboard membuat model tidak hanya prediktif, tetapi juga bisa dijelaskan.

---

## 23. Kesimpulan Teknis

Notebook ini membentuk sistem prediksi kondisi keuangan bank yang lengkap, dengan karakteristik:

- **multi-source**
- **multi-model**
- **multi-horizon**
- **time-series aware**
- **interpretabel**
- **siap dipakai untuk EWS**

Kekuatan utamanya ada pada kombinasi:

1. parser universal untuk data OJK,
2. RBBR/PK sebagai target domain,
3. feature engineering lintas laporan,
4. model regresi, klasifikasi, dan forecasting yang saling melengkapi,
5. ensemble dinamis,
6. interpretabilitas SHAP dan dashboard visual.

---

## 24. Reproducibility Notes

Untuk menjalankan notebook secara konsisten:

1. Pastikan file Excel sumber berada di folder `data/raw`.
2. Pastikan Google Drive sudah ter-mount bila memakai Colab.
3. Jalankan cell setup terlebih dahulu.
4. Pastikan dependency sesuai versi notebook.
5. Jalankan cell secara berurutan karena output antar cell saling bergantung.
6. Periksa `PROJECT_PATH` agar sesuai dengan lokasi folder proyek.

---

## 25. Catatan Akhir

Dokumentasi ini diturunkan dari struktur notebook dan alur kodenya. Nilai metrik akhir yang spesifik tetap bergantung pada hasil eksekusi notebook, data yang tersedia, dan kondisi environment saat training.
