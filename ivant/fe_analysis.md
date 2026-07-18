# Laporan Analisis Feature Engineering: SSDS Dikeri-Leon

Laporan ini menyajikan hasil pemeriksaan mendalam terhadap 4 aspek Feature Engineering (FE) yang Anda ajukan (poin 5-8).

---

## 5. Konfirmasi Urutan Prediksi Rekursif Hulu & Hilir (Bug `upstream_tma_mean`)

Kami menganalisis bagaimana fitur `upstream_tma_mean` diproses dalam pelatihan dan inferensi rekursif.

### Temuan Penting
- **Ada Bug Kritis**: 
  - Fitur `upstream_tma_mean` dibuat pada tahap awal dengan melakukan pivot data TMA dari `train_df` dan menggabungkannya kembali hanya ke `train_df`.
  - Fitur ini **tidak pernah digabungkan** ke dalam data lingkungan `env_df` maupun data uji `test_df`.
  - Akibatnya, pada data uji (`test_full`), kolom `upstream_tma_mean` **bernilai `NaN` seluruhnya** ketika reindex dilakukan untuk mencocokkan kolom fitur model (`final_col_order`).
  - Selain itu, kode prediksi rekursif saat ini (`recursive_predict_test` dan `recursive_walk_forward_validation`) **tidak melakukan kalkulasi ulang secara dinamis** untuk `upstream_tma_mean` di setiap timestamp `dt` berdasarkan nilai TMA stasiun hulu yang baru diprediksi.
- **Dampak pada Model**:
  - Model tree-based (LGBM, XGBoost, CatBoost) tidak mengalami crash karena kemampuannya menangani nilai `NaN` secara otomatis. Namun, model kehilangan sinyal penting dari fitur hulu ini pada saat melakukan prediksi sesungguhnya.

---

## 6. Granulitas `MAIN_RIV` pada 30 Pos

Kami memetakan stasiun pemantauan ke ID sungai utama (`MAIN_RIV`) menggunakan database HydroRIVERS:

| MAIN_RIV | Jumlah Pos | Daftar Pos Stasiun |
| :--- | :---: | :--- |
| **`50387677`** | **25** | Ketonggo, Sekayu, Cepu, Karanggeneng, Bojonegoro - Kali Kethek, Badegan, Wonogiri Dam, Kajangan, Kali Anyar - Kreteg Abang, Colo Weir, Ngadipiro, Ngrembang, Karangnongko, Kali Pepe - Tugu Boto, Kedungupit, Brangkal, Kali Pepe - PTPN, Babat, Napel, Peren, Jurug, Jarum, Sumberrejo, Floodway Bridge C, Serenan |
| **`50392385`** | **2** | Boboh Kali Lamong, Bengkelolor |
| **`50419297`** | **2** | Gunungsari, Arjowinangun - Pacitan |
| **`50419937`** | **1** | Lorog |

### Temuan Tributary (Anak Sungai)
- **Granularitas Rendah**: ID `50387677` merepresentasikan sistem sungai utama **Bengawan Solo**. Karena 25 dari 30 stasiun berbagi ID sungai yang sama, pemetaan hubungan hulu-hilir berdasarkan jarak ke muara (`DIST_DN_KM`) menjadi bias.
- **Tributary Terpisah**: Beberapa stasiun seperti Kali Pepe (tributary) dan stasiun di Bengawan Solo utama diidentifikasi berada di sungai yang sama, sehingga stasiun di anak sungai dianggap "upstream" bagi stasiun utama, meskipun karakteristik laju alirannya tidak terhubung langsung secara linier.

---

## 7. Validitas Data Lingkungan di Periode Uji (Test Period)

Kami menganalisis nilai fitur-fitur di `data_lingkungan.csv` sepanjang periode uji (`2025-09-19` hingga `2026-05-18`):

- **Variabilitas Fitur**: Seluruh fitur cuaca dan tanah memiliki nilai yang sangat bervariasi dengan standar deviasi yang wajar (misalnya: `std_dev` untuk `rainfall_mm` = 1.25, `temperature_c` = 2.64, dan `soil_moisture` = 0.08).
- **Jumlah Nilai Unik**: Setiap fitur numerik memiliki ratusan nilai unik sepanjang periode uji (misalnya: `rainfall_mm` memiliki 205 nilai unik, `temperature_c` memiliki 166 nilai unik).
- **Kesimpulan**: Data lingkungan di periode uji **valid dan berisi data observasi/forecasting asli**, bukan nilai placeholder statis maupun carry-forward konstan. Fitur-fitur ini aman digunakan untuk pemodelan.

---

## 8. Duplikasi Kolom Koordinat (`latitude`/`longitude`)

Kami memeriksa ulang temuan Anda mengenai duplikasi kolom koordinat di `train_full`.

### Konfirmasi Temuan (Bug Terverifikasi)
- **Temuan Anda 100% Benar**: Pada notebook versi asli, kolom koordinat stasiun memang terduplikasi menjadi `latitude_x`, `latitude_y`, `longitude_x`, dan `longitude_y`.
- **Akar Masalah**:
  - Di Section 3.3, koordinat dari `koordinat_pos.csv` dimasukkan ke `train_df` dan `test_df` dengan nama kolom `latitude` and `longitude`.
  - Di Section 3.1, koordinat juga digabungkan ke `env_df`.
  - Pada sel penyiapan fitur, pengecualian kolom ditulis sebagai berikut:
    `env_feature_cols = [c for c in env_df.columns if c not in ['datetime', 'nama_pos', 'landcover_name', 'lat', 'lon']]`
    Karena kolom asli bernama `latitude` dan `longitude` (bukan `lat` dan `lon`), baris pengecualian ini **gagal mendeteksi dan mengeksklusi** koordinat dari data lingkungan. Akibatnya, saat penggabungan akhir (`train_full`), kolom koordinat dari `train_df` dan `env_df` bertabrakan sehingga pandas menambahkan akhiran `_x` dan `_y`.

### Perbaikan yang Telah Dilakukan
Kami telah menerapkan perbaikan langsung pada berkas [notebook.ipynb](file:///c:/Users/ASUS/Desktop/SSDS_Dikeri-Leon/ivant/notebook.ipynb) pada sel **34**, **62**, dan **80** dengan mengganti pengecualian `'lat', 'lon'` menjadi `'latitude', 'longitude'`:
```python
env_feature_cols = [
    c for c in env_df.columns 
    if c not in ['datetime', 'nama_pos', 'landcover_name', 'latitude', 'longitude']
]
```

### Hasil Verifikasi Pasca-Perbaikan
Kami menjalankan verifikasi ulang pada dataset pasca-perbaikan dan mengonfirmasi:
- Kolom berakhiran `_x` dan `_y` **berhasil dibersihkan sepenuhnya** dari data pelatihan (`train_full`).
- Kolom spasial stasiun kini direpresentasikan secara bersih oleh kolom tunggal `latitude` dan `longitude`.
