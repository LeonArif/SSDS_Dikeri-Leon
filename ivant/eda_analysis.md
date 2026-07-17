# Laporan Analisis EDA: SSDS Dikeri-Leon

Laporan ini menyajikan hasil pemeriksaan mendalam terhadap 4 aspek penting dari Exploratory Data Analysis (EDA) yang Anda ajukan.

---

## 1. Analisis Nilai Minimum Negatif (`tma_mdpl`)

Nilai minimum absolut `tma_mdpl` di dataset pelatihan adalah **-0.0597 mdpl**, terjadi pada stasiun **Ketonggo** pada tanggal **11 Oktober 2023 pukul 18:00**.

### Temuan Visual & Fisik
- **Curah Hujan**: Total akumulasi curah hujan (`rainfall_mm`) selama 15 hari berturut-turut sebelum kejadian nilai minimum ini adalah **hanya 0.50 mm** (hampir tidak ada hujan sama sekali).
- **Konteks Musim**: Bulan Oktober merupakan puncak musim kemarau ekstrem di Jawa, terutama diperparah oleh fenomena El Niño kuat pada tahun 2023. Oleh karena itu, penurunan TMA hingga sedikit di bawah nol mdpl di stasiun tertentu sangat masuk akal secara hidrologis sebagai kondisi kemarau ekstrem (drought) asli, bukan eror sensor mendadak.
- **Visualisasi Tren**: Tren penurunan TMA di sekitar tanggal tersebut berlangsung secara gradual (bukan lonjakan/drop instan yang biasa menandakan kegagalan sensor).

![Time Series TMA Min](file:///C:/Users/ASUS/.gemini/antigravity-ide/brain/aa75db40-6c6f-4ab9-ad88-1304cecbc404/tma_min_check.png)

---

## 2. Analisis Korelasi Sebelum vs Sesudah Demean (GroupBy `nama_pos`)

Untuk membuktikan apakah hubungan fitur-fitur cuaca dan spasial dengan TMA terdistorsi oleh elevasi absolut stasiun (confounding elevation), kami membandingkan korelasi fitur numerik dengan `tma_mdpl` sebelum dan sesudah dilakukan demeaning (transformasi nilai dikurangi rata-rata historis per pos stasiun).

### Perbandingan Top 15 Korelasi (Absolute Value)

| Fitur | Korelasi Sebelum Demean | Korelasi Sesudah Demean | Selisih Absolut | Interpretasi |
| :--- | :---: | :---: | :---: | :--- |
| **`tma_mdpl`** | 1.000000 | 1.000000 | 0.000000 | Target |
| **`surface_pressure_hpa`** | **-0.947417** | **-0.112179** | **0.835239** | Sebelum demean, korelasi sangat tinggi karena proxy dari elevasi (lapse rate tekanan udara). Setelah demean, korelasi turun drastis. |
| **`longitude`** | -0.747092 | **NaN** (3.20e-16)* | 0.747092 | Fitur statis per stasiun. Setelah demean, variansinya secara teoritis bernilai 0 sehingga korelasi bernilai NaN (tidak terdefinisi). |
| **`latitude`** | -0.565027 | **NaN** (2.47e-16)* | 0.565027 | Fitur statis per stasiun. Setelah demean, variansinya secara teoritis bernilai 0 sehingga korelasi bernilai NaN (tidak terdefinisi). |
| **`soil_moisture_7_28cm`** | 0.123752 | **0.273917** | 0.150166 | Meningkat pesat setelah demeaning. |
| **`soil_moisture_28_100cm`** | 0.126973 | **0.251360** | 0.124388 | Meningkat pesat setelah demeaning. |
| **`soil_moisture_0_7cm`** | 0.105786 | **0.250349** | 0.144563 | Meningkat pesat setelah demeaning. |
| **`dew_point_c`** | -0.121663 | **0.184403** | 0.306066 | Menjadi korelasi positif setelah demeaning. |
| **`rainfall_max_24h_mm`** | -0.025054 | **0.174170** | 0.199224 | Meningkat signifikan setelah demeaning. |

\* *Catatan Teknis: Kalkulasi numerik mentah menghasilkan angka sangat kecil (~10^-16) karena batas presisi representasi floating point IEEE 754 komputer saat melakukan `x - x.mean()`, yang secara praktis berarti 0 (variansi nol) dan secara teoretis menghasilkan NaN.*

### Kesimpulan Demean
Analisis ini memvalidasi penuh kekhawatiran Anda. Elevasi absolut adalah **confounder utama**. Sebelum demean, model rentan hanya "menghafal" posisi stasiun lewat tekanan udara dan koordinat. Setelah demean, dinamika waktu yang sesungguhnya terlihat: perubahan TMA sangat dipengaruhi oleh **kelembapan tanah (soil moisture)** di berbagai kedalaman, **curah hujan maksimum 24 jam**, dan **titik embun (dew point)**.


---

## 3. Variance Inflation Factor (VIF) & Konsistensi Stasiun

Kami menghitung VIF untuk mengukur tingkat multikolinearitas spasial stasiun:

- **`surface_pressure_hpa`**: VIF = **62,001.13**
- **`longitude`**: VIF = **60,867.72**
- **`latitude`**: VIF = **407.25**

### Temuan VIF & Stasiun
- Nilai VIF yang luar biasa tinggi (>10) membuktikan adanya redundansi informasi spasial yang sangat kuat. Model berpotensi mengalami overfitting spasial jika menggunakan koordinat dan tekanan secara bersamaan secara naif.
- **Train vs Test Stations**: Kami memverifikasi bahwa daftar 30 pos stasiun di data latih (train) **sama persis (100% identik)** dengan 30 pos stasiun di data uji (test). Tidak ada stasiun baru di test set.
- **Implikasi**: Meskipun model tidak perlu digeneralisasi ke stasiun baru (out-of-station), struktur kolinearitas ini menjelaskan mengapa model baseline non-residual sebelumnya mengalami degradasi performa yang sangat parah ketika memprediksi secara rekursif (RMSE melonjak hingga ~47.8), sedangkan model berbasis target residual (`y_train - tma_hist_mean_by_pos_hour`) berhasil menstabilkan prediksi rekursif ke RMSE ~0.45.

---

## 4. Verifikasi ACF/PACF pada DAS Berbeda

Untuk memvalidasi kesesuaian lag global `[3, 6, 9, 12, 21]`, kami memeriksa ACF (Autocorrelation Function) dan PACF (Partial Autocorrelation Function) pada 3 pos stasiun dengan karakteristik ketinggian rata-rata yang sangat berbeda:
1. **Arjowinangun - Pacitan** (TMA rata-rata rendah: ~1.11 mdpl)
2. **Kajangan** (TMA rata-rata sedang: ~84.7 mdpl)
3. **Ngadipiro** (TMA rata-rata tinggi: ~143.5 mdpl)

![ACF PACF Check](file:///C:/Users/ASUS/.gemini/antigravity-ide/brain/aa75db40-6c6f-4ab9-ad88-1304cecbc404/acf_pacf_check.png)

### Hasil Analisis ACF/PACF
- **Autokorelasi Tinggi**: Ketiga stasiun menunjukkan nilai autokorelasi (ACF) yang sangat persisten (decay sangat lambat), menandakan ketergantungan waktu yang panjang (long-term memory).
- **PACF Signifikan**: Pada grafik PACF, lag ke-1 dan ke-2 memiliki korelasi parsial yang sangat dominan di hampir seluruh stasiun. Namun, stasiun dengan karakteristik aliran cepat/DAS kecil (seperti stasiun ber-elevasi tinggi) menunjukkan pola cutoff PACF yang lebih cepat dibandingkan stasiun dengan DAS besar/aliran lambat di daerah hilir yang memiliki lag signifikansi lebih panjang.
- Penggunaan lag global `[3, 6, 9, 12, 21]` sudah cukup baik untuk menangkap tren umum, namun performa model dapat dioptimalkan lebih lanjut jika lag disesuaikan secara dinamis per pos stasiun berdasarkan karakteristik laju aliran sungai masing-masing.
