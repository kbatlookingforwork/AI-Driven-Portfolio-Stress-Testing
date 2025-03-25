![AI Driven Portfolio Stress Testing](portfolio%20stress.png)

# AI-Driven Portfolio Stress Testing Platform

Aplikasi berbasis web untuk melakukan stress testing dan analisis risiko portofolio saham menggunakan simulasi Monte Carlo dan model time series.

## Fitur

- Simulasi Monte Carlo untuk stress testing portofolio
- Perhitungan Value at Risk (VaR) dan Expected Shortfall
- Peramalan deret waktu dengan model ARIMA
- Analisis berbagai skenario ekonomi
- Visualisasi interaktif hasil analisis
- Dukungan untuk saham Indonesia (IDX) dan US (NYSE/NASDAQ)

## Cara Penggunaan

1. Input data portofolio:
   - Pilih "Portofolio Sampel" untuk mencoba dengan data contoh
   - Atau "Buat Portofolio" untuk membuat portofolio kustom

2. Konfigurasi parameter analisis:
   - Tentukan rentang data historis
   - Atur jumlah simulasi Monte Carlo
   - Pilih horizon waktu proyeksi
   - Pilih skenario ekonomi yang diinginkan

3. Jalankan analisis untuk melihat:
   - Komposisi portofolio
   - Hasil simulasi Monte Carlo
   - Metrik risiko (VaR, Expected Shortfall)
   - Peramalan nilai portofolio

## Skenario Ekonomi yang Tersedia

- Normal Market
- Market Crash
- Recession
- Inflation Surge
- Tech Bubble Burst
- Pandemic
- Devaluasi Rupiah
- Kebijakan Moneter Ketat


## Detail Metodologi 

### 1. Simulasi Monte Carlo
Simulasi Monte Carlo digunakan untuk stress testing portofolio dengan langkah-langkah:
- Menggunakan data historis return untuk mengestimasi parameter distribusi
- Mensimulasikan ribuan skenario pergerakan harga dengan random sampling
- Mempertimbangkan korelasi antar aset menggunakan matriks korelasi historis
- Menghitung distribusi return portofolio berdasarkan bobot aset
- Menghasilkan path simulasi untuk mengestimasi potensi kerugian

### 2. Perhitungan Metrik Risiko

#### Value at Risk (VaR)
- Menghitung potensi kerugian maksimum pada tingkat kepercayaan tertentu
- Menggunakan metode historis dan simulasi Monte Carlo
- Implementasi pada dua tingkat kepercayaan: 95% dan 99%
- Formula: VaR = -percentile(returns, 100 * (1 - confidence_level))

#### Expected Shortfall (ES)
- Menghitung rata-rata kerugian yang melebihi VaR
- Memberikan estimasi tail risk yang lebih konservatif
- Implementasi pada tingkat kepercayaan 95% dan 99%
- Formula: ES = mean(losses[losses > VaR])

### 3. Peramalan ARIMA
Model ARIMA (Autoregressive Integrated Moving Average) digunakan untuk peramalan dengan:
- Analisis stasioneritas data menggunakan uji ADF
- Penentuan parameter optimal (p,d,q) menggunakan kriteria AIC
- Validasi model dengan diagnostik residual
- Peramalan nilai portofolio dengan interval kepercayaan
- Rolling window prediction untuk meningkatkan akurasi

### 4. Skenario Ekonomi
Analisis dampak berbagai skenario ekonomi terhadap portofolio:

#### Parameter Penyesuaian
- Returns adjustment: Perubahan ekspektasi return
- Volatility adjustment: Perubahan tingkat volatilitas
- Correlation adjustment: Perubahan korelasi antar aset
- Drawdown adjustment: Penyesuaian potensi penurunan

#### Skenario yang Tersedia
1. Normal Market
   - Baseline tanpa penyesuaian khusus
   - Menggunakan parameter historis

2. Market Crash
   - Returns: -25%
   - Volatility: 2.5x
   - Correlation: +0.3
   - Sektor terpengaruh: Financial (-35%), Technology (-30%)

3. Recession
   - Returns: -15%
   - Volatility: 1.8x
   - Correlation: +0.2
   - Dampak bertahap pada semua sektor

4. Inflation Surge
   - Returns: -5%
   - Volatility: 1.4x
   - Energy & Materials: +5-10%
   - Consumer: -20%

5. Tech Bubble Burst
   - Technology: -40%
   - Volatility: 1.6x
   - Utilities: +5% (defensive)

6. Pandemic
   - Returns: -20%
   - Volatility: 2.2x
   - Healthcare: +15%
   - Energy: -30%
## Disclaimer

Aplikasi ini ditujukan untuk tujuan edukasi dan penelitian. Hasil analisis tidak menjamin kinerja investasi di masa depan.
