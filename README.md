
# AI-Driven Portfolio Stress Testing Platform

Aplikasi berbasis web untuk melakukan stress testing dan analisis risiko portofolio saham menggunakan simulasi Monte Carlo dan model time series.

## Fitur

- Simulasi Monte Carlo untuk stress testing portofolio
- Perhitungan Value at Risk (VaR) dan Expected Shortfall
- Peramalan deret waktu dengan model ARIMA
- Analisis berbagai skenario ekonomi
- Visualisasi interaktif hasil analisis
- Dukungan untuk saham Indonesia (IDX) dan US (NYSE/NASDAQ)

## Teknologi

- Python
- Streamlit
- Pandas
- NumPy
- Plotly
- yfinance

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

## Disclaimer

Aplikasi ini ditujukan untuk tujuan edukasi dan penelitian. Hasil analisis tidak menjamin kinerja investasi di masa depan.
