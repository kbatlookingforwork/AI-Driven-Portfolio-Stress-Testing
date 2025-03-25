import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import io
import datetime
from dateutil.relativedelta import relativedelta

from utils.data_processor import load_sample_data, validate_portfolio_data, process_portfolio_data, fetch_historical_data
from utils.monte_carlo import run_monte_carlo_simulation
from utils.risk_metrics import calculate_var, calculate_expected_shortfall
from utils.time_series import run_arima_forecast
from utils.visualization import (
    plot_portfolio_composition, 
    plot_monte_carlo_simulations, 
    plot_risk_metrics, 
    plot_time_series_forecast
)
from models.economic_scenarios import (
    ECONOMIC_SCENARIOS, 
    apply_economic_scenario
)

# Set page config
st.set_page_config(
    page_title="Portfolio Stress Testing Platform",
    page_icon="📊",
    layout="wide"
)

# Define app state
if 'portfolio_data' not in st.session_state:
    st.session_state.portfolio_data = None
if 'simulation_results' not in st.session_state:
    st.session_state.simulation_results = None
if 'risk_metrics' not in st.session_state:
    st.session_state.risk_metrics = None
if 'forecast_data' not in st.session_state:
    st.session_state.forecast_data = None

# Main title
st.title("AI-Driven Portfolio Stress Testing Platform")
st.markdown("""
    <div style="display: flex; align-items: center; gap: 10px; margin-top: 20px;">
        <p style="font-weight: bold; color: green;">Created by:</p>
        <a href="https://www.linkedin.com/in/danyyudha" target="_blank">
            <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" 
                 style="width: 20px; height: 20px;">
        </a>
        <p><b>Dany Yudha Putra Haque</b></p>
    </div>
""", unsafe_allow_html=True)
# Sidebar for inputs
with st.sidebar:
    st.header("Configuration")
    
    # Data input options
    st.subheader("Data Input")
    data_input_option = st.radio(
        "Pilih metode input data:",
        ["Buat Portofolio", "Portofolio Sampel"]
    )
    
    if data_input_option == "Portofolio Sampel":
        if st.button("Muat Portofolio Sampel"):
            st.session_state.portfolio_data = load_sample_data()
            st.success("Portofolio sampel berhasil dimuat!")
    else:  # Buat Portofolio
        if 'build_portfolio' not in st.session_state:
            st.session_state.build_portfolio = []
        
        # Saham Blue Chip Indonesia (tanpa akhiran .JK)
        indonesian_stocks = [
            # Banking & Financial
            'BBCA', 'BBRI', 'BMRI', 'BBNI', 'BJTM', 'BTPS', 'BRIS', 'BDMN', 'BNGA',
            # Telecommunication
            'TLKM', 'EXCL', 'ISAT', 'FREN',
            # Consumer Goods
            'UNVR', 'ICBP', 'INDF', 'KLBF', 'SIDO', 'MYOR', 'GGRM', 'HMSP', 'CPIN', 'JPFA',
            # Infrastructure & Construction
            'PGAS', 'JSMR', 'WIKA', 'WSKT', 'ADHI', 'PTPP',
            # Mining & Energy
            'ADRO', 'PTBA', 'ITMG', 'MEDC', 'ANTM', 'INCO', 'TINS',
            # Property & Real Estate
            'BSDE', 'CTRA', 'PWON', 'SMRA', 'LPKR',
            # Industry & Manufacturing
            'ASII', 'SRIL', 'INTP', 'SMGR', 'BRPT',
            # Technology & Others
            'GOTO', 'BUKA', 'EMTK', 'AKRA', 'MNCN'
        ]
        
        # Common US stocks - kept small for focus on Indonesian market
        us_stocks = [
            'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'NVDA'
        ]
        
        all_stocks = indonesian_stocks + us_stocks
        
        # Add new stock to portfolio
        with st.form("add_stock_form"):
            st.subheader("Tambahkan Saham")
            
            col1, col2 = st.columns(2)
            
            with col1:
                new_symbol = st.selectbox("Pilih Saham", all_stocks)
                
            with col2:
                custom_symbol = st.text_input("Atau masukkan kode saham", "")
            
            if custom_symbol:
                new_symbol = custom_symbol
            
            col1, col2 = st.columns(2)
            
            with col1:
                new_weight = st.number_input("Bobot (%)", min_value=0.0, max_value=100.0, value=10.0, step=1.0) / 100.0
            
            with col2:
                new_value = st.number_input("Nilai (Juta Rp)", min_value=0.0, value=100.0, step=10.0)
            
            submit_button = st.form_submit_button(label="Tambahkan Saham")
            
            if submit_button:
                # Add stock to the build list
                st.session_state.build_portfolio.append({
                    'Symbol': new_symbol,
                    'Weight': new_weight,
                    'Value': new_value
                })
                st.success(f"Saham {new_symbol} berhasil ditambahkan!")
        
        # Display current portfolio being built
        if st.session_state.build_portfolio:
            st.subheader("Portofolio Saat Ini")
            
            build_df = pd.DataFrame(st.session_state.build_portfolio)
            st.dataframe(build_df)
            
            total_weight = build_df['Weight'].sum()
            total_value = build_df['Value'].sum()
            
            st.write(f"Total Bobot: {total_weight:.2%} | Total Nilai: {total_value:.2f} Juta Rp")
            
            # Warning if weights don't sum to 1
            if abs(total_weight - 1.0) > 0.01:
                st.warning(f"Peringatan: Total bobot ({total_weight:.2%}) tidak sama dengan 100%. Bobot akan dinormalisasi untuk analisis.")
            
            # Create portfolio from built data
            if st.button("Buat Portofolio"):
                # Create DataFrame and process it
                portfolio_df = pd.DataFrame(st.session_state.build_portfolio)
                st.session_state.portfolio_data = process_portfolio_data(portfolio_df)
                st.success("Portofolio berhasil dibuat!")
            
            # Clear portfolio being built
            if st.button("Hapus Semua"):
                st.session_state.build_portfolio = []
                st.success("Portofolio berhasil dihapus.")
    
    # Analysis parameters
    if st.session_state.portfolio_data is not None:
        st.subheader("Parameter Analisis")
        
        # Date ranges
        end_date = datetime.date.today()
        start_date = end_date - relativedelta(years=5)
        
        date_range = st.date_input(
            "Rentang data historis",
            value=(start_date, end_date),
            help="Pilih rentang tanggal untuk analisis data historis"
        )
        
        # Monte Carlo parameters
        st.subheader("Simulasi Monte Carlo")
        num_simulations = st.slider(
            "Jumlah simulasi",
            min_value=100,
            max_value=10000,
            value=1000,
            step=100,
            help="Semakin banyak simulasi = semakin akurat tapi lebih lambat"
        )
        
        time_horizon = st.slider(
            "Horizon waktu (hari)",
            min_value=1,
            max_value=252,
            value=21,
            step=1,
            help="Horizon waktu simulasi dalam hari perdagangan (252 hari = 1 tahun)"
        )
        
        # Economic scenario selection
        st.subheader("Skenario Ekonomi")
        selected_scenario = st.selectbox(
            "Pilih skenario ekonomi",
            list(ECONOMIC_SCENARIOS.keys()),
            help="Skenario ekonomi yang telah ditentukan untuk stress testing"
        )
        
        # Run analysis button
        if st.button("Jalankan Analisis Stress Test"):
            with st.spinner("Menjalankan stress test dan analisis skenario..."):
                try:
                    # Get historical data
                    portfolio_df = st.session_state.portfolio_data
                    symbols = portfolio_df['Symbol'].tolist()
                    
                    # Handle date format
                    if isinstance(date_range, tuple) and len(date_range) == 2:
                        start_date, end_date = date_range
                    else:
                        # Fallback if date_input returns a single date
                        start_date = end_date - relativedelta(years=5)
                    
                    # Debug
                    st.write(f"Debug: Mengambil data untuk {len(symbols)} saham: {', '.join(symbols)}")
                    
                    # Get historical price data
                    try:
                        historical_data = fetch_historical_data(
                            symbols, 
                            start_date, 
                            end_date
                        )
                        st.write(f"Debug: Berhasil memperoleh data historis: {len(historical_data)} hari data")
                    except Exception as fetch_error:
                        st.error(f"Error dalam mengambil data historis: {str(fetch_error)}")
                        raise fetch_error
                    
                    # Apply economic scenario
                    scenario_params = ECONOMIC_SCENARIOS[selected_scenario]
                    adjusted_historical_data = apply_economic_scenario(
                        historical_data.copy(), 
                        scenario_params
                    )
                    
                    # Run Monte Carlo simulation
                    simulation_data = run_monte_carlo_simulation(
                        adjusted_historical_data,
                        portfolio_df,
                        num_simulations=num_simulations,
                        time_horizon=time_horizon
                    )
                    
                    # Calculate risk metrics
                    var_95 = calculate_var(simulation_data, confidence_level=0.95)
                    var_99 = calculate_var(simulation_data, confidence_level=0.99)
                    es_95 = calculate_expected_shortfall(simulation_data, confidence_level=0.95)
                    es_99 = calculate_expected_shortfall(simulation_data, confidence_level=0.99)
                    
                    risk_metrics = {
                        'VaR_95': var_95,
                        'VaR_99': var_99,
                        'ES_95': es_95,
                        'ES_99': es_99
                    }
                    
                    # Run time series forecast with ARIMA
                    try:
                        st.write("Debug: Menjalankan peramalan ARIMA...")
                        st.write(f"Debug: Data historis memiliki {len(adjusted_historical_data)} baris dan {len(adjusted_historical_data.columns)} kolom")
                        st.write(f"Debug: Portofolio memiliki {len(portfolio_df)} aset")
                        
                        forecast_data = run_arima_forecast(
                            adjusted_historical_data, 
                            portfolio_df,
                            forecast_periods=time_horizon
                        )
                        
                        st.write("Debug: Peramalan ARIMA berhasil")
                    except Exception as arima_error:
                        st.error(f"Error dalam peramalan ARIMA: {str(arima_error)}")
                        # Tetap lanjutkan meskipun ARIMA error
                        forecast_data = None
                    
                    # Store results in session state
                    st.session_state.simulation_results = simulation_data
                    st.session_state.risk_metrics = risk_metrics
                    st.session_state.forecast_data = forecast_data
                    
                    st.success("Analisis selesai!")
                    
                except Exception as e:
                    st.error(f"Terjadi kesalahan selama analisis: {str(e)}")

# Main content area
if st.session_state.portfolio_data is not None:
    st.header("Ikhtisar Portofolio")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Komposisi Portofolio")
        st.dataframe(st.session_state.portfolio_data)
    
    with col2:
        st.subheader("Alokasi Aset")
        fig = plot_portfolio_composition(st.session_state.portfolio_data)
        st.plotly_chart(fig, use_container_width=True)
    
    # Display analysis results if available
    if st.session_state.simulation_results is not None:
        st.markdown("---")
        st.header("Hasil Stress Test")
        
        # Monte Carlo simulation results
        st.subheader("Simulasi Monte Carlo")
        try:
            mc_fig = plot_monte_carlo_simulations(st.session_state.simulation_results)
            st.plotly_chart(mc_fig, use_container_width=True)
            
            # Add explanation of Monte Carlo simulation
            st.markdown("""
            ### Penjelasan Simulasi Monte Carlo
            
            Simulasi Monte Carlo yang ditampilkan di atas menunjukkan berbagai jalur yang mungkin diambil oleh nilai portofolio Anda di masa depan, berdasarkan data historis dan skenario ekonomi yang dipilih:
            
            - **Garis berwarna-warni**: Setiap garis mewakili satu simulasi pergerakan harga portofolio Anda selama periode yang ditentukan
            - **Garis median (hitam)**: Menunjukkan jalur nilai portofolio yang paling mungkin terjadi
            - **Nilai basis 100**: Semua simulasi dimulai dari nilai basis 100 untuk memudahkan interpretasi pertumbuhan relatif
            
            Simulasi ini menggunakan model statistik untuk menghasilkan ribuan kemungkinan skenario pasar berdasarkan karakteristik risiko-return portofolio Anda, termasuk:
            - Tingkat pengembalian historis
            - Volatilitas (fluktuasi harga)
            - Korelasi antar saham dalam portofolio
            - Pengaruh skenario ekonomi terhadap kelas aset yang berbeda
            """)
        except Exception as e:
            st.error(f"Terjadi kesalahan pada visualisasi simulasi Monte Carlo: {str(e)}")
            st.info("Tip: Coba jalankan analisis dengan jumlah simulasi yang berbeda atau periode waktu yang lebih pendek.")
        
        # Risk metrics
        st.subheader("Metrik Risiko")
        col1, col2 = st.columns(2)
        
        # Check if risk metrics are available
        if st.session_state.risk_metrics is not None:
            with col1:
                risk_fig = plot_risk_metrics(st.session_state.risk_metrics)
                st.plotly_chart(risk_fig, use_container_width=True)
            
            with col2:
                st.markdown("### Value at Risk (VaR)")
                st.markdown(f"**95% VaR:** {st.session_state.risk_metrics['VaR_95']:.2%} dari nilai portofolio")
                st.markdown(f"**99% VaR:** {st.session_state.risk_metrics['VaR_99']:.2%} dari nilai portofolio")
                
                st.markdown("### Expected Shortfall (ES)")
                st.markdown(f"**95% ES:** {st.session_state.risk_metrics['ES_95']:.2%} dari nilai portofolio")
                st.markdown(f"**99% ES:** {st.session_state.risk_metrics['ES_99']:.2%} dari nilai portofolio")
                
                st.markdown("""
                **Interpretasi Metrik Risiko:**
                - **Value at Risk (VaR)** menunjukkan kerugian maksimum yang diharapkan pada tingkat kepercayaan tertentu dalam kondisi pasar normal
                - **Expected Shortfall (ES)** adalah rata-rata kerugian yang diharapkan ketika kerugian melebihi nilai VaR
                - Semakin tinggi persentase, semakin besar potensi kerugian yang perlu diantisipasi dalam portofolio
                - VaR 99% berarti ada 1% kemungkinan kerugian akan melebihi nilai tersebut
                """)
        else:
            st.warning("Data metrik risiko belum tersedia. Silakan jalankan stress test terlebih dahulu.")
        
        # Time series forecast
        if st.session_state.forecast_data is not None:
            st.markdown("---")
            st.subheader("Peramalan Deret Waktu (ARIMA)")
            try:
                # Dapatkan total nilai portofolio dalam juta rupiah untuk skala grafik
                total_portfolio_value = st.session_state.portfolio_data['Value'].sum()
                
                # Gunakan total nilai portofolio sebagai parameter untuk grafik peramalan
                forecast_fig = plot_time_series_forecast(
                    st.session_state.forecast_data,
                    total_portfolio_value=total_portfolio_value
                )
                
                # Tampilkan grafik dengan ukuran besar
                st.plotly_chart(forecast_fig, use_container_width=True)
                
                # Tampilkan deskripsi skenario ekonomi yang dipilih
                from models.economic_scenarios import get_scenario_description
                
                st.markdown(f"""
                ### Skenario Ekonomi: {selected_scenario}
                
                **Deskripsi:**  
                {get_scenario_description(selected_scenario)}
                
                Skenario ekonomi ini memengaruhi peramalan dan simulasi dengan menyesuaikan tingkat pengembalian, volatilitas, dan 
                korelasi antar aset berdasarkan karakteristik skenario yang dipilih. Setiap sektor ekonomi (Teknologi, Keuangan, 
                Kesehatan, dll.) dipengaruhi secara berbeda oleh skenario ini.
                
                **Mengapa Data Historis Terlihat Berbeda pada Skenario Berbeda?**  
                Dalam stress testing, data historis aktual dimodifikasi untuk mensimulasikan bagaimana aset akan berperilaku dalam 
                skenario ekonomi tertentu. Saat Anda memilih skenario selain "Normal Market":
                
                1. **Penyesuaian Return**: Return historis disesuaikan untuk mencerminkan kondisi pasar dalam skenario tersebut
                2. **Peningkatan Volatilitas**: Fluktuasi harga diperbesar untuk menggambarkan ketidakpastian yang lebih tinggi
                3. **Perubahan Korelasi**: Hubungan antar aset dimodifikasi karena aset cenderung bergerak lebih serupa selama krisis
                4. **Dampak Sektoral**: Berbagai sektor (seperti Teknologi atau Keuangan) dipengaruhi dengan tingkat berbeda-beda
                
                Ini memungkinkan Anda melihat bagaimana portofolio Anda mungkin berperilaku jika kondisi historis tersebut terjadi 
                dalam konteks skenario ekonomi yang dipilih, bukan hanya memprediksi berdasarkan data historis normal.
                """)
                
                st.markdown("---")
                
                st.markdown(f"""
                **Interpretasi Peramalan ARIMA:**
                - Grafik menunjukkan perkiraan nilai portofolio dalam juta Rupiah dengan modal awal Rp {total_portfolio_value:.2f} juta
                - Garis biru menunjukkan nilai historis portofolio
                - Garis merah menunjukkan proyeksi nilai portofolio berdasarkan skenario ekonomi yang dipilih
                - Area yang diarsir menunjukkan interval kepercayaan peramalan (95%)
                """)
            except Exception as e:
                st.error(f"Terjadi kesalahan pada visualisasi peramalan: {str(e)}")
                st.info("Tip: Coba jalankan analisis dengan periode peramalan yang berbeda atau pilih skenario ekonomi lainnya.")
else:
    # Default welcome screen
    st.markdown("""
    ## Selamat Datang di Platform Stress Testing Portofolio berbasis AI
    
    Alat ini membantu Anda menilai dan memvisualisasikan risiko portofolio investasi di berbagai skenario ekonomi.
    
    ### Fitur:
    - Simulasi Monte Carlo untuk stress testing portofolio
    - Perhitungan Value at Risk (VaR) dan Expected Shortfall
    - Peramalan deret waktu dengan model ARIMA
    - Analisis skenario ekonomi yang berbeda
    
    ### Untuk Memulai:
    1. Pilih "Portofolio Sampel" atau "Buat Portofolio" di panel samping
    2. Atur parameter analisis
    3. Klik "Jalankan Analisis Stress Test"
    
    Untuk analisis yang lebih akurat, tambahkan lebih banyak saham ke portofolio Anda dengan bobot yang mencerminkan alokasi investasi Anda.
    """)