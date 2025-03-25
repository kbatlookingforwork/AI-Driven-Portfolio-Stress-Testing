import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List

def plot_portfolio_composition(portfolio_data: pd.DataFrame):
    """
    Create a pie chart visualization of portfolio composition.
    
    Args:
        portfolio_data: DataFrame with portfolio data
        
    Returns:
        Figure: Plotly figure object with portfolio composition
    """
    # Create a color palette for various sectors
    color_map = {
        # Blues for Financial
        'BBCA': '#1f77b4', 'BBRI': '#2a9df4', 'BMRI': '#6baed6', 'BBNI': '#9ecae1', 'BJTM': '#c6dbef',
        'BTPS': '#d0e0f3', 'BRIS': '#deebf7', 'BDMN': '#e8f4f9', 'BNGA': '#f7fbff',
        
        # Greens for Consumer
        'UNVR': '#2ca02c', 'ICBP': '#4daf4a', 'INDF': '#66bd63', 'KLBF': '#88d27a', 'SIDO': '#a6e08f',
        'MYOR': '#c2e699', 'GGRM': '#d9f0d3', 'HMSP': '#e6f5d0', 'CPIN': '#f7fcf5', 'JPFA': '#e5f5e0',
        
        # Purples for Technology/Telecom
        'TLKM': '#9467bd', 'EXCL': '#b279a2', 'ISAT': '#c994c7', 'FREN': '#df65b0', 'GOTO': '#e7298a',
        'BUKA': '#e6a0c4', 'EMTK': '#f4bfdb', 'AKRA': '#fde0ef', 'MNCN': '#fff7fc',
        
        # Oranges for Energy/Mining
        'ADRO': '#ff7f0e', 'PTBA': '#fd8d3c', 'ITMG': '#fdae6b', 'MEDC': '#fdd0a2', 'ANTM': '#fee6ce',
        'INCO': '#fff5eb', 'TINS': '#ffedd5',
        
        # Reds for Infrastructure/Property
        'SMGR': '#d62728', 'WIKA': '#e7474b', 'WSKT': '#f16d71', 'PTPP': '#fc8e93', 'ADHI': '#fcb5b9',
        'BSDE': '#fccec2', 'CTRA': '#fde5d9', 'PWON': '#fee9e5', 'SMRA': '#fff5f0', 'LPKR': '#fff0ed',
        
        # Other colors for other sectors
        'ASII': '#8c564b', 'SRIL': '#c49c94', 'INTP': '#e377c2', 'BRPT': '#f7b6d2', 'PGAS': '#7f7f7f',
        'JSMR': '#bcbd22'
    }
    
    # Create a list of colors based on symbols in portfolio
    colors = []
    for symbol in portfolio_data['Symbol']:
        if symbol in color_map:
            colors.append(color_map[symbol])
        else:
            # Default color for symbols not in map
            colors.append('#17becf')
    
    # Create a more visually appealing pie chart
    fig = px.pie(
        portfolio_data,
        values='Value',
        names='Symbol',
        title='Alokasi Portofolio berdasarkan Nilai Investasi',
        color_discrete_sequence=colors if colors else px.colors.qualitative.Plotly,
        hole=0.4,  # Create a donut chart
        hover_data=['Weight', 'Percentage']
    )
    
    fig.update_traces(
        textposition='inside', 
        textinfo='percent+label',
        hovertemplate='<b>%{label}</b><br>Nilai: Rp %{value} juta<br>Bobot: %{customdata[0]:.1%}<br>Persentase: %{customdata[1]:.1f}%',
        marker=dict(line=dict(color='#FFFFFF', width=1))
    )
    
    fig.update_layout(
        title_font=dict(size=20),
        font=dict(size=14),
        legend=dict(orientation='h', yanchor='bottom', y=-0.15),
        margin=dict(l=30, r=30, t=80, b=30),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def plot_monte_carlo_simulations(simulation_results: Dict):
    """
    Create a visualization of Monte Carlo simulation results.
    
    Args:
        simulation_results: Dict containing simulation data
        
    Returns:
        Figure: Plotly figure object with simulation paths
    """
    simulations = simulation_results['simulations']
    time_horizon = simulation_results['time_horizon']
    
    # Create figure
    fig = go.Figure()
    
    # Time points for x-axis
    time_points = np.arange(time_horizon)
    
    # Add simulation paths (randomly select 100 paths to avoid clutter)
    num_paths_to_show = min(100, len(simulations))
    indices = np.random.choice(len(simulations), num_paths_to_show, replace=False)
    
    for i in indices:
        fig.add_trace(
            go.Scatter(
                x=time_points,
                y=simulations[i],
                mode='lines',
                line=dict(width=0.5, color='rgba(0, 100, 180, 0.1)'),
                showlegend=False,
                hoverinfo='skip'
            )
        )
    
    # Calculate percentiles for each time point
    percentiles = {}
    for percentile in [5, 25, 50, 75, 95]:
        percentiles[percentile] = np.percentile(simulations, percentile, axis=0)
    
    # Add percentile lines
    percentile_colors = {
        5: 'rgba(255, 0, 0, 0.8)',   # Red
        25: 'rgba(255, 165, 0, 0.8)', # Orange
        50: 'rgba(0, 0, 0, 0.8)',     # Black
        75: 'rgba(0, 128, 0, 0.8)',   # Green
        95: 'rgba(0, 0, 255, 0.8)'    # Blue
    }
    
    percentile_names = {
        5: 'Persentil ke-5',
        25: 'Persentil ke-25',
        50: 'Median',
        75: 'Persentil ke-75',
        95: 'Persentil ke-95'
    }
    
    for percentile, values in percentiles.items():
        fig.add_trace(
            go.Scatter(
                x=time_points,
                y=values,
                mode='lines',
                line=dict(width=2, color=percentile_colors[percentile]),
                name=percentile_names[percentile],
                hovertemplate='Hari: %{x}<br>' + percentile_names[percentile] + ': %{y:.2%}'
            )
        )
    
    # Update layout
    fig.update_layout(
        title='Hasil Simulasi Monte Carlo',
        xaxis_title='Hari Perdagangan',
        yaxis_title='Return Kumulatif',
        yaxis_tickformat='.1%',
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        font=dict(size=14),
        margin=dict(l=30, r=30, t=80, b=30)
    )
    
    # Add zero line
    fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="black")
    
    return fig

def plot_risk_metrics(risk_metrics: Dict):
    """
    Create a visualization of risk metrics.
    
    Args:
        risk_metrics: Dict containing risk metrics
        
    Returns:
        Figure: Plotly figure object with risk metrics
    """
    # Extract metrics and prepare for plotting
    metrics_to_plot = ['VaR_95', 'VaR_99', 'ES_95', 'ES_99']
    values = [risk_metrics[metric] for metric in metrics_to_plot]
    labels = ['VaR (95%)', 'VaR (99%)', 'Expected Shortfall (95%)', 'Expected Shortfall (99%)']
    
    # Terjemahan untuk tooltip dan label
    tooltips = [
        'Value at Risk dengan tingkat kepercayaan 95%',
        'Value at Risk dengan tingkat kepercayaan 99%',
        'Expected Shortfall dengan tingkat kepercayaan 95%',
        'Expected Shortfall dengan tingkat kepercayaan 99%'
    ]
    
    # Create bar chart with hover information
    fig = go.Figure(data=[
        go.Bar(
            x=labels,
            y=values,
            text=[f"{v:.2%}" for v in values],
            textposition='auto',
            marker_color=['rgba(255, 99, 132, 0.7)', 'rgba(255, 49, 89, 0.9)', 
                          'rgba(255, 159, 64, 0.7)', 'rgba(255, 120, 24, 0.9)'],
            hovertemplate='%{x}<br>Nilai: %{y:.2%}<br><b>%{text}</b><extra></extra>',
            customdata=tooltips
        )
    ])
    
    # Update layout
    fig.update_layout(
        title='Metrik Risiko Portofolio',
        xaxis_title='Metrik',
        yaxis_title='Nilai (dalam % dari portofolio)',
        yaxis_tickformat='.1%',
        yaxis_range=[0, max(values) * 1.2],  # Add some padding at the top
        font=dict(size=14),
        margin=dict(l=30, r=30, t=80, b=150),  # Increased bottom margin for annotation
        xaxis=dict(
            tickangle=0,  # Horizontal labels
            tickfont=dict(size=11)
        )
    )
    
    # Ubah label sumbu x agar lebih singkat
    fig.update_xaxes(
        ticktext=['VaR (95%)', 'VaR (99%)', 'ES (95%)', 'ES (99%)'],
        tickvals=[0, 1, 2, 3]
    )
    
    # Tambahkan anotasi penjelasan pada dua baris, lebih ke bawah di bawah tulisan "metrik"
    fig.add_annotation(
        x=0.5,
        y=-0.4,
        xref="paper",
        yref="paper",
        text="Semakin tinggi nilai metrik risiko,",
        showarrow=False,
        font=dict(size=12),
        align="center",
    )
    
    fig.add_annotation(
        x=0.5,
        y=-0.45,
        xref="paper",
        yref="paper",
        text="semakin tinggi potensi kerugian portofolio",
        showarrow=False,
        font=dict(size=12),
        align="center",
    )
    
    return fig

def plot_time_series_forecast(forecast_data: Dict, total_portfolio_value: float = None):
    """
    Create a visualization of time series forecast.
    
    Args:
        forecast_data: Dict containing forecast data
        total_portfolio_value: Total portfolio value in million Rupiah (optional)
        
    Returns:
        Figure: Plotly figure object with forecast
    """
    try:
        # Extract data
        historical_values = forecast_data['historical_values']
        forecast_values = forecast_data['forecast_values']
        lower_ci = forecast_data['lower_ci']
        upper_ci = forecast_data['upper_ci']
        
        # Get total portfolio value in millions for scaling
        # If not provided, default to normalized values (basis 100)
        scale_in_millions = False
        if total_portfolio_value is not None and total_portfolio_value > 0:
            scale_in_millions = True
            base_value = total_portfolio_value
            # Don't normalize to 100, use actual value in millions
            if abs(historical_values.iloc[0] - 100) < 1:  # If already normalized to 100
                # Scale values by the total portfolio value
                scaling_factor = base_value / 100
                historical_values = historical_values * scaling_factor
                forecast_values = forecast_values * scaling_factor
                lower_ci = lower_ci * scaling_factor
                upper_ci = upper_ci * scaling_factor
        else:
            # Default case: normalize to 100 for percentage representation
            if abs(historical_values.iloc[0] - 100) > 1:  # If not already normalized
                print("Normalizing historical and forecast values to start at 100")
                norm_factor = 100 / historical_values.iloc[0]
                historical_values = historical_values * norm_factor
                forecast_values = forecast_values * norm_factor
                lower_ci = lower_ci * norm_factor
                upper_ci = upper_ci * norm_factor
        
        # Create larger figure with specific height and width
        fig = go.Figure(layout=dict(
            autosize=False,
            width=900,
            height=600
        ))
        
        # Add historical values with better formatting for millions
        hovertemplate = 'Tanggal: %{x|%Y-%m-%d}<br>Nilai: Rp %{y:.2f} juta' if scale_in_millions else 'Tanggal: %{x|%Y-%m-%d}<br>Nilai: %{y:.2f}'
        
        fig.add_trace(
            go.Scatter(
                x=historical_values.index,
                y=historical_values.values,
                mode='lines',
                name='Historis',
                line=dict(color='blue', width=2.5),
                hovertemplate=hovertemplate
            )
        )
        
        # Add forecast with better formatting
        forecast_hover = 'Tanggal: %{x|%Y-%m-%d}<br>Peramalan: Rp %{y:.2f} juta' if scale_in_millions else 'Tanggal: %{x|%Y-%m-%d}<br>Peramalan: %{y:.2f}'
        
        fig.add_trace(
            go.Scatter(
                x=forecast_values.index,
                y=forecast_values.values,
                mode='lines',
                name='Peramalan',
                line=dict(color='red', width=2.5),
                hovertemplate=forecast_hover
            )
        )
        
        # Add confidence interval as shaded area with better error handling
        try:
            # Convert indices to list to handle different index types safely
            x_values = list(forecast_values.index) + list(forecast_values.index)[::-1]
            y_values = list(upper_ci.values) + list(lower_ci.values)[::-1]
            
            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=y_values,
                    fill='toself',
                    fillcolor='rgba(255, 0, 0, 0.2)',
                    line=dict(color='rgba(255, 0, 0, 0)'),
                    name='Interval Kepercayaan 95%',
                    hoverinfo='skip'
                )
            )
        except Exception as ci_error:
            print(f"Error adding confidence interval as area: {str(ci_error)}")
            # Fall back to separate lines for confidence intervals
            ci_hover = 'Tanggal: %{x|%Y-%m-%d}<br>Batas Bawah: Rp %{y:.2f} juta' if scale_in_millions else 'Tanggal: %{x|%Y-%m-%d}<br>Batas Bawah: %{y:.2f}'
            
            fig.add_trace(
                go.Scatter(
                    x=lower_ci.index,
                    y=lower_ci.values,
                    mode='lines',
                    line=dict(color='rgba(255, 0, 0, 0.5)', width=1, dash='dash'),
                    name='Batas Bawah (95%)',
                    hovertemplate=ci_hover
                )
            )
            
            ci_hover_upper = 'Tanggal: %{x|%Y-%m-%d}<br>Batas Atas: Rp %{y:.2f} juta' if scale_in_millions else 'Tanggal: %{x|%Y-%m-%d}<br>Batas Atas: %{y:.2f}'
            
            fig.add_trace(
                go.Scatter(
                    x=upper_ci.index,
                    y=upper_ci.values,
                    mode='lines',
                    line=dict(color='rgba(255, 0, 0, 0.5)', width=1, dash='dash'),
                    name='Batas Atas (95%)',
                    hovertemplate=ci_hover_upper
                )
            )
        
        # Prepare title and axis labels based on scale
        y_axis_title = 'Nilai Portofolio (dalam juta Rupiah)' if scale_in_millions else 'Nilai Portofolio (basis 100)'
        
        # Update layout with improved Indonesian titles and formatting
        fig.update_layout(
            title={
                'text': 'Peramalan Deret Waktu Portfolio',
                'font': {'size': 20}
            },
            xaxis_title='Tanggal',
            yaxis_title=y_axis_title,
            hovermode='x unified',
            xaxis=dict(
                tickformat='%b %Y',  # Mempersingkat format tanggal
                tickangle=-30,       # Mengurangi sudut putar
                tickfont=dict(size=11),
                nticks=10            # Membatasi jumlah label
            ),
            yaxis=dict(
                tickformat=',.2f',   # Format dengan ribuan separator
                gridcolor='rgba(220, 220, 220, 0.5)',
                tickfont=dict(size=11)
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(size=12)
            ),
            font=dict(size=14),
            margin=dict(l=50, r=50, t=100, b=150),  # Memperbesar margin untuk akomodasi judul dan anotasi
            plot_bgcolor='rgba(250, 250, 250, 0.5)'
        )
        
        # Perbarui anotasi penjelasan sesuai dengan skala nilai
        annotation_text1 = f'Nilai menggunakan skala juta Rupiah dengan modal awal Rp {historical_values.iloc[0]:.2f} juta.' if scale_in_millions else 'Nilai dimulai dari basis 100 pada awal periode.'
        
        # Tambahkan anotasi penjelasan pada dua baris
        fig.add_annotation(
            xref='paper', yref='paper',
            x=0.5, y=-0.27,
            text=annotation_text1,
            showarrow=False,
            font=dict(size=12),
            align='center'
        )
        
        fig.add_annotation(
            xref='paper', yref='paper',
            x=0.5, y=-0.32,
            text='Interval kepercayaan menunjukkan rentang kemungkinan dengan tingkat kepercayaan 95%.',
            showarrow=False,
            font=dict(size=12),
            align='center'
        )
        
        return fig
        
    except Exception as e:
        # Create a minimal chart showing the error
        fig = go.Figure()
        
        # Add error text
        fig.add_annotation(
            x=0.5,
            y=0.5,
            text=f"Terjadi kesalahan saat membuat visualisasi forecast: {str(e)}",
            font=dict(size=14, color="red"),
            showarrow=False,
            xref="paper",
            yref="paper"
        )
        
        fig.update_layout(
            title='Peramalan Deret Waktu (Error)',
            xaxis_title='Tanggal',
            yaxis_title='Nilai',
            font=dict(size=14),
            margin=dict(l=30, r=30, t=80, b=30)
        )
        
        return fig