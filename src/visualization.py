"""
Visualization Module
Plotly chart generators untuk dashboard
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

from .config import RBBRConfig

# ========== DASHBOARD CHARTS ==========

def plot_heatmap_pk(df: pd.DataFrame, title="Heatmap Kondisi Bank", forecast_start_date=None):
    """Professional banking heatmap with proper layout"""
    
    # Find bank column
    bank_col = 'bank_label' if 'bank_label' in df.columns else ('nama_bank' if 'nama_bank' in df.columns else next((c for c in ['bank', 'kode_bank'] if c in df.columns), None))
    if not bank_col:
        fig = go.Figure()
        fig.add_annotation(text="Data tidak tersedia", showarrow=False, font=dict(size=16))
        return fig

    # Clean and prepare data
    plot_df = df.dropna(subset=[bank_col, 'periode', 'pk_prediksi']).copy()
    if plot_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="Tidak ada data", showarrow=False, font=dict(size=16))
        return fig

    plot_df['pk_prediksi'] = plot_df['pk_prediksi'].astype(int)
    
    # Create pivot table
    pivot = plot_df.pivot_table(index=bank_col, columns='periode', values='pk_prediksi', aggfunc='first', observed=False)
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)
    
    # Format labels
    x_labels = [d.strftime('%b %Y') for d in pivot.columns]
    y_labels = list(pivot.index)
    
    num_banks = len(y_labels)
    num_periods = len(x_labels)
    
    # Calculate proper dimensions
    row_height = 30
    chart_height = max(500, min(900, num_banks * row_height + 150))
    
    # Create discrete colorscale
    pk_colors = RBBRConfig.PK_COLORS
    colorscale = [
        [0.0, pk_colors[1]], [0.2, pk_colors[1]],
        [0.2, pk_colors[2]], [0.4, pk_colors[2]],
        [0.4, pk_colors[3]], [0.6, pk_colors[3]],
        [0.6, pk_colors[4]], [0.8, pk_colors[4]],
        [0.8, pk_colors[5]], [1.0, pk_colors[5]]
    ]
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=x_labels,
        y=y_labels,
        colorscale=colorscale,
        zmin=1,
        zmax=5,
        xgap=1,
        ygap=1,
        hovertemplate='<b>%{y}</b><br>%{x}<br>PK: %{z}<extra></extra>',
        colorbar=dict(
            title=dict(text='<b>Peringkat<br>Komposit</b>', font=dict(size=12)),
            tickvals=[1, 2, 3, 4, 5],
            ticktext=['1 - Sangat Sehat', '2 - Sehat', '3 - Cukup Sehat', '4 - Kurang Sehat', '5 - Tidak Sehat'],
            thickness=20,
            len=0.6,
            x=1.0,
            xpad=10
        )
    ))
    
    # Add PK numbers in cells
    for i, bank in enumerate(y_labels):
        for j, period in enumerate(x_labels):
            val = pivot.values[i, j]
            if not np.isnan(val):
                fig.add_annotation(
                    x=period, y=bank,
                    text=str(int(val)),
                    showarrow=False,
                    font=dict(size=10, color='white', family='Arial'),
                    xref='x', yref='y'
                )
    
    # Add forecast boundary line
    if forecast_start_date:
        forecast_start_date = pd.to_datetime(forecast_start_date)
        try:
            forecast_idx = next(i for i, d in enumerate(pivot.columns) if d >= forecast_start_date)
            if forecast_idx > 0:
                boundary_x = x_labels[forecast_idx]
                fig.add_vline(
                    x=boundary_x,
                    line=dict(color='#3b82f6', width=3, dash='dash')
                )
                fig.add_annotation(
                    x=boundary_x,
                    y=1.02,
                    text='<b>FORECAST</b>',
                    showarrow=False,
                    xref='x',
                    yref='paper',
                    font=dict(size=11, color='#1e40af'),
                    bgcolor='rgba(219, 234, 254, 0.95)',
                    bordercolor='#3b82f6',
                    borderwidth=2,
                    borderpad=6
                )
        except StopIteration:
            pass
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f'<b>{title}</b>',
            x=0.5,
            xanchor='center',
            font=dict(size=20, color='#1f2937')
        ),
        xaxis=dict(
            side='bottom',
            tickangle=-45,
            tickfont=dict(size=10),
            showgrid=False
        ),
        yaxis=dict(
            tickfont=dict(size=10),
            showgrid=False
        ),
        height=chart_height,
        margin=dict(l=150, r=150, t=80, b=100),
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(family='Arial, sans-serif')
    )
    
    return fig

def plot_pk_distribution(df: pd.DataFrame):
    """Bar chart of PK distribution"""

    pk_counts = df['pk_prediksi'].value_counts().sort_index()

    colors = [RBBRConfig.PK_COLORS.get(pk, '#6c757d') for pk in pk_counts.index]

    fig = go.Figure(go.Bar(
        x=[f'PK {pk}' for pk in pk_counts.index],
        y=pk_counts.values,
        marker_color=colors,
        text=pk_counts.values,
        textposition='auto',
    ))

    fig.update_layout(
        title='Distribusi Peringkat Komposit',
        xaxis_title='Peringkat',
        yaxis_title='Jumlah Bank',
        showlegend=False,
    )

    return fig

def plot_radar_rbbr(bank_data: pd.Series):
    """Radar chart for 4 RBBR pillars"""

    categories = ['Risk Profile', 'GCG', 'Rentabilitas', 'Permodalan']

    values = [
        bank_data.get('score_risk_profile', 2.5),
        bank_data.get('score_gcg', 2.0),
        bank_data.get('score_rentabilitas', 2.5),
        bank_data.get('score_permodalan', 2.5),
    ]

    # Close the radar
    values += values[:1]
    categories += categories[:1]

    bank_name = bank_data.get('nama_bank', bank_data.get('bank', 'Bank'))
    fig = go.Figure(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name=bank_name,
        line_color='#003d82',
        fillcolor='rgba(0, 61, 130, 0.3)',
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 5],
                tickvals=[1, 2, 3, 4, 5],
            )
        ),
        showlegend=False,
        title='RBBR 4 Pilar Assessment',
    )

    return fig

def plot_trend_forecast(historical: pd.DataFrame, forecast: pd.DataFrame, metric='CAR'):
    """Time series with forecast and transition line"""

    fig = go.Figure()

    # Historical data
    fig.add_trace(go.Scatter(
        x=historical['periode'],
        y=historical[metric],
        name='Aktual',
        mode='lines+markers',
        line=dict(color='#003d82', width=3),
        marker=dict(size=8)
    ))

    # Forecast
    if not forecast.empty:
        # Boundary Line (Vertical)
        forecast_start = forecast['periode'].min()
        fig.add_vline(
            x=forecast_start,
            line_width=2,
            line_dash="dash",
            line_color="#666"
        )

        fig.add_trace(go.Scatter(
            x=forecast['periode'],
            y=forecast[f'{metric}_pred'],
            name='Prediksi',
            mode='lines+markers',
            line=dict(color='#d62828', width=3, dash='dash'),
            marker=dict(size=8, symbol='diamond')
        ))

        # Confidence interval
        if f'{metric}_lower' in forecast.columns:
            fig.add_trace(go.Scatter(
                x=pd.concat([forecast['periode'], forecast['periode'][::-1]]),
                y=pd.concat([forecast[f'{metric}_upper'], forecast[f'{metric}_lower'][::-1]]),
                fill='toself',
                fillcolor='rgba(214, 40, 40, 0.1)',
                line=dict(color='rgba(255,255,255,0)'),
                name='Confidence Interval',
                showlegend=True,
            ))

    fig.update_layout(
        title=f'Tren & Proyeksi {metric}',
        xaxis_title='Periode',
        yaxis_title=metric,
        hovermode='x unified',
        template='plotly_white',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig