"""
OTP Management Dashboard - Executive Edition
Professional Performance Analytics for Top Management
Optimized for Speed and Clarity
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import numpy as np

# Configure page
st.set_page_config(
    page_title="Executive OTP Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Professional CSS styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        padding: 1rem 2rem;
    }
    
    .stMetric {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .stMetric label {
        color: rgba(255,255,255,0.9) !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
    }
    
    .stMetric [data-testid="metric-value"] {
        color: white !important;
        font-weight: 700 !important;
        font-size: 2rem !important;
    }
    
    .stMetric [data-testid="metric-delta"] {
        color: rgba(255,255,255,0.8) !important;
    }
    
    h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    
    h2 {
        color: #2d3748;
        font-weight: 600;
        margin-top: 2rem;
    }
    
    h3 {
        color: #4a5568;
        font-weight: 600;
    }
    
    .executive-summary {
        background: linear-gradient(135deg, #f6f8fb 0%, #e9ecef 100%);
        padding: 2rem;
        border-radius: 12px;
        border-left: 4px solid #667eea;
        margin: 2rem 0;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background: #f7fafc;
        padding: 0.5rem;
        border-radius: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        font-weight: 500;
        padding: 0.75rem 1.5rem;
        border-radius: 6px;
    }
    
    .stTabs [aria-selected="true"] {
        background: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'df_processed' not in st.session_state:
    st.session_state.df_processed = None

@st.cache_data
def process_uploaded_data(file_content, filename):
    """Process uploaded CSV/Excel file with optimized performance"""
    try:
        # Read file based on extension
        if filename.endswith('.csv'):
            df = pd.read_csv(file_content, thousands=',')
        else:
            df = pd.read_excel(file_content)
        
        # Clean and convert data types efficiently
        # Handle TOTAL CHARGES with comma thousands separator
        if 'TOTAL CHARGES' in df.columns:
            df['TOTAL CHARGES'] = pd.to_numeric(
                df['TOTAL CHARGES'].astype(str).str.replace(',', ''), 
                errors='coerce'
            ).fillna(0)
        
        # Convert dates
        date_cols = ['POD DATE/TIME', 'ACT PU', 'PICKUP DATE/TIME']
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Convert numeric columns
        numeric_cols = ['Time In Transit', 'PIECES', 'WEIGHT(KG)']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Filter for target countries only
        if 'PU CTRY' in df.columns:
            df = df[df['PU CTRY'].isin(['DE', 'IT', 'IL'])].copy()
        
        # Calculate OTP status (72 hours threshold)
        if 'Time In Transit' in df.columns:
            df['Is_On_Time'] = df['Time In Transit'] <= 72
        
        # Identify controllables (Agent, Customs, Warehouse in QC NAME)
        if 'QC NAME' in df.columns:
            df['Is_Controllable'] = df['QC NAME'].fillna('').str.contains(
                'Agent|Customs|Warehouse', case=False, regex=True
            )
        
        # Extract month from POD DATE/TIME
        if 'POD DATE/TIME' in df.columns:
            df['Delivery_Month'] = df['POD DATE/TIME'].dt.to_period('M').astype(str)
        
        # Extract month from ACT PU for pieces analysis
        if 'ACT PU' in df.columns:
            df['Pickup_Month'] = df['ACT PU'].dt.to_period('M').astype(str)
        
        return df
    
    except Exception as e:
        st.error(f"Data processing error: {str(e)}")
        return None

def calculate_kpis(df):
    """Calculate key performance indicators"""
    kpis = {}
    
    if df is None or df.empty:
        return kpis
    
    # Total metrics
    kpis['total_shipments'] = len(df)
    kpis['total_pieces'] = df['PIECES'].sum() if 'PIECES' in df.columns else 0
    kpis['total_revenue'] = df['TOTAL CHARGES'].sum() if 'TOTAL CHARGES' in df.columns else 0
    
    # OTP calculations
    if 'Is_On_Time' in df.columns:
        on_time = df['Is_On_Time'].sum()
        kpis['gross_otp'] = (on_time / len(df) * 100) if len(df) > 0 else 0
        
        # Net OTP (controllables only)
        if 'Is_Controllable' in df.columns:
            ctrl_df = df[df['Is_Controllable']]
            if len(ctrl_df) > 0:
                ctrl_on_time = ctrl_df['Is_On_Time'].sum()
                kpis['net_otp'] = (ctrl_on_time / len(ctrl_df) * 100)
                kpis['controllable_count'] = len(ctrl_df)
            else:
                kpis['net_otp'] = 0
                kpis['controllable_count'] = 0
    
    # Average transit time
    if 'Time In Transit' in df.columns:
        kpis['avg_transit'] = df['Time In Transit'].mean()
    
    return kpis

def create_executive_summary(kpis):
    """Generate executive summary with professional formatting"""
    
    gross_status = "✅ ACHIEVING TARGET" if kpis.get('gross_otp', 0) >= 95 else "⚠️ BELOW TARGET"
    net_status = "✅ ACHIEVING TARGET" if kpis.get('net_otp', 0) >= 95 else "⚠️ BELOW TARGET"
    
    summary_html = f"""
    <div class="executive-summary">
        <h2 style="margin-top: 0;">Executive Performance Summary</h2>
        <p style="font-size: 1.1rem; color: #2d3748; line-height: 1.8;">
            <strong>Reporting Period:</strong> {datetime.now().strftime('%B %Y')}<br>
            <strong>Total Shipments Analyzed:</strong> {kpis.get('total_shipments', 0):,}<br>
            <strong>Total Revenue Impact:</strong> ${kpis.get('total_revenue', 0):,.2f}<br>
            <strong>Average Transit Time:</strong> {kpis.get('avg_transit', 0):.1f} hours
        </p>
        <hr style="border: 0; border-top: 1px solid #e2e8f0; margin: 1.5rem 0;">
        <p style="font-size: 1.1rem; color: #2d3748;">
            <strong>Gross OTP Performance:</strong> {kpis.get('gross_otp', 0):.1f}% - {gross_status}<br>
            <strong>Net OTP Performance (Controllables):</strong> {kpis.get('net_otp', 0):.1f}% - {net_status}<br>
            <strong>Controllable Issues Identified:</strong> {kpis.get('controllable_count', 0)} shipments
        </p>
    </div>
    """
    
    return summary_html

def create_monthly_trend_chart(df):
    """Create professional monthly trend visualization"""
    if 'Delivery_Month' not in df.columns:
        return None
    
    # Aggregate monthly data
    monthly = df.groupby('Delivery_Month').agg({
        'Is_On_Time': ['count', 'sum'] if 'Is_On_Time' in df.columns else ['count'],
        'PIECES': 'sum' if 'PIECES' in df.columns else 'count',
        'TOTAL CHARGES': 'sum' if 'TOTAL CHARGES' in df.columns else 'count'
    }).reset_index()
    
    monthly.columns = ['Month'] + ['_'.join(col).strip() for col in monthly.columns[1:]]
    
    # Calculate OTP percentages
    if 'Is_On_Time_sum' in monthly.columns:
        monthly['Gross_OTP'] = (monthly['Is_On_Time_sum'] / monthly['Is_On_Time_count'] * 100)
    
    # Calculate Net OTP for controllables
    if 'Is_Controllable' in df.columns:
        ctrl_monthly = df[df['Is_Controllable']].groupby('Delivery_Month').agg({
            'Is_On_Time': ['count', 'sum']
        }).reset_index()
        ctrl_monthly.columns = ['Month', 'Ctrl_Count', 'Ctrl_OnTime']
        ctrl_monthly['Net_OTP'] = (ctrl_monthly['Ctrl_OnTime'] / ctrl_monthly['Ctrl_Count'] * 100)
        monthly = monthly.merge(ctrl_monthly[['Month', 'Net_OTP']], on='Month', how='left')
    
    # Sort by month
    monthly = monthly.sort_values('Month')
    
    # Create subplot figure
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            '<b>OTP Performance Trend</b>',
            '<b>Monthly Volume</b>',
            '<b>Revenue Trend</b>',
            '<b>Performance vs Target</b>'
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.15,
        specs=[[{"secondary_y": False}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "indicator"}]]
    )
    
    # 1. OTP Trend Lines
    if 'Gross_OTP' in monthly.columns:
        fig.add_trace(
            go.Scatter(
                x=monthly['Month'],
                y=monthly['Gross_OTP'],
                mode='lines+markers',
                name='Gross OTP',
                line=dict(color='#667eea', width=3),
                marker=dict(size=8)
            ),
            row=1, col=1
        )
    
    if 'Net_OTP' in monthly.columns:
        fig.add_trace(
            go.Scatter(
                x=monthly['Month'],
                y=monthly.get('Net_OTP', [95]*len(monthly)),
                mode='lines+markers',
                name='Net OTP',
                line=dict(color='#764ba2', width=3),
                marker=dict(size=8)
            ),
            row=1, col=1
        )
    
    # Add target line
    fig.add_trace(
        go.Scatter(
            x=monthly['Month'],
            y=[95] * len(monthly),
            mode='lines',
            name='Target',
            line=dict(color='red', width=2, dash='dash')
        ),
        row=1, col=1
    )
    
    # 2. Monthly Volume
    if 'PIECES_sum' in monthly.columns:
        fig.add_trace(
            go.Bar(
                x=monthly['Month'],
                y=monthly['PIECES_sum'],
                name='Pieces',
                marker_color='#48bb78',
                text=monthly['PIECES_sum'].astype(int),
                textposition='outside'
            ),
            row=1, col=2
        )
    
    # 3. Revenue Trend
    if 'TOTAL CHARGES_sum' in monthly.columns:
        fig.add_trace(
            go.Bar(
                x=monthly['Month'],
                y=monthly['TOTAL CHARGES_sum'],
                name='Revenue',
                marker_color='#ed8936',
                text=['$' + f'{v:,.0f}' for v in monthly['TOTAL CHARGES_sum']],
                textposition='outside'
            ),
            row=2, col=1
        )
    
    # 4. Current Month Gauge
    if 'Gross_OTP' in monthly.columns and len(monthly) > 0:
        current_otp = monthly.iloc[-1]['Gross_OTP']
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=current_otp,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Current OTP %"},
                delta={'reference': 95, 'increasing': {'color': "green"}},
                gauge={
                    'axis': {'range': [None, 100], 'tickwidth': 1},
                    'bar': {'color': "#667eea"},
                    'steps': [
                        {'range': [0, 80], 'color': '#fed7d7'},
                        {'range': [80, 95], 'color': '#feebc8'},
                        {'range': [95, 100], 'color': '#c6f6d5'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 95
                    }
                }
            ),
            row=2, col=2
        )
    
    # Update layout
    fig.update_layout(
        height=600,
        showlegend=True,
        title_text="<b>Monthly Performance Dashboard</b>",
        title_font_size=24,
        title_x=0.5,
        font=dict(family="Inter", size=12),
        plot_bgcolor='white',
        paper_bgcolor='#f7fafc'
    )
    
    # Update axes
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#e2e8f0')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#e2e8f0')
    
    return fig

def create_shipper_analysis(df):
    """Create shipper/account performance analysis"""
    if 'SHIPPER NAME' not in df.columns:
        return pd.DataFrame()
    
    shipper_metrics = []
    
    for shipper in df['SHIPPER NAME'].dropna().unique():
        shipper_df = df[df['SHIPPER NAME'] == shipper]
        
        metrics = {
            'Account': shipper[:50],  # Truncate long names
            'Shipments': len(shipper_df),
            'Pieces': shipper_df['PIECES'].sum() if 'PIECES' in shipper_df.columns else 0,
            'Revenue': shipper_df['TOTAL CHARGES'].sum() if 'TOTAL CHARGES' in shipper_df.columns else 0
        }
        
        if 'Is_On_Time' in shipper_df.columns:
            on_time = shipper_df['Is_On_Time'].sum()
            metrics['OTP %'] = (on_time / len(shipper_df) * 100) if len(shipper_df) > 0 else 0
        
        if 'Time In Transit' in shipper_df.columns:
            metrics['Avg Transit (hrs)'] = shipper_df['Time In Transit'].mean()
        
        shipper_metrics.append(metrics)
    
    return pd.DataFrame(shipper_metrics).sort_values('Revenue', ascending=False)

def create_country_analysis(df):
    """Create country performance comparison"""
    if 'PU CTRY' not in df.columns:
        return None
    
    country_stats = []
    
    for country in ['DE', 'IT', 'IL']:
        country_df = df[df['PU CTRY'] == country]
        if len(country_df) > 0:
            stats = {
                'Country': country,
                'Shipments': len(country_df),
                'Pieces': country_df['PIECES'].sum() if 'PIECES' in country_df.columns else 0,
                'Revenue': country_df['TOTAL CHARGES'].sum() if 'TOTAL CHARGES' in country_df.columns else 0
            }
            
            if 'Is_On_Time' in country_df.columns:
                on_time = country_df['Is_On_Time'].sum()
                stats['OTP %'] = (on_time / len(country_df) * 100)
            
            country_stats.append(stats)
    
    if not country_stats:
        return None
    
    country_df = pd.DataFrame(country_stats)
    
    # Create visualization
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('<b>OTP by Country</b>', '<b>Volume Distribution</b>', '<b>Revenue by Country</b>'),
        specs=[[{"type": "bar"}, {"type": "pie"}, {"type": "bar"}]]
    )
    
    # OTP comparison
    if 'OTP %' in country_df.columns:
        fig.add_trace(
            go.Bar(
                x=country_df['Country'],
                y=country_df['OTP %'],
                marker_color=['#667eea', '#764ba2', '#9f7aea'],
                text=[f'{v:.1f}%' for v in country_df['OTP %']],
                textposition='outside',
                name='OTP %'
            ),
            row=1, col=1
        )
    
    # Volume pie chart
    fig.add_trace(
        go.Pie(
            labels=country_df['Country'],
            values=country_df['Pieces'],
            hole=0.4,
            marker_colors=['#667eea', '#764ba2', '#9f7aea'],
            textinfo='label+percent',
            name='Pieces'
        ),
        row=1, col=2
    )
    
    # Revenue bars
    fig.add_trace(
        go.Bar(
            x=country_df['Country'],
            y=country_df['Revenue'],
            marker_color=['#48bb78', '#38a169', '#2f855a'],
            text=['$' + f'{v:,.0f}' for v in country_df['Revenue']],
            textposition='outside',
            name='Revenue'
        ),
        row=1, col=3
    )
    
    fig.update_layout(
        height=400,
        showlegend=False,
        title_text="<b>Country Performance Analysis</b>",
        title_font_size=20,
        title_x=0.5,
        font=dict(family="Inter", size=12),
        plot_bgcolor='white',
        paper_bgcolor='#f7fafc'
    )
    
    return fig

# Main Application
def main():
    # Header
    st.markdown("<h1>📊 Executive OTP Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 1.2rem; color: #718096;'>On-Time Performance Analytics for Top Management</p>", unsafe_allow_html=True)
    
    # File upload section
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        uploaded_file = st.file_uploader(
            "Upload Data File",
            type=['csv', 'xlsx', 'xls'],
            help="Upload CSV or Excel file with shipment data"
        )
        
        if uploaded_file is not None:
            if st.button("📊 Generate Report", type="primary", use_container_width=True):
                with st.spinner("Processing data..."):
                    df = process_uploaded_data(uploaded_file, uploaded_file.name)
                    if df is not None and not df.empty:
                        st.session_state.df_processed = df
                        st.session_state.data_loaded = True
                        st.success(f"✅ Successfully processed {len(df):,} shipments from DE, IT, and IL")
                    else:
                        st.error("No data found for DE, IT, or IL countries")
    
    # Main dashboard
    if st.session_state.data_loaded and st.session_state.df_processed is not None:
        df = st.session_state.df_processed
        kpis = calculate_kpis(df)
        
        # Executive Summary
        st.markdown(create_executive_summary(kpis), unsafe_allow_html=True)
        
        # Key Metrics Row
        st.markdown("<h2>Key Performance Indicators</h2>", unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            delta_gross = kpis.get('gross_otp', 0) - 95
            st.metric(
                label="GROSS OTP",
                value=f"{kpis.get('gross_otp', 0):.1f}%",
                delta=f"{delta_gross:+.1f}% vs Target"
            )
        
        with col2:
            delta_net = kpis.get('net_otp', 0) - 95
            st.metric(
                label="NET OTP (Controllables)",
                value=f"{kpis.get('net_otp', 0):.1f}%",
                delta=f"{delta_net:+.1f}% vs Target"
            )
        
        with col3:
            st.metric(
                label="TOTAL REVENUE",
                value=f"${kpis.get('total_revenue', 0):,.0f}",
                delta=f"{kpis.get('total_pieces', 0):,.0f} pieces"
            )
        
        with col4:
            st.metric(
                label="AVG TRANSIT TIME",
                value=f"{kpis.get('avg_transit', 0):.1f} hrs",
                delta=f"{kpis.get('total_shipments', 0):,} shipments"
            )
        
        # Tabbed Analysis
        st.markdown("<h2>Detailed Analytics</h2>", unsafe_allow_html=True)
        
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Trends", "🌍 Country Analysis", "👥 Account Performance", "📥 Export Data"])
        
        with tab1:
            monthly_chart = create_monthly_trend_chart(df)
            if monthly_chart:
                st.plotly_chart(monthly_chart, use_container_width=True)
            
            # Monthly breakdown table
            if 'Delivery_Month' in df.columns:
                st.markdown("<h3>Monthly Breakdown</h3>", unsafe_allow_html=True)
                
                monthly_table = df.groupby('Delivery_Month').agg({
                    'PIECES': 'sum',
                    'TOTAL CHARGES': 'sum',
                    'Is_On_Time': lambda x: (x.sum() / len(x) * 100) if 'Is_On_Time' in df.columns else 0
                }).round(2)
                
                monthly_table.columns = ['Total Pieces', 'Total Revenue ($)', 'OTP %']
                monthly_table = monthly_table.sort_index()
                
                st.dataframe(
                    monthly_table.style.format({
                        'Total Pieces': '{:,.0f}',
                        'Total Revenue ($)': '${:,.2f}',
                        'OTP %': '{:.1f}%'
                    }),
                    use_container_width=True
                )
        
        with tab2:
            country_chart = create_country_analysis(df)
            if country_chart:
                st.plotly_chart(country_chart, use_container_width=True)
            
            # Country metrics table
            st.markdown("<h3>Country Metrics</h3>", unsafe_allow_html=True)
            
            country_table = df.groupby('PU CTRY').agg({
                'REFER': 'count',
                'PIECES': 'sum',
                'TOTAL CHARGES': 'sum',
                'Is_On_Time': lambda x: (x.sum() / len(x) * 100) if 'Is_On_Time' in df.columns else 0
            }).round(2)
            
            country_table.columns = ['Shipments', 'Total Pieces', 'Total Revenue ($)', 'OTP %']
            country_table = country_table.sort_values('Total Revenue ($)', ascending=False)
            
            st.dataframe(
                country_table.style.format({
                    'Shipments': '{:,.0f}',
                    'Total Pieces': '{:,.0f}',
                    'Total Revenue ($)': '${:,.2f}',
                    'OTP %': '{:.1f}%'
                }),
                use_container_width=True
            )
        
        with tab3:
            st.markdown("<h3>Account Performance Analysis</h3>", unsafe_allow_html=True)
            
            shipper_df = create_shipper_analysis(df)
            
            if not shipper_df.empty:
                # Top performers
                st.markdown("**Top 10 Accounts by Revenue**")
                
                top_shippers = shipper_df.head(10)
                
                # Format the dataframe
                display_df = top_shippers.copy()
                display_df['Revenue'] = display_df['Revenue'].apply(lambda x: f'${x:,.2f}')
                display_df['Pieces'] = display_df['Pieces'].apply(lambda x: f'{x:,.0f}')
                if 'OTP %' in display_df.columns:
                    display_df['OTP %'] = display_df['OTP %'].apply(lambda x: f'{x:.1f}%')
                if 'Avg Transit (hrs)' in display_df.columns:
                    display_df['Avg Transit (hrs)'] = display_df['Avg Transit (hrs)'].apply(lambda x: f'{x:.1f}')
                
                st.dataframe(display_df, use_container_width=True, hide_index=True)
                
                # Performance distribution
                col1, col2 = st.columns(2)
                
                with col1:
                    if 'OTP %' in shipper_df.columns:
                        fig_otp = go.Figure(data=[
                            go.Histogram(
                                x=shipper_df['OTP %'],
                                nbinsx=20,
                                marker_color='#667eea',
                                opacity=0.8
                            )
                        ])
                        fig_otp.update_layout(
                            title="<b>OTP Distribution Across Accounts</b>",
                            xaxis_title="OTP %",
                            yaxis_title="Number of Accounts",
                            height=300,
                            showlegend=False,
                            plot_bgcolor='white',
                            paper_bgcolor='#f7fafc'
                        )
                        fig_otp.add_vline(x=95, line_dash="dash", line_color="red", 
                                         annotation_text="Target")
                        st.plotly_chart(fig_otp, use_container_width=True)
                
                with col2:
                    # Revenue concentration
                    top5_revenue = shipper_df.head(5)['Revenue'].sum()
                    total_revenue = shipper_df['Revenue'].sum()
                    revenue_concentration = (top5_revenue / total_revenue * 100) if total_revenue > 0 else 0
                    
                    fig_conc = go.Figure(data=[
                        go.Pie(
                            labels=['Top 5 Accounts', 'Others'],
                            values=[top5_revenue, total_revenue - top5_revenue],
                            hole=0.5,
                            marker_colors=['#667eea', '#e2e8f0']
                        )
                    ])
                    fig_conc.update_layout(
                        title="<b>Revenue Concentration</b>",
                        height=300,
                        showlegend=True,
                        plot_bgcolor='white',
                        paper_bgcolor='#f7fafc',
                        annotations=[dict(text=f'{revenue_concentration:.0f}%', x=0.5, y=0.5, 
                                        font_size=20, showarrow=False)]
                    )
                    st.plotly_chart(fig_conc, use_container_width=True)
        
        with tab4:
            st.markdown("<h3>Export Options</h3>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📊 Export Processed Data**")
                
                # Prepare export data
                export_df = df.copy()
                
                # Select columns for export
                export_columns = ['REFER', 'SHIPPER NAME', 'PU CTRY', 'DEL CTRY', 
                                 'POD DATE/TIME', 'Time In Transit', 'Is_On_Time',
                                 'QC NAME', 'Is_Controllable', 'PIECES', 'TOTAL CHARGES']
                
                available_cols = [col for col in export_columns if col in export_df.columns]
                export_df = export_df[available_cols]
                
                # Download as CSV
                csv = export_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download as CSV",
                    data=csv,
                    file_name=f"otp_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col2:
                st.markdown("**📈 Export Executive Report**")
                
                # Create summary report
                report = f"""
OTP EXECUTIVE REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

EXECUTIVE SUMMARY
================
Total Shipments: {kpis.get('total_shipments', 0):,}
Total Revenue: ${kpis.get('total_revenue', 0):,.2f}
Total Pieces: {kpis.get('total_pieces', 0):,.0f}

PERFORMANCE METRICS
==================
Gross OTP: {kpis.get('gross_otp', 0):.1f}% (Target: 95%)
Net OTP (Controllables): {kpis.get('net_otp', 0):.1f}% (Target: 95%)
Average Transit Time: {kpis.get('avg_transit', 0):.1f} hours
Controllable Issues: {kpis.get('controllable_count', 0)} shipments

COUNTRY BREAKDOWN
================
"""
                if 'PU CTRY' in df.columns:
                    for country in ['DE', 'IT', 'IL']:
                        country_data = df[df['PU CTRY'] == country]
                        if len(country_data) > 0:
                            country_otp = (country_data['Is_On_Time'].sum() / len(country_data) * 100) if 'Is_On_Time' in country_data.columns else 0
                            report += f"{country}: {len(country_data)} shipments, OTP: {country_otp:.1f}%\n"
                
                st.download_button(
                    label="📄 Download Executive Report",
                    data=report,
                    file_name=f"executive_report_{datetime.now().strftime('%Y%m%d')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            
            # Data quality summary
            st.markdown("**📋 Data Quality Summary**")
            
            quality_metrics = {
                'Total Records Processed': len(df),
                'Date Range': f"{df['POD DATE/TIME'].min().strftime('%Y-%m-%d')} to {df['POD DATE/TIME'].max().strftime('%Y-%m-%d')}" if 'POD DATE/TIME' in df.columns else 'N/A',
                'Countries Included': ', '.join(df['PU CTRY'].unique()) if 'PU CTRY' in df.columns else 'N/A',
                'Unique Shippers': df['SHIPPER NAME'].nunique() if 'SHIPPER NAME' in df.columns else 0,
                'Records with QC Issues': df['QC NAME'].notna().sum() if 'QC NAME' in df.columns else 0,
                'Controllable Issues': df['Is_Controllable'].sum() if 'Is_Controllable' in df.columns else 0
            }
            
            quality_df = pd.DataFrame(list(quality_metrics.items()), columns=['Metric', 'Value'])
            st.dataframe(quality_df, use_container_width=True, hide_index=True)
    
    else:
        # Welcome screen when no data is loaded
        st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 3rem; border-radius: 12px; color: white; margin: 2rem 0;'>
            <h2 style='color: white; margin-top: 0;'>Welcome to the Executive OTP Dashboard</h2>
            <p style='font-size: 1.1rem; line-height: 1.8;'>
                This professional dashboard provides comprehensive On-Time Performance analytics 
                for executive decision-making. Upload your shipment data to begin analysis.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div style='background: #f7fafc; padding: 1.5rem; border-radius: 8px; height: 200px;'>
                <h3>📊 Gross OTP</h3>
                <p>Overall on-time performance across all shipments with 95% target benchmark</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style='background: #f7fafc; padding: 1.5rem; border-radius: 8px; height: 200px;'>
                <h3>🎯 Net OTP</h3>
                <p>Performance for controllable factors (Agent, Customs, Warehouse)</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div style='background: #f7fafc; padding: 1.5rem; border-radius: 8px; height: 200px;'>
                <h3>📈 Analytics</h3>
                <p>Monthly trends, country analysis, and account performance metrics</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style='background: #edf2f7; padding: 2rem; border-radius: 8px; margin-top: 2rem;'>
            <h3>📋 Data Requirements</h3>
            <p><strong>Required Columns:</strong></p>
            <ul>
                <li>POD DATE/TIME - Actual delivery date</li>
                <li>ACT PU - Actual pickup date</li>
                <li>PU CTRY - Pickup country (DE, IT, IL)</li>
                <li>Time In Transit - Transit hours</li>
                <li>QC NAME - Quality control category</li>
                <li>PIECES - Number of pieces</li>
                <li>TOTAL CHARGES - Revenue amount</li>
                <li>SHIPPER NAME - Account identifier</li>
            </ul>
            <p><strong>Supported Formats:</strong> CSV, Excel (.xlsx, .xls)</p>
            <p><strong>Target Countries:</strong> Germany (DE), Italy (IT), Israel (IL)</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
