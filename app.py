"""
Professional OTP Management Dashboard
For Top Management Reporting
Version 1.0
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import io
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="OTP Management Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-left: 4px solid #0066cc;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .header-style {
        background: linear-gradient(90deg, #0066cc 0%, #004499 100%);
        padding: 2rem;
        border-radius: 0.5rem;
        color: white;
        margin-bottom: 2rem;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None

def load_and_process_data(uploaded_file) -> pd.DataFrame:
    """Load and process the Excel data with proper date handling and filtering"""
    try:
        # Read Excel file
        df = pd.read_excel(uploaded_file, engine='openpyxl')
        
        # Convert date columns to datetime
        date_columns = ['ORD CREATE', 'READY', 'QT PU', 'ACT PU', 'READY_1', 
                       'QT PU_1', 'PICKUP DATE/TIME', 'Depart Date / Time',
                       'Arrive Date / Time', 'QDT', 'UPD DEL', 'POD DATE/TIME']
        
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Extract month from POD DATE/TIME (actual delivery date)
        df['Delivery_Month'] = pd.to_datetime(df['POD DATE/TIME']).dt.to_period('M')
        df['Delivery_Month_Str'] = df['Delivery_Month'].astype(str)
        
        # Filter for specified countries (DE, IT, IL)
        target_countries = ['DE', 'IT', 'IL']
        df_filtered = df[df['PU CTRY'].isin(target_countries)].copy()
        
        # Clean numeric columns
        df_filtered['PIECES'] = pd.to_numeric(df_filtered['PIECES'], errors='coerce').fillna(0)
        df_filtered['TOTAL CHARGES'] = pd.to_numeric(df_filtered['TOTAL CHARGES'], errors='coerce').fillna(0)
        df_filtered['Time In Transit'] = pd.to_numeric(df_filtered['Time In Transit'], errors='coerce').fillna(0)
        
        # Calculate OTP status
        df_filtered['OTP_Objective'] = 95  # 95% objective
        df_filtered['Is_On_Time'] = df_filtered['Time In Transit'] <= 72  # Assuming 72 hours as on-time threshold
        
        # Identify controllable QC categories
        controllable_keywords = ['Agent', 'Customs', 'Warehouse']
        df_filtered['Is_Controllable'] = df_filtered['QC NAME'].apply(
            lambda x: any(keyword in str(x) for keyword in controllable_keywords) if pd.notna(x) else False
        )
        
        return df_filtered
    
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        return None

def calculate_otp_metrics(df: pd.DataFrame) -> Dict:
    """Calculate OTP Gross and Net metrics"""
    metrics = {}
    
    # Overall metrics
    total_shipments = len(df)
    on_time_shipments = df['Is_On_Time'].sum()
    
    # Gross OTP (all shipments)
    metrics['gross_otp'] = (on_time_shipments / total_shipments * 100) if total_shipments > 0 else 0
    
    # Net OTP (only controllable)
    controllable_df = df[df['Is_Controllable']]
    controllable_total = len(controllable_df)
    controllable_on_time = controllable_df['Is_On_Time'].sum()
    metrics['net_otp'] = (controllable_on_time / controllable_total * 100) if controllable_total > 0 else 0
    
    # Additional metrics
    metrics['total_shipments'] = total_shipments
    metrics['on_time_shipments'] = on_time_shipments
    metrics['controllable_shipments'] = controllable_total
    metrics['total_pieces'] = df['PIECES'].sum()
    metrics['total_charges'] = df['TOTAL CHARGES'].sum()
    metrics['avg_transit_time'] = df['Time In Transit'].mean()
    
    return metrics

def create_monthly_trend_chart(df: pd.DataFrame) -> go.Figure:
    """Create monthly OTP trend chart"""
    monthly_data = []
    
    for month in df['Delivery_Month_Str'].unique():
        if pd.notna(month):
            month_df = df[df['Delivery_Month_Str'] == month]
            
            # Calculate monthly OTP metrics
            total = len(month_df)
            on_time = month_df['Is_On_Time'].sum()
            gross_otp = (on_time / total * 100) if total > 0 else 0
            
            # Net OTP for controllables
            controllable_df = month_df[month_df['Is_Controllable']]
            ctrl_total = len(controllable_df)
            ctrl_on_time = controllable_df['Is_On_Time'].sum()
            net_otp = (ctrl_on_time / ctrl_total * 100) if ctrl_total > 0 else 0
            
            monthly_data.append({
                'Month': month,
                'Gross OTP': gross_otp,
                'Net OTP': net_otp,
                'Objective': 95,
                'Total Shipments': total,
                'Total Pieces': month_df['PIECES'].sum(),
                'Total Charges': month_df['TOTAL CHARGES'].sum()
            })
    
    monthly_df = pd.DataFrame(monthly_data).sort_values('Month')
    
    # Create figure with secondary y-axis
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('OTP Performance Trend', 'Volume & Revenue Trend'),
        vertical_spacing=0.15,
        specs=[[{"secondary_y": False}],
               [{"secondary_y": True}]]
    )
    
    # OTP Trend
    fig.add_trace(
        go.Scatter(x=monthly_df['Month'], y=monthly_df['Gross OTP'],
                  name='Gross OTP', mode='lines+markers',
                  line=dict(color='#0066cc', width=3),
                  marker=dict(size=10)),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=monthly_df['Month'], y=monthly_df['Net OTP'],
                  name='Net OTP (Controllables)', mode='lines+markers',
                  line=dict(color='#00aa44', width=3),
                  marker=dict(size=10)),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=monthly_df['Month'], y=monthly_df['Objective'],
                  name='Objective (95%)', mode='lines',
                  line=dict(color='red', width=2, dash='dash')),
        row=1, col=1
    )
    
    # Volume & Revenue Trend
    fig.add_trace(
        go.Bar(x=monthly_df['Month'], y=monthly_df['Total Pieces'],
               name='Total Pieces', marker_color='lightblue'),
        row=2, col=1, secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(x=monthly_df['Month'], y=monthly_df['Total Charges'],
                  name='Total Charges ($)', mode='lines+markers',
                  line=dict(color='orange', width=2),
                  marker=dict(size=8)),
        row=2, col=1, secondary_y=True
    )
    
    # Update layout
    fig.update_layout(
        height=700,
        showlegend=True,
        title_text="Monthly Performance Dashboard",
        title_font_size=20,
        hovermode='x unified',
        template='plotly_white'
    )
    
    fig.update_yaxes(title_text="OTP %", row=1, col=1, range=[0, 105])
    fig.update_yaxes(title_text="Pieces", row=2, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Revenue ($)", row=2, col=1, secondary_y=True)
    fig.update_xaxes(title_text="Month", row=2, col=1)
    
    return fig

def create_country_analysis(df: pd.DataFrame) -> go.Figure:
    """Create country-wise performance analysis"""
    country_metrics = []
    
    for country in df['PU CTRY'].unique():
        if pd.notna(country):
            country_df = df[df['PU CTRY'] == country]
            
            total = len(country_df)
            on_time = country_df['Is_On_Time'].sum()
            otp = (on_time / total * 100) if total > 0 else 0
            
            country_metrics.append({
                'Country': country,
                'OTP %': otp,
                'Total Shipments': total,
                'Total Pieces': country_df['PIECES'].sum(),
                'Total Charges': country_df['TOTAL CHARGES'].sum(),
                'Avg Transit Time': country_df['Time In Transit'].mean()
            })
    
    country_df = pd.DataFrame(country_metrics)
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('OTP by Country', 'Volume by Country', 
                       'Revenue by Country', 'Transit Time by Country'),
        specs=[[{"type": "bar"}, {"type": "pie"}],
               [{"type": "bar"}, {"type": "bar"}]]
    )
    
    # OTP by Country
    fig.add_trace(
        go.Bar(x=country_df['Country'], y=country_df['OTP %'],
               name='OTP %', marker_color='#0066cc',
               text=country_df['OTP %'].round(1),
               textposition='outside'),
        row=1, col=1
    )
    
    # Volume pie chart
    fig.add_trace(
        go.Pie(labels=country_df['Country'], values=country_df['Total Pieces'],
               name='Pieces Distribution'),
        row=1, col=2
    )
    
    # Revenue by Country
    fig.add_trace(
        go.Bar(x=country_df['Country'], y=country_df['Total Charges'],
               name='Revenue', marker_color='green',
               text=country_df['Total Charges'].round(0),
               textposition='outside'),
        row=2, col=1
    )
    
    # Transit Time by Country
    fig.add_trace(
        go.Bar(x=country_df['Country'], y=country_df['Avg Transit Time'],
               name='Avg Transit Time', marker_color='orange',
               text=country_df['Avg Transit Time'].round(1),
               textposition='outside'),
        row=2, col=2
    )
    
    fig.update_layout(height=700, showlegend=False, title_text="Country Performance Analysis",
                     title_font_size=20, template='plotly_white')
    
    return fig

def create_shipper_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Create shipper-wise analysis table"""
    shipper_metrics = []
    
    for shipper in df['SHIPPER NAME'].unique():
        if pd.notna(shipper):
            shipper_df = df[df['SHIPPER NAME'] == shipper]
            
            total = len(shipper_df)
            on_time = shipper_df['Is_On_Time'].sum()
            otp = (on_time / total * 100) if total > 0 else 0
            
            shipper_metrics.append({
                'Shipper': shipper[:50],  # Truncate long names
                'Total Shipments': total,
                'On-Time': on_time,
                'OTP %': round(otp, 1),
                'Total Pieces': int(shipper_df['PIECES'].sum()),
                'Total Charges ($)': round(shipper_df['TOTAL CHARGES'].sum(), 2),
                'Avg Transit (hrs)': round(shipper_df['Time In Transit'].mean(), 1)
            })
    
    return pd.DataFrame(shipper_metrics).sort_values('Total Shipments', ascending=False)

def generate_executive_summary(df: pd.DataFrame, metrics: Dict) -> str:
    """Generate executive summary text"""
    summary = f"""
    ## Executive Summary
    
    ### Performance Overview
    - **Gross OTP Performance**: {metrics['gross_otp']:.1f}% (Target: 95%)
    - **Net OTP Performance (Controllables)**: {metrics['net_otp']:.1f}% (Target: 95%)
    - **Total Shipments Analyzed**: {metrics['total_shipments']:,}
    - **On-Time Deliveries**: {metrics['on_time_shipments']:,}
    
    ### Operational Metrics
    - **Total Pieces Handled**: {metrics['total_pieces']:,.0f}
    - **Total Revenue Generated**: ${metrics['total_charges']:,.2f}
    - **Average Transit Time**: {metrics['avg_transit_time']:.1f} hours
    
    ### Key Insights
    - Gross OTP is {'meeting' if metrics['gross_otp'] >= 95 else 'below'} the 95% objective
    - Controllable factors account for {metrics['controllable_shipments']} shipments
    - {'Immediate action required on controllable factors' if metrics['net_otp'] < 95 else 'Controllable factors performing well'}
    """
    return summary

# Main Application
def main():
    # Header
    st.markdown("""
        <div class="header-style">
            <h1 style='text-align: center; color: white; margin-bottom: 0;'>
                📊 OTP Management Dashboard
            </h1>
            <p style='text-align: center; color: #f0f0f0; font-size: 18px;'>
                Professional Performance Reporting for Top Management
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for file upload
    with st.sidebar:
        st.markdown("### 📁 Data Upload")
        uploaded_file = st.file_uploader(
            "Upload Excel File",
            type=['xlsx', 'xls'],
            help="Upload the shipment data Excel file"
        )
        
        if uploaded_file is not None:
            if st.button("🔄 Process Data", type="primary"):
                with st.spinner("Processing data..."):
                    df = load_and_process_data(uploaded_file)
                    if df is not None:
                        st.session_state.processed_data = df
                        st.session_state.data_loaded = True
                        st.success("✅ Data processed successfully!")
        
        # Filters
        if st.session_state.data_loaded:
            st.markdown("### 🔍 Filters")
            df = st.session_state.processed_data
            
            # Date range filter
            if 'POD DATE/TIME' in df.columns:
                min_date = df['POD DATE/TIME'].min()
                max_date = df['POD DATE/TIME'].max()
                
                date_range = st.date_input(
                    "Select Date Range",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date
                )
            
            # Country filter
            countries = st.multiselect(
                "Select Countries",
                options=df['PU CTRY'].unique().tolist(),
                default=df['PU CTRY'].unique().tolist()
            )
            
            # Apply filters button
            if st.button("Apply Filters"):
                # Filter data based on selections
                filtered_df = df[df['PU CTRY'].isin(countries)]
                if len(date_range) == 2:
                    filtered_df = filtered_df[
                        (filtered_df['POD DATE/TIME'].dt.date >= date_range[0]) &
                        (filtered_df['POD DATE/TIME'].dt.date <= date_range[1])
                    ]
                st.session_state.processed_data = filtered_df
                st.rerun()
    
    # Main content area
    if st.session_state.data_loaded and st.session_state.processed_data is not None:
        df = st.session_state.processed_data
        
        # Calculate metrics
        metrics = calculate_otp_metrics(df)
        
        # Executive Summary
        st.markdown(generate_executive_summary(df, metrics))
        
        # Key Metrics Cards
        st.markdown("### 📈 Key Performance Indicators")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Gross OTP",
                value=f"{metrics['gross_otp']:.1f}%",
                delta=f"{metrics['gross_otp'] - 95:.1f}% vs Target",
                delta_color="normal" if metrics['gross_otp'] >= 95 else "inverse"
            )
        
        with col2:
            st.metric(
                label="Net OTP (Controllables)",
                value=f"{metrics['net_otp']:.1f}%",
                delta=f"{metrics['net_otp'] - 95:.1f}% vs Target",
                delta_color="normal" if metrics['net_otp'] >= 95 else "inverse"
            )
        
        with col3:
            st.metric(
                label="Total Shipments",
                value=f"{metrics['total_shipments']:,}",
                delta=f"{metrics['on_time_shipments']:,} On-Time"
            )
        
        with col4:
            st.metric(
                label="Total Revenue",
                value=f"${metrics['total_charges']:,.0f}",
                delta=f"{metrics['total_pieces']:.0f} Pieces"
            )
        
        # Tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Trends", "🌍 Country Analysis", "📦 Shipper Analysis", "📋 Detailed Data"])
        
        with tab1:
            st.markdown("### Monthly Performance Trends")
            fig_trend = create_monthly_trend_chart(df)
            st.plotly_chart(fig_trend, use_container_width=True)
        
        with tab2:
            st.markdown("### Country-wise Performance")
            fig_country = create_country_analysis(df)
            st.plotly_chart(fig_country, use_container_width=True)
        
        with tab3:
            st.markdown("### Shipper Performance Analysis")
            shipper_df = create_shipper_analysis(df)
            
            # Display as formatted table
            st.dataframe(
                shipper_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "OTP %": st.column_config.ProgressColumn(
                        "OTP %",
                        help="On-Time Performance Percentage",
                        format="%.1f%%",
                        min_value=0,
                        max_value=100,
                    ),
                    "Total Charges ($)": st.column_config.NumberColumn(
                        "Total Charges ($)",
                        help="Total charges in USD",
                        format="$%.2f",
                    ),
                }
            )
            
            # Download button for shipper analysis
            csv = shipper_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Shipper Analysis",
                data=csv,
                file_name=f"shipper_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
                mime='text/csv'
            )
        
        with tab4:
            st.markdown("### Detailed Shipment Data")
            
            # Display options
            col1, col2 = st.columns(2)
            with col1:
                show_controllables = st.checkbox("Show only controllables", value=False)
            with col2:
                show_delays = st.checkbox("Show only delays", value=False)
            
            # Filter detailed data
            detailed_df = df.copy()
            if show_controllables:
                detailed_df = detailed_df[detailed_df['Is_Controllable']]
            if show_delays:
                detailed_df = detailed_df[~detailed_df['Is_On_Time']]
            
            # Select columns to display
            display_columns = ['REFER', 'SHIPPER NAME', 'PU CTRY', 'DEL CTRY', 
                              'POD DATE/TIME', 'Time In Transit', 'Is_On_Time',
                              'QC NAME', 'Is_Controllable', 'PIECES', 'TOTAL CHARGES']
            
            available_columns = [col for col in display_columns if col in detailed_df.columns]
            
            st.dataframe(
                detailed_df[available_columns],
                use_container_width=True,
                hide_index=True
            )
            
            # Download full dataset
            csv_full = detailed_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Full Dataset",
                data=csv_full,
                file_name=f"otp_detailed_data_{datetime.now().strftime('%Y%m%d')}.csv",
                mime='text/csv'
            )
    
    else:
        # Instructions when no data is loaded
        st.info("""
        ### 📌 Getting Started
        
        1. **Upload your Excel file** using the sidebar
        2. **Click "Process Data"** to analyze the shipment data
        3. **Apply filters** as needed to focus on specific time periods or countries
        4. **Navigate through tabs** to explore different analytical views
        
        ### 📊 Dashboard Features:
        - **Gross OTP**: Overall on-time performance across all shipments
        - **Net OTP**: Performance for controllable factors (Agent, Customs, Warehouse)
        - **Monthly Trends**: Track performance over time
        - **Country Analysis**: Compare performance across DE, IT, and IL
        - **Shipper Analysis**: Detailed breakdown by account/shipper
        
        ### 📋 Data Requirements:
        Your Excel file should contain columns for:
        - POD DATE/TIME (actual delivery date)
        - PU CTRY (pickup country)
        - QC NAME (quality control categorization)
        - PIECES, TOTAL CHARGES, Time In Transit
        - SHIPPER NAME (for account analysis)
        """)
        
        # Sample data structure
        with st.expander("📝 View Expected Data Format"):
            sample_data = {
                'REFER': ['CI 220889', 'CI 221024'],
                'SHIPPER NAME': ['COMPANY A', 'COMPANY B'],
                'PU CTRY': ['DE', 'IT'],
                'DEL CTRY': ['US', 'GB'],
                'POD DATE/TIME': ['2025-07-10', '2025-07-22'],
                'Time In Transit': [33, 29],
                'QC NAME': ['Agent-Delay', 'Airline-FLT delay'],
                'PIECES': [4, 5],
                'TOTAL CHARGES': [1500.00, 2107.79]
            }
            st.dataframe(pd.DataFrame(sample_data))

if __name__ == "__main__":
    main()
