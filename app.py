"""
Professional OTP Management Dashboard - Complete Version
Supports both Excel and CSV files with full functionality
Version 3.0
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
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
if 'df_processed' not in st.session_state:
    st.session_state.df_processed = None
if 'df_filtered' not in st.session_state:
    st.session_state.df_filtered = None

@st.cache_data(show_spinner=False)
def load_file(file_content, file_name) -> pd.DataFrame:
    """Load Excel or CSV file with robust error handling"""
    try:
        if file_name.endswith('.csv'):
            # Read CSV
            df = pd.read_csv(io.BytesIO(file_content))
        else:
            # Try multiple methods for Excel
            try:
                df = pd.read_excel(io.BytesIO(file_content), engine='openpyxl')
            except:
                try:
                    df = pd.read_excel(io.BytesIO(file_content), engine='xlrd')
                except:
                    df = pd.read_excel(io.BytesIO(file_content))
        
        return df
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

@st.cache_data(show_spinner=False)
def process_data(df: pd.DataFrame) -> pd.DataFrame:
    """Process the data with OTP calculations"""
    try:
        # Convert date columns
        date_columns = ['POD DATE/TIME', 'ORD CREATE', 'READY', 'QT PU', 'ACT PU', 
                       'PICKUP DATE/TIME', 'Depart Date / Time', 'Arrive Date / Time']
        
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Extract month from POD DATE/TIME (actual delivery date)
        if 'POD DATE/TIME' in df.columns:
            df['Delivery_Month'] = pd.to_datetime(df['POD DATE/TIME']).dt.to_period('M')
            df['Delivery_Month_Str'] = df['Delivery_Month'].astype(str)
        
        # Clean numeric columns
        numeric_columns = ['PIECES', 'TOTAL CHARGES', 'Time In Transit', 'WEIGHT(KG)', 'TOT DST']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Calculate OTP status (72 hours threshold)
        if 'Time In Transit' in df.columns:
            df['Is_On_Time'] = df['Time In Transit'] <= 72
        
        # Identify controllable QC categories
        if 'QC NAME' in df.columns:
            controllable_keywords = ['Agent', 'Customs', 'Warehouse']
            df['Is_Controllable'] = df['QC NAME'].apply(
                lambda x: any(keyword in str(x) for keyword in controllable_keywords) if pd.notna(x) else False
            )
        
        # Filter for target countries (DE, IT, IL)
        if 'PU CTRY' in df.columns:
            target_countries = ['DE', 'IT', 'IL']
            df = df[df['PU CTRY'].isin(target_countries)].copy()
        
        return df
    
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        return df

@st.cache_data(show_spinner=False)
def calculate_otp_metrics(df: pd.DataFrame) -> Dict:
    """Calculate comprehensive OTP metrics"""
    metrics = {}
    
    if 'Is_On_Time' in df.columns:
        # Overall metrics
        total_shipments = len(df)
        on_time_shipments = df['Is_On_Time'].sum()
        
        # Gross OTP (all shipments)
        metrics['gross_otp'] = (on_time_shipments / total_shipments * 100) if total_shipments > 0 else 0
        
        # Net OTP (only controllable)
        if 'Is_Controllable' in df.columns:
            controllable_df = df[df['Is_Controllable']]
            controllable_total = len(controllable_df)
            controllable_on_time = controllable_df['Is_On_Time'].sum() if len(controllable_df) > 0 else 0
            metrics['net_otp'] = (controllable_on_time / controllable_total * 100) if controllable_total > 0 else 0
            metrics['controllable_shipments'] = controllable_total
        else:
            metrics['net_otp'] = 0
            metrics['controllable_shipments'] = 0
        
        metrics['total_shipments'] = total_shipments
        metrics['on_time_shipments'] = on_time_shipments
    else:
        metrics['gross_otp'] = 0
        metrics['net_otp'] = 0
        metrics['total_shipments'] = len(df)
        metrics['on_time_shipments'] = 0
        metrics['controllable_shipments'] = 0
    
    # Additional metrics
    metrics['total_pieces'] = df['PIECES'].sum() if 'PIECES' in df.columns else 0
    metrics['total_charges'] = df['TOTAL CHARGES'].sum() if 'TOTAL CHARGES' in df.columns else 0
    metrics['avg_transit_time'] = df['Time In Transit'].mean() if 'Time In Transit' in df.columns else 0
    
    return metrics

@st.cache_data(show_spinner=False)
def create_monthly_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Create monthly OTP and volume analysis"""
    if 'Delivery_Month_Str' not in df.columns:
        return pd.DataFrame()
    
    monthly_data = []
    
    for month in df['Delivery_Month_Str'].unique():
        if pd.notna(month):
            month_df = df[df['Delivery_Month_Str'] == month]
            
            # Calculate metrics
            total = len(month_df)
            if total > 0:
                # Gross OTP
                on_time = month_df['Is_On_Time'].sum() if 'Is_On_Time' in month_df.columns else 0
                gross_otp = (on_time / total * 100)
                
                # Net OTP (controllables only)
                if 'Is_Controllable' in month_df.columns:
                    controllable_df = month_df[month_df['Is_Controllable']]
                    ctrl_total = len(controllable_df)
                    ctrl_on_time = controllable_df['Is_On_Time'].sum() if ctrl_total > 0 else 0
                    net_otp = (ctrl_on_time / ctrl_total * 100) if ctrl_total > 0 else 0
                else:
                    net_otp = 0
                
                monthly_data.append({
                    'Month': month,
                    'Gross OTP': gross_otp,
                    'Net OTP': net_otp,
                    'Total Shipments': total,
                    'Total Pieces': month_df['PIECES'].sum() if 'PIECES' in month_df.columns else 0,
                    'Total Charges': month_df['TOTAL CHARGES'].sum() if 'TOTAL CHARGES' in month_df.columns else 0
                })
    
    return pd.DataFrame(monthly_data).sort_values('Month') if monthly_data else pd.DataFrame()

@st.cache_data(show_spinner=False)
def create_shipper_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Create detailed shipper/account analysis"""
    if 'SHIPPER NAME' not in df.columns:
        return pd.DataFrame()
    
    shipper_metrics = []
    
    for shipper in df['SHIPPER NAME'].unique():
        if pd.notna(shipper):
            shipper_df = df[df['SHIPPER NAME'] == shipper]
            
            total = len(shipper_df)
            if total > 0:
                metrics_dict = {
                    'Shipper': shipper[:50],  # Truncate long names
                    'Total Shipments': total
                }
                
                if 'Is_On_Time' in shipper_df.columns:
                    on_time = shipper_df['Is_On_Time'].sum()
                    metrics_dict['On-Time'] = on_time
                    metrics_dict['OTP %'] = round((on_time / total * 100), 1)
                
                if 'PIECES' in shipper_df.columns:
                    metrics_dict['Total Pieces'] = int(shipper_df['PIECES'].sum())
                
                if 'TOTAL CHARGES' in shipper_df.columns:
                    metrics_dict['Total Charges ($)'] = round(shipper_df['TOTAL CHARGES'].sum(), 2)
                
                if 'Time In Transit' in shipper_df.columns:
                    metrics_dict['Avg Transit (hrs)'] = round(shipper_df['Time In Transit'].mean(), 1)
                
                shipper_metrics.append(metrics_dict)
    
    return pd.DataFrame(shipper_metrics).sort_values('Total Shipments', ascending=False) if shipper_metrics else pd.DataFrame()

def create_performance_charts(monthly_df: pd.DataFrame) -> go.Figure:
    """Create comprehensive performance charts"""
    if monthly_df.empty:
        return go.Figure()
    
    # Create subplots
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
    
    # Add 95% objective line
    fig.add_trace(
        go.Scatter(x=monthly_df['Month'], y=[95] * len(monthly_df),
                  name='Objective (95%)', mode='lines',
                  line=dict(color='red', width=2, dash='dash')),
        row=1, col=1
    )
    
    # Volume & Revenue
    fig.add_trace(
        go.Bar(x=monthly_df['Month'], y=monthly_df['Total Pieces'],
               name='Total Pieces', marker_color='lightblue'),
        row=2, col=1, secondary_y=False
    )
    
    if 'Total Charges' in monthly_df.columns:
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
    """Create country performance analysis"""
    if 'PU CTRY' not in df.columns:
        return go.Figure()
    
    country_metrics = []
    
    for country in df['PU CTRY'].unique():
        if pd.notna(country):
            country_df = df[df['PU CTRY'] == country]
            
            metrics_dict = {
                'Country': country,
                'Total Shipments': len(country_df)
            }
            
            if 'Is_On_Time' in country_df.columns:
                on_time = country_df['Is_On_Time'].sum()
                metrics_dict['OTP %'] = (on_time / len(country_df) * 100) if len(country_df) > 0 else 0
            
            if 'PIECES' in country_df.columns:
                metrics_dict['Total Pieces'] = country_df['PIECES'].sum()
            
            if 'TOTAL CHARGES' in country_df.columns:
                metrics_dict['Total Charges'] = country_df['TOTAL CHARGES'].sum()
            
            if 'Time In Transit' in country_df.columns:
                metrics_dict['Avg Transit Time'] = country_df['Time In Transit'].mean()
            
            country_metrics.append(metrics_dict)
    
    if not country_metrics:
        return go.Figure()
    
    country_df = pd.DataFrame(country_metrics)
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('OTP by Country', 'Volume by Country', 
                       'Revenue by Country', 'Transit Time by Country'),
        specs=[[{"type": "bar"}, {"type": "pie"}],
               [{"type": "bar"}, {"type": "bar"}]]
    )
    
    # Add traces based on available data
    if 'OTP %' in country_df.columns:
        fig.add_trace(
            go.Bar(x=country_df['Country'], y=country_df['OTP %'],
                   name='OTP %', marker_color='#0066cc',
                   text=country_df['OTP %'].round(1),
                   textposition='outside'),
            row=1, col=1
        )
    
    if 'Total Pieces' in country_df.columns:
        fig.add_trace(
            go.Pie(labels=country_df['Country'], values=country_df['Total Pieces'],
                   name='Pieces Distribution'),
            row=1, col=2
        )
    
    if 'Total Charges' in country_df.columns:
        fig.add_trace(
            go.Bar(x=country_df['Country'], y=country_df['Total Charges'],
                   name='Revenue', marker_color='green',
                   text=country_df['Total Charges'].round(0),
                   textposition='outside'),
            row=2, col=1
        )
    
    if 'Avg Transit Time' in country_df.columns:
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
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📁 Data Upload")
        
        # File uploader - supports both Excel and CSV
        uploaded_file = st.file_uploader(
            "Upload Data File",
            type=['xlsx', 'xls', 'csv'],
            help="Upload Excel (.xlsx, .xls) or CSV file with shipment data"
        )
        
        if uploaded_file is not None:
            # Show file info
            file_size = uploaded_file.size / (1024 * 1024)  # MB
            st.info(f"📄 File: {uploaded_file.name}\n📊 Size: {file_size:.2f} MB")
            
            if st.button("🔄 Process Data", type="primary"):
                with st.spinner("Loading and processing data..."):
                    # Read file content
                    file_content = uploaded_file.read()
                    
                    # Load file
                    df_raw = load_file(file_content, uploaded_file.name)
                    
                    if df_raw is not None:
                        # Process data
                        df_processed = process_data(df_raw)
                        
                        if df_processed is not None:
                            st.session_state.df_processed = df_processed
                            st.session_state.df_filtered = df_processed.copy()
                            st.success(f"✅ Processed {len(df_processed)} records successfully!")
                            st.rerun()
        
        # Filters section
        if st.session_state.df_processed is not None:
            st.markdown("### 🔍 Filters")
            df = st.session_state.df_processed
            
            # Date filter
            if 'POD DATE/TIME' in df.columns:
                date_min = pd.to_datetime(df['POD DATE/TIME'].min())
                date_max = pd.to_datetime(df['POD DATE/TIME'].max())
                
                if pd.notna(date_min) and pd.notna(date_max):
                    date_range = st.date_input(
                        "Date Range",
                        value=(date_min.date(), date_max.date()),
                        min_value=date_min.date(),
                        max_value=date_max.date()
                    )
            
            # Country filter
            if 'PU CTRY' in df.columns:
                countries = st.multiselect(
                    "Countries",
                    options=sorted(df['PU CTRY'].dropna().unique()),
                    default=sorted(df['PU CTRY'].dropna().unique())
                )
            
            # Shipper filter
            if 'SHIPPER NAME' in df.columns:
                if st.checkbox("Filter by Shipper"):
                    shippers = st.multiselect(
                        "Shippers",
                        options=sorted(df['SHIPPER NAME'].dropna().unique()),
                        default=sorted(df['SHIPPER NAME'].dropna().unique())
                    )
                else:
                    shippers = df['SHIPPER NAME'].dropna().unique()
            
            # Apply filters
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Apply Filters", type="secondary"):
                    filtered_df = df.copy()
                    
                    # Apply date filter
                    if 'POD DATE/TIME' in df.columns and 'date_range' in locals():
                        if len(date_range) == 2:
                            filtered_df = filtered_df[
                                (pd.to_datetime(filtered_df['POD DATE/TIME']).dt.date >= date_range[0]) &
                                (pd.to_datetime(filtered_df['POD DATE/TIME']).dt.date <= date_range[1])
                            ]
                    
                    # Apply country filter
                    if 'PU CTRY' in df.columns and 'countries' in locals():
                        filtered_df = filtered_df[filtered_df['PU CTRY'].isin(countries)]
                    
                    # Apply shipper filter
                    if 'SHIPPER NAME' in df.columns and 'shippers' in locals():
                        filtered_df = filtered_df[filtered_df['SHIPPER NAME'].isin(shippers)]
                    
                    st.session_state.df_filtered = filtered_df
                    st.success("Filters applied!")
                    st.rerun()
            
            with col2:
                if st.button("Reset Filters"):
                    st.session_state.df_filtered = st.session_state.df_processed.copy()
                    st.rerun()
    
    # Main content
    if st.session_state.df_filtered is not None:
        df = st.session_state.df_filtered
        
        # Calculate metrics
        metrics = calculate_otp_metrics(df)
        
        # Executive Summary
        st.markdown("## Executive Summary")
        
        # Key metrics
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
                delta=f"{int(metrics['total_pieces'])} Pieces"
            )
        
        # Performance insights
        st.markdown("### 📊 Performance Insights")
        if metrics['gross_otp'] < 95:
            st.warning(f"⚠️ Gross OTP is {95 - metrics['gross_otp']:.1f}% below target")
        else:
            st.success(f"✅ Gross OTP is meeting the 95% objective")
        
        if metrics['net_otp'] < 95 and metrics['controllable_shipments'] > 0:
            st.warning(f"⚠️ Controllable factors need attention - Net OTP at {metrics['net_otp']:.1f}%")
        
        # Tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Trends", "🌍 Country Analysis", "📦 Shipper Analysis", "📋 Detailed Data"])
        
        with tab1:
            st.markdown("### Monthly Performance Trends")
            monthly_df = create_monthly_analysis(df)
            
            if not monthly_df.empty:
                fig = create_performance_charts(monthly_df)
                st.plotly_chart(fig, use_container_width=True)
                
                # Monthly data table
                with st.expander("View Monthly Data"):
                    st.dataframe(monthly_df, use_container_width=True)
            else:
                st.info("Monthly trend analysis requires POD DATE/TIME column")
        
        with tab2:
            st.markdown("### Country Performance Analysis")
            fig_country = create_country_analysis(df)
            if fig_country.data:
                st.plotly_chart(fig_country, use_container_width=True)
            else:
                st.info("Country analysis requires PU CTRY column")
        
        with tab3:
            st.markdown("### Shipper Performance Analysis")
            shipper_df = create_shipper_analysis(df)
            
            if not shipper_df.empty:
                st.dataframe(
                    shipper_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "OTP %": st.column_config.ProgressColumn(
                            "OTP %",
                            help="On-Time Performance",
                            format="%.1f%%",
                            min_value=0,
                            max_value=100,
                        ) if "OTP %" in shipper_df.columns else None,
                        "Total Charges ($)": st.column_config.NumberColumn(
                            "Total Charges ($)",
                            format="$%.2f",
                        ) if "Total Charges ($)" in shipper_df.columns else None,
                    }
                )
                
                # Download shipper analysis
                csv = shipper_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Shipper Analysis",
                    data=csv,
                    file_name=f"shipper_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            else:
                st.info("Shipper analysis requires SHIPPER NAME column")
        
        with tab4:
            st.markdown("### Detailed Shipment Data")
            
            # Filtering options
            col1, col2, col3 = st.columns(3)
            with col1:
                show_controllables = st.checkbox("Show only controllables")
            with col2:
                show_delays = st.checkbox("Show only delays")
            with col3:
                max_rows = st.number_input("Max rows to display", min_value=100, max_value=5000, value=500)
            
            # Apply detail filters
            detail_df = df.copy()
            if show_controllables and 'Is_Controllable' in detail_df.columns:
                detail_df = detail_df[detail_df['Is_Controllable']]
            if show_delays and 'Is_On_Time' in detail_df.columns:
                detail_df = detail_df[~detail_df['Is_On_Time']]
            
            # Display data
            st.info(f"Showing {min(len(detail_df), max_rows)} of {len(detail_df)} records")
            
            # Select columns to display
            available_cols = detail_df.columns.tolist()
            priority_cols = ['REFER', 'SHIPPER NAME', 'PU CTRY', 'DEL CTRY', 
                           'POD DATE/TIME', 'Time In Transit', 'QC NAME', 
                           'PIECES', 'TOTAL CHARGES']
            display_cols = [col for col in priority_cols if col in available_cols]
            
            st.dataframe(detail_df[display_cols].head(max_rows), use_container_width=True)
            
            # Download full data
            csv_full = detail_df.to_csv(index=False)
            st.download_button(
                "📥 Download Full Dataset",
                data=csv_full,
                file_name=f"otp_data_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    else:
        # Welcome screen
        st.info("""
        ### 📌 Getting Started
        
        1. **Upload your data file** (Excel or CSV) using the sidebar
        2. **Click "Process Data"** to analyze the shipment data
        3. **Apply filters** to focus on specific periods or segments
        4. **Navigate tabs** to explore different views
        
        ### 📊 Key Features:
        - **Gross OTP**: Overall on-time performance (95% target)
        - **Net OTP**: Controllable factors only (Agent, Customs, Warehouse)
        - **Monthly Analysis**: Performance trends over time
        - **Country Breakdown**: Analysis for DE, IT, IL
        - **Shipper Analysis**: Performance by account
        
        ### 📋 Required Columns:
        - `POD DATE/TIME` - Actual delivery date
        - `Time In Transit` - Transit hours
        - `PU CTRY` - Pickup country
        - `QC NAME` - Quality control category
        - `PIECES` - Number of pieces
        - `TOTAL CHARGES` - Revenue
        - `SHIPPER NAME` - Account name
        
        The dashboard will adapt to available columns and show relevant analyses.
        """)

if __name__ == "__main__":
    main()
