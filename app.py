"""
Optimized Professional OTP Management Dashboard
Executive Performance Monitoring System
Version 2.0 - Performance Optimized
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import warnings
import pickle
import hashlib
import os

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Executive OTP Performance Dashboard",
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
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    h1 {
        color: #1f77b4;
        font-family: 'Arial', sans-serif;
        font-weight: bold;
    }
    h2 {
        color: #2c3e50;
        font-family: 'Arial', sans-serif;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# Cache functions for better performance
@st.cache_data(show_spinner=False)
def load_excel_file(uploaded_file):
    """Load and cache the Excel file"""
    try:
        # Read only necessary columns to save memory
        df = pd.read_excel(
            uploaded_file,
            usecols=['REFER', 'STATUS', 'SHIPPER NAME', 'PU CTRY', 'DEL CTRY',
                    'ACT PU', 'POD DATE/TIME', 'QDT', 'UPD DEL',
                    'PIECES', 'TOTAL CHARGES', 'QC NAME', 'QCCODE']
        )
        return df
    except:
        # If specific columns fail, read all
        return pd.read_excel(uploaded_file)

@st.cache_data(show_spinner=False)
def preprocess_data(df):
    """Preprocess the dataframe for analysis - cached for performance"""
    
    # Filter for target countries FIRST to reduce data size
    target_countries = ['DE', 'IT', 'IL']
    df_filtered = df[df['PU CTRY'].isin(target_countries)].copy()
    
    if len(df_filtered) == 0:
        return pd.DataFrame()  # Return empty if no data
    
    # Convert Excel date serial numbers to datetime
    date_columns = ['ACT PU', 'UPD DEL', 'POD DATE/TIME', 'QDT']
    for col in date_columns:
        if col in df_filtered.columns:
            try:
                # Excel dates - handle errors gracefully
                df_filtered[col] = pd.to_datetime('1900-01-01') + pd.to_timedelta(df_filtered[col] - 2, unit='D', errors='coerce')
            except:
                pass
    
    # Extract month and year
    if 'POD DATE/TIME' in df_filtered.columns:
        df_filtered['Month_Year'] = df_filtered['POD DATE/TIME'].dt.strftime('%Y-%m')
        df_filtered['Month_Display'] = df_filtered['POD DATE/TIME'].dt.strftime('%b %Y')
    
    # Define controllables efficiently
    controllable_keywords = ['agent', 'agt', 'customs', 'warehouse', 'w/house', 'delivery agent']
    df_filtered['Is_Controllable'] = df_filtered['QC NAME'].fillna('').str.lower().str.contains('|'.join(controllable_keywords), regex=True)
    
    # Calculate OTP
    if 'POD DATE/TIME' in df_filtered.columns and 'UPD DEL' in df_filtered.columns:
        df_filtered['On_Time'] = df_filtered['POD DATE/TIME'] <= df_filtered['UPD DEL']
    elif 'POD DATE/TIME' in df_filtered.columns and 'QDT' in df_filtered.columns:
        df_filtered['On_Time'] = df_filtered['POD DATE/TIME'] <= df_filtered['QDT']
    
    # Clean numeric columns
    df_filtered['TOTAL CHARGES'] = pd.to_numeric(df_filtered['TOTAL CHARGES'], errors='coerce').fillna(0)
    df_filtered['PIECES'] = pd.to_numeric(df_filtered['PIECES'], errors='coerce').fillna(0)
    
    return df_filtered

@st.cache_data(show_spinner=False)
def calculate_monthly_metrics(df_filtered):
    """Calculate monthly metrics - cached"""
    if 'Month_Display' not in df_filtered.columns:
        return None, None
    
    # Gross OTP by month
    monthly_metrics = df_filtered.groupby('Month_Display').agg({
        'On_Time': ['sum', 'count'],
        'PIECES': 'sum',
        'TOTAL CHARGES': 'sum'
    }).round(2)
    
    monthly_metrics.columns = ['On_Time_Count', 'Total_Count', 'Pieces', 'Total_Charges']
    monthly_metrics['Gross_OTP'] = (monthly_metrics['On_Time_Count'] / monthly_metrics['Total_Count'] * 100).round(1)
    
    # Net OTP (controllables) by month
    monthly_controllables = df_filtered[df_filtered['Is_Controllable'] == True].groupby('Month_Display').agg({
        'On_Time': ['sum', 'count']
    })
    
    if len(monthly_controllables) > 0:
        monthly_controllables.columns = ['Controllable_On_Time', 'Controllable_Total']
        monthly_controllables['Net_OTP'] = (monthly_controllables['Controllable_On_Time'] / 
                                           monthly_controllables['Controllable_Total'] * 100).round(1)
        monthly_metrics = monthly_metrics.merge(monthly_controllables[['Net_OTP']], 
                                               left_index=True, right_index=True, how='left')
    
    monthly_metrics = monthly_metrics.reset_index()
    monthly_metrics['Month_Sort'] = pd.to_datetime(monthly_metrics['Month_Display'], format='%b %Y')
    monthly_metrics = monthly_metrics.sort_values('Month_Sort')
    
    return monthly_metrics, monthly_controllables if len(monthly_controllables) > 0 else None

@st.cache_data(show_spinner=False)
def calculate_account_metrics(df_filtered, top_n=20):
    """Calculate account metrics - cached"""
    account_metrics = df_filtered.groupby('SHIPPER NAME').agg({
        'On_Time': ['sum', 'count'],
        'PIECES': 'sum',
        'TOTAL CHARGES': 'sum'
    }).round(2)
    
    account_metrics.columns = ['On_Time_Count', 'Total_Shipments', 'Total_Pieces', 'Total_Revenue']
    account_metrics['OTP_%'] = (account_metrics['On_Time_Count'] / account_metrics['Total_Shipments'] * 100).round(1)
    
    return account_metrics.sort_values('Total_Revenue', ascending=False).head(top_n)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'df_processed' not in st.session_state:
    st.session_state.df_processed = None

# Header
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    st.title("📊 Executive OTP Performance Dashboard")
    st.markdown("**Operational Excellence Monitoring System - Optimized Version**")
    st.markdown(f"*Report Generated: {datetime.now().strftime('%B %d, %Y at %H:%M')}*")

st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("🔧 Dashboard Controls")
    
    st.subheader("📁 Data Upload")
    uploaded_file = st.file_uploader(
        "Select Excel Data File",
        type=['xlsx', 'xls'],
        help="Upload the gvExportData Excel file"
    )
    
    if uploaded_file is not None:
        try:
            # Check if file has changed
            file_hash = hashlib.md5(uploaded_file.read()).hexdigest()
            uploaded_file.seek(0)  # Reset file pointer
            
            if 'file_hash' not in st.session_state or st.session_state.file_hash != file_hash:
                with st.spinner('Loading data... This may take a moment for large files.'):
                    # Load and process data
                    df_raw = load_excel_file(uploaded_file)
                    df_processed = preprocess_data(df_raw)
                    
                    if len(df_processed) > 0:
                        st.session_state.df_processed = df_processed
                        st.session_state.data_loaded = True
                        st.session_state.file_hash = file_hash
                        st.success(f"✅ Data loaded! Found {len(df_processed):,} records for DE/IT/IL")
                    else:
                        st.error("❌ No data found for countries DE, IT, or IL")
                        st.session_state.data_loaded = False
        except Exception as e:
            st.error(f"❌ Error: {str(e)[:100]}")

# Main Dashboard
if st.session_state.data_loaded and st.session_state.df_processed is not None:
    df_filtered = st.session_state.df_processed.copy()
    
    # Sidebar filters
    with st.sidebar:
        st.markdown("---")
        st.subheader("🎯 Filters")
        
        # Date range filter
        if 'POD DATE/TIME' in df_filtered.columns:
            min_date = df_filtered['POD DATE/TIME'].min()
            max_date = df_filtered['POD DATE/TIME'].max()
            
            # Use date slider for better performance
            date_range = st.slider(
                "Select Date Range",
                min_value=min_date.date() if pd.notna(min_date) else datetime.now().date(),
                max_value=max_date.date() if pd.notna(max_date) else datetime.now().date(),
                value=(min_date.date() if pd.notna(min_date) else datetime.now().date(), 
                       max_date.date() if pd.notna(max_date) else datetime.now().date()),
                format="MMM YYYY"
            )
            
            # Apply filter
            mask = (df_filtered['POD DATE/TIME'].dt.date >= date_range[0]) & \
                   (df_filtered['POD DATE/TIME'].dt.date <= date_range[1])
            df_filtered = df_filtered[mask]
        
        # Country filter
        selected_countries = st.multiselect(
            "Countries",
            options=['DE', 'IT', 'IL'],
            default=['DE', 'IT', 'IL']
        )
        
        if selected_countries:
            df_filtered = df_filtered[df_filtered['PU CTRY'].isin(selected_countries)]
        
        st.markdown("---")
        otp_target = st.number_input("OTP Target %", min_value=0, max_value=100, value=95)
    
    # Calculate KPIs
    total_shipments = len(df_filtered)
    on_time_shipments = df_filtered['On_Time'].sum() if 'On_Time' in df_filtered.columns else 0
    gross_otp = (on_time_shipments / total_shipments * 100) if total_shipments > 0 else 0
    
    controllable_df = df_filtered[df_filtered['Is_Controllable'] == True]
    controllable_total = len(controllable_df)
    controllable_on_time = controllable_df['On_Time'].sum() if controllable_total > 0 else 0
    net_otp = (controllable_on_time / controllable_total * 100) if controllable_total > 0 else 0
    
    total_charges = df_filtered['TOTAL CHARGES'].sum()
    total_pieces = df_filtered['PIECES'].sum()
    
    # Executive Summary
    st.header("📊 Executive Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Gross OTP",
            f"{gross_otp:.1f}%",
            f"{gross_otp - otp_target:.1f}% vs target",
            delta_color="normal" if gross_otp >= otp_target else "inverse"
        )
    
    with col2:
        st.metric(
            "Net OTP (Controllables)",
            f"{net_otp:.1f}%",
            f"{net_otp - otp_target:.1f}% vs target",
            delta_color="normal" if net_otp >= otp_target else "inverse"
        )
    
    with col3:
        st.metric(
            "Total Shipments",
            f"{total_shipments:,}",
            f"{controllable_total:,} controllables"
        )
    
    with col4:
        st.metric(
            "Total Revenue",
            f"${total_charges:,.0f}",
            f"{total_pieces:,.0f} pieces"
        )
    
    st.markdown("---")
    
    # Monthly Trends
    st.header("📈 Performance Trends")
    
    monthly_metrics, monthly_controllables = calculate_monthly_metrics(df_filtered)
    
    if monthly_metrics is not None and len(monthly_metrics) > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            # OTP Trend
            fig_otp = go.Figure()
            
            fig_otp.add_trace(go.Scatter(
                x=monthly_metrics['Month_Display'],
                y=monthly_metrics['Gross_OTP'],
                mode='lines+markers',
                name='Gross OTP',
                line=dict(color='#1f77b4', width=3)
            ))
            
            if 'Net_OTP' in monthly_metrics.columns:
                fig_otp.add_trace(go.Scatter(
                    x=monthly_metrics['Month_Display'],
                    y=monthly_metrics['Net_OTP'],
                    mode='lines+markers',
                    name='Net OTP',
                    line=dict(color='#ff7f0e', width=3)
                ))
            
            fig_otp.add_hline(y=otp_target, line_dash="dash", line_color="red", 
                            annotation_text=f"Target {otp_target}%")
            
            fig_otp.update_layout(
                title="OTP Performance Trend",
                xaxis_title="Month",
                yaxis_title="OTP (%)",
                hovermode='x unified',
                height=400,
                yaxis=dict(range=[0, 105])
            )
            
            st.plotly_chart(fig_otp, use_container_width=True)
        
        with col2:
            # Volume & Revenue
            fig_volume = make_subplots(specs=[[{"secondary_y": True}]])
            
            fig_volume.add_trace(
                go.Bar(
                    x=monthly_metrics['Month_Display'],
                    y=monthly_metrics['Pieces'],
                    name='Pieces',
                    marker_color='lightblue'
                ),
                secondary_y=False
            )
            
            fig_volume.add_trace(
                go.Scatter(
                    x=monthly_metrics['Month_Display'],
                    y=monthly_metrics['Total_Charges'],
                    mode='lines+markers',
                    name='Revenue',
                    line=dict(color='green', width=3)
                ),
                secondary_y=True
            )
            
            fig_volume.update_xaxes(title_text="Month")
            fig_volume.update_yaxes(title_text="Pieces", secondary_y=False)
            fig_volume.update_yaxes(title_text="Revenue ($)", secondary_y=True)
            fig_volume.update_layout(title="Volume & Revenue Trend", height=400)
            
            st.plotly_chart(fig_volume, use_container_width=True)
    
    st.markdown("---")
    
    # Account Performance (Top 10 only for speed)
    st.header("💼 Top Account Performance")
    
    account_metrics = calculate_account_metrics(df_filtered, top_n=10)
    
    if len(account_metrics) > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            # Revenue chart
            fig_revenue = px.bar(
                account_metrics.head(10).sort_values('Total_Revenue'),
                x='Total_Revenue',
                y=account_metrics.head(10).sort_values('Total_Revenue').index,
                orientation='h',
                title='Top 10 Accounts by Revenue',
                labels={'Total_Revenue': 'Revenue ($)', 'y': 'Account'},
                color='OTP_%',
                color_continuous_scale='RdYlGn',
                color_continuous_midpoint=otp_target
            )
            fig_revenue.update_layout(height=400)
            st.plotly_chart(fig_revenue, use_container_width=True)
        
        with col2:
            # Volume chart
            fig_pieces = px.bar(
                account_metrics.head(10).sort_values('Total_Pieces'),
                x='Total_Pieces',
                y=account_metrics.head(10).sort_values('Total_Pieces').index,
                orientation='h',
                title='Top 10 Accounts by Volume',
                labels={'Total_Pieces': 'Pieces', 'y': 'Account'}
            )
            fig_pieces.update_layout(height=400)
            st.plotly_chart(fig_pieces, use_container_width=True)
    
    st.markdown("---")
    
    # Country Performance
    st.header("🌍 Country Performance")
    
    country_metrics = df_filtered.groupby('PU CTRY').agg({
        'On_Time': ['sum', 'count'],
        'PIECES': 'sum',
        'TOTAL CHARGES': 'sum'
    })
    
    country_metrics.columns = ['On_Time_Count', 'Total_Shipments', 'Total_Pieces', 'Total_Revenue']
    country_metrics['OTP_%'] = (country_metrics['On_Time_Count'] / country_metrics['Total_Shipments'] * 100).round(1)
    
    cols = st.columns(len(country_metrics))
    for i, (country, metrics) in enumerate(country_metrics.iterrows()):
        with cols[i]:
            st.markdown(f"### {country}")
            st.metric("OTP", f"{metrics['OTP_%']:.1f}%")
            st.metric("Shipments", f"{metrics['Total_Shipments']:,.0f}")
            st.metric("Revenue", f"${metrics['Total_Revenue']:,.0f}")
    
    st.markdown("---")
    
    # Quick Insights
    st.header("💡 Key Insights")
    
    insights = []
    if gross_otp < otp_target:
        insights.append(f"• Gross OTP at {gross_otp:.1f}% is {otp_target - gross_otp:.1f}% below target")
    
    if net_otp > gross_otp:
        insights.append(f"• Controllable factors performing better ({net_otp:.1f}%) than overall")
    
    worst_country = country_metrics.nsmallest(1, 'OTP_%').index[0] if len(country_metrics) > 0 else None
    if worst_country:
        insights.append(f"• {worst_country} requires attention with lowest OTP")
    
    for insight in insights[:5]:  # Limit to top 5 insights
        st.markdown(insight)
    
    # Export Section
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Summary export
        summary = {
            'Metric': ['Gross OTP', 'Net OTP', 'Total Shipments', 'Total Revenue'],
            'Value': [f"{gross_otp:.1f}%", f"{net_otp:.1f}%", total_shipments, f"${total_charges:,.0f}"]
        }
        summary_df = pd.DataFrame(summary)
        
        st.download_button(
            "📊 Download Summary",
            summary_df.to_csv(index=False),
            f"OTP_Summary_{datetime.now().strftime('%Y%m%d')}.csv",
            "text/csv"
        )
    
    with col2:
        if monthly_metrics is not None:
            st.download_button(
                "📈 Download Trends",
                monthly_metrics.to_csv(index=False),
                f"Monthly_Trends_{datetime.now().strftime('%Y%m%d')}.csv",
                "text/csv"
            )

else:
    # Landing page
    st.info("📤 Please upload the Excel data file to begin analysis")
    
    with st.expander("📖 Instructions"):
        st.markdown("""
        ### Quick Start Guide
        
        1. **Upload your Excel file** using the sidebar
        2. **Data will be automatically filtered** for DE, IT, IL countries
        3. **Review the dashboard** - all calculations are automatic
        4. **Export reports** as needed
        
        **Performance Optimizations:**
        - Data is cached after first load
        - Only relevant columns are processed
        - Visualizations are limited to essential metrics
        - Calculations are pre-computed for speed
        
        **OTP Calculations:**
        - **Gross OTP**: All shipments on-time percentage
        - **Net OTP**: Only controllable factors (Agent, Customs, Warehouse)
        - **Target**: 95% (adjustable in sidebar)
        """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; font-size: 12px;'>"
    "Executive Dashboard v2.0 - Optimized for Performance | © 2024"
    "</div>", 
    unsafe_allow_html=True
)
