"""
OTP Management Dashboard - Fast Optimized Version
Minimal dependencies, maximum performance
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import io

# Page config
st.set_page_config(
    page_title="OTP Dashboard",
    page_icon="📊",
    layout="wide"
)

# Simple styling
st.markdown("""
<style>
    .main {padding-top: 0;}
    div[data-testid="metric-container"] {
        background-color: #f0f2f6;
        border: 1px solid #ccc;
        border-radius: 5px;
        padding: 10px;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None

# Title
st.title("📊 OTP Management Dashboard")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("📁 Upload Data")
    
    uploaded_file = st.file_uploader(
        "Choose file",
        type=['csv', 'xlsx', 'xls']
    )
    
    if uploaded_file and st.button("Load Data", type="primary"):
        try:
            # Read file based on type
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            # Basic processing
            # Convert dates
            if 'POD DATE/TIME' in df.columns:
                df['POD DATE/TIME'] = pd.to_datetime(df['POD DATE/TIME'], errors='coerce')
                df['Month'] = df['POD DATE/TIME'].dt.strftime('%Y-%m')
            
            # Convert numerics
            for col in ['Time In Transit', 'PIECES', 'TOTAL CHARGES']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce', downcast='float')
            
            # Filter countries
            if 'PU CTRY' in df.columns:
                df = df[df['PU CTRY'].isin(['DE', 'IT', 'IL'])]
            
            # Calculate OTP
            if 'Time In Transit' in df.columns:
                df['On_Time'] = df['Time In Transit'] <= 72
            
            # Identify controllables
            if 'QC NAME' in df.columns:
                df['Controllable'] = df['QC NAME'].fillna('').str.contains('Agent|Customs|Warehouse', case=False)
            
            st.session_state.data = df
            st.success(f"✅ Loaded {len(df)} records")
            st.rerun()
            
        except Exception as e:
            st.error(f"Error: {str(e)[:100]}")

# Main area
if st.session_state.data is not None:
    df = st.session_state.data
    
    # Quick metrics
    col1, col2, col3, col4 = st.columns(4)
    
    # Calculate OTP metrics
    if 'On_Time' in df.columns:
        total = len(df)
        on_time = df['On_Time'].sum()
        gross_otp = (on_time / total * 100) if total > 0 else 0
        
        with col1:
            st.metric("Gross OTP", f"{gross_otp:.1f}%", f"{gross_otp-95:.1f}% vs 95%")
        
        # Net OTP (controllables only)
        if 'Controllable' in df.columns:
            ctrl_df = df[df['Controllable']]
            if len(ctrl_df) > 0:
                net_otp = (ctrl_df['On_Time'].sum() / len(ctrl_df) * 100)
                with col2:
                    st.metric("Net OTP", f"{net_otp:.1f}%", f"{net_otp-95:.1f}% vs 95%")
    
    with col3:
        st.metric("Total Shipments", f"{len(df):,}")
    
    if 'TOTAL CHARGES' in df.columns:
        with col4:
            st.metric("Total Revenue", f"${df['TOTAL CHARGES'].sum():,.0f}")
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📈 Monthly", "📦 Shipper", "📋 Data"])
    
    with tab1:
        if 'Month' in df.columns and 'On_Time' in df.columns:
            # Monthly metrics
            monthly = df.groupby('Month').agg({
                'On_Time': ['count', 'sum'],
                'PIECES': 'sum',
                'TOTAL CHARGES': 'sum'
            }).round(1)
            
            monthly.columns = ['Total', 'On-Time', 'Pieces', 'Charges']
            monthly['OTP%'] = (monthly['On-Time'] / monthly['Total'] * 100).round(1)
            
            # Simple plot
            fig = go.Figure()
            
            # OTP line
            fig.add_trace(go.Scatter(
                x=monthly.index,
                y=monthly['OTP%'],
                mode='lines+markers',
                name='OTP %',
                line=dict(color='blue', width=2)
            ))
            
            # 95% target line
            fig.add_hline(y=95, line_dash="dash", line_color="red", 
                         annotation_text="95% Target")
            
            fig.update_layout(
                title="Monthly OTP Trend",
                xaxis_title="Month",
                yaxis_title="OTP %",
                height=400,
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Monthly table
            st.subheader("Monthly Metrics")
            display_monthly = monthly[['Total', 'OTP%', 'Pieces', 'Charges']]
            st.dataframe(display_monthly, use_container_width=True)
            
            # Net OTP trend if available
            if 'Controllable' in df.columns:
                ctrl_monthly = df[df['Controllable']].groupby('Month').agg({
                    'On_Time': ['count', 'sum']
                })
                if len(ctrl_monthly) > 0:
                    ctrl_monthly.columns = ['Total', 'On-Time']
                    ctrl_monthly['Net_OTP%'] = (ctrl_monthly['On-Time'] / ctrl_monthly['Total'] * 100).round(1)
                    
                    st.subheader("Net OTP (Controllables Only)")
                    st.line_chart(ctrl_monthly['Net_OTP%'])
    
    with tab2:
        if 'SHIPPER NAME' in df.columns:
            # Shipper analysis
            shipper_stats = []
            
            for shipper in df['SHIPPER NAME'].dropna().unique():
                shipper_df = df[df['SHIPPER NAME'] == shipper]
                
                stats = {
                    'Shipper': shipper[:40],
                    'Shipments': len(shipper_df)
                }
                
                if 'On_Time' in shipper_df.columns:
                    stats['OTP%'] = round((shipper_df['On_Time'].sum() / len(shipper_df) * 100), 1)
                
                if 'PIECES' in shipper_df.columns:
                    stats['Pieces'] = int(shipper_df['PIECES'].sum())
                
                if 'TOTAL CHARGES' in shipper_df.columns:
                    stats['Revenue'] = round(shipper_df['TOTAL CHARGES'].sum(), 0)
                
                shipper_stats.append(stats)
            
            shipper_df = pd.DataFrame(shipper_stats).sort_values('Shipments', ascending=False)
            
            st.subheader("Shipper Performance")
            st.dataframe(shipper_df.head(50), use_container_width=True, hide_index=True)
            
            # Download
            csv = shipper_df.to_csv(index=False)
            st.download_button(
                "📥 Download Shipper Analysis",
                csv,
                f"shippers_{datetime.now().strftime('%Y%m%d')}.csv",
                "text/csv"
            )
    
    with tab3:
        st.subheader("Data View")
        
        # Filter options
        col1, col2 = st.columns(2)
        with col1:
            if 'Controllable' in df.columns:
                if st.checkbox("Controllables only"):
                    display_df = df[df['Controllable']]
                else:
                    display_df = df
            else:
                display_df = df
        
        with col2:
            if 'On_Time' in df.columns:
                if st.checkbox("Delays only"):
                    display_df = display_df[~display_df['On_Time']]
        
        # Show count
        st.info(f"Showing {len(display_df)} records")
        
        # Display key columns only
        key_cols = ['REFER', 'SHIPPER NAME', 'PU CTRY', 'POD DATE/TIME', 
                   'Time In Transit', 'QC NAME', 'PIECES', 'TOTAL CHARGES']
        show_cols = [c for c in key_cols if c in display_df.columns]
        
        st.dataframe(
            display_df[show_cols].head(500),
            use_container_width=True,
            hide_index=True
        )
        
        # Download full data
        csv = display_df.to_csv(index=False)
        st.download_button(
            "📥 Download Data",
            csv,
            f"otp_data_{datetime.now().strftime('%Y%m%d')}.csv",
            "text/csv"
        )

else:
    # Instructions
    st.info("""
    ### 👆 Upload your data file to start
    
    **Supported formats:** CSV, Excel (.xlsx, .xls)
    
    **Required columns:**
    - `POD DATE/TIME` - Delivery date
    - `Time In Transit` - Hours in transit
    - `PU CTRY` - Pickup country (DE, IT, IL)
    - `QC NAME` - Quality control category
    - `PIECES` - Number of pieces
    - `TOTAL CHARGES` - Revenue
    - `SHIPPER NAME` - Account name
    
    **OTP Calculation:**
    - **Gross OTP:** All shipments ≤72 hours
    - **Net OTP:** Only controllables (Agent/Customs/Warehouse issues)
    - **Target:** 95% for both metrics
    """)
    
    # Quick test data generator
    if st.button("Generate Test Data"):
        test_data = pd.DataFrame({
            'REFER': [f'TEST-{i}' for i in range(100)],
            'PU CTRY': ['DE'] * 40 + ['IT'] * 30 + ['IL'] * 30,
            'POD DATE/TIME': pd.date_range('2024-01-01', periods=100, freq='D'),
            'Time In Transit': [50 + i % 50 for i in range(100)],
            'QC NAME': ['Agent-Issue'] * 20 + ['Customs-Delay'] * 20 + ['Weather'] * 60,
            'PIECES': [10 + i % 20 for i in range(100)],
            'TOTAL CHARGES': [1000 + i * 50 for i in range(100)],
            'SHIPPER NAME': ['Company A'] * 50 + ['Company B'] * 50
        })
        
        csv = test_data.to_csv(index=False)
        st.download_button(
            "📥 Download Test Data",
            csv,
            "test_data.csv",
            "text/csv"
        )
