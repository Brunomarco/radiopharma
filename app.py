"""
Professional OTP Management Dashboard
Executive Performance Monitoring System
Version 1.0
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import warnings
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
    h3 {
        color: #34495e;
        font-family: 'Arial', sans-serif;
    }
    .reportview-container {
        background: linear-gradient(to bottom, #ffffff 0%, #f8f9fa 100%);
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'df' not in st.session_state:
    st.session_state.df = None

# Header with company branding
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    st.title("📊 Executive OTP Performance Dashboard")
    st.markdown("**Operational Excellence Monitoring System**")
    st.markdown(f"*Report Generated: {datetime.now().strftime('%B %d, %Y at %H:%M')}*")

st.markdown("---")

# Sidebar for file upload and filters
with st.sidebar:
    st.header("🔧 Dashboard Controls")
    
    # File upload
    st.subheader("📁 Data Upload")
    uploaded_file = st.file_uploader(
        "Select Excel Data File",
        type=['xlsx', 'xls'],
        help="Upload the gvExportData Excel file"
    )
    
    if uploaded_file is not None:
        try:
            with st.spinner('Processing data...'):
                # Read the Excel file
                df = pd.read_excel(uploaded_file)
                st.session_state.df = df
                st.session_state.data_loaded = True
                st.success("✅ Data loaded successfully!")
                st.info(f"📊 Total Records: {len(df):,}")
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")

# Main dashboard content
if st.session_state.data_loaded and st.session_state.df is not None:
    df = st.session_state.df.copy()
    
    # Data preprocessing
    def preprocess_data(df):
        """Preprocess the dataframe for analysis"""
        # Convert Excel date serial numbers to datetime
        date_columns = ['ACT PU', 'UPD DEL', 'POD DATE/TIME', 'QDT']
        for col in date_columns:
            if col in df.columns:
                # Excel dates start from 1900-01-01, serial number format
                df[col] = pd.to_datetime('1900-01-01') + pd.to_timedelta(df[col] - 2, unit='D')
        
        # Filter for target countries only (DE, IT, IL)
        target_countries = ['DE', 'IT', 'IL']
        df_filtered = df[df['PU CTRY'].isin(target_countries)].copy()
        
        # Extract month and year from delivery date
        if 'POD DATE/TIME' in df_filtered.columns:
            df_filtered['Delivery_Month'] = df_filtered['POD DATE/TIME'].dt.to_period('M')
            df_filtered['Month_Year'] = df_filtered['POD DATE/TIME'].dt.strftime('%Y-%m')
            df_filtered['Month_Display'] = df_filtered['POD DATE/TIME'].dt.strftime('%B %Y')
        
        # Define controllable QC codes (Agent, Customs, Warehouse)
        controllable_keywords = ['agent', 'agt', 'customs', 'warehouse', 'w/house', 'delivery agent']
        
        def is_controllable(qc_name):
            if pd.isna(qc_name):
                return False
            qc_lower = str(qc_name).lower()
            return any(keyword in qc_lower for keyword in controllable_keywords)
        
        df_filtered['Is_Controllable'] = df_filtered['QC NAME'].apply(is_controllable)
        
        # Calculate OTP (On-Time Performance)
        # Assuming delivery is on-time if POD DATE/TIME <= UPD DEL (Updated Delivery)
        if 'POD DATE/TIME' in df_filtered.columns and 'UPD DEL' in df_filtered.columns:
            df_filtered['On_Time'] = df_filtered['POD DATE/TIME'] <= df_filtered['UPD DEL']
        else:
            # Alternative: use QDT if UPD DEL not available
            df_filtered['On_Time'] = df_filtered['POD DATE/TIME'] <= df_filtered['QDT']
        
        # Clean numeric columns
        if 'TOTAL CHARGES' in df_filtered.columns:
            df_filtered['TOTAL CHARGES'] = pd.to_numeric(df_filtered['TOTAL CHARGES'], errors='coerce')
        if 'PIECES' in df_filtered.columns:
            df_filtered['PIECES'] = pd.to_numeric(df_filtered['PIECES'], errors='coerce')
        
        return df_filtered
    
    # Process the data
    df_processed = preprocess_data(df)
    
    if len(df_processed) == 0:
        st.warning("⚠️ No data found for countries DE, IT, or IL. Please check your data file.")
    else:
        # Sidebar filters (continued)
        with st.sidebar:
            st.markdown("---")
            st.subheader("🎯 Data Filters")
            
            # Date range filter
            if 'POD DATE/TIME' in df_processed.columns:
                min_date = df_processed['POD DATE/TIME'].min()
                max_date = df_processed['POD DATE/TIME'].max()
                
                date_range = st.date_input(
                    "Select Date Range",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date
                )
                
                # Apply date filter
                if len(date_range) == 2:
                    mask = (df_processed['POD DATE/TIME'].dt.date >= date_range[0]) & \
                           (df_processed['POD DATE/TIME'].dt.date <= date_range[1])
                    df_filtered = df_processed[mask].copy()
                else:
                    df_filtered = df_processed.copy()
            else:
                df_filtered = df_processed.copy()
            
            # Country filter
            selected_countries = st.multiselect(
                "Select Countries",
                options=['DE', 'IT', 'IL'],
                default=['DE', 'IT', 'IL']
            )
            
            if selected_countries:
                df_filtered = df_filtered[df_filtered['PU CTRY'].isin(selected_countries)]
            
            st.markdown("---")
            st.subheader("📈 OTP Target")
            otp_target = st.slider("OTP Objective (%)", 0, 100, 95)
        
        # Main Dashboard Content
        
        # Executive Summary Section
        st.header("📊 Executive Summary")
        
        # Calculate key metrics
        total_shipments = len(df_filtered)
        on_time_shipments = df_filtered['On_Time'].sum() if 'On_Time' in df_filtered.columns else 0
        
        # Gross OTP (all shipments)
        gross_otp = (on_time_shipments / total_shipments * 100) if total_shipments > 0 else 0
        
        # Net OTP (controllables only)
        controllable_df = df_filtered[df_filtered['Is_Controllable'] == True]
        controllable_total = len(controllable_df)
        controllable_on_time = controllable_df['On_Time'].sum() if 'On_Time' in controllable_df.columns and controllable_total > 0 else 0
        net_otp = (controllable_on_time / controllable_total * 100) if controllable_total > 0 else 0
        
        # Total charges and pieces
        total_charges = df_filtered['TOTAL CHARGES'].sum() if 'TOTAL CHARGES' in df_filtered.columns else 0
        total_pieces = df_filtered['PIECES'].sum() if 'PIECES' in df_filtered.columns else 0
        
        # Display KPI metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Gross OTP",
                value=f"{gross_otp:.1f}%",
                delta=f"{gross_otp - otp_target:.1f}% vs target",
                delta_color="normal" if gross_otp >= otp_target else "inverse"
            )
        
        with col2:
            st.metric(
                label="Net OTP (Controllables)",
                value=f"{net_otp:.1f}%",
                delta=f"{net_otp - otp_target:.1f}% vs target",
                delta_color="normal" if net_otp >= otp_target else "inverse"
            )
        
        with col3:
            st.metric(
                label="Total Shipments",
                value=f"{total_shipments:,}",
                delta=f"{controllable_total:,} controllables"
            )
        
        with col4:
            st.metric(
                label="Total Revenue",
                value=f"${total_charges:,.2f}",
                delta=f"{total_pieces:,.0f} pieces"
            )
        
        st.markdown("---")
        
        # Month-on-Month Performance Analysis
        st.header("📈 Month-on-Month Performance Trends")
        
        if 'Month_Display' in df_filtered.columns:
            # Calculate monthly metrics
            monthly_metrics = df_filtered.groupby('Month_Display').agg({
                'On_Time': ['sum', 'count'],
                'PIECES': 'sum',
                'TOTAL CHARGES': 'sum'
            }).round(2)
            
            # Calculate OTP percentages
            monthly_metrics.columns = ['On_Time_Count', 'Total_Count', 'Pieces', 'Total_Charges']
            monthly_metrics['Gross_OTP'] = (monthly_metrics['On_Time_Count'] / monthly_metrics['Total_Count'] * 100).round(1)
            
            # Calculate Net OTP (controllables only) by month
            monthly_controllables = df_filtered[df_filtered['Is_Controllable'] == True].groupby('Month_Display').agg({
                'On_Time': ['sum', 'count']
            })
            monthly_controllables.columns = ['Controllable_On_Time', 'Controllable_Total']
            monthly_controllables['Net_OTP'] = (monthly_controllables['Controllable_On_Time'] / 
                                                monthly_controllables['Controllable_Total'] * 100).round(1)
            
            # Merge the metrics
            monthly_metrics = monthly_metrics.merge(monthly_controllables[['Net_OTP']], 
                                                   left_index=True, right_index=True, how='left')
            
            # Sort by month
            monthly_metrics = monthly_metrics.reset_index()
            monthly_metrics['Month_Sort'] = pd.to_datetime(monthly_metrics['Month_Display'], format='%B %Y')
            monthly_metrics = monthly_metrics.sort_values('Month_Sort')
            
            # Create dual-axis chart for OTP trends
            col1, col2 = st.columns(2)
            
            with col1:
                # OTP Trend Chart
                fig_otp = go.Figure()
                
                # Add Gross OTP line
                fig_otp.add_trace(go.Scatter(
                    x=monthly_metrics['Month_Display'],
                    y=monthly_metrics['Gross_OTP'],
                    mode='lines+markers',
                    name='Gross OTP',
                    line=dict(color='#1f77b4', width=3),
                    marker=dict(size=8)
                ))
                
                # Add Net OTP line
                fig_otp.add_trace(go.Scatter(
                    x=monthly_metrics['Month_Display'],
                    y=monthly_metrics['Net_OTP'],
                    mode='lines+markers',
                    name='Net OTP (Controllables)',
                    line=dict(color='#ff7f0e', width=3),
                    marker=dict(size=8)
                ))
                
                # Add target line
                fig_otp.add_trace(go.Scatter(
                    x=monthly_metrics['Month_Display'],
                    y=[otp_target] * len(monthly_metrics),
                    mode='lines',
                    name='Target',
                    line=dict(color='red', width=2, dash='dash')
                ))
                
                fig_otp.update_layout(
                    title="OTP Performance Trend",
                    xaxis_title="Month",
                    yaxis_title="OTP (%)",
                    hovermode='x unified',
                    showlegend=True,
                    height=400,
                    yaxis=dict(range=[0, 105])
                )
                
                st.plotly_chart(fig_otp, use_container_width=True)
            
            with col2:
                # Volume Trend Chart
                fig_volume = make_subplots(specs=[[{"secondary_y": True}]])
                
                # Add Pieces bar
                fig_volume.add_trace(
                    go.Bar(
                        x=monthly_metrics['Month_Display'],
                        y=monthly_metrics['Pieces'],
                        name='Pieces',
                        marker_color='lightblue',
                        opacity=0.7
                    ),
                    secondary_y=False
                )
                
                # Add Revenue line
                fig_volume.add_trace(
                    go.Scatter(
                        x=monthly_metrics['Month_Display'],
                        y=monthly_metrics['Total_Charges'],
                        mode='lines+markers',
                        name='Revenue',
                        line=dict(color='green', width=3),
                        marker=dict(size=8)
                    ),
                    secondary_y=True
                )
                
                fig_volume.update_xaxes(title_text="Month")
                fig_volume.update_yaxes(title_text="Pieces", secondary_y=False)
                fig_volume.update_yaxes(title_text="Revenue ($)", secondary_y=True)
                fig_volume.update_layout(
                    title="Volume & Revenue Trend",
                    hovermode='x unified',
                    height=400
                )
                
                st.plotly_chart(fig_volume, use_container_width=True)
        
        st.markdown("---")
        
        # Performance by Account
        st.header("💼 Performance by Account")
        
        # Group by shipper name (account)
        account_metrics = df_filtered.groupby('SHIPPER NAME').agg({
            'On_Time': ['sum', 'count'],
            'PIECES': 'sum',
            'TOTAL CHARGES': 'sum'
        }).round(2)
        
        account_metrics.columns = ['On_Time_Count', 'Total_Shipments', 'Total_Pieces', 'Total_Revenue']
        account_metrics['OTP_%'] = (account_metrics['On_Time_Count'] / account_metrics['Total_Shipments'] * 100).round(1)
        account_metrics = account_metrics.sort_values('Total_Revenue', ascending=False).head(20)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Top Accounts by Revenue
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
            # Top Accounts by Volume
            fig_volume_acc = px.bar(
                account_metrics.head(10).sort_values('Total_Pieces'),
                x='Total_Pieces',
                y=account_metrics.head(10).sort_values('Total_Pieces').index,
                orientation='h',
                title='Top 10 Accounts by Volume',
                labels={'Total_Pieces': 'Pieces', 'y': 'Account'},
                color='OTP_%',
                color_continuous_scale='RdYlGn',
                color_continuous_midpoint=otp_target
            )
            fig_volume_acc.update_layout(height=400)
            st.plotly_chart(fig_volume_acc, use_container_width=True)
        
        # Detailed Account Table
        st.subheader("📋 Detailed Account Performance")
        
        # Format the dataframe for display
        display_df = account_metrics.copy()
        display_df['Total_Revenue'] = display_df['Total_Revenue'].apply(lambda x: f"${x:,.2f}")
        display_df['Total_Pieces'] = display_df['Total_Pieces'].apply(lambda x: f"{x:,.0f}")
        display_df['Total_Shipments'] = display_df['Total_Shipments'].apply(lambda x: f"{x:,.0f}")
        display_df['OTP_%'] = display_df['OTP_%'].apply(lambda x: f"{x:.1f}%")
        
        st.dataframe(
            display_df[['Total_Shipments', 'Total_Pieces', 'Total_Revenue', 'OTP_%']],
            use_container_width=True,
            height=400
        )
        
        st.markdown("---")
        
        # Country Performance Analysis
        st.header("🌍 Performance by Country")
        
        country_metrics = df_filtered.groupby('PU CTRY').agg({
            'On_Time': ['sum', 'count'],
            'PIECES': 'sum',
            'TOTAL CHARGES': 'sum'
        }).round(2)
        
        country_metrics.columns = ['On_Time_Count', 'Total_Shipments', 'Total_Pieces', 'Total_Revenue']
        country_metrics['OTP_%'] = (country_metrics['On_Time_Count'] / country_metrics['Total_Shipments'] * 100).round(1)
        
        col1, col2, col3 = st.columns(3)
        
        for i, country in enumerate(country_metrics.index):
            with [col1, col2, col3][i]:
                st.markdown(f"### {country}")
                otp = country_metrics.loc[country, 'OTP_%']
                shipments = country_metrics.loc[country, 'Total_Shipments']
                revenue = country_metrics.loc[country, 'Total_Revenue']
                
                # Create gauge chart for OTP
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=otp,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "OTP %"},
                    delta={'reference': otp_target},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "green" if otp >= otp_target else "orange"},
                        'steps': [
                            {'range': [0, otp_target-5], 'color': "lightgray"},
                            {'range': [otp_target-5, otp_target], 'color': "yellow"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': otp_target
                        }
                    }
                ))
                fig_gauge.update_layout(height=250)
                st.plotly_chart(fig_gauge, use_container_width=True)
                
                st.metric("Shipments", f"{shipments:,.0f}")
                st.metric("Revenue", f"${revenue:,.2f}")
        
        st.markdown("---")
        
        # Root Cause Analysis
        st.header("🔍 Root Cause Analysis - Controllable Delays")
        
        # Filter for late deliveries with controllable causes
        late_controllables = df_filtered[
            (df_filtered['Is_Controllable'] == True) & 
            (df_filtered['On_Time'] == False)
        ]
        
        if len(late_controllables) > 0:
            # Group by QC NAME to find top causes
            qc_analysis = late_controllables['QC NAME'].value_counts().head(15)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Pareto chart of delay causes
                fig_pareto = go.Figure()
                
                # Calculate cumulative percentage
                cumulative_percent = (qc_analysis.cumsum() / qc_analysis.sum() * 100).round(1)
                
                fig_pareto.add_trace(go.Bar(
                    x=qc_analysis.index,
                    y=qc_analysis.values,
                    name='Frequency',
                    marker_color='lightblue'
                ))
                
                fig_pareto.add_trace(go.Scatter(
                    x=qc_analysis.index,
                    y=cumulative_percent.values,
                    name='Cumulative %',
                    yaxis='y2',
                    line=dict(color='red', width=3),
                    marker=dict(size=8)
                ))
                
                fig_pareto.update_layout(
                    title="Top Controllable Delay Causes (Pareto Analysis)",
                    xaxis_title="Delay Cause",
                    yaxis=dict(title="Frequency"),
                    yaxis2=dict(title="Cumulative %", overlaying='y', side='right', range=[0, 105]),
                    hovermode='x unified',
                    height=400,
                    xaxis_tickangle=-45
                )
                
                st.plotly_chart(fig_pareto, use_container_width=True)
            
            with col2:
                # Category breakdown
                categories = {
                    'Agent Related': late_controllables[late_controllables['QC NAME'].str.contains('Agt|Agent', case=False, na=False)].shape[0],
                    'Customs Related': late_controllables[late_controllables['QC NAME'].str.contains('Customs', case=False, na=False)].shape[0],
                    'Warehouse Related': late_controllables[late_controllables['QC NAME'].str.contains('Warehouse|W/House', case=False, na=False)].shape[0]
                }
                
                fig_pie = px.pie(
                    values=list(categories.values()),
                    names=list(categories.keys()),
                    title="Delay Categories Distribution",
                    hole=0.4
                )
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                fig_pie.update_layout(height=400)
                st.plotly_chart(fig_pie, use_container_width=True)
        
        st.markdown("---")
        
        # Executive Recommendations
        st.header("💡 Executive Recommendations")
        
        # Calculate insights for recommendations
        avg_gross_otp = gross_otp
        avg_net_otp = net_otp
        gap_to_target = otp_target - avg_gross_otp
        
        # Find worst performing account
        if len(account_metrics) > 0:
            worst_account = account_metrics.nsmallest(1, 'OTP_%').index[0]
            worst_account_otp = account_metrics.loc[worst_account, 'OTP_%']
        
        # Generate recommendations based on data
        recommendations = []
        
        if avg_gross_otp < otp_target:
            recommendations.append(f"• **Immediate Action Required**: Gross OTP at {avg_gross_otp:.1f}% is {abs(gap_to_target):.1f}% below the {otp_target}% target.")
        
        if avg_net_otp < avg_gross_otp - 5:
            recommendations.append(f"• **Focus on Controllables**: Net OTP ({avg_net_otp:.1f}%) significantly trails Gross OTP, indicating internal process improvements needed.")
        
        if len(late_controllables) > 0:
            top_cause = qc_analysis.index[0] if len(qc_analysis) > 0 else "Unknown"
            recommendations.append(f"• **Address Primary Delay Cause**: '{top_cause}' accounts for {qc_analysis.iloc[0]:,} delays.")
        
        if 'worst_account' in locals():
            recommendations.append(f"• **Account Attention**: '{worst_account}' requires immediate intervention with OTP at {worst_account_otp:.1f}%.")
        
        # Display recommendations
        for rec in recommendations:
            st.markdown(rec)
        
        # Download section
        st.markdown("---")
        st.header("📥 Export Reports")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Prepare summary data for export
            summary_data = {
                'Metric': ['Gross OTP %', 'Net OTP %', 'Total Shipments', 'Total Revenue', 'Total Pieces'],
                'Value': [f"{gross_otp:.1f}%", f"{net_otp:.1f}%", total_shipments, f"${total_charges:,.2f}", total_pieces]
            }
            summary_df = pd.DataFrame(summary_data)
            
            csv = summary_df.to_csv(index=False)
            st.download_button(
                label="📊 Download Summary Report",
                data=csv,
                file_name=f"OTP_Summary_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        with col2:
            # Monthly metrics for export
            if 'monthly_metrics' in locals():
                csv_monthly = monthly_metrics[['Month_Display', 'Gross_OTP', 'Net_OTP', 'Pieces', 'Total_Charges']].to_csv(index=False)
                st.download_button(
                    label="📈 Download Monthly Trends",
                    data=csv_monthly,
                    file_name=f"OTP_Monthly_Trends_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
        
        with col3:
            # Account metrics for export
            csv_accounts = account_metrics.to_csv()
            st.download_button(
                label="💼 Download Account Report",
                data=csv_accounts,
                file_name=f"Account_Performance_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

else:
    # Landing page when no data is loaded
    st.info("📤 Please upload the Excel data file using the sidebar to begin analysis.")
    
    # Instructions
    with st.expander("📖 Dashboard Overview & Instructions"):
        st.markdown("""
        ### Welcome to the Executive OTP Performance Dashboard
        
        This professional dashboard provides comprehensive insights into On-Time Performance (OTP) metrics for top management decision-making.
        
        **Key Features:**
        - **Gross OTP**: Overall on-time performance across all shipments
        - **Net OTP**: Performance for controllable factors (Agent, Customs, Warehouse)
        - **Month-on-Month Analysis**: Track performance trends over time
        - **Account Performance**: Identify top performers and areas needing attention
        - **Country Analysis**: Performance breakdown for DE, IT, and IL markets
        - **Root Cause Analysis**: Identify primary causes of delays
        
        **How to Use:**
        1. Upload your gvExportData Excel file using the sidebar
        2. Apply filters as needed (date range, countries)
        3. Review the automated analysis and visualizations
        4. Export reports for further analysis or presentation
        
        **Data Requirements:**
        - Excel file with shipment data
        - Required columns: PU CTRY, QC NAME, POD DATE/TIME, TOTAL CHARGES, PIECES, SHIPPER NAME
        - Data will be automatically filtered for DE, IT, and IL countries
        
        **OTP Calculation:**
        - Deliveries are considered on-time when POD DATE/TIME ≤ Updated Delivery Date
        - Target OTP is set at 95% (adjustable in sidebar)
        - Controllables include delays related to Agents, Customs, and Warehouse operations
        """)
    
    # Sample metrics display (demo mode)
    st.markdown("---")
    st.subheader("🎯 Sample Dashboard Preview")
    
    demo_col1, demo_col2, demo_col3, demo_col4 = st.columns(4)
    
    with demo_col1:
        st.metric("Gross OTP", "94.5%", "-0.5%", delta_color="inverse")
    with demo_col2:
        st.metric("Net OTP", "96.2%", "+1.2%")
    with demo_col3:
        st.metric("Total Shipments", "12,450", "850 controllables")
    with demo_col4:
        st.metric("Total Revenue", "$1.2M", "3,200 pieces")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 12px;'>
    <p>Executive OTP Performance Dashboard v1.0 | Powered by Streamlit | © 2024</p>
    <p>For technical support or questions, please contact the Operations Excellence Team</p>
</div>
""", unsafe_allow_html=True)
