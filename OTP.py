import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------- Page ----------
st.set_page_config(page_title="Executive OTP — Simple", page_icon="📊", layout="wide", initial_sidebar_state="collapsed")

# Minimal, clean styling
st.markdown("""
<style>
.main {padding: 0rem 1rem;}
h1 {color:#1f2937;font-weight:700;border-bottom:3px solid #3b82f6;padding-bottom:10px;}
.kpi {padding:1rem;border:1px solid #e6e6e6;border-radius:12px}
.kpi-title {font-size:.9rem;color:#6b7280}
.kpi-val {font-size:1.6rem;font-weight:700}
</style>
""", unsafe_allow_html=True)

st.title("📊 Executive OTP — Gross vs Net (Controllables)")

# ---------- Config ----------
OTP_TARGET_DEFAULT = 95
SCOPE_PU = {"DE", "IT", "IL"}  # PU CTRY filter
# Controllables (Agent / Delivery agent / Customs / Warehouse / W/house / FDA hold)
CTRL_REGEX = re.compile(r"\b(del\s*agt|delivery\s*agent|customs|warehouse|w/house|fda\s*hold)\b", re.I)

# ---------- Helpers ----------
def excel_to_datetime(s: pd.Series) -> pd.Series:
    """Robust datetime: parse; if many NaT, try Excel serial conversion."""
    out = pd.to_datetime(s, errors="coerce")
    if out.isna().mean() > 0.5:
        num = pd.to_numeric(s, errors="coerce")
        out2 = pd.to_datetime("1899-12-30") + pd.to_timedelta(num, unit="D")
        out = out.where(~out.isna(), out2)
    return out

@st.cache_data(show_spinner=False)
def read_file(uploaded) -> pd.DataFrame:
    name = uploaded.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded)
    return pd.read_excel(uploaded)  # uses openpyxl

@st.cache_data(show_spinner=False)
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Filter and compute flags needed for OTP."""
    required = ["PU CTRY", "STATUS", "POD DATE/TIME"]
    if not all(c in df.columns for c in required):
        return pd.DataFrame()

    d = df.copy()

    # Scope filters: PU CTRY and 440-BILLED
    d["_pu"] = d["PU CTRY"].astype(str).str.strip()
    d["_status"] = d["STATUS"].astype(str).str.strip().str.lower()
    d = d[d["_pu"].isin(SCOPE_PU)]
    d = d[d["_status"] == "440-billed"]
    if d.empty:
        return d

    # Dates: Actual Delivery & Target (prefer UPD DEL, fallback QDT)
    d["_adate"] = excel_to_datetime(d["POD DATE/TIME"])
    if "UPD DEL" in d.columns and d["UPD DEL"].notna().any():
        d["_target"] = excel_to_datetime(d["UPD DEL"])
    elif "QDT" in d.columns:
        d["_target"] = excel_to_datetime(d["QDT"])
    else:
        d["_target"] = pd.NaT

    # Month keys
    d["_month"] = d["_adate"].dt.to_period("M").dt.to_timestamp()
    d["Month_Display"] = d["_adate"].dt.strftime("%b %Y")
    d["Month_Sort"] = pd.to_datetime(d["Month_Display"], format="%b %Y", errors="coerce")

    # Controllables from QC NAME
    if "QC NAME" in d.columns:
        d["Is_Controllable"] = d["QC NAME"].astype(str).str.contains(CTRL_REGEX, na=False)
    else:
        d["Is_Controllable"] = False

    # On-time rule: POD ≤ Target
    ok = d["_adate"].notna() & d["_target"].notna()
    d["On_Time"] = False
    d.loc[ok, "On_Time"] = d.loc[ok, "_adate"] <= d.loc[ok, "_target"]

    return d

@st.cache_data(show_spinner=False)
def monthly_otp(df: pd.DataFrame) -> pd.DataFrame:
    """Gross & Net OTP per month."""
    if df.empty:
        return pd.DataFrame(columns=["Month_Display","Gross_OTP","Net_OTP","On_Time","Total","Net_On_Time","Net_Total","Month_Sort"])

    m = df.groupby("Month_Display", as_index=False).agg(
        On_Time=("On_Time", "sum"),
        Total=("On_Time", "count")
    )
    ctrl = df[df["Is_Controllable"] == True]
    if not ctrl.empty:
        n = ctrl.groupby("Month_Display", as_index=False).agg(
            Net_On_Time=("On_Time", "sum"),
            Net_Total=("On_Time", "count")
        )
        m = m.merge(n, on="Month_Display", how="left")
    else:
        m["Net_On_Time"] = np.nan
        m["Net_Total"]   = np.nan

    m["Gross_OTP"] = (m["On_Time"] / m["Total"] * 100).round(1)
    m["Net_OTP"]   = (m["Net_On_Time"] / m["Net_Total"] * 100).round(1)
    m["Month_Sort"] = pd.to_datetime(m["Month_Display"], format="%b %Y", errors="coerce")
    return m.sort_values("Month_Sort")

# ---------- Sidebar ----------
with st.sidebar:
    up = st.file_uploader("Upload Excel (.xlsx) or CSV", type=["xlsx", "csv"])
    otp_target = st.number_input("OTP Target (%)", min_value=0, max_value=100, value=OTP_TARGET_DEFAULT, step=1)

if not up:
    st.info("👆 Upload your file to compute OTP (filters: DE/IT/IL & 440-BILLED).")
    st.stop()

# ---------- Pipeline ----------
try:
    raw = read_file(up)
except Exception as e:
    st.error(f"Could not read file: {e}")
    st.stop()

df = preprocess(raw)
if df.empty:
    st.error("No rows after filtering or required columns missing. "
             "Ensure columns: PU CTRY, STATUS, POD DATE/TIME (+ UPD DEL or QDT).")
    st.stop()

monthly = monthly_otp(df)

# Overall Gross/Net (across filtered data)
total = len(df)
gross_overall = (df["On_Time"].sum() / total * 100) if total else np.nan
ctrl_df = df[df["Is_Controllable"] == True]
net_overall = (ctrl_df["On_Time"].sum() / len(ctrl_df) * 100) if len(ctrl_df) else np.nan

# ---------- KPI ----------
st.markdown("### 🎯 Key OTP Metrics")
k1, k2 = st.columns(2)
with k1:
    st.markdown('<div class="kpi">', unsafe_allow_html=True)
    st.markdown('<div class="kpi-title">Gross OTP (All Shipments)</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="kpi-val">{"" if pd.isna(gross_overall) else f"{gross_overall:.1f}%"}'</
                f'div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
with k2:
    st.markdown('<div class="kpi">', unsafe_allow_html=True)
    st.markdown('<div class="kpi-title">Net OTP (Controllables)</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="kpi-val">{"" if pd.isna(net_overall) else f"{net_overall:.1f}%"}'</
                f'div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

# ---------- Row: OTP Analysis (your preferred col2 style) ----------
c1, c2 = st.columns(2)
with c1:
    st.subheader("📈 OTP Trend by Month")
    if not monthly.empty:
        fig = make_subplots(specs=[[{"secondary_y": False}]])
        fig.add_trace(go.Scatter(
            x=monthly["Month_Display"], y=monthly["Gross_OTP"],
            mode="lines+markers", name="Gross OTP", line=dict(width=3, color="#3b82f6")
        ))
        if monthly["Net_OTP"].notna().any():
            fig.add_trace(go.Scatter(
                x=monthly["Month_Display"], y=monthly["Net_OTP"],
                mode="lines+markers", name="Net OTP", line=dict(width=3, color="#10b981")
            ))
        fig.add_hline(y=otp_target, line_dash="dash", line_color="red",
                      annotation_text=f"Target {otp_target:.0f}%")
        fig.update_layout(height=380, yaxis_title="OTP (%)", xaxis_title="Month", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No monthly OTP available.")

with c2:
    st.subheader("🎯 OTP Performance Analysis")
    impact = (net_overall - gross_overall) if (pd.notna(net_overall) and pd.notna(gross_overall)) else np.nan
    otp_breakdown = pd.DataFrame({
        "Category": ["Gross OTP", "Net OTP", "Controllable Impact"],
        "Value": [
            gross_overall if pd.notna(gross_overall) else 0.0,
            net_overall if pd.notna(net_overall) else 0.0,
            impact if pd.notna(impact) else 0.0
        ],
        "Type": ["Actual", "Adjusted", "Opportunity"]
    })

    fig_otp = go.Figure()
    colors = ['#3b82f6', '#10b981', '#fbbf24']
    for i, row in otp_breakdown.iterrows():
        fig_otp.add_trace(go.Bar(
            x=[row['Value']],
            y=[row['Category']],
            orientation='h',
            name=row['Category'],
            marker_color=colors[i],
            text=f"{row['Value']:.1f}%",
            textposition='outside',
            showlegend=False
        ))
    fig_otp.update_layout(
        height=380,
        xaxis_title="Percentage (%)",
        yaxis_title="",
        xaxis=dict(range=[0, 105]),
        barmode='overlay'
    )
    fig_otp.add_vline(x=otp_target, line_dash="dash", line_color="red")
    st.plotly_chart(fig_otp, use_container_width=True)

st.markdown("---")

# ---------- Table + Download ----------
st.subheader("Monthly OTP Table (Filtered: DE/IT/IL & 440-BILLED)")
monthly_out = monthly[["Month_Display","Gross_OTP","Net_OTP","On_Time","Total","Net_On_Time","Net_Total"]].rename(
    columns={
        "Month_Display":"Month",
        "On_Time":"On_Time_Count",
        "Total":"Total_Count",
        "Net_On_Time":"Net_On_Time_Count",
        "Net_Total":"Net_Total_Count"
    }
)
st.dataframe(monthly_out, use_container_width=True)

csv = monthly_out.to_csv(index=False).encode("utf-8")
st.download_button("📥 Download Monthly OTP CSV", data=csv, file_name="monthly_otp.csv", mime="text/csv")

st.caption("Definitions — Gross: all shipments. Net: controllables only (QC NAME contains Agent / Delivery agent / Customs / Warehouse / W/house / FDA Hold). OTP = POD ≤ target (UPD DEL; fallback QDT).")
