import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ---------- Page ----------
st.set_page_config(
    page_title="Executive OTP Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Minimal styling
st.markdown("""
<style>
.main {padding: 0rem 1rem;}
h1 {color:#1f2937;font-weight:700;border-bottom:3px solid #3b82f6;padding-bottom:10px;}
.kpi {padding:1rem;border:1px solid #e6e6e6;border-radius:12px;margin-bottom:1rem;}
.kpi-title {font-size:.9rem;color:#6b7280}
.kpi-val {font-size:1.6rem;font-weight:700}
</style>
""", unsafe_allow_html=True)

st.title("📊 Executive OTP — Gross vs Net (Controllables)")

# ---------- Config ----------
OTP_TARGET_DEFAULT = 95
SCOPE_PU = {"DE", "IT", "IL"}
CTRL_REGEX = re.compile(r"\b(agent|del\s*agt|delivery\s*agent|customs|warehouse|w/house)\b", re.I)

# ---------- Helpers ----------
def excel_to_datetime(s):
    out = pd.to_datetime(s, errors="coerce")
    if out.isna().mean() > 0.5:
        num = pd.to_numeric(s, errors="coerce")
        out2 = pd.to_datetime("1899-12-30") + pd.to_timedelta(num, unit="D")
        out = out.where(~out.isna(), out2)
    return out

@st.cache_data(show_spinner=False)
def read_file(uploaded):
    if uploaded.name.lower().endswith(".csv"):
        return pd.read_csv(uploaded)
    return pd.read_excel(uploaded)

@st.cache_data(show_spinner=False)
def preprocess(df):
    required = ["PU CTRY", "STATUS", "POD DATE/TIME"]
    if not all(c in df.columns for c in required):
        return pd.DataFrame()

    d = df.copy()
    d["_pu"] = d["PU CTRY"].astype(str).str.strip()
    d["_status"] = d["STATUS"].astype(str).str.strip().str.lower()
    d = d[d["_pu"].isin(SCOPE_PU)]
    d = d[d["_status"] == "440-billed"]
    if d.empty:
        return d

    d["_adate"] = excel_to_datetime(d["POD DATE/TIME"])
    if "UPD DEL" in d.columns and d["UPD DEL"].notna().any():
        d["_target"] = excel_to_datetime(d["UPD DEL"])
    elif "QDT" in d.columns:
        d["_target"] = excel_to_datetime(d["QDT"])
    else:
        d["_target"] = pd.NaT

    d["Month_Display"] = d["_adate"].dt.strftime("%b %Y")
    d["Month_Sort"] = pd.to_datetime(d["Month_Display"], format="%b %Y", errors="coerce")

    if "QC NAME" in d.columns:
        d["QC_NAME_CLEAN"] = d["QC NAME"].astype(str)
        d["Is_Controllable"] = d["QC_NAME_CLEAN"].str.contains(CTRL_REGEX, na=False)
    else:
        d["QC_NAME_CLEAN"] = ""
        d["Is_Controllable"] = False

    ok = d["_adate"].notna() & d["_target"].notna()
    d["On_Time"] = False
    d.loc[ok, "On_Time"] = d.loc[ok, "_adate"] <= d.loc[ok, "_target"]
    return d

@st.cache_data(show_spinner=False)
def monthly_otp(df):
    if df.empty:
        return pd.DataFrame()

    m = df.groupby("Month_Display", as_index=False).agg(
        On_Time=("On_Time", "sum"),
        Total=("On_Time", "count")
    )
    ctrl = df[df["Is_Controllable"]]
    n = ctrl.groupby("Month_Display", as_index=False).agg(
        Net_On_Time=("On_Time", "sum"),
        Net_Total=("On_Time", "count")
    )
    m = m.merge(n, on="Month_Display", how="left")
    m["Gross_OTP"] = (m["On_Time"] / m["Total"] * 100).round(1)
    m["Net_OTP"] = (m["Net_On_Time"] / m["Net_Total"] * 100).round(1)
    m["Month_Sort"] = pd.to_datetime(m["Month_Display"], format="%b %Y")
    return m.sort_values("Month_Sort")

# ---------- Sidebar ----------
with st.sidebar:
    up = st.file_uploader("Upload Excel (.xlsx or .csv)", type=["xlsx", "csv"])
    otp_target = st.number_input("OTP Target (%)", 0, 100, OTP_TARGET_DEFAULT)

if not up:
    st.info("👆 Upload your file to compute OTP.")
    st.stop()

try:
    raw = read_file(up)
except Exception as e:
    st.error(f"Error reading file: {e}")
    st.stop()

df = preprocess(raw)
if df.empty:
    st.error("No rows after filtering (DE/IT/IL & 440-BILLED) or missing columns.")
    st.stop()

monthly = monthly_otp(df)

# ---------- KPIs ----------
total = len(df)
gross_overall = (df["On_Time"].sum() / total * 100) if total else np.nan
ctrl_df = df[df["Is_Controllable"]]
net_overall = (ctrl_df["On_Time"].sum() / len(ctrl_df) * 100) if len(ctrl_df) else np.nan
exceptions = total - df["On_Time"].sum()
controllable_count = ctrl_df.shape[0]
uncontrollable_count = total - controllable_count

st.markdown("### 📊 August Snapshot")
k1, k2, k3, k4 = st.columns(4)
k1.metric("📦 Volume", f"{total:,}")
k2.metric("⚠️ Exceptions", f"{exceptions:,}")
k3.metric("🟢 Controllables", f"{controllable_count:,}")
k4.metric("🔴 Uncontrollables", f"{uncontrollable_count:,}")

c1, c2, c3 = st.columns(3)
c1.metric("Raw OTP", f"{gross_overall:.2f}%" if pd.notna(gross_overall) else "N/A")
c2.metric("Controllable OTP", f"{net_overall:.2f}%" if pd.notna(net_overall) else "N/A")
c3.metric("Adjusted OTP", f"{max(gross_overall, net_overall):.2f}%" if pd.notna(gross_overall) else "N/A")

st.markdown("---")

# ---------- Monthly Net OTP Trend ----------
st.subheader("📈 Monthly Controllable OTP Trend")
if not monthly.empty:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=monthly["Month_Display"],
        y=monthly["Net_Total"],
        name="Volume",
        marker_color="#cbd5e1",
        yaxis="y"
    ))
    fig.add_trace(go.Scatter(
        x=monthly["Month_Display"],
        y=monthly["Net_OTP"],
        name="Net OTP %",
        mode="lines+markers",
        line=dict(color="#1f77b4", width=3),
        yaxis="y2"
    ))
    fig.add_hline(y=otp_target, line_dash="dash", line_color="red", yref="y2")
    fig.update_layout(
        barmode="overlay",
        yaxis=dict(title="Volume", side="left"),
        yaxis2=dict(title="Net OTP %", overlaying="y", side="right", range=[0, 105]),
        height=400,
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No monthly data to display.")

st.markdown("---")

# ---------- QC NAME Summary ----------
st.subheader("🔧 QC Name Classification")
qc_summary = df["QC_NAME_CLEAN"].value_counts().reset_index()
qc_summary.columns = ["QC Name", "Count"]
qc_summary["Control Type"] = np.where(
    qc_summary["QC Name"].str.contains(CTRL_REGEX, na=False),
    "Controllable",
    "Non-Controllable"
)
st.dataframe(qc_summary, use_container_width=True)
