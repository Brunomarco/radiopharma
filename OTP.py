# app.py
import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ---------------- Page & Style ----------------
st.set_page_config(
    page_title="Executive OTP — Gross vs Net (Controllables)",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)
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

# ---------------- Config ----------------
OTP_TARGET = 95
SCOPE_PU = {"DE", "IT", "IL"}  # PU CTRY filter
# Controllables in QC NAME (case-insensitive)
CTRL_REGEX = re.compile(r"\b(agent|del\s*agt|delivery\s*agent|customs|warehouse|w/house)\b", re.I)

# ---------------- Helpers ----------------
def _excel_to_dt(s: pd.Series) -> pd.Series:
    """Robust datetime: parse strings; if many NaT remain, try Excel serials."""
    out = pd.to_datetime(s, errors="coerce")
    if out.isna().mean() > 0.5:
        num = pd.to_numeric(s, errors="coerce")
        out2 = pd.to_datetime("1899-12-30") + pd.to_timedelta(num, unit="D")
        out = out.where(~out.isna(), out2)
    return out

def _get_target_series(df: pd.DataFrame) -> pd.Series | None:
    """Prefer UPD DEL; fallback QDT."""
    if "UPD DEL" in df.columns and df["UPD DEL"].notna().any():
        return df["UPD DEL"]
    if "QDT" in df.columns:
        return df["QDT"]
    return None

@st.cache_data(show_spinner=False)
def read_file(uploaded) -> pd.DataFrame:
    name = uploaded.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded)
    return pd.read_excel(uploaded)  # openpyxl

@st.cache_data(show_spinner=False)
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to 440-BILLED + DE/IT/IL, build dates, flags, month key."""
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

    # Actual & target dates
    d["_adate"] = _excel_to_dt(d["POD DATE/TIME"])
    tgt = _get_target_series(d)
    d["_target"] = _excel_to_dt(tgt) if tgt is not None else pd.NaT

    # Month
    d["Month_Display"] = d["_adate"].dt.strftime("%b %Y")
    d["Month_Sort"] = pd.to_datetime(d["Month_Display"], format="%b %Y", errors="coerce")

    # QC controllable flag from QC NAME
    if "QC NAME" in d.columns:
        d["QC_NAME_CLEAN"] = d["QC NAME"].astype(str)
        d["Is_Controllable"] = d["QC_NAME_CLEAN"].str.contains(CTRL_REGEX, na=False)
    else:
        d["QC_NAME_CLEAN"] = ""
        d["Is_Controllable"] = False

    # On-time (gross)
    ok = d["_adate"].notna() & d["_target"].notna()
    d["On_Time_Gross"] = False
    d.loc[ok, "On_Time_Gross"] = d.loc[ok, "_adate"] <= d.loc[ok, "_target"]

    # Net marks NON-controllable lates as on-time
    d["Late"] = ~d["On_Time_Gross"]
    d["On_Time_Net"] = d["On_Time_Gross"] | (d["Late"] & ~d["Is_Controllable"])

    return d

def calculate_otp(df: pd.DataFrame) -> tuple[float, float]:
    """Gross/Net OTP from preprocessed df (Net >= Gross by construction)."""
    valid = df.dropna(subset=["_adate", "_target"])
    if valid.empty:
        return 0.0, 0.0
    gross = valid["On_Time_Gross"].mean() * 100.0
    net = valid["On_Time_Net"].mean() * 100.0
    if net < gross:  # numerical safety
        net = gross
    return round(gross, 1), round(net, 1)

@st.cache_data(show_spinner=False)
def monthly_otp(df: pd.DataFrame) -> pd.DataFrame:
    """Gross & Net OTP per month."""
    valid = df.dropna(subset=["_adate", "_target"])
    if valid.empty:
        return pd.DataFrame(columns=["Month_Display","Gross_OTP","Net_OTP","Gross_On","Gross_Tot","Net_On","Net_Tot","Month_Sort"])

    g = valid.groupby("Month_Display", as_index=False).agg(
        Gross_On=("On_Time_Gross", "sum"),
        Gross_Tot=("On_Time_Gross", "count")
    )
    n = valid.groupby("Month_Display", as_index=False).agg(
        Net_On=("On_Time_Net", "sum"),
        Net_Tot=("On_Time_Net", "count")
    )
    m = g.merge(n, on="Month_Display", how="left")
    m["Gross_OTP"] = (m["Gross_On"] / m["Gross_Tot"] * 100).round(1)
    m["Net_OTP"]   = (m["Net_On"] / m["Net_Tot"] * 100).round(1)
    m["Month_Sort"] = pd.to_datetime(m["Month_Display"], format="%b %Y", errors="coerce")
    return m.sort_values("Month_Sort")

# ---------------- Sidebar ----------------
with st.sidebar:
    uploaded = st.file_uploader("Upload Excel (.xlsx) or CSV", type=["xlsx", "csv"])
    otp_target = st.number_input("OTP Target (%)", min_value=0, max_value=100, value=OTP_TARGET, step=1)

if not uploaded:
    st.info("👆 Upload your Excel/CSV to compute OTP (filters: DE/IT/IL & 440-BILLED).")
    st.stop()

# ---------------- Pipeline ----------------
try:
    raw = read_file(uploaded)
except Exception as e:
    st.error(f"Could not read file: {e}")
    st.stop()

df = preprocess(raw)
if df.empty:
    st.error("No rows after filtering or required columns missing. Needed: PU CTRY, STATUS, POD DATE/TIME (+ UPD DEL or QDT).")
    st.stop()

gross_otp, net_otp = calculate_otp(df)
m = monthly_otp(df)

# ---------------- KPI Row ----------------
st.markdown("### 🎯 Key OTP Metrics")
col4, col5, col6 = st.columns(3)
with col4:
    gross_display = f"{gross_otp:.1f}%"
    delta_display = f"{gross_otp-otp_target:.1f}%"
    st.metric("✅ OTP Gross", gross_display, delta=delta_display if gross_otp else None)
with col5:
    net_display = f"{net_otp:.1f}%"
    delta_net = f"+{net_otp-gross_otp:.1f}%" if net_otp > gross_otp else f"{net_otp-gross_otp:.1f}%"
    st.metric("🎯 OTP Net", net_display, delta=delta_net if net_otp != gross_otp else None)
with col6:
    improvement_potential = net_otp - gross_otp
    st.metric("📈 Improvement", f"{improvement_potential:.1f}%", help="Potential uplift when excluding non-controllable issues")

st.markdown("---")

# ---------------- Row: Trends & Analysis ----------------
c1, c2 = st.columns(2)

# A) Monthly Net OTP (line) over Volume bars (Net_Tot) — like your screenshot
with c1:
    st.subheader("📈 Monthly Controllable OTP Trend")
    if not m.empty:
        fig = go.Figure()
        # Bars = volume of rows used to compute NET (same as total valid rows)
        fig.add_trace(go.Bar(
            x=m["Month_Display"], y=m["Net_Tot"],
            name="Volume", marker_color="#cbd5e1", yaxis="y"
        ))
        # Line = Net OTP %
        fig.add_trace(go.Scatter(
            x=m["Month_Display"], y=m["Net_OTP"],
            name="Net OTP %", mode="lines+markers",
            line=dict(color="#1f77b4", width=3), yaxis="y2"
        ))
        # Target on secondary axis
        fig.add_hline(y=otp_target, line_dash="dash", line_color="red", yref="y2")
        fig.update_layout(
            height=400, hovermode="x unified",
            yaxis=dict(title="Volume", side="left"),
            yaxis2=dict(title="Net OTP %", overlaying="y", side="right", range=[0, 105]),
            legend=dict(orientation="h", yanchor="bottom", y=-0.25, x=0)
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No monthly data available.")

# B) Performance Analysis bar (Gross vs Net vs Impact)
with c2:
    st.subheader("🎯 OTP Performance Analysis")
    otp_breakdown = pd.DataFrame({
        "Category": ["Gross OTP", "Net OTP", "Controllable Impact"],
        "Value": [gross_otp, net_otp, net_otp - gross_otp],
        "Type": ["Actual", "Adjusted", "Opportunity"]
    })
    fig_otp = go.Figure()
    colors = ["#3b82f6", "#10b981", "#fbbf24"]
    for i, row in otp_breakdown.iterrows():
        fig_otp.add_trace(go.Bar(
            x=[row["Value"]], y=[row["Category"]],
            orientation="h", name=row["Category"],
            marker_color=colors[i],
            text=f"{row['Value']:.1f}%", textposition="outside",
            showlegend=False
        ))
    fig_otp.update_layout(
        height=400, xaxis_title="Percentage (%)", yaxis_title="",
        xaxis=dict(range=[0, 105]), barmode="overlay"
    )
    fig_otp.add_vline(x=otp_target, line_dash="dash", line_color="red", annotation_text=f"Target {otp_target}%")
    st.plotly_chart(fig_otp, use_container_width=True)

st.markdown("---")

# ---------------- QC NAME listing & split ----------------
st.subheader("🔧 QC NAME — Controllable vs Non-Controllable")
if "QC_NAME_CLEAN" in df.columns:
    qc_counts = df["QC_NAME_CLEAN"].replace("", np.nan).dropna().value_counts().reset_index()
    qc_counts.columns = ["QC Name", "Count"]
    qc_counts["Control Type"] = np.where(
        qc_counts["QC Name"].str.contains(CTRL_REGEX, na=False),
        "Controllable", "Non-Controllable"
    )

    # Show both combined and split tabs
    tab_all, tab_ctrl, tab_non = st.tabs(["All QC Names", "Controllable", "Non-Controllable"])
    with tab_all:
        st.dataframe(qc_counts, use_container_width=True)
    with tab_ctrl:
        st.dataframe(qc_counts[qc_counts["Control Type"] == "Controllable"], use_container_width=True)
    with tab_non:
        st.dataframe(qc_counts[qc_counts["Control Type"] == "Non-Controllable"], use_container_width=True)

    # Download
    st.download_button(
        "📥 Download QC Classification CSV",
        qc_counts.to_csv(index=False).encode("utf-8"),
        file_name="qc_name_classification.csv",
        mime="text/csv"
    )
else:
    st.info("No QC NAME column found.")

st.caption("Gross: all shipments with POD ≤ target. Net: only controllable lates (Agent / Del Agt / Delivery agent / Customs / Warehouse / W/house) count against OTP. Base filters: PU CTRY ∈ {DE, IT, IL} and STATUS = 440-BILLED.")
