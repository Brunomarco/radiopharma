import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Executive OTP Dashboard — Excel-based", layout="wide")

OTP_TARGET_DEFAULT = 95.0
SCOPE_PU = ["DE", "IT", "IL"]
CTRL_KEYWORDS_DEFAULT = ["agent", "agt", "customs", "warehouse", "w/house", "delivery agent"]

st.markdown("""
<style>
.kpi-card {padding: 1rem; border: 1px solid #e6e6e6; border-radius: 14px;}
.kpi-title {font-size: 0.9rem; color: #5a5a5a;}
.kpi-value {font-size: 1.6rem; font-weight: 700;}
.ok {color:#0b8a42;} .warn {color:#b26a00;} .bad {color:#b00020;}
</style>
""", unsafe_allow_html=True)

st.title("Executive OTP Dashboard — OTP Only (Excel-based)")
st.caption("Scope: PU CTRY {DE, IT, IL} AND STATUS = 440-BILLED. Gross vs Net (Controllables) by month (Actual Delivery).")

def excel_to_datetime(s: pd.Series) -> pd.Series:
    out = pd.to_datetime(s, errors="coerce")
    if out.isna().mean() > 0.5:
        try:
            numeric = pd.to_numeric(s, errors="coerce")
            out2 = pd.to_datetime("1899-12-30") + pd.to_timedelta(numeric, unit="D")
            out = out.where(~out.isna(), out2)
        except Exception:
            pass
    return out

@st.cache_data(show_spinner=False)
def read_uploaded(file) -> pd.DataFrame:
    name = file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(file)
    return pd.read_excel(file)

def guess_column(cols, candidates):
    lowered = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lowered:
            return lowered[cand.lower()]
    return None

@st.cache_data(show_spinner=False)
def preprocess_for_otp(df: pd.DataFrame,
                       pu_ctry_col: str,
                       status_col: str,
                       adate_col: str,
                       target_col: str,
                       qc_col: str,
                       ctrl_keywords: list):
    d = df.copy()

    # Scope: PU CTRY in DE/IT/IL
    d["_pu"] = d[pu_ctry_col].astype(str).str.strip()
    d = d[d["_pu"].isin(SCOPE_PU)]

    # STATUS = 440-BILLED (case-insensitive / whitespace safe)
    d["_status"] = d[status_col].astype(str).str.strip().str.lower()
    d = d[d["_status"] == "440-billed"]

    if d.empty:
        return d

    # Dates
    d["_adate"] = excel_to_datetime(d[adate_col])
    d["_target"] = excel_to_datetime(d[target_col]) if target_col else pd.NaT

    # Month key
    d["_month"] = d["_adate"].dt.to_period("M").dt.to_timestamp()

    # Controllables
    if qc_col:
        qc_series = d[qc_col].astype(str).str.lower()
        pattern = "|".join(re.escape(k) for k in ctrl_keywords) if ctrl_keywords else r"$."
        d["_is_ctrl"] = qc_series.str.contains(pattern, regex=True, na=False)
    else:
        d["_is_ctrl"] = False

    # On-time flag
    d["_on_time"] = False
    has_dates = d["_adate"].notna() & d["_target"].notna()
    d.loc[has_dates, "_on_time"] = d.loc[has_dates, "_adate"] <= d.loc[has_dates, "_target"]

    return d

@st.cache_data(show_spinner=False)
def compute_monthly_otp(d: pd.DataFrame):
    d = d.dropna(subset=["_month"])
    gross = d.groupby("_month", as_index=False).agg(
        On_Time=("_on_time", "sum"),
        Total=("_on_time", "count"),
    )
    gross["Gross_OTP_%"] = (gross["On_Time"] / gross["Total"] * 100).round(1)

    dn = d[d["_is_ctrl"]]
    if dn.empty:
        net = pd.DataFrame(columns=["_month", "On_Time", "Total", "Net_OTP_%"])
    else:
        net = dn.groupby("_month", as_index=False).agg(
            On_Time=("_on_time", "sum"),
            Total=("_on_time", "count"),
        )
        net["Net_OTP_%"] = (net["On_Time"] / net["Total"] * 100).round(1)
    return gross, net

def cls_for_target(pct, target):
    if pd.isna(pct): return "bad"
    if pct >= target: return "ok"
    if pct >= target - 5: return "warn"
    return "bad"

# Sidebar: Upload & Mapping
with st.sidebar:
    st.header("Upload")
    up = st.file_uploader("Upload Excel (.xlsx) or CSV", type=["xlsx", "csv"])

    st.header("Column Mapping (auto-guessed; adjust if needed)")
if not up:
    st.info("Please upload your Excel/CSV file.")
    st.stop()

try:
    raw = read_uploaded(up)
except Exception as e:
    st.error(f"Could not read file: {e}")
    st.stop()

if raw.empty:
    st.error("Uploaded file has no rows.")
    st.stop()

cols = list(raw.columns)
pu_guess     = guess_column(cols, ["PU CTRY", "PU Country", "Pickup Country", "PU Country Code"])
status_guess = guess_column(cols, ["STATUS", "Status"])
adate_guess  = guess_column(cols, ["POD DATE/TIME", "Actual Delivery Date", "Delivery Actual", "POD Date"])
target_guess = guess_column(cols, ["UPD DEL", "QDT", "Promised Date", "Target Delivery"])
qc_guess     = guess_column(cols, ["QC NAME", "QC name", "QC", "QC Category", "Exception Category"])

with st.sidebar:
    pu_col     = st.selectbox("Pickup Country (PU CTRY)", [pu_guess] + [c for c in cols if c != pu_guess] if pu_guess else cols)
    status_col = st.selectbox("STATUS", [status_guess] + [c for c in cols if c != status_guess] if status_guess else cols)
    adate_col  = st.selectbox("Actual Delivery (POD / ADATE)", [adate_guess] + [c for c in cols if c != adate_guess] if adate_guess else cols)
    target_col = st.selectbox("Target/Promised Delivery", ["— none —"] + cols, index=(0 if not target_guess else (["— none —"] + cols).index(target_guess)))
    qc_col     = st.selectbox("QC Name (for controllables)", ["— none —"] + cols, index=(0 if not qc_guess else (["— none —"] + cols).index(qc_guess)))

    st.markdown("---")
    otp_target = st.number_input("OTP Target %", min_value=0, max_value=100, value=int(OTP_TARGET_DEFAULT), step=1)
    ctrl_txt   = st.text_input("Controllable QC keywords", value=",".join(CTRL_KEYWORDS_DEFAULT))

if not pu_col or not status_col or not adate_col:
    st.error("Please map PU CTRY, STATUS, and Actual Delivery columns.")
    st.stop()

ctrl_keywords = [w.strip().lower() for w in ctrl_txt.split(",") if w.strip()]
target_sel = None if target_col == "— none —" else target_col
qc_sel = None if qc_col == "— none —" else qc_col

d = preprocess_for_otp(raw, pu_col, status_col, adate_col, target_sel, qc_sel, ctrl_keywords)
if d.empty:
    st.error("No rows after filtering (PU CTRY in DE/IT/IL and STATUS = 440-BILLED). Check mappings and data.")
    st.stop()

gross, net = compute_monthly_otp(d)

# KPIs (latest month)
latest_m = gross["_month"].max() if not gross.empty else None
gross_latest = float(gross.loc[gross["_month"] == latest_m, "Gross_OTP_%"].iloc[0]) if latest_m is not None else np.nan

latest_n_m = net["_month"].max() if not net.empty else None
net_latest = float(net.loc[net["_month"] == latest_n_m, "Net_OTP_%"].iloc[0]) if latest_n_m is not None else np.nan

c1, c2 = st.columns(2)
with c1:
    cl = cls_for_target(gross_latest, otp_target)
    txt = "" if pd.isna(gross_latest) else f"{gross_latest:.1f}%"
    st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
    st.markdown('<div class="kpi-title">Gross OTP (latest month)</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="kpi-value {cl}">{txt}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="kpi-title">Objective: {otp_target:.0f}%</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with c2:
    txt = "" if pd.isna(net_latest) else f"{net_latest:.1f}%"
    st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
    st.markdown('<div class="kpi-title">Net OTP (Controllables, latest month)</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="kpi-value">{txt}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="kpi-title">QC includes: {", ".join(ctrl_keywords) if ctrl_keywords else "—"}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

# Trend chart
st.subheader("OTP Trend by Month (Actual Delivery Date)")
fig = go.Figure()
if not gross.empty:
    fig.add_trace(go.Scatter(x=gross["_month"], y=gross["Gross_OTP_%"], mode="lines+markers", name="Gross OTP"))
if not net.empty:
    fig.add_trace(go.Scatter(x=net["_month"], y=net["Net_OTP_%"], mode="lines+markers", name="Net OTP (Controllables)"))
fig.add_hline(y=otp_target, line_dash="dash", line_color="red", annotation_text=f"Target {otp_target:.0f}%")
fig.update_layout(yaxis_title="OTP (%)", xaxis_title="Month", hovermode="x unified", height=420)
st.plotly_chart(fig, use_container_width=True)

# Tables + downloads
def to_csv_bytes(df_: pd.DataFrame) -> bytes:
    return df_.to_csv(index=False).encode("utf-8")

st.subheader("Monthly OTP — Gross (STATUS 440-BILLED)")
if gross.empty:
    st.info("No Gross OTP data available.")
else:
    gross_out = gross.rename(columns={"_month": "Month", "On_Time": "On_Time_Count", "Total": "Total_Count"})
    st.dataframe(gross_out, use_container_width=True)
    st.download_button("Download Gross OTP CSV", data=to_csv_bytes(gross_out), file_name="otp_gross_by_month.csv")

st.subheader("Monthly OTP — Net (Controllables, STATUS 440-BILLED)")
if net.empty:
    st.info("No Net OTP data available (no controllable rows).")
else:
    net_out = net.rename(columns={"_month": "Month", "On_Time": "On_Time_Count", "Total": "Total_Count"})
    st.dataframe(net_out, use_container_width=True)
    st.download_button("Download Net OTP CSV", data=to_csv_bytes(net_out), file_name="otp_net_by_month.csv")

st.caption("Filters: PU CTRY ∈ {DE, IT, IL} and STATUS = 440-BILLED. OTP: POD ≤ Target (UPD DEL or QDT). Net OTP: QC contains Agent/Customs/Warehouse.")
