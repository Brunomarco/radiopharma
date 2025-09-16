# OTP-only, fast, minimal dependencies
import re
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Executive OTP (Fast)", layout="wide")

OTP_TARGET = 95.0
SCOPE_PU = {"DE", "IT", "IL"}
CTRL_KEYWORDS = ["agent", "agt", "customs", "warehouse", "w/house", "delivery agent"]

st.title("Executive OTP — Fast, Minimal")
st.caption("Scope: PU CTRY ∈ {DE, IT, IL} AND STATUS = 440-BILLED. OTP based on Actual Delivery ≤ Target.")

# ---------- Helpers ----------
def excel_to_datetime(s: pd.Series) -> pd.Series:
    out = pd.to_datetime(s, errors="coerce")
    if out.isna().mean() > 0.5:
        # Fallback for Excel serials
        numeric = pd.to_numeric(s, errors="coerce")
        out2 = pd.to_datetime("1899-12-30") + pd.to_timedelta(numeric, unit="D")
        out = out.where(~out.isna(), out2)
    return out

@st.cache_data(show_spinner=False)
def read_uploaded(file) -> pd.DataFrame:
    name = file.name.lower()
    if name.endswith(".csv"):
        # Read only the columns we need, case-insensitive map after read
        df = pd.read_csv(file)
    else:
        # Openpyxl required for xlsx (keep in requirements if you need xlsx)
        try:
            df = pd.read_excel(
                file,
                # If you know exact headers, uncomment and adjust:
                # usecols=["PU CTRY","STATUS","POD DATE/TIME","UPD DEL","QDT","QC NAME"]
            )
        except Exception:
            df = pd.read_excel(file)  # fallback
    return df

def guess(cols, candidates):
    low = {c.lower(): c for c in cols}
    for c in candidates:
        if c.lower() in low:
            return low[c.lower()]
    return None

@st.cache_data(show_spinner=False)
def preprocess(df: pd.DataFrame,
               pu_col: str, status_col: str,
               adate_col: str, target_col: str | None,
               qc_col: str | None,
               ctrl_kw: list[str]) -> pd.DataFrame:
    d = df.copy()

    # Normalize early for speed
    d["_pu"] = d[pu_col].astype(str).str.strip()
    d["_status"] = d[status_col].astype(str).str.strip().str.lower()

    # Scope filter first
    d = d[d["_pu"].isin(SCOPE_PU)]
    d = d[d["_status"] == "440-billed"]
    if d.empty:
        return d

    # Dates
    d["_adate"] = excel_to_datetime(d[adate_col])
    d["_target"] = excel_to_datetime(d[target_col]) if target_col else pd.NaT
    d["_month"] = d["_adate"].dt.to_period("M").dt.to_timestamp()

    # Controllables
    if qc_col:
        pattern = "|".join(re.escape(k) for k in ctrl_kw) if ctrl_kw else r"$."
        d["_is_ctrl"] = d[qc_col].astype(str).str.lower().str.contains(pattern, regex=True, na=False)
    else:
        d["_is_ctrl"] = False

    # OTP flag
    d["_on_time"] = False
    mask = d["_adate"].notna() & d["_target"].notna()
    d.loc[mask, "_on_time"] = d.loc[mask, "_adate"] <= d.loc[mask, "_target"]

    return d

@st.cache_data(show_spinner=False)
def compute_otp_monthly(d: pd.DataFrame):
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

def status_cls(pct, target=OTP_TARGET):
    if pd.isna(pct): return "bad"
    if pct >= target: return "ok"
    if pct >= target - 5: return "warn"
    return "bad"

st.markdown("""
<style>
.kpi-card {padding:1rem;border:1px solid #e6e6e6;border-radius:14px}
.kpi-title {font-size:.9rem;color:#5a5a5a}
.kpi-value {font-size:1.6rem;font-weight:700}
.ok{color:#0b8a42}.warn{color:#b26a00}.bad{color:#b00020}
</style>
""", unsafe_allow_html=True)

# ---------- UI ----------
with st.sidebar:
    st.header("Upload")
    up = st.file_uploader("Upload .xlsx or .csv", type=["xlsx", "csv"])
    st.caption("Faster if you upload CSV.")

if not up:
    st.info("Upload a file to begin.")
    st.stop()

# Read
df = read_uploaded(up)
if df.empty:
    st.error("Uploaded file is empty.")
    st.stop()

cols = list(df.columns)

# Mapping (auto-guessed)
pu_col     = guess(cols, ["PU CTRY", "PU Country", "Pickup Country", "PU Country Code"]) or st.selectbox("PU CTRY", cols)
status_col = guess(cols, ["STATUS", "Status"]) or st.selectbox("STATUS", cols)
adate_col  = guess(cols, ["POD DATE/TIME", "Actual Delivery Date", "Delivery Actual", "POD Date"]) or st.selectbox("Actual Delivery", cols)
target_col_guess = guess(cols, ["UPD DEL", "QDT", "Promised Date", "Target Delivery"])
qc_col_guess     = guess(cols, ["QC NAME", "QC name", "QC", "QC Category", "Exception Category"])

with st.sidebar:
    st.header("Column Mapping")
    pu_col = st.selectbox("Pickup Country (PU)", options=cols, index=cols.index(pu_col) if pu_col in cols else 0)
    status_col = st.selectbox("STATUS", options=cols, index=cols.index(status_col) if status_col in cols else 0)
    adate_col = st.selectbox("Actual Delivery (POD)", options=cols, index=cols.index(adate_col) if adate_col in cols else 0)
    target_col = st.selectbox("Target/Promised Delivery", options=["— none —"] + cols,
                              index=0 if not target_col_guess else (["— none —"] + cols).index(target_col_guess))
    qc_col = st.selectbox("QC Name (for controllables)", options=["— none —"] + cols,
                          index=0 if not qc_col_guess else (["— none —"] + cols).index(qc_col_guess))
    ctrl_txt = st.text_input("Controllable keywords", value="agent,agt,customs,warehouse,w/house,delivery agent")
    otp_target = st.number_input("OTP Target %", 0, 100, int(OTP_TARGET), 1)

target_col = None if target_col == "— none —" else target_col
qc_col = None if qc_col == "— none —" else qc_col
ctrl_kw = [w.strip().lower() for w in ctrl_txt.split(",") if w.strip()]

# Preprocess + compute
d = preprocess(df, pu_col, status_col, adate_col, target_col, qc_col, ctrl_kw)
if d.empty:
    st.error("No rows after filtering (PU in DE/IT/IL & STATUS=440-BILLED). Check mappings/data.")
    st.stop()

gross, net = compute_otp_monthly(d)

# KPIs
latest = gross["_month"].max() if not gross.empty else None
gross_latest = float(gross.loc[gross["_month"]==latest, "Gross_OTP_%"].iloc[0]) if latest is not None else np.nan

latest_net = net["_month"].max() if not net.empty else None
net_latest = float(net.loc[net["_month"]==latest_net, "Net_OTP_%"].iloc[0]) if latest_net is not None else np.nan

c1, c2 = st.columns(2)
with c1:
    txt = "" if pd.isna(gross_latest) else f"{gross_latest:.1f}%"
    st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
    st.markdown('<div class="kpi-title">Gross OTP (latest month)</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="kpi-value {status_cls(gross_latest, otp_target)}">{txt}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="kpi-title">Objective: {otp_target:.0f}%</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with c2:
    txt = "" if pd.isna(net_latest) else f"{net_latest:.1f}%"
    st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
    st.markdown('<div class="kpi-title">Net OTP (Controllables, latest month)</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="kpi-value">{txt}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="kpi-title">QC includes: {", ".join(ctrl_kw) if ctrl_kw else "—"}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

# Trend charts (native Streamlit charts)
st.subheader("OTP Trend — Gross (Actual Delivery Month)")
if not gross.empty:
    gross_chart = gross.set_index("_month")[["Gross_OTP_%"]]
    st.line_chart(gross_chart)
else:
    st.info("No Gross OTP data.")

st.subheader("OTP Trend — Net (Controllables)")
if not net.empty:
    net_chart = net.set_index("_month")[["Net_OTP_%"]]
    st.line_chart(net_chart)
else:
    st.info("No Net OTP data.")

# Tables + downloads
def to_csv_bytes(df_): return df_.to_csv(index=False).encode("utf-8")

st.subheader("Monthly OTP — Gross")
if not gross.empty:
    gross_out = gross.rename(columns={"_month":"Month","On_Time":"On_Time_Count","Total":"Total_Count"})
    st.dataframe(gross_out, use_container_width=True)
    st.download_button("Download Gross OTP CSV", data=to_csv_bytes(gross_out), file_name="otp_gross_by_month.csv")
else:
    st.info("No rows.")

st.subheader("Monthly OTP — Net (Controllables)")
if not net.empty:
    net_out = net.rename(columns={"_month":"Month","On_Time":"On_Time_Count","Total":"Total_Count"})
    st.dataframe(net_out, use_container_width=True)
    st.download_button("Download Net OTP CSV", data=to_csv_bytes(net_out), file_name="otp_net_by_month.csv")
else:
    st.info("No controllable rows.")
