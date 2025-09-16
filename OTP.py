import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Executive OTP & QC Dashboard", layout="wide")

OTP_TARGET_DEFAULT = 95.0
SCOPE_PU = {"DE", "IT", "IL"}

# -------- Controllables mapping based on your QC NAME list --------
# CONTROLLABLE (Agent / Customs / Delivery Agent):
# - Del Agt-Late del, Del Agt-Late del-TFC, Delivery agent not available
# - Customs delay, Customs delay-FDA Hold
# NON-CONTROLLABLE:
# - Airline-* (slow offload, FLT delay, misrouted, failure to load, no show)
# - Consignee-Driver waiting at delivery
# - Customer-Shipment not ready, Shipment not ready

_CONTROLLABLE_PATTERNS = [
    r"\bdel\s*agt\b",                # Del Agt-...
    r"\bdelivery\s*agent\b",         # Delivery agent...
    r"\bcustoms\b",                  # Customs delay...
    r"\bfda\s*hold\b",               # FDA hold under customs
    # If you add warehouse-type codes later, extend with r"\bwarehouse\b|\bw/house\b"
]
_NONCONTROLLABLE_PATTERNS = [
    r"\bairline\b",                  # Airline-...
    r"\bconsignee\b",                # Consignee-...
    r"\bcustomer\b",                 # Customer-...
    r"(^|\b)shipment\s*not\s*ready\b" # Shipment not ready (standalone)
]

_ctrl_regex = re.compile("|".join(_CONTROLLABLE_PATTERNS), flags=re.I)
_nonctrl_regex = re.compile("|".join(_NONCONTROLLABLE_PATTERNS), flags=re.I)

def is_controllable(qc: str) -> bool:
    s = (qc or "").strip()
    if not s:
        return False
    if _ctrl_regex.search(s):
        return True
    if _nonctrl_regex.search(s):
        return False
    # Default conservative: not controllable unless clearly Agent/Customs/etc.
    return False

# ---------------- Helpers ----------------
def excel_to_datetime(series: pd.Series) -> pd.Series:
    """Robust datetime: parse; if many NaT remain, try Excel serial conversion."""
    out = pd.to_datetime(series, errors="coerce")
    if out.isna().mean() > 0.5:
        num = pd.to_numeric(series, errors="coerce")
        out2 = pd.to_datetime("1899-12-30") + pd.to_timedelta(num, unit="D")
        out = out.where(~out.isna(), out2)
    return out

@st.cache_data(show_spinner=False)
def read_uploaded(file) -> pd.DataFrame:
    name = file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(file)
    return pd.read_excel(file)  # openpyxl used (see requirements)

@st.cache_data(show_spinner=False)
def preprocess(df: pd.DataFrame, target_pick: str) -> pd.DataFrame:
    # Required columns
    required = ["PU CTRY", "STATUS", "POD DATE/TIME"]
    if not all(c in df.columns for c in required):
        return pd.DataFrame()

    d = df.copy()

    # Scope filters
    d["_pu"] = d["PU CTRY"].astype(str).str.strip()
    d["_status"] = d["STATUS"].astype(str).str.strip().str.lower()
    d = d[d["_pu"].isin(SCOPE_PU)]
    d = d[d["_status"] == "440-billed"]
    if d.empty:
        return d

    # Datetime fields
    d["_adate"] = excel_to_datetime(d["POD DATE/TIME"])

    tgt_col = None
    if target_pick == "UPD DEL" and "UPD DEL" in d.columns:
        tgt_col = "UPD DEL"
    elif target_pick == "QDT" and "QDT" in d.columns:
        tgt_col = "QDT"
    else:
        # Auto: prefer UPD DEL, else QDT
        if "UPD DEL" in d.columns:
            tgt_col = "UPD DEL"
        elif "QDT" in d.columns:
            tgt_col = "QDT"

    d["_target"] = excel_to_datetime(d[tgt_col]) if tgt_col else pd.NaT

    # Month fields
    d["_month"] = d["_adate"].dt.to_period("M").dt.to_timestamp()
    d["Month_Display"] = d["_adate"].dt.strftime("%b %Y")
    d["Month_Sort"] = pd.to_datetime(d["Month_Display"], format="%b %Y", errors="coerce")

    # Controllables
    d["Is_Controllable"] = d.get("QC NAME", pd.Series("", index=d.index)).astype(str).apply(is_controllable)

    # OTP flag: POD <= Target
    ok = d["_adate"].notna() & d["_target"].notna()
    d["On_Time"] = False
    d.loc[ok, "On_Time"] = d.loc[ok, "_adate"] <= d.loc[ok, "_target"]

    return d

@st.cache_data(show_spinner=False)
def monthly_otp(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    m = df.groupby("Month_Display", as_index=False).agg(
        On_Time=("On_Time", "sum"),
        Total=("On_Time", "count")
    )
    dn = df[df["Is_Controllable"] == True]
    if not dn.empty:
        n = dn.groupby("Month_Display", as_index=False).agg(
            Net_On_Time=("On_Time", "sum"),
            Net_Total=("On_Time", "count")
        )
        m = m.merge(n, on="Month_Display", how="left")
    else:
        m["Net_On_Time"] = np.nan
        m["Net_Total"] = np.nan

    m["Gross_OTP"] = (m["On_Time"] / m["Total"] * 100).round(1)
    m["Net_OTP"] = (m["Net_On_Time"] / m["Net_Total"] * 100).round(1)
    m["Month_Sort"] = pd.to_datetime(m["Month_Display"], format="%b %Y", errors="coerce")
    return m.sort_values("Month_Sort")

# ---------------- UI: Upload & Settings ----------------
st.title("Executive OTP & Quality Control Dashboard")
st.caption("Filters: PU CTRY ∈ {DE, IT, IL} and STATUS = 440-BILLED")

with st.sidebar:
    up = st.file_uploader("Upload Excel (.xlsx) or CSV", type=["xlsx", "csv"])
    target_pick = st.selectbox("Target/Promised Date column",
                               options=["Auto (UPD DEL→QDT)", "UPD DEL", "QDT"])
    otp_target = st.number_input("OTP Target (%)", min_value=0, max_value=100,
                                 value=int(OTP_TARGET_DEFAULT), step=1)

if not up:
    st.info("Upload your Excel/CSV file to begin.")
    st.stop()

df_raw = read_uploaded(up)
df = preprocess(df_raw, target_pick)
if df.empty:
    st.error("No rows after filtering or required columns missing. "
             "Ensure the file has PU CTRY, STATUS, POD DATE/TIME, and UPD DEL or QDT.")
    st.stop()

# ------------- Overall KPIs -------------
total_shipments = len(df)
gross_otp_overall = (df["On_Time"].sum() / total_shipments * 100) if total_shipments else np.nan
ctrl_df = df[df["Is_Controllable"] == True]
net_otp_overall = (ctrl_df["On_Time"].sum() / len(ctrl_df) * 100) if len(ctrl_df) else np.nan

monthly = monthly_otp(df)

st.markdown("---")
c1, c2 = st.columns(2)

with c1:
    st.subheader("📊 OTP Summary (Latest Month)")
    latest_m = monthly["Month_Sort"].max() if not monthly.empty else None
    if latest_m is not None:
        last = monthly.loc[monthly["Month_Sort"] == latest_m].iloc[0]
        gross_last = last["Gross_OTP"]
        net_last = last["Net_OTP"]

        st.metric("Gross OTP (Latest)",
                  f"{gross_last:.1f}%" if pd.notna(gross_last) else "—",
                  f"{(gross_last - otp_target):.1f}% vs target" if pd.notna(gross_last) else None,
                  delta_color="normal" if pd.notna(gross_last) and gross_last >= otp_target else "inverse")

        st.metric("Net OTP (Latest)",
                  f"{net_last:.1f}%" if pd.notna(net_last) else "—",
                  f"{(net_last - otp_target):.1f}% vs target" if pd.notna(net_last) else None,
                  delta_color="normal" if pd.notna(net_last) and net_last >= otp_target else "inverse")
    else:
        st.info("No monthly OTP could be computed; check delivery/target dates.")

with c2:
    # 🎯 OTP Performance Analysis (Gross vs Net vs Controllable Impact)
    st.subheader("🎯 OTP Performance Analysis")
    impact = (net_otp_overall - gross_otp_overall) if (pd.notna(net_otp_overall) and pd.notna(gross_otp_overall)) else np.nan
    otp_breakdown = pd.DataFrame({
        "Category": ["Gross OTP", "Net OTP", "Controllable Impact"],
        "Value": [
            gross_otp_overall if pd.notna(gross_otp_overall) else 0.0,
            net_otp_overall if pd.notna(net_otp_overall) else 0.0,
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
        height=400,
        xaxis_title="Percentage (%)",
        yaxis_title="",
        xaxis=dict(range=[0, 105]),
        barmode='overlay'
    )
    fig_otp.add_vline(x=otp_target, line_dash="dash", line_color="red")
    st.plotly_chart(fig_otp, use_container_width=True)

# ------------- Trend Chart -------------
st.markdown("---")
st.subheader("📈 OTP Trend by Month")
if not monthly.empty:
    fig_trend = make_subplots(specs=[[{"secondary_y": False}]])
    fig_trend.add_trace(go.Scatter(
        x=monthly["Month_Display"], y=monthly["Gross_OTP"],
        mode="lines+markers", name="Gross OTP",
        line=dict(width=3, color="#3b82f6")
    ))
    if monthly["Net_OTP"].notna().any():
        fig_trend.add_trace(go.Scatter(
            x=monthly["Month_Display"], y=monthly["Net_OTP"],
            mode="lines+markers", name="Net OTP",
            line=dict(width=3, color="#10b981")
        ))
    fig_trend.add_hline(y=otp_target, line_dash="dash", line_color="red",
                        annotation_text=f"Target {otp_target:.0f}%")
    fig_trend.update_layout(height=420, yaxis_title="OTP (%)", xaxis_title="Month", hovermode="x unified")
    st.plotly_chart(fig_trend, use_container_width=True)
else:
    st.info("No monthly OTP available to chart.")

# ------------- QC Analysis -------------
st.markdown("---")
st.subheader("🔧 Quality Control Issues Analysis")
if ("QCCODE" in df.columns) and ("QC NAME" in df.columns):
    qc_data = df[df["QCCODE"].notna()].copy()
    if len(qc_data) > 0:
        # Tag issues by controllability using classifier
        qc_data["Issue_Type"] = np.where(
            qc_data["QC NAME"].astype(str).apply(is_controllable),
            "Controllable (Internal)",
            "Non-Controllable (External)"
        )

        colA, colB = st.columns(2)
        with colA:
            st.markdown("#### Distribution by Control Type")
            type_counts = qc_data["Issue_Type"].value_counts()
            fig_qc_pie = px.pie(
                values=type_counts.values,
                names=type_counts.index,
                hole=0.4,
                color=type_counts.index,
                color_discrete_map={
                    "Controllable (Internal)": "#10b981",
                    "Non-Controllable (External)": "#ef4444"
                }
            )
            fig_qc_pie.update_traces(textposition='inside', textinfo='percent+label')
            fig_qc_pie.update_layout(height=350)
            st.plotly_chart(fig_qc_pie, use_container_width=True)

            c1a, c1b = st.columns(2)
            with c1a:
                controllable_pct = (type_counts.get("Controllable (Internal)", 0) / len(qc_data) * 100) if len(qc_data) else 0
                st.metric("🟢 Controllable", f"{controllable_pct:.1f}%")
            with c1b:
                non_controllable_pct = (type_counts.get("Non-Controllable (External)", 0) / len(qc_data) * 100) if len(qc_data) else 0
                st.metric("🔴 Non-Controllable", f"{non_controllable_pct:.1f}%")

        with colB:
            st.markdown("#### Top 10 Quality Control Issues")
            qc_counts = qc_data.groupby(["QC NAME", "Issue_Type"]).size().reset_index(name="Count")
            qc_counts = qc_counts.sort_values("Count", ascending=False).head(10)
            qc_counts_sorted = qc_counts.sort_values("Count", ascending=True)

            fig_qc_bar = px.bar(
                qc_counts_sorted,
                x="Count", y="QC NAME",
                orientation="h",
                color="Issue_Type",
                text="Count",
                color_discrete_map={
                    "Controllable (Internal)": "#10b981",
                    "Non-Controllable (External)": "#ef4444"
                }
            )
            fig_qc_bar.update_traces(texttemplate='%{text}', textposition='outside')
            fig_qc_bar.update_layout(
                height=350,
                xaxis_title="Occurrences",
                yaxis_title="",
                legend_title="Issue Type",
                legend=dict(orientation="h", yanchor="bottom", y=-0.35, x=0),
                margin=dict(b=80)
            )
            st.plotly_chart(fig_qc_bar, use_container_width=True)
    else:
        st.info("No quality control issues found (no QCCODE rows).")
else:
    st.info("No QC data available (missing QCCODE and/or QC NAME columns).")

st.caption("All metrics are computed from the uploaded file only. Scope and status filters applied first.")
