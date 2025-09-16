import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px  # used for optional pies/tables if you extend later

# ---------------- Page & Style ----------------
st.set_page_config(
    page_title="Radiopharma OTP",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Colors & typography close to your screenshots
NAVY = "#0b1f44"       # bars / gauge fill
GOLD = "#f0b429"       # line / accents
SLATE = "#334155"      # axis labels
CARD_BG = "#ffffff"

st.markdown("""
<style>
.main {padding: 0rem 1rem;}
h1 {color:#0b1f44;font-weight:800;letter-spacing:.2px;border-bottom:3px solid #2ecc71;padding-bottom:10px;}
h2 {color:#0b1f44;font-weight:700;margin-top:1.5rem;margin-bottom:.75rem;}
.kpi {background:#fff;border:1px solid #e6e6e6;border-radius:14px;padding:14px;}
.k-num {font-size:36px;font-weight:800;color:#0b1f44;line-height:1.0;}
.k-cap {font-size:13px;color:#6b7280;margin-top:4px;}
</style>
""", unsafe_allow_html=True)

st.title("Radiopharma OTP")

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

def _kfmt(n: float) -> str:
    if pd.isna(n): return ""
    if n >= 1000: return f"{n/1000:.1f}K"
    return f"{n:.0f}"

def make_semi_gauge(title: str, value: float) -> go.Figure:
    """Semi-donut gauge with centered percentage."""
    v = max(0, min(100, 0 if pd.isna(value) else value))
    fig = go.Figure()
    fig.add_trace(go.Pie(
        values=[v, 100 - v, 100],
        hole=0.75,
        sort=False,
        direction="clockwise",
        rotation=180,                 # semicircle
        textinfo="none",
        marker=dict(colors=[NAVY, "#d1d5db", "rgba(0,0,0,0)"]),
        showlegend=False
    ))
    fig.add_annotation(text=f"{v:.2f}%", x=0.5, y=0.4, showarrow=False,
                       font=dict(size=20, color=NAVY, family="Arial Black"))
    fig.update_layout(
        margin=dict(l=10, r=10, t=28, b=0),
        height=160,
        annotations=[dict(text=title, x=0.5, y=1.15, showarrow=False,
                          font=dict(size=14, color=SLATE, family="Arial"))]
    )
    return fig

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

    # Month (human-friendly, full; will not be cut)
    d["Month_Display"] = d["_adate"].dt.strftime("%b %Y")   # e.g., Aug 2025
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

def calc_summary(df: pd.DataFrame):
    """Gross/Net %, plus late exception split for tiles."""
    valid = df.dropna(subset=["_adate", "_target"])
    total_ship = int(len(valid))
    gross = valid["On_Time_Gross"].mean() * 100 if total_ship else np.nan
    net = valid["On_Time_Net"].mean() * 100 if total_ship else np.nan
    if pd.notna(gross) and pd.notna(net) and net < gross:
        net = gross  # numerical safety

    # Exceptions = Late (gross) rows among valid
    late_df = valid[valid["Late"]]
    exceptions = int(len(late_df))
    controllables = int(late_df["Is_Controllable"].sum())
    uncontrollables = int(exceptions - controllables)

    return (round(gross, 2) if pd.notna(gross) else np.nan,
            round(net, 2) if pd.notna(net) else np.nan,
            total_ship, exceptions, controllables, uncontrollables)

@st.cache_data(show_spinner=False)
def monthly_net_otp(df: pd.DataFrame) -> pd.DataFrame:
    """Monthly volume + Net OTP (%)."""
    valid = df.dropna(subset=["_adate", "_target"])
    if valid.empty:
        return pd.DataFrame(columns=["Month_Display","Volume","Net_OTP","Month_Sort"])

    mv = valid.groupby("Month_Display", as_index=False).agg(
        Volume=("On_Time_Net", "count"),
        Net_OTP=("On_Time_Net", "mean")
    )
    mv["Net_OTP"] = (mv["Net_OTP"] * 100).round(2)
    mv["Month_Sort"] = pd.to_datetime(mv["Month_Display"], format="%b %Y", errors="coerce")
    mv = mv.sort_values("Month_Sort")
    return mv

# ---------------- Sidebar ----------------
with st.sidebar:
    up = st.file_uploader("Upload Excel (.xlsx) or CSV", type=["xlsx", "csv"])
    otp_target = st.number_input("OTP Target (%)", min_value=0, max_value=100, value=OTP_TARGET, step=1)

if not up:
    st.info("👆 Upload your Excel/CSV to compute OTP (filters: DE/IT/IL & 440-BILLED).")
    st.stop()

# ---------------- Pipeline ----------------
try:
    raw = read_file(up)
except Exception as e:
    st.error(f"Could not read file: {e}")
    st.stop()

df = preprocess(raw)
if df.empty:
    st.error("No rows after filtering or required columns missing. Need: PU CTRY, STATUS, POD DATE/TIME (+ UPD DEL or QDT).")
    st.stop()

gross_otp, net_otp, total_ship, exceptions, controllables, uncontrollables = calc_summary(df)
mv = monthly_net_otp(df)

# ---------------- KPI column (left) and Gauges (right) ----------------
left, right = st.columns([1, 1.4])

with left:
    st.markdown(f'<div class="kpi"><div class="k-num">{total_ship:,}</div><div class="k-cap">Volume</div></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="kpi"><div class="k-num">{exceptions:,}</div><div class="k-cap">Exceptions Count</div></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="kpi"><div class="k-num">{controllables:,}</div><div class="k-cap">Controllables</div></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="kpi"><div class="k-num">{uncontrollables:,}</div><div class="k-cap">Uncontrollables</div></div>', unsafe_allow_html=True)

with right:
    g1, g2, g3 = st.columns(3)
    with g1:
        st.plotly_chart(make_semi_gauge("Adjusted OTP", max(gross_otp, net_otp)), use_container_width=True, config={"displayModeBar": False})
    with g2:
        st.plotly_chart(make_semi_gauge("Controllable OTP", net_otp), use_container_width=True, config={"displayModeBar": False})
    with g3:
        st.plotly_chart(make_semi_gauge("Raw OTP", gross_otp), use_container_width=True, config={"displayModeBar": False})

st.markdown("---")

# ---------------- "Controllable OTP by Volume" (Net OTP line + Volume bars) ----------------
st.subheader("Controllable OTP by Volume")

if not mv.empty:
    # Robust labels & data types for Plotly
    x_labels = mv["Month_Display"].astype(str).tolist()
    vol_vals = mv["Volume"].astype(float).tolist()
    net_vals = mv["Net_OTP"].astype(float).tolist()

    fig = go.Figure()

    # Bars = monthly volume (navy) with K-labels on top
    fig.add_trace(go.Bar(
        x=x_labels,
        y=vol_vals,
        name="Volume",
        marker_color=NAVY,
        text=[_kfmt(v) for v in vol_vals],
        textposition="outside",
        textfont=dict(size=12, color="#4b5563"),
        yaxis="y"
    ))

    # Line = Net OTP % (gold) with % labels
    fig.add_trace(go.Scatter(
        x=x_labels,
        y=net_vals,
        name="Controllable OTP",
        mode="lines+markers+text",
        line=dict(color=GOLD, width=3),
        marker=dict(size=8),
        text=[f"{v:.2f}%" for v in net_vals],
        textposition="middle center",
        textfont=dict(size=11, color="#111827"),
        yaxis="y2"
    ))

    # Target line on secondary axis (version-safe)
    try:
        fig.add_hline(y=float(otp_target), line_dash="dash", line_color="red", yref="y2")
    except Exception:
        fig.add_shape(
            type="line", x0=-0.5, x1=len(x_labels)-0.5,
            y0=float(otp_target), y1=float(otp_target),
            xref="x", yref="y2", line=dict(color="red", dash="dash")
        )

    fig.update_layout(
        height=460,
        hovermode="x unified",
        plot_bgcolor="white",
        margin=dict(l=40, r=40, t=40, b=80),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.0),
        xaxis=dict(
            title="",
            tickangle=-30,
            tickmode="array",
            tickvals=x_labels,
            ticktext=x_labels,
            automargin=True
        ),
        yaxis=dict(
            title="Volume",
            side="left",
            gridcolor="#f3f4f6"
        ),
        yaxis2=dict(
            title="Controllable OTP",
            overlaying="y",
            side="right",
            range=[0, 105]
        ),
        barmode="overlay"
    )

    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No monthly data available.")

st.markdown("---")

# ---------------- QC NAME split (Controllable vs Non-Controllable) ----------------
st.subheader("QC NAME — Classification")
if "QC_NAME_CLEAN" in df.columns:
    qc = df[df["QC_NAME_CLEAN"].ne("")].copy()
    qc["Control Type"] = np.where(qc["QC_NAME_CLEAN"].str.contains(CTRL_REGEX, na=False),
                                  "Controllable", "Non-Controllable")

    qc_all = qc["QC_NAME_CLEAN"].value_counts().reset_index()
    qc_all.columns = ["QC Name", "Count"]
    qc_all["Control Type"] = np.where(qc_all["QC Name"].str.contains(CTRL_REGEX, na=False),
                                      "Controllable", "Non-Controllable")

    late_only = df[df["Late"] & df["QC_NAME_CLEAN"].ne("")].copy()
    late_only["Control Type"] = np.where(late_only["QC_NAME_CLEAN"].str.contains(CTRL_REGEX, na=False),
                                         "Controllable", "Non-Controllable")
    qc_late = late_only["QC_NAME_CLEAN"].value_counts().reset_index()
    qc_late.columns = ["QC Name", "Count"]
    qc_late["Control Type"] = np.where(qc_late["QC Name"].str.contains(CTRL_REGEX, na=False),
                                       "Controllable", "Non-Controllable")

    tab_all, tab_late, tab_ctrl, tab_non = st.tabs(["All", "Late Only", "Controllable", "Non-Controllable"])
    with tab_all:
        st.dataframe(qc_all, use_container_width=True)
    with tab_late:
        st.dataframe(qc_late, use_container_width=True)
    with tab_ctrl:
        st.dataframe(qc_all[qc_all["Control Type"] == "Controllable"], use_container_width=True)
    with tab_non:
        st.dataframe(qc_all[qc_all["Control Type"] == "Non-Controllable"], use_container_width=True)

    st.download_button(
        "📥 Download QC NAME (All) CSV",
        qc_all.to_csv(index=False).encode("utf-8"),
        file_name="qc_name_all.csv",
        mime="text/csv"
    )
    st.download_button(
        "📥 Download QC NAME (Late Only) CSV",
        qc_late.to_csv(index=False).encode("utf-8"),
        file_name="qc_name_late_only.csv",
        mime="text/csv"
    )
else:
    st.info("No QC NAME column found.")

st.caption("Gross: POD ≤ target (UPD DEL → QDT). Net: only controllable lates (Agent / Del Agt / Delivery agent / Customs / Warehouse / W/house) count against OTP. Filters: PU CTRY ∈ {DE, IT, IL}, STATUS = 440-BILLED.")
