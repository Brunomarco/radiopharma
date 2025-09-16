import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ---------------- Page & Style ----------------
st.set_page_config(
    page_title="Radiopharma OTP",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Colors
NAVY = "#0b1f44"       # bars / gauge fill
GOLD = "#f0b429"       # net line
BLUE = "#1f77b4"       # gross line
GREEN = "#10b981"      # net line (trend)
SLATE = "#334155"
LIGHT_GRAY = "#e5e7eb"

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
CTRL_REGEX = re.compile(r"\b(agent|del\s*agt|delivery\s*agent|customs|warehouse|w/house)\b", re.I)  # QC controllables

# ---------------- Helpers ----------------
def _excel_to_dt(s: pd.Series) -> pd.Series:
    """Parse strings; if many NaT, try Excel serials."""
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
    """Semi-donut gauge with centered %."""
    v = max(0, min(100, 0 if pd.isna(value) else value))
    fig = go.Figure()
    fig.add_trace(go.Pie(
        values=[v, 100 - v, 100],
        hole=0.75,
        sort=False,
        direction="clockwise",
        rotation=180,
        textinfo="none",
        marker=dict(colors=[NAVY, "#d1d5db", "rgba(0,0,0,0)"]),
        showlegend=False
    ))
    fig.add_annotation(text=f"{v:.2f}%", x=0.5, y=0.5, xref="paper", yref="paper",
                       showarrow=False, font=dict(size=26, color=NAVY, family="Arial Black"))
    fig.add_annotation(text=title, x=0.5, y=1.18, xref="paper", yref="paper",
                       showarrow=False, font=dict(size=14, color=SLATE))
    fig.update_layout(margin=dict(l=10, r=10, t=36, b=0), height=180)
    return fig

@st.cache_data(show_spinner=False)
def read_file(uploaded) -> pd.DataFrame:
    name = uploaded.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded)
    return pd.read_excel(uploaded)  # openpyxl backend

@st.cache_data(show_spinner=False)
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters & flags:
      - PU CTRY in {DE, IT, IL} and STATUS=440-BILLED
      - Dates: POD (actual delivery), ACT PU, Target = UPD DEL → QDT
      - OTP: Gross = POD<=Target; Net = only controllable lates count
      - Month keys: POD and ACT PU for grouping
    """
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

    # Dates
    d["_adate"]  = _excel_to_dt(d["POD DATE/TIME"])                 # POD = actual delivery (OTP month)
    d["_pudate"] = _excel_to_dt(d["ACT PU"]) if "ACT PU" in d.columns else pd.NaT  # actual pickup (PIECES month)
    tgt = _get_target_series(d)
    d["_target"] = _excel_to_dt(tgt) if tgt is not None else pd.NaT

    # Month labels
    d["Month_POD_Display"] = d["_adate"].dt.strftime("%b %Y")
    d["Month_POD_Sort"]    = pd.to_datetime(d["Month_POD_Display"], format="%b %Y", errors="coerce")
    d["Month_PU_Display"]  = d["_pudate"].dt.strftime("%b %Y")
    d["Month_PU_Sort"]     = pd.to_datetime(d["Month_PU_Display"], format="%b %Y", errors="coerce")

    # QC controllable flag
    if "QC NAME" in d.columns:
        d["QC_NAME_CLEAN"] = d["QC NAME"].astype(str)
        d["Is_Controllable"] = d["QC_NAME_CLEAN"].str.contains(CTRL_REGEX, na=False)
    else:
        d["QC_NAME_CLEAN"] = ""
        d["Is_Controllable"] = False

    # OTP
    ok = d["_adate"].notna() & d["_target"].notna()
    d["On_Time_Gross"] = False
    d.loc[ok, "On_Time_Gross"] = d.loc[ok, "_adate"] <= d.loc[ok, "_target"]

    d["Late"] = ~d["On_Time_Gross"]
    d["On_Time_Net"] = d["On_Time_Gross"] | (d["Late"] & ~d["Is_Controllable"])

    # PIECES numeric
    if "PIECES" in d.columns:
        d["PIECES"] = pd.to_numeric(d["PIECES"], errors="coerce").fillna(0)
    else:
        d["PIECES"] = 0

    return d

def calc_summary(df: pd.DataFrame):
    """Gross%, Net%, total valid OTP shipments, exceptions, controllables, uncontrollables."""
    valid = df.dropna(subset=["_adate", "_target"])
    total_ship = int(len(valid))
    gross = valid["On_Time_Gross"].mean() * 100 if total_ship else np.nan
    net = valid["On_Time_Net"].mean() * 100 if total_ship else np.nan
    if pd.notna(gross) and pd.notna(net) and net < gross:
        net = gross  # safety

    late_df = valid[valid["Late"]]
    exceptions = int(len(late_df))
    controllables = int(late_df["Is_Controllable"].sum())
    uncontrollables = int(exceptions - controllables)

    return (round(gross, 2) if pd.notna(gross) else np.nan,
            round(net, 2) if pd.notna(net) else np.nan,
            total_ship, exceptions, controllables, uncontrollables)

@st.cache_data(show_spinner=False)
def monthly_otp_by_pod(df: pd.DataFrame) -> pd.DataFrame:
    """Gross & Net OTP by POD month (single source of truth for OTP across charts)."""
    valid = df.dropna(subset=["_adate", "_target"]).copy()
    if valid.empty:
        return pd.DataFrame(columns=["Month_Display","Gross_OTP","Net_OTP","Month_Sort"])

    g = valid.groupby("Month_POD_Display", as_index=False).agg(
        Gross_On=("On_Time_Gross", "sum"),
        Gross_Tot=("On_Time_Gross", "count"),
        Net_On=("On_Time_Net", "sum"),
        Net_Tot=("On_Time_Net", "count")
    )
    g["Gross_OTP"] = (g["Gross_On"] / g["Gross_Tot"] * 100).round(2)
    g["Net_OTP"]   = (g["Net_On"] / g["Net_Tot"] * 100).round(2)
    g["Month_Display"] = g["Month_POD_Display"]
    g["Month_Sort"]    = pd.to_datetime(g["Month_Display"], format="%b %Y", errors="coerce")
    return g[["Month_Display", "Gross_OTP", "Net_OTP", "Month_Sort"]].sort_values("Month_Sort")

@st.cache_data(show_spinner=False)
def monthly_volume_by_pod(df: pd.DataFrame) -> pd.DataFrame:
    """Order volume by POD month (count rows with POD)."""
    base = df.dropna(subset=["_adate"]).copy()
    if base.empty:
        return pd.DataFrame(columns=["Month_Display","Volume","Month_Sort"])
    v = base.groupby("Month_POD_Display", as_index=False).size()
    v.columns = ["Month_Display", "Volume"]
    v["Month_Sort"] = pd.to_datetime(v["Month_Display"], format="%b %Y", errors="coerce")
    return v.sort_values("Month_Sort")

@st.cache_data(show_spinner=False)
def monthly_pieces_by_pu(df: pd.DataFrame) -> pd.DataFrame:
    """PIECES summed by ACT PU month."""
    base = df.dropna(subset=["_pudate"]).copy()
    if base.empty:
        return pd.DataFrame(columns=["Month_Display","Pieces","Month_Sort"])
    p = base.groupby("Month_PU_Display", as_index=False)["PIECES"].sum()
    p.columns = ["Month_Display", "Pieces"]
    p["Month_Sort"] = pd.to_datetime(p["Month_Display"], format="%b %Y", errors="coerce")
    return p.sort_values("Month_Sort")

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

# Monthly frames (OTP always by POD)
otp_pod   = monthly_otp_by_pod(df)     # Gross & Net % by POD month
vol_pod   = monthly_volume_by_pod(df)  # Volume by POD month
pieces_pu = monthly_pieces_by_pu(df)   # Pieces by ACT PU month

# ---------------- KPI & Gauges ----------------
left, right = st.columns([1, 1.5])

with left:
    st.markdown(f'<div class="kpi"><div class="k-num">{total_ship:,}</div><div class="k-cap">Volume (valid for OTP)</div></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="kpi"><div class="k-num">{exceptions:,}</div><div class="k-cap">Exceptions (Gross Late)</div></div>', unsafe_allow_html=True)
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

with st.expander("How months are defined"):
    st.markdown("""
- **OTP (Gross & Net) month:** **Actual Delivery (POD)**. One source of truth, so OTP % per month **match across all charts**.
- **Volume (orders):** **POD** month.
- **PIECES:** **ACT PU** month. OTP line on that chart still comes from POD and is matched by month name.
    """)

st.markdown("---")

# ---------------- Chart 1: Controllable OTP by Volume (POD) ----------------
st.subheader("Controllable OTP by Volume (POD)")

if not vol_pod.empty:
    # Drop Month_Sort from right before merge to avoid suffixes, rebuild after
    mv = vol_pod.merge(otp_pod.drop(columns=["Month_Sort"], errors="ignore"),
                       on="Month_Display", how="left")
    mv["Month_Sort"] = pd.to_datetime(mv["Month_Display"], format="%b %Y", errors="coerce")
    mv = mv.sort_values("Month_Sort")

    x_labels = mv["Month_Display"].astype(str).tolist()
    vol_vals = mv["Volume"].astype(float).tolist()
    net_vals = mv["Net_OTP"].astype(float).tolist()

    fig = go.Figure()

    # Bars = monthly volume (navy) with labels
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

    # Line = Net OTP % (gold)
    fig.add_trace(go.Scatter(
        x=x_labels,
        y=net_vals,
        name="Controllable OTP",
        mode="lines+markers",
        line=dict(color=GOLD, width=3),
        marker=dict(size=8),
        yaxis="y2"
    ))

    # % labels — lift with pixel offset so months like Jun never hide
    for xi, yi in zip(x_labels, net_vals):
        if pd.notna(yi):
            fig.add_annotation(
                x=xi, y=yi, xref="x", yref="y2",
                yshift=20, text=f"{yi:.2f}%", showarrow=False,
                font=dict(size=12, color="#111827")
            )

    # Target line on secondary axis
    try:
        fig.add_hline(y=float(otp_target), line_dash="dash", line_color="red", yref="y2")
    except Exception:
        fig.add_shape(type="line", x0=-0.5, x1=len(x_labels)-0.5,
                      y0=float(otp_target), y1=float(otp_target),
                      xref="x", yref="y2", line=dict(color="red", dash="dash"))

    fig.update_layout(
        height=520,
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
        yaxis=dict(title="Volume (Orders)", side="left", gridcolor=LIGHT_GRAY),
        yaxis2=dict(title="Controllable OTP (%)", overlaying="y", side="right", range=[0, 130]),
        barmode="overlay"
    )

    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No monthly volume available.")

st.markdown("---")

# ---------------- Chart 1b: Controllable OTP by PIECES (PIECES by ACT PU, OTP by POD) ----------------
st.subheader("Controllable OTP by Pieces (PIECES by ACT PU, OTP by POD)")

if not pieces_pu.empty:
    mp = pieces_pu.merge(otp_pod.drop(columns=["Month_Sort"], errors="ignore"),
                         on="Month_Display", how="left")
    mp["Month_Sort"] = pd.to_datetime(mp["Month_Display"], format="%b %Y", errors="coerce")
    mp = mp.sort_values("Month_Sort")

    x_labels = mp["Month_Display"].astype(str).tolist()
    pieces_vals = mp["Pieces"].astype(float).tolist()
    net_vals_p = mp["Net_OTP"].astype(float).tolist()

    figp = go.Figure()

    # Bars = monthly PIECES (navy)
    figp.add_trace(go.Bar(
        x=x_labels,
        y=pieces_vals,
        name="Pieces (ACT PU)",
        marker_color=NAVY,
        text=[_kfmt(v) for v in pieces_vals],
        textposition="outside",
        textfont=dict(size=12, color="#4b5563"),
        yaxis="y"
    ))

    # Line = Net OTP % (gold; from POD)
    figp.add_trace(go.Scatter(
        x=x_labels,
        y=net_vals_p,
        name="Controllable OTP (POD)",
        mode="lines+markers",
        line=dict(color=GOLD, width=3),
        marker=dict(size=8),
        yaxis="y2"
    ))

    # % labels — lift with pixel offset (fixes August visibility, etc.)
    for xi, yi in zip(x_labels, net_vals_p):
        if pd.notna(yi):
            figp.add_annotation(
                x=xi, y=yi, xref="x", yref="y2",
                yshift=20, text=f"{yi:.2f}%", showarrow=False,
                font=dict(size=12, color="#111827")
            )

    # Target line
    try:
        figp.add_hline(y=float(otp_target), line_dash="dash", line_color="red", yref="y2")
    except Exception:
        figp.add_shape(type="line", x0=-0.5, x1=len(x_labels)-0.5,
                       y0=float(otp_target), y1=float(otp_target),
                       xref="x", yref="y2", line=dict(color="red", dash="dash"))

    figp.update_layout(
        height=520,
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
        yaxis=dict(title="Pieces (ACT PU)", side="left", gridcolor=LIGHT_GRAY),
        yaxis2=dict(title="Controllable OTP (%)", overlaying="y", side="right", range=[0, 130]),
        barmode="overlay"
    )

    st.plotly_chart(figp, use_container_width=True)
else:
    st.info("No monthly PIECES available (ACT PU).")

st.markdown("---")

# ---------------- Chart 2: Monthly OTP Trend (Gross vs Net by POD) ----------------
st.subheader("Monthly OTP Trend (Gross vs Net) — by Actual Delivery (POD)")

if not otp_pod.empty:
    x_labels = otp_pod["Month_Display"].astype(str).tolist()
    gross_vals = otp_pod["Gross_OTP"].astype(float).tolist()
    net_vals   = otp_pod["Net_OTP"].astype(float).tolist()

    fig2 = go.Figure()

    fig2.add_trace(go.Scatter(
        x=x_labels, y=gross_vals,
        mode="lines+markers",
        name="Gross OTP",
        line=dict(color=BLUE, width=3),
        marker=dict(size=7),
    ))
    fig2.add_trace(go.Scatter(
        x=x_labels, y=net_vals,
        mode="lines+markers",
        name="Net OTP",
        line=dict(color=GREEN, width=3),
        marker=dict(size=7),
    ))

    # % labels — raise with pixel offset so they never collide
    for xi, yi in zip(x_labels, gross_vals):
        if pd.notna(yi):
            fig2.add_annotation(x=xi, y=yi, xref="x", yref="y",
                                yshift=20, text=f"{yi:.2f}%", showarrow=False,
                                font=dict(size=12, color=BLUE))
    for xi, yi in zip(x_labels, net_vals):
        if pd.notna(yi):
            fig2.add_annotation(x=xi, y=yi, xref="x", yref="y",
                                yshift=20, text=f"{yi:.2f}%", showarrow=False,
                                font=dict(size=12, color=GREEN))

    # Target
    try:
        fig2.add_hline(y=float(otp_target), line_dash="dash", line_color="red")
    except Exception:
        fig2.add_shape(type="line", x0=-0.5, x1=len(x_labels)-0.5,
                       y0=float(otp_target), y1=float(otp_target),
                       xref="x", yref="y", line=dict(color="red", dash="dash"))

    fig2.update_layout(
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
        yaxis=dict(title="OTP (%)", range=[0, 130], gridcolor=LIGHT_GRAY)
    )
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("No monthly OTP trend available.")

st.caption(
    "Gross: POD ≤ target (UPD DEL → QDT). Net: only controllable lates (Agent / Del Agt / Delivery agent / Customs / Warehouse / W/house) count against OTP. "
    "Filters: PU CTRY ∈ {DE, IT, IL}, STATUS = 440-BILLED. OTP per month = POD; PIECES per month = ACT PU."
)
