import re
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

# Colors / styles
NAVY  = "#0b1f44"      # bars / gauge fill
GOLD  = "#f0b429"      # controllable/net line
BLUE  = "#1f77b4"      # gross line
GREEN = "#10b981"      # net (trend) alt
SLATE = "#334155"
GRID  = "#e5e7eb"

st.markdown("""
<style>
.main {padding: 0rem 1rem;}
h1 {color:#0b1f44;font-weight:800;letter-spacing:.2px;border-bottom:3px solid #2ecc71;padding-bottom:10px;}
h2 {color:#0b1f44;font-weight:700;margin-top:1.5rem;margin-bottom:.75rem;}
.kpi {background:#fff;border:1px solid #e6e6e6;border-radius:14px;padding:14px;}
.k-num {font-size:36px;font-weight:800;color:#0b1f44;line-height:1.0;}
.k-cap {font-size:13px;color:#6b7280;margin-top:4px;}
.small {font-size:12px;color:#6b7280;}
</style>
""", unsafe_allow_html=True)

st.title("Radiopharma OTP")

# ---------------- Config ----------------
OTP_TARGET = 95
SCOPE_PU   = {"DE", "IT", "IL"}
CTRL_REGEX = re.compile(r"\b(agent|del\s*agt|delivery\s*agent|customs|warehouse|w/house)\b", re.I)

# ---------------- Helpers ----------------
def _excel_to_dt(s: pd.Series) -> pd.Series:
    """
    Robust datetime:
    - Try normal parse.
    - If many NaT, try Excel serial conversion (origin 1899-12-30).
    """
    out = pd.to_datetime(s, errors="coerce")
    if out.isna().mean() > 0.5:
        num = pd.to_numeric(s, errors="coerce")
        out2 = pd.to_datetime("1899-12-30") + pd.to_timedelta(num, unit="D")
        out  = out.where(~out.isna(), out2)
    return out

def _get_target_series(df: pd.DataFrame) -> pd.Series | None:
    if "UPD DEL" in df.columns and df["UPD DEL"].notna().any():
        return df["UPD DEL"]
    if "QDT" in df.columns:
        return df["QDT"]
    return None

def _kfmt(n: float) -> str:
    if pd.isna(n): return ""
    try:
        n = float(n)
    except Exception:
        return ""
    if n >= 1000: return f"{n/1000:.1f}K"
    return f"{n:.0f}"

def make_semi_gauge(title: str, value: float) -> go.Figure:
    """Semi-donut gauge with centered %."""
    v = max(0.0, min(100.0, 0.0 if pd.isna(value) else float(value)))
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
    return pd.read_excel(uploaded)  # openpyxl

@st.cache_data(show_spinner=False)
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters & flags:
      - PU CTRY in {DE, IT, IL} and STATUS=440-BILLED
      - Dates:
          POD (actual delivery) = OTP/volume/pieces month basis
          Target = UPD DEL (preferred) → QDT
      - OTP:
          Gross = POD <= target
          Net   = Gross OR (Late & NOT controllable)  → net ≥ gross
      - Month keys from POD: Month_YYYY_MM (e.g. '2025-07'), Month_Display (e.g. 'Jul 2025'), Month_Sort
    """
    required = ["PU CTRY", "STATUS", "POD DATE/TIME"]
    if not all(c in df.columns for c in required):
        return pd.DataFrame()

    d = df.copy()

    # Scope
    d["_pu"]     = d["PU CTRY"].astype(str).str.strip()
    d["_status"] = d["STATUS"].astype(str).str.strip().str.lower()
    d = d[d["_pu"].isin(SCOPE_PU)]
    d = d[d["_status"] == "440-billed"]
    if d.empty:
        return d

    # Dates
    d["_adate"] = _excel_to_dt(d["POD DATE/TIME"])  # POD = single source of month
    tgt = _get_target_series(d)
    d["_target"] = _excel_to_dt(tgt) if tgt is not None else pd.NaT

    # Month keys from POD
    d["Month_YYYY_MM"] = d["_adate"].dt.to_period("M").astype(str)  # e.g., '2025-07'
    d["Month_Sort"]    = pd.to_datetime(d["Month_YYYY_MM"] + "-01", errors="coerce")
    d["Month_Display"] = d["Month_Sort"].dt.strftime("%b %Y")       # e.g., 'Jul 2025'

    # QC controllable flag
    if "QC NAME" in d.columns:
        d["QC_NAME_CLEAN"] = d["QC NAME"].astype(str)
        d["Is_Controllable"] = d["QC_NAME_CLEAN"].str.contains(CTRL_REGEX, na=False)
    else:
        d["QC_NAME_CLEAN"] = ""
        d["Is_Controllable"] = False

    # OTP flags
    ok = d["_adate"].notna() & d["_target"].notna()
    d["On_Time_Gross"] = False
    d.loc[ok, "On_Time_Gross"] = d.loc[ok, "_adate"] <= d.loc[ok, "_target"]
    d["Late"]        = ~d["On_Time_Gross"]
    d["On_Time_Net"] = d["On_Time_Gross"] | (d["Late"] & ~d["Is_Controllable"])  # net ≥ gross

    # PIECES numeric
    if "PIECES" in d.columns:
        d["PIECES"] = pd.to_numeric(d["PIECES"], errors="coerce").fillna(0)
    else:
        d["PIECES"] = 0

    return d

def calc_summary(df: pd.DataFrame):
    """Gross%, Net%, counts."""
    valid = df.dropna(subset=["_adate", "_target"])
    total_ship = int(len(valid))
    gross = (valid["On_Time_Gross"].mean() * 100) if total_ship else float("nan")
    net   = (valid["On_Time_Net"].mean()   * 100) if total_ship else float("nan")
    if pd.notna(gross) and pd.notna(net) and net < gross:
        net = gross  # safety guard matching your "net ≥ gross" requirement

    late_df         = valid[valid["Late"]]
    exceptions      = int(len(late_df))
    controllables   = int(late_df["Is_Controllable"].sum())
    uncontrollables = int(exceptions - controllables)

    return round(gross, 2) if pd.notna(gross) else float("nan"), \
           round(net,   2) if pd.notna(net)   else float("nan"), \
           total_ship, exceptions, controllables, uncontrollables

@st.cache_data(show_spinner=False)
def monthly_otp_pod(df: pd.DataFrame) -> pd.DataFrame:
    """Gross & Net OTP grouped by POD month (YYYY-MM key)."""
    valid = df.dropna(subset=["_adate", "_target"]).copy()
    if valid.empty:
        return pd.DataFrame(columns=["Month_YYYY_MM","Month_Display","Month_Sort","Gross_OTP","Net_OTP"])

    g = valid.groupby(["Month_YYYY_MM","Month_Display","Month_Sort"], as_index=False).agg(
        Gross_On = ("On_Time_Gross", "sum"),
        Gross_Tot= ("On_Time_Gross", "count"),
        Net_On   = ("On_Time_Net",   "sum"),
        Net_Tot  = ("On_Time_Net",   "count"),
    )
    g["Gross_OTP"] = (g["Gross_On"] / g["Gross_Tot"] * 100).round(2)
    g["Net_OTP"]   = (g["Net_On"]   / g["Net_Tot"]   * 100).round(2)
    return g.sort_values("Month_Sort")

@st.cache_data(show_spinner=False)
def monthly_volume_pod(df: pd.DataFrame) -> pd.DataFrame:
    """Order count grouped by POD month (YYYY-MM key)."""
    base = df.dropna(subset=["_adate"]).copy()
    if base.empty:
        return pd.DataFrame(columns=["Month_YYYY_MM","Month_Display","Month_Sort","Volume"])
    v = base.groupby(["Month_YYYY_MM","Month_Display","Month_Sort"], as_index=False).size()
    v.rename(columns={"size":"Volume"}, inplace=True)
    return v.sort_values("Month_Sort")

@st.cache_data(show_spinner=False)
def monthly_pieces_pod(df: pd.DataFrame) -> pd.DataFrame:
    """
    PIECES grouped by POD month (YYYY-MM key) — this matches your rule:
    '2025-07-10 20:32' -> '2025-07', then sum pieces per month.
    """
    base = df.dropna(subset=["_adate"]).copy()
    if base.empty:
        return pd.DataFrame(columns=["Month_YYYY_MM","Month_Display","Month_Sort","Pieces"])
    p = base.groupby(["Month_YYYY_MM","Month_Display","Month_Sort"], as_index=False)["PIECES"].sum()
    p.rename(columns={"PIECES":"Pieces"}, inplace=True)
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

# Monthly frames (ALL by POD month with the same keys)
otp_pod   = monthly_otp_pod(df)
vol_pod   = monthly_volume_pod(df)
pieces_pod= monthly_pieces_pod(df)

# ---------------- KPI & Gauges ----------------
left, right = st.columns([1, 1.5])

with left:
    st.markdown(f'<div class="kpi"><div class="k-num">{total_ship:,}</div><div class="k-cap">Volume (valid for OTP)</div></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="kpi"><div class="k-num">{exceptions:,}</div><div class="k-cap">Exceptions (Gross Late)</div></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="kpi"><div class="k-num">{controllables:,}</div><div class="k-cap">Controllables (QC)</div></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="kpi"><div class="k-num">{uncontrollables:,}</div><div class="k-cap">Uncontrollables (QC)</div></div>', unsafe_allow_html=True)

with right:
    g1, g2, g3 = st.columns(3)
    with g1: st.plotly_chart(make_semi_gauge("Adjusted OTP", max(gross_otp, net_otp)), use_container_width=True, config={"displayModeBar": False})
    with g2: st.plotly_chart(make_semi_gauge("Controllable OTP", net_otp), use_container_width=True, config={"displayModeBar": False})
    with g3: st.plotly_chart(make_semi_gauge("Raw OTP", gross_otp),      use_container_width=True, config={"displayModeBar": False})

with st.expander("How months are defined"):
    st.markdown("""
- **All** MoM groupings (**OTP**, **Volume**, **PIECES**) use the **POD (Actual Delivery) month**.
- Month key is **YYYY-MM** from `POD DATE/TIME` (e.g., `2025-07-10 20:32` → **2025-07**).
- Target date is **UPD DEL** when present, else **QDT**.
""")

st.markdown("---")

# ---------------- Chart 1: Controllable OTP by Volume (POD) ----------------
st.subheader("Controllable OTP by Volume (POD)")

if not vol_pod.empty:
    mv = vol_pod.merge(otp_pod[["Month_YYYY_MM","Net_OTP"]], on="Month_YYYY_MM", how="left") \
                .sort_values("Month_Sort")

    x = mv["Month_Display"].astype(str).tolist()
    y_vol = mv["Volume"].astype(float).tolist()
    y_net = mv["Net_OTP"].astype(float).tolist()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=x, y=y_vol, name="Volume (Orders)",
        marker_color=NAVY,
        text=[_kfmt(v) for v in y_vol],
        textposition="outside", textfont=dict(size=12, color="#4b5563"),
        yaxis="y"
    ))
    fig.add_trace(go.Scatter(
        x=x, y=y_net, name="Net OTP (Controllable)",
        mode="lines+markers", line=dict(color=GOLD, width=3),
        marker=dict(size=8), yaxis="y2"
    ))
    # % labels lifted for clarity
    for xi, yi in zip(x, y_net):
        if pd.notna(yi):
            fig.add_annotation(x=xi, y=yi, xref="x", yref="y2",
                               yshift=24, text=f"{yi:.2f}%", showarrow=False,
                               font=dict(size=12, color="#111827"))
    # Target line on y2
    try:
        fig.add_hline(y=float(otp_target), line_dash="dash", line_color="red", yref="y2")
    except Exception:
        fig.add_shape(type="line", x0=-0.5, x1=len(x)-0.5, y0=float(otp_target), y1=float(otp_target),
                      xref="x", yref="y2", line=dict(color="red", dash="dash"))

    fig.update_layout(
        height=520, hovermode="x unified", plot_bgcolor="white",
        margin=dict(l=40, r=40, t=40, b=80),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.0),
        xaxis=dict(title="", tickangle=-30, tickmode="array", tickvals=x, ticktext=x, automargin=True),
        yaxis=dict(title="Volume (Orders)", side="left", gridcolor=GRID),
        yaxis2=dict(title="Net OTP (%)", overlaying="y", side="right", range=[0, 130]),
        barmode="overlay"
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No monthly volume available.")

st.markdown("---")

# ---------------- Chart 1b: Controllable OTP by Pieces (POD) ----------------
st.subheader("Controllable OTP by Pieces (POD)")

if not pieces_pod.empty:
    mp = pieces_pod.merge(otp_pod[["Month_YYYY_MM","Net_OTP"]], on="Month_YYYY_MM", how="left") \
                   .sort_values("Month_Sort")

    x = mp["Month_Display"].astype(str).tolist()
    y_pcs = mp["Pieces"].astype(float).tolist()
    y_net = mp["Net_OTP"].astype(float).tolist()

    figp = go.Figure()
    figp.add_trace(go.Bar(
        x=x, y=y_pcs, name="Pieces",
        marker_color=NAVY,
        text=[_kfmt(v) for v in y_pcs],
        textposition="outside", textfont=dict(size=12, color="#4b5563"),
        yaxis="y"
    ))
    figp.add_trace(go.Scatter(
        x=x, y=y_net, name="Net OTP (Controllable)",
        mode="lines+markers", line=dict(color=GOLD, width=3),
        marker=dict(size=8), yaxis="y2"
    ))
    # % labels
    for xi, yi in zip(x, y_net):
        if pd.notna(yi):
            figp.add_annotation(x=xi, y=yi, xref="x", yref="y2",
                                yshift=24, text=f"{yi:.2f}%", showarrow=False,
                                font=dict(size=12, color="#111827"))
    # Target
    try:
        figp.add_hline(y=float(otp_target), line_dash="dash", line_color="red", yref="y2")
    except Exception:
        figp.add_shape(type="line", x0=-0.5, x1=len(x)-0.5, y0=float(otp_target), y1=float(otp_target),
                       xref="x", yref="y2", line=dict(color="red", dash="dash"))

    figp.update_layout(
        height=520, hovermode="x unified", plot_bgcolor="white",
        margin=dict(l=40, r=40, t=40, b=80),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.0),
        xaxis=dict(title="", tickangle=-30, tickmode="array", tickvals=x, ticktext=x, automargin=True),
        yaxis=dict(title="Pieces", side="left", gridcolor=GRID),
        yaxis2=dict(title="Net OTP (%)", overlaying="y", side="right", range=[0, 130]),
        barmode="overlay"
    )
    st.plotly_chart(figp, use_container_width=True)
else:
    st.info("No monthly PIECES available.")

st.markdown("---")

# ---------------- Chart 2: Monthly OTP Trend (Gross vs Net by POD) ----------------
st.subheader("Monthly OTP Trend (Gross vs Net) — POD Month")

if not otp_pod.empty:
    otp_sorted = otp_pod.sort_values("Month_Sort")
    x       = otp_sorted["Month_Display"].astype(str).tolist()
    gross_y = otp_sorted["Gross_OTP"].astype(float).tolist()
    net_y   = otp_sorted["Net_OTP"].astype(float).tolist()

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=x, y=gross_y, mode="lines+markers", name="Gross OTP",
        line=dict(color=BLUE, width=3), marker=dict(size=7),
    ))
    fig2.add_trace(go.Scatter(
        x=x, y=net_y, mode="lines+markers", name="Net OTP",
        line=dict(color=GREEN, width=3), marker=dict(size=7),
    ))
    # % labels lifted
    for xi, yi in zip(x, gross_y):
        if pd.notna(yi):
            fig2.add_annotation(x=xi, y=yi, xref="x", yref="y",
                                yshift=24, text=f"{yi:.2f}%", showarrow=False,
                                font=dict(size=12, color=BLUE))
    for xi, yi in zip(x, net_y):
        if pd.notna(yi):
            fig2.add_annotation(x=xi, y=yi, xref="x", yref="y",
                                yshift=24, text=f"{yi:.2f}%", showarrow=False,
                                font=dict(size=12, color=GREEN))
    # Target
    try:
        fig2.add_hline(y=float(otp_target), line_dash="dash", line_color="red")
    except Exception:
        fig2.add_shape(type="line", x0=-0.5, x1=len(x)-0.5,
                       y0=float(otp_target), y1=float(otp_target),
                       xref="x", yref="y", line=dict(color="red", dash="dash"))

    fig2.update_layout(
        height=460, hovermode="x unified", plot_bgcolor="white",
        margin=dict(l=40, r=40, t=40, b=80),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.0),
        xaxis=dict(title="", tickangle=-30, tickmode="array", tickvals=x, ticktext=x, automargin=True),
        yaxis=dict(title="OTP (%)", range=[0, 130], gridcolor=GRID)
    )
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("No monthly OTP trend available.")

st.caption(
    "Gross OTP: POD ≤ target (UPD DEL → QDT). Net OTP: controllable lates not penalized (QC NAME contains Agent / Delivery agent / Customs / Warehouse / W/house) → Net ≥ Gross. "
    "Filters: PU CTRY ∈ {DE, IT, IL}, STATUS = 440-BILLED. All MoM groupings (OTP, Volume, PIECES) use POD month (key = YYYY-MM)."
)

# ---------------- QC breakdown table (optional, helpful for audits) ----------------
if "QC NAME" in df.columns:
    qc = df.copy()
    qc["Control_Type"] = qc["Is_Controllable"].map({True: "Controllable", False: "Non-Controllable"})
    qc_grp = (qc.groupby(["Control_Type","QC_NAME_CLEAN"], dropna=False)
                .size().reset_index(name="Count")
                .sort_values(["Control_Type","Count"], ascending=[True, False]))
    with st.expander("QC NAME breakdown (controllable vs non-controllable)"):
        st.dataframe(qc_grp, use_container_width=True)
