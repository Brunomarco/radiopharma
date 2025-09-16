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

# Colors & typography styled to your screenshots
NAVY = "#0b1f44"       # bars / gauge fill
GOLD = "#f0b429"       # line / accents
BLUE = "#1f77b4"       # gross line
GREEN = "#10b981"      # net line
SLATE = "#334155"      # axis labels
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
    """K formatter for volumes/pieces."""
    if pd.isna(n): return ""
    if n >= 1000: return f"{n/1000:.1f}K"
    return f"{n:.0f}"

def make_semi_gauge(title: str, value: float) -> go.Figure:
    """Semi-donut gauge with large percentage in the center."""
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
    # Big percentage in the center (raised for visibility)
    fig.add_annotation(
        text=f"{v:.2f}%",
        x=0.5, y=0.60,  # higher than center so it never looks clipped
        showarrow=False,
        font=dict(size=26, color=NAVY, family="Arial Black")
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=36, b=0),
        height=180,
    )
    # Title just above the gauge
    fig.add_annotation(
        text=title,
        x=0.5, y=1.18, showarrow=False,
        font=dict(size=14, color=SLATE, family="Arial")
    )
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
      - Filters: PU CTRY in {DE, IT, IL} and STATUS=440-BILLED
      - Dates: POD (actual delivery) and ACT PU (pickup), Target = UPD DEL → QDT
      - OTP: Gross = POD<=Target; Net = only controllable lates count
      - Month keys for both POD and ACT PU
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

    # Actual delivery (POD) & pickup (ACT PU)
    d["_adate"] = _excel_to_dt(d["POD DATE/TIME"])
    d["_pudate"] = _excel_to_dt(d["ACT PU"]) if "ACT PU" in d.columns else pd.NaT

    # Target date
    tgt = _get_target_series(d)
    d["_target"] = _excel_to_dt(tgt) if tgt is not None else pd.NaT

    # Month (full text) for both bases
    d["Month_POD_Display"] = d["_adate"].dt.strftime("%b %Y")
    d["Month_POD_Sort"] = pd.to_datetime(d["Month_POD_Display"], format="%b %Y", errors="coerce")
    d["Month_PU_Display"] = d["_pudate"].dt.strftime("%b %Y")
    d["Month_PU_Sort"] = pd.to_datetime(d["Month_PU_Display"], format="%b %Y", errors="coerce")

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

    # PIECES numeric
    if "PIECES" in d.columns:
        d["PIECES"] = pd.to_numeric(d["PIECES"], errors="coerce").fillna(0)
    else:
        d["PIECES"] = 0

    return d

def calc_summary(df: pd.DataFrame):
    """Gross%, Net%, total volume (valid OTP rows), exceptions (gross-late), controllables, uncontrollables."""
    valid = df.dropna(subset=["_adate", "_target"])
    total_ship = int(len(valid))

    gross = valid["On_Time_Gross"].mean() * 100 if total_ship else np.nan
    net = valid["On_Time_Net"].mean() * 100 if total_ship else np.nan
    if pd.notna(gross) and pd.notna(net) and net < gross:
        net = gross  # numerical safety

    late_df = valid[valid["Late"]]
    exceptions = int(len(late_df))
    controllables = int(late_df["Is_Controllable"].sum())
    uncontrollables = int(exceptions - controllables)

    return (round(gross, 2) if pd.notna(gross) else np.nan,
            round(net, 2) if pd.notna(net) else np.nan,
            total_ship, exceptions, controllables, uncontrollables)

def _month_cols(basis: str):
    """Return (display_col, sort_col) for basis: 'pod' or 'pu'."""
    if basis == "pu":
        return "Month_PU_Display", "Month_PU_Sort"
    return "Month_POD_Display", "Month_POD_Sort"

@st.cache_data(show_spinner=False)
def monthly_otp_and_metric(df: pd.DataFrame, basis: str, metric_col: str):
    """
    Generic monthly aggregator:
      - basis: 'pod' (default) or 'pu' (use ACT PU)
      - metric_col: 'Volume' (count rows with non-null basis date) or 'Pieces' (sum PIECES with non-null basis date)
      - OTP% computed on rows with POD & Target present, grouped by the same basis month
    """
    disp_col, sort_col = _month_cols(basis)

    # base for metric (rows that have the basis date)
    base_date_col = "_adate" if basis == "pod" else "_pudate"
    base = df.dropna(subset=[base_date_col]).copy()

    if base.empty:
        cols = ["Month_Display", metric_col, "Gross_OTP", "Net_OTP", "Month_Sort"]
        return pd.DataFrame(columns=cols)

    if metric_col == "Volume":
        metric = base.groupby(disp_col, as_index=False).size()
        metric.columns = ["Month_Display", "Volume"]
    elif metric_col == "Pieces":
        metric = base.groupby(disp_col, as_index=False)["PIECES"].sum()
        metric.columns = ["Month_Display", "Pieces"]
    else:
        raise ValueError("metric_col must be 'Volume' or 'Pieces'")

    # OTP on valid rows (POD & Target present), grouped by the same basis month
    valid = df.dropna(subset=["_adate", "_target"]).copy()
    if valid.empty:
        out = metric.copy()
        out["Gross_OTP"] = np.nan
        out["Net_OTP"] = np.nan
    else:
        # add month label of chosen basis to valid set
        valid["__month"] = valid[disp_col]
        g = valid.groupby("__month", as_index=False).agg(
            Gross_On=("On_Time_Gross", "sum"),
            Gross_Tot=("On_Time_Gross", "count")
        )
        n = valid.groupby("__month", as_index=False).agg(
            Net_On=("On_Time_Net", "sum"),
            Net_Tot=("On_Time_Net", "count")
        )
        out = metric.merge(g, left_on="Month_Display", right_on="__month", how="left").merge(
            n, left_on="Month_Display", right_on="__month", how="left"
        )
        out.drop(columns=["__month_x", "__month_y"], inplace=True, errors="ignore")
        out["Gross_OTP"] = (out["Gross_On"] / out["Gross_Tot"] * 100).round(2)
        out["Net_OTP"]   = (out["Net_On"] / out["Net_Tot"] * 100).round(2)

    out["Month_Sort"] = pd.to_datetime(out["Month_Display"], format="%b %Y", errors="coerce")
    return out.sort_values("Month_Sort")

# ---------------- Sidebar ----------------
with st.sidebar:
    up = st.file_uploader("Upload Excel (.xlsx) or CSV", type=["xlsx", "csv"])
    otp_target = st.number_input("OTP Target (%)", min_value=0, max_value=100, value=OTP_TARGET, step=1)

    st.markdown("**Month basis selectors**")
    # Default per your specs: OTP by POD; Pieces by ACT PU
    basis_volume = st.selectbox("Volume + Net OTP month", ["Actual Delivery (POD)", "Actual Pickup (ACT PU)"], index=0)
    basis_pieces = st.selectbox("Pieces + Net OTP month", ["Actual Delivery (POD)", "Actual Pickup (ACT PU)"], index=1)

    basis_map = {"Actual Delivery (POD)": "pod", "Actual Pickup (ACT PU)": "pu"}
    basis_volume = basis_map[basis_volume]
    basis_pieces = basis_map[basis_pieces]

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

# Monthly frames
mv = monthly_otp_and_metric(df, basis=basis_volume, metric_col="Volume")  # Volume+OTP by selected basis (default POD)
mp = monthly_otp_and_metric(df, basis=basis_pieces, metric_col="Pieces")  # Pieces+OTP by selected basis (default ACT PU)

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

with st.expander("How do we place an order into a month?"):
    st.markdown("""
- **OTP month**: by default, charts use **Actual Delivery (POD)**, matching your requirement “OTP month-on-month based on the actual delivery date”.
- **Volume month**: configurable (POD or ACT PU). Default = **POD**.
- **Pieces month**: configurable (POD or ACT PU). Default = **ACT PU**, matching your earlier request.
You can change both in the sidebar.
    """)

st.markdown("---")

# ---------------- Chart 1: Controllable OTP by Volume ----------------
st.subheader("Controllable OTP by Volume")

if not mv.empty:
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

    # % labels — lifted more to be crystal clear ( +7% )
    for xi, yi in zip(x_labels, net_vals):
        if pd.notna(yi):
            fig.add_annotation(
                x=xi, y=yi + 7,
                xref="x", yref="y2",
                text=f"{yi:.2f}%",
                showarrow=False,
                font=dict(size=12, color="#111827")
            )

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
        height=500,
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
            title="Volume (Orders)",
            side="left",
            gridcolor=LIGHT_GRAY
        ),
        yaxis2=dict(
            title="Controllable OTP (%)",
            overlaying="y",
            side="right",
            range=[0, 120]  # extra headroom so labels never clip
        ),
        barmode="overlay"
    )

    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No monthly data available.")

st.markdown("---")

# ---------------- Chart 1b: Controllable OTP by Pieces ----------------
st.subheader("Controllable OTP by Pieces")

if not mp.empty:
    x_labels = mp["Month_Display"].astype(str).tolist()
    pieces_vals = mp["Pieces"].astype(float).tolist()
    net_vals_p = mp["Net_OTP"].astype(float).tolist()

    figp = go.Figure()

    # Bars = monthly PIECES (navy)
    figp.add_trace(go.Bar(
        x=x_labels,
        y=pieces_vals,
        name="Pieces",
        marker_color=NAVY,
        text=[_kfmt(v) for v in pieces_vals],
        textposition="outside",
        textfont=dict(size=12, color="#4b5563"),
        yaxis="y"
    ))

    # Line = Net OTP % (gold)
    figp.add_trace(go.Scatter(
        x=x_labels,
        y=net_vals_p,
        name="Controllable OTP",
        mode="lines+markers",
        line=dict(color=GOLD, width=3),
        marker=dict(size=8),
        yaxis="y2"
    ))

    # % labels — lifted more ( +7% )
    for xi, yi in zip(x_labels, net_vals_p):
        if pd.notna(yi):
            figp.add_annotation(
                x=xi, y=yi + 7,
                xref="x", yref="y2",
                text=f"{yi:.2f}%",
                showarrow=False,
                font=dict(size=12, color="#111827")
            )

    # Target line
    try:
        figp.add_hline(y=float(otp_target), line_dash="dash", line_color="red", yref="y2")
    except Exception:
        figp.add_shape(
            type="line", x0=-0.5, x1=len(x_labels)-0.5,
            y0=float(otp_target), y1=float(otp_target),
            xref="x", yref="y2", line=dict(color="red", dash="dash")
        )

    figp.update_layout(
        height=500,
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
            title="Pieces",
            side="left",
            gridcolor=LIGHT_GRAY
        ),
        yaxis2=dict(
            title="Controllable OTP (%)",
            overlaying="y",
            side="right",
            range=[0, 120]
        ),
        barmode="overlay"
    )

    st.plotly_chart(figp, use_container_width=True)
else:
    st.info("No monthly PIECES data available.")

st.markdown("---")

# ---------------- Chart 2: Monthly OTP Trend (Gross vs Net + 95% only) ----------------
st.subheader("Monthly OTP Trend (Gross vs Net) — by Actual Delivery (POD)")

# Build a Gross vs Net MoM frame using POD basis (your requirement for OTP)
m_pod = monthly_otp_and_metric(df, basis="pod", metric_col="Volume")
if not m_pod.empty:
    gross_vals = m_pod.get("Gross_OTP", pd.Series([np.nan]*len(m_pod))).astype(float).tolist()
    net_vals   = m_pod.get("Net_OTP", pd.Series([np.nan]*len(m_pod))).astype(float).tolist()
    x_labels   = m_pod["Month_Display"].astype(str).tolist()

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

    # % labels — lifted more ( +7% ) for both series
    for xi, yi in zip(x_labels, gross_vals):
        if pd.notna(yi):
            fig2.add_annotation(x=xi, y=yi + 7, text=f"{yi:.2f}%", showarrow=False,
                                xref="x", yref="y", font=dict(size=12, color=BLUE))
    for xi, yi in zip(x_labels, net_vals):
        if pd.notna(yi):
            fig2.add_annotation(x=xi, y=yi + 7, text=f"{yi:.2f}%", showarrow=False,
                                xref="x", yref="y", font=dict(size=12, color=GREEN))

    # Target
    try:
        fig2.add_hline(y=float(otp_target), line_dash="dash", line_color="red")
    except Exception:
        fig2.add_shape(
            type="line", x0=-0.5, x1=len(x_labels)-0.5,
            y0=float(otp_target), y1=float(otp_target),
            xref="x", yref="y", line=dict(color="red", dash="dash")
        )

    fig2.update_layout(
        height=440,
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
            title="OTP (%)",
            range=[0, 120],
            gridcolor=LIGHT_GRAY
        )
    )
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("No monthly OTP trend available.")

st.caption("Gross: POD ≤ target (UPD DEL → QDT). Net: only controllable lates (Agent / Del Agt / Delivery agent / Customs / Warehouse / W/house) count against OTP. Filters: PU CTRY ∈ {DE, IT, IL}, STATUS = 440-BILLED.")
