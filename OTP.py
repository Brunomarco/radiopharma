import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ---------------- Page & style ----------------
st.set_page_config(page_title="Radiopharma OTP", page_icon="📊",
                   layout="wide", initial_sidebar_state="collapsed")

NAVY  = "#0b1f44"      # bars / gauge
GOLD  = "#f0b429"      # net line
BLUE  = "#1f77b4"      # gross line
GREEN = "#10b981"      # alt net
SLATE = "#334155"
GRID  = "#e5e7eb"

st.markdown("""
<style>
.main {padding: 0rem 1rem;}
h1 {color:#0b1f44;font-weight:800;letter-spacing:.2px;border-bottom:3px solid #2ecc71;padding-bottom:10px;}
h2 {color:#0b1f44;font-weight:700;margin-top:1.2rem;margin-bottom:.6rem;}
.kpi {background:#fff;border:1px solid #e6e6e6;border-radius:14px;padding:14px;}
.k-num {font-size:36px;font-weight:800;color:#0b1f44;line-height:1.0;}
.k-cap {font-size:13px;color:#6b7280;margin-top:4px;}
</style>
""", unsafe_allow_html=True)

st.title("Radiopharma OTP")

# ---------------- Config ----------------
OTP_TARGET = 95
SCOPE_PU   = {"DE", "IT", "IL"}
CTRL_REGEX = re.compile(r"\b(agent|del\s*agt|delivery\s*agent|customs|warehouse|w/house)\b", re.I)

# ---------------- Helpers ----------------
def _excel_to_dt(s: pd.Series) -> pd.Series:
    """Robust datetime: parse; if many NaT, try Excel serials."""
    out = pd.to_datetime(s, errors="coerce")
    if out.isna().mean() > 0.5:
        num  = pd.to_numeric(s, errors="coerce")
        out2 = pd.to_datetime("1899-12-30") + pd.to_timedelta(num, unit="D")
        out  = out.where(~out.isna(), out2)
    return out

def _kfmt(n: float) -> str:
    if pd.isna(n): return ""
    try: n = float(n)
    except: return ""
    return f"{n/1000:.1f}K" if n >= 1000 else f"{n:.0f}"

def _make_gauge(title: str, value: float) -> go.Figure:
    v = max(0.0, min(100.0, 0.0 if pd.isna(value) else float(value)))
    fig = go.Figure()
    fig.add_trace(go.Pie(
        values=[v, 100 - v, 100],
        hole=0.75, sort=False, direction="clockwise", rotation=180,
        textinfo="none", showlegend=False,
        marker=dict(colors=[NAVY, "#d1d5db", "rgba(0,0,0,0)"])
    ))
    fig.add_annotation(text=f"{v:.2f}%", x=0.5, y=0.5, xref="paper", yref="paper",
                       showarrow=False, font=dict(size=26, color=NAVY, family="Arial Black"))
    fig.add_annotation(text=title, x=0.5, y=1.18, xref="paper", yref="paper",
                       showarrow=False, font=dict(size=14, color=SLATE))
    fig.update_layout(margin=dict(l=10, r=10, t=36, b=0), height=180)
    return fig

# ---------------- IO ----------------
@st.cache_data(show_spinner=False)
def read_file(uploaded) -> pd.DataFrame:
    name = uploaded.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded)
    return pd.read_excel(uploaded)  # openpyxl

# ---------------- Prep & Dedup ----------------
@st.cache_data(show_spinner=False)
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter to PU CTRY in {DE, IT, IL} and STATUS=440-BILLED.
    Parse POD/UPD DEL/QDT, PIECES numeric, QC NAME controllable flag.
    """
    required = ["PU CTRY", "STATUS", "POD DATE/TIME", "REFER"]
    if not all(c in df.columns for c in required):
        return pd.DataFrame()

    d = df.copy()
    d["_pu"]     = d["PU CTRY"].astype(str).str.strip()
    d["_status"] = d["STATUS"].astype(str).str.strip().str.lower()
    d = d[d["_pu"].isin(SCOPE_PU)]
    d = d[d["_status"] == "440-billed"]
    if d.empty:
        return d

    # Dates
    d["_pod"]   = _excel_to_dt(d["POD DATE/TIME"])
    d["_upd"]   = _excel_to_dt(d["UPD DEL"]) if "UPD DEL" in d.columns else pd.NaT
    d["_qdt"]   = _excel_to_dt(d["QDT"])     if "QDT"     in d.columns else pd.NaT
    d["_target"]= d["_upd"].where(d["_upd"].notna(), d["_qdt"])

    # PIECES
    d["PIECES"] = pd.to_numeric(d.get("PIECES", 0), errors="coerce").fillna(0)

    # QC NAME controllable?
    if "QC NAME" in d.columns:
        d["QC_NAME_CLEAN"] = d["QC NAME"].astype(str)
        d["Is_Controllable_Row"] = d["QC_NAME_CLEAN"].str.contains(CTRL_REGEX, na=False)
    else:
        d["QC_NAME_CLEAN"] = ""
        d["Is_Controllable_Row"] = False

    return d

@st.cache_data(show_spinner=False)
def dedup_by_refer(d: pd.DataFrame) -> pd.DataFrame:
    """
    One entry per REFER (avoid double counting the same client/shipment).
      - POD = latest POD
      - Target = latest UPD DEL else latest QDT
      - PIECES = max non-null pieces
      - Controllable = any controllable QC in that REFER
    """
    if d.empty:
        return d

    g = (d.groupby("REFER", as_index=False).agg(
        _pod        = ("_pod", "max"),
        _upd        = ("_upd", "max"),
        _qdt        = ("_qdt", "max"),
        PIECES      = ("PIECES", "max"),
        PU_CTRY     = ("PU CTRY", "first"),
        Is_Controll = ("Is_Controllable_Row", "any"),
    ))
    g["_target"] = g["_upd"].where(g["_upd"].notna(), g["_qdt"])

    # Month keys from POD (Actual Delivery)
    g["Month_YYYY_MM"] = g["_pod"].dt.to_period("M").astype(str)
    g["Month_Sort"]    = pd.to_datetime(g["Month_YYYY_MM"] + "-01", errors="coerce")
    g["Month_Display"] = g["Month_Sort"].dt.strftime("%b %Y")

    # Row-level OTP on deduped entries
    ok = g["_pod"].notna() & g["_target"].notna()
    g["On_Time_Gross"] = False
    g.loc[ok, "On_Time_Gross"] = g.loc[ok, "_pod"] <= g.loc[ok, "_target"]
    g["Late"]         = ~g["On_Time_Gross"]
    g["On_Time_Net"]  = g["On_Time_Gross"] | (g["Late"] & ~g["Is_Controll"])

    return g

# ---------------- Monthly builders ----------------
@st.cache_data(show_spinner=False)
def monthly_frames(ship: pd.DataFrame):
    """Monthly Volume (unique entries), Pieces, OTP — by POD month."""
    base = ship.dropna(subset=["_pod"]).copy()

    vol = (base.groupby(["Month_YYYY_MM","Month_Display","Month_Sort"], as_index=False)
                .size().rename(columns={"size":"Volume"}))

    pieces = (base.groupby(["Month_YYYY_MM","Month_Display","Month_Sort"], as_index=False)
                   .agg(Pieces=("PIECES","sum")))

    otp_base = ship.dropna(subset=["_pod","_target"]).copy()
    if otp_base.empty:
        otp = pd.DataFrame(columns=["Month_YYYY_MM","Month_Display","Month_Sort","Gross_OTP","Net_OTP"])
    else:
        otp = (otp_base.groupby(["Month_YYYY_MM","Month_Display","Month_Sort"], as_index=False)
                      .agg(Gross_On=("On_Time_Gross","sum"),
                           Gross_Tot=("On_Time_Gross","count"),
                           Net_On=("On_Time_Net","sum"),
                           Net_Tot=("On_Time_Net","count")))
        otp["Gross_OTP"] = (otp["Gross_On"]/otp["Gross_Tot"]*100).round(2)
        otp["Net_OTP"]   = (otp["Net_On"]/otp["Net_Tot"]*100).round(2)

    vol, pieces, otp = [x.sort_values("Month_Sort") for x in (vol, pieces, otp)]
    return vol, pieces, otp

def calc_summary(ship: pd.DataFrame):
    otp_base = ship.dropna(subset=["_pod","_target"])
    gross = otp_base["On_Time_Gross"].mean()*100 if len(otp_base) else np.nan
    net   = otp_base["On_Time_Net"].mean()*100   if len(otp_base) else np.nan
    if pd.notna(gross) and pd.notna(net) and net < gross:  # safety: Net ≥ Gross per spec
        net = gross
    late_df         = otp_base[otp_base["Late"]]
    exceptions      = int(len(late_df))
    controllables   = int(late_df["Is_Controll"].sum())
    uncontrollables = exceptions - controllables
    return round(gross,2) if pd.notna(gross) else np.nan, \
           round(net,2)   if pd.notna(net)   else np.nan, \
           int(len(ship.dropna(subset=["_pod"]))), exceptions, controllables, uncontrollables

# ---------------- Sidebar ----------------
with st.sidebar:
    up = st.file_uploader("Upload Excel (.xlsx) or CSV", type=["xlsx","csv"])
    otp_target = st.number_input("OTP Target (%)", min_value=0, max_value=100, value=OTP_TARGET, step=1)

if not up:
    st.info("👆 Upload your file. We apply: PU CTRY ∈ {DE, IT, IL}, STATUS = 440-BILLED, one entry per REFER.")
    st.stop()

# ---------------- Pipeline ----------------
try:
    raw = read_file(up)
except Exception as e:
    st.error(f"Could not read file: {e}")
    st.stop()

df = preprocess(raw)
if df.empty:
    st.error("No rows after filtering or required columns missing. Needed: PU CTRY, STATUS, POD DATE/TIME, REFER (+ UPD DEL or QDT).")
    st.stop()

# *** Key fix: one entry per REFER (no double counting) ***
ship = dedup_by_refer(df)

vol_pod, pieces_pod, otp_pod = monthly_frames(ship)
gross_otp, net_otp, volume_total, exceptions, controllables, uncontrollables = calc_summary(ship)

# ---------------- KPIs & Gauges ----------------
left, right = st.columns([1, 1.5])
with left:
    st.markdown(f'<div class="kpi"><div class="k-num">{volume_total:,}</div><div class="k-cap">Volume (unique entries)</div></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="kpi"><div class="k-num">{exceptions:,}</div><div class="k-cap">Exceptions (Gross Late)</div></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="kpi"><div class="k-num">{controllables:,}</div><div class="k-cap">Controllables (QC)</div></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="kpi"><div class="k-num">{uncontrollables:,}</div><div class="k-cap">Uncontrollables (QC)</div></div>', unsafe_allow_html=True)

with right:
    c1, c2, c3 = st.columns(3)
    with c1: st.plotly_chart(_make_gauge("Adjusted OTP", max(gross_otp, net_otp)), use_container_width=True, config={"displayModeBar": False})
    with c2: st.plotly_chart(_make_gauge("Controllable OTP", net_otp), use_container_width=True, config={"displayModeBar": False})
    with c3: st.plotly_chart(_make_gauge("Raw OTP", gross_otp),      use_container_width=True, config={"displayModeBar": False})

with st.expander("Month & logic"):
    st.markdown("""
- **One entry per REFER** (no double counting). If the export repeats a REFER, we keep:
  - **Latest POD** and **latest target** (UPD DEL else QDT),
  - **PIECES = max** non-null across that REFER,
  - **Controllable** if **any** row for that REFER is controllable.
- Month basis for Volume/PIECES/OTP: **POD DATE/TIME → YYYY-MM**.
- Net OTP does **not** penalize controllable lates (Agent / Delivery agent / Customs / Warehouse / W/house) ⇒ **Net ≥ Gross**.
""")

st.markdown("---")

# ---------------- Chart: Net OTP by Volume ----------------
st.subheader("Controllable (Net) OTP by Volume — POD Month")
if not vol_pod.empty:
    mv = vol_pod.merge(otp_pod[["Month_YYYY_MM","Net_OTP"]], on="Month_YYYY_MM", how="left").sort_values("Month_Sort")
    x = mv["Month_Display"].tolist()
    y_vol = mv["Volume"].astype(float).tolist()
    y_net = mv["Net_OTP"].astype(float).tolist()

    fig = go.Figure()
    fig.add_trace(go.Bar(x=x, y=y_vol, name="Volume (Unique Entries)", marker_color=NAVY,
                         text=[_kfmt(v) for v in y_vol], textposition="outside",
                         textfont=dict(size=12, color="#4b5563"), yaxis="y"))
    fig.add_trace(go.Scatter(x=x, y=y_net, name="Net OTP",
                             mode="lines+markers", line=dict(color=GOLD, width=3),
                             marker=dict(size=8), yaxis="y2"))
    for xi, yi in zip(x, y_net):
        if pd.notna(yi):
            fig.add_annotation(x=xi, y=yi, xref="x", yref="y2", yshift=28,
                               text=f"{yi:.2f}%", showarrow=False, font=dict(size=12, color="#111827"))
    try:
        fig.add_hline(y=float(otp_target), line_dash="dash", line_color="red", yref="y2")
    except Exception:
        fig.add_shape(type="line", x0=-0.5, x1=len(x)-0.5, y0=float(otp_target), y1=float(otp_target),
                      xref="x", yref="y2", line=dict(color="red", dash="dash"))
    fig.update_layout(height=520, hovermode="x unified", plot_bgcolor="white",
                      margin=dict(l=40, r=40, t=40, b=80),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.0),
                      xaxis=dict(title="", tickangle=-30, tickmode="array", tickvals=x, ticktext=x, automargin=True),
                      yaxis=dict(title="Volume (Unique Entries)", side="left", gridcolor=GRID),
                      yaxis2=dict(title="Net OTP (%)", overlaying="y", side="right", range=[0, 130]),
                      barmode="overlay")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No monthly volume available.")

st.markdown("---")

# ---------------- Chart: Net OTP by Pieces ----------------
st.subheader("Controllable (Net) OTP by Pieces — POD Month")
if not pieces_pod.empty:
    mp = pieces_pod.merge(otp_pod[["Month_YYYY_MM","Net_OTP"]], on="Month_YYYY_MM", how="left").sort_values("Month_Sort")
    x = mp["Month_Display"].tolist()
    y_pcs = mp["Pieces"].astype(float).tolist()
    y_net = mp["Net_OTP"].astype(float).tolist()

    figp = go.Figure()
    figp.add_trace(go.Bar(x=x, y=y_pcs, name="Pieces (Sum)", marker_color=NAVY,
                          text=[_kfmt(v) for v in y_pcs], textposition="outside",
                          textfont=dict(size=12, color="#4b5563"), yaxis="y"))
    figp.add_trace(go.Scatter(x=x, y=y_net, name="Net OTP",
                              mode="lines+markers", line=dict(color=GOLD, width=3),
                              marker=dict(size=8), yaxis="y2"))
    for xi, yi in zip(x, y_net):
        if pd.notna(yi):
            figp.add_annotation(x=xi, y=yi, xref="x", yref="y2", yshift=28,
                                text=f"{yi:.2f}%", showarrow=False, font=dict(size=12, color="#111827"))
    try:
        figp.add_hline(y=float(otp_target), line_dash="dash", line_color="red", yref="y2")
    except Exception:
        figp.add_shape(type="line", x0=-0.5, x1=len(x)-0.5, y0=float(otp_target), y1=float(otp_target),
                       xref="x", yref="y2", line=dict(color="red", dash="dash"))
    figp.update_layout(height=520, hovermode="x unified", plot_bgcolor="white",
                       margin=dict(l=40, r=40, t=40, b=80),
                       legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.0),
                       xaxis=dict(title="", tickangle=-30, tickmode="array", tickvals=x, ticktext=x, automargin=True),
                       yaxis=dict(title="Pieces (Sum)", side="left", gridcolor=GRID),
                       yaxis2=dict(title="Net OTP (%)", overlaying="y", side="right", range=[0, 130]),
                       barmode="overlay")
    st.plotly_chart(figp, use_container_width=True)
else:
    st.info("No monthly PIECES available.")

st.markdown("---")

# ---------------- Chart: Gross vs Net OTP ----------------
st.subheader("Monthly OTP Trend (Gross vs Net) — POD Month")
if not otp_pod.empty:
    otp_sorted = otp_pod.sort_values("Month_Sort")
    x       = otp_sorted["Month_Display"].tolist()
    gross_y = otp_sorted["Gross_OTP"].astype(float).tolist()
    net_y   = otp_sorted["Net_OTP"].astype(float).tolist()

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=x, y=gross_y, mode="lines+markers", name="Gross OTP",
                              line=dict(color=BLUE, width=3), marker=dict(size=7)))
    fig2.add_trace(go.Scatter(x=x, y=net_y, mode="lines+markers", name="Net OTP",
                              line=dict(color=GREEN, width=3), marker=dict(size=7)))
    for xi, yi in zip(x, gross_y):
        if pd.notna(yi):
            fig2.add_annotation(x=xi, y=yi, xref="x", yref="y", yshift=28,
                                text=f"{yi:.2f}%", showarrow=False, font=dict(size=12, color=BLUE))
    for xi, yi in zip(x, net_y):
        if pd.notna(yi):
            fig2.add_annotation(x=xi, y=yi, xref="x", yref="y", yshift=28,
                                text=f"{yi:.2f}%", showarrow=False, font=dict(size=12, color=GREEN))
    try:
        fig2.add_hline(y=float(otp_target), line_dash="dash", line_color="red")
    except Exception:
        fig2.add_shape(type="line", x0=-0.5, x1=len(x)-0.5,
                       y0=float(otp_target), y1=float(otp_target),
                       xref="x", yref="y", line=dict(color="red", dash="dash"))
    fig2.update_layout(height=460, hovermode="x unified", plot_bgcolor="white",
                       margin=dict(l=40, r=40, t=40, b=80),
                       legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.0),
                       xaxis=dict(title="", tickangle=-30, tickmode="array", tickvals=x, ticktext=x, automargin=True),
                       yaxis=dict(title="OTP (%)", range=[0, 130], gridcolor=GRID))
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("No monthly OTP trend available.")
