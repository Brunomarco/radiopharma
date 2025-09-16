
import io
import re
import hashlib
import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(page_title="Executive Logistics KPI Dashboard", layout="wide")

st.markdown(
    '''
    <style>
    .kpi-card {padding: 1rem; border: 1px solid #e6e6e6; border-radius: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.06);}
    .kpi-title {font-size: 0.9rem; color: #5a5a5a; margin-bottom: 0.25rem;}
    .kpi-value {font-size: 1.6rem; font-weight: 700; margin-bottom: 0.25rem;}
    .kpi-sub {font-size: 0.85rem; color: #6f6f6f;}
    .ok {color: #0b8a42;}
    .warn {color: #b26a00;}
    .bad {color: #b00020;}
    </style>
    ''', unsafe_allow_html=True
)

st.title("Executive Logistics KPI Dashboard — Optimized")

st.caption("Fast, executive-ready KPIs. Upload the export, map columns, and go. Heavy tables are behind expanders for speed.")

# ------------------ Helpers ------------------
def df_hash_bytes(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()

@st.cache_data(show_spinner=False)
def read_any_cached(file_bytes: bytes, name: str) -> pd.DataFrame:
    """Fast cached reader for CSV/XLSX/XLS using in-memory bytes."""
    name = name.lower()
    bio = io.BytesIO(file_bytes)
    if name.endswith('.csv'):
        return pd.read_csv(bio)
    elif name.endswith('.xlsx'):
        return pd.read_excel(bio, engine='openpyxl')
    elif name.endswith('.xls'):
        # xlrd<2.0 needed, but Streamlit cloud will install via requirements
        return pd.read_excel(bio, engine='xlrd')
    else:
        raise ValueError('Unsupported file type')

def to_datetime_safe(s):
    return pd.to_datetime(s, errors='coerce')

def to_numeric_safe(s):
    return pd.to_numeric(s, errors='coerce')

@st.cache_data(show_spinner=False)
def compute_aggregations(dfmini: pd.DataFrame,
                         allowed_pu: tuple,
                         otp_col: str,
                         otp_on_values: tuple,
                         target_col: str,
                         tolerance_h: int,
                         controllables: tuple):
    """All heavy groupbys cached. dfmini only has mapped columns to minimize hashing cost."""
    df = dfmini.copy()

    # Filter by PU
    df = df[df['PU_CTRY'].isin(allowed_pu)].copy()

    # Build months
    df['_month_adate'] = df['ADATE'].dt.to_period('M').dt.to_timestamp()
    df['_month_apu']   = df['APU'].dt.to_period('M').dt.to_timestamp()

    # OTP logic
    if otp_col:
        on_vals = {v.strip().lower() for v in otp_on_values if v.strip()}
        df['_otp'] = df['OTP_SRC'].astype(str).str.lower().isin(on_vals)
    elif target_col:
        tol = pd.to_timedelta(tolerance_h, unit='h')
        df['_otp'] = df['ADATE'] <= (df['TARGET'] + tol)
    else:
        raise ValueError('Need OTP status column or Target/Promised Date.')

    # Net controllables
    if controllables:
        pattern = re.compile('|'.join([re.escape(c.lower()) for c in controllables]), flags=re.I)
        df['_is_ctrl'] = df['QC'].astype(str).str.contains(pattern, regex=True, na=False)
    else:
        df['_is_ctrl'] = False

    # OTP (gross)
    g = df.dropna(subset=['_month_adate']).groupby('_month_adate', as_index=False).agg(
        total=('_otp','count'),
        on_time=('_otp','sum')
    )
    g['gross_otp_pct'] = (g['on_time'] / g['total']).replace([np.inf, -np.inf], np.nan)

    # OTP (net controllables)
    dfn = df[df['_is_ctrl']].dropna(subset=['_month_adate'])
    if dfn.empty:
        n = pd.DataFrame(columns=['_month_adate','total','on_time','net_otp_pct'])
    else:
        n = dfn.groupby('_month_adate', as_index=False).agg(
            total=('_otp','count'),
            on_time=('_otp','sum')
        )
        n['net_otp_pct'] = (n['on_time'] / n['total']).replace([np.inf, -np.inf], np.nan)

    # Pieces & Charges per month by APU
    tmp2 = df.dropna(subset=['_month_apu'])
    pieces = tmp2.groupby('_month_apu', as_index=False)['PIECES'].sum().rename(columns={'PIECES':'pieces'})
    charges = tmp2.groupby('_month_apu', as_index=False)['CHARGES'].sum().rename(columns={'CHARGES':'total_charges'})

    # Account splits (behind expander in UI)
    shipper_p = tmp2.groupby(['_month_apu','SHIPPER'], as_index=False)['PIECES'].sum().rename(columns={'_month_apu':'Month','SHIPPER':'Shipper','PIECES':'pieces'})
    shipper_c = tmp2.groupby(['_month_apu','SHIPPER'], as_index=False)['CHARGES'].sum().rename(columns={'_month_apu':'Month','SHIPPER':'Shipper','CHARGES':'total_charges'})

    return g, n, pieces, charges, shipper_p, shipper_c, df  # return df for totals

# ------------------ UI ------------------
uploaded = st.file_uploader("Upload Excel/CSV export", type=['xlsx','xls','csv'])

if uploaded is None:
    st.info("Upload a file to begin.")
    st.stop()

with st.spinner("Reading file..."):
    file_bytes = uploaded.read()
    raw = read_any_cached(file_bytes, uploaded.name)

st.success(f"Loaded {raw.shape[0]:,} rows, {raw.shape[1]:,} columns.")

# Preview (limited for speed)
with st.expander("Preview (first 50 rows)"):
    st.dataframe(raw.head(50), use_container_width=True)

# Column mapping
st.sidebar.header("Column Mapping")
cols = list(raw.columns)

def select_with_guess(label, guesses):
    default_idx = 0
    lowered = [c.lower() for c in cols]
    for g in guesses:
        if g.lower() in lowered:
            default_idx = lowered.index(g.lower()) + 1
            break
    return st.sidebar.selectbox(label, ["— select —"] + cols, index=default_idx)

pu_ctry = select_with_guess("Pickup Country (PU CTRY)", ["PU CTRY","PU Country","Pickup Country","PU Country Code"])
qc_name = select_with_guess("QC Name", ["QC name","QC Name","QC","QC Category","Exception Category"])
adate   = select_with_guess("Actual Delivery Date", ["Actual Delivery Date","Delivery Actual","POD Date","Delivered At"])
apu     = select_with_guess("Actual Pickup Date", ["Actual Pickup Date","Pickup Actual","PU Actual","Picked Up At"])
pieces  = select_with_guess("Pieces", ["Pieces","PKGS","Packages","PIECES","No. of Pieces"])
charges = select_with_guess("Total Charges", ["Total Charges","Total Charge","CHARGES","Total Cost","TOTAL_AMOUNT"])
shipper = select_with_guess("Shipper Name (Account)", ["Shipper Name","Shipper","Account","Customer","Client","Consignor"])

otp_status_col = st.sidebar.selectbox("OTP Status Column (preferred)", ["— none —"] + cols, index=0)
otp_values = st.sidebar.text_input("On-time values (comma-separated)", value="ON TIME,YES,Y")
target_col = st.sidebar.selectbox("Target/Promised Delivery (fallback)", ["— none —"] + cols, index=0)
tol_h = st.sidebar.number_input("Tolerance (hours)", min_value=0, max_value=168, value=0, step=1)

allowed = st.sidebar.multiselect("PU CTRY scope", options=sorted(raw[pu_ctry].dropna().astype(str).unique()) if pu_ctry != "— select —" else ["DE","IT","IL"], default=["DE","IT","IL"])
controllables_txt = st.sidebar.text_input("Controllable QC categories", value="Agent,Customs,Warehouse")

# Validate mapping
required = {"PU CTRY": pu_ctry, "QC Name": qc_name, "Actual Delivery": adate, "Actual Pickup": apu, "Pieces": pieces, "Total Charges": charges, "Shipper": shipper}
missing = [k for k,v in required.items() if v == "— select —"]
if missing:
    st.warning("Map all required columns in the sidebar: " + ", ".join(missing))
    st.stop()

# Build minimal dataframe for faster caching/hashing
with st.spinner("Preparing data..."):
    dfmini = pd.DataFrame({
        "PU_CTRY": raw[pu_ctry].astype(str).str.strip(),
        "QC": raw[qc_name].astype(str).str.strip(),
        "ADATE": pd.to_datetime(raw[adate], errors='coerce'),
        "APU": pd.to_datetime(raw[apu], errors='coerce'),
        "PIECES": pd.to_numeric(raw[pieces], errors='coerce').fillna(0).astype(np.float64),
        "CHARGES": pd.to_numeric(raw[charges], errors='coerce').fillna(0.0).astype(np.float64),
        "SHIPPER": raw[shipper].astype(str).str.strip()
    })
    # Optional OTP/Target sources
    if otp_status_col != "— none —":
        dfmini["OTP_SRC"] = raw[otp_status_col]
    else:
        dfmini["OTP_SRC"] = ""  # empty
    if target_col != "— none —":
        dfmini["TARGET"] = pd.to_datetime(raw[target_col], errors='coerce')
    else:
        dfmini["TARGET"] = pd.NaT

# Compute aggregations (cached)
with st.spinner("Computing KPIs..."):
    g, n, pieces_m, charges_m, shipper_p, shipper_c, scoped_df = compute_aggregations(
        dfmini=dfmini,
        allowed_pu=tuple(allowed),
        otp_col=(otp_status_col != "— none —"),
        otp_on_values=tuple([v.strip() for v in otp_values.split(",")]),
        target_col=(target_col != "— none —"),
        tolerance_h=int(tol_h),
        controllables=tuple([c.strip() for c in controllables_txt.split(",") if c.strip()]),
    )

# Executive KPI cards
objective = 0.95
def status_class(val, objective=objective):
    if pd.isna(val):
        return "bad"
    if val >= objective:
        return "ok"
    elif val >= objective - 0.05:
        return "warn"
    return "bad"

latest_g = g["_month_adate"].max() if not g.empty else None
gross_latest = float(g.loc[g["_month_adate"]==latest_g, "gross_otp_pct"].iloc[0]) if latest_g is not None and not g.empty else np.nan

latest_n = n["_month_adate"].max() if not n.empty else None
net_latest = float(n.loc[n["_month_adate"]==latest_n, "net_otp_pct"].iloc[0]) if latest_n is not None and not n.empty else np.nan

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
    st.markdown('<div class="kpi-title">Gross OTP (latest month)</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="kpi-value {status_class(gross_latest)}'>{("" if pd.isna(gross_latest) else f"{gross_latest*100:.1f}%")}</div>', unsafe_allow_html=True)
    st.markdown('<div class="kpi-sub">Objective: 95%</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with c2:
    st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
    st.markdown('<div class="kpi-title">Net OTP (controllables)</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="kpi-value">{("" if pd.isna(net_latest) else f"{net_latest*100:.1f}%")}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="kpi-sub">QC: {controllables_txt}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with c3:
    total_p = int(scoped_df["PIECES"].sum())
    total_c = float(scoped_df["CHARGES"].sum())
    st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
    st.markdown('<div class="kpi-title">Scope totals</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="kpi-value">{total_p:,} pcs</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="kpi-sub">Total Charges: {total_c:,.2f}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Charts
st.markdown("### OTP by Month (Actual Delivery Date) — Gross")
if not g.empty:
    st.line_chart(g.set_index("_month_adate")[["gross_otp_pct"]])
else:
    st.info("No data for Gross OTP.")

st.markdown("### OTP by Month (Actual Delivery Date) — Net (Controllables)")
if not n.empty:
    st.line_chart(n.set_index("_month_adate")[["net_otp_pct"]])
else:
    st.info("No data for Net OTP.")

c4, c5 = st.columns(2)
with c4:
    st.markdown("### Pieces per Month (Actual Pickup Date)")
    if not pieces_m.empty:
        st.bar_chart(pieces_m.set_index("_month_apu")[["pieces"]])
    else:
        st.info("No data for Pieces.")
with c5:
    st.markdown("### Total Charges per Month (Actual Pickup Date)")
    if not charges_m.empty:
        st.bar_chart(charges_m.set_index("_month_apu")[["total_charges"]])
    else:
        st.info("No data for Charges.")

# Heavy tables behind expanders
with st.expander("Account Breakdown — Pieces (by Shipper)", expanded=False):
    st.dataframe(shipper_p, use_container_width=True)

with st.expander("Account Breakdown — Total Charges (by Shipper)", expanded=False):
    st.dataframe(shipper_c, use_container_width=True)

st.caption("Optimized build — caching, minimized hashing, and deferred heavy tables for responsiveness.")
