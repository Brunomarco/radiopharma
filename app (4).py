
import io
import sys
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime

st.set_page_config(page_title="Executive Logistics KPI Dashboard", layout="wide")

# ---------- Helper styles ----------
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
    .section {margin-top: 1.5rem;}
    </style>
    ''',
    unsafe_allow_html=True
)

st.title("Executive Logistics KPI Dashboard")

st.caption("Professional, month-on-month view for top management. Upload the raw export and map columns once.")

with st.expander("Data & KPI Definitions (click to view)", expanded=False):
    st.markdown(
        """
        **Scope Filters**
        - **Pickup Country (PU CTRY):** Only Germany (DE), Italy (IT), and Israel (IL) are included.
        
        **On-Time Performance (OTP)**
        - **Gross OTP:** On-time ratio computed over **all** shipments.
        - **Net OTP:** On-time ratio **excluding uncontrollable delays**, by keeping only rows where the **QC Name** indicates a controllable category (defaults: *Agent, Customs, Warehouse*).
        - **Objective:** 95% for Gross OTP.
        
        **Time Bases**
        - **Monthly OTP:** grouped by **Actual Delivery Date** month.
        - **Monthly Pieces:** grouped by **Actual Pickup Date** month.
        - **Monthly Charges:** grouped by **Actual Pickup Date** month (aligns revenue/cost with pickup).
        
        **Accounts Split**
        - Pieces and Total Charges are additionally broken down by **Shipper Name** (account view).
        """
    )

# ---------- File upload ----------
uploaded = st.file_uploader("Upload Excel/CSV export", type=["xlsx", "xls", "csv"])

@st.cache_data(show_spinner=False)
def read_any(file):
    name = file.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(file)
    elif name.endswith(".xlsx"):
        df = pd.read_excel(file, engine="openpyxl")
    elif name.endswith(".xls"):
        # Requires xlrd<2.0 installed
        df = pd.read_excel(file, engine="xlrd")
    else:
        raise ValueError("Unsupported file type")
    return df

if uploaded is not None:
    # Read file
    try:
        raw = read_any(uploaded)
    except Exception as e:
        st.error(f"Could not read the file: {e}")
        st.stop()

    st.success(f"Loaded file with {raw.shape[0]:,} rows and {raw.shape[1]:,} columns.")

    # Show a peek
    with st.expander("Preview raw data (first 200 rows)"):
        st.dataframe(raw.head(200), use_container_width=True)

    # ---------- Column Mapping ----------
    st.sidebar.header("Column Mapping")

    cols = list(raw.columns)
    def pick(label, candidates):
        # Convenience: try to auto-select first matching candidate
        default = None
        lowered = [c.lower() for c in cols]
        for cand in candidates:
            if cand.lower() in lowered:
                default = cols[lowered.index(cand.lower())]
                break
        return st.sidebar.selectbox(label, options=["— select —"] + cols, index=(["— select —"] + cols).index(default) if default else 0)

    pu_ctry_col     = pick("Pickup Country (PU CTRY)", ["PU CTRY","PU Country","Pickup Country","PU Country Code","PICKUP_COUNTRY"])
    qc_name_col     = pick("QC Name (Reason/Category)", ["QC name","QC Name","QC","QC Category","Exception Category"])
    adate_col       = pick("Actual Delivery Date", ["Actual Delivery Date","Delivery Actual","DELIVERED_AT","Actual Delivery","POD Date"])
    apu_col         = pick("Actual Pickup Date", ["Actual Pickup Date","Pickup Actual","PICKED_UP_AT","Actual Pickup","PU Actual"])
    pieces_col      = pick("Pieces", ["Pieces","PKGS","Packages","No. of Pieces","PIECES"])
    charges_col     = pick("Total Charges", ["Total Charges","Total Charge","TOTAL_AMOUNT","Total Cost","CHARGES","Total Price"])
    shipper_col     = pick("Shipper Name (Account)", ["Shipper Name","Shipper","Account","Customer","Client","Consignor"])
    otp_col         = st.sidebar.selectbox("OTP Status Column (optional if you prefer date comparison)", options=["— none —"] + cols, index=0)
    otp_on_values   = st.sidebar.text_input("OTP On-Time Values (comma-separated, case-insensitive)", value="ON TIME,YES,Y")

    st.sidebar.header("Filtering & Rules")
    allowed_pu = st.sidebar.multiselect("PU CTRY scope", options=sorted(raw[pu_ctry_col].dropna().unique()) if pu_ctry_col!="— select —" else ["DE","IT","IL"], default=["DE","IT","IL"])
    controllables_default = ["Agent","Customs","Warehouse"]
    controllables = st.sidebar.text_input("Controllable QC categories (comma-separated)", value=",".join(controllables_default))

    st.sidebar.header("OTP by Date Comparison (fallback if no OTP column)")
    target_date_col = st.sidebar.selectbox("Target/Promised Delivery Date (optional)", options=["— none —"] + cols, index=0)
    late_tolerance_hours = st.sidebar.number_input("Tolerance (hours) for 'on time' vs target", min_value=0, max_value=168, value=0, step=1)

    # Validate required columns
    required = {
        "Pickup Country (PU CTRY)": pu_ctry_col,
        "QC Name": qc_name_col,
        "Actual Delivery Date": adate_col,
        "Actual Pickup Date": apu_col,
        "Pieces": pieces_col,
        "Total Charges": charges_col,
        "Shipper Name": shipper_col,
    }
    missing = [k for k,v in required.items() if v == "— select —"]
    if missing:
        st.warning("Please map all required columns in the sidebar: " + ", ".join(missing))
        st.stop()

    # ---------- Type casting ----------
    df = raw.copy()

    def to_datetime_safe(s):
        # Coerce to datetime, handle strings and numbers
        return pd.to_datetime(s, errors="coerce")

    df["_pu_ctry"]   = df[pu_ctry_col].astype(str).str.strip()
    df["_qc_name"]   = df[qc_name_col].astype(str).str.strip()
    df["_adate"]     = to_datetime_safe(df[adate_col])
    df["_apu"]       = to_datetime_safe(df[apu_col])
    # Numeric casting
    def to_numeric_safe(s):
        return pd.to_numeric(s, errors="coerce")
    df["_pieces"]    = to_numeric_safe(df[pieces_col]).fillna(0)
    df["_charges"]   = to_numeric_safe(df[charges_col]).fillna(0.0)
    df["_shipper"]   = df[shipper_col].astype(str).str.strip()

    # Filter by PU countries
    df = df[df["_pu_ctry"].isin(allowed_pu)].copy()

    # Build month keys
    df["_month_adate"] = df["_adate"].dt.to_period("M").dt.to_timestamp()
    df["_month_apu"]   = df["_apu"].dt.to_period("M").dt.to_timestamp()

    # ---------- OTP logic ----------
    # 1) If OTP column is provided, treat values in otp_on_values as "on time".
    # 2) Else, if a target date is provided, on time = actual_delivery <= target + tolerance.
    otp_series = None
    if otp_col != "— none —":
        on_values = [v.strip().lower() for v in otp_on_values.split(",") if v.strip()]
        otp_series = df[otp_col].astype(str).str.lower().isin(on_values)
    elif target_date_col != "— none —":
        tcol = target_date_col
        df["_target"] = to_datetime_safe(df[tcol])
        tol = pd.to_timedelta(late_tolerance_hours, unit="h")
        otp_series = (df["_adate"] <= (df["_target"] + tol))
    else:
        st.error("Please provide either an OTP status column or a Target Delivery Date to compute OTP.")
        st.stop()

    df["_otp"] = otp_series

    # Net controllables filter
    controllable_keys = [c.strip().lower() for c in controllables.split(",") if c.strip()]
    if controllable_keys:
        import re
        pattern = "|".join([re.escape(c) for c in controllable_keys])
        df["_is_controllable"] = df["_qc_name"].str.lower().str.contains(pattern)
    else:
        df["_is_controllable"] = False

    # ---------- Aggregations ----------
    # OTP by Actual Delivery Month
    tmp = df.dropna(subset=["_month_adate"]).copy()
    otp_grp = tmp.groupby("_month_adate", as_index=False).agg(
        total=("_otp", "count"),
        on_time=("_otp", "sum")
    )
    otp_grp["gross_otp_pct"] = (otp_grp["on_time"] / otp_grp["total"]).replace([np.inf, -np.inf], np.nan)

    # Net OTP by controllables only
    df_ctrl = tmp[tmp["_is_controllable"]].copy()
    otp_net_grp = df_ctrl.groupby("_month_adate", as_index=False).agg(
        total=("_otp", "count"),
        on_time=("_otp", "sum")
    )
    otp_net_grp["net_otp_pct"] = (otp_net_grp["on_time"] / otp_net_grp["total"]).replace([np.inf, -np.inf], np.nan)

    # Pieces per month by Actual Pickup
    tmp2 = df.dropna(subset=["_month_apu"]).copy()
    pieces_grp = tmp2.groupby("_month_apu", as_index=False).agg(
        pieces=("_pieces", "sum")
    )

    # Charges per month by Actual Pickup
    charges_grp = tmp2.groupby("_month_apu", as_index=False).agg(
        total_charges=("_charges", "sum")
    )

    # Account splits (by shipper)
    shipper_pieces = tmp2.groupby(["_month_apu","_shipper"], as_index=False).agg(
        pieces=("_pieces","sum")
    )
    shipper_charges = tmp2.groupby(["_month_apu","_shipper"], as_index=False).agg(
        total_charges=("_charges","sum")
    )

    # ---------- Executive KPI cards ----------
    latest_month = otp_grp["_month_adate"].max() if not otp_grp.empty else None
    latest_row = otp_grp.loc[otp_grp["_month_adate"]==latest_month] if latest_month is not None else pd.DataFrame()
    gross_latest = float(latest_row["gross_otp_pct"].iloc[0]) if not latest_row.empty else np.nan

    net_latest_month = otp_net_grp["_month_adate"].max() if not otp_net_grp.empty else None
    net_latest_row = otp_net_grp.loc[otp_net_grp["_month_adate"]==net_latest_month] if net_latest_month is not None else pd.DataFrame()
    net_latest = float(net_latest_row["net_otp_pct"].iloc[0]) if not net_latest_row.empty else np.nan

    objective = 0.95
    def status_class(val, objective):
        if np.isnan(val):
            return "bad"
        if val >= objective:
            return "ok"
        elif val >= objective - 0.05:
            return "warn"
        return "bad"

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="kpi-title">Gross OTP (latest month)</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="kpi-value {status_class(gross_latest, objective)}'>{"" if np.isnan(gross_latest) else f"{gross_latest*100:.1f}%"}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="kpi-sub">Objective: 95%</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="kpi-title">Net OTP (controllables)</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="kpi-value">{"" if np.isnan(net_latest) else f"{net_latest*100:.1f}%"}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="kpi-sub">QC includes: {", ".join(controllable_keys) if controllable_keys else "—"}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        total_pieces = df["_pieces"].sum()
        total_charges = df["_charges"].sum()
        st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="kpi-title">Scope</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="kpi-value">{int(total_pieces):,} pcs</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="kpi-sub">Total Charges: {total_charges:,.2f}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ---------- Charts & Tables ----------
    st.markdown("### OTP by Month (Actual Delivery Date)")
    if not otp_grp.empty:
        otp_chart = otp_grp.set_index("_month_adate")[["gross_otp_pct"]]
        st.line_chart(otp_chart)
    else:
        st.info("No data to plot for Gross OTP.")

    st.markdown("### Net OTP by Month (Controllables)")
    if not otp_net_grp.empty:
        otp_net_chart = otp_net_grp.set_index("_month_adate")[["net_otp_pct"]]
        st.line_chart(otp_net_chart)
    else:
        st.info("No data to plot for Net OTP.")

    colA, colB = st.columns(2)
    with colA:
        st.markdown("### Pieces per Month (Actual Pickup Date)")
        if not pieces_grp.empty:
            st.bar_chart(pieces_grp.set_index("_month_apu")[["pieces"]])
        else:
            st.info("No data to plot for Pieces.")
    with colB:
        st.markdown("### Total Charges per Month (Actual Pickup Date)")
        if not charges_grp.empty:
            st.bar_chart(charges_grp.set_index("_month_apu")[["total_charges"]])
        else:
            st.info("No data to plot for Charges.")

    st.markdown("### Account Breakdown (Shipper) – Pieces")
    st.dataframe(shipper_pieces.rename(columns={"_month_apu":"Month","_shipper":"Shipper"}), use_container_width=True)

    st.markdown("### Account Breakdown (Shipper) – Total Charges")
    st.dataframe(shipper_charges.rename(columns={"_month_apu":"Month","_shipper":"Shipper"}), use_container_width=True)

    # ---------- Downloads ----------
    def to_csv_bytes(df):
        return df.to_csv(index=False).encode("utf-8")

    st.download_button("Download OTP (Gross) CSV", data=to_csv_bytes(otp_grp.rename(columns={
        "_month_adate":"Month","gross_otp_pct":"Gross OTP %","on_time":"On Time","total":"Total"
    })), file_name="otp_gross_by_month.csv")

    st.download_button("Download OTP (Net, Controllables) CSV", data=to_csv_bytes(otp_net_grp.rename(columns={
        "_month_adate":"Month","net_otp_pct":"Net OTP %","on_time":"On Time","total":"Total"
    })), file_name="otp_net_by_month.csv")

    st.download_button("Download Pieces by Month CSV", data=to_csv_bytes(pieces_grp.rename(columns={
        "_month_apu":"Month","pieces":"Pieces"
    })), file_name="pieces_by_month.csv")

    st.download_button("Download Charges by Month CSV", data=to_csv_bytes(charges_grp.rename(columns={
        "_month_apu":"Month","total_charges":"Total Charges"
    })), file_name="charges_by_month.csv")

    st.caption("© Your Company — Executive Dashboard. Built with Streamlit.")
else:
    st.info("Upload an Excel/CSV file to get started.")
