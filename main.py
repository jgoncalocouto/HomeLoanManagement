# mortgage_app_streamlit_v50_amort.py
# New:
# - Amortizações antecipadas (custom schedule + strategy + fee)
# - Baseline vs with-amort KPIs (juros poupados, comissões, poupança líquida, meses poupados)
# - Markers for amortizations on charts; columns in tables
# - Scenario save/load includes amortizations
# - Keeps your "Posição atual" flow and DOES NOT compute TAEG (we'll revisit later)

import json
import os
from datetime import datetime, date

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# =================== EARLY: apply pending inputs BEFORE any widgets ===================
if "pending_inputs" in st.session_state:
    for k, v in st.session_state["pending_inputs"].items():
        st.session_state[k] = v
    del st.session_state["pending_inputs"]

# ========== App Config
st.set_page_config(page_title="Gestão de Crédito Habitação", page_icon="🏠", layout="wide")
st.title("🏠 Gestão de Crédito Habitação")

# ========== Defaults & Paths
DEF_EUR = 3.0
DEF_SPREAD = 1.2

DEFAULT_BASE_DIR = os.getenv("MORTGAGE_APP_DIR", os.path.join(os.path.expanduser("~"), "mortgage_app"))
st.session_state.setdefault("base_dir", DEFAULT_BASE_DIR)
st.session_state.setdefault("scen_dir", os.path.join(st.session_state["base_dir"], "scenarios"))
os.makedirs(st.session_state["scen_dir"], exist_ok=True)

# ========== Helpers
def eur(x: float) -> str:
    try:
        s = f"{x:,.2f}".replace(",", "_").replace(".", ",").replace("_", ".")
        return f"{s} €"
    except Exception:
        return f"{x} €"

def per(x: float) -> str:
    """Format percentage in pt-PT style (comma as decimal)."""
    try:
        s = f"{x*100:,.2f}".replace(",", "_").replace(".", ",").replace("_", ".")
        return f"{s} %"
    except Exception:
        return f"{x*100} %"

def pmt(monthly_rate: float, nper: int, pv: float) -> float:
    if nper <= 0:
        return 0.0
    if abs(monthly_rate) < 1e-12:
        return pv / nper
    return pv * (monthly_rate * (1 + monthly_rate) ** nper) / ((1 + monthly_rate) ** nper - 1)

def _coerce_df(val, fallback: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame from a Streamlit data_editor value or fallback."""
    if isinstance(val, pd.DataFrame):
        return val.copy()
    if isinstance(val, dict):
        if "data" in val and isinstance(val["data"], list):
            try:
                return pd.DataFrame(val["data"])
            except Exception:
                return fallback.copy()
        try:
            return pd.DataFrame(val)
        except Exception:
            return fallback.copy()
    if val is None:
        return fallback.copy()
    try:
        return pd.DataFrame(val)
    except Exception:
        return fallback.copy()

def _clean_euribor_points(pts: pd.DataFrame, default_pct: float) -> pd.DataFrame:
    """Return ['month','euribor'] table (month = month-start). Accepts 'euribor_pct', comma decimals."""
    df = pts.copy() if pts is not None else pd.DataFrame(columns=["date", "euribor"])
    if "euribor" not in df.columns and "euribor_pct" in df.columns:
        df = df.rename(columns={"euribor_pct": "euribor"})
    if "date" not in df.columns:
        df["date"] = pd.NaT
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp(how="start")
    if "euribor" not in df.columns:
        df["euribor"] = default_pct
    df["euribor"] = df["euribor"].astype(str).str.replace(",", ".", regex=False)
    df["euribor"] = pd.to_numeric(df["euribor"], errors="coerce")
    df = (
        df.dropna(subset=["month"])
          .sort_values("month")
          .drop_duplicates("month", keep="last")
          [["month", "euribor"]]
    )
    return df

def build_euribor_path(start_date: date, months: int, pts: pd.DataFrame, default_pct: float) -> pd.Series:
    """
    Monthly Euribor via merge_asof:
      - For each month, take last change-point ≤ month (backward).
      - If no past value at start, anchor with first future point (not default).
      - Only if the table is empty → default_pct across horizon.
    """
    idx = pd.date_range(start=start_date, periods=int(months), freq="MS")
    left = pd.DataFrame({"month": idx})
    df = _clean_euribor_points(pts, default_pct)
    if df.empty:
        return pd.Series(default_pct, index=idx, name="euribor_pct")
    right = df.sort_values("month").reset_index(drop=True)
    merged = pd.merge_asof(left, right, on="month", direction="backward", allow_exact_matches=True)
    first_future_val = right["euribor"].dropna().iloc[0] if not right["euribor"].dropna().empty else default_pct
    merged["euribor"].fillna(first_future_val, inplace=True)
    s = merged.set_index("month")["euribor"].astype(float)
    s.name = "euribor_pct"
    return s

def _enrich_with_now(df: pd.DataFrame, ref_date: date):
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.date
    out["cum_payment"]   = out["payment"].cumsum()
    out["cum_interest"]  = out["interest"].cumsum()
    out["cum_principal"] = out["principal"].cumsum()

    # if ref_date is before all dates → idx_now = 0
    # if ref_date is after all dates → idx_now = len(out)
    if (out["date"] >= ref_date).any():
        idx_now = int((out["date"] >= ref_date).idxmax())
    else:
        idx_now = len(out)

    out["is_past"] = out.index < idx_now
    out["is_future"] = out.index >= idx_now
    out["now_flag"] = ""
    if 0 <= idx_now < len(out):
        out.loc[idx_now, "now_flag"] = "← agora"

    return out, idx_now

# ---------- Amortizações helpers ----------
def _amort_map_from_df(start_date: date, months: int, amort_df: pd.DataFrame, fee_pct: float) -> dict:
    """
    Build a map {period_index (0-based) -> (amount, fee)} aligned to monthly schedule dates.
    Sums multiple amortizations in the same month. Ignores invalid rows.
    """
    if amort_df is None or amort_df.empty:
        return {}
    df = amort_df.copy()
    if "date" not in df.columns or "amount" not in df.columns:
        return {}
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
    df = df.dropna(subset=["date"])
    if df.empty:
        return {}

    schedule_idx = pd.date_range(start=start_date, periods=months, freq="MS")
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp(how="start")
    month_to_i = {d: i for i, d in enumerate(schedule_idx)}
    rows = {}
    for _, r in df.iterrows():
        m = r["month"]
        amt = max(float(r["amount"]), 0.0)
        if m in month_to_i and amt > 0:
            i = month_to_i[m]
            prev_amt, prev_fee = rows.get(i, (0.0, 0.0))
            fee = amt * (float(fee_pct) / 100.0)
            rows[i] = (prev_amt + amt, prev_fee + fee)
    return rows

def _apply_amortization(bal: float, pay: float, rm: float, keep_payment: bool, months_left: int):
    """
    Recompute payment optionally after amortization.
    keep_payment=True -> Reduzir prazo (keep installment), False -> Reduzir prestação (reprice).
    """
    if keep_payment:
        return pay, months_left
    if months_left <= 0:
        return 0.0, 0
    if abs(rm) < 1e-12:
        return bal / months_left, months_left
    new_pay = pmt(rm, months_left, bal)
    return new_pay, months_left

# ========== Schedules (amort-aware)
def schedule_fixed(start_date, principal, annual_rate_pct, years, monthly_extra_sum,
                   upfront_fees_sum, amort_map=None, amort_keep_payment=True):
    """
    amort_map: dict {i -> (amort_amount, amort_fee)}, i is 0-based month index.
    amort_keep_payment: True => Reduzir prazo (keep installment), False => Reduzir prestação.
    """
    months = int(years * 12)
    r_m = (annual_rate_pct / 100.0) / 12.0
    pay = pmt(r_m, months, principal)
    bal = principal
    dates = pd.date_range(start=start_date, periods=months, freq="MS")
    rows = []
    amort_map = amort_map or {}

    i = 0
    while i < months and bal > 0:
        # apply amortization at the start of the month
        amort_amt, amort_fee = amort_map.get(i, (0.0, 0.0))
        if amort_amt > 0:
            amort_amt = min(amort_amt, bal)
            bal -= amort_amt
            months_left = months - i
            pay, _ = _apply_amortization(bal, pay, r_m, keep_payment=amort_keep_payment, months_left=months_left)

        # monthly interest
        interest = bal * r_m
        principal_component = min(pay - interest, bal)
        last = (i == months - 1) or (principal_component >= bal)

        if last:
            principal_component = bal
            pay_eff = interest + principal_component
        else:
            pay_eff = pay

        new_bal = bal - principal_component

        rows.append({
            "period": i + 1,
            "date": dates[i].date(),
            "rate_annual_pct": round(annual_rate_pct, 4),
            "payment": round(pay_eff + monthly_extra_sum + amort_fee, 2),  # include fee in payment for that month
            "interest": round(interest, 2),
            "principal": round(principal_component, 2),
            "extras_monthly": round(monthly_extra_sum, 2),
            "amortization": round(amort_amt, 2),
            "amort_fee": round(amort_fee, 2),
            "balance": round(max(new_bal, 0.0), 2),
            "reset": False,
        })

        bal = new_bal
        i += 1
        if bal <= 0:
            break

    df = pd.DataFrame(rows)
    df.attrs["upfront_fees_total"] = float(upfront_fees_sum)
    return df

def schedule_variable(start_date, principal, spread_pct, years, reset_months, euribor_series,
                      monthly_extra_sum, upfront_fees_sum, amort_map=None, amort_keep_payment=True):
    months = int(years * 12)
    dates = pd.date_range(start=start_date, periods=months, freq="MS")
    rows, bal = [], principal
    amort_map = amort_map or {}
    k = 0
    while k < months and bal > 0:
        eur = float(euribor_series.iloc[k])
        annual_rate_pct = eur + spread_pct
        r_m = (annual_rate_pct / 100.0) / 12.0
        next_boundary = (k // reset_months + 1) * reset_months
        block_len = min(months - k, next_boundary - k)
        pay = pmt(r_m, months - k, bal)

        for off in range(block_len):
            i = k + off

            # apply amortization at start of month
            amort_amt, amort_fee = amort_map.get(i, (0.0, 0.0))
            if amort_amt > 0:
                amort_amt = min(amort_amt, bal)
                bal -= amort_amt
                months_left = months - i
                pay, _ = _apply_amortization(bal, pay, r_m, keep_payment=amort_keep_payment, months_left=months_left)

            interest = bal * r_m
            principal_component = min(pay - interest, bal)
            last = (i == months - 1) or (principal_component >= bal)

            if last:
                principal_component = bal
                pay_eff = interest + principal_component
            else:
                pay_eff = pay

            new_bal = bal - principal_component

            rows.append({
                "period": i + 1,
                "date": dates[i].date(),
                "rate_annual_pct": round(annual_rate_pct, 4),
                "payment": round(pay_eff + monthly_extra_sum + amort_fee, 2),
                "interest": round(interest, 2),
                "principal": round(principal_component, 2),
                "extras_monthly": round(monthly_extra_sum, 2),
                "amortization": round(amort_amt, 2),
                "amort_fee": round(amort_fee, 2),
                "balance": round(max(new_bal, 0.0), 2),
                "reset": (off == 0),
            })

            bal = new_bal
            if bal <= 0:
                break

        k += block_len
    df = pd.DataFrame(rows)
    df.attrs["upfront_fees_total"] = float(upfront_fees_sum)
    return df

# ========== Scenario loader (staged; rotate keys; reset current_df caches)
def apply_scenario_dict(obj: dict):
    ins = obj.get("inputs", {})
    pending = {}
    pending["principal"] = float(ins.get("principal", 250_000.0))
    pending["years"] = int(ins.get("years", 30))
    pending["start"] = pd.to_datetime(ins.get("start", pd.Timestamp.today().date())).date()
    pending["rate_type"] = ins.get("rate_type", "Fixa")
    if pending["rate_type"] == "Fixa":
        pending["fixed_rate_pct"] = float(ins.get("fixed_rate_pct", 4.0))
    else:
        pending["spread_pct"] = float(ins.get("spread_pct", DEF_SPREAD))
        pending["reset_months"] = int(ins.get("reset_months", 12))

    # New initial editor data for Euribor/Despesas
    if pending["rate_type"] != "Fixa":
        ef = pd.DataFrame(obj.get("euribor_forecast", []))
        if not ef.empty:
            if "euribor_pct" in ef.columns and "euribor" not in ef.columns:
                ef = ef.rename(columns={"euribor_pct": "euribor"})
            ef["date"] = pd.to_datetime(ef["date"], errors="coerce").dt.date
        else:
            ef = pd.DataFrame({"date": [pending["start"]], "euribor": [DEF_EUR]})
        pending["eur_init_df"] = ef
    pending["upfront_init_df"] = pd.DataFrame(obj.get("fees_upfront", [{"name": "Comissão", "value": 0.0}]))
    pending["monthly_init_df"] = pd.DataFrame(obj.get("fees_monthly", [{"name": "Seguro", "value": 0.0}]))

    stres = obj.get("stress", {})
    def _load_tbl(rec_key, cols, default):
        df = pd.DataFrame(stres.get(rec_key, []))
        if df.empty:
            return default
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
        return df[["date"] + cols]
    
    pending["income_init_df"] = _load_tbl("income_anchors", ["income"], pd.DataFrame({"date":[pending["start"]], "income":[2000.0]}))
    pending["income_current_df"] = pending["income_init_df"].copy()
    pending["income_growth_init_df"] = _load_tbl("income_growth", ["growth_pct"], pd.DataFrame({"date":[pending["start"]], "growth_pct":[0.0]}))
    pending["income_growth_current_df"] = pending["income_growth_init_df"].copy()
    
    pending["expenses_init_df"] = _load_tbl("expenses_anchors", ["expenses"], pd.DataFrame({"date":[pending["start"]], "expenses":[1200.0]}))
    pending["expenses_current_df"] = pending["expenses_init_df"].copy()
    pending["inflation_init_df"] = _load_tbl("inflation", ["inflation_pct"], pd.DataFrame({"date":[pending["start"]], "inflation_pct":[2.0]}))
    pending["inflation_current_df"] = pending["inflation_init_df"].copy()
    pending["effort_threshold_pct"] = float(stres.get("effort_threshold_pct", 35.0))
    
    # rotate keys
    stamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
    pending["income_tbl_key"] = f"income_tbl_{stamp}"
    pending["income_growth_tbl_key"] = f"income_growth_tbl_{stamp}"
    pending["expenses_tbl_key"] = f"expenses_tbl_{stamp}"
    pending["inflation_tbl_key"] = f"inflation_tbl_{stamp}"
    pending["eur_csv_upl_key"] = f"eur_csv_{stamp}"
    pending["upfront_csv_upl_key"]   = f"upfront_csv_{stamp}"
    pending["amort_csv_upl_key"]     = f"amort_csv_{stamp}"
    pending["inflation_csv_upl_key"] = f"inflation_csv_{stamp}"

    # Amortizações from scenario
    amdf = pd.DataFrame(obj.get("amortizations", []))
    if not amdf.empty:
        amdf["date"] = pd.to_datetime(amdf["date"], errors="coerce").dt.date
    else:
        amdf = pd.DataFrame({"date": [pending["start"]], "amount": [0.0]})
    pending["amort_init_df"] = amdf
    pending["amort_current_df"] = amdf.copy()
    pending["amort_strategy"] = obj.get("amort_strategy", "Reduzir prazo")
    pending["amort_fee_pct"] = float(obj.get("amort_fee_pct", 0.5))

    # Reset current caches to the new initial values so UI + compute align
    pending["eur_current_df"] = pending.get("eur_init_df", st.session_state.get("eur_init_df", pd.DataFrame()))
    pending["upfront_current_df"] = pending.get("upfront_init_df", st.session_state.get("upfront_init_df", pd.DataFrame()))
    pending["monthly_current_df"] = pending.get("monthly_init_df", st.session_state.get("monthly_init_df", pd.DataFrame()))

    # Rotate widget keys
    stamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
    pending["eur_tbl_key"] = f"eur_tbl_{stamp}"
    pending["upf_tbl_key"] = f"upf_tbl_{stamp}"
    pending["mon_tbl_key"] = f"mon_tbl_{stamp}"
    pending["amort_tbl_key"] = f"amort_tbl_{stamp}"

    pending["force_recalc"] = True
    st.session_state["pending_inputs"] = pending
    (st.rerun() if hasattr(st, "rerun") else st.experimental_rerun())

def _clean_anchor(df, val_col):
    """Normalize an anchors table into columns ['month', val_col]."""
    # Accept None, dict, DataFrame
    if df is None:
        df = pd.DataFrame()
    elif isinstance(df, dict):
        df = pd.DataFrame(df)
    else:
        df = df.copy()

    if "date" not in df.columns or val_col not in df.columns:
        return pd.DataFrame(columns=["month", val_col])

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp(how="start")
    df = (
        df.dropna(subset=["month", val_col])
          .sort_values("month")
          .drop_duplicates("month", keep="last")
          [["month", val_col]]
    )
    return df


def _series_from_anchors(start_date, months, anchor_df, val_col, default_val):
    """
    Build a monthly Series by forward-filling anchor values.
    Returns a Series indexed by month starts.
    """
    idx = pd.date_range(start=start_date, periods=int(months), freq="MS")
    right = _clean_anchor(anchor_df, val_col)
    if right.empty:
        return pd.Series(default_val, index=idx, name=val_col)

    left = pd.DataFrame({"month": idx})
    merged = pd.merge_asof(
        left.sort_values("month"),
        right.sort_values("month"),
        on="month",
        direction="backward",
        allow_exact_matches=True,
    )
    # If no past anchor at the very start, seed with the first future anchor's value
    first_val = right[val_col].iloc[0]
    merged[val_col] = merged[val_col].fillna(first_val)

    s = merged.set_index("month")[val_col].astype(float)
    s.name = val_col
    return s


def _annual_pct_to_monthly_series(start_date, months, pct_df, pct_col, default_pct):
    """
    Convert an annual-% anchors table into a monthly rate Series:
      r_m = (1 + r_annual/100)**(1/12) - 1
    """
    ann = _series_from_anchors(start_date, months, pct_df, pct_col, default_pct)
    rm = (1.0 + ann / 100.0) ** (1.0 / 12.0) - 1.0
    rm.name = pct_col + "_monthly"
    return rm


def _apply_growth_from_anchor(base_anchors_df, monthly_rate_series):
    """
    Build a grown series from base anchors + monthly growth rates.
    - Anchors are filtered to the index range, and we seed the first index month
      with the first available anchor value (in-range if present, otherwise the
      first future anchor) so compounding always has a base.
    - Compounds piecewise between anchors (resets base at each anchor month).
    """
    idx = monthly_rate_series.index  # DatetimeIndex of month-starts (MS)
    if base_anchors_df is None:
        base_anchors_df = pd.DataFrame()
    elif isinstance(base_anchors_df, dict):
        base_anchors_df = pd.DataFrame(base_anchors_df)
    else:
        base_anchors_df = base_anchors_df.copy()

    # Detect value column ('income' or 'expenses')
    if base_anchors_df.empty or "date" not in base_anchors_df.columns:
        return pd.Series(0.0, index=idx)
    value_cols = [c for c in base_anchors_df.columns if c != "date"]
    if not value_cols:
        return pd.Series(0.0, index=idx)
    val_col = value_cols[-1]

    # Normalize anchors -> ['month', val_col]
    anchors = base_anchors_df.copy()
    anchors["month"] = pd.to_datetime(anchors["date"], errors="coerce").dt.to_period("M").dt.to_timestamp(how="start")
    anchors[val_col] = pd.to_numeric(anchors[val_col], errors="coerce")
    anchors = (
        anchors.dropna(subset=["month", val_col])
               .sort_values("month")
               .drop_duplicates("month", keep="last")
               [["month", val_col]]
    )
    if anchors.empty:
        return pd.Series(0.0, index=idx)

    # Filter anchors to the schedule's index span
    lo, hi = idx.min(), idx.max()
    in_range = anchors[(anchors["month"] >= lo) & (anchors["month"] <= hi)].copy()

    # Seed: pick the value we’ll use at the very first index month
    if not in_range.empty:
        seed_val = float(in_range.iloc[0][val_col])
    else:
        # If no anchor falls inside the range, take the first future anchor after 'lo' if any
        future = anchors[anchors["month"] >= lo]
        if not future.empty:
            seed_val = float(future.iloc[0][val_col])
        else:
            # Or the last past anchor before 'lo'
            past = anchors[anchors["month"] <= lo]
            seed_val = float(past.iloc[-1][val_col]) if not past.empty else 0.0

    # Base series with NaNs, then put in-range anchors, and seed first month
    base = pd.Series(index=idx, dtype=float)
    if not in_range.empty:
        # Only assign months that definitely exist in the index
        valid_months = in_range["month"][in_range["month"].isin(idx)]
        base.loc[valid_months] = in_range.set_index("month").loc[valid_months, val_col].values
    base.iloc[0] = seed_val
    base = base.ffill()

    # Group id that increments at each anchor month (within range / seed included)
    is_anchor = base.index.isin(in_range["month"].values) if not in_range.empty else (base.index == base.index[0])
    grp = pd.Series(is_anchor, index=base.index).astype(int).cumsum()

    # Monthly cumulative growth from global start, normalized per group
    rm = monthly_rate_series.reindex(idx).fillna(0.0)
    cum = (1.0 + rm).cumprod()
    group_start_cum = cum.groupby(grp).transform("first").replace(0.0, 1.0)
    rel_factor = cum / group_start_cum

    out = base * rel_factor
    out.name = val_col
    # Final safety: fill any residual NaNs
    return out.ffill()

def _income_series_with_january_raises(income_anchors_df, growth_pct_df, start_date, months):
    """
    Build monthly income where raises are applied only in January.
    - Income anchors: date + income (overrides immediately when it occurs)
    - Growth schedule (annual %): applied on January of each year (except we DON'T
      auto-apply at the very first schedule month; the first month uses the seeded anchor).
    - Between anchors and between Januaries, income is FLAT.
    """
    idx = pd.date_range(start=start_date, periods=int(months), freq="MS")  # month-starts

    # --- normalize anchors (date + income)
    inc_df = income_anchors_df.copy() if isinstance(income_anchors_df, pd.DataFrame) else pd.DataFrame(income_anchors_df or {})
    if inc_df.empty or "date" not in inc_df.columns or "income" not in inc_df.columns:
        # no anchors -> zero income
        return pd.Series(0.0, index=idx, name="income")

    inc_df["month"] = pd.to_datetime(inc_df["date"], errors="coerce").dt.to_period("M").dt.to_timestamp(how="start")
    inc_df["income"] = pd.to_numeric(inc_df["income"], errors="coerce")
    inc_df = (inc_df.dropna(subset=["month", "income"])
                    .sort_values("month")
                    .drop_duplicates("month", keep="last"))[["month", "income"]]

    # --- seed value for the first schedule month
    lo, hi = idx.min(), idx.max()
    in_range = inc_df[(inc_df["month"] >= lo) & (inc_df["month"] <= hi)]
    if not in_range.empty:
        seed_val = float(in_range.iloc[0]["income"])
    else:
        future = inc_df[inc_df["month"] >= lo]
        if not future.empty:
            seed_val = float(future.iloc[0]["income"])
        else:
            past = inc_df[inc_df["month"] <= lo]
            seed_val = float(past.iloc[-1]["income"]) if not past.empty else 0.0

    # --- growth % per January (as-of that January)
    jan_idx = pd.date_range(start=pd.Timestamp(year=lo.year, month=1, day=1),
                            end=pd.Timestamp(year=hi.year, month=1, day=1),
                            freq="12MS")  # Jan 1 of each year
    # Use your existing helper to forward-fill annual growth; then pick values at Jan
    ann_growth_full = _series_from_anchors(start_date, months, growth_pct_df, "growth_pct", 0.0)
    # Align to jan_idx (use nearest past value)
    g_jan = pd.merge_asof(
        pd.DataFrame({"month": jan_idx}),
        ann_growth_full.reset_index().rename(columns={"index": "month", "growth_pct": "g"}),
        on="month",
        direction="backward",
        allow_exact_matches=True,
    )
    if g_jan["g"].isna().all():
        g_jan["g"] = 0.0
    g_jan["g"] = g_jan["g"].fillna(method="ffill").fillna(0.0)
    g_by_year = {ts.year: float(g) for ts, g in zip(g_jan["month"], g_jan["g"])}

    # --- build a quick map of anchor overrides that fall inside the index
    anchor_map = {pd.Timestamp(m): float(v) for m, v in zip(in_range["month"], in_range["income"])} if not in_range.empty else {}

    # --- iterate months: apply Jan raises, apply anchor overrides
    out = []
    current_income = seed_val
    first_month = idx[0]
    for m in idx:
        # If this is a January and not the very first month of the schedule -> apply raise
        if m.month == 1 and m != first_month:
            g = g_by_year.get(m.year, 0.0)
            current_income = current_income * (1.0 + g / 100.0)

        # If an anchor exists for this month, override
        if m in anchor_map:
            current_income = anchor_map[m]

        out.append(current_income)

    s = pd.Series(out, index=idx, name="income")
    return s

def _detect_decimal(sample_bytes: bytes) -> str:
    """Guess decimal separator from a small CSV/Excel sample (default dot)."""
    try:
        sample = sample_bytes[:2048].decode("utf-8", errors="ignore")
    except Exception:
        return "."
    import re
    # Look for number-like tokens
    nums = re.findall(r"[-+]?\d{1,3}(?:[.,]\d{3})*(?:[.,]\d+)?", sample)
    # If we see things like '123,45' without a dot, assume comma decimal
    for n in nums:
        if "," in n and not "." in n:
            return ","
    return "."

def _read_table_flex(file, rename_map, required, date_cols=(), numeric_cols=()):
    """
    Read CSV or Excel, normalize headers, parse date/numeric with EU/US separators.
    """
    import io, os
    raw = file.read()
    name = getattr(file, "name", "file").lower()
    ext = os.path.splitext(name)[1]

    df = None
    err = None

    if ext == ".csv":
        decimal = _detect_decimal(raw)
        buf = io.BytesIO(raw)
        try:
            df = pd.read_csv(buf, sep=None, engine="python", decimal=decimal)
        except Exception as e:
            return None, f"Erro a ler CSV: {e}"
    elif ext in [".xls", ".xlsx"]:
        buf = io.BytesIO(raw)
        try:
            df = pd.read_excel(buf)
        except Exception as e:
            return None, f"Erro a ler Excel: {e}"
    else:
        return None, "Formato não suportado (use CSV ou Excel)."

    # --- Normalize headers ---
    df.columns = [str(c).strip().lower() for c in df.columns]

    # Rename columns if necessary
    applied = {}
    for src, target in rename_map.items():
        src_low = src.strip().lower()
        if src_low in df.columns and target not in df.columns:
            applied[src_low] = target
    df = df.rename(columns=applied)

    keep = list(set(rename_map.values()) & set(df.columns))
    df = df[keep].copy()

    # Require only 'date' if asked
    if "date" in required and "date" not in df.columns:
        return None, "Coluna 'date' em falta no ficheiro."
    
    # If required has >1 col (e.g. ["date","inflation_pct"]) but the second is missing,
    # then just take the first non-date column as the numeric/label
    for col in required:
        if col != "date" and col not in df.columns:
            # pick the first column that's not 'date'
            candidates = [c for c in df.columns if c != "date"]
            if candidates:
                df = df.rename(columns={candidates[0]: col})
            else:
                return None, f"Coluna numérica em falta no ficheiro (esperava '{col}')"


    # Dates
    for c in date_cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce").dt.date

    # Numerics
    for c in numeric_cols:
        if c in df.columns:
            s = df[c].astype(str).str.replace(" ", "", regex=False)
            s = s.str.replace(",", ".", regex=False)  # force dot decimal
            df[c] = pd.to_numeric(s, errors="coerce")

    return df.dropna(how="all"), None


def _apply_import(df, init_key: str, current_key: str, tbl_key_key: str, tbl_prefix: str):
    """
    Put imported df into init & current state and rotate the table widget key
    so the editor shows the new data immediately.
    """
    st.session_state[init_key] = df.copy()
    st.session_state[current_key] = df.copy()
    stamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
    st.session_state[tbl_key_key] = f"{tbl_prefix}_{stamp}"




# ========== Initialize state (primitives + initial editor data + current caches + keys)
st.session_state.setdefault("principal", 250_000.0)
st.session_state.setdefault("years", 30)
st.session_state.setdefault("start", pd.Timestamp.today().date())
st.session_state.setdefault("rate_type", "Fixa")
st.session_state.setdefault("fixed_rate_pct", 4.0)
st.session_state.setdefault("spread_pct", DEF_SPREAD)
st.session_state.setdefault("reset_months", 12)

# CSV/XLSX uploader keys (one per import strip)
st.session_state.setdefault("eur_csv_upl_key",        "eur_csv_0")
st.session_state.setdefault("upfront_csv_upl_key",    "upfront_csv_0")
st.session_state.setdefault("monthly_csv_upl_key",    "monthly_csv_0")
st.session_state.setdefault("amort_csv_upl_key",      "amort_csv_0")
st.session_state.setdefault("income_csv_upl_key",     "income_csv_0")
st.session_state.setdefault("income_g_csv_upl_key",   "income_g_csv_0")
st.session_state.setdefault("expenses_csv_upl_key",   "expenses_csv_0")
st.session_state.setdefault("inflation_csv_upl_key",  "inflation_csv_0")


st.session_state.setdefault("eur_init_df", pd.DataFrame({"date": [st.session_state["start"]], "euribor": [DEF_EUR]}))
st.session_state.setdefault("upfront_init_df", pd.DataFrame({"name": ["Comissão"], "value": [0.0]}))
st.session_state.setdefault("monthly_init_df", pd.DataFrame({"name": ["Seguro"], "value": [0.0]}))

# Amortizações defaults
st.session_state.setdefault("amort_init_df", pd.DataFrame({"date": [st.session_state["start"]], "amount": [0.0]}))
st.session_state.setdefault("amort_current_df", st.session_state["amort_init_df"].copy())
st.session_state.setdefault("amort_tbl_key", "amort_tbl")
st.session_state.setdefault("amort_strategy", "Reduzir prazo")
st.session_state.setdefault("amort_fee_pct", 0.5)

# current_df caches (what we actually use on calculate)
st.session_state.setdefault("eur_current_df", st.session_state["eur_init_df"].copy())
st.session_state.setdefault("upfront_current_df", st.session_state["upfront_init_df"].copy())
st.session_state.setdefault("monthly_current_df", st.session_state["monthly_init_df"].copy())

st.session_state.setdefault("eur_tbl_key", "eur_tbl")
st.session_state.setdefault("upf_tbl_key", "upf_tbl")
st.session_state.setdefault("mon_tbl_key", "mon_tbl")

# Last computed artifacts
st.session_state.setdefault("last_sched", None)
st.session_state.setdefault("last_sched_base", None)
st.session_state.setdefault("last_banner", "Edite e clique **Calcular agora** para gerar o plano.")

# Stress analysis defaults
st.session_state.setdefault("income_init_df",   pd.DataFrame({"date":[st.session_state["start"]], "income":[2000.0]}))
st.session_state.setdefault("income_growth_init_df", pd.DataFrame({"date":[st.session_state["start"]], "growth_pct":[0.0]}))
st.session_state.setdefault("expenses_init_df", pd.DataFrame({"date":[st.session_state["start"]], "expenses":[1200.0]}))
st.session_state.setdefault("inflation_init_df", pd.DataFrame({"date":[st.session_state["start"]], "inflation_pct":[2.0]}))

st.session_state.setdefault("income_current_df",   st.session_state["income_init_df"].copy())
st.session_state.setdefault("income_growth_current_df", st.session_state["income_growth_init_df"].copy())
st.session_state.setdefault("expenses_current_df", st.session_state["expenses_init_df"].copy())
st.session_state.setdefault("inflation_current_df", st.session_state["inflation_init_df"].copy())

st.session_state.setdefault("income_tbl_key", "income_tbl")
st.session_state.setdefault("income_growth_tbl_key", "income_growth_tbl")
st.session_state.setdefault("expenses_tbl_key", "expenses_tbl")
st.session_state.setdefault("inflation_tbl_key", "inflation_tbl")

# Stress options
st.session_state.setdefault("effort_threshold_pct", 35.0)



# ========== Sidebar
with st.sidebar:
    st.header("Dados gerais")
    st.number_input("Capital em dívida", key="principal", step=1_000.0, min_value=0.0)
    st.number_input("Duração (anos)", key="years", step=1, min_value=1, max_value=50)
    st.date_input("Data de início", key="start")

    st.radio("Tipo de taxa", ["Fixa", "Variável (reset)"], key="rate_type")

    if st.session_state["rate_type"] == "Fixa":
        st.number_input("Taxa nominal anual (%)", key="fixed_rate_pct", step=0.05)
    else:
        st.number_input("Spread (anual, %)", key="spread_pct", step=0.05)
        st.selectbox("Periodicidade da revisão (meses)", [3,6, 12], key="reset_months")
    
        # --- Import strip (optional) ---
        st.caption("Calendário de Euribor (edite; só conta quando carregar em **Calcular agora**)")
        eur_csv = st.file_uploader(
            "Importar Euribor (CSV/XLSX: colunas: date, euribor)",
            type=["csv", "xlsx"],
            key=st.session_state["eur_csv_upl_key"],
            accept_multiple_files=False,
        )
        
        if eur_csv is not None:
            # IMPORTANT: don't eur_csv.read() here; let the helper read once.
            df_imp, err = _read_table_flex(
                eur_csv,
                rename_map={"date": "date", "data": "date", "euribor": "euribor", "euribor_pct": "euribor"},
                required=["date", "euribor"],
                date_cols=["date"],
                numeric_cols=["euribor"],
            )
            if err:
                st.error(err)
            else:
                # Apply to your state (init + current), rotate editor key so it remounts with imported data
                st.session_state["eur_init_df"] = df_imp.copy()
                st.session_state["eur_current_df"] = df_imp.copy()
                from datetime import datetime as _dt
                st.session_state["eur_tbl_key"] = f"eur_tbl_{_dt.now().strftime('%Y%m%d%H%M%S%f')}"
                st.success(f"Importado {eur_csv.name} ({len(df_imp)} linhas).")
                # No st.rerun(): avoids refresh loop while the file stays selected

    
        # --- Editor (unchanged pattern) ---
        eur_init = st.session_state["eur_init_df"].copy()
        eur_init["date"] = pd.to_datetime(eur_init["date"], errors="coerce").dt.date
        eur_val = st.data_editor(
            eur_init,
            num_rows="dynamic",
            width="stretch",  # Streamlit deprecation: replaces use_container_width=True
            key=st.session_state["eur_tbl_key"],
            column_config={
                "date": st.column_config.DateColumn("date"),
                "euribor": st.column_config.NumberColumn("euribor", step=0.05, format="%.2f"),
            },
        )
        # Cache the edited value robustly (no widget assignment!)
        st.session_state["eur_current_df"] = (
            eur_val.copy() if isinstance(eur_val, pd.DataFrame)
            else st.session_state["eur_current_df"]
        )    
    st.info("Carregar em calcular agora quando os parametros do calculo estiverem definidos.")

    st.divider()
    st.header("Custos Iniciais")
    st.caption("Importar custos iniciais (CSV/XLSX: colunas: name, value)")
    up_csv = st.file_uploader(
        "Importar custos iniciais",
        type=["csv", "xlsx"],
        key=st.session_state["upfront_csv_upl_key"],
        accept_multiple_files=False,
    )
    if up_csv is not None:
        df_imp, err = _read_table_flex(
            up_csv,
            rename_map={
                "name":"name", "descricao":"name", "descrição":"name", "label":"name",
                "value":"value", "valor":"value", "montante":"value"
            },
            required=["name","value"],
            date_cols=[],
            numeric_cols=["value"],
        )
        if err:
            st.error(err)
        else:
            st.session_state["upfront_init_df"]    = df_imp.copy()
            st.session_state["upfront_current_df"] = df_imp.copy()
            from datetime import datetime as _dt
            st.session_state["upf_tbl_key"] = f"upf_tbl_{_dt.now().strftime('%Y%m%d%H%M%S%f')}"
            st.success(f"Importado {up_csv.name} ({len(df_imp)} linhas).")

    up_val = st.data_editor(
        st.session_state["upfront_init_df"].copy(),
        num_rows="dynamic",
        use_container_width=True,
        key=st.session_state["upf_tbl_key"],
        column_config={
            "name": st.column_config.TextColumn("name"),
            "value": st.column_config.NumberColumn("value", step=1.0, format="%.2f"),
        },
    )
    st.session_state["upfront_current_df"] = _coerce_df(up_val, st.session_state["upfront_current_df"])
    st.info("Carregar em calcular agora quando os parametros do calculo estiverem definidos.")

    st.header("Outros Custos Mensais")
    st.caption("Para além da prestação (ex: Seguros, Comissões de conta,etc)")
    mo_csv = st.file_uploader(
        "Importar custos mensais (CSV/XLSX: colunas: name, value)",
        type=["csv", "xlsx"],
        key=st.session_state["monthly_csv_upl_key"],
        accept_multiple_files=False,
    )
    if mo_csv is not None:
        df_imp, err = _read_table_flex(
            mo_csv,
            rename_map={
                "name":"name", "descricao":"name", "descrição":"name", "label":"name",
                "value":"value", "valor":"value", "montante":"value"
            },
            required=["name","value"],
            date_cols=[],
            numeric_cols=["value"],
        )
        if err:
            st.error(err)
        else:
            st.session_state["monthly_init_df"]    = df_imp.copy()
            st.session_state["monthly_current_df"] = df_imp.copy()
            from datetime import datetime as _dt
            st.session_state["mon_tbl_key"] = f"mon_tbl_{_dt.now().strftime('%Y%m%d%H%M%S%f')}"
            st.success(f"Importado {mo_csv.name} ({len[df_imp]} linhas).")

    mo_val = st.data_editor(
        st.session_state["monthly_init_df"].copy(),
        num_rows="dynamic",
        use_container_width=True,
        key=st.session_state["mon_tbl_key"],
        column_config={
            "name": st.column_config.TextColumn("name"),
            "value": st.column_config.NumberColumn("value", step=1.0, format="%.2f"),
        },
    )
    st.session_state["monthly_current_df"] = _coerce_df(mo_val, st.session_state["monthly_current_df"])
    st.info("Carregar em calcular agora quando os parametros do calculo estiverem definidos.")

    st.divider()
    
    st.header("Rendimento líquido mensal")
    st.info("Evolução de rendimento calculada através de um rendimento base que varia consoante a variação anual calendarizada.")
    st.caption("Valor Base")
    inc_val = st.data_editor(
        st.session_state["income_init_df"].copy(),
        num_rows="dynamic", use_container_width=True,
        key=st.session_state["income_tbl_key"],
        column_config={
            "date": st.column_config.DateColumn("date"),
            "income": st.column_config.NumberColumn("income", step=50.0, format="%.2f"),
        },
    )
    st.session_state["income_current_df"] = _coerce_df(inc_val, st.session_state["income_current_df"])
    
    st.caption("Variação anual do rendimento (%)")
    incg_val = st.data_editor(
        st.session_state["income_growth_init_df"].copy(),
        num_rows="dynamic", use_container_width=True,
        key=st.session_state["income_growth_tbl_key"],
        column_config={
            "date": st.column_config.DateColumn("date"),
            "growth_pct": st.column_config.NumberColumn("growth_pct", step=0.5, format="%.2f"),
        },
    )
    st.session_state["income_growth_current_df"] = _coerce_df(incg_val, st.session_state["income_growth_current_df"])

    st.header("Despesas Mensais")
    st.info("Evolução das despesas calculada através de um valor base que varia consoante a inflação calendarizada.")
    
    st.caption("Valor Base")
    exp_val = st.data_editor(
        st.session_state["expenses_init_df"].copy(),
        num_rows="dynamic", use_container_width=True,
        key=st.session_state["expenses_tbl_key"],
        column_config={
            "date": st.column_config.DateColumn("date"),
            "expenses": st.column_config.NumberColumn("expenses", step=50.0, format="%.2f"),
        },
    )
    st.session_state["expenses_current_df"] = _coerce_df(exp_val, st.session_state["expenses_current_df"])

    
    st.caption("Inflação (%)")
    st.caption("Importar inflação (CSV/XLSX: colunas: date, inflation_pct)")
    infl_csv = st.file_uploader(
        "Importar inflação",
        type=["csv", "xlsx"],
        key=st.session_state["inflation_csv_upl_key"],
        accept_multiple_files=False,
    )
    if infl_csv is not None:
        df_imp, err = _read_table_flex(
            infl_csv,
            rename_map={"date":"date","data":"date","inflation_pct":"inflation_pct","inflacao":"inflation_pct","inflação":"inflation_pct","taxa":"inflation_pct"},
            required=["date","inflation_pct"],
            date_cols=["date"],
            numeric_cols=["inflation_pct"],
        )
        if err:
            st.error(err)
        else:
            st.session_state["inflation_init_df"]    = df_imp.copy()
            st.session_state["inflation_current_df"] = df_imp.copy()
            from datetime import datetime as _dt
            st.session_state["inflation_tbl_key"] = f"inflation_tbl_{_dt.now().strftime('%Y%m%d%H%M%S%f')}"
            st.success(f"Importado {infl_csv.name} ({len(df_imp)} linhas).")

    infl_val = st.data_editor(
        st.session_state["inflation_init_df"].copy(),
        num_rows="dynamic", use_container_width=True,
        key=st.session_state["inflation_tbl_key"],
        column_config={
            "date": st.column_config.DateColumn("date"),
            "inflation_pct": st.column_config.NumberColumn("inflation_pct", step=0.5, format="%.2f"),
        },
    )
    st.session_state["inflation_current_df"] = _coerce_df(infl_val, st.session_state["inflation_current_df"])



    st.divider()
    st.header("Amortizações antecipadas")
    st.caption("Importar amortizações (CSV/XLSX: colunas: date, amount)")
    am_csv = st.file_uploader(
        "Importar amortizações",
        type=["csv", "xlsx"],
        key=st.session_state["amort_csv_upl_key"],
        accept_multiple_files=False,
    )
    if am_csv is not None:
        df_imp, err = _read_table_flex(
            am_csv,
            rename_map={"date":"date","data":"date","amount":"amount","valor":"amount","montante":"amount"},
            required=["date","amount"],
            date_cols=["date"],
            numeric_cols=["amount"],
        )
        if err:
            st.error(err)
        else:
            st.session_state["amort_init_df"]    = df_imp.copy()
            st.session_state["amort_current_df"] = df_imp.copy()
            from datetime import datetime as _dt
            st.session_state["amort_tbl_key"] = f"amort_tbl_{_dt.now().strftime('%Y%m%d%H%M%S%f')}"
            st.success(f"Importado {am_csv.name} ({len(df_imp)} linhas).")

    st.caption("Introduza (data, montante) das amortizações parciais (liquidações antecipadas).")
    am_val = st.data_editor(
        st.session_state["amort_init_df"].copy(),
        num_rows="dynamic",
        use_container_width=True,
        key=st.session_state["amort_tbl_key"],
        column_config={
            "date": st.column_config.DateColumn("date"),
            "amount": st.column_config.NumberColumn("amount", step=100.0, format="%.2f"),
        },
    )

    st.session_state["amort_current_df"] = _coerce_df(am_val, st.session_state["amort_current_df"])
    st.session_state["amort_strategy"] = st.selectbox(
        "Estratégia após amortização",
        ["Reduzir prazo", "Reduzir prestação"],
        index=(0 if st.session_state["amort_strategy"] == "Reduzir prazo" else 1)
    )
    st.session_state["amort_fee_pct"] = st.number_input(
        "Comissão (%) sobre amortização", value=float(st.session_state["amort_fee_pct"]), step=0.1, min_value=0.0
    )


    st.divider()
    st.header("📍 Data Atual")
    use_today = st.toggle("Usar data de hoje", value=True)
    ref_date = pd.Timestamp.today().date() if use_today else st.date_input("Data de referência", value=pd.Timestamp.today().date(), key="ref_date_manual")

    st.header("📁 Cenários")
    base_in = st.text_input("Diretório base", value=st.session_state["base_dir"])
    if st.button("Usar esta pasta"):
        st.session_state["base_dir"] = base_in
        st.session_state["scen_dir"] = os.path.join(base_in, "scenarios")
        os.makedirs(st.session_state["scen_dir"], exist_ok=True)
        st.success(f"Pasta ativa: {st.session_state['scen_dir']}")
        st.rerun()

    try:
        scen_files = [f for f in os.listdir(st.session_state["scen_dir"]) if f.endswith(".json")]
    except Exception as e:
        scen_files = []
        st.error(f"Não foi possível listar a pasta: {e}")

    scen_name = st.text_input("Nome do cenário", value="meu_cenario", key="scen_name")
    if st.button("💾 Guardar cenário"):
        # Use the cached current tables at save time
        eur_df_for_save = st.session_state["eur_current_df"].copy()
        up_for_save = st.session_state["upfront_current_df"].copy()
        mo_for_save = st.session_state["monthly_current_df"].copy()
        am_for_save = st.session_state["amort_current_df"].copy()

        payload = {
            "inputs": {
                "rate_type": st.session_state["rate_type"],
                "principal": st.session_state["principal"],
                "years": st.session_state["years"],
                "start": str(st.session_state["start"]),
                "fixed_rate_pct": (st.session_state["fixed_rate_pct"] if st.session_state["rate_type"] == "Fixa" else None),
                "spread_pct": (st.session_state["spread_pct"] if st.session_state["rate_type"] != "Fixa" else None),
                "reset_months": (st.session_state["reset_months"] if st.session_state["rate_type"] != "Fixa" else None),
            },
            "euribor_forecast": (
                eur_df_for_save.assign(date=pd.to_datetime(eur_df_for_save["date"]).dt.strftime("%Y-%m-%d")).to_dict(orient="records")
                if st.session_state["rate_type"] != "Fixa" else []
            ),
            "fees_upfront": up_for_save.to_dict(orient="records"),
            "fees_monthly": mo_for_save.to_dict(orient="records"),
            
            "stress": {
                "income_anchors": st.session_state["income_current_df"].assign(date=pd.to_datetime(st.session_state["income_current_df"]["date"]).dt.strftime("%Y-%m-%d")
                    ).to_dict(orient="records"),
                    "income_growth": st.session_state["income_growth_current_df"].assign(
                        date=pd.to_datetime(st.session_state["income_growth_current_df"]["date"]).dt.strftime("%Y-%m-%d")
                    ).to_dict(orient="records"),
                    "expenses_anchors": st.session_state["expenses_current_df"].assign(
                        date=pd.to_datetime(st.session_state["expenses_current_df"]["date"]).dt.strftime("%Y-%m-%d")
                    ).to_dict(orient="records"),
                    "inflation": st.session_state["inflation_current_df"].assign(
                        date=pd.to_datetime(st.session_state["inflation_current_df"]["date"]).dt.strftime("%Y-%m-%d")
                    ).to_dict(orient="records"),
                    "effort_threshold_pct": float(st.session_state["effort_threshold_pct"]),
                },
            "amortizations": am_for_save.assign(
                date=pd.to_datetime(am_for_save["date"], errors="coerce").dt.strftime("%Y-%m-%d")
            ).replace({pd.NA: None}).to_dict(orient="records"),
            "amort_strategy": st.session_state["amort_strategy"],
            "amort_fee_pct": float(st.session_state["amort_fee_pct"]),
        }
        path = os.path.join(st.session_state["scen_dir"], f"{st.session_state['scen_name']}.json")
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            st.success(f"Guardado em {path}")
        except Exception as e:
            st.error(f"Falha ao guardar: {e}")

    if scen_files:
        pick = st.selectbox("Selecionar cenário da pasta", options=["—"] + sorted(scen_files), index=0, key="pick_scen")
        if st.button("📥 Carregar cenário selecionado") and pick != "—":
            try:
                with open(os.path.join(st.session_state["scen_dir"], pick), "r", encoding="utf-8") as f:
                    obj = json.load(f)
                apply_scenario_dict(obj)  # staged load + rerun
            except Exception as e:
                st.error(f"Erro ao carregar {pick}: {e}")

    st.divider()
    calc_btn = st.button("✅ Calcular agora", use_container_width=True)

# ========== Calculations (only when requested)
should_calc = calc_btn or st.session_state.get("force_recalc", False) or (st.session_state["last_sched"] is None)
if should_calc:
    st.session_state["force_recalc"] = False

    # Use the cached current editor values
    eur_df_input = st.session_state["eur_current_df"].copy()
    if not eur_df_input.empty and "date" in eur_df_input.columns:
        eur_df_input["date"] = pd.to_datetime(eur_df_input["date"], errors="coerce").dt.date

    upfront_df_input = st.session_state["upfront_current_df"].copy()
    monthly_df_input = st.session_state["monthly_current_df"].copy()
    amort_df_input = st.session_state["amort_current_df"].copy()

    months_total = int(st.session_state["years"] * 12)
    upfront_total = float(upfront_df_input.get("value", pd.Series(dtype=float)).fillna(0).sum())
    monthly_extra = float(monthly_df_input.get("value", pd.Series(dtype=float)).fillna(0).sum())

    # Build amortization map
    fee_pct = float(st.session_state["amort_fee_pct"])
    amap = _amort_map_from_df(st.session_state["start"], months_total, amort_df_input, fee_pct)
    keep_payment = (st.session_state["amort_strategy"] == "Reduzir prazo")

    if st.session_state["rate_type"] == "Fixa":
        # Baseline (no amort)
        sched_base = schedule_fixed(
            st.session_state["start"],
            st.session_state["principal"],
            st.session_state["fixed_rate_pct"],
            st.session_state["years"],
            monthly_extra,
            upfront_total,
            amort_map={},
            amort_keep_payment=True
        )
        # With amortizações
        sched = schedule_fixed(
            st.session_state["start"],
            st.session_state["principal"],
            st.session_state["fixed_rate_pct"],
            st.session_state["years"],
            monthly_extra,
            upfront_total,
            amort_map=amap,
            amort_keep_payment=keep_payment
        )
        banner = f"Modo: **Fixa** | Taxa: {st.session_state['fixed_rate_pct']:.2f}%"
    else:
        if eur_df_input.empty:
            eur_df_input = pd.DataFrame({"date": [st.session_state["start"]], "euribor": [DEF_EUR]})
        eur_series = build_euribor_path(st.session_state["start"], months_total, eur_df_input, DEF_EUR)

        sched_base = schedule_variable(
            st.session_state["start"],
            st.session_state["principal"],
            st.session_state["spread_pct"],
            st.session_state["years"],
            int(st.session_state["reset_months"]),
            eur_series,
            monthly_extra,
            upfront_total,
            amort_map={},
            amort_keep_payment=True
        )
        sched = schedule_variable(
            st.session_state["start"],
            st.session_state["principal"],
            st.session_state["spread_pct"],
            st.session_state["years"],
            int(st.session_state["reset_months"]),
            eur_series,
            monthly_extra,
            upfront_total,
            amort_map=amap,
            amort_keep_payment=keep_payment
        )
        eff0 = float(eur_series.iloc[0]) + float(st.session_state["spread_pct"])
        banner = (
            f"Modo: **Variável** | Euribor(1º mês): {eur_series.iloc[0]:.2f}% + "
            f"Spread: {st.session_state['spread_pct']:.2f}% = **{eff0:.2f}%**"
        )

    # Enrich + persist
    sched, idx_now = _enrich_with_now(sched, ref_date)
    st.session_state["last_sched"] = sched
    st.session_state["last_sched_base"] = sched_base
    st.session_state["last_banner"] = banner
    
    # ==== Stress series (Income / Expenses / Effort) ====
    months_total = int(st.session_state["years"] * 12)
    income_anchor = st.session_state["income_current_df"].copy()
    expenses_anchor = st.session_state["expenses_current_df"].copy()
    income_growth_pct = st.session_state["income_growth_current_df"].copy()
    inflation_pct = st.session_state["inflation_current_df"].copy()
    
    # Monthly inflation series for expenses (unchanged)
    rm_infl = _annual_pct_to_monthly_series(st.session_state["start"], months_total, inflation_pct, "inflation_pct", 0.0)
    expenses_series = _apply_growth_from_anchor(expenses_anchor, rm_infl)
    

    income_series = _income_series_with_january_raises(
        income_anchor,
        income_growth_pct,
        st.session_state["start"],
        months_total,
    )
 
    # Align to schedule months
    x_idx = pd.to_datetime(sched["date"]).dt.to_period("M").dt.to_timestamp(how="start")
    income_series   = income_series.reindex(x_idx, method="ffill").fillna(0.0)
    expenses_series = expenses_series.reindex(x_idx, method="ffill").fillna(0.0)
    
    sched["income"] = income_series.values
    sched["expenses"] = expenses_series.values
    sched["effort_pct"] = (sched["payment"] / sched["income"]).replace([pd.NA, pd.NaT], 0.0).fillna(0.0)
    sched["available_income"] = (sched["income"] - sched["payment"] - sched["expenses"]).fillna(0.0)


# ========== Display last results
st.info(st.session_state["last_banner"])

sched = st.session_state["last_sched"]
sched_base = st.session_state["last_sched_base"]

if sched is None:
    st.warning("Ainda não foi calculado. Edite os inputs e clique **Calcular agora**.")
else:
    # Slices
    # Ensure idx_now exists (from enrich step). If not (shouldn't happen), set to 0 safely.
    idx_now = next((i for i, v in enumerate(sched["now_flag"]) if v == "← agora"), 0)
    paid_slice = sched.iloc[:idx_now]           # completed periods
    future_slice = sched.iloc[idx_now:]         # from 'now' onwards
    bal_before = float(sched.iloc[idx_now-1]["balance"]) if idx_now > 0 else float(st.session_state["principal"])

    # Totals
    already_paid_loan = float(paid_slice["payment"].sum())
    already_paid_interest = float(paid_slice["interest"].sum())
    upfront_total_disp = float(
        st.session_state["upfront_current_df"].get("value", pd.Series(dtype=float)).fillna(0).sum()
    )
    already_paid_incl_upfront = already_paid_loan + upfront_total_disp
    remaining_payments = float(future_slice["payment"].sum())
    remaining_interest = float(future_slice["interest"].sum())

    # Context
    months_elapsed = int(paid_slice.shape[0])
    months_total   = int(sched.shape[0])
    months_left    = max(months_total - months_elapsed, 0)

    # Current rate (row idx_now if exists, else last row)
    row_rate = sched.iloc[min(idx_now, len(sched)-1)]
    current_rate = float(row_rate["rate_annual_pct"])

    # Next reset date
    next_reset_date = None
    if "reset" in sched.columns:
        rs = future_slice[future_slice["reset"]]
        if not rs.empty:
            next_reset_date = pd.to_datetime(rs.iloc[0]["date"]).date()

    # KPIs layout
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Prestação inicial", eur(float(sched.iloc[0]["payment"])))
    col2.metric("Juros totais", eur(float(sched["interest"].sum())))
    col3.metric("Custos iniciais", eur(upfront_total_disp))
    col4.metric("Total a pagar", eur(float(sched["payment"].sum() + upfront_total_disp)))
    total_to_pay = float(sched["payment"].sum() + upfront_total_disp)

    st.divider()
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Pago até agora", eur(already_paid_incl_upfront))
    c2.metric("Total Pago em Juros", eur(already_paid_interest))
    c3.metric("Total Pago (%)", per(already_paid_incl_upfront / total_to_pay if total_to_pay else 0.0))

    c5, c6, c7 = st.columns(3)
    c5.metric("Prestação Atual", eur(float(future_slice["payment"].iloc[0])) if not future_slice.empty else "—")
    c6.metric("Taxa atual", f"{current_rate:.2f}%")
    c7.metric("Meses decorridos / totais", f"{months_elapsed} / {months_total}" + (f" (faltam {months_left})" if months_left>0 else ""))

    if next_reset_date:
        st.caption(f"Próxima revisão: **{next_reset_date.strftime('%d/%m/%Y')}**.")
    
    st.divider()
    st.subheader("📊 Stress financeiro")
    
    thr = float(st.session_state["effort_threshold_pct"]) / 100.0
    
    # Whole horizon
    eff_avg = float(sched["effort_pct"].mean())
    eff_max = float(sched["effort_pct"].max())
    months_over = int((sched["effort_pct"] > thr).sum())
    min_buf = float(sched["available_income"].min())
    min_buf_date = pd.to_datetime(sched.loc[sched["available_income"].idxmin(), "date"]).date()
    
    # From 'Posição atual'
    idx_now = next((i for i, v in enumerate(sched["now_flag"]) if v == "← agora"), 0)
    eff_avg_now = float(sched["effort_pct"].iloc[idx_now:].mean()) if idx_now < len(sched) else 0.0
    eff_max_now = float(sched["effort_pct"].iloc[idx_now:].max()) if idx_now < len(sched) else 0.0
    months_over_now = int((sched["effort_pct"].iloc[idx_now:] > thr).sum()) if idx_now < len(sched) else 0
    
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Esforço médio", per(eff_avg))
    a2.metric("Esforço máximo", per(eff_max))
    a3.metric("Meses acima do limite", f"{months_over} (limite {int(thr*100)}%)")
    a4.metric("Buffer mínimo", eur(min_buf))
    
    b1, b2, b3 = st.columns(3)
    b1.metric("Esforço médio (desde agora)", per(eff_avg_now))
    b2.metric("Esforço máximo (desde agora)", per(eff_max_now))
    b3.metric("Meses acima (desde agora)", f"{months_over_now}")
    
    st.caption(f"Menor buffer no mês de **{min_buf_date.strftime('%d/%m/%Y')}**.")


    # ----- Efeito das Amortizações (delta vs baseline) -----
    st.divider()
    st.subheader("🎯 Efeito das Amortizações")
    if sched_base is None or sched_base.empty:
        st.info("Sem baseline calculado.")
    else:
        base_interest = float(sched_base["interest"].sum())
        with_am_interest = float(sched["interest"].sum())
        interest_saved = base_interest - with_am_interest
        fees_paid = float(sched.get("amort_fee", pd.Series(dtype=float)).sum())
        net_saving = interest_saved - fees_paid

        base_months = int(sched_base.shape[0])
        with_months = int(sched.shape[0])
        months_saved = max(base_months - with_months, 0)

        base_end = pd.to_datetime(sched_base["date"]).max().date() if not sched_base.empty else None
        with_end = pd.to_datetime(sched["date"]).max().date() if not sched.empty else None

        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Juros poupados", eur(interest_saved))
        d2.metric("Comissões (amort.)", eur(fees_paid))
        d3.metric("Poupança líquida", eur(net_saving))
        if st.session_state["amort_strategy"] == "Reduzir prazo":
            d4.metric("Meses poupados", f"{months_saved}")
        else:
            # show new installment right after first amort month (if exists)
            first_am_idxs = sched.index[sched.get("amortization", pd.Series(dtype=float)) > 0]
            if len(first_am_idxs) > 0 and first_am_idxs.min() + 1 < len(sched):
                new_pay = float(sched.loc[first_am_idxs.min() + 1, "payment"])
                d4.metric("Nova prestação (após amort.)", eur(new_pay))
            else:
                d4.metric("Nova prestação (após amort.)", "—")

        if base_end and with_end:
            st.caption(f"Data de liquidação — **antes:** {base_end.strftime('%d/%m/%Y')}  •  **com amortizações:** {with_end.strftime('%d/%m/%Y')}")

    st.divider()

    # Charts
    st.subheader("Gráficos")

    x = pd.to_datetime(sched["date"])

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=("Capital em divida","Balanço Mensal", "Prestação / Juro / Capital","Rendimento disponivel"),
    )

    # Panel 1: Balance
    fig.add_trace(
        go.Scatter(x=x, y=sched["balance"], name="Saldo", line=dict(width=3)),
        row=1, col=1
    )
    
    # Panel 2: flows as grouped bars
    fig.add_trace(
        go.Bar(x=x, y=sched["income"], name="Rendimentos",
               offsetgroup="p2",
               marker=dict(line=dict(width=0))),  # no outline (or set width=1)
        row=2, col=1
    )
    fig.add_trace(
        go.Bar(x=x, y=sched["payment"], name="Prestação",
               offsetgroup="p2",
               marker=dict(line=dict(width=0))),
        row=2, col=1
    )
    fig.add_trace(
        go.Bar(x=x, y=sched["expenses"], name="Outras Despesas",
               offsetgroup="p2",
               marker=dict(line=dict(width=0))),
        row=2, col=1
    )
    
    # Panel 3: components as bars; keep total as a line (recommended)
    fig.add_trace(
        go.Scatter(x=x, y=sched["payment"], name="Prestação",
                   line=dict(width=2.5)),  # line OK here
        row=3, col=1
    )
    fig.add_trace(
        go.Bar(x=x, y=sched["interest"], name="Juro",
               offsetgroup="p3",
               marker=dict(line=dict(width=0))),
        row=3, col=1
    )
    fig.add_trace(
        go.Bar(x=x, y=sched["principal"], name="Capital",
               offsetgroup="p3",
               marker=dict(line=dict(width=0))),
        row=3, col=1
    )
    
    # Panel 4: keep as a line (reads better), but if you insist on bars, remove line/dash
    fig.add_trace(
        go.Scatter(x=x, y=sched["available_income"], name="Rendimento Disponivel",
                   line=dict(width=2.5)),
        row=4, col=1
    )
    
    # Global bar layout
    fig.update_layout(
        barmode="group",           # grouped bars in panels 2 and 3
        bargap=0.15,
        bargroupgap=0.05,
    )


    # 'now' marker + shading
    if len(sched) > 0:
        row_date = pd.Timestamp(sched.iloc[min(idx_now, len(sched)-1)]["date"])
        ref_ts = min(row_date, pd.Timestamp.max)
        fig.add_vline(x=ref_ts, line=dict(width=2, color="rgba(0,0,0,0.5)"))
        fig.add_vrect(x0=x.min(), x1=ref_ts, fillcolor="rgba(0,128,0,0.05)", line_width=0, layer="below")
        fig.add_vrect(x0=ref_ts, x1=x.max(), fillcolor="rgba(128,0,0,0.04)", line_width=0, layer="below")

    # Reset markers (variable)
    if "reset" in sched.columns and sched["reset"].any():
        for _, r in sched[sched["reset"]].iterrows():
            fig.add_vline(x=pd.to_datetime(r["date"]), line_dash="dot", opacity=0.35)

    # Amortization markers (triangles)
    am_rows = sched[sched.get("amortization", pd.Series(dtype=float)) > 0]
    if not am_rows.empty:
        fig.add_trace(
            go.Scatter(
                x=pd.to_datetime(am_rows["date"]),
                y=am_rows["balance"],
                mode="markers",
                name="Amortização",
                marker_symbol="triangle-up",
                marker_size=10,
                marker_line_width=1,
            ),
            row=1, col=1
        )

    # Visual separation between panels
    fig.add_hrect(y0=0.52, y1=1.00, fillcolor="rgba(0,0,0,0.03)", line_width=0, layer="below", yref="paper")
    fig.add_hrect(y0=0.00, y1=0.48, fillcolor="rgba(0,0,0,0.02)", line_width=0, layer="below", yref="paper")
    fig.add_shape(type="line", x0=0, x1=1, y0=0.50, y1=0.50, xref="paper", yref="paper",
                  line=dict(width=1, color="rgba(0,0,0,0.10)"))

    # Compute anchors
    x_all = pd.to_datetime(sched["date"])
    min_d = x_all.min()
    max_d = x_all.max()

    # "now" as a Timestamp, clipped into the schedule span for safety
    if 0 <= idx_now < len(sched):
        now_ts = pd.Timestamp(sched.iloc[idx_now]["date"])
    elif idx_now <= 0:
        now_ts = pd.Timestamp(min_d)
    else:
        now_ts = pd.Timestamp(max_d)

    def _clamp(end_ts):
        return min(pd.Timestamp(end_ts), pd.Timestamp(max_d))

    # Axes
    fig.update_xaxes(
        title_text="Data",
        row=2, col=1,
        showgrid=True, gridcolor="rgba(0,0,0,0.08)",
    )
    fig.update_xaxes(
        row=1, col=1,
        showgrid=True, gridcolor="rgba(0,0,0,0.08)",
    )

    # Start-anchored buttons from the "Posição atual"
    fig.update_layout(
        updatemenus=[dict(
            type="buttons",
            direction="right",
            x=0, y=1.15, xanchor="left", yanchor="bottom",
            pad={"r": 10, "t": 0},
            showactive=False,  # keeps labels readable
            buttons=[
                dict(
                    label="1Y (agora→)",
                    method="relayout",
                    args=[{"xaxis.range": [now_ts, _clamp(now_ts + pd.DateOffset(years=1))]}],
                ),
                dict(
                    label="5Y",
                    method="relayout",
                    args=[{"xaxis.range": [now_ts, _clamp(now_ts + pd.DateOffset(years=5))]}],
                ),
                dict(
                    label="10Y",
                    method="relayout",
                    args=[{"xaxis.range": [now_ts, _clamp(now_ts + pd.DateOffset(years=10))]}],
                ),
                dict(
                    label="Total",
                    method="relayout",
                    args=[{"xaxis.range": [min_d, max_d]}],
                ),
            ],
        )],
        height=1800,
        template="plotly_dark",  # keeping your current theme
        margin=dict(t=80, r=20, b=30, l=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Plano de pagamentos")

    tab1, tab2 = st.tabs(["Plano completo", "A partir de hoje"])
    with tab1:
        st.dataframe(sched, use_container_width=True, hide_index=True)

    with tab2:
        view_cols = ["now_flag", "period", "date", "rate_annual_pct", "payment", "interest", "principal",
                     "extras_monthly", "amortization", "amort_fee", "balance"]
        st.dataframe(sched.iloc[idx_now:][view_cols], use_container_width=True, hide_index=True)
