#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
app_2.py — Error-fixed & lightweight

Keeps your original features:
  • Revenue upload & normalization (SAP RU→EN)   (no missing imports)
  • Expenditures upload & yearly comparison      (uses your finance_core normalize_expenditures)
  • Revenue range comparison (Actual vs Previous) with optional last-month forecast
  • Exports: Excel (bytes). PPT/PDF try to work if libs are present; otherwise warn (no crash).

Key fixes:
  - Removed calls to functions not present in your current finance_core.py
  - Added local wrappers: validate_revenue(), normalize_revenue()
  - Implemented months_between(), compare_ranges_revenue() locally
  - Implemented load_month_amount_file() locally (tolerant to RU/EN columns)
  - Exports: Excel as bytes here (so Streamlit can download)
"""

from __future__ import annotations

import io
import os
import calendar
import datetime as dt
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# ===================== Import only AVAILABLE things =======================
# Your finance_core.py (current revision) exposes these names:
from finance_core import (  # type: ignore
    read_excel_any,
    translate_columns,
    normalize,                         # (df, corr_map) -> (df_norm, dq)
    compute_g1_transport,
    SPECIAL_CORR_DEFAULT,
    CORRESPONDENT_MAP_DEFAULT,
    normalize_expenditures,
    validate_expenditures,
    compare_expenditures,
    parse_month_ru,
)

# We will NOT import: validate_revenue, normalize_revenue, compare_ranges_revenue,
# export_excel(bytes), build_pptx, build_pdf, load_month_amount_file
# because they aren't present in your current finance_core.py version.

# If your core exposes DATE_SOURCE, use it; otherwise default to "Data of Document"
try:
    from finance_core import DATE_SOURCE as _DATE_SOURCE_DEFAULT  # type: ignore
except Exception:
    _DATE_SOURCE_DEFAULT = "Data of Document"

# ============================ Helpers (local) =============================

VAT_RATE_DEFAULT = 0.12  # fallback default for UI input

@dataclass
class _ValResult:
    ok: bool
    missing: List[str]
    suggestions: str = ""

REQUIRED_COLS = [
    "Correspondent", "Number of Documents", "Data of Document", "Data of transaction",
    "Number of request", "Material/SAP Code", "Name", "Qty", "Measurement",
    "Amount", "Currency", "Warranty"
]

def validate_revenue(df: pd.DataFrame) -> _ValResult:
    """
    Simple validator for SAP revenue after translate_columns().
    We accept if at least these required columns are present.
    """
    # Try translation first (safe to run twice)
    d = translate_columns(df, verbose=False)
    missing = [c for c in REQUIRED_COLS if c not in d.columns]
    if missing:
        tips = ("Make sure the SAP file has either English headers or Russian headers "
                "that can be translated to: " + ", ".join(REQUIRED_COLS))
        return _ValResult(False, missing, tips)
    return _ValResult(True, [])

def normalize_revenue(df: pd.DataFrame,
                      corr_map: Dict[int, str] | None = None,
                      special_corr: List[int] | None = None) -> pd.DataFrame:
    """
    Wrapper that mirrors the original behavior using your current finance_core:
      - translate_columns()
      - normalize(df, corr_map)   -> adds parsed dates & coercions
      - compute_g1_transport()    -> adds 'g1_transport'
    """
    corr_map = corr_map or CORRESPONDENT_MAP_DEFAULT
    special_corr = special_corr or SPECIAL_CORR_DEFAULT

    d = translate_columns(df, verbose=False)
    d, _dq = normalize(d, corr_map)
    d = compute_g1_transport(d, special_corr)
    return d

def months_between(start_ym: str, end_ym: str) -> List[str]:
    """Inclusive list of YYYY-MM between start and end."""
    sy, sm = map(int, start_ym.split("-"))
    ey, em = map(int, end_ym.split("-"))
    cur = dt.date(sy, sm, 1)
    end = dt.date(ey, em, 1)
    out = []
    while cur <= end:
        out.append(f"{cur.year:04d}-{cur.month:02d}")
        # add 1 month
        if cur.month == 12:
            cur = dt.date(cur.year + 1, 1, 1)
        else:
            cur = dt.date(cur.year, cur.month + 1, 1)
    return out

def _calc_after_vat(amount_vat_incl: float, vat_rate: float, vat_mode: str) -> float:
    vm = (vat_mode or "extract").lower()
    return amount_vat_incl / (1.0 + float(vat_rate)) if vm == "extract" else amount_vat_incl * (1.0 - float(vat_rate))

def _pick_date_column(df: pd.DataFrame, date_cols: List[str]) -> str:
    """Pick the first existing date column from user selection."""
    for c in date_cols:
        if c in df.columns:
            return c
    # last resort: any datetime-like column
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            return c
    raise ValueError(f"No usable date columns found. Tried: {date_cols}")

def _ensure_month_key(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d["Month"] = d[date_col].dt.to_period("M").astype(str)
    return d

def _subset_by_months(df: pd.DataFrame, months: List[str]) -> pd.DataFrame:
    return df[df["Month"].isin(months)].copy()

def _days_in_month(ym: str) -> int:
    y, m = map(int, ym.split("-"))
    return calendar.monthrange(y, m)[1]

def _active_days_factor(df_month: pd.DataFrame,
                        date_col: str,
                        vat_rate: float,
                        vat_mode: str,
                        nonempty_only: bool,
                        exclude_sundays: bool) -> Tuple[int, int, float]:
    """
    Compute (active_days, month_days, factor) inside a single month slice.
    active_day = day with after_vat_excl_cc > 0 (optionally) and not Sunday (optionally).
    factor = month_days / active_days (or 0 if active_days == 0).
    """
    if df_month.empty:
        month_key = None
        act, m_days = 0, 0
        return act, m_days, 0.0

    # Month key from the first row
    month_key = str(df_month["Month"].iloc[0])
    y, m = map(int, month_key.split("-"))
    month_days = _days_in_month(month_key)

    # group by date (one day per row)
    d = df_month.copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d["_date"] = d[date_col].dt.date
    g = d.groupby("_date", dropna=False).agg(gross=("Amount", "sum"), g1=("g1_transport", "sum")).reset_index()

    # calc after VAT excl CC per day
    g["after_vat_excl_cc"] = _calc_after_vat(g["gross"] - g["g1"], vat_rate, vat_mode)

    def is_sun(dateobj: dt.date) -> bool:
        try:
            return dt.date(dateobj.year, dateobj.month, dateobj.day).weekday() == 6
        except Exception:
            return False

    if nonempty_only:
        mask = g["after_vat_excl_cc"] > 0
    else:
        mask = pd.Series(True, index=g.index)

    if exclude_sundays:
        mask = mask & (~g["_date"].apply(is_sun))

    active_days = int(mask.sum())
    # (optionally exclude Sundays from month_days as in your first project)
    if exclude_sundays:
        month_days = sum(
            1 for d0 in pd.date_range(dt.date(y, m, 1), dt.date(y, m, month_days))
            if d0.weekday() != 6
        )

    factor = (month_days / active_days) if active_days else 0.0
    return active_days, month_days, factor

def _read_any_cached(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Simple cached reader (Streamlit cache kept at call site with key=file hash)."""
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".csv":
        return pd.read_csv(io.BytesIO(file_bytes))
    # use finance_core.read_excel_any which supports file-like
    return read_excel_any(io.BytesIO(file_bytes))

def load_month_amount_file_local(file_like_or_bytes) -> pd.DataFrame:
    """
    Accepts xlsx/csv; tolerant to RU/EN columns:
      Month | Месяц
      Amount_USD | USD | Amount
    Returns df with columns: Month (YYYY-MM), Amount_USD (float)
    """
    if isinstance(file_like_or_bytes, (bytes, bytearray)):
        bio = io.BytesIO(file_like_or_bytes)
        try:
            df = pd.read_excel(bio)
        except Exception:
            bio.seek(0)
            df = pd.read_csv(bio)
    else:
        # assume path or buffer
        try:
            df = pd.read_excel(file_like_or_bytes)
        except Exception:
            df = pd.read_csv(file_like_or_bytes)

    cols = {c.strip(): c for c in df.columns}
    month_col = None
    for cand in ["Month", "Месяц"]:
        if cand in cols:
            month_col = cols[cand]; break
    if month_col is None:
        # try to derive from date column if exists
        for c in df.columns:
            if "month" in str(c).lower():
                month_col = c
                break

    amt_col = None
    for cand in ["Amount_USD", "USD", "Amount"]:
        if cand in cols:
            amt_col = cols[cand]; break
    if amt_col is None:
        # pick first numeric-like
        for c in df.columns:
            if pd.api.types.is_numeric_dtype(df[c]):
                amt_col = c
                break

    if month_col is None or amt_col is None:
        # return empty standardized frame
        return pd.DataFrame({"Month": [], "Amount_USD": []})

    # Parse RU month text if needed
    out_month = []
    for v in df[month_col].astype(str).tolist():
        _, _, _, ym = parse_month_ru(v)
        out_month.append(ym if ym else v.strip())

    amounts = pd.to_numeric(df[amt_col], errors="coerce").fillna(0.0)

    res = pd.DataFrame({"Month": out_month, "Amount_USD": amounts})
    # squeeze unknown formats "2024-1" -> "2024-01"
    def _fix_ym(s):
        try:
            y, m = s.split("-")
            return f"{int(y):04d}-{int(m):02d}"
        except Exception:
            return s
    res["Month"] = res["Month"].astype(str).str.strip().apply(_fix_ym)
    # aggregate if duplicates
    res = res.groupby("Month", as_index=False)["Amount_USD"].sum()
    return res

def _cc_amount_for_month(cc_df: Optional[pd.DataFrame], ym: str) -> float:
    if cc_df is None or cc_df.empty or "Month" not in cc_df.columns or "Amount_USD" not in cc_df.columns:
        return 0.0
    s = cc_df.loc[cc_df["Month"].astype(str) == ym, "Amount_USD"]
    return float(s.iloc[0]) if not s.empty and pd.notna(s.iloc[0]) else 0.0

def compare_ranges_revenue(
    df_rev: pd.DataFrame,
    ar_start: str, ar_end: str,
    pr_start: str, pr_end: str,
    date_cols: List[str],
    vat_rate: float, vat_mode: str,
    cc_actual_df: Optional[pd.DataFrame],
    cc_prev_df: Optional[pd.DataFrame],
    forecast_last_ym: Optional[str] = None,
    nonempty_only: bool = True,
    exclude_sundays: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build two outputs:
      df_cmp: per-overlap month comparison (After VAT, excl CC)
      df_tot: full-period totals (After VAT excl CC) for Actual & Previous
    Notes:
      - 'Amount' and 'g1_transport' used
      - CC tables optionally added ONLY to totals "incl CC" column; main figures stay "excl CC"
    """
    if df_rev is None or df_rev.empty:
        return pd.DataFrame(), pd.DataFrame()

    date_col = _pick_date_column(df_rev, date_cols)
    d = _ensure_month_key(df_rev, date_col)

    actual_months  = months_between(ar_start, ar_end)
    previous_months = months_between(pr_start, pr_end)

    d_actual  = _subset_by_months(d, actual_months)
    d_prev    = _subset_by_months(d, previous_months)

    # ---- per-month summaries (excl CC) ----
    def _per_month_after_vat(df_in: pd.DataFrame) -> pd.DataFrame:
        if df_in.empty:
            return pd.DataFrame({"Month": [], "AfterVAT_excl_CC": []})
        g = df_in.groupby("Month", as_index=False).agg(
            Gross=("Amount", "sum"),
            G1=("g1_transport", "sum")
        )
        g["AfterVAT_excl_CC"] = _calc_after_vat(g["Gross"] - g["G1"], vat_rate, vat_mode)
        return g[["Month", "AfterVAT_excl_CC"]]

    act_pm  = _per_month_after_vat(d_actual)
    prev_pm = _per_month_after_vat(d_prev)

    # ---- optional forecast for the last actual month ----
    if forecast_last_ym and (forecast_last_ym in actual_months):
        # slice only that month
        last_m_df = d_actual[d_actual["Month"] == forecast_last_ym]
        if not last_m_df.empty:
            act_days, mon_days, factor = _active_days_factor(
                last_m_df, date_col, vat_rate, vat_mode, nonempty_only, exclude_sundays
            )
            if factor and np.isfinite(factor):
                idx = act_pm["Month"] == forecast_last_ym
                if idx.any():
                    act_pm.loc[idx, "AfterVAT_excl_CC"] = act_pm.loc[idx, "AfterVAT_excl_CC"] * factor

    # ---- overlap comparison table ----
    overlap = sorted(set(act_pm["Month"]).intersection(set(prev_pm["Month"])))
    if overlap:
        cmp_df = (
            pd.DataFrame({"Month": overlap})
            .merge(act_pm, on="Month", how="left")
            .merge(prev_pm.rename(columns={"AfterVAT_excl_CC": "Prev_AfterVAT_excl_CC"}), on="Month", how="left")
        )
        cmp_df = cmp_df.fillna(0.0)
        cmp_df["Delta"] = cmp_df["AfterVAT_excl_CC"] - cmp_df["Prev_AfterVAT_excl_CC"]
        cmp_df["% vs Prev"] = np.where(
            cmp_df["Prev_AfterVAT_excl_CC"] != 0,
            (cmp_df["Delta"] / cmp_df["Prev_AfterVAT_excl_CC"]) * 100.0,
            np.nan,
        )
        df_cmp = cmp_df.sort_values("Month")
    else:
        df_cmp = pd.DataFrame(columns=["Month", "AfterVAT_excl_CC", "Prev_AfterVAT_excl_CC", "Delta", "% vs Prev"])

    # ---- totals (excl & incl CC) ----
    def _total_excl_cc(df_pm: pd.DataFrame) -> float:
        return float(pd.to_numeric(df_pm["AfterVAT_excl_CC"], errors="coerce").sum()) if not df_pm.empty else 0.0

    total_actual_excl = _total_excl_cc(act_pm)
    total_prev_excl   = _total_excl_cc(prev_pm)

    # Add Call Center (assumed VAT-included), convert to After VAT with same vat_mode
    total_actual_incl = total_actual_excl + sum(
        _calc_after_vat(_cc_amount_for_month(cc_actual_df, ym), vat_rate, vat_mode) for ym in actual_months
    )
    total_prev_incl = total_prev_excl + sum(
        _calc_after_vat(_cc_amount_for_month(cc_prev_df, ym), vat_rate, vat_mode) for ym in previous_months
    )

    df_tot = pd.DataFrame([
        {"Period": "Actual (Full Period)",
         "Total_AfterVAT": round(total_actual_excl, 2),
         "Total_AfterVAT_incl_CC": round(total_actual_incl, 2)},
        {"Period": "Previous (Full Period)",
         "Total_AfterVAT": round(total_prev_excl, 2),
         "Total_AfterVAT_incl_CC": round(total_prev_incl, 2)},
    ])

    return df_cmp, df_tot

# --------------------------- Export helpers ---------------------------

def export_excel(sheets: Dict[str, pd.DataFrame]) -> bytes:
    """
    Generate a single .xlsx bytes object containing all sheets (for Streamlit download).
    """
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="xlsxwriter") as writer:
        wb = writer.book
        hdr = wb.add_format({"bold": True, "bg_color": "#EFEFEF", "border": 1})
        money = wb.add_format({"num_format": "#,##0.00"})

        for name, df in sheets.items():
            df_out = df.copy()
            df_out.to_excel(writer, sheet_name=name[:31], index=False)
            ws = writer.sheets[name[:31]]
            # header style + autosize
            for j, col in enumerate(df_out.columns):
                ws.write(0, j, str(col), hdr)
                # crude autosize
                width = max(10, min(60, max([len(str(col))] + [len(str(v)) for v in df_out[col].head(200)]) + 2))
                ws.set_column(j, j, width, money if df_out[col].dtype.kind in "fc" else None)
    bio.seek(0)
    return bio.getvalue()

def build_pptx(title: str, subtitle: str, charts: Dict[str, pd.DataFrame], tables: Dict[str, pd.DataFrame]) -> bytes:
    """
    Lightweight PPTX: if python-pptx not installed, warn and return empty bytes to avoid crash.
    """
    try:
        from pptx import Presentation
        from pptx.util import Inches, Pt
    except Exception:
        st.warning("python-pptx is not installed; PPTX export is unavailable.")
        return b""

    prs = Presentation()
    # Title slide
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    slide.shapes.title.text = title
    slide.placeholders[1].text = subtitle

    # Add a couple of compact tables
    for idx, (name, df) in enumerate(tables.items()):
        if idx >= 2:
            break
        slide = prs.slides.add_slide(prs.slide_layouts[5])  # title only
        slide.shapes.title.text = name
        rows, cols = (min(12, len(df)) + 1, min(6, df.shape[1]))
        x, y, cx, cy = Inches(0.5), Inches(1.2), Inches(9.0), Inches(4.5)
        table = slide.shapes.add_table(rows, cols, x, y, cx, cy).table
        # headers
        for j, col in enumerate(df.columns[:cols]):
            table.cell(0, j).text = str(col)
        # rows
        for i in range(rows - 1):
            for j in range(cols):
                try:
                    val = df.iloc[i, j]
                except Exception:
                    val = ""
                table.cell(i + 1, j).text = "" if pd.isna(val) else str(val)
    bio = io.BytesIO()
    prs.save(bio)
    bio.seek(0)
    return bio.getvalue()

def build_pdf(title: str, subtitle: str, tables: Dict[str, pd.DataFrame]) -> bytes:
    """
    Minimal PDF via reportlab if available. Otherwise warn and return empty bytes.
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
        from reportlab.lib.units import cm
    except Exception:
        st.warning("reportlab is not installed; PDF export is unavailable.")
        return b""

    bio = io.BytesIO()
    c = canvas.Canvas(bio, pagesize=A4)
    width, height = A4

    # Title page
    c.setFont("Helvetica-Bold", 18)
    c.drawString(2 * cm, height - 3 * cm, title)
    c.setFont("Helvetica", 12)
    c.drawString(2 * cm, height - 4 * cm, subtitle)
    c.showPage()

    # A couple of tables (basic text)
    for idx, (name, df) in enumerate(tables.items()):
        if idx >= 2:
            break
        c.setFont("Helvetica-Bold", 14)
        c.drawString(2 * cm, height - 2 * cm, name)
        c.setFont("Helvetica", 9)
        y = height - 3 * cm
        # header
        cols = list(map(str, df.columns[:6]))
        c.drawString(2 * cm, y, " | ".join(cols))
        y -= 0.5 * cm
        # rows (max 12)
        for i in range(min(12, len(df))):
            vals = [str(df.iloc[i, j]) for j in range(min(6, df.shape[1]))]
            c.drawString(2 * cm, y, " | ".join(vals))
            y -= 0.5 * cm
            if y < 3 * cm:
                c.showPage()
                y = height - 3 * cm
        c.showPage()

    c.save()
    bio.seek(0)
    return bio.getvalue()

# ============================ PAGE ============================
st.set_page_config(page_title="Artel Financial Suite — Comparisons", page_icon="📊", layout="wide")
st.title("📊 Artel Financial Suite — Range & Yearly Comparisons")

# ============================ SIDEBAR ============================
st.sidebar.header("⚙️ Settings")
vat_mode = st.sidebar.selectbox("VAT Mode (Revenue SAP)", ["extract", "add"], index=0)
vat_rate = st.sidebar.number_input("VAT rate (Revenue)", min_value=0.0, max_value=1.0, value=float(VAT_RATE_DEFAULT), step=0.01)

date_sources_all = ["Data of Document", "Data of transaction"]
date_sources = st.sidebar.multiselect("Date Source(s) for Revenue", date_sources_all, default=date_sources_all)
nonempty_only = st.sidebar.checkbox("Forecast: use only non-empty days", value=True)
exclude_sundays = st.sidebar.checkbox("Forecast: exclude Sundays", value=True)

st.sidebar.caption("SPECIAL_CORR is applied in normalization (G1 'ВЫЗОВ' exception).")

# ============================ TABS ============================
tab_rev, tab_exp, tab_cmp, tab_export = st.tabs([
    "📊 Revenue",
    "💸 Expenditures (Yearly)",
    "🧮 Comparison (Ranges)",
    "📤 Export",
])

# ============================ STATE ============================
for key in ["rev_df", "exp_df", "cc_actual_df", "cc_prev_df", "rev_cmp", "rev_tot", "exp_cmp", "exp_tot"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ============================ REVENUE TAB ============================
with tab_rev:
    st.subheader("Revenue Upload (SAP)")
    rev_files = st.file_uploader(
        "Upload one or more SAP Revenue files (.xlsx/.xls/.csv)",
        type=["xlsx", "xls", "csv"],
        accept_multiple_files=True,
        key="rev_up",
    )

    if rev_files:
        try:
            # cache by file content to avoid re-reading on rerun
            dfs = []
            for f in rev_files:
                key = f"revcache::{f.name}::{len(f.getvalue())}"
                df = st.session_state.get(key)
                if df is None:
                    df = _read_any_cached(f.getvalue(), f.name)
                    st.session_state[key] = df
                dfs.append(df)
            d = pd.concat(dfs, ignore_index=True)

            # validate then normalize via local wrappers
            v = validate_revenue(d)
            if not v.ok:
                st.error("⚠️ Invalid Revenue file structure.")
                st.write("Missing required columns:", v.missing)
                if v.suggestions:
                    st.info(f"Suggestions: {v.suggestions}")
            else:
                st.success("✅ Revenue file(s) validated.")
                st.session_state.rev_df = normalize_revenue(d, corr_map=CORRESPONDENT_MAP_DEFAULT, special_corr=SPECIAL_CORR_DEFAULT)
                st.dataframe(st.session_state.rev_df.head(20), use_container_width=True)
        except Exception as e:
            st.exception(e)

    st.markdown("---")
    st.subheader("Call Center & Admin (Month files)")
    c1, c2 = st.columns(2)
    with c1:
        cc_actual = st.file_uploader("Call Center - Actual Period (Month | Amount_USD)", type=["xlsx", "csv"], key="cc_a")
        if cc_actual:
            try:
                df = load_month_amount_file_local(cc_actual.getvalue())
                st.session_state.cc_actual_df = df
                st.caption("Parsed CC (Actual):")
                st.dataframe(df, use_container_width=True)
            except Exception as e:
                st.exception(e)
    with c2:
        cc_prev = st.file_uploader("Call Center - Previous Period (Month | Amount_USD)", type=["xlsx", "csv"], key="cc_p")
        if cc_prev:
            try:
                df = load_month_amount_file_local(cc_prev.getvalue())
                st.session_state.cc_prev_df = df
                st.caption("Parsed CC (Previous):")
                st.dataframe(df, use_container_width=True)
            except Exception as e:
                st.exception(e)

# ============================ EXPENDITURES TAB ============================
with tab_exp:
    st.subheader("Expenditures Upload (RU headers)")
    exp_file = st.file_uploader(
        "Upload Expenditures file (.xlsx/.xls/.csv) with RU headers",
        type=["xlsx", "xls", "csv"], key="exp_up"
    )

    if exp_file:
        try:
            key = f"expcache::{exp_file.name}::{len(exp_file.getvalue())}"
            d = st.session_state.get(key)
            if d is None:
                d = _read_any_cached(exp_file.getvalue(), exp_file.name)
                st.session_state[key] = d
            v = validate_expenditures(d)
            if not v.ok:
                st.error("⚠️ Invalid Expenditures file.")
                st.write("Missing required columns (RU→EN):", v.missing)
            else:
                st.success("✅ Expenditures file validated.")
                st.session_state.exp_df = normalize_expenditures(d)
                st.dataframe(st.session_state.exp_df.head(20), use_container_width=True)
        except Exception as e:
            st.exception(e)

    st.markdown("### Yearly Ranges")
    c1, c2 = st.columns(2)
    with c1:
        a_start = st.text_input("Actual Start (YYYY-MM)", "2025-01")
        a_end   = st.text_input("Actual End (YYYY-MM)",   "2025-12")
    with c2:
        p_start = st.text_input("Previous Start (YYYY-MM)", "2024-01")
        p_end   = st.text_input("Previous End (YYYY-MM)",   "2024-12")

    if st.button("Compare Expenditures (Yearly)"):
        if st.session_state.exp_df is None:
            st.warning("Upload Expenditures first.")
        else:
            try:
                exp_cmp, exp_tot = compare_expenditures(
                    st.session_state.exp_df, a_start.strip(), a_end.strip(), p_start.strip(), p_end.strip()
                )
                if (exp_cmp is None or exp_cmp.empty) and (exp_tot is None or exp_tot.empty):
                    st.error("No rows matched the selected ranges. Check that 'Месяц' values are parseable (e.g., 'Январь 2025').")
                else:
                    st.subheader("By Category Comparison (Actual vs Previous)")
                    if exp_cmp is not None and not exp_cmp.empty:
                        st.dataframe(exp_cmp, use_container_width=True)
                    else:
                        st.info("No category comparison to display.")

                    st.subheader("Totals")
                    if exp_tot is not None and not exp_tot.empty:
                        st.dataframe(exp_tot, use_container_width=True)
                    else:
                        st.info("No totals to display.")

                    st.session_state.exp_cmp = exp_cmp
                    st.session_state.exp_tot = exp_tot
            except Exception as e:
                st.exception(e)

# ============================ COMPARISON (RANGES) TAB ============================
with tab_cmp:
    st.subheader("Revenue Range Comparison")
    c1, c2 = st.columns(2)
    with c1:
        ar_start = st.text_input("Actual Period Start (YYYY-MM)", "2025-01", key="ar_s")
        ar_end   = st.text_input("Actual Period End (YYYY-MM)",   "2025-10", key="ar_e")
    with c2:
        pr_start = st.text_input("Previous Period Start (YYYY-MM)", "2024-01", key="pr_s")
        pr_end   = st.text_input("Previous Period End (YYYY-MM)",   "2024-12", key="pr_e")

    forecast_last = st.checkbox("Forecast Actual End Month (active-days)", value=True)

    if st.button("Build Range Comparison (Revenue)"):
        if st.session_state.rev_df is None:
            st.warning("Upload Revenue first.")
        else:
            date_cols = [c for c in date_sources if c in st.session_state.rev_df.columns]
            if not date_cols:
                st.error(
                    "Selected date source(s) not found in data.\n\n"
                    f"Chosen: {date_sources}\n"
                    f"Available: {', '.join(map(str, st.session_state.rev_df.columns))}"
                )
            else:
                try:
                    df_cmp, df_tot = compare_ranges_revenue(
                        st.session_state.rev_df,
                        ar_start.strip(), ar_end.strip(),
                        pr_start.strip(), pr_end.strip(),
                        date_cols,
                        float(vat_rate), str(vat_mode),
                        st.session_state.cc_actual_df, st.session_state.cc_prev_df,
                        (ar_end.strip() if forecast_last else None),
                        bool(nonempty_only), bool(exclude_sundays)
                    )
                    if (df_cmp is None or df_cmp.empty) and (df_tot is None or df_tot.empty):
                        st.error("Revenue comparison returned no rows. Check date ranges and that your files contain those months.")
                    else:
                        st.subheader("Overlap Comparison (After VAT, excl CC)")
                        if df_cmp is not None and not df_cmp.empty:
                            st.dataframe(df_cmp, use_container_width=True)
                        else:
                            st.info("No overlap rows to display.")

                        st.subheader("Full-Period Totals")
                        if df_tot is not None and not df_tot.empty:
                            st.dataframe(df_tot, use_container_width=True)
                        else:
                            st.info("No totals to display.")

                        st.session_state.rev_cmp = df_cmp
                        st.session_state.rev_tot = df_tot
                except Exception as e:
                    st.exception(e)

# ============================ EXPORT TAB ============================
with tab_export:
    st.subheader("Export Options")
    export_choice = st.radio("Select Export Type", [
        "Full Report (Revenue + Expenditures + Combined)",
        "Revenue Only",
        "Expenditures Only",
        "Combined Summary",
    ])
    report_title = st.text_input("Report Title", "Artel Financial Overview")
    subtitle = st.text_input("Subtitle", "Generated by Artel Financial Suite")

    # Build the dict of sheets dynamically based on what we have
    sheets: Dict[str, pd.DataFrame] = {}
    if st.session_state.rev_cmp is not None: sheets["Revenue_Overlap"] = st.session_state.rev_cmp
    if st.session_state.rev_tot is not None: sheets["Revenue_Totals"] = st.session_state.rev_tot
    if st.session_state.exp_cmp is not None: sheets["EXP_ByCategory_Compare"] = st.session_state.exp_cmp
    if st.session_state.exp_tot is not None: sheets["EXP_Totals"] = st.session_state.exp_tot

    # Combined quick view
    if st.session_state.get("rev_tot") is not None and st.session_state.get("exp_tot") is not None:
        try:
            rev_tot = st.session_state.rev_tot
            exp_tot = st.session_state.exp_tot
            comb = pd.DataFrame({
                "Metric": [
                    "Revenue Total (Actual)", "Revenue Total (Prev)",
                    "Expenditures Total (Actual)", "Expenditures Total (Prev)"
                ],
                "Value": [
                    float(rev_tot.loc[0, "Total_AfterVAT"]) if not rev_tot.empty else 0.0,
                    float(rev_tot.loc[1, "Total_AfterVAT"]) if not rev_tot.empty else 0.0,
                    float(exp_tot.loc[exp_tot["Period"]=="Actual (Full Period)", "Total_Amount_USD"].values[0]) if not exp_tot.empty else 0.0,
                    float(exp_tot.loc[exp_tot["Period"]=="Previous (Full Period)", "Total_Amount_USD"].values[0]) if not exp_tot.empty else 0.0,
                ]
            })
            sheets["Combined_Summary"] = comb
        except Exception:
            pass

    # Filter by export choice
    def _filter(choice: str, all_sheets: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        if choice == "Revenue Only":
            return {k: v for k, v in all_sheets.items() if k.startswith("Revenue")}
        if choice == "Expenditures Only":
            return {k: v for k, v in all_sheets.items() if k.startswith("EXP_")}
        if choice == "Combined Summary":
            return {k: v for k, v in all_sheets.items() if k.startswith("Combined")}
        return all_sheets

    chosen = _filter(export_choice, sheets)

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("⬇️ Export Excel (.xlsx)"):
            if not chosen:
                st.warning("Nothing to export yet. Build a comparison first.")
            else:
                try:
                    xbytes = export_excel(chosen)
                    st.download_button(
                        "Download Excel",
                        data=xbytes,
                        file_name="financial_report.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )
                except Exception as e:
                    st.exception(e)
    with c2:
        if st.button("⬇️ Export PowerPoint (.pptx)"):
            if not chosen:
                st.warning("Nothing to export yet.")
            else:
                try:
                    # keep light: 2 small tables max
                    charts = {}
                    for name, df in chosen.items():
                        if df.shape[0] > 0 and df.shape[1] >= 2:
                            charts[name] = df.iloc[: min(12, len(df))]
                            if len(charts) >= 2:
                                break
                    pptx_bytes = build_pptx(report_title, subtitle, charts=charts, tables=chosen)
                    if pptx_bytes:
                        st.download_button(
                            "Download PPTX",
                            data=pptx_bytes,
                            file_name="financial_report.pptx",
                            mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                        )
                except Exception as e:
                    st.exception(e)
    with c3:
        if st.button("⬇️ Export PDF (.pdf)"):
            if not chosen:
                st.warning("Nothing to export yet.")
            else:
                try:
                    pdf_bytes = build_pdf(report_title, subtitle, chosen)
                    if pdf_bytes:
                        st.download_button(
                            "Download PDF",
                            data=pdf_bytes,
                            file_name="financial_report.pdf",
                            mime="application/pdf",
                        )
                except Exception as e:
                    st.exception(e)

# ============================ FOOTER ============================
st.markdown("---")
st.caption("Upload Revenue/Expenditures in the first tabs, run the comparisons, then export. All computations are guarded by buttons and error-safe.")
