#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
app_2.py — Lightweight UI for:
  • Revenue upload + range comparison (overlap + totals)
  • Expenditures upload + yearly comparison (by category + totals)
  • Exports to Excel / PPTX / PDF

Notes:
  - No autoruns. Heavy ops only run when you click their buttons.
  - File reads are cached by content bytes to avoid re-parsing on each rerun.
  - Clear error messages when columns/ranges are missing.
"""

import io
import os
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st

# ------------------ Backend (from finance_core.py) ------------------
from finance_core import (
    read_excel_any,                # used inside a cached wrapper
    validate_revenue,
    validate_expenditures,
    normalize_revenue,
    normalize_expenditures,
    compare_ranges_revenue,
    compare_expenditures,
    export_excel,                  # returns bytes for Excel (in your core)
    build_pptx,                    # returns bytes
    build_pdf,                     # returns bytes
    load_month_amount_file,        # used inside a cached wrapper
    SPECIAL_CORR_DEFAULT,
    VAT_RATE_DEFAULT,
)

# ============================ PAGE SETUP ============================
st.set_page_config(page_title="Artel Financial Suite — Comparisons", page_icon="📊", layout="wide")
st.title("📊 Artel Financial Suite — Range & Yearly Comparisons")

# ============================ CACHED IO ============================
@st.cache_data(show_spinner=False)
def _read_any_cached(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Cache wrapper around read_excel_any (accepts .xlsx/.xls/.csv)."""
    # Heuristic by extension
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".csv":
        return pd.read_csv(io.BytesIO(file_bytes))
    # Fallback to your core reader
    return read_excel_any(io.BytesIO(file_bytes))

@st.cache_data(show_spinner=False)
def _load_month_amount_cached(file_bytes: bytes, filename: str) -> pd.DataFrame:
    return load_month_amount_file(io.BytesIO(file_bytes))

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
if "rev_df" not in st.session_state: st.session_state.rev_df = None
if "exp_df" not in st.session_state: st.session_state.exp_df = None
if "cc_actual_df" not in st.session_state: st.session_state.cc_actual_df = None
if "cc_prev_df" not in st.session_state: st.session_state.cc_prev_df = None
if "rev_cmp" not in st.session_state: st.session_state.rev_cmp = None
if "rev_tot" not in st.session_state: st.session_state.rev_tot = None
if "exp_cmp" not in st.session_state: st.session_state.exp_cmp = None
if "exp_tot" not in st.session_state: st.session_state.exp_tot = None

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
            dfs = [_read_any_cached(f.getvalue(), f.name) for f in rev_files]
            d = pd.concat(dfs, ignore_index=True)
            v = validate_revenue(d)
            if not v.ok:
                st.error("⚠️ Invalid Revenue file structure.")
                st.write("Missing required columns:", v.missing)
                if getattr(v, "suggestions", None):
                    st.info(f"Suggestions: {v.suggestions}")
            else:
                st.success("✅ Revenue file(s) validated.")
                st.session_state.rev_df = normalize_revenue(d, corr_map={}, special_corr=SPECIAL_CORR_DEFAULT)
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
                st.session_state.cc_actual_df = _load_month_amount_cached(cc_actual.getvalue(), cc_actual.name)
                st.caption("Parsed CC (Actual):")
                st.dataframe(st.session_state.cc_actual_df, use_container_width=True)
            except Exception as e:
                st.exception(e)
    with c2:
        cc_prev = st.file_uploader("Call Center - Previous Period (Month | Amount_USD)", type=["xlsx", "csv"], key="cc_p")
        if cc_prev:
            try:
                st.session_state.cc_prev_df = _load_month_amount_cached(cc_prev.getvalue(), cc_prev.name)
                st.caption("Parsed CC (Previous):")
                st.dataframe(st.session_state.cc_prev_df, use_container_width=True)
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
            d = _read_any_cached(exp_file.getvalue(), exp_file.name)
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
                    st.error("No rows matched the selected ranges. Check that 'Месяц' has recognizable months (e.g., 'Январь 2025').")
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
            # ensure chosen date columns exist after normalization
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
                        st.subheader("Overlap Comparison")
                        if df_cmp is not None and not df_cmp.empty:
                            st.dataframe(df_cmp, use_container_width=True)
                        else:
                            st.info("No overlap rows to display.")

                        st.subheader("Full-Period Totals (no % because lengths can differ)")
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
            # Be tolerant if structures differ
            pass

    # Filter by export choice
    def filter_sheets(choice: str, all_sheets: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        if choice == "Revenue Only":
            return {k: v for k, v in all_sheets.items() if k.startswith("Revenue")}
        if choice == "Expenditures Only":
            return {k: v for k, v in all_sheets.items() if k.startswith("EXP_")}
        if choice == "Combined Summary":
            return {k: v for k, v in all_sheets.items() if k.startswith("Combined")}
        return all_sheets

    chosen = filter_sheets(export_choice, sheets)

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
                    # keep charts small/light: at most 2 compact tables
                    charts = {}
                    for name, df in chosen.items():
                        if df.shape[0] > 0 and df.shape[1] >= 2:
                            charts[name] = df.iloc[: min(12, len(df))]
                            if len(charts) >= 2:
                                break
                    pptx_bytes = build_pptx(report_title, subtitle, charts=charts, tables=chosen)
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
st.caption("Upload data in the first two tabs, then run comparisons in the other tabs. Exports are created from whatever results are currently in memory.")
